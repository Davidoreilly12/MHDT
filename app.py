import os
import pickle
from typing import Dict, List, Optional, Tuple

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import swin_v2_b, Swin_V2_B_Weights
from PIL import Image
from huggingface_hub import hf_hub_download

import shap

# =========================================================
# CONFIG
# =========================================================
HF_REPO = "DOReilly2/swin_regressor"             # Hugging Face repo for CLM extractor
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WEIGHTS_DIR = "weights"                           # contains beta_pos.npy, beta_neg.npy, beta_r.npy
MODELS_DIR  = "models"                            # contains DeepSets biomarker models
CLM_SCALER_PATH = "env_scaler.pkl"  # optional {'mu':(7,), 'std':(7,)}

# UI controls
TOPK_BIOMARKERS = st.sidebar.number_input("Top-K biomarkers (None=all)", min_value=0, max_value=10000, value=0, step=50)
RUN_SHAPLEY      = st.sidebar.checkbox("Compute Monte-Carlo Shapley over image subsets", value=True)
NUM_PERM         = st.sidebar.number_input("Shapley permutations", min_value=32, max_value=5000, value=512, step=32)
SHAPLEY_SEED     = st.sidebar.number_input("Shapley random seed", min_value=0, max_value=10**6, value=42, step=1)

RUN_CHANNEL_SHAP = st.sidebar.checkbox("Optional: channel-level SHAP (CLM → DeepSets → β·ŷ)", value=False)
SHAP_BG_MAX      = st.sidebar.number_input("Channel SHAP background size", min_value=1, max_value=200, value=20, step=1)
SHAP_NSAMPLES_TXT= st.sidebar.text_input("Channel SHAP nsamples (int or 'auto')", value="auto")

dimension_labels = [
    "Layers of the Landscape_embedding",
    "Landform_embedding",
    "Biodiversity_embedding",
    "Color and Light_embedding",
    "Compatibility_embedding",
    "Archetypal Elements_embedding",
    "Character of Peace and Silence_embedding"
]

# =========================================================
# UTILITIES
# =========================================================
def barplot(values: np.ndarray, labels: list, title: str):
    order = np.argsort(np.abs(values))[::-1]
    vals = values[order]
    labs = [labels[i] for i in order]
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in vals]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(range(len(vals)), vals, color=colors)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labs)
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=1)
    ax.set_title(title)
    ax.set_xlabel("Contribution")
    st.pyplot(fig)

def zscore(v: np.ndarray) -> np.ndarray:
    m = np.mean(v)
    s = np.std(v) + 1e-8
    return (v - m) / s

# =========================================================
# CLM EXTRACTOR
# =========================================================
@st.cache_resource
def load_context_embeddings():
    emb = {}
    for label in dimension_labels:
        path = hf_hub_download(repo_id=HF_REPO, filename=f"context_embeddings/{label}.pt")
        emb_t = torch.load(path, map_location="cpu")
        emb[label] = emb_t.squeeze()
    return emb

CONTEXT_EMBEDDINGS = load_context_embeddings()

class MultiContextSwinRegressor(nn.Module):
    def __init__(self, context_embeddings: dict):
        super().__init__()
        self.swin = swin_v2_b(weights=Swin_V2_B_Weights.IMAGENET1K_V1)
        self.swin.head = nn.Identity()
        self.context_embeddings = nn.ParameterDict({
            label: nn.Parameter(context_embeddings[label].float().unsqueeze(0), requires_grad=False).squeeze(0)
            for label in context_embeddings
        })
        self.fusion_heads = nn.ModuleDict({
            label: nn.Sequential(
                nn.Linear(1024 + self.context_embeddings[label].shape[0], 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 1)
            ) for label in self.context_embeddings
        })

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # [B, 1024]
        image_feat = self.swin(image)
        outs = []
        for label in self.context_embeddings:
            ctx = self.context_embeddings[label].expand(image_feat.size(0), -1)
            fused = torch.cat([image_feat, ctx], dim=1)
            score = self.fusion_heads[label](fused)  # [B,1]
            outs.append(score)
        return torch.cat(outs, dim=1)  # [B, 7]

@st.cache_resource
def load_clm_model():
    model_path = hf_hub_download(repo_id=HF_REPO, filename="swin_regressor.pt")
    state_dict = torch.load(model_path, map_location="cpu")
    model = MultiContextSwinRegressor(CONTEXT_EMBEDDINGS)
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE).eval()
    st.info("CLM model loaded.")
    return model

CLM_MODEL = load_clm_model()

def preprocess_image(image: Image.Image) -> torch.Tensor:
    val_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = image.convert("RGB")
    return val_transform(image).unsqueeze(0).to(DEVICE)

# optional CLM standardization
def maybe_load_clm_scaler():
    if os.path.exists(CLM_SCALER_PATH):
        with open(CLM_SCALER_PATH, "rb") as f:
            obj = pickle.load(f)
        mu = np.asarray(obj["mu"], dtype=np.float32).reshape(-1)
        sd = np.asarray(obj["std"], dtype=np.float32).reshape(-1)
        if mu.shape[0] == 7 and sd.shape[0] == 7:
            return mu, sd
    return None

CLM_SCALER = maybe_load_clm_scaler()

def clm_to_elements(clm_vec: np.ndarray) -> np.ndarray:
    """Turn 7-D CLM vector into DeepSets elements [[z, channel_id], ...]."""
    x = clm_vec.astype(np.float32)
    if CLM_SCALER is not None:
        mu, sd = CLM_SCALER
        x = (x - mu) / (sd + 1e-6)
    elems = np.array([[float(x[i]), float(i)] for i in range(7)], dtype=np.float32)  # (7,2)
    return elems

# =========================================================
# LOAD β VECTORS
# =========================================================
@st.cache_resource
def load_betas() -> Dict[str, np.ndarray]:
    beta_pos = np.load(os.path.join(WEIGHTS_DIR, "beta_pos.npy")).astype(np.float32).reshape(-1)
    beta_neg = np.load(os.path.join(WEIGHTS_DIR, "beta_neg.npy")).astype(np.float32).reshape(-1)
    beta_r   = np.load(os.path.join(WEIGHTS_DIR, "beta_r.npy")).astype(np.float32).reshape(-1)
    if not (len(beta_pos) == len(beta_neg) == len(beta_r)):
        raise ValueError("beta_pos/neg/r must have the same length D.")
    return {"positive": beta_pos, "negative": beta_neg, "r": beta_r}

BETAS = load_betas()
D_FULL = len(BETAS["positive"])

# =========================================================
# DEEPSETS LOADER (original biomarker models)
# =========================================================
class DeepSet(nn.Module):
    """
    Must match your training code:
      phi: MLP
      attn: scalar gate
      rho: MLP -> (mu, log_var) but we use mu only
    """
    def __init__(self, input_dim=2, phi_dim=64, rho_dim=64):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(input_dim, phi_dim), nn.ReLU(),
            nn.Linear(phi_dim, phi_dim), nn.ReLU()
        )
        self.attn = nn.Sequential(nn.Linear(phi_dim, 1), nn.Sigmoid())
        self.rho = nn.Sequential(
            nn.Linear(phi_dim, rho_dim), nn.ReLU(),
            nn.Linear(rho_dim, 2)
        )

    def forward(self, x: torch.Tensor):
        # x: (T,2) elements
        z = self.phi(x)                 # (T, phi_dim)
        w = self.attn(z)                # (T, 1) in [0,1]
        agg = (w * z).sum(dim=0)        # (phi_dim,)
        out = self.rho(agg)             # (2,)
        mu = out[0]
        log_var = torch.clamp(out[1], min=-3.0, max=3.0)
        return mu, log_var, agg

class DeepSetsBank:
    """
    Lazy-loads per-biomarker DeepSets from MODELS_DIR.
    Supports files:
      model_{d:04d}.pt / .pkl
      deepset_biomarker_{d}.pt / .pkl
    """
    def __init__(self, D: int, models_dir: str):
        self.D = D
        self.dir = models_dir
        self.cache: Dict[int, Optional[nn.Module]] = {}

    def _try_paths(self, d: int) -> List[str]:
        names = [
            f"model_{d:04d}.pt", f"deepset_biomarker_{d}.pt",
            f"model_{d:04d}.pkl", f"deepset_biomarker_{d}.pkl"
        ]
        return [os.path.join(self.dir, n) for n in names]

    def _load_one(self, d: int) -> Optional[nn.Module]:
        if d in self.cache:
            return self.cache[d]
        paths = self._try_paths(d)
        path = next((p for p in paths if os.path.exists(p)), None)
        if path is None:
            self.cache[d] = None
            return None

        ckpt = torch.load(path, map_location=DEVICE)
        # ckpt may be a dict with 'state_dict' and optional hidden dims
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            phi_h = ckpt.get("phi_hidden", 64)
            rho_h = ckpt.get("rho_hidden", 64)
            mdl = DeepSet(phi_dim=phi_h, rho_dim=rho_h).to(DEVICE)
            mdl.load_state_dict(ckpt["state_dict"])
        elif isinstance(ckpt, dict):
            # assume raw state_dict-compatible
            mdl = DeepSet(phi_dim=ckpt.get("phi_hidden", 64), rho_dim=ckpt.get("rho_hidden", 64)).to(DEVICE)
            mdl.load_state_dict(ckpt)
        else:
            # Unexpected format -> try to load as state_dict
            mdl = DeepSet().to(DEVICE)
            mdl.load_state_dict(ckpt)
        mdl.eval()
        self.cache[d] = mdl
        return mdl

    def predict_mu_batch(self, elements_batch: List[np.ndarray], selected_idx: Optional[np.ndarray]=None) -> np.ndarray:
        """
        elements_batch: list of (T,2) arrays per image
        selected_idx: biomarkers to predict; if None -> all D
        Returns: Yhat (N, K) predicted standardized biomarkers
        """
        if selected_idx is None:
            selected_idx = np.arange(self.D, dtype=int)
        N = len(elements_batch)
        K = len(selected_idx)
        Y = np.zeros((N, K), dtype=np.float32)
        with torch.no_grad():
            for j, d in enumerate(selected_idx):
                mdl = self._load_one(int(d))
                if mdl is None:
                    continue
                for i in range(N):
                    X = torch.tensor(elements_batch[i], dtype=torch.float32, device=DEVICE)
                    mu, _, _ = mdl(X)
                    Y[i, j] = float(mu.item())
        return Y

@st.cache_resource
def load_deepsets_bank(D: int) -> DeepSetsBank:
    return DeepSetsBank(D, MODELS_DIR)

DS_BANK = load_deepsets_bank(D_FULL)

# =========================================================
# β-WEIGHTED WELLNESS from DeepSets-reconstructed biomarkers
# =========================================================
def select_biomarkers_for_speed(beta: np.ndarray, topk: int) -> np.ndarray:
    if topk is None or topk <= 0 or topk >= len(beta):
        return np.arange(len(beta), dtype=int)
    order = np.argsort(-np.abs(beta))[:topk]
    return order.astype(int)

def compute_wellness_from_Y(Yhat: np.ndarray, beta: np.ndarray, selected_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Yhat: (N,K) predicted biomarkers (std) for K=|selected_idx|
    beta: (D,)
    selected_idx: indices of biomarkers used
    Returns: (wellness (N,), contributions (N,K)) with contrib = β * ŷ
    """
    b = beta[selected_idx].astype(np.float32)  # (K,)
    contrib = Yhat * b[None, :]
    wellness = contrib.sum(axis=1)
    return wellness, contrib

# =========================================================
# MONTE-CARLO SHAPLEY over image subsets
# f_k(S) = beta_k^T mean_Yhat(S) for Yhat from DeepSets given each image's CLM
# =========================================================
def shapley_wellness_mc_Y(Yhat: np.ndarray, beta: np.ndarray, selected_idx: np.ndarray,
                          num_perm: int = 512, seed: int = 42) -> Tuple[np.ndarray, float]:
    """
    Yhat: (N,K) for selected_idx, K = len(selected_idx)
    """
    rng = np.random.default_rng(seed)
    N, K = Yhat.shape
    b = beta[selected_idx].astype(np.float64)
    phi = np.zeros(N, dtype=np.float64)

    for _ in range(num_perm):
        order = rng.permutation(N)
        s = np.zeros(K, dtype=np.float64)
        t = 0
        for idx in order:
            y_i = Yhat[idx].astype(np.float64)
            if t == 0:
                delta = np.dot(b, y_i)
            else:
                delta = np.dot(b, ((s + y_i) / (t + 1) - s / t))
            phi[idx] += delta
            s += y_i
            t += 1

    phi /= num_perm
    f_all = float(np.dot(b.astype(np.float32), Yhat.mean(axis=0).astype(np.float32)))
    return phi.astype(np.float32), f_all

# =========================================================
# CHANNEL-LEVEL SHAP (optional, black-box) on CLM → DeepSets → β·ŷ
# =========================================================
def make_background(X: np.ndarray, max_bg: int = 20) -> np.ndarray:
    if X.shape[0] <= max_bg:
        return X.copy()
    idx = np.linspace(0, X.shape[0]-1, max_bg, dtype=int)
    return X[idx].copy()

def get_channel_explainer_union(X_bg: np.ndarray, beta: np.ndarray, sel_idx: np.ndarray, idx_union: np.ndarray):
    """
    f(X_clm) = β_sel^T yhat_sel(X_clm), with yhat from DeepSets using union cache.
    We re-run DeepSets inside this black-box; for speed, keep background small and nsamples modest.
    """
    def f(X):
        # Build elements for each row, predict over selected biomarkers
        elems_batch = [clm_to_elements(X[i]) for i in range(X.shape[0])]
        Y = DS_BANK.predict_mu_batch(elems_batch, selected_idx=sel_idx)  # (B, K_sel)
        b = beta[sel_idx].astype(np.float32)
        return (Y @ b).astype(np.float32)
    return shap.KernelExplainer(f, X_bg)

def compute_channel_shap(explainer, X: np.ndarray, nsamples="auto") -> np.ndarray:
    shap_vals = explainer.shap_values(X, nsamples=nsamples)
    return np.array(shap_vals, dtype=np.float32)

# =========================================================
# STREAMLIT APP
# =========================================================
st.title("Image → CLM → DeepSets biomarkers → β-weighted Wellness (per-image & batch, Monte-Carlo)")

uploaded_files = st.file_uploader("Upload landscape images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    # 1) CLM per image
    rows = []
    clm_list = []
    for uf in uploaded_files:
        im = Image.open(uf)
        x = preprocess_image(im)
        with torch.no_grad():
            clm_pred = CLM_MODEL(x)                 # [1,7]
            clm_pred = (clm_pred.squeeze(0) * 6.0).clamp(1.0, 6.0)
            clm_vec = clm_pred.cpu().numpy().astype(np.float32)  # (7,)
        clm_list.append(clm_vec)
        row = {"Image": uf.name}
        for i, lab in enumerate(dimension_labels):
            row[lab] = float(clm_vec[i])
        rows.append(row)

    batch_clm = np.stack(clm_list, axis=0)  # (N,7)
    N = batch_clm.shape[0]
    st.write(f"**Images uploaded:** {N}")

    # 2) Elements batch for DeepSets
    elems_batch = [clm_to_elements(batch_clm[i]) for i in range(N)]  # list of (7,2)

    # 3) Top-K biomarkers per index + union
    idx_pos = select_biomarkers_for_speed(BETAS["positive"], TOPK_BIOMARKERS)
    idx_neg = select_biomarkers_for_speed(BETAS["negative"], TOPK_BIOMARKERS)
    idx_r   = select_biomarkers_for_speed(BETAS["r"],        TOPK_BIOMARKERS)
    idx_union = np.unique(np.concatenate([idx_pos, idx_neg, idx_r])).astype(int)
    st.caption(f"Using {len(idx_union)} biomarkers (union across indices).")

    # 4) Predict biomarkers (DeepSets) on union
    Yhat_union = DS_BANK.predict_mu_batch(elems_batch, selected_idx=idx_union)  # (N, K_union)

    # 5) Per-image wellness per index
    wellness = {}
    contrib_store = {}
    for name, beta in BETAS.items():
        sel = idx_pos if name=="positive" else (idx_neg if name=="negative" else idx_r)
        # map sel -> columns in union
        col_idx = np.searchsorted(idx_union, sel)
        Y_sel = Yhat_union[:, col_idx]  # (N, K_sel)
        W, C = compute_wellness_from_Y(Y_sel, beta, sel)
        wellness[name] = W
        contrib_store[name] = (C, sel)

    # 6) Per-image table
    df = pd.DataFrame(rows)
    df["wellness_positive"] = wellness["positive"]
    df["wellness_negative"] = wellness["negative"]
    df["wellness_r"]        = wellness["r"]
    st.subheader("Per-Image CLM + β-weighted Wellness")
    st.dataframe(df, use_container_width=True)
    st.download_button("Download per-image table (CSV)", df.to_csv(index=False), "image_wellness.csv", "text/csv")

    # 7) Batch summaries
    st.subheader("Batch wellness (means)")
    st.write("- positive = {:.3f} | negative = {:.3f} | r = {:.3f}".format(
        float(np.mean(wellness["positive"])),
        float(np.mean(wellness["negative"])),
        float(np.mean(wellness["r"]))
    ))

    df["wellness_positive_z"] = zscore(wellness["positive"])
    df["wellness_negative_z"] = zscore(wellness["negative"])
    df["wellness_r_z"]        = zscore(wellness["r"])

    # 8) Monte-Carlo Shapley across image subsets
    if RUN_SHAPLEY:
        st.markdown("---")
        st.subheader("Monte-Carlo Shapley wellness across image subsets")

        phi_batch = {}
        f_all_batch = {}
        for name, beta in BETAS.items():
            sel = idx_pos if name=="positive" else (idx_neg if name=="negative" else idx_r)
            col_idx = np.searchsorted(idx_union, sel)
            Y_sel = Yhat_union[:, col_idx]  # (N, K_sel)
            phi, f_all = shapley_wellness_mc_Y(Y_sel, beta, sel, num_perm=NUM_PERM, seed=SHAPLEY_SEED)
            phi_batch[name]  = phi
            f_all_batch[name]= f_all

        df["wellness_positive_shap"] = phi_batch["positive"]
        df["wellness_negative_shap"] = phi_batch["negative"]
        df["wellness_r_shap"]        = phi_batch["r"]

        st.write(f"**Grand (whole-batch) wellness** — positive: {f_all_batch['positive']:.3f} | "
                 f"negative: {f_all_batch['negative']:.3f} | r: {f_all_batch['r']:.3f}")

        eff = (f"sum φ_pos={df['wellness_positive_shap'].sum():.3f} vs f_all={f_all_batch['positive']:.3f}; "
               f"sum φ_neg={df['wellness_negative_shap'].sum():.3f} vs f_all={f_all_batch['negative']:.3f}; "
               f"sum φ_r={df['wellness_r_shap'].sum():.3f} vs f_all={f_all_batch['r']:.3f}")
        st.caption("Shapley efficiency check: " + eff)

        st.download_button("Download per-image table (+Shapley) (CSV)",
                           df.to_csv(index=False), "image_wellness_shapley.csv", "text/csv")

    # 9) Per-image top biomarker contributions
    st.markdown("---")
    st.subheader("Per-image top biomarker contributions (β·ŷ)")
    sel_img = st.number_input("Select image row (0-based)", min_value=0, max_value=N-1, value=0, step=1)
    img_name = df.loc[sel_img, "Image"]
    for name in ["positive", "negative", "r"]:
        C, sel_idx = contrib_store[name]  # (N,K_sel), (K_sel,)
        rowC = C[sel_img]
        top = pd.DataFrame({"biomarker_index": sel_idx, "contribution": rowC}) \
              .sort_values("contribution", key=np.abs, ascending=False) \
              .head(25)
        st.write(f"**{name.capitalize()} — top biomarker contributions for {img_name}**")
        st.dataframe(top, use_container_width=True)

    # 10) Optional channel-level SHAP (CLM → DeepSets → β·ŷ)
    if RUN_CHANNEL_SHAP:
        st.markdown("---")
        st.subheader("Channel-level SHAP for end-to-end wellness")

        # parse nsamples
        try:
            nsamples = int(SHAP_NSAMPLES_TXT)
        except:
            nsamples = "auto"

        X_bg = make_background(batch_clm, max_bg=int(min(SHAP_BG_MAX, N)))

        for name, beta in BETAS.items():
            sel = idx_pos if name=="positive" else (idx_neg if name=="negative" else idx_r)
            explainer = get_channel_explainer_union(X_bg, beta, sel, idx_union)
            shap_vals = compute_channel_shap(explainer, batch_clm, nsamples=nsamples)  # (N,7)

            st.markdown(f"#### {name.capitalize()} — per-image channel SHAP")
            sel_img_2 = st.number_input(f"[{name}] Select image row (0-based)",
                                        min_value=0, max_value=N-1, value=0, step=1, key=f"sel_{name}")
            vals = shap_vals[sel_img_2]
            barplot(vals, dimension_labels, title=f"{name.capitalize()} SHAP for {df.loc[sel_img_2, 'Image']}")

            st.markdown(f"#### {name.capitalize()} — batch mean channel SHAP")
            barplot(shap_vals.mean(axis=0), dimension_labels, f"{name.capitalize()} mean SHAP (signed)")
            barplot(np.abs(shap_vals).mean(axis=0), dimension_labels, f"{name.capitalize()} mean |SHAP|")

            # CSV
            df_sh = pd.DataFrame(shap_vals, columns=dimension_labels)
            df_sh.insert(0, "Image", df["Image"])
            st.download_button(
                f"Download {name} channel SHAP per image (CSV)",
                df_sh.to_csv(index=False),
                file_name=f"channel_shap_{name}.csv",
                mime="text/csv"
            )

    st.markdown("---")
    st.success("Done. Scroll up for tables, plots and CSV downloads.")
