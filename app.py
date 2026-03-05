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

# =========================================================
# CONFIG
# =========================================================
HF_REPO = "DOReilly2/swin_regressor"                  # Hugging Face repo for CLM extractor
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS_DIR  = "models"                                # DeepSets biomarker models
WEIGHTS_DIR = "weights"                               # indices & betas (.npy); optional clm_scaler.pkl
CLM_SCALER_PATH = "env_scaler.pkl"  # optional {'mu','std'} for 7-D CLM

# Sidebar controls
st.sidebar.header("Monte-Carlo Shapley Controls")
NUM_PERM      = st.sidebar.number_input("Permutations", min_value=32, max_value=5000, value=512, step=32)
SHAPLEY_SEED  = st.sidebar.number_input("Random seed", min_value=0, max_value=10**6, value=42, step=1)

st.sidebar.header("Biomarker Loading")
SHOW_PROGRESS = st.sidebar.checkbox("Show per-biomarker loading progress", value=False)

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
# UTILS
# =========================================================
def zscore(v: np.ndarray) -> np.ndarray:
    m = np.mean(v)
    s = np.std(v) + 1e-8
    return (v - m) / s

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

# =========================================================
# 1) CLM EXTRACTOR
# =========================================================
@st.cache_resource
def load_context_embeddings():
    emb = {}
    for label in dimension_labels:
        p = hf_hub_download(repo_id=HF_REPO, filename=f"context_embeddings/{label}.pt")
        emb_t = torch.load(p, map_location="cpu")
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
        image_feat = self.swin(image)  # [B,1024]
        outs = []
        for label in self.context_embeddings:
            ctx = self.context_embeddings[label].expand(image_feat.size(0), -1)
            fused = torch.cat([image_feat, ctx], dim=1)
            score = self.fusion_headsfused  # [B,1]
            outs.append(score)
        return torch.cat(outs, dim=1)     # [B,7]

@st.cache_resource
def load_clm_model():
    path = hf_hub_download(repo_id=HF_REPO, filename="swin_regressor.pt")
    sd = torch.load(path, map_location="cpu")
    m = MultiContextSwinRegressor(CONTEXT_EMBEDDINGS)
    m.load_state_dict(sd, strict=True)
    m.to(DEVICE).eval()
    st.info("CLM model loaded.")
    return m

CLM_MODEL = load_clm_model()

def preprocess_image(img: Image.Image) -> torch.Tensor:
    tfm = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    img = img.convert("RGB")
    return tfm(img).unsqueeze(0).to(DEVICE)

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
# 2) LOAD AFFECT SPECS (indices + betas), with 1→0 index shift
# =========================================================
@st.cache_resource
def load_affect_specs(weights_dir: str):
    out = {}
    for name in ["positive", "negative", "r"]:
        idx = np.load(os.path.join(weights_dir, f"{name}_indices.npy")).astype(int)
        betas = np.load(os.path.join(weights_dir, f"{name}_betas.npy")).astype(np.float32)
        if len(idx) != len(betas):
            raise ValueError(f"{name}: indices and betas length mismatch ({len(idx)} vs {len(betas)})")
        idx = idx - 1  # 1-based -> 0-based
        out[name] = (idx, betas)
    return out

AFFECT_SPECS = load_affect_specs(WEIGHTS_DIR)

# =========================================================
# 3) DEEPSETS LOADER (original biomarker models)
# =========================================================
class DeepSet(nn.Module):
    def __init__(self, input_dim=2, phi_dim=64, rho_dim=64):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(input_dim, phi_dim), nn.ReLU(),
            nn.Linear(phi_dim, phi_dim), nn.ReLU()
        )
        self.attn = nn.Sequential(nn.Linear(phi_dim, 1), nn.Sigmoid())
        self.rho = nn.Sequential(
            nn.Linear(phi_dim, rho_dim), nn.ReLU(),
            nn.Linear(rho_dim, 2)  # [mu, log_var]
        )
    def forward(self, x: torch.Tensor):
        z = self.phi(x)                 # (T,phi)
        w = self.attn(z)                # (T,1)
        agg = (w * z).sum(dim=0)        # (phi,)
        out = self.rho(agg)             # (2,)
        mu = out[0]
        log_var = torch.clamp(out[1], min=-3.0, max=3.0)
        return mu, log_var, agg

class DeepSetsBank:
    """
    Lazy-load DeepSets per biomarker:
      models/model_{d:04d}.pt/.pkl  or  models/deepset_biomarker_{d}.pt/.pkl
    """
    def __init__(self, models_dir: str, show_progress: bool=False):
        self.dir = models_dir
        self.cache: Dict[int, Optional[nn.Module]] = {}
        self.show_progress = show_progress

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
        # Try standard formats
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            phi_h = ckpt.get("phi_hidden", 64)
            rho_h = ckpt.get("rho_hidden", 64)
            mdl = DeepSet(phi_dim=phi_h, rho_dim=rho_h).to(DEVICE)
            mdl.load_state_dict(ckpt["state_dict"])
        elif isinstance(ckpt, dict):
            mdl = DeepSet(phi_dim=ckpt.get("phi_hidden", 64), rho_dim=ckpt.get("rho_hidden", 64)).to(DEVICE)
            mdl.load_state_dict(ckpt)
        else:
            mdl = DeepSet().to(DEVICE)
            mdl.load_state_dict(ckpt)
        mdl.eval()
        self.cache[d] = mdl
        return mdl

    def predict_mu_batch(self, elements_batch: List[np.ndarray], selected_idx: np.ndarray) -> np.ndarray:
        """
        elements_batch: list of (7,2) arrays per image
        selected_idx: biomarkers to predict
        Returns: Yhat (N, K) standardized biomarkers
        """
        N = len(elements_batch)
        K = len(selected_idx)
        Y = np.zeros((N, K), dtype=np.float32)
        iterator = enumerate(selected_idx)
        if self.show_progress:
            iterator = stqdm(iterator, total=K, desc="Loading DeepSets & predicting")
        with torch.no_grad():
            for j, d in iterator:
                mdl = self._load_one(int(d))
                if mdl is None:
                    continue
                for i in range(N):
                    X = torch.tensor(elements_batch[i], dtype=torch.float32, device=DEVICE)
                    mu, _, _ = mdl(X)
                    Y[i, j] = float(mu.item())
        return Y

# a tiny progress wrapper if user wants (fallback if stqdm not installed)
def stqdm(it, total=None, desc=""):
    import itertools, time
    count = 0
    ph = st.empty()
    for x in it:
        count += 1
        if total:
            ph.info(f"{desc}: {count}/{total}")
        yield x
    ph.empty()

@st.cache_resource
def load_deepsets_bank(models_dir: str, show_progress: bool=False) -> DeepSetsBank:
    return DeepSetsBank(models_dir, show_progress=show_progress)

DS_BANK = load_deepsets_bank(MODELS_DIR, SHOW_PROGRESS)

# =========================================================
# 4) MONTE-CARLO (Option B): per-image Shapley on subset wellness
#     Using per-image biomarkers first, then subset mean → β-weighted
# =========================================================
def mc_shapley_optionB(
    Yhat_sel: np.ndarray, betas_sel: np.ndarray, num_perm: int = 512, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Yhat_sel: (N, K_sel) per-image biomarker preds for current affect
    betas_sel: (K_sel,) β for the same biomarkers (aligned order)
    Returns:
      phi_scalar: (N,) per-image Shapley on wellness
      phi_biom  : (N, K_sel) per-image, per-biomarker Shapley contributions
      f_all     : scalar grand wellness on whole batch
    """
    rng = np.random.default_rng(seed)
    N, K = Yhat_sel.shape
    b = betas_sel.astype(np.float64)
    phi_scalar = np.zeros(N, dtype=np.float64)
    phi_biom   = np.zeros((N, K), dtype=np.float64)

    for _ in range(num_perm):
        order = rng.permutation(N)
        s = np.zeros(K, dtype=np.float64)
        t = 0
        for idx in order:
            y_i = Yhat_sel[idx].astype(np.float64)
            if t == 0:
                diff_mean = y_i               # (K,)
            else:
                diff_mean = (s + y_i) / (t + 1) - s / t
            delta_biom = b * diff_mean        # (K,)
            phi_scalar[idx] += np.sum(delta_biom)
            phi_biom[idx]   += delta_biom
            s += y_i
            t += 1

    phi_scalar /= num_perm
    phi_biom   /= num_perm
    f_all = float(np.dot(b.astype(np.float32), Yhat_sel.mean(axis=0).astype(np.float32)))
    return phi_scalar.astype(np.float32), phi_biom.astype(np.float32), f_all

# =========================================================
# STREAMLIT APP
# =========================================================
st.title("Monte‑Carlo Wellness on Random Image Subsets (Option B) — DeepSets + β indices")

uploaded = st.file_uploader("Upload images", type=["jpg","jpeg","png"], accept_multiple_files=True)

if uploaded:
    # -- 1) CLM per image
    rows = []
    clm_list = []
    for uf in uploaded:
        img = Image.open(uf)
        x = preprocess_image(img)
        with torch.no_grad():
            clm_pred = CLM_MODEL(x)                 # [1,7]
            clm_pred = (clm_pred.squeeze(0) * 6.0).clamp(1.0, 6.0)
            clm = clm_pred.cpu().numpy().astype(np.float32)
        rows.append({"Image": uf.name, **{dimension_labels[i]: float(clm[i]) for i in range(7)}})
        clm_list.append(clm)

    batch_clm = np.stack(clm_list, axis=0)  # (N,7)
    N = batch_clm.shape[0]
    st.write(f"**Images:** {N}")

    # -- 2) Build elements per image (7x2) for DeepSets
    elems_batch = [clm_to_elements(batch_clm[i]) for i in range(N)]

    # -- 3) Load affect specs (indices & betas), create union for a single DeepSets pass
    idx_pos, betas_pos = AFFECT_SPECS["positive"]
    idx_neg, betas_neg = AFFECT_SPECS["negative"]
    idx_r,   betas_r   = AFFECT_SPECS["r"]

    idx_union = np.unique(np.concatenate([idx_pos, idx_neg, idx_r])).astype(int)
    st.caption(f"DeepSets will load {len(idx_union)} biomarker models (union across affects).")

    # -- 4) Predict biomarkers on the union set once
    Yhat_union = DS_BANK.predict_mu_batch(elems_batch, selected_idx=idx_union)  # (N, K_union)

    # Helpers to slice per affect
    def slice_affect(Y_union, idx_union, idx_affect, betas_affect):
        # map affect indices into union columns
        col = np.searchsorted(idx_union, idx_affect)
        return Y_union[:, col], betas_affect  # (N,K_sel), (K_sel,)

    # -- 5) If N == 1, do direct wellness (single image case)
    df = pd.DataFrame(rows)
    if N == 1:
        st.subheader("Single image detected → Direct DeepSets → β wellness")
        # Positive
        Yp, bp = slice_affect(Yhat_union, idx_union, idx_pos, betas_pos)
        Wp = float(np.dot(Yp[0], bp))
        Cp = Yp[0] * bp
        # Negative
        Yn, bn = slice_affect(Yhat_union, idx_union, idx_neg, betas_neg)
        Wn = float(np.dot(Yn[0], bn))
        Cn = Yn[0] * bn
        # r
        Yr, br = slice_affect(Yhat_union, idx_union, idx_r, betas_r)
        Wr = float(np.dot(Yr[0], br))
        Cr = Yr[0] * br

        df["wellness_positive"] = [Wp]
        df["wellness_negative"] = [Wn]
        df["wellness_r"]        = [Wr]
        st.dataframe(df, use_container_width=True)

        # Show top biomarker contributions
        def top_contrib(contrib, idx_affect, title, k=25):
            tdf = pd.DataFrame({"biomarker_index": idx_affect, "contribution": contrib}) \
                    .sort_values("contribution", key=np.abs, ascending=False).head(k)
            st.write(title)
            st.dataframe(tdf, use_container_width=True)

        top_contrib(Cp, idx_pos, "**Positive — top biomarker contributions**")
        top_contrib(Cn, idx_neg, "**Negative — top biomarker contributions**")
        top_contrib(Cr, idx_r,   "**r — top biomarker contributions**")

        st.download_button("Download single‑image table (CSV)", df.to_csv(index=False), "single_image_wellness.csv", "text/csv")
        st.stop()

    # -- 6) N > 1  →  Monte‑Carlo Shapley across images (subset‑based)
    st.subheader("Subset‑based per‑image wellness via Monte‑Carlo Shapley")

    # Build Y per affect
    Y_pos, b_pos = slice_affect(Yhat_union, idx_union, idx_pos, betas_pos)
    Y_neg, b_neg = slice_affect(Yhat_union, idx_union, idx_neg, betas_neg)
    Y_r,   b_r   = slice_affect(Yhat_union, idx_union, idx_r,   betas_r)

    # Run MC for each affect (Option B)
    phi_pos, psi_pos, F_pos = mc_shapley_optionB(Y_pos, b_pos, num_perm=NUM_PERM, seed=SHAPLEY_SEED)
    phi_neg, psi_neg, F_neg = mc_shapley_optionB(Y_neg, b_neg, num_perm=NUM_PERM, seed=SHAPLEY_SEED)
    phi_r,   psi_r,   F_r   = mc_shapley_optionB(Y_r,   b_r,   num_perm=NUM_PERM, seed=SHAPLEY_SEED)

    # Build output table
    df["wellness_positive_shap"] = phi_pos
    df["wellness_negative_shap"] = phi_neg
    df["wellness_r_shap"]        = phi_r

    st.dataframe(df, use_container_width=True)

    st.write(f"**Grand (whole‑batch) wellness** — positive: {F_pos:.3f} | negative: {F_neg:.3f} | r: {F_r:.3f}")
    eff = (f"sum φ_pos={df['wellness_positive_shap'].sum():.3f} vs f_all={F_pos:.3f}; "
           f"sum φ_neg={df['wellness_negative_shap'].sum():.3f} vs f_all={F_neg:.3f}; "
           f"sum φ_r={df['wellness_r_shap'].sum():.3f} vs f_all={F_r:.3f}")
    st.caption("Shapley efficiency check: " + eff)

    # z-scored Shapley wellness (within batch)
    df["wellness_positive_shap_z"] = zscore(phi_pos)
    df["wellness_negative_shap_z"] = zscore(phi_neg)
    df["wellness_r_shap_z"]        = zscore(phi_r)

    st.download_button("Download per‑image Shapley table (CSV)",
                       df.to_csv(index=False), "image_wellness_shapley.csv", "text/csv")

    # -- 7) Per‑image per‑biomarker Shapley contributions
    st.markdown("---")
    st.subheader("Per‑image per‑biomarker Shapley contributions (β·Δmean biomarker)")

    sel_img = st.number_input("Select image row (0‑based)", min_value=0, max_value=N-1, value=0, step=1)
    img_name = df.loc[sel_img, "Image"]

    def show_top_biomarker_shapley(psi_img: np.ndarray, idx_affect: np.ndarray, title: str, k: int = 25):
        # psi_img: (K_sel,) per‑biomarker Shapley for selected image
        tdf = pd.DataFrame({
            "biomarker_index": idx_affect,
            "shapley_contribution": psi_img
        }).sort_values("shapley_contribution", key=np.abs, ascending=False).head(k)
        st.write(f"{title} for **{img_name}**")
        st.dataframe(tdf, use_container_width=True)

    show_top_biomarker_shapley(psi_pos[sel_img], idx_pos, "**Positive**")
    show_top_biomarker_shapley(psi_neg[sel_img], idx_neg, "**Negative**")
    show_top_biomarker_shapley(psi_r[sel_img],   idx_r,   "**r**")

    # -- 8) Batch‑level biomarker importance (mean |Shapley|)
    st.markdown("#### Batch biomarker importance (mean |Shapley|)")
    def batch_biom_importance(psi: np.ndarray, idx_affect: np.ndarray, title: str, k: int = 25):
        mabs = np.abs(psi).mean(axis=0)     # (K_sel,)
        tdf = pd.DataFrame({
            "biomarker_index": idx_affect,
            "mean_abs_shapley": mabs
        }).sort_values("mean_abs_shapley", ascending=False).head(k)
        st.write(title)
        st.dataframe(tdf, use_container_width=True)

    batch_biom_importance(psi_pos, idx_pos, "**Positive — top biomarkers by mean |Shapley|**")
    batch_biom_importance(psi_neg, idx_neg, "**Negative — top biomarkers by mean |Shapley|**")
    batch_biom_importance(psi_r,   idx_r,   "**r — top biomarkers by mean |Shapley|**")
