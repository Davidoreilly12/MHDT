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
HF_REPO = "DOReilly2/swin_regressor"  # CLM extractor repo
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS_DIR = "models"                 # DeepSets biomarker models
WEIGHTS_DIR = "weights"               # indices & betas
CLM_SCALER_PATH = "env_scaler.pkl"    # optional CLM scaler {'mu','std'}

# Sidebar
st.sidebar.header("Monte-Carlo Shapley Controls")
NUM_PERM = st.sidebar.number_input("Permutations", min_value=32, max_value=5000, value=512, step=32)
SHAPLEY_SEED = st.sidebar.number_input("Random seed", min_value=0, max_value=10**6, value=42, step=1)
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
    m = np.mean(v, axis=0)
    s = np.std(v, axis=0) + 1e-8
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
            score = self.fusion_heads[label](fused)  # [B,1]
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
    x = clm_vec.astype(np.float32)
    if CLM_SCALER is not None:
        mu, sd = CLM_SCALER
        x = (x - mu) / (sd + 1e-6)
    elems = np.array([[float(x[i]), float(i)] for i in range(7)], dtype=np.float32)
    return elems

# =========================================================
# 2) LOAD AFFECT SPECS
# =========================================================
@st.cache_resource
def load_affect_specs(weights_dir: str):
    out = {}
    for name in ["positive", "negative", "r"]:
        idx = np.load(os.path.join(weights_dir, f"{name}_indices.npy")).astype(int)
        betas = np.load(os.path.join(weights_dir, f"{name}_betas.npy")).astype(np.float32)
        idx = idx - 1
        out[name] = (idx, betas)
    return out

AFFECT_SPECS = load_affect_specs(WEIGHTS_DIR)

# =========================================================
# 3) DEEPSETS
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
        z = self.phi(x)
        w = self.attn(z)
        agg = (w * z).sum(dim=0)
        out = self.rho(agg)
        mu = out[0]
        log_var = torch.clamp(out[1], min=-3.0, max=3.0)
        return mu, log_var, agg

class DeepSetsBank:
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
        if d in self.cache: return self.cache[d]
        path = next((p for p in self._try_paths(d) if os.path.exists(p)), None)
        if path is None: self.cache[d]=None; return None
        ckpt = torch.load(path, map_location=DEVICE)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            phi_h, rho_h = ckpt.get("phi_hidden",64), ckpt.get("rho_hidden",64)
            mdl = DeepSet(phi_dim=phi_h, rho_dim=rho_h).to(DEVICE)
            mdl.load_state_dict(ckpt["state_dict"])
        elif isinstance(ckpt, dict):
            mdl = DeepSet(phi_dim=ckpt.get("phi_hidden",64), rho_dim=ckpt.get("rho_hidden",64)).to(DEVICE)
            mdl.load_state_dict(ckpt)
        else:
            mdl = DeepSet().to(DEVICE)
            mdl.load_state_dict(ckpt)
        mdl.eval()
        self.cache[d] = mdl
        return mdl
    def predict_mu_sigma_batch(self, elements_batch: List[np.ndarray], selected_idx: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
        """Vectorized: returns yhat*sigma"""
        N,K = len(elements_batch), len(selected_idx)
        Yhat = np.zeros((N,K),dtype=np.float32)
        Sigma = np.zeros((N,K),dtype=np.float32)
        for j,d in enumerate(selected_idx):
            mdl = self._load_one(int(d))
            if mdl is None: continue
            for i in range(N):
                X = torch.tensor(elements_batch[i],dtype=torch.float32,device=DEVICE)
                mu, log_var, _ = mdl(X)
                sigma = torch.exp(0.5*log_var)
                Yhat[i,j] = float(mu.item())
                Sigma[i,j] = float(sigma.item())
        return Yhat,Sigma

@st.cache_resource
def load_deepsets_bank(models_dir: str, show_progress: bool=False) -> DeepSetsBank:
    return DeepSetsBank(models_dir, show_progress=show_progress)

DS_BANK = load_deepsets_bank(MODELS_DIR, SHOW_PROGRESS)

# =========================================================
# 4) MONTE CARLO SHAPLEY
# =========================================================
def mc_shapley(Yhat_z: np.ndarray, betas_sel: np.ndarray, num_perm:int=512, seed:int=42):
    rng = np.random.default_rng(seed)
    N,K = Yhat_z.shape
    b = betas_sel.astype(np.float64)
    phi_scalar = np.zeros(N,dtype=np.float64)
    phi_biom   = np.zeros((N,K),dtype=np.float64)
    for _ in range(num_perm):
        order = rng.permutation(N)
        s = np.zeros(K,dtype=np.float64)
        t = 0
        for idx in order:
            y_i = Yhat_z[idx].astype(np.float64)
            diff_mean = y_i if t==0 else (s+y_i)/(t+1) - s/t
            delta_biom = b*diff_mean
            phi_scalar[idx] += np.sum(delta_biom)
            phi_biom[idx]   += delta_biom
            s += y_i
            t += 1
    phi_scalar /= num_perm
    phi_biom   /= num_perm
    f_all = float(np.dot(b, Yhat_z.mean(axis=0)))
    return phi_scalar.astype(np.float32), phi_biom.astype(np.float32), f_all

# =========================================================
# STREAMLIT APP
# =========================================================
st.title("Monte-Carlo Wellness on Random Image Subsets — DeepSets + β indices + Z-score")

uploaded = st.file_uploader("Upload images", type=["jpg","jpeg","png"], accept_multiple_files=True)

if uploaded:
    rows = []
    clm_list = []
    for uf in uploaded:
        img = Image.open(uf)
        x = preprocess_image(img)
        with torch.no_grad():
            clm_pred = CLM_MODEL(x)                 # [1,7]
            clm_pred = (clm_pred.squeeze(0)*6.0).clamp(1.0,6.0)
            clm = clm_pred.cpu().numpy().astype(np.float32)
        rows.append({"Image":uf.name, **{dimension_labels[i]:float(clm[i]) for i in range(7)}})
        clm_list.append(clm)

    batch_clm = np.stack(clm_list,axis=0)
    N = batch_clm.shape[0]
    st.write(f"**Images:** {N}")

    elems_batch = [clm_to_elements(batch_clm[i]) for i in range(N)]

    idx_pos, betas_pos = AFFECT_SPECS["positive"]
    idx_neg, betas_neg = AFFECT_SPECS["negative"]
    idx_r,   betas_r   = AFFECT_SPECS["r"]

    idx_union = np.unique(np.concatenate([idx_pos,idx_neg,idx_r]))
    st.caption(f"DeepSets will load {len(idx_union)} biomarker models (union across affects).")

    Yhat_union,Sigma_union = DS_BANK.predict_mu_sigma_batch(elems_batch, idx_union)
    Yhat_sigma_union = Yhat_union * Sigma_union           # per-image biomarker estimates
    Yhat_z_union = zscore(Yhat_sigma_union)              # z-score across batch

    def slice_affect(Y,zscore_union, idx_union, idx_affect, betas_affect):
        cols = np.searchsorted(idx_union, idx_affect)
        return Y[:,cols], Yhat_z_union[:,cols], betas_affect

    df = pd.DataFrame(rows)
    # Single image case
    if N==1:
        st.subheader("Single image → direct wellness")
        Yp, Yp_z, bp = slice_affect(Yhat_sigma_union,Yhat_z_union, idx_union, idx_pos, betas_pos)
        Yn, Yn_z, bn = slice_affect(Yhat_sigma_union,Yhat_z_union, idx_union, idx_neg, betas_neg)
        Yr, Yr_z, br = slice_affect(Yhat_sigma_union,Yhat_z_union, idx_union, idx_r, betas_r)
        Wp = float(np.dot(Yp_z[0],bp)); Cp = Yp_z[0]*bp
        Wn = float(np.dot(Yn_z[0],bn)); Cn = Yn_z[0]*bn
        Wr = float(np.dot(Yr_z[0],br)); Cr = Yr_z[0]*br
        df["wellness_positive"] = [Wp]
        df["wellness_negative"] = [Wn]
        df["wellness_r"] = [Wr]
        st.dataframe(df,use_container_width=True)
        def show_top(contrib,idx_aff,title):
            tdf = pd.DataFrame({"biomarker_index":idx_aff,"contribution":contrib}).sort_values("contribution",key=np.abs,ascending=False)
            st.write(title); st.dataframe(tdf,use_container_width=True)
        show_top(Cp,idx_pos,"**Positive**")
        show_top(Cn,idx_neg,"**Negative**")
        show_top(Cr,idx_r,"**r**")
        st.download_button("Download CSV",df.to_csv(index=False),"single_image.csv","text/csv")
        st.stop()

    # N>1 case
    st.subheader("Monte Carlo Shapley per-image wellness")
    Y_pos,Y_pos_z,b_pos = slice_affect(Yhat_sigma_union,Yhat_z_union, idx_union, idx_pos, betas_pos)
    Y_neg,Y_neg_z,b_neg = slice_affect(Yhat_sigma_union,Yhat_z_union, idx_union, idx_neg, betas_neg)
    Y_r,Y_r_z,b_r       = slice_affect(Yhat_sigma_union,Yhat_z_union, idx_union, idx_r, betas_r)

    phi_pos, psi_pos, F_pos = mc_shapley(Y_pos_z,b_pos,NUM_PERM,SHAPLEY_SEED)
    phi_neg, psi_neg, F_neg = mc_shapley(Y_neg_z,b_neg,NUM_PERM,SHAPLEY_SEED)
    phi_r, psi_r, F_r       = mc_shapley(Y_r_z,b_r,NUM_PERM,SHAPLEY_SEED)

    df["wellness_positive_shap"] = phi_pos
    df["wellness_negative_shap"] = phi_neg
    df["wellness_r_shap"] = phi_r
    st.dataframe(df,use_container_width=True)
    st.write(f"**Grand wellness** — positive: {F_pos:.3f} | negative: {F_neg:.3f} | r: {F_r:.3f}")

    df["wellness_positive_shap_z"] = zscore(phi_pos)
    df["wellness_negative_shap_z"] = zscore(phi_neg)
    df["wellness_r_shap_z"] = zscore(phi_r)

    st.download_button("Download per-image Shapley table", df.to_csv(index=False),"image_wellness.csv","text/csv")

    # Top biomarker contributions per image
    sel_img = st.number_input("Select image row (0-based)",0,N-1,0,1)
    img_name = df.loc[sel_img,"Image"]
    def show_top_biomarker(psi_img,idx_aff,title):
        tdf = pd.DataFrame({"biomarker_index":idx_aff,"shapley_contribution":psi_img}).sort_values("shapley_contribution",key=np.abs,ascending=False)
        st.write(f"{title} for **{img_name}**"); st.dataframe(tdf,use_container_width=True)
    show_top_biomarker(psi_pos[sel_img],idx_pos,"**Positive**")
    show_top_biomarker(psi_neg[sel_img],idx_neg,"**Negative**")
    show_top_biomarker(psi_r[sel_img],idx_r,"**r**")

    # Batch-level biomarker importance
    st.markdown("#### Batch biomarker importance (mean |Shapley|)")
    def batch_biom_importance(psi,idx_aff,title):
        mabs = np.abs(psi).mean(axis=0)
        tdf = pd.DataFrame({"biomarker_index":idx_aff,"mean_abs_shapley":mabs}).sort_values("mean_abs_shapley",ascending=False)
        st.write(title); st.dataframe(tdf,use_container_width=True)
    batch_biom_importance(psi_pos,idx_pos,"**Positive — top biomarkers**")
    batch_biom_importance(psi_neg,idx_neg,"**Negative — top biomarkers**")
    batch_biom_importance(psi_r,idx_r,"**r — top biomarkers**")
