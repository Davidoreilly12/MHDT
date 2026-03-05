import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import swin_v2_b, Swin_V2_B_Weights
from PIL import Image
from huggingface_hub import hf_hub_download

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import random

# ---------------- CONFIG ----------------
HF_REPO = "DOReilly2/swin_regressor"   # Hugging Face repo
DEVICE = "cpu"                         # set "cuda" if available
WEIGHTS_DIR = "/content/drive/MyDrive/saved_models_std/"  # folder for alpha_pos/neg/r.npy

# affect SHAP settings (channel-level on 7D CLM)
SHAP_BG_MAX = 20           # background size
SHAP_NSAMPLES = "auto"     # or an int like 200 to speed up

dimension_labels = [
    "Layers of the Landscape_embedding",
    "Landform_embedding",
    "Biodiversity_embedding",
    "Color and Light_embedding",
    "Compatibility_embedding",
    "Archetypal Elements_embedding",
    "Character of Peace and Silence_embedding"
]

# ---------------- UTILITIES ----------------
def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Convert PIL image to normalized tensor for model."""
    val_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = image.convert("RGB")
    return val_transform(image).unsqueeze(0).to(DEVICE)

@st.cache_resource
def load_context_embeddings():
    embeddings = {}
    for label in dimension_labels:
        path = hf_hub_download(repo_id=HF_REPO, filename=f"context_embeddings/{label}.pt")
        emb = torch.load(path, map_location="cpu")
        embeddings[label] = emb.squeeze()
    return embeddings

context_embeddings = load_context_embeddings()

# ---------------- MODEL ----------------
class MultiContextSwinRegressor(nn.Module):
    def __init__(self, context_embeddings: dict):
        super().__init__()
        # SwinV2 pretrained feature extractor
        self.swin = swin_v2_b(weights=Swin_V2_B_Weights.IMAGENET1K_V1)
        self.swin.head = nn.Identity()  # remove classifier head

        # context embeddings
        self.context_embeddings = nn.ParameterDict({
            label: nn.Parameter(context_embeddings[label].float().unsqueeze(0), requires_grad=False).squeeze(0)
            for label in context_embeddings
        })

        # per-dimension fusion heads
        self.fusion_heads = nn.ModuleDict({
            label: nn.Sequential(
                nn.Linear(1024 + self.context_embeddings[label].shape[0], 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 1)
            )
            for label in self.context_embeddings
        })

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # extract image features
        image_feat = self.swin(image)  # [B, 1024]
        outputs = []
        for label in self.context_embeddings:
            context = self.context_embeddings[label].expand(image_feat.size(0), -1)
            fused = torch.cat([image_feat, context], dim=1)
            score = self.fusion_heads[label](fused)
            outputs.append(score)
        return torch.cat(outputs, dim=1)

@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id=HF_REPO, filename="swin_regressor.pt")
    state_dict = torch.load(model_path, map_location="cpu")

    model = MultiContextSwinRegressor(context_embeddings)
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()
    st.info("Model loaded successfully (raw state_dict).")
    return model

model = load_model()

# ---------------- AFFECT WEIGHTS ----------------
@st.cache_resource
def load_affect_channel_weights():
    """
    Load 7-D alpha vectors for: positive, negative, r.
    Files must be aligned with dimension_labels order.
    """
    alpha_pos = np.load(f"{WEIGHTS_DIR}/alpha_pos.npy").astype(np.float32).reshape(-1)
    alpha_neg = np.load(f"{WEIGHTS_DIR}/alpha_neg.npy").astype(np.float32).reshape(-1)
    alpha_r   = np.load(f"{WEIGHTS_DIR}/alpha_r.npy").astype(np.float32).reshape(-1)
    if not (len(alpha_pos) == len(alpha_neg) == len(alpha_r) == len(dimension_labels)):
        raise ValueError("alpha_pos/neg/r must each be length 7 and match dimension_labels.")
    return {
        "positive": alpha_pos,
        "negative": alpha_neg,
        "r": alpha_r
    }

AFFECT_ALPHA = load_affect_channel_weights()

# ---------------- SIMPLE WELLNESS ON 7-D CLM ----------------
def compute_wellness_batch(clm_batch: np.ndarray, affect_alpha: dict) -> dict:
    """
    clm_batch: (N, 7) CLM predictions per image.
    returns: dict with three (N,) arrays for positive/negative/r
    """
    out = {}
    for k, alpha in affect_alpha.items():
        out[k] = clm_batch @ alpha  # (N,)
    return out

# ---------------- MONTE-CARLO SHAPLEY WELLNESS ACROSS IMAGE SUBSETS ----------------
def shapley_wellness_mc(clm_batch: np.ndarray,
                        alpha: np.ndarray,
                        num_perm: int = 512,
                        seed: int = 42) -> Tuple[np.ndarray, float]:
    """
    Monte-Carlo Shapley for wellness index defined as:
        f(S) = alpha^T mean({x_i | i in S}); define f(∅)=0.
    Returns:
        phi: (N,) per-image Shapley wellness
        f_all: scalar grand wellness for the whole batch (alpha^T mean(batch))
    Complexity: O(num_perm * N * 7). For N<=200, perms up to ~1k is typically fine on CPU.

    Derivation for marginal at step t (coalition size t >= 0):
        delta_i = alpha^T [ (s + x_i)/(t+1) - s/t ]  for t>0
        delta_i = alpha^T x_i                         for t=0
      where s is the running sum of CLMs in coalition.
    """
    rng = np.random.default_rng(seed)
    X = clm_batch.astype(np.float32)       # (N,7)
    N = X.shape[0]
    phi = np.zeros(N, dtype=np.float64)

    # precompute linear form for speed
    # but we still need (s/t) term; so we keep s (7,) accumulating
    for _ in range(num_perm):
        order = rng.permutation(N)
        s = np.zeros(7, dtype=np.float64)
        t = 0
        for idx in order:
            x_i = X[idx].astype(np.float64)
            if t == 0:
                delta = np.dot(alpha, x_i)
            else:
                delta = np.dot(alpha, ( (s + x_i)/(t+1) - s/t ))
            phi[idx] += delta
            s += x_i
            t += 1

    phi /= num_perm  # average over permutations
    f_all = float(np.dot(alpha, X.mean(axis=0)))
    return phi.astype(np.float32), f_all

def shapley_all_indices(clm_batch: np.ndarray,
                        affect_alpha: dict,
                        num_perm: int,
                        seed: int) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Return a dict:
      { 'positive': {'phi': (N,), 'f_all': scalar},
        'negative': {...},
        'r': {...} }
    """
    out = {}
    for k, alpha in affect_alpha.items():
        phi, f_all = shapley_wellness_mc(clm_batch, alpha, num_perm=num_perm, seed=seed)
        out[k] = {"phi": phi, "f_all": f_all}
    return out

# ---------------- CLM→Wellness Channel-SHAP (optional) ----------------
def make_background(X: np.ndarray, max_bg: int = 20) -> np.ndarray:
    if X.shape[0] <= max_bg:
        return X.copy()
    idx = np.linspace(0, X.shape[0]-1, max_bg, dtype=int)
    return X[idx].copy()

def get_kernel_explainer(X_bg: np.ndarray, alpha: np.ndarray):
    def f(X):
        return X @ alpha
    explainer = shap.KernelExplainer(f, X_bg)
    return explainer

def compute_shap_for_index(explainer, X: np.ndarray, nsamples="auto") -> np.ndarray:
    shap_vals = explainer.shap_values(X, nsamples=nsamples)
    return np.array(shap_vals, dtype=np.float32)

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

# ---------------- STREAMLIT UI ----------------
st.title("CLASS 2.0 — Image → CLM → Wellness (per-image, subset-combinations, & SHAP)")

uploaded_files = st.file_uploader(
    "Upload landscape images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

col0, col1, col2 = st.columns(3)
with col0:
    run_shapley = st.checkbox("Compute subset-combination wellness (Shapley)", value=True)
with col1:
    num_perm = st.number_input("Shapley permutations", min_value=32, max_value=5000, value=512, step=32)
with col2:
    shapley_seed = st.number_input("Shapley random seed", min_value=0, max_value=999999, value=42, step=1)

st.markdown("---")
st.subheader("Optional: channel-level SHAP (on 7 CLM channels)")
col3, col4 = st.columns(2)
with col3:
    run_channel_shap = st.checkbox("Run channel SHAP", value=False)
with col4:
    shap_bg_n = st.number_input("SHAP background size", min_value=1, max_value=200, value=SHAP_BG_MAX, step=1)
shap_nsamples_str = st.text_input("SHAP nsamples (int or 'auto')", value=str(SHAP_NSAMPLES))

if uploaded_files:
    # collect per-image predictions
    rows = []
    clm_list = []

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        image_tensor = preprocess_image(image)

        with torch.no_grad():
            predicted_scores = model(image_tensor)  # [1, 7]
            # scale & clamp as in your original code
            predicted_scores = (predicted_scores.squeeze(0) * 6.0).clamp(1.0, 6.0)
            scores = predicted_scores.cpu().numpy().astype(np.float32)  # (7,)

        clm_list.append(scores)
        row = {"Image": uploaded_file.name}
        for i, lab in enumerate(dimension_labels):
            row[lab] = float(scores[i])
        rows.append(row)

    # Build batch matrix
    batch_clm = np.stack(clm_list, axis=0)  # (N,7)
    N = batch_clm.shape[0]
    S = batch_clm.shape[1]
    assert S == len(dimension_labels) == 7

    # ----- Linear CLM→Wellness (per image) -----
    wellness_lin = compute_wellness_batch(batch_clm, AFFECT_ALPHA)  # dict of 3 arrays (N,)

    # Base per-image table
    df = pd.DataFrame(rows)
    df["wellness_positive_lin"] = wellness_lin["positive"]
    df["wellness_negative_lin"] = wellness_lin["negative"]
    df["wellness_r_lin"]        = wellness_lin["r"]

    # ----- Shapley (subset-combination) CLM→Wellness (per image & grand) -----
    if run_shapley:
        st.markdown("### Subset-combination wellness (Monte-Carlo Shapley)")
        shap_out = shapley_all_indices(batch_clm, AFFECT_ALPHA, num_perm=num_perm, seed=shapley_seed)

        # Add per-image Shapley scores
        df["wellness_positive_shap"] = shap_out["positive"]["phi"]
        df["wellness_negative_shap"] = shap_out["negative"]["phi"]
        df["wellness_r_shap"]        = shap_out["r"]["phi"]

        # Batch grand (whole) wellness per index
        grand_pos = shap_out["positive"]["f_all"]
        grand_neg = shap_out["negative"]["f_all"]
        grand_r   = shap_out["r"]["f_all"]

        st.write(f"**Grand wellness (whole batch)** — "
                 f"positive: {grand_pos:.3f}, "
                 f"negative: {grand_neg:.3f}, "
                 f"r: {grand_r:.3f}")

        # Check efficiency property: sum_i Shapley_i ≈ f(all) (numerical tolerance)
        tol_info = (
            f"sum(Shapley_pos)={df['wellness_positive_shap'].sum():.3f} vs f_all={grand_pos:.3f}; "
            f"sum(Shapley_neg)={df['wellness_negative_shap'].sum():.3f} vs f_all={grand_neg:.3f}; "
            f"sum(Shapley_r)={df['wellness_r_shap'].sum():.3f} vs f_all={grand_r:.3f}"
        )
        st.caption("Shapley efficiency check: " + tol_info)

    # ----- Present per-image table -----
    st.subheader("Per-Image Predictions (CLM + Wellness)")
    st.dataframe(df, use_container_width=True)

    # CSV (Excel-friendly) download
    csv_text = df.to_csv(index=False)
    st.download_button("Download per-image table (CSV)", csv_text, "image_clm_wellness.csv", "text/csv")

    # ----- Batch summaries -----
    st.subheader("Batch-level summaries")
    def zscore(v):
        m = np.mean(v)
        s = np.std(v) + 1e-8
        return (v - m) / s

    st.write(f"- **Images**: {N}")
    st.write("- **Linear wellness means**: "
             f"positive = {np.mean(wellness_lin['positive']):.3f}, "
             f"negative = {np.mean(wellness_lin['negative']):.3f}, "
             f"r = {np.mean(wellness_lin['r']):.3f}")
    if run_shapley:
        st.write("- **Shapley wellness means**: "
                 f"positive = {np.mean(df['wellness_positive_shap']):.3f}, "
                 f"negative = {np.mean(df['wellness_negative_shap']):.3f}, "
                 f"r = {np.mean(df['wellness_r_shap']):.3f}")

    # z-scores for linear wellness (within batch)
    df["wellness_positive_lin_z"] = zscore(wellness_lin["positive"])
    df["wellness_negative_lin_z"] = zscore(wellness_lin["negative"])
    df["wellness_r_lin_z"]        = zscore(wellness_lin["r"])

    # z-scores for shapley wellness (within batch)
    if run_shapley:
        df["wellness_positive_shap_z"] = zscore(df["wellness_positive_shap"].values)
        df["wellness_negative_shap_z"] = zscore(df["wellness_negative_shap"].values)
        df["wellness_r_shap_z"]        = zscore(df["wellness_r_shap"].values)

    # ----- Optional channel-level SHAP on 7D CLM -----
    if run_channel_shap:
        st.markdown("---")
        st.subheader("Channel-level SHAP on CLM → Wellness")

        # nsamples parse
        try:
            nsamples = int(shap_nsamples_str)
        except:
            nsamples = "auto"

        # background from this batch
        X_bg = make_background(batch_clm, max_bg=int(min(shap_bg_n, N)))

        # Build explainers
        explainers = {}
        for k in ["positive", "negative", "r"]:
            explainers[k] = get_kernel_explainer(X_bg, AFFECT_ALPHA[k])

        shap_batch = {}
        for k in ["positive", "negative", "r"]:
            shap_vals = compute_shap_for_index(explainers[k], batch_clm, nsamples=nsamples)  # (N,7)
            shap_batch[k] = shap_vals

        # Per-image SHAP channel plot
        st.markdown("#### Per-image channel contributions")
        sel_idx = st.number_input("Select image row (0-based)", min_value=0, max_value=N-1, value=0, step=1)
        img_name = df.loc[sel_idx, "Image"] if "Image" in df.columns else f"img_{sel_idx}"
        st.write(f"**Image:** {img_name}")

        for k in ["positive", "negative", "r"]:
            st.write(f"**{k.capitalize()} wellness — channel SHAP**")
            vals = shap_batch[k][sel_idx]  # (7,)
            barplot(vals, dimension_labels, title=f"{k.capitalize()} — channel SHAP")

            # show a small table
            per_img_tab = pd.DataFrame({
                "channel": dimension_labels,
                "shap": vals
            }).sort_values("shap", key=np.abs, ascending=False)
            st.dataframe(per_img_tab, use_container_width=True)

        # Batch mean SHAP
        st.markdown("#### Batch-level SHAP summaries")
        for k in ["positive", "negative", "r"]:
            vals_mean = shap_batch[k].mean(axis=0)             # signed mean
            vals_mean_abs = np.abs(shap_batch[k]).mean(axis=0) # mean |SHAP|
            st.write(f"**{k.capitalize()} — mean channel SHAP** (signed)")
            barplot(vals_mean, dimension_labels, title=f"{k.capitalize()} — mean SHAP (signed)")
            st.write(f"**{k.capitalize()} — mean |channel SHAP|**")
            barplot(vals_mean_abs, dimension_labels, title=f"{k.capitalize()} — mean |SHAP|")

        # Downloads
        st.markdown("#### Download SHAP tables")
        for k in ["positive", "negative", "r"]:
            df_shap = pd.DataFrame(shap_batch[k], columns=dimension_labels)
            df_shap.insert(0, "Image", df["Image"])
            st.download_button(
                f"Download {k} SHAP per image (CSV)",
                df_shap.to_csv(index=False),
                file_name=f"shap_{k}_per_image.csv",
                mime="text/csv"
            )

    st.markdown("---")
    st.success("Done. You can scroll up for per-image table download and SHAP exports.")
