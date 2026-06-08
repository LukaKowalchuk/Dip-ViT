import os
import json
import base64
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
import open_clip
from peft import LoraConfig, get_peft_model
from PIL import Image
from sklearn.neighbors import NearestNeighbors

import streamlit as st
import plotly.graph_objects as go
from sklearn.manifold import TSNE
import umap
import cv2
import matplotlib.cm as cm

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path("set to directory/app_files")

LOGO_UVA_PATH = BASE_DIR / "uvalogo_regular_p_en.jpg"
LOGO_WFA_PATH = BASE_DIR / "WFA-logo.webp"

CHECKPOINTS = {
    ("BioCLIP-1", "flies"):   BASE_DIR / "BioCLIP-1_lora_best_flies.pt",
    ("BioCLIP-1", "maggots"): BASE_DIR / "BioCLIP-1_lora_best_maggots.pt",
    ("BioCLIP-2", "flies"):   BASE_DIR / "BioCLIP-2_lora_best_flies.pt",
    ("BioCLIP-2", "maggots"): BASE_DIR / "BioCLIP-2_lora_best_maggots.pt",
}

TRAIN_EMBEDDINGS = {
    ("BioCLIP-1", "flies"):   BASE_DIR / "train_embeddings_BioCLIP-1_flies.npy",
    ("BioCLIP-1", "maggots"): BASE_DIR / "train_embeddings_BioCLIP-1_maggots.npy",
    ("BioCLIP-2", "flies"):   BASE_DIR / "train_embeddings_BioCLIP-2_flies.npy",
    ("BioCLIP-2", "maggots"): BASE_DIR / "train_embeddings_BioCLIP-2_maggots.npy",
}

TRAIN_LABELS = {
    ("BioCLIP-1", "flies"):   BASE_DIR / "train_labels_BioCLIP-1_flies.npy",
    ("BioCLIP-1", "maggots"): BASE_DIR / "train_labels_BioCLIP-1_maggots.npy",
    ("BioCLIP-2", "flies"):   BASE_DIR / "train_labels_BioCLIP-2_flies.npy",
    ("BioCLIP-2", "maggots"): BASE_DIR / "train_labels_BioCLIP-2_maggots.npy",
}

CLASS_NAMES = {
    ("BioCLIP-1", "flies"):   BASE_DIR / "class_names_BioCLIP-1_flies.json",
    ("BioCLIP-1", "maggots"): BASE_DIR / "class_names_BioCLIP-1_maggots.json",
    ("BioCLIP-2", "flies"):   BASE_DIR / "class_names_BioCLIP-2_flies.json",
    ("BioCLIP-2", "maggots"): BASE_DIR / "class_names_BioCLIP-2_maggots.json",
}

THRESHOLDS = {
    ("BioCLIP-1", "flies"):   0.2110, ("BioCLIP-1", "maggots"): 0.2355,
    ("BioCLIP-2", "flies"):   0.1124, ("BioCLIP-2", "maggots"):  0.1184,
}

LORA_CONFIG = LoraConfig(r=8, lora_alpha=16, target_modules=["out_proj", "c_fc", "c_proj"], lora_dropout=0.1, bias="none")
HF_MODELS = {"BioCLIP-1": "hf-hub:imageomics/bioclip", "BioCLIP-2": "hf-hub:imageomics/bioclip-2"}

# ============================================================
# LOGO ENCODING
# ============================================================
def get_base64_logo(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None

uva_b64 = get_base64_logo(LOGO_UVA_PATH)
wfa_b64 = get_base64_logo(LOGO_WFA_PATH)

# ============================================================
# INTERFACE STYLING (BEIGE & GREEN)
# ============================================================
st.set_page_config(page_title="Forensic Entomology Classifier", page_icon="🪰", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Fraunces:ital,wght@0,300;0,600;1,300&display=swap');

.stApp {
    background-color: #fdfcf0;
    color: #3e4a36;
}

section[data-testid="stSidebar"] {
    background-color: #e8ede1;
    border-right: 1px solid #ccd4c1;
}

html, body, [class*="css"] { font-family: 'DM Mono', monospace; }
h1, h2, h3 { font-family: 'Fraunces', serif; font-weight: 600; color: #2c3627; }

.result-box {
    border: 1px solid #c2ccb8;
    border-radius: 12px;
    padding: 28px;
    background: #ffffff;
    margin-top: 16px;
    box-shadow: 0 4px 12px rgba(62, 74, 54, 0.08);
}

.result-known { border-left: 10px solid #77966d; }
.result-unknown { border-left: 10px solid #b56b6b; }

.species-name { font-size: 2rem; font-weight: 700; color: #2c3627; margin: 0; }
.label-grey { font-size: 0.85rem; color: #7a8574; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 15px; }
.value-highlight { font-size: 1.4rem; color: #5a7352; font-weight: bold; }

.tag {
    display: inline-block; padding: 6px 14px; border-radius: 6px;
    font-size: 0.8rem; font-weight: bold; text-transform: uppercase; margin-top: 12px;
}
.tag-known { background: #f0f7ed; color: #4b6343; }
.tag-unknown { background: #fff0f0; color: #914242; }

hr { border: 0; border-top: 1px solid #ccd4c1; margin: 25px 0; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# ATTENTION ROLLOUT
# ============================================================
def get_attention_rollout(model, image_tensor, device, discard_ratio=0.9):
    hooks = []

    for block in model.visual.transformer.resblocks:
        original_forward = block.attn.forward
        store = {}

        def make_hook(s, orig):
            def patched(query, key, value, **kwargs):
                kwargs['need_weights'] = True
                kwargs['average_attn_weights'] = False
                out, weights = orig(query, key, value, **kwargs)
                s['weights'] = weights
                return out, weights
            return patched

        block.attn.forward = make_hook(store, original_forward)
        hooks.append((block, original_forward, store))

    model.eval()
    with torch.no_grad():
        _ = model.encode_image(image_tensor.unsqueeze(0).to(device))

    all_attentions = []
    for block, orig_fwd, store in hooks:
        block.attn.forward = orig_fwd
        if 'weights' in store:
            attn = store['weights'].squeeze(0).mean(0).cpu().numpy()
            all_attentions.append(attn)

    if len(all_attentions) == 0:
        raise RuntimeError("No attention weights captured")

    result = np.eye(all_attentions[0].shape[0])
    for attn in all_attentions:
        flat      = attn.flatten()
        threshold = np.percentile(flat, discard_ratio * 100)
        attn      = np.where(attn >= threshold, attn, 0)
        attn      = attn / (attn.sum(axis=-1, keepdims=True) + 1e-8)
        attn_adj  = 0.5 * attn + 0.5 * np.eye(attn.shape[0])
        result    = attn_adj @ result

    cls_attn    = result[0, 1:]
    num_patches = cls_attn.shape[0]
    grid_size   = int(num_patches ** 0.5)
    attn_map    = cls_attn.reshape(grid_size, grid_size)
    attn_map    = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    return attn_map  # shape (grid, grid), float [0,1]


def render_heatmap_overlay(pil_img, attn_map, alpha=0.5):
    """
    Blend jet heatmap over the image at the same 224x224 resolution
    the model saw, then scale back up to the original display size.
    """
    # Step 1: centre-crop + resize to 224x224 to match what the model saw
    img_224 = np.array(pil_img.resize((224, 224), Image.LANCZOS)).astype(float) / 255.0

    # Step 2: upsample attention map to 224x224
    heatmap_224 = cv2.resize(attn_map, (224, 224), interpolation=cv2.INTER_LINEAR)

    # Step 3: apply jet colormap and blend
    colored = cm.jet(heatmap_224)[:, :, :3]                          # (224,224,3) float [0,1]
    blended = np.clip(alpha * colored + (1 - alpha) * img_224, 0, 1) # (224,224,3)

    # Step 4: scale back up to original image size for display
    orig_w, orig_h = pil_img.size
    blended_uint8 = (blended * 255).astype(np.uint8)
    result = Image.fromarray(blended_uint8).resize((orig_w, orig_h), Image.LANCZOS)
    return result


# ============================================================
# HELPERS
# ============================================================
@st.cache_resource
def load_model(version, mode):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(HF_MODELS[version])
    model = get_peft_model(model, LORA_CONFIG)
    model.load_state_dict(torch.load(CHECKPOINTS[(version, mode)], map_location=device))
    return model.to(device).eval(), preprocess, device

@st.cache_resource
def load_reference(version, mode):
    emb, lbl = np.load(TRAIN_EMBEDDINGS[(version, mode)]), np.load(TRAIN_LABELS[(version, mode)])
    with open(CLASS_NAMES[(version, mode)]) as f: names = json.load(f)
    knn = NearestNeighbors(n_neighbors=5, metric="cosine").fit(emb)
    return knn, emb, lbl, names

def make_projection_plot(train_emb, train_labels, query_emb, query_class, class_names, method="UMAP"):
    all_emb = np.vstack([train_emb, query_emb[np.newaxis, :]])
    if method == "UMAP":
        emb_2d = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(all_emb)
    else:
        emb_2d = TSNE(n_components=2, perplexity=min(30, len(train_emb)-1), random_state=42, max_iter=500).fit_transform(all_emb)

    train_2d, query_2d = emb_2d[:-1, :], emb_2d[-1, :]

    colors = ["#77966d", "#a39171", "#5d6d7e", "#9b59b6", "#e67e22", "#16a085", "#2980b9", "#7f8c8d"]
    fig = go.Figure()

    for i, lbl in enumerate(sorted(set(train_labels))):
        mask = train_labels == lbl
        fig.add_trace(go.Scatter(x=train_2d[mask, 0], y=train_2d[mask, 1], mode="markers",
                                 name=class_names[lbl], marker=dict(color=colors[i % len(colors)], size=7, opacity=0.7)))

    fig.add_trace(go.Scatter(x=[query_2d[0]], y=[query_2d[1]], mode="markers", name="Uploaded Specimen",
                             marker=dict(color="#2c3627", size=16, symbol="star", line=dict(color="white", width=2))))

    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="white", height=550, font=dict(color="#3e4a36"),
                      margin=dict(l=10, r=10, t=30, b=10), xaxis=dict(showgrid=False, showticklabels=False), yaxis=dict(showgrid=False, showticklabels=False))
    return fig

# ============================================================
# MAIN UI
# ============================================================
logo_cols = st.columns([2, 3, 1])
with logo_cols[0]:
    if uva_b64: st.markdown(f'<img src="data:image/jpg;base64,{uva_b64}" width="350">', unsafe_allow_html=True)
with logo_cols[2]:
    if wfa_b64: st.markdown(f'<div style="text-align:right;"><img src="data:image/webp;base64,{wfa_b64}" width="150"></div>', unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; margin-top:-10px;'>Dipteran Forensic Classifier</h1>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("App Configuration")
    version = st.selectbox("BioCLIP Version", ["BioCLIP-1", "BioCLIP-2"])
    mode = st.radio("Group", ["flies", "maggots"])
    st.markdown("---")
    proj_method = st.selectbox("Projection Method", ["UMAP", "t-SNE"])
    show_proj = st.checkbox("Show Feature Map", value=True)
    st.markdown("---")
    show_heatmap = st.checkbox("Show Attention Heatmap", value=False)
    if show_heatmap:
        heatmap_alpha = st.slider("Heatmap Opacity", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

# Upload & Results
col1, col2 = st.columns(2, gap="large")
with col1:
    uploaded = st.file_uploader("Upload specimen image", type=["jpg", "png"], label_visibility="collapsed")
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, width='stretch', caption="Query Specimen")

with col2:
    if uploaded:
        with st.spinner("Analyzing image..."):
            model, preprocess, device = load_model(version, mode)
            knn, train_emb, train_labels, class_names = load_reference(version, mode)

            tensor = preprocess(img).unsqueeze(0).to(device)
            with torch.no_grad():
                query_emb = F.normalize(model.encode_image(tensor), dim=-1).squeeze().cpu().numpy()

            dist, idx = knn.kneighbors([query_emb])
            mean_dist = dist[0].mean()
            pred_idx = np.bincount(train_labels[idx[0]]).argmax()
            is_known = mean_dist < THRESHOLDS[(version, mode)]

            res_class = "result-known" if is_known else "result-unknown"

            st.markdown(f"""
                <div class='result-box {res_class}'>
                    <p class='species-name'>{class_names[pred_idx] if is_known else "Unknown Specimen"}</p>
                    <span class='tag {"tag-known" if is_known else "tag-unknown"}'>{"Identified" if is_known else "Outside Distribution"}</span>
                    <p class='label-grey'>Mean KNN Distance</p>
                    <p class='value-highlight'>{mean_dist:.4f}</p>
                    {f"<hr><p class='label-grey'>Closest known species match</p><p class='value-highlight' style='color:#7a8574;'>{class_names[pred_idx]}</p>" if not is_known else ""}
                </div>
            """, unsafe_allow_html=True)

# ── Attention Heatmap ─────────────────────────────────────────────────────────
if uploaded and show_heatmap:
    st.markdown("### Attention Rollout Heatmap (After LoRA)")
    with st.spinner("Computing attention rollout..."):
        img_tensor = preprocess(img)
        attn_map = get_attention_rollout(model, img_tensor, device)
        overlay_img = render_heatmap_overlay(img, attn_map, alpha=heatmap_alpha)

    h_col1, h_col2 = st.columns(2, gap="large")
    with h_col1:
        st.image(img, caption="Original", width='stretch')
    with h_col2:
        st.image(overlay_img, caption="Attention Overlay", width='stretch')

# ── Feature Projection ────────────────────────────────────────────────────────
if uploaded and show_proj:
    st.markdown("### Feature Distribution (Global Context)")
    st.plotly_chart(
        make_projection_plot(train_emb, train_labels, query_emb, class_names[pred_idx], class_names, proj_method),
        width='stretch'
    )
