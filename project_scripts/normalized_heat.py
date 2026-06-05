"""
bioclip_area_heat_analysis.py
==============================
1. Loads a fine-tuned BioCLIP-2 + LoRA model (weights-only .pt)
2. Runs attention rollout on every image in IMAGE_DIR
3. Reads matching LabelMe JSONs from ANNOTATION_DIR
4. Computes per-polygon metrics:
     - raw_heat          : sum of heatmap values inside the mask
     - relative_heat     : raw_heat / total_heat  (0–1, no area correction)
     - normalized_heat   : relative_heat / area_px  (current metric, area-corrected)
     - enrichment        : relative_heat / (area_px / total_px)  (enrichment ratio)
5. Saves results to a CSV + prints per-label averages across all images

Outputs
-------
  OUTPUT_DIR/heatmaps_raw_npy/       – raw patch-grid .npy  (e.g. 16x16)
  OUTPUT_DIR/heatmaps_resized_npy/   – resized .npy  (orig image resolution)
  OUTPUT_DIR/overlays_png/           – original | rollout side-by-side
  OUTPUT_DIR/mask_overlays_png/      – jet heatmap + white polygon overlay
  OUTPUT_DIR/heat_results.csv        – per-image metrics
  OUTPUT_DIR/heat_averages.csv       – per-label averages across all images

CSV columns (heat_results.csv)
-----------
  image, label, area_px, raw_heat, relative_heat, normalized_heat, enrichment

Usage:
    python bioclip_area_heat_analysis.py
"""

import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
from tqdm import tqdm
from collections import defaultdict
import open_clip as oc
from peft import get_peft_model, LoraConfig

# ── CONFIG ────────────────────────────────────────────────────────────────────
ANNOTATION_DIR = Path(r"C:\Users\lukak\Downloads\Entomology app\KowalchukLuka_research_project-main\bioclip_test\heatmap_labelme_test_folder_flies")
IMAGE_DIR      = Path(r"C:\Users\lukak\Downloads\Entomology app\KowalchukLuka_research_project-main\bioclip_test\heatmap_labelme_test_folder_flies")
MODEL_PATH     = Path(r"C:\Users\lukak\Downloads\Entomology app\KowalchukLuka_research_project-main\app_files\BioCLIP-2_lora_best_flies.pt")
OUTPUT_DIR     = Path(r"C:\Users\lukak\Downloads\Entomology app\KowalchukLuka_research_project-main\bioclip_test\normalized_heatmap_storage")
MAX_IMAGES     = 102

DISCARD_RATIO  = 0.9
IMG_SIZE       = 224
IMG_EXTS       = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

# LoRA config — must match training
LORA_RANK      = 8
LORA_ALPHA     = 16
LORA_DROPOUT   = 0.1
LORA_TARGETS   = ["out_proj", "c_fc", "c_proj"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────────────────────────────────────────

# Output subdirs
HEATMAP_RAW_DIR     = OUTPUT_DIR / "heatmaps_raw_npy"
HEATMAP_RESIZED_DIR = OUTPUT_DIR / "heatmaps_resized_npy"
OVERLAY_DIR         = OUTPUT_DIR / "overlays_png"
MASK_OVERLAY_DIR    = OUTPUT_DIR / "mask_overlays_png"
for d in (HEATMAP_RAW_DIR, HEATMAP_RESIZED_DIR, OVERLAY_DIR, MASK_OVERLAY_DIR):
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# MODEL — always reconstructs architecture then loads weights
# =============================================================================

def load_model(model_path, device):
    print("Building BioCLIP-2 + LoRA architecture...")
    base_model, _, preprocess = oc.create_model_and_transforms(
        "hf-hub:imageomics/bioclip-2"
    )

    lora_config = LoraConfig(
        r              = LORA_RANK,
        lora_alpha     = LORA_ALPHA,
        target_modules = LORA_TARGETS,
        lora_dropout   = LORA_DROPOUT,
        bias           = "none",
    )
    model = get_peft_model(base_model, lora_config)

    print(f"Loading weights from {model_path} ...")
    state_dict = torch.load(model_path, map_location=device)

    if isinstance(state_dict, torch.nn.Module):
        state_dict = state_dict.state_dict()

    if not isinstance(state_dict, dict):
        raise TypeError(
            f"Expected a state-dict (dict/OrderedDict) in {model_path}, "
            f"got {type(state_dict)}."
        )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys    ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    model = model.to(device).eval()
    print("Model ready.\n")
    return model, preprocess


# =============================================================================
# ATTENTION ROLLOUT
# =============================================================================

def get_attention_rollout(model, image_tensor, device, discard_ratio=DISCARD_RATIO):
    resblocks = model.base_model.model.visual.transformer.resblocks
    hooks = []

    for block in resblocks:
        original_forward = block.attn.forward
        store = {}

        def make_hook(s, orig):
            def patched(query, key, value, **kwargs):
                kwargs["need_weights"]         = True
                kwargs["average_attn_weights"] = False
                out, weights = orig(query, key, value, **kwargs)
                s["weights"] = weights
                return out, weights
            return patched

        block.attn.forward = make_hook(store, original_forward)
        hooks.append((block, original_forward, store))

    with torch.no_grad():
        _ = model.encode_image(image_tensor.unsqueeze(0).to(device))

    all_attentions = []
    for block, orig_fwd, store in hooks:
        block.attn.forward = orig_fwd
        if "weights" in store:
            attn = store["weights"].squeeze(0).mean(0).cpu().numpy()
            all_attentions.append(attn)

    if not all_attentions:
        raise RuntimeError("No attention weights captured — check model architecture.")

    result = np.eye(all_attentions[0].shape[0])
    for attn in all_attentions:
        threshold = np.percentile(attn.flatten(), discard_ratio * 100)
        attn      = np.where(attn >= threshold, attn, 0)
        attn      = attn / (attn.sum(axis=-1, keepdims=True) + 1e-8)
        attn_adj  = 0.5 * attn + 0.5 * np.eye(attn.shape[0])
        result    = attn_adj @ result

    cls_attn  = result[0, 1:]
    grid_size = int(cls_attn.shape[0] ** 0.5)
    attn_map  = cls_attn.reshape(grid_size, grid_size)
    attn_map  = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    return attn_map.astype(np.float32)


# =============================================================================
# MASK UTILITIES
# =============================================================================

def polygon_to_mask(points, img_h, img_w):
    mask = Image.new("L", (img_w, img_h), 0)
    flat = [coord for xy in points for coord in xy]
    ImageDraw.Draw(mask).polygon(flat, outline=1, fill=1)
    return np.array(mask, dtype=bool)


def build_masks(json_path: Path, override_hw=None):
    with open(json_path) as f:
        data = json.load(f)

    # Use corrected dimensions if provided (EXIF rotation may swap H/W)
    if override_hw is not None:
        img_h, img_w = override_hw
    else:
        img_h, img_w = data["imageHeight"], data["imageWidth"]

    per_label        = {}
    per_label_points = {}

    for shape in data["shapes"]:
        label      = shape["label"]
        shape_type = shape.get("shape_type", "polygon")
        points     = shape["points"]

        if shape_type == "rectangle":
            (x1, y1), (x2, y2) = points
            points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        elif shape_type != "polygon":
            print(f"  Skipping unsupported shape '{shape_type}' in {json_path.name}")
            continue

        mask = polygon_to_mask(points, img_h, img_w)
        per_label[label] = per_label[label] | mask if label in per_label else mask

        if label not in per_label_points:
            per_label_points[label] = []
        per_label_points[label].append(points)

    combined = np.zeros((img_h, img_w), dtype=bool)
    for m in per_label.values():
        combined |= m

    return per_label, per_label_points, combined, img_h, img_w


def resize_mask_to(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    if mask.shape == (target_h, target_w):
        return mask
    return np.array(
        Image.fromarray(mask.astype(np.uint8)).resize(
            (target_w, target_h), resample=Image.NEAREST
        ), dtype=bool
    )


# =============================================================================
# HEAT METRICS
# =============================================================================

def compute_heat_metrics(heatmap: np.ndarray, per_label: dict, combined: np.ndarray):
    """
    raw_heat        = sum of heatmap values inside the mask
    relative_heat   = raw_heat / total_heat          (0–1, no area correction)
    normalized_heat = relative_heat / area_px        (area-corrected density)
    enrichment      = relative_heat / (area_px / total_px)
                      values around 1 = neutral, >1 = enriched, <1 = depleted
    """
    H, W       = heatmap.shape
    total_heat = float(heatmap.sum()) + 1e-8
    total_px   = H * W
    rows       = []

    for label, mask in list(per_label.items()) + [("COMBINED", combined)]:
        m        = resize_mask_to(mask, H, W)
        area_px  = int(m.sum())
        raw_heat = float(heatmap[m].sum())

        relative_heat   = raw_heat / total_heat
        normalized_heat = relative_heat / (area_px + 1e-8)
        enrichment      = relative_heat / ((area_px / total_px) + 1e-8)

        rows.append({
            "label":           label,
            "area_px":         area_px,
            "raw_heat":        raw_heat,
            "relative_heat":   relative_heat,
            "normalized_heat": normalized_heat,
            "enrichment":      enrichment,
        })

    return rows


# =============================================================================
# OVERLAY FIGURES
# =============================================================================

def save_overlay(stem, orig_np, attn_resized):
    """Standard side-by-side: original | jet heatmap overlay at 224x224."""
    overlay = np.clip(
        0.5 * orig_np / 255.0 + 0.5 * plt.cm.jet(attn_resized)[:, :, :3], 0, 1
    )
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(orig_np);  axes[0].axis("off"); axes[0].set_title("Original")
    axes[1].imshow(overlay);  axes[1].axis("off"); axes[1].set_title(f"Rollout (discard={DISCARD_RATIO})")
    fig.suptitle(stem, fontsize=10)
    plt.tight_layout()
    plt.savefig(OVERLAY_DIR / f"{stem}.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_mask_overlay(stem, orig_img, attn_raw, per_label_points):
    """
    Renders at original image resolution so polygon coords need no scaling.
    Derives resize target from the actual numpy array shape to handle both
    portrait and landscape images correctly.
      - white semi-transparent fill inside each polygon
      - solid white outline around each polygon
      - label text at the polygon centroid
    """
    orig_np = np.array(orig_img)
    h, w    = orig_np.shape[:2]

    attn_full = cv2.resize(attn_raw, (w, h))

    base = np.clip(
        0.5 * orig_np / 255.0 + 0.5 * plt.cm.jet(attn_full)[:, :, :3], 0, 1
    )

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(base)
    ax.axis("off")

    legend_handles = []
    for label, polygons in per_label_points.items():
        for points in polygons:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]

            poly_patch = plt.Polygon(
                points,
                closed=True,
                facecolor=(1, 1, 1, 0.25),
                edgecolor=(1, 1, 1, 0.9),
                linewidth=1.5,
            )
            ax.add_patch(poly_patch)

            cx, cy = np.mean(xs), np.mean(ys)
            ax.text(
                cx, cy, label,
                color="white", fontsize=7, fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.4, lw=0),
            )

        legend_handles.append(
            mpatches.Patch(facecolor=(1, 1, 1, 0.25), edgecolor="white", label=label)
        )

    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="upper right",
            fontsize=7,
            framealpha=0.5,
            facecolor="black",
            labelcolor="white",
        )

    ax.set_title(f"{stem}  —  heatmap + annotations", fontsize=9)
    plt.tight_layout()
    plt.savefig(MASK_OVERLAY_DIR / f"{stem}_mask_overlay.png", dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    model, preprocess = load_model(MODEL_PATH, DEVICE)

    image_paths = sorted(
        p for p in IMAGE_DIR.iterdir() if p.suffix in IMG_EXTS
    )[:MAX_IMAGES]

    print(f"Found {len(image_paths)} image(s) in {IMAGE_DIR}\n")

    all_rows = []
    failed   = []

    for img_path in tqdm(image_paths, desc="Processing"):
        stem      = img_path.stem
        json_path = ANNOTATION_DIR / (stem + ".json")

        if not json_path.exists():
            print(f"  No annotation for {stem}, skipping.")
            continue

        try:
            # 1. Load image at original resolution, correcting for EXIF rotation
            orig_img   = Image.open(img_path).convert("RGB")
            orig_img   = ImageOps.exif_transpose(orig_img)   # bakes rotation into pixels
            img_tensor = preprocess(orig_img)
            orig_np    = np.array(orig_img.resize((IMG_SIZE, IMG_SIZE)))

            # 2. Attention rollout
            attn_raw     = get_attention_rollout(model, img_tensor, DEVICE)
            attn_resized = cv2.resize(attn_raw, (IMG_SIZE, IMG_SIZE))

            # 3. Save heatmaps
            np.save(HEATMAP_RAW_DIR     / f"{stem}_raw.npy",     attn_raw)
            np.save(HEATMAP_RESIZED_DIR / f"{stem}_resized.npy", attn_resized)

            # 4. Save standard overlay (224x224)
            save_overlay(stem, orig_np, attn_resized)

            # 5. Build masks from JSON, passing corrected H/W so that if EXIF
            #    rotation swapped portrait/landscape the mask dimensions match
            corrected_hw = (orig_img.height, orig_img.width)
            per_label, per_label_points, combined, img_h, img_w = build_masks(
                json_path, override_hw=corrected_hw
            )

            # 6. Save mask overlay — derives dimensions from numpy array,
            #    handles portrait/landscape correctly
            save_mask_overlay(stem, orig_img, attn_raw, per_label_points)

            # 7. Resize heatmap to original image resolution for accurate masking
            attn_full = cv2.resize(attn_raw, (img_w, img_h))

            # 8. Compute heat metrics
            rows = compute_heat_metrics(attn_full, per_label, combined)
            for row in rows:
                row["image"] = stem
                all_rows.append(row)
                print(f"  {stem}  |  {row['label']:20s}  "
                      f"area={row['area_px']:>7,}px  "
                      f"rel={row['relative_heat']:.4f}  "
                      f"norm={row['normalized_heat']:.6f}  "
                      f"enrich={row['enrichment']:.4f}")

        except Exception as e:
            print(f"  ERROR on {stem}: {e}")
            failed.append(stem)

    # 9. Write heat_results.csv
    results_csv = OUTPUT_DIR / "heat_results.csv"
    fieldnames  = ["image", "label", "area_px", "raw_heat",
                   "relative_heat", "normalized_heat", "enrichment"]
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nPer-image results saved → {results_csv}")

    # 10. Compute per-label averages and write heat_averages.csv
    label_accum = defaultdict(lambda: {
        "raw_heat": [], "relative_heat": [], "normalized_heat": [],
        "enrichment": [], "area_px": []
    })
    for row in all_rows:
        label_accum[row["label"]]["raw_heat"].append(row["raw_heat"])
        label_accum[row["label"]]["relative_heat"].append(row["relative_heat"])
        label_accum[row["label"]]["normalized_heat"].append(row["normalized_heat"])
        label_accum[row["label"]]["enrichment"].append(row["enrichment"])
        label_accum[row["label"]]["area_px"].append(row["area_px"])

    averages_csv   = OUTPUT_DIR / "heat_averages.csv"
    avg_fieldnames = ["label", "n_images", "mean_area_px",
                      "mean_raw_heat",
                      "mean_relative_heat",   "std_relative_heat",
                      "mean_normalized_heat", "std_normalized_heat",
                      "mean_enrichment",      "std_enrichment"]

    with open(averages_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=avg_fieldnames)
        writer.writeheader()
        print("\n── Per-label averages across all images ──")
        for label, vals in sorted(label_accum.items()):
            n       = len(vals["normalized_heat"])
            row_avg = {
                "label":                label,
                "n_images":             n,
                "mean_area_px":         round(np.mean(vals["area_px"]),          2),
                "mean_raw_heat":        round(np.mean(vals["raw_heat"]),          6),
                "mean_relative_heat":   round(np.mean(vals["relative_heat"]),     6),
                "std_relative_heat":    round(np.std(vals["relative_heat"]),      6),
                "mean_normalized_heat": round(np.mean(vals["normalized_heat"]),   8),
                "std_normalized_heat":  round(np.std(vals["normalized_heat"]),    8),
                "mean_enrichment":      round(np.mean(vals["enrichment"]),        6),
                "std_enrichment":       round(np.std(vals["enrichment"]),         6),
            }
            writer.writerow(row_avg)
            print(f"  {label:20s}  n={n:>3}  "
                  f"rel={row_avg['mean_relative_heat']:.6f}  "
                  f"norm={row_avg['mean_normalized_heat']:.8f}  "
                  f"enrich={row_avg['mean_enrichment']:.6f}")

    print(f"\nAverages saved → {averages_csv}")

    if failed:
        print(f"\nFailed ({len(failed)}): {failed}")

    print("\nDone.")


if __name__ == "__main__":
    main()