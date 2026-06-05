"""
LabelMe annotation → binary mask converter
-------------------------------------------
Point IMAGE_DIR at your folder of fly images, run labelme on them,
then run this script to load the resulting JSONs and visualise the masks.

Usage:
    pip install labelme matplotlib numpy pillow
    python labelme_mask_demo.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image, ImageDraw

# ── CONFIG ────────────────────────────────────────────────────────────────────
IMAGE_DIR   = Path(r"C:\Users\lukak\Downloads\Entomology app\KowalchukLuka_research_project-main\bioclip_test\heatmap_labelme_test_folder_flies")   #  folder with images + labelme JSONs
MAX_IMAGES  = 102                                  # how many to process
# ─────────────────────────────────────────────────────────────────────────────


def polygon_to_mask(points: list, img_h: int, img_w: int) -> np.ndarray:
    """Rasterise a polygon (list of [x, y] vertices) onto a boolean mask."""
    mask = Image.new("L", (img_w, img_h), 0)
    flat = [coord for xy in points for coord in xy]   # [[x,y],...] → [x,y,x,y,...]
    ImageDraw.Draw(mask).polygon(flat, outline=1, fill=1)
    return np.array(mask, dtype=bool)


def load_annotation(json_path: Path):
    """
    Load a LabelMe JSON and return:
        image       – numpy array (H, W, 3)
        shapes      – list of shape dicts from the JSON
        img_h, img_w
    """
    with open(json_path) as f:
        data = json.load(f)

    img_h = data["imageHeight"]
    img_w = data["imageWidth"]

    # LabelMe can embed the image as base64; if not, load from disk
    if data.get("imageData"):
        import base64, io
        img_bytes = base64.b64decode(data["imageData"])
        image = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
    else:
        img_path = json_path.parent / data["imagePath"]
        image = np.array(Image.open(img_path).convert("RGB"))

    return image, data["shapes"], img_h, img_w


def process_annotation(json_path: Path) -> dict:
    """
    For one JSON return:
        image           – (H, W, 3) uint8
        masks           – dict of label → boolean mask (H, W)
        combined_mask   – union of all masks
    """
    image, shapes, img_h, img_w = load_annotation(json_path)

    masks = {}
    for shape in shapes:
        label      = shape["label"]
        shape_type = shape.get("shape_type", "polygon")
        points     = shape["points"]

        if shape_type in ("polygon", "rectangle"):
            if shape_type == "rectangle":
                # LabelMe stores rectangles as two corner points; expand to 4
                (x1, y1), (x2, y2) = points
                points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            mask = polygon_to_mask(points, img_h, img_w)
        else:
            print(f"  Skipping unsupported shape type '{shape_type}' in {json_path.name}")
            continue

        # Multiple shapes with the same label → union
        if label in masks:
            masks[label] = masks[label] | mask
        else:
            masks[label] = mask

    combined = np.zeros((img_h, img_w), dtype=bool)
    for m in masks.values():
        combined |= m

    return {"image": image, "masks": masks, "combined_mask": combined}


def visualise(result: dict, title: str):
    """Show the original image, each labelled mask, and the combined mask."""
    labels   = list(result["masks"].keys())
    n_panels = 1 + len(labels) + 1          # image | per-label masks | combined
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=11)

    # Original image
    axes[0].imshow(result["image"])
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Per-label masks overlaid on image
    colours = plt.cm.tab10.colors
    for i, label in enumerate(labels):
        ax    = axes[i + 1]
        mask  = result["masks"][label]
        overlay = result["image"].copy().astype(float)
        colour_rgb = np.array(colours[i % 10][:3]) * 255
        overlay[mask] = overlay[mask] * 0.5 + colour_rgb * 0.5
        ax.imshow(overlay.astype(np.uint8))
        ax.set_title(f'"{label}"')
        ax.axis("off")

    # Combined mask
    ax = axes[-1]
    ax.imshow(result["image"])
    combined_rgba = np.zeros((*result["combined_mask"].shape, 4), dtype=np.uint8)
    combined_rgba[result["combined_mask"]] = [255, 50, 50, 160]
    ax.imshow(combined_rgba)
    ax.set_title("Combined")
    ax.axis("off")

    plt.tight_layout()
    plt.show()


def summarise(result: dict, name: str):
    """Print pixel coverage stats for each label."""
    total_px = result["image"].shape[0] * result["image"].shape[1]
    print(f"\n── {name} ──")
    for label, mask in result["masks"].items():
        pct = mask.sum() / total_px * 100
        print(f"  {label:25s}  {mask.sum():>7,} px  ({pct:.1f}% of image)")
    comb_pct = result["combined_mask"].sum() / total_px * 100
    print(f"  {'COMBINED':25s}  {result['combined_mask'].sum():>7,} px  ({comb_pct:.1f}% of image)")


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    json_files = sorted(ANNOTATION_DIR.glob("*.json"))[:MAX_IMAGES]

    if not json_files:
        print(f"No LabelMe JSON files found in: {IMAGE_DIR}")
        print("Make sure you've run labelme on your images first.")
    else:
        print(f"Found {len(json_files)} annotation(s). Processing...\n")

    for json_path in json_files:
        print(f"Loading: {json_path.name}")
        try:
            result = process_annotation(json_path)
            summarise(result, json_path.stem)
            visualise(result, json_path.stem)
        except Exception as e:
            print(f"  ERROR in {json_path.name}: {e}")