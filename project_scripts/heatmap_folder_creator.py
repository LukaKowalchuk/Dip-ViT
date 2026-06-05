import os
import random
import shutil
import csv

# ─────────────────────────────────────────────
# CONFIG — set your directory here
# ─────────────────────────────────────────────
DATASET_DIR = r"C:\Users\lukak\Downloads\Entomology app\KowalchukLuka_research_project-main\datasets\final_dataset"  # <-- change this

MIN_IMAGES_FLIES   = 282
MIN_IMAGES_MAGGOTS = 423
SAMPLE_SIZE        = 17

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# Output folders are placed two levels above DATASET_DIR (i.e. at C:/path/to/)
PARENT_DIR     = os.path.dirname(os.path.dirname(DATASET_DIR))
OUTPUT_FLIES   = os.path.join(PARENT_DIR, "heatmap_labelme_test_folder_flies")
OUTPUT_MAGGOTS = os.path.join(PARENT_DIR, "heatmap_labelme_test_folder_maggots")
# ─────────────────────────────────────────────


def get_images(folder):
    """Return all image file paths directly inside a folder (non-recursive)."""
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
        and os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ]


def collect_samples():
    fly_samples    = []
    maggot_samples = []

    subfolders = [
        os.path.join(DATASET_DIR, d)
        for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ]

    for folder in subfolders:
        name = os.path.basename(folder)
        is_maggot = "maggot" in name.lower()

        images    = get_images(folder)
        threshold = MIN_IMAGES_MAGGOTS if is_maggot else MIN_IMAGES_FLIES

        if len(images) < threshold:
            print(f"  SKIP  '{name}' — {len(images)} images (need {threshold})")
            continue

        selected = random.sample(images, SAMPLE_SIZE)
        print(f"  OK    '{name}' — {len(images)} images → {SAMPLE_SIZE} selected")

        if is_maggot:
            maggot_samples.extend(selected)
        else:
            fly_samples.extend(selected)

    return fly_samples, maggot_samples


def copy_samples(samples, output_dir, label):
    os.makedirs(output_dir, exist_ok=True)
    random.shuffle(samples)

    print(f"\nCopying {len(samples)} {label} images to:\n  {output_dir}")

    manifest = []  # rows for the CSV: (new_filename, original_filename, original_path)

    for i, src in enumerate(samples, start=1):
        ext      = os.path.splitext(src)[1].lower()
        dst_name = f"{i:04d}{ext}"
        dst      = os.path.join(output_dir, dst_name)

        # Avoid overwriting if name already exists (safety guard)
        counter = 1
        while os.path.exists(dst):
            dst_name = f"{i:04d}_{counter}{ext}"
            dst      = os.path.join(output_dir, dst_name)
            counter += 1

        shutil.copy2(src, dst)
        manifest.append((dst_name, os.path.basename(src), src))

    # Write manifest CSV into the output folder
    csv_path = os.path.join(output_dir, "sample_manifest.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["new_filename", "original_filename", "original_filepath"])
        writer.writerows(manifest)

    print(f"  Done — {len(samples)} files written.")
    print(f"  Manifest saved → {csv_path}")


def main():
    random.seed()  # truly random each run; set a number here for reproducibility

    print("=" * 55)
    print("Scanning subfolders...")
    print("=" * 55)

    fly_samples, maggot_samples = collect_samples()

    print(f"\nTotal fly images selected:    {len(fly_samples)}")
    print(f"Total maggot images selected: {len(maggot_samples)}")

    if not fly_samples and not maggot_samples:
        print("\nNo eligible folders found. Check your thresholds or directory.")
        return

    copy_samples(fly_samples,    OUTPUT_FLIES,   "fly")
    copy_samples(maggot_samples, OUTPUT_MAGGOTS, "maggot")

    print("\n" + "=" * 55)
    print("All done!")
    print(f"  Flies   → {OUTPUT_FLIES}")
    print(f"  Maggots → {OUTPUT_MAGGOTS}")
    print("=" * 55)


if __name__ == "__main__":
    main()
