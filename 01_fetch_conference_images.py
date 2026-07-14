"""
Fetch conference/meeting room images from SUN397 for dataset augmentation.

The original Princeton host (vision.princeton.edu/.../SUN397.tar.gz) is dead
(404) as of mid-2026 -- this also breaks torchvision's built-in downloader,
since it points at the same URL. This version pulls from the Hugging Face
mirror instead, using streaming mode so you only pull the conference/meeting
room images rather than the full 37GB archive.

Install requirements:
    pip install datasets pillow opencv-python tqdm huggingface_hub

Usage:
    python 01_fetch_conference_images.py --target-count 300 --out-dir data/images_candidate
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# Full 397-class SUN397 mirror on Hugging Face (image + ClassLabel columns).
HF_DATASET_ID = "1aurent/SUN397"

TARGET_LABEL_KEYWORDS = ["conference_room", "meeting_room"]

MIN_WIDTH = 640
MIN_HEIGHT = 480
BLUR_VARIANCE_THRESHOLD = 80.0  # Laplacian variance; lower = blurrier


def is_blurry(pil_image: Image.Image) -> bool:
    rgb = np.array(pil_image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < BLUR_VARIANCE_THRESHOLD


def passes_quality_check(pil_image: Image.Image) -> bool:
    width, height = pil_image.size
    if width < MIN_WIDTH or height < MIN_HEIGHT:
        return False
    if is_blurry(pil_image):
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-count", type=int, default=300)
    parser.add_argument("--out-dir", type=str, default="data/images_candidate")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Streaming {HF_DATASET_ID} from Hugging Face (no full download)...")
    ds = load_dataset(HF_DATASET_ID, split="train", streaming=True)

    label_feature = ds.features["label"]
    matching_label_ids = {
        i for i, name in enumerate(label_feature.names)
        if any(kw in name.lower() for kw in TARGET_LABEL_KEYWORDS)
    }
    matching_label_names = {label_feature.names[i] for i in matching_label_ids}
    print(f"Matched categories: {sorted(matching_label_names)}")

    if not matching_label_ids:
        print("No matching categories found. Check label_feature.names manually:")
        print(label_feature.names)
        return

    accepted = 0
    scanned = 0
    pbar = tqdm(desc="Scanning + filtering", unit="img")

    for example in ds:
        scanned += 1
        pbar.update(1)
        if example["label"] not in matching_label_ids:
            continue

        img = example["image"]  # PIL.Image
        if not passes_quality_check(img):
            continue

        dest_path = out_dir / f"sun397_conf_{accepted:04d}.jpg"
        img.convert("RGB").save(dest_path, quality=95)
        accepted += 1

        if accepted >= args.target_count:
            break

    pbar.close()
    print(f"\nScanned {scanned} images, accepted {accepted} after quality filtering.")
    print(f"Saved to {out_dir}")
    print("Next: run 02_sanity_check_detections.py against this folder before merging "
          "into data/images/ or relabeling.")


if __name__ == "__main__":
    main()
