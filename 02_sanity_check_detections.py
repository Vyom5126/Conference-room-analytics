"""
Sanity-check YOLO detection quality on a candidate image folder BEFORE
spending time relabeling or merging into the training set.

Runs your existing src/detect.py (unmodified) against the candidate folder,
then flags images where detection likely failed or looks implausible for a
conference room photo.

Usage:
    python 02_sanity_check_detections.py \
        --images-dir data/images_candidate \
        --output-dir outputs/candidate_detections \
        --report-out outputs/candidate_detections/sanity_report.csv
"""

import argparse
import csv
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if not SRC_DIR.exists():
    # Fallback: script lives one level above src/ instead of alongside it
    SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))
from detect import run_detection  # noqa: E402
from utils import ROOT, load_config  # noqa: E402

STRICT_CONF_THRESHOLD = 0.5
MIN_PLAUSIBLE_OBJECTS = 1
MAX_PLAUSIBLE_OBJECTS = 40
FURNITURE_CLASSES = {"chair", "dining table"}


def evaluate_detections(dets: list[dict]) -> list[str]:
    flags = []

    if len(dets) == 0:
        flags.append("ZERO_DETECTIONS")
        return flags

    confidences = [d["confidence"] for d in dets]
    if max(confidences) < STRICT_CONF_THRESHOLD:
        flags.append("LOW_CONFIDENCE")

    if len(dets) < MIN_PLAUSIBLE_OBJECTS or len(dets) > MAX_PLAUSIBLE_OBJECTS:
        flags.append("SUSPECT_COUNT")

    class_names = {d["class"].lower() for d in dets}
    if not (class_names & FURNITURE_CLASSES):
        flags.append("NO_CHAIRS_OR_TABLE")

    return flags


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/candidate_detections"))
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--report-out", type=Path, default=Path("outputs/candidate_detections/sanity_report.csv"))
    args = p.parse_args()

    config = load_config(args.config)
    all_dets = run_detection(
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        config=config,
        save_per_image=False,
    )

    rows = []
    flagged_count = 0
    for filename, dets in all_dets.items():
        flags = evaluate_detections(dets)
        if flags:
            flagged_count += 1
        rows.append({
            "image": filename,
            "num_detections": len(dets),
            "flags": ";".join(flags),
        })

    rows.sort(key=lambda r: r["flags"], reverse=True)  # flagged rows float to top

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "num_detections", "flags"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{flagged_count}/{len(rows)} images flagged for manual review.")
    print(f"Report: {args.report_out}")
    print("Open the flagged images and check whether it's a real image problem "
          "(bad angle, occlusion, poor lighting) or a genuine detector gap. "
          "Only merge unflagged + manually-cleared images into data/images/.")


if __name__ == "__main__":
    main()
