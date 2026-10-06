"""
detect.py
Run YOLOv8 on every image in data/images/ and save detections to a JSON file.

Usage:
    python src/detect.py
    python src/detect.py --images-dir data/images --output-dir outputs/detections
"""
import argparse, json, logging, sys, time
from pathlib import Path

from utils import ROOT, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

_yolo_cache: dict = {}   # model name → loaded YOLO model (one load per process)


def load_detector(config):
    """Load (cached) YOLO model. Returns (model, set of class IDs to keep)."""
    from ultralytics import YOLO
    cfg  = config.get("detection", {})
    keep = set(cfg["relevant_classes"])
    name = cfg.get("model_name", "yolov8n.pt")
    if name not in _yolo_cache:
        _yolo_cache[name] = YOLO(name)
    model = _yolo_cache[name]
    return model, {cid for cid, n in model.names.items() if n in keep}


def detect_image(image, config):
    """Detect relevant objects in one image (path or BGR array). Returns a list of dicts."""
    cfg = config.get("detection", {})
    model, keep_ids = load_detector(config)
    res = model.predict(source=str(image) if isinstance(image, Path) else image,
                        conf=float(cfg.get("confidence_threshold", 0.30)),
                        iou=float(cfg.get("iou_threshold", 0.45)),
                        imgsz=int(cfg.get("image_size", 640)),
                        device=cfg.get("device", ""), verbose=False)[0]
    h, w = res.orig_shape[:2]

    # Keep only relevant objects and store their position + confidence
    dets = []
    for box in res.boxes:
        cid = int(box.cls.item())
        if cid not in keep_ids:
            continue
        xc, yc, bw, bh = box.xywhn[0].tolist()
        dets.append({"class": model.names[cid],
                     "x_center": round(xc, 5), "y_center": round(yc, 5),
                     "width": round(bw, 5),    "height":   round(bh, 5),
                     "confidence": round(float(box.conf.item()), 4),
                     "img_width": w, "img_height": h})
    return dets


def run_detection(images_dir, output_dir, config, save_per_image=False):
    """Detect objects in every image. Returns {filename: [list of objects]}."""
    model, keep_ids = load_detector(config)
    log.info("Loaded model. Tracking: %s", sorted(model.names[i] for i in keep_ids))

    # Collect image paths
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        log.error("No images found in %s", images_dir); sys.exit(1)
    log.info("Found %d images", len(images))
    output_dir.mkdir(parents=True, exist_ok=True)

    all_dets, total, t0 = {}, 0, time.time()

    for i, img in enumerate(images, 1):
        dets = detect_image(img, config)
        all_dets[img.name] = dets
        total += len(dets)

        if save_per_image:
            with open(output_dir / f"{img.stem}_det.json", "w") as f:
                json.dump(dets, f, indent=2)

        if i % 10 == 0 or i == len(images):
            log.info("[%d/%d] %.1fs | %.2f obj/img", i, len(images), time.time()-t0, total/i)

    out = output_dir / "detections.json"
    with open(out, "w") as f:
        json.dump(all_dets, f, indent=2)
    log.info("✓ Saved %d objects from %d images → %s", total, len(images), out)
    return all_dets


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--config", type=Path)
    p.add_argument("--save-per-image", action="store_true")
    args = p.parse_args()

    config  = load_config(args.config)
    paths   = config.get("paths", {})
    run_detection(
        images_dir = args.images_dir or ROOT / paths.get("images_dir", "data/images"),
        output_dir = args.output_dir or ROOT / paths.get("detections_dir", "outputs/detections"),
        config     = config,
        save_per_image = args.save_per_image,
    )

if __name__ == "__main__":
    main()
