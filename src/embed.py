"""
embed.py
Compute image embeddings for every image in data/images/ and save them to
features/embeddings.npz. Ridge "linear probes" on these embeddings are the
main quality models (see score.py).

The backbone (CLIP, SigLIP or DINOv2) and the views averaged per image are set
in config.yaml → embedding. Views:
  center   the backbone's default preprocessing (DINOv2/CLIP centre-crop to a square)
  pad      whole image padded to a square, so nothing is cropped
  flip     the padded image mirrored
  tiles    the four quarters of the image, each padded to a square
The main embedding is the L2-normalised mean of the per-view embeddings.

A second "table close-up" embedding is computed from the largest table YOLO
detected (outputs/detections/detections.json), padded to a square. It helps the
table-clutter score.

Usage:
    python src/detect.py      # table boxes are needed first
    python src/embed.py
"""
import argparse, json, logging, time
from pathlib import Path
import cv2, numpy as np
from utils import ROOT, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TABLE_CONF   = 0.3     # minimum YOLO confidence for the table used in the close-up
TABLE_MARGIN = 1.1     # table box is enlarged by 10% so items on its edge are kept

_model_cache: dict = {}   # model name → backbone dict, loaded once per process


def load_backbone(model_name):
    """
    Return a cached dict {kind, model, proc, encode}; encode maps a list of PIL images
    to an (n, d) torch tensor of raw embeddings.
    """
    if model_name in _model_cache:
        return _model_cache[model_name]
    import torch
    from transformers import AutoImageProcessor, AutoModel, AutoProcessor

    name = model_name.lower()
    if "clip" in name or "siglip" in name:
        model, proc, kind = AutoModel.from_pretrained(model_name).eval(), AutoProcessor.from_pretrained(model_name), "clip"
        def encode(images):
            out = model.get_image_features(**proc(images=images, return_tensors="pt"))
            return out if torch.is_tensor(out) else out.pooler_output
    elif "dinov2" in name:   # CLS token + mean patch token
        model, proc, kind = AutoModel.from_pretrained(model_name).eval(), AutoImageProcessor.from_pretrained(model_name), "dinov2"
        def encode(images):
            h = model(**proc(images=images, return_tensors="pt")).last_hidden_state
            return torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1)
    else:
        raise ValueError(f"Unsupported embedding model: {model_name} (use a CLIP, SigLIP or DINOv2 checkpoint)")
    _model_cache[model_name] = {"kind": kind, "model": model, "proc": proc, "encode": encode}
    return _model_cache[model_name]


def to_pil(im):
    """Path or BGR array → RGB PIL image."""
    from PIL import Image
    if isinstance(im, (str, Path)):
        return Image.open(im).convert("RGB")
    return Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))


def pad_square(im):
    """Pad a PIL image to a square with its mean colour (no cropping)."""
    from PIL import Image
    w, h = im.size
    s = max(w, h)
    canvas = Image.new("RGB", (s, s), tuple(int(c) for c in np.asarray(im).reshape(-1, 3).mean(0)))
    canvas.paste(im, ((s - w) // 2, (s - h) // 2))
    return canvas


def tile_boxes(w, h):
    """The four quarter boxes (left, top, right, bottom) used by the 'tiles' view, in reading order."""
    return [(x*w//2, y*h//2, (x+1)*w//2, (y+1)*h//2) for y in (0, 1) for x in (0, 1)]


def make_views(im, views):
    """Expand one PIL image into the configured views."""
    from PIL import ImageOps
    out, w, h = [], *im.size
    for v in views:
        if v == "center":
            out.append(im)
        elif v == "pad":
            out.append(pad_square(im))
        elif v == "flip":
            out.append(ImageOps.mirror(pad_square(im)))
        elif v == "tiles":
            out += [pad_square(im.crop(b)) for b in tile_boxes(w, h)]
        else:
            raise ValueError(f"Unknown view: {v}")
    return out


def table_box(dets, w, h):
    """Pixel box of the largest confidently detected table, enlarged by TABLE_MARGIN; None if there is none."""
    tables = [d for d in dets if d["class"] == "dining table" and d["confidence"] >= TABLE_CONF]
    if not tables:
        return None
    t = max(tables, key=lambda d: d["width"] * d["height"])
    half_w, half_h = t["width"] * TABLE_MARGIN / 2, t["height"] * TABLE_MARGIN / 2
    return (int(max(0, t["x_center"] - half_w) * w), int(max(0, t["y_center"] - half_h) * h),
            int(min(1, t["x_center"] + half_w) * w), int(min(1, t["y_center"] + half_h) * h))


def embed_images(images, emb_cfg):
    """
    Main L2-normalised embeddings for a list of images (paths or BGR arrays).
    `emb_cfg` is the config.yaml → embedding section (model_name, views).
    Returns an (n, d) float32 array.
    """
    import torch
    encode = load_backbone(emb_cfg["model_name"])["encode"]
    views  = emb_cfg.get("views", ["center"])
    result = []
    for im in images:
        with torch.no_grad():
            e = torch.nn.functional.normalize(encode(make_views(to_pil(im), views)), dim=-1).mean(0)
        result.append(torch.nn.functional.normalize(e, dim=-1).numpy())
    return np.stack(result).astype(np.float32)


def embed_tables(images, dets_list, emb_cfg):
    """
    Table close-up embeddings: the largest detected table (or the whole image if
    none was found), padded to a square. Returns ((n, d) float32 array, has_table list).
    """
    import torch
    encode = load_backbone(emb_cfg["model_name"])["encode"]
    crops, has = [], []
    for im, dets in zip(images, dets_list):
        pil = to_pil(im)
        box = table_box(dets, *pil.size)
        crops.append(pad_square(pil.crop(box) if box else pil)); has.append(box is not None)
    with torch.no_grad():
        e = torch.nn.functional.normalize(encode(crops), dim=-1)
    return e.numpy().astype(np.float32), has


def embedding_cols(dim, prefix="emb"):
    """Column names used for embedding features in models and payloads ('emb' main, 'tab' table close-up)."""
    return [f"{prefix}_{i:03d}" for i in range(dim)]


def build_embeddings(images_dir, output_path, config, detections_path):
    """Embed every image in images_dir (main + table close-up) and save everything to an .npz file."""
    files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise FileNotFoundError(f"No images found in {images_dir}")
    if not detections_path.exists():
        raise FileNotFoundError(f"Run detect.py first (table boxes are needed): {detections_path}")
    with open(detections_path, encoding="utf-8") as f:
        dets = json.load(f)
    emb_cfg = config["embedding"]
    log.info("Embedding %d images with %s, views %s + table close-up…",
             len(files), emb_cfg["model_name"], emb_cfg.get("views", ["center"]))

    t0, main, table, has, bs = time.time(), [], [], [], int(emb_cfg.get("batch_size", 16))
    for i in range(0, len(files), bs):
        batch = files[i:i+bs]
        main.append(embed_images(batch, emb_cfg))
        t, h = embed_tables(batch, [dets.get(f.name, []) for f in batch], emb_cfg)
        table.append(t); has += h
        log.info("[%d/%d] %.1fs", min(i+bs, len(files)), len(files), time.time()-t0)

    main, table = np.concatenate(main), np.concatenate(table)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, filenames=np.array([f.name for f in files]), embeddings=main,
             table_embeddings=table, has_table=np.array(has),
             model_name=emb_cfg["model_name"], views=np.array(emb_cfg.get("views", ["center"])))
    log.info("✓ Saved %s main + %s table embeddings (%d images with a table) → %s",
             main.shape, table.shape, sum(has), output_path)
    return main


def load_embeddings(path, emb_cfg=None, kind="main"):
    """
    Load the .npz written by build_embeddings as a DataFrame indexed by filename.
    kind = "main" (emb_* columns) or "table" (tab_* columns).
    If emb_cfg is given, refuse files built with a different model or views.
    """
    import pandas as pd
    d = np.load(path)
    if emb_cfg is not None and "model_name" in d:
        built = (str(d["model_name"]), list(d["views"]))
        wanted = (emb_cfg["model_name"], list(emb_cfg.get("views", ["center"])))
        if built != wanted:
            raise ValueError(f"{path} was built with {built}, config asks for {wanted}. Rerun: python src/embed.py")
    key, prefix = ("embeddings", "emb") if kind == "main" else ("table_embeddings", "tab")
    if key not in d:
        raise ValueError(f"{path} has no {kind} embeddings. Rerun: python src/embed.py")
    return pd.DataFrame(d[key], index=d["filenames"], columns=embedding_cols(d[key].shape[1], prefix))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", type=Path)
    p.add_argument("--output",     type=Path)
    p.add_argument("--detections", type=Path)
    p.add_argument("--config",     type=Path)
    args = p.parse_args()

    config = load_config(args.config)
    paths  = config.get("paths", {})
    build_embeddings(
        images_dir      = args.images_dir or ROOT / paths.get("images_dir", "data/images"),
        output_path     = args.output     or ROOT / paths.get("embeddings_npz", "features/embeddings.npz"),
        config          = config,
        detections_path = args.detections or ROOT / paths.get("detections_dir", "outputs/detections") / "detections.json",
    )

if __name__ == "__main__":
    main()
