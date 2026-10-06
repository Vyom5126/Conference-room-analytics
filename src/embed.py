"""
embed.py
Compute CLIP image embeddings for every image in data/images/ and save them to
features/clip_embeddings.npz. A ridge "linear probe" on these embeddings is the
main quality model (see score.py).

Usage:
    python src/embed.py
"""
import argparse, logging, time
from pathlib import Path
import cv2, numpy as np
from utils import ROOT, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

_clip_cache: dict = {}   # model name → (model, processor), loaded once per process


def load_clip(config):
    """Load (cached) CLIP model and processor named in config.embedding.model_name."""
    from transformers import CLIPModel, CLIPProcessor
    name = config["embedding"]["model_name"]
    if name not in _clip_cache:
        _clip_cache[name] = (CLIPModel.from_pretrained(name).eval(), CLIPProcessor.from_pretrained(name))
    return _clip_cache[name]


def embed_images(images, config):
    """
    L2-normalised CLIP image embeddings for a list of images (paths or BGR arrays).
    Returns an (n, d) float32 array.
    """
    import torch
    from PIL import Image
    model, proc = load_clip(config)
    pil = []
    for im in images:
        if isinstance(im, (str, Path)):
            pil.append(Image.open(im).convert("RGB"))
        else:
            pil.append(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
    with torch.no_grad():
        out = model.get_image_features(**proc(images=pil, return_tensors="pt"))
    out = out if torch.is_tensor(out) else out.pooler_output
    return torch.nn.functional.normalize(out, dim=-1).numpy().astype(np.float32)


def embedding_cols(dim):
    """Column names used for embedding features in models and payloads."""
    return [f"emb_{i:03d}" for i in range(dim)]


def build_embeddings(images_dir, output_path, config):
    """Embed every image in images_dir and save {filenames, embeddings} to an .npz file."""
    files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not files:
        raise FileNotFoundError(f"No images found in {images_dir}")
    bs = int(config["embedding"].get("batch_size", 16))
    log.info("Embedding %d images with %s…", len(files), config["embedding"]["model_name"])

    t0, chunks = time.time(), []
    for i in range(0, len(files), bs):
        chunks.append(embed_images(files[i:i+bs], config))
        log.info("[%d/%d] %.1fs", min(i+bs, len(files)), len(files), time.time()-t0)

    emb = np.concatenate(chunks)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, filenames=np.array([f.name for f in files]), embeddings=emb)
    log.info("✓ Saved %s embeddings → %s", emb.shape, output_path)
    return emb


def load_embeddings(path):
    """Load the .npz written by build_embeddings as a DataFrame indexed by filename."""
    import pandas as pd
    d = np.load(path)
    return pd.DataFrame(d["embeddings"], index=d["filenames"], columns=embedding_cols(d["embeddings"].shape[1]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", type=Path)
    p.add_argument("--output",     type=Path)
    p.add_argument("--config",     type=Path)
    args = p.parse_args()

    config = load_config(args.config)
    paths  = config.get("paths", {})
    build_embeddings(
        images_dir  = args.images_dir or ROOT / paths.get("images_dir", "data/images"),
        output_path = args.output     or ROOT / paths.get("embeddings_npz", "features/clip_embeddings.npz"),
        config      = config,
    )

if __name__ == "__main__":
    main()
