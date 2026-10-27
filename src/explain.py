"""
explain.py
Experimental "where did the score go?" heatmap for the DINOv2 linear probe.

Validation (74 photos with YOLO-detected cups/bottles/laptops): those regions are
redder than the rest of the photo in 89% of photos (p < 0.001), but equally so in
tidy rooms where the objects are place settings. It localises objects; it cannot
judge whether they are mess.

The main embedding is  x = m/‖m‖,  m = mean_v z_v/‖z_v‖,  z_v = [CLS_v ; mean_i p_vi]
(one z per view, p_vi = patch tokens). The probe is linear in the standardised
embedding, so holding the norms fixed the patch-average half splits exactly into
one contribution per patch:

    c_vi = w_p · p_vi / (N · V · ‖z_v‖ · ‖m‖)

where w_p are the probe weights (divided by the scaler's std) on the patch-average
dimensions. Each patch is mapped back through its view's geometry (padding, flip,
tile, resize and centre-crop) onto the photo. The map shows each region relative
to the photo's own average: red areas pull the score down. The CLS half of the
embedding (global context) is not spatial and is not shown.
"""
import cv2, numpy as np
from embed import load_backbone, make_views, pad_square, tile_boxes, to_pil


def view_geometry(views, w, h):
    """
    For each view image make_views() produces: (src_box, canvas_size, paste_offset, flipped).
    src_box = region of the original photo shown; the canvas is that region (padded to a square
    for pad/flip/tiles) with the region pasted at paste_offset.
    """
    geo = []
    for v in views:
        if v == "center":
            geo.append(((0, 0, w, h), (w, h), (0, 0), False))
        elif v in ("pad", "flip"):
            s = max(w, h)
            geo.append(((0, 0, w, h), (s, s), ((s - w) // 2, (s - h) // 2), v == "flip"))
        elif v == "tiles":
            for (l, t, r, b) in tile_boxes(w, h):
                tw, th = r - l, b - t
                s = max(tw, th)
                geo.append(((l, t, r, b), (s, s), ((s - tw) // 2, (s - th) // 2), False))
        else:
            raise ValueError(f"Unknown view: {v}")
    return geo


def patch_boxes(geo, shortest_edge, crop, patch):
    """
    Image-space boxes (l, t, r, b) of every patch of one view, in the token order
    DINOv2 uses (row-major over the crop×crop grid). Boxes may fall partly on padding.
    """
    (sl, st, sr, sb), (cw, ch), (ox, oy), flipped = geo
    scale = shortest_edge / min(cw, ch)
    rw, rh = cw * scale, ch * scale
    cx0, cy0 = (rw - crop) / 2, (rh - crop) / 2          # centre-crop origin in resized coordinates
    g = crop // patch
    boxes = []
    for r in range(g):
        for c in range(g):
            x0 = (cx0 + c * patch) / scale - ox; x1 = (cx0 + (c + 1) * patch) / scale - ox
            y0 = (cy0 + r * patch) / scale - oy; y1 = (cy0 + (r + 1) * patch) / scale - oy
            if flipped:   # canvas x was mirrored: region x → (region width - x)
                x0, x1 = (sr - sl) - x1, (sr - sl) - x0
            boxes.append((sl + x0, st + y0, sl + x1, st + y1))
    return boxes


def patch_contributions(hidden, w_patch):
    """
    hidden: (V, 1+N, d) last hidden states; w_patch: (d,) effective probe weights on the patch-average half.
    Returns (contributions (V, N), main embedding x (2d,)).
    """
    cls, p = hidden[:, 0], hidden[:, 1:]
    z = np.concatenate([cls, p.mean(1)], axis=1)                 # (V, 2d)
    zn = np.linalg.norm(z, axis=1)                               # (V,)
    m = (z / zn[:, None]).mean(0)
    mn = np.linalg.norm(m)
    V, N = p.shape[:2]
    contrib = (p @ w_patch) / (N * V * zn[:, None] * mn)         # (V, N)
    return contrib, (m / mn).astype(np.float32)


def effective_weights(payload, dim):
    """Probe weights on the main embedding columns, in raw-embedding units (coef / scaler std)."""
    pipe = payload["model"]
    sc, ridge = pipe.named_steps["sc"], pipe.named_steps["m"]
    cols = payload["feature_cols"]
    idx = [cols.index(f"emb_{i:03d}") for i in range(dim)]
    return np.asarray(ridge.coef_)[idx] / sc.scale_[idx]


def explain(image, payload, cell=8):
    """
    Heatmap for one photo (path or BGR array) and a DINOv2 probe payload.
    Returns (overlay BGR image, density map, main embedding x) or None if the model can't be explained.
    """
    import torch
    emb_cfg = payload.get("embedding") or {}
    if payload.get("feature_set") not in ("embed", "embed_table", "hybrid") or "dinov2" not in emb_cfg.get("model_name", "").lower():
        return None
    bb = load_backbone(emb_cfg["model_name"])
    model, proc = bb["model"], bb["proc"]
    pil = to_pil(image)
    w, h = pil.size
    views = emb_cfg.get("views", ["center"])
    with torch.no_grad():
        hidden = model(**proc(images=make_views(pil, views), return_tensors="pt")).last_hidden_state.numpy()
    d = hidden.shape[2]
    wts = effective_weights(payload, 2 * d)
    contrib, x = patch_contributions(hidden, wts[d:])

    shortest = (proc.size.get("shortest_edge") or min(proc.size.get("height", 224), proc.size.get("width", 224)))
    crop = proc.crop_size["height"] if getattr(proc, "do_center_crop", True) else shortest
    patch = model.config.patch_size

    H, W = max(1, h // cell), max(1, w // cell)
    density = np.zeros((H, W)); weight = np.zeros((H, W))
    for vi, geo in enumerate(view_geometry(views, w, h)):
        for c, (l, t, r, b) in zip(contrib[vi], patch_boxes(geo, shortest, crop, patch)):
            x0, x1 = int(np.clip(l / cell, 0, W)), int(np.clip(np.ceil(r / cell), 0, W))
            y0, y1 = int(np.clip(t / cell, 0, H)), int(np.clip(np.ceil(b / cell), 0, H))
            if x1 <= x0 or y1 <= y0:
                continue                                  # patch lies on padding only
            area = max(1e-9, (r - l) * (b - t) / cell**2)
            density[y0:y1, x0:x1] += c / area
            weight[y0:y1, x0:x1] += 1
    density = cv2.GaussianBlur(density, (0, 0), sigmaX=1.5)
    rel = density - density[weight > 0].mean() if (weight > 0).any() else density

    # Overlay: red where the region pulls the score below the photo's average
    neg = np.clip(-rel, 0, None)
    scale = np.percentile(neg[neg > 0], 99) if (neg > 0).any() else 1.0
    alpha = cv2.resize(np.clip(neg / (scale + 1e-12), 0, 1), (w, h), interpolation=cv2.INTER_LINEAR)[..., None] * 0.6
    bgr = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR).astype(np.float32)
    red = np.zeros_like(bgr); red[..., 2] = 255
    overlay = (bgr * (1 - alpha) + red * alpha).astype(np.uint8)
    return overlay, rel, x
