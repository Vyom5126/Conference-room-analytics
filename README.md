# Conference Room Visual Quality Assessor

[![tests](https://github.com/Vyom5126/Conference-room-analytics/actions/workflows/tests.yml/badge.svg)](https://github.com/Vyom5126/Conference-room-analytics/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-violet)](https://ultralytics.com)
[![DINOv2](https://img.shields.io/badge/DINOv2-base-orange)](https://huggingface.co/facebook/dinov2-base)

Scores how ready a conference room looks for the next meeting from a single photo. It rates four things on a 1–5 scale: overall readiness, cleanliness, chair alignment and table clutter.

Every score comes with:
- **a likely range,** calibrated on held-out errors,
- **a warning** when the photo doesn't look like a conference room,
- **the detected objects,** drawn on the photo,
- **an optional heatmap** (experimental) of the regions pulling the score down.

> **Status: prototype.**
> - The 300 training images were rated by an AI judge (Claude) against a written rubric ([`data/rating_rubric.md`](data/rating_rubric.md)). These ratings are the project's labels; they have not been checked against human raters.
> - The scores below measure how well the models reproduce that judge.
> - A drift check found no sign of the judge changing standards across the 300 images (ρ = 0.05 between file order and score, vs 0.07 for a CLIP control).
> - An optional [human check](#validating-the-labels) takes about 30 minutes.

---

## How it works

```
                 ┌─ DINOv2, 6 views averaged ──────────────┐
Image ──────────┼─ DINOv2, close-up of the detected table ─┼─→ Ridge model per dimension → score + likely range
                 └─ YOLOv8 detections ─────────────────────┘          │
                          │                                           ├─→ "unlike the training photos?" check
                          └─→ annotated image, object counts          └─→ experimental heatmap
```

- **Main scoring model:** a ridge regression ("linear probe") on DINOv2 image embeddings.
  - Each photo is embedded six ways and averaged: padded to a square (nothing cropped), its mirror image, and its four quarters (so small clutter fills more of the frame).
  - Each trained model stores the embedding settings it was trained with, so the app always embeds photos exactly as in training.
- **Table close-up:** a crop of the largest table YOLO finds, embedded separately. It improves the table-clutter score (ρ 0.62 → 0.65) but not the others.
- **Automatic model choice:** training picks, for each dimension, the simplest model within 0.005 MAE of the best and saves the choice in `models/selection.json`. Today that's the plain embedding model for overall, cleanliness and chair alignment, and embeddings + table close-up for clutter.
- **YOLOv8:** draws the boxes, counts chairs and small objects, spots occupied rooms, and supplies the table crop. Its 13 engineered features are kept as an interpretable baseline.

### History

1. **v1** trained on labels **generated from its own detection features**, so its metrics only showed that a model could recover that formula.
2. **v2** introduced rubric-based AI ratings of the actual photos. The old synthetic labels correlated only ρ = 0.18 with them, and the YOLO-feature pipeline turned out to be the weakest approach: even zero-shot CLIP ranks rooms better. A CLIP B/32 linear probe became the main model (ρ 0.67).
3. **v3** compared four image models and several view and training tricks (see [Experiments](#experiments)). DINOv2 with multi-view averaging won (ρ 0.76), and it is also the smallest model tried.
4. **v4 (current)** added the table close-up, calibrated score ranges, the out-of-domain warning, the experimental heatmap and CI.

The synthetic v1 labels are kept in `data/labels_synthetic.csv` for reference.

---

## Labels

[`data/rating_rubric.md`](data/rating_rubric.md) defines four scores (5 = best):

| Column | What it rates |
|---|---|
| `overall_score` | Holistic "ready for the next meeting" (half points) |
| `cleanliness_score` | Floor, walls, whiteboard, cables, general condition |
| `chair_alignment_score` | Chairs tucked in and evenly arranged |
| `clutter_score` | Table clear. Deliberate place settings (glasses, notepads) are **not** clutter. |

Every image also records `occupied`, `valid_room`, `confidence` and a short note on what drove the score.
- **Invalid rooms:** 12 images are banquet halls, auditoriums or dining rooms (`valid_room = 0`) and are left out of training.
- **Score distribution:** overall scores span 1.5–5 (mean 4.27, std 0.75). Only 33 of the 300 photos score 3 or below, because SUN397 photos are mostly staged, tidy rooms.

---

## Results

All models use repeated 5×5-fold cross-validation on 288 images and are reported as mean ± std over 25 folds.

### Overall score

| Model | Input | MAE | Spearman ρ | R² |
|---|---|---|---|---|
| Predict the mean | — | 0.554 ± 0.050 | — | −0.03 |
| CLIP zero-shot (no training) | image | — | 0.42 | — |
| XGBoost | YOLO features | 0.508 ± 0.063 | 0.29 ± 0.11 | 0.07 |
| Random Forest | YOLO features | 0.465 ± 0.060 | 0.33 ± 0.12 | 0.15 |
| Ridge | YOLO features | 0.455 ± 0.049 | 0.33 ± 0.11 | 0.22 |
| **Ridge (linear probe)** | **DINOv2 embedding** | **0.324 ± 0.029** | **0.76 ± 0.06** | **0.66** |
| Ridge | DINOv2 + table close-up | 0.321 ± 0.029 | 0.76 ± 0.06 | 0.68 |
| Ridge | DINOv2 + YOLO | 0.329 ± 0.030 | 0.75 ± 0.06 | 0.64 |

The zero-shot CLIP row comes from `src/clip_baseline.py`: 4 positive vs 4 negative prompts, scored on the same 288 images.

### Every dimension: the model the app uses vs the best YOLO-feature model

| Target | Predict mean (MAE) | Best YOLO model (MAE / ρ) | App model | App model (MAE / ρ) |
|---|---|---|---|---|
| Overall | 0.554 | 0.453 / 0.34 | DINOv2 | **0.324 / 0.76** |
| Cleanliness | 0.590 | 0.550 / 0.23 | DINOv2 | **0.417 / 0.61** |
| Chair alignment | 0.530 | 0.511 / 0.28 | DINOv2 | **0.372 / 0.71** |
| Table clutter | 0.678 | 0.464 / 0.44 | DINOv2 + table | **0.386 / 0.65** |

### Score ranges

Each score is shown with a range meant to contain the rating about 80% of the time.
- **Method:** split-conformal ([`src/intervals.py`](src/intervals.py)), calibrated on out-of-fold errors.
- **Split at a predicted 4.0:** messy rooms are over-predicted, so a single ± range misses most of them. Each side of 4.0 gets its own range, and each range is asymmetric.
- **How coverage was checked:** calibrate on a random half, measure on the other half, repeat 200 times.

| Target | Average width | Coverage, all rooms | Coverage, rooms rated ≤ 3 |
|---|---|---|---|
| Overall | 0.90 | 80% | 71% (one symmetric range: 36%) |
| Cleanliness | 1.09 | 80% | 21% |
| Chair alignment | 1.09 | 80% | 62% |
| Table clutter | 0.70 | 79% | 52% |

Cleanliness ranges still miss most messy rooms: the model rarely predicts low cleanliness, so the lower range seldom kicks in.

### "Is this a conference room?" check

- **How it works:** each photo's DINOv2 embedding is compared with its 5 nearest training photos (mean cosine similarity, [`src/ood.py`](src/ood.py)).
- **Threshold:** 0.59, the 1st percentile of the same score for the training photos themselves, each left out in turn; the typical training photo scores 0.79.
- **What it catches:** the Ultralytics sample photos of a bus and a person score 0.13 and 0.15 and are flagged. Banquet halls and auditoriums are near misses: 3 of the 12 in the dataset are flagged.
- **What the app does:** flagged photos still get scores, plus a clear "scores unreliable" warning.

### Heatmap (experimental, off by default)

- **What it shows:** because the model is linear, the patch-average half of the DINOv2 embedding splits exactly into one contribution per image patch ([`src/explain.py`](src/explain.py)). Mapping those back onto the photo shows which regions pull the score below the photo's average.
- **Validation:** on the 74 photos where YOLO found cups, bottles or laptops, those regions are redder than the rest of the photo in 89% of photos (p < 0.001). But that holds equally in tidy rooms (92%), where the objects are deliberate place settings.
- **Causal check:** erasing the detected objects (OpenCV inpainting) and re-scoring barely changes the score. The median gain from clearing *every* detected item is about +0.01 points, no different from erasing the same area of plain background (see [Experiments](#things-that-did-not-make-the-cut)).
- **Verdict:** it shows *where objects are*, not *what to fix*. The model grades the room as a whole (layout, people, and clutter YOLO can't detect), so red regions are not "remove this and the score goes up". That's why it is opt-in and labelled experimental.

### Where it fails

- **Messy rooms still score too high.** Rooms rated 3 or below are predicted about 0.5 too high on average, because there are only 33 such examples. The worst misses are lunch meetings and workshops with food and laptops everywhere.
- **Occupied rooms are harder.** MAE is 0.47 for occupied rooms vs 0.29 for empty ones.
- **YOLO counts place settings as clutter.** A table laid with glasses and notepads looks "cluttered" to the detector, which is one reason the YOLO features score poorly.

---

## Experiments

All rows below use the same repeated 5×5-fold CV on the overall score unless noted. Each figure is ρ / MAE.

### Image backbone and views

| Backbone | Size | Default preprocessing | Best view setting |
|---|---|---|---|
| CLIP ViT-B/32 | 0.6 GB | 0.671 / 0.356 | 0.680 / 0.349 (center + tiles) |
| SigLIP 2 base | 1.5 GB | 0.693 / 0.342 | 0.702 / 0.340 (center + pad/flip) |
| CLIP ViT-L/14 | 1.7 GB | 0.726 / 0.320 | 0.735 / 0.317 (center + tiles) |
| **DINOv2 base** | **0.35 GB** | 0.733 / 0.329 | **0.759 / 0.311 (pad + flip + tiles)** |

- **Padding instead of cropping** helps DINOv2 but slightly hurts both CLIP models: the hypothesis that CLIP's centre crop was hiding clutter did not hold.
- **Combining backbones** (DINOv2 + CLIP or SigLIP) was slightly worse than DINOv2 alone.

### Table close-up (adopted for clutter)

| Target | 6 views | 6 views + table close-up |
|---|---|---|
| Table clutter | 0.607 / 0.356 | **0.645 / 0.332** |
| Overall | 0.761 / 0.308 | 0.763 / 0.302 (tie) |
| Cleanliness | 0.622 / 0.382 | 0.606 / 0.394 (worse) |

### Things that did not make the cut

| Idea | Result | Verdict |
|---|---|---|
| Linear calibration of predictions | Messy-room bias +0.75 → +0.66, MAE unchanged | Not worth the complexity |
| Up-weighting rare low scores | Bias +0.68 → +0.50, but ρ 0.680 → 0.662 and tidy rooms worse | Trade-off, not a fix |
| Synthetic clutter: real cutouts pasted onto 122 tidy tables, used as "original > cluttered" ranking pairs | Bias +0.61 → +0.53, but ρ 0.760 → 0.727 | Cutouts look pasted; the model likely learns the artefacts |
| Per-object "this item costs X points": erase each YOLO-detected object, re-score (models trained without that photo), 227 objects in 74 photos | Objects on messy tables cost more than on tidy ones (+0.07 vs −0.01, p = 0.0002), but no more than erasing a random background patch of the same size (p = 0.37). Clearing *all* items: median +0.01 points vs control (p = 0.06) | The grade barely depends on objects YOLO can see; per-object costs would be noise |

The common lesson: the messy-room bias comes from too few messy examples, and post-hoc fixes only trade it against overall accuracy.

---

## Quick start

```bash
git clone https://github.com/Vyom5126/Conference-room-analytics.git
cd Conference-room-analytics
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS/Linux
pip install -r requirements-dev.txt
```

### Score a photo

```bash
python src/evaluate.py --image data/images/sun397_conf_0089.jpg              # report + annotated image
python src/evaluate.py --image data/images/sun397_conf_0089.jpg --explain    # also save the heatmap
streamlit run streamlit_app.py                                               # web UI
```

Scoring takes about 5.5 s per photo on a laptop CPU, and the first photo takes about 15 s while the models load. The first run downloads `facebook/dinov2-base` (about 350 MB).

### Rebuild everything

```bash
python src/detect.py          # YOLO → outputs/detections/detections.json
python src/features.py        # → features/features.csv
python src/embed.py           # DINOv2: 6 views + table close-up → features/embeddings.npz (~12 min on CPU)
python src/score.py           # CV all models × 4 targets; saves models, ranges, selection.json, reference.npz
python src/clip_baseline.py   # zero-shot CLIP vs trained models
python -m pytest tests        # 24 tests, no model downloads needed (also run by GitHub Actions)
```

**Other backbones:** CLIP and SigLIP checkpoints also work. Change `config.yaml → embedding`, then rerun `embed.py` and `score.py`. (The heatmap supports DINOv2 only.)

---

## Validating the labels

The labels come from a single AI rater. If you want to check them against your own judgment:

```bash
python src/human_check.py sample   # 50 images, stratified so messy rooms are included
streamlit run rate_app.py          # rate them blind (AI scores are hidden) → data/labels_human.csv
python src/human_check.py report   # Spearman / MAE / % within ±0.5: you vs AI judge vs model
```

- **If agreement is close to typical human–human agreement** (ρ around 0.6–0.8, most scores within ±0.5): the labels are a reasonable target. Quote that figure alongside the results.
- **If it isn't:** relabel with your own ratings and rerun `python src/score.py`.

---

## Project structure

```
├── config.yaml                     # paths, classes, backbone + views, targets, CV, ranges, OOD, prompts
├── streamlit_app.py                # web UI
├── rate_app.py                     # blind human-rating page for the label check
├── .github/workflows/tests.yml     # runs pytest on every push
├── data/
│   ├── images/                     # 300 SUN397 conference-room photos
│   ├── labels.csv                  # AI-judge ratings (4 scores + occupied/valid_room/confidence/notes)
│   ├── rating_rubric.md            # the rubric used for labels.csv
│   ├── human_check_sample.csv      # the 50 images to rate yourself
│   ├── labels_synthetic.csv        # old feature-derived labels (reference only)
│   └── generate_labels*.py, download_images.py
├── src/
│   ├── detect.py                   # YOLOv8 inference (batch + single image)
│   ├── features.py                 # 13 features from detections
│   ├── embed.py                    # multi-view + table close-up embeddings (DINOv2 / CLIP / SigLIP)
│   ├── score.py                    # repeated K-fold CV, model selection, final models for every target
│   ├── intervals.py                # calibrated score ranges
│   ├── ood.py                      # "unlike the training photos" check
│   ├── explain.py                  # experimental heatmap
│   ├── evaluate.py                 # score one image end to end
│   ├── clip_baseline.py            # zero-shot CLIP comparison
│   └── human_check.py              # sample + agreement report for the human check
├── models/
│   ├── <target>/<model>.joblib     # trained models (each with its range and embedding settings)
│   ├── selection.json              # model used per target in "auto" mode
│   └── reference.npz               # training embeddings + threshold for the OOD check
├── tests/                          # pytest (24 tests)
└── notebooks/                      # 01_eda, 02_model_analysis (with saved outputs)
```

---

## Known limitations

- **One AI rater, no human check.** The labels reflect a single AI judge's reading of the rubric.
- **Too few messy rooms.** This is the main source of error (see [Experiments](#experiments)). Adding 50–100 genuinely messy photos, ideally before/after-meeting pairs of the same room, would help more than any model change.
- **Ranges are weakest where it matters.** About 80% coverage overall, but lower for messy rooms (21–71% depending on the dimension).
- **COCO detector blind spots.** COCO has no class for paper, trash, cables or whiteboard writing, so the YOLO features and object counts miss most real clutter.
- **Research-only data.** SUN397 images are for research use. Check the licence before any commercial use.

---

## Roadmap

1. **Collect more messy rooms.** Your own office photos, or more SUN397 / Places365 conference rooms selected by the current model's lowest scores, rated with the same rubric.
2. **Fine-tune the backbone** with a pairwise ranking loss once there are about 1,000 rated images. With 288, a frozen backbone plus ridge is the safer choice.
3. **Detect what COCO can't.** Use open-vocabulary detection (YOLO-World / OWLv2) for paper, trash and cables, so observations can say *what* is wrong and the heatmap can tell leftovers from place settings.
4. **Make the grade explainable.** Per-object costs and the heatmap show the current model is holistic. Real "where" answers need local supervision: region-level ratings (table / floor / chairs crops) or an open-vocabulary detector whose counts feed the grade directly.
5. **Make it faster.** A 2-view "fast mode" or an ONNX export of DINOv2 for CPU deployment.
6. **Deploy with fixed cameras.** Compare each room against its own "clean reference" photo and score rooms after each booking.

---

## License

No license file has been added yet. All rights reserved by the author until one is.
