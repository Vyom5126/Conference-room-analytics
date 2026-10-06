# Conference Room Visual Quality Assessor

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-violet)](https://ultralytics.com)
[![CLIP](https://img.shields.io/badge/CLIP-ViT--B%2F32-orange)](https://huggingface.co/openai/clip-vit-base-patch32)

Scores how ready a conference room looks for the next meeting from a single photo. It rates four things on a 1–5 scale: overall readiness, cleanliness, chair alignment and table clutter. It also shows the objects it detected.

> **Status: prototype.**
> - The 300 training images were rated by an AI judge (Claude) against a written rubric ([`data/rating_rubric.md`](data/rating_rubric.md)). They have not yet been checked against human raters.
> - The scores below measure how well the models reproduce that judge.
> - A drift check found no sign of the judge changing standards across the 300 images (ρ = 0.05 between file order and score, vs 0.07 for a CLIP control).
> - The [human check](#validating-the-labels) is the next step and takes about 30 minutes.

---

## How it works

```
                 ┌─ CLIP ViT-B/32 image embedding (512-d) ─┐
Image ──────────┤                                          ├─→ Ridge models (one per dimension) → 4 scores
                 └─ YOLOv8 detections → 13 features ───────┘
                          │
                          └─→ annotated image, object counts, observations
```

- **Scoring:** a ridge regression ("linear probe") on CLIP image embeddings. This is the main model, `clip_ridge`.
- **YOLOv8:** draws the boxes, counts chairs and small objects, and spots occupied rooms. Its 13 engineered features are kept as an interpretable baseline. Adding them to the CLIP model makes no measurable difference.

### History

The first version trained on labels **generated from the detection features themselves**, so its metrics only showed that a model could recover that formula. Once rubric-based AI ratings of the actual photos existed:
- The old synthetic labels correlated only ρ = 0.18 with them.
- The YOLO-feature pipeline turned out to be the weakest approach. Even zero-shot CLIP, with no training at all, ranks rooms better.

The synthetic labels are kept in `data/labels_synthetic.csv` for reference.

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
| **Ridge (linear probe)** | **CLIP embedding** | **0.360 ± 0.046** | **0.67 ± 0.08** | **0.54** |
| Ridge | CLIP + YOLO | 0.356 ± 0.045 | 0.67 ± 0.07 | 0.55 |

The zero-shot CLIP row comes from `src/clip_baseline.py`: 4 positive vs 4 negative prompts, scored on the same 288 images.

### Every dimension: CLIP linear probe vs best YOLO-feature model

| Target | Predict mean (MAE) | Best YOLO model (MAE / ρ) | CLIP probe (MAE / ρ) |
|---|---|---|---|
| Overall | 0.554 | 0.453 / 0.34 | **0.360 / 0.67** |
| Cleanliness | 0.590 | 0.550 / 0.23 | **0.430 / 0.57** |
| Chair alignment | 0.530 | 0.511 / 0.28 | **0.402 / 0.64** |
| Table clutter | 0.678 | 0.464 / 0.44 | **0.423 / 0.57** |

### Where it fails

- **Messy rooms score too high.** Rooms rated 3 or below are predicted about 0.8 too high on average, because there are only 33 such examples to learn from. The worst misses are lunch meetings and workshops with food and laptops everywhere (rated 1.5, predicted about 3.2).
- **Occupied rooms are harder.** MAE is 0.51 for occupied rooms vs 0.34 for empty ones.
- **YOLO miscounts place settings as clutter.** A table laid with glasses and notepads looks "cluttered" to the detector, which is one reason the YOLO features score poorly.

---

## Validating the labels

The labels come from a single AI rater, so check them against your own judgment before trusting the numbers:

```bash
python src/human_check.py sample   # 50 images, stratified so messy rooms are included
streamlit run rate_app.py          # rate them blind (AI scores are hidden) → data/labels_human.csv
python src/human_check.py report   # Spearman / MAE / % within ±0.5: you vs AI judge vs model
```

- **If agreement is close to typical human–human agreement** (ρ around 0.6–0.8, most scores within ±0.5): the labels are a reasonable target. Quote that figure alongside the results.
- **If it isn't:** relabel with your own ratings and rerun `python src/score.py`.

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

### Run the pipeline

```bash
python src/detect.py          # YOLO → outputs/detections/detections.json
python src/features.py        # → features/features.csv
python src/embed.py           # CLIP → features/clip_embeddings.npz
python src/score.py           # CV all models × 4 targets, save models/<target>/<model>.joblib
python src/evaluate.py --image data/images/sun397_conf_0089.jpg
python src/clip_baseline.py   # zero-shot CLIP vs trained models
streamlit run streamlit_app.py
python -m pytest tests
```

The first CLIP run downloads `openai/clip-vit-base-patch32` (about 600 MB).

---

## Project structure

```
├── config.yaml                     # paths, classes, CLIP model, targets, CV settings, prompts
├── streamlit_app.py                # web UI
├── rate_app.py                     # blind human-rating page for the label check
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
│   ├── embed.py                    # CLIP image embeddings
│   ├── score.py                    # repeated K-fold CV + final models for every target
│   ├── evaluate.py                 # score one image end to end
│   ├── clip_baseline.py            # zero-shot CLIP comparison
│   └── human_check.py              # sample + agreement report for the human check
├── models/<target>/<model>.joblib  # trained models
├── tests/                          # pytest
└── notebooks/                      # 01_eda, 02_model_analysis
```

---

## Known limitations

- **One AI rater, no human check yet.** The rater (Claude) and the main model (CLIP) are both vision-language models and may share blind spots. The human check above addresses this.
- **Too few messy rooms.** Adding 50–100 genuinely messy photos, ideally before/after-meeting pairs of the same room, would help more than any model change.
- **COCO detector blind spots.** COCO has no class for paper, trash, cables or whiteboard writing, so the YOLO features miss most real clutter.
- **Research-only data.** SUN397 images are for research use. Check the licence before any commercial use.

---

## Roadmap

1. **Validate the labels.** Rate the 50-image sample and report agreement.
2. **Collect more messy rooms** and fine-tune with a pairwise ranking loss ("which room is tidier?").
3. **Try stronger backbones.** SigLIP 2 or DINOv2 embeddings are a one-line change in `config.yaml → embedding.model_name` (SigLIP needs its own processor).
4. **Detect what COCO can't.** Use open-vocabulary detection (YOLO-World / OWLv2) for paper, trash and cables, so the observations can say *what* is wrong.
5. **Deploy with fixed cameras.** Compare each room against its own "clean reference" photo and score rooms after each booking.

---

## License

No license file has been added yet. All rights reserved by the author until one is.
