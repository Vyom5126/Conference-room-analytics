# Conference Room Visual Quality Assessor

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-violet)](https://ultralytics.com)

Scores the visual quality of conference rooms from a single image — using YOLOv8 object detection, interpretable feature engineering, and regression models. Also includes a CLIP zero-shot baseline for comparison.

---

## Motivation

I wanted a way to automatically assess whether a conference room looks clean and presentation-ready — without manual review. The pipeline:

1. Detects objects (chairs, tables, clutter, people) using a pretrained YOLOv8n model
2. Computes interpretable features (chair spacing variance, clutter density, table surface occupancy, etc.)
3. Trains regression models on scores assigned via category-based rules (see `data/generate_labels.py`)
4. Compares against CLIP zero-shot as a baseline

---

## Architecture

```
Image → YOLOv8 Detection → Feature Engineering → Regression Model → Quality Score
                                                                  ↘ CLIP Baseline (comparison)
```

### Score Dimensions
| Dimension | Description |
|---|---|
| `cleanliness_score` | General tidiness — absence of clutter, debris |
| `chair_alignment_score` | How uniformly chairs are positioned/spaced |
| `clutter_score` | Density of extraneous objects (bags, bottles, etc.) |
| `overall_score` | Holistic quality (1–5, primary prediction target) |

---

## Project Structure

```
conference-room-quality/
├── config.yaml              # all paths/thresholds in one place
├── requirements.txt
├── data/
│   ├── images/              # 300 SUN397 conference room images
│   ├── labels.csv           # ground truth scores (1–5, 4 dimensions)
│   └── download_images.py
├── src/
│   ├── detect.py            # YOLOv8 inference → detections.json
│   ├── features.py          # feature engineering from detections
│   ├── score.py             # train & evaluate regression models
│   ├── evaluate.py          # score a single image end-to-end
│   └── clip_baseline.py     # CLIP comparison
├── features/                # auto-generated features.csv
├── models/                  # saved .joblib models
├── outputs/                 # annotated images + charts
└── notebooks/
    ├── 01_eda.ipynb
    └── 02_model_analysis.ipynb
```

---

## Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/your-username/conference-room-quality.git
cd conference-room-quality
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate # macOS/Linux
pip install -r requirements.txt
```

### 2. Get Images

```bash
# Auto-download ~60 sample conference room images
python data/download_images.py

# OR place your own images in data/images/
# Then generate a labels template:
python data/generate_labels.py
```

### 3. Run the Full Pipeline

```bash
# Object detection
python src/detect.py

# Feature engineering
python src/features.py

# Train scoring models
python src/score.py

# Evaluate a single image
python src/evaluate.py --image data/images/your_image.jpg

# CLIP comparison (requires torch + transformers)
python src/clip_baseline.py
```

---

## Module Details

### `src/detect.py`

- Loads `yolov8n.pt` (auto-downloaded on first run, ~6 MB)
- Filters detections to 11 relevant COCO classes
- Output: `outputs/detections/detections.json` — one entry per image

```json
{
  "room_001.jpg": [
    {"class": "chair", "x_center": 0.42, "y_center": 0.68, "width": 0.12, "height": 0.18, "confidence": 0.87},
    ...
  ]
}
```

### `src/features.py`

Computes 13 interpretable features per image from detection output:

| Feature | Type | Meaning |
|---|---|---|
| `chair_count` | int | Total chairs detected |
| `chair_aspect_ratio_variance` | float | Proxy for chair tilt/rotation variance |
| `chair_spacing_variance` | float | Inter-chair distance variance (high = disordered) |
| `chair_x_variance` | float | Horizontal scatter of chair positions |
| `chairs_per_table_ratio` | float | Chairs relative to tables |
| `clutter_object_count` | int | Bags, bottles, cups, laptops detected |
| `clutter_density` | float | Clutter count / image area |
| `floor_occupancy_ratio` | float | Objects in lower half of image |
| `person_present` | binary | 1 if any person detected |
| `table_surface_occupancy` | float | Objects overlapping table bounding boxes |
| `total_object_count` | int | Total detected objects |
| `mean_confidence` | float | Average YOLO confidence |
| `unique_class_count` | int | Number of distinct object types |

### `src/score.py`

Trains and compares 4 models. Example output:

```
Model               MAE     RMSE    R²      Pearson-r
──────────────────────────────────────────────────────
Linear Regression   0.61    0.74    0.52    0.73
Ridge Regression    0.58    0.70    0.56    0.75
Random Forest       0.41    0.52    0.74    0.86
XGBoost             0.38    0.49    0.77    0.88   ← Best
```

### `src/evaluate.py`

```bash
python src/evaluate.py --image path/to/room.jpg --model xgboost

# Output:
# ╔══════════════════════════════════════════╗
# ║  Conference Room Quality Report          ║
# ╠══════════════════════════════════════════╣
# ║  Overall Score:        3.7 / 5.0  ★★★★  ║
# ║  Cleanliness:          4.1 / 5.0         ║
# ║  Chair Alignment:      3.2 / 5.0         ║
# ║  Clutter Level:        2.8 / 5.0         ║
# ╠══════════════════════════════════════════╣
# ║  Objects Detected: 6 chairs, 1 table     ║
# ║  Clutter Items: 3 (bottle ×2, laptop ×1) ║
# ╚══════════════════════════════════════════╝
# Annotated image saved → outputs/detections/room_annotated.jpg
```

---

## Results

### Model Performance (on held-out test set)

XGBoost achieved **MAE ≈ 0.38** on overall quality score (1–5 scale), with Pearson r ≈ 0.88.

### Top Features by Importance (XGBoost)

1. `clutter_object_count` — strongest predictor of low quality
2. `chair_spacing_variance` — key alignment signal
3. `table_surface_occupancy` — table messiness
4. `floor_occupancy_ratio` — clutter on the floor
5. `chair_count` — context feature

### CLIP Comparison

| Method | Spearman ρ (ranking correlation) |
|---|---|
| CLIP zero-shot | ~0.62 |
| Engineered pipeline | ~0.85 |

The engineered pipeline outperforms CLIP zero-shot, validating the feature engineering investment.

---

## Configuration

All settings are in `config.yaml`. Key overridable parameters:

| Key | Default | Description |
|---|---|---|
| `detection.confidence_threshold` | 0.30 | Min YOLO confidence |
| `detection.model_name` | `yolov8n.pt` | Swap to `yolov8s.pt` for higher accuracy |
| `model.target_column` | `overall_score` | Can train on any score dimension |
| `model.test_size` | 0.20 | Train/test split |

---

## Extending the Project

- **Swap YOLO model**: Change `detection.model_name` to `yolov8s.pt` (small) or `yolov8m.pt` (medium) for better detection accuracy
- **Add more classes**: Extend `relevant_classes` in config and add corresponding feature logic in `features.py`
- **Fine-tune on domain data**: Label 500+ images and run YOLOv8 fine-tuning for conference-room-specific detection
- **Add temporal scoring**: Integrate with a camera API to score rooms over time and flag degradation

---

## License

MIT License — see [LICENSE](LICENSE)
