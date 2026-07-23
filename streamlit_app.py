"""
streamlit_app.py
Conference Room Quality Assessor — Streamlit UI
Upload a room image → YOLO detection → feature extraction → ML quality score
"""
import sys, tempfile, platform
from pathlib import Path

# On Linux/cloud: use real Pillow. On Windows: inject stub to avoid Smart App Control.
if platform.system() == "Windows":
    sys.path.insert(0, str(Path(__file__).parent / "src" / "pil_stub"))
sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import numpy as np
import streamlit as st

from utils import ROOT, load_config
import evaluate as ev

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Conference Room Quality Assessor",
    page_icon="🏢",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.score-hero {
    background: linear-gradient(135deg, #1a1d2e 0%, #0f1020 100%);
    border: 1px solid rgba(108,99,255,0.3);
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
    margin-bottom: 20px;
}
.score-num  { font-size: 3.5rem; font-weight: 800; color: #a78bfa; line-height: 1; }
.score-denom { font-size: 1rem; color: #7c82a0; }
.score-stars { font-size: 1.8rem; margin: 8px 0; }
.score-label { font-size: 1.1rem; font-weight: 600; color: #e8eaf0; }

.warn-box {
    background: rgba(251,191,36,0.1);
    border: 1px solid rgba(251,191,36,0.3);
    border-radius: 10px;
    padding: 10px 14px;
    color: #fbbf24;
    font-size: 0.88rem;
    margin-bottom: 6px;
}
.section-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #a78bfa;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# ── Cache config & models ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading pipeline…")
def get_config():
    return load_config()

config = get_config()

# ── Helper: annotated image bytes → numpy for st.image ───────────────────────
def read_annotated(stem: str) -> np.ndarray | None:
    paths    = config.get("paths", {})
    dets_dir = ROOT / paths.get("detections_dir", "outputs/detections")
    ann_path = dets_dir / f"{stem}_annotated.jpg"
    if not ann_path.exists():
        return None
    img = cv2.imread(str(ann_path))
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None

def stars(score: float) -> str:
    n = round(score)
    return "★" * n + "☆" * (5 - n)

def score_label(score: float) -> str:
    return ["", "Poor", "Fair", "Good", "Very Good", "Excellent"][round(score)]

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🏢 Conference Room Quality Assessor")
st.caption("Upload a conference room photo and get an AI-powered quality score.")

uploaded = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    label_visibility="collapsed",
)

if uploaded:
    col_img, col_ctrl = st.columns([2, 1])

    with col_img:
        st.image(uploaded, caption="Uploaded image", use_container_width=True)

    with col_ctrl:
        model_name = st.selectbox(
            "Model",
            ["xgboost", "random_forest", "ridge_regression", "linear_regression"],
            format_func=lambda x: x.replace("_", " ").title(),
        )
        run = st.button("🔍 Evaluate Room", use_container_width=True, type="primary")

    if run:
        img_bytes = uploaded.read()
        suffix = Path(uploaded.name).suffix.lower() or ".jpg"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(img_bytes)
            tmp_path = Path(tmp.name)

        with st.spinner("Detecting objects and scoring…"):
            try:
                result = ev.run(tmp_path, model_name, save_outputs=True, config=config)
            except Exception as exc:
                st.error(f"❌ Error: {exc}")
                tmp_path.unlink(missing_ok=True)
                st.stop()

        tmp_path.unlink(missing_ok=True)

        overall   = result["overall_score"]
        breakdown = result["breakdown"]
        feat      = result["features"]
        warns     = result["warnings"]

        # ── Score hero ───────────────────────────────────────────────────────
        st.markdown(f"""
        <div class="score-hero">
            <div class="score-num">{overall:.1f}</div>
            <div class="score-denom">/ 5.0</div>
            <div class="score-stars">{stars(overall)}</div>
            <div class="score-label">{score_label(overall)}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Annotated image ──────────────────────────────────────────────────
        ann = read_annotated(tmp_path.stem)
        if ann is not None:
            st.markdown('<p class="section-label">Detected Objects</p>', unsafe_allow_html=True)
            st.image(ann, use_container_width=True)

        # ── Score breakdown gauges ───────────────────────────────────────────
        st.markdown('<p class="section-label">Score Breakdown</p>', unsafe_allow_html=True)
        for dim, val in breakdown.items():
            st.text(f"{dim}: {val:.1f} / 5.0")
            st.progress(max(0.0, min(1.0, (val - 1) / 4)))

        # ── Key stats ────────────────────────────────────────────────────────
        st.markdown('<p class="section-label">Detection Summary</p>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Objects",  int(result["detections_count"]))
        c2.metric("Chairs",   int(feat.get("chair_count", 0)))
        c3.metric("Clutter",  int(feat.get("clutter_object_count", 0)))
        c4.metric("Occupied", "Yes" if feat.get("person_present") else "No")

        # ── Warnings ─────────────────────────────────────────────────────────
        if warns:
            st.markdown('<p class="section-label">Observations</p>', unsafe_allow_html=True)
            for w in warns:
                st.markdown(f'<div class="warn-box">{w}</div>', unsafe_allow_html=True)

        # ── Model badge ──────────────────────────────────────────────────────
        st.caption(f"Model: {model_name.replace('_',' ').title()}")
