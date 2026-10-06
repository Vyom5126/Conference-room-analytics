"""
streamlit_app.py
Conference Room Quality Assessor — Streamlit UI
Upload a room image → YOLO detection → feature extraction → ML quality score
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import cv2
import numpy as np
import streamlit as st

from utils import load_config
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

MODEL_LABELS = {
    "clip_ridge":       "CLIP linear probe (best)",
    "hybrid_ridge":     "CLIP + YOLO features",
    "ridge_regression": "YOLO features only (baseline)",
}

def score_label(score: float) -> str:
    return ["", "Poor", "Fair", "Good", "Very Good", "Excellent"][int(score + 0.5)]

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🏢 Conference Room Quality Assessor")
st.caption("Upload a conference room photo and get an AI-powered quality score.")
st.info("Prototype: models are trained on 288 photos rated by an AI judge (Claude) against a written "
        "rubric. The ratings have not yet been checked against human raters, so treat scores as indicative.")

uploaded = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    label_visibility="collapsed",
)

if uploaded:
    col_img, col_ctrl = st.columns([2, 1])

    with col_img:
        st.image(uploaded, caption="Uploaded image", width="stretch")

    with col_ctrl:
        model_name = st.selectbox(
            "Model",
            list(MODEL_LABELS),
            format_func=MODEL_LABELS.get,
        )
        run = st.button("🔍 Evaluate Room", width="stretch", type="primary")

    if run:
        img = cv2.imdecode(np.frombuffer(uploaded.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            st.error("❌ Could not decode the uploaded image.")
            st.stop()

        with st.spinner("Detecting objects and scoring… (the first run downloads the CLIP model)"):
            try:
                result = ev.run(img, model_name, save_outputs=False, config=config,
                                name=uploaded.name, quiet=True)
            except Exception as exc:
                st.error(f"❌ Error: {exc}")
                st.stop()

        overall   = result["overall_score"]
        breakdown = result["breakdown"]
        feat      = result["features"]
        warns     = result["warnings"]

        # ── Score hero ───────────────────────────────────────────────────────
        st.markdown(f"""
        <div class="score-hero">
            <div class="score-num">{overall:.1f}</div>
            <div class="score-denom">/ 5.0</div>
            <div class="score-stars">{ev.stars(overall)}</div>
            <div class="score-label">{score_label(overall)}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Annotated image ──────────────────────────────────────────────────
        st.markdown('<p class="section-label">Detected Objects</p>', unsafe_allow_html=True)
        st.image(cv2.cvtColor(result["annotated"], cv2.COLOR_BGR2RGB), width="stretch")

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
        c3.metric("Small items", int(feat.get("clutter_object_count", 0)))
        c4.metric("Occupied", "Yes" if feat.get("person_present") else "No")

        # ── Warnings ─────────────────────────────────────────────────────────
        if warns:
            st.markdown('<p class="section-label">Observations</p>', unsafe_allow_html=True)
            for w in warns:
                st.markdown(f'<div class="warn-box">{w}</div>', unsafe_allow_html=True)

        # ── Model badge ──────────────────────────────────────────────────────
        st.caption(f"Model: {MODEL_LABELS[model_name]}")
