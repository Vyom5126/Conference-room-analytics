"""
rate_app.py
Blind rating page for the human check: shows each sampled image (no AI scores)
and saves your ratings to data/labels_human.csv after every image.

    python src/human_check.py sample
    streamlit run rate_app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd
import streamlit as st

from human_check import HUMAN_CSV, SAMPLE_CSV, SCORES
from utils import ROOT, load_config

st.set_page_config(page_title="Rate Conference Rooms", page_icon="📝", layout="centered")
config = load_config()

if not SAMPLE_CSV.exists():
    st.error("No sample yet. Run `python src/human_check.py sample` first.")
    st.stop()

sample = pd.read_csv(SAMPLE_CSV)["filename"].tolist()
done = pd.read_csv(HUMAN_CSV) if HUMAN_CSV.exists() else pd.DataFrame(columns=["filename", *SCORES])
todo = [f for f in sample if f not in set(done["filename"])]

st.title("📝 Rate conference rooms")
st.progress(len(done) / len(sample), text=f"{len(done)} / {len(sample)} rated")
with st.expander("Rubric (5 = best)"):
    st.markdown((ROOT / "data" / "rating_rubric.md").read_text(encoding="utf-8"))

if not todo:
    st.success("All done. Run `python src/human_check.py report` to see the agreement.")
    st.stop()

fname = todo[0]
st.image(str(ROOT / config["paths"]["images_dir"] / fname), caption=fname, width="stretch")

with st.form(key=fname):
    vals = {
        "overall_score":         st.select_slider("Overall readiness", [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5], value=4),
        "cleanliness_score":     st.select_slider("Cleanliness (floor, walls, whiteboard)", [1, 2, 3, 4, 5], value=4),
        "chair_alignment_score": st.select_slider("Chair alignment", [1, 2, 3, 4, 5], value=4),
        "clutter_score":         st.select_slider("Table clear of clutter", [1, 2, 3, 4, 5], value=4),
    }
    if st.form_submit_button("Save and next", type="primary"):
        row = pd.DataFrame([{"filename": fname, **vals}])
        pd.concat([done, row]).to_csv(HUMAN_CSV, index=False)
        st.rerun()
