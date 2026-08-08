"""
CropSense — An Explainable Crop Recommendation System
Using Hybrid Feature Fusion and Stacking Ensemble Learning.

Streamlit dashboard entry point. Five pages, routed via a top navbar:
    Home · Predict · Batch · Dashboard · About

Run:
    streamlit run app.py
"""
from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

from utils.styles import GLOBAL_CSS

ROOT = Path(__file__).parent
LOGO_PATH = ROOT / "assets" / "logo_256.png"

st.set_page_config(
    page_title="CropSense — Explainable Crop Recommendation",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

MODELS_DIR = ROOT / "models"
MODEL_READY = (MODELS_DIR / "production_model.joblib").exists()


@st.cache_resource(show_spinner=False)
def _logo_b64() -> str:
    return base64.b64encode((ROOT / "assets" / "logo_64.png").read_bytes()).decode()


LOGO_B64 = _logo_b64()

NAV_ITEMS = [
    ("home", "Home"),
    ("single", "Predict"),
    ("batch", "Batch"),
    ("performance", "Dashboard"),
    ("about", "About"),
]
NAV_SLUGS = {slug for slug, _ in NAV_ITEMS}

if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "home"

if "nav" in st.query_params:
    target = st.query_params["nav"]
    if target in NAV_SLUGS:
        st.session_state["nav_page"] = target
    st.query_params.clear()

current = st.session_state["nav_page"]

nav_links_html = "".join(
    f'<a href="?nav={slug}" target="_self" class="{"active" if slug == current else ""}">{label}</a>'
    for slug, label in NAV_ITEMS
)
st.markdown(f"""
<div class="topnav">
    <div class="topnav-brand">
        <img class="topnav-logo" src="data:image/png;base64,{LOGO_B64}" alt="CropSense logo">
        <span>CropSense</span>
    </div>
    <div class="topnav-links">{nav_links_html}</div>
</div>
""", unsafe_allow_html=True)

if not MODEL_READY:
    st.warning("Models not trained yet. Run `python train_model.py` in the terminal first.",
               icon=":material/warning:")

# Route
if current == "home":
    from pages_impl import home
    home.render(MODEL_READY)
elif current == "single":
    from pages_impl import single_prediction
    single_prediction.render(MODEL_READY)
elif current == "batch":
    from pages_impl import batch_prediction
    batch_prediction.render(MODEL_READY)
elif current == "performance":
    from pages_impl import performance
    performance.render(MODEL_READY)
elif current == "about":
    from pages_impl import about
    about.render()
