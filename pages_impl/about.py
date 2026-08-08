"""About / methodology page."""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from utils.styles import page_header

MODELS_DIR = Path(__file__).parent.parent / "models"

MODEL_ORDER = [
    "Logistic Regression", "Decision Tree", "Random Forest", "KNN",
    "Naive Bayes", "XGBoost", "First Stacking", "Second Stacking", "Voting",
]

MODEL_DESCRIPTIONS = {
    "Logistic Regression": "a linear baseline classifier",
    "Decision Tree": "a single decision-tree baseline",
    "Random Forest": "an ensemble of 200 decision trees",
    "KNN": "a k-nearest-neighbours classifier (k=5)",
    "Naive Bayes": "a Gaussian Naive Bayes classifier",
    "XGBoost": "a gradient-boosted tree ensemble",
    "First Stacking": "a stacking ensemble combining KNN, Random Forest, and Naive Bayes "
                       "through a Logistic Regression meta-learner",
    "Second Stacking": "a stacking ensemble combining KNN, Bagging, and Naive Bayes "
                        "through a Logistic Regression meta-learner",
    "Voting": "a soft-voting ensemble of Logistic Regression, XGBoost, Random Forest, "
              "Decision Tree, and Naive Bayes",
}


@st.cache_data(show_spinner=False)
def _champion() -> str:
    path = MODELS_DIR / "metrics.json"
    if not path.exists():
        return "the champion stacking ensemble"
    with open(path) as f:
        return json.load(f).get("champion", "the champion stacking ensemble")


def render() -> None:
    champ = _champion()
    champ_desc = MODEL_DESCRIPTIONS.get(champ, "")

    page_header("ℹ️", "About this project",
                "The methodology, models, and data behind CropSense.")

    st.markdown("""
    <div class="about-section">
        <h3>🌱&nbsp; What is CropSense?</h3>
        <p>
            CropSense is an explainable crop recommendation system that helps smallholder
            farmers choose the most suitable crop for their land based on seven soil and
            climate measurements. Unlike a black-box model, it explains the reasoning behind
            every recommendation so that farmers can trust and act on the result.
        </p>
    </div>

    <div class="about-section">
        <h3>⚙️&nbsp; How it works</h3>
        <p>The system runs a five-step pipeline every time a recommendation is made:</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="arch-flow">
        <div class="arch-step">7 inputs</div>
        <span class="arch-arrow">→</span>
        <div class="arch-step">Scale</div>
        <span class="arch-arrow">→</span>
        <div class="arch-step">Fuse to 22</div>
        <span class="arch-arrow">→</span>
        <div class="arch-step green">{champ}</div>
        <span class="arch-arrow">→</span>
        <div class="arch-step">SHAP explain</div>
        <span class="arch-arrow">→</span>
        <div class="arch-step">Crop + reason</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="about-section">
        <h3>🧬&nbsp; Hybrid feature fusion</h3>
        <p>
            The seven original inputs are expanded into 22 features by combining three methods
            that each work on a different principle. Principal Component Analysis (PCA) creates
            five new components through mathematical transformation. Mutual Information (MI)
            selects the five most informative original features. Recursive Feature Elimination
            (RFE) selects five features using a Random Forest model. Together they give the
            classifier a richer representation than any single method alone.
        </p>
    </div>

    <div class="about-section">
        <h3>🏆&nbsp; The models</h3>
        <p>
            Nine classifiers are trained and compared on the same 22 fused features to prove
            which performs best:
        </p>
    </div>
    """, unsafe_allow_html=True)

    chips_html = "".join(
        f'<span class="model-chip" style="background:var(--primary); color:white;">{m} ★</span>'
        if m == champ else f'<span class="model-chip">{m}</span>'
        for m in MODEL_ORDER
    )
    st.markdown(f'<div style="margin: 0.5rem 0 1rem;">{chips_html}</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="about-section">
        <p>
            The <b>{champ}</b> model — {champ_desc} — is used as the production model because
            it achieved the highest accuracy on the held-out test set among all 9 models compared,
            with ties broken by 5-fold cross-validation stability. See the
            <a href="?nav=performance" target="_self">Dashboard</a> for the full comparison.
        </p>
    </div>

    <div class="about-section">
        <h3>🔍&nbsp; Explainability with SHAP</h3>
        <p>
            Rather than depend on the production model's own architecture — which may not be
            tree-based and can change between training runs — SHAP is applied to a dedicated
            Random Forest trained on the same 22 fused features. Its predictions closely track
            the production model's, so its SHAP values reliably explain what drives each
            recommendation — shown as a bar chart of the top contributing features on the
            prediction page.
        </p>
    </div>

    <div class="about-section">
        <h3>📊&nbsp; Dataset</h3>
        <p>
            The system is trained on the Kaggle Crop Recommendation dataset — 2,200 records,
            7 features (N, P, K, temperature, humidity, pH, rainfall), and 22 balanced crop
            classes with 100 samples each. This is the standard benchmark used across recent
            crop recommendation research, which makes the results directly comparable to
            published studies.
        </p>
    </div>

    <div class="about-section">
        <h3>🧰&nbsp; Technology stack</h3>
        <p>
            Built entirely with open-source Python tools: Streamlit for the web interface,
            scikit-learn and XGBoost for the models, SHAP for explanations, and Plotly for
            interactive charts. The whole system runs on a standard laptop.
        </p>
    </div>

    <div class="footer-row" style="margin-top:1.5rem;">
        <div>An Explainable Crop Recommendation System Using Hybrid Feature Fusion and Stacking Ensemble Learning</div>
        <div>Final Year Project</div>
    </div>
    """, unsafe_allow_html=True)
