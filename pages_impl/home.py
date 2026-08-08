"""Home / landing page."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent.parent
FEATURE_FUSION_IMG = "app/static/feature-fusion.jpg"
STACKING_IMG = "app/static/stacking.jpg"
SHAP_IMG = "app/static/shap.jpg"

# Champion -> home-page description, kept consistent with about.py wording.
_CHAMP_COPY = {
    "First Stacking": (
        "A First Stacking ensemble combines KNN, Random Forest, and Naive Bayes through a "
        "Logistic Regression meta-learner. Because each model makes mistakes differently, "
        "combining them produces a more accurate and stable recommendation."
    ),
    "Second Stacking": (
        "A Second Stacking ensemble combines KNN, Bagging, and Naive Bayes through a "
        "Logistic Regression meta-learner. Because each model makes mistakes differently, "
        "combining them produces a more accurate and stable recommendation."
    ),
    "Voting": (
        "A soft-voting ensemble blends five diverse classifiers, averaging their "
        "probability estimates so that no single model's blind spot decides the outcome."
    ),
}
_FALLBACK_COPY = (
    "A stacking ensemble blends several diverse models through a meta-learner. Because each "
    "model makes mistakes differently, combining them produces a more accurate and stable "
    "recommendation."
)


@st.cache_data(show_spinner=False)
def _champion_info() -> tuple[str | None, float | None]:
    """(champion_name, test_accuracy) from metrics.json, or (None, None) pre-training."""
    path = ROOT / "models" / "metrics.json"
    if not path.exists():
        return None, None
    try:
        m = json.loads(path.read_text())
        champ = m.get("champion")
        acc = m.get("models", {}).get(champ, {}).get("accuracy")
        return champ, acc
    except Exception:
        return None, None


def render(model_ready: bool) -> None:
    champ, champ_acc = _champion_info()
    ensemble_text = _CHAMP_COPY.get(champ, _FALLBACK_COPY)
    ensemble_kicker = f"{champ} Ensemble" if champ else "Stacking Ensemble"

    live_chip = ""
    if champ and champ_acc:
        live_chip = (
            f'<div class="hero-live-chip"><span class="dot"></span>'
            f'Production model: <b>{champ}</b> · <b>{champ_acc * 100:.2f}%</b> test accuracy</div>'
        )

    # Hero
    st.markdown(f"""
    <div class="hero-wrap">
        <div class="hero-badge reveal">🌱 Explainable AI for Agriculture</div>
        <div class="hero-title reveal reveal-1">Know exactly <span class="accent">what to plant</span><br>and why.</div>
        <div class="hero-sub reveal reveal-2">
            CropSense recommends the best crop for your land using soil and climate data —
            and shows you the reason behind every recommendation, in plain terms.
        </div>
        <div class="hero-actions reveal reveal-3">
            <a class="btn-pill btn-pill-primary" href="?nav=single" target="_self">Get a recommendation →</a>
            <a class="btn-pill btn-pill-secondary" href="?nav=about" target="_self">How it works</a>
        </div>
        <div class="reveal reveal-4">{live_chip}</div>
    </div>
    """, unsafe_allow_html=True)

    # Stat strip — real headline number when metrics exist
    acc_stat = f"{champ_acc * 100:.1f}%" if champ_acc else "9"
    acc_label = "test accuracy" if champ_acc else "ML models compared"
    st.markdown(f"""
    <div class="stat-strip">
        <div class="stat-item"><div class="num">{acc_stat}</div><div class="lbl">{acc_label}</div></div>
        <div class="stat-item"><div class="num">22</div><div class="lbl">crop types</div></div>
        <div class="stat-item"><div class="num">9</div><div class="lbl">models benchmarked</div></div>
        <div class="stat-item"><div class="num">7</div><div class="lbl">soil &amp; climate inputs</div></div>
        <div class="stat-item"><div class="num">&lt;1s</div><div class="lbl">per recommendation</div></div>
    </div>
    """, unsafe_allow_html=True)

    # How it works — 3 steps
    st.markdown("""
    <div style="text-align:center; margin-top:0.5rem;">
        <span class="eyebrow">How it works</span>
        <h2 style="border:none; margin-top:0.2rem;">From soil test to decision in three steps</h2>
    </div>
    <div class="steps-grid">
        <div class="step-card reveal reveal-1">
            <div class="step-num">01</div>
            <div class="step-icon">🧪</div>
            <div class="step-title">Enter your values</div>
            <div class="step-text">Seven numbers: N, P, K and pH from a Department of Agriculture
            soil test, plus temperature, humidity and rainfall from MetMalaysia.</div>
        </div>
        <div class="step-card reveal reveal-2">
            <div class="step-num">02</div>
            <div class="step-icon">⚙️</div>
            <div class="step-title">The ensemble decides</div>
            <div class="step-text">Your inputs are fused into 22 engineered features and scored
            by a stacking ensemble benchmarked against 8 other models.</div>
        </div>
        <div class="step-card reveal reveal-3">
            <div class="step-num">03</div>
            <div class="step-icon">💡</div>
            <div class="step-title">Read the reasoning</div>
            <div class="step-text">SHAP breaks the decision down into the exact soil and climate
            values that drove it — in plain language, not a black box.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature rows
    st.markdown(f"""
    <div class="feature-row">
        <div class="feature-copy">
            <div class="feature-icon-badge">🧬</div>
            <div class="feature-index">01 / 03</div>
            <div class="feature-kicker">Hybrid Feature Fusion</div>
            <div class="feature-title">Three methods, one richer picture</div>
            <div class="feature-text">
                Your 7 inputs are expanded into 22 features using PCA, Mutual Information,
                and Recursive Feature Elimination. Combining methods that work on different
                principles gives the model a fuller view of your land than any single method could.
            </div>
        </div>
        <div class="feature-visual"><div class="feature-visual-card">
            <img src="{FEATURE_FUSION_IMG}" alt="PCA, Mutual Information, and RFE feature fusion">
        </div></div>
    </div>

    <div class="feature-row reverse">
        <div class="feature-copy">
            <div class="feature-icon-badge">🏆</div>
            <div class="feature-index">02 / 03</div>
            <div class="feature-kicker">{ensemble_kicker}</div>
            <div class="feature-title">Many models, one trusted answer</div>
            <div class="feature-text">{ensemble_text}</div>
        </div>
        <div class="feature-visual"><div class="feature-visual-card">
            <img src="{STACKING_IMG}" alt="Base learners stacked through a meta-learner">
        </div></div>
    </div>

    <div class="feature-row">
        <div class="feature-copy">
            <div class="feature-icon-badge">🔍</div>
            <div class="feature-index">03 / 03</div>
            <div class="feature-kicker">Explainable with SHAP</div>
            <div class="feature-title">See why, not just what</div>
            <div class="feature-text">
                Every recommendation comes with a SHAP explanation showing which of your
                inputs pushed the decision toward the recommended crop — so you can trust
                the result instead of taking it on faith.
            </div>
        </div>
        <div class="feature-visual"><div class="feature-visual-card">
            <img src="{SHAP_IMG}" alt="SHAP analytics highlighting the top contributing feature">
        </div></div>
    </div>
    """, unsafe_allow_html=True)

    # Tech credibility strip
    st.markdown("""
    <div class="tech-strip">
        <div class="tech-strip-label">Built with research-grade tooling</div>
        <span class="tech-chip">🐍 scikit-learn</span>
        <span class="tech-chip">⚡ XGBoost</span>
        <span class="tech-chip">🎯 SHAP</span>
        <span class="tech-chip">📊 Plotly</span>
        <span class="tech-chip">🎈 Streamlit</span>
        <span class="tech-chip">🗂️ Kaggle Crop Dataset</span>
    </div>
    """, unsafe_allow_html=True)

    # CTA banner
    st.markdown("""
    <div class="cta-banner">
        <h2>Ready to see what your land is telling you?</h2>
        <p>Enter seven values. Get a recommendation — and a reason.</p>
        <a class="btn-pill btn-pill-primary" href="?nav=single" target="_self">Get a recommendation →</a>
    </div>
    """, unsafe_allow_html=True)

    # Footer
    logo_b64 = base64.b64encode((ROOT / "assets" / "logo_32.png").read_bytes()).decode()
    st.markdown(f"""
    <div class="site-footer">
        <div class="site-footer-grid">
            <div>
                <div class="site-footer-brand">
                    <img src="data:image/png;base64,{logo_b64}" alt=""> CropSense
                </div>
                <div class="site-footer-tag">
                    An Explainable Crop Recommendation System using Hybrid Feature Fusion
                    and Stacking Ensemble Learning. Built as a Final Year Project.
                </div>
            </div>
            <div>
                <h5>Product</h5>
                <div class="site-footer-links">
                    <a href="?nav=single" target="_self">Single prediction</a>
                    <a href="?nav=batch" target="_self">Batch prediction</a>
                    <a href="?nav=performance" target="_self">Model performance</a>
                </div>
            </div>
            <div>
                <h5>Research</h5>
                <div class="site-footer-links">
                    <a href="?nav=about" target="_self">Methodology</a>
                    <a href="?nav=performance" target="_self">Benchmark results</a>
                </div>
            </div>
        </div>
        <div class="site-footer-bottom">
            <div>© 2026 CropSense — CSP600/CSP650 Final Year Project</div>
            <div>Powered by a stacking ensemble · Explained by SHAP</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
