"""Single prediction page — the core crop recommendation experience."""
from __future__ import annotations

import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.explainer import load_recommender
from utils.preprocessing import build_raw_input
from utils.styles import CROP_EMOJI, FEATURE_INPUTS, page_header

_BASE_LABELS = {key: label for key, label, *_ in FEATURE_INPUTS}


def _confidence_message(conf: float) -> None:
    pct = conf * 100
    if pct >= 90:
        st.success(f"**High confidence ({pct:.1f}%).** The model is very sure about this "
                   f"recommendation. The soil and climate values are a strong match for this crop.",
                   icon=":material/check_circle:")
    elif pct >= 60:
        st.info(f"**Moderate confidence ({pct:.1f}%).** This crop is a reasonable match, but "
                f"consider the alternative crops below before deciding.",
                icon=":material/info:")
    else:
        st.warning(f"**Low confidence ({pct:.1f}%).** The values don't strongly match any single "
                   f"crop. Please double-check your inputs, or consult an agriculture extension "
                   f"officer before making a decision.",
                   icon=":material/warning:")


def _probability_chart(top_crops):
    crops = [c.capitalize() for c, _ in top_crops][::-1]
    probs = [p * 100 for _, p in top_crops][::-1]
    colors = ["#3f7d52" if i == len(crops) - 1 else "#a9cbb2" for i in range(len(crops))]
    fig = go.Figure(go.Bar(
        x=probs, y=crops, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{p:.1f}%" for p in probs], textposition="outside",
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        height=260, margin=dict(l=10, r=40, t=10, b=10),
        xaxis=dict(range=[0, 105], showgrid=True, gridcolor="#eee3dc", title=None, ticksuffix="%"),
        yaxis=dict(title=None),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#6b6862", size=13),
    )
    return fig


def _friendly_feature_label(feat: str) -> str:
    """Turn an internal feature code (e.g. 'RFE_rainfall') into a farmer-readable label."""
    if feat.startswith("PCA_"):
        return f"Combined soil/climate pattern {feat.split('_', 1)[1]}"
    if feat.startswith("MI_"):
        key = feat.split("_", 1)[1]
        return f"{_BASE_LABELS.get(key, key)} (MI)"
    if feat.startswith("RFE_"):
        key = feat.split("_", 1)[1]
        return f"{_BASE_LABELS.get(key, key)} (RFE)"
    return _BASE_LABELS.get(feat, feat)


def _shap_narrative(shap_factors, crop: str) -> str:
    """Plain-language summary of the SHAP factors, grouped by underlying soil/climate value."""
    groups: dict[str, dict] = {}
    for feat, val in shap_factors:
        if feat.startswith("PCA_"):
            key, label = "_pattern", "the combined soil & climate pattern"
        elif feat.startswith(("MI_", "RFE_")):
            key = feat.split("_", 1)[1]
            label = _BASE_LABELS.get(key, key)
        else:
            key, label = feat, _BASE_LABELS.get(feat, feat)
        g = groups.setdefault(key, {"label": label, "value": 0.0})
        g["value"] += val

    ranked = sorted(groups.values(), key=lambda g: abs(g["value"]), reverse=True)
    favor = [g for g in ranked if g["value"] > 0][:2]
    against = [g for g in ranked if g["value"] < 0][:1]

    sentences = []
    if favor:
        names = " and ".join(f"**{g['label'].lower()}**" for g in favor)
        sentences.append(
            f"Your {names} matched well with what **{crop}** typically needs — "
            f"these were the biggest reasons it was recommended."
        )
    if against:
        names = " and ".join(f"**{g['label'].lower()}**" for g in against)
        sentences.append(
            f"Your {names} was a little outside the usual range for **{crop}**, which pulled "
            f"slightly against the recommendation — though not enough to change it."
        )
    if not sentences:
        sentences.append(
            f"Your soil and climate values were an even, balanced match for **{crop}** overall."
        )
    return " ".join(sentences)


def _shap_chart(shap_factors):
    """Horizontal diverging bar chart of SHAP contributions."""
    feats = [_friendly_feature_label(f) for f, _ in shap_factors][::-1]
    vals = [v for _, v in shap_factors][::-1]
    colors = ["#3f7d52" if v >= 0 else "#b5563f" for v in vals]
    fig = go.Figure(go.Bar(
        x=vals, y=feats, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        height=340, margin=dict(l=10, r=20, t=10, b=10, pad=6),
        xaxis=dict(title="SHAP value (impact on prediction)", zeroline=True,
                   zerolinecolor="#d8d4ca", zerolinewidth=1.5,
                   showgrid=True, gridcolor="#eee3dc"),
        yaxis=dict(title=None),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#6b6862", size=13),
    )
    return fig


def render(model_ready: bool) -> None:
    page_header("🌾", "Crop recommendation",
                "Enter your soil and climate values, then get an explained recommendation.")

    if not model_ready:
        st.markdown('<div class="empty-state-card"><div class="icon">⚙️</div>'
                    '<b>Models not trained yet.</b><br>Run <code>python train_model.py</code> first.</div>',
                    unsafe_allow_html=True)
        return

    left, right = st.columns([1, 1.15], gap="large")

    # ---- Input panel ---- #
    with left:
        with st.container(border=True):
            st.markdown("#### Your land's values")
            values = {}
            for key, label, lo, hi, default, step, unit, help_txt in FEATURE_INPUTS:
                unit_lbl = f" ({unit})" if unit else ""
                values[key] = st.slider(
                    f"{label}{unit_lbl}", min_value=lo, max_value=hi,
                    value=default, step=step, help=help_txt,
                )
            predict = st.button("Get recommendation", type="primary", width="stretch",
                                 icon=":material/spa:")

        st.markdown("""
        <div class="info-note" style="margin-top:1rem;">
            <b>Where to get these values?</b><br>
            Soil values (N, P, K, pH) — from a Department of Agriculture soil test (free) or a
            private lab. Climate values (temperature, humidity, rainfall) — from the
            MetMalaysia website for your district.
        </div>
        """, unsafe_allow_html=True)

    # ---- Results panel ---- #
    with right:
        if not predict and "last_result" not in st.session_state:
            st.markdown('<div class="empty-state-card"><div class="icon">🌾</div>'
                        '<b>Your recommendation will appear here.</b><br>'
                        'Set your values on the left and press the button.</div>',
                        unsafe_allow_html=True)
            return

        if predict:
            with st.spinner("Analysing your soil and climate values..."):
                start = time.time()
                rec = load_recommender()
                X_raw = build_raw_input(**values)
                st.session_state["last_result"] = rec.predict_and_explain(X_raw)
                # Prediction is near-instant; keep the spinner up briefly so the user sees it work.
                time.sleep(max(0.0, 1.2 - (time.time() - start)))

        result = st.session_state["last_result"]
        crop = result["crop"]
        conf = result["confidence"]
        emoji = CROP_EMOJI.get(crop, "🌱")

        # result card
        st.markdown(f"""
        <div class="result-card-dark">
            <div class="result-emoji">{emoji}</div>
            <div>
                <div class="result-crop-name">{crop}</div>
                <div class="result-crop-sub">Recommended crop for your land</div>
            </div>
            <div class="result-confidence">
                <div class="value">{conf * 100:.1f}%</div>
                <div class="label">confidence</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        _confidence_message(conf)

        # tabs: alternatives + explanation
        with st.container(border=True):
            tab1, tab2 = st.tabs([":material/bar_chart: Probability", ":material/search: Why this crop?"])

            with tab1:
                st.markdown("Top 5 candidate crops by probability:")
                st.plotly_chart(_probability_chart(result["top_crops"]), width="stretch",
                                config={"displayModeBar": False})

            with tab2:
                st.markdown(
                    f'<div class="shap-narrative">🌱&nbsp; {_shap_narrative(result["shap_factors"], crop)}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown("Which values drove this recommendation:")
                st.plotly_chart(_shap_chart(result["shap_factors"]), width="stretch",
                                config={"displayModeBar": False})
                st.caption("Green bars push toward the recommended crop; red bars push away. "
                           "MI/RFE mean the same value picked up by a different feature-selection method; "
                           "PCA is a blended pattern across several values.")
