"""Model performance dashboard — metrics, comparisons, confusion matrix."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from utils.styles import page_header

MODELS_DIR = Path(__file__).parent.parent / "models"

MODEL_ORDER = [
    "Logistic Regression", "Decision Tree", "Random Forest", "KNN",
    "Naive Bayes", "XGBoost", "First Stacking", "Second Stacking", "Voting",
]


@st.cache_data(show_spinner=False)
def _load_metrics():
    with open(MODELS_DIR / "metrics.json") as f:
        return json.load(f)


def _comparison_chart(models: dict, champ: str):
    names = [m for m in MODEL_ORDER if m in models]
    accs = [models[m]["accuracy"] * 100 for m in names]
    colors = ["#3f7d52" if m == champ else "#a9cbb2" for m in names]
    fig = go.Figure(go.Bar(
        x=names, y=accs, marker=dict(color=colors),
        text=[f"{a:.2f}%" for a in accs], textposition="outside",
        hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=20, b=80),
        yaxis=dict(range=[min(accs) - 2, 101], title="Accuracy (%)",
                   showgrid=True, gridcolor="#eee"),
        xaxis=dict(tickangle=-40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#1c1b19"),
    )
    return fig


def _confusion_heatmap(cm, classes):
    z = np.array(cm)
    fig = go.Figure(go.Heatmap(
        z=z, x=classes, y=classes, colorscale="Greens",
        hovertemplate="True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>",
        showscale=True,
    ))
    fig.update_layout(
        height=560, margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(title="Predicted", tickangle=-90, tickfont=dict(size=9)),
        yaxis=dict(title="Actual", autorange="reversed", tickfont=dict(size=9)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#1c1b19"),
    )
    return fig


def render(model_ready: bool) -> None:
    page_header("📊", "Model performance",
                "How the 9 models compare, and how the champion behaves on held-out data.")

    if not model_ready or not (MODELS_DIR / "metrics.json").exists():
        st.markdown('<div class="empty-state-card"><div class="icon">📊</div>'
                    '<b>No metrics yet.</b><br>Run <code>python train_model.py</code> to generate results.</div>',
                    unsafe_allow_html=True)
        return

    metrics = _load_metrics()
    models = metrics["models"]
    champ = metrics.get("champion") or max(
        models, key=lambda name: (models[name]["accuracy"], models[name].get("cv_mean", 0))
    )
    champ_m = models[champ]

    # champion banner
    st.markdown(f"""
    <div class="champion-banner">
        <div class="champion-trophy">🏆</div>
        <div>
            <div class="champion-banner-label">Champion model</div>
            <div class="champion-banner-name">{champ}</div>
            <div class="champion-banner-sub">Highest test accuracy, tie-broken by cross-validation stability</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # champion metric cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{champ_m['accuracy'] * 100:.2f}%")
    c2.metric("Precision", f"{champ_m['precision'] * 100:.2f}%")
    c3.metric("Recall", f"{champ_m['recall'] * 100:.2f}%")
    c4.metric("F1 score", f"{champ_m['f1'] * 100:.2f}%")

    if "cv_mean" in champ_m:
        st.caption(f"5-fold cross-validation accuracy: "
                   f"{champ_m['cv_mean'] * 100:.2f}% ± {champ_m['cv_std'] * 100:.2f}%  "
                   f"(low variance means stable performance).")

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        ":material/bar_chart: Model comparison",
        ":material/grid_on: Confusion matrix",
        ":material/table_chart: Full metrics",
    ])

    with tab1:
        st.markdown("Accuracy across all 9 models on the held-out test set:")
        st.plotly_chart(_comparison_chart(models, champ), width="stretch",
                        config={"displayModeBar": False})
        st.caption(f"🏆 highlights **{champ}** — the model with the highest test accuracy "
                   "(ties broken by cross-validation mean).")

    with tab2:
        st.markdown(f"Confusion matrix for {champ} across all 22 crop classes:")
        st.plotly_chart(
            _confusion_heatmap(metrics["confusion_matrix"], metrics["class_names"]),
            width="stretch", config={"displayModeBar": False})
        st.caption("The strong diagonal shows most crops are predicted correctly.")

    with tab3:
        st.markdown("Full results for every model (test set + cross-validation):")
        rows = []
        for m in MODEL_ORDER:
            if m not in models:
                continue
            r = models[m]
            rows.append({
                "Model": m,
                "Accuracy": f"{r['accuracy'] * 100:.2f}%",
                "Precision": f"{r['precision'] * 100:.2f}%",
                "Recall": f"{r['recall'] * 100:.2f}%",
                "F1": f"{r['f1'] * 100:.2f}%",
                "CV Mean": f"{r.get('cv_mean', 0) * 100:.2f}%",
                "CV Std": f"±{r.get('cv_std', 0) * 100:.2f}%",
            })
        st.dataframe(rows, width="stretch", hide_index=True)

        st.markdown(f"""
        <div class="info-note" style="margin-top:1rem;">
            <b>Training set:</b> {metrics['n_train']} samples &nbsp;·&nbsp;
            <b>Test set:</b> {metrics['n_test']} samples &nbsp;·&nbsp;
            <b>Fused features:</b> {metrics['n_fused_features']} &nbsp;·&nbsp;
            <b>PCA variance captured:</b> {metrics.get('pca_explained_variance', 0) * 100:.1f}%
        </div>
        """, unsafe_allow_html=True)
