"""Batch prediction page — upload a CSV and predict crops for many rows."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.explainer import load_recommender
from utils.preprocessing import BASE_FEATURES, dataframe_to_raw
from utils.styles import CROP_EMOJI, page_header


def _sample_csv() -> str:
    sample = pd.DataFrame([
        {"N": 90, "P": 42, "K": 43, "temperature": 21.0, "humidity": 82.0, "ph": 6.5, "rainfall": 203.0},
        {"N": 20, "P": 130, "K": 200, "temperature": 23.0, "humidity": 92.0, "ph": 6.0, "rainfall": 110.0},
        {"N": 40, "P": 70, "K": 80, "temperature": 26.0, "humidity": 55.0, "ph": 7.2, "rainfall": 90.0},
    ])
    return sample.to_csv(index=False)


def render(model_ready: bool) -> None:
    page_header("📋", "Batch prediction",
                "Upload a CSV with many rows to get recommendations for all of them at once.")

    if not model_ready:
        st.markdown('<div class="empty-state-card"><div class="icon">⚙️</div>'
                    '<b>Models not trained yet.</b><br>Run <code>python train_model.py</code> first.</div>',
                    unsafe_allow_html=True)
        return

    st.markdown(f"""
    <div class="info-note">
        Your CSV must contain these 7 columns (exact names):
        <b>{', '.join(BASE_FEATURES)}</b>. Each row is one plot of land.
    </div>
    """, unsafe_allow_html=True)

    st.download_button("Download sample CSV", data=_sample_csv(), icon=":material/download:",
                       file_name="sample_batch_input.csv", mime="text/csv")

    uploaded = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded is None:
        st.markdown('<div class="empty-state-card" style="margin-top:1rem;"><div class="icon">📄</div>'
                    '<b>No file uploaded yet.</b><br>Upload a CSV or download the sample above to try it.</div>',
                    unsafe_allow_html=True)
        return

    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read the file: {e}", icon=":material/error:")
        return

    missing = [c for c in BASE_FEATURES if c not in df.columns]
    if missing:
        st.error(f"Your CSV is missing these required columns: {', '.join(missing)}",
                 icon=":material/error:")
        return

    df = df.dropna(subset=BASE_FEATURES).reset_index(drop=True)
    if len(df) == 0:
        st.warning("No valid rows found after removing empty values.", icon=":material/warning:")
        return

    rec = load_recommender()
    X_raw = dataframe_to_raw(df)
    labels, proba = rec.predict(X_raw)
    confidences = proba.max(axis=1)

    out = df.copy()
    out.insert(0, "Row", out.index + 1)
    out["Recommended Crop"] = [str(c).capitalize() for c in labels]
    out["Confidence (%)"] = confidences * 100

    st.markdown(f"#### Results — {len(out)} rows")

    # quick summary metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows processed", len(out))
    c2.metric("Unique crops", out["Recommended Crop"].nunique())
    c3.metric("Avg. confidence", f"{confidences.mean() * 100:.1f}%")

    # crop mix — a compact chip summary. With many rows and few repeats, a bar per
    # crop is mostly bars of length 1-2 and doesn't carry much signal on its own.
    crop_counts = out["Recommended Crop"].value_counts()
    chips = "".join(
        f'<span class="model-chip">{CROP_EMOJI.get(c.lower(), "🌱")} {c}'
        f'{f" ×{n}" if n > 1 else ""}</span>'
        for c, n in crop_counts.items()
    )
    st.markdown(f"**Crops recommended:** {chips}", unsafe_allow_html=True)

    GOOD, CAUTION, REVIEW = "#3f7d52", "#c99a3f", "#b5563f"

    def _bucket_color(v: float) -> str:
        if v >= 75:
            return GOOD
        if v >= 50:
            return CAUTION
        return REVIEW

    def _status_label(v: float) -> str:
        if v >= 75:
            return "🟢 High"
        if v >= 50:
            return "🟡 Moderate"
        return "🔴 Review"

    out["Status"] = out["Confidence (%)"].apply(_status_label)

    # model confidence per row — this is the part worth a closer look: which
    # recommendations is the model least sure about, and might deserve a manual check.
    n_show = min(len(out), 20)
    shown = out.nsmallest(n_show, "Confidence (%)").sort_values("Confidence (%)", ascending=False)

    labels = [f"{CROP_EMOJI.get(c.lower(), '🌱')} {c} · Row {r}"
              for r, c in zip(shown["Row"], shown["Recommended Crop"])]
    colors = [_bucket_color(v) for v in shown["Confidence (%)"]]

    fig = go.Figure(go.Bar(
        x=shown["Confidence (%)"], y=labels,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.1f}%" for v in shown["Confidence (%)"]], textposition="outside",
        hovertemplate="%{y}: %{x:.1f}% confidence<extra></extra>",
    ))
    fig.update_layout(
        height=max(200, 30 * len(shown) + 60),
        margin=dict(l=10, r=40, t=10, b=10),
        xaxis=dict(title="Model confidence", range=[0, 105], showgrid=True, gridcolor="#eee3dc"),
        yaxis=dict(title=None),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#6b6862", size=13),
    )
    with st.container(border=True):
        title = ("**Rows worth double-checking — lowest model confidence:**" if len(out) > n_show
                  else "**Model confidence per row, lowest first:**")
        st.markdown(title)
        st.markdown(
            '<span style="font-size:0.82rem;color:var(--text-muted);">'
            f'<span style="color:{GOOD};">●</span> ≥75% high &nbsp;'
            f'<span style="color:{CAUTION};">●</span> 50–75% moderate &nbsp;'
            f'<span style="color:{REVIEW};">●</span> below 50% — worth reviewing</span>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        if len(out) > n_show:
            st.caption(f"Showing the {n_show} lowest-confidence rows out of {len(out)} total. "
                       "Every row's status is in the table below (sort by Confidence to find the rest).")

    st.dataframe(
        out, width="stretch", hide_index=True,
        column_config={"Confidence (%)": st.column_config.NumberColumn("Confidence (%)", format="%.1f%%")},
    )

    csv_bytes = out.to_csv(index=False).encode("utf-8")
    st.download_button("Download results as CSV", data=csv_bytes, icon=":material/download:",
                       file_name="crop_recommendations.csv", mime="text/csv",
                       type="primary")
