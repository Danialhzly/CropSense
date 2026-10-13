"""Check my crop page — the farmer picks a crop, and the system says what to fix and what the risks are."""
from __future__ import annotations

import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from utils.explainer import load_recommender
from utils.preprocessing import BASE_FEATURES, build_raw_input
from utils.styles import CROP_EMOJI, FEATURE_INPUTS, page_header

ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "Crop_recommendation.csv"

# A crop's "typical range" is the middle 90% of its samples in the training data,
# so a handful of extreme samples do not stretch the range.
LOW_Q, HIGH_Q = 0.05, 0.95

SOIL = ["N", "P", "K", "ph"]                        # can be corrected with fertiliser / lime
CLIMATE = ["temperature", "humidity", "rainfall"]   # cannot be corrected, so reported as risks

LABELS = {key: label for key, label, *_ in FEATURE_INPUTS}
UNITS = {key: unit for key, _, _, _, _, _, unit, _ in FEATURE_INPUTS}

# General agronomic guidance per soil value and direction. Deliberately names the
# type of input only, never a dose: doses depend on farm size and local soil tests.
SOIL_ADVICE = {
    ("N", "low"): "Add a **nitrogen fertiliser** (for example urea).",
    ("N", "high"): "**Hold back nitrogen fertiliser** this season. Too much nitrogen causes "
                   "excessive leafy growth and weak stems.",
    ("P", "low"): "Add a **phosphate fertiliser** (for example triple superphosphate or rock phosphate).",
    ("P", "high"): "**Skip phosphate fertiliser** this season. Excess phosphorus can block the "
                   "uptake of other nutrients.",
    ("K", "low"): "Add a **potash fertiliser** (for example muriate of potash, MOP).",
    ("K", "high"): "**Skip potash fertiliser** this season. Excess potassium can reduce the "
                   "uptake of magnesium and calcium.",
    ("ph", "low"): "Your soil is **too acidic**. Apply **agricultural lime** (for example ground "
                   "magnesium limestone, GML) to raise the pH.",
    ("ph", "high"): "Your soil is **too alkaline**. Apply **elemental sulphur** or an acidifying "
                    "fertiliser (such as ammonium sulphate) to lower the pH.",
}

CLIMATE_RISK = {
    ("temperature", "low"): "It is **cooler** than {crop} usually grows in. Expect slower growth and a lower yield.",
    ("temperature", "high"): "It is **hotter** than {crop} usually grows in. Heat stress can reduce flowering and yield.",
    ("humidity", "low"): "The air is **drier** than {crop} usually needs. Plants lose water faster, so watch for wilting.",
    ("humidity", "high"): "The air is **more humid** than {crop} usually needs. This raises the risk of fungal disease.",
    ("rainfall", "low"): "There is **less rain** than {crop} needs. Plan irrigation, or the crop may suffer water stress.",
    ("rainfall", "high"): "There is **more rain** than {crop} needs. There is a risk of waterlogging and root rot, "
                          "so make sure the field drains well.",
}

PAGE_CSS = """
<style>
.cc-step { font-size: 0.78rem; font-weight: 700; color: var(--primary); text-transform: uppercase;
           letter-spacing: 0.06em; margin: 1.4rem 0 0.4rem; }
.cc-score-row { display: flex; gap: 0.8rem; flex-wrap: wrap; margin-bottom: 0.6rem; }
.cc-score { flex: 1 1 160px; background: white; border: 1px solid var(--border); border-radius: 14px;
            padding: 0.9rem 1.1rem; }
.cc-score .big { font-size: 1.7rem; font-weight: 800; color: var(--text-primary); }
.cc-score .small { font-size: 0.82rem; color: var(--text-muted); }
.cc-item { background: white; border: 1px solid var(--border); border-left: 4px solid var(--primary);
           border-radius: 10px; padding: 0.75rem 1rem; margin-bottom: 0.55rem; font-size: 0.93rem;
           line-height: 1.5; color: var(--text-primary); }
.cc-item.risk { border-left-color: #c99a3f; }
.cc-item .meta { display: block; font-size: 0.8rem; color: var(--text-muted); margin-top: 0.2rem; }
.cc-badge { display: inline-block; white-space: nowrap; border-radius: 999px; padding: 0.25rem 0.8rem; font-size: 0.8rem;
            font-weight: 700; }
.cc-badge.low { background: #eaf3ec; color: #2f6140; }
.cc-badge.medium { background: #faf3e4; color: #8a6414; }
.cc-badge.high { background: #f7e9e5; color: #963f2b; }
</style>
"""


@st.cache_data(show_spinner=False)
def crop_ranges() -> pd.DataFrame:
    """Typical (5th-95th percentile) range of every feature, for every crop."""
    df = pd.read_csv(DATA_PATH)
    q = df.groupby("label")[BASE_FEATURES].quantile([LOW_Q, HIGH_Q]).unstack()
    return q  # columns: (feature, quantile)


def _bold(text: str) -> str:
    """The advice text uses **bold**; convert it to HTML because it is rendered inside HTML cards."""
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)


def _fmt(key: str, v: float) -> str:
    unit = UNITS[key]
    num = f"{v:.1f}" if key in ("ph", "temperature", "humidity") else f"{v:.0f}"
    return f"{num} {unit}".strip()


def _check(crop: str, values: dict) -> dict:
    """Compare the farmer's values with the crop's typical range and score the land with the model."""
    rec = load_recommender()
    ranges = crop_ranges().loc[crop]
    idx = rec.class_names.index(crop)

    rows, fixed = [], dict(values)
    for key in BASE_FEATURES:
        lo, hi, v = float(ranges[(key, LOW_Q)]), float(ranges[(key, HIGH_Q)]), values[key]
        status = "low" if v < lo else "high" if v > hi else "ok"
        rows.append({"key": key, "value": v, "lo": lo, "hi": hi, "status": status})
        if key in SOIL and status != "ok":
            fixed[key] = min(max(v, lo), hi)   # smallest change that brings it into range

    _, proba_now = rec.predict(build_raw_input(**values))
    _, proba_fixed = rec.predict(build_raw_input(**fixed))
    now, after = float(proba_now[0][idx]), float(proba_fixed[0][idx])

    # The model score drives the risk, but climate problems cannot be fixed by the farmer,
    # so each one raises the risk: one climate problem means at least medium, two or more mean high.
    n_climate = sum(1 for r in rows if r["key"] in CLIMATE and r["status"] != "ok")
    if now >= 0.6:
        risk = "low"
    elif after >= 0.6:
        risk = "medium"
    else:
        risk = "high"
    if n_climate >= 2:
        risk = "high"
    elif n_climate == 1 and risk == "low":
        risk = "medium"

    order = np.argsort(proba_now[0])[::-1]
    alternatives = [(rec.class_names[i], float(proba_now[0][i])) for i in order if rec.class_names[i] != crop][:3]
    return {"crop": crop, "rows": rows, "now": now, "after": after, "risk": risk,
            "alternatives": alternatives, "fixed": fixed}


def _render_result(r: dict) -> None:
    crop = r["crop"]
    name = crop.capitalize()
    emoji = CROP_EMOJI.get(crop, "🌱")
    soil_issues = [x for x in r["rows"] if x["key"] in SOIL and x["status"] != "ok"]
    climate_issues = [x for x in r["rows"] if x["key"] in CLIMATE and x["status"] != "ok"]

    risk_text = {
        "low": ("Low risk", f"Your land suits {name} well."
                + (" The soil fixes below will make it even better." if soil_issues else "")),
        "medium": ("Medium risk", f"{name} can work here, but follow the advice and watch the risks below."),
        "high": ("High risk", f"Your climate or soil is a poor match for {name}. See the risks below."),
    }[r["risk"]]

    st.markdown(f"""
    <div class="result-card-dark">
        <div class="result-emoji">{emoji}</div>
        <div>
            <div class="result-crop-name">{name}</div>
            <div class="result-crop-sub">{risk_text[1]}</div>
        </div>
        <div class="result-confidence">
            <span class="cc-badge {r['risk']}">{risk_text[0]}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Suitability now vs after fixing the soil
    st.markdown('<div class="cc-step">1 · How well your land suits this crop</div>', unsafe_allow_html=True)
    after_note = ("after the soil fixes below" if soil_issues else "no soil changes needed")
    st.markdown(f"""
    <div class="cc-score-row">
        <div class="cc-score"><div class="big">{r['now'] * 100:.1f}%</div>
            <div class="small">Suitability now</div></div>
        <div class="cc-score"><div class="big">{r['after'] * 100:.1f}%</div>
            <div class="small">Suitability {after_note}</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Suitability is the trained model's probability for this crop, given your values. "
               "The second score re-runs the model with your soil values moved into the crop's typical range.")

    # 2. Soil: what to improve
    st.markdown('<div class="cc-step">2 · What to improve in your soil</div>', unsafe_allow_html=True)
    if soil_issues:
        for x in soil_issues:
            st.markdown(f"""<div class="cc-item">{_bold(SOIL_ADVICE[(x['key'], x['status'])])}
                <span class="meta">{LABELS[x['key']]}: yours is {_fmt(x['key'], x['value'])};
                {name} usually grows at {_fmt(x['key'], x['lo'])} to {_fmt(x['key'], x['hi'])}.</span></div>""",
                        unsafe_allow_html=True)
    else:
        st.success(f"Your nitrogen, phosphorus, potassium and pH are already within the usual range for {name}.",
                   icon=":material/check_circle:")

    # 3. Climate: risks
    st.markdown('<div class="cc-step">3 · Risks you cannot fix with fertiliser</div>', unsafe_allow_html=True)
    if climate_issues:
        for x in climate_issues:
            text = CLIMATE_RISK[(x["key"], x["status"])].format(crop=name)
            st.markdown(f"""<div class="cc-item risk">{_bold(text)}
                <span class="meta">{LABELS[x['key']]}: yours is {_fmt(x['key'], x['value'])};
                {name} usually grows at {_fmt(x['key'], x['lo'])} to {_fmt(x['key'], x['hi'])}.</span></div>""",
                        unsafe_allow_html=True)
    else:
        st.success(f"Your temperature, humidity and rainfall suit {name}.", icon=":material/check_circle:")

    # 4. Alternatives when the chosen crop is risky
    if r["risk"] != "low":
        st.markdown('<div class="cc-step">4 · Crops that suit your land better right now</div>',
                    unsafe_allow_html=True)
        chips = "".join(
            f'<span class="model-chip">{CROP_EMOJI.get(c, "🌱")} {c.capitalize()} · {p * 100:.0f}%</span>'
            for c, p in r["alternatives"])
        st.markdown(chips, unsafe_allow_html=True)

    with st.expander("See all 7 values compared with the usual range"):
        status_text = {"ok": "✅ Within range", "low": "⬇️ Too low", "high": "⬆️ Too high"}
        table = pd.DataFrame([{
            "Value": LABELS[x["key"]],
            "Yours": _fmt(x["key"], x["value"]),
            f"Usual for {name}": f"{_fmt(x['key'], x['lo'])} – {_fmt(x['key'], x['hi'])}",
            "Status": status_text[x["status"]],
            "Type": "Soil (can be improved)" if x["key"] in SOIL else "Climate (risk)",
        } for x in r["rows"]])
        st.dataframe(table, hide_index=True, width="stretch")

    st.markdown("""
    <div class="info-note" style="margin-top:1rem;">
        This is general guidance based on the typical growing conditions in the dataset. It names the
        <b>type</b> of fertiliser, not the amount. Exact amounts depend on your farm size and soil test, so
        please confirm with a Department of Agriculture extension officer before applying.
    </div>
    """, unsafe_allow_html=True)


def render(model_ready: bool) -> None:
    page_header("🧪", "Check my crop",
                "Choose the crop you want to plant, enter your values, and see what to improve and what the risks are.")

    if not model_ready:
        st.markdown('<div class="empty-state-card"><div class="icon">⚙️</div>'
                    '<b>Models not trained yet.</b><br>Run <code>python train_model.py</code> first.</div>',
                    unsafe_allow_html=True)
        return

    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    crops = sorted(crop_ranges().index)
    left, right = st.columns([1, 1.15], gap="large")

    with left:
        with st.container(border=True):
            st.markdown("#### Your crop and land")
            crop = st.selectbox("Crop you want to plant", crops, index=crops.index("rice"),
                                format_func=lambda c: f"{CROP_EMOJI.get(c, '🌱')} {c.capitalize()}",
                                key="cc_crop")
            values = {}
            for key, label, lo, hi, default, step, unit, help_txt in FEATURE_INPUTS:
                unit_lbl = f" ({unit})" if unit else ""
                values[key] = st.slider(f"{label}{unit_lbl}", min_value=lo, max_value=hi, value=default,
                                        step=step, help=help_txt, key=f"cc_{key}")
            check = st.button("Check my crop", type="primary", width="stretch",
                              icon=":material/fact_check:", key="cc_button")

    with right:
        if not check and "cc_result" not in st.session_state:
            st.markdown('<div class="empty-state-card"><div class="icon">🧪</div>'
                        '<b>Your crop check will appear here.</b><br>'
                        'Choose a crop, set your values on the left and press the button.</div>',
                        unsafe_allow_html=True)
            return

        if check:
            with st.spinner(f"Checking your land for {crop}..."):
                start = time.time()
                st.session_state["cc_result"] = _check(crop, values)
                time.sleep(max(0.0, 1.2 - (time.time() - start)))

        _render_result(st.session_state["cc_result"])
