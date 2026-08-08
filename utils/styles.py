"""Shared constants, feature metadata, and the global CSS for CropSense."""
from __future__ import annotations

import streamlit as st


def page_header(icon: str, title: str, subtitle: str) -> None:
    """Render the shared icon-badge + title + subtitle header used on every inner page."""
    st.markdown(f"""
    <div class="page-header">
        <div class="page-header-icon">{icon}</div>
        <div class="page-header-text">
            <h2>{title}</h2>
            <p>{subtitle}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


# Slider configuration for the 7 input features: (label, min, max, default, step, unit, help)
FEATURE_INPUTS = [
    ("N",           "Nitrogen (N)",     0.0,   140.0,  90.0,  1.0,  "mg/kg",
     "Soil nitrogen content. Get this from a soil test."),
    ("P",           "Phosphorus (P)",   5.0,   145.0,  42.0,  1.0,  "mg/kg",
     "Soil phosphorus content."),
    ("K",           "Potassium (K)",    5.0,   205.0,  43.0,  1.0,  "mg/kg",
     "Soil potassium content."),
    ("temperature", "Temperature",      8.0,   44.0,   21.0,  0.1,  "°C",
     "Average temperature. Available from MetMalaysia."),
    ("humidity",    "Humidity",         14.0,  100.0,  82.0,  0.1,  "%",
     "Relative humidity."),
    ("ph",          "Soil pH",          3.5,   10.0,   6.5,   0.1,  "",
     "Soil acidity/alkalinity. 7 is neutral."),
    ("rainfall",    "Rainfall",         20.0,  300.0,  203.0, 1.0,  "mm",
     "Rainfall in the growing period."),
]

# Emoji per crop for nicer result cards
CROP_EMOJI = {
    "rice": "🌾", "maize": "🌽", "chickpea": "🫘", "kidneybeans": "🫘",
    "pigeonpeas": "🫛", "mothbeans": "🫘", "mungbean": "🫛", "blackgram": "🫘",
    "lentil": "🫘", "pomegranate": "🍎", "banana": "🍌", "mango": "🥭",
    "grapes": "🍇", "watermelon": "🍉", "muskmelon": "🍈", "apple": "🍎",
    "orange": "🍊", "papaya": "🍈", "coconut": "🥥", "cotton": "🌱",
    "jute": "🌿", "coffee": "☕",
}

# The full green-themed CSS. Injected once in app.py.
# Font is loaded once via .streamlit/config.toml's theme.font — no @import needed here.
GLOBAL_CSS = """
<style>
:root {
    --primary: #3f7d52;
    --primary-hover: #2f6140;
    --gradient: linear-gradient(135deg, #3f7d52, #7fb587);
    --gradient-2: linear-gradient(135deg, #5a9c6d 0%, #3f7d52 55%, #2f6140 100%);
    --tint: #eef2ec;
    --gold: #c99a3f;
    --gold-bg: #faf3e4;
    --text-primary: #1c1b19;
    --text-secondary: #6b6862;
    --text-muted: #8a877e;
    --border: #e6e3dc;
    --border-input: #d8d4ca;
    --negative: #b5563f;
    --blue: #3f6d9e;
    --blue-bg: #eaf1f8;
    --bg-main: #faf9f6;
    --shadow-sm: 0 1px 3px rgba(28,27,25,0.06);
    --shadow-md: 0 10px 30px rgba(28,27,25,0.08);
    --shadow-lg: 0 24px 60px rgba(28,27,25,0.14);
    --shadow-glow: 0 20px 45px rgba(63,125,82,0.28);
    --radius-card: 18px;
    --radius-xl: 28px;
    --radius-pill: 999px;
    --ease: cubic-bezier(0.16, 1, 0.3, 1);
}

* { -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale; }
html { scroll-behavior: smooth; }
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-primary);
}
::selection { background: var(--primary); color: white; }
.stApp { background: var(--bg-main); }
[data-testid="stMainBlockContainer"] { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }
h1, h2, h3 { color: var(--text-primary); font-weight: 800; letter-spacing: -0.025em; }
h2 { padding-bottom: 0.35rem; border-bottom: 1px solid var(--border); margin-bottom: 1.1rem !important; }
p, li { color: var(--text-primary); }

.kicker {
    display: inline-block; text-transform: uppercase; letter-spacing: 0.08em;
    font-size: 0.78rem; font-weight: 700; color: var(--primary);
}
.pill-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: var(--tint); color: var(--primary);
    border-radius: var(--radius-pill); padding: 0.35rem 0.9rem;
    font-size: 0.85rem; font-weight: 600;
}
.card {
    background: white; border: 1px solid var(--border);
    border-radius: var(--radius-card); box-shadow: var(--shadow-sm);
    padding: 1.5rem;
}
.text-muted { color: var(--text-muted); }
.text-secondary { color: var(--text-secondary); }

/* hide default sidebar — we use a top navbar */
section[data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"] { display: none !important; }

/* hide Streamlit's default header/toolbar/status bar — we use a custom top navbar */
header[data-testid="stHeader"],
div[data-testid="stDecoration"],
div[data-testid="stToolbar"],
div[data-testid="stStatusWidget"] { display: none !important; }
[data-testid="stAppViewContainer"] { padding-top: 0 !important; }
[data-testid="stMainBlockContainer"] { padding-top: 0 !important; }

/* top navbar */
.topnav {
    display: grid; grid-template-columns: 1fr auto 1fr;
    align-items: center; gap: 1rem;
    padding: 0.9rem 1.2rem; margin: 0 0 1.4rem;
    position: sticky; top: 0; z-index: 999;
    background: rgba(250,249,246,0.72);
    backdrop-filter: blur(20px) saturate(180%);
    -webkit-backdrop-filter: blur(20px) saturate(180%);
    border-bottom: 1px solid var(--border);
    border-radius: 0 0 20px 20px;
}
.topnav-brand { display: flex; align-items: center; gap: 0.6rem; justify-self: start; }
.topnav-logo {
    width: 34px; height: 34px; object-fit: contain; flex-shrink: 0;
}
.topnav-brand span { font-weight: 800; font-size: 1.1rem; letter-spacing: -0.02em; color: var(--text-primary) !important; }
.topnav-links { display: flex; gap: 1.8rem; justify-self: center; }
.topnav-links a {
    color: var(--text-secondary); text-decoration: none; font-size: 0.95rem;
    font-weight: 500; transition: color .2s var(--ease); position: relative;
}
.topnav-links a::after {
    content: ""; position: absolute; left: 0; right: 0; bottom: -6px; height: 2px;
    background: var(--primary); border-radius: 2px; transform: scaleX(0);
    transition: transform .25s var(--ease);
}
.topnav-links a:hover { color: var(--text-primary); }
.topnav-links a.active { color: var(--primary); font-weight: 700; }
.topnav-links a.active::after { transform: scaleX(1); }
.topnav-cta { justify-self: end; }
@media (max-width: 900px) {
    .topnav { grid-template-columns: auto 1fr; }
    .topnav-links, .topnav-cta { display: none; }
}

/* buttons */
.stButton > button, .stDownloadButton > button {
    border-radius: var(--radius-pill); font-weight: 600; transition: all .2s var(--ease);
    border: 1px solid var(--border-input); padding: 0.5rem 1.4rem;
}
.stButton > button[kind="primary"] {
    background: var(--primary); border: none; box-shadow: var(--shadow-sm); color: white;
}
.stButton > button[kind="primary"]:hover {
    background: var(--primary-hover); transform: translateY(-2px) scale(1.01);
    box-shadow: 0 10px 22px rgba(63,125,82,0.32);
}
.stButton > button[kind="primary"]:active { transform: translateY(0) scale(0.99); }
.stButton > button:not([kind="primary"]):hover,
.stDownloadButton > button:hover { border-color: var(--primary); color: var(--primary); transform: translateY(-1px); }
a.btn-pill {
    display: inline-block; text-decoration: none; border-radius: var(--radius-pill);
    font-weight: 600; padding: 0.68rem 1.6rem; font-size: 0.95rem; transition: all .25s var(--ease);
}
a.btn-pill-primary { background: var(--primary); color: white !important; border: none; box-shadow: 0 8px 20px rgba(63,125,82,0.28); }
a.btn-pill-primary:hover { background: var(--primary-hover); transform: translateY(-2px); box-shadow: 0 12px 26px rgba(63,125,82,0.36); }
a.btn-pill-secondary { background: transparent; color: var(--text-primary) !important; border: 1px solid var(--border-input); }
a.btn-pill-secondary:hover { border-color: var(--primary); color: var(--primary) !important; transform: translateY(-2px); background: white; }

/* metric cards */
div[data-testid="stMetric"] {
    background: white; padding: 1.1rem 1.2rem; border-radius: var(--radius-card);
    box-shadow: var(--shadow-sm); border: 1px solid var(--border);
}
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; }

/* tabs */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 10px 10px 0 0; padding: 0.5rem 1.1rem; background: var(--tint); font-weight: 600;
}
.stTabs [aria-selected="true"] { background: var(--primary) !important; color: white !important; }

/* dataframes / expanders / alerts */
div[data-testid="stAlertContainer"] { border-radius: 12px; }
[data-testid="stDataFrame"], [data-testid="stTable"] {
    border-radius: 12px; overflow: hidden; border: 1px solid var(--border);
}
div[data-testid="stExpander"] {
    border-radius: 12px; border: 1px solid var(--border); box-shadow: var(--shadow-sm); overflow: hidden;
}

/* scrollbar */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-input); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: var(--primary); }

/* hero */
@keyframes fadeUp { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: translateY(0); } }
@keyframes auroraDrift {
    0%, 100% { transform: translate(0, 0) scale(1); }
    50% { transform: translate(3%, -4%) scale(1.08); }
}
.hero-wrap {
    position: relative; padding: 2.5rem 0 1rem; text-align: center;
    animation: fadeUp .7s var(--ease) both;
}
.hero-wrap::before, .hero-wrap::after {
    content: ""; position: absolute; z-index: -1; border-radius: 50%;
    filter: blur(110px); opacity: 0.28; animation: auroraDrift 14s ease-in-out infinite;
}
.hero-wrap::before {
    width: 720px; height: 720px; top: -320px; left: -8%;
    background: radial-gradient(circle, #7fb587, transparent 72%);
}
.hero-wrap::after {
    width: 680px; height: 680px; top: -280px; right: -8%;
    background: radial-gradient(circle, #c99a3f, transparent 72%);
    animation-delay: -7s;
}
.hero-badge {
    display: inline-flex; align-items: center; gap: 0.4rem; background: var(--tint);
    color: var(--primary); border-radius: 999px; padding: 0.4rem 1rem;
    font-size: 0.85rem; font-weight: 600; margin-bottom: 1.1rem;
    box-shadow: inset 0 0 0 1px rgba(63,125,82,0.12);
}
.hero-title {
    font-size: clamp(2.4rem, 6vw, 4.2rem); font-weight: 800; letter-spacing: -0.035em;
    line-height: 1.05; margin: 0.4rem 0 1rem; color: var(--text-primary) !important;
}
.hero-title .accent {
    background: var(--gradient-2); -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero-sub { max-width: 660px; margin: 0 auto 1.6rem; font-size: 1.12rem; color: var(--text-secondary); line-height: 1.55; }
.hero-actions { display: flex; justify-content: center; gap: 0.8rem; flex-wrap: wrap; margin-bottom: 2rem; }

/* stat strip */
.stat-strip {
    display: flex; justify-content: center; gap: 1.2rem; flex-wrap: wrap; padding: 1.6rem 1rem;
    background: white; border: 1px solid var(--border); border-radius: var(--radius-xl);
    box-shadow: var(--shadow-md); margin: 1rem 0 3rem;
    animation: fadeUp .7s var(--ease) .1s both;
}
.stat-item { text-align: center; padding: 0.4rem 1.3rem; flex: 1 1 140px; transition: transform .25s var(--ease); }
.stat-item:hover { transform: translateY(-3px); }
.stat-item .num { font-size: 2.1rem; font-weight: 800; color: var(--primary); letter-spacing: -0.02em; }
.stat-item .lbl { font-size: 0.85rem; color: var(--text-muted); }

/* feature rows */
.feature-row { display: flex; align-items: center; gap: 3.5rem; padding: 2.8rem 0; border-top: 1px solid var(--border); }
.feature-row.reverse { flex-direction: row-reverse; }
.feature-visual, .feature-copy { flex: 1 1 0; min-width: 0; }
.feature-icon-badge {
    width: 60px; height: 60px; border-radius: 16px; background: var(--tint);
    display: flex; align-items: center; justify-content: center; font-size: 1.7rem; margin-bottom: 1.1rem;
    box-shadow: inset 0 0 0 1px rgba(63,125,82,0.1);
}
.feature-kicker { color: var(--primary); font-weight: 700; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.5rem; }
.feature-title { font-size: clamp(1.4rem, 2.6vw, 2rem); font-weight: 800; letter-spacing: -0.02em; margin-bottom: 0.7rem; }
.feature-text { color: var(--text-secondary); font-size: 1.02rem; line-height: 1.6; }
.feature-visual-card {
    position: relative; background: var(--gradient-2); border-radius: var(--radius-xl); height: 280px;
    display: flex; align-items: center; justify-content: center; box-shadow: var(--shadow-glow);
    overflow: hidden; transition: transform .4s var(--ease), box-shadow .4s var(--ease);
}
.feature-visual-card:hover { transform: translateY(-6px) scale(1.015); box-shadow: 0 28px 55px rgba(63,125,82,0.36); }
.feature-visual-card::before {
    content: ""; position: absolute; inset: 0;
    background: radial-gradient(circle at 30% 20%, rgba(255,255,255,0.22), transparent 55%);
    pointer-events: none;
}
.feature-visual-card svg { position: relative; width: 82%; height: 82%; }
.feature-visual-card img { position: relative; width: 100%; height: 100%; object-fit: cover; }

/* cta banner + footer */
.cta-banner {
    position: relative; background: var(--text-primary); border-radius: var(--radius-xl);
    padding: 3.4rem 2rem; text-align: center; margin: 3rem 0 2rem; color: white; overflow: hidden;
    box-shadow: var(--shadow-lg);
}
.cta-banner::before {
    content: ""; position: absolute; inset: 0;
    background: radial-gradient(circle at 20% 0%, rgba(127,181,135,0.35), transparent 55%),
                radial-gradient(circle at 85% 100%, rgba(201,154,63,0.25), transparent 55%);
    pointer-events: none;
}
.cta-banner > * { position: relative; }
.cta-banner h2 { color: white; border: none; font-size: clamp(1.6rem, 3.4vw, 2.2rem); margin-bottom: 0.7rem !important; }
.cta-banner p { color: #c9c6bd; font-size: 1.05rem; margin-bottom: 1.6rem; }
.footer-row { display: flex; justify-content: space-between; align-items: center; padding-top: 1.5rem; border-top: 1px solid var(--border); color: var(--text-muted); font-size: 0.86rem; flex-wrap: wrap; gap: 0.6rem; }

/* prediction result cards */
.result-card-dark {
    background: var(--text-primary); border-radius: var(--radius-card);
    padding: 1.6rem 1.8rem; display: flex; align-items: center; gap: 1.1rem; margin-bottom: 1.2rem;
}
.result-emoji { font-size: 2.6rem; flex-shrink: 0; }
.result-crop-name { color: white; font-size: 1.7rem; font-weight: 800; letter-spacing: -0.02em; text-transform: capitalize; }
.result-crop-sub { color: #a8a49b; font-size: 0.85rem; }
.result-confidence { margin-left: auto; text-align: right; }
.result-confidence .value { color: #7fb587; font-size: 1.8rem; font-weight: 800; }
.result-confidence .label { color: #a8a49b; font-size: 0.78rem; }

.alt-crop-card {
    background: white; border: 1px solid var(--border); border-radius: 14px;
    padding: 0.8rem 1.1rem; display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem;
}
.alt-crop-card .name { font-weight: 700; font-size: 0.96rem; text-transform: capitalize; }
.alt-crop-card .pct { color: var(--text-muted); font-size: 0.86rem; }

/* SHAP factor rows */
.factor-row { display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.6rem; }
.factor-label { flex: 0 0 170px; font-size: 0.85rem; color: var(--text-secondary); font-weight: 500; }
.factor-bar-track { flex: 1 1 auto; height: 12px; background: var(--tint); border-radius: 999px; position: relative; }
.factor-bar-fill { height: 100%; border-radius: 999px; position: absolute; top: 0; }
.factor-tag { flex: 0 0 auto; font-size: 0.72rem; font-weight: 700; padding: 0.15rem 0.6rem; border-radius: 999px; text-align: center; min-width: 78px; }
.factor-tag.supports { background: #eaf3ec; color: var(--primary); }
.factor-tag.limiting { background: #f7e9e5; color: var(--negative); }

/* empty state */
.empty-state-card {
    border: 1.5px dashed var(--border-input); border-radius: var(--radius-card);
    background: white; padding: 3.5rem 2rem; text-align: center; color: var(--text-muted);
}
.empty-state-card .icon { font-size: 2.4rem; margin-bottom: 0.8rem; }

/* about page */
.about-section { padding: 1.6rem 0; border-top: 1px solid var(--border); }
.about-section:first-of-type { border-top: none; }
.about-section h3 { font-size: 1.25rem; margin-bottom: 0.6rem; }
.about-section p { color: var(--text-secondary); line-height: 1.7; font-size: 1rem; }
.arch-flow { display: flex; align-items: center; gap: 0.5rem; margin: 1.4rem 0; flex-wrap: wrap; }
.arch-step { background: white; border: 1px solid var(--border); border-radius: 12px; padding: 0.6rem 0.9rem; font-size: 0.82rem; font-weight: 700; text-align: center; }
.arch-step.green { background: var(--primary); color: white; border: none; }
.arch-arrow { color: var(--text-muted); font-size: 1.2rem; }
.model-chip { display: inline-block; background: var(--tint); color: var(--primary); border-radius: 999px; padding: 0.3rem 0.8rem; font-size: 0.8rem; font-weight: 700; margin: 0.2rem; }
.info-note {
    background: var(--blue-bg); border-left: 3px solid var(--blue); border-radius: 8px;
    padding: 0.9rem 1.1rem; font-size: 0.9rem; color: var(--text-secondary);
}
.shap-narrative {
    background: var(--tint); border-left: 3px solid var(--primary); border-radius: 8px;
    padding: 0.9rem 1.1rem; font-size: 0.95rem; line-height: 1.55;
    color: var(--text-primary) !important; margin-bottom: 1rem;
}
.shap-narrative strong, .shap-narrative b { color: var(--primary-hover); }

/* generic card hover (used by .card + result summaries) */
.card { transition: transform .25s var(--ease), box-shadow .25s var(--ease); }
.card:hover { transform: translateY(-2px); box-shadow: var(--shadow-md); }

/* page header — kicker + icon badge + title + subtitle, used on every inner page */
.page-header { display: flex; align-items: center; gap: 1rem; margin: 0.4rem 0 1.6rem; animation: fadeUp .6s var(--ease) both; }
.page-header-icon {
    width: 52px; height: 52px; border-radius: 16px; background: var(--gradient-2); flex-shrink: 0;
    display: flex; align-items: center; justify-content: center; font-size: 1.5rem;
    box-shadow: var(--shadow-glow);
}
.page-header-text h2 { border: none; margin: 0 !important; padding: 0; font-size: clamp(1.5rem, 3vw, 2rem); }
.page-header-text p { color: var(--text-secondary); margin: 0.2rem 0 0; font-size: 1.02rem; }

/* champion banner (performance dashboard) */
.champion-banner {
    position: relative; overflow: hidden; display: flex; align-items: center; gap: 1.2rem;
    background: var(--text-primary); border-radius: var(--radius-xl); padding: 1.6rem 2rem;
    margin-bottom: 1.6rem; box-shadow: var(--shadow-lg); color: white;
}
.champion-banner::before {
    content: ""; position: absolute; inset: 0;
    background: radial-gradient(circle at 15% 20%, rgba(127,181,135,0.4), transparent 55%),
                radial-gradient(circle at 90% 90%, rgba(201,154,63,0.28), transparent 55%);
    pointer-events: none;
}
.champion-banner > * { position: relative; }
.champion-trophy {
    width: 56px; height: 56px; border-radius: 16px; flex-shrink: 0; font-size: 1.7rem;
    background: rgba(255,255,255,0.1); display: flex; align-items: center; justify-content: center;
    box-shadow: inset 0 0 0 1px rgba(255,255,255,0.14);
}
.champion-banner-label { color: #a8a49b; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; }
.champion-banner-name { color: white; font-size: 1.5rem; font-weight: 800; letter-spacing: -0.02em; }
.champion-banner-sub { color: #c9c6bd; font-size: 0.92rem; margin-top: 0.15rem; }

/* feature row eyebrow numbering */
.feature-index {
    font-size: 0.78rem; font-weight: 700; color: var(--text-muted); letter-spacing: 0.08em;
    margin-bottom: 0.4rem;
}

/* sliders */
div[data-testid="stSlider"] { padding-top: 0.2rem; padding-bottom: 0.6rem; }
div[data-testid="stSlider"] label p { font-weight: 600; color: var(--text-primary) !important; }
div[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    box-shadow: 0 2px 8px rgba(63,125,82,0.35); transition: transform .18s var(--ease), box-shadow .18s var(--ease);
}
div[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"]:hover,
div[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"]:focus {
    transform: scale(1.15); box-shadow: 0 4px 14px rgba(63,125,82,0.45);
}
div[data-testid="stTickBar"] { font-size: 0.72rem; color: var(--text-muted); }

/* file uploader dropzone */
[data-testid="stFileUploaderDropzone"] {
    background: var(--tint) !important; border: 1.5px dashed var(--border-input) !important;
    border-radius: var(--radius-card) !important; transition: border-color .2s var(--ease), background .2s var(--ease);
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--primary) !important; background: #e5ede2 !important; }
[data-testid="stFileUploaderDropzone"] button { border-radius: var(--radius-pill) !important; }

/* selectbox / text input / number input polish */
div[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input {
    border-radius: 12px !important;
}
div[data-baseweb="select"] > div:focus-within,
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: var(--primary) !important; box-shadow: 0 0 0 1px var(--primary) !important;
}

/* checkbox / radio / toggle accent already inherit primaryColor via theme */

/* accessible focus ring on interactive elements */
a:focus-visible, button:focus-visible, [role="tab"]:focus-visible, [role="slider"]:focus-visible {
    outline: 2px solid var(--primary); outline-offset: 2px;
}

/* section eyebrow pill reused for standalone content blocks */
.eyebrow {
    display: inline-flex; align-items: center; gap: 0.4rem; font-size: 0.78rem; font-weight: 700;
    color: var(--primary); text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 0.5rem;
}

/* ---- v2 additions: staggered reveals, steps, tech strip, footer ---- */
.reveal   { animation: fadeUp .7s var(--ease) both; }
.reveal-1 { animation-delay: .05s; } .reveal-2 { animation-delay: .15s; }
.reveal-3 { animation-delay: .25s; } .reveal-4 { animation-delay: .35s; }

/* hero live-accuracy chip */
.hero-live-chip {
    display: inline-flex; align-items: center; gap: 0.45rem;
    background: white; border: 1px solid var(--border); border-radius: var(--radius-pill);
    padding: 0.4rem 1rem; font-size: 0.85rem; font-weight: 600; color: var(--text-secondary);
    box-shadow: var(--shadow-sm); margin-top: 0.4rem;
}
.hero-live-chip .dot {
    width: 8px; height: 8px; border-radius: 50%; background: var(--primary);
    box-shadow: 0 0 0 0 rgba(63,125,82,0.5); animation: pulseDot 2s infinite;
}
@keyframes pulseDot {
    0% { box-shadow: 0 0 0 0 rgba(63,125,82,0.45); }
    70% { box-shadow: 0 0 0 9px rgba(63,125,82,0); }
    100% { box-shadow: 0 0 0 0 rgba(63,125,82,0); }
}
.hero-live-chip b { color: var(--primary); font-weight: 800; }

/* how-it-works steps */
.steps-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin: 0.6rem 0 1rem; }
@media (max-width: 820px) { .steps-grid { grid-template-columns: 1fr; } }
.step-card {
    position: relative; background: white; border: 1px solid var(--border);
    border-radius: var(--radius-card); padding: 1.5rem 1.4rem 1.4rem;
    box-shadow: var(--shadow-sm); transition: transform .25s var(--ease), box-shadow .25s var(--ease);
}
.step-card:hover { transform: translateY(-4px); box-shadow: var(--shadow-md); }
.step-num {
    position: absolute; top: 1.1rem; right: 1.2rem; font-size: 2rem; font-weight: 800;
    color: var(--tint); -webkit-text-stroke: 1.5px rgba(63,125,82,0.35); letter-spacing: -0.03em;
}
.step-icon {
    width: 46px; height: 46px; border-radius: 13px; background: var(--tint);
    display: flex; align-items: center; justify-content: center; font-size: 1.3rem; margin-bottom: 0.9rem;
}
.step-title { font-weight: 800; font-size: 1.05rem; letter-spacing: -0.01em; margin-bottom: 0.35rem; }
.step-text  { color: var(--text-secondary); font-size: 0.93rem; line-height: 1.55; }

/* tech credibility strip */
.tech-strip {
    display: flex; justify-content: center; align-items: center; gap: 0.7rem; flex-wrap: wrap;
    padding: 1.2rem 1rem; margin: 0.4rem 0 0.6rem;
}
.tech-strip-label {
    flex-basis: 100%; text-align: center; font-size: 0.75rem; font-weight: 700;
    color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.3rem;
}
.tech-chip {
    display: inline-flex; align-items: center; gap: 0.45rem;
    background: white; border: 1px solid var(--border); border-radius: var(--radius-pill);
    padding: 0.45rem 1.05rem; font-size: 0.88rem; font-weight: 600; color: var(--text-secondary);
    transition: all .22s var(--ease);
}
.tech-chip:hover { border-color: var(--primary); color: var(--primary); transform: translateY(-2px); box-shadow: var(--shadow-sm); }

/* tech-company footer */
.site-footer { border-top: 1px solid var(--border); margin-top: 3rem; padding: 2.2rem 0 1.4rem; }
.site-footer-grid { display: grid; grid-template-columns: 1.4fr 1fr 1fr; gap: 2rem; margin-bottom: 1.6rem; }
@media (max-width: 820px) { .site-footer-grid { grid-template-columns: 1fr; } }
.site-footer-brand { display: flex; align-items: center; gap: 0.55rem; font-weight: 800; font-size: 1.05rem; margin-bottom: 0.5rem; }
.site-footer-brand img { width: 26px; height: 26px; }
.site-footer-tag { color: var(--text-muted); font-size: 0.88rem; line-height: 1.55; max-width: 320px; }
.site-footer h5 { font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted); margin: 0 0 0.7rem; }
.site-footer-links { display: flex; flex-direction: column; gap: 0.45rem; }
.site-footer-links a { color: var(--text-secondary); text-decoration: none; font-size: 0.92rem; transition: color .2s var(--ease); }
.site-footer-links a:hover { color: var(--primary); }
.site-footer-bottom { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.5rem; padding-top: 1.2rem; border-top: 1px solid var(--border); color: var(--text-muted); font-size: 0.82rem; }
</style>
"""
