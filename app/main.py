"""Streamlit entry point — run with: streamlit run app/main.py"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


import streamlit as st

st.set_page_config(
    page_title="Economic Policy Analyst",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Reset & base ─────────────────────────────────────────────── */
html, body, [class*="css"], .stMarkdown, .stText {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Hide Streamlit chrome */
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
[data-testid="stToolbar"]    { display: none; }

/* ── Page background ──────────────────────────────────────────── */
.stApp {
    background:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(74,158,255,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 90% 80%, rgba(167,139,250,0.08) 0%, transparent 55%),
        #080C14;
}

.main .block-container {
    padding-top: 1rem;
    padding-bottom: 3rem;
    max-width: 1280px;
}

/* ── Glass cards ──────────────────────────────────────────────── */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(16, 21, 35, 0.75) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(255, 255, 255, 0.07) !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 32px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255,255,255,0.05) !important;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    animation: fadeSlideIn 0.4s ease-out both;
}

[data-testid="stVerticalBlockBorderWrapper"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 48px rgba(0, 0, 0, 0.55), inset 0 1px 0 rgba(255,255,255,0.07) !important;
}

/* ── Form ─────────────────────────────────────────────────────── */
[data-testid="stForm"] {
    background: rgba(16, 21, 35, 0.65) !important;
    border: 1px solid rgba(74, 158, 255, 0.18) !important;
    border-radius: 20px !important;
    padding: 2rem !important;
    box-shadow: 0 0 60px rgba(74, 158, 255, 0.06);
}

textarea {
    background: rgba(8, 12, 20, 0.85) !important;
    border: 1.5px solid rgba(74, 158, 255, 0.25) !important;
    border-radius: 12px !important;
    color: #E8EAF0 !important;
    font-size: 1rem !important;
    line-height: 1.65 !important;
    padding: 0.85rem 1rem !important;
    transition: border-color 0.2s, box-shadow 0.2s;
    resize: none;
}

textarea:focus {
    border-color: rgba(74, 158, 255, 0.6) !important;
    box-shadow: 0 0 0 3px rgba(74, 158, 255, 0.12) !important;
    outline: none !important;
}

/* ── Primary button ───────────────────────────────────────────── */
button[kind="primaryFormSubmit"],
button[kind="primary"] {
    background: linear-gradient(135deg, #3B82F6 0%, #7C3AED 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    color: #fff !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    padding: 0.65rem 2.5rem !important;
    box-shadow: 0 4px 20px rgba(59, 130, 246, 0.35) !important;
    transition: opacity 0.2s, transform 0.15s, box-shadow 0.2s !important;
}

button[kind="primaryFormSubmit"]:hover,
button[kind="primary"]:hover {
    opacity: 0.92 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 28px rgba(59, 130, 246, 0.45) !important;
}

/* ── Selectbox ────────────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
    background: rgba(8,12,20,0.85) !important;
    border: 1.5px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
}

/* ── Divider ──────────────────────────────────────────────────── */
hr {
    border: none !important;
    border-top: 1px solid rgba(255,255,255,0.07) !important;
    margin: 1.75rem 0 !important;
}

/* ── Expander (Sources) ───────────────────────────────────────── */
[data-testid="stExpander"] {
    background: rgba(8,12,20,0.5) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 10px !important;
}

[data-testid="stExpander"] summary {
    font-size: 0.8rem !important;
    color: rgba(232,234,240,0.45) !important;
    letter-spacing: 0.05em;
}

/* ── Alerts ───────────────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border-left-width: 3px !important;
}

/* ── Sidebar nav ──────────────────────────────────────────────── */
[data-testid="stSidebarNav"] { display: none; }

/* ── Animations ───────────────────────────────────────────────── */
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.35; }
}

@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position:  200% center; }
}
</style>
""", unsafe_allow_html=True)

pg = st.navigation([
    st.Page("pages/query.py", title="Query", icon="🔍"),
    st.Page("pages/eval_dashboard.py", title="Eval Dashboard", icon="📈"),
])
pg.run()
