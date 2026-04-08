"""AI Share of Voice — Streamlit interface."""
from __future__ import annotations

import asyncio
import pathlib
import sys
import threading
from datetime import date

import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from src.config import CompanyEntry
from src.db import (
    get_mentions_for_run,
    get_queries_for_run,
    get_run,
    init_db,
    insert_mention,
    insert_query,
    insert_run,
)
from src.detector import CompanyRef, detect_all_mentions
from src.prompts import auto_generate_prompts
from src.query_engine import run_queries
from src.report import generate_report

# ── Async helpers ─────────────────────────────────────────────────────────────

def _run_async(coro):
    """Run a coroutine in a daemon thread and block until done."""
    result = [None]; exc = [None]
    def _t():
        try:    result[0] = asyncio.run(coro)
        except Exception as e: exc[0] = e
    t = threading.Thread(target=_t, daemon=True); t.start(); t.join()
    if exc[0]: raise exc[0]
    return result[0]


def _run_async_polling(coro, on_tick=None, interval: float = 0.35):
    """Run a coroutine in a daemon thread, calling on_tick() from the main thread
    every ~interval seconds so Streamlit UI elements can update live."""
    result = [None]; exc = [None]
    def _t():
        try:    result[0] = asyncio.run(coro)
        except Exception as e: exc[0] = e
    t = threading.Thread(target=_t, daemon=True)
    t.start()
    while t.is_alive():
        if on_tick: on_tick()
        t.join(timeout=interval)
    if exc[0]: raise exc[0]
    return result[0]


# ── Constants ─────────────────────────────────────────────────────────────────

AVAILABLE_MODELS: list[tuple[str, str]] = [
    # OpenAI
    ("openai/gpt-5.4",                         "GPT-5.4"),
    ("openai/gpt-4o",                          "GPT-4o"),
    # Anthropic
    ("anthropic/claude-opus-4",                "Claude Opus 4"),
    ("anthropic/claude-sonnet-4.6",            "Claude Sonnet 4.6"),
    ("anthropic/claude-sonnet-4.5",            "Claude Sonnet 4.5"),
    ("anthropic/claude-haiku-4.5",             "Claude Haiku 4.5"),
    # Google
    ("google/gemini-2.5-pro-preview",          "Gemini 2.5 Pro"),
    ("google/gemini-2.5-flash",                "Gemini 2.5 Flash"),
    ("google/gemini-2.0-flash-001",            "Gemini 2.0 Flash"),
    # Perplexity
    ("perplexity/sonar-pro",                   "Perplexity Sonar Pro"),
    ("perplexity/sonar",                       "Perplexity Sonar"),
    # Meta
    ("meta-llama/llama-3.3-70b-instruct",      "Llama 3.3 70B"),
    # X.AI
    ("x-ai/grok-3",                            "Grok-3"),
    # Mistral
    ("mistralai/mistral-large",                "Mistral Large"),
]
_DEFAULT_MODEL_IDS = [
    "openai/gpt-4o",
    "anthropic/claude-sonnet-4.6",
    "google/gemini-2.5-flash",
    "perplexity/sonar-pro",
]
_LABEL_TO_ID = {label: mid for mid, label in AVAILABLE_MODELS}

DB_PATH       = "data/sov.db"
TEMPLATE_PATH = "templates/report.html.j2"
REPORTS_DIR   = "reports"

_DEFAULT_COMPETITORS = ["McKinsey & Company", "Boston Consulting Group", "Bain & Company"]
_DEFAULT_ALIASES     = ["Roland Berger Strategy Consultants"]

def _auto_period() -> str:
    d = date.today(); return f"Q{(d.month-1)//3+1} {d.year}"

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Share of Voice",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><rect width='100' height='100' fill='%23000'/><text x='50' y='68' font-size='58' font-family='Arial,sans-serif' font-weight='700' text-anchor='middle' fill='%2300aac9'>B</text></svg>",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── RBDesign typeface — loaded directly from rolandberger.com CDN ── */
@font-face {
  font-family: 'RBDesign';
  src: url('https://www.rolandberger.com/fonts/RBDesign/RBDesign-Light.otf') format('opentype');
  font-weight: 300; font-style: normal; font-display: swap;
}
@font-face {
  font-family: 'RBDesign';
  src: url('https://www.rolandberger.com/fonts/RBDesign/RBDesign-LightItalic.otf') format('opentype');
  font-weight: 300; font-style: italic; font-display: swap;
}
@font-face {
  font-family: 'RBDesign';
  src: url('https://www.rolandberger.com/fonts/RBDesign/RBDesign-Regular.otf') format('opentype');
  font-weight: 400; font-style: normal; font-display: swap;
}
@font-face {
  font-family: 'RBDesign';
  src: url('https://www.rolandberger.com/fonts/RBDesign/RBDesign-Medium.otf') format('opentype');
  font-weight: 500; font-style: normal; font-display: swap;
}
@font-face {
  font-family: 'RBDesign';
  src: url('https://www.rolandberger.com/fonts/RBDesign/RBDesign-Semibold.otf') format('opentype');
  font-weight: 600; font-style: normal; font-display: swap;
}
@font-face {
  font-family: 'RBDesign';
  src: url('https://www.rolandberger.com/fonts/RBDesign/RBDesign-Bold.otf') format('opentype');
  font-weight: 700; font-style: normal; font-display: swap;
}
@font-face {
  font-family: 'RBDesign';
  src: url('https://www.rolandberger.com/fonts/RBDesign/RBDesign-BoldItalic.otf') format('opentype');
  font-weight: 700; font-style: italic; font-display: swap;
}

/* IBM Plex Mono for numeric data */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&display=swap');

/* ── Roland Berger design tokens (from rolandberger.com bundle.css) ── */
:root {
  --accent:       #00aac9;   /* RB primary cyan — most-used accent */
  --accent-light: #e3f6fc;
  --accent-dim:   #0092ae;
  --bg:           #eff0f1;   /* RB actual page background */
  --surface:      #ffffff;
  --text:         #000000;   /* pure black — RB body color */
  --muted:        #8d9399;   /* RB gray */
  --border:       #dee0e3;   /* RB border color */
  --sidebar-bg:   #000000;   /* RB header = black */
  --sidebar-text: #ffffff;
  --sidebar-muted:#a0a0a0;
  --sidebar-border:#2a2a2a;
}

html, body, .stApp, [class*="block-container"] { background-color: var(--bg) !important; }

/* RBDesign — proprietary Roland Berger typeface */
* { font-family: 'RBDesign', Arial, sans-serif !important; }

/* ── Restore Streamlit icon fonts ── */
.material-symbols-rounded,
.material-symbols-outlined,
.material-symbols-sharp,
.material-icons,
.material-icons-round,
[data-testid="stIconMaterial"] {
  font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
}

#MainMenu, footer, .stDeployButton,
[data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }

/* ─── Hide sidebar collapse/expand toggle ─────────────── */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
button[data-testid="baseButton-header"] { display: none !important; }

/* ─── Sidebar — black, like RB's fixed header nav ──────── */
[data-testid="stSidebar"] {
  background: var(--sidebar-bg) !important;
  border-right: none !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 2rem 1.25rem !important; }

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div { color: var(--sidebar-text) !important; }

[data-testid="stSidebar"] [data-testid="stTextInput"] input,
[data-testid="stSidebar"] [data-testid="stTextArea"] textarea {
  background: #1a1a1a !important;
  border: 1px solid var(--sidebar-border) !important;
  border-radius: 0 !important;
  color: var(--sidebar-text) !important;
  font-size: 14px !important;
}
[data-testid="stSidebar"] [data-testid="stTextInput"] input:focus,
[data-testid="stSidebar"] [data-testid="stTextArea"] textarea:focus {
  border-color: var(--accent) !important;
  box-shadow: none !important; outline: none !important;
}
[data-testid="stSidebar"] [data-testid="stTextInput"] label,
[data-testid="stSidebar"] [data-testid="stTextArea"] label,
[data-testid="stSidebar"] [data-testid="stSlider"] label {
  color: var(--sidebar-muted) !important;
  font-size: 10px !important; font-weight: 600 !important;
  letter-spacing: 0.12em !important; text-transform: uppercase !important;
}

[data-testid="stSidebar"] [data-testid="stMultiSelect"] [data-baseweb="select"] > div {
  background: #1a1a1a !important;
  border: 1px solid var(--sidebar-border) !important;
  border-radius: 0 !important; color: var(--sidebar-text) !important;
}
[data-testid="stSidebar"] [data-testid="stMultiSelect"] [data-baseweb="select"] > div:focus-within {
  border-color: var(--accent) !important; box-shadow: none !important;
}
[data-testid="stSidebar"] [data-baseweb="tag"] {
  background: rgba(0,170,201,0.2) !important; border-radius: 0 !important;
}
[data-testid="stSidebar"] [data-baseweb="tag"] span { color: var(--accent) !important; font-size: 12px !important; }

[data-testid="stSidebar"] [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
  background: var(--accent) !important; border-color: var(--accent) !important;
}

/* Sidebar labels */
.sov-label {
  font-size: 10px; font-weight: 600; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--sidebar-muted);
  margin: 1.5rem 0 0.4rem; display: block;
}
.sov-hint {
  font-size: 12px; color: var(--sidebar-muted); line-height: 1.55;
  margin: 0 0 8px; display: block; font-weight: 300;
}

/* ─── Main area inputs ────────────────────────────────── */
.main [data-testid="stTextInput"] input,
.main [data-testid="stTextArea"] textarea {
  border: 1px solid var(--border) !important;
  border-radius: 0 !important; font-size: 15px !important; font-weight: 300 !important;
  color: var(--text) !important; background: var(--surface) !important;
  transition: border-color 0.15s !important;
  padding: 10px 12px !important;
}
.main [data-testid="stTextInput"] input:focus,
.main [data-testid="stTextArea"] textarea:focus {
  border-color: var(--text) !important; box-shadow: none !important; outline: none !important;
}
.main [data-testid="stTextInput"] label,
.main [data-testid="stTextArea"] label,
.main [data-testid="stSlider"] label {
  font-size: 10px !important; font-weight: 600 !important; letter-spacing: 0.12em !important;
  text-transform: uppercase !important; color: var(--muted) !important; margin-bottom: 6px !important;
}

/* ─── Main area multiselect ───────────────────────────── */
.main [data-testid="stMultiSelect"] [data-baseweb="select"] > div {
  border: 1px solid var(--border) !important; border-radius: 0 !important;
  background: var(--surface) !important; font-size: 14px !important; font-weight: 300 !important;
}
.main [data-testid="stMultiSelect"] [data-baseweb="select"] > div:focus-within {
  border-color: var(--text) !important; box-shadow: none !important;
}
.main [data-baseweb="tag"] { background: var(--accent-light) !important; border-radius: 0 !important; }
.main [data-baseweb="tag"] span { color: var(--accent-dim) !important; font-size: 12px !important; }

/* ─── Primary button — RB black button style ──────────── */
/* RB: black bg, white text, uppercase, 48px, letter-spacing, NO border-radius */
/* RB hover: 4px accent underline slides up from bottom */
.main .stButton > button[kind="primary"] {
  background: var(--text) !important; color: #fff !important;
  border: none !important; border-radius: 0 !important;
  font-size: 12px !important; font-weight: 700 !important;
  letter-spacing: 0.1em !important; text-transform: uppercase !important;
  height: 48px !important; padding: 0 24px !important;
  position: relative !important; overflow: hidden !important;
  transition: none !important; box-shadow: none !important;
}
.main .stButton > button[kind="primary"]::after {
  content: '' !important; position: absolute !important;
  bottom: 0 !important; left: 0 !important;
  width: 100% !important; height: 4px !important;
  background: var(--accent) !important;
  transform: translateY(100%) !important;
  transition: transform 0.15s cubic-bezier(0.39,0.575,0.565,1) !important;
}
.main .stButton > button[kind="primary"]:hover::after {
  transform: translateY(0%) !important;
}
.main .stButton > button[kind="primary"]:disabled {
  background: var(--border) !important; color: var(--muted) !important;
}

/* ─── Secondary buttons ───────────────────────────────── */
.main .stButton > button[kind="secondary"] {
  background: var(--surface) !important; color: var(--text) !important;
  border: 1px solid var(--border) !important; border-radius: 0 !important;
  font-size: 12px !important; font-weight: 600 !important;
  letter-spacing: 0.08em !important; text-transform: uppercase !important;
  transition: border-color 0.15s !important;
}
.main .stButton > button[kind="secondary"]:hover {
  border-color: var(--text) !important; background: var(--surface) !important;
}

/* ─── Download button — filled black like RB CTA ─────── */
.stDownloadButton > button {
  background: var(--text) !important; color: #fff !important;
  border: none !important; border-radius: 0 !important;
  font-size: 12px !important; font-weight: 700 !important;
  letter-spacing: 0.1em !important; text-transform: uppercase !important;
  height: 48px !important; padding: 0 24px !important;
  position: relative !important; overflow: hidden !important;
}
.stDownloadButton > button::after {
  content: '' !important; position: absolute !important;
  bottom: 0 !important; left: 0 !important;
  width: 100% !important; height: 4px !important;
  background: var(--accent) !important;
  transform: translateY(100%) !important;
  transition: transform 0.15s cubic-bezier(0.39,0.575,0.565,1) !important;
}
.stDownloadButton > button:hover::after { transform: translateY(0%) !important; }

/* ─── Sidebar: + button ───────────────────────────────── */
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
  background: var(--accent) !important; color: #fff !important;
  border: none !important; border-radius: 0 !important;
  height: 36px !important; min-height: 36px !important;
  padding: 0 14px !important; font-size: 18px !important;
  font-weight: 400 !important; line-height: 1 !important;
  box-shadow: none !important; transform: none !important;
  letter-spacing: 0 !important; text-transform: none !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"]::after { display: none !important; }
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
  background: var(--accent-dim) !important;
}

/* ─── Sidebar: × remove buttons ───────────────────────── */
[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
  background: transparent !important; color: var(--sidebar-muted) !important;
  border: 1px solid var(--sidebar-border) !important; border-radius: 0 !important;
  height: 36px !important; min-height: 36px !important;
  padding: 0 10px !important; font-size: 15px !important;
  font-weight: 400 !important; line-height: 1 !important;
  letter-spacing: 0 !important; text-transform: none !important;
  transition: border-color 0.12s, color 0.12s !important;
}
[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
  border-color: var(--accent) !important; color: var(--accent) !important;
}

.comp-tag {
  font-size: 14px; font-weight: 300; color: var(--sidebar-text);
  padding: 8px 0 8px 2px; line-height: 1.3;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}

/* ─── Slider ──────────────────────────────────────────── */
.main [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
  background: var(--text) !important; border-color: var(--text) !important;
  border-radius: 0 !important;
}

/* ─── Divider ─────────────────────────────────────────── */
hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 1.5rem 0 !important; }

/* ─── Expander ────────────────────────────────────────── */
[data-testid="stExpander"] {
  border: 1px solid var(--border) !important; border-radius: 0 !important;
  background: var(--surface) !important; overflow: hidden !important;
}
[data-testid="stExpander"] summary {
  font-size: 13px !important; font-weight: 500 !important;
  color: var(--text) !important; padding: 14px 18px !important;
}
[data-testid="stExpander"] summary:hover { background: var(--bg) !important; }

/* ─── Status / alerts ─────────────────────────────────── */
[data-testid="stStatusWidget"], [data-testid="stStatus"] {
  border: 1px solid var(--border) !important; border-radius: 0 !important;
  background: var(--surface) !important; font-size: 14px !important;
}
[data-testid="stAlert"] { border-radius: 0 !important; font-size: 14px !important; }

/* ─── Progress bar — flat, RB cyan fill ──────────────── */
[data-testid="stProgressBar"] > div > div {
  background: var(--accent) !important; border-radius: 0 !important;
}
[data-testid="stProgressBar"] > div {
  border-radius: 0 !important; background: var(--border) !important; height: 3px !important;
}
.progress-label {
  font-size: 10px; font-weight: 600; color: var(--muted);
  letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 8px;
}

/* ─── Main block ──────────────────────────────────────── */
.main .block-container {
  padding-top: 2.5rem !important; padding-bottom: 3rem !important;
  max-width: 900px !important;
}

/* ─── Page header ─────────────────────────────────────── */
.sov-header {
  padding-bottom: 2rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 1.75rem;
}
/* RB-style wordmark bar */
.sov-logo-bar {
  display: flex; align-items: center; gap: 0; margin-bottom: 24px;
}
.sov-logo-bar img {
  border-right: 1px solid var(--border);
  padding-right: 14px;
}
.sov-logo-product {
  font-size: 11px; font-weight: 300; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--muted); line-height: 1;
}
/* Live indicator */
.sov-badge {
  display: inline-flex; align-items: center; gap: 7px;
  font-size: 10px; font-weight: 600; letter-spacing: 0.14em;
  text-transform: uppercase; color: var(--accent); margin-bottom: 12px;
}
.sov-badge::before {
  content: ''; width: 5px; height: 5px;
  background: var(--accent); display: block; flex-shrink: 0;
  animation: pulse 2.4s ease-in-out infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.25} }
/* RBDesign-style heading: light weight, tight tracking */
.sov-title {
  font-size: 40px; font-weight: 300; letter-spacing: -0.01em;
  line-height: 1.05; color: var(--text); margin: 0 0 12px;
}
.sov-title strong { font-weight: 700; }
.sov-subtitle {
  font-size: 16px; color: var(--muted); margin: 0; max-width: 560px;
  line-height: 1.7; font-weight: 300;
}

/* ─── Step indicator ──────────────────────────────────── */
.sov-steps {
  display: flex; align-items: center;
  padding: 1.25rem 0 1.5rem; border-bottom: 1px solid var(--border);
  margin-bottom: 1.75rem;
}
.sov-step { display: flex; align-items: center; gap: 8px; flex-shrink: 0; }
.sov-step-num {
  width: 22px; height: 22px; border-radius: 0;
  border: 1px solid var(--border); background: var(--bg);
  color: var(--muted); font-size: 10px; font-weight: 700;
  display: flex; align-items: center; justify-content: center;
  letter-spacing: 0;
}
.sov-step-label {
  font-size: 11px; font-weight: 600; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--muted); white-space: nowrap;
}
.sov-step.active .sov-step-num { background: var(--text); border-color: var(--text); color: #fff; }
.sov-step.active .sov-step-label { color: var(--text); }
.sov-step.done .sov-step-num { background: var(--accent); border-color: var(--accent); color: #fff; }
.sov-connector { flex:1; min-width:12px; max-width:40px; height:1px; background:var(--border); margin:0 6px; }

/* ─── Prompt review ───────────────────────────────────── */
.prompt-review-meta { display: flex; align-items: center; gap: 12px; margin-bottom: 14px; }
.prompt-count-badge {
  font-size: 10px; font-weight: 700; color: var(--accent);
  background: var(--accent-light); border-radius: 0;
  padding: 3px 10px; white-space: nowrap; letter-spacing: 0.1em; text-transform: uppercase;
}
.prompt-review-hint { font-size: 13px; color: var(--muted); font-weight: 300; }

/* ─── Report section ──────────────────────────────────── */
.report-meta { flex: 1; }
.report-meta-title { font-size: 15px; font-weight: 600; color: var(--text); letter-spacing:0.02em; margin:0; text-transform: uppercase; font-size: 11px; }
.report-meta-file { font-size: 12px; color: var(--muted); font-family:'IBM Plex Mono',monospace !important; margin-top:4px; font-weight: 400; }

/* ─── Empty state ─────────────────────────────────────── */
.sov-empty {
  display: flex; flex-direction: column; align-items: center;
  justify-content: center; padding: 80px 24px; text-align: center;
  border: 1px solid var(--border); background: var(--surface); margin-top: 2rem;
}
.sov-empty-icon {
  width: 4px; height: 48px; background: var(--accent);
  margin-bottom: 24px;
}
.sov-empty-title { font-size: 18px; font-weight: 300; color: var(--text); margin: 0 0 8px; letter-spacing: -0.01em; }
.sov-empty-desc { font-size: 14px; color: var(--muted); margin: 0; max-width: 360px; line-height: 1.7; font-weight: 300; }

/* ─── Misc ────────────────────────────────────────────── */
[data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace !important; }

.sov-watermark {
  position: fixed; bottom: 18px; left: 22px; font-size: 10px;
  font-weight: 400; color: var(--muted); letter-spacing: 0.08em;
  text-transform: uppercase;
  opacity: 0.6; z-index: 9999; pointer-events: none;
}

/* Sidebar scrollbar */
[data-testid="stSidebar"] ::-webkit-scrollbar { width: 3px; }
[data-testid="stSidebar"] ::-webkit-scrollbar-track { background: transparent; }
[data-testid="stSidebar"] ::-webkit-scrollbar-thumb { background: #2a2a2a; }

/* ─── Mobile ──────────────────────────────────────── */
@media (max-width: 768px) {
  .main .block-container {
    padding-left: 1rem !important; padding-right: 1rem !important;
    padding-top: 1.25rem !important; max-width: 100% !important;
  }
  .sov-title    { font-size: 28px !important; }
  .sov-subtitle { font-size: 14px !important; }
  .sov-header   { padding-bottom: 1.25rem; margin-bottom: 1.25rem; }
  .sov-steps    { flex-wrap: wrap; gap: 8px; padding: 1rem 0 1.25rem; }
  .sov-connector { display: none; }
  .sov-step-label { font-size: 10px !important; }
  .prompt-review-meta { flex-wrap: wrap; gap: 6px; }
  .report-meta-title { font-size: 10px !important; }
  .report-meta-file  { font-size: 11px !important; }
  .sov-watermark { display: none !important; }
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="sov-watermark">developed by gregor.weindorf</div>', unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────

_ss = st.session_state
_init = {
    "app_stage":        "config",
    "report_html":      None,
    "report_path":      None,
    "pending_prompts":  [],
    "pending_config":   {},
    "prompt_ver":       0,
    "competitors":      list(_DEFAULT_COMPETITORS),
    "comp_input_key":   0,
    "aliases":          list(_DEFAULT_ALIASES),
    "alias_input_key":  0,
    "focus_brief":      "",
    "custom_prompts_raw": "",
}
for k, v in _init.items():
    if k not in _ss: _ss[k] = v

# ── Callbacks ─────────────────────────────────────────────────────────────────

def _remove_comp(i): _ss.competitors.pop(i) if 0 <= i < len(_ss.competitors) else None
def _add_comp():
    v = _ss.get(f"nc_{_ss.comp_input_key}", "").strip()
    if v and v not in _ss.competitors: _ss.competitors.append(v)
    _ss.comp_input_key += 1

def _remove_alias(i): _ss.aliases.pop(i) if 0 <= i < len(_ss.aliases) else None
def _add_alias():
    v = _ss.get(f"na_{_ss.alias_input_key}", "").strip()
    if v and v not in _ss.aliases: _ss.aliases.append(v)
    _ss.alias_input_key += 1

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding:0 0 20px;border-bottom:1px solid #2a2a2a;margin-bottom:20px;">
      <img src="https://www.rolandberger.com/img/assets/RoBe_Logotype_White_Digital.png"
           alt="Roland Berger"
           style="height:16px;display:block;margin-bottom:10px;">
      <div style="font-size:9px;color:#666;letter-spacing:0.16em;text-transform:uppercase;font-family:'RBDesign',Arial,sans-serif;font-weight:600;">
        AI Share of Voice
      </div>
    </div>""", unsafe_allow_html=True)

    # API key ──────────────────────────────────────────────────────────────────
    api_key = st.secrets.get("OPENROUTER_API_KEY", "")
    if api_key:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:8px;background:#CCFBF1;
             border:1px solid #99F6E4;border-radius:8px;padding:10px 12px;margin-bottom:1rem">
          <div style="width:6px;height:6px;border-radius:50%;background:#0D9488;flex-shrink:0"></div>
          <span style="font-size:12px;font-weight:500;color:#0F766E">API key active</span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(
            '<span class="sov-label">API Key</span>'
            '<span class="sov-hint">Needed to query AI models. Free key at openrouter.ai</span>',
            unsafe_allow_html=True,
        )
        api_key = st.text_input("key", type="password",
                                placeholder="sk-or-v1-...", label_visibility="collapsed")

    st.markdown('<hr style="margin:1.25rem 0">', unsafe_allow_html=True)

    # Your Target ──────────────────────────────────────────────────────────────
    st.markdown(
        '<span class="sov-label">Your Target</span>'
        '<span class="sov-hint">The brand or company whose AI visibility you want to measure.</span>',
        unsafe_allow_html=True,
    )
    target_name = st.text_input("name", value="Roland Berger",
                                placeholder="Company or brand name",
                                label_visibility="collapsed")

    # Aliases tag input
    st.markdown(
        '<span style="font-size:11px;color:#71717A;display:block;margin:6px 0 4px">'
        'Alternative names — how else might AI refer to this brand?</span>',
        unsafe_allow_html=True,
    )
    for i, alias in enumerate(_ss.aliases):
        ca, cb = st.columns([5, 1])
        with ca: st.markdown(f'<div class="comp-tag">{alias}</div>', unsafe_allow_html=True)
        with cb: st.button("×", key=f"ra_{i}", on_click=_remove_alias, args=(i,))
    ca2, cb2 = st.columns([5, 1])
    with ca2:
        st.text_input("alias_in", key=f"na_{_ss.alias_input_key}",
                      label_visibility="collapsed", placeholder="Add alias…")
    with cb2:
        st.button("+", key="add_alias", on_click=_add_alias, type="primary")

    st.markdown('<hr style="margin:1.25rem 0">', unsafe_allow_html=True)

    # Competitors ──────────────────────────────────────────────────────────────
    st.markdown(
        '<span class="sov-label">Competitors</span>'
        '<span class="sov-hint">Companies to benchmark your visibility against.</span>',
        unsafe_allow_html=True,
    )
    for i, comp in enumerate(_ss.competitors):
        cc, cd = st.columns([5, 1])
        with cc: st.markdown(f'<div class="comp-tag">{comp}</div>', unsafe_allow_html=True)
        with cd: st.button("×", key=f"rm_{i}", on_click=_remove_comp, args=(i,))
    ce, cf = st.columns([5, 1])
    with ce:
        st.text_input("comp_in", key=f"nc_{_ss.comp_input_key}",
                      label_visibility="collapsed", placeholder="Add competitor…")
    with cf:
        st.button("+", key="add_comp", on_click=_add_comp, type="primary")

    st.markdown('<hr style="margin:1.25rem 0">', unsafe_allow_html=True)

    # Scan Settings ────────────────────────────────────────────────────────────
    st.markdown('<span class="sov-label">Scan Settings</span>', unsafe_allow_html=True)
    st.markdown(
        '<span style="font-size:11px;color:#71717A;display:block;margin-bottom:4px">'
        'What topic should AI models be asked about?</span>',
        unsafe_allow_html=True,
    )
    topic = st.text_input("topic", value="strategy consulting germany",
                          label_visibility="collapsed",
                          placeholder="e.g. cloud ERP software, luxury SUVs in France")

    num_prompts = st.slider("Questions per AI model", min_value=2, max_value=50, value=10,
                            help="Each question goes to every selected model. 10 is good for a quick test; 20+ for reliable production data.")

    # AI Model selector ────────────────────────────────────────────────────────
    st.markdown(
        '<span style="font-size:11px;color:#71717A;display:block;margin:8px 0 4px">'
        'AI models to query (up to 4)</span>',
        unsafe_allow_html=True,
    )
    all_labels     = [label for _, label in AVAILABLE_MODELS]
    default_labels = [label for mid, label in AVAILABLE_MODELS if mid in _DEFAULT_MODEL_IDS]
    selected_labels = st.multiselect(
        "models",
        options=all_labels,
        default=default_labels,
        max_selections=4,
        label_visibility="collapsed",
        key="model_multiselect",
    )
    models = [(mid, label) for mid, label in AVAILABLE_MODELS if label in selected_labels]

    n_est = num_prompts * max(len(models), 1)
    st.markdown(
        f'<div style="font-size:11px;color:#71717A;margin-top:2px">'
        f'~{n_est} total API requests</div>',
        unsafe_allow_html=True,
    )

    st.markdown("""
    <div style="margin-top:1.5rem;padding-top:1.25rem;border-top:1px solid #E4E4E7">
      <div style="font-size:11px;color:#71717A;line-height:1.6">
        Queries via <a href="https://openrouter.ai" target="_blank"
        style="color:#0D9488;text-decoration:none">OpenRouter</a>.
        Reports saved locally.
      </div>
    </div>""", unsafe_allow_html=True)

# ── Step indicator ────────────────────────────────────────────────────────────

def _step_indicator(active: int):
    steps = ["Configure", "Review Questions", "Scan"]
    parts = []
    for i, label in enumerate(steps):
        icon = ("&#10003;" if i < active else str(i + 1))
        cls  = ("done" if i < active else "active" if i == active else "")
        parts.append(
            f'<div class="sov-step {cls}">'
            f'<div class="sov-step-num">{icon}</div>'
            f'<div class="sov-step-label">{label}</div>'
            f'</div>'
        )
        if i < len(steps) - 1:
            parts.append(f'<div class="sov-connector"></div>')
    st.markdown(f'<div class="sov-steps">{"".join(parts)}</div>', unsafe_allow_html=True)

# ── Page header ───────────────────────────────────────────────────────────────

st.markdown("""
<div class="sov-header">
  <div class="sov-logo-bar">
    <img src="https://www.rolandberger.com/img/assets/RoBe_Logotype_White_Digital.png"
         alt="Roland Berger"
         style="height:14px;display:block;margin-right:14px;filter:brightness(0);">
    <span class="sov-logo-product">AI Share of Voice</span>
  </div>
  <div class="sov-badge">Live Intelligence</div>
  <h1 class="sov-title"><strong>Share of Voice</strong> Scanner</h1>
  <p class="sov-subtitle">
    Track how AI models mention your brand — benchmark against competitors
    and identify gaps in your visibility.
  </p>
</div>
""", unsafe_allow_html=True)

# ── Stage routing ─────────────────────────────────────────────────────────────

stage = _ss.app_stage

# ══════════════════════════════════════════════════════════════════════════════
# STAGE: config
# ══════════════════════════════════════════════════════════════════════════════

if stage == "config":
    _step_indicator(active=0)

    with st.expander("Optional: Focus the generated questions", expanded=False):
        st.markdown("""
        <div style="padding:4px 0 12px;font-size:13px;color:#71717A;line-height:1.65">
          Narrow the questions to a specific persona, region, or deal type.
          Leave blank for broad coverage.
        </div>""", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div style="font-size:12px;font-weight:500;color:#71717A;margin-bottom:4px">Focus note</div>',
                        unsafe_allow_html=True)
            focus_brief = st.text_input(
                "focus_brief", label_visibility="collapsed",
                placeholder="e.g. CFO persona, DACH region, mid-market deals",
                value=_ss.focus_brief,
            )
        with col_b:
            st.markdown('<div style="font-size:12px;font-weight:500;color:#71717A;margin-bottom:4px">'
                        'Your own questions (one per line)</div>', unsafe_allow_html=True)
            custom_prompts_raw = st.text_area(
                "custom_prompts", label_visibility="collapsed",
                placeholder="Which consulting firms do German CFOs trust for restructuring?\nBest strategy advisors for mid-size industrials in DACH?",
                height=96, value=_ss.custom_prompts_raw,
                help="Always included first; also used as style examples for AI-generated questions.",
            )

    st.write("")

    if not api_key:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:10px;padding:12px 16px;
             background:#FFF7ED;border:1px solid #FED7AA;border-radius:10px;margin-bottom:16px">
          <div style="font-size:13px;color:#92400E">
            Add your OpenRouter API key in the sidebar to continue.
            Free key at <a href="https://openrouter.ai" target="_blank"
            style="color:#B45309;font-weight:500">openrouter.ai</a>.
          </div>
        </div>""", unsafe_allow_html=True)

    if not models:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:10px;padding:12px 16px;
             background:#FFF7ED;border:1px solid #FED7AA;border-radius:10px;margin-bottom:16px">
          <div style="font-size:13px;color:#92400E">
            Select at least one AI model in the sidebar to continue.
          </div>
        </div>""", unsafe_allow_html=True)

    can_generate = bool(api_key and topic.strip() and target_name.strip() and models)
    gen_clicked  = st.button("Generate Questions", type="primary",
                             disabled=not can_generate, use_container_width=True)

    if gen_clicked:
        custom_prompts = [p.strip() for p in custom_prompts_raw.splitlines() if p.strip()]
        n_to_generate  = max(0, num_prompts - len(custom_prompts))

        all_names = [target_name] + list(_ss.aliases) + list(_ss.competitors)
        with st.spinner(f"Generating {n_to_generate} questions for '{topic}'…"):
            generated = _run_async(auto_generate_prompts(
                topic=topic, count=n_to_generate, api_key=api_key,
                brief=focus_brief, examples=custom_prompts or None,
                exclude_names=all_names,
            )) if n_to_generate > 0 else []

        _ss.pending_prompts = list(dict.fromkeys(custom_prompts + generated))
        _ss.pending_config  = {
            "target_name":    target_name,
            "alias_list":     list(_ss.aliases),
            "competitor_list": list(_ss.competitors),
            "topic":          topic,
            "api_key":        api_key,
            "models":         models,
        }
        _ss.focus_brief        = focus_brief
        _ss.custom_prompts_raw = custom_prompts_raw
        _ss.prompt_ver        += 1
        _ss.app_stage          = "review"
        st.rerun()

    # ── Report section ────────────────────────────────────────────────────────
    if _ss.report_html:
        st.markdown('<hr style="margin:2rem 0">', unsafe_allow_html=True)
        report_name = pathlib.Path(_ss.report_path).name

        st.markdown(f"""
        <div class="report-meta">
          <div class="report-meta-title">Scan complete</div>
          <div class="report-meta-file">{report_name}</div>
        </div>""", unsafe_allow_html=True)

        st.write("")
        col_dl, col_new = st.columns(2)
        with col_dl:
            st.download_button("Download Report", data=_ss.report_html,
                               file_name=report_name, mime="text/html",
                               use_container_width=True)
        with col_new:
            if st.button("New Scan", type="primary", use_container_width=True):
                _ss.report_html = None
                _ss.report_path = None
                st.rerun()

    elif can_generate:
        st.markdown("""
        <div class="sov-empty">
          <div class="sov-empty-icon">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none"
                 stroke="#0D9488" stroke-width="1.75" stroke-linecap="round">
              <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
            </svg>
          </div>
          <div class="sov-empty-title">No report yet</div>
          <div class="sov-empty-desc">
            Click <strong>Generate Questions</strong> to preview the questions
            before the scan starts.
          </div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STAGE: review
# ══════════════════════════════════════════════════════════════════════════════

elif stage == "review":
    _step_indicator(active=1)

    prompts = _ss.pending_prompts
    cfg     = _ss.pending_config
    n_p     = len(prompts)

    model_names = ", ".join(label for _, label in cfg.get("models", []))
    st.markdown(f"""
    <div class="prompt-review-meta">
      <span class="prompt-count-badge">{n_p} questions</span>
      <span class="prompt-review-hint">
        Edit freely — remove off-topic lines, add your own.
        Will be sent to: <strong>{model_names}</strong>.
      </span>
    </div>""", unsafe_allow_html=True)

    textarea_val = st.text_area(
        "questions", value="\n".join(prompts),
        height=min(520, max(240, n_p * 44)),
        label_visibility="collapsed",
        key=f"review_ta_{_ss.prompt_ver}",
        help="One question per line. Edit, delete, or reorder freely.",
        placeholder="One question per line…",
    )
    approved = [l.strip() for l in textarea_val.splitlines() if l.strip()]

    if len(approved) != n_p:
        st.markdown(
            f'<div style="font-size:12px;color:#71717A;margin-top:4px">'
            f'{len(approved)} questions selected</div>', unsafe_allow_html=True)

    st.write("")
    col_back, col_run = st.columns([1, 3])
    with col_back:
        if st.button("Back", use_container_width=True):
            _ss.app_stage = "config"; st.rerun()
    with col_run:
        lbl = (f"Run Scan — {len(approved)} questions × {len(cfg.get('models',[]))} models"
               if approved else "Add at least one question to continue")
        if st.button(lbl, type="primary", disabled=not approved,
                     use_container_width=True):
            _ss.pending_prompts = approved
            _ss.app_stage       = "scanning"
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STAGE: scanning
# ══════════════════════════════════════════════════════════════════════════════

elif stage == "scanning":
    _step_indicator(active=2)

    cfg          = _ss.pending_config
    prompt_list  = _ss.pending_prompts
    api_key_run  = cfg["api_key"]
    topic_run    = cfg["topic"]
    models_run   = cfg["models"]

    companies = [CompanyEntry(name=cfg["target_name"], aliases=cfg["alias_list"], is_target=True)] + [
        CompanyEntry(name=c, aliases=[], is_target=False) for c in cfg["competitor_list"]
    ]
    company_refs = [CompanyRef(name=c.name, aliases=c.aliases, is_target=c.is_target) for c in companies]

    init_db(DB_PATH)
    prog_label = st.empty()
    prog_bar   = st.progress(0)

    def _prog(pct: float, label: str):
        prog_label.markdown(f'<div class="progress-label">{label}</div>', unsafe_allow_html=True)
        prog_bar.progress(pct)

    _prog(0.05, "Starting…")

    try:
        with st.status("Running scan…", expanded=True) as status:
            real_total = len(prompt_list) * len(models_run)
            q_done     = [0]   # shared counter updated by on_progress callback

            query_status = st.empty()
            query_status.write(f"Querying {len(models_run)} models — {real_total} requests in flight…")
            _prog(0.12, f"Starting {real_total} queries across {len(models_run)} models…")

            def _on_q_progress(done: int, total: int):
                q_done[0] = done

            def _tick():
                done = q_done[0]
                pct  = 0.12 + (done / real_total) * 0.60  # 12 % → 72 %
                _prog(pct, f"Querying models — {done} / {real_total} responses received…")

            results = _run_async_polling(
                run_queries(prompt_list, models_run, api_key_run,
                            max_concurrent=5, on_progress=_on_q_progress),
                on_tick=_tick,
                interval=0.35,
            )

            # Per-model error breakdown
            from collections import Counter as _Counter
            model_errors = _Counter(r.model_label for r in results if r.error)
            ok_count     = sum(1 for r in results if not r.error)

            query_status.write(f"**{ok_count} / {real_total} responses received**")
            if model_errors:
                for model_label, n_err in sorted(model_errors.items()):
                    st.warning(
                        f"**{model_label}** — {n_err} / {len(prompt_list)} requests failed. "
                        f"The model ID may be unavailable on OpenRouter."
                    )
            _prog(0.75, f"{ok_count}/{real_total} responses — detecting mentions…")

            st.write("Detecting brand mentions…")
            run_id = insert_run(DB_PATH, topic=topic_run, period=_auto_period())
            for r in results:
                qid = insert_query(DB_PATH, run_id=run_id,
                                   model_id=r.model_id, model_label=r.model_label,
                                   prompt=r.prompt, response=r.response)
                if r.response:
                    for hit in detect_all_mentions(r.response, company_refs):
                        insert_mention(DB_PATH, query_id=qid,
                                       company_name=hit["company_name"], is_target=hit["is_target"],
                                       match_type=hit["type"], excerpt=hit["excerpt"])
            _prog(0.88, "Mentions detected — building report…")

            st.write("Generating report…")
            queries    = get_queries_for_run(DB_PATH, run_id)
            mentions   = get_mentions_for_run(DB_PATH, run_id)
            run_record = get_run(DB_PATH, run_id)

            report_path = generate_report(
                run=run_record, queries=queries, mentions=mentions, companies=companies,
                template_path=TEMPLATE_PATH, output_dir=REPORTS_DIR,
            )
            _ss.report_html      = pathlib.Path(report_path).read_text(encoding="utf-8")
            _ss.report_path      = report_path
            _prog(1.0, "Scan complete")
            status.update(label="Scan complete", state="complete")

    except Exception as exc:
        st.error(f"Scan failed: {exc}")
        if st.button("Back to Configuration"):
            _ss.app_stage = "config"; st.rerun()
    else:
        _ss.app_stage = "config"
        st.rerun()
