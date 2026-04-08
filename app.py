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


# ── Error helpers ─────────────────────────────────────────────────────────────

def _classify_error(err: str) -> tuple[str, str]:
    """
    Turn a raw exception string into (short_label, user_guidance).
    Returns a tuple suitable for display in st.error / st.warning.
    """
    e = err.lower()
    if "401" in e or "authentication" in e or "invalid api key" in e or "unauthorized" in e:
        return (
            "Authentication failed",
            "Your OpenRouter API key was rejected. Check that the key is correct and hasn't expired. "
            "You can generate a new key at openrouter.ai/keys.",
        )
    if "402" in e or "insufficient" in e or "credit" in e or "balance" in e:
        return (
            "Insufficient credits",
            "Your OpenRouter account has run out of credits. Top up at openrouter.ai/credits.",
        )
    if "403" in e or "forbidden" in e or "not allowed" in e:
        return (
            "Access denied",
            "Your API key doesn't have permission to use this model. Some models require a paid plan.",
        )
    if "404" in e or "not found" in e or "no such model" in e:
        return (
            "Model not found",
            "This model ID isn't available on OpenRouter. It may have been renamed or removed. "
            "Check openrouter.ai/models for the current list.",
        )
    if "429" in e or "rate limit" in e or "too many requests" in e:
        return (
            "Rate limited",
            "Too many requests sent too quickly. Try reducing the number of questions or models, "
            "or wait a moment before retrying.",
        )
    if "timeout" in e or "timed out" in e or "read timeout" in e:
        return (
            "Request timed out",
            "The model took too long to respond (>90 s). This can happen with large thinking models. "
            "Try Gemini 2.5 Flash or GPT-4o instead.",
        )
    if "connection" in e or "network" in e or "name or service not known" in e or "ssl" in e:
        return (
            "Network error",
            "Could not reach the OpenRouter API. Check your internet connection and try again.",
        )
    if "context" in e or "token" in e and "limit" in e:
        return (
            "Context limit exceeded",
            "The prompt is too long for this model. Reduce the number of questions per model.",
        )
    return ("Unexpected error", err[:300])


def _error_html(label: str, detail: str) -> str:
    return f"""
    <div style="border:1px solid #fca5a5;background:#fef2f2;padding:14px 16px;margin:8px 0;">
      <div style="font-size:11px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;
                  color:#b91c1c;margin-bottom:6px;">{label}</div>
      <div style="font-size:13px;color:#7f1d1d;line-height:1.55;">{detail}</div>
    </div>"""


def _warning_html(label: str, detail: str) -> str:
    return f"""
    <div style="border:1px solid #fcd34d;background:#fffbeb;padding:14px 16px;margin:8px 0;">
      <div style="font-size:11px;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;
                  color:#92400e;margin-bottom:6px;">{label}</div>
      <div style="font-size:13px;color:#78350f;line-height:1.55;">{detail}</div>
    </div>"""

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

/* ════════════════════════════════════════════════════════
   BUTTONS — target data-testid which beats emotion-css
   generated class names from Streamlit's primaryColor theme
   ════════════════════════════════════════════════════════ */

/* ─── Global reset: kill rounding & shadows on every button */
button[data-testid^="baseButton"] {
  border-radius: 0 !important;
  box-shadow: none !important;
}

/* ─── Primary CTA (main area) — black, RB style ──────── */
.main button[data-testid="baseButton-primary"] {
  background:       #000000 !important;
  background-color: #000000 !important;
  color:            #ffffff !important;
  border:           none !important;
  border-radius:    0 !important;
  font-size:        12px !important;
  font-weight:      700 !important;
  letter-spacing:   0.1em !important;
  text-transform:   uppercase !important;
  height:           48px !important;
  padding:          0 24px !important;
  position:         relative !important;
  overflow:         hidden !important;
  transition:       none !important;
  box-shadow:       none !important;
}
.main button[data-testid="baseButton-primary"]::after {
  content: '' !important; position: absolute !important;
  bottom: 0 !important; left: 0 !important;
  width: 100% !important; height: 4px !important;
  background: var(--accent) !important;
  transform: translateY(100%) !important;
  transition: transform 0.15s cubic-bezier(0.39,0.575,0.565,1) !important;
}
.main button[data-testid="baseButton-primary"]:hover::after {
  transform: translateY(0%) !important;
}
.main button[data-testid="baseButton-primary"] *,
.main button[kind="primary"] * { color: #ffffff !important; }
.main button[data-testid="baseButton-primary"]:disabled {
  background:       var(--border) !important;
  background-color: var(--border) !important;
  color:            var(--muted) !important;
}

/* ─── Secondary (Back / cancel — main area) ──────────── */
.main [data-testid="stButton"] button[data-testid="baseButton-secondary"] {
  background:       #ffffff !important;
  background-color: #ffffff !important;
  color:            #000000 !important;
  border:           1px solid var(--border) !important;
  border-radius:    0 !important;
  font-size:        12px !important;
  font-weight:      600 !important;
  letter-spacing:   0.08em !important;
  text-transform:   uppercase !important;
  height:           48px !important;
  transition:       border-color 0.15s !important;
  box-shadow:       none !important;
}
.main [data-testid="stButton"] button[data-testid="baseButton-secondary"] *,
.main [data-testid="stButton"] button[kind="secondary"] * { color: #000000 !important; }
.main [data-testid="stButton"] button[data-testid="baseButton-secondary"]:hover {
  border-color: #000000 !important;
  background:       #ffffff !important;
  background-color: #ffffff !important;
}

/* ─── Download button — same black RB style as primary ── */
[data-testid="stDownloadButton"] button {
  background:       #000000 !important;
  background-color: #000000 !important;
  color:            #ffffff !important;
  border:           none !important;
  border-radius:    0 !important;
  font-size:        12px !important;
  font-weight:      700 !important;
  letter-spacing:   0.1em !important;
  text-transform:   uppercase !important;
  height:           48px !important;
  padding:          0 24px !important;
  position:         relative !important;
  overflow:         hidden !important;
  transition:       none !important;
  box-shadow:       none !important;
}
[data-testid="stDownloadButton"] button::after {
  content: '' !important; position: absolute !important;
  bottom: 0 !important; left: 0 !important;
  width: 100% !important; height: 4px !important;
  background: var(--accent) !important;
  transform: translateY(100%) !important;
  transition: transform 0.15s cubic-bezier(0.39,0.575,0.565,1) !important;
}
[data-testid="stDownloadButton"] button:hover::after {
  transform: translateY(0%) !important;
}
[data-testid="stDownloadButton"] button * { color: #ffffff !important; }

/* ─── Sidebar: + add button (cyan) ───────────────────── */
[data-testid="stSidebar"] button[data-testid="baseButton-primary"] {
  background:       var(--accent) !important;
  background-color: var(--accent) !important;
  color:            #ffffff !important;
  border:           none !important;
  border-radius:    0 !important;
  height:           36px !important;
  min-height:       36px !important;
  padding:          0 14px !important;
  font-size:        18px !important;
  font-weight:      400 !important;
  line-height:      1 !important;
  letter-spacing:   0 !important;
  text-transform:   none !important;
  box-shadow:       none !important;
}
[data-testid="stSidebar"] button[data-testid="baseButton-primary"]::after {
  display: none !important;
}
[data-testid="stSidebar"] button[data-testid="baseButton-primary"] *,
[data-testid="stSidebar"] button[kind="primary"] * { color: #ffffff !important; }
[data-testid="stSidebar"] button[data-testid="baseButton-primary"]:hover {
  background:       var(--accent-dim) !important;
  background-color: var(--accent-dim) !important;
}

/* ─── Sidebar: × remove button ───────────────────────── */
[data-testid="stSidebar"] button[data-testid="baseButton-secondary"] {
  background:       #1a1a1a !important;
  background-color: #1a1a1a !important;
  color:            #a0a0a0 !important;
  border:           1px solid #444444 !important;
  border-radius:    0 !important;
  height:           36px !important;
  min-height:       36px !important;
  padding:          0 10px !important;
  font-size:        15px !important;
  font-weight:      400 !important;
  line-height:      1 !important;
  letter-spacing:   0 !important;
  text-transform:   none !important;
  transition:       border-color 0.12s, color 0.12s !important;
}
/* Use * to cover div/span/p — Streamlit uses div inside buttons in some versions.
   Also include [kind="secondary"] as fallback for builds without data-testid. */
[data-testid="stSidebar"] button[data-testid="baseButton-secondary"] *,
[data-testid="stSidebar"] button[kind="secondary"] * {
  color: #a0a0a0 !important;
}
[data-testid="stSidebar"] button[data-testid="baseButton-secondary"]:hover,
[data-testid="stSidebar"] button[kind="secondary"]:hover {
  background:       #222222 !important;
  background-color: #222222 !important;
  border-color:     var(--accent) !important;
  color:            var(--accent) !important;
}
[data-testid="stSidebar"] button[data-testid="baseButton-secondary"]:hover *,
[data-testid="stSidebar"] button[kind="secondary"]:hover * {
  color: var(--accent) !important;
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
  position: fixed; bottom: 18px; right: 22px; font-size: 10px;
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
           style="height:32px;display:block;margin-bottom:10px;">
      <div style="font-size:9px;color:#666;letter-spacing:0.16em;text-transform:uppercase;font-family:'RBDesign',Arial,sans-serif;font-weight:600;">
        AI Share of Voice
      </div>
    </div>""", unsafe_allow_html=True)

    # API key ──────────────────────────────────────────────────────────────────
    api_key = st.secrets.get("OPENROUTER_API_KEY", "")
    if api_key:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:8px;
             border-left:3px solid #00aac9;padding:8px 12px;margin-bottom:1rem">
          <div style="width:5px;height:5px;background:#00aac9;flex-shrink:0"></div>
          <span style="font-size:11px;font-weight:600;letter-spacing:0.1em;
                       text-transform:uppercase;color:#8d9399;">API key active</span>
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
    _MAX_COMPETITORS = 3
    st.markdown(
        '<span class="sov-label">Competitors</span>'
        '<span class="sov-hint">Up to 3 companies to benchmark your visibility against.</span>',
        unsafe_allow_html=True,
    )
    for i, comp in enumerate(_ss.competitors):
        cc, cd = st.columns([5, 1])
        with cc: st.markdown(f'<div class="comp-tag">{comp}</div>', unsafe_allow_html=True)
        with cd: st.button("×", key=f"rm_{i}", on_click=_remove_comp, args=(i,))
    if len(_ss.competitors) < _MAX_COMPETITORS:
        ce, cf = st.columns([5, 1])
        with ce:
            st.text_input("comp_in", key=f"nc_{_ss.comp_input_key}",
                          label_visibility="collapsed", placeholder="Add competitor…")
        with cf:
            st.button("+", key="add_comp", on_click=_add_comp, type="primary")
    else:
        st.markdown(
            '<span style="font-size:11px;color:#555;display:block;margin-top:4px;">'
            'Maximum of 3 competitors reached.</span>',
            unsafe_allow_html=True,
        )

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
         style="height:28px;display:block;margin-right:14px;filter:brightness(0);">
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
        try:
            with st.spinner(f"Generating {n_to_generate} questions for '{topic}'…"):
                generated = _run_async(auto_generate_prompts(
                    topic=topic, count=n_to_generate, api_key=api_key,
                    brief=focus_brief, examples=custom_prompts or None,
                    exclude_names=all_names,
                )) if n_to_generate > 0 else []
        except Exception as _gen_exc:
            _lbl, _detail = _classify_error(str(_gen_exc))
            st.markdown(_error_html(
                f"Question generation failed — {_lbl}",
                _detail,
            ), unsafe_allow_html=True)
            st.stop()

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
                 stroke="#00aac9" stroke-width="1.75" stroke-linecap="round">
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

    # Always derive models from the live sidebar selection — not from the
    # snapshot taken at "Generate Questions" time. The sidebar multiselect
    # is visible on every stage, so the user may change it before hitting
    # "Run Scan". Using the live value ensures the scan matches what they see.
    live_labels  = _ss.get("model_multiselect", [])
    live_models  = [(mid, label) for mid, label in AVAILABLE_MODELS if label in live_labels]
    # Fall back to pending_config if session key isn't populated yet
    if not live_models:
        live_models = cfg.get("models", [])

    model_names = ", ".join(label for _, label in live_models)
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

    if not live_models:
        st.markdown(_warning_html(
            "No models selected",
            "Select at least one AI model in the sidebar before running the scan.",
        ), unsafe_allow_html=True)

    st.write("")
    col_back, col_run = st.columns([1, 3])
    with col_back:
        if st.button("Back", use_container_width=True):
            _ss.app_stage = "config"; st.rerun()
    with col_run:
        can_run = bool(approved and live_models)
        lbl = (f"Run Scan — {len(approved)} questions × {len(live_models)} models"
               if can_run else "Add at least one question and select a model to continue")
        if st.button(lbl, type="primary", disabled=not can_run,
                     use_container_width=True):
            # Sync models to whatever is currently selected in the sidebar
            cfg["models"] = live_models
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

            # ── Response quality checks ───────────────────────────────────────
            from collections import Counter as _Counter, defaultdict as _defaultdict

            ok_count      = sum(1 for r in results if not r.error)
            error_results = [r for r in results if r.error]
            empty_results = [r for r in results if not r.error and not r.response]

            query_status.write(f"**{ok_count} / {real_total} responses received**")

            # Group errors by model and by error class so each model gets one
            # clear explanation rather than N identical warnings.
            if error_results:
                by_model: dict = _defaultdict(list)
                for r in error_results:
                    by_model[r.model_label].append(r.error or "")
                for model_label, errs in sorted(by_model.items()):
                    # Pick the most representative error for this model
                    sample = errs[0]
                    lbl, detail = _classify_error(sample)
                    st.markdown(_warning_html(
                        f"{model_label} — {len(errs)}/{len(prompt_list)} requests failed · {lbl}",
                        detail,
                    ), unsafe_allow_html=True)

            # Warn about models that responded but returned empty content
            if empty_results:
                empty_models = sorted({r.model_label for r in empty_results})
                st.markdown(_warning_html(
                    "Empty responses received",
                    f"{', '.join(empty_models)} returned responses with no text content. "
                    "This can happen with thinking/reasoning models whose answer appears in a "
                    "separate field. Try switching to a non-thinking variant (e.g. Gemini 2.5 Flash).",
                ), unsafe_allow_html=True)

            if ok_count == 0:
                st.markdown(_error_html(
                    "All requests failed",
                    "No model returned a usable response. Check your API key and model selection, "
                    "then try again.",
                ), unsafe_allow_html=True)
                st.stop()

            _prog(0.75, f"{ok_count}/{real_total} responses — detecting mentions…")

            st.write("Detecting brand mentions…")
            run_id = insert_run(DB_PATH, topic=topic_run, period=_auto_period())
            total_mentions = 0
            for r in results:
                qid = insert_query(DB_PATH, run_id=run_id,
                                   model_id=r.model_id, model_label=r.model_label,
                                   prompt=r.prompt, response=r.response)
                if r.response:
                    hits = detect_all_mentions(r.response, company_refs)
                    total_mentions += len(hits)
                    for hit in hits:
                        insert_mention(DB_PATH, query_id=qid,
                                       company_name=hit["company_name"], is_target=hit["is_target"],
                                       match_type=hit["type"], excerpt=hit["excerpt"])

            if total_mentions == 0:
                st.markdown(_warning_html(
                    "No brand mentions detected",
                    f"None of the {ok_count} responses contained '{cfg['target_name']}' or any "
                    f"competitor name. This usually means: (1) the topic or questions are too broad "
                    f"and AI models answered generically without naming companies, or (2) the company "
                    f"names / aliases in the sidebar don't match how AI models refer to them. "
                    f"Try adding aliases or refining the topic.",
                ), unsafe_allow_html=True)

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
        _lbl, _detail = _classify_error(str(exc))
        st.markdown(_error_html(f"Scan failed — {_lbl}", _detail), unsafe_allow_html=True)
        with st.expander("Technical details"):
            st.code(str(exc), language=None)
        if st.button("Back to Configuration", type="primary"):
            _ss.app_stage = "config"; st.rerun()
    else:
        _ss.app_stage = "config"
        st.rerun()
