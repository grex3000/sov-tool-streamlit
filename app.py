"""AI Share of Voice — Streamlit interface."""
from __future__ import annotations

import asyncio
import json
import pathlib
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
from datetime import date

import streamlit as st
import streamlit.components.v1 as components

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

# ── Async helper ──────────────────────────────────────────────────────────────

def _run_async(coro):
    result = [None]; exc = [None]
    def _t():
        try:    result[0] = asyncio.run(coro)
        except Exception as e: exc[0] = e
    t = threading.Thread(target=_t, daemon=True); t.start(); t.join()
    if exc[0]: raise exc[0]
    return result[0]

# ── Share helper ──────────────────────────────────────────────────────────────

def _upload_gist(html: str, filename: str, token: str) -> str:
    """Upload HTML as a secret GitHub Gist. Returns an htmlpreview.github.io URL."""
    body = json.dumps({
        "description": "AI Share of Voice Report",
        "public": False,
        "files": {filename: {"content": html}},
    }).encode()
    req = urllib.request.Request(
        "https://api.github.com/gists",
        data=body,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
            "User-Agent": "sov-scanner/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())
    owner   = data["owner"]["login"]
    gist_id = data["id"]
    raw     = f"https://gist.githubusercontent.com/{owner}/{gist_id}/raw/{filename}"
    return f"https://htmlpreview.github.io/?{urllib.parse.quote(raw, safe=':/')}"

# ── Constants ─────────────────────────────────────────────────────────────────

AVAILABLE_MODELS: list[tuple[str, str]] = [
    ("openai/gpt-5.4",                          "GPT-5.4"),
    ("openai/gpt-4o",                           "GPT-4o"),
    ("anthropic/claude-sonnet-4.6",             "Claude Sonnet 4.6"),
    ("anthropic/claude-opus-4",                 "Claude Opus 4"),
    ("anthropic/claude-haiku-4-5-20251001",     "Claude Haiku 4.5"),
    ("google/gemini-3.1-pro-preview",           "Gemini 3.1 Pro"),
    ("google/gemini-2.5-flash-preview:thinking","Gemini 2.5 Flash"),
    ("perplexity/sonar-pro",                    "Perplexity Sonar Pro"),
    ("perplexity/sonar",                        "Perplexity Sonar"),
    ("meta-llama/llama-3.3-70b-instruct",       "Llama 3.3 70B"),
]
_DEFAULT_MODEL_IDS = [
    "openai/gpt-5.4",
    "anthropic/claude-sonnet-4.6",
    "google/gemini-3.1-pro-preview",
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
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><circle cx='50' cy='50' r='45' fill='%230D9488'/></svg>",
    layout="wide",
)

# ── CSS ────────────────────────────────────────────────────────────────────────

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">

<style>
:root {
  --accent:       #0D9488;
  --accent-light: #CCFBF1;
  --accent-dim:   #0F766E;
  --bg:           #F6F7F8;
  --surface:      #FFFFFF;
  --text:         #18181B;
  --muted:        #71717A;
  --border:       #E4E4E7;
}

html, body, .stApp, [class*="block-container"] { background-color: var(--bg) !important; }

/* Apply Outfit broadly but restore icon fonts below */
* { font-family: 'Outfit', system-ui, sans-serif !important; }

/* ── Restore Streamlit icon fonts (fixes arrow_right/arrow_down text) ── */
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

/* ─── Sidebar ─────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 2rem 1.25rem !important; }

.sov-label {
  font-size: 11px; font-weight: 600; letter-spacing: 0.1em;
  text-transform: uppercase; color: var(--muted);
  margin: 1.5rem 0 0.25rem; display: block;
}
.sov-hint {
  font-size: 11.5px; color: var(--muted); line-height: 1.55;
  margin: 0 0 8px; display: block;
}

/* ─── Inputs ──────────────────────────────────────────── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
  border: 1px solid var(--border) !important;
  border-radius: 8px !important; font-size: 14px !important;
  color: var(--text) !important; background: var(--surface) !important;
  transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
  padding: 10px 12px !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(13,148,136,0.12) !important;
  outline: none !important;
}
[data-testid="stTextInput"] label,
[data-testid="stTextArea"] label,
[data-testid="stSlider"] label {
  font-size: 12px !important; font-weight: 500 !important;
  color: var(--muted) !important; margin-bottom: 4px !important;
}

/* ─── Multiselect ─────────────────────────────────────── */
[data-testid="stMultiSelect"] [data-baseweb="select"] > div {
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  background: var(--surface) !important;
  font-size: 13px !important;
}
[data-testid="stMultiSelect"] [data-baseweb="select"] > div:focus-within {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px rgba(13,148,136,0.12) !important;
}
[data-baseweb="tag"] {
  background: var(--accent-light) !important;
  border-radius: 5px !important;
}
[data-baseweb="tag"] span { color: var(--accent-dim) !important; font-size: 12px !important; }

/* ─── Primary button (main area) ──────────────────────── */
.main .stButton > button[kind="primary"] {
  background: var(--accent) !important; color: #fff !important;
  border: none !important; border-radius: 10px !important;
  font-size: 14px !important; font-weight: 600 !important;
  letter-spacing: -0.01em !important; height: 48px !important;
  transition: background 0.15s, transform 0.1s, box-shadow 0.15s !important;
  box-shadow: 0 1px 3px rgba(13,148,136,0.25) !important;
}
.main .stButton > button[kind="primary"]:hover {
  background: var(--accent-dim) !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 16px rgba(13,148,136,0.22) !important;
}
.main .stButton > button[kind="primary"]:active {
  transform: scale(0.99) translateY(0) !important; box-shadow: none !important;
}
.main .stButton > button[kind="primary"]:disabled {
  background: #D1D5DB !important; box-shadow: none !important; transform: none !important;
}

/* ─── Secondary buttons ───────────────────────────────── */
.stButton > button[kind="secondary"] {
  background: var(--surface) !important; color: var(--text) !important;
  border: 1px solid var(--border) !important; border-radius: 8px !important;
  font-size: 13px !important; font-weight: 500 !important;
  transition: border-color 0.15s, background 0.15s !important;
}
.stButton > button[kind="secondary"]:hover {
  background: var(--bg) !important; border-color: #A1A1AA !important;
}

/* ─── Download button ─────────────────────────────────── */
.stDownloadButton > button {
  background: var(--surface) !important; color: var(--accent) !important;
  border: 1.5px solid var(--accent) !important; border-radius: 8px !important;
  font-size: 13px !important; font-weight: 600 !important;
  transition: background 0.15s, border-color 0.15s !important;
}
.stDownloadButton > button:hover {
  background: var(--accent-light) !important; border-color: var(--accent-dim) !important;
}

/* ─── Sidebar: + button (primary) ─────────────────────── */
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
  background: var(--accent) !important; color: #fff !important;
  border: none !important; border-radius: 7px !important;
  height: 36px !important; min-height: 36px !important;
  padding: 0 12px !important; font-size: 18px !important;
  font-weight: 400 !important; line-height: 1 !important;
  box-shadow: none !important; transform: none !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
  background: var(--accent-dim) !important; transform: none !important;
}

/* ─── Sidebar: × remove buttons ────────────────────────── */
[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
  background: transparent !important; color: var(--muted) !important;
  border: 1px solid var(--border) !important; border-radius: 7px !important;
  height: 36px !important; min-height: 36px !important;
  padding: 0 10px !important; font-size: 15px !important;
  font-weight: 400 !important; line-height: 1 !important;
  transition: background 0.12s, border-color 0.12s, color 0.12s !important;
}
[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
  background: #FFF1F2 !important; border-color: #FECDD3 !important;
  color: #E11D48 !important;
}

.comp-tag {
  font-size: 13px; font-weight: 500; color: var(--text);
  padding: 8px 0 8px 2px; line-height: 1.3;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}

/* ─── Slider ──────────────────────────────────────────── */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
  background: var(--accent) !important; border-color: var(--accent) !important;
}

/* ─── Divider ─────────────────────────────────────────── */
hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 1.5rem 0 !important; }

/* ─── Expander ────────────────────────────────────────── */
[data-testid="stExpander"] {
  border: 1px solid var(--border) !important; border-radius: 10px !important;
  background: var(--surface) !important; overflow: hidden !important;
}
[data-testid="stExpander"] summary {
  font-size: 13px !important; font-weight: 500 !important;
  color: var(--text) !important; padding: 14px 18px !important;
}
[data-testid="stExpander"] summary:hover { background: var(--bg) !important; }

/* ─── Status / alerts ─────────────────────────────────── */
[data-testid="stStatusWidget"], [data-testid="stStatus"] {
  border: 1px solid var(--border) !important; border-radius: 10px !important;
  background: var(--surface) !important; font-size: 14px !important;
}
[data-testid="stAlert"] { border-radius: 10px !important; font-size: 13px !important; }

/* ─── Progress bar ────────────────────────────────────── */
[data-testid="stProgressBar"] > div > div {
  background: var(--accent) !important; border-radius: 4px !important;
}
[data-testid="stProgressBar"] > div {
  border-radius: 4px !important; background: var(--accent-light) !important;
}
.progress-label {
  font-size: 12px; font-weight: 500; color: var(--muted);
  letter-spacing: 0.01em; margin-bottom: 6px;
}

/* ─── Main block ──────────────────────────────────────── */
.main .block-container {
  padding-top: 2.5rem !important; padding-bottom: 3rem !important;
  max-width: 900px !important;
}

/* ─── Page header ─────────────────────────────────────── */
.sov-header { padding-bottom: 1.75rem; border-bottom: 1px solid var(--border); margin-bottom: 1.5rem; }
.sov-badge {
  display: inline-flex; align-items: center; gap: 8px;
  font-size: 11px; font-weight: 600; letter-spacing: 0.1em;
  text-transform: uppercase; color: var(--accent); margin-bottom: 12px;
}
.sov-badge::before {
  content: ''; width: 6px; height: 6px; border-radius: 50%;
  background: var(--accent); display: block;
  animation: pulse 2.4s ease-in-out infinite;
}
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.4;transform:scale(.75)} }
.sov-title {
  font-size: 34px; font-weight: 700; letter-spacing: -0.03em;
  line-height: 1.1; color: var(--text); margin: 0 0 8px;
}
.sov-subtitle { font-size: 15px; color: var(--muted); margin: 0; max-width: 540px; }

/* ─── Step indicator ──────────────────────────────────── */
.sov-steps {
  display: flex; align-items: center;
  padding: 1.25rem 0 1.5rem; border-bottom: 1px solid var(--border);
  margin-bottom: 1.75rem;
}
.sov-step { display: flex; align-items: center; gap: 7px; flex-shrink: 0; }
.sov-step-num {
  width: 24px; height: 24px; border-radius: 50%;
  border: 1.5px solid var(--border); background: var(--bg);
  color: var(--muted); font-size: 11px; font-weight: 700;
  display: flex; align-items: center; justify-content: center;
}
.sov-step-label { font-size: 12px; font-weight: 500; color: var(--muted); white-space: nowrap; }
.sov-step.active .sov-step-num { background: var(--accent); border-color: var(--accent); color: #fff; }
.sov-step.active .sov-step-label { color: var(--text); font-weight: 600; }
.sov-step.done .sov-step-num { background: var(--accent-light); border-color: var(--accent-light); color: var(--accent); }
.sov-connector { flex:1; min-width:12px; max-width:40px; height:1px; background:var(--border); margin:0 4px; }

/* ─── Prompt review ───────────────────────────────────── */
.prompt-review-meta { display: flex; align-items: center; gap: 12px; margin-bottom: 14px; }
.prompt-count-badge {
  font-size: 12px; font-weight: 700; color: var(--accent);
  background: var(--accent-light); border-radius: 20px; padding: 3px 11px; white-space: nowrap;
}
.prompt-review-hint { font-size: 12px; color: var(--muted); }

/* ─── Share URL box ───────────────────────────────────── */
.share-url-box {
  background: var(--accent-light); border: 1px solid #99F6E4;
  border-radius: 10px; padding: 14px 16px; margin-top: 8px;
}
.share-url-label {
  font-size: 11px; font-weight: 600; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--accent-dim); margin-bottom: 6px;
}

/* ─── Report section ──────────────────────────────────── */
.report-meta { flex: 1; }
.report-meta-title { font-size: 15px; font-weight: 600; color: var(--text); letter-spacing:-0.01em; margin:0; }
.report-meta-file { font-size: 12px; color: var(--muted); font-family:'JetBrains Mono',monospace !important; margin-top:2px; }

/* ─── Empty state ─────────────────────────────────────── */
.sov-empty {
  display: flex; flex-direction: column; align-items: center;
  justify-content: center; padding: 72px 24px; text-align: center;
  border: 1px dashed var(--border); border-radius: 16px;
  background: var(--surface); margin-top: 2rem;
}
.sov-empty-icon {
  width: 48px; height: 48px; border-radius: 12px;
  background: var(--accent-light); display: flex;
  align-items: center; justify-content: center; margin-bottom: 20px;
}
.sov-empty-title { font-size: 16px; font-weight: 600; color: var(--text); margin: 0 0 8px; }
.sov-empty-desc { font-size: 14px; color: var(--muted); margin: 0; max-width: 360px; line-height: 1.65; }

/* ─── Misc ────────────────────────────────────────────── */
[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace !important; }

.sov-watermark {
  position: fixed; bottom: 18px; right: 22px; font-size: 11px;
  font-weight: 500; color: var(--muted); letter-spacing: 0.03em;
  opacity: 0.55; z-index: 9999; pointer-events: none;
  font-family: 'JetBrains Mono', monospace !important;
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
    "report_share_url": None,
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
    <div style="margin-bottom:1.5rem">
      <div style="font-size:13px;font-weight:700;letter-spacing:-0.02em;color:#18181B">SOV Scanner</div>
      <div style="font-size:12px;color:#71717A;margin-top:2px">Configuration</div>
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
  <div class="sov-badge">AI Visibility Intelligence</div>
  <h1 class="sov-title">Share of Voice Scanner</h1>
  <p class="sov-subtitle">
    Track how AI models mention your brand — benchmark against competitors
    and find where you're missing.
  </p>
</div>
""", unsafe_allow_html=True)

# ── Stage routing ─────────────────────────────────────────────────────────────

stage = _ss.app_stage
github_token = st.secrets.get("GITHUB_TOKEN", "")

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

        with st.spinner(f"Generating {n_to_generate} questions for '{topic}'…"):
            generated = _run_async(auto_generate_prompts(
                topic=topic, count=n_to_generate, api_key=api_key,
                brief=focus_brief, examples=custom_prompts or None,
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

        # Action row
        col_meta, col_dl, col_share, col_new = st.columns([3, 1.3, 1.3, 1])
        with col_meta:
            st.markdown(f"""
            <div class="report-meta">
              <div class="report-meta-title">Latest Report</div>
              <div class="report-meta-file">{report_name}</div>
            </div>""", unsafe_allow_html=True)
        with col_dl:
            st.download_button("Download Report", data=_ss.report_html,
                               file_name=report_name, mime="text/html",
                               use_container_width=True)
        with col_share:
            if github_token:
                share_label = "Shared" if _ss.report_share_url else "Share Link"
                share_disabled = bool(_ss.report_share_url)
                if st.button(share_label, disabled=share_disabled,
                             use_container_width=True):
                    with st.spinner("Publishing report…"):
                        try:
                            url = _upload_gist(_ss.report_html, report_name, github_token)
                            _ss.report_share_url = url
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Could not share: {exc}")
            else:
                st.markdown(
                    '<div style="font-size:11px;color:#71717A;padding-top:10px">'
                    'Add GITHUB_TOKEN to secrets to enable sharing.</div>',
                    unsafe_allow_html=True,
                )
        with col_new:
            if st.button("New Scan", use_container_width=True):
                _ss.report_html      = None
                _ss.report_path      = None
                _ss.report_share_url = None
                st.rerun()

        # Share URL
        if _ss.report_share_url:
            st.markdown("""
            <div class="share-url-box">
              <div class="share-url-label">Shareable link — copy and send to anyone</div>
            </div>""", unsafe_allow_html=True)
            st.code(_ss.report_share_url, language=None)
            st.markdown(
                '<div style="font-size:11px;color:#71717A;margin-top:4px">'
                'This is a private link hosted on GitHub Gist. '
                'Anyone with this URL can view the report.</div>',
                unsafe_allow_html=True,
            )

        st.write("")
        components.html(_ss.report_html, height=1080, scrolling=True)

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
            st.write(f"Querying {len(models_run)} models — {real_total} total requests…")
            _prog(0.15, f"Querying {len(models_run)} models × {len(prompt_list)} questions…")

            results = _run_async(run_queries(prompt_list, models_run, api_key_run, max_concurrent=5))
            errors  = sum(1 for r in results if r.error)
            st.write(f"**{len(results) - errors} responses received**" +
                     (f" — {errors} failed" if errors else ""))
            _prog(0.75, f"{len(results) - errors}/{real_total} responses — detecting mentions…")

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
            _ss.report_share_url = None  # clear any old share URL
            _prog(1.0, "Scan complete")
            status.update(label="Scan complete", state="complete")

    except Exception as exc:
        st.error(f"Scan failed: {exc}")
        if st.button("Back to Configuration"):
            _ss.app_stage = "config"; st.rerun()
    else:
        _ss.app_stage = "config"
        st.rerun()
