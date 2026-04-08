"""AI Share of Voice — Streamlit interface."""
from __future__ import annotations

import asyncio
import pathlib
import sys
import threading
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
    result = [None]
    exc    = [None]
    def _target():
        try:    result[0] = asyncio.run(coro)
        except Exception as e: exc[0] = e
    t = threading.Thread(target=_target, daemon=True)
    t.start(); t.join()
    if exc[0]: raise exc[0]
    return result[0]

# ── Constants ─────────────────────────────────────────────────────────────────

MODELS = [
    ("openai/gpt-5.4",               "GPT-5.4"),
    ("anthropic/claude-sonnet-4.6",  "Claude Sonnet 4.6"),
    ("google/gemini-3.1-pro-preview","Gemini 3.1 Pro"),
    ("perplexity/sonar-pro",         "Perplexity Sonar Pro"),
]
DB_PATH       = "data/sov.db"
TEMPLATE_PATH = "templates/report.html.j2"
REPORTS_DIR   = "reports"

_DEFAULT_COMPETITORS = ["McKinsey & Company", "Boston Consulting Group", "Bain & Company"]

def _auto_period() -> str:
    d = date.today()
    return f"Q{(d.month - 1) // 3 + 1} {d.year}"

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
*, *::before, *::after { font-family: 'Outfit', system-ui, sans-serif !important; }

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

/* ─── Main-area primary button ────────────────────────── */
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

/* ─── Secondary / back button ─────────────────────────── */
.stButton > button[kind="secondary"] {
  background: var(--surface) !important; color: var(--text) !important;
  border: 1px solid var(--border) !important; border-radius: 8px !important;
  font-size: 13px !important; font-weight: 500 !important;
  transition: border-color 0.15s, background 0.15s !important;
}
.stButton > button[kind="secondary"]:hover {
  background: var(--bg) !important; border-color: #A1A1AA !important;
}

/* ─── Download button (prominent teal outline) ─────────── */
.stDownloadButton > button {
  background: var(--surface) !important; color: var(--accent) !important;
  border: 1.5px solid var(--accent) !important; border-radius: 8px !important;
  font-size: 13px !important; font-weight: 600 !important;
  transition: background 0.15s, border-color 0.15s !important;
}
.stDownloadButton > button:hover {
  background: var(--accent-light) !important;
  border-color: var(--accent-dim) !important;
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
.prompt-review-meta {
  display: flex; align-items: center; gap: 12px; margin-bottom: 14px;
}
.prompt-count-badge {
  font-size: 12px; font-weight: 700; color: var(--accent);
  background: var(--accent-light); border-radius: 20px;
  padding: 3px 11px; white-space: nowrap;
}
.prompt-review-hint { font-size: 12px; color: var(--muted); }

/* ─── Report section ──────────────────────────────────── */
.report-header {
  display: flex; align-items: center; gap: 16px;
  padding: 0 0 1.25rem; border-bottom: 1px solid var(--border);
  margin-bottom: 1.5rem;
}
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

/* ─── Monospace metrics ───────────────────────────────── */
[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace !important; }

/* ─── Dev watermark ───────────────────────────────────── */
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
if "app_stage"        not in _ss: _ss.app_stage        = "config"   # config|review|scanning
if "report_html"      not in _ss: _ss.report_html      = None
if "report_path"      not in _ss: _ss.report_path      = None
if "pending_prompts"  not in _ss: _ss.pending_prompts  = []
if "pending_config"   not in _ss: _ss.pending_config   = {}
if "competitors"      not in _ss: _ss.competitors      = list(_DEFAULT_COMPETITORS)
if "comp_input_key"   not in _ss: _ss.comp_input_key   = 0
if "prompt_ver"       not in _ss: _ss.prompt_ver       = 0

# ── Callbacks ─────────────────────────────────────────────────────────────────

def _remove_comp(idx: int):
    if 0 <= idx < len(_ss.competitors):
        _ss.competitors.pop(idx)

def _add_comp():
    val = _ss.get(f"nc_{_ss.comp_input_key}", "").strip()
    if val and val not in _ss.competitors:
        _ss.competitors.append(val)
    _ss.comp_input_key += 1

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="margin-bottom:1.5rem">
      <div style="font-size:13px;font-weight:700;letter-spacing:-0.02em;color:#18181B">SOV Scanner</div>
      <div style="font-size:12px;color:#71717A;margin-top:2px">Configuration</div>
    </div>""", unsafe_allow_html=True)

    # ── API key ────────────────────────────────────────────────────────────
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
        api_key = st.text_input("API Key", type="password",
                                placeholder="sk-or-v1-...", label_visibility="collapsed")

    st.markdown('<hr style="margin:1.25rem 0">', unsafe_allow_html=True)

    # ── Target ─────────────────────────────────────────────────────────────
    st.markdown(
        '<span class="sov-label">Your Target</span>'
        '<span class="sov-hint">The brand or company whose AI visibility you want to measure.</span>',
        unsafe_allow_html=True,
    )
    target_name = st.text_input("Name", value="Roland Berger",
                                placeholder="Company or brand name",
                                label_visibility="collapsed")
    st.markdown(
        '<span style="font-size:11px;color:#71717A;display:block;margin-bottom:4px">'
        'Alternative names (separated by |)</span>',
        unsafe_allow_html=True,
    )
    target_aliases = st.text_input("Aliases",
                                   value="Roland Berger Strategy Consultants",
                                   placeholder="Full name | Short name | Abbreviation",
                                   label_visibility="collapsed",
                                   help="Other names AI might use — e.g. abbreviations or trading names")

    st.markdown('<hr style="margin:1.25rem 0">', unsafe_allow_html=True)

    # ── Competitors ────────────────────────────────────────────────────────
    st.markdown(
        '<span class="sov-label">Competitors</span>'
        '<span class="sov-hint">Add companies to benchmark your visibility against.</span>',
        unsafe_allow_html=True,
    )

    for i, comp in enumerate(_ss.competitors):
        c_name, c_btn = st.columns([5, 1])
        with c_name:
            st.markdown(f'<div class="comp-tag">{comp}</div>', unsafe_allow_html=True)
        with c_btn:
            st.button("×", key=f"rm_{i}", on_click=_remove_comp, args=(i,))

    c_in, c_add = st.columns([5, 1])
    with c_in:
        st.text_input("new_comp", key=f"nc_{_ss.comp_input_key}",
                      label_visibility="collapsed", placeholder="Add competitor…")
    with c_add:
        st.button("+", key="add_comp", on_click=_add_comp, type="primary")

    st.markdown('<hr style="margin:1.25rem 0">', unsafe_allow_html=True)

    # ── Scan settings ──────────────────────────────────────────────────────
    st.markdown(
        '<span class="sov-label">Scan Settings</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<span style="font-size:11px;color:#71717A;display:block;margin-bottom:4px">'
        'Topic — what should AI be asked about?</span>',
        unsafe_allow_html=True,
    )
    topic = st.text_input("Topic", value="strategy consulting germany",
                          label_visibility="collapsed",
                          placeholder="e.g. ERP software for retail, strategy consulting")

    num_prompts = st.slider("Questions per AI model", min_value=2, max_value=50, value=10,
                            help="Each question is sent to all 4 AI models. 10 is a good starting point; use 20+ for production-quality results.")

    # Estimated request count
    n_est = num_prompts * len(MODELS)
    st.markdown(
        f'<div style="font-size:11px;color:#71717A;margin-top:-6px;margin-bottom:8px">'
        f'Approx. {n_est} API requests total ({len(MODELS)} models)</div>',
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

# ── Helpers: step indicator ───────────────────────────────────────────────────

def _step_indicator(active: int):
    """Render a 3-step indicator. active = 0|1|2"""
    steps = ["Configure", "Review Questions", "Scan"]
    parts = []
    for i, label in enumerate(steps):
        if i < active:
            cls = "done"
            icon = "✓"
        elif i == active:
            cls = "active"
            icon = str(i + 1)
        else:
            cls = ""
            icon = str(i + 1)
        parts.append(
            f'<div class="sov-step {cls}">'
            f'<div class="sov-step-num">{icon}</div>'
            f'<div class="sov-step-label">{label}</div>'
            f'</div>'
        )
        if i < len(steps) - 1:
            done_cls = "done" if i < active else ""
            parts.append(f'<div class="sov-connector {done_cls}"></div>')
    st.markdown(f'<div class="sov-steps">{"".join(parts)}</div>', unsafe_allow_html=True)

# ── Page header (always shown) ────────────────────────────────────────────────

st.markdown("""
<div class="sov-header">
  <div class="sov-badge">AI Visibility Intelligence</div>
  <h1 class="sov-title">Share of Voice Scanner</h1>
  <p class="sov-subtitle">
    Track how AI models mention your brand across any topic — benchmark against competitors
    and find where you're missing.
  </p>
</div>
""", unsafe_allow_html=True)

# ── Stage routing ─────────────────────────────────────────────────────────────

stage = _ss.app_stage

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Configure
# ══════════════════════════════════════════════════════════════════════════════

if stage == "config":
    _step_indicator(active=0)

    # Prompt focus expander
    with st.expander("Optional: Focus the generated questions", expanded=False):
        st.markdown("""
        <div style="padding:4px 0 12px;font-size:13px;color:#71717A;line-height:1.65">
          Tell the tool to focus on a specific angle — persona, region, or deal type.
          Leave blank for broad, varied coverage.
        </div>""", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div style="font-size:12px;font-weight:500;color:#71717A;margin-bottom:4px">Focus note</div>',
                        unsafe_allow_html=True)
            focus_brief = st.text_input(
                "focus_brief", label_visibility="collapsed",
                placeholder="e.g. CFO persona, DACH region, mid-market deals",
                value=_ss.get("focus_brief", ""),
            )
        with col_b:
            st.markdown('<div style="font-size:12px;font-weight:500;color:#71717A;margin-bottom:4px">'
                        'Your own questions (one per line)</div>', unsafe_allow_html=True)
            custom_prompts_raw = st.text_area(
                "custom_prompts", label_visibility="collapsed",
                placeholder="Which consulting firms do German CFOs trust for restructuring?\nBest strategy advisors for mid-size industrials in DACH?",
                height=96,
                value=_ss.get("custom_prompts_raw", ""),
                help="These are always included first and used as style examples for AI-generated questions.",
            )

    st.write("")

    if not api_key:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:10px;padding:12px 16px;
             background:#FFF7ED;border:1px solid #FED7AA;border-radius:10px;margin-bottom:16px">
          <div style="font-size:13px;color:#92400E">
            Add your OpenRouter API key in the sidebar to continue.
            Get a free key at <a href="https://openrouter.ai" target="_blank"
            style="color:#B45309;font-weight:500">openrouter.ai</a>.
          </div>
        </div>""", unsafe_allow_html=True)

    can_generate = bool(api_key and topic.strip() and target_name.strip())
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
            "alias_list":     [a.strip() for a in target_aliases.split("|") if a.strip()],
            "competitor_list": list(_ss.competitors),
            "topic":          topic,
            "api_key":        api_key,
        }
        _ss.focus_brief        = focus_brief
        _ss.custom_prompts_raw = custom_prompts_raw
        _ss.prompt_ver        += 1
        _ss.app_stage          = "review"
        st.rerun()

    # ── Report (if a previous scan exists) ───────────────────────────────────
    if _ss.report_html:
        st.markdown('<hr style="margin:2rem 0">', unsafe_allow_html=True)

        report_name = pathlib.Path(_ss.report_path).name
        col_meta, col_dl, col_new = st.columns([3, 1.4, 1.1])
        with col_meta:
            st.markdown(f"""
            <div class="report-meta">
              <div class="report-meta-title">Latest Report</div>
              <div class="report-meta-file">{report_name}</div>
            </div>""", unsafe_allow_html=True)
        with col_dl:
            st.download_button(
                label="Download Report",
                data=_ss.report_html,
                file_name=report_name,
                mime="text/html",
                use_container_width=True,
            )
        with col_new:
            if st.button("New Scan", use_container_width=True):
                _ss.report_html = None
                _ss.report_path = None
                st.rerun()

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
# STAGE 2 — Review prompts
# ══════════════════════════════════════════════════════════════════════════════

elif stage == "review":
    _step_indicator(active=1)

    prompts = _ss.pending_prompts
    cfg     = _ss.pending_config

    # Meta row: count badge + hint
    n_prompts  = len(prompts)
    n_requests = n_prompts * len(MODELS)
    st.markdown(f"""
    <div class="prompt-review-meta">
      <span class="prompt-count-badge">{n_prompts} questions</span>
      <span class="prompt-review-hint">
        Review and edit below — remove anything off-topic, add your own.
        These will be sent to <strong>{len(MODELS)} AI models</strong>
        ({n_requests} requests total).
      </span>
    </div>""", unsafe_allow_html=True)

    # Editable text area (one prompt per line)
    textarea_val = st.text_area(
        "questions",
        value="\n".join(prompts),
        height=min(520, max(240, n_prompts * 44)),
        label_visibility="collapsed",
        key=f"review_ta_{_ss.prompt_ver}",
        help="One question per line. Edit, delete, or reorder freely.",
        placeholder="Add or edit questions here, one per line…",
    )

    approved = [l.strip() for l in textarea_val.splitlines() if l.strip()]
    n_approved = len(approved)

    # Show live count if it changed
    if n_approved != n_prompts:
        st.markdown(
            f'<div style="font-size:12px;color:#71717A;margin-top:4px">'
            f'{n_approved} questions selected</div>',
            unsafe_allow_html=True,
        )

    st.write("")
    col_back, col_run = st.columns([1, 3])
    with col_back:
        if st.button("Back", use_container_width=True):
            _ss.app_stage = "config"
            st.rerun()
    with col_run:
        run_label = (
            f"Run Scan — {n_approved} questions × {len(MODELS)} models"
            if n_approved > 0 else "Add at least one question to continue"
        )
        if st.button(run_label, type="primary", disabled=n_approved == 0,
                     use_container_width=True):
            _ss.pending_prompts = approved
            _ss.app_stage       = "scanning"
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — Scanning
# ══════════════════════════════════════════════════════════════════════════════

elif stage == "scanning":
    _step_indicator(active=2)

    cfg         = _ss.pending_config
    prompt_list = _ss.pending_prompts
    api_key_run = cfg["api_key"]
    topic_run   = cfg["topic"]

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

    _prog(0.05, "Starting scan…")

    try:
        with st.status("Running scan…", expanded=True) as status:

            # 1 — Querying models
            real_total = len(prompt_list) * len(MODELS)
            st.write(f"Querying {len(MODELS)} AI models — {real_total} total requests…")
            _prog(0.15, f"Querying {len(MODELS)} models across {len(prompt_list)} questions…")
            results = _run_async(run_queries(prompt_list, MODELS, api_key_run, max_concurrent=5))
            errors  = sum(1 for r in results if r.error)
            st.write(f"**{len(results) - errors} responses received**" +
                     (f" — {errors} failed" if errors else ""))
            _prog(0.75, f"{len(results) - errors}/{real_total} responses — detecting mentions…")

            # 2 — Detect mentions
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

            # 3 — Report
            st.write("Generating report…")
            queries    = get_queries_for_run(DB_PATH, run_id)
            mentions   = get_mentions_for_run(DB_PATH, run_id)
            run_record = get_run(DB_PATH, run_id)

            report_path = generate_report(
                run=run_record, queries=queries, mentions=mentions, companies=companies,
                template_path=TEMPLATE_PATH, output_dir=REPORTS_DIR,
            )
            _ss.report_html = pathlib.Path(report_path).read_text(encoding="utf-8")
            _ss.report_path = report_path
            _prog(1.0, "Scan complete")
            status.update(label="Scan complete", state="complete")

    except Exception as exc:
        st.error(f"Scan failed: {exc}")
        if st.button("Back to Configuration"):
            _ss.app_stage = "config"
            st.rerun()
    else:
        _ss.app_stage = "config"
        st.rerun()
