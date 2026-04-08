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
    q = (d.month - 1) // 3 + 1
    return f"Q{q} {d.year}"

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Share of Voice",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><circle cx='50' cy='50' r='45' fill='%230D9488'/></svg>",
    layout="wide",
)

# ── CSS injection ─────────────────────────────────────────────────────────────

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

/* ─── Base ────────────────────────────────────────── */
html, body, .stApp, [class*="block-container"] {
  background-color: var(--bg) !important;
}
*, *::before, *::after {
  font-family: 'Outfit', system-ui, sans-serif !important;
}

/* ─── Hide Streamlit chrome ───────────────────────── */
#MainMenu, footer, .stDeployButton,
[data-testid="stToolbar"], [data-testid="stDecoration"] {
  display: none !important;
}

/* ─── Sidebar ──────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child {
  padding: 2rem 1.25rem 2rem !important;
}

/* ─── Sidebar section labels ──────────────────────── */
.sov-label {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--muted);
  margin: 1.5rem 0 0.35rem;
  display: block;
}
.sov-label:first-child { margin-top: 0.5rem; }

/* ─── Inputs ───────────────────────────────────────── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  font-size: 14px !important;
  color: var(--text) !important;
  background: var(--surface) !important;
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
  font-size: 12px !important;
  font-weight: 500 !important;
  color: var(--muted) !important;
  margin-bottom: 4px !important;
}

/* ─── Primary button (main area) ────────────────────── */
.main .stButton > button[kind="primary"] {
  background: var(--accent) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 10px !important;
  font-size: 14px !important;
  font-weight: 600 !important;
  letter-spacing: -0.01em !important;
  height: 48px !important;
  transition: background 0.15s ease, transform 0.1s ease, box-shadow 0.15s ease !important;
  box-shadow: 0 1px 3px rgba(13,148,136,0.25) !important;
}
.main .stButton > button[kind="primary"]:hover {
  background: var(--accent-dim) !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 16px rgba(13,148,136,0.22) !important;
}
.main .stButton > button[kind="primary"]:active {
  transform: scale(0.99) translateY(0) !important;
  box-shadow: none !important;
}
.main .stButton > button[kind="primary"]:disabled {
  background: #D1D5DB !important;
  box-shadow: none !important;
  transform: none !important;
}

/* ─── Secondary / download button ───────────────────── */
.stButton > button[kind="secondary"],
.stDownloadButton > button {
  background: var(--surface) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  transition: border-color 0.15s ease, background 0.15s ease !important;
}
.stButton > button[kind="secondary"]:hover,
.stDownloadButton > button:hover {
  background: var(--bg) !important;
  border-color: #A1A1AA !important;
}

/* ─── Sidebar competitor "+ " button (primary) ────── */
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
  background: var(--accent) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 7px !important;
  height: 36px !important;
  min-height: 36px !important;
  padding: 0 12px !important;
  font-size: 18px !important;
  font-weight: 400 !important;
  line-height: 1 !important;
  box-shadow: none !important;
  transform: none !important;
  transition: background 0.15s ease !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
  background: var(--accent-dim) !important;
  transform: none !important;
  box-shadow: none !important;
}

/* ─── Sidebar competitor "×" remove buttons ──────── */
[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
  background: transparent !important;
  color: var(--muted) !important;
  border: 1px solid var(--border) !important;
  border-radius: 7px !important;
  height: 36px !important;
  min-height: 36px !important;
  padding: 0 10px !important;
  font-size: 15px !important;
  font-weight: 400 !important;
  line-height: 1 !important;
  transition: background 0.12s ease, border-color 0.12s ease, color 0.12s ease !important;
}
[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
  background: #FFF1F2 !important;
  border-color: #FECDD3 !important;
  color: #E11D48 !important;
}

/* ─── Competitor tag name ─────────────────────────── */
.comp-tag {
  font-size: 13px;
  font-weight: 500;
  color: var(--text);
  padding: 8px 0 8px 2px;
  line-height: 1.3;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* ─── Slider ─────────────────────────────────────────── */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
  background: var(--accent) !important;
  border-color: var(--accent) !important;
}

/* ─── Divider ────────────────────────────────────────── */
hr {
  border: none !important;
  border-top: 1px solid var(--border) !important;
  margin: 1.5rem 0 !important;
}

/* ─── Expander ───────────────────────────────────────── */
[data-testid="stExpander"] {
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  background: var(--surface) !important;
  overflow: hidden !important;
}
[data-testid="stExpander"] summary {
  font-size: 13px !important;
  font-weight: 500 !important;
  color: var(--text) !important;
  padding: 14px 18px !important;
}
[data-testid="stExpander"] summary:hover {
  background: var(--bg) !important;
}

/* ─── Status widget ──────────────────────────────────── */
[data-testid="stStatusWidget"],
[data-testid="stStatus"] {
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  background: var(--surface) !important;
  font-size: 14px !important;
}

/* ─── Progress bar ───────────────────────────────────── */
[data-testid="stProgressBar"] > div > div {
  background: var(--accent) !important;
  border-radius: 4px !important;
}
[data-testid="stProgressBar"] > div {
  border-radius: 4px !important;
  background: var(--accent-light) !important;
}
.progress-label {
  font-size: 12px;
  font-weight: 500;
  color: var(--muted);
  letter-spacing: 0.01em;
  margin-bottom: 6px;
}

/* ─── Alert / info box ───────────────────────────────── */
[data-testid="stAlert"] {
  border-radius: 10px !important;
  font-size: 13px !important;
}

/* ─── Main content padding ───────────────────────────── */
.main .block-container {
  padding-top: 2.5rem !important;
  padding-bottom: 3rem !important;
  max-width: 900px !important;
}

/* ─── Custom header ──────────────────────────────────── */
.sov-header {
  padding-bottom: 2rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 2rem;
}
.sov-badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--accent);
  margin-bottom: 14px;
}
.sov-badge::before {
  content: '';
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--accent);
  display: block;
  animation: pulse 2.4s ease-in-out infinite;
}
@keyframes pulse {
  0%,100% { opacity:1; transform:scale(1); }
  50%      { opacity:.4; transform:scale(.75); }
}
.sov-title {
  font-size: 34px;
  font-weight: 700;
  letter-spacing: -0.03em;
  line-height: 1.1;
  color: var(--text);
  margin: 0 0 8px;
}
.sov-subtitle {
  font-size: 16px;
  font-weight: 400;
  color: var(--muted);
  margin: 0;
  max-width: 560px;
}

/* ─── Empty state ────────────────────────────────────── */
.sov-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 72px 24px;
  text-align: center;
  border: 1px dashed var(--border);
  border-radius: 16px;
  background: var(--surface);
  margin-top: 2rem;
}
.sov-empty-icon {
  width: 48px; height: 48px;
  border-radius: 12px;
  background: var(--accent-light);
  display: flex; align-items: center; justify-content: center;
  margin-bottom: 20px;
}
.sov-empty-title {
  font-size: 16px;
  font-weight: 600;
  color: var(--text);
  margin: 0 0 8px;
}
.sov-empty-desc {
  font-size: 14px;
  color: var(--muted);
  margin: 0;
  max-width: 360px;
  line-height: 1.65;
}

/* ─── Report header ──────────────────────────────────── */
.sov-report-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 0 1.25rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 1.5rem;
}
.sov-report-title {
  font-size: 15px;
  font-weight: 600;
  color: var(--text);
  margin: 0;
  letter-spacing: -0.01em;
}
.sov-report-meta {
  font-size: 12px;
  color: var(--muted);
  font-family: 'JetBrains Mono', monospace !important;
  margin-top: 2px;
}

/* ─── Monospace numbers ───────────────────────────────── */
[data-testid="stMetricValue"] {
  font-family: 'JetBrains Mono', monospace !important;
}

/* ─── Developer watermark ────────────────────────────── */
.sov-watermark {
  position: fixed;
  bottom: 18px;
  right: 22px;
  font-size: 11px;
  font-weight: 500;
  color: var(--muted);
  letter-spacing: 0.03em;
  opacity: 0.55;
  z-index: 9999;
  pointer-events: none;
  font-family: 'JetBrains Mono', monospace !important;
}
</style>
""", unsafe_allow_html=True)

# ── Watermark ─────────────────────────────────────────────────────────────────

st.markdown('<div class="sov-watermark">developed by gregor.weindorf</div>', unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────

if "report_html"         not in st.session_state: st.session_state.report_html         = None
if "report_path"         not in st.session_state: st.session_state.report_path         = None
if "focus_brief"         not in st.session_state: st.session_state.focus_brief         = ""
if "custom_prompts_raw"  not in st.session_state: st.session_state.custom_prompts_raw  = ""
if "competitors"         not in st.session_state: st.session_state.competitors         = list(_DEFAULT_COMPETITORS)
if "comp_input_key"      not in st.session_state: st.session_state.comp_input_key      = 0

# ── Competitor callbacks ───────────────────────────────────────────────────────

def _remove_comp(idx: int):
    if 0 <= idx < len(st.session_state.competitors):
        st.session_state.competitors.pop(idx)

def _add_comp():
    key = f"nc_{st.session_state.comp_input_key}"
    val = st.session_state.get(key, "").strip()
    if val and val not in st.session_state.competitors:
        st.session_state.competitors.append(val)
    st.session_state.comp_input_key += 1   # resets the input widget

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="margin-bottom:1.5rem">
      <div style="font-size:13px;font-weight:700;letter-spacing:-0.02em;color:#18181B">
        SOV Scanner
      </div>
      <div style="font-size:12px;color:#71717A;margin-top:2px">Configuration</div>
    </div>
    """, unsafe_allow_html=True)

    # API key
    api_key = st.secrets.get("OPENROUTER_API_KEY", "")
    if api_key:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:8px;background:#CCFBF1;
             border:1px solid #99F6E4;border-radius:8px;padding:10px 12px;margin-bottom:1rem">
          <div style="width:6px;height:6px;border-radius:50%;background:#0D9488;flex-shrink:0"></div>
          <span style="font-size:12px;font-weight:500;color:#0F766E">API key active</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<span class="sov-label">OpenRouter API Key</span>', unsafe_allow_html=True)
        api_key = st.text_input("API Key", type="password",
                                placeholder="sk-or-v1-...", label_visibility="collapsed",
                                help="Free key at openrouter.ai")

    st.markdown('<hr style="margin:1.25rem 0">', unsafe_allow_html=True)

    st.markdown('<span class="sov-label">Target</span>', unsafe_allow_html=True)
    target_name    = st.text_input("Name", value="Roland Berger", label_visibility="collapsed",
                                   placeholder="Your brand or entity name")
    target_aliases = st.text_input("Aliases", value="Roland Berger Strategy Consultants",
                                   label_visibility="collapsed",
                                   placeholder="Aliases separated by |",
                                   help="Alternative names the detector should match (separate with |)")

    st.markdown('<hr style="margin:1.25rem 0">', unsafe_allow_html=True)

    # ── Competitors tag list ─────────────────────────────────────────────────
    st.markdown('<span class="sov-label">Competitors</span>', unsafe_allow_html=True)

    for i, comp in enumerate(st.session_state.competitors):
        c_name, c_btn = st.columns([5, 1])
        with c_name:
            st.markdown(f'<div class="comp-tag">{comp}</div>', unsafe_allow_html=True)
        with c_btn:
            st.button("×", key=f"rm_{i}", on_click=_remove_comp, args=(i,))

    c_in, c_add = st.columns([5, 1])
    with c_in:
        st.text_input(
            "new_comp",
            key=f"nc_{st.session_state.comp_input_key}",
            label_visibility="collapsed",
            placeholder="Add competitor…",
        )
    with c_add:
        st.button("+", key="add_comp", on_click=_add_comp, type="primary")

    st.markdown('<hr style="margin:1.25rem 0">', unsafe_allow_html=True)

    st.markdown('<span class="sov-label">Scan Settings</span>', unsafe_allow_html=True)
    topic = st.text_input("Topic", value="strategy consulting germany",
                          label_visibility="collapsed",
                          placeholder="e.g. restructuring, best EV brand, cloud ERP")

    num_prompts = st.slider("Prompts per model", min_value=2, max_value=50, value=10,
                            help="More prompts = more reliable scores. 10 for quick tests; 20+ for production.")

    st.markdown("""
    <div style="margin-top:2rem;padding-top:1.25rem;border-top:1px solid #E4E4E7">
      <div style="font-size:11px;color:#71717A;line-height:1.6">
        Queries are sent via <a href="https://openrouter.ai" target="_blank"
        style="color:#0D9488;text-decoration:none">OpenRouter</a>.
        Reports are stored locally.
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Main area ─────────────────────────────────────────────────────────────────

st.markdown("""
<div class="sov-header">
  <div class="sov-badge">AI Visibility Intelligence</div>
  <h1 class="sov-title">Share of Voice Scanner</h1>
  <p class="sov-subtitle">
    Track how AI models mention your brand across any topic.
    Benchmark against competitors and identify where you're missing.
  </p>
</div>
""", unsafe_allow_html=True)

# Prompt focus expander
with st.expander("Prompt Focus — narrow the scope of generated queries", expanded=False):
    st.markdown("""
    <div style="padding:4px 0 12px;font-size:13px;color:#71717A;line-height:1.65">
      Describe the angle you want to focus on. The generator will produce more targeted queries.
      Leave both fields blank for broad, varied coverage.
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div style="font-size:12px;font-weight:500;color:#71717A;margin-bottom:4px">Focus brief</div>', unsafe_allow_html=True)
        focus_brief = st.text_input("focus_brief", label_visibility="collapsed",
                                    placeholder="e.g. DACH region, CFO persona, mid-market industrials",
                                    value=st.session_state.focus_brief)
    with col_b:
        st.markdown('<div style="font-size:12px;font-weight:500;color:#71717A;margin-bottom:4px">Your own prompts (one per line)</div>', unsafe_allow_html=True)
        custom_prompts_raw = st.text_area("custom_prompts", label_visibility="collapsed",
                                          placeholder="Which firms do CFOs in Germany trust for post-merger restructuring?\nTop restructuring advisors for mid-size industrials in DACH?",
                                          height=96,
                                          value=st.session_state.custom_prompts_raw,
                                          help="Included first and used as style examples for the generator.")

st.write("")

# Run button
can_run = bool(api_key and topic.strip() and target_name.strip())
run_clicked = st.button("Run Scan", type="primary", disabled=not can_run,
                        use_container_width=True)

if not api_key:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding:12px 16px;
         background:#FFF7ED;border:1px solid #FED7AA;border-radius:10px;margin-top:12px">
      <div style="font-size:13px;color:#92400E">
        Add your OpenRouter API key in the sidebar to get started.
        Get a free key at <a href="https://openrouter.ai" target="_blank"
        style="color:#B45309;font-weight:500">openrouter.ai</a>.
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Scan execution ────────────────────────────────────────────────────────────

if run_clicked:
    alias_list     = [a.strip() for a in target_aliases.split("|") if a.strip()]
    competitor_list = list(st.session_state.competitors)
    custom_prompts  = [p.strip() for p in custom_prompts_raw.splitlines() if p.strip()]

    st.session_state.focus_brief        = focus_brief
    st.session_state.custom_prompts_raw = custom_prompts_raw
    st.session_state.report_html        = None

    companies = [CompanyEntry(name=target_name, aliases=alias_list, is_target=True)] + [
        CompanyEntry(name=c, aliases=[], is_target=False) for c in competitor_list
    ]
    company_refs = [CompanyRef(name=c.name, aliases=c.aliases, is_target=c.is_target) for c in companies]

    total_requests = num_prompts * len(MODELS)

    init_db(DB_PATH)

    # ── Progress bar ─────────────────────────────────────────────────────────
    prog_label = st.empty()
    prog_bar   = st.progress(0)

    def _prog(pct: float, label: str):
        prog_label.markdown(f'<div class="progress-label">{label}</div>', unsafe_allow_html=True)
        prog_bar.progress(pct)

    _prog(0.02, "Initializing…")

    with st.status("Running scan…", expanded=True) as status:

        # 1 — Prompts
        n_to_generate = max(0, num_prompts - len(custom_prompts))
        if n_to_generate > 0:
            st.write(f"Generating {n_to_generate} prompts for **{topic}**…")
            _prog(0.08, "Generating prompts…")
            generated = _run_async(auto_generate_prompts(
                topic=topic, count=n_to_generate, api_key=api_key,
                brief=focus_brief, examples=custom_prompts or None,
            ))
        else:
            generated = []

        prompt_list = list(dict.fromkeys(custom_prompts + generated))
        src_note = (f" — {len(custom_prompts)} custom + {len(generated)} generated"
                    if custom_prompts and generated else "")
        st.write(f"**{len(prompt_list)} prompts ready{src_note}**")
        _prog(0.20, f"{len(prompt_list)} prompts ready — querying {len(MODELS)} models…")

        # 2 — Query models
        real_total = len(prompt_list) * len(MODELS)
        st.write(f"Querying {len(MODELS)} models ({real_total} total requests)…")
        results = _run_async(run_queries(prompt_list, MODELS, api_key, max_concurrent=5))
        errors  = sum(1 for r in results if r.error)
        st.write(f"**{len(results) - errors} responses received**" +
                 (f" — {errors} failed" if errors else ""))
        _prog(0.75, f"{len(results) - errors}/{real_total} responses received — detecting mentions…")

        # 3 — Detect mentions
        st.write("Detecting mentions…")
        run_id = insert_run(DB_PATH, topic=topic, period=_auto_period())
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

        # 4 — Report
        st.write("Generating report…")
        queries    = get_queries_for_run(DB_PATH, run_id)
        mentions   = get_mentions_for_run(DB_PATH, run_id)
        run_record = get_run(DB_PATH, run_id)

        report_path = generate_report(
            run=run_record, queries=queries, mentions=mentions, companies=companies,
            template_path=TEMPLATE_PATH, output_dir=REPORTS_DIR,
        )
        st.session_state.report_html = pathlib.Path(report_path).read_text(encoding="utf-8")
        st.session_state.report_path = report_path

        _prog(1.0, "Scan complete")
        status.update(label="Scan complete", state="complete")

# ── Report / empty state ──────────────────────────────────────────────────────

if st.session_state.report_html:
    report_name = pathlib.Path(st.session_state.report_path).name

    st.markdown(f"""
    <div class="sov-report-header">
      <div>
        <div class="sov-report-title">Report</div>
        <div class="sov-report-meta">{report_name}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.download_button(
        label="Download HTML report",
        data=st.session_state.report_html,
        file_name=report_name,
        mime="text/html",
    )

    st.write("")
    components.html(st.session_state.report_html, height=960, scrolling=True)

else:
    st.markdown("""
    <div class="sov-empty">
      <div class="sov-empty-icon">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none"
             stroke="#0D9488" stroke-width="1.75" stroke-linecap="round">
          <circle cx="11" cy="11" r="8"/>
          <line x1="21" y1="21" x2="16.65" y2="16.65"/>
        </svg>
      </div>
      <div class="sov-empty-title">No scan run yet</div>
      <div class="sov-empty-desc">
        Configure your target and competitors in the sidebar,
        then click <strong>Run Scan</strong> to generate your first report.
      </div>
    </div>
    """, unsafe_allow_html=True)
