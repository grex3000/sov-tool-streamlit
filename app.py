"""AI Share of Voice — Streamlit interface."""
from __future__ import annotations

import asyncio
import pathlib
import sys
import threading

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

# ── Async helper (Streamlit-safe) ─────────────────────────────────────────────

def _run_async(coro):
    """Run a coroutine in a fresh event loop inside a daemon thread."""
    result = [None]
    exc = [None]

    def _target():
        try:
            result[0] = asyncio.run(coro)
        except Exception as e:
            exc[0] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join()
    if exc[0]:
        raise exc[0]
    return result[0]


# ── Models ────────────────────────────────────────────────────────────────────

MODELS = [
    ("openai/gpt-5.4",               "GPT-5.4"),
    ("anthropic/claude-sonnet-4.6",  "Claude Sonnet 4.6"),
    ("google/gemini-3.1-pro-preview","Gemini 3.1 Pro"),
    ("perplexity/sonar-pro",         "Perplexity Sonar Pro"),
]

DB_PATH       = "data/sov.db"
TEMPLATE_PATH = "templates/report.html.j2"
REPORTS_DIR   = "reports"

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Share of Voice",
    page_icon="🎯",
    layout="wide",
)

# ── Session state ─────────────────────────────────────────────────────────────

if "report_html" not in st.session_state:
    st.session_state.report_html = None
if "report_path" not in st.session_state:
    st.session_state.report_path = None

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Configuration")

    # API key — from Streamlit secrets or manual input
    api_key = st.secrets.get("OPENROUTER_API_KEY", "")
    if api_key:
        st.success("API key loaded", icon="🔑")
    else:
        api_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            placeholder="sk-or-v1-...",
            help="Get a free key at openrouter.ai",
        )

    st.divider()

    st.subheader("🎯 Target")
    target_name = st.text_input("Name", value="Roland Berger")
    target_aliases = st.text_input(
        "Aliases",
        value="Roland Berger Strategy Consultants",
        help="Separate multiple aliases with |",
    )

    st.subheader("🏁 Competitors")
    competitors_raw = st.text_area(
        "One per line",
        value="McKinsey & Company\nBoston Consulting Group\nBain & Company",
        height=110,
    )

    st.divider()

    st.subheader("📊 Scan Settings")
    topic = st.text_input(
        "Topic",
        value="strategy consulting germany",
        placeholder="e.g. restructuring, best EV brand, cloud ERP",
    )
    num_prompts = st.slider(
        "Number of prompts", min_value=2, max_value=50, value=10,
        help="More prompts = more reliable scores. 10 is a quick test; 20+ for production.",
    )
    period = st.text_input(
        "Period label", value="Q2 2026",
        help="Appears in the report header — use consistently for trend tracking.",
    )

# ── Main area ─────────────────────────────────────────────────────────────────

st.title("🎯 AI Share of Voice")
st.caption(
    "Measure how often your target is mentioned by AI models — "
    "benchmark against competitors on any topic."
)

st.divider()

# Prompt focus expander
with st.expander("🔍 Prompt Focus *(optional — recommended for first runs)*", expanded=False):
    st.markdown(
        "Describe the angle you want to focus on. The generator will use this to produce "
        "more targeted queries. Leave blank for broad, varied coverage."
    )
    col_a, col_b = st.columns(2)
    with col_a:
        focus_brief = st.text_input(
            "Focus brief",
            placeholder="e.g. DACH region, CFO persona, mid-market industrial companies",
        )
    with col_b:
        custom_prompts_raw = st.text_area(
            "Your own prompts (one per line)",
            placeholder=(
                "Which firms do CFOs in Germany trust for post-merger restructuring?\n"
                "Top restructuring advisors for mid-size industrials in DACH?"
            ),
            height=100,
            help="Included first and used as style examples for the generator.",
        )

st.write("")

# Run button
run_clicked = st.button(
    "🚀 Run Scan",
    type="primary",
    disabled=not api_key or not topic.strip() or not target_name.strip(),
    use_container_width=True,
)

if not api_key:
    st.info("Add your OpenRouter API key in the sidebar to get started.", icon="🔑")

# ── Scan execution ────────────────────────────────────────────────────────────

if run_clicked:
    alias_list      = [a.strip() for a in target_aliases.split("|") if a.strip()]
    competitor_list = [c.strip() for c in competitors_raw.splitlines() if c.strip()]
    custom_prompts  = [p.strip() for p in custom_prompts_raw.splitlines() if p.strip()]

    companies = [CompanyEntry(name=target_name, aliases=alias_list, is_target=True)] + [
        CompanyEntry(name=c, aliases=[], is_target=False) for c in competitor_list
    ]
    company_refs = [
        CompanyRef(name=c.name, aliases=c.aliases, is_target=c.is_target)
        for c in companies
    ]

    init_db(DB_PATH)

    with st.status("Running scan…", expanded=True) as status:

        # 1 — Prompts
        n_to_generate = max(0, num_prompts - len(custom_prompts))
        if n_to_generate > 0:
            st.write(f"Generating {n_to_generate} prompts for **{topic}**…")
            generated = _run_async(
                auto_generate_prompts(
                    topic=topic,
                    count=n_to_generate,
                    api_key=api_key,
                    brief=focus_brief,
                    examples=custom_prompts or None,
                )
            )
        else:
            generated = []

        prompt_list = list(dict.fromkeys(custom_prompts + generated))
        src_note = (
            f" ({len(custom_prompts)} custom + {len(generated)} generated)"
            if custom_prompts and generated else ""
        )
        st.write(f"✓ {len(prompt_list)} prompts ready{src_note}")

        # 2 — Query models
        st.write(f"Querying {len(MODELS)} models…")
        results = _run_async(
            run_queries(prompt_list, MODELS, api_key, max_concurrent=5)
        )
        errors = sum(1 for r in results if r.error)
        st.write(f"✓ {len(results) - errors} responses received" +
                 (f" ({errors} errors)" if errors else ""))

        # 3 — Persist + detect
        st.write("Detecting mentions…")
        run_id = insert_run(DB_PATH, topic=topic, period=period)
        for r in results:
            qid = insert_query(
                DB_PATH, run_id=run_id,
                model_id=r.model_id, model_label=r.model_label,
                prompt=r.prompt, response=r.response,
            )
            if r.response:
                for hit in detect_all_mentions(r.response, company_refs):
                    insert_mention(
                        DB_PATH, query_id=qid,
                        company_name=hit["company_name"], is_target=hit["is_target"],
                        match_type=hit["type"], excerpt=hit["excerpt"],
                    )

        # 4 — Report
        st.write("Generating report…")
        queries    = get_queries_for_run(DB_PATH, run_id)
        mentions   = get_mentions_for_run(DB_PATH, run_id)
        run_record = get_run(DB_PATH, run_id)

        report_path = generate_report(
            run=run_record,
            queries=queries,
            mentions=mentions,
            companies=companies,
            template_path=TEMPLATE_PATH,
            output_dir=REPORTS_DIR,
        )

        st.session_state.report_html = pathlib.Path(report_path).read_text(encoding="utf-8")
        st.session_state.report_path = report_path

        status.update(label="✅ Scan complete!", state="complete")

# ── Report display ────────────────────────────────────────────────────────────

if st.session_state.report_html:
    st.divider()

    col1, col2 = st.columns([6, 1])
    with col1:
        st.subheader("📊 Report")
    with col2:
        st.download_button(
            label="⬇️ Download HTML",
            data=st.session_state.report_html,
            file_name=pathlib.Path(st.session_state.report_path).name,
            mime="text/html",
            use_container_width=True,
        )

    components.html(st.session_state.report_html, height=950, scrolling=True)
