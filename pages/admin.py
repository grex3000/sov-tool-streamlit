"""SOV Admin — password-protected dashboard showing all historical scans."""
from __future__ import annotations

import os
import pathlib
import sys

import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.config import CompanyEntry
from src.db import (
    get_admin_run_summary,
    get_companies_for_run,
    get_mentions_for_run,
    get_queries_for_run,
    init_db,
)
from src.report import compute_gap_analysis, compute_sov

DB_PATH = "data/sov.db"

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="SOV Admin", page_icon="📊", layout="wide")

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@font-face {
  font-family: 'RBDesign';
  src: url('https://www.rolandberger.com/fonts/RBDesign/RBDesign-Light.otf') format('opentype');
  font-weight: 300; font-style: normal; font-display: swap;
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
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --accent:        #00aac9;
  --accent-light:  #e3f6fc;
  --bg:            #eff0f1;
  --surface:       #ffffff;
  --text:          #000000;
  --muted:         #8d9399;
  --border:        #dee0e3;
  --sidebar-bg:    #000000;
  --sidebar-text:  #ffffff;
  --sidebar-muted: #a0a0a0;
  --sidebar-border:#2a2a2a;
  --score-high:    #00aac9;
  --score-mid:     #f59e0b;
  --score-low:     #ef4444;
}

html, body, .stApp, [class*="block-container"] { background-color: var(--bg) !important; }
* { font-family: 'RBDesign', Arial, sans-serif !important; }

.material-symbols-rounded, .material-symbols-outlined,
.material-symbols-sharp, .material-icons, .material-icons-round,
[data-testid="stIconMaterial"] {
  font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
}

#MainMenu, footer, .stDeployButton,
[data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
[data-testid="stSidebarNav"] { display: none !important; }
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
button[data-testid="baseButton-header"] { display: none !important; }

/* Sidebar */
[data-testid="stSidebar"] { background: var(--sidebar-bg) !important; border-right: none !important; }
[data-testid="stSidebar"] * { color: var(--sidebar-text) !important; }

/* Inputs */
[data-testid="stTextInput"] input {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 0 !important;
  color: var(--text) !important;
  -webkit-text-fill-color: var(--text) !important;
  font-size: 15px !important;
}
[data-testid="stTextInput"] input:focus {
  border-color: var(--text) !important;
  box-shadow: none !important; outline: none !important;
}
[data-testid="stTextInput"] input::placeholder {
  color: var(--muted) !important;
  -webkit-text-fill-color: var(--muted) !important;
}

/* Primary button — black RB style */
button[data-testid="baseButton-primary"] {
  background: #000000 !important;
  background-color: #000000 !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: 0 !important;
  font-size: 12px !important;
  font-weight: 700 !important;
  letter-spacing: 0.1em !important;
  text-transform: uppercase !important;
  height: 48px !important;
  padding: 0 24px !important;
  box-shadow: none !important;
}
button[data-testid="baseButton-primary"] * { color: #ffffff !important; }

/* Metrics */
[data-testid="stMetric"] { background: var(--surface); padding: 16px 20px; border: 1px solid var(--border); }
[data-testid="stMetricLabel"] { font-size: 10px !important; font-weight: 600 !important;
  letter-spacing: 0.12em !important; text-transform: uppercase !important; color: var(--muted) !important; }
[data-testid="stMetricValue"] { font-size: 28px !important; font-weight: 300 !important; color: var(--text) !important; }

/* Expander */
[data-testid="stExpander"] { border: 1px solid var(--border) !important;
  border-radius: 0 !important; background: var(--surface) !important; margin-bottom: 4px !important; }
[data-testid="stExpander"] summary { font-weight: 400 !important; font-size: 14px !important; }

/* Admin table */
.admin-table { width:100%; border-collapse:collapse; font-size:13px; font-weight:300; }
.admin-table th {
  font-size:10px; font-weight:600; letter-spacing:.12em; text-transform:uppercase;
  color:var(--muted); padding:6px 10px; border-bottom:2px solid var(--border);
  text-align:left; background:var(--surface);
}
.admin-table td { padding:8px 10px; border-bottom:1px solid var(--border); vertical-align:top; }
.admin-table tr:last-child td { border-bottom:none; }
.admin-table tr.target-row td { font-weight:600; }
.sov-pct { font-family:'IBM Plex Mono',monospace !important; font-size:13px; }
.sov-high { color:var(--score-high); }
.sov-mid  { color:var(--score-mid); }
.sov-low  { color:var(--score-low); }
.admin-tag {
  display:inline-block; font-size:11px; font-weight:500;
  padding:2px 8px; margin:2px 3px 2px 0;
  background:var(--accent-light); color:var(--accent);
}
.admin-tag.target { background:#000; color:#fff; }
.admin-section-label {
  font-size:10px; font-weight:600; letter-spacing:.12em; text-transform:uppercase;
  color:var(--muted); margin:20px 0 8px; display:block;
}
</style>
""", unsafe_allow_html=True)

# ── Password gate ─────────────────────────────────────────────────────────────

_ss = st.session_state
if "admin_authed" not in _ss:
    _ss.admin_authed = False

if not _ss.admin_authed:
    st.markdown("""
    <div style="max-width:400px;margin:80px auto 0;">
      <img src="https://www.rolandberger.com/img/assets/RoBe_Logotype_White_Digital.png"
           style="height:28px;filter:brightness(0);margin-bottom:32px;display:block;">
      <div style="font-size:22px;font-weight:300;letter-spacing:-.02em;margin-bottom:24px;">
        Admin Access
      </div>
    </div>
    """, unsafe_allow_html=True)

    col, _ = st.columns([1, 2])
    with col:
        pwd = st.text_input("Password", type="password",
                            placeholder="Enter admin password…",
                            label_visibility="collapsed")
        if st.button("Enter", type="primary", use_container_width=True):
            expected = st.secrets.get("ADMIN_PASSWORD", os.environ.get("ADMIN_PASSWORD", ""))
            if expected and pwd == expected:
                _ss.admin_authed = True
                st.rerun()
            elif not expected:
                st.error("ADMIN_PASSWORD not configured in secrets.toml or environment.")
            else:
                st.error("Incorrect password.")
    st.stop()

# ── Init DB & load data ───────────────────────────────────────────────────────

init_db(DB_PATH)
runs = get_admin_run_summary(DB_PATH)

# Enrich each run with company info (target + competitors)
for r in runs:
    companies = get_companies_for_run(DB_PATH, r["id"])
    r["_target"] = next((c["company_name"] for c in companies if c["is_target"]), "—")
    r["_competitors"] = [c["company_name"] for c in companies if not c["is_target"]]

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="display:flex;align-items:center;gap:20px;padding:0 0 24px;border-bottom:1px solid #dee0e3;margin-bottom:24px;">
  <img src="https://www.rolandberger.com/img/assets/RoBe_Logotype_White_Digital.png"
       style="height:28px;filter:brightness(0);">
  <div>
    <div style="font-size:9px;font-weight:600;letter-spacing:.16em;text-transform:uppercase;color:#8d9399;">
      AI Share of Voice
    </div>
    <div style="font-size:20px;font-weight:300;letter-spacing:-.02em;line-height:1.1;">
      Admin Dashboard
    </div>
  </div>
  <div style="margin-left:auto;">
    <a href="/" style="font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;
                       color:#8d9399;text-decoration:none;">← Back to App</a>
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPI metrics ───────────────────────────────────────────────────────────────

unique_topics  = len({r["topic"] for r in runs})
unique_targets = len({r["_target"] for r in runs if r["_target"] != "—"})
total_qs       = sum(r["query_count"] or 0 for r in runs)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Scans",     len(runs))
c2.metric("Unique Topics",   unique_topics)
c3.metric("Unique Targets",  unique_targets)
c4.metric("Total Questions", total_qs)

st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)

# ── Runs list ─────────────────────────────────────────────────────────────────

if not runs:
    st.info("No scans recorded yet. Run a scan from the main app to see data here.")
    st.stop()

st.markdown('<span class="admin-section-label">Scan History</span>', unsafe_allow_html=True)


def _sov_class(pct: float | None) -> str:
    if pct is None:
        return ""
    if pct >= 40:
        return "sov-high"
    if pct >= 20:
        return "sov-mid"
    return "sov-low"


def _fmt_sov(pct: float | None) -> str:
    if pct is None:
        return "—"
    return f"{pct:.0f}%"


for r in runs:
    date_str      = r["created_at"][:10]
    target        = r["_target"]
    competitors   = r["_competitors"]
    sov_pct       = r.get("target_sov_pct")
    sov_label     = _fmt_sov(sov_pct)
    sov_cls       = _sov_class(sov_pct)
    comp_str      = ", ".join(competitors) if competitors else "—"
    q_count       = r["query_count"] or 0
    m_count       = r["model_count"] or 0

    expander_title = f"{date_str}  ·  {r['topic']}  ·  {target}  ·  SOV {sov_label}"

    with st.expander(expander_title):
        # ── Summary row ──────────────────────────────────────────────────────
        meta_cols = st.columns([1.5, 2.5, 2, 2.5, 0.8, 0.8])
        with meta_cols[0]:
            st.markdown(f'<span class="admin-section-label">Date</span>{date_str}', unsafe_allow_html=True)
        with meta_cols[1]:
            st.markdown(f'<span class="admin-section-label">Topic</span>{r["topic"]}', unsafe_allow_html=True)
        with meta_cols[2]:
            st.markdown(
                f'<span class="admin-section-label">Target</span>'
                f'<span class="admin-tag target">{target}</span>',
                unsafe_allow_html=True,
            )
        with meta_cols[3]:
            tags = "".join(f'<span class="admin-tag">{c}</span>' for c in competitors) or "—"
            st.markdown(f'<span class="admin-section-label">Competitors</span>{tags}', unsafe_allow_html=True)
        with meta_cols[4]:
            st.markdown(f'<span class="admin-section-label">Questions</span>{q_count}', unsafe_allow_html=True)
        with meta_cols[5]:
            st.markdown(f'<span class="admin-section-label">Models</span>{m_count}', unsafe_allow_html=True)

        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

        # ── Load detail data ──────────────────────────────────────────────────
        queries       = get_queries_for_run(DB_PATH, r["id"])
        mentions      = get_mentions_for_run(DB_PATH, r["id"])
        company_rows  = get_companies_for_run(DB_PATH, r["id"])
        company_objs  = [
            CompanyEntry(name=c["company_name"], aliases=[], is_target=bool(c["is_target"]))
            for c in company_rows
        ]

        if not queries or not company_objs:
            st.info("No query data recorded for this run.")
            continue

        sov = compute_sov(queries, mentions, company_objs)
        gap = compute_gap_analysis(queries, mentions, company_objs)

        # ── SOV table ─────────────────────────────────────────────────────────
        st.markdown('<span class="admin-section-label">Share of Voice by Model</span>', unsafe_allow_html=True)

        models = sov["models"]
        header_cells = "".join(f"<th>{m['label']}</th>" for m in models)
        rows_html = ""
        for company in sov["companies"]:
            tgt_cls = "target-row" if company["is_target"] else ""
            model_cells = ""
            for m in models:
                bm = company["by_model"].get(m["id"], {})
                pct = bm.get("sov_pct")
                cls = _sov_class(pct)
                model_cells += f'<td><span class="sov-pct {cls}">{_fmt_sov(pct)}</span></td>'
            avg_cls = _sov_class(company["avg_sov"])
            rows_html += (
                f'<tr class="{tgt_cls}">'
                f'<td>{company["name"]}</td>'
                f'<td><span class="sov-pct {avg_cls}">{company["avg_sov"]}%</span></td>'
                f'{model_cells}'
                f'</tr>'
            )

        st.markdown(f"""
        <table class="admin-table">
          <thead><tr>
            <th>Company</th><th>Avg SOV</th>{header_cells}
          </tr></thead>
          <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)

        # ── GAP analysis ──────────────────────────────────────────────────────
        if gap:
            st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
            st.markdown('<span class="admin-section-label">GAP Analysis</span>', unsafe_allow_html=True)

            g1, g2, g3 = st.columns(3)
            g1.metric("Gap Rate",        f"{gap['gap_pct']}%")
            g2.metric("Gap Prompts",     gap["gap_prompt_count"])
            g3.metric("Consensus Gaps",  len(gap.get("consensus_gaps", [])))

            if gap["by_competitor"]:
                comp_rows = "".join(
                    f"<tr><td>{c['name']}</td>"
                    f"<td><span class='sov-pct sov-low'>{c['gap_pct']}%</span></td>"
                    f"<td>{c['gap_count']}</td></tr>"
                    for c in gap["by_competitor"]
                )
                st.markdown(f"""
                <table class="admin-table" style="max-width:480px;margin-top:10px;">
                  <thead><tr><th>Competitor</th><th>Gap %</th><th>Prompts</th></tr></thead>
                  <tbody>{comp_rows}</tbody>
                </table>
                """, unsafe_allow_html=True)

        # ── Questions asked ───────────────────────────────────────────────────
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        st.markdown('<span class="admin-section-label">Questions Asked</span>', unsafe_allow_html=True)

        unique_prompts = list(dict.fromkeys(q["prompt"] for q in queries))
        prompt_rows = "".join(
            f'<tr><td style="color:var(--muted);font-size:12px;width:32px;">{i+1}.</td>'
            f'<td>{p}</td></tr>'
            for i, p in enumerate(unique_prompts)
        )
        st.markdown(f"""
        <table class="admin-table">
          <tbody>{prompt_rows}</tbody>
        </table>
        """, unsafe_allow_html=True)
