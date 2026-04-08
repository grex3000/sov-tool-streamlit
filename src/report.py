from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, FileSystemLoader
from markupsafe import Markup, escape

if TYPE_CHECKING:
    from src.config import CompanyEntry


# ─── Helpers ──────────────────────────────────────────────────────────────────

def score_class(pct: float) -> str:
    if pct >= 40:
        return "score-high"
    if pct >= 20:
        return "score-mid"
    return "score-low"


def highlight_excerpt(excerpt: str, companies: list[CompanyEntry]) -> Markup:
    """
    HTML-escape an excerpt, then wrap every company/alias occurrence in a
    coloured <span>. Uses a single-pass combined regex so longer names are
    matched first and aliases can't double-wrap an already-replaced span.
    Returns a Markup object safe to use with | safe in Jinja2.
    """
    escaped = str(escape(excerpt))

    # Build flat list of (name, css_class), longest first
    names: list[tuple[str, str]] = []
    for c in companies:
        css = "mention-target" if c.is_target else "mention-competitor"
        names.append((c.name, css))
        for alias in c.aliases:
            names.append((alias, css))
    names.sort(key=lambda t: len(t[0]), reverse=True)

    if not names:
        return Markup(escaped)

    # Single combined regex — alternation tries longest match first
    combined = re.compile(
        "|".join(re.escape(n) for n, _ in names),
        re.IGNORECASE,
    )
    name_to_css = {n.lower(): css for n, css in names}

    def _replace(m: re.Match) -> str:
        css = name_to_css.get(m.group().lower(), "mention-target")
        return f'<span class="{css}">{escape(m.group())}</span>'

    return Markup(combined.sub(_replace, escaped))


# ─── SOV computation ──────────────────────────────────────────────────────────

def compute_sov(
    queries: list[dict],
    mentions: list[dict],
    companies: list[CompanyEntry],
) -> dict:
    """
    Returns a dict with everything the report template needs:

    {
      "companies": [          # sorted by avg_sov desc
        {
          "name", "is_target", "avg_sov", "rank", "score_class",
          "by_model": {model_id: {"label","sov_pct","score_class","mentioned","total"}}
        }, ...
      ],
      "models": [{"id","label"}, ...],
      "target": {...},         # shortcut to the target company entry
      "by_model_spotlight": { model_id: {"label", "companies": [...ranked...]} },
      "citations": [...],      # up to 5 target-company mention dicts (pre-highlighted)
    }
    """
    # query_id → set of company names mentioned
    mentioned_by_qid: dict[int, set[str]] = defaultdict(set)
    for m in mentions:
        mentioned_by_qid[m["query_id"]].add(m["company_name"])

    # Ordered models from queries (preserve insertion order)
    models_seen: dict[str, str] = {}  # id → label
    for q in queries:
        models_seen.setdefault(q["model_id"], q["model_label"])

    # ── Per company ────────────────────────────────────────────────────────
    results: list[dict] = []
    for company in companies:
        by_model: dict[str, dict] = {}
        model_pcts: list[float] = []

        for model_id, model_label in models_seen.items():
            model_qs = [q for q in queries if q["model_id"] == model_id]
            total = len(model_qs)
            mentioned = sum(
                1 for q in model_qs
                if company.name in mentioned_by_qid.get(q["id"], set())
            )
            pct = round(mentioned / total * 100, 1) if total else 0.0
            model_pcts.append(pct)
            by_model[model_id] = {
                "label": model_label,
                "sov_pct": pct,
                "score_class": score_class(pct),
                "mentioned": mentioned,
                "total": total,
            }

        avg = round(sum(model_pcts) / len(model_pcts), 1) if model_pcts else 0.0
        results.append(
            {
                "name": company.name,
                "is_target": company.is_target,
                "avg_sov": avg,
                "score_class": score_class(avg),
                "by_model": by_model,
            }
        )

    # Rank by avg_sov descending
    results.sort(key=lambda x: x["avg_sov"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    target = next((r for r in results if r["is_target"]), results[0] if results else {})

    # ── Model spotlight: per model, all companies ranked ───────────────────
    by_model_spotlight: dict[str, dict] = {}
    for model_id, model_label in models_seen.items():
        spotlight_companies = sorted(
            [
                {
                    "name": r["name"],
                    "is_target": r["is_target"],
                    "sov_pct": r["by_model"][model_id]["sov_pct"],
                    "score_class": r["by_model"][model_id]["score_class"],
                    "mentioned": r["by_model"][model_id]["mentioned"],
                    "total": r["by_model"][model_id]["total"],
                }
                for r in results
                if model_id in r["by_model"]
            ],
            key=lambda x: x["sov_pct"],
            reverse=True,
        )
        by_model_spotlight[model_id] = {
            "label": model_label,
            "companies": spotlight_companies,
        }

    # ── Citations: target mentions, pre-highlighted, up to 5 ──────────────
    raw_citations = [m for m in mentions if m.get("is_target")][:5]
    citations = []
    for c in raw_citations:
        if c.get("excerpt"):
            c = dict(c)
            c["excerpt_html"] = highlight_excerpt(c["excerpt"], companies)
        citations.append(c)

    return {
        "companies": results,
        "models": [{"id": mid, "label": lbl} for mid, lbl in models_seen.items()],
        "target": target,
        "by_model_spotlight": by_model_spotlight,
        "citations": citations,
    }


# ─── GAP Analysis ─────────────────────────────────────────────────────────────

def _generate_recommendations(
    target_name: str,
    gap_pct: float,
    by_competitor: list[dict],
    by_model: list[dict],
    consensus_gaps: list[dict],
    total_prompts: int,
) -> list[str]:
    if gap_pct == 0:
        return ["No visibility gaps detected — your target appears wherever competitors do."]

    recs = []

    if by_competitor:
        top = by_competitor[0]
        recs.append(
            f"Your largest visibility gap is against <strong>{top['name']}</strong>, which appears "
            f"without you in {top['gap_count']} of {total_prompts} queries ({top['gap_pct']}%). "
            f"Audit how {top['name']} is described online and identify content themes you are missing."
        )

    if by_model:
        top_model = by_model[0]
        recs.append(
            f"<strong>{top_model['label']}</strong> is your weakest channel — competitors appear "
            f"without you in {top_model['gap_count']} of {top_model['total']} queries "
            f"({top_model['gap_pct']}%). Different models draw from different training data; "
            f"improving your indexed presence on sources that feed this model will help most."
        )

    if consensus_gaps:
        recs.append(
            f"<strong>{len(consensus_gaps)} {'query' if len(consensus_gaps) == 1 else 'queries'}</strong> "
            f"triggered competitor mentions across <em>all</em> models without mentioning you — "
            f"every model has learned to recommend others but not you for these phrasings. "
            f"Review them below and create targeted content to close these gaps first."
        )

    if gap_pct >= 50:
        recs.append(
            f"An overall gap rate of {gap_pct}% is high. Publish structured, citable content "
            f"(case studies, thought leadership, press mentions) that clearly associates "
            f"<strong>{target_name}</strong> with the topic — this is the primary lever for AI visibility."
        )
    elif gap_pct >= 25:
        recs.append(
            f"A gap rate of {gap_pct}% indicates meaningful room for improvement. "
            f"Focus on increasing high-quality, citable web presence on the topic "
            f"to build training signal for future model updates."
        )

    return recs


def compute_gap_analysis(
    queries: list[dict],
    mentions: list[dict],
    companies: list,
) -> dict | None:
    """
    Compute GAP analysis: prompts where competitors were mentioned but the target was not.

    Returns:
    {
      "total_prompts": int,
      "gap_prompt_count": int,
      "gap_pct": float,
      "by_competitor": [{"name", "gap_count", "gap_pct"}, ...],
      "by_model": [{"id", "label", "gap_count", "total", "gap_pct"}, ...],
      "consensus_gaps": [{"prompt", "competitors_mentioned"}, ...],  # max 5
      "recommendations": [str, ...],
    }
    Returns None if there is no target company.
    """
    target = next((c for c in companies if c.is_target), None)
    if not target:
        return None

    competitor_names = {c.name for c in companies if not c.is_target}

    # query_id → set of company names mentioned
    mentioned_by_qid: dict[int, set[str]] = defaultdict(set)
    for m in mentions:
        mentioned_by_qid[m["query_id"]].add(m["company_name"])

    # Build per-prompt, per-model data
    # prompt → {model_id: {"target_mentioned": bool, "competitors_mentioned": [str]}}
    prompt_model_data: dict[str, dict[str, dict]] = defaultdict(dict)
    model_labels: dict[str, str] = {}
    for q in queries:
        model_labels[q["model_id"]] = q["model_label"]
        mentioned = mentioned_by_qid.get(q["id"], set())
        prompt_model_data[q["prompt"]][q["model_id"]] = {
            "target_mentioned": target.name in mentioned,
            "competitors_mentioned": [c for c in competitor_names if c in mentioned],
        }

    total_prompts = len(prompt_model_data)
    gap_by_competitor: dict[str, int] = defaultdict(int)
    gap_by_model: dict[str, int] = defaultdict(int)
    total_by_model: dict[str, int] = defaultdict(int)
    gap_prompts: list[dict] = []

    for prompt, model_data in prompt_model_data.items():
        prompt_gap_competitors: set[str] = set()
        any_target = False
        any_competitor = False

        for model_id, data in model_data.items():
            total_by_model[model_id] += 1
            if data["target_mentioned"]:
                any_target = True
            if data["competitors_mentioned"]:
                any_competitor = True
            # per-model gap: this model mentioned a competitor but not the target
            if not data["target_mentioned"] and data["competitors_mentioned"]:
                gap_by_model[model_id] += 1
                for c in data["competitors_mentioned"]:
                    gap_by_competitor[c] += 1
                    prompt_gap_competitors.add(c)

        if prompt_gap_competitors:
            # consensus = no model at all mentioned the target, but at least one mentioned a competitor
            is_consensus = not any_target and any_competitor
            gap_prompts.append({
                "prompt": prompt,
                "competitors_mentioned": sorted(prompt_gap_competitors),
                "is_consensus": is_consensus,
            })

    consensus_gaps = [p for p in gap_prompts if p["is_consensus"]]

    by_competitor = sorted(
        [
            {
                "name": name,
                "gap_count": cnt,
                "gap_pct": round(cnt / total_prompts * 100, 1) if total_prompts else 0.0,
            }
            for name, cnt in gap_by_competitor.items()
        ],
        key=lambda x: x["gap_count"],
        reverse=True,
    )

    by_model = sorted(
        [
            {
                "id": mid,
                "label": model_labels.get(mid, mid),
                "gap_count": cnt,
                "total": total_by_model[mid],
                "gap_pct": round(cnt / total_by_model[mid] * 100, 1) if total_by_model[mid] else 0.0,
            }
            for mid, cnt in gap_by_model.items()
        ],
        key=lambda x: x["gap_count"],
        reverse=True,
    )

    gap_prompt_count = len(gap_prompts)
    gap_pct = round(gap_prompt_count / total_prompts * 100, 1) if total_prompts else 0.0

    recommendations = _generate_recommendations(
        target.name, gap_pct, by_competitor, by_model, consensus_gaps, total_prompts
    )

    return {
        "total_prompts": total_prompts,
        "gap_prompt_count": gap_prompt_count,
        "gap_pct": gap_pct,
        "by_competitor": by_competitor,
        "by_model": by_model,
        "consensus_gaps": consensus_gaps[:5],
        "gap_prompts": gap_prompts,
        "recommendations": recommendations,
    }


# ─── Response log ─────────────────────────────────────────────────────────────

def _build_response_log(
    queries: list[dict],
    mentions: list[dict],
    companies: list[CompanyEntry],
) -> list[dict]:
    """
    Return one entry per unique prompt, each containing per-model responses
    with highlighted HTML and company mention tags.

    Shape: [
      {
        "prompt": str,
        "mentioned_companies": [{"name", "is_target"}, ...],   # union across all models
        "responses": [
          {
            "model_id", "model_label", "response", "response_html",
            "mentions": [{"company_name", "is_target", "match_type"}, ...]
          }, ...
        ]
      }, ...
    ]
    """
    mentions_by_qid: dict[int, list[dict]] = defaultdict(list)
    for m in mentions:
        mentions_by_qid[m["query_id"]].append(m)

    # Collect unique prompts preserving encounter order
    prompts_ordered: list[str] = []
    seen_prompts: set[str] = set()
    for q in queries:
        if q["prompt"] not in seen_prompts:
            seen_prompts.add(q["prompt"])
            prompts_ordered.append(q["prompt"])

    prompt_index = {p: i for i, p in enumerate(prompts_ordered)}
    entries: list[dict] = [{"prompt": p, "responses": []} for p in prompts_ordered]

    for q in queries:
        idx = prompt_index[q["prompt"]]
        q_mentions = mentions_by_qid.get(q["id"], [])
        response_html = (
            highlight_excerpt(q["response"], companies)
            if q.get("response")
            else Markup("<em style='color:var(--low-text)'>No response (API error)</em>")
        )
        entries[idx]["responses"].append(
            {
                "model_id": q["model_id"],
                "model_label": q["model_label"],
                "response": q.get("response") or "",
                "response_html": response_html,
                "mentions": q_mentions,
            }
        )

    for entry in entries:
        # Sort responses by model label for consistent column order
        entry["responses"].sort(key=lambda r: r["model_label"])

        # Collect unique companies mentioned across all model responses for this prompt
        seen: dict[str, bool] = {}
        for r in entry["responses"]:
            for m in r["mentions"]:
                seen.setdefault(m["company_name"], bool(m["is_target"]))
        entry["mentioned_companies"] = [
            {"name": name, "is_target": is_tgt} for name, is_tgt in seen.items()
        ]

    return entries


# ─── Report generation ────────────────────────────────────────────────────────

def generate_report(
    run: dict,
    queries: list[dict],
    mentions: list[dict],
    companies: list[CompanyEntry],
    template_path: str,
    output_dir: str,
) -> str:
    """Render the HTML report and save to output_dir. Returns the output file path."""
    sov = compute_sov(queries, mentions, companies)
    gap = compute_gap_analysis(queries, mentions, companies)

    target = sov["target"]
    competitor_count = sum(1 for c in sov["companies"] if not c["is_target"])

    env = Environment(
        loader=FileSystemLoader(str(Path(template_path).parent)),
        autoescape=True,
    )
    template = env.get_template(Path(template_path).name)

    response_log = _build_response_log(queries, mentions, companies)

    html = template.render(
        company_name=next((c.name for c in companies if c.is_target), "—"),
        topic=run["topic"],
        period=run.get("period") or "—",
        generated_date=datetime.now(timezone.utc).strftime("%B %d, %Y"),
        run=run,
        # KPIs
        target_avg_sov=target.get("avg_sov", 0),
        target_rank=target.get("rank", "—"),
        target_score_class=target.get("score_class", "score-low"),
        company_count=len(sov["companies"]),
        competitor_count=competitor_count,
        model_count=len(sov["models"]),
        query_count=len(queries),
        # Full data
        companies=sov["companies"],
        models=sov["models"],
        by_model_spotlight=sov["by_model_spotlight"],
        citations=sov["citations"],
        # GAP analysis
        gap=gap,
        # Appendix
        prompts_list=[e["prompt"] for e in response_log],
        response_log=response_log,
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M")
    slug = run["topic"].lower().replace(" ", "-").replace("/", "-")
    filename = f"sov-{slug}-{ts}.html"
    out_path = str(Path(output_dir) / filename)
    Path(out_path).write_text(html, encoding="utf-8")
    return out_path
