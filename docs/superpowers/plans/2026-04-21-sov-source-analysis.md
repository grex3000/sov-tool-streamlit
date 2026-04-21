# SOV Source Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture the sources each LLM cites per query, count source-based company mentions toward SOV scores, show sources in the HTML report, and let users export all citations as CSV/Excel.

**Architecture:** A new `src/source_extractor.py` module extracts citations three ways (structured API field, URL regex, text-pattern). `QueryResult` gains a `citations` field for Perplexity's structured URLs. Source citations that match a company are inserted into the DB as mentions with `match_type="source_citation"` (so SOV maths require no changes). `generate_report` accepts an optional `source_citations` list to render a domain-frequency section and per-query source panels. Export buttons in `app.py` produce in-memory CSV/Excel from the same list.

**Tech Stack:** Python stdlib (`re`, `urllib.parse`, `csv`, `io`), `openpyxl` (new), existing `rapidfuzz`/`jinja2` stack.

---

## File Map

| Action | Path | What changes |
|--------|------|--------------|
| Create | `src/source_extractor.py` | `SourceCitation` dataclass + `extract_sources()` + `_parse_domain()` + `_match_company()` |
| Create | `tests/test_source_extractor.py` | Unit tests for all three extraction methods |
| Create | `tests/test_domain_stats.py` | Unit tests for `compute_domain_stats` |
| Modify | `src/query_engine.py` | Add `citations: list[str]` to `QueryResult`; capture from API response |
| Modify | `src/report.py` | Add `compute_domain_stats()`; update `_build_response_log` + `generate_report` to accept `source_citations` |
| Modify | `templates/report.html.j2` | Per-query sources panel inside response-log cards; new domain-frequency section after gap analysis |
| Modify | `app.py` | Call `extract_sources` in scan loop; insert source mentions to DB; store citations in session; pass to `generate_report`; add export buttons |
| Modify | `requirements.txt` | Add `openpyxl>=3.1.0` |

---

## Task 1: Add openpyxl to requirements.txt

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add dependency**

Open `requirements.txt` and add at the end:
```
openpyxl>=3.1.0
```

- [ ] **Step 2: Install**

```bash
pip install openpyxl>=3.1.0
```
Expected: installs without error.

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: add openpyxl dependency for source export"
```

---

## Task 2: Create `src/source_extractor.py` with tests

**Files:**
- Create: `src/source_extractor.py`
- Create: `tests/__init__.py` (empty)
- Create: `tests/test_source_extractor.py`

- [ ] **Step 1: Write failing tests**

Create `tests/__init__.py` (empty file), then create `tests/test_source_extractor.py`:

```python
from __future__ import annotations

import pytest
from src.config import CompanyEntry
from src.source_extractor import SourceCitation, _match_company, _parse_domain, extract_sources

_COMPANIES = [
    CompanyEntry(name="Roland Berger", aliases=["Roland Berger Strategy Consultants"], is_target=True),
    CompanyEntry(name="McKinsey & Company", aliases=["McKinsey"], is_target=False),
    CompanyEntry(name="Boston Consulting Group", aliases=["BCG"], is_target=False),
]


def test_parse_domain_strips_www():
    assert _parse_domain("https://www.mckinsey.com/article") == "mckinsey.com"


def test_parse_domain_no_www():
    assert _parse_domain("https://bcg.com/insights") == "bcg.com"


def test_parse_domain_invalid_returns_input():
    assert _parse_domain("not-a-url") == "not-a-url"


def test_match_company_finds_mckinsey():
    assert _match_company("mckinsey.com", _COMPANIES) == "McKinsey & Company"


def test_match_company_finds_by_alias():
    assert _match_company("bcg.com", _COMPANIES) == "Boston Consulting Group"


def test_match_company_no_match():
    assert _match_company("hbr.org", _COMPANIES) is None


def test_extract_sources_structured_url():
    citations = extract_sources(
        response_text="No URLs here.",
        structured_urls=["https://www.mckinsey.com/report"],
        companies=_COMPANIES,
        query_prompt="Who are top strategy consultants?",
        model_id="perplexity/sonar-pro",
        model_label="Perplexity Sonar Pro",
    )
    assert len(citations) == 1
    assert citations[0].domain == "mckinsey.com"
    assert citations[0].match_type == "structured"
    assert citations[0].company_match == "McKinsey & Company"


def test_extract_sources_url_in_text():
    citations = extract_sources(
        response_text="See https://www.bcg.com/insights for more.",
        structured_urls=[],
        companies=_COMPANIES,
        query_prompt="Best consultants?",
        model_id="openai/gpt-4o",
        model_label="GPT-4o",
    )
    assert any(c.domain == "bcg.com" and c.match_type == "url" for c in citations)


def test_extract_sources_text_pattern():
    citations = extract_sources(
        response_text="According to McKinsey, digital transformation is key.",
        structured_urls=[],
        companies=_COMPANIES,
        query_prompt="Digital transformation trends?",
        model_id="openai/gpt-4o",
        model_label="GPT-4o",
    )
    assert any(c.match_type == "text_pattern" and c.company_match == "McKinsey & Company" for c in citations)


def test_extract_sources_deduplication():
    # Same domain via two methods (structured + text url) → one structured, one url → both kept (different match_type)
    citations = extract_sources(
        response_text="See https://www.mckinsey.com/article for details.",
        structured_urls=["https://mckinsey.com/overview"],
        companies=_COMPANIES,
        query_prompt="Consultants?",
        model_id="perplexity/sonar-pro",
        model_label="Perplexity Sonar Pro",
    )
    types = {c.match_type for c in citations if c.domain == "mckinsey.com"}
    assert "structured" in types
    assert "url" in types


def test_extract_sources_empty_response():
    citations = extract_sources(
        response_text="",
        structured_urls=[],
        companies=_COMPANIES,
        query_prompt="Any query",
        model_id="openai/gpt-4o",
        model_label="GPT-4o",
    )
    assert citations == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/gregor/sov-streamlit && python -m pytest tests/test_source_extractor.py -v 2>&1 | head -30
```
Expected: `ModuleNotFoundError: No module named 'src.source_extractor'`

- [ ] **Step 3: Create `src/source_extractor.py`**

```python
from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse

from src.config import CompanyEntry

_URL_RE = re.compile(r'https?://[^\s\)\]"\'<>]+')

_TEXT_PATTERN_TEMPLATES = [
    r'\baccording to\s+{name}',
    r'\bsource:\s+{name}',
    r'\bcited (?:in|by)\s+{name}',
    r'\bpublished by\s+{name}',
    r'\bper\s+{name}',
]


@dataclass
class SourceCitation:
    url: str
    domain: str
    company_match: str | None
    match_type: str  # "structured", "url", or "text_pattern"
    query_prompt: str
    model_id: str
    model_label: str


def _parse_domain(url: str) -> str:
    """Return SLD+TLD from a URL, e.g. 'https://www.mckinsey.com/a' → 'mckinsey.com'."""
    try:
        host = urlparse(url).netloc
        if not host:
            return url
        parts = host.split(".")
        return ".".join(parts[-2:]) if len(parts) >= 2 else host
    except Exception:
        return url


def _match_company(text: str, companies: list[CompanyEntry]) -> str | None:
    """Return company name if text contains a company name/alias stem."""
    normalized = text.lower().replace(" ", "").replace("-", "").replace("&", "").replace(",", "")
    for company in companies:
        for token in [company.name] + company.aliases:
            stem = token.lower().replace(" ", "").replace("-", "").replace("&", "").replace(",", "")
            if len(stem) >= 4 and stem in normalized:
                return company.name
    return None


def extract_sources(
    response_text: str,
    structured_urls: list[str],
    companies: list[CompanyEntry],
    query_prompt: str,
    model_id: str,
    model_label: str,
) -> list[SourceCitation]:
    """
    Extract source citations from a model response using three methods.
    Dedup key: (domain, match_type) — same pair counted once per call.
    """
    seen: set[tuple[str, str]] = set()
    results: list[SourceCitation] = []

    def _add(url: str, domain: str, match_type: str) -> None:
        key = (domain, match_type)
        if key in seen:
            return
        seen.add(key)
        results.append(SourceCitation(
            url=url,
            domain=domain,
            company_match=_match_company(domain, companies),
            match_type=match_type,
            query_prompt=query_prompt,
            model_id=model_id,
            model_label=model_label,
        ))

    # 1. Structured citations (Perplexity API field)
    for url in structured_urls:
        domain = _parse_domain(url)
        if domain:
            _add(url, domain, "structured")

    # 2. URLs extracted from response text
    for m in _URL_RE.finditer(response_text):
        url = m.group().rstrip(".,;)")
        domain = _parse_domain(url)
        if domain:
            _add(url, domain, "url")

    # 3. Text-pattern company name mentions
    for company in companies:
        for name in [company.name] + company.aliases:
            if len(name) < 4:
                continue
            matched = False
            for tmpl in _TEXT_PATTERN_TEMPLATES:
                pattern = tmpl.format(name=re.escape(name))
                if re.search(pattern, response_text, re.IGNORECASE):
                    key = (name.lower(), "text_pattern")
                    if key not in seen:
                        seen.add(key)
                        results.append(SourceCitation(
                            url="",
                            domain=name.lower(),
                            company_match=company.name,
                            match_type="text_pattern",
                            query_prompt=query_prompt,
                            model_id=model_id,
                            model_label=model_label,
                        ))
                    matched = True
                    break
            if matched:
                break  # one match per company is enough

    return results
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/gregor/sov-streamlit && python -m pytest tests/test_source_extractor.py -v
```
Expected: all 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/source_extractor.py tests/__init__.py tests/test_source_extractor.py
git commit -m "feat: add source_extractor module with tests"
```

---

## Task 3: Add `citations` field to `QueryResult` in `query_engine.py`

**Files:**
- Modify: `src/query_engine.py`

- [ ] **Step 1: Update `QueryResult` dataclass**

In `src/query_engine.py`, change the `QueryResult` dataclass (lines 18–25) from:

```python
@dataclass
class QueryResult:
    model_id: str
    model_label: str
    prompt: str
    response: str | None
    error: str | None = None
```

to:

```python
from dataclasses import dataclass, field

@dataclass
class QueryResult:
    model_id: str
    model_label: str
    prompt: str
    response: str | None
    error: str | None = None
    citations: list[str] = field(default_factory=list)
```

(Also add `field` to the `from dataclasses import dataclass` line at top.)

- [ ] **Step 2: Capture citations from API response**

In `_query_one`, replace the return statement inside the `try` block (currently lines 54–58):

```python
            content = resp.choices[0].message.content
            return QueryResult(
                model_id=model_id, model_label=model_label,
                prompt=prompt, response=content if content else "",
            )
```

with:

```python
            content = resp.choices[0].message.content
            raw_citations = []
            if hasattr(resp, "model_extra") and resp.model_extra:
                raw_citations = [str(u) for u in resp.model_extra.get("citations", []) if u]
            return QueryResult(
                model_id=model_id, model_label=model_label,
                prompt=prompt, response=content if content else "",
                citations=raw_citations,
            )
```

- [ ] **Step 3: Verify the app still imports correctly**

```bash
cd /Users/gregor/sov-streamlit && python -c "from src.query_engine import QueryResult, run_queries; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/query_engine.py
git commit -m "feat: capture structured citations from OpenRouter API response"
```

---

## Task 4: Add `compute_domain_stats` to `report.py` with tests

**Files:**
- Modify: `src/report.py`
- Create: `tests/test_domain_stats.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_domain_stats.py`:

```python
from __future__ import annotations

from src.report import compute_domain_stats
from src.source_extractor import SourceCitation


def _cite(domain, model_id="m1", model_label="M1", company_match=None):
    return SourceCitation(
        url="",
        domain=domain,
        company_match=company_match,
        match_type="url",
        query_prompt="test",
        model_id=model_id,
        model_label=model_label,
    )


def test_compute_domain_stats_aggregate():
    citations = [
        _cite("mckinsey.com", company_match="McKinsey"),
        _cite("mckinsey.com", company_match="McKinsey"),
        _cite("hbr.org"),
    ]
    stats = compute_domain_stats(citations)
    agg = {e["domain"]: e for e in stats["aggregate"]}
    assert agg["mckinsey.com"]["count"] == 2
    assert agg["mckinsey.com"]["company_match"] == "McKinsey"
    assert agg["hbr.org"]["count"] == 1


def test_compute_domain_stats_by_model():
    citations = [
        _cite("bcg.com", model_id="m1", model_label="GPT-4o"),
        _cite("bcg.com", model_id="m2", model_label="Claude"),
        _cite("hbr.org", model_id="m1", model_label="GPT-4o"),
    ]
    stats = compute_domain_stats(citations)
    m1 = {e["domain"]: e for e in stats["by_model"]["m1"]["domains"]}
    assert m1["bcg.com"]["count"] == 1
    assert "m2" in stats["by_model"]


def test_compute_domain_stats_empty():
    stats = compute_domain_stats([])
    assert stats["aggregate"] == []
    assert stats["by_model"] == {}


def test_compute_domain_stats_top10_limit():
    citations = [_cite(f"site{i}.com") for i in range(15)]
    stats = compute_domain_stats(citations)
    assert len(stats["aggregate"]) <= 10
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/gregor/sov-streamlit && python -m pytest tests/test_domain_stats.py -v 2>&1 | head -20
```
Expected: `ImportError` or `AttributeError` since `compute_domain_stats` doesn't exist yet.

- [ ] **Step 3: Add `compute_domain_stats` to `src/report.py`**

Add this function after the `score_class` function (after line 23 in `report.py`):

```python
def compute_domain_stats(source_citations: list) -> dict:
    """
    Aggregate source citations by domain for report display.

    Returns:
    {
      "aggregate": [{"domain", "count", "company_match"}, ...],  # top 10, sorted by count
      "by_model": {model_id: {"label": str, "domains": [{"domain", "count", "company_match"}, ...]}}
    }
    """
    from collections import defaultdict

    agg: dict[str, dict] = {}
    by_model: dict[str, dict] = {}

    for c in source_citations:
        # aggregate
        if c.domain not in agg:
            agg[c.domain] = {"domain": c.domain, "count": 0, "company_match": c.company_match}
        agg[c.domain]["count"] += 1

        # per model
        if c.model_id not in by_model:
            by_model[c.model_id] = {"label": c.model_label, "domains": {}}
        model_domains = by_model[c.model_id]["domains"]
        if c.domain not in model_domains:
            model_domains[c.domain] = {"domain": c.domain, "count": 0, "company_match": c.company_match}
        model_domains[c.domain]["count"] += 1

    aggregate = sorted(agg.values(), key=lambda x: x["count"], reverse=True)[:10]

    by_model_out = {
        mid: {
            "label": data["label"],
            "domains": sorted(data["domains"].values(), key=lambda x: x["count"], reverse=True)[:10],
        }
        for mid, data in by_model.items()
    }

    return {"aggregate": aggregate, "by_model": by_model_out}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/gregor/sov-streamlit && python -m pytest tests/test_domain_stats.py -v
```
Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/report.py tests/test_domain_stats.py
git commit -m "feat: add compute_domain_stats to report module"
```

---

## Task 5: Update `generate_report` and `_build_response_log` to accept source citations

**Files:**
- Modify: `src/report.py`

- [ ] **Step 1: Update `_build_response_log` signature and attach sources per response**

Replace the `_build_response_log` function signature (line ~361) and its response-building loop.

Change the function signature from:

```python
def _build_response_log(
    queries: list[dict],
    mentions: list[dict],
    companies: list[CompanyEntry],
) -> list[dict]:
```

to:

```python
def _build_response_log(
    queries: list[dict],
    mentions: list[dict],
    companies: list[CompanyEntry],
    source_citations: list | None = None,
) -> list[dict]:
```

Then, just before `return entries` at the end of `_build_response_log`, add:

```python
    # Attach per-response source citations
    if source_citations:
        # Index: (prompt, model_id) → list[SourceCitation]
        from collections import defaultdict as _dd
        src_index: dict[tuple[str, str], list] = _dd(list)
        for sc in source_citations:
            src_index[(sc.query_prompt, sc.model_id)].append(sc)

        for entry in entries:
            for r in entry["responses"]:
                r["sources"] = src_index.get((entry["prompt"], r["model_id"]), [])
    else:
        for entry in entries:
            for r in entry["responses"]:
                r["sources"] = []
```

- [ ] **Step 2: Update `generate_report` to accept and pass through `source_citations`**

Change the `generate_report` function signature (line ~433) from:

```python
def generate_report(
    run: dict,
    queries: list[dict],
    mentions: list[dict],
    companies: list[CompanyEntry],
    template_path: str,
    output_dir: str,
) -> str:
```

to:

```python
def generate_report(
    run: dict,
    queries: list[dict],
    mentions: list[dict],
    companies: list[CompanyEntry],
    template_path: str,
    output_dir: str,
    source_citations: list | None = None,
) -> str:
```

Update the `compute_sov` call and `_build_response_log` call inside `generate_report`:

Change:
```python
    response_log = _build_response_log(queries, mentions, companies)
```

to:
```python
    response_log = _build_response_log(queries, mentions, companies, source_citations)
    domain_stats = compute_domain_stats(source_citations or [])
```

And in the `template.render(...)` call, add two new keyword arguments after `gap=gap,`:

```python
        domain_stats=domain_stats,
        source_citations=source_citations or [],
```

- [ ] **Step 3: Verify no import errors**

```bash
cd /Users/gregor/sov-streamlit && python -c "from src.report import generate_report, compute_domain_stats; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Run all existing tests still pass**

```bash
cd /Users/gregor/sov-streamlit && python -m pytest tests/ -v
```
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/report.py
git commit -m "feat: thread source_citations through report generation pipeline"
```

---

## Task 6: Integrate source extraction into scan loop in `app.py`

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add `source_citations` to session state initialization**

In `app.py` around line 707, in the `_init` dict, add:

```python
    "source_citations":  [],
```

So the dict becomes:
```python
_init = {
    "app_stage":         "config",
    "report_html":       None,
    "report_path":       None,
    "source_citations":  [],
    ...
}
```

- [ ] **Step 2: Import `extract_sources` at the top of `app.py`**

After the existing `from src.detector import CompanyRef, detect_all_mentions` line, add:

```python
from src.source_extractor import extract_sources
```

- [ ] **Step 3: Extract sources and insert source-citation mentions during the scan loop**

In `app.py`, locate the scan loop (around lines 1228–1238):

```python
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
```

Replace with:

```python
            total_mentions = 0
            all_source_citations = []
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

                    # Source citation extraction
                    src_hits = extract_sources(
                        response_text=r.response,
                        structured_urls=r.citations,
                        companies=companies,
                        query_prompt=r.prompt,
                        model_id=r.model_id,
                        model_label=r.model_label,
                    )
                    all_source_citations.extend(src_hits)
                    for sc in src_hits:
                        if sc.company_match:
                            company_obj = next(
                                (c for c in companies if c.name == sc.company_match), None
                            )
                            if company_obj:
                                insert_mention(
                                    DB_PATH, query_id=qid,
                                    company_name=sc.company_match,
                                    is_target=company_obj.is_target,
                                    match_type="source_citation",
                                    excerpt=sc.url or sc.domain,
                                )
                                total_mentions += 1
```

- [ ] **Step 4: Store source citations in session and pass to `generate_report`**

After `_ss.source_citations = all_source_citations` — locate the `generate_report` call (around line 1257):

```python
            report_path = generate_report(
                run=run_record, queries=queries, mentions=mentions, companies=companies,
                template_path=TEMPLATE_PATH, output_dir=REPORTS_DIR,
            )
            _ss.report_html      = pathlib.Path(report_path).read_text(encoding="utf-8")
            _ss.report_path      = report_path
```

Replace with:

```python
            _ss.source_citations = all_source_citations
            report_path = generate_report(
                run=run_record, queries=queries, mentions=mentions, companies=companies,
                template_path=TEMPLATE_PATH, output_dir=REPORTS_DIR,
                source_citations=all_source_citations,
            )
            _ss.report_html      = pathlib.Path(report_path).read_text(encoding="utf-8")
            _ss.report_path      = report_path
```

- [ ] **Step 5: Verify syntax**

```bash
cd /Users/gregor/sov-streamlit && python -c "import ast; ast.parse(open('app.py').read()); print('OK')"
```
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add app.py
git commit -m "feat: extract sources during scan and insert source-citation mentions"
```

---

## Task 7: Add per-query sources panel and domain frequency section to HTML template

**Files:**
- Modify: `templates/report.html.j2`

- [ ] **Step 1: Add CSS for source elements**

In `templates/report.html.j2`, find the closing `</style>` tag (just before `</head>`). Add these styles immediately before `</style>`:

```css
    /* ── Source Citations ─────────────────────────────── */
    .source-list { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; padding-top: 8px; border-top: 1px solid var(--border); }
    .source-tag { display: inline-flex; align-items: center; gap: 5px; font-size: 11px; padding: 3px 8px; border: 1px solid var(--border); background: var(--surface); color: var(--muted); max-width: 240px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .source-tag a { color: var(--accent); text-decoration: none; overflow: hidden; text-overflow: ellipsis; }
    .source-tag a:hover { text-decoration: underline; }
    .source-badge { font-size: 9px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; padding: 2px 5px; border-radius: 2px; }
    .source-badge.structured { background: #dbeafe; color: #1e40af; }
    .source-badge.url        { background: var(--accent-light); color: var(--accent-dim); }
    .source-badge.text_pattern { background: #f3e8ff; color: #7c3aed; }
    .domain-table { width: 100%; border-collapse: collapse; font-size: 13px; margin-bottom: 24px; }
    .domain-table th { padding: 9px 12px; text-align: left; font-size: 11px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); border-bottom: 2px solid var(--border); }
    .domain-table td { padding: 9px 12px; border-bottom: 1px solid var(--border); }
    .domain-table tbody tr:last-child td { border-bottom: none; }
    .domain-table tbody tr:hover td { background: rgba(0,0,0,.018); }
    .domain-count { font-family: var(--mono); font-weight: 600; color: var(--text); }
    .domain-company { font-size: 11px; font-weight: 600; padding: 2px 7px; }
    .domain-company.is-target { background: #dcfce7; color: #166534; }
    .domain-company.is-competitor { background: var(--accent-light); color: var(--accent-dim); }
    .domain-tabs { display: flex; gap: 2px; margin-bottom: 16px; border-bottom: 2px solid var(--border); }
    .domain-tab { padding: 8px 16px; font-size: 12px; font-weight: 600; cursor: pointer; border: none; background: none; color: var(--muted); border-bottom: 2px solid transparent; margin-bottom: -2px; }
    .domain-tab.active { color: var(--text); border-bottom-color: var(--text); }
    .domain-panel { display: none; }
    .domain-panel.active { display: block; }
```

- [ ] **Step 2: Add per-query sources to response-log model cards**

In the template, find the response-log model card content (around line 741):

```html
          <div class="response-text">{{ r.response_html }}</div>
        </div>
```

Replace with:

```html
          <div class="response-text">{{ r.response_html }}</div>
          {% if r.sources %}
          <div class="source-list">
            {% for s in r.sources %}
            <span class="source-tag">
              <span class="source-badge {{ s.match_type }}">{{ s.match_type | replace('_', ' ') }}</span>
              {% if s.url %}
              <a href="{{ s.url }}" target="_blank" rel="noopener">{{ s.domain }}</a>
              {% else %}
              <span>{{ s.domain }}</span>
              {% endif %}
            </span>
            {% endfor %}
          </div>
          {% endif %}
        </div>
```

- [ ] **Step 3: Add domain frequency section after gap analysis**

In the template, find the line `{# ════════════════════════ 05 · EXAMPLE CITATIONS ═════════════════════════ #}` (around line 602) and insert a new section immediately before it:

```html
  {# ════════════════════════ 04b · SOURCE DOMAINS ════════════════════════════ #}
  {% if domain_stats and domain_stats.aggregate %}
  <section class="section" id="source-domains">
    <div class="section-header">
      <span class="sec-num">04b</span>
      <span class="sec-title">Source Domains</span>
    </div>
    <p class="sec-desc">
      Domains and publications cited by AI models when answering the queries.
      Company-matched domains are highlighted.
    </p>

    <div class="domain-tabs" id="domain-tabs">
      <button class="domain-tab active" data-panel="aggregate" onclick="switchDomainTab(this)">All Models</button>
      {% for mid, data in domain_stats.by_model.items() %}
      <button class="domain-tab" data-panel="{{ mid }}" onclick="switchDomainTab(this)">{{ data.label }}</button>
      {% endfor %}
    </div>

    <div class="domain-panel active" id="dp-aggregate">
      <table class="domain-table">
        <thead><tr><th>Domain</th><th>Company</th><th style="text-align:right">Citations</th></tr></thead>
        <tbody>
          {% for entry in domain_stats.aggregate %}
          <tr>
            <td>{{ entry.domain }}</td>
            <td>
              {% if entry.company_match %}
              {% set is_tgt = companies | selectattr('name', 'equalto', entry.company_match) | map(attribute='is_target') | first | default(false) %}
              <span class="domain-company {{ 'is-target' if is_tgt else 'is-competitor' }}">{{ entry.company_match }}</span>
              {% endif %}
            </td>
            <td style="text-align:right"><span class="domain-count">{{ entry.count }}</span></td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>

    {% for mid, data in domain_stats.by_model.items() %}
    <div class="domain-panel" id="dp-{{ mid }}">
      <table class="domain-table">
        <thead><tr><th>Domain</th><th>Company</th><th style="text-align:right">Citations</th></tr></thead>
        <tbody>
          {% for entry in data.domains %}
          <tr>
            <td>{{ entry.domain }}</td>
            <td>
              {% if entry.company_match %}
              {% set is_tgt = companies | selectattr('name', 'equalto', entry.company_match) | map(attribute='is_target') | first | default(false) %}
              <span class="domain-company {{ 'is-target' if is_tgt else 'is-competitor' }}">{{ entry.company_match }}</span>
              {% endif %}
            </td>
            <td style="text-align:right"><span class="domain-count">{{ entry.count }}</span></td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
    {% endfor %}
  </section>
  {% endif %}

```

- [ ] **Step 4: Add tab-switching JavaScript**

Find the closing `</body>` tag (near the very end of the template) and add this script just before it:

```html
<script>
function switchDomainTab(btn) {
  const tabs = document.querySelectorAll('#domain-tabs .domain-tab');
  const panels = document.querySelectorAll('#source-domains .domain-panel');
  tabs.forEach(t => t.classList.remove('active'));
  panels.forEach(p => p.classList.remove('active'));
  btn.classList.add('active');
  const target = document.getElementById('dp-' + btn.dataset.panel);
  if (target) target.classList.add('active');
}
</script>
```

- [ ] **Step 5: Verify template renders without error (dry run)**

```bash
cd /Users/gregor/sov-streamlit && python -c "
from jinja2 import Environment, FileSystemLoader
env = Environment(loader=FileSystemLoader('templates'), autoescape=True)
tmpl = env.get_template('report.html.j2')
html = tmpl.render(
    company_name='Test Co', topic='test', period='Q2 2026',
    generated_date='April 21, 2026', run={},
    target_avg_sov=50, target_rank=1, target_score_class='score-high',
    company_count=2, competitor_count=1, model_count=2, query_count=10,
    companies=[], models=[], by_model_spotlight={}, citations=[],
    gap=None, domain_stats={'aggregate': [], 'by_model': {}},
    source_citations=[],
    prompts_list=[], response_log=[],
)
print('OK', len(html), 'chars')
"
```
Expected: `OK NNNNN chars`

- [ ] **Step 6: Commit**

```bash
git add templates/report.html.j2
git commit -m "feat: add source domains section and per-query source panels to report template"
```

---

## Task 8: Add CSV and Excel export buttons in `app.py`

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Add export helper functions**

In `app.py`, after the `_warning_html` function (around line 173), add two helper functions:

```python
def _build_sources_csv(source_citations: list) -> str:
    """Return sources as a UTF-8 CSV string."""
    import csv
    import io
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["query", "model", "domain", "url", "match_type", "company_match"])
    for sc in source_citations:
        writer.writerow([
            sc.query_prompt,
            sc.model_label,
            sc.domain,
            sc.url,
            sc.match_type,
            sc.company_match or "",
        ])
    return buf.getvalue()


def _build_sources_excel(source_citations: list) -> bytes:
    """Return sources as an in-memory Excel (.xlsx) file."""
    import io
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sources"
    ws.append(["Query", "Model", "Domain", "URL", "Match Type", "Company Match"])
    for sc in source_citations:
        ws.append([
            sc.query_prompt,
            sc.model_label,
            sc.domain,
            sc.url,
            sc.match_type,
            sc.company_match or "",
        ])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
```

- [ ] **Step 2: Add export buttons to the report section**

In `app.py`, find the report download section (around lines 1031–1041):

```python
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
```

Replace with:

```python
        col_dl, col_csv, col_xlsx, col_new = st.columns(4)
        with col_dl:
            st.download_button("Download Report", data=_ss.report_html,
                               file_name=report_name, mime="text/html",
                               use_container_width=True)
        with col_csv:
            if _ss.source_citations:
                st.download_button(
                    "Export Sources (CSV)",
                    data=_build_sources_csv(_ss.source_citations),
                    file_name=report_name.replace(".html", "-sources.csv"),
                    mime="text/csv",
                    use_container_width=True,
                )
        with col_xlsx:
            if _ss.source_citations:
                st.download_button(
                    "Export Sources (Excel)",
                    data=_build_sources_excel(_ss.source_citations),
                    file_name=report_name.replace(".html", "-sources.xlsx"),
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
        with col_new:
            if st.button("New Scan", type="primary", use_container_width=True):
                _ss.report_html = None
                _ss.report_path = None
                _ss.source_citations = []
                st.rerun()
```

- [ ] **Step 3: Clear source_citations on new scan**

Also ensure `source_citations` is cleared when "New Scan" resets state. The replacement in Step 2 already adds `_ss.source_citations = []` before `st.rerun()`.

- [ ] **Step 4: Verify syntax**

```bash
cd /Users/gregor/sov-streamlit && python -c "import ast; ast.parse(open('app.py').read()); print('OK')"
```
Expected: `OK`

- [ ] **Step 5: Run all tests**

```bash
cd /Users/gregor/sov-streamlit && python -m pytest tests/ -v
```
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add app.py
git commit -m "feat: add CSV and Excel source export buttons"
```

---

## Self-Review

**Spec coverage check:**
- Source extraction (structured/URL/text-pattern) → Task 2 ✓
- Structured citations from API → Task 3 ✓
- SOV integration (source citations count as mentions) → Task 6 ✓
- Per-query sources in report → Task 7 Step 2 ✓
- Domain frequency section in report → Task 7 Step 3 ✓
- CSV export → Task 8 ✓
- Excel export → Task 8 ✓
- `openpyxl` dependency → Task 1 ✓
- No DB persistence (per-run only via session state) → Task 6 ✓

**Placeholder scan:** None found.

**Type consistency check:**
- `SourceCitation` fields (`url`, `domain`, `company_match`, `match_type`, `query_prompt`, `model_id`, `model_label`) are consistent across Tasks 2, 4, 5, 6, 7, 8.
- `compute_domain_stats(source_citations)` → `{"aggregate": [...], "by_model": {...}}` matches template usage in Task 7.
- `_build_response_log` receives `source_citations` list and attaches `r["sources"]` → template accesses `r.sources` ✓
- `generate_report` accepts `source_citations=` and passes `domain_stats=` + `source_citations=` to template ✓
