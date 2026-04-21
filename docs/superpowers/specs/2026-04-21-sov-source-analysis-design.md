# SOV Source Analysis — Design Spec

**Date:** 2026-04-21  
**Status:** Approved

## Overview

Extend the SOV scanner to capture, analyze, and export the sources each LLM cites when answering queries. Source citations count toward company SOV scores and are displayed in the report alongside a domain frequency analysis. A per-run CSV/Excel export is available.

---

## Source Extraction (`src/source_extractor.py`)

A new module handles three extraction methods, producing a unified list of `SourceCitation` objects.

**`SourceCitation` fields:**
- `url` — raw URL string (empty string if text-pattern match only)
- `domain` — parsed domain (e.g. `mckinsey.com`)
- `company_match` — matched company name from alias list, or `None`
- `match_type` — one of `"structured"`, `"url"`, `"text_pattern"`
- `query_id` — links back to the originating query
- `model_id` — model that produced the response

**Extraction methods (applied in order, deduplicated by URL/domain):**

1. **Structured citations** — parse `citations` field from OpenRouter API response. Perplexity models return this as a list of URLs. `query_engine.py` passes it through alongside response text.

2. **URL regex** — scan response text for `https?://[^\s\)\]"']+` patterns. Parse domain using `urllib.parse.urlparse`, stripping `www.` prefix.

3. **Text patterns** — regex match against known company names/aliases for phrases like:
   - `According to {company}`
   - `Source: {company}`
   - `cited (in|by) {company}`
   - `published by {company}`
   - `per {company}`

   Uses the same company alias list already in memory for the run.

All three methods run for every query response. Results are deduplicated: same domain within the same query/model counts once per match_type.

---

## SOV Integration (`src/detector.py`)

After existing text-based mention detection runs, source citations with a `company_match` are appended as `Mention` objects with `match_type="source_citation"`. This feeds directly into the existing SOV percentage calculation — no separate scoring track. A company whose domain is cited as a source is counted as mentioned in that query.

---

## Report Additions (`src/report.py` + `templates/report.html.j2`)

### Per-query sources panel
Below each query's response excerpt, show a compact list of sources found. Only rendered for queries with at least one citation. Each entry shows: domain, full URL (if available), and match_type badge (`structured` / `url` / `text`).

### Domain frequency section
A new report section showing the top 10 cited domains per model and in aggregate. Columns: domain, citation count, company match (if any). Positioned after the gap analysis section.

---

## Export (`app.py`)

After a scan completes, two download buttons appear in the Streamlit UI alongside the report:
- **Export Sources (CSV)** — generated via Python's `csv` module
- **Export Sources (Excel)** — generated via `openpyxl`

Both are in-memory (no file written to disk). Export is flat, one row per source citation:

| Column | Description |
|--------|-------------|
| `run_id` | Scan run identifier |
| `query` | The prompt text sent to the model |
| `model` | Model ID |
| `domain` | Parsed domain |
| `url` | Full URL (empty if text-pattern only) |
| `match_type` | `structured`, `url`, or `text_pattern` |
| `company_match` | Matched company name, or blank |

---

## Dependencies

- `openpyxl` — add to `requirements.txt` for Excel export
- No other new dependencies

---

## Out of Scope

- Persistent storage of sources in SQLite (per-run only)
- Sentiment analysis of cited sources
- Fetching/crawling cited URLs
