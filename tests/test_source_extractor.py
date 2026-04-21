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
