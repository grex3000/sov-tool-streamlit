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
