from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse

from src.config import CompanyEntry

_URL_RE = re.compile(r'https?://[^\s\)\]"\'<>]+')

_TWO_PART_TLDS = {"co", "com", "org", "net", "gov", "ac", "edu"}

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
        host = urlparse(url).netloc.split(":")[0]  # strip port
        if not host:
            return url
        parts = host.split(".")
        if len(parts) >= 3 and parts[-2] in _TWO_PART_TLDS:
            return ".".join(parts[-3:])
        return ".".join(parts[-2:]) if len(parts) >= 2 else host
    except Exception:
        return url


def _match_company(text: str, companies: list[CompanyEntry]) -> str | None:
    """Return company name if text contains a company name/alias stem."""
    normalized = text.lower().replace(" ", "").replace("-", "").replace("&", "").replace(",", "")
    for company in companies:
        for token in [company.name] + company.aliases:
            stem = token.lower().replace(" ", "").replace("-", "").replace("&", "").replace(",", "")
            # minimum 3 to support 3-char firm aliases such as BCG and PwC
            if len(stem) >= 3 and stem in normalized:
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
            # skip names shorter than 3 chars (same threshold as _match_company)
            if len(name) < 3:
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
