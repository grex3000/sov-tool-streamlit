from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz

_MIN_FUZZY_LEN = 5    # names shorter than this skip fuzzy matching
_FUZZY_THRESHOLD = 88
_MIN_SENTENCE_LEN = 25  # sentences shorter than this skip fuzzy matching
                        # prevents 1-char markdown fragments ("g", "S") from
                        # scoring 100% via partial_ratio on a contained letter


@dataclass
class CompanyRef:
    name: str
    aliases: list[str]
    is_target: bool = False


def detect_mention(
    text: str,
    name: str,
    aliases: list[str],
) -> dict | None:
    """
    Detect a single company in text using exact then fuzzy matching.
    Returns {"type", "matched", "excerpt", "score"} or None.
    """
    all_names = [name] + (aliases or [])
    text_lower = text.lower()

    # ── Exact / case-insensitive ─────────────────────────────────────
    for n in all_names:
        idx = text_lower.find(n.lower())
        if idx != -1:
            start = max(0, idx - 80)
            end = min(len(text), idx + len(n) + 80)
            return {
                "type": "exact",
                "matched": n,
                "excerpt": text[start:end].strip(),
                "score": 100,
            }

    # ── Fuzzy per sentence ───────────────────────────────────────────
    # Split on sentence-ending punctuation OR a blank line, not on every
    # bare newline — this prevents single-char markdown fragments
    # (e.g. "g" from "**Roland Berger**\ng\nMore text") from being tested.
    for sentence in re.split(r"(?<=[.!?])\s+|\n{2,}", text):
        sentence = sentence.strip()
        if len(sentence) < _MIN_SENTENCE_LEN:
            continue
        for n in all_names:
            if len(n) < _MIN_FUZZY_LEN:
                continue
            score = fuzz.partial_ratio(n.lower(), sentence.lower())
            if score >= _FUZZY_THRESHOLD:
                return {
                    "type": "fuzzy",
                    "matched": n,
                    "excerpt": sentence,
                    "score": score,
                }

    return None


def detect_all_mentions(
    text: str,
    companies: list[CompanyRef],
) -> list[dict]:
    """
    Run detection for every company. Returns a list of mention dicts, one per
    company that was found. Companies with no match are omitted.

    Each dict: {"company_name", "is_target", "type", "matched", "excerpt", "score"}
    """
    results = []
    for company in companies:
        hit = detect_mention(text, company.name, company.aliases)
        if hit:
            results.append(
                {
                    "company_name": company.name,
                    "is_target": company.is_target,
                    **hit,
                }
            )
    return results
