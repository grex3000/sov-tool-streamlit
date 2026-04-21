from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

_SENTIMENT_MODEL = "openai/gpt-4o-mini"
_SYSTEM = (
    "You are a sentiment classifier. Given a text and a list of company names, "
    "classify how the text portrays each company as one of: positive, neutral, negative. "
    "Reply with a JSON object only, mapping each company name to its sentiment value."
)


async def _score_one(
    client: AsyncOpenAI,
    response_text: str,
    company_names: list[str],
    query_id: int,
) -> dict[str, str]:
    if not response_text.strip():
        return {name: "neutral" for name in company_names}
    prompt = (
        f"Text:\n{response_text[:3000]}\n\n"
        f"Companies: {json.dumps(company_names)}\n\nReturn JSON only."
    )
    try:
        resp = await client.chat.completions.create(
            model=_SENTIMENT_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
            timeout=30,
        )
        raw = resp.choices[0].message.content or "{}"
        # Strip markdown fences if present
        raw = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(raw)
        result = {}
        for name in company_names:
            val = str(parsed.get(name, "neutral")).lower()
            result[name] = val if val in ("positive", "neutral", "negative") else "neutral"
        return result
    except Exception as exc:
        logger.warning("Sentiment scoring failed for query %s: %s", query_id, exc)
        return {name: "neutral" for name in company_names}


async def score_sentiments(
    queries: list[dict],
    mentions: list[dict],
    api_key: str,
) -> list[dict]:
    """
    Attach a 'sentiment' key to each mention dict in-place.
    Makes one LLM call per query that has at least one mention.
    Returns the (mutated) mentions list.
    """
    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    query_response: dict[int, str] = {q["id"]: q.get("response") or "" for q in queries}
    mentions_by_qid: dict[int, list[dict]] = defaultdict(list)
    for m in mentions:
        mentions_by_qid[m["query_id"]].append(m)

    async def _score_query(qid: int, qmentions: list[dict]) -> None:
        names = list({m["company_name"] for m in qmentions})
        sentiment_map = await _score_one(client, query_response.get(qid, ""), names, qid)
        for m in qmentions:
            m["sentiment"] = sentiment_map.get(m["company_name"], "neutral")

    await asyncio.gather(*[
        _score_query(qid, qmentions)
        for qid, qmentions in mentions_by_qid.items()
    ])

    for m in mentions:
        if "sentiment" not in m:
            m["sentiment"] = "neutral"

    return mentions
