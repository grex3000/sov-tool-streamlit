from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from openai import AsyncOpenAI
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"


@dataclass
class QueryResult:
    model_id: str
    model_label: str
    prompt: str
    response: str | None
    error: str | None = None


def _make_client(api_key: str) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=api_key,
        base_url=_OPENROUTER_BASE,
        default_headers={"HTTP-Referer": "https://sov-tool", "X-Title": "SOV Tool"},
    )


async def _query_one(
    client: AsyncOpenAI,
    model_id: str,
    model_label: str,
    prompt: str,
    semaphore: asyncio.Semaphore,
) -> QueryResult:
    """Execute a single prompt against one model. Returns a QueryResult (never raises)."""
    async with semaphore:
        try:
            resp = await client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=900,
                timeout=45,
            )
            return QueryResult(
                model_id=model_id, model_label=model_label,
                prompt=prompt, response=resp.choices[0].message.content or "",
            )
        except Exception as exc:
            return QueryResult(
                model_id=model_id, model_label=model_label,
                prompt=prompt, response=None, error=str(exc),
            )


async def run_queries(
    prompts: list[str],
    models: list[tuple[str, str]],   # [(model_id, label), ...]
    api_key: str,
    max_concurrent: int = 5,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[QueryResult]:
    """
    Query every model for every prompt concurrently. Returns a flat list of results.

    on_progress(completed, total) is called from within the asyncio loop after each
    request completes — safe to update a shared counter read by the calling thread.
    """
    client    = _make_client(api_key)
    semaphore = asyncio.Semaphore(max_concurrent)
    total     = len(prompts) * len(models)
    completed = [0]

    with Progress(
        TextColumn("  [bold cyan]{task.description}"),
        BarColumn(bar_width=36),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
    ) as rich_progress:
        task_id = rich_progress.add_task("Querying models", total=total)

        async def _run_one(mid: str, label: str, prompt: str) -> QueryResult:
            result = await _query_one(client, mid, label, prompt, semaphore)
            rich_progress.advance(task_id)
            completed[0] += 1
            if on_progress:
                on_progress(completed[0], total)
            return result

        coros = [
            _run_one(mid, label, prompt)
            for mid, label in models
            for prompt in prompts
        ]
        results = await asyncio.gather(*coros)

    return list(results)
