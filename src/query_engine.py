from __future__ import annotations

import asyncio
from dataclasses import dataclass

from openai import AsyncOpenAI
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
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
    progress: Progress,
    task_id: TaskID,
) -> QueryResult:
    async with semaphore:
        try:
            resp = await client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=900,
                timeout=45,
            )
            text = resp.choices[0].message.content or ""
            result = QueryResult(
                model_id=model_id, model_label=model_label,
                prompt=prompt, response=text,
            )
        except Exception as exc:
            result = QueryResult(
                model_id=model_id, model_label=model_label,
                prompt=prompt, response=None, error=str(exc),
            )
        finally:
            progress.advance(task_id)
        return result


async def run_queries(
    prompts: list[str],
    models: list[tuple[str, str]],   # [(model_id, label), ...]
    api_key: str,
    max_concurrent: int = 5,
) -> list[QueryResult]:
    """Query every model for every prompt concurrently. Returns flat list of results."""
    client = _make_client(api_key)
    semaphore = asyncio.Semaphore(max_concurrent)
    total = len(prompts) * len(models)

    with Progress(
        TextColumn("  [bold cyan]{task.description}"),
        BarColumn(bar_width=36),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        TimeElapsedColumn(),
    ) as progress:
        task_id = progress.add_task("Querying models", total=total)
        coros = [
            _query_one(client, mid, label, prompt, semaphore, progress, task_id)
            for mid, label in models
            for prompt in prompts
        ]
        results = await asyncio.gather(*coros)

    return list(results)
