from __future__ import annotations

import asyncio
import json

from openai import AsyncOpenAI
from rich.console import Console
from rich.prompt import Prompt

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"
_GENERATE_MODEL  = "openai/gpt-5.4"
_CLARIFY_MODEL   = "openai/gpt-5.4"

# ─── Prompt generation system prompt ──────────────────────────────────────────

_GENERATE_SYSTEM = (
    "You generate diverse, realistic queries that people ask AI assistants when researching a specific topic.\n"
    "Vary phrasing, persona, intent, geography, scale, and context.\n"
    'Return a JSON object with a single key "prompts" whose value is an array of strings.'
)

# ─── Clarification system prompt ──────────────────────────────────────────────

_CLARIFY_SYSTEM = """\
You are a research assistant helping someone design focused search queries for an AI Share of Voice scan.
The tool sends queries to AI models and measures how often specific companies or products are mentioned.
Your job: understand their scanning goals through a brief conversation — maximum 2 questions.

Rules:
- Ask one short, concrete question at a time
- If the user's first answer already gives you enough context, skip straight to the BRIEF
- Once you have enough context, respond with exactly: BRIEF: <one sentence>

Example BRIEF outputs:
  BRIEF: Focus on DACH region, CFO persona, mid-market industrial companies considering post-acquisition restructuring
  BRIEF: Global scope, consumer audience comparing electric car brands on reliability and value
  BRIEF: UK market, B2B SaaS buyers evaluating CRM tools for sales teams under 50 people\
"""


# ─── Interactive intent gathering ─────────────────────────────────────────────

async def _run_clarification(topic: str, api_key: str, console: Console) -> str:
    """
    Run a short LLM-driven conversation to extract a focus brief.
    Returns the brief string (empty string on failure).
    """
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=_OPENROUTER_BASE,
        default_headers={"HTTP-Referer": "https://sov-tool", "X-Title": "SOV Tool"},
    )

    messages: list[dict] = [
        {"role": "system", "content": _CLARIFY_SYSTEM},
        {"role": "user",   "content": f'I want to run an AI Share of Voice scan on the topic: "{topic}"'},
    ]

    for turn in range(3):
        resp = await client.chat.completions.create(
            model=_CLARIFY_MODEL,
            messages=messages,
            max_tokens=300,
        )
        assistant_msg = resp.choices[0].message.content.strip()
        messages.append({"role": "assistant", "content": assistant_msg})

        # Check if the LLM produced the brief
        if "BRIEF:" in assistant_msg:
            brief = assistant_msg.split("BRIEF:", 1)[1].strip()
            return brief

        # Print the question and get user answer
        console.print(f"\n  [bold cyan]◆[/bold cyan] {assistant_msg}")
        answer = Prompt.ask("  ").strip()
        if not answer:
            break
        messages.append({"role": "user", "content": answer})

    # Fallback: ask the LLM to summarise what it learned
    messages.append({
        "role": "user",
        "content": "Based on our conversation, produce the BRIEF now.",
    })
    try:
        resp = await client.chat.completions.create(
            model=_CLARIFY_MODEL,
            messages=messages,
            max_tokens=150,
        )
        text = resp.choices[0].message.content.strip()
        if "BRIEF:" in text:
            return text.split("BRIEF:", 1)[1].strip()
        return text
    except Exception:
        return ""


def gather_intent(topic: str, api_key: str) -> tuple[str, list[str]]:
    """
    Interactive session: LLM-driven clarification + optional custom prompts.

    Returns:
        brief         - one-sentence focus description for the generator
        custom_prompts - prompts the user typed in directly
    """
    console = Console()
    console.print()
    console.rule("[bold]Prompt Focus Assistant[/bold]", style="dim")
    console.print(f"  Clarifying goals for [cyan]{topic!r}[/cyan]...\n")

    # Run LLM clarification
    try:
        brief = asyncio.run(_run_clarification(topic, api_key, console))
    except Exception as exc:
        console.print(f"  [yellow]⚠[/yellow]  Clarification failed ({exc}), continuing without brief.\n")
        brief = ""

    if brief:
        console.print(f"\n  [green]✓[/green] Focus: [italic]{brief}[/italic]\n")

    # Custom prompts
    console.print("  Add your own prompts — one per line, empty line to finish (or Enter to skip):")
    custom: list[str] = []
    while True:
        try:
            line = input("  > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            break
        custom.append(line)

    if custom:
        console.print(f"  [green]✓[/green] {len(custom)} custom prompt{'s' if len(custom) != 1 else ''} added\n")
    else:
        console.print()

    return brief, custom


# ─── Prompt generation ────────────────────────────────────────────────────────

async def auto_generate_prompts(
    topic: str,
    count: int,
    api_key: str,
    brief: str = "",
    examples: list[str] | None = None,
) -> list[str]:
    """
    Ask an LLM to generate `count` varied prompts for the given topic.

    brief    - optional focus description from the clarification session
    examples - optional list of user-provided prompts to use as style/angle examples
    """
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=_OPENROUTER_BASE,
        default_headers={"HTTP-Referer": "https://sov-tool", "X-Title": "SOV Tool"},
    )

    focus_block = f"\nFocus/constraints: {brief}" if brief else ""

    examples_block = ""
    if examples:
        ex_lines = "\n".join(f"  - {e}" for e in examples[:5])
        examples_block = (
            f"\nThe user has provided these example prompts — match their angle and style:\n{ex_lines}"
        )

    resp = await client.chat.completions.create(
        model=_GENERATE_MODEL,
        messages=[
            {"role": "system", "content": _GENERATE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f'Topic: "{topic}"{focus_block}\n'
                    f"Generate {count} varied, realistic queries a person might ask an AI assistant about this topic.\n"
                    "Vary along these dimensions:\n"
                    "- Different personas (buyer, researcher, journalist, professional, casual user)\n"
                    "- Different intents (seeking a recommendation, comparing options, discovering what exists)\n"
                    "- Different phrasings (direct question, 'best X', 'who/what is known for Y', 'compare A vs B')\n"
                    "- Different geographies or markets where relevant\n"
                    "- Different scale or context (large/small, premium/budget, B2B/consumer, etc.)\n"
                    "- Different levels of specificity (broad overview vs. narrow niche use case)\n"
                    f"{examples_block}\n"
                    "Make each query sound like something a real person would naturally type or say to an AI."
                ),
            },
        ],
        response_format={"type": "json_object"},
        max_tokens=2000,
    )

    raw = json.loads(resp.choices[0].message.content)

    if isinstance(raw, list):
        prompts = raw
    else:
        prompts = raw.get("prompts") or raw.get("queries") or []
        if not prompts:
            for v in raw.values():
                if isinstance(v, list):
                    prompts = v
                    break

    return [str(p) for p in prompts[:count]]


def load_prompts_from_file(path: str) -> list[str]:
    """Load prompts from a plain-text file — one per line. Lines starting with # are ignored."""
    with open(path) as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.strip().startswith("#")
        ]
