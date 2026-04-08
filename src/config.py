from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()


@dataclass
class CompanyEntry:
    name: str
    aliases: list[str]
    is_target: bool = False


@dataclass
class ModelEntry:
    id: str
    label: str


@dataclass
class AppConfig:
    target: CompanyEntry
    competitors: list[CompanyEntry]
    models: list[ModelEntry]
    queries_per_topic: int
    max_concurrent_requests: int
    report_template: str
    reports_dir: str
    db_path: str
    api_key: str

    @property
    def all_companies(self) -> list[CompanyEntry]:
        """Target first, then competitors."""
        return [self.target] + self.competitors


def load_config(path: str = "config.yaml") -> AppConfig:
    raw = yaml.safe_load(Path(path).read_text())

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Copy .env.example to .env and fill in your key."
        )

    target_raw = raw["target"]
    target = CompanyEntry(
        name=target_raw["name"],
        aliases=target_raw.get("aliases", []),
        is_target=True,
    )

    competitors = [
        CompanyEntry(name=c["name"], aliases=c.get("aliases", []), is_target=False)
        for c in raw.get("competitors", [])
    ]

    models = [ModelEntry(id=m["id"], label=m["label"]) for m in raw["models"]]

    return AppConfig(
        target=target,
        competitors=competitors,
        models=models,
        queries_per_topic=raw.get("queries_per_topic", 20),
        max_concurrent_requests=raw.get("max_concurrent_requests", 5),
        report_template=raw.get("report_template", "templates/report.html.j2"),
        reports_dir=raw.get("reports_dir", "reports"),
        db_path=raw.get("db_path", "data/sov.db"),
        api_key=api_key,
    )
