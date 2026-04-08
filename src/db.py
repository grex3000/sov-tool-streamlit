from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db(db_path: str) -> None:
    conn = _connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS runs (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at   TEXT    NOT NULL,
            topic        TEXT    NOT NULL,
            period       TEXT,
            notes        TEXT
        );

        CREATE TABLE IF NOT EXISTS queries (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id      INTEGER NOT NULL,
            model_id    TEXT    NOT NULL,
            model_label TEXT    NOT NULL,
            prompt      TEXT    NOT NULL,
            response    TEXT,
            created_at  TEXT    NOT NULL,
            FOREIGN KEY (run_id) REFERENCES runs(id)
        );

        CREATE TABLE IF NOT EXISTS mentions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            query_id     INTEGER NOT NULL,
            company_name TEXT    NOT NULL,
            is_target    INTEGER NOT NULL DEFAULT 0,
            match_type   TEXT,
            excerpt      TEXT,
            FOREIGN KEY (query_id) REFERENCES queries(id)
        );
    """)
    conn.commit()
    conn.close()


def insert_run(
    db_path: str,
    topic: str,
    period: str | None = None,
    notes: str | None = None,
) -> int:
    conn = _connect(db_path)
    cur = conn.execute(
        "INSERT INTO runs (created_at, topic, period, notes) VALUES (?,?,?,?)",
        (datetime.now(timezone.utc).isoformat(), topic, period, notes),
    )
    conn.commit()
    run_id = cur.lastrowid
    conn.close()
    return run_id


def insert_query(
    db_path: str,
    run_id: int,
    model_id: str,
    model_label: str,
    prompt: str,
    response: str | None,
) -> int:
    conn = _connect(db_path)
    cur = conn.execute(
        """INSERT INTO queries (run_id, model_id, model_label, prompt, response, created_at)
           VALUES (?,?,?,?,?,?)""",
        (run_id, model_id, model_label, prompt, response,
         datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    query_id = cur.lastrowid
    conn.close()
    return query_id


def insert_mention(
    db_path: str,
    query_id: int,
    company_name: str,
    is_target: bool,
    match_type: str | None = None,
    excerpt: str | None = None,
) -> None:
    conn = _connect(db_path)
    conn.execute(
        """INSERT INTO mentions (query_id, company_name, is_target, match_type, excerpt)
           VALUES (?,?,?,?,?)""",
        (query_id, company_name, int(is_target), match_type, excerpt),
    )
    conn.commit()
    conn.close()


def get_run(db_path: str, run_id: int) -> dict[str, Any]:
    conn = _connect(db_path)
    row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    conn.close()
    return dict(row) if row else {}


def list_runs(db_path: str) -> list[dict[str, Any]]:
    conn = _connect(db_path)
    rows = conn.execute("SELECT * FROM runs ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_queries_for_run(db_path: str, run_id: int) -> list[dict[str, Any]]:
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT * FROM queries WHERE run_id = ? ORDER BY model_id, id",
        (run_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_mentions_for_run(db_path: str, run_id: int) -> int:
    """Delete all stored mentions for a run. Returns the number of rows deleted."""
    conn = _connect(db_path)
    cur = conn.execute(
        "DELETE FROM mentions WHERE query_id IN (SELECT id FROM queries WHERE run_id = ?)",
        (run_id,),
    )
    conn.commit()
    deleted = cur.rowcount
    conn.close()
    return deleted


def get_mentions_for_run(db_path: str, run_id: int) -> list[dict[str, Any]]:
    """Return all mention rows for all queries in a run, joined with query metadata."""
    conn = _connect(db_path)
    rows = conn.execute(
        """SELECT m.*, q.model_id, q.model_label, q.prompt, q.created_at AS query_date
           FROM mentions m
           JOIN queries q ON m.query_id = q.id
           WHERE q.run_id = ?
           ORDER BY q.model_id, m.company_name""",
        (run_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
