"""SQLite storage: perfume catalogue + interactions captured by POST /log_interaction."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS perfumes (
    item_idx          INTEGER PRIMARY KEY,
    parent_asin       TEXT UNIQUE NOT NULL,
    title             TEXT,
    store             TEXT,
    notes             TEXT,
    image             TEXT,
    train_popularity  INTEGER
);
CREATE TABLE IF NOT EXISTS interactions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    TEXT NOT NULL,
    item_id    TEXT,               -- parent_asin; NULL for quiz events
    event      TEXT NOT NULL CHECK (event IN ('quiz','click','like','not_for_me','view')),
    payload    TEXT,               -- JSON (quiz answers, rank position, ...)
    timestamp  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX IF NOT EXISTS idx_interactions_user ON interactions(user_id);
"""


@contextmanager
def connect(db_path: Path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)


def load_catalogue(db_path: Path, items_parquet: Path) -> int:
    items = pd.read_parquet(items_parquet)
    cols = ["item_idx", "parent_asin", "title", "store", "notes", "image", "train_popularity"]
    with connect(db_path) as conn:
        conn.execute("DELETE FROM perfumes")
        conn.executemany(f"INSERT INTO perfumes ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                         items[cols].itertuples(index=False, name=None))
    return len(items)


def log_interaction(db_path: Path, user_id: str, event: str, item_id: str | None = None,
                    payload: str | None = None) -> int:
    with connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO interactions (user_id, item_id, event, payload) VALUES (?,?,?,?)",
            (user_id, item_id, event, payload))
        return int(cur.lastrowid)


def get_perfumes(db_path: Path, item_idx: list[int]) -> dict[int, dict]:
    if not item_idx:
        return {}
    with connect(db_path) as conn:
        rows = conn.execute(
            f"SELECT * FROM perfumes WHERE item_idx IN ({','.join('?' * len(item_idx))})",
            [int(i) for i in item_idx]).fetchall()
    return {r["item_idx"]: dict(r) for r in rows}


def count_interactions(db_path: Path) -> int:
    with connect(db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
