"""Q5 evidence (Week 3): latency of the live API and proof that a click reaches SQLite.

Needs the API running (scripts/run_demo.sh or uvicorn). Sends 100 requests per endpoint,
reports mean / p95 / max in milliseconds, then logs a "like" and shows the SQL row count
before and after.

Run:  python scripts/benchmark_api.py [--api http://localhost:8000]
Out:  reports/latency.json, reports/db_proof.txt
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
KNOWN_USER = "AEJI7MSPXQCTPRTWBSQYK4NU7AOQ"


def timed(fn, n: int = 100) -> dict:
    ms = []
    for _ in range(n):
        t = time.perf_counter()
        r = fn()
        ms.append((time.perf_counter() - t) * 1000)
        r.raise_for_status()
    ms.sort()
    return {"n": n, "mean_ms": round(statistics.mean(ms), 1), "p95_ms": round(ms[int(0.95 * n) - 1], 1),
            "max_ms": round(ms[-1], 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    api = ap.parse_args().api
    health = requests.get(f"{api}/health", timeout=10).json()
    quiz = {"answers": {"moment": "Evening out", "taste": "Sweet", "season": "Winter"}}
    first = requests.post(f"{api}/recommendations/quiz", json=quiz, timeout=10).json()["items"]
    asin = first[0]["parent_asin"]
    sid = f"bench-{int(time.time())}"
    requests.post(f"{api}/log_interaction", json={"user_id": sid, "event": "quiz", "payload": quiz}, timeout=10)
    requests.post(f"{api}/log_interaction", json={"user_id": sid, "item_id": asin, "event": "like"}, timeout=10)
    res = {
        "health": health,
        "GET /recommendations/{user_id} (known user, NLGCL)": timed(
            lambda: requests.get(f"{api}/recommendations/{KNOWN_USER}?k=10", timeout=10)),
        "POST /recommendations/quiz (new visitor)": timed(
            lambda: requests.post(f"{api}/recommendations/quiz", json=quiz, timeout=10)),
        "POST /recommendations/session (quiz + like, read from SQLite)": timed(
            lambda: requests.post(f"{api}/recommendations/session", json={"user_id": sid}, timeout=10)),
        "GET /perfumes/{asin}/similar": timed(lambda: requests.get(f"{api}/perfumes/{asin}/similar", timeout=10)),
        "POST /log_interaction": timed(lambda: requests.post(
            f"{api}/log_interaction", json={"user_id": sid, "item_id": asin, "event": "view"}, timeout=10)),
    }
    (ROOT / "reports" / "latency.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))

    # proof that a "like" sent through the API is stored in SQLite
    db = ROOT / "data" / "scentmatch.db"
    q = "SELECT COUNT(*) FROM interactions WHERE event = 'like'"
    with sqlite3.connect(db) as c:
        before = c.execute(q).fetchone()[0]
    proof_user = f"proof-{int(time.time())}"
    r = requests.post(f"{api}/log_interaction", json={"user_id": proof_user, "item_id": asin, "event": "like"}, timeout=10)
    with sqlite3.connect(db) as c:
        after = c.execute(q).fetchone()[0]
        row = c.execute("SELECT id, user_id, item_id, event, timestamp FROM interactions WHERE user_id = ?",
                        (proof_user,)).fetchone()
    text = (f"$ sqlite3 data/scentmatch.db \"{q}\"   -> {before}\n"
            f"POST /log_interaction {{user_id: {proof_user}, item_id: {asin}, event: like}} -> HTTP {r.status_code} {r.json()}\n"
            f"$ sqlite3 data/scentmatch.db \"{q}\"   -> {after}\n"
            f"new row: {row}\n")
    (ROOT / "reports" / "db_proof.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
