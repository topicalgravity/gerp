"""Persistent store for account requests and approvals.

SQLite on a Render persistent disk (GERP_DATA_DIR, mounted at /var/data in
prod) — the Stage-3 storage brought forward so account approval is a one-click
link in the owner's notification email instead of a Render env-var edit.
Stdlib sqlite3 only; no new dependency.

One table doubles as the request log and the approval list: a row with
approved_at set is an approved account. The env lists in auth.py
(GERP_APPROVED_EMAILS / GERP_FRONTIER_ALLOWLIST) remain as bootstrap/override.

The DB path is resolved at call time (not import) so tests can point
GERP_DATA_DIR at a temp dir. Connections are short-lived per call: one gunicorn
worker with 8 threads doesn't need pooling, and WAL mode keeps readers and the
occasional write from blocking each other.
"""

from __future__ import annotations

import datetime as _dt
import os
import sqlite3
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS account_requests (
    email        TEXT PRIMARY KEY,
    first_name   TEXT,
    last_name    TEXT,
    company      TEXT,
    requested_at TEXT NOT NULL,
    approved_at  TEXT
);
"""


def _db_path() -> Path:
    data_dir = Path(os.environ.get("GERP_DATA_DIR",
                                   Path(__file__).parent / "data"))
    return data_dir / "gerp.db"


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(_SCHEMA)
    return conn


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def init_db() -> None:
    """Create the data dir + schema. Called once at app startup; every other
    function also ensures the schema, so this is a fail-fast nicety."""
    _connect().close()


def record_request(first: str, last: str, email: str, company: str) -> None:
    """Upsert an account request. A re-request refreshes the details but never
    clears an existing approval."""
    email = email.strip().lower()
    with _connect() as conn:
        conn.execute(
            """INSERT INTO account_requests
                   (email, first_name, last_name, company, requested_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(email) DO UPDATE SET
                   first_name = excluded.first_name,
                   last_name = excluded.last_name,
                   company = excluded.company,
                   requested_at = excluded.requested_at""",
            (email, first, last, company, _now()))


def approve(email: str) -> str:
    """Mark an email approved. Returns "approved" (newly), "already", or
    "unknown" — the last still inserts+approves, so the owner can approve an
    address that never came through the form."""
    email = email.strip().lower()
    with _connect() as conn:
        row = conn.execute(
            "SELECT approved_at FROM account_requests WHERE email = ?",
            (email,)).fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO account_requests (email, requested_at, approved_at) "
                "VALUES (?, ?, ?)", (email, _now(), _now()))
            return "unknown"
        if row[0]:
            return "already"
        conn.execute(
            "UPDATE account_requests SET approved_at = ? WHERE email = ?",
            (_now(), email))
        return "approved"


def is_approved(email: str) -> bool:
    with _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM account_requests "
            "WHERE email = ? AND approved_at IS NOT NULL",
            (email.strip().lower(),)).fetchone()
    return row is not None


def count_approved() -> int:
    with _connect() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM account_requests "
            "WHERE approved_at IS NOT NULL").fetchone()[0]
