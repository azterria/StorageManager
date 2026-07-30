import datetime
import logging
import pathlib
import sqlite3

import src.parser

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    dev          INTEGER NOT NULL,
    ino          INTEGER NOT NULL,
    path         TEXT NOT NULL,
    dir_name     TEXT NOT NULL,
    date         TEXT,
    time         TEXT,
    timestamp    TEXT,
    registration TEXT,
    event_type   TEXT NOT NULL,
    runway       TEXT,
    classified   INTEGER NOT NULL,
    status       TEXT NOT NULL DEFAULT 'indexing',
    archive_path TEXT,
    mtime        REAL NOT NULL,
    first_seen   TEXT NOT NULL,
    last_seen    TEXT NOT NULL,
    UNIQUE(dev, ino)
);
CREATE INDEX IF NOT EXISTS idx_events_registration ON events(registration);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_status ON events(status);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def _to_timestamp(date: str | None, time: str | None) -> str | None:
    if not date or not time:
        return None
    return f"{date[0:4]}-{date[4:6]}-{date[6:8]}T{time[0:2]}:{time[2:4]}:{time[4:6]}"


class Index:
    """SQLite-backed durable index, keyed by (dev, ino) so renames don't create duplicates."""

    def __init__(self, db_path: str):
        pathlib.Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        logger.info("Index opened at %s", db_path)

    def upsert_event(
        self,
        dev: int,
        ino: int,
        path: str,
        dir_name: str,
        parsed: src.parser.ParsedEvent,
        mtime: float,
        status: str,
        now: datetime.datetime | None = None,
    ) -> int:
        """Insert or update the row for (dev, ino). Never downgrades an 'archived' row."""
        now_iso = (now or datetime.datetime.now(datetime.timezone.utc)).isoformat()
        timestamp = _to_timestamp(parsed.date, parsed.time)
        existing = self._conn.execute(
            "SELECT id, status FROM events WHERE dev = ? AND ino = ?", (dev, ino)
        ).fetchone()
        if existing is None:
            cur = self._conn.execute(
                """
                INSERT INTO events (
                    dev, ino, path, dir_name, date, time, timestamp,
                    registration, event_type, runway, classified,
                    status, mtime, first_seen, last_seen
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    dev, ino, path, dir_name, parsed.date, parsed.time, timestamp,
                    parsed.registration, parsed.event, parsed.runway, int(parsed.classified),
                    status, mtime, now_iso, now_iso,
                ),
            )
            self._conn.commit()
            return cur.lastrowid

        row_id = existing["id"]
        new_status = existing["status"] if existing["status"] == "archived" else status
        self._conn.execute(
            """
            UPDATE events SET
                path = ?, dir_name = ?, date = ?, time = ?, timestamp = ?,
                registration = ?, event_type = ?, runway = ?, classified = ?,
                status = ?, mtime = ?, last_seen = ?
            WHERE id = ?
            """,
            (
                path, dir_name, parsed.date, parsed.time, timestamp,
                parsed.registration, parsed.event, parsed.runway, int(parsed.classified),
                new_status, mtime, now_iso, row_id,
            ),
        )
        self._conn.commit()
        return row_id

    def get_event(self, event_id: int) -> dict | None:
        row = self._conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        return dict(row) if row is not None else None

    def query_events(
        self,
        registration: str | None = None,
        since: str | None = None,
        until: str | None = None,
        timestamp_prefix: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[dict], int]:
        clauses = []
        params: list = []
        if registration:
            clauses.append("registration LIKE ?")
            params.append(f"{registration}%")
        if since:
            clauses.append("timestamp >= ?")
            params.append(since)
        if until:
            clauses.append("timestamp <= ?")
            params.append(until)
        if timestamp_prefix:
            clauses.append("timestamp LIKE ?")
            params.append(f"{timestamp_prefix}%")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        total = self._conn.execute(
            f"SELECT COUNT(*) FROM events {where}", params
        ).fetchone()[0]
        rows = self._conn.execute(
            f"SELECT * FROM events {where} ORDER BY timestamp DESC, id DESC LIMIT ? OFFSET ?",
            [*params, limit, offset],
        ).fetchall()
        return [dict(r) for r in rows], total

    def local_events_older_than(self, cutoff_timestamp: str, limit: int) -> list[dict]:
        rows = self._conn.execute(
            """
            SELECT * FROM events
            WHERE status = 'local' AND timestamp IS NOT NULL AND timestamp < ?
            ORDER BY timestamp ASC
            LIMIT ?
            """,
            (cutoff_timestamp, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def oldest_local_events(self, limit: int, exclude_ids: frozenset[int] = frozenset()) -> list[dict]:
        rows = self._conn.execute(
            """
            SELECT * FROM events
            WHERE status = 'local' AND timestamp IS NOT NULL
            ORDER BY timestamp ASC
            LIMIT ?
            """,
            (limit + len(exclude_ids),),
        ).fetchall()
        result = [dict(r) for r in rows if r["id"] not in exclude_ids]
        return result[:limit]

    def count_local_settled(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM events WHERE status = 'local'"
        ).fetchone()[0]

    def count_by_status(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT status, COUNT(*) AS n FROM events GROUP BY status"
        ).fetchall()
        return {r["status"]: r["n"] for r in rows}

    def count_unclassified(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM events WHERE classified = 0"
        ).fetchone()[0]

    def set_archived(self, event_id: int, archive_path: str, now: datetime.datetime | None = None) -> None:
        now_iso = (now or datetime.datetime.now(datetime.timezone.utc)).isoformat()
        self._conn.execute(
            "UPDATE events SET status = 'archived', archive_path = ?, last_seen = ? WHERE id = ?",
            (archive_path, now_iso, event_id),
        )
        self._conn.commit()

    def get_meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row is not None else None

    def set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
