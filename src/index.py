import datetime
import logging
import pathlib
import sqlite3
import threading

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
    debug_cleaned INTEGER NOT NULL DEFAULT 0,
    failure_stage TEXT,
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

CREATE TABLE IF NOT EXISTS pending_renames (
    event_id         INTEGER PRIMARY KEY,
    new_registration TEXT NOT NULL,
    requested_at     TEXT NOT NULL,
    FOREIGN KEY (event_id) REFERENCES events(id)
);
"""


def _to_timestamp(date: str | None, time: str | None) -> str | None:
    if not date or not time:
        return None
    return f"{date[0:4]}-{date[4:6]}-{date[6:8]}T{time[0:2]}:{time[2:4]}:{time[4:6]}"


class Index:
    """SQLite-backed durable index, keyed by (dev, ino) so renames don't create duplicates.

    The connection is shared across threads (the periodic scan runs in a thread-pool
    thread; FastAPI request handlers run in their own thread-pool threads too), and
    sqlite3 connections aren't safe for concurrent use from multiple threads even with
    check_same_thread=False — that flag only disables the same-thread check, it doesn't
    add synchronization. `_lock` serializes every method below so two threads never
    interleave statements on the same connection.
    """

    def __init__(self, db_path: str):
        self._lock = threading.Lock()
        pathlib.Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._migrate_debug_cleaned_column()
        self._migrate_failure_stage_column()
        self._conn.commit()
        logger.info("Index opened at %s", db_path)

    def _migrate_debug_cleaned_column(self) -> None:
        """CREATE TABLE IF NOT EXISTS doesn't add columns to a pre-existing table,
        so DBs created before debug_cleaned existed need it added explicitly. Called
        only from __init__, before any other thread can see this Index, so no lock
        needed here."""
        columns = {row["name"] for row in self._conn.execute("PRAGMA table_info(events)")}
        if "debug_cleaned" not in columns:
            self._conn.execute(
                "ALTER TABLE events ADD COLUMN debug_cleaned INTEGER NOT NULL DEFAULT 0"
            )

    def _migrate_failure_stage_column(self) -> None:
        """Same rationale as _migrate_debug_cleaned_column, for failure_stage."""
        columns = {row["name"] for row in self._conn.execute("PRAGMA table_info(events)")}
        if "failure_stage" not in columns:
            self._conn.execute("ALTER TABLE events ADD COLUMN failure_stage TEXT")

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
        """Insert or update the row for (dev, ino). Never downgrades an 'archived' or
        'failed' row — a 'failed' event's directory is left untouched on disk, so
        without this the next scan would see what still looks like a plain local/
        indexing directory and silently flip status back, undoing the failure mark
        and re-queuing the event for another doomed attempt next cycle."""
        now_iso = (now or datetime.datetime.now(datetime.timezone.utc)).isoformat()
        timestamp = _to_timestamp(parsed.date, parsed.time)
        with self._lock:
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
            new_status = existing["status"] if existing["status"] in ("archived", "failed") else status
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

    def get_known_states(self) -> dict[tuple[int, int], dict]:
        """Return {(dev, ino): {status, mtime, path}} for every indexed row in one query.

        Lets the scanner skip re-parsing and re-upserting directories that haven't
        changed since the last scan, instead of writing (and fsync-committing) every
        row on every cycle regardless of whether anything moved.
        """
        with self._lock:
            rows = self._conn.execute("SELECT dev, ino, status, mtime, path FROM events").fetchall()
        return {(r["dev"], r["ino"]): {"status": r["status"], "mtime": r["mtime"], "path": r["path"]} for r in rows}

    def get_event(self, event_id: int) -> dict | None:
        with self._lock:
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

        with self._lock:
            total = self._conn.execute(
                f"SELECT COUNT(*) FROM events {where}", params
            ).fetchone()[0]
            rows = self._conn.execute(
                f"SELECT * FROM events {where} ORDER BY timestamp DESC, id DESC LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
        return [dict(r) for r in rows], total

    def local_events_older_than(self, cutoff_timestamp: str, limit: int) -> list[dict]:
        with self._lock:
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
        with self._lock:
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

    def local_events_for_debug_cleanup(self, cutoff_timestamp: str, limit: int) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM events
                WHERE status = 'local' AND debug_cleaned = 0
                    AND timestamp IS NOT NULL AND timestamp < ?
                ORDER BY timestamp ASC
                LIMIT ?
                """,
                (cutoff_timestamp, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def set_debug_cleaned(self, event_id: int) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE events SET debug_cleaned = 1 WHERE id = ?", (event_id,)
            )
            self._conn.commit()

    def local_events_by_type_older_than(self, event_type: str, cutoff_timestamp: str, limit: int) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM events
                WHERE status = 'local' AND event_type = ? AND timestamp IS NOT NULL AND timestamp < ?
                ORDER BY timestamp ASC
                LIMIT ?
                """,
                (event_type, cutoff_timestamp, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_event(self, event_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM events WHERE id = ?", (event_id,))
            self._conn.commit()

    def count_local_settled(self) -> int:
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM events WHERE status = 'local'"
            ).fetchone()[0]

    def count_by_status(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM events GROUP BY status"
            ).fetchall()
        return {r["status"]: r["n"] for r in rows}

    def count_unclassified(self) -> int:
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM events WHERE classified = 0"
            ).fetchone()[0]

    def find_events_by_identity(self, event_type: str, runway: str, date: str, time: str) -> list[dict]:
        """Look up events by the fields a directory name is built from, other than
        registration — used by /rename to locate an event without trusting whatever
        (possibly wrong) registration is currently on file."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM events WHERE event_type = ? AND runway = ? AND date = ? AND time = ?",
                (event_type, runway, date, time),
            ).fetchall()
        return [dict(r) for r in rows]

    def rename_event(self, event_id: int, new_registration: str, new_dir_name: str, new_path: str,
                      now: datetime.datetime | None = None) -> None:
        """Update the row after a physical directory rename. dev/ino (the identity key)
        are unaffected by an in-place rename, so only the renamed fields need updating."""
        now_iso = (now or datetime.datetime.now(datetime.timezone.utc)).isoformat()
        with self._lock:
            self._conn.execute(
                "UPDATE events SET registration = ?, dir_name = ?, path = ?, last_seen = ? WHERE id = ?",
                (new_registration, new_dir_name, new_path, now_iso, event_id),
            )
            self._conn.commit()

    def queue_rename(self, event_id: int, new_registration: str, now: datetime.datetime | None = None) -> None:
        """Record a rename request for an event that's still 'indexing', to be applied
        once it settles to 'local' (see rename.apply_pending_renames). A later call for
        the same event_id overwrites the pending registration rather than stacking."""
        now_iso = (now or datetime.datetime.now(datetime.timezone.utc)).isoformat()
        with self._lock:
            self._conn.execute(
                "INSERT INTO pending_renames (event_id, new_registration, requested_at) VALUES (?, ?, ?) "
                "ON CONFLICT(event_id) DO UPDATE SET "
                "new_registration = excluded.new_registration, requested_at = excluded.requested_at",
                (event_id, new_registration, now_iso),
            )
            self._conn.commit()

    def get_pending_renames_for_local_events(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT pending_renames.event_id AS event_id,
                       pending_renames.new_registration AS new_registration
                FROM pending_renames
                JOIN events ON events.id = pending_renames.event_id
                WHERE events.status = 'local'
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def clear_pending_rename(self, event_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM pending_renames WHERE event_id = ?", (event_id,))
            self._conn.commit()

    def set_archived(self, event_id: int, archive_path: str, now: datetime.datetime | None = None) -> None:
        now_iso = (now or datetime.datetime.now(datetime.timezone.utc)).isoformat()
        with self._lock:
            self._conn.execute(
                "UPDATE events SET status = 'archived', archive_path = ?, last_seen = ? WHERE id = ?",
                (archive_path, now_iso, event_id),
            )
            self._conn.commit()

    def set_failed(self, event_id: int, stage: str, now: datetime.datetime | None = None) -> None:
        """Mark an event as permanently failed at a given maintenance stage ('archive'
        or 'debug_cleanup'). Sticky like 'archived' (see upsert_event) — recovery is
        manual."""
        now_iso = (now or datetime.datetime.now(datetime.timezone.utc)).isoformat()
        with self._lock:
            self._conn.execute(
                "UPDATE events SET status = 'failed', failure_stage = ?, last_seen = ? WHERE id = ?",
                (stage, now_iso, event_id),
            )
            self._conn.commit()

    def get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row is not None else None

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
