import asyncio
import datetime
import logging
import os
import pathlib
import re

import src.index
import src.parser

logger = logging.getLogger(__name__)

_DATE_DIR_RE = re.compile(r"^\d{8}$")

# Written by the tracker once an event directory's final output (the re-encoded
# recording) has landed. Required for settling in addition to the mtime check below,
# since a directory's mtime only moves when entries are added/removed/renamed — not
# when an already-created file has bytes appended to it.
_COMPLETE_MARKER_NAME = "complete"

# If a directory's mtime has been quiet this long with no tracker-written `complete`
# marker, we self-declare it complete rather than leave it stuck in 'indexing' forever.
# Covers events written before the marker convention existed, and trackers that crashed
# without ever writing one. Deliberately much larger than any realistic settle_seconds
# so we're confident the tracker isn't just still working on it.
_STALE_EVENT_AGE_SECONDS = 86400


def _maybe_declare_stale_complete(
    entry: pathlib.Path, st: os.stat_result, now_ts: float
) -> os.stat_result | None:
    """Write the `complete` marker ourselves if `entry` is stale with no tracker marker.

    Returns the post-write stat result if a marker was written, else None. Only called
    when no marker exists yet, so this never overwrites a tracker-written one.
    """
    age = now_ts - st.st_mtime
    if age < _STALE_EVENT_AGE_SECONDS:
        return None
    try:
        (entry / _COMPLETE_MARKER_NAME).write_text(
            f"declared complete by storage_manager: no tracker marker after {age:.0f}s "
            "of mtime inactivity\n"
        )
        return entry.stat()
    except OSError:
        logger.exception("Failed to write stale-complete marker for %s", entry)
        return None


def scan_once(root: pathlib.Path, index: src.index.Index, settle_seconds: float) -> dict:
    """Walk `{root}/{yyyymmdd}/*` once, upserting each run directory by (dev, ino).

    A directory is "settled" (status='local') once the tracker's `complete` marker
    file is present *and* the directory's mtime hasn't moved for `settle_seconds` —
    otherwise it's still being written to (status='indexing') and excluded from
    archive candidacy. The mtime check is a safety margin after the marker appears
    (belt-and-suspenders against a marker written just before a crash mid-copy), not
    the primary signal. Renamed directories are picked up because the lookup key is
    the inode, not the path.

    If no marker is present and the directory has been quiet for
    `_STALE_EVENT_AGE_SECONDS`, we write the marker ourselves (see
    `_maybe_declare_stale_complete`) so the directory can settle on a later scan. That
    write bumps the directory's own mtime, so it still goes through the normal
    settle_seconds wait afterward rather than settling in this same scan — same
    lifecycle as a tracker-written marker, just self-authored.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    now_ts = now.timestamp()
    summary = {"seen": 0, "settled": 0, "unsettled": 0}

    if not root.is_dir():
        logger.warning("Scan root does not exist: %s", root)
        return summary

    for date_dir in root.iterdir():
        if not date_dir.is_dir() or not _DATE_DIR_RE.match(date_dir.name):
            continue
        try:
            entries = list(date_dir.iterdir())
        except OSError:
            logger.exception("Failed to list %s", date_dir)
            continue
        for entry in entries:
            if not entry.is_dir():
                continue
            try:
                st = entry.stat()
            except OSError:
                logger.exception("Failed to stat %s", entry)
                continue

            summary["seen"] += 1
            has_marker = (entry / _COMPLETE_MARKER_NAME).exists()
            if not has_marker:
                restat = _maybe_declare_stale_complete(entry, st, now_ts)
                if restat is not None:
                    st, has_marker = restat, True

            is_settled = has_marker and (now_ts - st.st_mtime) >= settle_seconds
            summary["settled" if is_settled else "unsettled"] += 1

            parsed = src.parser.parse(entry.name, parent_date=date_dir.name)
            index.upsert_event(
                dev=st.st_dev,
                ino=st.st_ino,
                path=str(entry),
                dir_name=entry.name,
                parsed=parsed,
                mtime=st.st_mtime,
                status="local" if is_settled else "indexing",
                now=now,
            )

    index.set_meta("last_scan_time", now.isoformat())
    return summary


async def run_periodic(
    root: pathlib.Path,
    index: src.index.Index,
    interval_seconds: float,
    settle_seconds: float,
    stop_event: asyncio.Event,
) -> None:
    """Rescan on a timer until `stop_event` is set.

    A running `scan_once` call is a blocking, thread-pooled sqlite3 call that cannot
    be interrupted mid-flight — cancelling this task while a scan is in progress
    would let the caller close the index's connection out from under that thread.
    So shutdown must set `stop_event` and then await this task to completion rather
    than cancelling it, and this loop checks the event only between scans.
    """
    while not stop_event.is_set():
        try:
            summary = await asyncio.to_thread(scan_once, root, index, settle_seconds)
            logger.info(
                "Scan complete: %d dirs (%d settled, %d unsettled)",
                summary["seen"], summary["settled"], summary["unsettled"],
            )
        except Exception:
            logger.exception("Scan failed")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
        except asyncio.TimeoutError:
            pass
