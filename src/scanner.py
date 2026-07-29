import asyncio
import datetime
import logging
import pathlib
import re

import src.index
import src.parser

logger = logging.getLogger(__name__)

_DATE_DIR_RE = re.compile(r"^\d{8}$")


def scan_once(root: pathlib.Path, index: src.index.Index, settle_seconds: float) -> dict:
    """Walk `{root}/{yyyymmdd}/*` once, upserting each run directory by (dev, ino).

    A directory is "settled" (status='local') once its mtime hasn't moved for
    `settle_seconds` — otherwise it's still being written to (status='indexing') and
    excluded from archive candidacy. Renamed directories are picked up because the
    lookup key is the inode, not the path.
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
            is_settled = (now_ts - st.st_mtime) >= settle_seconds
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
