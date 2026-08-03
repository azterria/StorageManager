import datetime
import fnmatch
import logging
import pathlib
import shutil
import subprocess

import src.index

logger = logging.getLogger(__name__)

DEFAULT_CRF = 28

# Top-level event-directory entries worth keeping past debug cleanup and into the
# archive: the recording, structured per-frame data, and small logs. Everything
# else (the per-frame jpg dumps under associators/, camera/, detector/, sanity/,
# search_frames/) is reconstructable from a fresh PlaneTracker run and is the
# actual disk hog, so it's dropped rather than kept indefinitely.
_DEBUG_KEEP_PATTERNS = (
    "*_recording.mp4", "*.csv", "*.jsonl", "*.json", "debug.log", "atc.mp3",
    "complete", "*.jpg",
)


def _keep_in_debug_cleanup(name: str) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in _DEBUG_KEEP_PATTERNS)


def cleanup_debug_data(event_dir: pathlib.Path) -> None:
    """Remove top-level entries of a local event directory that aren't in the
    debug keep-list. Subdirectories of per-frame debug images (associators/,
    camera/, detector/, sanity/, search_frames/) are the intended target; the
    keep-list matches only files, so any subdirectory is removed unconditionally."""
    for entry in event_dir.iterdir():
        if _keep_in_debug_cleanup(entry.name):
            continue
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def _disk_pressure(filespace_root: pathlib.Path, threshold_pct: float) -> bool:
    usage = shutil.disk_usage(filespace_root)
    percent_used = usage.used / usage.total * 100
    return percent_used >= threshold_pct


def select_candidates(
    index: src.index.Index,
    age_days: float,
    disk_threshold_pct: float,
    filespace_root: pathlib.Path,
    batch_size: int,
) -> list[dict]:
    """Age-based primary trigger, with disk pressure pulling forward the oldest
    not-yet-archived events (regardless of age) to fill out the batch."""
    cutoff = (
        datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=age_days)
    ).isoformat(timespec="seconds")
    candidates = index.local_events_older_than(cutoff, limit=batch_size)

    if len(candidates) < batch_size and _disk_pressure(filespace_root, disk_threshold_pct):
        exclude_ids = frozenset(c["id"] for c in candidates)
        extra = index.oldest_local_events(batch_size - len(candidates), exclude_ids=exclude_ids)
        candidates.extend(extra)

    return candidates


def select_debug_cleanup_candidates(
    index: src.index.Index,
    debug_cleanup_days: float,
    batch_size: int,
) -> list[dict]:
    cutoff = (
        datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=debug_cleanup_days)
    ).isoformat(timespec="seconds")
    return index.local_events_for_debug_cleanup(cutoff, limit=batch_size)


def _reencode(src_video: pathlib.Path, dest_video: pathlib.Path, crf: int) -> None:
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(src_video),
        "-vsync", "vfr",
        "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
        str(dest_video),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg re-encode failed (rc={result.returncode}): {result.stderr.strip()}")


def archive_event(row: dict, archive_root: pathlib.Path, crf: int = DEFAULT_CRF) -> str:
    """Reduce and move one event directory into the archive, atomically from the
    index's point of view: the local original is only removed after the fully
    staged replacement has landed at its final destination via `shutil.move`."""
    src_dir = pathlib.Path(row["path"])
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Event directory missing: {src_dir}")

    dest_dir = pathlib.Path(archive_root) / (row["date"] or "unclassified") / row["dir_name"]
    tmp_dest = dest_dir.parent / f"{dest_dir.name}.archiving"
    if tmp_dest.exists():
        shutil.rmtree(tmp_dest)
    tmp_dest.mkdir(parents=True, exist_ok=True)

    video_paths = list(src_dir.glob("*_recording.mp4"))
    video_names = {v.name for v in video_paths}
    for entry in src_dir.iterdir():
        if entry.name in video_names:
            continue
        # Debug-only subdirectories (associators/, camera/, detector/, sanity/,
        # search_frames/) never belong in the archive, whether or not
        # cleanup_debug_data has already run on this event.
        if not _keep_in_debug_cleanup(entry.name):
            continue
        if entry.is_dir():
            shutil.copytree(entry, tmp_dest / entry.name)
        else:
            shutil.copy2(entry, tmp_dest / entry.name)

    for video_path in video_paths:
        _reencode(video_path, tmp_dest / video_path.name, crf)

    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(tmp_dest), str(dest_dir))

    shutil.rmtree(src_dir)
    logger.info("Archived event id=%s path=%s -> %s", row["id"], src_dir, dest_dir)
    return str(dest_dir)


def run_maintenance_cycle(
    index: src.index.Index,
    filespace_root: pathlib.Path,
    archive_root: pathlib.Path,
    age_days: float,
    disk_threshold_pct: float,
    batch_size: int,
    crf: int = DEFAULT_CRF,
    debug_cleanup_days: float | None = None,
) -> dict:
    """Process one bounded batch of archive candidates, plus (if enabled) a batch
    of debug cleanups on still-local events. Idempotent/resumable: a failure on
    one event is logged and skipped, leaving it eligible for the next call rather
    than aborting the whole batch."""
    debug_cleaned = 0
    if debug_cleanup_days is not None:
        debug_candidates = select_debug_cleanup_candidates(index, debug_cleanup_days, batch_size)
        logger.info("Debug cleanup starting: %d candidate(s)", len(debug_candidates))
        for row in debug_candidates:
            try:
                cleanup_debug_data(pathlib.Path(row["path"]))
                index.set_debug_cleaned(row["id"])
                debug_cleaned += 1
            except Exception:
                logger.exception("Failed to clean debug data for event id=%s path=%s", row["id"], row["path"])

    candidates = select_candidates(index, age_days, disk_threshold_pct, filespace_root, batch_size)
    logger.info("Maintenance cycle starting: %d candidate(s)", len(candidates))

    processed = 0
    for row in candidates:
        try:
            archive_path = archive_event(row, archive_root, crf)
            index.set_archived(row["id"], archive_path)
            processed += 1
        except Exception:
            logger.exception("Failed to archive event id=%s path=%s", row["id"], row["path"])

    remaining = len(
        select_candidates(index, age_days, disk_threshold_pct, filespace_root, batch_size=1_000_000)
    )
    logger.info(
        "Maintenance cycle complete: processed=%d remaining=%d debug_cleaned=%d",
        processed, remaining, debug_cleaned,
    )
    return {"status": "ok", "processed": processed, "remaining": remaining, "debug_cleaned": debug_cleaned}
