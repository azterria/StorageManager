import asyncio
import contextlib
import datetime
import logging
import os
import pathlib
import shutil

import fastapi
import fastapi.responses
import uvicorn

import src.archive
import src.index
import src.logging_config
import src.models
import src.scanner
import src.thumbnail

src.logging_config.setup()

logger = logging.getLogger(__name__)

PORT = int(os.getenv("STORAGE_MANAGER_PORT", "8020"))

_index: src.index.Index | None = None
_ready = False
_scan_task: asyncio.Task | None = None
_scan_stop_event: asyncio.Event | None = None

_filespace_root: pathlib.Path
_archive_root: pathlib.Path
_archive_age_days: float
_archive_disk_threshold_pct: float
_maintenance_batch_size: int
_archive_crf: int
_thumbnail_cache_dir: pathlib.Path


def _load_config() -> None:
    """Read env-based config fresh at startup (rather than at import time) so tests
    can point each run at its own tmp_path fixtures via monkeypatched env vars."""
    global _filespace_root, _archive_root, _archive_age_days, _archive_disk_threshold_pct
    global _maintenance_batch_size, _archive_crf, _thumbnail_cache_dir

    _filespace_root = pathlib.Path(os.environ["FILESPACE_ROOT"])
    _archive_root = pathlib.Path(os.environ["ARCHIVE_ROOT"])
    _archive_age_days = float(os.getenv("ARCHIVE_AGE_DAYS", "30"))
    _archive_disk_threshold_pct = float(os.getenv("ARCHIVE_DISK_THRESHOLD_PCT", "85"))
    _maintenance_batch_size = int(os.getenv("MAINTENANCE_BATCH_SIZE", "20"))
    _archive_crf = int(os.getenv("ARCHIVE_CRF", str(src.archive.DEFAULT_CRF)))

    index_db_path = pathlib.Path(os.getenv("INDEX_DB_PATH", "data/index/storage_manager.db"))
    _thumbnail_cache_dir = index_db_path.parent / "thumbnails"


def _maybe_fresh_start() -> None:
    """If FRESH_START=1, wipe the index DB and thumbnail cache before anything
    opens them, so the service starts as if from a clean deploy. Must run after
    _load_config() (needs _thumbnail_cache_dir) and before src.index.Index() opens
    the DB file."""
    if os.getenv("FRESH_START") != "1":
        return

    index_db_path = pathlib.Path(os.getenv("INDEX_DB_PATH", "data/index/storage_manager.db"))
    if index_db_path.exists():
        index_db_path.unlink()
        logger.info("FRESH_START: deleted index DB at %s", index_db_path)

    if _thumbnail_cache_dir.exists():
        shutil.rmtree(_thumbnail_cache_dir)
        logger.info("FRESH_START: deleted thumbnail cache at %s", _thumbnail_cache_dir)


def _event_settle_seconds() -> float:
    return float(os.getenv("EVENT_SETTLE_SECONDS", "120"))


def _scan_interval_seconds() -> float:
    return float(os.getenv("SCAN_INTERVAL_SECONDS", "30"))


@contextlib.asynccontextmanager
async def _lifespan(app: fastapi.FastAPI):
    global _index, _ready, _scan_task, _scan_stop_event

    _load_config()
    _maybe_fresh_start()
    index_db_path = os.getenv("INDEX_DB_PATH", "data/index/storage_manager.db")
    db_existed = pathlib.Path(index_db_path).exists()
    _index = src.index.Index(index_db_path)

    settle_seconds = _event_settle_seconds()
    if db_existed:
        # DB already has data to serve; let the periodic scan loop (which scans
        # immediately on its first iteration) catch up in the background instead
        # of blocking /health on a full scan.
        _ready = True
    else:
        # No DB to fall back on yet, so /health must wait for a full scan before
        # there's anything to report.
        await asyncio.to_thread(src.scanner.scan_once, _filespace_root, _index, settle_seconds)
        _ready = True

    _scan_stop_event = asyncio.Event()
    _scan_task = asyncio.create_task(
        src.scanner.run_periodic(
            _filespace_root, _index, _scan_interval_seconds(), settle_seconds, _scan_stop_event
        )
    )
    logger.info("StorageManager ready: filespace=%s archive=%s", _filespace_root, _archive_root)
    yield

    _ready = False
    # Signal-and-await rather than cancel(): a scan may be mid-flight in the thread
    # pool, and cancelling the task wouldn't stop that thread before we close the
    # index's connection out from under it.
    _scan_stop_event.set()
    await _scan_task
    _index.close()


app = fastapi.FastAPI(lifespan=_lifespan)


def _row_to_summary(row: dict) -> src.models.EventSummary:
    return src.models.EventSummary(
        id=row["id"],
        date=row["date"],
        time=row["time"],
        registration=row["registration"],
        event=row["event_type"],
        runway=row["runway"],
        status=row["status"],
        in_progress=row["status"] == "indexing",
    )


@app.get("/health")
def health(response: fastapi.Response):
    if not _ready:
        response.status_code = 503
        return {"status": "starting"}
    return {"status": "ok"}


@app.get("/status", response_model=src.models.StatusResponse)
def status(response: fastapi.Response):
    if not _ready:
        response.status_code = 503
        return src.models.StatusResponse(
            status="starting", index_size=0, last_scan_time=None, archive_backlog=0,
            unclassified_count=0,
        )
    counts = _index.count_by_status()
    return src.models.StatusResponse(
        status="ok",
        index_size=sum(counts.values()),
        last_scan_time=_index.get_meta("last_scan_time"),
        archive_backlog=counts.get("local", 0),
        unclassified_count=_index.count_unclassified(),
    )


_TIME_FIELDS = ("year", "month", "day", "hour", "minute", "second")


def _build_timestamp_prefix(year: int, month: int | None, day: int | None,
                             hour: int | None, minute: int | None, second: int | None) -> str:
    prefix = f"{year:04d}"
    if month is None:
        return prefix
    prefix += f"-{month:02d}"
    if day is None:
        return prefix
    prefix += f"-{day:02d}"
    if hour is None:
        return prefix
    prefix += f"T{hour:02d}"
    if minute is None:
        return prefix
    prefix += f":{minute:02d}"
    if second is None:
        return prefix
    return prefix + f":{second:02d}"


@app.get("/events", response_model=src.models.EventList)
def list_events(
    registration: str | None = None,
    year: int | None = None,
    month: int | None = fastapi.Query(None, ge=1, le=12),
    day: int | None = fastapi.Query(None, ge=1, le=31),
    hour: int | None = fastapi.Query(None, ge=0, le=23),
    minute: int | None = fastapi.Query(None, ge=0, le=59),
    second: int | None = fastapi.Query(None, ge=0, le=59),
    window_seconds: int | None = None,
    limit: int = 50,
    offset: int = 0,
):
    values = (year, month, day, hour, minute, second)
    given = [v is not None for v in values]
    if any(given):
        depth = max(i for i, g in enumerate(given) if g)
        if not all(given[: depth + 1]):
            missing = _TIME_FIELDS[next(i for i, g in enumerate(given) if not g)]
            raise fastapi.HTTPException(
                400, f"Missing '{missing}': time fields must be given from 'year' downward with no gaps"
            )

    if window_seconds is not None and second is None:
        raise fastapi.HTTPException(400, "'window_seconds' requires 'second' to be given")

    since = until = timestamp_prefix = None
    if year is not None:
        if second is not None and window_seconds is not None:
            try:
                at_dt = datetime.datetime(year, month, day, hour, minute, second)
            except ValueError as exc:
                raise fastapi.HTTPException(400, f"Invalid timestamp: {exc}") from exc
            delta = datetime.timedelta(seconds=window_seconds)
            since = (at_dt - delta).isoformat()
            until = (at_dt + delta).isoformat()
        else:
            timestamp_prefix = _build_timestamp_prefix(year, month, day, hour, minute, second)

    rows, total = _index.query_events(
        registration=registration,
        since=since,
        until=until,
        timestamp_prefix=timestamp_prefix,
        limit=limit,
        offset=offset,
    )
    return src.models.EventList(
        events=[_row_to_summary(r) for r in rows], total=total, limit=limit, offset=offset
    )


@app.get("/events/{event_id}", response_model=src.models.EventSummary)
def get_event(event_id: int):
    row = _index.get_event(event_id)
    if row is None:
        raise fastapi.HTTPException(404, "Event not found")
    return _row_to_summary(row)


def _resolve_event_dir(row: dict) -> pathlib.Path:
    return pathlib.Path(row["archive_path"] or row["path"])


@app.get("/events/{event_id}/video")
def get_video(event_id: int):
    row = _index.get_event(event_id)
    if row is None:
        logger.warning("Video requested for unknown event id=%s", event_id)
        raise fastapi.HTTPException(404, "Event not found")
    candidates = sorted(_resolve_event_dir(row).glob("*_recording.mp4"))
    if not candidates:
        logger.warning("No recording available for event id=%s", event_id)
        raise fastapi.HTTPException(404, "No recording available for this event")
    return fastapi.responses.FileResponse(candidates[0], media_type="video/mp4")


@app.get("/events/{event_id}/thumbnail")
def get_thumbnail(event_id: int):
    row = _index.get_event(event_id)
    if row is None:
        logger.warning("Thumbnail requested for unknown event id=%s", event_id)
        raise fastapi.HTTPException(404, "Event not found")
    cache_path = _thumbnail_cache_dir / f"{event_id}.jpg"
    try:
        thumbnail_path = src.thumbnail.get_or_create_thumbnail(_resolve_event_dir(row), cache_path)
    except FileNotFoundError as exc:
        raise fastapi.HTTPException(404, "No recording available to generate a thumbnail") from exc
    except RuntimeError as exc:
        logger.exception("Thumbnail generation failed for event id=%s", event_id)
        raise fastapi.HTTPException(500, str(exc)) from exc
    return fastapi.responses.FileResponse(thumbnail_path, media_type="image/jpeg")


@app.post("/maintenance_cycle", response_model=src.models.MaintenanceCycleResponse)
def maintenance_cycle():
    logger.info("Maintenance cycle triggered via API")
    result = src.archive.run_maintenance_cycle(
        _index,
        _filespace_root,
        _archive_root,
        _archive_age_days,
        _archive_disk_threshold_pct,
        _maintenance_batch_size,
        _archive_crf,
    )
    return src.models.MaintenanceCycleResponse(**result)


if __name__ == "__main__":
    uvicorn.run("src.service:app", host="0.0.0.0", port=PORT, reload=False)
