import asyncio
import contextlib
import datetime
import logging
import os
import pathlib

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


def _event_settle_seconds() -> float:
    return float(os.getenv("EVENT_SETTLE_SECONDS", "120"))


def _scan_interval_seconds() -> float:
    return float(os.getenv("SCAN_INTERVAL_SECONDS", "30"))


@contextlib.asynccontextmanager
async def _lifespan(app: fastapi.FastAPI):
    global _index, _ready, _scan_task, _scan_stop_event

    _load_config()
    index_db_path = os.getenv("INDEX_DB_PATH", "data/index/storage_manager.db")
    _index = src.index.Index(index_db_path)

    settle_seconds = _event_settle_seconds()
    # Run one synchronous scan up front so /health can become ready without waiting
    # a full scan interval.
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
            status="starting", index_size=0, last_scan_time=None, archive_backlog=0
        )
    counts = _index.count_by_status()
    return src.models.StatusResponse(
        status="ok",
        index_size=sum(counts.values()),
        last_scan_time=_index.get_meta("last_scan_time"),
        archive_backlog=counts.get("local", 0),
    )


@app.get("/events", response_model=src.models.EventList)
def list_events(
    registration: str | None = None,
    since: str | None = None,
    until: str | None = None,
    at: str | None = None,
    window_seconds: int | None = None,
    limit: int = 50,
    offset: int = 0,
):
    if at is not None:
        if since is not None or until is not None:
            raise fastapi.HTTPException(
                400, "Use either 'at' + 'window_seconds' or 'since'/'until', not both"
            )
        if window_seconds is None:
            raise fastapi.HTTPException(400, "'window_seconds' is required when 'at' is given")
        try:
            at_dt = datetime.datetime.fromisoformat(at)
        except ValueError as exc:
            raise fastapi.HTTPException(400, f"Invalid 'at' timestamp: {at}") from exc
        delta = datetime.timedelta(seconds=window_seconds)
        since = (at_dt - delta).isoformat()
        until = (at_dt + delta).isoformat()

    rows, total = _index.query_events(
        registration=registration, since=since, until=until, limit=limit, offset=offset
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
        raise fastapi.HTTPException(404, "Event not found")
    candidates = sorted(_resolve_event_dir(row).glob("*_recording.mp4"))
    if not candidates:
        raise fastapi.HTTPException(404, "No recording available for this event")
    return fastapi.responses.FileResponse(candidates[0], media_type="video/mp4")


@app.get("/events/{event_id}/thumbnail")
def get_thumbnail(event_id: int):
    row = _index.get_event(event_id)
    if row is None:
        raise fastapi.HTTPException(404, "Event not found")
    cache_path = _thumbnail_cache_dir / f"{event_id}.jpg"
    try:
        thumbnail_path = src.thumbnail.get_or_create_thumbnail(_resolve_event_dir(row), cache_path)
    except FileNotFoundError as exc:
        raise fastapi.HTTPException(404, "No recording available to generate a thumbnail") from exc
    except RuntimeError as exc:
        raise fastapi.HTTPException(500, str(exc)) from exc
    return fastapi.responses.FileResponse(thumbnail_path, media_type="image/jpeg")


@app.post("/maintenance_cycle", response_model=src.models.MaintenanceCycleResponse)
def maintenance_cycle():
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
