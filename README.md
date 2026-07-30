# StorageManager

A read-only event API and archive maintenance service for PlaneTracker
recordings. StorageManager watches a filespace of dated event directories,
indexes them into SQLite, serves event metadata/video/thumbnails over HTTP,
and periodically re-encodes and relocates aged events into an archive to
free up local storage.

## How it works

- **Scanner** (`src/scanner.py`) walks `{FILESPACE_ROOT}/{yyyymmdd}/*` on a
  timer, upserting each run directory into the index keyed by `(dev, ino)`
  so renames don't create duplicates. A directory is "settled" (`local`)
  once the tracker's `complete` marker file is present *and* its mtime
  hasn't moved for `EVENT_SETTLE_SECONDS`; otherwise it's still being
  written (`indexing`) and excluded from archiving.
- **Parser** (`src/parser.py`) extracts registration, event type, runway,
  and timestamp from PlaneTracker's directory naming convention. Names it
  doesn't recognize are kept as `unclassified` rather than rejected.
- **Index** (`src/index.py`) is a durable SQLite-backed store of event
  metadata and status (`indexing` / `local` / `archived`).
- **Archive** (`src/archive.py`) runs maintenance cycles that select aged
  or (under disk pressure) oldest `local` events, re-encode their videos
  with `ffmpeg`, and move the result into `ARCHIVE_ROOT`. This is
  idempotent/resumable: a failure on one event is logged and skipped.
- **Thumbnail** (`src/thumbnail.py`) generates and caches a JPEG frame per
  event, picked from the highest-confidence accepted detection where
  available.
- **Service** (`src/service.py`) is the FastAPI app tying it all together.

## API

- `GET /health` — liveness/readiness probe.
- `GET /status` — index size, last scan time, archive backlog, count of
  unclassified events.
- `GET /events` — list events, filterable by `registration` and by time
  fields (`year`, `month`, `day`, `hour`, `minute`, `second`, given from
  `year` downward with no gaps), optionally widened with `window_seconds`
  around a fully-specified timestamp. Supports `limit`/`offset`.
- `GET /events/{id}` — a single event's metadata.
- `GET /events/{id}/video` — the event's recording (MP4).
- `GET /events/{id}/thumbnail` — a generated/cached thumbnail (JPEG).
- `POST /maintenance_cycle` — trigger one bounded batch of archiving.

## Configuration

Set via environment variables:

| Variable | Default | Description |
|---|---|---|
| `FILESPACE_ROOT` | *(required)* | Root of live event directories |
| `ARCHIVE_ROOT` | *(required)* | Root to move archived events into |
| `STORAGE_MANAGER_PORT` | `8020` | HTTP port |
| `INDEX_DB_PATH` | `data/index/storage_manager.db` | SQLite index location |
| `ARCHIVE_AGE_DAYS` | `30` | Age threshold to archive an event |
| `ARCHIVE_DISK_THRESHOLD_PCT` | `85` | Disk usage that pulls forward older events |
| `MAINTENANCE_BATCH_SIZE` | `20` | Max events per maintenance cycle |
| `ARCHIVE_CRF` | `28` | ffmpeg CRF used when re-encoding |
| `EVENT_SETTLE_SECONDS` | `120` | Quiet time before an event is considered settled |
| `SCAN_INTERVAL_SECONDS` | `30` | Delay between periodic scans |

## Running

### Docker (recommended)

```bash
FILESPACE_ROOT=/path/to/filespace ARCHIVE_ROOT=/path/to/archive \
    scripts/build_and_run.sh
```

This builds the image and runs it, mounting `FILESPACE_ROOT` and
`ARCHIVE_ROOT` as volumes. See `scripts/platform/run_x86_64.sh` for the
full set of env vars forwarded into the container.

### Local development

```bash
scripts/environment/build_environment.sh   # creates a conda env "StorageManager"
conda activate StorageManager
FILESPACE_ROOT=/path/to/filespace ARCHIVE_ROOT=/path/to/archive \
    python3 src/service.py
```

Requires `ffmpeg` on `PATH` for archiving.

## Testing

```bash
pytest
```
