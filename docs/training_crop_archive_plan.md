# Plan: Training-data crop pipeline for archived events

Status: not started, ready to implement.

## Context

`archive_event` (`src/archive.py`) already re-encodes `*_recording.mp4` at a
lower CRF during archiving and keeps the per-event `detections.jsonl` plus a
sparse (~5-15fps) sequence of full-frame `.jpg` stills that PlaneTracker wrote
while steering the camera (one still per `detections.jsonl` line, filename
suffix = ms since tracking started).

The goal is to keep training data for ostrack (a tracker) without keeping the
uncompressed stills forever: replace the raw jpg sequence with a single
720p-cropped mp4 (each still cropped to a fixed-size window centered on that
frame's tracked position) plus a `detections_cropped.jsonl` whose boxes are
translated into the new crop's coordinate space and indexed to match the new
video's frames. `detections.jsonl` (full-frame boxes) is kept as-is — it's
lightweight and already survives archiving.

This was worked through with Matthew across several rounds (verified against
a real example event, `N6884H_touch_and_go_24_20260724_183304`, including a
side-by-side CRF-28 thumbnail/tail-number legibility check). Decisions locked
in:

1. Crop and assemble in a temp staging dir, move the finished artifacts into
   place, then delete the originals — reusing `archive_event`'s existing
   stage-then-swap-then-delete pattern (nothing new needed for atomicity). On
   any failure — this step, any other step of `archive_event`, or a
   `cleanup_debug_data` failure — the event stops being retried
   automatically rather than silently re-attempting forever: write an empty
   marker file into the still-in-place event directory (`failed_archive` for
   an `archive_event` failure, `failed_debug_cleanup` for a debug-cleanup
   failure), set the index row's `status` to `failed`, and log at WARNING.
   This replaces the bare `logger.exception` in both of
   `run_maintenance_cycle`'s existing except-blocks (debug cleanup and
   archive) — see "Failure surfacing" below. `tilt_calibration` deletion
   failures are unaffected (different failure mode — delete, not create —
   and not mentioned as in scope).
2. Frame ordering: filename suffix is elapsed ms since tracking started,
   monotonically increasing but not sequential/regular — must not assume a
   fixed frame rate or use timestamp-based ffmpeg frame matching (no
   `-vsync vfr` on a still sequence); use an explicit ordered file list so no
   frame is silently dropped or duplicated.
3. Frames with no detection (empty `boxes`) still carry a carried-forward
   `hint_cx`/`hint_cy` — crop on the hint anyway, so every detection record
   maps to exactly one crop frame (no gaps to reconcile between the mp4 frame
   count and the jsonl line count).
4. Crop window is fixed-size; if centering it on the hint would run off the
   source image, translate (clamp) the window's position to stay fully
   inside — never pad. The hint itself is guaranteed to never leave the
   image.
5. When a frame has multiple detections, pick the one to report by
   `accepted_mask` first, then highest confidence among accepted (falling
   back to highest confidence overall if none are accepted). This is a new
   rule, not a port of `thumbnail.py`'s `_select_best_frame` — that function
   solves a different problem (picking one frame among many, by latest
   accepted `elapsed_ms`, confidence only as a no-accepted-detections
   fallback) rather than picking one detection among several within a
   single frame.
6. `maintenance_cycle` runs at midnight during a quiet window — no compute
   budget concern from adding a per-frame crop + one more ffmpeg encode.
7. Thumbnails read from `*_recording.mp4` via `detections.jsonl` timestamps
   (`thumbnail.py`), never from the raw jpgs — confirmed via a real
   before/after CRF-28 comparison. No jpg needs to be preserved for
   thumbnail purposes.
8. Rounding for every float→int pixel conversion (crop origin `x0`/`y0`,
   translated box/hint coordinates) is always `floor`, never `round`. Floor
   is the conservative direction — it never pushes a coordinate further out
   of bounds than the source float already implied — and the existing
   clamp-to-crop-bounds step (decision 4) is the deliberate fallback for
   anything that still lands outside `[0, crop_w]`/`[0, crop_h]` after
   translation, so no separate bounds-check needs to be added on top.

## Approach

All new logic lives in `src/archive.py`, following the existing pattern of
`_reencode` + the copy loop inside `archive_event`. No index/DB schema
changes needed — this is purely new files inside the archived event
directory.

**New constants** (next to `DEFAULT_CRF` at `src/archive.py:12`):
```python
DEFAULT_CROP_WIDTH = 1280
DEFAULT_CROP_HEIGHT = 720
DEFAULT_TRAINING_CLIP_FPS = 10
```
(Reuses the same `crf` value already threaded through `archive_event` for
the training clip's encode — one fewer knob, and the CRF-28 test already
showed no visible quality loss.)

**New helpers in `src/archive.py`** (after `_reencode`, before
`archive_event`):

- `_frame_elapsed_ms(stem: str) -> int` — same trailing-`_<digits>` regex
  parse as `thumbnail._elapsed_ms_from_stem`. Duplicated rather than
  imported/shared: this codebase already duplicates this exact parse
  independently in `thumbnail.py`, so matching that precedent rather than
  introducing a new shared module.
- `_iter_detection_records(event_dir) -> list[dict]` — parse
  `detections.jsonl`, sorted by `_frame_elapsed_ms(record["frame"])`.
- `_primary_detection(record) -> tuple[list[float], int] | None` — implements
  the accepted-then-confidence rule from decision 5; returns the chosen box
  and its index into `boxes`, or `None` if `boxes` is empty.
- `_crop_origin(cx, cy, img_w, img_h, crop_w, crop_h) -> tuple[int, int]` —
  `x0 = clamp(floor(cx - crop_w/2), 0, img_w - crop_w)`, same for `y0`
  (floor per decision 8). Raises if `img_w < crop_w or img_h < crop_h` (fail
  loud rather than silently producing a mismatched crop size — every real
  source frame is 1920x1080, confirmed against the real example event, so
  this should never trigger; if it does, that event's archive fails and is
  surfaced per "Failure surfacing" below, like any other `archive_event`
  failure).
- `_build_training_clip(event_dir, tmp_dest, crop_w, crop_h, fps, crf) -> int`:
  1. `records = _iter_detection_records(event_dir)`; if empty, skip
     (no jpgs to process — leave no cropped-training artifacts for this
     event rather than erroring).
  2. For each record, in order: load `{frame}.jpg`, compute crop origin from
     `hint_cx`/`hint_cy` via `_crop_origin`, slice the fixed-size window
     (`img[y0:y0+crop_h, x0:x0+crop_w]`), write it to a scratch dir
     (`tmp_dest / "_crop_scratch"`) as a sequentially-numbered jpg.
     Translate every box in `record["boxes"]` into crop-local coordinates
     (`floor(x) - x0, floor(y) - y0`, per decision 8). A box with no
     positive-area overlap against
     `[0, crop_w] x [0, crop_h]` (i.e. it falls entirely outside the crop
     window — possible for a non-primary detection elsewhere in the source
     frame) is dropped from the record rather than clamped into a
     degenerate zero-area sliver at the crop edge; drop the corresponding
     entries from `classes`/`confidences`/`accepted_mask` too, keeping
     index alignment. Surviving boxes are then clamped to
     `[0, crop_w]` / `[0, crop_h]` (same abut-don't-pad rule as the crop
     window itself).

     **Correction found during the real-data spot-check** (see below): the
     primary box does *not* always survive the crop. `hint_cx`/`hint_cy` is
     carried forward from the actual track, independent of any single
     frame's detections — a frame can have its only box be a low-
     confidence, unaccepted false positive elsewhere in the image (real
     example: `Hudson_Valley_PTZ_1_53973` in the `N6884H_touch_and_go_...`
     event, box centered around `(435, 1055)` while the hint — and the
     crop — sits around `(1144, 61)`). `_primary_detection`'s result
     ("the one to report", decision 5) is therefore surfaced as a
     `primary_box_index` field on the output record — the surviving-boxes
     index of the primary detection, or `null` if it didn't survive the
     crop — rather than asserted as an invariant. Translate `hint_cx`/
     `hint_cy` the same way as the crop origin (no drop applies — the hint
     is guaranteed inside the source image and the crop is built to
     contain it). Build the corresponding `detections_cropped.jsonl`
     record: original fields (`method`, `classes`, `confidences`,
     `accepted_mask`) filtered as above, plus `boxes`/`hint_cx`/`hint_cy`
     translated, plus new `primary_box_index`, `frame_index` (0-based
     position in the assembled clip), `source_frame` (original stem, kept
     for traceability), `crop_x0`, `crop_y0`.
  3. Assemble the scratch jpgs into `tmp_dest / "training_crop.mp4"` via
     ffmpeg's `concat` demuxer fed an explicit ordered file list (`file
     '<path>'` per line) at `fps` — guarantees exact 1:1, order-preserving
     frame mapping regardless of the original irregular spacing, unlike
     feeding a real video through `-vsync`. Same encode settings as
     `_reencode` (`-c:v libx264 -preset medium -crf {crf}`), just fed via
     `-f concat -safe 0 -r {fps} -i <filelist>` instead of `-i <source
     video>`.
  4. Verify via `ffprobe -count_frames` that the assembled clip's frame count
     equals `len(records)`; raise `RuntimeError` if not (aborts this event's
     archive before the swap — same failure contract as a bad `_reencode`).
  5. Write `detections_cropped.jsonl` to `tmp_dest`.
  6. Delete the scratch dir.
  7. Return `len(records)` (for logging).

**Changes to `archive_event`** (`src/archive.py:115-154`):
- In the copy loop (lines 131-142), also skip `*.jpg` files (in addition to
  the existing video-name skip) — they're superseded by the generated
  `training_crop.mp4`. (`cleanup_debug_data`'s keep-list is untouched: jpgs
  still need to survive until archive time so this step can read them from
  `src_dir`.)
- After the `_reencode` loop (line 145), call
  `_build_training_clip(src_dir, tmp_dest, crop_w, crop_h, fps, crf)` —
  reading stills from `src_dir` (the original, not-yet-moved event dir) and
  writing into `tmp_dest`. Any exception here propagates up before
  `shutil.move`/`shutil.rmtree(src_dir)`, so a failure leaves the original
  event untouched and `status='local'` for retry.
- Thread `crop_w`, `crop_h`, `fps` as new parameters on `archive_event` with
  the `DEFAULT_*` values as defaults, same as `crf` today.

**Threading through `run_maintenance_cycle`** (`src/archive.py:157-225`) and
**`service.py`**: mirror exactly how `ARCHIVE_CRF` is wired today
(`_load_config()` around `service.py:43-57`, module globals, passed
positionally into `run_maintenance_cycle` from the `/maintenance_cycle`
handler at `service.py:344-358`). New env vars: `ARCHIVE_CROP_WIDTH`,
`ARCHIVE_CROP_HEIGHT`, `ARCHIVE_TRAINING_CLIP_FPS`.

## Failure surfacing

Currently `run_maintenance_cycle`'s per-event except-blocks (debug cleanup
and archive) just `logger.exception` and leave `status='local'`, so a
permanently-failing event (e.g. a corrupt jpg, a `_build_training_clip`
frame-count mismatch) gets silently re-attempted — and re-fails — every
maintenance cycle forever, with no visibility. This round adds a `failed`
status so a broken event stops being retried and shows up somewhere a human
will look.

**Schema** (`src/index.py`, `_SCHEMA`): add a nullable `failure_stage TEXT`
column to `events`, alongside the existing `status`/`archive_path`/
`debug_cleaned` columns.

**`Index.set_failed(event_id: int, stage: str) -> None`** (new method, next
to `set_archived`/`set_debug_cleaned`): `UPDATE events SET status =
'failed', failure_stage = ? WHERE id = ?`.

**`Index.list_failed() -> list[dict]`** (new method): `SELECT * FROM events
WHERE status = 'failed'`, powers `/failed_jobs`.

**`Index.upsert_event`** (`src/index.py:126`): the never-downgrades check
`new_status = existing["status"] if existing["status"] == "archived" else
status` becomes `if existing["status"] in ("archived", "failed") else
status`. Without this, the next periodic scan would see the (untouched,
still-`local`-looking-on-disk) event directory and flip `status` straight
back to `local`/`indexing`, silently undoing the failure mark and putting
the event right back in next cycle's candidate query — the same infinite
silent-retry problem this feature exists to fix. As with `archived`,
`failed` becomes one-way from the index's perspective: recovery is manual
(fix whatever's wrong on disk, then reset the row's `status` directly —
no reset endpoint is being added here).

**`src/archive.py`**: new helper `_mark_failed(index, row, stage)` —
touches an empty `failed_archive` (stage=`"archive"`) or
`failed_debug_cleanup` (stage=`"debug_cleanup"`) marker file in
`row["path"]`, then calls `index.set_failed(row["id"], stage)`. Both
except-blocks in `run_maintenance_cycle` call this and `logger.warning`
(with the exception) instead of today's `logger.exception`.

**`src/service.py` / `src/models.py`**:
- `StatusResponse` gets a new `failed_count: int` field; `/status` sets it
  from `counts.get("failed", 0)` (no new `Index` method needed — the
  existing `count_by_status` group-by already includes `failed` once rows
  exist with that status).
- New `FailedJob` / `FailedJobList` models (id, path, dir_name,
  registration, event, failure_stage, last_seen) and a new `GET
  /failed_jobs` endpoint backed by `Index.list_failed()`, following the
  existing `EventList`/`/events` shape.

Out of scope for this round (per Matthew — keep the failure record minimal
for now): no stored error message/traceback, no automatic retry/backoff, no
`/failed_jobs/{id}/retry` endpoint.

## Files touched

- `src/archive.py` — all new logic described above, plus `_mark_failed` and
  the two updated except-blocks in `run_maintenance_cycle`.
- `src/index.py` — `failure_stage` column, `Index.set_failed`,
  `Index.list_failed`, the `upsert_event` sticky-status fix.
- `src/models.py` — `failed_count` on `StatusResponse`, new `FailedJob` /
  `FailedJobList`.
- `src/service.py` — three new env-var reads in `_load_config()`, three new
  globals, threaded into the existing `run_maintenance_cycle(...)` call;
  `failed_count` wired into `/status`; new `GET /failed_jobs` endpoint.
- `tests/test_archive.py` — extend `_make_event_with_video` (or add a
  sibling fixture) to also write a handful of `{frame}.jpg` stills matching
  `detections.jsonl` lines with real `hint_cx`/`hint_cy`/`boxes`, since the
  current fixture's `detections.jsonl` has no matching jpgs at all. Add
  tests for: `_crop_origin` clamping (center window fits fully inside
  bounds even when hint is near an edge, floor rounding per decision 8),
  `_primary_detection`'s accepted/confidence precedence, `_build_training_clip`
  producing a frame count matching input record count, `archive_event` no
  longer copying raw `*.jpg` into the archived dir but producing
  `training_crop.mp4` + `detections_cropped.jsonl` instead, and a failure
  path (e.g. corrupt/missing jpg) leaving the source directory untouched,
  writing `failed_archive`, and setting `status='failed'`/`failure_stage=
  'archive'` on the index row. Add a debug-cleanup-failure test (e.g. an
  unreadable subdirectory) covering the same `failed_debug_cleanup` path.
- `tests/test_index.py` — `upsert_event` no longer downgrades a `failed`
  row on rescan (mirrors the existing `archived`-never-downgrades test).
- `tests/test_service.py` — `/status` reports `failed_count`; `/failed_jobs`
  lists a failed row with its `failure_stage`.

## Verification

- Run the existing suite plus new tests: `pytest tests/test_archive.py -q`
  (real ffmpeg/OpenCV, per existing convention — no mocking).
- Run the full suite before considering this done, per project convention:
  `pytest -q`.
- Manual spot-check against the real example event
  (`N6884H_touch_and_go_24_20260724_183304`, already used for the CRF-28
  comparison this session): run the new pipeline against a copy of it,
  visually confirm a handful of `training_crop.mp4` frames still show the
  aircraft centered/legible, and confirm `detections_cropped.jsonl`'s
  translated boxes land inside the crop frame when overlaid.