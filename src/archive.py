import datetime
import fnmatch
import json
import logging
import math
import pathlib
import re
import shutil
import subprocess

import cv2

import src.index

logger = logging.getLogger(__name__)

DEFAULT_CRF = 28
DEFAULT_CROP_WIDTH = 1280
DEFAULT_CROP_HEIGHT = 720
DEFAULT_TRAINING_CLIP_FPS = 10

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


def select_tilt_calibration_deletion_candidates(
    index: src.index.Index,
    tilt_calibration_cleanup_days: float,
    batch_size: int,
) -> list[dict]:
    cutoff = (
        datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=tilt_calibration_cleanup_days)
    ).isoformat(timespec="seconds")
    return index.local_events_by_type_older_than("tilt_calibration", cutoff, limit=batch_size)


def delete_tilt_calibration_event(row: dict) -> None:
    """Remove a tilt_calibration event directory outright. Unlike archived event
    types, calibration runs carry no data worth keeping past debug cleanup, so
    they're deleted rather than routed through archive_event."""
    event_dir = pathlib.Path(row["path"])
    if event_dir.is_dir():
        shutil.rmtree(event_dir)


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


_FRAME_STEM_RE = re.compile(r"^(.+)_(\d+)$")


def _frame_elapsed_ms(stem: str) -> int:
    """Same trailing-`_<digits>` parse as thumbnail._elapsed_ms_from_stem, duplicated
    rather than shared since this codebase already duplicates this exact parse
    independently in thumbnail.py."""
    match = _FRAME_STEM_RE.match(stem)
    return int(match.group(2)) if match else 0


def _iter_detection_records(event_dir: pathlib.Path) -> list[dict]:
    path = event_dir / "detections.jsonl"
    records = []
    if path.exists():
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    records.sort(key=lambda r: _frame_elapsed_ms(r["frame"]))
    return records


def _primary_detection(record: dict) -> tuple[list[float], int] | None:
    """Pick the detection to report for a frame with multiple boxes: accepted first,
    then highest confidence among accepted, falling back to highest confidence
    overall if none are accepted. Returns (box, index into boxes), or None if the
    frame has no boxes at all."""
    boxes = record.get("boxes", [])
    if not boxes:
        return None
    confidences = record.get("confidences", [])
    accepted_mask = record.get("accepted_mask", [])

    def confidence(i: int) -> float:
        return float(confidences[i]) if i < len(confidences) else float("-inf")

    accepted_indices = [i for i in range(len(boxes)) if i < len(accepted_mask) and accepted_mask[i]]
    candidates = accepted_indices if accepted_indices else list(range(len(boxes)))
    best_index = max(candidates, key=confidence)
    return boxes[best_index], best_index


def _crop_origin(
    cx: float, cy: float, img_w: int, img_h: int, crop_w: int, crop_h: int
) -> tuple[int, int]:
    """Fixed-size crop window origin, centered on (cx, cy) and clamped (never
    padded) to stay fully inside the image. Float coordinates always round toward
    floor (never round/ceil) so a clamp is never skipped by a coordinate rounding
    just past its bound."""
    if img_w < crop_w or img_h < crop_h:
        raise RuntimeError(
            f"source frame ({img_w}x{img_h}) smaller than crop window ({crop_w}x{crop_h})"
        )
    x0 = min(max(math.floor(cx - crop_w / 2), 0), img_w - crop_w)
    y0 = min(max(math.floor(cy - crop_h / 2), 0), img_h - crop_h)
    return x0, y0


def _encode_training_clip(file_list_path: pathlib.Path, dest_video: pathlib.Path, fps: int, crf: int) -> None:
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "concat", "-safe", "0", "-r", str(fps),
        "-i", str(file_list_path),
        "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
        str(dest_video),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg training-clip encode failed (rc={result.returncode}): {result.stderr.strip()}"
        )


def _count_frames(video_path: pathlib.Path) -> int:
    command = [
        "ffprobe", "-v", "error", "-count_frames",
        "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames",
        "-of", "csv=p=0", str(video_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe frame count failed (rc={result.returncode}): {result.stderr.strip()}")
    return int(result.stdout.strip())


def _translate_boxes(
    record: dict, x0: int, y0: int, crop_w: int, crop_h: int
) -> tuple[list[list[float]], list, list, list, list[int]]:
    """Translate every box into crop-local coordinates, dropping any box with no
    positive-area overlap against the crop window rather than clamping it into a
    degenerate zero-area sliver at the edge. classes/confidences/accepted_mask are
    filtered in lockstep to keep index alignment. Returns the surviving arrays plus
    the list of original box indices that survived (for the primary-detection
    invariant check in _build_training_clip)."""
    src_boxes = record.get("boxes", [])
    src_classes = record.get("classes", [])
    src_confidences = record.get("confidences", [])
    src_accepted = record.get("accepted_mask", [])

    boxes, classes, confidences, accepted_mask, surviving_indices = [], [], [], [], []
    for j, box in enumerate(src_boxes):
        bx0 = math.floor(box[0]) - x0
        by0 = math.floor(box[1]) - y0
        bx1 = math.floor(box[2]) - x0
        by1 = math.floor(box[3]) - y0
        if bx1 <= 0 or by1 <= 0 or bx0 >= crop_w or by0 >= crop_h or bx1 <= bx0 or by1 <= by0:
            continue
        boxes.append([
            max(0, min(bx0, crop_w)), max(0, min(by0, crop_h)),
            max(0, min(bx1, crop_w)), max(0, min(by1, crop_h)),
        ])
        if j < len(src_classes):
            classes.append(src_classes[j])
        if j < len(src_confidences):
            confidences.append(src_confidences[j])
        if j < len(src_accepted):
            accepted_mask.append(src_accepted[j])
        surviving_indices.append(j)

    return boxes, classes, confidences, accepted_mask, surviving_indices


def _build_training_clip(
    event_dir: pathlib.Path,
    tmp_dest: pathlib.Path,
    crop_w: int,
    crop_h: int,
    fps: int,
    crf: int,
) -> int:
    """Build training_crop.mp4 + detections_cropped.jsonl in tmp_dest from the raw
    per-frame jpg stills and detections.jsonl in event_dir. Returns the number of
    frames processed (0 if there were no detection records at all — in which case
    no training-crop artifacts are written for this event)."""
    records = _iter_detection_records(event_dir)
    if not records:
        return 0

    scratch_dir = tmp_dest / "_crop_scratch"
    scratch_dir.mkdir()
    try:
        cropped_records = []
        file_list_lines = []
        for i, record in enumerate(records):
            frame_path = event_dir / f"{record['frame']}.jpg"
            img = cv2.imread(str(frame_path))
            if img is None:
                raise RuntimeError(f"Could not read training-crop source frame {frame_path}")
            img_h, img_w = img.shape[:2]

            hint_cx, hint_cy = record["hint_cx"], record["hint_cy"]
            x0, y0 = _crop_origin(hint_cx, hint_cy, img_w, img_h, crop_w, crop_h)
            crop = img[y0:y0 + crop_h, x0:x0 + crop_w]

            scratch_path = scratch_dir / f"{i:06d}.jpg"
            if not cv2.imwrite(str(scratch_path), crop):
                raise RuntimeError(f"Failed to write scratch crop frame {scratch_path}")
            file_list_lines.append(f"file '{scratch_path.resolve()}'\n")

            boxes, classes, confidences, accepted_mask, surviving_indices = _translate_boxes(
                record, x0, y0, crop_w, crop_h
            )

            # The detection decision 5 says to "report" among several in a frame.
            # The crop is centered on the hint, not on this box, so it is not
            # guaranteed to land inside the crop (e.g. a low-confidence, unaccepted
            # detection elsewhere in frame while hint_cx/hint_cy carries forward the
            # real track) — None here means "the primary detection didn't survive
            # the crop", not an error.
            primary = _primary_detection(record)
            primary_box_index = None
            if primary is not None and primary[1] in surviving_indices:
                primary_box_index = surviving_indices.index(primary[1])

            cropped_records.append({
                "method": record.get("method"),
                "boxes": boxes,
                "classes": classes,
                "confidences": confidences,
                "accepted_mask": accepted_mask,
                "primary_box_index": primary_box_index,
                "hint_cx": hint_cx - x0,
                "hint_cy": hint_cy - y0,
                "frame_index": i,
                "source_frame": record["frame"],
                "crop_x0": x0,
                "crop_y0": y0,
            })

        file_list_path = scratch_dir / "files.txt"
        file_list_path.write_text("".join(file_list_lines))

        clip_path = tmp_dest / "training_crop.mp4"
        _encode_training_clip(file_list_path, clip_path, fps, crf)

        frame_count = _count_frames(clip_path)
        if frame_count != len(records):
            raise RuntimeError(
                f"training clip frame count mismatch: expected {len(records)}, got {frame_count}"
            )

        with (tmp_dest / "detections_cropped.jsonl").open("w") as f:
            for cropped_record in cropped_records:
                f.write(json.dumps(cropped_record) + "\n")
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)

    return len(records)


def _mark_failed(index: src.index.Index, row: dict, stage: str) -> None:
    """Stop a permanently-failing event from being retried every maintenance cycle:
    write an on-disk marker (mirrors the `complete` marker convention) and set
    status='failed' in the index, which upsert_event treats as sticky like
    'archived'. Recovery is manual."""
    marker_name = "failed_archive" if stage == "archive" else "failed_debug_cleanup"
    event_dir = pathlib.Path(row["path"])
    try:
        (event_dir / marker_name).touch()
    except OSError:
        logger.warning("Could not write %s marker in %s", marker_name, event_dir)
    index.set_failed(row["id"], stage)


def archive_event(
    row: dict,
    archive_root: pathlib.Path,
    crf: int = DEFAULT_CRF,
    crop_w: int = DEFAULT_CROP_WIDTH,
    crop_h: int = DEFAULT_CROP_HEIGHT,
    training_clip_fps: int = DEFAULT_TRAINING_CLIP_FPS,
) -> str:
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
        # Raw per-frame jpg stills are superseded by training_crop.mp4, built below
        # from these same stills while they're still readable from src_dir.
        if fnmatch.fnmatch(entry.name, "*.jpg"):
            continue
        if entry.is_dir():
            shutil.copytree(entry, tmp_dest / entry.name)
        else:
            shutil.copy2(entry, tmp_dest / entry.name)

    for video_path in video_paths:
        _reencode(video_path, tmp_dest / video_path.name, crf)

    _build_training_clip(src_dir, tmp_dest, crop_w, crop_h, training_clip_fps, crf)

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
    tilt_calibration_cleanup_days: float | None = None,
    crop_w: int = DEFAULT_CROP_WIDTH,
    crop_h: int = DEFAULT_CROP_HEIGHT,
    training_clip_fps: int = DEFAULT_TRAINING_CLIP_FPS,
) -> dict:
    """Process one bounded batch of archive candidates, plus (if enabled) a batch
    of debug cleanups on still-local events. Idempotent/resumable in the sense that
    a failure on one event doesn't abort the whole batch — but the event itself is
    marked 'failed' (see _mark_failed) rather than left for automatic retry, since a
    permanent failure (bad data) would otherwise just fail again every cycle."""
    debug_cleaned = 0
    if debug_cleanup_days is not None:
        debug_candidates = select_debug_cleanup_candidates(index, debug_cleanup_days, batch_size)
        logger.info("Debug cleanup starting: %d candidate(s)", len(debug_candidates))
        for row in debug_candidates:
            try:
                cleanup_debug_data(pathlib.Path(row["path"]))
                index.set_debug_cleaned(row["id"])
                debug_cleaned += 1
            except Exception as exc:
                logger.warning(
                    "Failed to clean debug data for event id=%s path=%s: %s",
                    row["id"], row["path"], exc, exc_info=True,
                )
                _mark_failed(index, row, "debug_cleanup")

    tilt_calibration_deleted = 0
    if tilt_calibration_cleanup_days is not None:
        tilt_candidates = select_tilt_calibration_deletion_candidates(
            index, tilt_calibration_cleanup_days, batch_size
        )
        logger.info("Tilt calibration cleanup starting: %d candidate(s)", len(tilt_candidates))
        for row in tilt_candidates:
            try:
                delete_tilt_calibration_event(row)
                index.delete_event(row["id"])
                tilt_calibration_deleted += 1
            except Exception:
                logger.exception(
                    "Failed to delete tilt_calibration event id=%s path=%s", row["id"], row["path"]
                )

    candidates = select_candidates(index, age_days, disk_threshold_pct, filespace_root, batch_size)
    logger.info("Maintenance cycle starting: %d candidate(s)", len(candidates))

    processed = 0
    for row in candidates:
        try:
            archive_path = archive_event(row, archive_root, crf, crop_w, crop_h, training_clip_fps)
            index.set_archived(row["id"], archive_path)
            processed += 1
        except Exception as exc:
            logger.warning(
                "Failed to archive event id=%s path=%s: %s", row["id"], row["path"], exc, exc_info=True
            )
            _mark_failed(index, row, "archive")

    remaining = len(
        select_candidates(index, age_days, disk_threshold_pct, filespace_root, batch_size=1_000_000)
    )
    logger.info(
        "Maintenance cycle complete: processed=%d remaining=%d debug_cleaned=%d tilt_calibration_deleted=%d",
        processed, remaining, debug_cleaned, tilt_calibration_deleted,
    )
    return {
        "status": "ok",
        "processed": processed,
        "remaining": remaining,
        "debug_cleaned": debug_cleaned,
        "tilt_calibration_deleted": tilt_calibration_deleted,
    }
