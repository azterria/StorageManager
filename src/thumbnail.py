import json
import logging
import pathlib
import re
import shutil

import cv2
import numpy

logger = logging.getLogger(__name__)

_FRAME_STEM_RE = re.compile(r"^(.+)_(\d+)$")


def _elapsed_ms_from_stem(stem: str) -> int:
    match = _FRAME_STEM_RE.match(stem)
    return int(match.group(2)) if match else 0


def _find_video(event_dir: pathlib.Path) -> pathlib.Path:
    candidates = sorted(event_dir.glob("*_recording.mp4"))
    if not candidates:
        logger.warning("No *_recording.mp4 found in %s", event_dir)
        raise FileNotFoundError(f"No *_recording.mp4 found in {event_dir}")
    return candidates[0]


def _select_best_frame(event_dir: pathlib.Path) -> int:
    """Pick the last frame (by elapsed_ms) with an accepted detection.

    Falls back to the highest-confidence detection of any kind, then to the first
    frame of the video (elapsed_ms=0) if there are no usable detection records at all.
    """
    path = event_dir.joinpath("detections.jsonl")
    last_accepted_ms: int | None = None
    best_fallback_score: float | None = None
    best_fallback_stem: str | None = None
    if path.exists():
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                stem = record.get("frame")
                if stem is None:
                    continue
                confidences = record.get("confidences", [])
                accepted_mask = record.get("accepted_mask", [])
                for i, confidence in enumerate(confidences):
                    accepted = bool(accepted_mask[i]) if i < len(accepted_mask) else False
                    if accepted:
                        elapsed_ms = _elapsed_ms_from_stem(stem)
                        if last_accepted_ms is None or elapsed_ms > last_accepted_ms:
                            last_accepted_ms = elapsed_ms
                    if best_fallback_score is None or float(confidence) > best_fallback_score:
                        best_fallback_score = float(confidence)
                        best_fallback_stem = stem
    if last_accepted_ms is not None:
        return last_accepted_ms
    if best_fallback_stem is None:
        return 0
    return _elapsed_ms_from_stem(best_fallback_stem)


def get_or_create_thumbnail(event_dir: pathlib.Path, cache_path: pathlib.Path) -> pathlib.Path:
    """Return the cached thumbnail JPEG for an event, generating it on first request."""
    if cache_path.exists():
        return cache_path

    video_path = _find_video(event_dir)
    elapsed_ms = _select_best_frame(event_dir)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("Cannot open video %s", video_path)
        raise RuntimeError(f"Cannot open video {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(elapsed_ms))
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        logger.error("Could not extract frame at %dms from %s", elapsed_ms, video_path)
        raise RuntimeError(f"Could not extract frame at {elapsed_ms}ms from {video_path}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(cache_path.stem + ".tmp" + cache_path.suffix)
    _write_jpeg(tmp_path, frame)
    shutil.move(str(tmp_path), str(cache_path))
    logger.info("Generated thumbnail %s from %s at %dms", cache_path, video_path, elapsed_ms)
    return cache_path


def _write_jpeg(path: pathlib.Path, frame: numpy.ndarray) -> None:
    ok = cv2.imwrite(str(path), frame)
    if not ok:
        logger.error("Failed to write JPEG to %s", path)
        raise RuntimeError(f"Failed to write JPEG to {path}")
