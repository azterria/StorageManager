import json

import cv2
import numpy
import pytest

import src.thumbnail

FPS = 10
FRAME_SIZE = (64, 48)  # (width, height)

# BGR colors, one per frame, 100ms apart.
_COLORS = [
    (0, 0, 255),   # frame at 0ms: red
    (0, 255, 0),   # frame at 100ms: green
    (255, 0, 0),   # frame at 200ms: blue
]


def _make_video(event_dir, name="cam_recording.mp4"):
    path = event_dir / name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, FPS, FRAME_SIZE)
    for color in _COLORS:
        frame = numpy.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=numpy.uint8)
        frame[:, :] = color
        writer.write(frame)
    writer.release()
    return path


def _write_detections(event_dir, records):
    path = event_dir / "detections.jsonl"
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def _dominant_bgr_channel(image) -> int:
    means = [float(image[:, :, i].mean()) for i in range(3)]
    return int(numpy.argmax(means))


def test_thumbnail_extracts_highest_confidence_accepted_frame(tmp_path):
    event_dir = tmp_path / "event"
    event_dir.mkdir()
    _make_video(event_dir)
    _write_detections(event_dir, [
        {"frame": "cam_0", "boxes": [[0, 0, 10, 10]], "confidences": [0.95], "accepted_mask": [False]},
        {"frame": "cam_100", "boxes": [[0, 0, 10, 10]], "confidences": [0.5], "accepted_mask": [True]},
        {"frame": "cam_200", "boxes": [[0, 0, 10, 10]], "confidences": [0.1], "accepted_mask": [False]},
    ])

    cache_path = tmp_path / "thumb.jpg"
    result = src.thumbnail.get_or_create_thumbnail(event_dir, cache_path)

    assert result == cache_path
    assert cache_path.exists()
    image = cv2.imread(str(cache_path))
    # frame at 100ms (green) should have been selected — its accepted_mask is True
    # despite lower confidence than the unaccepted frames.
    assert _dominant_bgr_channel(image) == 1  # green channel


def test_thumbnail_is_cached_after_first_request(tmp_path):
    event_dir = tmp_path / "event"
    event_dir.mkdir()
    _make_video(event_dir)
    _write_detections(event_dir, [
        {"frame": "cam_0", "boxes": [[0, 0, 10, 10]], "confidences": [0.9], "accepted_mask": [True]},
    ])
    cache_path = tmp_path / "thumb.jpg"

    src.thumbnail.get_or_create_thumbnail(event_dir, cache_path)
    mtime_first = cache_path.stat().st_mtime_ns

    src.thumbnail.get_or_create_thumbnail(event_dir, cache_path)
    mtime_second = cache_path.stat().st_mtime_ns

    assert mtime_first == mtime_second


def test_thumbnail_no_detections_falls_back_to_first_frame(tmp_path):
    event_dir = tmp_path / "event"
    event_dir.mkdir()
    _make_video(event_dir)

    cache_path = tmp_path / "thumb.jpg"
    result = src.thumbnail.get_or_create_thumbnail(event_dir, cache_path)

    assert result.exists()
    image = cv2.imread(str(cache_path))
    assert _dominant_bgr_channel(image) == 2  # red channel, first frame


def test_thumbnail_missing_video_raises(tmp_path):
    event_dir = tmp_path / "event"
    event_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        src.thumbnail.get_or_create_thumbnail(event_dir, tmp_path / "thumb.jpg")
