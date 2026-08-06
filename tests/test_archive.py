import datetime
import json
import pathlib
import shutil

import cv2
import numpy
import pytest

import src.archive
import src.index
import src.parser


@pytest.fixture
def idx(tmp_path):
    index = src.index.Index(str(tmp_path / "index.db"))
    yield index
    index.close()


def _insert(idx, event_dir, date, time, status="local", registration="N123AB"):
    parsed = src.parser.ParsedEvent(
        registration=registration, event="landing", runway="24", date=date, time=time, classified=True
    )
    return idx.upsert_event(
        dev=1, ino=event_dir.stat().st_ino, path=str(event_dir), dir_name=event_dir.name,
        parsed=parsed, mtime=1.0, status=status,
    )


def _make_event_with_video(base, date, name, with_debug_subfolders=False):
    event_dir = base / date / name
    event_dir.mkdir(parents=True)
    # Empty (no detection records) rather than a real record with no matching jpg —
    # _build_training_clip short-circuits to a no-op on an empty detections.jsonl,
    # which keeps this fixture usable by tests that don't care about the training-
    # crop pipeline. See _make_event_with_training_stills for that pipeline's tests.
    (event_dir / "detections.jsonl").write_text("")
    (event_dir / "debug.log").write_text("log line\n")
    video_path = event_dir / "cam_recording.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 5, (32, 24))
    for _ in range(5):
        writer.write(numpy.zeros((24, 32, 3), dtype=numpy.uint8))
    writer.release()

    if with_debug_subfolders:
        (event_dir / "cam_1.jpg").write_bytes(b"jpg")
        for subfolder in ("associators", "camera", "detector", "sanity", "search_frames"):
            sub = event_dir / subfolder
            sub.mkdir()
            (sub / "frame_0.jpg").write_bytes(b"jpg")

    return event_dir


def _make_event_with_training_stills(base, date, name, num_frames=3, img_size=(64, 48)):
    """Like _make_event_with_video, but with real per-frame jpg stills and matching
    detections.jsonl records (hint_cx/hint_cy + one box each), for exercising
    _build_training_clip / the training-crop path of archive_event."""
    event_dir = base / date / name
    event_dir.mkdir(parents=True)
    w, h = img_size

    lines = []
    for i in range(num_frames):
        frame_name = f"cam_{i * 100}"
        cv2.imwrite(str(event_dir / f"{frame_name}.jpg"), numpy.zeros((h, w, 3), dtype=numpy.uint8))
        cx, cy = w / 2 + i, h / 2 + i
        lines.append(json.dumps({
            "frame": frame_name,
            "method": "yolo",
            "boxes": [[cx - 2, cy - 2, cx + 2, cy + 2]],
            "classes": [0],
            "confidences": [0.9],
            "accepted_mask": [True],
            "hint_cx": cx,
            "hint_cy": cy,
        }))
    (event_dir / "detections.jsonl").write_text("\n".join(lines) + "\n")
    (event_dir / "debug.log").write_text("log line\n")

    video_path = event_dir / "cam_recording.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 5, (w, h))
    for _ in range(5):
        writer.write(numpy.zeros((h, w, 3), dtype=numpy.uint8))
    writer.release()

    return event_dir


def _make_event_with_broken_training_data(base, date, name):
    """detections.jsonl references a frame with no matching jpg on disk, to exercise
    the archive/build-training-clip failure path."""
    event_dir = base / date / name
    event_dir.mkdir(parents=True)
    (event_dir / "detections.jsonl").write_text(
        json.dumps({
            "frame": "missing_frame_0", "boxes": [], "confidences": [], "hint_cx": 5, "hint_cy": 5,
        }) + "\n"
    )
    video_path = event_dir / "cam_recording.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 5, (32, 24))
    for _ in range(5):
        writer.write(numpy.zeros((24, 32, 3), dtype=numpy.uint8))
    writer.release()
    return event_dir


def test_select_candidates_age_based(tmp_path, idx):
    filespace = tmp_path / "filespace"
    today = datetime.datetime.now(datetime.timezone.utc)
    old_date = (today - datetime.timedelta(days=365)).strftime("%Y%m%d")
    new_date = today.strftime("%Y%m%d")
    old_dir = _make_event_with_video(filespace, old_date, f"old_landing_24_{old_date}_010203")
    new_dir = _make_event_with_video(filespace, new_date, f"new_landing_24_{new_date}_010203")
    _insert(idx, old_dir, old_date, "010203")
    _insert(idx, new_dir, new_date, "010203")

    candidates = src.archive.select_candidates(
        idx, age_days=30, disk_threshold_pct=99.9, filespace_root=filespace, batch_size=10
    )

    assert len(candidates) == 1
    assert candidates[0]["dir_name"] == old_dir.name


def test_select_candidates_disk_pressure_pulls_forward_recent(tmp_path, idx, monkeypatch):
    filespace = tmp_path / "filespace"
    recent_dir = _make_event_with_video(filespace, "20260101", "recent_landing_24_20260101_010203")
    _insert(idx, recent_dir, "20260101", "010203")

    monkeypatch.setattr(src.archive, "_disk_pressure", lambda root, threshold: True)
    candidates = src.archive.select_candidates(
        idx, age_days=3650, disk_threshold_pct=1.0, filespace_root=filespace, batch_size=10
    )

    assert len(candidates) == 1
    assert candidates[0]["dir_name"] == "recent_landing_24_20260101_010203"


def test_archive_event_moves_and_reencodes(tmp_path):
    filespace = tmp_path / "filespace"
    archive_root = tmp_path / "archive"
    event_dir = _make_event_with_video(filespace, "20200101", "old_landing_24_20200101_010203")
    row = {"id": 1, "path": str(event_dir), "date": "20200101", "dir_name": event_dir.name}

    dest = src.archive.archive_event(row, archive_root)

    dest_path = tmp_path / "archive" / "20200101" / event_dir.name
    assert dest == str(dest_path)
    assert dest_path.is_dir()
    assert (dest_path / "cam_recording.mp4").exists()
    assert (dest_path / "detections.jsonl").exists()
    assert (dest_path / "debug.log").exists()
    assert not event_dir.exists()


def test_cleanup_debug_data_removes_only_debug_subfolders(tmp_path):
    filespace = tmp_path / "filespace"
    event_dir = _make_event_with_video(
        filespace, "20200101", "old_landing_24_20200101_010203", with_debug_subfolders=True
    )

    src.archive.cleanup_debug_data(event_dir)

    for subfolder in ("associators", "camera", "detector", "sanity", "search_frames"):
        assert not (event_dir / subfolder).exists()
    assert (event_dir / "cam_recording.mp4").exists()
    assert (event_dir / "detections.jsonl").exists()
    assert (event_dir / "debug.log").exists()
    assert (event_dir / "cam_1.jpg").exists()


def test_archive_event_skips_debug_subfolders(tmp_path):
    filespace = tmp_path / "filespace"
    archive_root = tmp_path / "archive"
    event_dir = _make_event_with_video(
        filespace, "20200101", "old_landing_24_20200101_010203", with_debug_subfolders=True
    )
    row = {"id": 1, "path": str(event_dir), "date": "20200101", "dir_name": event_dir.name}

    dest = src.archive.archive_event(row, archive_root)

    dest_path = tmp_path / "archive" / "20200101" / event_dir.name
    assert dest == str(dest_path)
    for subfolder in ("associators", "camera", "detector", "sanity", "search_frames"):
        assert not (dest_path / subfolder).exists()
    assert (dest_path / "cam_recording.mp4").exists()
    # Raw jpgs (including top-level ones like cam_1.jpg, not just the debug
    # subfolders) are dropped in favor of the generated training_crop.mp4.
    assert not (dest_path / "cam_1.jpg").exists()


def test_run_maintenance_cycle_cleans_debug_data_before_archive_age(tmp_path, idx):
    filespace = tmp_path / "filespace"
    archive_root = tmp_path / "archive"
    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")
    event_dir = _make_event_with_video(
        filespace, today, f"recent_landing_24_{today}_010203", with_debug_subfolders=True
    )
    event_id = _insert(idx, event_dir, today, "010203")

    result = src.archive.run_maintenance_cycle(
        idx, filespace, archive_root, age_days=30, disk_threshold_pct=99.9, batch_size=10,
        debug_cleanup_days=0,
    )

    assert result["debug_cleaned"] == 1
    assert not (event_dir / "associators").exists()
    assert event_dir.exists()
    row = idx.get_event(event_id)
    assert row["status"] == "local"
    assert row["debug_cleaned"] == 1


def _insert_tilt_calibration(idx, event_dir, date, time_):
    parsed = src.parser.ParsedEvent(
        registration=None, event="tilt_calibration", runway=None, date=date, time=time_, classified=True
    )
    return idx.upsert_event(
        dev=1, ino=event_dir.stat().st_ino, path=str(event_dir), dir_name=event_dir.name,
        parsed=parsed, mtime=1.0, status="local",
    )


def test_run_maintenance_cycle_deletes_old_tilt_calibration(tmp_path, idx):
    filespace = tmp_path / "filespace"
    old_date = "20200101"
    event_dir = filespace / old_date / f"tilt_calibration_{old_date}_010203"
    event_dir.mkdir(parents=True)
    (event_dir / "debug.log").write_text("log line\n")
    event_id = _insert_tilt_calibration(idx, event_dir, old_date, "010203")

    result = src.archive.run_maintenance_cycle(
        idx, filespace, tmp_path / "archive", age_days=30, disk_threshold_pct=99.9, batch_size=10,
        tilt_calibration_cleanup_days=1,
    )

    assert result["tilt_calibration_deleted"] == 1
    assert not event_dir.exists()
    assert idx.get_event(event_id) is None


def test_run_maintenance_cycle_keeps_recent_tilt_calibration(tmp_path, idx):
    filespace = tmp_path / "filespace"
    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")
    now_str = datetime.datetime.now(datetime.timezone.utc).strftime("%H%M%S")
    event_dir = filespace / today / f"tilt_calibration_{today}_{now_str}"
    event_dir.mkdir(parents=True)
    event_id = _insert_tilt_calibration(idx, event_dir, today, now_str)

    result = src.archive.run_maintenance_cycle(
        idx, filespace, tmp_path / "archive", age_days=30, disk_threshold_pct=99.9, batch_size=10,
        tilt_calibration_cleanup_days=1,
    )

    assert result["tilt_calibration_deleted"] == 0
    assert event_dir.exists()
    assert idx.get_event(event_id) is not None


def test_archive_event_missing_source_raises(tmp_path):
    row = {"id": 1, "path": str(tmp_path / "does_not_exist"), "date": "20200101", "dir_name": "x"}
    with pytest.raises(FileNotFoundError):
        src.archive.archive_event(row, tmp_path / "archive")


def test_run_maintenance_cycle_processes_batch_and_updates_index(tmp_path, idx):
    filespace = tmp_path / "filespace"
    archive_root = tmp_path / "archive"
    old_dir = _make_event_with_video(filespace, "20200101", "old_landing_24_20200101_010203")
    event_id = _insert(idx, old_dir, "20200101", "010203")

    result = src.archive.run_maintenance_cycle(
        idx, filespace, archive_root, age_days=30, disk_threshold_pct=99.9, batch_size=10
    )

    assert result["status"] == "ok"
    assert result["processed"] == 1
    assert result["remaining"] == 0
    row = idx.get_event(event_id)
    assert row["status"] == "archived"
    assert row["archive_path"] is not None


def test_run_maintenance_cycle_is_idempotent_on_empty_backlog(tmp_path, idx):
    filespace = tmp_path / "filespace"
    filespace.mkdir()
    result = src.archive.run_maintenance_cycle(
        idx, filespace, tmp_path / "archive", age_days=30, disk_threshold_pct=99.9, batch_size=10
    )
    assert result == {
        "status": "ok", "processed": 0, "remaining": 0, "debug_cleaned": 0, "tilt_calibration_deleted": 0,
    }


def test_run_maintenance_cycle_skips_failed_event_and_continues(tmp_path, idx):
    filespace = tmp_path / "filespace"
    archive_root = tmp_path / "archive"
    good_dir = _make_event_with_video(filespace, "20200101", "good_landing_24_20200101_010203")
    good_id = _insert(idx, good_dir, "20200101", "010203")

    # A second row points at a directory that no longer exists on disk.
    bad_id = idx.upsert_event(
        dev=1, ino=999, path=str(filespace / "20200101" / "missing"), dir_name="missing",
        parsed=src.parser.ParsedEvent(None, "landing", "24", "20200101", "010203", True),
        mtime=1.0, status="local",
    )

    result = src.archive.run_maintenance_cycle(
        idx, filespace, archive_root, age_days=30, disk_threshold_pct=99.9, batch_size=10
    )

    assert result["processed"] == 1
    assert idx.get_event(good_id)["status"] == "archived"
    bad_row = idx.get_event(bad_id)
    assert bad_row["status"] == "failed"
    assert bad_row["failure_stage"] == "archive"


def test_crop_origin_centers_when_far_from_edges():
    assert src.archive._crop_origin(50, 50, 100, 100, 20, 20) == (40, 40)


def test_crop_origin_clamps_near_top_left():
    assert src.archive._crop_origin(2, 2, 100, 100, 20, 20) == (0, 0)


def test_crop_origin_clamps_near_bottom_right():
    assert src.archive._crop_origin(97, 97, 100, 100, 20, 20) == (80, 80)


def test_crop_origin_floors_rather_than_rounds():
    # 50.6 - 10 = 40.6: floor -> 40, round -> 41. Flooring must win (decision 8).
    assert src.archive._crop_origin(50.6, 50.6, 100, 100, 20, 20) == (40, 40)


def test_crop_origin_raises_when_crop_larger_than_source():
    with pytest.raises(RuntimeError):
        src.archive._crop_origin(10, 10, 50, 50, 100, 100)


def test_primary_detection_empty_boxes_returns_none():
    assert src.archive._primary_detection({"boxes": []}) is None


def test_primary_detection_prefers_accepted_over_higher_confidence():
    record = {
        "boxes": [[0, 0, 1, 1], [2, 2, 3, 3], [4, 4, 5, 5]],
        "confidences": [0.9, 0.5, 0.99],
        "accepted_mask": [False, True, False],
    }
    box, index = src.archive._primary_detection(record)
    assert index == 1
    assert box == [2, 2, 3, 3]


def test_primary_detection_picks_highest_confidence_among_accepted():
    record = {
        "boxes": [[0, 0, 1, 1], [2, 2, 3, 3]],
        "confidences": [0.5, 0.9],
        "accepted_mask": [True, True],
    }
    _, index = src.archive._primary_detection(record)
    assert index == 1


def test_primary_detection_falls_back_to_confidence_when_none_accepted():
    record = {
        "boxes": [[0, 0, 1, 1], [2, 2, 3, 3]],
        "confidences": [0.5, 0.9],
        "accepted_mask": [False, False],
    }
    _, index = src.archive._primary_detection(record)
    assert index == 1


def test_build_training_clip_no_records_is_a_noop(tmp_path):
    event_dir = tmp_path / "filespace" / "20200101" / "evt"
    event_dir.mkdir(parents=True)
    (event_dir / "detections.jsonl").write_text("")
    tmp_dest = tmp_path / "staging"
    tmp_dest.mkdir()

    count = src.archive._build_training_clip(event_dir, tmp_dest, crop_w=16, crop_h=16, fps=10, crf=28)

    assert count == 0
    assert not (tmp_dest / "training_crop.mp4").exists()
    assert not (tmp_dest / "detections_cropped.jsonl").exists()


def test_build_training_clip_frame_count_matches_records(tmp_path):
    event_dir = _make_event_with_training_stills(tmp_path / "filespace", "20200101", "evt", num_frames=4)
    tmp_dest = tmp_path / "staging"
    tmp_dest.mkdir()

    count = src.archive._build_training_clip(event_dir, tmp_dest, crop_w=16, crop_h=16, fps=10, crf=28)

    assert count == 4
    clip_path = tmp_dest / "training_crop.mp4"
    assert clip_path.exists()
    assert src.archive._count_frames(clip_path) == 4

    cropped_lines = (tmp_dest / "detections_cropped.jsonl").read_text().strip().split("\n")
    assert len(cropped_lines) == 4
    first = json.loads(cropped_lines[0])
    assert first["frame_index"] == 0
    assert first["source_frame"] == "cam_0"
    assert len(first["boxes"]) == 1
    assert first["primary_box_index"] == 0
    box = first["boxes"][0]
    assert 0 <= box[0] < box[2] <= 16
    assert 0 <= box[1] < box[3] <= 16
    assert not (tmp_dest / "_crop_scratch").exists()


def test_build_training_clip_primary_box_index_null_when_dropped_by_crop(tmp_path):
    """Regression test: hint_cx/hint_cy is carried forward from the real track
    independent of any single frame's detections, so a frame's only box can be a
    low-confidence, unaccepted false positive far from the hint-centered crop.
    The primary detection then falls outside the crop and must be dropped like any
    other out-of-window box, not treated as an invariant violation."""
    event_dir = tmp_path / "filespace" / "20200101" / "evt"
    event_dir.mkdir(parents=True)
    w, h = 64, 48
    cv2.imwrite(str(event_dir / "cam_0.jpg"), numpy.zeros((h, w, 3), dtype=numpy.uint8))
    (event_dir / "detections.jsonl").write_text(json.dumps({
        "frame": "cam_0",
        "method": "yolo",
        "boxes": [[0, 0, 4, 4]],  # far from the hint below
        "classes": [0],
        "confidences": [0.3],
        "accepted_mask": [False],
        "hint_cx": w - 2,
        "hint_cy": h - 2,
    }) + "\n")
    tmp_dest = tmp_path / "staging"
    tmp_dest.mkdir()

    count = src.archive._build_training_clip(event_dir, tmp_dest, crop_w=8, crop_h=8, fps=10, crf=28)

    assert count == 1
    record = json.loads((tmp_dest / "detections_cropped.jsonl").read_text().strip())
    assert record["boxes"] == []
    assert record["primary_box_index"] is None


def test_archive_event_produces_training_clip_and_drops_raw_jpgs(tmp_path):
    filespace = tmp_path / "filespace"
    archive_root = tmp_path / "archive"
    event_dir = _make_event_with_training_stills(
        filespace, "20200101", "evt_landing_24_20200101_010203", num_frames=3
    )
    row = {"id": 1, "path": str(event_dir), "date": "20200101", "dir_name": event_dir.name}

    dest = src.archive.archive_event(row, archive_root, crop_w=16, crop_h=16, training_clip_fps=10)

    dest_path = pathlib.Path(dest)
    assert (dest_path / "training_crop.mp4").exists()
    assert (dest_path / "detections_cropped.jsonl").exists()
    assert not list(dest_path.glob("*.jpg"))
    assert (dest_path / "detections.jsonl").exists()


def test_archive_event_raises_and_leaves_source_untouched_on_missing_jpg(tmp_path):
    filespace = tmp_path / "filespace"
    archive_root = tmp_path / "archive"
    event_dir = _make_event_with_broken_training_data(
        filespace, "20200101", "broken_landing_24_20200101_010203"
    )
    row = {"id": 1, "path": str(event_dir), "date": "20200101", "dir_name": event_dir.name}

    with pytest.raises(RuntimeError):
        src.archive.archive_event(row, archive_root)

    assert event_dir.exists()
    assert not (archive_root / "20200101" / event_dir.name).exists()


def test_run_maintenance_cycle_marks_archive_failure(tmp_path, idx):
    filespace = tmp_path / "filespace"
    archive_root = tmp_path / "archive"
    event_dir = _make_event_with_broken_training_data(
        filespace, "20200101", "broken_landing_24_20200101_010203"
    )
    event_id = _insert(idx, event_dir, "20200101", "010203")

    result = src.archive.run_maintenance_cycle(
        idx, filespace, archive_root, age_days=30, disk_threshold_pct=99.9, batch_size=10,
    )

    assert result["processed"] == 0
    row = idx.get_event(event_id)
    assert row["status"] == "failed"
    assert row["failure_stage"] == "archive"
    assert (event_dir / "failed_archive").exists()


def test_run_maintenance_cycle_marks_debug_cleanup_failure(tmp_path, idx, monkeypatch):
    filespace = tmp_path / "filespace"
    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d")
    event_dir = _make_event_with_video(filespace, today, f"recent_landing_24_{today}_010203")
    event_id = _insert(idx, event_dir, today, "010203")

    def _boom(_event_dir):
        raise OSError("simulated cleanup failure")

    monkeypatch.setattr(src.archive, "cleanup_debug_data", _boom)

    result = src.archive.run_maintenance_cycle(
        idx, filespace, tmp_path / "archive", age_days=30, disk_threshold_pct=99.9, batch_size=10,
        debug_cleanup_days=0,
    )

    assert result["debug_cleaned"] == 0
    row = idx.get_event(event_id)
    assert row["status"] == "failed"
    assert row["failure_stage"] == "debug_cleanup"
    assert (event_dir / "failed_debug_cleanup").exists()
