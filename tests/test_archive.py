import datetime
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


def _make_event_with_video(base, date, name):
    event_dir = base / date / name
    event_dir.mkdir(parents=True)
    (event_dir / "detections.jsonl").write_text('{"frame": "cam_0", "boxes": [], "confidences": []}\n')
    (event_dir / "debug.log").write_text("log line\n")
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
    assert result == {"status": "ok", "processed": 0, "remaining": 0}


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
    assert idx.get_event(bad_id)["status"] == "local"
