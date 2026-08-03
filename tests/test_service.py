import os
import time

import cv2
import numpy
from fastapi.testclient import TestClient

import src.archive
import src.service


def _make_event_dir(filespace, date, name, mtime, with_video=False, with_detections=False, complete=True):
    event_dir = filespace / date / name
    event_dir.mkdir(parents=True)
    if complete:
        (event_dir / "complete").touch()
    if with_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(event_dir / "cam_recording.mp4"), fourcc, 5, (32, 24))
        for _ in range(5):
            writer.write(numpy.zeros((24, 32, 3), dtype=numpy.uint8))
        writer.release()
    if with_detections:
        (event_dir / "detections.jsonl").write_text(
            '{"frame": "cam_0", "boxes": [[0,0,5,5]], "confidences": [0.9], "accepted_mask": [true]}\n'
        )
    os.utime(event_dir, (mtime, mtime))
    return event_dir


def test_health_and_status_ready(configured_env):
    with TestClient(src.service.app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok"}

        status = client.get("/status")
        assert status.status_code == 200
        body = status.json()
        assert body["status"] == "ok"
        assert body["index_size"] == 0
        assert body["last_scan_time"] is not None
        assert body["unclassified_count"] == 0


def test_status_reports_unclassified_count(configured_env):
    filespace = configured_env["filespace"]
    now = time.time()
    _make_event_dir(filespace, "20260101", "N123AB_landing_24_20260101_010203", now - 1000)
    _make_event_dir(filespace, "20260101", "some_weird_name", now - 1000)

    with TestClient(src.service.app) as client:
        body = client.get("/status").json()
        assert body["index_size"] == 2
        assert body["unclassified_count"] == 1


def test_events_filtering_by_registration_and_time_window(configured_env):
    filespace = configured_env["filespace"]
    now = time.time()
    _make_event_dir(filespace, "20260101", "N123AB_landing_24_20260101_010203", now - 1000)
    _make_event_dir(filespace, "20260601", "N999ZZ_takeoff_06_20260601_120000", now - 1000)

    with TestClient(src.service.app) as client:
        by_reg = client.get("/events", params={"registration": "N123"})
        assert by_reg.status_code == 200
        body = by_reg.json()
        assert body["total"] == 1
        assert body["events"][0]["registration"] == "N123AB"

        by_month = client.get("/events", params={"year": 2026, "month": 6})
        body = by_month.json()
        assert body["total"] == 1
        assert body["events"][0]["registration"] == "N999ZZ"

        by_year = client.get("/events", params={"year": 2026})
        body = by_year.json()
        assert body["total"] == 2

        by_exact_with_window = client.get(
            "/events",
            params={
                "year": 2026, "month": 6, "day": 1, "hour": 12, "minute": 0, "second": 0,
                "window_seconds": 60,
            },
        )
        body = by_exact_with_window.json()
        assert body["total"] == 1
        assert body["events"][0]["registration"] == "N999ZZ"


def test_events_window_seconds_requires_second(configured_env):
    with TestClient(src.service.app) as client:
        resp = client.get(
            "/events",
            params={"year": 2026, "month": 6, "day": 1, "hour": 12, "minute": 0, "window_seconds": 60},
        )
        assert resp.status_code == 400


def test_events_time_fields_reject_gaps(configured_env):
    with TestClient(src.service.app) as client:
        resp = client.get("/events", params={"year": 2026, "day": 1})
        assert resp.status_code == 400


def test_events_pagination(configured_env):
    filespace = configured_env["filespace"]
    now = time.time()
    for i in range(5):
        _make_event_dir(filespace, "20260101", f"landing_{i:02d}_20260101_{i:02d}0000", now - 1000)

    with TestClient(src.service.app) as client:
        page = client.get("/events", params={"limit": 2, "offset": 0})
        body = page.json()
        assert body["total"] == 5
        assert len(body["events"]) == 2


def test_get_event_not_found(configured_env):
    with TestClient(src.service.app) as client:
        resp = client.get("/events/999")
        assert resp.status_code == 404


def test_get_event_marks_unsettled_as_in_progress(configured_env, monkeypatch):
    monkeypatch.setenv("EVENT_SETTLE_SECONDS", "120")
    filespace = configured_env["filespace"]
    _make_event_dir(filespace, "20260101", "landing_24_20260101_010203", time.time())

    with TestClient(src.service.app) as client:
        events = client.get("/events").json()["events"]
        assert events[0]["in_progress"] is True
        assert events[0]["status"] == "indexing"


def test_video_and_thumbnail_roundtrip(configured_env):
    filespace = configured_env["filespace"]
    _make_event_dir(
        filespace, "20260101", "landing_24_20260101_010203", time.time() - 1000,
        with_video=True, with_detections=True,
    )

    with TestClient(src.service.app) as client:
        event_id = client.get("/events").json()["events"][0]["id"]

        video_resp = client.get(f"/events/{event_id}/video")
        assert video_resp.status_code == 200
        assert video_resp.headers["content-type"] == "video/mp4"

        thumb_resp = client.get(f"/events/{event_id}/thumbnail")
        assert thumb_resp.status_code == 200
        assert thumb_resp.headers["content-type"] == "image/jpeg"


def test_fresh_start_wipes_db_and_thumbnail_cache(configured_env, monkeypatch):
    filespace = configured_env["filespace"]
    _make_event_dir(
        filespace, "20260101", "landing_24_20260101_010203", time.time() - 1000,
        with_video=True, with_detections=True,
    )

    with TestClient(src.service.app) as client:
        event_id = client.get("/events").json()["events"][0]["id"]
        assert client.get(f"/events/{event_id}/thumbnail").status_code == 200

    index_db_path = os.environ["INDEX_DB_PATH"]
    thumbnail_cache_dir = os.path.dirname(index_db_path) + "/thumbnails"
    assert os.path.exists(index_db_path)
    assert os.path.exists(thumbnail_cache_dir)

    monkeypatch.setenv("FRESH_START", "1")
    with TestClient(src.service.app):
        pass

    assert not os.path.exists(thumbnail_cache_dir)

    monkeypatch.delenv("FRESH_START")
    with TestClient(src.service.app) as client:
        # DB was wiped, so this is rebuilt from the still-present filespace data
        # rather than being empty.
        status = client.get("/status").json()
        assert status["index_size"] == 1


def test_video_404_when_no_recording(configured_env):
    filespace = configured_env["filespace"]
    _make_event_dir(filespace, "20260101", "landing_24_20260101_010203", time.time() - 1000)

    with TestClient(src.service.app) as client:
        event_id = client.get("/events").json()["events"][0]["id"]
        resp = client.get(f"/events/{event_id}/video")
        assert resp.status_code == 404


def test_maintenance_cycle_delegates_to_archive_module(configured_env, monkeypatch):
    calls = []

    def fake_run_maintenance_cycle(
        index, filespace_root, archive_root, age_days, disk_threshold_pct, batch_size, crf,
        debug_cleanup_days, tilt_calibration_cleanup_days,
    ):
        calls.append((filespace_root, archive_root))
        return {
            "status": "ok", "processed": 3, "remaining": 7, "debug_cleaned": 2, "tilt_calibration_deleted": 1,
        }

    monkeypatch.setattr(src.archive, "run_maintenance_cycle", fake_run_maintenance_cycle)

    with TestClient(src.service.app) as client:
        first = client.post("/maintenance_cycle")
        second = client.post("/maintenance_cycle")

    expected = {
        "status": "ok", "processed": 3, "remaining": 7, "debug_cleaned": 2, "tilt_calibration_deleted": 1,
    }
    assert first.status_code == 200
    assert first.json() == expected
    assert second.json() == expected
    assert len(calls) == 2
