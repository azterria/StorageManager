import os
import time

import pytest

import src.index
import src.scanner


@pytest.fixture
def idx(tmp_path):
    index = src.index.Index(str(tmp_path / "index.db"))
    yield index
    index.close()


def _make_event_dir(root, date, name, mtime=None, complete=True):
    event_dir = root / date / name
    event_dir.mkdir(parents=True)
    if complete:
        (event_dir / "complete").touch()
    if mtime is not None:
        os.utime(event_dir, (mtime, mtime))
    return event_dir


def test_scan_finds_and_classifies_settled_event(tmp_path, idx):
    root = tmp_path / "filespace"
    _make_event_dir(root, "20260724", "N123AB_landing_24_20260724_183304", mtime=time.time() - 1000)

    summary = src.scanner.scan_once(root, idx, settle_seconds=120)

    assert summary == {"seen": 1, "settled": 1, "unsettled": 0}
    results, total = idx.query_events()
    assert total == 1
    assert results[0]["registration"] == "N123AB"
    assert results[0]["event_type"] == "landing"
    assert results[0]["status"] == "local"


def test_scan_marks_recent_dir_as_indexing(tmp_path, idx):
    root = tmp_path / "filespace"
    _make_event_dir(root, "20260724", "N123AB_landing_24_20260724_183304", mtime=time.time())

    summary = src.scanner.scan_once(root, idx, settle_seconds=120)

    assert summary == {"seen": 1, "settled": 0, "unsettled": 1}
    results, _ = idx.query_events()
    assert results[0]["status"] == "indexing"


def test_scan_marks_dir_without_complete_marker_as_indexing(tmp_path, idx):
    root = tmp_path / "filespace"
    _make_event_dir(
        root, "20260724", "N123AB_landing_24_20260724_183304",
        mtime=time.time() - 1000, complete=False,
    )

    summary = src.scanner.scan_once(root, idx, settle_seconds=120)

    assert summary == {"seen": 1, "settled": 0, "unsettled": 1}
    results, _ = idx.query_events()
    assert results[0]["status"] == "indexing"


def test_scan_ignores_non_date_dirs_and_files(tmp_path, idx):
    root = tmp_path / "filespace"
    root.mkdir()
    (root / "not_a_date").mkdir()
    (root / "stray_file.txt").write_text("hello")

    summary = src.scanner.scan_once(root, idx, settle_seconds=0)

    assert summary == {"seen": 0, "settled": 0, "unsettled": 0}


def test_scan_missing_root_does_not_crash(tmp_path, idx):
    summary = src.scanner.scan_once(tmp_path / "does_not_exist", idx, settle_seconds=0)
    assert summary == {"seen": 0, "settled": 0, "unsettled": 0}


def test_rename_mid_scan_retains_same_event_id(tmp_path, idx):
    root = tmp_path / "filespace"
    old_mtime = time.time() - 1000
    event_dir = _make_event_dir(root, "20260724", "landing_24_20260724_183304", mtime=old_mtime)

    src.scanner.scan_once(root, idx, settle_seconds=120)
    results, _ = idx.query_events()
    original_id = results[0]["id"]
    assert results[0]["event_type"] == "landing"
    assert results[0]["registration"] is None

    new_dir = event_dir.parent / "touch_and_go_24_20260724_183304"
    event_dir.rename(new_dir)
    os.utime(new_dir, (old_mtime, old_mtime))

    src.scanner.scan_once(root, idx, settle_seconds=120)
    results, total = idx.query_events()

    assert total == 1
    assert results[0]["id"] == original_id
    assert results[0]["event_type"] == "touch_and_go"
    assert results[0]["path"] == str(new_dir)


def test_unsettled_then_settled_transition(tmp_path, idx):
    root = tmp_path / "filespace"
    event_dir = _make_event_dir(root, "20260724", "landing_24_20260724_183304", mtime=time.time())

    src.scanner.scan_once(root, idx, settle_seconds=120)
    results, _ = idx.query_events()
    assert results[0]["status"] == "indexing"
    event_id = results[0]["id"]

    old_mtime = time.time() - 1000
    os.utime(event_dir, (old_mtime, old_mtime))
    src.scanner.scan_once(root, idx, settle_seconds=120)
    results, _ = idx.query_events()
    assert results[0]["id"] == event_id
    assert results[0]["status"] == "local"


def test_scan_never_downgrades_archived_status(tmp_path, idx):
    root = tmp_path / "filespace"
    event_dir = _make_event_dir(root, "20260724", "landing_24_20260724_183304", mtime=time.time() - 1000)
    src.scanner.scan_once(root, idx, settle_seconds=120)
    results, _ = idx.query_events()
    event_id = results[0]["id"]
    idx.set_archived(event_id, "/archive/whatever")

    src.scanner.scan_once(root, idx, settle_seconds=120)
    row = idx.get_event(event_id)
    assert row["status"] == "archived"
