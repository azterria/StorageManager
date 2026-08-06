import datetime

import pytest

import src.index
import src.parser


@pytest.fixture
def idx(tmp_path):
    index = src.index.Index(str(tmp_path / "index.db"))
    yield index
    index.close()


def _parsed(event="landing", registration="N123AB", runway="24", date="20260724", time="183304"):
    return src.parser.ParsedEvent(
        registration=registration, event=event, runway=runway, date=date, time=time, classified=True
    )


def test_insert_new_event(idx):
    event_id = idx.upsert_event(
        dev=1, ino=100, path="/root/20260724/N123AB_landing_24_20260724_183304",
        dir_name="N123AB_landing_24_20260724_183304", parsed=_parsed(), mtime=1.0, status="local",
    )
    row = idx.get_event(event_id)
    assert row["registration"] == "N123AB"
    assert row["event_type"] == "landing"
    assert row["status"] == "local"
    assert row["timestamp"] == "2026-07-24T18:33:04"


def test_upsert_same_inode_updates_in_place(idx):
    first_id = idx.upsert_event(
        dev=1, ino=100, path="/root/20260724/landing_24_20260724_183304",
        dir_name="landing_24_20260724_183304", parsed=_parsed(registration=None), mtime=1.0, status="indexing",
    )
    second_id = idx.upsert_event(
        dev=1, ino=100, path="/root/20260724/touch_and_go_24_20260724_183304",
        dir_name="touch_and_go_24_20260724_183304", parsed=_parsed(event="touch_and_go", registration=None),
        mtime=2.0, status="local",
    )
    assert first_id == second_id
    row = idx.get_event(first_id)
    assert row["event_type"] == "touch_and_go"
    assert row["path"] == "/root/20260724/touch_and_go_24_20260724_183304"
    assert row["status"] == "local"


def test_upsert_never_downgrades_archived(idx):
    event_id = idx.upsert_event(
        dev=1, ino=100, path="/root/x", dir_name="x", parsed=_parsed(), mtime=1.0, status="local",
    )
    idx.set_archived(event_id, "/archive/x")
    idx.upsert_event(
        dev=1, ino=100, path="/root/x", dir_name="x", parsed=_parsed(), mtime=3.0, status="local",
    )
    row = idx.get_event(event_id)
    assert row["status"] == "archived"
    assert row["archive_path"] == "/archive/x"


def test_upsert_never_downgrades_failed(idx):
    event_id = idx.upsert_event(
        dev=1, ino=100, path="/root/x", dir_name="x", parsed=_parsed(), mtime=1.0, status="local",
    )
    idx.set_failed(event_id, "archive")
    idx.upsert_event(
        dev=1, ino=100, path="/root/x", dir_name="x", parsed=_parsed(), mtime=3.0, status="local",
    )
    row = idx.get_event(event_id)
    assert row["status"] == "failed"
    assert row["failure_stage"] == "archive"


def test_query_by_registration_prefix(idx):
    idx.upsert_event(dev=1, ino=1, path="/a", dir_name="a", parsed=_parsed(registration="N123AB"), mtime=1.0, status="local")
    idx.upsert_event(dev=1, ino=2, path="/b", dir_name="b", parsed=_parsed(registration="N999ZZ"), mtime=1.0, status="local")
    results, total = idx.query_events(registration="N123")
    assert total == 1
    assert results[0]["registration"] == "N123AB"


def test_query_by_time_window(idx):
    idx.upsert_event(dev=1, ino=1, path="/a", dir_name="a", parsed=_parsed(date="20260101", time="120000"), mtime=1.0, status="local")
    idx.upsert_event(dev=1, ino=2, path="/b", dir_name="b", parsed=_parsed(date="20260601", time="120000"), mtime=1.0, status="local")
    results, total = idx.query_events(since="2026-03-01T00:00:00", until="2026-12-31T00:00:00")
    assert total == 1
    assert results[0]["date"] == "20260601"


def test_query_pagination(idx):
    for i in range(5):
        idx.upsert_event(
            dev=1, ino=i, path=f"/{i}", dir_name=str(i),
            parsed=_parsed(date="20260101", time=f"{i:02d}0000"), mtime=1.0, status="local",
        )
    results, total = idx.query_events(limit=2, offset=0)
    assert total == 5
    assert len(results) == 2


def test_local_events_older_than(idx):
    idx.upsert_event(dev=1, ino=1, path="/a", dir_name="a", parsed=_parsed(date="20260101", time="120000"), mtime=1.0, status="local")
    idx.upsert_event(dev=1, ino=2, path="/b", dir_name="b", parsed=_parsed(date="20260601", time="120000"), mtime=1.0, status="local")
    old = idx.local_events_older_than("2026-03-01T00:00:00", limit=10)
    assert len(old) == 1
    assert old[0]["date"] == "20260101"


def test_local_events_older_than_excludes_indexing_and_archived(idx):
    idx.upsert_event(dev=1, ino=1, path="/a", dir_name="a", parsed=_parsed(date="20260101", time="120000"), mtime=1.0, status="indexing")
    archived_id = idx.upsert_event(dev=1, ino=2, path="/b", dir_name="b", parsed=_parsed(date="20260101", time="120000"), mtime=1.0, status="local")
    idx.set_archived(archived_id, "/archive/b")
    old = idx.local_events_older_than("2026-03-01T00:00:00", limit=10)
    assert old == []


def test_count_by_status(idx):
    idx.upsert_event(dev=1, ino=1, path="/a", dir_name="a", parsed=_parsed(), mtime=1.0, status="local")
    idx.upsert_event(dev=1, ino=2, path="/b", dir_name="b", parsed=_parsed(), mtime=1.0, status="indexing")
    counts = idx.count_by_status()
    assert counts == {"local": 1, "indexing": 1}


def test_count_unclassified(idx):
    idx.upsert_event(dev=1, ino=1, path="/a", dir_name="a", parsed=_parsed(), mtime=1.0, status="local")
    idx.upsert_event(
        dev=1, ino=2, path="/b", dir_name="b",
        parsed=src.parser.ParsedEvent(
            registration=None, event="unclassified", runway=None, date=None, time=None, classified=False,
        ),
        mtime=1.0, status="local",
    )
    assert idx.count_unclassified() == 1


def test_find_events_by_identity(idx):
    idx.upsert_event(dev=1, ino=1, path="/a", dir_name="a", parsed=_parsed(registration="705"), mtime=1.0, status="local")
    idx.upsert_event(dev=1, ino=2, path="/b", dir_name="b", parsed=_parsed(registration="N999ZZ", runway="06"), mtime=1.0, status="local")

    matches = idx.find_events_by_identity("landing", "24", "20260724", "183304")
    assert len(matches) == 1
    assert matches[0]["registration"] == "705"

    assert idx.find_events_by_identity("takeoff", "24", "20260724", "183304") == []


def test_rename_event_updates_registration_path_and_dir_name(idx):
    event_id = idx.upsert_event(
        dev=1, ino=1, path="/root/20260724/705_landing_24_20260724_183304",
        dir_name="705_landing_24_20260724_183304", parsed=_parsed(registration="705"), mtime=1.0, status="local",
    )
    idx.rename_event(
        event_id, "N72705",
        "N72705_landing_24_20260724_183304", "/root/20260724/N72705_landing_24_20260724_183304",
    )
    row = idx.get_event(event_id)
    assert row["registration"] == "N72705"
    assert row["dir_name"] == "N72705_landing_24_20260724_183304"
    assert row["path"] == "/root/20260724/N72705_landing_24_20260724_183304"


def test_queue_rename_upserts_and_only_surfaces_for_local_events(idx):
    indexing_id = idx.upsert_event(dev=1, ino=1, path="/a", dir_name="a", parsed=_parsed(registration="705"), mtime=1.0, status="indexing")
    local_id = idx.upsert_event(dev=1, ino=2, path="/b", dir_name="b", parsed=_parsed(registration="706"), mtime=1.0, status="local")

    idx.queue_rename(indexing_id, "N72705")
    idx.queue_rename(local_id, "N72706")
    assert idx.get_pending_renames_for_local_events() == [{"event_id": local_id, "new_registration": "N72706"}]

    idx.queue_rename(indexing_id, "N72705X")  # overwrite, not stack
    idx.clear_pending_rename(local_id)
    idx.upsert_event(dev=1, ino=1, path="/a", dir_name="a", parsed=_parsed(registration="705"), mtime=1.0, status="local")
    assert idx.get_pending_renames_for_local_events() == [{"event_id": indexing_id, "new_registration": "N72705X"}]


def test_meta_roundtrip(idx):
    assert idx.get_meta("last_scan_time") is None
    idx.set_meta("last_scan_time", "2026-07-29T00:00:00+00:00")
    assert idx.get_meta("last_scan_time") == "2026-07-29T00:00:00+00:00"
    idx.set_meta("last_scan_time", "2026-07-29T01:00:00+00:00")
    assert idx.get_meta("last_scan_time") == "2026-07-29T01:00:00+00:00"
