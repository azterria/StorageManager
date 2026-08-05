import pytest

import src.index
import src.parser
import src.rename


@pytest.fixture
def idx(tmp_path):
    index = src.index.Index(str(tmp_path / "index.db"))
    yield index
    index.close()


def _parsed(event="landing", registration="705", runway="24", date="20260724", time="183304"):
    return src.parser.ParsedEvent(
        registration=registration, event=event, runway=runway, date=date, time=time, classified=True
    )


@pytest.mark.parametrize(
    "registration,event,runway,date,time,message_fragment",
    [
        ("N72 705", "landing", "24", "20260724", "183304", "registration"),
        ("N72705", "", "24", "20260724", "183304", "event"),
        ("N72705", "landing", "", "20260724", "183304", "runway"),
        ("N72705", "landing", "24", "2026724", "183304", "date"),
        ("N72705", "landing", "24", "20260724", "18330", "time"),
    ],
)
def test_validate_rename_request_rejects_bad_input(registration, event, runway, date, time, message_fragment):
    with pytest.raises(src.rename.RenameError, match=message_fragment):
        src.rename.validate_rename_request(registration, event, runway, date, time)


def test_validate_rename_request_accepts_good_input():
    src.rename.validate_rename_request("N72705", "landing", "24", "20260724", "183304")


def test_apply_rename_moves_directory_and_updates_index(tmp_path, idx):
    event_dir = tmp_path / "20260724" / "705_landing_24_20260724_183304"
    event_dir.mkdir(parents=True)
    (event_dir / "cam_recording.mp4").write_text("video")

    event_id = idx.upsert_event(
        dev=1, ino=1, path=str(event_dir), dir_name=event_dir.name,
        parsed=_parsed(), mtime=1.0, status="local",
    )
    row = idx.get_event(event_id)

    updated = src.rename.apply_rename(idx, row, "N72705")

    new_dir = event_dir.parent / "N72705_landing_24_20260724_183304"
    assert not event_dir.exists()
    assert new_dir.is_dir()
    assert (new_dir / "cam_recording.mp4").exists()
    assert updated["registration"] == "N72705"
    assert updated["dir_name"] == new_dir.name
    assert updated["path"] == str(new_dir)


def test_apply_rename_raises_if_source_missing(tmp_path, idx):
    event_dir = tmp_path / "20260724" / "705_landing_24_20260724_183304"
    event_id = idx.upsert_event(
        dev=1, ino=1, path=str(event_dir), dir_name=event_dir.name,
        parsed=_parsed(), mtime=1.0, status="local",
    )
    row = idx.get_event(event_id)

    with pytest.raises(src.rename.RenameError, match="missing"):
        src.rename.apply_rename(idx, row, "N72705")


def test_apply_rename_raises_if_target_already_exists(tmp_path, idx):
    event_dir = tmp_path / "20260724" / "705_landing_24_20260724_183304"
    event_dir.mkdir(parents=True)
    (event_dir.parent / "N72705_landing_24_20260724_183304").mkdir()

    event_id = idx.upsert_event(
        dev=1, ino=1, path=str(event_dir), dir_name=event_dir.name,
        parsed=_parsed(), mtime=1.0, status="local",
    )
    row = idx.get_event(event_id)

    with pytest.raises(src.rename.RenameError, match="already exists"):
        src.rename.apply_rename(idx, row, "N72705")


def test_apply_pending_renames_applies_only_to_local_events(tmp_path, idx):
    indexing_dir = tmp_path / "20260724" / "705_landing_24_20260724_183304"
    indexing_dir.mkdir(parents=True)
    local_dir = tmp_path / "20260724" / "706_takeoff_24_20260724_183304"
    local_dir.mkdir(parents=True)

    indexing_id = idx.upsert_event(
        dev=1, ino=1, path=str(indexing_dir), dir_name=indexing_dir.name,
        parsed=_parsed(registration="705"), mtime=1.0, status="indexing",
    )
    local_id = idx.upsert_event(
        dev=1, ino=2, path=str(local_dir), dir_name=local_dir.name,
        parsed=_parsed(event="takeoff", registration="706"), mtime=1.0, status="local",
    )
    idx.queue_rename(indexing_id, "N72705")
    idx.queue_rename(local_id, "N72706")

    src.rename.apply_pending_renames(idx)

    assert indexing_dir.exists()  # untouched: still 'indexing'
    assert idx.get_event(indexing_id)["registration"] == "705"
    assert idx.get_pending_renames_for_local_events() == []  # local one cleared

    local_row = idx.get_event(local_id)
    assert local_row["registration"] == "N72706"
    assert not local_dir.exists()
    assert (tmp_path / "20260724" / "N72706_takeoff_24_20260724_183304").is_dir()
