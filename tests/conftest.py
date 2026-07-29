import pytest


@pytest.fixture
def configured_env(tmp_path, monkeypatch):
    """Point StorageManager's env-based config at fresh tmp_path directories."""
    filespace = tmp_path / "filespace"
    archive_root = tmp_path / "archive"
    filespace.mkdir()
    archive_root.mkdir()

    monkeypatch.setenv("FILESPACE_ROOT", str(filespace))
    monkeypatch.setenv("ARCHIVE_ROOT", str(archive_root))
    monkeypatch.setenv("INDEX_DB_PATH", str(tmp_path / "index" / "storage_manager.db"))
    monkeypatch.setenv("EVENT_SETTLE_SECONDS", "0")
    monkeypatch.setenv("SCAN_INTERVAL_SECONDS", "3600")
    monkeypatch.setenv("ARCHIVE_AGE_DAYS", "30")
    monkeypatch.setenv("ARCHIVE_DISK_THRESHOLD_PCT", "99.9")
    monkeypatch.setenv("MAINTENANCE_BATCH_SIZE", "10")

    return {"filespace": filespace, "archive_root": archive_root}
