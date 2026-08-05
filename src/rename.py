import logging
import pathlib
import re

import src.index

logger = logging.getLogger(__name__)

# Deliberately conservative: registrations land in a directory name segment
# (`{registration}_{event}_{runway}_{date}_{time}`), so anything that could be
# misread as a path separator or as another segment's delimiter is rejected
# up front rather than trusted to round-trip through parser.parse later.
_REGISTRATION_RE = re.compile(r"^[A-Za-z0-9-]+$")
_DATE_RE = re.compile(r"^\d{8}$")
_TIME_RE = re.compile(r"^\d{6}$")


class RenameError(Exception):
    """Raised for any /rename request that can't be satisfied; message is user-facing."""


def validate_rename_request(registration: str, event: str, runway: str, date: str, time: str) -> None:
    if not _REGISTRATION_RE.match(registration):
        raise RenameError(f"Invalid registration: {registration!r}")
    if not event:
        raise RenameError("event must not be empty")
    if not runway:
        raise RenameError("runway must not be empty")
    if not _DATE_RE.match(date):
        raise RenameError(f"Invalid date (expected yyyymmdd): {date!r}")
    if not _TIME_RE.match(time):
        raise RenameError(f"Invalid time (expected hhmmss): {time!r}")


def _build_dir_name(registration: str, event: str, runway: str, date: str, time: str) -> str:
    return f"{registration}_{event}_{runway}_{date}_{time}"


def apply_rename(index: src.index.Index, row: dict, new_registration: str) -> dict:
    """Rename a settled ('local') event's directory in place and update its index row.

    `pathlib.Path.rename` is a same-filesystem move, so dev/ino — the index's identity
    key — are unchanged; only path, dir_name, and registration need updating.
    """
    old_path = pathlib.Path(row["path"])
    if not old_path.is_dir():
        raise RenameError(f"Event directory missing: {old_path}")

    new_dir_name = _build_dir_name(new_registration, row["event_type"], row["runway"], row["date"], row["time"])
    new_path = old_path.parent / new_dir_name
    if new_path != old_path and new_path.exists():
        raise RenameError(f"Target directory already exists: {new_path}")

    old_path.rename(new_path)
    index.rename_event(row["id"], new_registration, new_dir_name, str(new_path))
    logger.info("Renamed event id=%s %s -> %s", row["id"], old_path, new_path)
    return index.get_event(row["id"])


def apply_pending_renames(index: src.index.Index) -> None:
    """Apply any queued renames whose event has since settled to 'local'.

    Called from scan_once (after it upserts the current pass), so a rename requested
    while an event was still 'indexing' lands as soon as the same scan settles it,
    rather than waiting for a separate trigger.
    """
    for pending in index.get_pending_renames_for_local_events():
        event_id = pending["event_id"]
        row = index.get_event(event_id)
        if row is None or row["status"] != "local":
            continue
        try:
            apply_rename(index, row, pending["new_registration"])
        except Exception:
            logger.exception("Failed to apply pending rename for event id=%s", event_id)
            continue
        index.clear_pending_rename(event_id)