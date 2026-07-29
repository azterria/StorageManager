import dataclasses
import logging
import re

logger = logging.getLogger(__name__)

# Event tokens as they literally appear in PlaneTracker directory names. Longer/more
# specific tokens are listed first only for readability — regex alternation here is
# unambiguous since each token is anchored immediately before "_{runway}_{date}_{time}".
_EVENT_TOKENS = (
    "touch_and_go",
    "home_watch",
    "takeoff",
    "landing",
    "activity",
    "manual",
    "unknown",
    "home",
)

# {registration_}{event}_{runway}_{yyyymmdd}_{hhmmss}, per TrackingController._run_track
# and the touch-and-go / home-watch renames.
_STANDARD_RE = re.compile(
    r"^(?:(?P<registration>.+)_)?"
    r"(?P<event>" + "|".join(_EVENT_TOKENS) + r")"
    r"_(?P<runway>[^_]+)_(?P<date>\d{8})_(?P<time>\d{6})$"
)

# tilt_calibration_{yyyymmdd}_{hhmmss}, no registration/runway.
_TILT_CALIBRATION_RE = re.compile(r"^tilt_calibration_(?P<date>\d{8})_(?P<time>\d{6})$")

# Non-template run kinds with no timestamp/runway in the directory name itself.
_BARE_LITERALS = frozenset({"runway_test", "calibration", "stream_reconnect"})

# The home watcher's background loop names its directories "home_{runway}_{stem}" (not
# "home_watch_..."), so normalize the bare "home" token to the EventType value.
_EVENT_NORMALIZE = {"home": "home_watch"}


@dataclasses.dataclass(frozen=True)
class ParsedEvent:
    registration: str | None
    event: str
    runway: str | None
    date: str | None
    time: str | None
    classified: bool


def parse(dir_name: str, parent_date: str | None = None) -> ParsedEvent:
    """Parse a PlaneTracker run-directory name into structured fields.

    `parent_date` is the enclosing `{yyyymmdd}` directory name, used as a fallback for
    run kinds whose own name carries no date (and for the unclassified fallback).
    Never raises — unrecognised names are returned as `event="unclassified"` rather
    than rejected, since PlaneTracker directory names are not a closed set.
    """
    match = _STANDARD_RE.match(dir_name)
    if match:
        event = _EVENT_NORMALIZE.get(match.group("event"), match.group("event"))
        return ParsedEvent(
            registration=match.group("registration"),
            event=event,
            runway=match.group("runway"),
            date=match.group("date"),
            time=match.group("time"),
            classified=True,
        )

    match = _TILT_CALIBRATION_RE.match(dir_name)
    if match:
        return ParsedEvent(
            registration=None,
            event="tilt_calibration",
            runway=None,
            date=match.group("date"),
            time=match.group("time"),
            classified=True,
        )

    if dir_name in _BARE_LITERALS:
        return ParsedEvent(
            registration=None,
            event=dir_name,
            runway=None,
            date=parent_date,
            time=None,
            classified=True,
        )

    logger.warning("Unclassified directory name: %s", dir_name)
    return ParsedEvent(
        registration=None,
        event="unclassified",
        runway=None,
        date=parent_date,
        time=None,
        classified=False,
    )
