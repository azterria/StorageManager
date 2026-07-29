import src.parser


def test_standard_template_with_registration():
    result = src.parser.parse("N6884H_touch_and_go_24_20260724_183304")
    assert result.registration == "N6884H"
    assert result.event == "touch_and_go"
    assert result.runway == "24"
    assert result.date == "20260724"
    assert result.time == "183304"
    assert result.classified is True


def test_standard_template_no_registration():
    result = src.parser.parse("landing_24_20260724_183304")
    assert result.registration is None
    assert result.event == "landing"
    assert result.runway == "24"
    assert result.date == "20260724"
    assert result.time == "183304"
    assert result.classified is True


def test_all_event_types():
    for event in ("takeoff", "landing", "activity", "manual", "unknown"):
        result = src.parser.parse(f"{event}_06_20260101_120000")
        assert result.event == event
        assert result.runway == "06"


def test_home_watch_event_via_run_track():
    result = src.parser.parse("N123AB_home_watch_06_20260101_120000")
    assert result.registration == "N123AB"
    assert result.event == "home_watch"
    assert result.runway == "06"


def test_home_watcher_background_loop_unlabelled():
    result = src.parser.parse("home_24_20260724_183304")
    assert result.registration is None
    assert result.event == "home_watch"
    assert result.runway == "24"
    assert result.classified is True


def test_home_watcher_background_loop_labelled_after_rename():
    result = src.parser.parse("N6884H_home_24_20260724_183304")
    assert result.registration == "N6884H"
    assert result.event == "home_watch"
    assert result.runway == "24"


def test_tilt_calibration():
    result = src.parser.parse("tilt_calibration_20260724_183304")
    assert result.event == "tilt_calibration"
    assert result.registration is None
    assert result.runway is None
    assert result.date == "20260724"
    assert result.time == "183304"
    assert result.classified is True


def test_bare_literal_run_kinds_use_parent_date():
    for name in ("runway_test", "calibration", "stream_reconnect"):
        result = src.parser.parse(name, parent_date="20260724")
        assert result.event == name
        assert result.date == "20260724"
        assert result.time is None
        assert result.classified is True


def test_unclassified_fallback_never_raises():
    result = src.parser.parse("some_totally_unexpected_name", parent_date="20260724")
    assert result.event == "unclassified"
    assert result.classified is False
    assert result.date == "20260724"
    assert result.registration is None
    assert result.runway is None


def test_registration_with_hyphen():
    result = src.parser.parse("G-ABCD_takeoff_09_20260101_010203")
    assert result.registration == "G-ABCD"
    assert result.event == "takeoff"
