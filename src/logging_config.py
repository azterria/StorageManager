import json
import logging.config
import pathlib

_CONFIG_PATH = pathlib.Path(__file__).resolve().parent.parent / "config" / "logging.json"
_LOG_DIR = pathlib.Path("logs")


def setup():
    _LOG_DIR.mkdir(exist_ok=True)
    with _CONFIG_PATH.open() as f:
        config = json.load(f)
    logging.config.dictConfig(config)
