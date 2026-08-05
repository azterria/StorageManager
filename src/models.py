import pydantic


class EventSummary(pydantic.BaseModel):
    id: int
    date: str | None
    time: str | None
    registration: str | None
    event: str
    runway: str | None
    status: str
    in_progress: bool


class EventList(pydantic.BaseModel):
    events: list[EventSummary]
    total: int
    limit: int
    offset: int


class StatusResponse(pydantic.BaseModel):
    status: str
    index_size: int
    last_scan_time: str | None
    archive_backlog: int
    unclassified_count: int


class MaintenanceCycleResponse(pydantic.BaseModel):
    status: str
    processed: int
    remaining: int
    debug_cleaned: int
    tilt_calibration_deleted: int


class RenameRequest(pydantic.BaseModel):
    registration: str
    event: str
    runway: str
    date: str
    time: str


class RenameResponse(pydantic.BaseModel):
    status: str
    event: EventSummary
