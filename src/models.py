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


class MaintenanceCycleResponse(pydantic.BaseModel):
    status: str
    processed: int
    remaining: int
