# Notes to self on StorageManager

This file is for things I'd otherwise have to re-derive every session: risks
I've spotted but that aren't fixed (because I wasn't asked to), design
tensions, and open questions. Not a spec — README and docstrings already
cover the "how it works." The existing docstrings in this codebase are
unusually good about explaining *why* (dev/ino keying, non-cancelling
shutdown, archived-never-downgrades, atomic archive move) — if I add code,
match that style rather than leaving the reasoning implicit.

## Things that look like bugs but I haven't confirmed are

- **Events without a `timestamp` may be unarchivable forever.**
  `local_events_older_than` and `oldest_local_events` (src/index.py) both
  filter on `timestamp IS NOT NULL`. `timestamp` is only set when *both*
  parsed date and time exist (`_to_timestamp`). Bare-literal run kinds
  (`runway_test`, `calibration`, `stream_reconnect`) get a date but no time,
  so their timestamp is always `None`. Unclassified directories are the same.
  That means these never get selected as archive candidates, not even under
  disk pressure — they'd sit in `local` status and consume filespace
  indefinitely. Worth checking whether this is intentional (maybe these
  directories are expected to be tiny/rare) before treating it as a bug.

- **No locking around the shared SQLite connection.** `Index` opens one
  connection with `check_same_thread=False` and hands it to both the
  periodic scanner (via `asyncio.to_thread`) and every sync FastAPI endpoint
  (which uvicorn/Starlette also runs in a threadpool). Python's `sqlite3`
  doesn't serialize access across threads on your behalf just because the
  thread-check is disabled — concurrent `execute()` calls from different
  threads on the same connection is asking for trouble (this is why the SQLite
  docs steer people toward one-connection-per-thread, or an explicit lock).
  It's probably fine in practice because SQLite operations here are short and
  requests are low-frequency, but if `/maintenance_cycle` is ever called
  concurrently with itself or during a scan, this is the first place I'd look
  for a flaky failure.

- **Directory mtime settling assumes new entries, not just data writes.**
  `scan_once` uses the *directory's* mtime to decide "settled." A directory's
  mtime updates when entries are added/removed/renamed inside it, not when an
  existing file's contents are appended to. If PlaneTracker ever opens
  `*_recording.mp4` once and streams writes into it without touching the
  directory entry list again, `EVENT_SETTLE_SECONDS` could elapse while the
  video is still being written, and the event would look "local"
  (archive-eligible) while incomplete. I don't know PlaneTracker's write
  pattern well enough to say if this is real — flag it if archived videos
  ever turn up truncated.

## Design tensions worth remembering

- `dev`/`ino` as the identity key only works within one filesystem. If
  `FILESPACE_ROOT` ever becomes a union of multiple mounts, or a container
  restart remaps a bind mount such that inodes are reused differently, the
  uniqueness assumption breaks quietly (silent misattribution, not a crash).
- Archive is a one-way move: once `status='archived'`, the index never
  revisits that row (see `upsert_event`'s "never downgrades" comment). If
  something external deletes or moves `archive_path` contents afterward,
  `/events/{id}/video` and `/thumbnail` fail with no self-healing path —
  there's no re-scan of the archive tree, only of `FILESPACE_ROOT`.
- Thumbnails are cached forever once generated (keyed by event id, no TTL,
  no invalidation). If `_select_best_frame`'s heuristic changes, every
  existing cached thumbnail is now "wrong" by the new logic but will never
  regenerate. Any future change to frame-selection should probably come with
  a story for busting the cache (version the filename? wipe the dir?).
- `/maintenance_cycle` is caller-triggered, not scheduled inside the service
  itself — there's no cron/timer calling it. Whatever deploys this is
  presumably responsible for hitting the endpoint periodically. Worth
  confirming that's actually wired up somewhere outside this repo before
  assuming archiving happens at all in production.

## Open questions I'd ask if I were pairing on this

- Is there a reason `PORT` is read at *module import time* in service.py
  (`PORT = int(os.getenv(...))`) while every other config value is
  deliberately deferred to `_load_config()` for testability? Looks like an
  oversight rather than a choice, but it's harmless today since tests don't
  exercise the `__main__` block.
- What actually calls `POST /maintenance_cycle` in production, and how
  often? Not visible from this repo.
