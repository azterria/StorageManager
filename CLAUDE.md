# Notes to self on StorageManager

This file is for things I'd otherwise have to re-derive every session: risks
I've spotted but that aren't fixed (because I wasn't asked to), design
tensions, and open questions. Not a spec — README and docstrings already
cover the "how it works." The existing docstrings in this codebase are
unusually good about explaining *why* (dev/ino keying, non-cancelling
shutdown, archived-never-downgrades, atomic archive move) — if I add code,
match that style rather than leaving the reasoning implicit.

## Resolved (2026-08-05)

- **No locking around the shared SQLite connection.** Previously deliberately left
  alone (deployments are single-consumer, idle ~90% of the time) — accepted only up
  to the point a second concurrent writer showed up per deployment. That happened
  when `/rename` was added: its handler runs in a request-thread-pool thread and
  can write to the index at the same time the
  periodic scan thread does, on the same `sqlite3.Connection`. This wasn't
  hypothetical; it reproduced immediately in tests (`cannot commit - no transaction
  is active`). Fixed by wrapping every `Index` method body in a `threading.Lock`
  (`Index._lock`, see `index.py`), since `check_same_thread=False` disables the
  same-thread check but adds no actual synchronization. If another concurrent
  writer path is added later, no further locking work should be needed — it's
  already whole-connection, not per-method.

- **No way to correct an event's registration after the fact.** `/rename` matches
  an event by `event`/`runway`/`date`/`time` (deliberately not by registration, since
  a wrong/partial registration is exactly what it's correcting) and updates
  `registration` in place, physically renaming the event directory to match.
  Renaming an `indexing` event is deferred rather than
  done immediately — the tracker may still hold that directory open by path — so the
  request is queued in a `pending_renames` table and applied once the event settles
  to `local` (`rename.apply_pending_renames`, called at the end of every
  `scanner.scan_once`). `archived` events are rejected outright, consistent with
  archiving being treated as one-way everywhere else in this service.

## Resolved (2026-07-30)

- **Timestampless events (`runway_test`, `calibration`, `stream_reconnect`,
  `unclassified`) being unarchivable forever.** Confirmed with Matthew: in
  production these always land in a dated directory with a real timestamp —
  the no-timestamp path shouldn't actually occur for the kinds we care
  about. Rather than change archive eligibility, `/status` now reports
  `unclassified_count` (see `Index.count_unclassified`) so an unexpected
  buildup is visible instead of silent. If this count is ever nonzero and
  growing, that's the signal something upstream changed.

- **Directory mtime settling assumed new entries, not just data writes.**
  Confirmed: PlaneTracker writes `*_recording.mp4` last, as a re-encode of
  raw output after the event finishes — so appending to an open file
  without touching the directory entry list was a real risk. Fixed by
  requiring a tracker-written `complete` marker file in the event directory
  *in addition to* the mtime-quiet check (`scanner.py`,
  `_COMPLETE_MARKER_NAME`). The mtime check stays as a safety margin after
  the marker appears, in case a marker gets written just before a crash
  mid-copy. This is a two-sided contract: the tracker must actually create
  `complete` as its last write, or events never settle. If events start
  piling up stuck in `indexing` after a tracker change, check that first.

- **Events from before the `complete` marker existed stuck in `indexing`
  forever.** Same two-sided contract as above, but for old data: directories
  written before the marker convention (or by a tracker that crashed before
  writing one) have no marker to wait for. `scan_once` now self-declares
  completion (`_maybe_declare_stale_complete`) once a directory's mtime has
  been quiet for `_STALE_EVENT_AGE_SECONDS` (1 day, hardcoded) with no
  marker present — it writes `complete` itself, with a note in the file
  distinguishing it from a tracker-written one. That write bumps the
  directory's mtime, so the directory still goes through one more
  `settle_seconds` wait afterward on a later scan rather than settling in
  the same pass — deliberately the same lifecycle as a real marker, to avoid
  a directory flipping to `local` and back to `indexing` once the write's
  own mtime bump is observed.

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
