#!/bin/bash
set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/../.." &> /dev/null && pwd)

STORAGE_MANAGER_PORT="${STORAGE_MANAGER_PORT:-8020}"
LOGS_DIR="${LOGS_DIR:-${PROJECT_ROOT}/logs}"
mkdir -p "${LOGS_DIR}"

FILESPACE_ROOT="${FILESPACE_ROOT:?FILESPACE_ROOT environment variable is required}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:?ARCHIVE_ROOT environment variable is required}"

exec docker run --rm \
    --name "${CONTAINER_NAME:-storage-manager}" \
    --network host \
    --env STORAGE_MANAGER_PORT="${STORAGE_MANAGER_PORT}" \
    --env FILESPACE_ROOT=/data/filespace \
    --env ARCHIVE_ROOT=/data/archive \
    --env INDEX_DB_PATH="${INDEX_DB_PATH:-/data/index/storage_manager.db}" \
    ${ARCHIVE_AGE_DAYS:+-e ARCHIVE_AGE_DAYS="${ARCHIVE_AGE_DAYS}"} \
    ${ARCHIVE_DISK_THRESHOLD_PCT:+-e ARCHIVE_DISK_THRESHOLD_PCT="${ARCHIVE_DISK_THRESHOLD_PCT}"} \
    ${EVENT_SETTLE_SECONDS:+-e EVENT_SETTLE_SECONDS="${EVENT_SETTLE_SECONDS}"} \
    --volume "${FILESPACE_ROOT}:/data/filespace" \
    --volume "${ARCHIVE_ROOT}:/data/archive" \
    --volume "${PROJECT_ROOT}/data/index:/data/index" \
    --volume "${PROJECT_ROOT}/config:/home/fltech/Workspace/StorageManager/config:ro" \
    --volume "${LOGS_DIR}:/home/fltech/Workspace/StorageManager/logs" \
    storage-manager:x86_64
