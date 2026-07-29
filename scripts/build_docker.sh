#!/bin/bash
set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "${SCRIPT_DIR}/../" || exit

GIT_HASH=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --git-hash) GIT_HASH="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

architecture=$(uname -i)

if [ "${architecture}" = "x86_64" ]; then
  echo "Building for 64 bit desktop (${architecture})"
  docker build -f docker/dockerfile.x86_64 --build-arg UID="$(id -u)" --build-arg GID="$(id -g)" ${GIT_HASH:+--build-arg GIT_HASH="$GIT_HASH"} -t storage-manager:x86_64 .
else
  echo "Unrecognised architecture ${architecture}, building not completed"
fi
