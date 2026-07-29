#!/bin/bash
set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

ARCH=$(uname -m)

case "$ARCH" in
    x86_64)
        exec "${SCRIPT_DIR}/platform/run_x86_64.sh" "$@"
        ;;
    *)
        echo "Unsupported architecture: $ARCH" >&2
        exit 1
        ;;
esac
