#!/bin/bash
set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

"${SCRIPT_DIR}/build_docker.sh" "$@"
exec "${SCRIPT_DIR}/run_docker.sh"
