#! /bin/bash -l

python_version="3.13"

eval "$(conda shell.bash hook)"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SCRIPTS_DIR="$(dirname "$SCRIPT_DIR")"
WORKSPACE_DIR="$(dirname "$SCRIPTS_DIR")"

conda create -n StorageManager python=="${python_version}" -y
conda activate StorageManager
pip install -r "${WORKSPACE_DIR}/requirements.txt" -r "${WORKSPACE_DIR}/requirements-dev.txt"

echo "${WORKSPACE_DIR}" > "${CONDA_PREFIX}/lib/python${python_version}/site-packages/develop.pth"
