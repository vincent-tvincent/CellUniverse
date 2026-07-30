#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python3}"

"${python_bin}" -m venv "${script_dir}/.venv"
"${script_dir}/.venv/bin/python" -m pip install --upgrade pip
"${script_dir}/.venv/bin/python" -m pip install -r "${script_dir}/requirements.txt"

printf 'Initializer environment is ready: %s\n' "${script_dir}/.venv"
