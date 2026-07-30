#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_path="${script_dir}/.venv/bin/python"

if [[ ! -x "${python_path}" ]]; then
  printf 'Initializer environment is missing. Run %s/setup_venv.sh first.\n' "${script_dir}" >&2
  exit 2
fi

exec "${python_path}" "${script_dir}/initialize_frame.py" "$@"
