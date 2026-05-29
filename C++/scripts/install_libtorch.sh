#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/install_libtorch.sh [options]

Detect the current machine and install the most appropriate LibTorch package
under external/libtorch by default.

Options:
  --prefix DIR        Install under DIR/libtorch instead of external/libtorch.
  --target TARGET    auto, cpu, cuda, macos-arm64, or linux-cpu.
  --cuda-tag TAG     CUDA package tag for Linux CUDA builds, e.g. cu128, cu126, cu121.
  --dry-run          Print the selected package without downloading it.
  --resolve-only     Resolve the live package URL without downloading it.
  --force            Replace an existing libtorch directory.
  -h, --help         Show this help.

Auto-selection:
  macOS arm64             -> macOS Apple Silicon LibTorch
  Linux x86_64 + NVIDIA hardware + CUDA toolkit -> Linux CUDA LibTorch
  Linux x86_64 otherwise  -> Linux CPU LibTorch
USAGE
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"

PREFIX="$PROJECT_ROOT/external"
TARGET="auto"
CUDA_TAG=""
FORCE=0
DRY_RUN=0
RESOLVE_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix)
      [[ $# -ge 2 ]] || { echo "error: --prefix requires a directory" >&2; exit 2; }
      PREFIX="$2"
      shift 2
      ;;
    --target)
      [[ $# -ge 2 ]] || { echo "error: --target requires a value" >&2; exit 2; }
      TARGET="$2"
      shift 2
      ;;
    --cuda-tag)
      [[ $# -ge 2 ]] || { echo "error: --cuda-tag requires a value" >&2; exit 2; }
      CUDA_TAG="$2"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --resolve-only)
      RESOLVE_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

require_command() {
  local name="$1"
  if ! command -v "$name" >/dev/null 2>&1; then
    echo "error: required command not found: $name" >&2
    exit 1
  fi
}

download_file() {
  local url="$1"
  local output="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 --retry-delay 2 -o "$output" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$output" "$url"
  else
    echo "error: install curl or wget first" >&2
    exit 1
  fi
}

find_nvcc() {
  if command -v nvcc >/dev/null 2>&1; then
    command -v nvcc
    return 0
  fi

  local candidate
  for candidate in /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc; do
    if [[ -x "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

nvidia_driver_available() {
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1
}

has_nvidia_hardware() {
  if nvidia_driver_available; then
    return 0
  fi
  command -v lspci >/dev/null 2>&1 && lspci | grep -Eiq 'NVIDIA.*(VGA|3D|Display|controller)'
}

has_cuda_toolkit() {
  find_nvcc >/dev/null 2>&1
}

detect_cuda_tag() {
  if [[ -n "$CUDA_TAG" ]]; then
    echo "$CUDA_TAG"
    return
  fi

  local version_line major minor
  local nvcc_bin
  nvcc_bin="$(find_nvcc)"
  version_line="$("$nvcc_bin" --version | sed -n 's/.*release \([0-9][0-9]*\)\.\([0-9][0-9]*\).*/\1 \2/p' | tail -n 1)"
  if [[ -z "$version_line" ]]; then
    echo "cu128"
    return
  fi

  read -r major minor <<<"$version_line"
  if [[ "$major" -gt 12 || ( "$major" -eq 12 && "$minor" -ge 8 ) ]]; then
    echo "cu128"
  elif [[ "$major" -eq 12 && "$minor" -ge 6 ]]; then
    echo "cu126"
  elif [[ "$major" -eq 12 && "$minor" -ge 1 ]]; then
    echo "cu121"
  else
    echo "error: CUDA $major.$minor is too old for the default LibTorch packages; use --target cpu or --cuda-tag manually" >&2
    exit 1
  fi
}

select_latest_from_index() {
  local index_url="$1"
  local regex="$2"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "${index_url} <latest package matching ${regex}>"
    return 0
  fi

  require_command python3
  python3 - "$index_url" "$regex" <<'PY'
import html
import re
import sys
import urllib.request
from urllib.parse import quote, unquote, urljoin

index_url, regex = sys.argv[1], re.compile(sys.argv[2])
with urllib.request.urlopen(index_url, timeout=30) as response:
    page = response.read().decode("utf-8", errors="replace")

raw_matches = re.findall(r'''[^"'<>\s]*libtorch[^"'<>\s]*\.zip''', page)
candidate_map = {}
for raw in raw_matches:
    raw = html.unescape(raw)
    filename = raw.rsplit("/", 1)[-1]
    decoded = unquote(filename)
    candidate_map[decoded] = filename

names = sorted(
    decoded
    for decoded in candidate_map
    if regex.search(decoded) and ".dev" not in decoded
)
if not names:
    preview = "\n  ".join(sorted(candidate_map)[:20])
    detail = f"\nCandidates seen:\n  {preview}" if preview else ""
    raise SystemExit(f"no matching LibTorch packages found at {index_url}{detail}")

def version_key(name):
    match = re.search(r'-(\d+(?:\.\d+){1,3})(?:%2B|\+|\.zip)', name)
    if not match:
        return ()
    return tuple(int(part) for part in match.group(1).split("."))

chosen = max(names, key=version_key)
encoded = candidate_map[chosen]
if "+" in encoded:
    encoded = quote(encoded, safe="-_.~/")
print(urljoin(index_url.rstrip("/") + "/", encoded))
PY
}

detect_target() {
  local os arch
  os="$(uname -s)"
  arch="$(uname -m)"

  if [[ "$os" == "Darwin" && "$arch" == "arm64" ]]; then
    echo "macos-arm64"
    return
  fi

  if [[ "$os" == "Linux" && ( "$arch" == "x86_64" || "$arch" == "amd64" ) ]]; then
    if has_nvidia_hardware && has_cuda_toolkit; then
      echo "cuda"
    else
      echo "linux-cpu"
    fi
    return
  fi

  echo "error: unsupported system: $os $arch" >&2
  exit 1
}

if [[ "$TARGET" == "auto" ]]; then
  TARGET="$(detect_target)"
fi

case "$TARGET" in
  cpu)
    case "$(uname -s)" in
      Darwin) TARGET="macos-arm64" ;;
      Linux) TARGET="linux-cpu" ;;
      *) echo "error: unsupported CPU target for $(uname -s)" >&2; exit 1 ;;
    esac
    ;;
  cuda|linux-cpu|macos-arm64)
    ;;
  *)
    echo "error: unsupported target: $TARGET" >&2
    exit 2
    ;;
esac

case "$TARGET" in
  macos-arm64)
    URL="$(select_latest_from_index \
      "https://download.pytorch.org/libtorch/cpu/" \
      '^libtorch-macos-arm64-[0-9][^/]*\.zip$')"
    DEVICE_HINT="cpu"
    ;;
  linux-cpu)
    URL="$(select_latest_from_index \
      "https://download.pytorch.org/libtorch/cpu/" \
      '^libtorch-cxx11-abi-shared-with-deps-[0-9][^/]*\+cpu\.zip$')"
    DEVICE_HINT="cpu"
    ;;
  cuda)
    if ! has_nvidia_hardware; then
      echo "error: --target cuda requested, but no NVIDIA hardware was detected" >&2
      exit 1
    fi
    if ! has_cuda_toolkit; then
      echo "error: --target cuda requested, but nvcc was not found on PATH or under /usr/local/cuda*" >&2
      exit 1
    fi
    CUDA_TAG="$(detect_cuda_tag)"
    if ! nvidia_driver_available; then
      echo "warning: NVIDIA hardware and nvcc were found, but nvidia-smi cannot talk to the driver." >&2
      echo "warning: CUDA LibTorch will be installed, but GPU runtime will fail until the NVIDIA driver is working." >&2
    fi
    URL="$(select_latest_from_index \
      "https://download.pytorch.org/libtorch/${CUDA_TAG}/" \
      "^libtorch-cxx11-abi-shared-with-deps-[0-9][^/]*\+${CUDA_TAG}\.zip$")"
    DEVICE_HINT="cuda"
    ;;
esac

INSTALL_DIR="$PREFIX/libtorch"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Selected target: $TARGET"
  if [[ "$TARGET" == "cuda" ]]; then
    echo "Selected CUDA package tag: $CUDA_TAG"
    if nvidia_driver_available; then
      echo "NVIDIA driver currently accessible: yes"
    else
      echo "NVIDIA driver currently accessible: no"
    fi
  fi
  echo "Download URL: $URL"
  echo "Install directory: $INSTALL_DIR"
  echo "Suggested n2v2 device: $DEVICE_HINT"
  exit 0
fi

if [[ "$RESOLVE_ONLY" -eq 1 ]]; then
  echo "Selected target: $TARGET"
  if [[ "$TARGET" == "cuda" ]]; then
    echo "Selected CUDA package tag: $CUDA_TAG"
  fi
  echo "Resolved download URL: $URL"
  echo "Install directory: $INSTALL_DIR"
  echo "Suggested n2v2 device: $DEVICE_HINT"
  exit 0
fi

mkdir -p "$PREFIX"

if [[ -e "$INSTALL_DIR" ]]; then
  if [[ "$FORCE" -ne 1 ]]; then
    echo "error: $INSTALL_DIR already exists; pass --force to replace it" >&2
    exit 1
  fi
  rm -rf "$INSTALL_DIR"
fi

TMP_DIR="$(mktemp -d "$PREFIX/.libtorch-install.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT
ZIP_PATH="$TMP_DIR/libtorch.zip"

echo "Selected target: $TARGET"
if [[ "$TARGET" == "cuda" ]]; then
  echo "Selected CUDA package tag: $CUDA_TAG"
  if nvidia_driver_available; then
    echo "NVIDIA driver currently accessible: yes"
  else
    echo "NVIDIA driver currently accessible: no"
  fi
fi
echo "Downloading: $URL"
download_file "$URL" "$ZIP_PATH"

require_command unzip
echo "Extracting to: $PREFIX"
unzip -q "$ZIP_PATH" -d "$TMP_DIR/extract"

if [[ ! -d "$TMP_DIR/extract/libtorch" ]]; then
  echo "error: archive did not contain a libtorch directory" >&2
  exit 1
fi

mv "$TMP_DIR/extract/libtorch" "$INSTALL_DIR"
cat > "$INSTALL_DIR/CELLUNIVERSE_INSTALL.txt" <<EOF
Installed by scripts/install_libtorch.sh
Target: $TARGET
URL: $URL
Suggested n2v2 device: $DEVICE_HINT
EOF

cat <<EOF

LibTorch installed at:
  $INSTALL_DIR

Configure this project with:
  JOBS="\$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)"
  cmake -S . -B build -DCMAKE_PREFIX_PATH="$INSTALL_DIR"
  cmake --build build -j "\$JOBS"

For N2V2 config, use:
  simulation.preprocessing_pipeline: n2v2
  simulation.n2v2_preprocess.device: $DEVICE_HINT
EOF
