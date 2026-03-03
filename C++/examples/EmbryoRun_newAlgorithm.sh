#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CPP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

INPUT_DIR="/Users/wangyiding/CellUniverse/C++/examples/input/C.elegans_developing embryo_Fluo-N3DH-CE_Training/01"
CONFIG_FILE="$CPP_ROOT/examples/config.yaml"
INITIAL_FILE="$SCRIPT_DIR/initial_embryo.csv"
OUTPUT_BASE="/Users/wangyiding/CellUniverse/C++/output"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$OUTPUT_BASE/output_embryo_$STAMP"
LOG_FILE="$OUT_DIR/runlog_$STAMP.txt"

BUILD_DIR="$CPP_ROOT/build"
FALLBACK_BUILD_DIR="$CPP_ROOT/cmake-build-debug"

MIN_FRAME=1
MAX_FRAME=10
MODE="new"

mkdir -p "$OUT_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "*******************************************************************************************************"
echo "Cell Universe Embryo Run (clean + rebuild + run)  [NEW ALGORITHM]"
echo "*******************************************************************************************************"
echo "CPP Root    : $CPP_ROOT"
echo "Build Dir   : $BUILD_DIR"
echo "Input Dir   : $INPUT_DIR"
echo "Initial CSV : $INITIAL_FILE"
echo "Config File : $CONFIG_FILE"
echo "Output Dir  : $OUT_DIR"
echo "Log File    : $LOG_FILE"
echo "Frames      : $MIN_FRAME .. $MAX_FRAME"
echo "Mode        : $MODE"
echo "========================================="

[ -d "$CPP_ROOT" ] || { echo "[FATAL] CPP root not found: $CPP_ROOT"; exit 1; }
[ -d "$INPUT_DIR" ] || { echo "[FATAL] input dir not found: $INPUT_DIR"; exit 1; }
[ -f "$CONFIG_FILE" ] || { echo "[FATAL] config not found: $CONFIG_FILE"; exit 1; }
[ -f "$INITIAL_FILE" ] || { echo "[FATAL] initial csv not found: $INITIAL_FILE"; exit 1; }
[ -f "$CPP_ROOT/CMakeLists.txt" ] || { echo "[FATAL] CMakeLists.txt not found in: $CPP_ROOT"; exit 1; }

echo "[STEP] Cleaning previous build artifacts..."
if [ -d "$BUILD_DIR" ]; then
  echo "  - Removing: $BUILD_DIR"
  rm -rf "$BUILD_DIR"
fi
if [ -d "$FALLBACK_BUILD_DIR" ]; then
  echo "  - Removing: $FALLBACK_BUILD_DIR"
  rm -rf "$FALLBACK_BUILD_DIR"
fi
mkdir -p "$BUILD_DIR"

echo "[STEP] Reconfiguring CMake..."
cmake -S "$CPP_ROOT" -B "$BUILD_DIR"

echo "[STEP] Building..."
cmake --build "$BUILD_DIR" -- -j"$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"

BIN="$BUILD_DIR/celluniverse"
[ -x "$BIN" ] || { echo "[FATAL] build succeeded but binary not found/executable: $BIN"; exit 1; }

echo "[STEP] Sanity-checking input frames..."
fmin="$INPUT_DIR/t$(printf '%03d' "$MIN_FRAME").tif"
fmax="$INPUT_DIR/t$(printf '%03d' "$MAX_FRAME").tif"
[ -f "$fmin" ] || { echo "[FATAL] missing first frame: $fmin"; exit 1; }
[ -f "$fmax" ] || { echo "[FATAL] missing last frame:  $fmax"; exit 1; }

FIRST_FRAME_FILE="$(basename "$fmin")"
export CELLUNIVERSE_INITIAL_FRAME_FILE="$FIRST_FRAME_FILE"
echo "[INFO] CELLUNIVERSE_INITIAL_FRAME_FILE=$CELLUNIVERSE_INITIAL_FRAME_FILE"

echo "[STEP] Running tracker (NEW algorithm)..."
echo "[CMD] $BIN $MIN_FRAME $MAX_FRAME \"$INPUT_DIR/t%03d.tif\" \"$OUT_DIR\" \"$CONFIG_FILE\" \"$INITIAL_FILE\" $MODE"

"$BIN" \
  "$MIN_FRAME" \
  "$MAX_FRAME" \
  "$INPUT_DIR/t%03d.tif" \
  "$OUT_DIR" \
  "$CONFIG_FILE" \
  "$INITIAL_FILE" \
  "$MODE" 2> >(grep -v "TIFF_Warning TIFFReadDirectory: Unknown field with tag 6500" >&2)

echo "========================================="
echo "Run finished (exit=0)."
echo "Results saved to: $OUT_DIR"
echo "Log saved to:     $LOG_FILE"
echo "========================================="