#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CPP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_ROOT="$CPP_ROOT/outputs"

# run_celluniverse.sh -i
# run_celluniverse.sh <preset_config.ini> [preset_name]
#
# Mode 1:
#   ./run_celluniverse.sh /path/to/user_input_configurations.ini preset_name
# Mode 2:
#   ./run_celluniverse.sh -i
#   -> interactive loop that asks for config path and preset.

if [ -t 2 ]; then
  RED="\033[31m"
  GREEN="\033[32m"
  RESET="\033[0m"
else
  RED=""
  GREEN=""
  RESET=""
fi

err() {
  printf "%b%s%b\n" "$RED" "$*" "$RESET" >&2
}

green_text() {
  printf "%b%s%b" "$GREEN" "$1" "$RESET"
}

red_text() {
  printf "%b%s%b" "$RED" "$1" "$RESET"
}

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

expand_home() {
  local p="$1"
  if [[ "$p" == "~/"* ]]; then
    printf '%s/%s\n' "$HOME" "${p#~/}"
  else
    printf '%s\n' "$p"
  fi
}

resolve_path() {
  local p="$1"
  local base="$2"
  p="$(expand_home "$p")"
  if [[ "$p" == /* ]]; then
    printf '%s\n' "$p"
  else
    printf '%s/%s\n' "$base" "$p"
  fi
}

validate_input_path() {
  local p="$1"
  local dir_part=""

  # printf-pattern input like frame%03d.tif: validate its parent directory.
  if [[ "$p" == *%* ]]; then
    dir_part="$(dirname "$p")"
    [ -d "$dir_part" ]
    return
  fi

  # Normal path: can be a directory or a single file.
  [ -e "$p" ]
}

ini_get() {
  local ini_file="$1"
  local section="$2"
  local key="$3"
  awk -v target_section="$section" -v target_key="$key" '
    BEGIN { in_section=0; }
    /^[[:space:]]*[#;]/ { next; }
    /^[[:space:]]*\[/ {
      sec=$0
      gsub(/^[[:space:]]*\[/, "", sec)
      gsub(/\][[:space:]]*$/, "", sec)
      in_section=(sec==target_section)
      next
    }
    in_section {
      split($0, parts, "=")
      if (length(parts) < 2) next
      k=parts[1]
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", k)
      if (k==target_key) {
        v=substr($0, index($0, "=")+1)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", v)
        print v
        exit
      }
    }
  ' "$ini_file"
}

list_presets() {
  local ini_file="$1"
  awk '
    /^[[:space:]]*[#;]/ { next; }
    /^[[:space:]]*\[/ {
      sec=$0
      gsub(/^[[:space:]]*\[/, "", sec)
      gsub(/\][[:space:]]*$/, "", sec)
      if (length(sec) > 0) print sec
    }
  ' "$ini_file"
}

section_has_required_keys() {
  local ini_file="$1"
  local section="$2"
  local required_keys="build_dir input_path output_base_dir output_name_rule cell_config_file initial_csv_file first_frame last_frame"
  local key=""
  for key in $required_keys; do
    if [ -z "$(ini_get "$ini_file" "$section" "$key")" ]; then
      return 1
    fi
  done
  return 0
}

list_valid_presets() {
  local ini_file="$1"
  local section=""
  while IFS= read -r section; do
    [ -n "$section" ] || continue
    if section_has_required_keys "$ini_file" "$section"; then
      printf '%s\n' "$section"
    fi
  done < <(list_presets "$ini_file")
}

is_valid_runauto_ini() {
  local ini_file="$1"
  local meta_app=""
  local section_count=0
  local section=""

  meta_app="$(ini_get "$ini_file" "meta" "app")"
  if [ -n "$meta_app" ] && [ "$meta_app" != "celluniverse-runauto" ]; then
    return 1
  fi

  while IFS= read -r section; do
    [ -n "$section" ] && section_count=$((section_count + 1))
  done < <(list_valid_presets "$ini_file")

  [ "$section_count" -gt 0 ]
}

choose_config_file() {
  local answer=""
  local cfg=""
  local -a ini_candidates=()
  local i=1
  local f=""

  while true; do
    ini_candidates=()
    for f in ./*.ini; do
      [ -f "$f" ] || continue
      ini_candidates+=("${f#./}")
    done

    if [ "${#ini_candidates[@]}" -gt 0 ]; then
      echo "INI files in current directory ($(pwd)):" >&2
      i=1
      for f in "${ini_candidates[@]}"; do
        if is_valid_runauto_ini "$f"; then
          printf "  %s) %s %s\n" "$i" "$(green_text "$f")" "$(green_text "[valid]")" >&2
        else
          printf "  %s) %s %s\n" "$i" "$(red_text "$f")" "$(red_text "[not runauto]")" >&2
        fi
        i=$((i + 1))
      done
    else
      echo "No .ini files found in current directory ($(pwd))." >&2
    fi

    if ! read -r -p "Enter config file number or path (*.ini): " answer; then
      err "[FATAL] no input provided."
      exit 1
    fi

    answer="$(trim "$answer")"
    if [[ "$answer" =~ ^[0-9]+$ ]] && [ "${#ini_candidates[@]}" -gt 0 ]; then
      if [ "$answer" -ge 1 ] && [ "$answer" -le "${#ini_candidates[@]}" ]; then
        cfg="${ini_candidates[$((answer - 1))]}"
      else
        echo "[WARN] invalid file number: $answer" >&2
        continue
      fi
    else
      cfg="$answer"
    fi

    cfg="$(expand_home "$(trim "$cfg")")"
    if [ -f "$cfg" ]; then
      if is_valid_runauto_ini "$cfg"; then
        printf '%s\n' "$cfg"
        return 0
      fi
      echo "[WARN] not a valid runauto INI: $(red_text "$cfg")" >&2
      echo "       It must contain at least one section with required keys." >&2
      continue
    fi

    if [ -n "$cfg" ]; then
      echo "[WARN] file not found: $cfg" >&2
    else
      echo "[WARN] empty input. Please enter a file number or path." >&2
    fi
  done
}

choose_preset() {
  local ini_file="$1"
  local requested="${2:-}"
  local -a presets=()
  local i=1
  local answer=""
  local line=""
  local selected_preset=""

  while IFS= read -r line; do
    [ -n "$line" ] && presets+=("$line")
  done < <(list_valid_presets "$ini_file")
  [ "${#presets[@]}" -gt 0 ] || { err "[FATAL] no valid presets found in $ini_file"; exit 1; }

  if [ -n "$requested" ]; then
    for p in "${presets[@]}"; do
      if [ "$p" = "$requested" ]; then
        printf '%s\n' "$p"
        return 0
      fi
    done
    err "[FATAL] preset not found (or invalid): $requested"
    exit 1
  fi

  if [ -t 0 ] && [ -t 2 ]; then
    if selected_preset="$(choose_preset_arrow_menu presets)"; then
      printf '%s\n' "$selected_preset"
      return 0
    fi
  fi

  show_preset_list presets

  while true; do
    if ! read -r -p "Select preset (name, number, or 'list'; default 1): " answer; then
      err "[FATAL] no input provided."
      exit 1
    fi

    answer="$(trim "$answer")"
    if [ -z "$answer" ]; then
      printf '%s\n' "${presets[0]}"
      return 0
    fi

    if [ "$answer" = "list" ] || [ "$answer" = "l" ] || [ "$answer" = "?" ]; then
      show_preset_list presets
      continue
    fi

    if [[ "$answer" =~ ^[0-9]+$ ]]; then
      if [ "$answer" -ge 1 ] && [ "$answer" -le "${#presets[@]}" ]; then
        printf '%s\n' "${presets[$((answer - 1))]}"
        return 0
      fi
    else
      for p in "${presets[@]}"; do
        if [ "$p" = "$answer" ]; then
          printf '%s\n' "$p"
          return 0
        fi
      done
    fi
    echo "[WARN] invalid preset selection." >&2
  done
}

validate_core_count() {
  local value="$1"
  [[ "$value" =~ ^[0-9]+$ ]] && [ "$value" -gt 0 ]
}

choose_core_count() {
  local requested="${1:-}"
  local answer=""

  if [ -n "$requested" ]; then
    if validate_core_count "$requested"; then
      printf '%s\n' "$requested"
      return 0
    fi
    err "[FATAL] invalid core count: $requested"
    exit 1
  fi

  while true; do
    if ! read -r -p "How many CPU cores to use? (blank = YAML default): " answer; then
      err "[FATAL] no input provided."
      exit 1
    fi

    answer="$(trim "$answer")"
    if [ -z "$answer" ]; then
      printf '\n'
      return 0
    fi
    if validate_core_count "$answer"; then
      printf '%s\n' "$answer"
      return 0
    fi
    echo "[WARN] invalid core count. Enter a positive integer, or blank for YAML default." >&2
  done
}

build_unique_output_dir() {
  local base_dir="$1"
  local name_rule="$2"
  local preset="$3"
  local stamp
  local dir
  local copy_idx=1

  stamp="$(date +%Y%m%d_%H%M%S)"
  name_rule="${name_rule//\{preset\}/$preset}"
  name_rule="${name_rule//\{timestamp\}/$stamp}"
  dir="$base_dir/$name_rule"

  while [ -e "$dir" ]; do
    dir="$base_dir/${name_rule}_copy${copy_idx}"
    copy_idx=$((copy_idx + 1))
  done
  printf '%s\n' "$dir"
}

load_extra_args() {
  local args_file="$1"
  local line=""
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%%#*}"
    line="$(trim "$line")"
    [ -n "$line" ] || continue
    EXTRA_ARGS+=("$line")
  done < "$args_file"
}

get_term_width() {
  local cols=80
  if command -v tput >/dev/null 2>&1; then
    cols="$(tput cols 2>/dev/null || echo 80)"
  fi
  if ! [[ "$cols" =~ ^[0-9]+$ ]]; then
    cols=80
  fi
  if [ "$cols" -lt 40 ]; then
    cols=40
  fi
  printf '%s\n' "$cols"
}

get_term_height() {
  local rows=24
  if command -v tput >/dev/null 2>&1; then
    rows="$(tput lines 2>/dev/null || echo 24)"
  fi
  if ! [[ "$rows" =~ ^[0-9]+$ ]]; then
    rows=24
  fi
  if [ "$rows" -lt 10 ]; then
    rows=10
  fi
  printf '%s\n' "$rows"
}

show_preset_list() {
  local -n preset_ref=$1
  local count="${#preset_ref[@]}"
  local i=1
  local cols rows page_limit

  cols="$(get_term_width)"
  rows="$(get_term_height)"
  page_limit=$((rows - 6))
  if [ "$page_limit" -lt 8 ]; then
    page_limit=8
  fi

  {
    echo "Available presets ($count total):"
    if [ "$count" -gt "$page_limit" ]; then
      echo "Hint: more presets may be below. Type a number/name and Enter, or type 'list' to reopen this list."
    fi
    for p in "${preset_ref[@]}"; do
      printf '  %3d) %s\n' "$i" "$p"
      i=$((i + 1))
    done
  } | fold -s -w "$cols" >&2
}

choose_preset_arrow_menu() {
  local -n preset_ref=$1
  local count="${#preset_ref[@]}"
  local selected=0
  local offset=0
  local rows visible key rest i idx prefix line cols max_name typed_number viewport_cap

  [ "$count" -gt 0 ] || return 1
  [ -r /dev/tty ] && [ -w /dev/tty ] || return 1

  rows="$(get_term_height)"
  cols="$(get_term_width)"
  visible=$((rows - 8))
  viewport_cap=18
  if [ "$visible" -lt 8 ]; then
    visible=8
  fi
  if [ "$visible" -gt "$viewport_cap" ]; then
    visible="$viewport_cap"
  fi
  if [ "$visible" -gt "$count" ]; then
    visible="$count"
  fi
  max_name=$((cols - 10))
  if [ "$max_name" -lt 20 ]; then
    max_name=20
  fi
  typed_number=""

  draw_menu() {
    printf '\033[H\033[J' > /dev/tty
    printf 'Available presets (%s total)\n' "$count" > /dev/tty
    printf 'Keys: Up/Down or j/k = move | PageUp/PageDown = jump | number+Enter = select by number\n' > /dev/tty
    printf '      Enter = select highlighted | Backspace = edit number | q = text prompt\n' > /dev/tty
    printf 'Hint: scroll down to see more presets below.\n\n' > /dev/tty
    i=0
    while [ "$i" -lt "$visible" ]; do
      idx=$((offset + i))
      [ "$idx" -lt "$count" ] || break
      line="${preset_ref[$idx]}"
      if [ "${#line}" -gt "$max_name" ]; then
        line="${line:0:$((max_name - 3))}..."
      fi
      if [ "$idx" -eq "$selected" ]; then
        prefix=">"
      else
        prefix=" "
      fi
      printf '%s %3d) %s\n' "$prefix" "$((idx + 1))" "$line" > /dev/tty
      i=$((i + 1))
    done
    printf '\nShowing %d-%d of %d.\n' "$((offset + 1))" "$((offset + visible))" "$count" > /dev/tty
    if [ -n "$typed_number" ]; then
      printf 'Typed number: %s\n' "$typed_number" > /dev/tty
    fi
  }

  printf '\033[?25l' > /dev/tty
  trap "printf '\033[?25h' > /dev/tty" RETURN
  while true; do
    draw_menu
    IFS= read -rsn1 key < /dev/tty || { printf '\033[?25h' > /dev/tty; trap - RETURN; return 1; }
    case "$key" in
      "")
        if [ -n "$typed_number" ]; then
          if [ "$typed_number" -ge 1 ] && [ "$typed_number" -le "$count" ]; then
            selected=$((typed_number - 1))
            printf '\033[?25h\033[H\033[J' > /dev/tty
            printf '%s\n' "${preset_ref[$selected]}"
            trap - RETURN
            return 0
          fi
          typed_number=""
          continue
        fi
        printf '\033[?25h\033[H\033[J' > /dev/tty
        printf '%s\n' "${preset_ref[$selected]}"
        trap - RETURN
        return 0
        ;;
      q|Q)
        if [ -n "$typed_number" ]; then
          typed_number=""
          continue
        fi
        printf '\033[?25h\033[H\033[J' > /dev/tty
        trap - RETURN
        return 1
        ;;
      $'\177'|$'\b')
        typed_number="${typed_number%?}"
        continue
        ;;
      [0-9])
        typed_number="${typed_number}${key}"
        if [ "$typed_number" -ge 1 ] && [ "$typed_number" -le "$count" ]; then
          selected=$((typed_number - 1))
          if [ "$selected" -lt "$offset" ]; then
            offset="$selected"
          elif [ "$selected" -ge $((offset + visible)) ]; then
            offset=$((selected - visible + 1))
          fi
        fi
        continue
        ;;
      j)
        key="B"
        ;;
      k)
        key="A"
        ;;
      $'\033')
        IFS= read -rsn1 rest < /dev/tty || rest=""
        if [ "$rest" = "[" ]; then
          IFS= read -rsn1 key < /dev/tty || key=""
          if [[ "$key" =~ [0-9] ]]; then
            IFS= read -rsn1 rest < /dev/tty || rest=""
          fi
        fi
        ;;
    esac

    case "$key" in
      A)
        typed_number=""
        [ "$selected" -gt 0 ] && selected=$((selected - 1))
        ;;
      B)
        typed_number=""
        [ "$selected" -lt $((count - 1)) ] && selected=$((selected + 1))
        ;;
      5)
        typed_number=""
        selected=$((selected - visible))
        [ "$selected" -lt 0 ] && selected=0
        ;;
      6)
        typed_number=""
        selected=$((selected + visible))
        [ "$selected" -ge "$count" ] && selected=$((count - 1))
        ;;
    esac

    if [ "$selected" -lt "$offset" ]; then
      offset="$selected"
    elif [ "$selected" -ge $((offset + visible)) ]; then
      offset=$((selected - visible + 1))
    fi
  done
}

hr() {
  local ch="${1:--}"
  local cols
  cols="$(get_term_width)"
  printf '%*s\n' "$cols" "" | tr ' ' "$ch"
}

print_kv() {
  local key="$1"
  local value="$2"
  local label_w=22
  local cols indent wrap_w

  cols="$(get_term_width)"
  indent=$((label_w + 2))
  wrap_w=$((cols - indent))
  if [ "$wrap_w" -lt 20 ]; then
    wrap_w=20
  fi

  printf "%-${label_w}s: " "$key"
  printf '%s\n' "$value" | fold -s -w "$wrap_w" | awk -v ind="$indent" '
    NR == 1 { print; next }
    { printf "%*s%s\n", ind, "", $0 }
  '
}

INTERACTIVE=0
INI_FILE=""
PRESET_ARG=""
CORES_ARG=""

usage() {
  echo "Usage:" >&2
  echo "  $0 -i [--cores N]" >&2
  echo "  $0 <preset_config.ini> [preset_name] [cores]" >&2
  echo "  $0 <preset_config.ini> [preset_name] --cores N" >&2
}

POSITIONAL=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    -i)
      INTERACTIVE=1
      shift
      ;;
    --cores|--threads|-j)
      if [ "$#" -lt 2 ]; then
        err "[FATAL] $1 requires a positive integer argument."
        usage
        exit 1
      fi
      CORES_ARG="$2"
      shift 2
      ;;
    --cores=*|--threads=*)
      CORES_ARG="${1#*=}"
      shift
      ;;
    -*)
      err "[FATAL] unknown option: $1"
      usage
      exit 1
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

if [ "$INTERACTIVE" -eq 1 ]; then
  if [ "${#POSITIONAL[@]}" -ne 0 ]; then
    err "[FATAL] -i does not accept positional arguments."
    usage
    exit 1
  fi
elif [ "${#POSITIONAL[@]}" -eq 0 ]; then
  err "[FATAL] missing arguments."
  usage
  exit 1
elif [ "${#POSITIONAL[@]}" -le 3 ]; then
  INI_FILE="${POSITIONAL[0]}"
  PRESET_ARG="${POSITIONAL[1]:-}"
  if [ "${#POSITIONAL[@]}" -eq 3 ]; then
    if [ -n "$CORES_ARG" ]; then
      err "[FATAL] core count provided both positionally and with --cores."
      usage
      exit 1
    fi
    CORES_ARG="${POSITIONAL[2]}"
  fi
else
  err "[FATAL] invalid arguments."
  usage
  exit 1
fi

if [ -n "$CORES_ARG" ] && ! validate_core_count "$CORES_ARG"; then
  err "[FATAL] invalid core count: $CORES_ARG"
  usage
  exit 1
fi

if [ "$INTERACTIVE" -eq 1 ]; then
  INI_FILE="$(choose_config_file)"
else
  INI_FILE="$(expand_home "$INI_FILE")"
  [ -f "$INI_FILE" ] || { err "[FATAL] preset config not found: $INI_FILE"; exit 1; }
  is_valid_runauto_ini "$INI_FILE" || {
    err "[FATAL] not a valid runauto INI: $INI_FILE"
    echo "        It must contain at least one section with required keys." >&2
    exit 1
  }
fi

INI_DIR="$(cd "$(dirname "$INI_FILE")" && pwd)"
PRESET="$(choose_preset "$INI_FILE" "$PRESET_ARG")"
if [ "$INTERACTIVE" -eq 1 ]; then
  CORES_ARG="$(choose_core_count "$CORES_ARG")"
fi

BUILD_DIR_RAW="$(ini_get "$INI_FILE" "$PRESET" "build_dir")"
INPUT_PATH_RAW="$(ini_get "$INI_FILE" "$PRESET" "input_path")"
OUTPUT_BASE_RAW="$(ini_get "$INI_FILE" "$PRESET" "output_base_dir")"
OUTPUT_RULE="$(ini_get "$INI_FILE" "$PRESET" "output_name_rule")"
CELL_CONFIG_RAW="$(ini_get "$INI_FILE" "$PRESET" "cell_config_file")"
INITIAL_RAW="$(ini_get "$INI_FILE" "$PRESET" "initial_csv_file")"
CLI_ARGS_RAW="$(ini_get "$INI_FILE" "$PRESET" "cli_args_file")"
FIRST_FRAME="$(ini_get "$INI_FILE" "$PRESET" "first_frame")"
LAST_FRAME="$(ini_get "$INI_FILE" "$PRESET" "last_frame")"
# Optional checkpoint-resume fields (2026-04-22). Moved out of config.yaml so
# presets can set resume per-run without editing the YAML. Missing keys →
# resume disabled (celluniverse treats 0 / empty as no-resume).
RESUME_FROM_RAW="$(ini_get "$INI_FILE" "$PRESET" "resume_from")"
RESUME_SRC_RAW="$(ini_get "$INI_FILE" "$PRESET" "resume_source_dir")"

[ -n "$BUILD_DIR_RAW" ] || { err "[FATAL] missing key: build_dir"; exit 1; }
[ -n "$INPUT_PATH_RAW" ] || { err "[FATAL] missing key: input_path"; exit 1; }
[ -n "$OUTPUT_BASE_RAW" ] || { err "[FATAL] missing key: output_base_dir"; exit 1; }
[ -n "$OUTPUT_RULE" ] || { err "[FATAL] missing key: output_name_rule"; exit 1; }
[ -n "$CELL_CONFIG_RAW" ] || { err "[FATAL] missing key: cell_config_file"; exit 1; }
[ -n "$INITIAL_RAW" ] || { err "[FATAL] missing key: initial_csv_file"; exit 1; }
[ -n "$FIRST_FRAME" ] || { err "[FATAL] missing key: first_frame"; exit 1; }
[ -n "$LAST_FRAME" ] || { err "[FATAL] missing key: last_frame"; exit 1; }

BUILD_DIR="$(resolve_path "$BUILD_DIR_RAW" "$INI_DIR")"
INPUT_PATH="$(resolve_path "$INPUT_PATH_RAW" "$INI_DIR")"
OUTPUT_BASE_DIR="$(resolve_path "$OUTPUT_BASE_RAW" "$INI_DIR")"
CELL_CONFIG_FILE="$(resolve_path "$CELL_CONFIG_RAW" "$INI_DIR")"
INITIAL_FILE="$(resolve_path "$INITIAL_RAW" "$INI_DIR")"

CLI_ARGS_FILE=""
if [ -n "$CLI_ARGS_RAW" ]; then
  CLI_ARGS_FILE="$(resolve_path "$CLI_ARGS_RAW" "$INI_DIR")"
  if [ ! -f "$CLI_ARGS_FILE" ]; then
    echo "[WARN] cli_args_file not found, continue without extra args: $CLI_ARGS_FILE" >&2
    CLI_ARGS_FILE=""
  fi
fi

OUT_DIR="$(build_unique_output_dir "$OUTPUT_BASE_DIR" "$OUTPUT_RULE" "$PRESET")"

hr "="
echo "Cell Universe Auto Run"
hr "="
print_kv "Preset selected" "$PRESET"
print_kv "Configuration File" "$INI_FILE"
print_kv "Initial CSV" "$INITIAL_FILE"
print_kv "CLI Args File" "${CLI_ARGS_FILE:-<none>}"
print_kv "Config YAML" "$CELL_CONFIG_FILE"
print_kv "CPU cores" "${CORES_ARG:-YAML default}"
hr "-"
print_kv "Build Path" "$BUILD_DIR"
print_kv "Frames" "$FIRST_FRAME to $LAST_FRAME"
print_kv "Input Path" "$INPUT_PATH"
print_kv "Output Path" "$OUT_DIR"
hr "="
echo

[ -d "$BUILD_DIR" ] || { err "[FATAL] build dir not found: $BUILD_DIR"; exit 1; }
validate_input_path "$INPUT_PATH" || { err "[FATAL] input path invalid or not found: $INPUT_PATH"; exit 1; }
[ -f "$CELL_CONFIG_FILE" ] || { err "[FATAL] config yaml not found: $CELL_CONFIG_FILE"; exit 1; }
[ -f "$INITIAL_FILE" ] || { err "[FATAL] initial csv not found: $INITIAL_FILE"; exit 1; }

mkdir -p "$OUT_DIR"
cp "$INI_FILE" "$OUT_DIR/run_preset_used.ini"
[ -n "$CLI_ARGS_FILE" ] && cp "$CLI_ARGS_FILE" "$OUT_DIR/run_cli_args_used.txt"

EXTRA_ARGS=()
if [ -n "$CLI_ARGS_FILE" ]; then
  load_extra_args "$CLI_ARGS_FILE"
fi

cd "$BUILD_DIR"

CMD=(./celluniverse "$FIRST_FRAME" "$LAST_FRAME" "$INPUT_PATH" "$OUT_DIR" "$CELL_CONFIG_FILE" "$INITIAL_FILE")

# Append optional resume args only when both are set. If resume_source_dir is
# a relative path in the INI, resolve it like the other path fields. A
# resume_from value of 0 (or missing) means no-resume; pass nothing so the
# binary also sees no-resume. If resume_from > 0 we pass both args (source
# dir defaults to empty string, which celluniverse treats as disabled).
RESUME_FROM="${RESUME_FROM_RAW:-0}"
RESUME_SOURCE_DIR=""
if [ -n "$RESUME_SRC_RAW" ]; then
  RESUME_SOURCE_DIR="$(resolve_path "$RESUME_SRC_RAW" "$INI_DIR")"
fi
if [ "$RESUME_FROM" -gt 0 ] 2>/dev/null && [ -n "$RESUME_SOURCE_DIR" ]; then
  CMD+=("$RESUME_FROM" "$RESUME_SOURCE_DIR")
  print_kv "Resume from frame" "$RESUME_FROM"
  print_kv "Resume source dir" "$RESUME_SOURCE_DIR"
fi

if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

echo "[CMD] ${CMD[*]}"
DEBUG_LOG="$OUT_DIR/debug_log.txt"
echo "[CMD] ${CMD[*]}" > "$DEBUG_LOG"
echo "Started: $(date)" >> "$DEBUG_LOG"
echo "CPU cores: ${CORES_ARG:-YAML default}" >> "$DEBUG_LOG"
echo "==========================================" >> "$DEBUG_LOG"
if [ -n "$CORES_ARG" ]; then
  export CELLUNIVERSE_THREADS="$CORES_ARG"
else
  unset CELLUNIVERSE_THREADS
fi
if "${CMD[@]}" > >(tee -a "$DEBUG_LOG") 2> >(grep -Ev "TIFF_Warning TIFFReadDirectory: Unknown field with tag 6500(0|1)?" | tee -a "$DEBUG_LOG" >&2); then
  :
else
  err "[FATAL] celluniverse failed."
  echo "[FATAL] celluniverse failed." >> "$DEBUG_LOG"
  exit 1
fi
echo "==========================================" >> "$DEBUG_LOG"
echo "Finished: $(date)" >> "$DEBUG_LOG"

# # Convert output PNGs to multi-page TIFFs for easy viewing.
# CONVERT_SCRIPT="$SCRIPT_DIR/convert_png_to_tiff.py"
# if [ -f "$CONVERT_SCRIPT" ] && command -v python3 >/dev/null 2>&1; then
#   echo "Converting PNGs to TIFFs..."
#   if [ -d "$OUT_DIR/real" ]; then
#     python3 "$CONVERT_SCRIPT" --i "$OUT_DIR/real" --o "$OUT_DIR/real_tiff" && echo "  real_tiff done." || echo "  [WARN] real_tiff conversion failed."
#   fi
#   if [ -d "$OUT_DIR/synth" ]; then
#     python3 "$CONVERT_SCRIPT" --i "$OUT_DIR/synth" --o "$OUT_DIR/synth_tiff" && echo "  synth_tiff done." || echo "  [WARN] synth_tiff conversion failed."
#   fi
# fi

hr "="
echo "Run finished (exit=0)."
echo "Results saved to:"
echo "$OUT_DIR"
echo "Debug log: $DEBUG_LOG"
hr "="
