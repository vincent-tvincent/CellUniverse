#!/usr/bin/env python3
"""Run standalone Cell Lumen in a true rolling prior loop and audit centers.

This experimental helper is intentionally separate from the main tracking
pipeline. It starts from one initial CSV, then uses each frame's own exported
Cell Lumen result as the next frame prior. Ground truth is read only after each
frame is written, only for auditing miss/extra behavior.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import shlex
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/Users/wangyiding/CellUniverse")
DEFAULT_PYTHON = Path(
    "/Users/wangyiding/.cache/codex-runtimes/"
    "codex-primary-runtime/dependencies/python/bin/python3"
)
DEFAULT_EXE = ROOT / "C++/build/celluniverse"
DEFAULT_EXPORT = ROOT / "C++/scripts/export_fine_shape_masks_from_cells.py"
DEFAULT_INPUT_DIR = ROOT / (
    "C++/examples/input/"
    "C.elegans_developing embryo_Fluo-N3DH-CE_Training/01"
)
DEFAULT_CONFIG = ROOT / (
    "C++/config/C.elegans developing embryo/"
    "Concentrated/C_elegans_DensityAuto_Best.yaml"
)
DEFAULT_INITIAL = ROOT / (
    "C++/config/C.elegans developing embryo/"
    "C.elegans_initial/initial_files/00_core_start_points/initial_embryo_0.csv"
)
DEFAULT_GT = ROOT / (
    "C++/config/C.elegans developing embryo/"
    "C.elegans_initial/ground_truth/embryo_FixedGroundTruth.csv"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/Volumes/T9/🦠Cell Universe/🟣Output/"
    "CellLumen_AllRun_07062026/📁Rolling_CenterAudit"
)

BIOPHYSICS_PROFILE_HEADER = "standalone_cell_lumen_biophysics:"


def _parse_profile_scalar(text: str):
    value = text.strip()
    if not value:
        return ""
    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"null", "none", "~"}:
        return None
    if (value.startswith("'") and value.endswith("'")) or (
        value.startswith('"') and value.endswith('"')
    ):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_standalone_biophysics_profile(path: Path) -> dict[str, object]:
    """Read the flat standalone profile without adding a PyYAML dependency."""
    profile: dict[str, object] = {}
    in_section = False
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        stripped = raw_line.strip()
        if not in_section:
            if raw_line == BIOPHYSICS_PROFILE_HEADER:
                in_section = True
            continue
        if stripped and not raw_line.startswith((" ", "\t")):
            break
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if indent != 2 or ":" not in stripped:
            raise ValueError(
                f"Malformed {BIOPHYSICS_PROFILE_HEADER} entry at {path}:{line_number}"
            )
        key, value = stripped.split(":", 1)
        profile[key.strip()] = _parse_profile_scalar(value)
    return profile


def profile_value(profile: dict[str, object], key: str, default):
    value = profile.get(key, default)
    if isinstance(default, bool) and not isinstance(value, bool):
        raise TypeError(f"Biophysics profile key {key} must be boolean")
    if isinstance(default, int) and not isinstance(default, bool):
        if not isinstance(value, int):
            raise TypeError(f"Biophysics profile key {key} must be integer")
    if isinstance(default, float) and not isinstance(value, (int, float)):
        raise TypeError(f"Biophysics profile key {key} must be numeric")
    return value


def parse_frame_spec(text: str) -> list[int]:
    frames: list[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = chunk.split("-", 1)
            frames.extend(range(int(start), int(end) + 1))
        else:
            frames.append(int(chunk))
    return sorted(set(frames))


def read_gt(path: Path, frames: list[int]) -> dict[int, list[dict[str, float | str]]]:
    wanted = set(frames)
    by_frame: dict[int, list[dict[str, float | str]]] = {frame: [] for frame in frames}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            frame = int(row["frame"])
            if frame not in wanted:
                continue
            by_frame[frame].append(
                {
                    "label": row["label_id"],
                    "parent": row.get("parent_label", ""),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "z": float(row["z_interp"]),
                }
            )
    return by_frame


def distance(left: dict[str, float | str], right: dict[str, float | str]) -> float:
    return math.sqrt(
        (float(left["x"]) - float(right["x"])) ** 2
        + (float(left["y"]) - float(right["y"])) ** 2
        + (float(left["z"]) - float(right["z"])) ** 2
    )


def read_final_summary(path: Path) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "name": row["cell_name"],
                    "x": float(row["center_x"]),
                    "y": float(row["center_y"]),
                    "z": float(row["center_z_scaled"]),
                    "rx": float(row["radius_x"]),
                    "ry": float(row["radius_y"]),
                    "rz": float(row["radius_z_scaled"]),
                    "voxels": int(float(row.get("voxels") or 0)),
                    "mean_intensity": float(row.get("mean_intensity") or 0.0),
                }
            )
    return rows


def read_raw_candidates(path: Path, z_scale: float) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "name": row["name"],
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "z": float(row["z"]) * z_scale,
                }
            )
    return rows


def read_prior(path: Path, z_scale: float) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if "center_x" in row:
                rows.append(
                    {
                        "name": row.get("cell_name", ""),
                        "x": float(row["center_x"]),
                        "y": float(row["center_y"]),
                        "z": float(row["center_z_scaled"]),
                    }
                )
            elif "file" in row and "x" in row and "z" in row:
                rows.append(
                    {
                        "name": row.get("name", ""),
                        "x": float(row["x"]),
                        "y": float(row["y"]),
                        "z": float(row["z"]) * z_scale,
                    }
                )
            elif "cellType" in row and "x" in row and "z" in row:
                rows.append(
                    {
                        "name": row.get("cellType", ""),
                        "x": float(row["x"]),
                        "y": float(row["y"]),
                        "z": float(row["z"]) * z_scale,
                    }
                )
    return rows


def greedy_match(gt_rows, pred_rows, threshold: float):
    # Keep the historical function name for imports, but use maximum bipartite
    # matching instead of nearest-pair greediness. Close daughter pairs often
    # have two predictions within threshold of two GT labels; greedy nearest can
    # consume the shared best prediction first and report a false missing cell.
    edges: list[list[tuple[float, int]]] = []
    for gt in gt_rows:
        gt_edges = []
        for pi, pred in enumerate(pred_rows):
            dist = distance(gt, pred)
            if dist <= threshold:
                gt_edges.append((dist, pi))
        gt_edges.sort(key=lambda item: item[0])
        edges.append(gt_edges)

    match_pred_to_gt: dict[int, int] = {}

    def augment(gi: int, seen_pred: set[int]) -> bool:
        for _dist, pi in edges[gi]:
            if pi in seen_pred:
                continue
            seen_pred.add(pi)
            if pi not in match_pred_to_gt or augment(match_pred_to_gt[pi], seen_pred):
                match_pred_to_gt[pi] = gi
                return True
        return False

    gt_order = sorted(range(len(gt_rows)), key=lambda gi: (len(edges[gi]), gi))
    for gi in gt_order:
        augment(gi, set())

    used_gt: set[int] = set(match_pred_to_gt.values())
    used_pred: set[int] = set(match_pred_to_gt.keys())
    matches = [
        (distance(gt_rows[gi], pred_rows[pi]), gt_rows[gi], pred_rows[pi])
        for pi, gi in match_pred_to_gt.items()
    ]
    matches.sort(key=lambda item: item[0])

    missing = [gt_rows[i] for i in range(len(gt_rows)) if i not in used_gt]
    extra = [pred_rows[i] for i in range(len(pred_rows)) if i not in used_pred]
    return matches, missing, extra


def nearest(row, rows):
    if not rows:
        return "", float("inf")
    best = min(rows, key=lambda item: distance(row, item))
    return best.get("name") or best.get("label") or "", distance(row, best)


def write_prior_from_summary(
    summary_path: Path,
    prior_path: Path,
    frame: int,
    z_scale: float,
) -> Path:
    rows = read_final_summary(summary_path)
    prior_path.parent.mkdir(parents=True, exist_ok=True)
    with prior_path.open("w", newline="") as handle:
        fieldnames = ["file", "name", "x", "y", "z", "majorRadius", "bRadius", "minorRadius"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "file": f"t{frame:03d}.tif",
                    "name": row["name"],
                    "x": f"{float(row['x']):.9g}",
                    "y": f"{float(row['y']):.9g}",
                    "z": f"{float(row['z']) / z_scale:.9g}",
                    "majorRadius": f"{float(row['rx']):.9g}",
                    "bRadius": f"{float(row['ry']):.9g}",
                    "minorRadius": f"{float(row['rz']) / z_scale:.9g}",
                }
            )
    return prior_path


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    return str(value)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(data), indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(json_safe(data), sort_keys=True) + "\n")


def command_text(command: list[str]) -> str:
    return shlex.join([str(token) for token in command])


def capture_command(command: list[str], timeout_sec: int = 20) -> dict[str, str | int]:
    try:
        proc = subprocess.run(
            command,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        return {
            "command": command,
            "returncode": proc.returncode,
            "output": proc.stdout.strip(),
        }
    except Exception as exc:
        return {"command": command, "returncode": -1, "output": repr(exc)}


def git_metadata() -> dict:
    return {
        "head": capture_command(["git", "rev-parse", "HEAD"]),
        "branch": capture_command(["git", "branch", "--show-current"]),
        "status_short": capture_command(["git", "status", "--short"]),
        "diff_stat": capture_command(["git", "diff", "--stat"]),
    }


def disk_metadata(path: Path) -> dict[str, int | str]:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    }


def file_metadata(path: Path) -> dict[str, int | str | bool]:
    if not path.exists():
        return {"path": str(path), "exists": False, "size_bytes": 0}
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def count_final_summary(path: Path) -> int:
    return len(read_final_summary(path))


def replace_option_value(command: list[str], option: str, value: str) -> list[str]:
    updated = list(command)
    for index, token in enumerate(updated):
        if token == option and index + 1 < len(updated):
            updated[index + 1] = value
            return updated
    updated.extend([option, value])
    return updated


def remove_options(
    command: list[str],
    flags: set[str],
    valued: set[str],
) -> list[str]:
    cleaned: list[str] = []
    skip_next = False
    for token in command:
        if skip_next:
            skip_next = False
            continue
        if token in flags:
            continue
        if token in valued:
            skip_next = True
            continue
        cleaned.append(token)
    return cleaned


def main() -> int:
    config_probe = argparse.ArgumentParser(add_help=False)
    config_probe.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    config_probe_args, _ = config_probe.parse_known_args()
    biophysics_profile = load_standalone_biophysics_profile(config_probe_args.config)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", default="1-60")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--config", type=Path, default=config_probe_args.config)
    parser.add_argument("--initial", type=Path, default=DEFAULT_INITIAL)
    parser.add_argument("--gt-csv", type=Path, default=DEFAULT_GT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--executable", type=Path, default=DEFAULT_EXE)
    parser.add_argument("--export-script", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument(
        "--threads",
        type=int,
        default=profile_value(biophysics_profile, "rolling_threads", 8),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=profile_value(biophysics_profile, "rolling_threshold", 25.0),
    )
    parser.add_argument(
        "--z-scale",
        type=float,
        default=profile_value(biophysics_profile, "rolling_z_scale", 7.0),
    )
    parser.add_argument("--unique-initial-celltype-names", action="store_true")
    parser.add_argument("--preserve-rolling-parent-names", action="store_true")
    parser.add_argument(
        "--no-prefuse",
        action="store_true",
        default=profile_value(biophysics_profile, "rolling_no_prefuse", False),
    )
    parser.add_argument(
        "--ultrack-style",
        action="store_true",
        default=profile_value(biophysics_profile, "rolling_ultrack_style", False),
    )
    parser.add_argument("--stop-on-fail", action="store_true")
    parser.add_argument(
        "--skip-fine-shape-runs-csv",
        action="store_true",
        help=(
            "Pass --skip-runs-csv to the fine shape exporter so long rolling "
            "audits keep summary/log/prior files without writing large RLE "
            "cache CSVs."
        ),
    )
    parser.add_argument(
        "--prior-reach-distance",
        type=float,
        default=profile_value(biophysics_profile, "rolling_prior_reach_distance", 50.0),
    )
    parser.add_argument(
        "--prior-dynamic-reach-from-radius",
        action="store_true",
        default=profile_value(
            biophysics_profile,
            "rolling_prior_dynamic_reach_from_radius",
            False,
        ),
    )
    parser.add_argument(
        "--prior-dynamic-reach-radius-scale",
        type=float,
        default=profile_value(
            biophysics_profile,
            "rolling_prior_dynamic_reach_radius_scale",
            1.25,
        ),
    )
    parser.add_argument(
        "--prior-dynamic-reach-margin",
        type=float,
        default=profile_value(
            biophysics_profile,
            "rolling_prior_dynamic_reach_margin",
            0.0,
        ),
    )
    parser.add_argument(
        "--prior-parent-region-max-norm",
        type=float,
        default=profile_value(
            biophysics_profile,
            "rolling_prior_parent_region_max_norm",
            2.2,
        ),
    )
    parser.add_argument("--prior-allow-multi-parent-candidates", action="store_true")
    parser.add_argument("--prior-multi-parent-max-assignments", type=int, default=None)
    parser.add_argument("--prior-multi-parent-distance-slack", type=float, default=None)
    parser.add_argument(
        "--prior-split-min-separation",
        type=float,
        default=profile_value(
            biophysics_profile,
            "rolling_prior_split_min_separation",
            27.0,
        ),
    )
    parser.add_argument(
        "--prior-root-parent-split-min-separation",
        type=float,
        default=profile_value(
            biophysics_profile,
            "rolling_prior_root_parent_split_min_separation",
            37.0,
        ),
    )
    parser.add_argument(
        "--prior-split-max-volume-ratio",
        type=float,
        default=profile_value(
            biophysics_profile,
            "rolling_prior_split_max_volume_ratio",
            8.0,
        ),
    )
    parser.add_argument("--prior-max-candidates-per-parent", type=int, default=None)
    parser.add_argument("--prior-same-parent-duplicate-distance", type=float, default=None)
    parser.add_argument("--prior-duplicate-radius-fraction", type=float, default=None)
    parser.add_argument(
        "--fuse-representative-mode",
        choices=["largest", "centroid", "medoid"],
        default=profile_value(
            biophysics_profile,
            "rolling_fuse_representative_mode",
            "largest",
        ),
    )
    parser.add_argument(
        "--fuse-preserve-split-hypotheses",
        action="store_true",
        default=profile_value(
            biophysics_profile,
            "rolling_fuse_preserve_split_hypotheses",
            False,
        ),
    )
    parser.add_argument("--fuse-preserve-pair-min-separation", type=float, default=None)
    parser.add_argument("--fuse-preserve-pair-max-valley-ratio", type=float, default=None)
    parser.add_argument("--fuse-preserve-require-future-support", action="store_true")
    parser.add_argument("--fuse-preserve-require-distinct-future-support", action="store_true")
    parser.add_argument("--fuse-preserve-min-future-support", type=float, default=None)
    parser.add_argument("--fuse-preserve-z-column-alternatives", action="store_true")
    parser.add_argument("--fuse-z-column-alternative-min-dz", type=float, default=None)
    parser.add_argument("--fuse-z-column-alternative-lateral", type=float, default=None)
    parser.add_argument("--fuse-z-column-alternative-max-extra-per-group", type=int, default=None)
    parser.add_argument("--fuse-preserve-representative-alternatives", action="store_true")
    parser.add_argument("--fuse-representative-alternative-min-distance", type=float, default=None)
    parser.add_argument("--fuse-representative-alternative-max-extra-per-group", type=int, default=None)
    parser.add_argument("--fuse-split-large-z-span-groups", action="store_true")
    parser.add_argument("--fuse-max-group-z-span", type=float, default=None)
    parser.add_argument("--fuse-z-split-gap", type=float, default=None)
    parser.add_argument(
        "--prior-soft-division-penalty",
        type=float,
        default=biophysics_profile.get("rolling_prior_soft_division_penalty"),
    )
    parser.add_argument(
        "--prior-soft-split-margin",
        type=float,
        default=biophysics_profile.get("rolling_prior_soft_split_margin"),
    )
    parser.add_argument("--prior-soft-root-split-shortfall-penalty", type=float, default=None)
    parser.add_argument("--prior-bridge-target-valley-ratio", type=float, default=None)
    parser.add_argument("--prior-bridge-bonus-weight", type=float, default=None)
    parser.add_argument("--prior-bridge-penalty-weight", type=float, default=None)
    parser.add_argument("--prior-preserve-z-column-alternatives", action="store_true")
    parser.add_argument("--prior-z-column-alternative-min-dz", type=float, default=None)
    parser.add_argument("--prior-z-column-alternative-lateral", type=float, default=None)
    parser.add_argument("--prior-preserve-single-z-alt-runner-up", action="store_true")
    parser.add_argument("--prior-single-z-alt-runner-up-max-score-gap", type=float, default=None)
    parser.add_argument("--prior-preserve-representative-alternatives", action="store_true")
    parser.add_argument("--prior-representative-alternative-min-distance", type=float, default=None)
    parser.add_argument("--prior-representative-alternative-max-score-gap", type=float, default=None)
    parser.add_argument("--prior-unresolved-close-split-duplicate", action="store_true")
    parser.add_argument("--prior-unresolved-close-split-max-sep", type=float, default=None)
    parser.add_argument("--prior-unresolved-close-split-max-valley", type=float, default=None)
    parser.add_argument("--prior-duplicate-single-on-future-split-pressure", action="store_true")
    parser.add_argument("--prior-single-future-split-duplicate-min-pressure", type=float, default=None)
    parser.add_argument("--prior-collapse-coordinate-copies", action="store_true")
    parser.add_argument("--prior-coordinate-copy-epsilon", type=float, default=None)
    parser.add_argument("--prior-coordinate-copy-keep-unresolved", action="store_true")
    parser.add_argument("--prior-coordinate-copy-keep-future", action="store_true")
    parser.add_argument("--prior-coordinate-copy-max-future-kept", type=int, default=None)
    parser.add_argument("--prior-coordinate-copy-future-projection-separation", type=float, default=None)
    parser.add_argument("--prior-close-bright-bridge-penalty-weight", type=float, default=None)
    parser.add_argument("--prior-close-bright-bridge-target-valley", type=float, default=None)
    parser.add_argument("--prior-close-bright-bridge-close-sep", type=float, default=None)
    parser.add_argument(
        "--delayed-future-support",
        action="store_true",
        default=profile_value(
            biophysics_profile,
            "rolling_delayed_future_support",
            False,
        ),
        help=(
            "Experimental one-frame delayed commit: use the current frame's "
            "provisional Cell Lumen result to probe the next frame, then "
            "re-export the current frame with that next-frame raw candidate "
            "set as temporal evidence. Ground truth is still used only for audit."
        ),
    )
    parser.add_argument("--prior-future-support-weight", type=float, default=None)
    parser.add_argument("--prior-pair-future-support-weight", type=float, default=None)
    parser.add_argument("--prior-pair-future-distinct-bonus", type=float, default=None)
    parser.add_argument("--prior-pair-future-same-penalty", type=float, default=None)
    parser.add_argument("--prior-pair-require-distinct-future-support", action="store_true")
    parser.add_argument("--prior-pair-future-required-penalty", type=float, default=None)
    parser.add_argument("--prior-pair-preserved-split-allow-shared-future", action="store_true")
    parser.add_argument(
        "--prior-pair-preserved-shared-future-penalty-scale",
        type=float,
        default=None,
    )
    parser.add_argument("--prior-single-future-pressure-penalty", type=float, default=None)
    parser.add_argument("--prior-future-split-pressure-distance", type=float, default=None)
    parser.add_argument("--prior-future-split-pressure-min-separation", type=float, default=None)
    parser.add_argument("--prior-future-motion-balance-weight", type=float, default=None)
    parser.add_argument("--prior-future-motion-balance-tolerance", type=float, default=None)
    parser.add_argument("--prior-future-motion-balance-single-only", action="store_true")
    parser.add_argument("--prior-soft-pair-parent-volume-ratio-target", type=float, default=None)
    parser.add_argument("--prior-soft-pair-parent-volume-ratio-penalty", type=float, default=None)
    parser.add_argument("--prior-global-target-count", type=int, default=None)
    parser.add_argument(
        "--auto-global-count-budget",
        action="store_true",
        default=profile_value(
            biophysics_profile,
            "rolling_auto_global_count_budget",
            False,
        ),
        help=(
            "Experimental two-pass budget: first export a conservative centroid "
            "solution without preserved split hypotheses, then allow preserved "
            "splits only up to a data-driven global count target."
        ),
    )
    args = parser.parse_args()

    frames = parse_frame_spec(args.frames)
    stamp = time.strftime("%Y%m%d.%H%M%S")
    mode = "UltrackStyle" if args.ultrack_style else "BioFuseV5"
    if args.no_prefuse:
        mode += "_NoPreFuse"
    run_name = args.run_name or f"⚠️EXPERIMENT_{frames[0]:03d}-{frames[-1]:03d}_TrueRolling_{mode}_{stamp}"
    run_dir = args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    gt_by_frame = read_gt(args.gt_csv, frames)
    prior_path = args.initial
    prior_frame_name = ""
    summary_rows: list[dict] = []
    failure_rows: list[dict] = []

    env = os.environ.copy()
    env["CELLUNIVERSE_CELL_LUMEN_SKIP_TIFF"] = "1"
    env["CELLUNIVERSE_THREADS"] = str(args.threads)

    manifest_path = run_dir / "RUN_MANIFEST.json"
    frame_manifest_path = run_dir / "FRAME_EXECUTION_MANIFEST.jsonl"
    commands_path = run_dir / "REPRODUCE_COMMANDS.sh"
    effective_env = {
        key: env.get(key, "")
        for key in (
            "CELLUNIVERSE_THREADS",
            "CELLUNIVERSE_SEED",
            "CELLUNIVERSE_CELL_LUMEN_SKIP_TIFF",
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
        )
    }
    top_command = [sys.executable, *sys.argv]
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "schema": "cell_lumen_true_rolling_audit_manifest_v1",
        "purpose": (
            "Full reproducibility manifest for standalone Cell Lumen rolling "
            "center audit. Ground truth is used only after frame output for "
            "miss/extra audit."
        ),
        "run_dir": run_dir,
        "frames": frames,
        "top_level_command": top_command,
        "top_level_command_text": command_text(top_command),
        "args": vars(args),
        "standalone_cell_lumen_biophysics": biophysics_profile,
        "input_files": {
            "config": file_metadata(args.config),
            "initial": file_metadata(args.initial),
            "ground_truth_audit_only": file_metadata(args.gt_csv),
            "executable": file_metadata(args.executable),
            "rolling_script": file_metadata(Path(__file__).resolve()),
            "export_script": file_metadata(args.export_script),
        },
        "root": ROOT,
        "python_executable": sys.executable,
        "platform": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "git": git_metadata(),
        "environment": effective_env,
        "output_policy": {
            "cell_lumen_tiff_disabled": env["CELLUNIVERSE_CELL_LUMEN_SKIP_TIFF"] == "1",
            "fine_shape_runs_csv_retained": not args.skip_fine_shape_runs_csv,
            "label_tiff_requested": False,
            "visualization_regeneration_data": (
                "Each frame keeps raw CellLumen CSV, BioFuse fine shape summary, "
                "rolling prior CSV, logs, and, unless --skip-fine-shape-runs-csv "
                "is used, fine_shape_runs.csv RLE rows that can regenerate label TIFFs."
            ),
        },
        "disk_before": {
            "output_root": disk_metadata(args.output_root),
            "run_dir_parent": disk_metadata(run_dir.parent),
            "workspace": disk_metadata(ROOT),
        },
    }
    write_json(manifest_path, manifest)
    commands_path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"cd {shlex.quote(str(ROOT))}\n"
        f"export CELLUNIVERSE_THREADS={shlex.quote(str(args.threads))}\n"
        "export CELLUNIVERSE_CELL_LUMEN_SKIP_TIFF=1\n"
        f"{command_text(top_command)}\n"
    )
    commands_path.chmod(0o755)
    print(f"RUN_MANIFEST={manifest_path}", flush=True)
    print(f"FRAME_MANIFEST={frame_manifest_path}", flush=True)

    for frame in frames:
        frame_start = time.perf_counter()
        frame_dir = run_dir / f"f{frame:03d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        image = args.input_dir / f"t{frame:03d}.tif"
        raw_csv = frame_dir / f"t{frame:03d}_cell_lumen_independent.csv"
        cell_log = frame_dir / "cell_lumen.log"
        fine_dir = frame_dir / "fine_shape_biofused"
        bio_log = frame_dir / "biofuse.log"
        summary_csv = fine_dir / f"t{frame:03d}_fine_shape_summary.csv"
        frame_record = {
            "frame": frame,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            "paths": {
                "frame_dir": frame_dir,
                "image": image,
                "raw_csv": raw_csv,
                "cell_log": cell_log,
                "fine_dir": fine_dir,
                "bio_log": bio_log,
                "summary_csv": summary_csv,
                "prior_input": prior_path,
            },
            "commands": {},
            "processes": {},
            "outputs": {},
            "audit": {},
        }

        cell_cmd = [
            str(args.executable),
            "--cell-lumen",
            str(image),
            str(frame_dir),
            str(args.config),
            str(raw_csv),
            str(prior_path),
        ]
        frame_record["commands"]["cell_lumen"] = {
            "argv": cell_cmd,
            "text": command_text(cell_cmd),
        }
        cell_start = time.perf_counter()
        with cell_log.open("w") as log:
            cell_proc = subprocess.run(
                cell_cmd,
                cwd=str(ROOT),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        cell_sec = time.perf_counter() - cell_start
        frame_record["processes"]["cell_lumen"] = {
            "returncode": cell_proc.returncode,
            "seconds": cell_sec,
            "raw_csv_exists": raw_csv.exists(),
            "log": cell_log,
        }
        if cell_proc.returncode != 0 or not raw_csv.exists():
            frame_record["status"] = "RUN_FAIL_CELL_LUMEN"
            frame_record["disk_after_frame"] = disk_metadata(run_dir.parent)
            append_jsonl(frame_manifest_path, frame_record)
            print(f"f{frame:03d} RUN_FAIL cell_lumen return={cell_proc.returncode}", flush=True)
            return 2

        export_cmd = [
            str(args.python),
            str(args.export_script),
            "--image",
            str(image),
            "--cells",
            str(raw_csv),
            "--output-dir",
            str(fine_dir),
            "--frame-name",
            f"t{frame:03d}.tif",
            "--filter-frame",
            "--z-scaling",
            str(args.z_scale),
            "--z-mode",
            "raw",
            "--radius-z-mode",
            "scaled",
            "--biological-prior-cells",
            str(prior_path),
            "--biological-prior-z-mode",
            "auto",
            "--biological-prior-radius-z-mode",
            "auto",
            "--prior-reach-distance",
            str(args.prior_reach_distance),
            "--prior-parent-region-max-norm",
            str(args.prior_parent_region_max_norm),
            "--prior-split-min-separation",
            str(args.prior_split_min_separation),
            "--prior-root-parent-split-min-separation",
            str(args.prior_root_parent_split_min_separation),
            "--prior-split-max-volume-ratio",
            str(args.prior_split_max_volume_ratio),
        ]
        if args.skip_fine_shape_runs_csv:
            export_cmd.append("--skip-runs-csv")
        if args.prior_same_parent_duplicate_distance is not None:
            export_cmd.extend(
                [
                    "--prior-same-parent-duplicate-distance",
                    str(args.prior_same_parent_duplicate_distance),
                ]
            )
        if args.prior_duplicate_radius_fraction is not None:
            export_cmd.extend(
                [
                    "--prior-duplicate-radius-fraction",
                    str(args.prior_duplicate_radius_fraction),
                ]
            )
        if args.prior_dynamic_reach_from_radius:
            export_cmd.extend(
                [
                    "--prior-dynamic-reach-from-radius",
                    "--prior-dynamic-reach-radius-scale",
                    str(args.prior_dynamic_reach_radius_scale),
                    "--prior-dynamic-reach-margin",
                    str(args.prior_dynamic_reach_margin),
                ]
            )
        if not args.no_prefuse:
            export_cmd.extend(
                [
                    "--fuse-biological-duplicates",
                    "--fuse-z-column-lateral",
                    "16",
                    "--fuse-z-column-max-dz",
                    "30",
                    "--fuse-min-scaled-distance",
                    "20",
                    "--fuse-representative-mode",
                    args.fuse_representative_mode,
                ]
            )
            if args.fuse_preserve_split_hypotheses:
                export_cmd.append("--fuse-preserve-split-hypotheses")
            if args.fuse_preserve_pair_min_separation is not None:
                export_cmd.extend(
                    [
                        "--fuse-preserve-pair-min-separation",
                        str(args.fuse_preserve_pair_min_separation),
                    ]
                )
            if args.fuse_preserve_pair_max_valley_ratio is not None:
                export_cmd.extend(
                    [
                        "--fuse-preserve-pair-max-valley-ratio",
                        str(args.fuse_preserve_pair_max_valley_ratio),
                    ]
                )
            if args.fuse_preserve_require_future_support:
                export_cmd.append("--fuse-preserve-require-future-support")
            if args.fuse_preserve_require_distinct_future_support:
                export_cmd.append("--fuse-preserve-require-distinct-future-support")
            if args.fuse_preserve_min_future_support is not None:
                export_cmd.extend(
                    [
                        "--fuse-preserve-min-future-support",
                        str(args.fuse_preserve_min_future_support),
                    ]
                )
            if args.fuse_preserve_z_column_alternatives:
                export_cmd.append("--fuse-preserve-z-column-alternatives")
            if args.fuse_z_column_alternative_min_dz is not None:
                export_cmd.extend(
                    [
                        "--fuse-z-column-alternative-min-dz",
                        str(args.fuse_z_column_alternative_min_dz),
                    ]
                )
            if args.fuse_z_column_alternative_lateral is not None:
                export_cmd.extend(
                    [
                        "--fuse-z-column-alternative-lateral",
                        str(args.fuse_z_column_alternative_lateral),
                    ]
                )
            if args.fuse_z_column_alternative_max_extra_per_group is not None:
                export_cmd.extend(
                    [
                        "--fuse-z-column-alternative-max-extra-per-group",
                        str(args.fuse_z_column_alternative_max_extra_per_group),
                    ]
                )
            if args.fuse_preserve_representative_alternatives:
                export_cmd.append("--fuse-preserve-representative-alternatives")
            if args.fuse_representative_alternative_min_distance is not None:
                export_cmd.extend(
                    [
                        "--fuse-representative-alternative-min-distance",
                        str(args.fuse_representative_alternative_min_distance),
                    ]
                )
            if args.fuse_representative_alternative_max_extra_per_group is not None:
                export_cmd.extend(
                    [
                        "--fuse-representative-alternative-max-extra-per-group",
                        str(args.fuse_representative_alternative_max_extra_per_group),
                    ]
                )
            if args.fuse_split_large_z_span_groups:
                export_cmd.append("--fuse-split-large-z-span-groups")
            if args.fuse_max_group_z_span is not None:
                export_cmd.extend(["--fuse-max-group-z-span", str(args.fuse_max_group_z_span)])
            if args.fuse_z_split_gap is not None:
                export_cmd.extend(["--fuse-z-split-gap", str(args.fuse_z_split_gap)])
        if args.ultrack_style:
            export_cmd.append("--prior-ultrack-style-selection")
        optional_soft_args = {
            "--prior-max-candidates-per-parent": args.prior_max_candidates_per_parent,
            "--prior-multi-parent-max-assignments": args.prior_multi_parent_max_assignments,
            "--prior-multi-parent-distance-slack": args.prior_multi_parent_distance_slack,
            "--prior-soft-division-penalty": args.prior_soft_division_penalty,
            "--prior-soft-split-margin": args.prior_soft_split_margin,
            "--prior-bridge-target-valley-ratio": args.prior_bridge_target_valley_ratio,
            "--prior-bridge-bonus-weight": args.prior_bridge_bonus_weight,
            "--prior-bridge-penalty-weight": args.prior_bridge_penalty_weight,
            "--prior-z-column-alternative-min-dz": args.prior_z_column_alternative_min_dz,
            "--prior-z-column-alternative-lateral": args.prior_z_column_alternative_lateral,
            "--prior-single-z-alt-runner-up-max-score-gap": (
                args.prior_single_z_alt_runner_up_max_score_gap
            ),
            "--prior-representative-alternative-min-distance": (
                args.prior_representative_alternative_min_distance
            ),
            "--prior-representative-alternative-max-score-gap": (
                args.prior_representative_alternative_max_score_gap
            ),
            "--prior-unresolved-close-split-max-sep": (
                args.prior_unresolved_close_split_max_sep
            ),
            "--prior-unresolved-close-split-max-valley": (
                args.prior_unresolved_close_split_max_valley
            ),
            "--prior-single-future-split-duplicate-min-pressure": (
                args.prior_single_future_split_duplicate_min_pressure
            ),
            "--prior-coordinate-copy-epsilon": args.prior_coordinate_copy_epsilon,
            "--prior-coordinate-copy-max-future-kept": (
                args.prior_coordinate_copy_max_future_kept
            ),
            "--prior-coordinate-copy-future-projection-separation": (
                args.prior_coordinate_copy_future_projection_separation
            ),
            "--prior-close-bright-bridge-penalty-weight": (
                args.prior_close_bright_bridge_penalty_weight
            ),
            "--prior-close-bright-bridge-target-valley": (
                args.prior_close_bright_bridge_target_valley
            ),
            "--prior-close-bright-bridge-close-sep": (
                args.prior_close_bright_bridge_close_sep
            ),
            "--prior-soft-root-split-shortfall-penalty": args.prior_soft_root_split_shortfall_penalty,
            "--prior-future-support-weight": args.prior_future_support_weight,
            "--prior-pair-future-support-weight": args.prior_pair_future_support_weight,
            "--prior-pair-future-distinct-bonus": args.prior_pair_future_distinct_bonus,
            "--prior-pair-future-same-penalty": args.prior_pair_future_same_penalty,
            "--prior-pair-future-required-penalty": args.prior_pair_future_required_penalty,
            "--prior-pair-preserved-shared-future-penalty-scale": (
                args.prior_pair_preserved_shared_future_penalty_scale
            ),
            "--prior-single-future-pressure-penalty": args.prior_single_future_pressure_penalty,
            "--prior-future-split-pressure-distance": args.prior_future_split_pressure_distance,
            "--prior-future-split-pressure-min-separation": args.prior_future_split_pressure_min_separation,
            "--prior-future-motion-balance-weight": args.prior_future_motion_balance_weight,
            "--prior-future-motion-balance-tolerance": args.prior_future_motion_balance_tolerance,
            "--prior-soft-pair-parent-volume-ratio-target": (
                args.prior_soft_pair_parent_volume_ratio_target
            ),
            "--prior-soft-pair-parent-volume-ratio-penalty": (
                args.prior_soft_pair_parent_volume_ratio_penalty
            ),
            "--prior-global-target-count": args.prior_global_target_count,
        }
        for option, value in optional_soft_args.items():
            if value is not None:
                export_cmd.extend([option, str(value)])
        if args.prior_preserve_z_column_alternatives:
            export_cmd.append("--prior-preserve-z-column-alternatives")
        if args.prior_allow_multi_parent_candidates:
            export_cmd.append("--prior-allow-multi-parent-candidates")
        if args.prior_preserve_single_z_alt_runner_up:
            export_cmd.append("--prior-preserve-single-z-alt-runner-up")
        if args.prior_preserve_representative_alternatives:
            export_cmd.append("--prior-preserve-representative-alternatives")
        if args.prior_unresolved_close_split_duplicate:
            export_cmd.append("--prior-unresolved-close-split-duplicate")
        if args.prior_duplicate_single_on_future_split_pressure:
            export_cmd.append("--prior-duplicate-single-on-future-split-pressure")
        if args.prior_collapse_coordinate_copies:
            export_cmd.append("--prior-collapse-coordinate-copies")
        if args.prior_coordinate_copy_keep_unresolved:
            export_cmd.append("--prior-coordinate-copy-keep-unresolved")
        if args.prior_coordinate_copy_keep_future:
            export_cmd.append("--prior-coordinate-copy-keep-future")
        if args.prior_pair_require_distinct_future_support:
            export_cmd.append("--prior-pair-require-distinct-future-support")
        if args.prior_pair_preserved_split_allow_shared_future:
            export_cmd.append("--prior-pair-preserved-split-allow-shared-future")
        if args.prior_future_motion_balance_single_only:
            export_cmd.append("--prior-future-motion-balance-single-only")
        if args.unique_initial_celltype_names:
            export_cmd.append("--unique-initial-celltype-names")
        if args.preserve_rolling_parent_names:
            export_cmd.append("--preserve-rolling-parent-names")
        if prior_frame_name:
            export_cmd.extend(["--biological-prior-frame-name", prior_frame_name])
        frame_record["commands"]["biofuse_first_export"] = {
            "argv": export_cmd,
            "text": command_text(export_cmd),
            "fine_shape_runs_csv_expected": not args.skip_fine_shape_runs_csv,
        }

        bio_start = time.perf_counter()
        with bio_log.open("w") as log:
            bio_proc = subprocess.run(
                export_cmd,
                cwd=str(ROOT),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        bio_sec = time.perf_counter() - bio_start
        runs_csv = fine_dir / f"t{frame:03d}_fine_shape_runs.csv"
        frame_record["processes"]["biofuse_first_export"] = {
            "returncode": bio_proc.returncode,
            "seconds": bio_sec,
            "summary_exists": summary_csv.exists(),
            "runs_csv": file_metadata(runs_csv),
            "log": bio_log,
        }
        if bio_proc.returncode != 0 or not summary_csv.exists():
            frame_record["status"] = "RUN_FAIL_BIOFUSE"
            frame_record["disk_after_frame"] = disk_metadata(run_dir.parent)
            append_jsonl(frame_manifest_path, frame_record)
            print(f"f{frame:03d} RUN_FAIL biofuse return={bio_proc.returncode}", flush=True)
            return 3
        future_probe_sec = 0.0
        if args.delayed_future_support and frame < frames[-1]:
            # This is the experimental Cell Lumen version of a tiny Ultrack-like
            # rolling window. The provisional current-frame output is used only
            # to probe next-frame brightness candidates; GT is still withheld
            # until after the final current-frame export below.
            firstpass_summary = frame_dir / f"firstpass_t{frame:03d}_fine_shape_summary.csv"
            firstpass_prior = frame_dir / f"firstpass_prior_t{frame:03d}_for_future_probe.csv"
            shutil.copy2(summary_csv, firstpass_summary)
            if bio_log.exists():
                shutil.move(str(bio_log), str(frame_dir / "biofuse_firstpass.log"))
            write_prior_from_summary(summary_csv, firstpass_prior, frame, args.z_scale)

            future_frame = frame + 1
            future_dir = frame_dir / f"future_probe_f{future_frame:03d}"
            future_dir.mkdir(parents=True, exist_ok=True)
            future_image = args.input_dir / f"t{future_frame:03d}.tif"
            future_raw_csv = future_dir / f"t{future_frame:03d}_cell_lumen_future_probe.csv"
            future_log = future_dir / "cell_lumen_future_probe.log"
            future_cmd = [
                str(args.executable),
                "--cell-lumen",
                str(future_image),
                str(future_dir),
                str(args.config),
                str(future_raw_csv),
                str(firstpass_prior),
            ]
            frame_record["commands"]["future_probe_cell_lumen"] = {
                "argv": future_cmd,
                "text": command_text(future_cmd),
                "future_frame": future_frame,
            }
            future_start = time.perf_counter()
            with future_log.open("w") as log:
                future_proc = subprocess.run(
                    future_cmd,
                    cwd=str(ROOT),
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            future_probe_sec = time.perf_counter() - future_start
            frame_record["processes"]["future_probe_cell_lumen"] = {
                "returncode": future_proc.returncode,
                "seconds": future_probe_sec,
                "future_raw_csv_exists": future_raw_csv.exists(),
                "future_raw_csv": future_raw_csv,
                "log": future_log,
            }
            if future_proc.returncode != 0 or not future_raw_csv.exists():
                frame_record["status"] = "RUN_FAIL_FUTURE_PROBE"
                frame_record["disk_after_frame"] = disk_metadata(run_dir.parent)
                append_jsonl(frame_manifest_path, frame_record)
                print(
                    f"f{frame:03d} RUN_FAIL future_probe "
                    f"frame={future_frame} return={future_proc.returncode}",
                    flush=True,
                )
                return 4

            final_export_cmd = list(export_cmd)
            final_export_cmd.extend(
                [
                    "--future-candidate-cells",
                    str(future_raw_csv),
                    "--future-candidate-frame-name",
                    f"t{future_frame:03d}.tif",
                    "--future-candidate-z-mode",
                    "raw",
                    "--future-candidate-radius-z-mode",
                    "scaled",
                ]
            )
            frame_record["commands"]["biofuse_final_export_with_future"] = {
                "argv": final_export_cmd,
                "text": command_text(final_export_cmd),
                "future_raw_csv": future_raw_csv,
            }
            if args.auto_global_count_budget:
                conservative_dir = frame_dir / "conservative_budget_probe"
                conservative_summary = conservative_dir / f"t{frame:03d}_fine_shape_summary.csv"
                conservative_log = frame_dir / "biofuse_conservative_budget_probe.log"
                conservative_cmd = remove_options(
                    final_export_cmd,
                    flags={
                        "--fuse-preserve-split-hypotheses",
                        "--fuse-preserve-require-future-support",
                        "--fuse-preserve-require-distinct-future-support",
                        "--prior-pair-preserved-split-allow-shared-future",
                    },
                    valued={
                        "--fuse-preserve-pair-min-separation",
                        "--fuse-preserve-pair-max-valley-ratio",
                        "--fuse-preserve-min-future-support",
                        "--prior-pair-preserved-shared-future-penalty-scale",
                        "--prior-global-target-count",
                    },
                )
                conservative_cmd = replace_option_value(
                    conservative_cmd,
                    "--output-dir",
                    str(conservative_dir),
                )
                frame_record["commands"]["biofuse_conservative_budget_probe"] = {
                    "argv": conservative_cmd,
                    "text": command_text(conservative_cmd),
                }
                conservative_start = time.perf_counter()
                with conservative_log.open("w") as log:
                    conservative_proc = subprocess.run(
                        conservative_cmd,
                        cwd=str(ROOT),
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                conservative_sec = time.perf_counter() - conservative_start
                bio_sec += conservative_sec
                frame_record["processes"]["biofuse_conservative_budget_probe"] = {
                    "returncode": conservative_proc.returncode,
                    "seconds": conservative_sec,
                    "summary_exists": conservative_summary.exists(),
                    "summary": conservative_summary,
                    "log": conservative_log,
                }
                if conservative_proc.returncode != 0 or not conservative_summary.exists():
                    frame_record["status"] = "RUN_FAIL_CONSERVATIVE_BUDGET_PROBE"
                    frame_record["disk_after_frame"] = disk_metadata(run_dir.parent)
                    append_jsonl(frame_manifest_path, frame_record)
                    print(
                        f"f{frame:03d} RUN_FAIL conservative_budget_probe "
                        f"return={conservative_proc.returncode}",
                        flush=True,
                    )
                    return 6
                conservative_count = count_final_summary(conservative_summary)
                prior_count = len(read_prior(prior_path, args.z_scale))
                auto_target = conservative_count + max(0, conservative_count - prior_count)
                final_export_cmd = replace_option_value(
                    final_export_cmd,
                    "--prior-global-target-count",
                    str(auto_target),
                )
                frame_record["commands"]["biofuse_final_export_with_future"] = {
                    "argv": final_export_cmd,
                    "text": command_text(final_export_cmd),
                    "future_raw_csv": future_raw_csv,
                }
                frame_record["auto_global_count_budget"] = {
                    "prior_count": prior_count,
                    "conservative_count": conservative_count,
                    "target": auto_target,
                }
                print(
                    f"f{frame:03d} auto_budget prior={prior_count} "
                    f"conservative={conservative_count} target={auto_target}",
                    flush=True,
                )
            final_bio_start = time.perf_counter()
            with bio_log.open("w") as log:
                final_bio_proc = subprocess.run(
                    final_export_cmd,
                    cwd=str(ROOT),
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            final_bio_sec = time.perf_counter() - final_bio_start
            bio_sec += future_probe_sec + final_bio_sec
            runs_csv = fine_dir / f"t{frame:03d}_fine_shape_runs.csv"
            frame_record["processes"]["biofuse_final_export_with_future"] = {
                "returncode": final_bio_proc.returncode,
                "seconds": final_bio_sec,
                "summary_exists": summary_csv.exists(),
                "runs_csv": file_metadata(runs_csv),
                "log": bio_log,
            }
            if final_bio_proc.returncode != 0 or not summary_csv.exists():
                frame_record["status"] = "RUN_FAIL_FINAL_BIOFUSE_WITH_FUTURE"
                frame_record["disk_after_frame"] = disk_metadata(run_dir.parent)
                append_jsonl(frame_manifest_path, frame_record)
                print(
                    f"f{frame:03d} RUN_FAIL final_biofuse_with_future "
                    f"return={final_bio_proc.returncode}",
                    flush=True,
                )
                return 5

        gt_rows = gt_by_frame[frame]
        final_rows = read_final_summary(summary_csv)
        raw_rows = read_raw_candidates(raw_csv, args.z_scale)
        prior_rows = read_prior(prior_path, args.z_scale)
        matches, missing, extra = greedy_match(gt_rows, final_rows, args.threshold)
        max_dist = max((item[0] for item in matches), default=0.0)
        mean_dist = sum(item[0] for item in matches) / len(matches) if matches else 0.0
        status = "PASS" if not missing and not extra else "FAIL"
        total_sec = time.perf_counter() - frame_start
        runs_csv = fine_dir / f"t{frame:03d}_fine_shape_runs.csv"
        frame_record["outputs"].update(
            {
                "raw_csv": file_metadata(raw_csv),
                "summary_csv": file_metadata(summary_csv),
                "runs_csv": file_metadata(runs_csv),
                "cell_log": file_metadata(cell_log),
                "bio_log": file_metadata(bio_log),
            }
        )
        frame_record["audit"] = {
            "status": status,
            "gt": len(gt_rows),
            "raw": len(raw_rows),
            "pred": len(final_rows),
            "matched": len(matches),
            "missing": len(missing),
            "extra": len(extra),
            "max_distance": max_dist,
            "mean_distance": mean_dist,
            "missing_labels": [row["label"] for row in missing],
            "extra_names": [row["name"] for row in extra],
            "threshold": args.threshold,
            "z_scale": args.z_scale,
        }
        frame_record["timing_seconds"] = {
            "cell_lumen": cell_sec,
            "biofuse_total": bio_sec,
            "future_probe": future_probe_sec,
            "frame_total": total_sec,
        }

        print(
            f"f{frame:03d} {status} raw={len(raw_rows)} pred={len(final_rows)} "
            f"gt={len(gt_rows)} match={len(matches)} miss={len(missing)} "
            f"extra={len(extra)} max={max_dist:.2f} mean={mean_dist:.2f} "
            f"cell={cell_sec:.2f}s bio={bio_sec:.2f}s "
            f"future={future_probe_sec:.2f}s total={total_sec:.2f}s",
            flush=True,
        )
        if missing:
            print("  missing=" + ",".join(str(row["label"]) for row in missing), flush=True)
        if extra:
            names = [str(row["name"]) for row in extra[:8]]
            if len(extra) > 8:
                names.append(f"...+{len(extra) - 8}")
            print("  extra=" + ",".join(names), flush=True)

        summary_rows.append(
            {
                "frame": frame,
                "status": status,
                "gt": len(gt_rows),
                "raw": len(raw_rows),
                "pred": len(final_rows),
                "matched": len(matches),
                "missing": len(missing),
                "extra": len(extra),
                "max_distance": f"{max_dist:.6f}",
                "mean_distance": f"{mean_dist:.6f}",
                "cell_lumen_sec": f"{cell_sec:.6f}",
                "biofuse_sec": f"{bio_sec:.6f}",
                "future_probe_sec": f"{future_probe_sec:.6f}",
                "total_sec": f"{total_sec:.6f}",
                "prior_file": str(prior_path),
                "raw_csv": str(raw_csv),
                "summary_csv": str(summary_csv),
                "missing_labels": ";".join(str(row["label"]) for row in missing),
                "extra_names": ";".join(str(row["name"]) for row in extra),
            }
        )

        for row in missing:
            nearest_raw, raw_dist = nearest(row, raw_rows)
            nearest_final, final_dist = nearest(row, final_rows)
            nearest_prior, prior_dist = nearest(row, prior_rows)
            failure_rows.append(
                {
                    "frame": frame,
                    "type": "missing",
                    "label_or_name": row["label"],
                    "parent": row.get("parent", ""),
                    "category": "raw_present" if raw_dist <= args.threshold else "raw_far_or_absent",
                    "nearest_raw": nearest_raw,
                    "dist_raw": f"{raw_dist:.3f}",
                    "nearest_final": nearest_final,
                    "dist_final": f"{final_dist:.3f}",
                    "nearest_prior": nearest_prior,
                    "dist_prior": f"{prior_dist:.3f}",
                    "voxels": "",
                    "nearest_gt": "",
                    "dist_gt": "",
                }
            )
        for row in extra:
            nearest_gt, gt_dist = nearest(row, gt_rows)
            nearest_prior, prior_dist = nearest(row, prior_rows)
            if gt_dist <= args.threshold:
                category = "same_gt_internal_duplicate"
            elif gt_dist <= 40.0:
                category = "borderline_edge_lobe"
            else:
                category = "remote_or_false_fragment"
            failure_rows.append(
                {
                    "frame": frame,
                    "type": "extra",
                    "label_or_name": row["name"],
                    "parent": "",
                    "category": category,
                    "nearest_raw": "",
                    "dist_raw": "",
                    "nearest_final": "",
                    "dist_final": "",
                    "nearest_prior": nearest_prior,
                    "dist_prior": f"{prior_dist:.3f}",
                    "voxels": row.get("voxels", ""),
                    "nearest_gt": nearest_gt,
                    "dist_gt": f"{gt_dist:.3f}",
                }
            )

        next_prior = frame_dir / f"rolling_prior_t{frame:03d}_for_next.csv"
        prior_path = write_prior_from_summary(summary_csv, next_prior, frame, args.z_scale)
        prior_frame_name = f"t{frame:03d}.tif"
        frame_record["outputs"]["next_prior"] = file_metadata(next_prior)
        frame_record["disk_after_frame"] = disk_metadata(run_dir.parent)
        frame_record["status"] = status
        append_jsonl(frame_manifest_path, frame_record)

        write_csv(
            run_dir / f"rolling_center_audit_summary_f{frames[0]:03d}_f{frames[-1]:03d}.csv",
            summary_rows,
            list(summary_rows[0].keys()),
        )
        if failure_rows:
            write_csv(
                run_dir / f"rolling_failure_analysis_f{frames[0]:03d}_f{frames[-1]:03d}.csv",
                failure_rows,
                list(failure_rows[0].keys()),
            )
        if args.stop_on_fail and status != "PASS":
            break

    manifest["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    manifest["disk_after"] = {
        "output_root": disk_metadata(args.output_root),
        "run_dir_parent": disk_metadata(run_dir.parent),
        "workspace": disk_metadata(ROOT),
    }
    manifest["summary_csv"] = str(run_dir / f"rolling_center_audit_summary_f{frames[0]:03d}_f{frames[-1]:03d}.csv")
    manifest["failure_analysis_csv"] = str(run_dir / f"rolling_failure_analysis_f{frames[0]:03d}_f{frames[-1]:03d}.csv")
    manifest["frame_manifest_jsonl"] = str(frame_manifest_path)
    write_json(manifest_path, manifest)
    print(f"RUN_DIR={run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
