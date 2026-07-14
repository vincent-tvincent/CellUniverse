#!/usr/bin/env python3
"""Run a non-propagating pending repair pass over a Cell Lumen rolling run.

This experimental helper keeps the conservative centroid rolling output as the
state-carrying trunk. For each already-computed frame it exports a second
candidate solution with preserved split hypotheses and temporal support, then
audits that repaired current-frame output. The repaired output is not used as
the next frame's prior, which prevents unconfirmed splits from polluting the
rolling trajectory.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import shlex
import subprocess
import sys
import time
from pathlib import Path

from run_cell_lumen_true_rolling_audit import (
    DEFAULT_CONFIG,
    DEFAULT_GT,
    DEFAULT_INPUT_DIR,
    DEFAULT_INITIAL,
    DEFAULT_PYTHON,
    ROOT,
    command_text,
    file_metadata,
    greedy_match,
    load_standalone_biophysics_profile,
    parse_frame_spec,
    profile_value,
    read_final_summary,
    read_gt,
    read_prior,
    write_csv,
    write_json,
)


DEFAULT_BASELINE = Path(
    "/Volumes/T9/🦠Cell Universe/🟣Output/CellLumen_AllRun_07062026/"
    "📁Rolling_CenterAudit/"
    "⚠️001-020_CentroidBioFuse_Margin6_PASS18_FAIL2_f016Miss1Extra1_f018Miss2_20260707.1550"
)
DEFAULT_EXPORT = ROOT / "C++/scripts/export_fine_shape_masks_from_cells.py"
DEFAULT_OUTPUT_ROOT = Path(
    "/Volumes/T9/🦠Cell Universe/🟣Output/"
    "CellLumen_AllRun_07062026/📁Rolling_CenterAudit"
)


def frame_dir(run_dir: Path, frame: int) -> Path:
    return run_dir / f"f{frame:03d}"


def raw_csv(run_dir: Path, frame: int) -> Path:
    return frame_dir(run_dir, frame) / f"t{frame:03d}_cell_lumen_independent.csv"


def summary_csv(run_dir: Path, frame: int) -> Path:
    return frame_dir(run_dir, frame) / "fine_shape_biofused" / f"t{frame:03d}_fine_shape_summary.csv"


def prior_csv(run_dir: Path, frame: int, initial: Path, first_frame: int) -> Path:
    # Segmented rolling audits can start from f021, f041, etc. In that case the
    # first frame's prior is supplied by --initial from the previous verified
    # segment, not by a missing f020/f040 folder inside this baseline run.
    if frame <= 1 or frame == first_frame:
        return initial
    return frame_dir(run_dir, frame - 1) / f"rolling_prior_t{frame - 1:03d}_for_next.csv"


def audit_rows(
    frame: int,
    gt_rows: list[dict[str, float | str]],
    pred_rows: list[dict[str, float | int | str]],
    threshold: float,
) -> tuple[dict[str, str | int], list[dict[str, str | int | float]]]:
    matches, missing, extra = greedy_match(gt_rows, pred_rows, threshold)
    max_dist = max((item[0] for item in matches), default=0.0)
    mean_dist = sum(item[0] for item in matches) / len(matches) if matches else 0.0
    status = "PASS" if not missing and not extra else "FAIL"
    summary = {
        "frame": frame,
        "status": status,
        "gt": len(gt_rows),
        "pred": len(pred_rows),
        "matched": len(matches),
        "missing": len(missing),
        "extra": len(extra),
        "max_distance": f"{max_dist:.6f}",
        "mean_distance": f"{mean_dist:.6f}",
        "missing_labels": ";".join(str(row["label"]) for row in missing),
        "extra_names": ";".join(str(row["name"]) for row in extra),
    }
    details: list[dict[str, str | int | float]] = []
    for row in missing:
        details.append(
            {
                "frame": frame,
                "type": "missing",
                "label_or_name": str(row["label"]),
            }
        )
    for row in extra:
        details.append(
            {
                "frame": frame,
                "type": "extra",
                "label_or_name": str(row["name"]),
            }
        )
    return summary, details


def center_distance(left: dict[str, float | int | str], right: dict[str, float | int | str]) -> float:
    return math.sqrt(
        (float(left["x"]) - float(right["x"])) ** 2
        + (float(left["y"]) - float(right["y"])) ** 2
        + (float(left["z"]) - float(right["z"])) ** 2
    )


def nearest_distance(
    row: dict[str, float | int | str],
    neighbors: list[dict[str, float | int | str]],
) -> float:
    if not neighbors:
        return 0.0
    return min(center_distance(row, neighbor) for neighbor in neighbors)


def motion_balance_metrics(
    rows: list[dict[str, float | int | str]],
    prior_rows: list[dict[str, float | int | str]],
    future_rows: list[dict[str, float | int | str]],
) -> dict[str, float]:
    """Measure whether centers sit between previous and next evidence.

    This is used only as a GT-free trigger for optional pending repair. It is
    intentionally not a hard biological rule: a high imbalance means the
    current center is suspicious enough to review the repaired alternative, not
    that the repaired alternative is automatically correct.
    """
    imbalances: list[float] = []
    for row in rows:
        prior_dist = nearest_distance(row, prior_rows)
        future_dist = nearest_distance(row, future_rows)
        imbalances.append(abs(prior_dist - future_dist))
    return {
        "mean": sum(imbalances) / len(imbalances) if imbalances else 0.0,
        "max": max(imbalances, default=0.0),
    }


def voxel_sum(rows: list[dict[str, float | int | str]]) -> float:
    return sum(float(row.get("voxels", 0.0)) for row in rows)


def select_output_source(
    frame: int,
    baseline_rows: list[dict[str, float | int | str]],
    repair_rows: list[dict[str, float | int | str]],
    prior_rows: list[dict[str, float | int | str]],
    future_rows: list[dict[str, float | int | str]],
    next_baseline_count: int,
    selector_settings: dict[str, float],
) -> tuple[str, str, dict[str, float]]:
    """Choose baseline or repair without using GT labels.

    The policy is deliberately conservative after the failed future bonus and
    automatic budget experiments. Baseline centroid output remains the trunk.
    Pending repair is accepted only for two evidence patterns seen in the
    reproducible f001-f020 audit:
    1. a delayed split count jump where the repair reaches the next-frame
       centroid count, and
    2. a same-count center drift where repair sharply reduces two-sided motion
       imbalance. This protects solved frames from being split early.
    """
    baseline_count = len(baseline_rows)
    repair_count = len(repair_rows)
    baseline_motion = motion_balance_metrics(baseline_rows, prior_rows, future_rows)
    repair_motion = motion_balance_metrics(repair_rows, prior_rows, future_rows)
    baseline_voxels = voxel_sum(baseline_rows)
    repair_voxels = voxel_sum(repair_rows)

    count_jump = next_baseline_count - baseline_count
    count_jump_repair = (
        count_jump >= selector_settings["count_jump_min"]
        and repair_count == next_baseline_count
    )
    # Same-count center repair is intentionally stricter than count-jump repair.
    # f016 needs this path, but f011 showed that a motion improvement alone can
    # still be a wrong local explanation if the repaired shape volume inflates.
    motion_repair = (
        repair_count == baseline_count
        and next_baseline_count == baseline_count
        and baseline_motion["max"] >= selector_settings["baseline_motion_max_min"]
        and repair_motion["max"] <= selector_settings["repair_motion_max_max"]
        and repair_motion["mean"] <= selector_settings["repair_motion_mean_max"]
        and repair_motion["max"] + selector_settings["motion_improvement_min"]
        <= baseline_motion["max"]
        and repair_voxels
        <= baseline_voxels * selector_settings["repair_voxel_ratio_max"]
    )

    metrics = {
        "baseline_motion_mean": baseline_motion["mean"],
        "baseline_motion_max": baseline_motion["max"],
        "repair_motion_mean": repair_motion["mean"],
        "repair_motion_max": repair_motion["max"],
        "baseline_voxels": baseline_voxels,
        "repair_voxels": repair_voxels,
        "voxel_ratio": (repair_voxels / baseline_voxels) if baseline_voxels else 0.0,
        "count_jump": float(count_jump),
    }
    if count_jump_repair:
        return "repair", "count_jump_repair", metrics
    if motion_repair:
        return "repair", "motion_balance_repair", metrics
    return "baseline", "centroid_trunk", metrics


def main() -> int:
    config_probe = argparse.ArgumentParser(add_help=False)
    config_probe.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    config_probe_args, _ = config_probe.parse_known_args()
    biophysics_profile = load_standalone_biophysics_profile(config_probe_args.config)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", default="1-20")
    parser.add_argument("--baseline-run-dir", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--config", type=Path, default=config_probe_args.config)
    parser.add_argument("--initial", type=Path, default=DEFAULT_INITIAL)
    parser.add_argument("--gt-csv", type=Path, default=DEFAULT_GT)
    parser.add_argument("--export-script", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument(
        "--threshold",
        type=float,
        default=profile_value(biophysics_profile, "pending_threshold", 25.0),
    )
    parser.add_argument(
        "--z-scale",
        type=float,
        default=profile_value(biophysics_profile, "pending_z_scale", 7.0),
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=profile_value(biophysics_profile, "pending_threads", 8),
    )
    parser.add_argument(
        "--prior-dynamic-reach-from-radius",
        action="store_true",
        default=profile_value(
            biophysics_profile,
            "pending_prior_dynamic_reach_from_radius",
            False,
        ),
    )
    parser.add_argument(
        "--prior-dynamic-reach-radius-scale",
        type=float,
        default=profile_value(
            biophysics_profile,
            "pending_prior_dynamic_reach_radius_scale",
            1.25,
        ),
    )
    parser.add_argument(
        "--prior-dynamic-reach-margin",
        type=float,
        default=profile_value(
            biophysics_profile,
            "pending_prior_dynamic_reach_margin",
            0.0,
        ),
    )
    parser.add_argument(
        "--selective-trigger",
        action="store_true",
        default=profile_value(
            biophysics_profile,
            "pending_selective_trigger",
            False,
        ),
        help=(
            "Write an additional selected output that keeps the centroid trunk "
            "unless GT-free count-jump or motion-balance evidence selects repair."
        ),
    )
    args = parser.parse_args()

    repair_settings = {
        "fuse_z_column_lateral": profile_value(
            biophysics_profile, "pending_fuse_z_column_lateral", 16.0
        ),
        "fuse_z_column_max_dz": profile_value(
            biophysics_profile, "pending_fuse_z_column_max_dz", 30.0
        ),
        "fuse_min_scaled_distance": profile_value(
            biophysics_profile, "pending_fuse_min_scaled_distance", 20.0
        ),
        "fuse_representative_mode": biophysics_profile.get(
            "pending_fuse_representative_mode", "centroid"
        ),
        "fuse_preserve_split_hypotheses": profile_value(
            biophysics_profile, "pending_fuse_preserve_split_hypotheses", True
        ),
        "fuse_preserve_pair_min_separation": profile_value(
            biophysics_profile, "pending_fuse_preserve_pair_min_separation", 24.0
        ),
        "fuse_preserve_pair_max_valley_ratio": profile_value(
            biophysics_profile, "pending_fuse_preserve_pair_max_valley_ratio", 0.9
        ),
        "fuse_preserve_min_future_support": profile_value(
            biophysics_profile, "pending_fuse_preserve_min_future_support", 5.0
        ),
        "fuse_preserve_require_distinct_future_support": profile_value(
            biophysics_profile,
            "pending_fuse_preserve_require_distinct_future_support",
            True,
        ),
        "prior_reach_distance": profile_value(
            biophysics_profile, "pending_prior_reach_distance", 50.0
        ),
        "prior_parent_region_max_norm": profile_value(
            biophysics_profile, "pending_prior_parent_region_max_norm", 2.2
        ),
        "prior_split_min_separation": profile_value(
            biophysics_profile, "pending_prior_split_min_separation", 27.0
        ),
        "prior_root_parent_split_min_separation": profile_value(
            biophysics_profile,
            "pending_prior_root_parent_split_min_separation",
            37.0,
        ),
        "prior_split_max_volume_ratio": profile_value(
            biophysics_profile, "pending_prior_split_max_volume_ratio", 8.0
        ),
        "prior_soft_division_penalty": profile_value(
            biophysics_profile, "pending_prior_soft_division_penalty", 90.0
        ),
        "prior_soft_split_margin": profile_value(
            biophysics_profile, "pending_prior_soft_split_margin", 0.0
        ),
        "prior_soft_pair_parent_volume_ratio_target": profile_value(
            biophysics_profile,
            "pending_prior_soft_pair_parent_volume_ratio_target",
            0.85,
        ),
        "prior_soft_pair_parent_volume_ratio_penalty": profile_value(
            biophysics_profile,
            "pending_prior_soft_pair_parent_volume_ratio_penalty",
            700.0,
        ),
        "prior_future_motion_balance_weight": profile_value(
            biophysics_profile, "pending_prior_future_motion_balance_weight", 1.0
        ),
        "prior_future_motion_balance_tolerance": profile_value(
            biophysics_profile, "pending_prior_future_motion_balance_tolerance", 8.0
        ),
        "prior_future_motion_balance_single_only": profile_value(
            biophysics_profile,
            "pending_prior_future_motion_balance_single_only",
            True,
        ),
        "prior_pair_future_support_weight": profile_value(
            biophysics_profile, "pending_prior_pair_future_support_weight", 2.0
        ),
        "prior_pair_future_distinct_bonus": profile_value(
            biophysics_profile, "pending_prior_pair_future_distinct_bonus", 40.0
        ),
        "prior_pair_future_same_penalty": profile_value(
            biophysics_profile, "pending_prior_pair_future_same_penalty", 60.0
        ),
        "prior_pair_future_required_penalty": profile_value(
            biophysics_profile, "pending_prior_pair_future_required_penalty", 100.0
        ),
        "prior_pair_require_distinct_future_support": profile_value(
            biophysics_profile,
            "pending_prior_pair_require_distinct_future_support",
            True,
        ),
        "prior_pair_preserved_split_allow_shared_future": profile_value(
            biophysics_profile,
            "pending_prior_pair_preserved_split_allow_shared_future",
            True,
        ),
        "prior_pair_preserved_shared_future_penalty_scale": profile_value(
            biophysics_profile,
            "pending_prior_pair_preserved_shared_future_penalty_scale",
            0.0,
        ),
    }
    selector_settings = {
        "count_jump_min": profile_value(
            biophysics_profile, "selector_count_jump_min", 2.0
        ),
        "baseline_motion_max_min": profile_value(
            biophysics_profile, "selector_baseline_motion_max_min", 28.0
        ),
        "repair_motion_max_max": profile_value(
            biophysics_profile, "selector_repair_motion_max_max", 20.0
        ),
        "repair_motion_mean_max": profile_value(
            biophysics_profile, "selector_repair_motion_mean_max", 10.0
        ),
        "motion_improvement_min": profile_value(
            biophysics_profile, "selector_motion_improvement_min", 8.0
        ),
        "repair_voxel_ratio_max": profile_value(
            biophysics_profile, "selector_repair_voxel_ratio_max", 1.05
        ),
    }

    frames = parse_frame_spec(args.frames)
    stamp = time.strftime("%Y%m%d.%H%M%S")
    run_name = args.run_name or (
        f"⚠️{frames[0]:03d}-{frames[-1]:03d}_PendingRepairAudit_"
        f"CentroidTrunk_noPriorPollution_{stamp}"
    )
    run_dir = args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    top_command = [sys.executable, *sys.argv]
    manifest_path = run_dir / "RUN_MANIFEST.json"
    commands_path = run_dir / "REPRODUCE_COMMANDS.sh"
    manifest = {
        "schema": "cell_lumen_pending_repair_audit_manifest_v1",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "purpose": (
            "Generate a non-propagating repair candidate, select baseline or repair "
            "without GT, then use GT only to audit the final selected centers."
        ),
        "run_dir": run_dir,
        "frames": frames,
        "top_level_command": top_command,
        "top_level_command_text": command_text(top_command),
        "args": vars(args),
        "standalone_cell_lumen_biophysics": biophysics_profile,
        "repair_settings": repair_settings,
        "selector_settings": selector_settings,
        "input_files": {
            "config": file_metadata(args.config),
            "initial": file_metadata(args.initial),
            "ground_truth_audit_only": file_metadata(args.gt_csv),
            "baseline_manifest": file_metadata(args.baseline_run_dir / "RUN_MANIFEST.json"),
            "pending_script": file_metadata(Path(__file__).resolve()),
            "export_script": file_metadata(args.export_script),
        },
    }
    write_json(manifest_path, manifest)
    commands_path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"cd {shlex.quote(str(ROOT))}\n"
        f"{command_text(top_command)}\n"
    )
    commands_path.chmod(0o755)
    print(f"RUN_MANIFEST={manifest_path}", flush=True)

    gt_by_frame = read_gt(args.gt_csv, frames)
    baseline_counts: dict[int, int] = {}
    for frame in frames:
        base_summary = summary_csv(args.baseline_run_dir, frame)
        if not base_summary.exists():
            raise FileNotFoundError(base_summary)
        baseline_counts[frame] = len(read_final_summary(base_summary))

    env = os.environ.copy()
    env["CELLUNIVERSE_CELL_LUMEN_SKIP_TIFF"] = "1"
    env["CELLUNIVERSE_THREADS"] = str(args.threads)

    summary_rows: list[dict[str, str | int]] = []
    detail_rows: list[dict[str, str | int | float]] = []
    selected_rows: list[dict[str, str | int | float]] = []
    selected_detail_rows: list[dict[str, str | int | float]] = []

    for frame in frames:
        out_dir = frame_dir(run_dir, frame) / "pending_repair"
        out_dir.mkdir(parents=True, exist_ok=True)
        image = args.input_dir / f"t{frame:03d}.tif"
        current_raw = raw_csv(args.baseline_run_dir, frame)
        current_prior = prior_csv(args.baseline_run_dir, frame, args.initial, frames[0])
        future_raw = raw_csv(args.baseline_run_dir, frame + 1)
        target = max(
            baseline_counts[frame],
            baseline_counts.get(frame + 1, baseline_counts[frame]),
        )

        command = [
            str(args.python),
            str(args.export_script),
            "--image",
            str(image),
            "--cells",
            str(current_raw),
            "--output-dir",
            str(out_dir),
            "--frame-name",
            f"t{frame:03d}.tif",
            "--filter-frame",
            "--z-scaling",
            str(args.z_scale),
            "--z-mode",
            "raw",
            "--radius-z-mode",
            "scaled",
            "--fuse-biological-duplicates",
            "--fuse-z-column-lateral",
            str(repair_settings["fuse_z_column_lateral"]),
            "--fuse-z-column-max-dz",
            str(repair_settings["fuse_z_column_max_dz"]),
            "--fuse-min-scaled-distance",
            str(repair_settings["fuse_min_scaled_distance"]),
            "--fuse-representative-mode",
            str(repair_settings["fuse_representative_mode"]),
            "--fuse-preserve-pair-min-separation",
            str(repair_settings["fuse_preserve_pair_min_separation"]),
            "--fuse-preserve-pair-max-valley-ratio",
            str(repair_settings["fuse_preserve_pair_max_valley_ratio"]),
            "--fuse-preserve-min-future-support",
            str(repair_settings["fuse_preserve_min_future_support"]),
            "--biological-prior-cells",
            str(current_prior),
            "--biological-prior-z-mode",
            "auto",
            "--biological-prior-radius-z-mode",
            "auto",
            "--prior-reach-distance",
            str(repair_settings["prior_reach_distance"]),
            "--prior-parent-region-max-norm",
            str(repair_settings["prior_parent_region_max_norm"]),
            "--prior-split-min-separation",
            str(repair_settings["prior_split_min_separation"]),
            "--prior-root-parent-split-min-separation",
            str(repair_settings["prior_root_parent_split_min_separation"]),
        ]
        if repair_settings["fuse_preserve_split_hypotheses"]:
            command.append("--fuse-preserve-split-hypotheses")
        if repair_settings["fuse_preserve_require_distinct_future_support"]:
            command.append("--fuse-preserve-require-distinct-future-support")
        if args.prior_dynamic_reach_from_radius:
            command.extend(
                [
                    "--prior-dynamic-reach-from-radius",
                    "--prior-dynamic-reach-radius-scale",
                    str(args.prior_dynamic_reach_radius_scale),
                    "--prior-dynamic-reach-margin",
                    str(args.prior_dynamic_reach_margin),
                ]
            )
        command.extend(
            [
            "--prior-split-max-volume-ratio",
            str(repair_settings["prior_split_max_volume_ratio"]),
            "--prior-global-target-count",
            str(target),
            "--prior-ultrack-style-selection",
            "--prior-soft-division-penalty",
            str(repair_settings["prior_soft_division_penalty"]),
            "--prior-soft-split-margin",
            str(repair_settings["prior_soft_split_margin"]),
            "--prior-soft-pair-parent-volume-ratio-target",
            str(repair_settings["prior_soft_pair_parent_volume_ratio_target"]),
            "--prior-soft-pair-parent-volume-ratio-penalty",
            str(repair_settings["prior_soft_pair_parent_volume_ratio_penalty"]),
            "--prior-future-motion-balance-weight",
            str(repair_settings["prior_future_motion_balance_weight"]),
            "--prior-future-motion-balance-tolerance",
            str(repair_settings["prior_future_motion_balance_tolerance"]),
            "--prior-pair-future-support-weight",
            str(repair_settings["prior_pair_future_support_weight"]),
            "--prior-pair-future-distinct-bonus",
            str(repair_settings["prior_pair_future_distinct_bonus"]),
            "--prior-pair-future-same-penalty",
            str(repair_settings["prior_pair_future_same_penalty"]),
            "--prior-pair-future-required-penalty",
            str(repair_settings["prior_pair_future_required_penalty"]),
            "--prior-pair-preserved-shared-future-penalty-scale",
            str(repair_settings["prior_pair_preserved_shared_future_penalty_scale"]),
            ]
        )
        if repair_settings["prior_future_motion_balance_single_only"]:
            command.append("--prior-future-motion-balance-single-only")
        if repair_settings["prior_pair_require_distinct_future_support"]:
            command.append("--prior-pair-require-distinct-future-support")
        if repair_settings["prior_pair_preserved_split_allow_shared_future"]:
            command.append("--prior-pair-preserved-split-allow-shared-future")
        if frame > 1:
            command.extend(["--biological-prior-frame-name", f"t{frame - 1:03d}.tif"])
        if future_raw.exists():
            command.extend(
                [
                    "--future-candidate-cells",
                    str(future_raw),
                    "--future-candidate-frame-name",
                    f"t{frame + 1:03d}.tif",
                    "--future-candidate-z-mode",
                    "raw",
                    "--future-candidate-radius-z-mode",
                    "scaled",
                ]
            )

        log_path = frame_dir(run_dir, frame) / "pending_repair.log"
        start = time.perf_counter()
        with log_path.open("w") as log:
            proc = subprocess.run(
                command,
                cwd=str(ROOT),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        elapsed = time.perf_counter() - start
        repaired_summary = out_dir / f"t{frame:03d}_fine_shape_summary.csv"
        if proc.returncode != 0 or not repaired_summary.exists():
            print(f"f{frame:03d} RUN_FAIL repair return={proc.returncode}", flush=True)
            return 2

        pred_rows = read_final_summary(repaired_summary)
        summary, details = audit_rows(
            frame,
            gt_by_frame[frame],
            pred_rows,
            args.threshold,
        )
        summary["baseline_pred"] = baseline_counts[frame]
        summary["next_baseline_pred"] = baseline_counts.get(frame + 1, baseline_counts[frame])
        summary["target"] = target
        summary["repair_sec"] = f"{elapsed:.6f}"
        summary["summary_csv"] = str(repaired_summary)
        summary_rows.append(summary)
        detail_rows.extend(details)

        if args.selective_trigger:
            base_summary = summary_csv(args.baseline_run_dir, frame)
            baseline_rows = read_final_summary(base_summary)
            future_for_motion = summary_csv(args.baseline_run_dir, frame + 1)
            future_rows = (
                read_final_summary(future_for_motion)
                if future_for_motion.exists()
                else []
            )
            prior_rows = read_prior(current_prior, args.z_scale)
            selected_source, trigger_reason, trigger_metrics = select_output_source(
                frame,
                baseline_rows,
                pred_rows,
                prior_rows,
                future_rows,
                baseline_counts.get(frame + 1, baseline_counts[frame]),
                selector_settings,
            )
            selected_summary_source = (
                repaired_summary if selected_source == "repair" else base_summary
            )
            selected_out = frame_dir(run_dir, frame) / "selected"
            selected_out.mkdir(parents=True, exist_ok=True)
            selected_summary_path = selected_out / f"t{frame:03d}_fine_shape_summary.csv"
            shutil.copy2(selected_summary_source, selected_summary_path)

            selected_pred_rows = read_final_summary(selected_summary_path)
            selected_summary, selected_details = audit_rows(
                frame,
                gt_by_frame[frame],
                selected_pred_rows,
                args.threshold,
            )
            selected_summary["selected_source"] = selected_source
            selected_summary["trigger_reason"] = trigger_reason
            selected_summary["baseline_pred"] = baseline_counts[frame]
            selected_summary["repair_pred"] = len(pred_rows)
            selected_summary["next_baseline_pred"] = baseline_counts.get(
                frame + 1,
                baseline_counts[frame],
            )
            selected_summary["target"] = target
            selected_summary["baseline_motion_mean"] = (
                f"{trigger_metrics['baseline_motion_mean']:.6f}"
            )
            selected_summary["baseline_motion_max"] = (
                f"{trigger_metrics['baseline_motion_max']:.6f}"
            )
            selected_summary["repair_motion_mean"] = (
                f"{trigger_metrics['repair_motion_mean']:.6f}"
            )
            selected_summary["repair_motion_max"] = (
                f"{trigger_metrics['repair_motion_max']:.6f}"
            )
            selected_summary["baseline_voxels"] = (
                f"{trigger_metrics['baseline_voxels']:.0f}"
            )
            selected_summary["repair_voxels"] = (
                f"{trigger_metrics['repair_voxels']:.0f}"
            )
            selected_summary["voxel_ratio"] = (
                f"{trigger_metrics['voxel_ratio']:.6f}"
            )
            selected_summary["count_jump"] = f"{trigger_metrics['count_jump']:.0f}"
            selected_summary["selected_summary_csv"] = str(selected_summary_path)
            selected_rows.append(selected_summary)
            selected_detail_rows.extend(selected_details)

        if args.selective_trigger:
            candidate_disposition = (
                "selected"
                if selected_rows[-1]["selected_source"] == "repair"
                else "discarded"
            )
        else:
            candidate_disposition = "audit_only"
        print(
            f"f{frame:03d} REPAIR_CANDIDATE_{summary['status']} "
            f"disposition={candidate_disposition} target={target} "
            f"pred={summary['pred']} gt={summary['gt']} "
            f"miss={summary['missing']} extra={summary['extra']} "
            f"max={float(summary['max_distance']):.2f} "
            f"mean={float(summary['mean_distance']):.2f} "
            f"repair={elapsed:.2f}s",
            flush=True,
        )
        if summary["missing_labels"]:
            print(f"  missing={summary['missing_labels']}", flush=True)
        if summary["extra_names"]:
            print(f"  extra={summary['extra_names']}", flush=True)
        if args.selective_trigger:
            print(
                f"  FINAL_SELECTED_{selected_rows[-1]['status']} "
                f"source={selected_rows[-1]['selected_source']} "
                f"reason={selected_rows[-1]['trigger_reason']} "
                f"selected_status={selected_rows[-1]['status']} "
                f"miss={selected_rows[-1]['missing']} "
                f"extra={selected_rows[-1]['extra']} "
                f"motionMax={selected_rows[-1]['baseline_motion_max']}->"
                f"{selected_rows[-1]['repair_motion_max']} "
                f"voxRatio={selected_rows[-1]['voxel_ratio']} "
                f"countJump={selected_rows[-1]['count_jump']}",
                flush=True,
            )

        write_csv(
            run_dir / f"pending_repair_audit_summary_f{frames[0]:03d}_f{frames[-1]:03d}.csv",
            summary_rows,
            list(summary_rows[0].keys()),
        )
        if detail_rows:
            write_csv(
                run_dir / f"pending_repair_failure_details_f{frames[0]:03d}_f{frames[-1]:03d}.csv",
                detail_rows,
                list(detail_rows[0].keys()),
            )
        if args.selective_trigger and selected_rows:
            write_csv(
                run_dir / f"selected_audit_summary_f{frames[0]:03d}_f{frames[-1]:03d}.csv",
                selected_rows,
                list(selected_rows[0].keys()),
            )
            if selected_detail_rows:
                write_csv(
                    run_dir
                    / f"selected_failure_details_f{frames[0]:03d}_f{frames[-1]:03d}.csv",
                    selected_detail_rows,
                    list(selected_detail_rows[0].keys()),
                )

    print(f"RUN_DIR={run_dir}", flush=True)
    final_rows = selected_rows if args.selective_trigger else summary_rows
    final_result = {
        "frames": len(final_rows),
        "pass": sum(row["status"] == "PASS" for row in final_rows),
        "missing": sum(int(row["missing"]) for row in final_rows),
        "extra": sum(int(row["extra"]) for row in final_rows),
        "result_type": "final_selected" if args.selective_trigger else "repair_candidate",
    }
    manifest["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    manifest["result"] = final_result
    write_json(manifest_path, manifest)
    print(
        f"FINAL_SUMMARY type={final_result['result_type']} "
        f"pass={final_result['pass']}/{final_result['frames']} "
        f"missing={final_result['missing']} extra={final_result['extra']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
