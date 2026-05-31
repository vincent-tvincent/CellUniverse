#!/usr/bin/env python3
"""Original-pipeline Fluo GT watcher.

This wrapper reuses the vetted r26 Fluo GT evaluator/orchestrator, but keeps a
separate namespace and forces the non-CellLumen original pipeline:

  - simulation.preprocess_mode = n2v2
  - simulation.n2v2_preprocess.enabled = true
  - cell_lumen.enabled = false
  - cell_lumen.fusionEnabled = false
  - 3 concurrent tuning workers, 32 threads each

GT is available for frames 0..194. Runs can continue to frame 200, but live
failure decisions are made only where GT exists.
"""

import argparse
import csv
import importlib.util
import os
import shutil
import sys
import time
from pathlib import Path


REPO_ROOT = Path("/home/puv/celluniverse/CellUniverse/C++")
SOURCE_DRIVER = Path("/home/puv/output_fluo/tuning_fluo_gt_20260529/continuous_gt_watch_r26/continuous_gt_watch_r26.py")
ROOT = Path("/home/puv/output_fluo/original_fluo_gt_watch_20260531")
CONFIG_SRC = REPO_ROOT / "config/config.yaml"
LAST_RUN_FRAME = 200


def load_driver():
    spec = importlib.util.spec_from_file_location("fluo_gt_watch_r26", SOURCE_DRIVER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["fluo_gt_watch_r26"] = module
    spec.loader.exec_module(module)
    patch_driver(module)
    return module


def patch_driver(d):
    d.ROOT = ROOT
    d.CONFIG_SRC = CONFIG_SRC
    d.CONT_THREADS = 32
    d.BATCH_THREADS = 32
    d.MAX_BATCH_WORKERS = 3
    d.CPU_BUDGET = 96
    d.CPUSET = "0-95"
    d.REWIND = 5
    d.LOOKAHEAD = 3
    d.DIRS = {
        "configs": ROOT / "configs",
        "runs": ROOT / "runs",
        "metrics": ROOT / "metrics",
        "logs": ROOT / "logs",
        "docs": ROOT / "docs",
        "finalists": ROOT / "finalists",
        "gt": ROOT / "metrics" / "gt_reference",
        "initials": ROOT / "initials",
    }
    d.LOGBOOK = d.DIRS["docs"] / "original_pipeline_tuning_logbook.md"
    d.CODE_CHANGE_LOG = d.DIRS["docs"] / "code_change_log.md"
    d.DECISIONS = ROOT / "decisions.jsonl"
    d.FAILURES = ROOT / "failure_events.jsonl"
    d.DEBUG_SUMMARY = ROOT / "debug_reason_summaries.csv"
    d.STATUS = ROOT / "status.json"

    d.COMMON_OVERRIDES = {
        "simulation.preprocess_mode": "n2v2",
        "simulation.n2v2_preprocess.enabled": True,
        "simulation.n2v2_preprocess.enable_network": True,
        "simulation.n2v2_preprocess.contrast.gamma": 1.45,
        "simulation.export_frame_png": False,
        "simulation.export_frame_tiff": False,
        "simulation.export_preprocessed_images": False,
        "simulation.export_signal_debug_images": False,
        "simulation.export_perturb_debug_images": False,
        "simulation.export_perturb_cell_center_debug_images": False,
        "simulation.release_analyzed_exported_frames": True,
        "simulation.prepare_analyze_one_frame": True,
        "simulation.parallel_threads": 32,
        "cell_lumen.enabled": False,
        "cell_lumen.fusionEnabled": False,
    }

    def original_candidate_overrides(failure_kind):
        split_recall = [
            ("n2v2_gamma145_edge055_split_recall", {
                "simulation.n2v2_preprocess.contrast.gamma": 1.45,
                "simulation.light_preprocess_gamma": 1.0,
                "prob.bio_bridge_min_edge_brightness_absolute": 0.055,
                "prob.split_cost": 260.0,
                "prob.split_cost_fraction": 0.004,
                "prob.P_split_base": 0.08,
                "prob.P_split_max": 0.70,
                "prob.split_candidates_per_attempt": 42,
                "prob.split_bridge_cost_rescue_enabled": True,
                "prob.split_bridge_cost_rescue_max_positive_fraction": 0.30,
                "prob.split_bridge_cost_rescue_max_valley_ratio": 0.65,
            }, "N2V2 with prior successful gamma family and moderate edge_too_dim relaxation"),
            ("n2v2_gamma150_edge050_bridge_open", {
                "simulation.n2v2_preprocess.contrast.gamma": 1.50,
                "simulation.light_preprocess_gamma": 1.0,
                "prob.bio_bridge_min_edge_brightness_absolute": 0.050,
                "prob.split_bridge_cost_rescue_enabled": True,
                "prob.split_bridge_cost_rescue_max_positive_fraction": 0.35,
                "prob.split_bridge_cost_rescue_max_valley_ratio": 0.75,
                "prob.split_bridge_cost_rescue_max_gap_density": 0.10,
                "prob.bio_bridge_no_valley_hard_threshold": 1.15,
                "prob.P_split_base": 0.08,
                "prob.P_split_max": 0.70,
            }, "N2V2 gamma 1.50 bridge-opening probe; avoids sub-1 environment lift"),
            ("n2v2_gamma145_edge045_control", {
                "simulation.n2v2_preprocess.contrast.gamma": 1.45,
                "simulation.light_preprocess_gamma": 1.0,
                "prob.bio_bridge_min_edge_brightness_absolute": 0.045,
                "prob.split_cost": 260.0,
                "prob.split_cost_fraction": 0.004,
                "prob.P_split_base": 0.08,
                "prob.P_split_max": 0.70,
                "prob.split_bridge_cost_rescue_enabled": True,
                "prob.split_bridge_cost_rescue_max_positive_fraction": 0.30,
                "prob.split_bridge_cost_rescue_max_valley_ratio": 0.65,
            }, "separate direct edge-threshold effect while holding N2V2 gamma in prior successful range"),
        ]
        extra_guard = [
            ("false_split_guard", {
                "prob.split_cost": 700.0,
                "prob.split_cost_fraction": 0.018,
                "prob.P_split_base": 0.02,
                "prob.P_split_max": 0.40,
                "prob.split_geometry_gate_enabled": True,
                "prob.split_max_daughter_seed_drift_fraction": 0.85,
                "prob.split_max_daughter_axis_expansion": 1.25,
                "prob.split_max_daughter_overlap_fraction": 0.03,
            }, "suppress extra cells/new-cell events caused by over-eager classic splits"),
            ("position_stability_guard", {
                "prob.position_prior_weight": 110.0,
                "prob.position_prior_threshold": 14.0,
                "prob.split_axis_alignment_gate_enabled": True,
                "prob.split_axis_alignment_min_angle_degrees": 28.0,
                "prob.split_daughter_refit_min_radius_fraction": 0.75,
                "prob.split_daughter_refit_max_radius_fraction": 1.00,
            }, "reduce center drift and geometry-walking false positives"),
            ("raw_contrast_conservative", {
                "simulation.frame_intensity_scale_low_percentile": 0.02,
                "simulation.frame_intensity_scale_high_percentile": 0.990,
                "simulation.contrast_background_floor": 0.025,
                "simulation.asymmetric_cost_threshold": 0.04,
            }, "make raw-frame contrast slightly more conservative around noisy frames"),
        ]
        center_stability = [
            ("center_stability_prior", {
                "prob.position_prior_weight": 100.0,
                "prob.position_prior_threshold": 12.0,
                "simulation.signal_guided_position_enabled": True,
                "simulation.signal_map_enabled": True,
                "simulation.signal_map_perturb_guidance_enabled": True,
            }, "stabilize matched-cell centers without changing CellLumen state"),
            ("raw_percentile_balance", {
                "simulation.frame_intensity_scale_low_percentile": 0.005,
                "simulation.frame_intensity_scale_high_percentile": 0.997,
                "simulation.frame_intensity_hard_max": 512.0,
                "prob.position_prior_weight": 90.0,
                "prob.position_prior_threshold": 16.0,
            }, "test whether frame brightness spread is driving centroid drift"),
            ("geometry_refit_tight", {
                "prob.split_geometry_gate_enabled": True,
                "prob.split_daughter_refit_iterations": 5,
                "prob.split_daughter_refit_min_radius_fraction": 0.80,
                "prob.split_daughter_refit_max_radius_fraction": 1.00,
                "prob.split_daughter_volume_scale": 0.78,
            }, "tighten daughter geometry when drift follows split refit"),
        ]
        if failure_kind == "missed_recall":
            return split_recall
        if failure_kind == "extra_suppression":
            return extra_guard
        return center_stability

    d.candidate_overrides = original_candidate_overrides

    original_parse_debug = d.parse_debug

    def parse_original_debug(run_dir):
        debug = original_parse_debug(run_dir)
        log_path = Path(run_dir) / "debug_log.txt"
        text = log_path.read_text(errors="ignore") if log_path.exists() else ""
        if "mode=none n2v2_enabled=0" in text or "preprocess_mode: none" in text:
            debug["preprocess_mode"] = "none"
        elif "mode=n2v2" in text or "preprocess_mode: n2v2" in text:
            debug["preprocess_mode"] = "n2v2"
        return debug

    d.parse_debug = parse_original_debug


    def corrected_gt_birth_rows(gt, frame):
        previous_labels = {g.get("label") for g in gt.get(frame - 1, [])} if frame > 0 else set()
        return [g for g in gt.get(frame, []) if g.get("parent", 0) > 0 and g.get("label") not in previous_labels]

    def corrected_evaluate_frames(run_dir, start, end, candidate, stage):
        gt = d.load_gt()
        pred = d.load_predictions(Path(run_dir) / "cells.csv", start, end)
        rows = []
        split_rows = []
        summary = {
            "candidate": candidate,
            "stage": stage,
            "run_dir": str(run_dir),
            "start": start,
            "end": end,
            "passed": True,
            "score": 0.0,
            "missing_cells": 0,
            "extra_cells": 0,
            "missed_splits": 0,
            "extra_splits": 0,
            "matched_cells": 0,
            "mean_matched_distance": 0.0,
            "max_matched_distance": 0.0,
            "duplicate_tolerated": 0,
            "failed_frames": [],
        }
        total_dist = 0.0
        for frame in range(start, end + 1):
            gt_rows = gt.get(frame, [])
            pred_rows = pred.get(frame, [])
            threshold = d.frame_match_threshold(frame)
            matches, missing, extra = d.greedy_match(gt_rows, pred_rows, threshold)
            missing, tolerated = d.tolerate_duplicate_gt(missing, matches)
            frame_dists = [m[2] for m in matches]
            gt_birth_rows = corrected_gt_birth_rows(gt, frame)
            pred_birth_rows = d.split_events_from_predictions(pred, frame)
            split_matches, split_missing, split_extra = d.greedy_match(gt_birth_rows, pred_birth_rows, threshold)
            if frame == start:
                split_missing, split_extra = [], []
            frame_pass = not missing and not extra and not split_missing and not split_extra and (max(frame_dists) if frame_dists else 0.0) <= threshold
            rows.append({
                "candidate": candidate,
                "stage": stage,
                "frame": frame,
                "gt_count": len(gt_rows),
                "pred_count": len(pred_rows),
                "matched": len(matches),
                "missing_cells": len(missing),
                "extra_cells": len(extra),
                "missed_splits": len(split_missing),
                "extra_splits": len(split_extra),
                "duplicate_tolerated": len(tolerated),
                "mean_matched_distance": f"{(sum(frame_dists) / len(frame_dists)) if frame_dists else 0.0:.6f}",
                "max_matched_distance": f"{max(frame_dists) if frame_dists else 0.0:.6f}",
                "match_threshold": f"{threshold:.6f}",
                "frame_pass": int(frame_pass),
            })
            for item in tolerated:
                split_rows.append({"frame": frame, "type": "gt_duplicate_suspect", "detail": d.json.dumps(item, sort_keys=True)})
            for item in missing:
                split_rows.append({"frame": frame, "type": "missing_cell", "detail": d.json.dumps(item, sort_keys=True)})
            for item in extra:
                split_rows.append({"frame": frame, "type": "extra_cell", "detail": d.json.dumps(item, sort_keys=True)})
            for item in split_missing:
                split_rows.append({"frame": frame, "type": "missed_split", "detail": d.json.dumps(item, sort_keys=True)})
            for item in split_extra:
                split_rows.append({"frame": frame, "type": "extra_new_cell", "detail": d.json.dumps(item, sort_keys=True)})
            summary["missing_cells"] += len(missing)
            summary["extra_cells"] += len(extra)
            summary["missed_splits"] += len(split_missing)
            summary["extra_splits"] += len(split_extra)
            summary["matched_cells"] += len(matches)
            summary["duplicate_tolerated"] += len(tolerated)
            total_dist += sum(frame_dists)
            summary["max_matched_distance"] = max(summary["max_matched_distance"], max(frame_dists) if frame_dists else 0.0)
            if not frame_pass:
                summary["passed"] = False
                summary["failed_frames"].append(frame)
        summary["mean_matched_distance"] = total_dist / summary["matched_cells"] if summary["matched_cells"] else 9999.0
        summary["score"] = 1000.0 * summary["missed_splits"] + 1000.0 * summary["extra_splits"] + 100.0 * summary["missing_cells"] + 100.0 * summary["extra_cells"] + summary["mean_matched_distance"]
        metrics_dir = d.DIRS["metrics"] / stage
        metrics_dir.mkdir(parents=True, exist_ok=True)
        d.write_csv(metrics_dir / f"{candidate}_metrics.csv", [
            "candidate", "stage", "frame", "gt_count", "pred_count", "matched",
            "missing_cells", "extra_cells", "missed_splits", "extra_splits",
            "duplicate_tolerated", "mean_matched_distance", "max_matched_distance", "match_threshold", "frame_pass",
        ], rows)
        d.write_csv(metrics_dir / f"{candidate}_events.csv", ["frame", "type", "detail"], split_rows)
        d.write_json(metrics_dir / f"{candidate}_summary.json", summary)
        d.append_jsonl(d.DECISIONS, {"time": d.now(), "event": "scored", **summary})
        return summary, rows

    def corrected_evaluate_frame(run_dir, frame, candidate, stage):
        first_pred_frame = 0
        cells_csv = Path(run_dir) / "cells.csv"
        if cells_csv.exists():
            with cells_csv.open(newline="") as f:
                reader = csv.DictReader(f)
                seen_frames = [d.parse_prediction_frame(row.get("file", row.get("frame", ""))) for row in reader]
            seen_frames = [fr for fr in seen_frames if fr is not None]
            if seen_frames:
                first_pred_frame = min(seen_frames)
        start = max(first_pred_frame, frame - 1)
        summary, rows = corrected_evaluate_frames(run_dir, start, frame, candidate, stage)
        frame_row = next((row for row in rows if int(row["frame"]) == frame), None)
        if frame_row is not None:
            frame_summary = dict(summary)
            frame_summary["passed"] = bool(int(frame_row["frame_pass"]))
            frame_summary["missing_cells"] = int(frame_row["missing_cells"])
            frame_summary["extra_cells"] = int(frame_row["extra_cells"])
            frame_summary["missed_splits"] = int(frame_row["missed_splits"])
            frame_summary["extra_splits"] = int(frame_row["extra_splits"])
            return frame_summary, frame_row
        return summary, None

    d.evaluate_frames = corrected_evaluate_frames
    d.evaluate_frame = corrected_evaluate_frame

    original_start_continuous = d.start_continuous

    def start_original(stop_old=False):
        d.ensure_dirs()
        d.generate_gt(force=True)
        if stop_old:
            d.stop_old_workers()
        baseline_config = d.make_config(
            "original_baseline_n2v2_noncelllumen_v001",
            CONFIG_SRC,
            {},
            "initial original-tracker scout from frame 0; N2V2 on and CellLumen disabled",
            "original CellUniverse tracker with N2V2 preprocessing, GT watched through frame 194",
        )
        initial, _ = d.initial_csv_for(0)
        run = d.Run(
            "original_baseline_n2v2_noncelllumen_v001",
            "original_n2v2_continuation_000_200_gt_to_194",
            0,
            LAST_RUN_FRAME,
            baseline_config,
            initial,
            d.CONT_THREADS,
        )
        run.launch()
        parent_config = baseline_config
        while True:
            failure_frame, frame_summary, debug = d.poll_run_until_failure(run, max(0, run.passed_until + 1))
            if failure_frame is None:
                if run.proc.poll() is None and run.passed_until < d.LAST_GT_FRAME:
                    time.sleep(10)
                    continue
                d.append_log("Original continuous run finished", [
                    f"candidate: `{run.name}`",
                    f"passed until GT frame: `{run.passed_until}`",
                    f"run requested through frame: `{LAST_RUN_FRAME}`",
                    f"run: `{run.run_dir}`",
                    "GT ends at frame 194; frames 195-200 are output-only unless another GT source is provided.",
                ])
                break
            run.terminate(f"failure at frame {failure_frame}")
            new_run, new_config = d.run_candidate_batch(failure_frame, frame_summary, debug, parent_config, source_run=run.run_dir)
            if new_run is None:
                d.log_status(status="blocked", blocked_frame=failure_frame)
                break
            run = new_run
            parent_config = new_config
        d.log_status(status="finished_or_blocked")

    def continue_original_from(run, parent_config):
        while True:
            failure_frame, frame_summary, debug = d.poll_run_until_failure(run, max(0, run.passed_until + 1))
            if failure_frame is None:
                if run.proc.poll() is None and run.passed_until < d.LAST_GT_FRAME:
                    time.sleep(10)
                    continue
                d.append_log("Original continuous run finished", [
                    f"candidate: `{run.name}`",
                    f"passed until GT frame: `{run.passed_until}`",
                    f"run requested through frame: `{LAST_RUN_FRAME}`",
                    f"run: `{run.run_dir}`",
                    "GT ends at frame 194; frames 195-200 are output-only unless another GT source is provided.",
                ])
                break
            run.terminate(f"failure at frame {failure_frame}")
            new_run, new_config = d.run_candidate_batch(failure_frame, frame_summary, debug, parent_config, source_run=run.run_dir)
            if new_run is None:
                d.log_status(status="blocked", blocked_frame=failure_frame)
                break
            run = new_run
            parent_config = new_config
        d.log_status(status="finished_or_blocked")

    def start_brightness1(stop_old=False):
        d.ensure_dirs()
        d.generate_gt(force=True)
        if stop_old:
            d.stop_old_workers()
        config = d.make_config(
            "original_baseline_n2v2_noncelllumen_initbright1_v002",
            CONFIG_SRC,
            {
                "cell.initialBrightness": 1.0,
                "cell.maxBrightness": 1.0,
                "simulation.n2v2_preprocess.contrast.gamma": 1.45,
                "simulation.light_preprocess_gamma": 1.0,
            },
            "restart corrected original/N2V2 non-CellLumen baseline with initial cell brightness forced to 1",
            "test whether high initial cell brightness prevents N2V2-shaped cells from absorbing surrounding environment and inflating radii",
        )
        initial, _ = d.initial_csv_for(0)
        run = d.Run(
            "original_baseline_n2v2_noncelllumen_initbright1_v002",
            "original_n2v2_initbright1_000_200_gt_to_194",
            0,
            LAST_RUN_FRAME,
            config,
            initial,
            d.CONT_THREADS,
        )
        d.append_log("Starting initial-brightness-1 N2V2 baseline", [
            "correction: original pipeline keeps N2V2 preprocessing enabled and CellLumen disabled",
            "override: `cell.initialBrightness=1.0` and `cell.maxBrightness=1.0`",
            "gamma caution: N2V2 contrast gamma held at `1.45`; raw light gamma held at `1.0`",
            f"config: `{config}`",
            f"initial csv: `{initial}`",
        ])
        run.launch()
        continue_original_from(run, config)

    def start_brightness1_trash_guard(stop_old=False):
        d.ensure_dirs()
        d.generate_gt(force=True)
        if stop_old:
            d.stop_old_workers()
        parent = d.DIRS["configs"] / "original_baseline_n2v2_noncelllumen_initbright1_v002.yaml"
        if not parent.exists():
            parent = CONFIG_SRC
        config = d.make_config(
            "original_baseline_n2v2_noncelllumen_initbright1_trashguard_v003",
            parent,
            {
                "cell.initialBrightness": 1.0,
                "cell.maxBrightness": 1.0,
                "cell.trashRemovalEnabled": False,
                "cell.trashRemovalBrightnessThreshold": 0.0,
                "cell.trashPcaShapeFitEnabled": False,
                "cell.trashPcaShapeUpdatePosition": False,
                "cell.trashPcaShapeMaxOriginalRadiusFactor": 1.0,
                "simulation.n2v2_preprocess.contrast.gamma": 1.45,
                "simulation.light_preprocess_gamma": 1.0,
            },
            "restart from frame 0 after early trash object elongated/disappeared too soon",
            "keep trash support alive as a non-scored environment guard and prevent trash PCA elongation from stealing surrounding signal",
        )
        initial, _ = d.initial_csv_for(0)
        run = d.Run(
            "original_baseline_n2v2_noncelllumen_initbright1_trashguard_v003",
            "original_n2v2_initbright1_trashguard_000_200_gt_to_194",
            0,
            LAST_RUN_FRAME,
            config,
            initial,
            d.CONT_THREADS,
        )
        d.append_log("Starting initial-brightness-1 trash-guard N2V2 baseline", [
            "user diagnosis: a trash/guard cell elongated too much and disappeared too early, already at frame 0",
            "overrides: keep `cell.initialBrightness=1.0`; disable trash removal; disable trash PCA shape fitting; hold trash max radius factor at 1.0",
            "gamma caution: N2V2 contrast gamma held at `1.45`; raw light gamma held at `1.0`",
            f"parent config: `{parent}`",
            f"config: `{config}`",
            f"initial csv: `{initial}`",
        ])
        run.launch()
        continue_original_from(run, config)

    def start_brightness1_trash_elong125(stop_old=False):
        d.ensure_dirs()
        d.generate_gt(force=True)
        if stop_old:
            d.stop_old_workers()
        parent = d.DIRS["configs"] / "original_baseline_n2v2_noncelllumen_initbright1_v002.yaml"
        if not parent.exists():
            parent = CONFIG_SRC
        config = d.make_config(
            "original_baseline_n2v2_noncelllumen_initbright1_trashmaxelong125_v004",
            parent,
            {
                "cell.initialBrightness": 1.0,
                "cell.maxBrightness": 1.0,
                "cell.trashPcaShapeMaxElongation": 1.25,
                "simulation.n2v2_preprocess.contrast.gamma": 1.45,
                "simulation.light_preprocess_gamma": 1.0,
            },
            "restart from frame 0 with known trash max elongation fix",
            "cap trash-cell elongation at 1.25 so the early trash guard does not elongate/disappear and destabilize later fits",
        )
        initial, _ = d.initial_csv_for(0)
        run = d.Run(
            "original_baseline_n2v2_noncelllumen_initbright1_trashmaxelong125_v004",
            "original_n2v2_initbright1_trashmaxelong125_000_200_gt_to_194",
            0,
            LAST_RUN_FRAME,
            config,
            initial,
            d.CONT_THREADS,
        )
        d.append_log("Starting known trash max elongation fix", [
            "user-known fix: set trash max elongation to `1.25` because a trash elongated too much and disappeared too early from frame 0",
            "overrides: `cell.initialBrightness=1.0`, `cell.maxBrightness=1.0`, `cell.trashPcaShapeMaxElongation=1.25`",
            "gamma caution: N2V2 contrast gamma held at `1.45`; raw light gamma held at `1.0`",
            f"parent config: `{parent}`",
            f"config: `{config}`",
            f"initial csv: `{initial}`",
        ])
        run.launch()
        continue_original_from(run, config)

    def start_brightness1_oldinit_trash_elong125(stop_old=False):
        d.ensure_dirs()
        d.generate_gt(force=True)
        if stop_old:
            d.stop_old_workers()
        parent = d.DIRS["configs"] / "original_baseline_n2v2_noncelllumen_initbright1_trashmaxelong125_v004.yaml"
        if not parent.exists():
            parent = d.DIRS["configs"] / "original_baseline_n2v2_noncelllumen_initbright1_v002.yaml"
        if not parent.exists():
            parent = CONFIG_SRC
        config = d.make_config(
            "original_baseline_n2v2_noncelllumen_initbright1_oldinit_trashmaxelong125_v005",
            parent,
            {
                "cell.initialBrightness": 1.0,
                "cell.maxBrightness": 1.0,
                "cell.trashPcaShapeMaxElongation": 1.25,
                "simulation.n2v2_preprocess.contrast.gamma": 1.45,
                "simulation.light_preprocess_gamma": 1.0,
            },
            "restart from frame 0 with old-init trash seeds and known trash elongation cap",
            "test the user-known trash fix under the same frame-0 trash/guard condition seen in prior successful records",
        )
        old_initial = ROOT.parent / "tuning_fluo_gt_20260529/continuous_gt_watch_r26/runs/r26_oldinit_t004_compare/r26_oldinit_t004_compare_classic_oldinit_currentcost/initial_used.csv"
        initial = d.DIRS["initials"] / "initial_000_oldinit_with_trash_v005.csv"
        shutil.copy2(old_initial, initial)
        run = d.Run(
            "original_baseline_n2v2_noncelllumen_initbright1_oldinit_trashmaxelong125_v005",
            "original_n2v2_initbright1_oldinit_trashmaxelong125_000_200_gt_to_194",
            0,
            LAST_RUN_FRAME,
            config,
            initial,
            d.CONT_THREADS,
        )
        d.append_log("Starting old-init trash max elongation fix", [
            "user-known fix: `cell.trashPcaShapeMaxElongation=1.25`",
            "critical correction: v004 used the four-cell embryo initial and had no explicit trash row, so the trash cap could not fire",
            "old-init source includes `trash_1` and `trash_2` at frame 0, matching prior successful records",
            "overrides: `cell.initialBrightness=1.0`, `cell.maxBrightness=1.0`, `cell.trashPcaShapeMaxElongation=1.25`",
            "gamma caution: N2V2 contrast gamma held at `1.45`; raw light gamma held at `1.0`",
            f"parent config: `{parent}`",
            f"config: `{config}`",
            f"initial csv: `{initial}`",
        ])
        run.launch()
        continue_original_from(run, config)

    def start_brightness1_oldinit_trash_elong125_v006(stop_old=False):
        d.ensure_dirs()
        d.generate_gt(force=True)
        if stop_old:
            d.stop_old_workers()
        parent = d.DIRS["configs"] / "original_baseline_n2v2_noncelllumen_initbright1_oldinit_trashmaxelong125_v005.yaml"
        if not parent.exists():
            parent = d.DIRS["configs"] / "original_baseline_n2v2_noncelllumen_initbright1_trashmaxelong125_v004.yaml"
        if not parent.exists():
            parent = CONFIG_SRC
        config = d.make_config(
            "original_baseline_n2v2_noncelllumen_initbright1_oldinit_trashmaxelong125_v006",
            parent,
            {
                "cell.initialBrightness": 1.0,
                "cell.maxBrightness": 1.0,
                "cell.trashPcaShapeMaxElongation": 1.25,
                "simulation.n2v2_preprocess.contrast.gamma": 1.45,
                "simulation.light_preprocess_gamma": 1.0,
            },
            "restart from frame 0 with normalized old-init trash seeds and trash elongation cap",
            "preserve prior old-init trash/guard condition while converting blank radius fields into current CSV schema",
        )
        old_initial = ROOT.parent / "tuning_fluo_gt_20260529/continuous_gt_watch_r26/runs/r26_oldinit_t004_compare/r26_oldinit_t004_compare_classic_oldinit_currentcost/initial_used.csv"
        initial = d.DIRS["initials"] / "initial_000_oldinit_with_trash_normalized_v006.csv"
        with old_initial.open(newline="") as src, initial.open("w", newline="") as dst:
            reader = csv.DictReader(src)
            fields = ["file", "name", "x", "y", "z", "aRadius", "bRadius", "cRadius", "theta_x", "theta_y", "theta_z", "isTrash"]
            writer = csv.DictWriter(dst, fieldnames=fields)
            writer.writeheader()
            for row in reader:
                radius = row.get("bRadius") or row.get("majorRadius") or row.get("minorRadius") or "10"
                name = row.get("name", "")
                writer.writerow({
                    "file": row.get("file", "t000.tif"),
                    "name": name,
                    "x": row.get("x", ""),
                    "y": row.get("y", ""),
                    "z": row.get("z", ""),
                    "aRadius": row.get("majorRadius") or radius,
                    "bRadius": radius,
                    "cRadius": row.get("minorRadius") or radius,
                    "theta_x": row.get("theta_x", "0"),
                    "theta_y": row.get("theta_y", "0"),
                    "theta_z": row.get("theta_z", "0"),
                    "isTrash": "1" if name.startswith("trash") else "0",
                })
        run = d.Run(
            "original_baseline_n2v2_noncelllumen_initbright1_oldinit_trashmaxelong125_v006",
            "original_n2v2_initbright1_oldinit_trashmaxelong125_000_200_gt_to_194_v006",
            0,
            LAST_RUN_FRAME,
            config,
            initial,
            d.CONT_THREADS,
        )
        d.append_log("Starting normalized old-init trash max elongation fix", [
            "v005 result: startup failed before tracking because the historical old-init CSV had blank radius fields in the current named-column schema",
            "normalization: copied old-init rows, mapped blank `aRadius/cRadius` to the stored `bRadius`, and marked names starting with `trash` as `isTrash=1`",
            "user-known fix: `cell.trashPcaShapeMaxElongation=1.25`",
            "overrides: `cell.initialBrightness=1.0`, `cell.maxBrightness=1.0`, `cell.trashPcaShapeMaxElongation=1.25`",
            "gamma caution: N2V2 contrast gamma held at `1.45`; raw light gamma held at `1.0`",
            f"parent config: `{parent}`",
            f"config: `{config}`",
            f"initial csv: `{initial}`",
        ])
        run.launch()
        continue_original_from(run, config)

    def start_brightness1_oldinit_trash_elong125_v007_newcode(stop_old=False):
        d.ensure_dirs()
        d.generate_gt(force=True)
        if stop_old:
            d.stop_old_workers()
        parent = d.DIRS["configs"] / "original_baseline_n2v2_noncelllumen_initbright1_oldinit_trashmaxelong125_v006.yaml"
        if not parent.exists():
            parent = d.DIRS["configs"] / "original_baseline_n2v2_noncelllumen_initbright1_trashmaxelong125_v004.yaml"
        if not parent.exists():
            parent = CONFIG_SRC
        config = d.make_config(
            "original_baseline_n2v2_noncelllumen_initbright1_oldinit_trashmaxelong125_newcode_v007",
            parent,
            {
                "cell.initialBrightness": 1.0,
                "cell.maxBrightness": 1.0,
                "cell.trashPcaShapeMaxElongation": 1.25,
                "simulation.n2v2_preprocess.contrast.gamma": 1.45,
                "simulation.light_preprocess_gamma": 1.0,
            },
            "fresh restart after debug/recompile with normalized old-init trash seeds and trash elongation cap",
            "use the newly compiled binary while preserving original pipeline, N2V2 preprocessing, CellLumen off, brightness 1, and the known trash elongation fix",
        )
        old_initial = ROOT.parent / "tuning_fluo_gt_20260529/continuous_gt_watch_r26/runs/r26_oldinit_t004_compare/r26_oldinit_t004_compare_classic_oldinit_currentcost/initial_used.csv"
        initial = d.DIRS["initials"] / "initial_000_oldinit_with_trash_normalized_newcode_v007.csv"
        with old_initial.open(newline="") as src, initial.open("w", newline="") as dst:
            reader = csv.DictReader(src)
            fields = ["file", "name", "x", "y", "z", "aRadius", "bRadius", "cRadius", "theta_x", "theta_y", "theta_z", "isTrash"]
            writer = csv.DictWriter(dst, fieldnames=fields)
            writer.writeheader()
            for row in reader:
                radius = row.get("bRadius") or row.get("majorRadius") or row.get("minorRadius") or "10"
                name = row.get("name", "")
                writer.writerow({
                    "file": row.get("file", "t000.tif"),
                    "name": name,
                    "x": row.get("x", ""),
                    "y": row.get("y", ""),
                    "z": row.get("z", ""),
                    "aRadius": row.get("majorRadius") or radius,
                    "bRadius": radius,
                    "cRadius": row.get("minorRadius") or radius,
                    "theta_x": row.get("theta_x", "0"),
                    "theta_y": row.get("theta_y", "0"),
                    "theta_z": row.get("theta_z", "0"),
                    "isTrash": "1" if name.startswith("trash") else "0",
                })
        run = d.Run(
            "original_baseline_n2v2_noncelllumen_initbright1_oldinit_trashmaxelong125_newcode_v007",
            "original_n2v2_initbright1_oldinit_trashmaxelong125_newcode_000_200_gt_to_194_v007",
            0,
            LAST_RUN_FRAME,
            config,
            initial,
            d.CONT_THREADS,
        )
        d.append_log("Starting fresh new-code old-init trash max elongation fix", [
            "user instruction: prune existing old runs and start a new run based on the newly debugged/recompiled code",
            "process prune: stopped active v006 watcher/CellUniverse and stale gamma canary workers before launch",
            "initial normalization: old-init rows converted to current `aRadius,bRadius,cRadius,isTrash` schema; `trash_1` and `trash_2` retained",
            "user-known fix: `cell.trashPcaShapeMaxElongation=1.25`",
            "overrides: `cell.initialBrightness=1.0`, `cell.maxBrightness=1.0`, `cell.trashPcaShapeMaxElongation=1.25`",
            "gamma caution: N2V2 contrast gamma held at `1.45`; raw light gamma held at `1.0`",
            f"parent config: `{parent}`",
            f"config: `{config}`",
            f"initial csv: `{initial}`",
        ])
        run.launch()
        continue_original_from(run, config)

    def start_brightness1_oldinit_trash_elong125_v008_newcode_restart(stop_old=False):
        d.ensure_dirs()
        d.generate_gt(force=True)
        if stop_old:
            d.stop_old_workers()
        parent = d.DIRS["configs"] / "original_baseline_n2v2_noncelllumen_initbright1_oldinit_trashmaxelong125_v006.yaml"
        if not parent.exists():
            parent = d.DIRS["configs"] / "original_baseline_n2v2_noncelllumen_initbright1_trashmaxelong125_v004.yaml"
        if not parent.exists():
            parent = CONFIG_SRC
        config = d.make_config(
            "original_baseline_n2v2_noncelllumen_initbright1_oldinit_trashmaxelong125_newcode_restart_v008",
            parent,
            {
                "cell.initialBrightness": 1.0,
                "cell.maxBrightness": 1.0,
                "cell.trashPcaShapeMaxElongation": 1.25,
                "simulation.n2v2_preprocess.contrast.gamma": 1.45,
                "simulation.light_preprocess_gamma": 1.0,
            },
            "fresh restart after debug/recompile with normalized old-init trash seeds and trash elongation cap",
            "use the newly compiled binary while preserving original pipeline, N2V2 preprocessing, CellLumen off, brightness 1, and the known trash elongation fix",
        )
        old_initial = ROOT.parent / "tuning_fluo_gt_20260529/continuous_gt_watch_r26/runs/r26_oldinit_t004_compare/r26_oldinit_t004_compare_classic_oldinit_currentcost/initial_used.csv"
        initial = d.DIRS["initials"] / "initial_000_oldinit_with_trash_normalized_newcode_restart_v008.csv"
        with old_initial.open(newline="") as src, initial.open("w", newline="") as dst:
            reader = csv.DictReader(src)
            fields = ["file", "name", "x", "y", "z", "aRadius", "bRadius", "cRadius", "theta_x", "theta_y", "theta_z", "isTrash"]
            writer = csv.DictWriter(dst, fieldnames=fields)
            writer.writeheader()
            for row in reader:
                radius = row.get("bRadius") or row.get("majorRadius") or row.get("minorRadius") or "10"
                name = row.get("name", "")
                writer.writerow({
                    "file": row.get("file", "t000.tif"),
                    "name": name,
                    "x": row.get("x", ""),
                    "y": row.get("y", ""),
                    "z": row.get("z", ""),
                    "aRadius": row.get("majorRadius") or radius,
                    "bRadius": radius,
                    "cRadius": row.get("minorRadius") or radius,
                    "theta_x": row.get("theta_x", "0"),
                    "theta_y": row.get("theta_y", "0"),
                    "theta_z": row.get("theta_z", "0"),
                    "isTrash": "1" if name.startswith("trash") else "0",
                })
        run = d.Run(
            "original_baseline_n2v2_noncelllumen_initbright1_oldinit_trashmaxelong125_newcode_restart_v008",
            "original_n2v2_initbright1_oldinit_trashmaxelong125_newcode_restart_000_200_gt_to_194_v008",
            0,
            LAST_RUN_FRAME,
            config,
            initial,
            d.CONT_THREADS,
        )
        d.append_log("Starting v008 restart after interrupted v007", [
            "restart reason: v007 passed frame 0 but was interrupted during frame 1, so restart from frame 0 in a fresh namespace",
            "process prune: stopped active v006 watcher/CellUniverse and stale gamma canary workers before launch",
            "initial normalization: old-init rows converted to current `aRadius,bRadius,cRadius,isTrash` schema; `trash_1` and `trash_2` retained",
            "user-known fix: `cell.trashPcaShapeMaxElongation=1.25`",
            "overrides: `cell.initialBrightness=1.0`, `cell.maxBrightness=1.0`, `cell.trashPcaShapeMaxElongation=1.25`",
            "gamma caution: N2V2 contrast gamma held at `1.45`; raw light gamma held at `1.0`",
            f"parent config: `{parent}`",
            f"config: `{config}`",
            f"initial csv: `{initial}`",
        ])
        run.launch()
        continue_original_from(run, config)

    def start_gamma_resume_frame4(stop_old=False):
        d.ensure_dirs()
        d.generate_gt(force=True)
        if stop_old:
            d.stop_old_workers()
        parent_config = d.DIRS["configs"] / "original_baseline_n2v2_noncelllumen_v001.yaml"
        if not parent_config.exists():
            parent_config = d.make_config(
                "original_baseline_n2v2_noncelllumen_v001",
                CONFIG_SRC,
                {},
                "initial original-tracker scout from frame 0; N2V2 on and CellLumen disabled",
                "original CellUniverse tracker with N2V2 preprocessing, GT watched through frame 194",
            )
        source_run = d.DIRS["runs"] / "original_n2v2_continuation_000_200_gt_to_194" / "original_baseline_n2v2_noncelllumen_v001"
        frame_summary, frame_row = d.evaluate_frame(source_run, 4, "original_baseline_n2v2_noncelllumen_v001", "original_n2v2_resume_source_t004")
        debug = d.parse_debug(source_run)
        d.append_log("Gamma-focused resume at frame 4", [
            "reason: first original-pipeline failure was two missing GT cells / missed split births at frame 4",
            "debug diagnosis: split candidates rejected by dim bridge-edge evidence (`edge_too_dim`) and daughter overlap",
            "gamma plan: keep N2V2 gamma in the prior successful >=1.45 range; do not use sub-1 gamma because it can absorb surrounding environment",
            f"parent config: `{parent_config}`",
            f"source run: `{source_run}`",
            f"source frame summary: `{frame_row}`",
            f"source debug: `{debug}`",
        ])
        new_run, new_config = d.run_candidate_batch(4, frame_row or frame_summary, debug, parent_config, source_run=source_run)
        if new_run is None:
            d.log_status(status="blocked", blocked_frame=4)
            d.append_log("Gamma-focused resume blocked at frame 4", [
                "all gamma-aware candidates failed before completing the rewind/check window",
                "next likely knob: further lower `prob.bio_bridge_min_edge_brightness_absolute` or inspect split overlap geometry constraints",
            ])
            return
        continue_original_from(new_run, new_config)

    d.continue_original_from = continue_original_from
    d.start_brightness1 = start_brightness1
    d.start_brightness1_trash_guard = start_brightness1_trash_guard
    d.start_brightness1_trash_elong125 = start_brightness1_trash_elong125
    d.start_brightness1_oldinit_trash_elong125 = start_brightness1_oldinit_trash_elong125
    d.start_brightness1_oldinit_trash_elong125_v006 = start_brightness1_oldinit_trash_elong125_v006
    d.start_brightness1_oldinit_trash_elong125_v007_newcode = start_brightness1_oldinit_trash_elong125_v007_newcode
    d.start_brightness1_oldinit_trash_elong125_v008_newcode_restart = start_brightness1_oldinit_trash_elong125_v008_newcode_restart
    d.start_gamma_resume_frame4 = start_gamma_resume_frame4
    d.start_original = start_original
    d.start_continuous = original_start_continuous


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup")
    p_start = sub.add_parser("start")
    p_start.add_argument("--stop-old", action="store_true")
    p_gamma = sub.add_parser("resume-gamma4")
    sub.add_parser("start-n2v2")
    p_bright = sub.add_parser("start-brightness1")
    p_bright.add_argument("--stop-old", action="store_true")
    p_trash = sub.add_parser("start-brightness1-trashguard")
    p_trash.add_argument("--stop-old", action="store_true")
    p_trash_elong = sub.add_parser("start-brightness1-trash-elong125")
    p_trash_elong.add_argument("--stop-old", action="store_true")
    p_oldinit_trash_elong = sub.add_parser("start-brightness1-oldinit-trash-elong125")
    p_oldinit_trash_elong.add_argument("--stop-old", action="store_true")
    p_oldinit_trash_elong_v006 = sub.add_parser("start-brightness1-oldinit-trash-elong125-v006")
    p_oldinit_trash_elong_v006.add_argument("--stop-old", action="store_true")
    p_oldinit_trash_elong_v007 = sub.add_parser("start-brightness1-oldinit-trash-elong125-v007-newcode")
    p_oldinit_trash_elong_v007.add_argument("--stop-old", action="store_true")
    p_oldinit_trash_elong_v008 = sub.add_parser("start-brightness1-oldinit-trash-elong125-v008-newcode-restart")
    p_oldinit_trash_elong_v008.add_argument("--stop-old", action="store_true")
    p_gamma.add_argument("--stop-old", action="store_true")
    sub.add_parser("status")
    args = parser.parse_args()

    d = load_driver()
    if args.cmd == "setup":
        d.ensure_dirs()
        d.generate_gt(force=True)
        print(f"setup complete: {ROOT}")
    elif args.cmd == "start":
        d.start_original(stop_old=args.stop_old)
    elif args.cmd == "resume-gamma4":
        d.start_gamma_resume_frame4(stop_old=args.stop_old)
    elif args.cmd == "start-n2v2":
        d.start_original(stop_old=False)
    elif args.cmd == "start-brightness1":
        d.start_brightness1(stop_old=args.stop_old)
    elif args.cmd == "start-brightness1-trashguard":
        d.start_brightness1_trash_guard(stop_old=args.stop_old)
    elif args.cmd == "start-brightness1-trash-elong125":
        d.start_brightness1_trash_elong125(stop_old=args.stop_old)
    elif args.cmd == "start-brightness1-oldinit-trash-elong125":
        d.start_brightness1_oldinit_trash_elong125(stop_old=args.stop_old)
    elif args.cmd == "start-brightness1-oldinit-trash-elong125-v006":
        d.start_brightness1_oldinit_trash_elong125_v006(stop_old=args.stop_old)
    elif args.cmd == "start-brightness1-oldinit-trash-elong125-v007-newcode":
        d.start_brightness1_oldinit_trash_elong125_v007_newcode(stop_old=args.stop_old)
    elif args.cmd == "start-brightness1-oldinit-trash-elong125-v008-newcode-restart":
        d.start_brightness1_oldinit_trash_elong125_v008_newcode_restart(stop_old=args.stop_old)
    elif args.cmd == "status":
        d.print_status()


if __name__ == "__main__":
    main()
