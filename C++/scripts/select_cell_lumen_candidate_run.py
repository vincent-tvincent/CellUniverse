#!/usr/bin/env python3
"""Select between a stable Cell Lumen trunk run and one optional repair run.

This helper exists because several experimental repairs are useful only in
specific temporal patterns. For example, dynamic parent-radius reach fixes a
large-parent daughter that persists in later frames, but it can also introduce
a one-frame extra. The selector therefore keeps the conservative trunk by
default and accepts the candidate run only when the candidate count is
temporally stable without looking at ground-truth labels.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import time
from pathlib import Path

from run_cell_lumen_pending_repair_audit import audit_rows
from run_cell_lumen_true_rolling_audit import (
    DEFAULT_GT,
    DEFAULT_INITIAL,
    DEFAULT_OUTPUT_ROOT,
    greedy_match,
    parse_frame_spec,
    read_final_summary,
    read_gt,
    write_csv,
)


def frame_dir(run_dir: Path, frame: int) -> Path:
    return run_dir / f"f{frame:03d}"


def summary_csv(run_dir: Path, frame: int) -> Path:
    return frame_dir(run_dir, frame) / "fine_shape_biofused" / f"t{frame:03d}_fine_shape_summary.csv"


def rolling_prior_csv(run_dir: Path, frame: int) -> Path:
    return frame_dir(run_dir, frame) / f"rolling_prior_t{frame:03d}_for_next.csv"


def copy_frame_source(source_run: Path, selected_run: Path, frame: int) -> Path:
    source_summary = summary_csv(source_run, frame)
    if not source_summary.exists():
        raise FileNotFoundError(source_summary)
    selected_frame = frame_dir(selected_run, frame)
    selected_summary_dir = selected_frame / "fine_shape_biofused"
    selected_summary_dir.mkdir(parents=True, exist_ok=True)
    selected_summary = selected_summary_dir / source_summary.name
    shutil.copy2(source_summary, selected_summary)

    source_prior = rolling_prior_csv(source_run, frame)
    if source_prior.exists():
        shutil.copy2(source_prior, selected_frame / source_prior.name)
    return selected_summary


def select_source(
    frame: int,
    frames: list[int],
    baseline_counts: dict[int, int],
    candidate_counts: dict[int, int],
    previous_selected_count: int | None,
) -> tuple[str, str]:
    baseline_count = baseline_counts[frame]
    candidate_count = candidate_counts[frame]
    next_frame = frame + 1
    next_candidate_count = candidate_counts.get(next_frame)

    if candidate_count <= baseline_count:
        return "baseline", "centroid_trunk"

    # Accept a candidate count rescue only when the added object is not an
    # isolated one-frame count spike. This is intentionally GT-free: it checks
    # whether the candidate hypothesis is temporally self-consistent, not
    # whether it matches the audit labels.
    if next_candidate_count is not None and candidate_count == next_candidate_count:
        return "candidate", "stable_candidate_count_rescue"

    # At the right boundary of a segment, use the already selected previous
    # count as the continuity reference. This avoids rejecting the final frame
    # of a stable repaired chain just because the next frame was not run yet.
    if frame == frames[-1] and previous_selected_count == candidate_count:
        return "candidate", "boundary_continues_selected_count"

    return "baseline", "candidate_count_spike_rejected"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", required=True)
    parser.add_argument("--baseline-run-dir", type=Path, required=True)
    parser.add_argument("--candidate-run-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--gt-csv", type=Path, default=DEFAULT_GT)
    parser.add_argument("--threshold", type=float, default=25.0)
    parser.add_argument("--initial", type=Path, default=DEFAULT_INITIAL)
    args = parser.parse_args()

    frames = parse_frame_spec(args.frames)
    stamp = time.strftime("%Y%m%d.%H%M%S")
    run_name = args.run_name or (
        f"⚠️{frames[0]:03d}-{frames[-1]:03d}_SelectedCandidateRun_{stamp}"
    )
    run_dir = args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    baseline_counts = {
        frame: len(read_final_summary(summary_csv(args.baseline_run_dir, frame)))
        for frame in frames
    }
    candidate_counts = {
        frame: len(read_final_summary(summary_csv(args.candidate_run_dir, frame)))
        for frame in frames
    }
    gt_by_frame = read_gt(args.gt_csv, frames)

    selected_rows: list[dict[str, str | int]] = []
    detail_rows: list[dict[str, str | int | float]] = []
    previous_selected_count: int | None = None

    for frame in frames:
        source_name, reason = select_source(
            frame,
            frames,
            baseline_counts,
            candidate_counts,
            previous_selected_count,
        )
        source_run = args.candidate_run_dir if source_name == "candidate" else args.baseline_run_dir
        selected_summary = copy_frame_source(source_run, run_dir, frame)
        pred_rows = read_final_summary(selected_summary)
        summary, details = audit_rows(frame, gt_by_frame[frame], pred_rows, args.threshold)
        summary["selected_source"] = source_name
        summary["selection_reason"] = reason
        summary["baseline_pred"] = baseline_counts[frame]
        summary["candidate_pred"] = candidate_counts[frame]
        summary["selected_summary_csv"] = str(selected_summary)
        selected_rows.append(summary)
        detail_rows.extend(details)
        previous_selected_count = len(pred_rows)

        print(
            f"f{frame:03d} {summary['status']} selected={source_name} "
            f"reason={reason} baseline={baseline_counts[frame]} "
            f"candidate={candidate_counts[frame]} pred={summary['pred']} "
            f"miss={summary['missing']} extra={summary['extra']} "
            f"max={float(summary['max_distance']):.2f}",
            flush=True,
        )
        if summary["missing_labels"]:
            print(f"  missing={summary['missing_labels']}", flush=True)
        if summary["extra_names"]:
            print(f"  extra={summary['extra_names']}", flush=True)

    if selected_rows:
        write_csv(
            run_dir
            / f"selected_candidate_audit_summary_f{frames[0]:03d}_f{frames[-1]:03d}.csv",
            selected_rows,
            list(selected_rows[0].keys()),
        )
    if detail_rows:
        write_csv(
            run_dir / f"selected_candidate_audit_details_f{frames[0]:03d}_f{frames[-1]:03d}.csv",
            detail_rows,
            list(detail_rows[0].keys()),
        )
    print(f"RUN_DIR={run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
