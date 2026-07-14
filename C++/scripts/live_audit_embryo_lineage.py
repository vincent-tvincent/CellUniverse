#!/usr/bin/env python3
"""Live center and lineage audit for CellUniverse embryo runs.

This script is an OpenLab friendly wrapper around
audit_embryo_lineage_topology.py. It waits for each completed frame, audits
center matching and parent child topology against fixed GT, prints one compact
line per frame, and writes an incremental CSV summary. It never writes tracker
inputs and never feeds GT information back into CellUniverse.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from audit_embryo_lineage_topology import (  # noqa: E402
    DEFAULT_GT,
    audit_topology,
    matched_maps,
    parse_frames,
    read_gt_with_life,
    write_summary,
)
from validate_embryo_centers import read_prediction  # noqa: E402


def checkpoint_path(run_dir: Path, frame: int) -> Path:
    return run_dir / "checkpoints" / f"frame_{frame:03d}.txt"


def frame_has_prediction(cells_csv: Path, frame: int) -> bool:
    expected_file = f"t{frame:03d}.tif"
    if not cells_csv.exists():
        return False
    try:
        with cells_csv.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("file") == expected_file:
                    return True
    except OSError:
        return False
    return False


def wait_for_stable_file(path: Path, stable_polls: int, poll_sec: float) -> bool:
    """Wait until a file size is unchanged for a few polls.

    The preferred completion marker is the checkpoint, but OpenLab users often
    want to watch a partially copied output folder. A short size stability check
    makes the cells-only mode less likely to read the file mid-write.
    """
    if not path.exists():
        return False
    last_size = -1
    stable_count = 0
    while stable_count < stable_polls:
        try:
            current_size = path.stat().st_size
        except OSError:
            return False
        if current_size == last_size:
            stable_count += 1
        else:
            stable_count = 0
            last_size = current_size
        time.sleep(poll_sec)
    return True


def frame_complete(
    run_dir: Path,
    cells_csv: Path,
    frame: int,
    completion_marker: str,
    stable_polls: int,
    poll_sec: float,
) -> bool:
    if completion_marker == "checkpoint":
        return checkpoint_path(run_dir, frame).exists()
    if not frame_has_prediction(cells_csv, frame):
        return False
    return wait_for_stable_file(cells_csv, stable_polls, poll_sec)


def audit_one_frame(
    cells_csv: Path,
    gt_csv: Path,
    frame: int,
    threshold: float,
    previous_maps: dict[str, object] | None,
    details: bool,
) -> tuple[dict[str, object], dict[str, object]]:
    pred = read_prediction(cells_csv, frame)
    gt = read_gt_with_life(gt_csv, frame)
    maps = matched_maps(pred, gt, threshold)
    missing: list[str] = maps["missing"]  # type: ignore[assignment]
    extra: list[str] = maps["extra"]  # type: ignore[assignment]
    center_status = "PASS" if not missing and not extra else "FAIL"
    lineage_errors = audit_topology(frame, gt, maps, previous_maps)
    lineage_status = "PASS" if not lineage_errors else "FAIL"
    row = {
        "frame": frame,
        "center_status": center_status,
        "lineage_status": lineage_status,
        "pred": len(pred),
        "gt": len(gt),
        "matched": maps["matched"],
        "missing": len(missing),
        "extra": len(extra),
        "mean_distance": f"{float(maps['mean_distance']):.6f}",
        "max_distance": f"{float(maps['max_distance']):.6f}",
        "missing_labels": ",".join(missing),
        "extra_names": ",".join(extra),
        "lineage_errors": " | ".join(lineage_errors),
    }

    print(
        f"f{frame:03d} center={center_status} lineage={lineage_status} "
        f"pred={len(pred)} gt={len(gt)} matched={maps['matched']} "
        f"missing={len(missing)} extra={len(extra)} "
        f"mean={float(maps['mean_distance']):.3f} "
        f"max={float(maps['max_distance']):.3f}",
        flush=True,
    )
    if details:
        if missing:
            print(f"  missing_labels={','.join(missing)}", flush=True)
        if extra:
            print(f"  extra_names={','.join(extra)}", flush=True)
        for error in lineage_errors[:12]:
            print(f"  lineage_error={error}", flush=True)
        if len(lineage_errors) > 12:
            print(f"  lineage_error=...+{len(lineage_errors) - 12}", flush=True)
    return row, maps


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Watch CellUniverse cells.csv output and print live center plus "
            "lineage audit results for every completed frame."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--frames", required=True, help="Frame list or range, for example 18-171")
    parser.add_argument("--gt", type=Path, default=Path(DEFAULT_GT))
    parser.add_argument("--threshold", type=float, default=25.0)
    parser.add_argument(
        "--previous-cells",
        type=Path,
        help="cells.csv containing the frame immediately before the first audited frame.",
    )
    parser.add_argument(
        "--previous-frame",
        type=int,
        help="Frame number read from --previous-cells for resume topology audit.",
    )
    parser.add_argument(
        "--completion-marker",
        choices=("checkpoint", "cells"),
        default="checkpoint",
        help="Use checkpoints for safest frame completion detection, or cells for copied folders without checkpoints.",
    )
    parser.add_argument("--poll-sec", type=float, default=2.0)
    parser.add_argument("--timeout-sec", type=float, default=0.0)
    parser.add_argument("--stable-polls", type=int, default=2)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--details", action="store_true")
    parser.add_argument("--stop-on-fail", action="store_true")
    args = parser.parse_args()

    frames = parse_frames(args.frames)
    if not frames:
        parser.error("--frames did not select any frame")
    if args.previous_cells is not None or args.previous_frame is not None:
        if args.previous_cells is None or args.previous_frame is None:
            parser.error("--previous-cells and --previous-frame must be provided together")

    run_dir = args.run_dir.expanduser().resolve()
    cells_csv = run_dir / "cells.csv"
    summary_csv = args.summary_csv
    if summary_csv is None:
        summary_csv = run_dir / "live_center_lineage_audit_summary.csv"

    print(
        f"[live-audit] run_dir={run_dir} frames={frames[0]}-{frames[-1]} "
        f"gt={args.gt} threshold={args.threshold:g} "
        f"completion_marker={args.completion_marker}",
        flush=True,
    )

    previous_maps: dict[str, object] | None = None
    if args.previous_cells is not None and args.previous_frame is not None:
        previous_pred = read_prediction(args.previous_cells, args.previous_frame)
        previous_gt = read_gt_with_life(args.gt, args.previous_frame)
        previous_maps = matched_maps(previous_pred, previous_gt, args.threshold)
        previous_missing: list[str] = previous_maps["missing"]  # type: ignore[assignment]
        previous_extra: list[str] = previous_maps["extra"]  # type: ignore[assignment]
        print(
            f"previous f{args.previous_frame:03d} "
            f"center={'PASS' if not previous_missing and not previous_extra else 'FAIL'} "
            f"pred={len(previous_pred)} gt={len(previous_gt)} "
            f"matched={previous_maps['matched']} "
            f"missing={len(previous_missing)} extra={len(previous_extra)}",
            flush=True,
        )

    rows: list[dict[str, object]] = []
    started = time.time()
    for frame in frames:
        while not frame_complete(
            run_dir,
            cells_csv,
            frame,
            args.completion_marker,
            max(1, args.stable_polls),
            max(0.1, args.poll_sec),
        ):
            if args.timeout_sec > 0 and time.time() - started > args.timeout_sec:
                print(f"[live-audit] timeout waiting for f{frame:03d}", file=sys.stderr)
                if rows:
                    write_summary(summary_csv, rows)
                return 2
            time.sleep(max(0.1, args.poll_sec))

        row, previous_maps = audit_one_frame(
            cells_csv,
            args.gt,
            frame,
            args.threshold,
            previous_maps,
            args.details,
        )
        rows.append(row)
        write_summary(summary_csv, rows)
        if args.stop_on_fail and (
            row["center_status"] != "PASS" or row["lineage_status"] != "PASS"
        ):
            print(f"[live-audit] stopping at f{frame:03d} due to audit failure", flush=True)
            return 1

    all_ok = all(
        row["center_status"] == "PASS" and row["lineage_status"] == "PASS"
        for row in rows
    )
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
