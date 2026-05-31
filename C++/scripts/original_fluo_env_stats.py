#!/usr/bin/env python3
"""Extract environment statistics for original Fluo tuning runs."""

import argparse
import csv
import json
import re
from pathlib import Path


ROOT = Path("/home/puv/output_fluo/original_fluo_gt_watch_20260531")


PRE_RE = re.compile(
    r"\[Preprocess\] file=t(?P<frame>\d+)\.tif stage=(?P<stage>\S+) "
    r".*?min=(?P<min>[-+0-9.eE]+) max=(?P<max>[-+0-9.eE]+) "
    r"mean=(?P<mean>[-+0-9.eE]+) stddev=(?P<stddev>[-+0-9.eE]+)"
)
BG_RE = re.compile(
    r"\[Adaptive Background\] frame (?P<frame>\d+) base=(?P<base>[-+0-9.eE]+) "
    r"ratio=(?P<ratio>[-+0-9.eE]+) background=(?P<background>[-+0-9.eE]+)"
)
SAVED_RE = re.compile(r"Saved\s+(?P<count>\d+)\s+cells\s+for\s+frame\s+(?P<frame>\d+)")


def latest_run(root):
    runs = sorted((root / "runs").glob("*/*/debug_log.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError(f"no debug logs under {root / 'runs'}")
    return runs[0].parent


def load_metrics(root, run_dir):
    out = {}
    try:
        stage = run_dir.parent.name
        candidate = run_dir.name
    except Exception:
        return out
    path = root / "metrics" / stage / f"{candidate}_metrics.csv"
    if not path.exists():
        return out
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            frame = int(row["frame"])
            out[frame] = {
                "gt_count": row.get("gt_count", ""),
                "pred_count_scored": row.get("pred_count", ""),
                "missing_cells": row.get("missing_cells", ""),
                "extra_cells": row.get("extra_cells", ""),
                "missed_splits": row.get("missed_splits", ""),
                "extra_splits": row.get("extra_splits", ""),
                "max_matched_distance": row.get("max_matched_distance", ""),
                "frame_pass": row.get("frame_pass", ""),
            }
    return out


def extract(run_dir, root):
    text = (run_dir / "debug_log.txt").read_text(errors="ignore")
    rows = {}
    for m in PRE_RE.finditer(text):
        if m.group("stage") not in {"processed_sequence", "post_interpolation"}:
            continue
        frame = int(m.group("frame"))
        prefix = m.group("stage")
        row = rows.setdefault(frame, {"frame": frame})
        for key in ("min", "max", "mean", "stddev"):
            row[f"{prefix}_{key}"] = m.group(key)
    for m in BG_RE.finditer(text):
        frame = int(m.group("frame"))
        row = rows.setdefault(frame, {"frame": frame})
        row["adaptive_background_base"] = m.group("base")
        row["adaptive_background_ratio"] = m.group("ratio")
        row["adaptive_background"] = m.group("background")
    for m in SAVED_RE.finditer(text):
        frame = int(m.group("frame"))
        row = rows.setdefault(frame, {"frame": frame})
        row["saved_cell_count"] = m.group("count")

    metrics = load_metrics(root, run_dir)
    for frame, metric in metrics.items():
        rows.setdefault(frame, {"frame": frame}).update(metric)
    return [rows[k] for k in sorted(rows)]


def write_outputs(root, run_dir, rows):
    out_dir = root / "docs"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "environment_stats_latest.csv"
    md_path = out_dir / "environment_observations.md"
    fields = [
        "frame",
        "processed_sequence_mean",
        "processed_sequence_stddev",
        "post_interpolation_mean",
        "post_interpolation_stddev",
        "adaptive_background",
        "saved_cell_count",
        "gt_count",
        "pred_count_scored",
        "missing_cells",
        "extra_cells",
        "missed_splits",
        "extra_splits",
        "max_matched_distance",
        "frame_pass",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    failed = [r for r in rows if str(r.get("frame_pass", "")).strip() == "0"]
    latest = rows[-1] if rows else {}
    with md_path.open("w") as f:
        f.write("# Original Fluo Environment Observations\n\n")
        f.write(f"- run: `{run_dir}`\n")
        f.write(f"- latest parsed frame: `{latest.get('frame', '')}`\n")
        f.write(f"- first failed scored frame: `{failed[0].get('frame') if failed else ''}`\n")
        f.write("- tracked fields: processed/post-interpolation brightness mean and standard deviation, adaptive background, saved cell count, GT count, and GT mismatch counts.\n\n")
        f.write("## Recent Frames\n\n")
        f.write("| frame | mean | stddev | saved | GT | miss | extra | missed split | extra split | pass |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows[-12:]:
            f.write(
                f"| {row.get('frame', '')} | {row.get('post_interpolation_mean', row.get('processed_sequence_mean', ''))} "
                f"| {row.get('post_interpolation_stddev', row.get('processed_sequence_stddev', ''))} "
                f"| {row.get('saved_cell_count', '')} | {row.get('gt_count', '')} "
                f"| {row.get('missing_cells', '')} | {row.get('extra_cells', '')} "
                f"| {row.get('missed_splits', '')} | {row.get('extra_splits', '')} "
                f"| {row.get('frame_pass', '')} |\n"
            )
    return csv_path, md_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir or latest_run(args.root)
    rows = extract(run_dir, args.root)
    csv_path, md_path = write_outputs(args.root, run_dir, rows)
    print(json.dumps({"run_dir": str(run_dir), "rows": len(rows), "csv": str(csv_path), "md": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
