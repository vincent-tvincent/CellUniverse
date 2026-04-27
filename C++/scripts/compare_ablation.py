#!/usr/bin/env python3
"""Ablation-test comparison tool.

Reads two run output directories (cells.csv + debug_log.txt) and emits a
side-by-side metrics delta report. Intended for use with the late-frame
ablation protocol documented at
`C++/docs/plans/2026-04-22-late-frame-ablation-plan.md`.

Usage:
  python3 compare_ablation.py <baseline_dir> <variant_dir> [--from-frame N]
"""

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path


def load_cells_csv(path):
    """Return {frame: {name: {x, y, z, aR, bR, cR}}}"""
    frames = defaultdict(dict)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            fn = int(row["file"].replace("frame", "").replace(".tif", ""))
            frames[fn][row["name"]] = {
                "x": float(row["x"]),
                "y": float(row["y"]),
                "z": float(row["z"]),
                "aR": float(row["aRadius"]),
                "bR": float(row["bRadius"]),
                "cR": float(row["cRadius"]),
            }
    return frames


def extract_splits(log_path):
    """Yield (frame_num, cell_name, drift1, drift2, cost) for each Split Accepted.

    frame_num is the 1-indexed frame where the accept log appears (the Split
    Accepted line prints before the [Optimize Done] line for that same frame).
    """
    splits = []
    current_frame = 0  # will become 1 on the first Optimize Done line
    pending = []
    with open(log_path) as f:
        for line in f:
            m_opt = re.match(r"\[Optimize Done\] frame (\d+)", line)
            if m_opt:
                fn = int(m_opt.group(1))
                for p in pending:
                    p["frame"] = fn
                    splits.append(p)
                pending = []
                continue
            m_acc = re.match(
                r"\[Split Accepted\] (\S+) costDiff=([-\d.e+]+) "
                r"bestLabel=(\S+) .*?drift1=([\d.]+) .*?drift2=([\d.]+)",
                line,
            )
            if m_acc:
                pending.append(
                    {
                        "name": m_acc.group(1),
                        "cost_diff": float(m_acc.group(2)),
                        "label": m_acc.group(3),
                        "drift1": float(m_acc.group(4)),
                        "drift2": float(m_acc.group(5)),
                    }
                )
    return splits


def extract_bio_rejects(log_path):
    """Yield (reason, cell_name) for each bio reject, across the whole run."""
    rej = defaultdict(int)
    with open(log_path) as f:
        for line in f:
            m = re.match(r"\[Split Reject bio\] (\S+) reason=(\S+)", line)
            if m:
                reason = m.group(2).split("_")[0:2]
                # Group by leading 2 tokens: "d1_buried", "d2_bridging", "bridge_flat", "edge_too"
                key = "_".join(reason[:2]) if len(reason) >= 2 else reason[0]
                rej[key] += 1
    return dict(rej)


def bloat_metrics(frames, birth_a=30.0):
    """Per-frame bloat indicators.

    birth_a is a rough average aR at birth; more accurate would be per-cell
    birth lookup, but the average suffices for ablation delta reporting.
    """
    out = {}
    for fn, cells in frames.items():
        if not cells:
            continue
        aRs = [c["aR"] for c in cells.values()]
        bloated = sum(1 for a in aRs if a > 1.3 * birth_a)
        out[fn] = {
            "n_cells": len(cells),
            "max_aR": max(aRs),
            "mean_aR": sum(aRs) / len(aRs),
            "bloated_count": bloated,
        }
    return out


def frame_drift(frames):
    """For each (name, frame) return displacement from the previous frame.

    Skips the first frame a cell appears (no previous position).
    """
    out = defaultdict(list)  # frame -> list of drifts
    prev = {}
    for fn in sorted(frames):
        for name, cell in frames[fn].items():
            if name in prev:
                p = prev[name]
                d = math.sqrt(
                    (cell["x"] - p["x"]) ** 2
                    + (cell["y"] - p["y"]) ** 2
                    + (cell["z"] - p["z"]) ** 2
                )
                out[fn].append((name, d))
        prev = {name: cell for name, cell in frames[fn].items()}
    return out


def fmt_line(label, base_val, var_val, fmt="{}"):
    bs = fmt.format(base_val)
    vs = fmt.format(var_val)
    return f"  {label:36} {bs:>14}  {vs:>14}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline_dir")
    ap.add_argument("variant_dir")
    ap.add_argument("--from-frame", type=int, default=0,
                    help="Only compare frames >= this number")
    args = ap.parse_args()

    base = Path(args.baseline_dir)
    var = Path(args.variant_dir)

    base_cells = load_cells_csv(base / "cells.csv")
    var_cells = load_cells_csv(var / "cells.csv")
    base_splits = extract_splits(base / "debug_log.txt")
    var_splits = extract_splits(var / "debug_log.txt")
    base_rej = extract_bio_rejects(base / "debug_log.txt")
    var_rej = extract_bio_rejects(var / "debug_log.txt")

    min_f = args.from_frame
    base_bloat = {
        fn: v for fn, v in bloat_metrics(base_cells).items() if fn >= min_f
    }
    var_bloat = {
        fn: v for fn, v in bloat_metrics(var_cells).items() if fn >= min_f
    }
    base_drift = {
        fn: v for fn, v in frame_drift(base_cells).items() if fn >= min_f
    }
    var_drift = {
        fn: v for fn, v in frame_drift(var_cells).items() if fn >= min_f
    }

    base_splits_f = [s for s in base_splits if s["frame"] >= min_f]
    var_splits_f = [s for s in var_splits if s["frame"] >= min_f]

    print(f"=== Ablation comparison (from f{min_f}) ===")
    print(f"  baseline: {base.name}")
    print(f"  variant : {var.name}")
    print()

    # --- Cell-count trajectory ---
    frames = sorted(set(base_bloat) | set(var_bloat))
    print(f"  {'frame':<6}  {'base n':>6} {'var n':>6}  "
          f"{'base maxaR':>10} {'var maxaR':>10}  "
          f"{'base bloat':>10} {'var bloat':>10}")
    for fn in frames:
        b = base_bloat.get(fn, {})
        v = var_bloat.get(fn, {})
        print(
            f"  {fn:<6}  "
            f"{b.get('n_cells', '-'):>6} {v.get('n_cells', '-'):>6}  "
            f"{b.get('max_aR', 0):>10.1f} {v.get('max_aR', 0):>10.1f}  "
            f"{b.get('bloated_count', 0):>10} {v.get('bloated_count', 0):>10}"
        )
    print()

    # --- Aggregate metrics ---
    def agg_max_aR(bloat):
        if not bloat: return 0.0
        return max(b["max_aR"] for b in bloat.values())
    def agg_mean_aR(bloat):
        if not bloat: return 0.0
        vals = [b["mean_aR"] for b in bloat.values()]
        return sum(vals) / len(vals)
    def agg_bloat_frames(bloat):
        return sum(1 for b in bloat.values() if b["bloated_count"] > 0)

    def agg_drift_over(drift, thresh):
        return sum(
            sum(1 for _, d in frame_list if d > thresh)
            for frame_list in drift.values()
        )
    def agg_drift_mean(drift):
        all_d = [d for frame_list in drift.values() for _, d in frame_list]
        return sum(all_d) / max(1, len(all_d))

    print("=== Aggregate summary ===")
    print(f"  {'metric':<36} {'baseline':>14} {'variant':>14}")
    print(fmt_line("total splits accepted", len(base_splits_f), len(var_splits_f)))
    print(fmt_line("splits with drift1 or drift2 > 15",
                   sum(1 for s in base_splits_f if max(s['drift1'], s['drift2']) > 15),
                   sum(1 for s in var_splits_f if max(s['drift1'], s['drift2']) > 15)))
    print(fmt_line("max aR across all frames",
                   agg_max_aR(base_bloat), agg_max_aR(var_bloat), "{:.1f}"))
    print(fmt_line("mean-of-frame-mean aR",
                   agg_mean_aR(base_bloat), agg_mean_aR(var_bloat), "{:.1f}"))
    print(fmt_line("frames with any bloated cell (aR > 1.3 × 30)",
                   agg_bloat_frames(base_bloat), agg_bloat_frames(var_bloat)))
    print(fmt_line("cell-frames with drift > 15 vx",
                   agg_drift_over(base_drift, 15),
                   agg_drift_over(var_drift, 15)))
    print(fmt_line("cell-frames with drift > 25 vx",
                   agg_drift_over(base_drift, 25),
                   agg_drift_over(var_drift, 25)))
    print(fmt_line("mean per-frame drift (vx)",
                   agg_drift_mean(base_drift), agg_drift_mean(var_drift), "{:.2f}"))
    print()

    # --- Bio reject counts ---
    print("=== Bio-reject reasons ===")
    print(f"  {'reason':<36} {'baseline':>14} {'variant':>14}")
    reasons = sorted(set(base_rej) | set(var_rej))
    for r in reasons:
        print(fmt_line(r, base_rej.get(r, 0), var_rej.get(r, 0)))
    print()

    # --- Split diff ---
    def split_key(s):
        return (s["name"], s["frame"])
    base_keys = {split_key(s): s for s in base_splits_f}
    var_keys = {split_key(s): s for s in var_splits_f}
    only_base = sorted(base_keys.keys() - var_keys.keys(), key=lambda k: k[1])
    only_var = sorted(var_keys.keys() - base_keys.keys(), key=lambda k: k[1])
    print("=== Split diffs (cell, frame) ===")
    if only_base:
        print("  Only in baseline:")
        for (name, fn) in only_base[:15]:
            s = base_keys[(name, fn)]
            short = name[:8] + (':' + name[32:] if len(name) > 32 else '')
            print(f"    f{fn:<3} {short:14} d1={s['drift1']:.1f} d2={s['drift2']:.1f} cost={s['cost_diff']:.0f}")
        if len(only_base) > 15:
            print(f"    ... and {len(only_base) - 15} more")
    if only_var:
        print("  Only in variant:")
        for (name, fn) in only_var[:15]:
            s = var_keys[(name, fn)]
            short = name[:8] + (':' + name[32:] if len(name) > 32 else '')
            print(f"    f{fn:<3} {short:14} d1={s['drift1']:.1f} d2={s['drift2']:.1f} cost={s['cost_diff']:.0f}")
        if len(only_var) > 15:
            print(f"    ... and {len(only_var) - 15} more")


if __name__ == "__main__":
    main()
