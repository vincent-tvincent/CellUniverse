#!/usr/bin/env python3
"""Report unusually large frame-to-frame jumps inside a GT centroid CSV.

This helper is only for audit sanity checks. It never changes tracking output
or the ground truth file. It is useful when a single failing frame may come from
a local GT identity/centroid discontinuity rather than a tracker decision.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def parse_point(row: dict[str, str]) -> tuple[float, float, float]:
    z_value = row.get("z_interp") or row.get("z") or row.get("z_scaled") or row.get("z_raw")
    if z_value is None:
        raise KeyError("GT row has no z, z_interp, z_scaled, or z_raw column")
    return float(row["x"]), float(row["y"]), float(z_value)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find large consecutive-frame centroid jumps in a GT CSV."
    )
    parser.add_argument("--gt", required=True, type=Path)
    parser.add_argument("--first-frame", type=int, default=None)
    parser.add_argument("--last-frame", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=25.0)
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args()

    by_label: dict[str, list[tuple[int, tuple[float, float, float]]]] = {}
    with args.gt.open(newline="") as f:
        for row in csv.DictReader(f):
            frame = int(float(row["frame"]))
            if args.first_frame is not None and frame < args.first_frame:
                continue
            if args.last_frame is not None and frame > args.last_frame:
                continue
            label = str(row["label_id"])
            by_label.setdefault(label, []).append((frame, parse_point(row)))

    jumps: list[tuple[float, str, int, int, tuple[float, float, float], tuple[float, float, float]]] = []
    for label, values in by_label.items():
        values.sort()
        for (prev_frame, prev_point), (frame, point) in zip(values, values[1:]):
            if frame != prev_frame + 1:
                continue
            distance = math.sqrt(sum((prev_point[i] - point[i]) ** 2 for i in range(3)))
            if distance >= args.threshold:
                jumps.append((distance, label, prev_frame, frame, prev_point, point))

    jumps.sort(reverse=True)
    print(
        f"GT jump audit: file={args.gt} frames={args.first_frame}-{args.last_frame} "
        f"threshold={args.threshold} jumps={len(jumps)}"
    )
    for distance, label, prev_frame, frame, prev_point, point in jumps[: args.top]:
        p0 = ",".join(f"{value:.3f}" for value in prev_point)
        p1 = ",".join(f"{value:.3f}" for value in point)
        print(
            f"label={label} {prev_frame}->{frame} jump={distance:.3f} "
            f"from=({p0}) to=({p1})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
