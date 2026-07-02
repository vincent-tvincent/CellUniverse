#!/usr/bin/env python3
"""Export center anchored fine shape masks from an existing cell CSV.

This is a review and research helper for the optional fine shape model. It does
not change Cell Universe tracking, PCA fitting, or CellLumen detection. Given a
raw TIFF stack and a cells or initial CSV, it grows a local brightness connected
component around each provided center and writes compact RLE mask rows that can
be rendered by render_fine_shape_masks.py.
"""

from __future__ import annotations

import argparse
import csv
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from statistics import quantiles

import cv2
import numpy as np


@dataclass
class CellRow:
    frame: str
    name: str
    x: float
    y: float
    z_scaled: float
    r1: float
    r2: float
    r3: float


def parse_float(row: dict[str, str], names: tuple[str, ...], default: float = 0.0) -> float:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return float(value)
    return default


def load_cells(path: Path, frame_name: str | None) -> list[CellRow]:
    rows: list[CellRow] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("isTrash", "0").lower() in {"1", "true", "yes"}:
                continue
            frame = row.get("file") or row.get("frame") or ""
            if frame_name and frame and frame != frame_name:
                continue
            rows.append(
                CellRow(
                    frame=frame or frame_name or "",
                    name=row.get("name") or row.get("cell_name") or f"cell_{len(rows) + 1:04d}",
                    x=parse_float(row, ("x", "center_x")),
                    y=parse_float(row, ("y", "center_y")),
                    z_scaled=parse_float(row, ("z", "z_scaled", "center_z", "center_z_scaled")),
                    r1=parse_float(row, ("majorRadius", "aRadius", "radius_scaled"), 18.0),
                    r2=parse_float(row, ("bRadius", "radius_scaled"), 18.0),
                    r3=parse_float(row, ("minorRadius", "cRadius", "radius_scaled"), 18.0),
                )
            )
    return rows


def load_tiff_stack(path: Path) -> np.ndarray:
    ok, pages = cv2.imreadmulti(str(path), flags=cv2.IMREAD_UNCHANGED)
    if not ok or not pages:
        raise RuntimeError(f"Failed to read TIFF stack: {path}")
    stack = np.stack([page.astype(np.float32) for page in pages], axis=0)
    if stack.max() > stack.min():
        stack = (stack - stack.min()) / (stack.max() - stack.min()) * 255.0
    return stack


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    clipped_q = min(1.0, max(0.0, q))
    values = sorted(values)
    pos = clipped_q * (len(values) - 1)
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    if lo == hi:
        return values[lo]
    t = pos - lo
    return values[lo] * (1.0 - t) + values[hi] * t


def csv_cell(value: str) -> str:
    if any(ch in value for ch in ',\"\n\r'):
        return '"' + value.replace('"', '""') + '"'
    return value


def export_masks(args: argparse.Namespace) -> None:
    stack = load_tiff_stack(args.image)
    z_count, rows, cols = stack.shape
    frame_name = args.frame_name or args.image.name
    cells = load_cells(args.cells, frame_name if args.filter_frame else None)
    if not cells:
        raise RuntimeError(f"No cells loaded from {args.cells}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / f"{args.image.stem}_fine_shape_summary.csv"
    runs_path = args.output_dir / f"{args.image.stem}_fine_shape_runs.csv"

    with summary_path.open("w", newline="") as summary_f, runs_path.open("w", newline="") as runs_f:
        summary = summary_f
        runs = runs_f
        summary.write(
            "frame,cell_name,center_x,center_y,center_z_scaled,radius_scaled,"
            "threshold,seed_value,voxels,mean_intensity,truncated,"
            "bbox_x0,bbox_y0,bbox_z0,bbox_x1,bbox_y1,bbox_z1\n"
        )
        runs.write("frame,cell_name,z,y,x0,x1\n")

        exported = 0
        total_voxels = 0
        for cell in cells:
            base_radius = max(cell.r1, cell.r2, cell.r3, args.min_radius)
            radius = min(args.max_radius, max(args.min_radius, base_radius * args.radius_scale))
            radius_sq = radius * radius

            x0 = max(0, int(np.floor(cell.x - radius)))
            x1 = min(cols - 1, int(np.ceil(cell.x + radius)))
            y0 = max(0, int(np.floor(cell.y - radius)))
            y1 = min(rows - 1, int(np.ceil(cell.y + radius)))
            z0 = max(0, int(np.floor((cell.z_scaled - radius) / args.z_scaling)))
            z1 = min(z_count - 1, int(np.ceil((cell.z_scaled + radius) / args.z_scaling)))
            if x0 > x1 or y0 > y1 or z0 > z1:
                continue

            values: list[float] = []
            for z in range(z0, z1 + 1):
                dz = z * args.z_scaling - cell.z_scaled
                for y in range(y0, y1 + 1):
                    dy = y - cell.y
                    for x in range(x0, x1 + 1):
                        dx = x - cell.x
                        if dx * dx + dy * dy + dz * dz <= radius_sq:
                            values.append(float(stack[z, y, x]))
            if not values:
                continue

            threshold = max(args.threshold_floor, percentile(values, args.threshold_quantile))
            seed = None
            seed_value = -1.0
            seed_radius_sq = args.seed_search_radius * args.seed_search_radius
            for z in range(z0, z1 + 1):
                dz = z * args.z_scaling - cell.z_scaled
                for y in range(y0, y1 + 1):
                    dy = y - cell.y
                    for x in range(x0, x1 + 1):
                        dx = x - cell.x
                        dist_sq = dx * dx + dy * dy + dz * dz
                        if dist_sq > radius_sq or dist_sq > seed_radius_sq:
                            continue
                        value = float(stack[z, y, x])
                        if value > seed_value:
                            seed_value = value
                            seed = (x, y, z)
            if seed is None:
                continue
            if seed_value < threshold:
                threshold = max(args.threshold_floor, seed_value * args.seed_fallback_fraction)

            local_shape = (z1 - z0 + 1, y1 - y0 + 1, x1 - x0 + 1)
            visited = np.zeros(local_shape, dtype=np.bool_)
            mask = np.zeros(local_shape, dtype=np.bool_)
            q: deque[tuple[int, int, int]] = deque([seed])
            visited[seed[2] - z0, seed[1] - y0, seed[0] - x0] = True

            bbox = [cols, rows, z_count, -1, -1, -1]
            voxels = 0
            intensity_sum = 0.0
            truncated = False

            while q:
                x, y, z = q.popleft()
                value = float(stack[z, y, x])
                if value < threshold:
                    continue
                mask[z - z0, y - y0, x - x0] = True
                voxels += 1
                intensity_sum += value
                bbox[0] = min(bbox[0], x)
                bbox[1] = min(bbox[1], y)
                bbox[2] = min(bbox[2], z)
                bbox[3] = max(bbox[3], x)
                bbox[4] = max(bbox[4], y)
                bbox[5] = max(bbox[5], z)
                if args.max_voxels_per_cell > 0 and voxels >= args.max_voxels_per_cell:
                    truncated = True
                    break

                for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if nx < x0 or nx > x1 or ny < y0 or ny > y1 or nz < z0 or nz > z1:
                        continue
                    if visited[nz - z0, ny - y0, nx - x0]:
                        continue
                    ddx = nx - cell.x
                    ddy = ny - cell.y
                    ddz = nz * args.z_scaling - cell.z_scaled
                    if ddx * ddx + ddy * ddy + ddz * ddz > radius_sq:
                        continue
                    visited[nz - z0, ny - y0, nx - x0] = True
                    q.append((nx, ny, nz))

            if voxels <= 0:
                continue

            for z in range(z0, z1 + 1):
                for y in range(y0, y1 + 1):
                    run_start = -1
                    for x in range(x0, x1 + 1):
                        in_mask = bool(mask[z - z0, y - y0, x - x0])
                        if in_mask and run_start < 0:
                            run_start = x
                        if (not in_mask or x == x1) and run_start >= 0:
                            run_end = x if in_mask and x == x1 else x - 1
                            runs.write(
                                f"{csv_cell(args.image.stem)},{csv_cell(cell.name)},{z},{y},{run_start},{run_end}\n"
                            )
                            run_start = -1

            summary.write(
                f"{csv_cell(args.image.stem)},{csv_cell(cell.name)},"
                f"{cell.x:.9g},{cell.y:.9g},{cell.z_scaled:.9g},{radius:.9g},"
                f"{threshold:.9g},{seed_value:.9g},{voxels},{intensity_sum / voxels:.9g},"
                f"{1 if truncated else 0},{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},{bbox[4]},{bbox[5]}\n"
            )
            exported += 1
            total_voxels += voxels

    print(
        f"[FineShape CSV Export] image={args.image} cells={len(cells)} exported={exported} "
        f"total_voxels={total_voxels} summary={summary_path} runs={runs_path}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export center anchored fine shape RLE masks from an existing cell CSV."
    )
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--cells", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--frame-name", help="Frame name to match in the CSV, default is image filename.")
    parser.add_argument("--filter-frame", action="store_true", help="Only use rows matching the frame name.")
    parser.add_argument("--z-scaling", type=float, default=7.0)
    parser.add_argument("--radius-scale", type=float, default=1.25)
    parser.add_argument("--min-radius", type=float, default=8.0)
    parser.add_argument("--max-radius", type=float, default=64.0)
    parser.add_argument("--threshold-quantile", type=float, default=0.55)
    parser.add_argument("--threshold-floor", type=float, default=0.08)
    parser.add_argument("--seed-search-radius", type=float, default=10.0)
    parser.add_argument("--seed-fallback-fraction", type=float, default=0.85)
    parser.add_argument("--max-voxels-per-cell", type=int, default=200000)
    args = parser.parse_args()
    export_masks(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
