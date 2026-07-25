#!/usr/bin/env python3
"""Render CellLumen fine shape RLE masks as a Napari-friendly TIFF stack.

The fine shape exporter intentionally writes compact CSV RLE rows instead of
changing the main PCA ellipsoid path. This script is a review helper: it turns
those rows back into a 3D label or binary TIFF so the new shape model can be
inspected without touching tracking results.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def read_reference_shape(path: Path) -> tuple[int, int, int] | None:
    # Prefer OpenCV when available because the C++ tracker also uses OpenCV for
    # TIFF export. Fall back to tifffile so Napari-only environments can still
    # render full-size review masks instead of silently cropping to the mask
    # bounding box inferred from RLE rows.
    try:
        import cv2
    except Exception:
        cv2 = None

    if cv2 is not None:
        ok, pages = cv2.imreadmulti(str(path), flags=cv2.IMREAD_UNCHANGED)
        if ok and pages:
            first = pages[0]
            if first.ndim >= 2:
                return len(pages), int(first.shape[0]), int(first.shape[1])

    try:
        import tifffile
    except Exception:
        return None

    try:
        arr = tifffile.imread(str(path))
    except Exception:
        return None
    if arr.ndim == 2:
        return 1, int(arr.shape[0]), int(arr.shape[1])
    if arr.ndim >= 3:
        return int(arr.shape[0]), int(arr.shape[-2]), int(arr.shape[-1])
    return None


def infer_shape_from_runs(runs_path: Path) -> tuple[int, int, int]:
    max_z = max_y = max_x = -1
    with runs_path.open(newline="") as f:
        for row in csv.DictReader(f):
            max_z = max(max_z, int(row["z"]))
            max_y = max(max_y, int(row["y"]))
            max_x = max(max_x, int(row["x1"]))
    if max_z < 0 or max_y < 0 or max_x < 0:
        raise ValueError(f"No mask rows found in {runs_path}")
    return max_z + 1, max_y + 1, max_x + 1


def load_label_map(summary_path: Path) -> dict[str, int]:
    mapping: dict[str, int] = {}
    with summary_path.open(newline="") as f:
        for index, row in enumerate(csv.DictReader(f), start=1):
            mapping[row["cell_name"]] = index
    return mapping


def write_tiff(path: Path, volume: np.ndarray, compression: str | None) -> None:
    try:
        import tifffile
    except Exception as exc:
        raise RuntimeError("tifffile is required to write the review TIFF") from exc

    def write_once(target: Path, compression_name: str | None) -> None:
        kwargs = {"photometric": "minisblack"}
        if compression_name:
            kwargs["compression"] = compression_name
        tifffile.imwrite(str(target), volume, **kwargs)

    # Leave this uncompressed by default for maximum reader compatibility.
    # Review runs can opt into zlib/deflate, which is lossless and usually
    # shrinks binary fine shape masks dramatically without changing pixels. Some
    # local imagecodecs installs are ABI-mismatched, so compressed writes are
    # attempted on a temp path and safely fall back to uncompressed output.
    tmp_path = path.with_name(path.name + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    if compression:
        try:
            write_once(tmp_path, compression)
            tmp_path.replace(path)
            return
        except Exception as exc:
            if tmp_path.exists():
                tmp_path.unlink()
            print(
                f"[FineShape Render] compression_failed={compression} "
                f"fallback=uncompressed reason={exc}"
            )

    write_once(tmp_path, None)
    tmp_path.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render CellLumen fine shape RLE CSV files as a TIFF stack."
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--reference-tif",
        type=Path,
        help="Optional real/synth TIFF whose stack size should be matched.",
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Write a smaller uint8 binary mask instead of uint16 labels.",
    )
    parser.add_argument(
        "--compression",
        choices=["none", "zlib", "deflate"],
        default="none",
        help="Optional lossless TIFF compression for the review stack.",
    )
    args = parser.parse_args()

    shape = read_reference_shape(args.reference_tif) if args.reference_tif else None
    if shape is None:
        shape = infer_shape_from_runs(args.runs)

    dtype = np.uint8 if args.binary else np.uint16
    volume = np.zeros(shape, dtype=dtype)
    label_map = load_label_map(args.summary)

    clipped_rows = 0
    painted_rows = 0
    with args.runs.open(newline="") as f:
        for row in csv.DictReader(f):
            z = int(row["z"])
            y = int(row["y"])
            x0 = int(row["x0"])
            x1 = int(row["x1"])
            if z < 0 or z >= shape[0] or y < 0 or y >= shape[1] or x1 < 0 or x0 >= shape[2]:
                clipped_rows += 1
                continue
            x0 = max(0, x0)
            x1 = min(shape[2] - 1, x1)
            if x0 > x1:
                clipped_rows += 1
                continue
            value = 255 if args.binary else label_map.get(row["cell_name"], 0)
            if value == 0:
                clipped_rows += 1
                continue
            volume[z, y, x0 : x1 + 1] = value
            painted_rows += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    compression = None if args.compression == "none" else args.compression
    write_tiff(args.output, volume, compression)
    print(
        f"[FineShape Render] output={args.output} shape={shape} dtype={volume.dtype} "
        f"painted_rows={painted_rows} clipped_rows={clipped_rows} "
        f"nonzero={int(np.count_nonzero(volume))} compression={args.compression}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
