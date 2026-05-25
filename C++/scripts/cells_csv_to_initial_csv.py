#!/usr/bin/env python3
"""Convert a CellUniverse cells.csv frame snapshot into an initial CSV.

The generated file keeps theta columns so CsvHandler's `initial_z_space: auto`
recognizes the z coordinates as already being in optimizer/scaled space.
Current CellFactory does not restore theta from initial CSVs, but the columns
prevent accidental z re-scaling.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REQUIRED_COLUMNS = {
    "file",
    "name",
    "x",
    "y",
    "z",
    "aRadius",
    "bRadius",
    "cRadius",
    "isTrash",
}

OUTPUT_COLUMNS = [
    "file",
    "name",
    "x",
    "y",
    "z",
    "aRadius",
    "bRadius",
    "cRadius",
    "theta_x",
    "theta_y",
    "theta_z",
    "isTrash",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract one frame from CellUniverse cells.csv as an initial CSV."
    )
    parser.add_argument("cells_csv", type=Path, help="Path to an exported cells.csv.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output initial CSV path.",
    )
    parser.add_argument(
        "--frame",
        help="Frame file to extract, e.g. t024.tif. Defaults to the latest frame in file order.",
    )
    parser.add_argument(
        "--output-file",
        help="Override the file column in the generated CSV, useful when remapping to a new run's first frame.",
    )
    parser.add_argument(
        "--drop-theta",
        action="store_true",
        help="Write only file/name/position/radii/isTrash. Use only with simulation.initial_z_space: scaled.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    with args.cells_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"{args.cells_csv} is empty or missing a header")

        missing = REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise SystemExit(f"{args.cells_csv} is missing required columns: {missing_text}")

        rows = list(reader)

    if not rows:
        raise SystemExit(f"{args.cells_csv} contains no data rows")

    frame = args.frame if args.frame else rows[-1]["file"]
    selected = [row for row in rows if row["file"] == frame]
    if not selected:
        raise SystemExit(f"frame {frame!r} was not found in {args.cells_csv}")

    output_columns = [
        "file",
        "name",
        "x",
        "y",
        "z",
        "aRadius",
        "bRadius",
        "cRadius",
        "isTrash",
    ] if args.drop_theta else OUTPUT_COLUMNS

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_columns)
        writer.writeheader()
        for row in selected:
            output_row = {column: row.get(column, "0") for column in output_columns}
            if args.output_file:
                output_row["file"] = args.output_file
            writer.writerow(output_row)

    print(
        f"wrote {len(selected)} cells from {frame} to {args.output}"
        + (f" with file column remapped to {args.output_file}" if args.output_file else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
