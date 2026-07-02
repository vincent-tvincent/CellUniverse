"""Parsers for CellUniverse output artifacts.

All readers are pure (filesystem in, dataclasses out) so they can be tested
against recorded run fixtures without a GUI.
"""

from __future__ import annotations

import csv
import re
import shlex
from pathlib import Path

from .model import Cell, FrameData, LumenCenter, SplitEvent

# ``t085.tif`` -> 85 ; also accepts a bare ``85``.
_FILE_FRAME_RE = re.compile(r"t?(\d+)")
_LUMEN_ID_RE = re.compile(r"lumen_candidate_id=(\d+)")


def _to_float(s: str, default: float = 0.0) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


def frame_of_file(file_field: str) -> int | None:
    m = _FILE_FRAME_RE.search(file_field)
    return int(m.group(1)) if m else None


def read_cells_csv(path: Path, frame: int) -> list[Cell]:
    """Read cells.csv rows belonging to ``frame``.

    cells.csv accumulates every frame; we filter by the ``file`` column.
    """
    cells: list[Cell] = []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if frame_of_file(row.get("file", "")) != frame:
                continue
            cells.append(
                Cell(
                    name=row["name"],
                    x=_to_float(row["x"]),
                    y=_to_float(row["y"]),
                    z=_to_float(row["z"]),
                    a_radius=_to_float(row["aRadius"]),
                    b_radius=_to_float(row["bRadius"]),
                    c_radius=_to_float(row["cRadius"]),
                    theta_x=_to_float(row["theta_x"]),
                    theta_y=_to_float(row["theta_y"]),
                    theta_z=_to_float(row["theta_z"]),
                    is_trash=str(row.get("isTrash", "0")).strip() in ("1", "true", "True"),
                )
            )
    return cells


def read_candidate_graph(path: Path) -> tuple[list[LumenCenter], list[SplitEvent], list[str]]:
    """Parse a candidate_graph/frame_NN_candidates.csv.

    Returns ``(lumen_centers, accepted_splits, fit_order)`` where ``fit_order``
    is the sequence of cell names in file row order (continuations + split
    parents) — the v1 fit-order proxy.
    """
    lumen: list[LumenCenter] = []
    splits: list[SplitEvent] = []
    fit_order: list[str] = []
    if not path.exists():
        return lumen, splits, fit_order
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            kind = row.get("kind", "")
            note = row.get("note", "")
            if kind == "lumen_center":
                cid = _LUMEN_ID_RE.search(note)
                lumen.append(
                    LumenCenter(
                        x=_to_float(row["x1"]),
                        y=_to_float(row["y1"]),
                        z=_to_float(row["z1"]),
                        voxels=_to_float(row.get("vox_a", "0")),
                        signal=_to_float(row.get("signal_a", "0")),
                        candidate_id=int(cid.group(1)) if cid else None,
                    )
                )
            elif kind == "split_pair" and str(row.get("selected", "0")).strip() == "1":
                splits.append(
                    SplitEvent(
                        parent=row.get("parent", ""),
                        d1=(_to_float(row["x1"]), _to_float(row["y1"]), _to_float(row["z1"])),
                        d2=(_to_float(row["x2"]), _to_float(row["y2"]), _to_float(row["z2"])),
                        score=_to_float(row.get("score", "0")),
                        source=row.get("source", ""),
                        note=note,
                    )
                )
            elif kind == "continuation":
                name = row.get("parent", "")
                if name:
                    fit_order.append(name)
    # Split parents enter the fit sequence too, after their continuation slot.
    for s in splits:
        if s.parent and s.parent not in fit_order:
            fit_order.append(s.parent)
    return lumen, splits, fit_order


def read_checkpoint_zinfo(path: Path) -> tuple[int | None, int | None]:
    """Return ``(z_slices, maxZ)`` from a checkpoint header, or (None, None)."""
    z_slices = max_z = None
    if not path.exists():
        return None, None
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "z_slices":
                z_slices = int(float(parts[1]))
            elif len(parts) >= 2 and parts[0] == "maxZ":
                max_z = int(float(parts[1]))
            elif parts and parts[0] == "cell":
                break
    return z_slices, max_z


def read_checkpoint_brightness(path: Path) -> dict[str, float]:
    """Map cell name -> brightness from a checkpoint's quoted ``cell`` lines.

    Line format: ``cell "Cell type 1_310" x y z aR bR cR tx ty tz brightness isTrash``.
    cells.csv has no brightness column, so this is the source for synth shading.
    """
    out: dict[str, float] = {}
    if not path.exists():
        return out
    with open(path) as fh:
        for line in fh:
            if not line.startswith("cell "):
                continue
            try:
                toks = shlex.split(line)
            except ValueError:
                continue
            # toks: ['cell', name, x,y,z, aR,bR,cR, tx,ty,tz, brightness, isTrash]
            if len(toks) >= 13:
                out[toks[1]] = _to_float(toks[11], 0.98)
    return out


def load_frame(run_dir: Path, frame: int) -> FrameData:
    """Assemble a FrameData from a run's on-disk artifacts."""
    cells = read_cells_csv(run_dir / "cells.csv", frame)
    bright = read_checkpoint_brightness(run_dir / "checkpoints" / f"frame_{frame:03d}.txt")
    if bright:
        from dataclasses import replace
        cells = [replace(c, brightness=bright.get(c.name, c.brightness)) for c in cells]
    cg = run_dir / "candidate_graph" / f"frame_{frame}_candidates.csv"
    lumen, splits, fit_order = read_candidate_graph(cg)
    return FrameData(
        frame=frame,
        cells=cells,
        lumen_centers=lumen,
        splits=splits,
        fit_order=fit_order,
    )
