#!/usr/bin/env python3
"""Recover CellUniverse compact-export frames into typed TIFF layers.

By default, every selected frame materializes exactly two image volumes, each
in its own content folder:

* ``background/frame_NNNNNN.background.tif``: background only
* ``cells/frame_NNNNNN.cells.tif``: cells only, with zero outside cells

It also writes ``recovery/frame_NNNNNN.recovery.json`` as a
completion/provenance record.  Normal Web UI previews should consume compact
data in memory instead of invoking this recovery tool.

``--all-frames --center-id-sidecar`` additionally writes one sparse
``centers/cell_centers_ids.csv`` table for a time-aware cell-center/ID display
layer. ``--center-id-stacks`` independently writes a forced-Deflate uint8 TIFF
per frame under ``center_id_stacks/``. Each optional raster layer contains a
small sphere or cube at every current cell center and its decimal string ID in
a deterministic bitmap font beside the marker.

Compact-export v1 rotations are radians.  Centers and radii are expressed in
the interpolated optimizer XYZ voxel coordinate system described by
``coordinates``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import stat
import struct
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

try:
    import numpy as np
    import tifffile
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by CLI hosts
    raise SystemExit(
        "recover_compact_export.py requires NumPy and tifffile "
        f"(missing {exc.name!r})"
    ) from exc


MANIFEST_SCHEMA = "celluniverse.compact.manifest"
FRAME_SCHEMA = "celluniverse.compact.frame"
RECOVERY_SCHEMA = "celluniverse.compact.recovery"
BATCH_RECOVERY_SCHEMA = "celluniverse.compact.recovery-batch"
CENTER_ID_RECOVERY_SCHEMA = "celluniverse.compact.center-id-recovery"
FORMAT_VERSION = 1

CUBM_HEADER = struct.Struct("<4sHHIIIQ")
CUBM_MAGIC = b"CUBM"
CUBM_VERSION = 1
CUBM_FLAGS_LSB_FIRST_ZYX = 1
FRAME_NAME_PATTERN = re.compile(r"frame_(\d{6,})\.json")
RECOVERED_NAME_PATTERN = re.compile(
    r"frame_(\d{6,})\.(?:background\.tif|cells\.tif|recovery\.json)"
)
CENTER_ID_NAME_PATTERN = re.compile(
    r"frame_(\d{6,})\.center_ids\.(?:tif|recovery\.json)"
)
MAX_CPP_FRAME = 2_147_483_647
PIPELINE_MODES = {
    "traditional",
    "celluniverse2",
    "celluniverse3",
    "celllumen",
    "cell_lumen_fusion",
}
Z_INTERPOLATION_SOURCES = {"config", "initial_csv", "cell_lumen_profile"}
RENDER_CONTRACT = {
    "id": "ellipsoid_rz_ry_rx_overwrite_cv8u_v1",
    "rotation": "Rz*Ry*Rx",
    "membership": "squared_local_radius<=1",
    "overlap": "later_draw_order_overwrites",
    "output": "opencv_float32_to_uint8_scale_255",
}

DEFAULT_MAX_MATERIALIZED_BYTES = 10_000_000_000
TIFF_OVERHEAD_RESERVE = 16 * 1024 * 1024
JSON_OVERHEAD_RESERVE = 1024 * 1024
NEXT_PAGE_RESERVE = 2 * 1024 * 1024
BATCH_TIFF_OVERHEAD_RESERVE = 2 * 1024 * 1024
BATCH_SIDECAR_RESERVE = 16 * 1024 * 1024
CENTER_ID_COMPRESSED_RESERVE = 8 * 1024 * 1024
CENTER_ID_CSV_NAME = "cell_centers_ids.csv"
BATCH_RECOVERY_NAME = "recovery_batch.json"
BACKGROUND_DIR_NAME = "background"
CELLS_DIR_NAME = "cells"
CENTERS_DIR_NAME = "centers"
RECOVERY_DIR_NAME = "recovery"
TOOLS_DIR_NAME = "tools"
CENTER_ID_STACK_DIR_NAME = "center_id_stacks"
OUTPUT_LAYOUT = {
    "version": 1,
    "background": BACKGROUND_DIR_NAME,
    "cells": CELLS_DIR_NAME,
    "centers": CENTERS_DIR_NAME,
    "recovery": RECOVERY_DIR_NAME,
    "tools": TOOLS_DIR_NAME,
}
DATA_DIR_NAMES = (
    BACKGROUND_DIR_NAME,
    CELLS_DIR_NAME,
    CENTERS_DIR_NAME,
    RECOVERY_DIR_NAME,
    CENTER_ID_STACK_DIR_NAME,
)
CENTER_ID_STACK_COMPRESSION = "deflate"
CENTER_ID_FONT_SCALE = 1
CENTER_ID_TEXT_Z_HALF_THICKNESS = 1
CENTER_ID_TEXT_GAP = 1
CENTER_ID_RENDER_CONTRACT = {
    "id": "sphere_or_cube_with_5x7_decimal_id_v1",
    "dtype": "uint8",
    "background_value": 0,
    "marker_value": 255,
    "text_value": 192,
    "center_rounding": "floor(float32(value)+0.5)",
    "glyphs": "builtin_decimal_5x7",
    "glyph_scale": CENTER_ID_FONT_SCALE,
    "text_z_half_thickness": CENTER_ID_TEXT_Z_HALF_THICKNESS,
    "text_gap_voxels": CENTER_ID_TEXT_GAP,
    "text_placement": (
        "right_left_down_up_then_diagonals_clamped_without_bbox_overlap"
    ),
    "compression": CENTER_ID_STACK_COMPRESSION,
}
DIGIT_GLYPHS_5X7 = {
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
}

# This is reserved per committed TIFF for the whole-batch estimate.  The
# larger TIFF_OVERHEAD_RESERVE remains the atomic, current-frame runtime guard
# inside recover().  Aggregating that temporary reserve for every frame would
# incorrectly assume that all frame temporary files coexist.


class RecoveryError(RuntimeError):
    """Raised for invalid compact data or unsafe recovery requests."""


@dataclass(frozen=True)
class CellSpec:
    draw_order: int
    name: str
    center_xyz: tuple[float, float, float]
    radii_abc: tuple[float, float, float]
    rotation_xyz: tuple[float, float, float]
    brightness: float
    is_trash: bool


@dataclass(frozen=True)
class FrameSpec:
    selected_frame: int
    source_frame: str
    pipeline_mode: str
    shape_zyx: tuple[int, int, int]
    coordinates: Mapping[str, Any]
    background: Mapping[str, Any]
    cells: tuple[CellSpec, ...]
    compact_root: Path
    manifest_path: Path
    frame_path: Path


@dataclass(frozen=True)
class CenterIdAnnotation:
    cell_id: str
    center_zyx: tuple[int, int, int]
    text_placement: str
    text_blocks: tuple[tuple[int, int, int, int], ...]
    text_bbox_zyx: tuple[int, int, int, int, int, int]


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        size = path.stat().st_size
    except FileNotFoundError as exc:
        raise RecoveryError(f"{label} does not exist: {path}") from exc
    if size > 64 * 1024 * 1024:
        raise RecoveryError(f"{label} is unexpectedly large ({size} bytes): {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RecoveryError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RecoveryError(f"{label} must contain a JSON object: {path}")
    return value


def _check_schema(document: Mapping[str, Any], schema: str, label: str) -> None:
    if document.get("schema") != schema:
        raise RecoveryError(
            f"{label} schema must be {schema!r}, got {document.get('schema')!r}"
        )
    version = document.get("version")
    if version != FORMAT_VERSION or isinstance(version, bool):
        raise RecoveryError(
            f"{label} version must be integer {FORMAT_VERSION}, got {version!r}"
        )


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RecoveryError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise RecoveryError(f"{label} must be finite")
    return result


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RecoveryError(f"{label} must be an integer")
    if value < minimum:
        raise RecoveryError(f"{label} must be >= {minimum}")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise RecoveryError(f"{label} must be an object")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RecoveryError(f"{label} must be a non-empty string")
    return value


def _xyz(value: Any, label: str) -> tuple[float, float, float]:
    obj = _mapping(value, label)
    return (
        _finite_number(obj.get("x"), f"{label}.x"),
        _finite_number(obj.get("y"), f"{label}.y"),
        _finite_number(obj.get("z"), f"{label}.z"),
    )


def _radii(value: Any, label: str) -> tuple[float, float, float]:
    obj = _mapping(value, label)
    result = (
        _finite_number(obj.get("a"), f"{label}.a"),
        _finite_number(obj.get("b"), f"{label}.b"),
        _finite_number(obj.get("c"), f"{label}.c"),
    )
    if any(radius <= 0.0 for radius in result):
        raise RecoveryError(f"{label} radii must all be > 0")
    return result


def _rotation(value: Any, label: str) -> tuple[float, float, float]:
    obj = _mapping(value, label)
    return (
        _finite_number(obj.get("theta_x"), f"{label}.theta_x"),
        _finite_number(obj.get("theta_y"), f"{label}.theta_y"),
        _finite_number(obj.get("theta_z"), f"{label}.theta_z"),
    )


def _validate_offsets(background: Mapping[str, Any], label: str) -> None:
    if "offset_updates" in background:
        updates = background["offset_updates"]
        if not isinstance(updates, list):
            raise RecoveryError(f"{label}.offset_updates must be an array")
        for index, update in enumerate(updates):
            _finite_number(update, f"{label}.offset_updates[{index}]")
    elif "additive_offset" in background:
        _finite_number(background["additive_offset"], f"{label}.additive_offset")


def _validate_background(
    background_value: Any, compact_root: Path
) -> Mapping[str, Any]:
    background = _mapping(background_value, "frame.background")
    kind = _nonempty_string(background.get("kind"), "frame.background.kind")
    _validate_offsets(background, "frame.background")

    if kind == "scalar":
        _finite_number(background.get("value"), "frame.background.value")
    elif kind == "rotated_soft_ellipsoid":
        _xyz(background.get("center"), "frame.background.center")
        _radii(background.get("radii"), "frame.background.radii")
        _rotation(background.get("rotation"), "frame.background.rotation")
        _finite_number(background.get("cold"), "frame.background.cold")
        _finite_number(background.get("hot"), "frame.background.hot")
        soft_margin = _finite_number(
            background.get("soft_margin"), "frame.background.soft_margin"
        )
        if soft_margin < 0.0:
            raise RecoveryError("frame.background.soft_margin must be >= 0")
    elif kind == "binary_mask":
        _finite_number(background.get("cold"), "frame.background.cold")
        _finite_number(background.get("hot"), "frame.background.hot")
        mask_format = _nonempty_string(
            background.get("mask_format"), "frame.background.mask_format"
        )
        if mask_format != "CUBM1":
            raise RecoveryError("frame.background.mask_format must be 'CUBM1'")
        mask_path_value = _nonempty_string(
            background.get("mask_path"), "frame.background.mask_path"
        )
        mask_relative = Path(mask_path_value)
        if mask_relative.is_absolute():
            raise RecoveryError("frame.background.mask_path must be package-relative")
        mask_path = (compact_root / mask_relative).resolve()
        try:
            mask_path.relative_to(compact_root)
        except ValueError as exc:
            raise RecoveryError(
                "frame.background.mask_path escapes the compact package"
            ) from exc
        if not mask_path.is_file():
            raise RecoveryError(f"CUBM mask does not exist: {mask_path}")
    else:
        raise RecoveryError(f"unsupported frame.background.kind: {kind!r}")
    return background


def _load_contract(compact: Path, selected_frame: int) -> FrameSpec:
    compact = compact.expanduser().resolve()
    if compact.is_dir():
        compact_root = compact
        manifest_path = compact / "manifest.json"
    else:
        manifest_path = compact
        compact_root = compact.parent
    compact_root = compact_root.resolve()

    manifest = _read_json(manifest_path, "compact manifest")
    _check_schema(manifest, MANIFEST_SCHEMA, "compact manifest")
    if manifest.get("frame_schema") != FRAME_SCHEMA:
        raise RecoveryError(
            f"compact manifest frame_schema must be {FRAME_SCHEMA!r}"
        )
    if manifest.get("mask_schema") != "CUBM1":
        raise RecoveryError("compact manifest mask_schema must be 'CUBM1'")

    if selected_frame < 0 or selected_frame > MAX_CPP_FRAME:
        raise RecoveryError(
            f"--frame must be between 0 and {MAX_CPP_FRAME}"
        )
    frame_entries = manifest.get("frames")
    if not isinstance(frame_entries, list):
        raise RecoveryError("compact manifest frames must be an array")
    selected_relative: str | None = None
    seen_frames: set[int] = set()
    for index, raw_entry in enumerate(frame_entries):
        label = f"compact manifest frames[{index}]"
        entry = _mapping(raw_entry, label)
        entry_frame = _integer(entry.get("frame"), f"{label}.frame")
        if entry_frame in seen_frames:
            raise RecoveryError(f"compact manifest repeats frame {entry_frame}")
        seen_frames.add(entry_frame)
        entry_path = _nonempty_string(entry.get("path"), f"{label}.path")
        path_name = Path(entry_path).name
        match = FRAME_NAME_PATTERN.fullmatch(path_name)
        if match is None or int(match.group(1)) != entry_frame:
            raise RecoveryError(
                f"compact manifest frame/path mismatch for frame {entry_frame}"
            )
        if entry_frame == selected_frame:
            selected_relative = entry_path
    if selected_relative is None:
        raise RecoveryError(
            f"frame {selected_frame} is not listed in compact manifest"
        )
    relative_path = Path(selected_relative)
    if relative_path.is_absolute():
        raise RecoveryError("compact manifest frame path must be package-relative")
    frame_path = (compact_root / relative_path).resolve()
    try:
        frame_path.relative_to(compact_root)
    except ValueError as exc:
        raise RecoveryError(
            "compact manifest frame path escapes the compact package"
        ) from exc
    frame = _read_json(frame_path, "compact frame")
    _check_schema(frame, FRAME_SCHEMA, "compact frame")

    record_frame = _integer(frame.get("frame"), "frame.frame")
    if record_frame != selected_frame:
        raise RecoveryError(
            f"compact frame number mismatch: expected {selected_frame}, "
            f"found {record_frame}"
        )
    source_frame = _nonempty_string(
        frame.get("source_frame"), "frame.source_frame"
    )
    if Path(source_frame).name != source_frame or source_frame in {".", ".."}:
        raise RecoveryError("frame.source_frame must be a filename")
    pipeline_mode = _nonempty_string(
        frame.get("pipeline_mode"), "frame.pipeline_mode"
    )
    if pipeline_mode not in PIPELINE_MODES:
        raise RecoveryError(
            f"unsupported frame.pipeline_mode: {pipeline_mode!r}"
        )

    dimensions = _mapping(frame.get("dimensions"), "frame.dimensions")
    x_size = _integer(dimensions.get("x"), "frame.dimensions.x", minimum=1)
    y_size = _integer(dimensions.get("y"), "frame.dimensions.y", minimum=1)
    z_size = _integer(dimensions.get("z"), "frame.dimensions.z", minimum=1)
    if any(size > 1_000_000 for size in (x_size, y_size, z_size)):
        raise RecoveryError("compact frame dimension is unreasonably large")

    coordinates = _mapping(frame.get("coordinates"), "frame.coordinates")
    if (
        coordinates.get("cell_order") != "xyz"
        or coordinates.get("volume_order") != "zyx"
    ):
        raise RecoveryError("unsupported compact coordinate ordering")
    if coordinates.get("space") != "interpolated":
        raise RecoveryError("frame.coordinates.space must be 'interpolated'")
    _integer(
        coordinates.get("z_interpolation_ratio"),
        "frame.coordinates.z_interpolation_ratio",
        minimum=1,
    )
    _nonempty_string(
        coordinates.get("initial_z_space"), "frame.coordinates.initial_z_space"
    )
    ratio_source = _nonempty_string(
        coordinates.get("z_interpolation_source"),
        "frame.coordinates.z_interpolation_source",
    )
    if ratio_source not in Z_INTERPOLATION_SOURCES:
        raise RecoveryError(
            f"unsupported z interpolation source: {ratio_source!r}"
        )

    render_contract = _mapping(
        frame.get("render_contract"), "frame.render_contract"
    )
    for key, expected in RENDER_CONTRACT.items():
        if render_contract.get(key) != expected:
            raise RecoveryError(
                f"unsupported frame.render_contract.{key}: "
                f"{render_contract.get(key)!r}"
            )

    background = _validate_background(frame.get("background"), compact_root)

    cells_value = frame.get("cells")
    if not isinstance(cells_value, list):
        raise RecoveryError("frame.cells must be an ordered array")
    cells: list[CellSpec] = []
    previous_draw_order: int | None = None
    for index, raw_cell in enumerate(cells_value):
        label = f"frame.cells[{index}]"
        cell = _mapping(raw_cell, label)
        draw_order = _integer(cell.get("draw_order"), f"{label}.draw_order")
        if previous_draw_order is not None and draw_order <= previous_draw_order:
            raise RecoveryError(
                "frame.cells must be strictly increasing by draw_order"
            )
        previous_draw_order = draw_order
        is_trash = cell.get("is_trash")
        if not isinstance(is_trash, bool):
            raise RecoveryError(f"{label}.is_trash must be boolean")
        cells.append(
            CellSpec(
                draw_order=draw_order,
                name=_nonempty_string(cell.get("name"), f"{label}.name"),
                center_xyz=_xyz(cell.get("center"), f"{label}.center"),
                radii_abc=_radii(cell.get("radii"), f"{label}.radii"),
                rotation_xyz=_rotation(
                    cell.get("rotation"), f"{label}.rotation"
                ),
                brightness=_finite_number(
                    cell.get("brightness"), f"{label}.brightness"
                ),
                is_trash=is_trash,
            )
        )

    return FrameSpec(
        selected_frame=selected_frame,
        source_frame=source_frame,
        pipeline_mode=pipeline_mode,
        shape_zyx=(z_size, y_size, x_size),
        coordinates=coordinates,
        background=background,
        cells=tuple(cells),
        compact_root=compact_root,
        manifest_path=manifest_path,
        frame_path=frame_path,
    )


def _manifest_frame_ids(compact: Path) -> tuple[int, ...]:
    compact = compact.expanduser().resolve()
    if compact.is_dir():
        compact_root = compact
        manifest_path = compact / "manifest.json"
    else:
        manifest_path = compact
        compact_root = compact.parent
    compact_root = compact_root.resolve()

    manifest = _read_json(manifest_path, "compact manifest")
    _check_schema(manifest, MANIFEST_SCHEMA, "compact manifest")
    if manifest.get("frame_schema") != FRAME_SCHEMA:
        raise RecoveryError(
            f"compact manifest frame_schema must be {FRAME_SCHEMA!r}"
        )
    if manifest.get("mask_schema") != "CUBM1":
        raise RecoveryError("compact manifest mask_schema must be 'CUBM1'")

    frame_entries = manifest.get("frames")
    if not isinstance(frame_entries, list):
        raise RecoveryError("compact manifest frames must be an array")
    frame_ids: list[int] = []
    seen_frames: set[int] = set()
    for index, raw_entry in enumerate(frame_entries):
        label = f"compact manifest frames[{index}]"
        entry = _mapping(raw_entry, label)
        frame = _integer(entry.get("frame"), f"{label}.frame")
        if frame in seen_frames:
            raise RecoveryError(f"compact manifest repeats frame {frame}")
        seen_frames.add(frame)
        entry_path = _nonempty_string(entry.get("path"), f"{label}.path")
        relative_path = Path(entry_path)
        if relative_path.is_absolute():
            raise RecoveryError(
                "compact manifest frame path must be package-relative"
            )
        resolved_path = (compact_root / relative_path).resolve()
        try:
            resolved_path.relative_to(compact_root)
        except ValueError as exc:
            raise RecoveryError(
                "compact manifest frame path escapes the compact package"
            ) from exc
        match = FRAME_NAME_PATTERN.fullmatch(relative_path.name)
        if match is None or int(match.group(1)) != frame:
            raise RecoveryError(
                f"compact manifest frame/path mismatch for frame {frame}"
            )
        frame_ids.append(frame)
    if not frame_ids:
        raise RecoveryError("compact manifest contains no frames")
    return tuple(sorted(frame_ids))


def _rotation_matrix(rotation_xyz: Sequence[float]) -> np.ndarray:
    """Return the forward local-to-world rotation Rz * Ry * Rx."""

    # Compact values round-trip C++ float state.  Cast back to float32 before
    # the C++-equivalent double-precision trigonometry.
    rx, ry, rz = (float(np.float32(value)) for value in rotation_xyz)
    sx, cx = math.sin(rx), math.cos(rx)
    sy, cy = math.sin(ry), math.cos(ry)
    sz, cz = math.sin(rz), math.cos(rz)
    rx_matrix = np.array(
        ((1.0, 0.0, 0.0), (0.0, cx, -sx), (0.0, sx, cx)), dtype=np.float64
    )
    ry_matrix = np.array(
        ((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy)), dtype=np.float64
    )
    rz_matrix = np.array(
        ((cz, -sz, 0.0), (sz, cz, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    return rz_matrix @ ry_matrix @ rx_matrix


def _ellipsoid_radial_slice(
    *,
    z_index: int,
    y_coordinates: np.ndarray,
    x_coordinates: np.ndarray,
    center_xyz: Sequence[float],
    radii_abc: Sequence[float],
    rotation_xyz: Sequence[float],
) -> np.ndarray:
    center_x, center_y, center_z = (
        float(np.float32(value)) for value in center_xyz
    )
    radius_a, radius_b, radius_c = (
        float(np.float32(value)) for value in radii_abc
    )
    rotation = _rotation_matrix(rotation_xyz)
    # BackgroundRegionTracker subtracts cv::Point3f values before promoting
    # the displacement to double for rotation.
    dx = (
        np.asarray(x_coordinates, dtype=np.float32) - np.float32(center_x)
    ).astype(np.float64)
    dy = (
        np.asarray(y_coordinates, dtype=np.float32) - np.float32(center_y)
    ).astype(np.float64)
    dz = float(np.float32(z_index) - np.float32(center_z))
    # Inverse rotation is R^T.  Array rows are Y and columns are X.
    local_x = rotation[0, 0] * dx + rotation[1, 0] * dy + rotation[2, 0] * dz
    local_y = rotation[0, 1] * dx + rotation[1, 1] * dy + rotation[2, 1] * dz
    local_z = rotation[0, 2] * dx + rotation[1, 2] * dy + rotation[2, 2] * dz
    return np.sqrt(
        (local_x / radius_a) ** 2
        + (local_y / radius_b) ** 2
        + (local_z / radius_c) ** 2
    )


def _apply_background_offsets(
    values: np.ndarray, background: Mapping[str, Any]
) -> np.ndarray:
    """Apply normalized offsets in order, clamping to [0, 1] after each."""

    result = np.asarray(values, dtype=np.float32)
    if "offset_updates" in background:
        updates: Iterable[Any] = background["offset_updates"]
    elif "additive_offset" in background:
        updates = (background["additive_offset"],)
    else:
        updates = ()
    for update in updates:
        result = np.asarray(
            result + np.float32(update),
            dtype=np.float32,
        )
        result = np.minimum(
            np.maximum(result, np.float32(0.0)), np.float32(1.0)
        )
    return result


def _opencv_u8(values: np.ndarray) -> np.ndarray:
    """Match saturating OpenCV ``convertTo(CV_8U, 255.0)`` semantics."""

    scaled = np.rint(np.clip(values, 0.0, 1.0) * 255.0)
    return scaled.astype(np.uint8)


class CubmMask:
    """Read CUBM v1 mask slices without materializing the full mask."""

    def __init__(self, path: Path, expected_shape_zyx: Sequence[int]) -> None:
        self.path = path
        self.expected_shape_zyx = tuple(int(value) for value in expected_shape_zyx)
        self._file: Any | None = None
        self.payload_offset = CUBM_HEADER.size

    def __enter__(self) -> "CubmMask":
        try:
            handle = self.path.open("rb")
            header = handle.read(CUBM_HEADER.size)
        except OSError as exc:
            raise RecoveryError(f"cannot open CUBM mask {self.path}: {exc}") from exc
        if len(header) != CUBM_HEADER.size:
            handle.close()
            raise RecoveryError(f"CUBM header is truncated: {self.path}")
        magic, version, flags, width, height, depth, voxel_count = (
            CUBM_HEADER.unpack(header)
        )
        expected_z, expected_y, expected_x = self.expected_shape_zyx
        expected_count = expected_z * expected_y * expected_x
        if magic != CUBM_MAGIC:
            handle.close()
            raise RecoveryError(f"CUBM magic is invalid: {self.path}")
        if version != CUBM_VERSION:
            handle.close()
            raise RecoveryError(
                f"CUBM version must be {CUBM_VERSION}, got {version}"
            )
        if flags != CUBM_FLAGS_LSB_FIRST_ZYX:
            handle.close()
            raise RecoveryError(
                "CUBM flags must declare LSB-first ZYX payload (flags=1)"
            )
        if (depth, height, width) != self.expected_shape_zyx:
            handle.close()
            raise RecoveryError(
                "CUBM dimensions do not match frame dimensions: "
                f"{(depth, height, width)} != {self.expected_shape_zyx}"
            )
        if voxel_count != expected_count:
            handle.close()
            raise RecoveryError(
                f"CUBM voxelCount {voxel_count} != expected {expected_count}"
            )
        expected_bytes = CUBM_HEADER.size + ((expected_count + 7) // 8)
        actual_bytes = self.path.stat().st_size
        if actual_bytes != expected_bytes:
            handle.close()
            raise RecoveryError(
                f"CUBM file size {actual_bytes} != expected {expected_bytes}"
            )
        self._file = handle
        return self

    def __exit__(self, *_: Any) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def slice(self, z_index: int) -> np.ndarray:
        if self._file is None:
            raise RecoveryError("CUBM reader is not open")
        z_size, y_size, x_size = self.expected_shape_zyx
        if z_index < 0 or z_index >= z_size:
            raise RecoveryError(f"CUBM z index is out of range: {z_index}")
        slice_voxels = y_size * x_size
        first_bit = z_index * slice_voxels
        first_byte = first_bit // 8
        bit_offset = first_bit % 8
        byte_count = (bit_offset + slice_voxels + 7) // 8
        self._file.seek(self.payload_offset + first_byte)
        payload = self._file.read(byte_count)
        if len(payload) != byte_count:
            raise RecoveryError(f"CUBM payload is truncated at z={z_index}")
        raw = np.frombuffer(payload, dtype=np.uint8)
        bit_indices = np.arange(bit_offset, bit_offset + slice_voxels)
        bits = (raw[bit_indices >> 3] >> (bit_indices & 7)) & 1
        return bits.astype(bool, copy=False).reshape((y_size, x_size))


def _mask_path(spec: FrameSpec) -> Path:
    path = (spec.compact_root / str(spec.background["mask_path"])).resolve()
    try:
        path.relative_to(spec.compact_root)
    except ValueError as exc:  # defensive; validation already checks this
        raise RecoveryError("CUBM mask path escapes compact root") from exc
    return path


def _background_slices(spec: FrameSpec) -> Iterator[np.ndarray]:
    z_size, y_size, x_size = spec.shape_zyx
    background = spec.background
    kind = str(background["kind"])

    if kind == "binary_mask":
        cold = np.float32(background["cold"])
        hot = np.float32(background["hot"])
        with CubmMask(_mask_path(spec), spec.shape_zyx) as mask:
            for z_index in range(z_size):
                values = np.where(mask.slice(z_index), hot, cold).astype(
                    np.float32, copy=False
                )
                yield _opencv_u8(_apply_background_offsets(values, background))
        return

    if kind == "scalar":
        for _ in range(z_size):
            values = np.full(
                (y_size, x_size),
                np.float32(background["value"]),
                dtype=np.float32,
            )
            yield _opencv_u8(_apply_background_offsets(values, background))
        return

    center = _xyz(background["center"], "frame.background.center")
    radii = _radii(background["radii"], "frame.background.radii")
    rotation = _rotation(background["rotation"], "frame.background.rotation")
    cold = np.float32(background["cold"])
    hot = np.float32(background["hot"])
    soft_margin = float(np.float32(background["soft_margin"]))
    y_grid, x_grid = np.ogrid[:y_size, :x_size]
    for z_index in range(z_size):
        radial = _ellipsoid_radial_slice(
            z_index=z_index,
            y_coordinates=y_grid,
            x_coordinates=x_grid,
            center_xyz=center,
            radii_abc=radii,
            rotation_xyz=rotation,
        )
        if soft_margin <= 1.0e-8:
            weight = (radial <= 1.0).astype(np.float32)
        else:
            u = np.clip(
                (1.0 + soft_margin - radial) / (2.0 * soft_margin), 0.0, 1.0
            )
            weight = (u * u * (3.0 - 2.0 * u)).astype(np.float32)
        hot_minus_cold = np.float32(hot - cold)
        values = np.asarray(cold + hot_minus_cold * weight, dtype=np.float32)
        yield _opencv_u8(_apply_background_offsets(values, background))


def _cells_slices(spec: FrameSpec) -> Iterator[np.ndarray]:
    z_size, y_size, x_size = spec.shape_zyx
    prepared: list[
        tuple[
            CellSpec,
            np.ndarray,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            int,
            int,
            int,
            int,
        ]
    ] = []
    for cell in spec.cells:
        center_x, center_y, center_z = (
            float(np.float32(value)) for value in cell.center_xyz
        )
        radius_a, radius_b, radius_c = (
            float(np.float32(value)) for value in cell.radii_abc
        )
        max_radius = float(
            max(np.float32(radius_a), np.float32(radius_b), np.float32(radius_c))
        )
        x_low = max(
            0,
            math.floor(float(np.float32(center_x) - np.float32(max_radius))),
        )
        x_high = min(
            x_size - 1,
            math.ceil(float(np.float32(center_x) + np.float32(max_radius))),
        )
        y_low = max(
            0,
            math.floor(float(np.float32(center_y) - np.float32(max_radius))),
        )
        y_high = min(
            y_size - 1,
            math.ceil(float(np.float32(center_y) + np.float32(max_radius))),
        )
        prepared.append(
            (
                cell,
                _rotation_matrix(cell.rotation_xyz).T,
                center_x,
                center_y,
                center_z,
                max_radius,
                1.0 / (radius_a * radius_a),
                1.0 / (radius_b * radius_b),
                1.0 / (radius_c * radius_c),
                x_low,
                x_high,
                y_low,
                y_high,
            )
        )

    for z_index in range(z_size):
        values = np.zeros((y_size, x_size), dtype=np.float32)
        for (
            cell,
            inverse_rotation,
            center_x,
            center_y,
            center_z,
            max_radius,
            inv_a2,
            inv_b2,
            inv_c2,
            x_low,
            x_high,
            y_low,
            y_high,
        ) in prepared:
            float_z = np.float32(z_index)
            if abs(float(np.float32(float_z - np.float32(center_z)))) > max_radius:
                continue
            dz = float(float_z) - center_z
            base_dx = float(x_low) - center_x
            step_lx = float(inverse_rotation[0, 0])
            step_ly = float(inverse_rotation[1, 0])
            step_lz = float(inverse_rotation[2, 0])
            row_width = x_high - x_low + 1
            for y_index in range(y_low, y_high + 1):
                dy = float(y_index) - center_y
                start_lx = (
                    inverse_rotation[0, 0] * base_dx
                    + inverse_rotation[0, 1] * dy
                    + inverse_rotation[0, 2] * dz
                )
                start_ly = (
                    inverse_rotation[1, 0] * base_dx
                    + inverse_rotation[1, 1] * dy
                    + inverse_rotation[1, 2] * dz
                )
                start_lz = (
                    inverse_rotation[2, 0] * base_dx
                    + inverse_rotation[2, 1] * dy
                    + inverse_rotation[2, 2] * dz
                )

                # np.cumsum over [start, step, step, ...] reproduces the
                # legacy scanner's repeated double addition along X.
                local_x = np.full(row_width, step_lx, dtype=np.float64)
                local_y = np.full(row_width, step_ly, dtype=np.float64)
                local_z = np.full(row_width, step_lz, dtype=np.float64)
                local_x[0] = start_lx
                local_y[0] = start_ly
                local_z[0] = start_lz
                np.cumsum(local_x, out=local_x)
                np.cumsum(local_y, out=local_y)
                np.cumsum(local_z, out=local_z)
                squared_radius = (
                    (local_x * local_x) * inv_a2
                    + (local_y * local_y) * inv_b2
                    + (local_z * local_z) * inv_c2
                )
                row = values[y_index, x_low : x_high + 1]
                # Ordered overwrite is intentional, including trash cells.
                row[squared_radius <= 1.0] = np.float32(cell.brightness)
        yield _opencv_u8(values)


def _center_voxel(value: float) -> int:
    return math.floor(float(np.float32(value)) + 0.5)


def _bbox_intersects(
    first: tuple[int, int, int, int, int, int],
    second: tuple[int, int, int, int, int, int],
) -> bool:
    return (
        first[0] < second[1]
        and second[0] < first[1]
        and first[2] < second[3]
        and second[2] < first[3]
        and first[4] < second[5]
        and second[4] < first[5]
    )


def _text_blocks(
    cell_id: str,
    *,
    origin_y: int,
    origin_x: int,
) -> tuple[tuple[int, int, int, int], ...]:
    scale = CENTER_ID_FONT_SCALE
    blocks: list[tuple[int, int, int, int]] = []
    stride = (5 + 1) * scale
    for glyph_index, character in enumerate(cell_id):
        glyph = DIGIT_GLYPHS_5X7[character]
        glyph_x = origin_x + glyph_index * stride
        for row_index, row in enumerate(glyph):
            for column_index, value in enumerate(row):
                if value != "1":
                    continue
                y0 = origin_y + row_index * scale
                x0 = glyph_x + column_index * scale
                blocks.append((y0, y0 + scale, x0, x0 + scale))
    return tuple(blocks)


def _center_id_annotations(
    spec: FrameSpec,
    *,
    marker_radius: int,
) -> tuple[CenterIdAnnotation, ...]:
    if marker_radius < 1 or marker_radius > 16:
        raise RecoveryError("center marker radius must be between 1 and 16")
    depth, height, width = spec.shape_zyx
    glyph_height = 7 * CENTER_ID_FONT_SCALE
    text_depth = 2 * CENTER_ID_TEXT_Z_HALF_THICKNESS + 1
    if depth < text_depth:
        raise RecoveryError(
            f"frame {spec.selected_frame} depth {depth} cannot fit the "
            f"{text_depth}-slice raster ID thickness"
        )
    centers: list[tuple[str, int, int, int]] = []
    marker_boxes: list[tuple[int, int, int, int, int, int]] = []
    for cell in spec.cells:
        unsupported = sorted(set(cell.name) - set(DIGIT_GLYPHS_5X7))
        if unsupported:
            raise RecoveryError(
                f"frame {spec.selected_frame} cell ID {cell.name!r} contains "
                "unsupported raster glyphs; center-ID TIFF mode currently "
                "supports decimal string IDs only"
            )
        if not cell.name:
            raise RecoveryError(
                f"frame {spec.selected_frame} contains an empty cell ID"
            )
        center_x, center_y, center_z = cell.center_xyz
        center = (
            _center_voxel(center_z),
            _center_voxel(center_y),
            _center_voxel(center_x),
        )
        center_z_i, center_y_i, center_x_i = center
        if not (
            0 <= center_z_i < depth
            and 0 <= center_y_i < height
            and 0 <= center_x_i < width
        ):
            raise RecoveryError(
                f"frame {spec.selected_frame} cell {cell.name!r} center "
                f"{center} is outside ZYX shape {spec.shape_zyx}"
            )
        centers.append((cell.name, center_z_i, center_y_i, center_x_i))
        marker_boxes.append(
            (
                max(0, center_z_i - marker_radius),
                min(depth, center_z_i + marker_radius + 1),
                max(0, center_y_i - marker_radius),
                min(height, center_y_i + marker_radius + 1),
                max(0, center_x_i - marker_radius),
                min(width, center_x_i + marker_radius + 1),
            )
        )

    annotations: list[CenterIdAnnotation] = []
    text_boxes: list[tuple[int, int, int, int, int, int]] = []
    for cell_index, (cell_id, center_z, center_y, center_x) in enumerate(centers):
        glyph_width = (
            len(cell_id) * 5 + max(0, len(cell_id) - 1)
        ) * CENTER_ID_FONT_SCALE
        if glyph_height > height or glyph_width > width:
            raise RecoveryError(
                f"frame {spec.selected_frame} cell {cell_id!r} raster ID "
                f"{glyph_height}x{glyph_width} does not fit YX shape "
                f"{height}x{width}"
            )
        centered_y = min(
            max(center_y - glyph_height // 2, 0),
            height - glyph_height,
        )
        centered_x = min(
            max(center_x - glyph_width // 2, 0),
            width - glyph_width,
        )
        right_x = center_x + marker_radius + 1 + CENTER_ID_TEXT_GAP
        left_x = (
            center_x
            - marker_radius
            - CENTER_ID_TEXT_GAP
            - glyph_width
        )
        down_y = center_y + marker_radius + 1 + CENTER_ID_TEXT_GAP
        up_y = (
            center_y
            - marker_radius
            - CENTER_ID_TEXT_GAP
            - glyph_height
        )
        candidates = (
            ("right", centered_y, right_x),
            ("left", centered_y, left_x),
            ("down", down_y, centered_x),
            ("up", up_y, centered_x),
            ("down-right", down_y, right_x),
            ("down-left", down_y, left_x),
            ("up-right", up_y, right_x),
            ("up-left", up_y, left_x),
        )
        selected: tuple[
            str,
            int,
            int,
            tuple[int, int, int, int, int, int],
        ] | None = None
        text_z0 = min(
            max(center_z - CENTER_ID_TEXT_Z_HALF_THICKNESS, 0),
            depth - text_depth,
        )
        text_z1 = text_z0 + text_depth
        for placement, origin_y, origin_x in candidates:
            if (
                origin_y < 0
                or origin_x < 0
                or origin_y + glyph_height > height
                or origin_x + glyph_width > width
            ):
                continue
            bbox = (
                text_z0,
                text_z1,
                origin_y,
                origin_y + glyph_height,
                origin_x,
                origin_x + glyph_width,
            )
            other_markers = (
                marker_boxes[:cell_index] + marker_boxes[cell_index + 1 :]
            )
            if any(_bbox_intersects(bbox, marker) for marker in other_markers):
                continue
            if any(_bbox_intersects(bbox, prior) for prior in text_boxes):
                continue
            selected = (placement, origin_y, origin_x, bbox)
            break
        if selected is None:
            raise RecoveryError(
                f"frame {spec.selected_frame} cell {cell_id!r} has no "
                "collision-free in-bounds position for its raster ID"
            )
        placement, origin_y, origin_x, text_bbox = selected
        annotations.append(
            CenterIdAnnotation(
                cell_id=cell_id,
                center_zyx=(center_z, center_y, center_x),
                text_placement=placement,
                text_blocks=_text_blocks(
                    cell_id,
                    origin_y=origin_y,
                    origin_x=origin_x,
                ),
                text_bbox_zyx=text_bbox,
            )
        )
        text_boxes.append(text_bbox)
    return tuple(annotations)


def _center_id_slices(
    spec: FrameSpec,
    *,
    marker_shape: str,
    marker_radius: int,
) -> Iterator[np.ndarray]:
    if marker_shape not in {"sphere", "cube"}:
        raise RecoveryError("center marker shape must be 'sphere' or 'cube'")
    depth, height, width = spec.shape_zyx
    annotations = _center_id_annotations(
        spec,
        marker_radius=marker_radius,
    )
    marker_value = np.uint8(CENTER_ID_RENDER_CONTRACT["marker_value"])
    text_value = np.uint8(CENTER_ID_RENDER_CONTRACT["text_value"])
    for z_index in range(depth):
        image = np.zeros((height, width), dtype=np.uint8)
        for annotation in annotations:
            center_z, center_y, center_x = annotation.center_zyx
            delta_z = z_index - center_z
            if abs(delta_z) > marker_radius:
                continue
            if marker_shape == "cube":
                y0 = max(0, center_y - marker_radius)
                y1 = min(height, center_y + marker_radius + 1)
                x0 = max(0, center_x - marker_radius)
                x1 = min(width, center_x + marker_radius + 1)
                image[y0:y1, x0:x1] = marker_value
                continue
            in_plane_squared = marker_radius * marker_radius - delta_z * delta_z
            for delta_y in range(-marker_radius, marker_radius + 1):
                remaining = in_plane_squared - delta_y * delta_y
                if remaining < 0:
                    continue
                y_index = center_y + delta_y
                if not 0 <= y_index < height:
                    continue
                half_width = math.isqrt(remaining)
                x0 = max(0, center_x - half_width)
                x1 = min(width, center_x + half_width + 1)
                image[y_index, x0:x1] = marker_value
        for annotation in annotations:
            z0, z1, _, _, _, _ = annotation.text_bbox_zyx
            if not z0 <= z_index < z1:
                continue
            for y0, y1, x0, x1 in annotation.text_blocks:
                image[y0:y1, x0:x1] = text_value
        yield image


def _tree_regular_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    total = 0
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        # Never recurse through directory symlinks.
        directory_names[:] = [
            name
            for name in directory_names
            if not Path(directory, name).is_symlink()
        ]
        for name in file_names:
            path = Path(directory, name)
            try:
                info = path.stat()
            except FileNotFoundError:
                continue
            if stat.S_ISREG(info.st_mode):
                total += info.st_size
    return total


def _legacy_flat_outputs(output_dir: Path) -> list[Path]:
    if not output_dir.is_dir():
        return []
    legacy: list[Path] = []
    for path in output_dir.iterdir():
        if (
            RECOVERED_NAME_PATTERN.fullmatch(path.name) is not None
            or CENTER_ID_NAME_PATTERN.fullmatch(path.name) is not None
            or path.name in {CENTER_ID_CSV_NAME, BATCH_RECOVERY_NAME}
        ):
            legacy.append(path)
    return sorted(legacy)


def _validate_grouped_output_layout(output_dir: Path) -> None:
    if os.path.lexists(output_dir):
        if output_dir.is_symlink() or not output_dir.is_dir():
            raise RecoveryError(
                f"output directory must be a real directory: {output_dir}"
            )
    legacy = _legacy_flat_outputs(output_dir)
    if legacy:
        preview = ", ".join(path.name for path in legacy[:8])
        suffix = "" if len(legacy) <= 8 else ", ..."
        raise RecoveryError(
            "legacy flat recovery output must be migrated before writing or "
            f"resuming grouped output: {preview}{suffix}"
        )
    for name in DATA_DIR_NAMES:
        path = output_dir / name
        if os.path.lexists(path) and (path.is_symlink() or not path.is_dir()):
            raise RecoveryError(
                f"output category must be a real directory: {path}"
            )


def _ensure_real_directory(path: Path) -> None:
    if os.path.lexists(path):
        if path.is_symlink() or not path.is_dir():
            raise RecoveryError(f"cannot use non-directory output path: {path}")
        return
    path.mkdir(parents=True, exist_ok=False)
    if path.is_symlink() or not path.is_dir():
        raise RecoveryError(f"cannot create real output directory: {path}")


def _budget_plan(
    output_dir: Path, shape_zyx: Sequence[int], max_bytes: int
) -> Mapping[str, int]:
    voxel_count = math.prod(shape_zyx)
    existing_bytes = _tree_regular_bytes(output_dir)
    one_tiff_reserved = voxel_count + TIFF_OVERHEAD_RESERVE
    planned_peak = (
        existing_bytes + 2 * one_tiff_reserved + JSON_OVERHEAD_RESERVE
    )
    if planned_peak > max_bytes:
        raise RecoveryError(
            "recovery would exceed materialization budget: "
            f"existing={existing_bytes}, reserved_peak={planned_peak}, "
            f"limit={max_bytes}"
        )
    return {
        "limit_bytes": max_bytes,
        "existing_bytes": existing_bytes,
        "voxel_count_per_output": voxel_count,
        "tiff_overhead_reserve_per_output": TIFF_OVERHEAD_RESERVE,
        "recovery_json_reserve": JSON_OVERHEAD_RESERVE,
        "reserved_peak_bytes": planned_peak,
    }


def _guard_current_budget(
    output_dir: Path, max_bytes: int, *, upcoming_bytes: int = 0
) -> None:
    current = _tree_regular_bytes(output_dir)
    if current + upcoming_bytes > max_bytes:
        raise RecoveryError(
            "materialization budget changed or was exhausted during recovery: "
            f"current={current}, upcoming={upcoming_bytes}, limit={max_bytes}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _temp_path(output_dir: Path, target: Path) -> Path:
    del output_dir
    return target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"


def _relative_output_path(output_dir: Path, target: Path) -> str:
    return target.relative_to(output_dir).as_posix()


def _stream_tiff(
    destination: Path,
    shape_zyx: Sequence[int],
    slices: Iterator[np.ndarray],
    *,
    compression: str,
    output_dir: Path,
    max_bytes: int,
) -> None:
    expected_z, expected_y, expected_x = shape_zyx
    compression_value: str | None = None if compression == "none" else "deflate"

    def guarded_slices() -> Iterator[np.ndarray]:
        count = 0
        for count, image in enumerate(slices, start=1):
            array = np.asarray(image)
            if array.dtype != np.uint8 or array.shape != (expected_y, expected_x):
                raise RecoveryError(
                    "slice provider returned unexpected dtype/shape: "
                    f"{array.dtype} {array.shape}"
                )
            _guard_current_budget(
                output_dir,
                max_bytes,
                upcoming_bytes=array.nbytes + NEXT_PAGE_RESERVE,
            )
            yield np.ascontiguousarray(array)
            _guard_current_budget(output_dir, max_bytes)
        if count != expected_z:
            raise RecoveryError(
                f"slice provider produced {count} slices, expected {expected_z}"
            )

    voxel_count = math.prod(shape_zyx)
    try:
        with tifffile.TiffWriter(
            destination, bigtiff=voxel_count >= (2**32 - 2**25)
        ) as writer:
            writer.write(
                data=guarded_slices(),
                shape=tuple(shape_zyx),
                dtype=np.uint8,
                photometric="minisblack",
                compression=compression_value,
                metadata={"axes": "ZYX"},
            )
    except RecoveryError:
        raise
    except Exception as exc:
        raise RecoveryError(f"cannot write TIFF {destination}: {exc}") from exc
    try:
        with destination.open("rb") as handle:
            os.fsync(handle.fileno())
    except OSError as exc:
        raise RecoveryError(f"cannot fsync TIFF {destination}: {exc}") from exc
    _guard_current_budget(output_dir, max_bytes)


def _output_targets(output_dir: Path, frame: int) -> tuple[Path, Path, Path]:
    stem = f"frame_{frame:06d}"
    return (
        output_dir / BACKGROUND_DIR_NAME / f"{stem}.background.tif",
        output_dir / CELLS_DIR_NAME / f"{stem}.cells.tif",
        output_dir / RECOVERY_DIR_NAME / f"{stem}.recovery.json",
    )


def _center_id_targets(output_dir: Path, frame: int) -> tuple[Path, Path]:
    stem = f"frame_{frame:06d}.center_ids"
    return (
        output_dir / CENTER_ID_STACK_DIR_NAME / f"{stem}.tif",
        output_dir / RECOVERY_DIR_NAME / f"{stem}.recovery.json",
    )


def _completed_recovery(
    output_dir: Path,
    spec: FrameSpec,
    *,
    compression: str,
) -> bool:
    frame = spec.selected_frame
    background_target, cells_target, recovery_target = _output_targets(
        output_dir, frame
    )
    targets = (background_target, cells_target, recovery_target)
    present = tuple(os.path.lexists(path) for path in targets)
    if not any(present):
        return False
    if not all(present):
        raise RecoveryError(
            f"cannot resume frame {frame}: recovery output is incomplete"
        )
    if any(path.is_symlink() or not path.is_file() for path in targets):
        raise RecoveryError(
            f"cannot resume frame {frame}: recovery outputs must be regular files"
        )

    record = _read_json(recovery_target, "recovery completion record")
    _check_schema(record, RECOVERY_SCHEMA, "recovery completion record")
    if record.get("status") != "complete":
        raise RecoveryError(
            f"cannot resume frame {frame}: recovery status is not complete"
        )
    if record.get("selected_frame") != frame:
        raise RecoveryError(
            f"cannot resume frame {frame}: completion record frame mismatch"
        )
    expected_record_fields: tuple[tuple[str, Any], ...] = (
        ("source_frame", spec.source_frame),
        ("pipeline_mode", spec.pipeline_mode),
        ("shape_zyx", list(spec.shape_zyx)),
        ("coordinates", dict(spec.coordinates)),
        ("cell_count", len(spec.cells)),
        ("trash_cell_count", sum(cell.is_trash for cell in spec.cells)),
        ("background_kind", spec.background["kind"]),
        ("compression", compression),
    )
    for key, expected in expected_record_fields:
        if record.get(key) != expected:
            raise RecoveryError(
                f"cannot resume frame {frame}: {key} no longer matches "
                "the compact input"
            )
    recorded_layout = record.get("output_layout")
    if recorded_layout is not None and recorded_layout != OUTPUT_LAYOUT:
        raise RecoveryError(
            f"cannot resume frame {frame}: output layout no longer matches"
        )
    inputs = _mapping(record.get("inputs"), "recovery inputs")
    expected_inputs = {
        "manifest": str(spec.manifest_path),
        "frame": str(spec.frame_path),
        "frame_sha256": _sha256(spec.frame_path),
    }
    if spec.background["kind"] == "binary_mask":
        mask_path = _mask_path(spec)
        expected_inputs["mask"] = str(mask_path)
        expected_inputs["mask_sha256"] = _sha256(mask_path)
    for key, expected in expected_inputs.items():
        if inputs.get(key) != expected:
            raise RecoveryError(
                f"cannot resume frame {frame}: input {key} no longer matches "
                "the compact package"
            )
    outputs = _mapping(record.get("outputs"), "recovery outputs")
    checks = (
        (
            background_target,
            "background_tiff",
            "background_bytes",
            "background_sha256",
        ),
        (cells_target, "cells_tiff", "cells_bytes", "cells_sha256"),
    )
    for path, name_key, bytes_key, sha_key in checks:
        recorded_path = outputs.get(name_key)
        accepted_paths = {
            path.name,
            _relative_output_path(output_dir, path),
        }
        if recorded_path not in accepted_paths:
            raise RecoveryError(
                f"cannot resume frame {frame}: {name_key} mismatch"
            )
        expected_bytes = _integer(outputs.get(bytes_key), bytes_key)
        if path.stat().st_size != expected_bytes:
            raise RecoveryError(
                f"cannot resume frame {frame}: {path.name} size mismatch"
            )
        expected_sha = _nonempty_string(outputs.get(sha_key), sha_key)
        if _sha256(path) != expected_sha:
            raise RecoveryError(
                f"cannot resume frame {frame}: {path.name} checksum mismatch"
            )
    recovery_paths = {
        recovery_target.name,
        _relative_output_path(output_dir, recovery_target),
    }
    if outputs.get("recovery_json") not in recovery_paths:
        raise RecoveryError(
            f"cannot resume frame {frame}: recovery_json mismatch"
        )
    return True


def _completed_center_id_stack(
    output_dir: Path,
    spec: FrameSpec,
    *,
    marker_shape: str,
    marker_radius: int,
) -> bool:
    tiff_target, recovery_target = _center_id_targets(
        output_dir,
        spec.selected_frame,
    )
    targets = (tiff_target, recovery_target)
    present = tuple(os.path.lexists(path) for path in targets)
    if not any(present):
        return False
    if not all(present):
        raise RecoveryError(
            f"cannot resume center-ID frame {spec.selected_frame}: "
            "output is incomplete"
        )
    if any(path.is_symlink() or not path.is_file() for path in targets):
        raise RecoveryError(
            f"cannot resume center-ID frame {spec.selected_frame}: "
            "outputs must be regular files"
        )

    record = _read_json(recovery_target, "center-ID completion record")
    _check_schema(
        record,
        CENTER_ID_RECOVERY_SCHEMA,
        "center-ID completion record",
    )
    expected_fields: tuple[tuple[str, Any], ...] = (
        ("status", "complete"),
        ("selected_frame", spec.selected_frame),
        ("source_frame", spec.source_frame),
        ("pipeline_mode", spec.pipeline_mode),
        ("shape_zyx", list(spec.shape_zyx)),
        ("coordinates", dict(spec.coordinates)),
        ("cell_count", len(spec.cells)),
        ("marker_shape", marker_shape),
        ("marker_radius", marker_radius),
        ("render_contract", dict(CENTER_ID_RENDER_CONTRACT)),
        ("compression", CENTER_ID_STACK_COMPRESSION),
    )
    for key, expected in expected_fields:
        if record.get(key) != expected:
            raise RecoveryError(
                f"cannot resume center-ID frame {spec.selected_frame}: "
                f"{key} no longer matches; use "
                "--overwrite-center-id-stacks to replace only this layer"
            )
    inputs = _mapping(record.get("inputs"), "center-ID recovery inputs")
    expected_inputs = {
        "frame": str(spec.frame_path),
        "frame_sha256": _sha256(spec.frame_path),
    }
    for key, expected in expected_inputs.items():
        if inputs.get(key) != expected:
            raise RecoveryError(
                f"cannot resume center-ID frame {spec.selected_frame}: "
                f"input {key} no longer matches"
            )
    outputs = _mapping(record.get("outputs"), "center-ID recovery outputs")
    accepted_tiff_paths = {
        tiff_target.name,
        _relative_output_path(output_dir, tiff_target),
    }
    if outputs.get("center_id_tiff") not in accepted_tiff_paths:
        raise RecoveryError(
            f"cannot resume center-ID frame {spec.selected_frame}: "
            "TIFF path mismatch"
        )
    accepted_record_paths = {
        recovery_target.name,
        _relative_output_path(output_dir, recovery_target),
    }
    if outputs.get("recovery_json") not in accepted_record_paths:
        raise RecoveryError(
            f"cannot resume center-ID frame {spec.selected_frame}: "
            "completion path mismatch"
        )
    expected_bytes = _integer(
        outputs.get("center_id_bytes"),
        "center_id_bytes",
    )
    if tiff_target.stat().st_size != expected_bytes:
        raise RecoveryError(
            f"cannot resume center-ID frame {spec.selected_frame}: "
            "TIFF size mismatch"
        )
    expected_sha = _nonempty_string(
        outputs.get("center_id_sha256"),
        "center_id_sha256",
    )
    if _sha256(tiff_target) != expected_sha:
        raise RecoveryError(
            f"cannot resume center-ID frame {spec.selected_frame}: "
            "TIFF checksum mismatch"
        )
    return True


def _existing_recovery_frame_ids(output_dir: Path) -> set[int]:
    if not output_dir.is_dir():
        return set()
    frame_ids: set[int] = set()
    for path in output_dir.rglob("*"):
        if not path.is_file():
            continue
        match = RECOVERED_NAME_PATTERN.fullmatch(path.name)
        if match is not None:
            frame_ids.add(int(match.group(1)))
    return frame_ids


def _existing_center_id_frame_ids(output_dir: Path) -> set[int]:
    if not output_dir.is_dir():
        return set()
    frame_ids: set[int] = set()
    for path in output_dir.rglob("*"):
        if not path.is_file():
            continue
        match = CENTER_ID_NAME_PATTERN.fullmatch(path.name)
        if match is not None:
            frame_ids.add(int(match.group(1)))
    return frame_ids


def _center_id_csv(specs: Sequence[FrameSpec]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(
        (
            "frame",
            "source_frame",
            "cell_id",
            "z",
            "y",
            "x",
            "draw_order",
            "is_trash",
            "brightness",
        )
    )
    for spec in specs:
        for cell in spec.cells:
            center_x, center_y, center_z = cell.center_xyz
            writer.writerow(
                (
                    spec.selected_frame,
                    spec.source_frame,
                    cell.name,
                    format(center_z, ".9g"),
                    format(center_y, ".9g"),
                    format(center_x, ".9g"),
                    cell.draw_order,
                    "true" if cell.is_trash else "false",
                    format(cell.brightness, ".9g"),
                )
            )
    encoded = stream.getvalue().encode("utf-8")
    if len(encoded) > BATCH_SIDECAR_RESERVE:
        raise RecoveryError(
            "cell center/ID CSV exceeds its reserved byte budget"
        )
    return encoded


def _atomic_write_bytes(
    target: Path,
    encoded: bytes,
    *,
    output_dir: Path,
    max_bytes: int,
) -> None:
    _ensure_real_directory(target.parent)
    temp = _temp_path(output_dir, target)
    try:
        _guard_current_budget(
            output_dir, max_bytes, upcoming_bytes=len(encoded)
        )
        with temp.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _guard_current_budget(output_dir, max_bytes)
        os.replace(temp, target)
        _guard_current_budget(output_dir, max_bytes)
    except Exception:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass
        raise


def recover(
    *,
    compact: Path,
    frame: int,
    output_dir: Path,
    compression: str = "none",
    max_materialized_bytes: int = DEFAULT_MAX_MATERIALIZED_BYTES,
    overwrite: bool = False,
    dry_run: bool = False,
) -> Mapping[str, Any]:
    if max_materialized_bytes <= 0:
        raise RecoveryError("--max-materialized-bytes must be > 0")
    spec = _load_contract(compact, frame)
    output_dir = output_dir.expanduser().resolve()
    _validate_grouped_output_layout(output_dir)
    background_target, cells_target, recovery_target = _output_targets(
        output_dir, frame
    )
    targets = (background_target, cells_target, recovery_target)
    if not overwrite:
        collisions = [path for path in targets if os.path.lexists(path)]
        if collisions:
            raise RecoveryError(
                "output exists; pass --overwrite to replace it: "
                + ", ".join(str(path) for path in collisions)
            )

    # Validate CUBM header/payload before budget reporting or any output write.
    if spec.background["kind"] == "binary_mask":
        with CubmMask(_mask_path(spec), spec.shape_zyx):
            pass

    budget = _budget_plan(
        output_dir, spec.shape_zyx, max_materialized_bytes
    )
    plan: dict[str, Any] = {
        "schema": RECOVERY_SCHEMA,
        "version": FORMAT_VERSION,
        "status": "dry-run" if dry_run else "complete",
        "selected_frame": frame,
        "source_frame": spec.source_frame,
        "pipeline_mode": spec.pipeline_mode,
        "shape_zyx": list(spec.shape_zyx),
        "coordinates": dict(spec.coordinates),
        "cell_count": len(spec.cells),
        "trash_cell_count": sum(cell.is_trash for cell in spec.cells),
        "background_kind": spec.background["kind"],
        "compression": compression,
        "output_layout": dict(OUTPUT_LAYOUT),
        "budget": dict(budget),
        "inputs": {
            "manifest": str(spec.manifest_path),
            "manifest_sha256": _sha256(spec.manifest_path),
            "frame": str(spec.frame_path),
            "frame_sha256": _sha256(spec.frame_path),
        },
        "outputs": {
            "background_tiff": _relative_output_path(
                output_dir, background_target
            ),
            "cells_tiff": _relative_output_path(output_dir, cells_target),
            "recovery_json": _relative_output_path(
                output_dir, recovery_target
            ),
        },
    }
    if spec.background["kind"] == "binary_mask":
        mask_path = _mask_path(spec)
        plan["inputs"]["mask"] = str(mask_path)
        plan["inputs"]["mask_sha256"] = _sha256(mask_path)

    if dry_run:
        return plan

    _ensure_real_directory(output_dir)
    for target in targets:
        _ensure_real_directory(target.parent)
    background_temp = _temp_path(output_dir, background_target)
    cells_temp = _temp_path(output_dir, cells_target)
    recovery_temp = _temp_path(output_dir, recovery_target)
    temporary_paths = (background_temp, cells_temp, recovery_temp)
    try:
        _stream_tiff(
            background_temp,
            spec.shape_zyx,
            _background_slices(spec),
            compression=compression,
            output_dir=output_dir,
            max_bytes=max_materialized_bytes,
        )
        _stream_tiff(
            cells_temp,
            spec.shape_zyx,
            _cells_slices(spec),
            compression=compression,
            output_dir=output_dir,
            max_bytes=max_materialized_bytes,
        )
        plan["outputs"]["background_sha256"] = _sha256(background_temp)
        plan["outputs"]["background_bytes"] = background_temp.stat().st_size
        plan["outputs"]["cells_sha256"] = _sha256(cells_temp)
        plan["outputs"]["cells_bytes"] = cells_temp.stat().st_size

        encoded = (
            json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        if len(encoded) > JSON_OVERHEAD_RESERVE:
            raise RecoveryError("recovery JSON exceeds its reserved byte budget")
        _guard_current_budget(
            output_dir, max_materialized_bytes, upcoming_bytes=len(encoded)
        )
        with recovery_temp.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _guard_current_budget(output_dir, max_materialized_bytes)

        # Each rename is atomic.  The recovery JSON is committed last and acts
        # as the completion marker for the two already-fsynced TIFFs.
        os.replace(background_temp, background_target)
        os.replace(cells_temp, cells_target)
        os.replace(recovery_temp, recovery_target)
        _guard_current_budget(output_dir, max_materialized_bytes)
    except Exception:
        for path in temporary_paths:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise
    return plan


def recover_center_id_stack(
    *,
    spec: FrameSpec,
    output_dir: Path,
    marker_shape: str,
    marker_radius: int,
    max_materialized_bytes: int,
    overwrite: bool = False,
    dry_run: bool = False,
) -> Mapping[str, Any]:
    if max_materialized_bytes <= 0:
        raise RecoveryError("--max-materialized-bytes must be > 0")
    if marker_shape not in {"sphere", "cube"}:
        raise RecoveryError("center marker shape must be 'sphere' or 'cube'")
    annotations = _center_id_annotations(spec, marker_radius=marker_radius)

    output_dir = output_dir.expanduser().resolve()
    _validate_grouped_output_layout(output_dir)
    tiff_target, recovery_target = _center_id_targets(
        output_dir,
        spec.selected_frame,
    )
    targets = (tiff_target, recovery_target)
    if not overwrite:
        collisions = [path for path in targets if os.path.lexists(path)]
        if collisions:
            raise RecoveryError(
                "center-ID output exists; pass "
                "--overwrite-center-id-stacks to replace it: "
                + ", ".join(str(path) for path in collisions)
            )

    existing_bytes = _tree_regular_bytes(output_dir)
    reserved_peak = (
        existing_bytes
        + CENTER_ID_COMPRESSED_RESERVE
        + JSON_OVERHEAD_RESERVE
    )
    if reserved_peak > max_materialized_bytes:
        raise RecoveryError(
            "center-ID recovery reserve would exceed materialization budget: "
            f"existing={existing_bytes}, reserved_peak={reserved_peak}, "
            f"limit={max_materialized_bytes}"
        )
    plan: dict[str, Any] = {
        "schema": CENTER_ID_RECOVERY_SCHEMA,
        "version": FORMAT_VERSION,
        "status": "dry-run" if dry_run else "complete",
        "selected_frame": spec.selected_frame,
        "source_frame": spec.source_frame,
        "pipeline_mode": spec.pipeline_mode,
        "shape_zyx": list(spec.shape_zyx),
        "coordinates": dict(spec.coordinates),
        "cell_count": len(spec.cells),
        "annotation_count": len(annotations),
        "text_placement_counts": {
            placement: sum(
                annotation.text_placement == placement
                for annotation in annotations
            )
            for placement in (
                "right",
                "left",
                "down",
                "up",
                "down-right",
                "down-left",
                "up-right",
                "up-left",
            )
        },
        "marker_shape": marker_shape,
        "marker_radius": marker_radius,
        "render_contract": dict(CENTER_ID_RENDER_CONTRACT),
        "compression": CENTER_ID_STACK_COMPRESSION,
        "budget": {
            "limit_bytes": max_materialized_bytes,
            "existing_bytes": existing_bytes,
            "uncompressed_payload_bytes": math.prod(spec.shape_zyx),
            "compressed_output_reserve_bytes": CENTER_ID_COMPRESSED_RESERVE,
            "recovery_json_reserve": JSON_OVERHEAD_RESERVE,
            "reserved_peak_bytes": reserved_peak,
        },
        "inputs": {
            "frame": str(spec.frame_path),
            "frame_sha256": _sha256(spec.frame_path),
        },
        "outputs": {
            "center_id_tiff": _relative_output_path(
                output_dir,
                tiff_target,
            ),
            "recovery_json": _relative_output_path(
                output_dir,
                recovery_target,
            ),
        },
    }
    if dry_run:
        return plan

    _ensure_real_directory(output_dir)
    for target in targets:
        _ensure_real_directory(target.parent)
    tiff_temp = _temp_path(output_dir, tiff_target)
    recovery_temp = _temp_path(output_dir, recovery_target)
    temporary_paths = (tiff_temp, recovery_temp)
    try:
        _stream_tiff(
            tiff_temp,
            spec.shape_zyx,
            _center_id_slices(
                spec,
                marker_shape=marker_shape,
                marker_radius=marker_radius,
            ),
            compression=CENTER_ID_STACK_COMPRESSION,
            output_dir=output_dir,
            max_bytes=max_materialized_bytes,
        )
        plan["outputs"]["center_id_sha256"] = _sha256(tiff_temp)
        plan["outputs"]["center_id_bytes"] = tiff_temp.stat().st_size
        encoded = (
            json.dumps(plan, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        if len(encoded) > JSON_OVERHEAD_RESERVE:
            raise RecoveryError(
                "center-ID recovery JSON exceeds its reserved byte budget"
            )
        _guard_current_budget(
            output_dir,
            max_materialized_bytes,
            upcoming_bytes=len(encoded),
        )
        with recovery_temp.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _guard_current_budget(output_dir, max_materialized_bytes)

        os.replace(tiff_temp, tiff_target)
        os.replace(recovery_temp, recovery_target)
        _guard_current_budget(output_dir, max_materialized_bytes)
    except Exception:
        for path in temporary_paths:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise
    return plan


def recover_all(
    *,
    compact: Path,
    output_dir: Path,
    compression: str = "none",
    max_materialized_bytes: int = DEFAULT_MAX_MATERIALIZED_BYTES,
    overwrite: bool = False,
    resume: bool = False,
    write_center_id_sidecar: bool = False,
    write_center_id_stacks: bool = False,
    overwrite_center_id_stacks: bool = False,
    center_marker_shape: str = "sphere",
    center_marker_radius: int = 2,
    dry_run: bool = False,
) -> Mapping[str, Any]:
    if max_materialized_bytes <= 0:
        raise RecoveryError("--max-materialized-bytes must be > 0")
    if overwrite and resume:
        raise RecoveryError("--overwrite and --resume cannot be used together")
    if overwrite_center_id_stacks and not write_center_id_stacks:
        raise RecoveryError(
            "--overwrite-center-id-stacks requires --center-id-stacks"
        )
    if overwrite_center_id_stacks and not resume:
        raise RecoveryError(
            "--overwrite-center-id-stacks requires --resume so existing "
            "background and cells outputs are verified and preserved"
        )
    if overwrite and overwrite_center_id_stacks:
        raise RecoveryError(
            "--overwrite and --overwrite-center-id-stacks cannot be combined"
        )
    if center_marker_shape not in {"sphere", "cube"}:
        raise RecoveryError(
            "--center-marker-shape must be 'sphere' or 'cube'"
        )
    if center_marker_radius < 1 or center_marker_radius > 16:
        raise RecoveryError(
            "--center-marker-radius must be between 1 and 16"
        )

    frame_ids = _manifest_frame_ids(compact)
    specs = tuple(_load_contract(compact, frame) for frame in frame_ids)
    for spec in specs:
        if spec.background["kind"] == "binary_mask":
            with CubmMask(_mask_path(spec), spec.shape_zyx):
                pass

    output_dir = output_dir.expanduser().resolve()
    _validate_grouped_output_layout(output_dir)
    center_target = output_dir / CENTERS_DIR_NAME / CENTER_ID_CSV_NAME
    batch_target = output_dir / RECOVERY_DIR_NAME / BATCH_RECOVERY_NAME
    skipped_frames: list[int] = []
    pending_specs: list[FrameSpec] = []
    center_id_skipped_frames: list[int] = []
    center_id_pending_specs: list[FrameSpec] = []

    if resume:
        stale_frames = sorted(
            _existing_recovery_frame_ids(output_dir) - set(frame_ids)
        )
        if stale_frames:
            preview = ", ".join(map(str, stale_frames[:8]))
            suffix = "" if len(stale_frames) <= 8 else ", ..."
            raise RecoveryError(
                "cannot resume: output contains frames that are no longer "
                f"in the compact manifest: {preview}{suffix}"
            )
        for spec in specs:
            if _completed_recovery(
                output_dir, spec, compression=compression
            ):
                skipped_frames.append(spec.selected_frame)
            else:
                pending_specs.append(spec)
    else:
        pending_specs.extend(specs)
        if not overwrite:
            collisions: list[Path] = []
            for spec in specs:
                collisions.extend(
                    path
                    for path in _output_targets(
                        output_dir, spec.selected_frame
                    )
                    if os.path.lexists(path)
                )
            sidecars = [batch_target]
            if write_center_id_sidecar:
                sidecars.append(center_target)
            collisions.extend(
                path for path in sidecars if os.path.lexists(path)
            )
            if collisions:
                preview = ", ".join(str(path) for path in collisions[:8])
                suffix = "" if len(collisions) <= 8 else ", ..."
                raise RecoveryError(
                    "batch output exists; pass --resume or --overwrite: "
                    + preview
                    + suffix
                )

    if write_center_id_stacks:
        for spec in specs:
            _center_id_annotations(
                spec,
                marker_radius=center_marker_radius,
            )
        stale_center_id_frames = sorted(
            _existing_center_id_frame_ids(output_dir) - set(frame_ids)
        )
        if stale_center_id_frames:
            preview = ", ".join(map(str, stale_center_id_frames[:8]))
            suffix = (
                ""
                if len(stale_center_id_frames) <= 8
                else ", ..."
            )
            raise RecoveryError(
                "cannot recover center-ID stacks: output contains frames "
                f"that are no longer in the compact manifest: {preview}{suffix}"
            )
        if resume and not overwrite_center_id_stacks:
            for spec in specs:
                if _completed_center_id_stack(
                    output_dir,
                    spec,
                    marker_shape=center_marker_shape,
                    marker_radius=center_marker_radius,
                ):
                    center_id_skipped_frames.append(spec.selected_frame)
                else:
                    center_id_pending_specs.append(spec)
        else:
            center_id_pending_specs.extend(specs)
            if not overwrite_center_id_stacks:
                center_id_collisions: list[Path] = []
                for spec in specs:
                    center_id_collisions.extend(
                        path
                        for path in _center_id_targets(
                            output_dir,
                            spec.selected_frame,
                        )
                        if os.path.lexists(path)
                    )
                if center_id_collisions:
                    preview = ", ".join(
                        str(path) for path in center_id_collisions[:8]
                    )
                    suffix = (
                        ""
                        if len(center_id_collisions) <= 8
                        else ", ..."
                    )
                    raise RecoveryError(
                        "center-ID output exists; pass --resume or "
                        "--overwrite-center-id-stacks: "
                        + preview
                        + suffix
                    )

    center_bytes = _center_id_csv(specs) if write_center_id_sidecar else b""
    existing_bytes = _tree_regular_bytes(output_dir)
    image_payload_bytes = sum(
        2 * math.prod(spec.shape_zyx) for spec in pending_specs
    )
    center_id_uncompressed_payload_bytes = sum(
        math.prod(spec.shape_zyx) for spec in center_id_pending_specs
    )
    center_id_reserve_bytes = len(center_id_pending_specs) * (
        CENTER_ID_COMPRESSED_RESERVE + JSON_OVERHEAD_RESERVE
    )
    batch_reserve_bytes = len(pending_specs) * (
        2 * BATCH_TIFF_OVERHEAD_RESERVE + JSON_OVERHEAD_RESERVE
    )
    batch_reserve_bytes += center_id_reserve_bytes
    batch_reserve_bytes += JSON_OVERHEAD_RESERVE
    if write_center_id_sidecar:
        batch_reserve_bytes += BATCH_SIDECAR_RESERVE
    projected_bytes = (
        existing_bytes + image_payload_bytes + batch_reserve_bytes
    )
    if projected_bytes > max_materialized_bytes:
        raise RecoveryError(
            "batch recovery preflight would exceed materialization budget: "
            f"existing={existing_bytes}, image_payload={image_payload_bytes}, "
            f"center_id_reserve={center_id_reserve_bytes}, "
            f"reserved_peak={projected_bytes}, "
            f"limit={max_materialized_bytes}"
        )

    unique_shapes = sorted({spec.shape_zyx for spec in specs})
    compact_path = compact.expanduser().resolve()
    manifest_path = (
        compact_path / "manifest.json"
        if compact_path.is_dir()
        else compact_path
    )
    result: dict[str, Any] = {
        "schema": BATCH_RECOVERY_SCHEMA,
        "version": FORMAT_VERSION,
        "status": "dry-run" if dry_run else "in-progress",
        "frame_count": len(specs),
        "frame_ids": list(frame_ids),
        "frames_to_recover": [
            spec.selected_frame for spec in pending_specs
        ],
        "frames_resumed": skipped_frames,
        "center_id_frames_to_recover": [
            spec.selected_frame for spec in center_id_pending_specs
        ],
        "center_id_frames_resumed": center_id_skipped_frames,
        "center_id_stacks": {
            "enabled": write_center_id_stacks,
            "directory": (
                CENTER_ID_STACK_DIR_NAME if write_center_id_stacks else None
            ),
            "marker_shape": (
                center_marker_shape if write_center_id_stacks else None
            ),
            "marker_radius": (
                center_marker_radius if write_center_id_stacks else None
            ),
            "render_contract": (
                dict(CENTER_ID_RENDER_CONTRACT)
                if write_center_id_stacks
                else None
            ),
        },
        "pipeline_modes": sorted({spec.pipeline_mode for spec in specs}),
        "shapes_zyx": [list(shape) for shape in unique_shapes],
        "cell_center_count": sum(len(spec.cells) for spec in specs),
        "compression": compression,
        "output_layout": dict(OUTPUT_LAYOUT),
        "budget": {
            "limit_bytes": max_materialized_bytes,
            "existing_bytes": existing_bytes,
            "uncompressed_image_payload_bytes": image_payload_bytes,
            "uncompressed_center_id_payload_bytes": (
                center_id_uncompressed_payload_bytes
            ),
            "center_id_committed_reserve_bytes": center_id_reserve_bytes,
            "committed_output_reserve_bytes": batch_reserve_bytes,
            "estimated_committed_peak_bytes": projected_bytes,
            "runtime_tiff_overhead_reserve_per_output": (
                TIFF_OVERHEAD_RESERVE
            ),
        },
        "inputs": {
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
        },
        "outputs": {
            "output_dir": str(output_dir),
            "batch_recovery_json": _relative_output_path(
                output_dir, batch_target
            ),
            "center_id_csv": (
                _relative_output_path(output_dir, center_target)
                if write_center_id_sidecar
                else None
            ),
            "center_id_stack_dir": (
                CENTER_ID_STACK_DIR_NAME if write_center_id_stacks else None
            ),
        },
    }
    if dry_run:
        if write_center_id_sidecar:
            result["outputs"]["center_id_csv_bytes"] = len(center_bytes)
        return result

    _ensure_real_directory(output_dir)
    in_progress = (
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if len(in_progress) > JSON_OVERHEAD_RESERVE:
        raise RecoveryError("batch recovery JSON exceeds its reserved byte budget")
    _atomic_write_bytes(
        batch_target,
        in_progress,
        output_dir=output_dir,
        max_bytes=max_materialized_bytes,
    )

    for index, spec in enumerate(pending_specs, start=1):
        print(
            f"[{index}/{len(pending_specs)}] recovering frame "
            f"{spec.selected_frame}",
            file=sys.stderr,
            flush=True,
        )
        recover(
            compact=compact,
            frame=spec.selected_frame,
            output_dir=output_dir,
            compression=compression,
            max_materialized_bytes=max_materialized_bytes,
            overwrite=overwrite,
            dry_run=False,
        )

    for index, spec in enumerate(center_id_pending_specs, start=1):
        print(
            f"[{index}/{len(center_id_pending_specs)}] recovering center-ID "
            f"stack for frame {spec.selected_frame}",
            file=sys.stderr,
            flush=True,
        )
        recover_center_id_stack(
            spec=spec,
            output_dir=output_dir,
            marker_shape=center_marker_shape,
            marker_radius=center_marker_radius,
            max_materialized_bytes=max_materialized_bytes,
            overwrite=overwrite_center_id_stacks,
            dry_run=False,
        )

    if write_center_id_sidecar:
        _atomic_write_bytes(
            center_target,
            center_bytes,
            output_dir=output_dir,
            max_bytes=max_materialized_bytes,
        )
        result["outputs"]["center_id_csv_bytes"] = center_target.stat().st_size
        result["outputs"]["center_id_csv_sha256"] = _sha256(center_target)

    result["status"] = "complete"
    existing_batch_bytes = (
        batch_target.stat().st_size if batch_target.exists() else 0
    )
    bytes_without_batch = _tree_regular_bytes(output_dir) - existing_batch_bytes
    final_bytes = bytes_without_batch
    encoded = b""
    for _ in range(8):
        result["budget"]["final_regular_bytes"] = final_bytes
        encoded = (
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        if len(encoded) > JSON_OVERHEAD_RESERVE:
            raise RecoveryError(
                "batch recovery JSON exceeds its reserved byte budget"
            )
        next_final_bytes = bytes_without_batch + len(encoded)
        if next_final_bytes == final_bytes:
            break
        final_bytes = next_final_bytes
    else:
        raise RecoveryError("batch recovery JSON size did not converge")
    _atomic_write_bytes(
        batch_target,
        encoded,
        output_dir=output_dir,
        max_bytes=max_materialized_bytes,
    )
    if _tree_regular_bytes(output_dir) != final_bytes:
        raise RecoveryError("final output size changed while committing batch record")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recover one or all compact-export frames into background-only "
            "and cells-only ZYX uint8 TIFF volumes."
        )
    )
    parser.add_argument(
        "--compact",
        required=True,
        type=Path,
        help="Compact package directory or its manifest.json",
    )
    frames = parser.add_mutually_exclusive_group(required=True)
    frames.add_argument(
        "--frame",
        type=int,
        help="Compact frame id used by frames/frame_NNNNNN.json",
    )
    frames.add_argument(
        "--all-frames",
        action="store_true",
        help="Recover every frame listed in the compact manifest",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--compression", choices=("none", "deflate"), default="none"
    )
    parser.add_argument(
        "--max-materialized-bytes",
        type=int,
        default=DEFAULT_MAX_MATERIALIZED_BYTES,
        help="Hard total output-root budget; default is decimal 10 GB",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "With --all-frames, checksum and skip completed frame triplets; "
            "fail on partial or corrupt outputs"
        ),
    )
    parser.add_argument(
        "--center-id-sidecar",
        action="store_true",
        help=(
            "With --all-frames, write sparse centers/cell_centers_ids.csv "
            "in TZYX display order"
        ),
    )
    parser.add_argument(
        "--center-id-stacks",
        action="store_true",
        help=(
            "With --all-frames, write independently resumable Deflate "
            "center_id_stacks/frame_NNNNNN.center_ids.tif volumes containing "
            "a center marker and adjacent raster ID for each current cell"
        ),
    )
    parser.add_argument(
        "--overwrite-center-id-stacks",
        action="store_true",
        help=(
            "Replace only center-ID TIFFs and their completion records; "
            "requires --all-frames --resume --center-id-stacks"
        ),
    )
    parser.add_argument(
        "--center-marker-shape",
        choices=("sphere", "cube"),
        default="sphere",
        help="Marker geometry for --center-id-stacks; default is sphere",
    )
    parser.add_argument(
        "--center-marker-radius",
        type=int,
        default=2,
        help="Marker radius in interpolated voxels; default is 2",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the plan without creating output directories/files",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.all_frames:
            result = recover_all(
                compact=args.compact,
                output_dir=args.output_dir,
                compression=args.compression,
                max_materialized_bytes=args.max_materialized_bytes,
                overwrite=args.overwrite,
                resume=args.resume,
                write_center_id_sidecar=args.center_id_sidecar,
                write_center_id_stacks=args.center_id_stacks,
                overwrite_center_id_stacks=args.overwrite_center_id_stacks,
                center_marker_shape=args.center_marker_shape,
                center_marker_radius=args.center_marker_radius,
                dry_run=args.dry_run,
            )
        else:
            if args.resume:
                raise RecoveryError("--resume requires --all-frames")
            if args.center_id_sidecar:
                raise RecoveryError(
                    "--center-id-sidecar requires --all-frames"
                )
            if args.center_id_stacks:
                raise RecoveryError(
                    "--center-id-stacks requires --all-frames"
                )
            if args.overwrite_center_id_stacks:
                raise RecoveryError(
                    "--overwrite-center-id-stacks requires "
                    "--all-frames --center-id-stacks"
                )
            result = recover(
                compact=args.compact,
                frame=args.frame,
                output_dir=args.output_dir,
                compression=args.compression,
                max_materialized_bytes=args.max_materialized_bytes,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
    except RecoveryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
