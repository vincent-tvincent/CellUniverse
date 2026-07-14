#!/usr/bin/env python3
"""Export experimental Cell Lumen fine shape masks from an existing cell CSV.

This is a review and research helper for the optional fine shape model. It does
not change Cell Universe tracking, PCA fitting, or CellLumen detection. Given a
raw TIFF stack and a cells or initial CSV, it grows a local brightness connected
component around each provided center and writes compact RLE mask rows that can
be rendered by render_fine_shape_masks.py. It can also write a Napari-friendly
label TIFF directly for visual review.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageSequence


@dataclass
class CellRow:
    frame: str
    name: str
    x: float
    y: float
    z_scaled: float
    rx: float
    ry: float
    rz_scaled: float
    z_mode_used: str


def scaled_distance(left: CellRow, right: CellRow) -> tuple[float, float, float]:
    dx = left.x - right.x
    dy = left.y - right.y
    dz = left.z_scaled - right.z_scaled
    lateral = float(np.hypot(dx, dy))
    distance = float(np.sqrt(dx * dx + dy * dy + dz * dz))
    return lateral, abs(dz), distance


def parse_float(row: dict[str, str], names: tuple[str, ...], default: float = 0.0) -> float:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return float(value)
    return default


def resolve_z_scaled(
    row: dict[str, str],
    z_value: float,
    z_mode: str,
    default_z_scaling: float,
    z_count: int,
) -> tuple[float, str, float]:
    voxel_z = parse_float(row, ("voxel_size_z", "z_scaling", "zScale"), default_z_scaling)
    if voxel_z <= 0:
        voxel_z = default_z_scaling

    if z_mode == "scaled":
        return z_value, "scaled", voxel_z
    if z_mode == "raw":
        return z_value * voxel_z, "raw", voxel_z

    has_explicit_scaled = any(
        row.get(name) not in (None, "")
        for name in ("z_scaled", "center_z_scaled")
    )
    if has_explicit_scaled:
        return z_value, "scaled", voxel_z

    # Cell Genesis Studio manual initial CSVs store z in slice coordinates and
    # provide voxel_size_z. Normal CellUniverse cells.csv files store z already
    # stretched and do not carry voxel_size_z. Keep this heuristic local to this
    # review tool so tracking behavior is not affected.
    if voxel_z > 1.01 and -1.0 <= z_value <= max(float(z_count) + 2.0, float(z_count) * 1.2):
        return z_value * voxel_z, "raw_auto", voxel_z
    return z_value, "scaled_auto", voxel_z


def resolve_z_radius_scaled(
    row: dict[str, str],
    rz_value: float,
    rx: float,
    ry: float,
    radius_z_mode: str,
    voxel_z: float,
) -> tuple[float, str]:
    if radius_z_mode == "scaled":
        return rz_value, "scaled"
    if radius_z_mode == "raw":
        return rz_value * voxel_z, "raw"

    if voxel_z > 1.01 and rz_value <= max(rx, ry) * 0.55:
        return rz_value * voxel_z, "raw_auto"
    return rz_value, "scaled_auto"


def load_cells(
    path: Path,
    frame_name: str | None,
    z_mode: str,
    radius_z_mode: str,
    default_z_scaling: float,
    z_count: int,
    unique_celltype_names: bool = False,
) -> list[CellRow]:
    rows: list[CellRow] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("isTrash", "0").lower() in {"1", "true", "yes"}:
                continue
            frame = row.get("file") or row.get("frame") or ""
            if frame_name and frame and frame != frame_name:
                continue
            z_value = parse_float(row, ("z", "z_scaled", "center_z", "center_z_scaled"))
            z_scaled, z_mode_used, voxel_z = resolve_z_scaled(
                row,
                z_value,
                z_mode,
                default_z_scaling,
                z_count,
            )
            rx = parse_float(row, ("majorRadius", "aRadius", "radius_scaled"), 18.0)
            ry = parse_float(row, ("bRadius", "radius_scaled"), rx)
            rz_value = parse_float(row, ("minorRadius", "cRadius", "radius_scaled"), min(rx, ry))
            rz_scaled, radius_mode_used = resolve_z_radius_scaled(
                row,
                rz_value,
                rx,
                ry,
                radius_z_mode,
                voxel_z,
            )
            explicit_name = row.get("name") or row.get("cell_name")
            if explicit_name:
                cell_name = explicit_name
            elif unique_celltype_names and row.get("cellType"):
                # C. elegans initial CSVs carry only a broad cellType label.
                # Mirror the main workflow's need for unique root identities so
                # rolling biological priors do not collapse all roots together.
                cell_name = f"{row['cellType']}_{len(rows) + 1}"
            else:
                cell_name = f"cell_{len(rows) + 1:04d}"
            rows.append(
                CellRow(
                    frame=frame or frame_name or "",
                    name=cell_name,
                    x=parse_float(row, ("x", "center_x")),
                    y=parse_float(row, ("y", "center_y")),
                    z_scaled=z_scaled,
                    rx=rx,
                    ry=ry,
                    rz_scaled=rz_scaled,
                    z_mode_used=f"{z_mode_used}/radius_{radius_mode_used}",
                )
            )
    return rows


def load_tiff_stack(path: Path) -> np.ndarray:
    try:
        image = Image.open(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read TIFF stack: {path}") from exc
    pages = []
    for page in ImageSequence.Iterator(image):
        arr = np.asarray(page)
        if arr.ndim == 3:
            arr = arr[..., 0]
        pages.append(arr.astype(np.float32))
    if not pages:
        raise RuntimeError(f"TIFF stack has no pages: {path}")
    stack = np.stack(pages, axis=0)
    if stack.max() > stack.min():
        stack = (stack - stack.min()) / (stack.max() - stack.min()) * 255.0
    return stack


def write_tiff_stack(path: Path, volume: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(volume[z]) for z in range(volume.shape[0])]
    if not images:
        raise RuntimeError(f"Cannot write empty TIFF stack: {path}")
    images[0].save(str(path), save_all=True, append_images=images[1:])


def stretch_z(volume: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return volume
    return np.repeat(volume, factor, axis=0)


def neighbor_offsets(connectivity: int) -> tuple[tuple[int, int, int], ...]:
    if connectivity == 6:
        return (
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        )
    offsets = []
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue
                offsets.append((dx, dy, dz))
    return tuple(offsets)


def cell_volume_score(cell: CellRow) -> float:
    return max(cell.rx, 1.0) * max(cell.ry, 1.0) * max(cell.rz_scaled, 1.0)


def ellipsoid_norm_distance(cell: CellRow, prior: CellRow, args: argparse.Namespace) -> float:
    rx = max(1.0e-6, prior.rx * args.prior_parent_region_scale + args.prior_parent_region_margin)
    ry = max(1.0e-6, prior.ry * args.prior_parent_region_scale + args.prior_parent_region_margin)
    rz = max(1.0e-6, prior.rz_scaled * args.prior_parent_region_scale + args.prior_parent_region_margin)
    dx = (cell.x - prior.x) / rx
    dy = (cell.y - prior.y) / ry
    dz = (cell.z_scaled - prior.z_scaled) / rz
    return float(np.sqrt(dx * dx + dy * dy + dz * dz))


def biological_prior_score(cell: CellRow, prior: CellRow, distance: float, norm: float, args: argparse.Namespace) -> float:
    # Prefer candidates with normal cell-sized support, but mildly penalize
    # candidates that are far from the previous-frame parent. This review-only
    # score intentionally avoids using GT or current-frame cell count.
    return (
        cell_volume_score(cell)
        - args.prior_distance_score_weight * distance
        - args.prior_norm_score_weight * norm
    )


def effective_prior_reach_distance(prior: CellRow, args: argparse.Namespace) -> float:
    """Return the reach distance for one previous-frame parent cell.

    A fixed global reach suppresses remote bright fragments, but f035 showed
    that a real daughter from a large parent can sit just outside the old 50 px
    reach. This optional rule keeps the old default unless explicitly enabled,
    then scales reach by the actual parent radius instead of adding another
    frame-specific threshold.
    """
    base_reach = args.prior_reach_distance
    if not getattr(args, "prior_dynamic_reach_from_radius", False):
        return base_reach
    parent_radius = max(prior.rx, prior.ry, prior.rz_scaled)
    dynamic_reach = (
        parent_radius * args.prior_dynamic_reach_radius_scale
        + args.prior_dynamic_reach_margin
    )
    return max(base_reach, dynamic_reach)


def cell_support_radius_score(cell: CellRow) -> float:
    return float(cell_volume_score(cell) ** (1.0 / 3.0))


def renamed_cell(cell: CellRow, name: str) -> CellRow:
    return CellRow(
        frame=cell.frame,
        name=name,
        x=cell.x,
        y=cell.y,
        z_scaled=cell.z_scaled,
        rx=cell.rx,
        ry=cell.ry,
        rz_scaled=cell.rz_scaled,
        z_mode_used=cell.z_mode_used,
    )


def coordinate_copy_key(cell: CellRow, epsilon: float) -> tuple[int, int, int]:
    scale = max(epsilon, 1.0e-9)
    return (
        int(round(cell.x / scale)),
        int(round(cell.y / scale)),
        int(round(cell.z_scaled / scale)),
    )


def coordinate_copy_priority(cell: CellRow) -> tuple[int, int, int, int, float, str]:
    name = cell.name
    # These suffixes are generated by experimental rescue paths. When several
    # rows have the exact same center, prefer the original physical candidate
    # as the representative and keep rescue copies only when an explicit
    # pending close-split option asks for one.
    return (
        1 if "futureSplitDup" in name else 0,
        1 if "multiParent" in name else 0,
        1 if "singleZAlt" in name else 0,
        1 if "unresolvedSplitDup" in name else 0,
        -cell_volume_score(cell),
        name,
    )


def future_split_projection_rank(
    cell: CellRow,
    args: argparse.Namespace,
    stack: np.ndarray | None,
    future_cells: list[CellRow] | None,
) -> tuple[float, float, float, str]:
    if stack is None or not future_cells:
        return (float("inf"), float("inf"), 0.0, cell.name)

    nearby: list[tuple[float, CellRow]] = []
    radius = max(1.0e-6, args.prior_future_split_pressure_distance)
    for future in future_cells:
        _lateral, _dz, distance = scaled_distance(cell, future)
        if distance <= radius:
            nearby.append((distance, future))
    if len(nearby) < 2:
        return (float("inf"), float("inf"), 0.0, cell.name)

    best: tuple[float, float, float, str] | None = None
    projection_sep = max(1.0, args.prior_coordinate_copy_future_projection_separation)
    for i, (left_distance, left_future) in enumerate(nearby):
        for right_distance, right_future in nearby[i + 1:]:
            _lateral, _dz, future_sep = scaled_distance(left_future, right_future)
            if future_sep < args.prior_future_split_pressure_min_separation:
                continue
            midpoint = CellRow(
                frame=cell.frame,
                name=f"{left_future.name}+{right_future.name}_midpoint",
                x=(left_future.x + right_future.x) * 0.5,
                y=(left_future.y + right_future.y) * 0.5,
                z_scaled=(left_future.z_scaled + right_future.z_scaled) * 0.5,
                rx=cell.rx,
                ry=cell.ry,
                rz_scaled=cell.rz_scaled,
                z_mode_used=cell.z_mode_used,
            )
            _mid_lateral, _mid_dz, midpoint_distance = scaled_distance(cell, midpoint)
            axis_left = CellRow(
                frame=cell.frame,
                name=f"{cell.name}_futureAxisA",
                x=cell.x - (right_future.x - left_future.x) * projection_sep / (2.0 * future_sep),
                y=cell.y - (right_future.y - left_future.y) * projection_sep / (2.0 * future_sep),
                z_scaled=cell.z_scaled
                - (right_future.z_scaled - left_future.z_scaled) * projection_sep / (2.0 * future_sep),
                rx=cell.rx,
                ry=cell.ry,
                rz_scaled=cell.rz_scaled,
                z_mode_used=cell.z_mode_used,
            )
            axis_right = CellRow(
                frame=cell.frame,
                name=f"{cell.name}_futureAxisB",
                x=cell.x + (right_future.x - left_future.x) * projection_sep / (2.0 * future_sep),
                y=cell.y + (right_future.y - left_future.y) * projection_sep / (2.0 * future_sep),
                z_scaled=cell.z_scaled
                + (right_future.z_scaled - left_future.z_scaled) * projection_sep / (2.0 * future_sep),
                rx=cell.rx,
                ry=cell.ry,
                rz_scaled=cell.rz_scaled,
                z_mode_used=cell.z_mode_used,
            )
            valley_ratio, gap_drop = bridge_slab_valley_ratio(stack, axis_left, axis_right, args)
            pressure = max(0.0, radius - max(left_distance, right_distance)) + 0.05 * future_sep
            rank = (valley_ratio, midpoint_distance, -pressure, cell.name)
            if best is None or rank < best:
                best = rank
    if best is None:
        return (float("inf"), float("inf"), 0.0, cell.name)
    return best


def collapse_coordinate_copies(
    cells: list[CellRow],
    args: argparse.Namespace,
    stack: np.ndarray | None = None,
    future_cells: list[CellRow] | None = None,
) -> list[CellRow]:
    if not getattr(args, "prior_collapse_coordinate_copies", False):
        return cells
    groups: dict[tuple[int, int, int], list[CellRow]] = {}
    for cell in cells:
        groups.setdefault(coordinate_copy_key(cell, args.prior_coordinate_copy_epsilon), []).append(cell)

    collapsed: list[CellRow] = []
    future_candidates: list[tuple[tuple[float, float, float, str], CellRow]] = []
    removed = 0
    kept_unresolved = 0
    kept_future = 0
    for group in groups.values():
        ordered = sorted(group, key=coordinate_copy_priority)
        collapsed.append(ordered[0])
        allowed_extra = 0
        if getattr(args, "prior_coordinate_copy_keep_unresolved", False):
            unresolved = [cell for cell in ordered if "unresolvedSplitDup" in cell.name]
            if unresolved:
                collapsed.append(unresolved[0])
                kept_unresolved += 1
                allowed_extra += 1
        if getattr(args, "prior_coordinate_copy_keep_future", False):
            future = [cell for cell in ordered if "futureSplitDup" in cell.name]
            if future:
                future_candidates.append(
                    (
                        future_split_projection_rank(future[0], args, stack, future_cells),
                        future[0],
                    )
                )
                allowed_extra += 1
        removed += max(0, len(group) - 1 - allowed_extra)

    if future_candidates:
        future_candidates.sort(key=lambda item: item[0])
        max_future = int(getattr(args, "prior_coordinate_copy_max_future_kept", -1))
        if max_future >= 0:
            future_candidates = future_candidates[:max_future]
        collapsed.extend(cell for _rank, cell in future_candidates)
        kept_future = len(future_candidates)

    if removed or kept_unresolved or kept_future:
        print(
            f"[FineShape Coordinate Copy Collapse] before={len(cells)} "
            f"after={len(collapsed)} groups={len(groups)} removed={removed} "
            f"kept_unresolved={kept_unresolved} kept_future={kept_future} "
            f"future_budget={getattr(args, 'prior_coordinate_copy_max_future_kept', -1)} "
            f"epsilon={args.prior_coordinate_copy_epsilon}"
        )
    return collapsed


def daughter_name(parent_name: str, daughter_index: int) -> str:
    if "_" not in parent_name:
        return f"{parent_name}_{daughter_index}"
    prefix, suffix = parent_name.rsplit("_", 1)
    if suffix.isdigit():
        return f"{prefix}_{suffix}{daughter_index}"
    return f"{parent_name}_{daughter_index}"


def ultrack_style_candidate_score(
    item: tuple[CellRow, float, float, float],
    args: argparse.Namespace,
    future_cells: list[CellRow] | None = None,
    include_motion_balance: bool = True,
) -> float:
    cell, distance, norm, _old_score = item
    support = min(args.prior_soft_support_clip, cell_support_radius_score(cell))
    future_support, _future_distance, _future_name = future_candidate_support(cell, future_cells, args)
    motion_balance_penalty = (
        future_motion_balance_penalty(cell, distance, future_cells, args)
        if include_motion_balance
        else 0.0
    )
    return (
        args.prior_soft_support_weight * support
        - args.prior_soft_distance_weight * distance
        - args.prior_soft_norm_weight * norm
        + args.prior_future_support_weight * future_support
        - motion_balance_penalty
    )


def future_candidate_support(
    cell: CellRow,
    future_cells: list[CellRow] | None,
    args: argparse.Namespace,
) -> tuple[float, float, str]:
    """Soft temporal support from the next frame's raw Cell Lumen candidates.

    This is an experimental one-frame analog of Ultrack's temporal consistency:
    a current-frame candidate is more trustworthy when a nearby brightness
    candidate persists into the next image. It is a soft score, not a gate, so
    candidates without future data can still be selected when this feature is
    disabled or the evidence is otherwise strong.
    """
    if not future_cells:
        return 0.0, float("inf"), ""
    best_cell = min(future_cells, key=lambda other: scaled_distance(cell, other)[2])
    _lateral, _dz, best_distance = scaled_distance(cell, best_cell)
    radius = max(1.0e-6, args.prior_future_support_distance)
    support = max(0.0, radius - best_distance)
    return float(support), float(best_distance), best_cell.name


def future_motion_balance_penalty(
    cell: CellRow,
    prior_distance: float,
    future_cells: list[CellRow] | None,
    args: argparse.Namespace,
) -> float:
    """Softly penalize edge lobes that are near the prior but do not persist.

    This keeps the experimental Cell Lumen selector anchored to the successful
    centroid BioFuse baseline while adding a small Ultrack-style two-sided
    temporal consistency term. The f016 regression case needs this exact kind
    of evidence: the false edge lobe is close to the previous center but far
    from the next-frame continuation, while the true center has balanced
    previous and next distances. The weight defaults to zero so existing review
    behavior is unchanged unless a rolling audit opts in.
    """
    weight = getattr(args, "prior_future_motion_balance_weight", 0.0)
    if weight <= 0.0 or not future_cells:
        return 0.0
    _future_support, future_distance, _future_name = future_candidate_support(cell, future_cells, args)
    if not math.isfinite(future_distance):
        return 0.0
    tolerance = max(0.0, getattr(args, "prior_future_motion_balance_tolerance", 8.0))
    imbalance = max(0.0, abs(prior_distance - future_distance) - tolerance)
    return weight * imbalance


def future_pair_support(
    left: CellRow,
    right: CellRow,
    future_cells: list[CellRow] | None,
    args: argparse.Namespace,
) -> tuple[float, bool, str, str]:
    """Score whether two current candidates have two separate future witnesses.

    Ultrack does not accept a division only because each daughter has a nearby
    future point; the two daughter hypotheses also need separate outgoing links.
    This experimental helper keeps that idea local to Cell Lumen: a close pair
    gets extra soft evidence only when each side links to a distinct next-frame
    raw candidate. The default weights are zero, so existing review behavior is
    unchanged unless the rolling audit opts in.
    """
    left_support, _left_distance, left_name = future_candidate_support(left, future_cells, args)
    right_support, _right_distance, right_name = future_candidate_support(right, future_cells, args)
    distinct = bool(left_name and right_name and left_name != right_name)
    return float(min(left_support, right_support)), distinct, left_name, right_name


def merged_future_pair_support(
    left: CellRow,
    right: CellRow,
    future_cells: list[CellRow] | None,
    args: argparse.Namespace,
) -> tuple[float, float, str]:
    """Soft support for close daughters represented by one future raw peak.

    The raw Cell Lumen detector can merge very close daughters in the next
    frame even when the current BioFuse group already contains a plausible
    splitA/splitB pair. A full Ultrack graph would keep a merged future
    hypothesis as a node; this local experimental version gives soft support
    when one future brightness candidate lies near the pair midpoint.
    """
    if not future_cells:
        return 0.0, float("inf"), ""
    midpoint = CellRow(
        frame=left.frame,
        name=f"{left.name}+{right.name}_midpoint",
        x=(left.x + right.x) * 0.5,
        y=(left.y + right.y) * 0.5,
        z_scaled=(left.z_scaled + right.z_scaled) * 0.5,
        rx=max(left.rx, right.rx),
        ry=max(left.ry, right.ry),
        rz_scaled=max(left.rz_scaled, right.rz_scaled),
        z_mode_used=left.z_mode_used,
    )
    best_cell = min(future_cells, key=lambda other: scaled_distance(midpoint, other)[2])
    _lateral, _dz, best_distance = scaled_distance(midpoint, best_cell)
    radius = max(1.0e-6, args.prior_future_support_distance)
    return max(0.0, radius - best_distance), float(best_distance), best_cell.name


def future_split_pressure(
    cell: CellRow,
    future_cells: list[CellRow] | None,
    args: argparse.Namespace,
) -> float:
    """Return soft evidence that one current center may become two centers.

    This is not a GT count or a hard split trigger. It only penalizes a
    continuation hypothesis when the next raw Cell Lumen frame already contains
    two separated nearby brightness candidates. That mirrors Ultrack's
    continuation-versus-division competition without copying its full ILP.
    """
    if not future_cells:
        return 0.0
    nearby: list[tuple[float, CellRow]] = []
    radius = max(1.0e-6, args.prior_future_split_pressure_distance)
    for future in future_cells:
        _lateral, _dz, dist = scaled_distance(cell, future)
        if dist <= radius:
            nearby.append((dist, future))
    if len(nearby) < 2:
        return 0.0
    best = 0.0
    for i, (left_dist, left) in enumerate(nearby):
        for right_dist, right in nearby[i + 1:]:
            _lateral, _dz, sep = scaled_distance(left, right)
            if sep < args.prior_future_split_pressure_min_separation:
                continue
            support = max(0.0, radius - max(left_dist, right_dist))
            best = max(best, support + 0.05 * sep)
    return float(best)


def _sample_stack_nearest(stack: np.ndarray, x: float, y: float, z_scaled: float, z_scaling: float) -> float:
    z = int(round(z_scaled / max(z_scaling, 1.0e-6)))
    yy = int(round(y))
    xx = int(round(x))
    if z < 0 or z >= stack.shape[0] or yy < 0 or yy >= stack.shape[1] or xx < 0 or xx >= stack.shape[2]:
        return 0.0
    return float(stack[z, yy, xx])


def bridge_slab_valley_ratio(
    stack: np.ndarray | None,
    left: CellRow,
    right: CellRow,
    args: argparse.Namespace,
) -> tuple[float, float]:
    """Estimate a 3D saddle ratio between two candidate centers.

    This is a lightweight experimental analog of the CellUniverse split bridge
    preview. It samples small cross-section slabs around the daughter axis
    before the final rolling selection so ambiguous two-center hypotheses are
    scored instead of being accepted just because they pass a distance gate.
    """
    if stack is None:
        return 1.0, 0.0

    vec = np.array([right.x - left.x, right.y - left.y, right.z_scaled - left.z_scaled], dtype=np.float64)
    length = float(np.linalg.norm(vec))
    if length <= 1.0e-6:
        return 1.0, 0.0
    axis = vec / length
    # Build a stable orthonormal basis in scaled xyz coordinates.
    ref = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(axis, ref))) > 0.92:
        ref = np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, ref)
    u /= max(np.linalg.norm(u), 1.0e-6)
    v = np.cross(axis, u)
    v /= max(np.linalg.norm(v), 1.0e-6)

    cross_radius = max(0.0, args.prior_bridge_cross_radius)
    samples = max(5, int(args.prior_bridge_samples))
    offsets: list[tuple[float, float]] = [(0.0, 0.0)]
    for radius in (cross_radius * 0.5, cross_radius):
        if radius <= 0.0:
            continue
        for angle in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
            offsets.append((float(np.cos(angle) * radius), float(np.sin(angle) * radius)))

    profile: list[float] = []
    for t in np.linspace(0.0, 1.0, samples):
        center = np.array(
            [
                left.x + vec[0] * t,
                left.y + vec[1] * t,
                left.z_scaled + vec[2] * t,
            ],
            dtype=np.float64,
        )
        values = []
        for ou, ov in offsets:
            point = center + u * ou + v * ov
            values.append(_sample_stack_nearest(stack, point[0], point[1], point[2], args.z_scaling))
        profile.append(float(np.mean(values)))

    if len(profile) < 3:
        return 1.0, 0.0
    edge_count = max(1, len(profile) // 4)
    left_edge = max(profile[:edge_count])
    right_edge = max(profile[-edge_count:])
    edge_bright = max(left_edge, right_edge, 1.0e-6)
    mid_start = len(profile) // 3
    mid_end = len(profile) - mid_start
    valley = min(profile[mid_start:mid_end]) if mid_start < mid_end else min(profile)
    valley_ratio = valley / edge_bright
    gap_drop = edge_bright - valley
    return float(valley_ratio), float(gap_drop)


def candidates_overlap_as_duplicates(left: CellRow, right: CellRow, args: argparse.Namespace) -> bool:
    _, _, distance = scaled_distance(left, right)
    left_radius = cell_volume_score(left) ** (1.0 / 3.0)
    right_radius = cell_volume_score(right) ** (1.0 / 3.0)
    overlap_limit = (left_radius + right_radius) * args.prior_duplicate_radius_fraction
    return distance <= min(args.prior_same_parent_duplicate_distance, overlap_limit)


def preserved_split_group_key(name: str) -> str | None:
    for marker in ("_biofused_splitA", "_biofused_splitB"):
        if marker in name:
            return name.split(marker, 1)[0]
    return None


def preserved_split_siblings(left: CellRow, right: CellRow) -> bool:
    left_is_a = "_biofused_splitA" in left.name
    left_is_b = "_biofused_splitB" in left.name
    right_is_a = "_biofused_splitA" in right.name
    right_is_b = "_biofused_splitB" in right.name
    if (left_is_a and right_is_b) or (left_is_b and right_is_a):
        return True
    left_key = preserved_split_group_key(left.name)
    right_key = preserved_split_group_key(right.name)
    return bool(left_key and right_key and left_key == right_key and left.name != right.name)


def z_column_alternative_siblings(left: CellRow, right: CellRow, args: argparse.Namespace) -> bool:
    if not getattr(args, "prior_preserve_z_column_alternatives", False):
        return False
    lateral, dz, _distance = scaled_distance(left, right)
    return (
        lateral <= args.prior_z_column_alternative_lateral
        and dz >= args.prior_z_column_alternative_min_dz
    )


def representative_alternative_siblings(left: CellRow, right: CellRow, args: argparse.Namespace) -> bool:
    if not getattr(args, "prior_preserve_representative_alternatives", False):
        return False
    if "_repAlt" not in left.name and "_repAlt" not in right.name:
        return False
    _lateral, _dz, distance = scaled_distance(left, right)
    return distance >= args.prior_representative_alternative_min_distance


def lineage_suffix_length(name: str) -> int:
    if "_" not in name:
        return 0
    return len(name.rsplit("_", 1)[-1])


def choose_prior_group_candidates(
    prior: CellRow,
    items: list[tuple[CellRow, float, float, float]],
    args: argparse.Namespace,
    stack: np.ndarray | None = None,
    future_cells: list[CellRow] | None = None,
) -> list[CellRow]:
    if not items:
        return []

    rank_score = (
        (lambda item: ultrack_style_candidate_score(item, args, future_cells))
        if args.prior_ultrack_style_selection
        else (lambda item: item[3])
    )
    pair_rank_score = rank_score
    if args.prior_ultrack_style_selection and getattr(
        args,
        "prior_future_motion_balance_single_only",
        False,
    ):
        pair_rank_score = lambda item: ultrack_style_candidate_score(
            item,
            args,
            future_cells,
            include_motion_balance=False,
        )
    ranked = sorted(items, key=rank_score, reverse=True)
    deduped: list[tuple[CellRow, float, float, float]] = []
    for item in ranked:
        candidate = item[0]
        if any(
            candidates_overlap_as_duplicates(candidate, kept[0], args)
            and not preserved_split_siblings(candidate, kept[0])
            and not z_column_alternative_siblings(candidate, kept[0], args)
            and not representative_alternative_siblings(candidate, kept[0], args)
            for kept in deduped
        ):
            continue
        deduped.append(item)

    if args.prior_max_candidates_per_parent <= 1 or len(deduped) <= 1:
        selected = [deduped[0][0]]
        if getattr(args, "prior_duplicate_single_on_future_split_pressure", False):
            split_pressure = future_split_pressure(deduped[0][0], future_cells, args)
            if split_pressure >= args.prior_single_future_split_duplicate_min_pressure:
                # The single-candidate fast path still needs the same recall
                # protection as the Ultrack-style branch: one current raw
                # center can represent two close daughters that only become
                # distinguishable in the next frame.
                selected.append(
                    renamed_cell(
                        deduped[0][0],
                        f"{deduped[0][0].name}_futureSplitDup",
                    )
                )
        return selected

    if args.prior_ultrack_style_selection:
        def single_score(item: tuple[CellRow, float, float, float]) -> float:
            split_pressure = future_split_pressure(item[0], future_cells, args)
            return rank_score(item) - args.prior_single_future_pressure_penalty * split_pressure

        best_single = max(deduped, key=single_score)
        best_single_score = single_score(best_single)
        best_pair: tuple[float, tuple[CellRow, CellRow], float, float, float] | None = None
        split_min_separation = args.prior_split_min_separation
        if (
            args.prior_root_parent_split_min_separation > 0.0
            and lineage_suffix_length(prior.name) <= 1
        ):
            split_min_separation = max(split_min_separation, args.prior_root_parent_split_min_separation)
        for i, left in enumerate(deduped):
            for right in deduped[i + 1:]:
                _, _, sep = scaled_distance(left[0], right[0])
                left_volume = cell_volume_score(left[0])
                right_volume = cell_volume_score(right[0])
                volume_ratio = max(left_volume, right_volume) / max(1.0, min(left_volume, right_volume))
                valley_ratio, gap_drop = bridge_slab_valley_ratio(stack, left[0], right[0], args)
                sep_shortfall = max(0.0, split_min_separation - sep)
                volume_excess = 0.0
                if args.prior_split_max_volume_ratio > 0.0:
                    volume_excess = max(0.0, volume_ratio - args.prior_split_max_volume_ratio)
                parent_volume_excess = 0.0
                if (
                    args.prior_soft_pair_parent_volume_ratio_target > 0.0
                    and args.prior_soft_pair_parent_volume_ratio_penalty > 0.0
                ):
                    parent_volume = max(1.0, cell_volume_score(prior))
                    pair_parent_volume_ratio = (left_volume + right_volume) / parent_volume
                    parent_volume_excess = max(
                        0.0,
                        pair_parent_volume_ratio
                        - args.prior_soft_pair_parent_volume_ratio_target,
                    )
                bridge_dark_bonus = max(0.0, args.prior_bridge_target_valley_ratio - valley_ratio)
                bridge_bright_penalty = max(0.0, valley_ratio - args.prior_bridge_target_valley_ratio)
                close_bright_bridge_penalty = 0.0
                if (
                    args.prior_close_bright_bridge_penalty_weight > 0.0
                    and args.prior_close_bright_bridge_close_sep > 0.0
                    and sep < args.prior_close_bright_bridge_close_sep
                ):
                    # Experimental soft cost for close pair hypotheses whose
                    # 3D bridge remains bright. It targets internal duplicate
                    # peaks without hard-rejecting low-valley close daughters.
                    valley_excess = max(
                        0.0,
                        valley_ratio - args.prior_close_bright_bridge_target_valley,
                    )
                    sep_deficit = args.prior_close_bright_bridge_close_sep - sep
                    close_bright_bridge_penalty = (
                        args.prior_close_bright_bridge_penalty_weight
                        * valley_excess
                        * sep_deficit
                        / max(1.0, args.prior_close_bright_bridge_close_sep)
                    )
                pair_future_support, pair_future_distinct, _left_future_name, _right_future_name = (
                    future_pair_support(left[0], right[0], future_cells, args)
                )
                shared_future_penalty_scale = 1.0
                if (
                    args.prior_pair_preserved_split_allow_shared_future
                    and not pair_future_distinct
                    and preserved_split_siblings(left[0], right[0])
                ):
                    # Close daughters can still be represented by one raw
                    # brightness candidate in the next frame. Keep the
                    # Ultrack-style distinct-link preference soft for BioFuse
                    # splitA/splitB hypotheses instead of hard-rejecting the
                    # earliest biologically plausible division frame.
                    shared_future_penalty_scale = max(
                        0.0,
                        args.prior_pair_preserved_shared_future_penalty_scale,
                    )
                    merged_support, _merged_distance, _merged_name = merged_future_pair_support(
                        left[0],
                        right[0],
                        future_cells,
                        args,
                    )
                    pair_future_support = max(pair_future_support, merged_support)
                pair_future_term = (
                    args.prior_pair_future_support_weight * pair_future_support
                    + (
                        args.prior_pair_future_distinct_bonus
                        if pair_future_distinct
                        else -args.prior_pair_future_same_penalty * shared_future_penalty_scale
                    )
                )
                if args.prior_pair_require_distinct_future_support and not pair_future_distinct:
                    pair_future_term -= (
                        args.prior_pair_future_required_penalty
                        * shared_future_penalty_scale
                    )
                shortfall_penalty = args.prior_soft_split_shortfall_penalty
                if (
                    args.prior_soft_root_split_shortfall_penalty > 0.0
                    and lineage_suffix_length(prior.name) <= 1
                ):
                    shortfall_penalty = args.prior_soft_root_split_shortfall_penalty
                pair_score = (
                    pair_rank_score(left)
                    + pair_rank_score(right)
                    + pair_future_term
                    + args.prior_soft_pair_separation_weight * min(sep, args.prior_soft_pair_separation_clip)
                    + args.prior_bridge_bonus_weight * bridge_dark_bonus
                    + args.prior_bridge_gap_weight * max(0.0, gap_drop)
                    - args.prior_bridge_penalty_weight * bridge_bright_penalty
                    - close_bright_bridge_penalty
                    - shortfall_penalty * sep_shortfall
                    - args.prior_soft_volume_ratio_penalty * volume_excess
                    - args.prior_soft_pair_parent_volume_ratio_penalty * parent_volume_excess
                    - args.prior_soft_division_penalty
                )
                if best_pair is None or pair_score > best_pair[0]:
                    best_pair = (pair_score, (left[0], right[0]), sep, valley_ratio, volume_ratio)

        if best_pair is not None and best_pair[0] > best_single_score + args.prior_soft_split_margin:
            selected = list(best_pair[1])
            selected.sort(key=lambda cell: (cell.y, cell.x, cell.z_scaled, cell.name))
            added_unresolved_duplicate = False
            if (
                args.prior_unresolved_close_split_duplicate
                and best_pair[2] <= args.prior_unresolved_close_split_max_sep
                and best_pair[3] <= args.prior_unresolved_close_split_max_valley
            ):
                # Some newly divided daughters are closer than the brightness
                # data can resolve into two clean centers. Keep an extra copy of
                # the best single-center evidence as a candidate hypothesis so
                # Cell Lumen does not permanently miss one daughter. This is a
                # recall-first option; downstream selection can remove the
                # duplicate if it is unsupported.
                duplicate_source = best_single[0]
                selected.append(
                    renamed_cell(
                        duplicate_source,
                        f"{duplicate_source.name}_unresolvedSplitDup",
                    )
                )
                added_unresolved_duplicate = True
            print(
                f"[FineShape UltrackStyle Select] parent={prior.name} choice=split "
                f"pair_score={best_pair[0]:.6g} single_score={best_single_score:.6g} "
                f"sep={best_pair[2]:.6g} bridge_valley={best_pair[3]:.6g} "
                f"volume_ratio={best_pair[4]:.6g} "
                f"future_support_weight={args.prior_future_support_weight:.6g} "
                f"unresolved_duplicate={int(added_unresolved_duplicate)}"
            )
            limit = args.prior_max_candidates_per_parent + (1 if added_unresolved_duplicate else 0)
            return selected[:limit]

        print(
            f"[FineShape UltrackStyle Select] parent={prior.name} choice=single "
            f"single_score={best_single_score:.6g} "
            f"best_pair_score={(best_pair[0] if best_pair else float('-inf')):.6g}"
        )
        selected = [best_single[0]]
        if getattr(args, "prior_preserve_single_z_alt_runner_up", False):
            runner_candidates: list[tuple[float, CellRow]] = []
            for item in deduped:
                if item[0].name == best_single[0].name:
                    continue
                if not z_column_alternative_siblings(best_single[0], item[0], args):
                    continue
                score_gap = best_single_score - rank_score(item)
                if score_gap <= args.prior_single_z_alt_runner_up_max_score_gap:
                    runner_candidates.append((score_gap, item[0]))
            runner_candidates.sort(key=lambda item: (item[0], item[1].y, item[1].x, item[1].z_scaled, item[1].name))
            if runner_candidates:
                # Recall-first approximation of an Ultrack node choice: a
                # single fused representative is not allowed to delete a close
                # z-layer alternative when the parent assignment is ambiguous.
                selected.append(
                    renamed_cell(
                        runner_candidates[0][1],
                        f"{runner_candidates[0][1].name}_singleZAlt",
                    )
                )
        if getattr(args, "prior_preserve_representative_alternatives", False):
            rep_alt_candidates: list[tuple[float, CellRow]] = []
            for item in deduped:
                if item[0].name == best_single[0].name:
                    continue
                if not representative_alternative_siblings(best_single[0], item[0], args):
                    continue
                score_gap = best_single_score - rank_score(item)
                if score_gap <= args.prior_representative_alternative_max_score_gap:
                    rep_alt_candidates.append((score_gap, item[0]))
            rep_alt_candidates.sort(
                key=lambda item: (item[0], item[1].y, item[1].x, item[1].z_scaled, item[1].name)
            )
            if rep_alt_candidates:
                # A BioFuse representative alternative is not a split accept.
                # It keeps one local raw-lobe hypothesis alive when largest,
                # medoid, and centroid choices disagree but have comparable
                # parent support.
                selected.append(
                    renamed_cell(
                        rep_alt_candidates[0][1],
                        f"{rep_alt_candidates[0][1].name}_repAltKeep",
                    )
                )
        return selected[: max(1, args.prior_max_candidates_per_parent + len(selected) - 1)]

    best_pair: tuple[float, tuple[CellRow, CellRow]] | None = None
    split_min_separation = args.prior_split_min_separation
    if (
        args.prior_root_parent_split_min_separation > 0.0
        and lineage_suffix_length(prior.name) <= 1
    ):
        split_min_separation = max(split_min_separation, args.prior_root_parent_split_min_separation)
    for i, left in enumerate(deduped):
        for right in deduped[i + 1:]:
            _, _, sep = scaled_distance(left[0], right[0])
            if sep < split_min_separation:
                continue
            if args.prior_split_max_volume_ratio > 0.0:
                left_volume = cell_volume_score(left[0])
                right_volume = cell_volume_score(right[0])
                volume_ratio = max(left_volume, right_volume) / max(1.0, min(left_volume, right_volume))
                if volume_ratio > args.prior_split_max_volume_ratio:
                    continue
            pair_score = left[3] + right[3] + args.prior_split_separation_bonus * sep
            if best_pair is None or pair_score > best_pair[0]:
                best_pair = (pair_score, (left[0], right[0]))

    if best_pair is not None:
        selected = list(best_pair[1])
        selected.sort(key=lambda cell: (cell.y, cell.x, cell.z_scaled, cell.name))
        return selected[: args.prior_max_candidates_per_parent]

    return [deduped[0][0]]


def apply_biological_prior(
    cells: list[CellRow],
    prior_cells: list[CellRow],
    args: argparse.Namespace,
    stack: np.ndarray | None = None,
    future_cells: list[CellRow] | None = None,
) -> list[CellRow]:
    if not cells or not prior_cells:
        return cells

    groups: dict[str, list[tuple[CellRow, float, float, float]]] = {}
    rejected_remote = 0
    dynamic_reach_rescued = 0
    rejected_parent_region = 0

    for cell in cells:
        prior_matches: list[tuple[CellRow, float, float, float]] = []
        for prior in prior_cells:
            _, _, distance = scaled_distance(cell, prior)
            norm = ellipsoid_norm_distance(cell, prior, args)
            effective_reach = effective_prior_reach_distance(prior, args)
            if distance <= effective_reach and norm <= args.prior_parent_region_max_norm:
                prior_matches.append((prior, distance, norm, effective_reach))
        if not prior_matches:
            rejected_remote += 1
            continue
        prior_matches.sort(key=lambda item: (item[1], item[2], item[0].name))
        if getattr(args, "prior_allow_multi_parent_candidates", False):
            best_distance = prior_matches[0][1]
            distance_limit = best_distance + args.prior_multi_parent_distance_slack
            prior_matches = [
                item for item in prior_matches if item[1] <= distance_limit
            ][: max(1, args.prior_multi_parent_max_assignments)]
        else:
            prior_matches = prior_matches[:1]

        for assignment_index, (prior, distance, norm, _effective_reach) in enumerate(prior_matches):
            if distance > args.prior_reach_distance:
                dynamic_reach_rescued += 1
            assigned_cell = cell
            if assignment_index > 0:
                # Ultrack-style graph approximation: a high-recall brightness
                # node can be a plausible continuation for more than one prior
                # cell in dense close-daughter regions. Keep a renamed copy in
                # the alternate parent group rather than committing to nearest
                # neighbor ownership before scoring.
                assigned_cell = renamed_cell(cell, f"{cell.name}_multiParent{assignment_index}")
            score = biological_prior_score(assigned_cell, prior, distance, norm, args)
            groups.setdefault(prior.name, []).append((assigned_cell, distance, norm, score))

    choice_records: list[tuple[CellRow, list[CellRow], list[tuple[CellRow, float, float, float]]]] = []
    multi_parent_groups = 0
    prior_by_name = {prior.name: prior for prior in prior_cells}
    for prior_name, items in groups.items():
        if len(items) > 1:
            multi_parent_groups += 1
        prior = prior_by_name[prior_name]
        chosen = choose_prior_group_candidates(
            prior,
            items,
            args,
            stack,
            future_cells,
        )
        choice_records.append((prior, chosen, items))

    target_count = getattr(args, "prior_global_target_count", 0)
    if target_count > 0:
        total_selected = sum(len(chosen) for _prior, chosen, _items in choice_records)
        excess = total_selected - target_count
        if excess > 0:
            # This is a small experimental analog of a global ILP complexity
            # budget. Parent groups can propose splits locally, but the final
            # frame is not allowed to grow beyond a data-driven target count.
            # We collapse the weakest split groups first instead of rejecting
            # candidates during BioFuse.
            def item_rank(
                item: tuple[CellRow, float, float, float],
            ) -> float:
                if args.prior_ultrack_style_selection:
                    return ultrack_style_candidate_score(item, args, future_cells)
                return item[3]

            collapse_options: list[tuple[float, int, CellRow]] = []
            for index, (_prior, chosen, items) in enumerate(choice_records):
                if len(chosen) <= 1:
                    continue
                chosen_names = {cell.name for cell in chosen}
                chosen_scores = [
                    item_rank(item)
                    for item in items
                    if item[0].name in chosen_names
                ]
                weakest_chosen_score = min(chosen_scores) if chosen_scores else 0.0
                best_single = max(items, key=item_rank)[0]
                collapse_options.append((weakest_chosen_score, index, best_single))
            collapse_options.sort(key=lambda option: option[0])
            collapsed_groups = 0
            for _score, index, best_single in collapse_options:
                if excess <= 0:
                    break
                prior, chosen, items = choice_records[index]
                removed = len(chosen) - 1
                if removed <= 0:
                    continue
                choice_records[index] = (prior, [best_single], items)
                excess -= removed
                collapsed_groups += 1
            if collapsed_groups:
                print(
                    f"[FineShape Biological Prior GlobalBudget] "
                    f"target={target_count} initial={total_selected} "
                    f"collapsed_groups={collapsed_groups} "
                    f"remaining={sum(len(chosen) for _prior, chosen, _items in choice_records)}"
                )

    selected: list[CellRow] = []
    for prior, chosen, _items in choice_records:
        if args.preserve_rolling_parent_names:
            # Optional rolling-graph experiment: preserve parent identity so the
            # next frame can reason over lineage hypotheses. It is default-off
            # because the established center-audit baseline used raw Cell Lumen
            # candidate names and must stay reproducible.
            if len(chosen) <= 1:
                selected.extend(renamed_cell(cell, prior.name) for cell in chosen)
            else:
                ordered = sorted(chosen, key=lambda cell: (cell.y, cell.x, cell.z_scaled, cell.name))
                selected.extend(
                    renamed_cell(cell, daughter_name(prior.name, index))
                    for index, cell in enumerate(ordered)
                )
        else:
            selected.extend(chosen)

    selected = collapse_coordinate_copies(selected, args, stack, future_cells)
    selected.sort(key=lambda cell: (cell.y, cell.x, cell.z_scaled, cell.name))
    print(
        f"[FineShape Biological Prior] before={len(cells)} after={len(selected)} "
        f"prior_cells={len(prior_cells)} parent_groups={len(groups)} "
        f"multi_parent_groups={multi_parent_groups} "
        f"rejected_remote={rejected_remote} "
        f"dynamic_reach_rescued={dynamic_reach_rescued} "
        f"rejected_parent_region={rejected_parent_region} "
        f"global_target={target_count} "
        f"reach_distance={args.prior_reach_distance} "
        f"dynamic_reach={int(bool(getattr(args, 'prior_dynamic_reach_from_radius', False)))} "
        f"parent_region_max_norm={args.prior_parent_region_max_norm} "
        f"max_candidates_per_parent={args.prior_max_candidates_per_parent}"
    )
    return selected


def fused_representative(
    members: list[CellRow],
    args: argparse.Namespace,
    suffix: str,
) -> CellRow:
    """Create one review candidate for a physically overlapping local group.

    The original experimental BioFuse kept the largest bright lobe. That is
    useful for removing tiny duplicates, but it can bias the center toward an
    edge lobe. The centroid and medoid modes are optional Ultrack-inspired
    alternatives: keep the hypothesis compact, but avoid prematurely committing
    to the largest fragment before temporal/parent evidence has a chance to
    score it.
    """
    if len(members) == 1:
        return members[0]

    largest = max(members, key=lambda cell: (cell_volume_score(cell), -cell.z_scaled, cell.name))
    mode = getattr(args, "fuse_representative_mode", "largest")
    if mode == "largest":
        center_source = largest
        x = largest.x
        y = largest.y
        z_scaled = largest.z_scaled
    elif mode == "medoid":
        center_source = min(
            members,
            key=lambda cell: sum(scaled_distance(cell, other)[2] for other in members),
        )
        x = center_source.x
        y = center_source.y
        z_scaled = center_source.z_scaled
    else:
        center_source = largest
        x = float(sum(cell.x for cell in members) / len(members))
        y = float(sum(cell.y for cell in members) / len(members))
        z_scaled = float(sum(cell.z_scaled for cell in members) / len(members))

    return CellRow(
        frame=center_source.frame,
        name=f"{center_source.name}_{suffix}{len(members)}",
        x=x,
        y=y,
        z_scaled=z_scaled,
        rx=max(cell.rx for cell in members),
        ry=max(cell.ry for cell in members),
        rz_scaled=max(cell.rz_scaled for cell in members),
        z_mode_used=center_source.z_mode_used + f"/{suffix}_{len(members)}_{mode}",
    )


def split_members_by_seed_pair(
    members: list[CellRow],
    left_seed: CellRow,
    right_seed: CellRow,
) -> tuple[list[CellRow], list[CellRow]]:
    left_cluster: list[CellRow] = []
    right_cluster: list[CellRow] = []
    for member in members:
        left_distance = scaled_distance(member, left_seed)[2]
        right_distance = scaled_distance(member, right_seed)[2]
        if left_distance <= right_distance:
            left_cluster.append(member)
        else:
            right_cluster.append(member)
    return left_cluster, right_cluster


def split_large_z_span_fusion_group(
    members: list[CellRow],
    args: argparse.Namespace,
) -> list[list[CellRow]]:
    if (
        not getattr(args, "fuse_split_large_z_span_groups", False)
        or len(members) <= 1
    ):
        return [members]
    z_values = [cell.z_scaled for cell in members]
    if max(z_values) - min(z_values) <= args.fuse_max_group_z_span:
        return [members]

    ordered = sorted(members, key=lambda cell: (cell.z_scaled, cell.y, cell.x, cell.name))
    groups: list[list[CellRow]] = [[ordered[0]]]
    for cell in ordered[1:]:
        previous = groups[-1][-1]
        group_start_z = groups[-1][0].z_scaled
        if (
            cell.z_scaled - previous.z_scaled >= args.fuse_z_split_gap
            or cell.z_scaled - group_start_z > args.fuse_max_group_z_span
        ):
            # Default-off safeguard against union-find chain merging. A long
            # ladder of z-adjacent duplicate peaks can have no single large
            # gap, but still span too much z as a group. Keep each chunk local
            # so the later parent-aware selector can choose the biological
            # continuation without an averaged-away center.
            groups.append([cell])
        else:
            groups[-1].append(cell)
    return groups


def z_column_alternatives_for_representative(
    members: list[CellRow],
    representative: CellRow,
    args: argparse.Namespace,
) -> list[CellRow]:
    if not getattr(args, "fuse_preserve_z_column_alternatives", False):
        return []
    max_extra = max(0, int(getattr(args, "fuse_z_column_alternative_max_extra_per_group", 1)))
    if max_extra == 0 or len(members) <= 1:
        return []
    candidates: list[tuple[float, CellRow]] = []
    for member in members:
        if member.name == representative.name:
            continue
        lateral, dz, _distance = scaled_distance(member, representative)
        if (
            lateral <= args.fuse_z_column_alternative_lateral
            and dz >= args.fuse_z_column_alternative_min_dz
        ):
            candidates.append((dz, member))
    candidates.sort(key=lambda item: (-item[0], item[1].y, item[1].x, item[1].z_scaled, item[1].name))
    # Recall-first Cell Lumen option: z-column duplicate groups often contain
    # one true low/high daughter center plus one internal bright layer. Keep a
    # tiny number of raw seed alternatives so later temporal/global selection
    # can decide, instead of permanently deleting the true center at BioFuse.
    return [
        renamed_cell(member, f"{member.name}_zAlt")
        for _dz, member in candidates[:max_extra]
    ]


def representative_alternatives_for_fused_group(
    members: list[CellRow],
    representative: CellRow,
    args: argparse.Namespace,
) -> list[CellRow]:
    if not getattr(args, "fuse_preserve_representative_alternatives", False):
        return []
    max_extra = max(
        0,
        int(getattr(args, "fuse_representative_alternative_max_extra_per_group", 1)),
    )
    if max_extra == 0 or len(members) <= 1:
        return []

    min_distance = max(0.0, getattr(args, "fuse_representative_alternative_min_distance", 8.0))
    candidates: list[tuple[float, float, CellRow]] = []
    for member in members:
        if member.name == representative.name:
            continue
        _lateral, _dz, distance = scaled_distance(member, representative)
        if distance < min_distance:
            continue
        # BioFuse representative choice is intentionally experimental. Keeping
        # one clearly separated raw lobe lets the parent-aware rolling selector
        # compare largest/medoid style hypotheses later instead of making this
        # local duplicate merge a hard irreversible decision.
        candidates.append((-cell_volume_score(member), -distance, member))
    candidates.sort(key=lambda item: (item[0], item[1], item[2].y, item[2].x, item[2].z_scaled, item[2].name))
    return [
        renamed_cell(member, f"{member.name}_repAlt")
        for _volume, _distance, member in candidates[:max_extra]
    ]


def append_fused_member_group(
    fused: list[CellRow],
    members: list[CellRow],
    args: argparse.Namespace,
    suffix: str,
) -> None:
    representative = fused_representative(members, args, suffix)
    fused.append(representative)
    fused.extend(representative_alternatives_for_fused_group(members, representative, args))
    fused.extend(z_column_alternatives_for_representative(members, representative, args))


def preserved_split_pair_for_fusion_group(
    members: list[CellRow],
    args: argparse.Namespace,
    stack: np.ndarray | None,
    future_cells: list[CellRow] | None = None,
) -> tuple[CellRow, CellRow] | None:
    """Return a plausible two-peak hypothesis that should survive BioFuse.

    This is deliberately not a final split acceptor. It only prevents early
    duplicate fusion from deleting close daughter alternatives before the
    later parent-aware selector compares split versus continuation.
    """
    if not getattr(args, "fuse_preserve_split_hypotheses", False):
        return None
    if len(members) < 3 or stack is None:
        return None

    min_separation = max(0.0, getattr(args, "fuse_preserve_pair_min_separation", 34.0))
    max_valley = getattr(args, "fuse_preserve_pair_max_valley_ratio", 0.82)
    best: tuple[float, CellRow, CellRow] | None = None
    for i, left in enumerate(members):
        for right in members[i + 1:]:
            _, _, separation = scaled_distance(left, right)
            if separation < min_separation:
                continue
            valley_ratio, gap_drop = bridge_slab_valley_ratio(stack, left, right, args)
            if valley_ratio > max_valley:
                continue
            if getattr(args, "fuse_preserve_require_future_support", False):
                left_future, _left_future_distance, _left_future_name = future_candidate_support(
                    left,
                    future_cells,
                    args,
                )
                right_future, _right_future_distance, _right_future_name = future_candidate_support(
                    right,
                    future_cells,
                    args,
                )
                min_future = getattr(args, "fuse_preserve_min_future_support", 5.0)
                if min(left_future, right_future) < min_future:
                    continue
            if getattr(args, "fuse_preserve_require_distinct_future_support", False):
                _pair_future, pair_future_distinct, _left_future_name, _right_future_name = (
                    future_pair_support(left, right, future_cells, args)
                )
                if not pair_future_distinct:
                    continue
            support = cell_support_radius_score(left) + cell_support_radius_score(right)
            score = (
                separation
                + 0.15 * support
                + getattr(args, "prior_bridge_bonus_weight", 90.0)
                * max(0.0, max_valley - valley_ratio)
                + getattr(args, "prior_bridge_gap_weight", 0.03) * max(0.0, gap_drop)
            )
            if best is None or score > best[0]:
                best = (score, left, right)
    if best is None:
        return None
    return best[1], best[2]


def fuse_biological_duplicates(
    cells: list[CellRow],
    args: argparse.Namespace,
    stack: np.ndarray | None = None,
    future_cells: list[CellRow] | None = None,
) -> list[CellRow]:
    if not cells:
        return cells

    parent = list(range(len(cells)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for i, left in enumerate(cells):
        for j in range(i + 1, len(cells)):
            right = cells[j]
            lateral, dz, distance = scaled_distance(left, right)
            same_column_duplicate = (
                lateral <= args.fuse_z_column_lateral
                and dz <= args.fuse_z_column_max_dz
                and distance <= args.fuse_z_column_max_distance
            )
            physically_impossible_close = distance <= args.fuse_min_scaled_distance
            local_overlap_duplicate = (
                lateral <= args.fuse_near_lateral
                and distance <= args.fuse_near_scaled_distance
            )
            if same_column_duplicate or physically_impossible_close or local_overlap_duplicate:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for index in range(len(cells)):
        groups.setdefault(find(index), []).append(index)

    fused: list[CellRow] = []
    merged_group_count = 0
    preserved_split_group_count = 0
    z_span_split_group_count = 0
    for group_indices in groups.values():
        members = [cells[index] for index in group_indices]
        if len(members) == 1:
            fused.append(members[0])
            continue
        merged_group_count += 1
        member_groups = split_large_z_span_fusion_group(members, args)
        if len(member_groups) > 1:
            z_span_split_group_count += 1
        preserved_pair = preserved_split_pair_for_fusion_group(members, args, stack, future_cells)
        if preserved_pair is not None:
            left_cluster, right_cluster = split_members_by_seed_pair(
                members,
                preserved_pair[0],
                preserved_pair[1],
            )
            if left_cluster and right_cluster:
                preserved_split_group_count += 1
                for sub_cluster in split_large_z_span_fusion_group(left_cluster, args):
                    append_fused_member_group(fused, sub_cluster, args, "biofused_splitA")
                for sub_cluster in split_large_z_span_fusion_group(right_cluster, args):
                    append_fused_member_group(fused, sub_cluster, args, "biofused_splitB")
                continue
        for member_group in member_groups:
            append_fused_member_group(fused, member_group, args, "biofused")

    fused.sort(key=lambda cell: (cell.y, cell.x, cell.z_scaled, cell.name))
    print(
        f"[FineShape BioFuse] before={len(cells)} after={len(fused)} "
        f"merged_groups={merged_group_count} "
        f"preserved_split_groups={preserved_split_group_count} "
        f"z_span_split_groups={z_span_split_group_count} "
        f"representative_mode={getattr(args, 'fuse_representative_mode', 'largest')} "
        f"z_column_lateral={args.fuse_z_column_lateral} "
        f"z_column_max_dz={args.fuse_z_column_max_dz} "
        f"min_scaled_distance={args.fuse_min_scaled_distance}"
    )
    return fused


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
    cells = load_cells(
        args.cells,
        frame_name if args.filter_frame else None,
        args.z_mode,
        args.radius_z_mode,
        args.z_scaling,
        z_count,
        args.unique_initial_celltype_names,
    )
    if not cells:
        raise RuntimeError(f"No cells loaded from {args.cells}")
    input_cell_count = len(cells)
    future_cells: list[CellRow] = []
    if args.future_candidate_cells:
        future_cells = load_cells(
            args.future_candidate_cells,
            args.future_candidate_frame_name,
            args.future_candidate_z_mode,
            args.future_candidate_radius_z_mode,
            args.z_scaling,
            z_count,
            args.unique_initial_celltype_names,
        )
    if args.fuse_biological_duplicates:
        cells = fuse_biological_duplicates(cells, args, stack, future_cells)
    if args.biological_prior_cells:
        prior_cells = load_cells(
            args.biological_prior_cells,
            args.biological_prior_frame_name,
            args.biological_prior_z_mode,
            args.biological_prior_radius_z_mode,
            args.z_scaling,
            z_count,
            args.unique_initial_celltype_names,
        )
        if not prior_cells:
            raise RuntimeError(f"No prior cells loaded from {args.biological_prior_cells}")
        cells = apply_biological_prior(cells, prior_cells, args, stack, future_cells)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / f"{args.image.stem}_fine_shape_summary.csv"
    runs_path = args.output_dir / f"{args.image.stem}_fine_shape_runs.csv"
    label_volume = None
    claim_score = None
    if args.label_output:
        label_volume = np.zeros(stack.shape, dtype=np.uint16)
        claim_score = np.full(stack.shape, -np.inf, dtype=np.float32)
    neighbors = neighbor_offsets(args.connectivity)

    runs_handle = None
    if not args.skip_runs_csv:
        runs_handle = runs_path.open("w", newline="")
    with summary_path.open("w", newline="") as summary_f:
        summary = summary_f
        runs = runs_handle
        summary.write(
            "frame,cell_name,center_x,center_y,center_z_scaled,radius_scaled,"
            "radius_x,radius_y,radius_z_scaled,z_mode,threshold,seed_value,voxels,mean_intensity,truncated,"
            "bbox_x0,bbox_y0,bbox_z0,bbox_x1,bbox_y1,bbox_z1\n"
        )
        if runs is not None:
            runs.write("frame,cell_name,z,y,x0,x1\n")

        exported = 0
        total_voxels = 0
        for cell in cells:
            if args.window_mode == "sphere":
                base_radius = max(cell.rx, cell.ry, cell.rz_scaled, args.min_radius)
                rx = ry = rz_scaled = min(
                    args.max_radius,
                    max(args.min_radius, base_radius * args.radius_scale),
                )
            else:
                rx = min(args.max_radius, max(args.min_radius, cell.rx * args.radius_scale))
                ry = min(args.max_radius, max(args.min_radius, cell.ry * args.radius_scale))
                rz_scaled = min(
                    args.max_radius,
                    max(args.min_radius, cell.rz_scaled * args.radius_scale),
                )

            x0 = max(0, int(np.floor(cell.x - rx)))
            x1 = min(cols - 1, int(np.ceil(cell.x + rx)))
            y0 = max(0, int(np.floor(cell.y - ry)))
            y1 = min(rows - 1, int(np.ceil(cell.y + ry)))
            z0 = max(0, int(np.floor((cell.z_scaled - rz_scaled) / args.z_scaling)))
            z1 = min(z_count - 1, int(np.ceil((cell.z_scaled + rz_scaled) / args.z_scaling)))
            if x0 > x1 or y0 > y1 or z0 > z1:
                continue

            def inside_window(x: int, y: int, z: int) -> bool:
                dx = (x - cell.x) / max(rx, 1.0e-6)
                dy = (y - cell.y) / max(ry, 1.0e-6)
                dz = (z * args.z_scaling - cell.z_scaled) / max(rz_scaled, 1.0e-6)
                return dx * dx + dy * dy + dz * dz <= 1.0

            def window_norm(x: int, y: int, z: int) -> float:
                dx = (x - cell.x) / max(rx, 1.0e-6)
                dy = (y - cell.y) / max(ry, 1.0e-6)
                dz = (z * args.z_scaling - cell.z_scaled) / max(rz_scaled, 1.0e-6)
                return float(np.sqrt(dx * dx + dy * dy + dz * dz))

            values: list[float] = []
            for z in range(z0, z1 + 1):
                for y in range(y0, y1 + 1):
                    for x in range(x0, x1 + 1):
                        if inside_window(x, y, z):
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
                        if not inside_window(x, y, z) or dist_sq > seed_radius_sq:
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

                for dx, dy, dz in neighbors:
                    nx, ny, nz = x + dx, y + dy, z + dz
                    if nx < x0 or nx > x1 or ny < y0 or ny > y1 or nz < z0 or nz > z1:
                        continue
                    if visited[nz - z0, ny - y0, nx - x0]:
                        continue
                    if not inside_window(nx, ny, nz):
                        continue
                    visited[nz - z0, ny - y0, nx - x0] = True
                    q.append((nx, ny, nz))

            if voxels <= 0:
                continue

            mean_intensity = intensity_sum / voxels
            if (
                args.min_mask_mean_intensity >= 0.0
                and mean_intensity < args.min_mask_mean_intensity
            ):
                print(
                    f"[FineShape Skip LowMean] cell={cell.name} "
                    f"mean_intensity={mean_intensity:.6g} "
                    f"min_mask_mean_intensity={args.min_mask_mean_intensity}"
                )
                continue

            output_label = exported + 1
            if label_volume is not None and claim_score is not None:
                contrast_denominator = max(1.0e-6, seed_value - threshold)
                for z in range(z0, z1 + 1):
                    for y in range(y0, y1 + 1):
                        for x in range(x0, x1 + 1):
                            if not mask[z - z0, y - y0, x - x0]:
                                continue
                            intensity_score = (float(stack[z, y, x]) - threshold) / contrast_denominator
                            score = intensity_score - args.claim_distance_penalty * window_norm(x, y, z)
                            if score > float(claim_score[z, y, x]):
                                claim_score[z, y, x] = score
                                label_volume[z, y, x] = output_label

            for z in range(z0, z1 + 1):
                for y in range(y0, y1 + 1):
                    run_start = -1
                    for x in range(x0, x1 + 1):
                        in_mask = bool(mask[z - z0, y - y0, x - x0])
                        if in_mask and run_start < 0:
                            run_start = x
                        if (not in_mask or x == x1) and run_start >= 0:
                            run_end = x if in_mask and x == x1 else x - 1
                            if runs is not None:
                                runs.write(
                                    f"{csv_cell(args.image.stem)},{csv_cell(cell.name)},{z},{y},{run_start},{run_end}\n"
                                )
                            run_start = -1

            summary.write(
                f"{csv_cell(args.image.stem)},{csv_cell(cell.name)},"
                f"{cell.x:.9g},{cell.y:.9g},{cell.z_scaled:.9g},{max(rx, ry, rz_scaled):.9g},"
                f"{rx:.9g},{ry:.9g},{rz_scaled:.9g},{csv_cell(cell.z_mode_used)},"
                f"{threshold:.9g},{seed_value:.9g},{voxels},{mean_intensity:.9g},"
                f"{1 if truncated else 0},{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},{bbox[4]},{bbox[5]}\n"
            )
            exported += 1
            total_voxels += voxels
    if runs_handle is not None:
        runs_handle.close()

    if label_volume is not None and args.label_output:
        output_volume = stretch_z(label_volume, max(1, args.label_z_stretch))
        write_tiff_stack(args.label_output, output_volume)
        print(
            f"[FineShape Label TIFF] output={args.label_output} "
            f"shape={output_volume.shape} raw_shape={label_volume.shape} "
            f"labels={int(label_volume.max())} "
            f"nonzero={int(np.count_nonzero(output_volume))} "
            f"label_z_stretch={args.label_z_stretch}"
        )

    if args.stretched_image_output:
        image_volume = stretch_z(np.clip(stack, 0, 255).astype(np.uint8), max(1, args.label_z_stretch))
        write_tiff_stack(args.stretched_image_output, image_volume)
        print(
            f"[FineShape Stretched Image TIFF] output={args.stretched_image_output} "
            f"shape={image_volume.shape} z_stretch={args.label_z_stretch}"
        )

    print(
        f"[FineShape CSV Export] image={args.image} input_cells={input_cell_count} "
        f"cells={len(cells)} future_cells={len(future_cells)} exported={exported} "
        f"total_voxels={total_voxels} summary={summary_path} "
        f"runs={runs_path if not args.skip_runs_csv else '<skipped>'} "
        f"window_mode={args.window_mode} z_mode={args.z_mode}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export experimental Cell Lumen fine shape RLE masks from an existing cell CSV."
    )
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--cells", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--frame-name", help="Frame name to match in the CSV, default is image filename.")
    parser.add_argument("--filter-frame", action="store_true", help="Only use rows matching the frame name.")
    parser.add_argument("--z-scaling", type=float, default=7.0)
    parser.add_argument(
        "--unique-initial-celltype-names",
        action="store_true",
        help="Experimental rolling option: convert repeated initial cellType rows into unique root-like names.",
    )
    parser.add_argument(
        "--preserve-rolling-parent-names",
        action="store_true",
        help="Experimental rolling option: rename selected candidates as parent continuations or daughters.",
    )
    parser.add_argument(
        "--z-mode",
        choices=["auto", "raw", "scaled"],
        default="auto",
        help="How to interpret CSV z coordinates. auto treats Cell Genesis rows with voxel_size_z as raw slice z.",
    )
    parser.add_argument(
        "--radius-z-mode",
        choices=["auto", "raw", "scaled"],
        default="auto",
        help="How to interpret CSV z radii. auto converts small cRadius values with voxel_size_z to scaled units.",
    )
    parser.add_argument(
        "--window-mode",
        choices=["ellipsoid", "sphere"],
        default="ellipsoid",
        help="Local search window. The output mask is still an irregular brightness component.",
    )
    parser.add_argument(
        "--connectivity",
        choices=[6, 26],
        default=26,
        type=int,
        help="3D flood fill connectivity for the brightness component.",
    )
    parser.add_argument("--radius-scale", type=float, default=1.25)
    parser.add_argument("--min-radius", type=float, default=8.0)
    parser.add_argument("--max-radius", type=float, default=64.0)
    parser.add_argument("--threshold-quantile", type=float, default=0.55)
    parser.add_argument("--threshold-floor", type=float, default=0.08)
    parser.add_argument("--seed-search-radius", type=float, default=10.0)
    parser.add_argument("--seed-fallback-fraction", type=float, default=0.85)
    parser.add_argument("--max-voxels-per-cell", type=int, default=200000)
    parser.add_argument(
        "--skip-runs-csv",
        action="store_true",
        help=(
            "Skip the large per-voxel/RLE fine_shape_runs.csv cache. The "
            "summary CSV, logs, and optional label TIFF are still written. "
            "This changes only review artifacts, not candidate selection."
        ),
    )
    parser.add_argument(
        "--fuse-biological-duplicates",
        action="store_true",
        help="Experimental review option that fuses physically impossible duplicate Cell Lumen candidates before mask export.",
    )
    parser.add_argument("--fuse-z-column-lateral", type=float, default=4.0)
    parser.add_argument("--fuse-z-column-max-dz", type=float, default=24.0)
    parser.add_argument("--fuse-z-column-max-distance", type=float, default=26.0)
    parser.add_argument("--fuse-min-scaled-distance", type=float, default=18.0)
    parser.add_argument("--fuse-near-lateral", type=float, default=8.0)
    parser.add_argument("--fuse-near-scaled-distance", type=float, default=18.5)
    parser.add_argument(
        "--fuse-split-large-z-span-groups",
        action="store_true",
        help=(
            "Default-off safeguard: split BioFuse groups whose z span is too "
            "large before choosing representatives, preventing transitive "
            "z-column chains from averaging away low-z centers."
        ),
    )
    parser.add_argument("--fuse-max-group-z-span", type=float, default=45.0)
    parser.add_argument("--fuse-z-split-gap", type=float, default=18.0)
    parser.add_argument(
        "--fuse-representative-mode",
        choices=["largest", "centroid", "medoid"],
        default="largest",
        help="Experimental BioFuse representative choice for a duplicate group.",
    )
    parser.add_argument(
        "--fuse-preserve-split-hypotheses",
        action="store_true",
        help=(
            "Experimental option: when a duplicate group contains two separated "
            "3D peaks with bridge evidence, keep two candidates for later "
            "parent-aware split scoring instead of collapsing them to one."
        ),
    )
    parser.add_argument("--fuse-preserve-pair-min-separation", type=float, default=34.0)
    parser.add_argument("--fuse-preserve-pair-max-valley-ratio", type=float, default=0.82)
    parser.add_argument(
        "--fuse-preserve-z-column-alternatives",
        action="store_true",
        help=(
            "Default-off recall option: keep a small number of raw seed "
            "alternatives from z-column duplicate groups instead of trusting "
            "one fused representative to choose the correct z layer."
        ),
    )
    parser.add_argument("--fuse-z-column-alternative-min-dz", type=float, default=10.0)
    parser.add_argument("--fuse-z-column-alternative-lateral", type=float, default=16.0)
    parser.add_argument("--fuse-z-column-alternative-max-extra-per-group", type=int, default=1)
    parser.add_argument(
        "--fuse-preserve-representative-alternatives",
        action="store_true",
        help=(
            "Default-off recall option: when BioFuse collapses a duplicate "
            "group, also keep one separated raw lobe so representative mode "
            "selection is not an irreversible local decision."
        ),
    )
    parser.add_argument("--fuse-representative-alternative-min-distance", type=float, default=8.0)
    parser.add_argument("--fuse-representative-alternative-max-extra-per-group", type=int, default=1)
    parser.add_argument(
        "--fuse-preserve-require-future-support",
        action="store_true",
        help="Require both preserved duplicate-group split peaks to have next-frame raw candidate support.",
    )
    parser.add_argument(
        "--fuse-preserve-require-distinct-future-support",
        action="store_true",
        help="Require preserved duplicate-group split peaks to link to two distinct next-frame raw candidates.",
    )
    parser.add_argument("--fuse-preserve-min-future-support", type=float, default=5.0)
    parser.add_argument(
        "--biological-prior-cells",
        type=Path,
        help="Optional previous-frame cells or initial CSV used as an experimental biological prior.",
    )
    parser.add_argument(
        "--biological-prior-frame-name",
        help="Frame name to read from --biological-prior-cells, such as t017.tif.",
    )
    parser.add_argument(
        "--biological-prior-z-mode",
        choices=["auto", "raw", "scaled"],
        default="auto",
        help="How to interpret prior CSV z coordinates.",
    )
    parser.add_argument(
        "--biological-prior-radius-z-mode",
        choices=["auto", "raw", "scaled"],
        default="auto",
        help="How to interpret prior CSV z radii.",
    )
    parser.add_argument(
        "--prior-reach-distance",
        type=float,
        default=50.0,
        help="Reject candidates farther than this scaled 3D distance from every previous-frame prior cell.",
    )
    parser.add_argument(
        "--prior-dynamic-reach-from-radius",
        action="store_true",
        help=(
            "Experimental option: allow large previous-frame parents to use a "
            "larger reach distance derived from their radius, instead of the "
            "fixed --prior-reach-distance only."
        ),
    )
    parser.add_argument(
        "--prior-dynamic-reach-radius-scale",
        type=float,
        default=1.25,
        help="Radius multiplier used when --prior-dynamic-reach-from-radius is enabled.",
    )
    parser.add_argument(
        "--prior-dynamic-reach-margin",
        type=float,
        default=0.0,
        help="Additive scaled-pixel margin for dynamic parent reach.",
    )
    parser.add_argument(
        "--prior-parent-region-scale",
        type=float,
        default=1.6,
        help="Scale previous-frame radii when building the parent occupancy region.",
    )
    parser.add_argument(
        "--prior-parent-region-margin",
        type=float,
        default=8.0,
        help="Additive margin in scaled pixels for the parent occupancy region.",
    )
    parser.add_argument(
        "--prior-parent-region-max-norm",
        type=float,
        default=1.0,
        help="Maximum ellipsoid-normalized distance allowed inside a previous-frame parent region.",
    )
    parser.add_argument(
        "--prior-allow-multi-parent-candidates",
        action="store_true",
        help=(
            "Default-off Ultrack-style recall option: assign one brightness "
            "candidate to multiple plausible previous-frame parents when dense "
            "close daughters make nearest-parent ownership ambiguous."
        ),
    )
    parser.add_argument("--prior-multi-parent-max-assignments", type=int, default=2)
    parser.add_argument("--prior-multi-parent-distance-slack", type=float, default=12.0)
    parser.add_argument(
        "--prior-max-candidates-per-parent",
        type=int,
        default=2,
        help="Maximum candidates kept for each previous-frame parent, allowing possible split hypotheses.",
    )
    parser.add_argument(
        "--prior-global-target-count",
        type=int,
        default=0,
        help=(
            "Experimental default-off global count budget. When positive, "
            "parent groups may propose splits locally, but weakest split groups "
            "are collapsed back to one candidate until the frame count is at or "
            "below this target."
        ),
    )
    parser.add_argument(
        "--prior-split-min-separation",
        type=float,
        default=30.0,
        help="Minimum scaled 3D separation needed to keep two candidates for one previous-frame parent.",
    )
    parser.add_argument(
        "--prior-root-parent-split-min-separation",
        type=float,
        default=-1.0,
        help="Optional stricter split separation for root-like parents with a one-character lineage suffix.",
    )
    parser.add_argument(
        "--prior-same-parent-duplicate-distance",
        type=float,
        default=26.0,
        help="Within one parent group, candidates closer than this are treated as duplicate centers.",
    )
    parser.add_argument(
        "--prior-preserve-z-column-alternatives",
        action="store_true",
        help=(
            "Default-off recall option: do not deduplicate close candidates "
            "when they are mainly separated along z, preserving low/high z "
            "hypotheses for rolling Cell Lumen selection."
        ),
    )
    parser.add_argument("--prior-z-column-alternative-min-dz", type=float, default=10.0)
    parser.add_argument("--prior-z-column-alternative-lateral", type=float, default=16.0)
    parser.add_argument(
        "--prior-preserve-single-z-alt-runner-up",
        action="store_true",
        help=(
            "Default-off recall option: when a parent chooses a single center, "
            "also keep the best close z-layer alternative if it is not much "
            "weaker. This prevents one representative from deleting the true "
            "z layer in dense rolling runs."
        ),
    )
    parser.add_argument("--prior-single-z-alt-runner-up-max-score-gap", type=float, default=35.0)
    parser.add_argument(
        "--prior-preserve-representative-alternatives",
        action="store_true",
        help=(
            "Default-off recall option: when BioFuse emitted a _repAlt raw "
            "lobe, do not let same-parent duplicate cleanup make the local "
            "representative mode an irreversible decision."
        ),
    )
    parser.add_argument("--prior-representative-alternative-min-distance", type=float, default=8.0)
    parser.add_argument("--prior-representative-alternative-max-score-gap", type=float, default=35.0)
    parser.add_argument(
        "--prior-duplicate-radius-fraction",
        type=float,
        default=0.80,
        help="Duplicate limit as a fraction of the two candidate support radii sum.",
    )
    parser.add_argument("--prior-distance-score-weight", type=float, default=15.0)
    parser.add_argument("--prior-norm-score-weight", type=float, default=250.0)
    parser.add_argument(
        "--prior-split-separation-bonus",
        type=float,
        default=25.0,
        help="Small bonus for separated two-candidate parent groups so close real split hypotheses can survive.",
    )
    parser.add_argument(
        "--prior-split-max-volume-ratio",
        type=float,
        default=-1.0,
        help="When positive, two candidates for one prior parent can both survive only if their support-volume ratio is below this limit.",
    )
    parser.add_argument(
        "--prior-ultrack-style-selection",
        action="store_true",
        help="Experimental option: compare continuation and split hypotheses with a soft objective instead of selecting every valid pair.",
    )
    parser.add_argument("--prior-soft-support-weight", type=float, default=4.0)
    parser.add_argument("--prior-soft-support-clip", type=float, default=60.0)
    parser.add_argument("--prior-soft-distance-weight", type=float, default=0.35)
    parser.add_argument("--prior-soft-norm-weight", type=float, default=5.0)
    parser.add_argument("--prior-soft-pair-separation-weight", type=float, default=0.15)
    parser.add_argument("--prior-soft-pair-separation-clip", type=float, default=55.0)
    parser.add_argument("--prior-soft-division-penalty", type=float, default=115.0)
    parser.add_argument("--prior-soft-split-margin", type=float, default=0.0)
    parser.add_argument("--prior-soft-split-shortfall-penalty", type=float, default=3.0)
    parser.add_argument("--prior-soft-root-split-shortfall-penalty", type=float, default=0.0)
    parser.add_argument("--prior-soft-volume-ratio-penalty", type=float, default=18.0)
    parser.add_argument(
        "--prior-unresolved-close-split-duplicate",
        action="store_true",
        help=(
            "Default-off recall option: for a close, low-valley split whose two "
            "daughters may still share one brightness center, keep one extra "
            "copy of the best single-center evidence as a candidate hypothesis."
        ),
    )
    parser.add_argument("--prior-unresolved-close-split-max-sep", type=float, default=30.0)
    parser.add_argument("--prior-unresolved-close-split-max-valley", type=float, default=0.4)
    parser.add_argument(
        "--prior-duplicate-single-on-future-split-pressure",
        action="store_true",
        help=(
            "Default-off recall option: if next-frame raw Cell Lumen candidates "
            "show two separated future witnesses for one current single center, "
            "duplicate the current hypothesis as an unresolved split placeholder."
        ),
    )
    parser.add_argument("--prior-single-future-split-duplicate-min-pressure", type=float, default=5.0)
    parser.add_argument(
        "--prior-collapse-coordinate-copies",
        action="store_true",
        help=(
            "Default-off cleanup for experimental rescue copies. Collapse rows "
            "with the same 3D center so multi-parent or z-alt aliases do not "
            "become persistent extra cells. This does not merge nearby but "
            "distinct centers."
        ),
    )
    parser.add_argument("--prior-coordinate-copy-epsilon", type=float, default=0.001)
    parser.add_argument(
        "--prior-coordinate-copy-keep-unresolved",
        action="store_true",
        help=(
            "When coordinate-copy collapse is enabled, keep one unresolved "
            "close-split duplicate as a pending placeholder for under-resolved "
            "close daughters."
        ),
    )
    parser.add_argument(
        "--prior-coordinate-copy-keep-future",
        action="store_true",
        help=(
            "When coordinate-copy collapse is enabled, keep one future-split "
            "duplicate. This is recall-first and can increase extras."
        ),
    )
    parser.add_argument(
        "--prior-coordinate-copy-max-future-kept",
        type=int,
        default=-1,
        help=(
            "When future coordinate copies are kept, optionally cap how many "
            "survive per frame. A non-negative value ranks them by current-frame "
            "projected 3D bridge evidence along the future split direction."
        ),
    )
    parser.add_argument(
        "--prior-coordinate-copy-future-projection-separation",
        type=float,
        default=24.0,
        help="Scaled separation used when projecting future split direction into the current image.",
    )
    parser.add_argument(
        "--prior-soft-pair-parent-volume-ratio-target",
        type=float,
        default=0.0,
        help=(
            "Experimental default-off soft penalty target for split pairs whose "
            "two support volumes together exceed the previous-frame parent "
            "support volume. This helps suppress large internal bright lobes "
            "without blocking small close daughters."
        ),
    )
    parser.add_argument(
        "--prior-soft-pair-parent-volume-ratio-penalty",
        type=float,
        default=0.0,
    )
    parser.add_argument("--prior-bridge-cross-radius", type=float, default=5.0)
    parser.add_argument("--prior-bridge-samples", type=int, default=9)
    parser.add_argument("--prior-bridge-target-valley-ratio", type=float, default=0.78)
    parser.add_argument("--prior-bridge-bonus-weight", type=float, default=90.0)
    parser.add_argument("--prior-bridge-penalty-weight", type=float, default=260.0)
    parser.add_argument("--prior-bridge-gap-weight", type=float, default=0.03)
    parser.add_argument(
        "--prior-close-bright-bridge-penalty-weight",
        type=float,
        default=0.0,
        help=(
            "Default-off soft penalty for close split pairs whose bridge "
            "remains bright. This is a complexity cost, not a hard veto."
        ),
    )
    parser.add_argument(
        "--prior-close-bright-bridge-target-valley",
        type=float,
        default=0.35,
        help="Valley ratio above which the close bright bridge penalty starts.",
    )
    parser.add_argument(
        "--prior-close-bright-bridge-close-sep",
        type=float,
        default=35.0,
        help="Scaled distance below which the close bright bridge penalty applies.",
    )
    parser.add_argument(
        "--future-candidate-cells",
        type=Path,
        help="Optional next-frame raw Cell Lumen candidates used only as soft temporal support.",
    )
    parser.add_argument("--future-candidate-frame-name")
    parser.add_argument(
        "--future-candidate-z-mode",
        choices=["auto", "raw", "scaled"],
        default="raw",
    )
    parser.add_argument(
        "--future-candidate-radius-z-mode",
        choices=["auto", "raw", "scaled"],
        default="scaled",
    )
    parser.add_argument("--prior-future-support-distance", type=float, default=32.0)
    parser.add_argument("--prior-future-support-weight", type=float, default=0.0)
    parser.add_argument("--prior-future-motion-balance-weight", type=float, default=0.0)
    parser.add_argument("--prior-future-motion-balance-tolerance", type=float, default=8.0)
    parser.add_argument(
        "--prior-future-motion-balance-single-only",
        action="store_true",
        help="Apply the optional motion-balance penalty to continuation choices but not to split pair scoring.",
    )
    parser.add_argument("--prior-pair-future-support-weight", type=float, default=0.0)
    parser.add_argument("--prior-pair-future-distinct-bonus", type=float, default=0.0)
    parser.add_argument("--prior-pair-future-same-penalty", type=float, default=0.0)
    parser.add_argument(
        "--prior-pair-preserved-split-allow-shared-future",
        action="store_true",
        help=(
            "For BioFuse splitA/splitB hypotheses, keep same future raw-candidate "
            "support as a scaled soft penalty rather than a full distinct-link "
            "failure. This is experimental and default-off."
        ),
    )
    parser.add_argument(
        "--prior-pair-preserved-shared-future-penalty-scale",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--prior-pair-require-distinct-future-support",
        action="store_true",
        help="Softly punish split hypotheses whose two daughters do not link to distinct next-frame raw candidates.",
    )
    parser.add_argument("--prior-pair-future-required-penalty", type=float, default=200.0)
    parser.add_argument("--prior-single-future-pressure-penalty", type=float, default=0.0)
    parser.add_argument("--prior-future-split-pressure-distance", type=float, default=38.0)
    parser.add_argument("--prior-future-split-pressure-min-separation", type=float, default=24.0)
    parser.add_argument(
        "--min-mask-mean-intensity",
        type=float,
        default=-1.0,
        help="Experimental review filter. Skip exported masks with mean intensity below this normalized 0-255 value.",
    )
    parser.add_argument(
        "--label-output",
        type=Path,
        help="Optional uint16 label TIFF with overlap resolved by brightness and distance score.",
    )
    parser.add_argument(
        "--label-z-stretch",
        type=int,
        default=1,
        help="Repeat label TIFF slices along z for direct Napari viewing without manually setting layer scale.",
    )
    parser.add_argument(
        "--stretched-image-output",
        type=Path,
        help="Optional raw image review TIFF repeated by --label-z-stretch so it aligns with stretched labels.",
    )
    parser.add_argument(
        "--claim-distance-penalty",
        type=float,
        default=0.15,
        help="Penalty used only when resolving overlapping label TIFF voxels.",
    )
    args = parser.parse_args()
    export_masks(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
