#!/usr/bin/env python3
"""Audit an experimental Cell Lumen rolling window selector.

This script is intentionally separate from the main Cell Universe tracker and
from run_cell_lumen_true_rolling_audit.py. It reads an existing Cell Lumen raw
candidate cache, then solves a small windowed selection problem:

* each previous-frame prior cell must choose one continuation hypothesis or one
  split hypothesis;
* hypotheses from different parents cannot claim the same local raw candidate
  or physically duplicate candidates;
* split hypotheses pay a complexity penalty and must be supported by distinct
  future candidates to beat a continuation.

The goal is to test Ultrack-like global selection ideas without changing the
established Cell Lumen candidate generator, YAML, PCA shape logic, or rolling
baseline. Ground truth is used only after a frame is selected, for audit.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path("/Users/wangyiding/CellUniverse")
DEFAULT_CACHE_RUN = Path(
    "/Volumes/T9/🦠Cell Universe/🟣Output/CellLumen_AllRun_07062026/"
    "📁Rolling_CenterAudit/"
    "⚠️001-020_CentroidBioFuse_Margin6_PASS18_FAIL2_f016Miss1Extra1_f018Miss2_20260707.1550"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/Volumes/T9/🦠Cell Universe/🟣Output/CellLumen_AllRun_07062026/"
    "📁Rolling_CenterAudit"
)
DEFAULT_INPUT_DIR = ROOT / (
    "C++/examples/input/"
    "C.elegans_developing embryo_Fluo-N3DH-CE_Training/01"
)
DEFAULT_INITIAL = ROOT / (
    "C++/config/C.elegans developing embryo/"
    "C.elegans_initial/initial_files/00_core_start_points/initial_embryo_0.csv"
)
DEFAULT_GT = ROOT / (
    "C++/config/C.elegans developing embryo/"
    "C.elegans_initial/ground_truth/embryo_FixedGroundTruth.csv"
)


@dataclass(frozen=True)
class Cell:
    name: str
    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float
    source: str
    birth_frame: int = 0


@dataclass(frozen=True)
class Hypothesis:
    parent_name: str
    kind: str
    cells: tuple[Cell, ...]
    score: float
    detail: str


def parse_frames(text: str) -> list[int]:
    frames: list[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = chunk.split("-", 1)
            frames.extend(range(int(start), int(end) + 1))
        else:
            frames.append(int(chunk))
    return sorted(set(frames))


def parse_float(row: dict[str, str], names: tuple[str, ...], default: float = 0.0) -> float:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return float(value)
    return default


def load_initial(path: Path, z_scale: float, initial_root_age: int = 0) -> list[Cell]:
    cells: list[Cell] = []
    initial_birth_frame = -max(0, int(initial_root_age))
    with path.open(newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle), start=1):
            x = parse_float(row, ("x", "center_x"))
            y = parse_float(row, ("y", "center_y"))
            z_raw = parse_float(row, ("z", "center_z"))
            name = row.get("name") or row.get("cell_name") or row.get("cellType") or "root"
            cells.append(
                Cell(
                    name=f"{name}_{index}",
                    x=x,
                    y=y,
                    z=z_raw * z_scale,
                    rx=18.0,
                    ry=18.0,
                    rz=18.0,
                    source="initial",
                    birth_frame=initial_birth_frame,
                )
            )
    return cells


def load_raw_candidates(path: Path, z_scale: float) -> list[Cell]:
    cells: list[Cell] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            cells.append(
                Cell(
                    name=row["name"],
                    x=float(row["x"]),
                    y=float(row["y"]),
                    z=float(row["z"]) * z_scale,
                    rx=parse_float(row, ("majorRadius", "aRadius", "radius_scaled"), 18.0),
                    ry=parse_float(row, ("bRadius", "radius_scaled"), 18.0),
                    # Cell Lumen raw candidate radii are already treated as
                    # scaled by the current fine-shape audit path.
                    rz=parse_float(row, ("minorRadius", "cRadius", "radius_scaled"), 18.0),
                    source="raw",
                )
            )
    return cells


def load_summary_candidates(path: Path) -> list[Cell]:
    cells: list[Cell] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            cells.append(
                Cell(
                    name=row["cell_name"],
                    x=float(row["center_x"]),
                    y=float(row["center_y"]),
                    z=float(row["center_z_scaled"]),
                    rx=parse_float(row, ("radius_x", "radius_scaled"), 18.0),
                    ry=parse_float(row, ("radius_y", "radius_scaled"), 18.0),
                    rz=parse_float(row, ("radius_z_scaled", "radius_scaled"), 18.0),
                    source="summary",
                )
            )
    return cells


def load_gt(path: Path, frames: list[int]) -> dict[int, list[dict[str, float | str]]]:
    wanted = set(frames)
    by_frame: dict[int, list[dict[str, float | str]]] = {frame: [] for frame in frames}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            frame = int(row["frame"])
            if frame not in wanted:
                continue
            by_frame[frame].append(
                {
                    "label": row["label_id"],
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "z": float(row["z_interp"]),
                }
            )
    return by_frame


def load_bridge_helpers():
    import sys

    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from export_fine_shape_masks_from_cells import (  # type: ignore
        CellRow as BridgeCellRow,
        bridge_slab_valley_ratio,
        load_tiff_stack,
    )

    return BridgeCellRow, bridge_slab_valley_ratio, load_tiff_stack


def to_bridge_cell(cell: Cell, bridge_cell_row_type) -> object:
    return bridge_cell_row_type(
        frame="",
        name=cell.name,
        x=cell.x,
        y=cell.y,
        z_scaled=cell.z,
        rx=cell.rx,
        ry=cell.ry,
        rz_scaled=cell.rz,
        z_mode_used=cell.source,
    )


def distance(left: Cell | dict[str, float | str], right: Cell | dict[str, float | str]) -> float:
    lx = float(left.x if isinstance(left, Cell) else left["x"])
    ly = float(left.y if isinstance(left, Cell) else left["y"])
    lz = float(left.z if isinstance(left, Cell) else left["z"])
    rx = float(right.x if isinstance(right, Cell) else right["x"])
    ry = float(right.y if isinstance(right, Cell) else right["y"])
    rz = float(right.z if isinstance(right, Cell) else right["z"])
    return math.sqrt((lx - rx) ** 2 + (ly - ry) ** 2 + (lz - rz) ** 2)


def ellipsoid_norm(cell: Cell, prior: Cell, scale: float, margin: float) -> float:
    rx = max(1.0, prior.rx * scale + margin)
    ry = max(1.0, prior.ry * scale + margin)
    rz = max(1.0, prior.rz * scale + margin)
    dx = (cell.x - prior.x) / rx
    dy = (cell.y - prior.y) / ry
    dz = (cell.z - prior.z) / rz
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def support_radius(cell: Cell) -> float:
    return max(1.0, (max(1.0, cell.rx) * max(1.0, cell.ry) * max(1.0, cell.rz)) ** (1.0 / 3.0))


def nearest_support(cell: Cell, future_cells: list[Cell], radius: float) -> tuple[float, float, str]:
    if not future_cells:
        return 0.0, float("inf"), ""
    best = min(future_cells, key=lambda other: distance(cell, other))
    best_dist = distance(cell, best)
    return max(0.0, radius - best_dist), best_dist, best.name


def pair_future_support(
    left: Cell,
    right: Cell,
    future_cells_by_offset: list[list[Cell]],
    args: argparse.Namespace,
) -> tuple[float, int, str]:
    total = 0.0
    distinct_offsets = 0
    detail_parts: list[str] = []
    for offset, future_cells in enumerate(future_cells_by_offset, start=1):
        if not future_cells:
            continue
        left_support, left_dist, left_name = nearest_support(left, future_cells, args.future_link_distance)
        right_support, right_dist, right_name = nearest_support(right, future_cells, args.future_link_distance)
        distinct = bool(left_name and right_name and left_name != right_name)
        support = min(left_support, right_support)
        if distinct:
            distinct_offsets += 1
            total += support * (args.future_decay ** (offset - 1))
        else:
            total -= args.same_future_penalty
        detail_parts.append(
            f"o{offset}:support={support:.3g}:distinct={int(distinct)}:"
            f"ld={left_dist:.3g}:rd={right_dist:.3g}"
        )
    return total, distinct_offsets, "|".join(detail_parts)


def single_future_support(
    cell: Cell,
    future_cells_by_offset: list[list[Cell]],
    args: argparse.Namespace,
) -> tuple[float, str]:
    total = 0.0
    detail_parts: list[str] = []
    for offset, future_cells in enumerate(future_cells_by_offset, start=1):
        support, best_dist, _name = nearest_support(cell, future_cells, args.future_link_distance)
        weighted = support * (args.future_decay ** (offset - 1))
        total += weighted
        detail_parts.append(f"o{offset}:support={support:.3g}:dist={best_dist:.3g}")
    return total, "|".join(detail_parts)


def has_future_candidates(future_cells_by_offset: list[list[Cell]]) -> bool:
    return any(bool(cells) for cells in future_cells_by_offset)


def local_split_pressure(
    cell: Cell,
    future_cells_by_offset: list[list[Cell]],
    args: argparse.Namespace,
) -> float:
    if not future_cells_by_offset:
        return 0.0
    future_cells = future_cells_by_offset[0]
    nearby = [candidate for candidate in future_cells if distance(cell, candidate) <= args.future_link_distance]
    if len(nearby) < 2:
        return 0.0
    best = 0.0
    for i, left in enumerate(nearby):
        for right in nearby[i + 1 :]:
            sep = distance(left, right)
            if sep < args.split_pressure_min_separation:
                continue
            left_support, _, _ = nearest_support(left, [cell], args.future_link_distance)
            right_support, _, _ = nearest_support(right, [cell], args.future_link_distance)
            best = max(best, min(left_support, right_support) + 0.05 * sep)
    return best


def base_candidate_score(
    candidate: Cell,
    prior: Cell,
    future_cells_by_offset: list[list[Cell]],
    args: argparse.Namespace,
) -> tuple[float, str]:
    dist = distance(candidate, prior)
    norm = ellipsoid_norm(candidate, prior, args.parent_region_scale, args.parent_region_margin)
    future_support, future_detail = single_future_support(candidate, future_cells_by_offset, args)
    split_pressure = local_split_pressure(candidate, future_cells_by_offset, args)
    source_bonus = args.summary_candidate_bonus if candidate.source == "summary" else 0.0
    future_shortfall = (
        max(0.0, args.single_future_target_support - future_support)
        if has_future_candidates(future_cells_by_offset)
        else 0.0
    )
    score = (
        args.support_weight * min(args.support_clip, support_radius(candidate))
        - args.distance_weight * dist
        - args.norm_weight * norm
        + args.future_support_weight * future_support
        - args.single_split_pressure_penalty * split_pressure
        - args.single_future_shortfall_penalty_weight * future_shortfall
        + source_bonus
    )
    detail = (
        f"dist={dist:.3g};norm={norm:.3g};support_radius={support_radius(candidate):.3g};"
        f"future={future_support:.3g};split_pressure={split_pressure:.3g};"
        f"future_shortfall={future_shortfall:.3g};"
        f"source={candidate.source};source_bonus={source_bonus:.3g};{future_detail}"
    )
    return score, detail


def build_parent_hypotheses(
    prior: Cell,
    raw_candidates: list[Cell],
    future_cells_by_offset: list[list[Cell]],
    args: argparse.Namespace,
    frame: int,
    bridge_stack=None,
    bridge_cell_row_type=None,
    bridge_ratio_func=None,
) -> list[Hypothesis]:
    held = Cell(
        name=f"{prior.name}_held",
        x=prior.x,
        y=prior.y,
        z=prior.z,
        rx=prior.rx,
        ry=prior.ry,
        rz=prior.rz,
        source="held_prior",
        birth_frame=prior.birth_frame,
    )
    fallback = Hypothesis(
        parent_name=prior.name,
        kind="hold",
        cells=(held,),
        score=-args.hold_penalty,
        detail="fallback hold; used only when raw candidates are weaker or globally conflicting",
    )
    items: list[tuple[float, Cell, str]] = []
    for candidate in raw_candidates:
        dist = distance(candidate, prior)
        if dist > args.reach_distance:
            continue
        norm = ellipsoid_norm(candidate, prior, args.parent_region_scale, args.parent_region_margin)
        if norm > args.parent_region_max_norm:
            continue
        score, detail = base_candidate_score(candidate, prior, future_cells_by_offset, args)
        items.append((score, candidate, detail))

    if not items:
        return [
            Hypothesis(
                parent_name=prior.name,
                kind="hold",
                cells=(held,),
                score=-args.missing_candidate_penalty,
                detail="no reachable raw candidate; holding previous center",
            )
        ]

    items.sort(key=lambda item: item[0], reverse=True)
    top = items[: args.top_k_per_parent]
    hypotheses = [
        Hypothesis(prior.name, "single", (candidate,), score, detail)
        for score, candidate, detail in top[: args.top_single_per_parent]
    ]
    hypotheses.append(fallback)

    for i, (left_score, left, left_detail) in enumerate(top):
        for right_score, right, right_detail in top[i + 1 :]:
            sep = distance(left, right)
            if sep < args.physical_min_split_separation:
                continue
            pair_support, distinct_offsets, pair_future_detail = pair_future_support(
                left,
                right,
                future_cells_by_offset,
                args,
            )
            midpoint = Cell(
                name="pair_midpoint",
                x=(left.x + right.x) * 0.5,
                y=(left.y + right.y) * 0.5,
                z=(left.z + right.z) * 0.5,
                rx=1.0,
                ry=1.0,
                rz=1.0,
                source="midpoint",
            )
            midpoint_distance = distance(midpoint, prior)
            midpoint_excess = max(0.0, midpoint_distance - args.pair_midpoint_target_distance)
            left_parent_distance = distance(left, prior)
            right_parent_distance = distance(right, prior)
            balance = abs(left_parent_distance - right_parent_distance) / max(1.0, sep)
            raw_only_penalty = (
                args.raw_only_split_penalty
                if left.source == "raw" and right.source == "raw"
                else 0.0
            )
            parent_age = max(0, frame - prior.birth_frame)
            young_parent_shortfall = max(0.0, args.min_split_parent_age - parent_age)
            young_parent_penalty = args.young_parent_split_penalty * young_parent_shortfall
            bridge_valley = 1.0
            bridge_gap = 0.0
            bridge_term = 0.0
            if bridge_stack is not None and bridge_cell_row_type is not None and bridge_ratio_func is not None:
                left_bridge = to_bridge_cell(left, bridge_cell_row_type)
                right_bridge = to_bridge_cell(right, bridge_cell_row_type)
                bridge_valley, bridge_gap = bridge_ratio_func(
                    bridge_stack,
                    left_bridge,
                    right_bridge,
                    args,
                )
                bridge_dark = max(0.0, args.bridge_target_valley_ratio - bridge_valley)
                bridge_bright = max(0.0, bridge_valley - args.bridge_target_valley_ratio)
                bridge_term = (
                    args.bridge_dark_bonus_weight * bridge_dark
                    + args.bridge_gap_bonus_weight * max(0.0, bridge_gap)
                    - args.bridge_bright_penalty_weight * bridge_bright
                )
            shortfall = max(0.0, args.soft_split_target_separation - sep)
            low_distinct_penalty = (
                args.required_distinct_future_penalty
                if distinct_offsets < args.min_distinct_future_offsets
                else 0.0
            )
            pair_score = (
                left_score
                + right_score
                + args.pair_future_support_weight * pair_support
                + args.pair_separation_weight * min(sep, args.pair_separation_clip)
                + bridge_term
                - args.split_shortfall_penalty * shortfall
                - args.pair_midpoint_penalty_weight * midpoint_excess
                - args.pair_balance_penalty_weight * max(0.0, balance - args.pair_balance_target)
                - raw_only_penalty
                - young_parent_penalty
                - args.division_penalty
                - low_distinct_penalty
            )
            ordered = tuple(sorted((left, right), key=lambda cell: (cell.y, cell.x, cell.z, cell.name)))
            detail = (
                f"sep={sep:.3g};pair_future={pair_support:.3g};distinct_offsets={distinct_offsets};"
                f"shortfall={shortfall:.3g};midpoint_distance={midpoint_distance:.3g};"
                f"midpoint_excess={midpoint_excess:.3g};balance={balance:.3g};"
                f"raw_only_penalty={raw_only_penalty:.3g};parent_age={parent_age};"
                f"young_parent_penalty={young_parent_penalty:.3g};"
                f"bridge_valley={bridge_valley:.3g};bridge_gap={bridge_gap:.3g};"
                f"bridge_term={bridge_term:.3g};"
                f"left=({left_detail});right=({right_detail});"
                f"pair_future_detail={pair_future_detail}"
            )
            hypotheses.append(Hypothesis(prior.name, "split", ordered, pair_score, detail))

    hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)
    kept = [hyp for hyp in hypotheses if hyp.kind != "hold"][: args.max_hypotheses_per_parent]
    # Always keep the fallback hold outside the top-k cut. Otherwise a heavily
    # over-split state can make the MILP infeasible because every remaining
    # hypothesis for a parent conflicts with another selected parent.
    kept.append(fallback)
    return kept


def candidates_conflict(left: Cell, right: Cell, args: argparse.Namespace) -> bool:
    if left.name == right.name:
        return True
    return distance(left, right) < args.global_duplicate_distance


def hypotheses_conflict(left: Hypothesis, right: Hypothesis, args: argparse.Namespace) -> bool:
    # Hold hypotheses do not claim a raw Cell Lumen candidate. Keeping them
    # conflict-free guarantees a complete ILP-style solution exists, which is
    # safer for auditing than silently returning an empty frame.
    if left.kind == "hold" or right.kind == "hold":
        return False
    for left_cell in left.cells:
        for right_cell in right.cells:
            if candidates_conflict(left_cell, right_cell, args):
                return True
    return False


def solve_global_parent_choice(
    hypotheses_by_parent: dict[str, list[Hypothesis]],
    args: argparse.Namespace,
) -> list[Hypothesis]:
    if args.solver_backend in {"auto", "scipy"}:
        try:
            return solve_global_parent_choice_scipy(hypotheses_by_parent, args)
        except Exception as exc:
            if args.solver_backend == "scipy":
                raise
            print(f"[WindowILPStyle Solver] scipy_milp_unavailable fallback=branch reason={exc}")

    return solve_global_parent_choice_branch(hypotheses_by_parent, args)


def solve_global_parent_choice_scipy(
    hypotheses_by_parent: dict[str, list[Hypothesis]],
    args: argparse.Namespace,
) -> list[Hypothesis]:
    """Solve the parent-hypothesis set packing problem with SciPy/HiGHS.

    This is the first real MIP backend for the Cell Lumen window selector. It
    mirrors the Ultrack idea at the local hypothesis level: every parent must
    choose exactly one continuation/split/hold state, and physically conflicting
    states across parents cannot co-exist.
    """
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    hypotheses: list[Hypothesis] = []
    parent_to_indices: dict[str, list[int]] = {}
    for parent, parent_hypotheses in sorted(hypotheses_by_parent.items()):
        parent_to_indices[parent] = []
        for hyp in parent_hypotheses:
            parent_to_indices[parent].append(len(hypotheses))
            hypotheses.append(hyp)

    if not hypotheses:
        return []

    conflict_pairs: list[tuple[int, int]] = []
    for i, left in enumerate(hypotheses):
        for j in range(i + 1, len(hypotheses)):
            right = hypotheses[j]
            if left.parent_name == right.parent_name:
                continue
            if hypotheses_conflict(left, right, args):
                conflict_pairs.append((i, j))

    row_count = len(parent_to_indices) + len(conflict_pairs)
    matrix = lil_matrix((row_count, len(hypotheses)), dtype=float)
    lower = np.full(row_count, -np.inf, dtype=float)
    upper = np.full(row_count, np.inf, dtype=float)

    row_index = 0
    for indices in parent_to_indices.values():
        for index in indices:
            matrix[row_index, index] = 1.0
        lower[row_index] = 1.0
        upper[row_index] = 1.0
        row_index += 1

    for left, right in conflict_pairs:
        matrix[row_index, left] = 1.0
        matrix[row_index, right] = 1.0
        lower[row_index] = -np.inf
        upper[row_index] = 1.0
        row_index += 1

    objective = np.array([-hyp.score for hyp in hypotheses], dtype=float)
    result = milp(
        c=objective,
        integrality=np.ones(len(hypotheses), dtype=int),
        bounds=Bounds(0.0, 1.0),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options={
            "time_limit": args.solver_time_limit_sec,
            "mip_rel_gap": args.solver_mip_gap,
            "disp": False,
        },
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"SciPy MILP failed: status={result.status} message={result.message}")

    selected = [
        hyp
        for hyp, value in zip(hypotheses, result.x)
        if value >= 0.5
    ]
    selected.sort(key=lambda hyp: hyp.parent_name)
    print(
        f"[WindowILPStyle Solver] backend=scipy_milp hypotheses={len(hypotheses)} "
        f"parents={len(parent_to_indices)} conflicts={len(conflict_pairs)} "
        f"objective={-float(result.fun):.6g}"
    )
    return selected


def solve_global_parent_choice_branch(
    hypotheses_by_parent: dict[str, list[Hypothesis]],
    args: argparse.Namespace,
) -> list[Hypothesis]:
    parent_names = sorted(hypotheses_by_parent, key=lambda name: len(hypotheses_by_parent[name]))
    suffix_best: list[float] = [0.0] * (len(parent_names) + 1)
    for index in range(len(parent_names) - 1, -1, -1):
        best_local = max((hyp.score for hyp in hypotheses_by_parent[parent_names[index]]), default=0.0)
        suffix_best[index] = suffix_best[index + 1] + best_local

    best_score = -float("inf")
    best_solution: list[Hypothesis] = []
    expanded_nodes = 0
    search_truncated = False

    def greedy_solution() -> list[Hypothesis]:
        chosen: list[Hypothesis] = []
        for parent in parent_names:
            selected = None
            for hyp in hypotheses_by_parent[parent]:
                if any(hypotheses_conflict(hyp, other, args) for other in chosen):
                    continue
                selected = hyp
                break
            if selected is None:
                selected = hypotheses_by_parent[parent][-1]
            chosen.append(selected)
        chosen.sort(key=lambda hyp: hyp.parent_name)
        return chosen

    def search(index: int, chosen: list[Hypothesis], score: float) -> None:
        nonlocal best_score, best_solution, expanded_nodes, search_truncated
        if search_truncated:
            return
        expanded_nodes += 1
        if expanded_nodes > args.max_search_nodes:
            search_truncated = True
            return
        if score + suffix_best[index] <= best_score:
            return
        if index >= len(parent_names):
            if score > best_score:
                best_score = score
                best_solution = list(chosen)
            return
        parent = parent_names[index]
        for hyp in hypotheses_by_parent[parent]:
            if any(hypotheses_conflict(hyp, selected, args) for selected in chosen):
                continue
            chosen.append(hyp)
            search(index + 1, chosen, score + hyp.score)
            chosen.pop()

    search(0, [], 0.0)
    if search_truncated or not best_solution:
        return greedy_solution()
    best_solution.sort(key=lambda hyp: hyp.parent_name)
    return best_solution


def greedy_match(gt_rows: list[dict[str, float | str]], pred_rows: list[Cell], threshold: float):
    pairs: list[tuple[float, int, int]] = []
    for gi, gt in enumerate(gt_rows):
        for pi, pred in enumerate(pred_rows):
            pairs.append((distance(gt, pred), gi, pi))
    pairs.sort(key=lambda item: item[0])
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[tuple[float, dict[str, float | str], Cell]] = []
    for dist, gi, pi in pairs:
        if dist > threshold:
            break
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        matches.append((dist, gt_rows[gi], pred_rows[pi]))
    missing = [gt_rows[index] for index in range(len(gt_rows)) if index not in used_gt]
    extra = [pred_rows[index] for index in range(len(pred_rows)) if index not in used_pred]
    return matches, missing, extra


def write_selected_csv(path: Path, frame: int, cells: list[Cell], z_scale: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        fieldnames = ["file", "name", "x", "y", "z", "majorRadius", "bRadius", "minorRadius"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for cell in cells:
            writer.writerow(
                {
                    "file": f"t{frame:03d}.tif",
                    "name": cell.name,
                    "x": f"{cell.x:.9g}",
                    "y": f"{cell.y:.9g}",
                    "z": f"{cell.z / z_scale:.9g}",
                    "majorRadius": f"{cell.rx:.9g}",
                    "bRadius": f"{cell.ry:.9g}",
                    "minorRadius": f"{cell.rz:.9g}",
                }
            )


def make_output_cell(
    parent_name: str,
    hyp: Hypothesis,
    cell: Cell,
    index: int,
    frame: int,
    parent_birth_frame: int,
) -> Cell:
    if hyp.kind == "split":
        name = f"{parent_name}_{index}"
        birth_frame = frame
    elif hyp.kind == "hold":
        name = parent_name
        birth_frame = parent_birth_frame
    else:
        name = parent_name
        birth_frame = parent_birth_frame
    return Cell(
        name=name,
        x=cell.x,
        y=cell.y,
        z=cell.z,
        rx=cell.rx,
        ry=cell.ry,
        rz=cell.rz,
        source=f"window_{hyp.kind}:{cell.name}",
        birth_frame=birth_frame,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-run-dir", type=Path, default=DEFAULT_CACHE_RUN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--initial", type=Path, default=DEFAULT_INITIAL)
    parser.add_argument("--gt-csv", type=Path, default=DEFAULT_GT)
    parser.add_argument("--frames", default="1-20")
    parser.add_argument("--threshold", type=float, default=25.0)
    parser.add_argument("--z-scale", type=float, default=7.0)
    parser.add_argument("--window-size", type=int, default=3)
    parser.add_argument(
        "--solver-backend",
        choices=["auto", "scipy", "branch"],
        default="auto",
        help="Use scipy.optimize.milp when available, otherwise the bounded diagnostic branch search.",
    )
    parser.add_argument("--solver-time-limit-sec", type=float, default=60.0)
    parser.add_argument("--solver-mip-gap", type=float, default=0.001)
    parser.add_argument(
        "--include-summary-candidates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include BioFuse summary candidates as stable merged hypotheses beside raw Cell Lumen peaks.",
    )
    parser.add_argument("--reach-distance", type=float, default=54.0)
    parser.add_argument("--parent-region-scale", type=float, default=1.7)
    parser.add_argument("--parent-region-margin", type=float, default=10.0)
    parser.add_argument("--parent-region-max-norm", type=float, default=2.5)
    parser.add_argument("--top-k-per-parent", type=int, default=8)
    parser.add_argument("--top-single-per-parent", type=int, default=4)
    parser.add_argument("--max-hypotheses-per-parent", type=int, default=18)
    parser.add_argument("--support-weight", type=float, default=2.0)
    parser.add_argument("--support-clip", type=float, default=26.0)
    parser.add_argument("--summary-candidate-bonus", type=float, default=18.0)
    parser.add_argument("--distance-weight", type=float, default=0.38)
    parser.add_argument("--norm-weight", type=float, default=4.5)
    parser.add_argument("--future-link-distance", type=float, default=34.0)
    parser.add_argument("--future-decay", type=float, default=0.72)
    parser.add_argument("--future-support-weight", type=float, default=0.65)
    parser.add_argument(
        "--single-future-target-support",
        type=float,
        default=0.0,
        help="Soft target for continuation support when future candidates exist.",
    )
    parser.add_argument(
        "--single-future-shortfall-penalty-weight",
        type=float,
        default=0.0,
        help="Penalty weight for continuation candidates that do not persist into the future window.",
    )
    parser.add_argument("--pair-future-support-weight", type=float, default=1.45)
    parser.add_argument("--same-future-penalty", type=float, default=8.0)
    parser.add_argument("--min-distinct-future-offsets", type=int, default=1)
    parser.add_argument("--required-distinct-future-penalty", type=float, default=28.0)
    parser.add_argument("--single-split-pressure-penalty", type=float, default=0.35)
    parser.add_argument("--split-pressure-min-separation", type=float, default=18.0)
    parser.add_argument("--physical-min-split-separation", type=float, default=8.0)
    parser.add_argument("--soft-split-target-separation", type=float, default=24.0)
    parser.add_argument("--pair-separation-weight", type=float, default=0.34)
    parser.add_argument("--pair-separation-clip", type=float, default=48.0)
    parser.add_argument("--split-shortfall-penalty", type=float, default=0.9)
    parser.add_argument("--pair-midpoint-target-distance", type=float, default=14.0)
    parser.add_argument("--pair-midpoint-penalty-weight", type=float, default=3.0)
    parser.add_argument("--pair-balance-target", type=float, default=0.35)
    parser.add_argument("--pair-balance-penalty-weight", type=float, default=16.0)
    parser.add_argument("--raw-only-split-penalty", type=float, default=0.0)
    parser.add_argument(
        "--min-split-parent-age",
        type=float,
        default=0.0,
        help="Soft refractory age in frames before a rolling parent is allowed to split cheaply.",
    )
    parser.add_argument(
        "--initial-root-age",
        type=int,
        default=0,
        help="Treat initial CSV roots as this many frames old for soft split-age scoring.",
    )
    parser.add_argument(
        "--young-parent-split-penalty",
        type=float,
        default=0.0,
        help="Penalty per missing frame below --min-split-parent-age for a split hypothesis.",
    )
    parser.add_argument("--use-bridge-evidence", action="store_true")
    parser.add_argument("--bridge-target-valley-ratio", type=float, default=0.82)
    parser.add_argument("--bridge-dark-bonus-weight", type=float, default=30.0)
    parser.add_argument("--bridge-bright-penalty-weight", type=float, default=80.0)
    parser.add_argument("--bridge-gap-bonus-weight", type=float, default=0.02)
    parser.add_argument("--prior-bridge-cross-radius", type=float, default=4.0)
    parser.add_argument("--prior-bridge-samples", type=int, default=17)
    parser.add_argument("--division-penalty", type=float, default=132.0)
    parser.add_argument("--global-duplicate-distance", type=float, default=14.0)
    parser.add_argument("--missing-candidate-penalty", type=float, default=120.0)
    parser.add_argument("--hold-penalty", type=float, default=95.0)
    parser.add_argument(
        "--max-search-nodes",
        type=int,
        default=250000,
        help="Branch search cap before falling back to greedy conflict-aware selection.",
    )
    args = parser.parse_args()
    # Reuse the existing bridge_slab_valley_ratio helper, whose option names
    # come from the fine-shape exporter.
    args.z_scaling = args.z_scale
    bridge_cell_row_type = None
    bridge_ratio_func = None
    load_tiff_stack_func = None
    if args.use_bridge_evidence:
        bridge_cell_row_type, bridge_ratio_func, load_tiff_stack_func = load_bridge_helpers()
        print(
            f"[WindowILPStyle Bridge] enabled input_dir={args.input_dir} "
            f"target_valley={args.bridge_target_valley_ratio}",
            flush=True,
        )

    frames = parse_frames(args.frames)
    stamp = time.strftime("%Y%m%d.%H%M%S")
    run_name = args.run_name or (
        f"⚠️EXPERIMENT_{frames[0]:03d}-{frames[-1]:03d}_"
        f"WindowILPStyle_cacheAudit_{stamp}"
    )
    run_dir = args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_by_frame: dict[int, list[Cell]] = {}
    summary_by_frame: dict[int, list[Cell]] = {}
    for frame in frames:
        raw_path = args.cache_run_dir / f"f{frame:03d}" / f"t{frame:03d}_cell_lumen_independent.csv"
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing raw candidate cache: {raw_path}")
        raw_by_frame[frame] = load_raw_candidates(raw_path, args.z_scale)
        summary_path = (
            args.cache_run_dir
            / f"f{frame:03d}"
            / "fine_shape_biofused"
            / f"t{frame:03d}_fine_shape_summary.csv"
        )
        if args.include_summary_candidates and summary_path.exists():
            summary_by_frame[frame] = load_summary_candidates(summary_path)
        else:
            summary_by_frame[frame] = []

    gt_by_frame = load_gt(args.gt_csv, frames)
    prior_cells = load_initial(args.initial, args.z_scale, args.initial_root_age)
    summary_rows: list[dict[str, str | int | float]] = []
    detail_rows: list[dict[str, str | int | float]] = []

    for frame in frames:
        frame_start_time = time.perf_counter()
        frame_dir = run_dir / f"f{frame:03d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        bridge_load_seconds = 0.0
        bridge_stack = None
        if args.use_bridge_evidence:
            if load_tiff_stack_func is None:
                raise RuntimeError("Bridge evidence requested but TIFF loader was not initialized.")
            bridge_start_time = time.perf_counter()
            bridge_stack = load_tiff_stack_func(args.input_dir / f"t{frame:03d}.tif")
            bridge_load_seconds = time.perf_counter() - bridge_start_time
        future_cells_by_offset = [
            raw_by_frame.get(frame + offset, []) + summary_by_frame.get(frame + offset, [])
            for offset in range(1, max(1, args.window_size))
        ]
        current_candidates = raw_by_frame[frame] + summary_by_frame.get(frame, [])
        hypotheses_by_parent: dict[str, list[Hypothesis]] = {}
        build_start_time = time.perf_counter()
        for prior in prior_cells:
            hypotheses_by_parent[prior.name] = build_parent_hypotheses(
                prior,
                current_candidates,
                future_cells_by_offset,
                args,
                frame,
                bridge_stack=bridge_stack,
                bridge_cell_row_type=bridge_cell_row_type,
                bridge_ratio_func=bridge_ratio_func,
            )
            for rank, hyp in enumerate(hypotheses_by_parent[prior.name], start=1):
                detail_rows.append(
                    {
                        "frame": frame,
                        "parent": prior.name,
                        "rank": rank,
                        "kind": hyp.kind,
                        "score": f"{hyp.score:.9g}",
                        "cell_names": ";".join(cell.name for cell in hyp.cells),
                        "cell_xyz": ";".join(
                            f"{cell.x:.3f},{cell.y:.3f},{cell.z:.3f}" for cell in hyp.cells
                        ),
                        "detail": hyp.detail,
                    }
                )
        build_seconds = time.perf_counter() - build_start_time

        solve_start_time = time.perf_counter()
        solution = solve_global_parent_choice(hypotheses_by_parent, args)
        solve_seconds = time.perf_counter() - solve_start_time
        selected: list[Cell] = []
        prior_by_name = {cell.name: cell for cell in prior_cells}
        for hyp in solution:
            parent_birth_frame = prior_by_name[hyp.parent_name].birth_frame
            for index, cell in enumerate(hyp.cells):
                selected.append(
                    make_output_cell(
                        hyp.parent_name,
                        hyp,
                        cell,
                        index,
                        frame,
                        parent_birth_frame,
                    )
                )
        selected.sort(key=lambda cell: (cell.y, cell.x, cell.z, cell.name))

        selected_csv = frame_dir / f"t{frame:03d}_window_ilp_selected.csv"
        write_selected_csv(selected_csv, frame, selected, args.z_scale)
        audit_start_time = time.perf_counter()
        matches, missing, extra = greedy_match(gt_by_frame[frame], selected, args.threshold)
        audit_seconds = time.perf_counter() - audit_start_time
        status = "PASS" if not missing and not extra else "FAIL"
        max_dist = max((item[0] for item in matches), default=0.0)
        mean_dist = sum(item[0] for item in matches) / len(matches) if matches else 0.0
        frame_seconds = time.perf_counter() - frame_start_time
        print(
            f"f{frame:03d} {status} pred={len(selected)} gt={len(gt_by_frame[frame])} "
            f"match={len(matches)} miss={len(missing)} extra={len(extra)} "
            f"mean={mean_dist:.3f} max={max_dist:.3f} time={frame_seconds:.2f}s "
            f"build={build_seconds:.2f}s solve={solve_seconds:.2f}s bridge={bridge_load_seconds:.2f}s",
            flush=True,
        )
        if missing:
            print("  missing=" + ",".join(str(row["label"]) for row in missing), flush=True)
        if extra:
            print("  extra=" + ",".join(cell.name for cell in extra), flush=True)

        summary_rows.append(
            {
                "frame": frame,
                "status": status,
                "gt": len(gt_by_frame[frame]),
                "raw": len(raw_by_frame[frame]),
                "pred": len(selected),
                "matched": len(matches),
                "missing": len(missing),
                "extra": len(extra),
                "mean_distance": f"{mean_dist:.9g}",
                "max_distance": f"{max_dist:.9g}",
                "selected_csv": str(selected_csv),
                "missing_labels": ";".join(str(row["label"]) for row in missing),
                "extra_names": ";".join(cell.name for cell in extra),
                "frame_seconds": f"{frame_seconds:.9g}",
                "bridge_load_seconds": f"{bridge_load_seconds:.9g}",
                "hypothesis_build_seconds": f"{build_seconds:.9g}",
                "solver_seconds": f"{solve_seconds:.9g}",
                "audit_seconds": f"{audit_seconds:.9g}",
            }
        )
        prior_cells = selected

    summary_path = run_dir / f"window_ilp_audit_summary_f{frames[0]:03d}_f{frames[-1]:03d}.csv"
    with summary_path.open("w", newline="") as handle:
        fieldnames = [
            "frame",
            "status",
            "gt",
            "raw",
            "pred",
            "matched",
            "missing",
            "extra",
            "mean_distance",
            "max_distance",
            "selected_csv",
            "missing_labels",
            "extra_names",
            "frame_seconds",
            "bridge_load_seconds",
            "hypothesis_build_seconds",
            "solver_seconds",
            "audit_seconds",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    detail_path = run_dir / f"window_ilp_hypotheses_f{frames[0]:03d}_f{frames[-1]:03d}.csv"
    with detail_path.open("w", newline="") as handle:
        fieldnames = ["frame", "parent", "rank", "kind", "score", "cell_names", "cell_xyz", "detail"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detail_rows)

    fail_rows = [row for row in summary_rows if row["status"] != "PASS"]
    print(
        f"[WindowILPStyle Audit] run_dir={run_dir} "
        f"pass={len(summary_rows) - len(fail_rows)} fail={len(fail_rows)} "
        f"summary={summary_path} hypotheses={detail_path}",
        flush=True,
    )
    return 1 if fail_rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
