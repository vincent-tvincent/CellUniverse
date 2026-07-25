#!/usr/bin/env python3
"""Audit a true temporal graph MIP selector for Cell Lumen candidates.

This is an experimental research script. It does not change CellUniverse C++,
the AutoDensity Best YAML, PCA fitting, or the existing rolling Cell Lumen
baseline. Unlike cell_lumen_window_ilp_audit.py, this script does not commit one
frame at a time. It builds one small temporal graph over all requested frames:

* candidate node variables choose Cell Lumen raw/BioFuse centers;
* link variables connect selected candidates across adjacent frames;
* appearance/disappearance slack variables keep the graph feasible when a local
  candidate or edge is missing, but charge a biological continuity penalty;
* division variables charge extra cost when one selected node links to two
  selected next-frame nodes;
* conflict constraints prevent duplicate centers in the same frame.

The goal is to test the Ultrack-style idea that temporal consistency should be
solved as a joint graph problem while preserving Cell Lumen's original
brightness-based high-recall candidate generator.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cell_lumen_window_ilp_audit import (  # type: ignore
    Cell,
    DEFAULT_CACHE_RUN,
    DEFAULT_GT,
    DEFAULT_INITIAL,
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_ROOT,
    distance,
    greedy_match,
    load_gt,
    load_initial,
    load_raw_candidates,
    load_summary_candidates,
    parse_frames,
    support_radius,
    write_selected_csv,
)


def load_bridge_helpers():
    from export_fine_shape_masks_from_cells import (  # type: ignore
        CellRow as BridgeCellRow,
        bridge_slab_valley_ratio,
        load_tiff_stack,
    )

    return BridgeCellRow, bridge_slab_valley_ratio, load_tiff_stack


@dataclass(frozen=True)
class Node:
    frame: int
    index: int
    cell: Cell
    score: float


@dataclass(frozen=True)
class Edge:
    left: int
    right: int
    score: float
    dist: float


@dataclass(frozen=True)
class SourceEdge:
    source: int
    node: int
    score: float
    dist: float


@dataclass(frozen=True)
class DivisionPair:
    node: int
    edge_a: int
    edge_b: int
    valley_ratio: float
    gap_drop: float
    child_separation: float
    child_overlap: float
    parent_child_dist_a: float
    parent_child_dist_b: float
    motion_imbalance: float
    score: float


def to_bridge_cell(cell: Cell, frame: int):
    BridgeCellRow, _bridge_slab_valley_ratio, _load_tiff_stack = load_bridge_helpers()
    return BridgeCellRow(
        frame=f"t{frame:03d}",
        name=cell.name,
        x=cell.x,
        y=cell.y,
        z_scaled=cell.z,
        rx=cell.rx,
        ry=cell.ry,
        rz_scaled=cell.rz,
        z_mode_used="temporal_mip_scaled",
    )


def node_score(cell: Cell, args: argparse.Namespace) -> float:
    source_bonus = args.summary_candidate_bonus if cell.source == "summary" else 0.0
    return args.node_support_weight * min(args.node_support_clip, support_radius(cell)) + source_bonus


def edge_score(left: Cell, right: Cell, args: argparse.Namespace) -> tuple[float, float] | None:
    dist = distance(left, right)
    if dist > args.link_distance:
        return None
    score = args.link_base_reward - args.link_distance_weight * dist
    return score, dist


def source_edge_score(source: Cell, node: Cell, args: argparse.Namespace) -> tuple[float, float] | None:
    dist = distance(source, node)
    if dist > args.source_link_distance:
        return None
    score = args.source_link_base_reward - args.source_link_distance_weight * dist
    return score, dist


def build_division_pairs(
    args: argparse.Namespace,
    nodes: list[Node],
    edges: list[Edge],
    outgoing_edges: dict[int, list[int]],
) -> list[DivisionPair]:
    if not (args.use_division_bridge_pair_score or args.use_division_bridge_filter):
        return []

    _BridgeCellRow, bridge_slab_valley_ratio, load_tiff_stack = load_bridge_helpers()
    bridge_args = argparse.Namespace(
        prior_bridge_cross_radius=args.prior_bridge_cross_radius,
        prior_bridge_samples=args.prior_bridge_samples,
        z_scaling=args.z_scale,
    )
    stack_cache: dict[int, object] = {}
    pairs: list[DivisionPair] = []
    for node_index, edge_indices in outgoing_edges.items():
        if len(edge_indices) < 2:
            continue
        for first_pos, edge_a in enumerate(edge_indices):
            right_a = nodes[edges[edge_a].right]
            for edge_b in edge_indices[first_pos + 1 :]:
                right_b = nodes[edges[edge_b].right]
                if right_a.frame != right_b.frame:
                    continue
                stack = stack_cache.get(right_a.frame)
                if stack is None:
                    stack_path = args.input_dir / f"t{right_a.frame:03d}.tif"
                    stack = load_tiff_stack(stack_path)
                    stack_cache[right_a.frame] = stack
                left_bridge = to_bridge_cell(right_a.cell, right_a.frame)
                right_bridge = to_bridge_cell(right_b.cell, right_b.frame)
                valley_ratio, gap_drop = bridge_slab_valley_ratio(stack, left_bridge, right_bridge, bridge_args)
                child_separation = distance(right_a.cell, right_b.cell)
                radius_a = (right_a.cell.rx + right_a.cell.ry + right_a.cell.rz) / 3.0
                radius_b = (right_b.cell.rx + right_b.cell.ry + right_b.cell.rz) / 3.0
                child_overlap = max(0.0, radius_a + radius_b - child_separation)
                parent = nodes[node_index]
                parent_child_dist_a = distance(parent.cell, right_a.cell)
                parent_child_dist_b = distance(parent.cell, right_b.cell)
                motion_imbalance = abs(parent_child_dist_a - parent_child_dist_b)
                dark_bonus = max(0.0, args.division_bridge_target_valley - valley_ratio)
                bright_penalty = max(0.0, valley_ratio - args.division_bridge_target_valley)
                score = (
                    args.division_bridge_bonus_weight * dark_bonus
                    + args.division_bridge_gap_weight * max(0.0, gap_drop)
                    - args.division_bridge_penalty_weight * bright_penalty
                    - args.division_pair_overlap_penalty_weight * child_overlap
                    - args.division_pair_motion_imbalance_penalty_weight * motion_imbalance
                )
                pairs.append(
                    DivisionPair(
                        node=node_index,
                        edge_a=edge_a,
                        edge_b=edge_b,
                        valley_ratio=valley_ratio,
                        gap_drop=gap_drop,
                        child_separation=child_separation,
                        child_overlap=child_overlap,
                        parent_child_dist_a=parent_child_dist_a,
                        parent_child_dist_b=parent_child_dist_b,
                        motion_imbalance=motion_imbalance,
                        score=score,
                    )
                )
    return pairs


def load_candidates_for_frames(args: argparse.Namespace, frames: list[int]) -> dict[int, list[Cell]]:
    by_frame: dict[int, list[Cell]] = {}
    for frame in frames:
        raw_path = args.cache_run_dir / f"f{frame:03d}" / f"t{frame:03d}_cell_lumen_independent.csv"
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing raw candidate cache: {raw_path}")
        candidates = load_raw_candidates(raw_path, args.z_scale)
        summary_path = (
            args.cache_run_dir
            / f"f{frame:03d}"
            / "fine_shape_biofused"
            / f"t{frame:03d}_fine_shape_summary.csv"
        )
        if args.include_summary_candidates and summary_path.exists():
            candidates += load_summary_candidates(summary_path)
        by_frame[frame] = candidates
    return by_frame


def build_graph(
    args: argparse.Namespace,
    frames: list[int],
    candidates_by_frame: dict[int, list[Cell]],
    sources: list[Cell],
) -> tuple[list[Node], list[Edge], list[SourceEdge], dict[int, list[int]]]:
    nodes: list[Node] = []
    node_indices_by_frame: dict[int, list[int]] = {frame: [] for frame in frames}
    for frame in frames:
        scored: list[Node] = []
        for candidate in candidates_by_frame[frame]:
            scored.append(Node(frame=frame, index=-1, cell=candidate, score=node_score(candidate, args)))
        scored.sort(key=lambda node: node.score, reverse=True)
        for local_index, node in enumerate(scored[: args.max_candidates_per_frame]):
            global_index = len(nodes)
            nodes.append(Node(frame=frame, index=local_index, cell=node.cell, score=node.score))
            node_indices_by_frame[frame].append(global_index)

    edges: list[Edge] = []
    for left_frame, right_frame in zip(frames, frames[1:]):
        for left_index in node_indices_by_frame[left_frame]:
            left_node = nodes[left_index]
            for right_index in node_indices_by_frame[right_frame]:
                right_node = nodes[right_index]
                scored = edge_score(left_node.cell, right_node.cell, args)
                if scored is None:
                    continue
                score, dist = scored
                edges.append(Edge(left=left_index, right=right_index, score=score, dist=dist))

    source_edges: list[SourceEdge] = []
    first_frame = frames[0]
    for source_index, source in enumerate(sources):
        for node_index in node_indices_by_frame[first_frame]:
            scored = source_edge_score(source, nodes[node_index].cell, args)
            if scored is None:
                continue
            score, dist = scored
            source_edges.append(SourceEdge(source=source_index, node=node_index, score=score, dist=dist))
    return nodes, edges, source_edges, node_indices_by_frame


def solve_temporal_mip(
    args: argparse.Namespace,
    frames: list[int],
    nodes: list[Node],
    edges: list[Edge],
    source_edges: list[SourceEdge],
    node_indices_by_frame: dict[int, list[int]],
    source_count: int,
) -> tuple[set[int], set[int], set[int], set[int], list[DivisionPair]]:
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    first_frame = frames[0]
    last_frame = frames[-1]

    incoming_edges: dict[int, list[int]] = {i: [] for i in range(len(nodes))}
    outgoing_edges: dict[int, list[int]] = {i: [] for i in range(len(nodes))}
    for edge_index, edge in enumerate(edges):
        outgoing_edges[edge.left].append(edge_index)
        incoming_edges[edge.right].append(edge_index)
    incoming_sources: dict[int, list[int]] = {i: [] for i in range(len(nodes))}
    source_outgoing: dict[int, list[int]] = {i: [] for i in range(source_count)}
    for source_edge_index, source_edge in enumerate(source_edges):
        incoming_sources[source_edge.node].append(source_edge_index)
        source_outgoing[source_edge.source].append(source_edge_index)

    division_pairs = build_division_pairs(args, nodes, edges, outgoing_edges)
    division_pairs_by_node: dict[int, list[int]] = {i: [] for i in range(len(nodes))}
    for pair_index, pair in enumerate(division_pairs):
        division_pairs_by_node[pair.node].append(pair_index)

    node_offset = 0
    edge_offset = len(nodes)
    source_edge_offset = edge_offset + len(edges)
    appear_offset = source_edge_offset + len(source_edges)
    disappear_offset = appear_offset + len(nodes)
    division_offset = disappear_offset + len(nodes)
    division_pair_offset = division_offset + len(nodes)
    var_count = division_pair_offset + len(division_pairs)

    objective = np.zeros(var_count, dtype=float)
    for i, node in enumerate(nodes):
        objective[node_offset + i] = -node.score
    for i, edge in enumerate(edges):
        objective[edge_offset + i] = -edge.score
    for i, source_edge in enumerate(source_edges):
        objective[source_edge_offset + i] = -source_edge.score
    for i, node in enumerate(nodes):
        if node.frame != first_frame:
            objective[appear_offset + i] = args.appear_penalty
        if node.frame != last_frame:
            objective[disappear_offset + i] = args.disappear_penalty
    for i, _node in enumerate(nodes):
        objective[division_offset + i] = args.division_penalty
    for i, pair in enumerate(division_pairs):
        objective[division_pair_offset + i] = -pair.score

    rows: list[dict[int, float]] = []
    lower: list[float] = []
    upper: list[float] = []

    def add_constraint(coeffs: dict[int, float], lo: float, hi: float) -> None:
        rows.append(coeffs)
        lower.append(lo)
        upper.append(hi)

    # Each initial root must choose exactly one candidate in the first frame.
    for source_index in range(source_count):
        coeffs = {
            source_edge_offset + edge_index: 1.0
            for edge_index in source_outgoing[source_index]
        }
        add_constraint(coeffs, 1.0, 1.0)

    # A node is selected exactly when it has one incoming source/track edge, or
    # pays an appearance slack. This mirrors Ultrack's flow model and prevents a
    # single missing local link from making the whole temporal graph infeasible.
    for node_index, node in enumerate(nodes):
        coeffs = {node_offset + node_index: -1.0}
        if node.frame == first_frame:
            for source_edge_index in incoming_sources[node_index]:
                coeffs[source_edge_offset + source_edge_index] = 1.0
        else:
            for edge_index in incoming_edges[node_index]:
                coeffs[edge_offset + edge_index] = 1.0
            coeffs[appear_offset + node_index] = 1.0
        add_constraint(coeffs, 0.0, 0.0)

    # Selected nonterminal nodes either continue to one next-frame node,
    # disappear with a slack penalty, or divide into two outgoing links and pay
    # the division penalty.
    for node_index, node in enumerate(nodes):
        if node.frame == last_frame:
            continue
        outgoing = outgoing_edges[node_index]
        # out + disappear = x + division
        coeffs_div = {
            node_offset + node_index: -1.0,
            division_offset + node_index: -1.0,
            disappear_offset + node_index: 1.0,
        }
        for edge_index in outgoing:
            coeffs_div[edge_offset + edge_index] = 1.0
        add_constraint(coeffs_div, 0.0, 0.0)
        # out <= 2x
        coeffs_max = {node_offset + node_index: -2.0}
        for edge_index in outgoing:
            coeffs_max[edge_offset + edge_index] = 1.0
        add_constraint(coeffs_max, -float("inf"), 0.0)
        # division <= x
        add_constraint(
            {
                division_offset + node_index: 1.0,
                node_offset + node_index: -1.0,
            },
            -float("inf"),
            0.0,
        )
        # disappear <= x
        add_constraint(
            {
                disappear_offset + node_index: 1.0,
                node_offset + node_index: -1.0,
            },
            -float("inf"),
            0.0,
        )
        # A selected cell can either divide or disappear, not both. Without this
        # conservation term the MIP can pay a division penalty while also using
        # disappearance slack, which is not a valid biological event.
        add_constraint(
            {
                disappear_offset + node_index: 1.0,
                division_offset + node_index: 1.0,
                node_offset + node_index: -1.0,
            },
            -float("inf"),
            0.0,
        )
    for node_index, node in enumerate(nodes):
        if node.frame == first_frame:
            # Initial source cells already explain the first frame; free
            # appearances here would let the MIP invent extra root tracks.
            add_constraint({appear_offset + node_index: 1.0}, 0.0, 0.0)
        if node.frame == last_frame:
            add_constraint({disappear_offset + node_index: 1.0}, 0.0, 0.0)

    filtered_division_pair_count = 0
    if division_pairs:
        # Pair variables make division evidence pair-specific: the graph does
        # not merely ask whether a parent has two outgoing links, it scores the
        # exact daughter pair by its 3D bridge valley.
        for pair_index, pair in enumerate(division_pairs):
            pair_var = division_pair_offset + pair_index
            edge_a_var = edge_offset + pair.edge_a
            edge_b_var = edge_offset + pair.edge_b
            add_constraint({pair_var: 1.0, edge_a_var: -1.0}, -float("inf"), 0.0)
            add_constraint({pair_var: 1.0, edge_b_var: -1.0}, -float("inf"), 0.0)
            add_constraint(
                {
                    pair_var: 1.0,
                    edge_a_var: -1.0,
                    edge_b_var: -1.0,
                },
                -1.0,
                float("inf"),
            )
            if args.use_division_bridge_filter and pair.valley_ratio > args.division_bridge_max_valley:
                add_constraint(
                    {
                        edge_a_var: 1.0,
                        edge_b_var: 1.0,
                    },
                    -float("inf"),
                    1.0,
                )
                filtered_division_pair_count += 1

        for node_index, node in enumerate(nodes):
            if node.frame == last_frame:
                continue
            coeffs = {division_offset + node_index: -1.0}
            for pair_index in division_pairs_by_node[node_index]:
                coeffs[division_pair_offset + pair_index] = 1.0
            add_constraint(coeffs, 0.0, 0.0)

    # Same-frame duplicate centers cannot both be selected.
    conflict_count = 0
    for frame in frames:
        frame_nodes = node_indices_by_frame[frame]
        for i, left_index in enumerate(frame_nodes):
            for right_index in frame_nodes[i + 1 :]:
                if distance(nodes[left_index].cell, nodes[right_index].cell) >= args.duplicate_distance:
                    continue
                add_constraint(
                    {
                        node_offset + left_index: 1.0,
                        node_offset + right_index: 1.0,
                    },
                    -float("inf"),
                    1.0,
                )
                conflict_count += 1

    matrix = lil_matrix((len(rows), var_count), dtype=float)
    for row_index, coeffs in enumerate(rows):
        for col_index, value in coeffs.items():
            matrix[row_index, col_index] = value

    result = milp(
        c=objective,
        integrality=np.ones(var_count, dtype=int),
        bounds=Bounds(0.0, 1.0),
        constraints=LinearConstraint(matrix.tocsr(), np.array(lower), np.array(upper)),
        options={
            "time_limit": args.solver_time_limit_sec,
            "mip_rel_gap": args.solver_mip_gap,
            "disp": False,
        },
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Temporal graph MILP failed: status={result.status} message={result.message}")

    selected_nodes = {i for i in range(len(nodes)) if result.x[node_offset + i] >= 0.5}
    selected_edges = {i for i in range(len(edges)) if result.x[edge_offset + i] >= 0.5}
    selected_divisions = {i for i in range(len(nodes)) if result.x[division_offset + i] >= 0.5}
    selected_division_pairs = {
        i for i in range(len(division_pairs)) if result.x[division_pair_offset + i] >= 0.5
    }
    selected_appearances = sum(1 for i in range(len(nodes)) if result.x[appear_offset + i] >= 0.5)
    selected_disappearances = sum(1 for i in range(len(nodes)) if result.x[disappear_offset + i] >= 0.5)
    print(
        f"[TemporalGraphMIP] nodes={len(nodes)} edges={len(edges)} source_edges={len(source_edges)} "
        f"division_pairs={len(division_pairs)} bridge_filtered_pairs={filtered_division_pair_count} "
        f"conflicts={conflict_count} vars={var_count} constraints={len(rows)} "
        f"appear={selected_appearances} disappear={selected_disappearances} "
        f"objective={-float(result.fun):.6g}",
        flush=True,
    )
    return selected_nodes, selected_edges, selected_divisions, selected_division_pairs, division_pairs


def output_name_for_node(node_index: int) -> str:
    return f"mip_cell_{node_index:05d}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-run-dir", type=Path, default=DEFAULT_CACHE_RUN)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--initial", type=Path, default=DEFAULT_INITIAL)
    parser.add_argument("--gt-csv", type=Path, default=DEFAULT_GT)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--frames", default="1-20")
    parser.add_argument("--threshold", type=float, default=25.0)
    parser.add_argument("--z-scale", type=float, default=7.0)
    parser.add_argument("--include-summary-candidates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-candidates-per-frame", type=int, default=80)
    parser.add_argument("--node-support-weight", type=float, default=2.0)
    parser.add_argument("--node-support-clip", type=float, default=26.0)
    parser.add_argument("--summary-candidate-bonus", type=float, default=12.0)
    parser.add_argument("--source-link-distance", type=float, default=70.0)
    parser.add_argument("--source-link-base-reward", type=float, default=48.0)
    parser.add_argument("--source-link-distance-weight", type=float, default=0.45)
    parser.add_argument("--link-distance", type=float, default=42.0)
    parser.add_argument("--link-base-reward", type=float, default=36.0)
    parser.add_argument("--link-distance-weight", type=float, default=0.55)
    parser.add_argument("--duplicate-distance", type=float, default=13.0)
    parser.add_argument("--division-penalty", type=float, default=72.0)
    parser.add_argument("--appear-penalty", type=float, default=90.0)
    parser.add_argument("--disappear-penalty", type=float, default=90.0)
    parser.add_argument("--use-division-bridge-pair-score", action="store_true")
    parser.add_argument("--use-division-bridge-filter", action="store_true")
    parser.add_argument("--division-bridge-target-valley", type=float, default=0.58)
    parser.add_argument("--division-bridge-max-valley", type=float, default=0.90)
    parser.add_argument("--division-bridge-bonus-weight", type=float, default=120.0)
    parser.add_argument("--division-bridge-penalty-weight", type=float, default=180.0)
    parser.add_argument("--division-bridge-gap-weight", type=float, default=0.02)
    parser.add_argument("--division-pair-overlap-penalty-weight", type=float, default=0.0)
    parser.add_argument("--division-pair-motion-imbalance-penalty-weight", type=float, default=0.0)
    parser.add_argument("--prior-bridge-cross-radius", type=float, default=4.0)
    parser.add_argument("--prior-bridge-samples", type=int, default=17)
    parser.add_argument("--initial-root-age", type=int, default=100)
    parser.add_argument("--solver-time-limit-sec", type=float, default=120.0)
    parser.add_argument("--solver-mip-gap", type=float, default=0.001)
    args = parser.parse_args()

    frames = parse_frames(args.frames)
    stamp = time.strftime("%Y%m%d.%H%M%S")
    run_name = args.run_name or (
        f"⚠️EXPERIMENT_{frames[0]:03d}-{frames[-1]:03d}_TemporalGraphMIP_{stamp}"
    )
    run_dir = args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    candidates_by_frame = load_candidates_for_frames(args, frames)
    sources = load_initial(args.initial, args.z_scale, args.initial_root_age)
    gt_by_frame = load_gt(args.gt_csv, frames)
    nodes, edges, source_edges, node_indices_by_frame = build_graph(args, frames, candidates_by_frame, sources)
    selected_nodes, selected_edges, selected_divisions, selected_division_pairs, division_pairs = solve_temporal_mip(
        args,
        frames,
        nodes,
        edges,
        source_edges,
        node_indices_by_frame,
        len(sources),
    )

    summary_rows: list[dict[str, str | int | float]] = []
    for frame in frames:
        frame_dir = run_dir / f"f{frame:03d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        selected_cells: list[Cell] = []
        for node_index in node_indices_by_frame[frame]:
            if node_index not in selected_nodes:
                continue
            node = nodes[node_index]
            selected_cells.append(
                Cell(
                    name=output_name_for_node(node_index),
                    x=node.cell.x,
                    y=node.cell.y,
                    z=node.cell.z,
                    rx=node.cell.rx,
                    ry=node.cell.ry,
                    rz=node.cell.rz,
                    source=f"temporal_mip:{node.cell.name}",
                )
            )
        selected_cells.sort(key=lambda cell: (cell.y, cell.x, cell.z, cell.name))
        selected_csv = frame_dir / f"t{frame:03d}_temporal_graph_mip_selected.csv"
        write_selected_csv(selected_csv, frame, selected_cells, args.z_scale)
        matches, missing, extra = greedy_match(gt_by_frame[frame], selected_cells, args.threshold)
        status = "PASS" if not missing and not extra else "FAIL"
        mean_dist = sum(item[0] for item in matches) / len(matches) if matches else 0.0
        max_dist = max((item[0] for item in matches), default=0.0)
        print(
            f"f{frame:03d} {status} pred={len(selected_cells)} gt={len(gt_by_frame[frame])} "
            f"match={len(matches)} miss={len(missing)} extra={len(extra)} "
            f"mean={mean_dist:.3f} max={max_dist:.3f}",
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
                "pred": len(selected_cells),
                "matched": len(matches),
                "missing": len(missing),
                "extra": len(extra),
                "mean_distance": f"{mean_dist:.9g}",
                "max_distance": f"{max_dist:.9g}",
                "selected_csv": str(selected_csv),
                "missing_labels": ";".join(str(row["label"]) for row in missing),
                "extra_names": ";".join(cell.name for cell in extra),
            }
        )

    summary_path = run_dir / f"temporal_graph_mip_summary_f{frames[0]:03d}_f{frames[-1]:03d}.csv"
    with summary_path.open("w", newline="") as handle:
        fieldnames = [
            "frame",
            "status",
            "gt",
            "pred",
            "matched",
            "missing",
            "extra",
            "mean_distance",
            "max_distance",
            "selected_csv",
            "missing_labels",
            "extra_names",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    edge_path = run_dir / f"temporal_graph_mip_selected_edges_f{frames[0]:03d}_f{frames[-1]:03d}.csv"
    with edge_path.open("w", newline="") as handle:
        fieldnames = ["left_frame", "left_node", "left_name", "right_frame", "right_node", "right_name", "dist", "division_left"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for edge_index in sorted(selected_edges):
            edge = edges[edge_index]
            left = nodes[edge.left]
            right = nodes[edge.right]
            writer.writerow(
                {
                    "left_frame": left.frame,
                    "left_node": edge.left,
                    "left_name": left.cell.name,
                    "right_frame": right.frame,
                    "right_node": edge.right,
                    "right_name": right.cell.name,
                    "dist": f"{edge.dist:.9g}",
                    "division_left": int(edge.left in selected_divisions),
                }
            )

    pair_path = run_dir / f"temporal_graph_mip_selected_division_pairs_f{frames[0]:03d}_f{frames[-1]:03d}.csv"
    with pair_path.open("w", newline="") as handle:
        fieldnames = [
            "left_node",
            "left_frame",
            "left_name",
            "right_a",
            "right_b",
            "valley_ratio",
            "gap_drop",
            "child_separation",
            "child_overlap",
            "parent_child_dist_a",
            "parent_child_dist_b",
            "motion_imbalance",
            "pair_score",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for pair_index in sorted(selected_division_pairs):
            pair = division_pairs[pair_index]
            left = nodes[pair.node]
            right_a = nodes[edges[pair.edge_a].right]
            right_b = nodes[edges[pair.edge_b].right]
            writer.writerow(
                {
                    "left_node": pair.node,
                    "left_frame": left.frame,
                    "left_name": left.cell.name,
                    "right_a": f"{right_a.frame}:{right_a.cell.name}",
                    "right_b": f"{right_b.frame}:{right_b.cell.name}",
                    "valley_ratio": f"{pair.valley_ratio:.9g}",
                    "gap_drop": f"{pair.gap_drop:.9g}",
                    "child_separation": f"{pair.child_separation:.9g}",
                    "child_overlap": f"{pair.child_overlap:.9g}",
                    "parent_child_dist_a": f"{pair.parent_child_dist_a:.9g}",
                    "parent_child_dist_b": f"{pair.parent_child_dist_b:.9g}",
                    "motion_imbalance": f"{pair.motion_imbalance:.9g}",
                    "pair_score": f"{pair.score:.9g}",
                }
            )

    fail_rows = [row for row in summary_rows if row["status"] != "PASS"]
    print(
        f"[TemporalGraphMIP Audit] run_dir={run_dir} "
        f"pass={len(summary_rows) - len(fail_rows)} fail={len(fail_rows)} "
        f"summary={summary_path} edges={edge_path}",
        flush=True,
    )
    return 1 if fail_rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
