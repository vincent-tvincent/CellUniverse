#!/usr/bin/env python3
"""Audit CellUniverse embryo centers and lineage topology against fixed GT.

This script is a review tool only. It never writes tracker inputs and never
feeds GT information back into CellUniverse. The topology audit first matches
predicted centers to fixed GT labels, then checks whether predicted parent
branches map to the same GT parent branches. It deliberately does not require
literal CellUniverse names to match GT label strings.
"""

from __future__ import annotations

import argparse
import csv
import heapq
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from validate_embryo_centers import (  # noqa: E402
    point_distance,
    read_prediction,
)


DEFAULT_GT = (
    "/Users/wangyiding/CellUniverse/C++/config/C.elegans developing embryo/"
    "C.elegans_initial/ground_truth/embryo_FixedGroundTruth.csv"
)


def parse_frames(text: str) -> list[int]:
    frames: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            frames.extend(range(int(start), int(end) + 1))
        else:
            frames.append(int(part))
    return sorted(set(frames))


def read_gt_with_life(path: Path, frame: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["frame"]) != frame:
                continue
            rows.append(
                {
                    "label": row["label_id"],
                    "parent": row["parent_label"],
                    "start_frame": int(row["start_frame"]),
                    "end_frame": int(row["end_frame"]),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "z": float(row["z_interp"]),
                }
            )
    return rows


def predicted_parent_name(name: str) -> str | None:
    if "_" not in name:
        return None
    prefix, lineage_code = name.rsplit("_", 1)
    if len(lineage_code) <= 1:
        return None
    return f"{prefix}_{lineage_code[:-1]}"


def matched_maps(
    pred: list[dict[str, object]],
    gt: list[dict[str, object]],
    threshold: float,
) -> dict[str, object]:
    # Dense late frames can contain several plausible centers within the audit
    # threshold. A greedy nearest-edge assignment can report false missing/extra
    # cells just because one early local choice steals a neighbor. Use a small
    # min-cost max-flow matcher so the audit measures the best global one-to-one
    # center assignment before checking lineage topology.
    edges: list[tuple[int, int, int]] = []
    for pred_index, pred_row in enumerate(pred):
        for gt_index, gt_row in enumerate(gt):
            distance = point_distance(pred_row, gt_row)
            if distance <= threshold:
                edges.append((int(round(distance * 1000.0)), pred_index, gt_index))

    pair_pred, pair_gt = global_min_cost_assignment(len(pred), len(gt), edges)

    matched = sum(1 for gt_index in pair_pred if gt_index != -1)
    distances = [
        point_distance(pred[pred_index], gt[gt_index])
        for pred_index, gt_index in enumerate(pair_pred)
        if gt_index != -1
    ]
    pred_to_gt = {
        str(pred[pred_index]["name"]): str(gt[gt_index]["label"])
        for pred_index, gt_index in enumerate(pair_pred)
        if gt_index != -1
    }
    gt_to_pred = {
        str(gt[gt_index]["label"]): str(pred[pred_index]["name"])
        for gt_index, pred_index in enumerate(pair_gt)
        if pred_index != -1
    }
    missing = [str(gt[index]["label"]) for index, partner in enumerate(pair_gt) if partner == -1]
    extra = [str(pred[index]["name"]) for index, partner in enumerate(pair_pred) if partner == -1]
    return {
        "pair_pred": pair_pred,
        "pair_gt": pair_gt,
        "matched": matched,
        "pred_to_gt": pred_to_gt,
        "gt_to_pred": gt_to_pred,
        "missing": missing,
        "extra": extra,
        "mean_distance": sum(distances) / len(distances) if distances else 0.0,
        "max_distance": max(distances) if distances else 0.0,
    }


def global_min_cost_assignment(
    pred_count: int,
    gt_count: int,
    candidate_edges: list[tuple[int, int, int]],
) -> tuple[list[int], list[int]]:
    class Edge:
        __slots__ = ("to", "rev", "cap", "cost")

        def __init__(self, to: int, rev: int, cap: int, cost: int) -> None:
            self.to = to
            self.rev = rev
            self.cap = cap
            self.cost = cost

    total_nodes = 1 + pred_count + gt_count + 1
    source = 0
    pred_base = 1
    gt_base = pred_base + pred_count
    sink = total_nodes - 1
    graph: list[list[Edge]] = [[] for _ in range(total_nodes)]

    def add_edge(src: int, dst: int, cap: int, cost: int) -> None:
        graph[src].append(Edge(dst, len(graph[dst]), cap, cost))
        graph[dst].append(Edge(src, len(graph[src]) - 1, 0, -cost))

    for pred_index in range(pred_count):
        add_edge(source, pred_base + pred_index, 1, 0)
    for gt_index in range(gt_count):
        add_edge(gt_base + gt_index, sink, 1, 0)
    for cost, pred_index, gt_index in candidate_edges:
        add_edge(pred_base + pred_index, gt_base + gt_index, 1, cost)

    potentials = [0] * total_nodes
    while True:
        dist = [10**18] * total_nodes
        prev_node = [-1] * total_nodes
        prev_edge = [-1] * total_nodes
        dist[source] = 0
        heap: list[tuple[int, int]] = [(0, source)]
        while heap:
            cur_dist, node = heapq.heappop(heap)
            if cur_dist != dist[node]:
                continue
            for edge_index, edge in enumerate(graph[node]):
                if edge.cap <= 0:
                    continue
                next_dist = cur_dist + edge.cost + potentials[node] - potentials[edge.to]
                if next_dist < dist[edge.to]:
                    dist[edge.to] = next_dist
                    prev_node[edge.to] = node
                    prev_edge[edge.to] = edge_index
                    heapq.heappush(heap, (next_dist, edge.to))

        if prev_node[sink] == -1:
            break
        for node in range(total_nodes):
            if dist[node] < 10**18:
                potentials[node] += dist[node]

        node = sink
        while node != source:
            src = prev_node[node]
            edge = graph[src][prev_edge[node]]
            edge.cap -= 1
            graph[node][edge.rev].cap += 1
            node = src

    pair_pred = [-1] * pred_count
    pair_gt = [-1] * gt_count
    for pred_index in range(pred_count):
        pred_node = pred_base + pred_index
        for edge in graph[pred_node]:
            if gt_base <= edge.to < gt_base + gt_count and edge.cap == 0:
                gt_index = edge.to - gt_base
                pair_pred[pred_index] = gt_index
                pair_gt[gt_index] = pred_index
                break
    return pair_pred, pair_gt


def gt_by_label(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {str(row["label"]): row for row in rows}


def audit_topology(
    frame: int,
    current_gt: list[dict[str, object]],
    current_maps: dict[str, object],
    previous_maps: dict[str, object] | None,
) -> list[str]:
    if previous_maps is None:
        return []

    errors: list[str] = []
    pred_to_gt_prev: dict[str, str] = previous_maps["pred_to_gt"]  # type: ignore[assignment]
    pred_to_gt_cur: dict[str, str] = current_maps["pred_to_gt"]  # type: ignore[assignment]
    gt_to_pred_cur: dict[str, str] = current_maps["gt_to_pred"]  # type: ignore[assignment]
    gt_rows = gt_by_label(current_gt)

    for gt_label, pred_name in sorted(gt_to_pred_cur.items(), key=lambda item: int(item[0])):
        gt_row = gt_rows[gt_label]
        parent_label = str(gt_row.get("parent", "0"))
        start_frame = int(gt_row.get("start_frame", frame))
        predicted_parent = predicted_parent_name(pred_name)
        mapped_parent = pred_to_gt_prev.get(predicted_parent or "")

        if start_frame == frame and parent_label not in {"", "0"}:
            if mapped_parent != parent_label:
                errors.append(
                    "birth_parent_mismatch:"
                    f"gt={gt_label}:gt_parent={parent_label}:"
                    f"pred={pred_name}:pred_parent={predicted_parent or 'NONE'}:"
                    f"pred_parent_gt={mapped_parent or 'NONE'}"
                )
            continue

        previous_same_name_gt = pred_to_gt_prev.get(pred_name)
        if previous_same_name_gt == gt_label:
            continue
        if predicted_parent and pred_to_gt_prev.get(predicted_parent) == gt_label:
            errors.append(
                "unexpected_split_for_continuing_gt:"
                f"gt={gt_label}:pred={pred_name}:pred_parent={predicted_parent}"
            )
        elif previous_same_name_gt and previous_same_name_gt != gt_label:
            errors.append(
                "identity_swap:"
                f"pred={pred_name}:prev_gt={previous_same_name_gt}:current_gt={gt_label}"
            )
        else:
            errors.append(
                "unexpected_new_prediction_for_continuing_gt:"
                f"gt={gt_label}:pred={pred_name}:pred_parent={predicted_parent or 'NONE'}"
            )

    return errors


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame",
        "center_status",
        "lineage_status",
        "pred",
        "gt",
        "matched",
        "missing",
        "extra",
        "mean_distance",
        "max_distance",
        "missing_labels",
        "extra_names",
        "lineage_errors",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--frames", required=True)
    parser.add_argument("--gt", type=Path, default=Path(DEFAULT_GT))
    parser.add_argument("--threshold", type=float, default=25.0)
    parser.add_argument(
        "--previous-cells",
        type=Path,
        help="Optional cells.csv that contains the frame immediately before the first audited frame.",
    )
    parser.add_argument(
        "--previous-frame",
        type=int,
        help="Frame number to read from --previous-cells for resume topology auditing.",
    )
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--details", action="store_true")
    parser.add_argument("--stop-on-fail", action="store_true")
    args = parser.parse_args()

    frames = parse_frames(args.frames)
    previous_maps: dict[str, object] | None = None
    if args.previous_cells is not None or args.previous_frame is not None:
        if args.previous_cells is None or args.previous_frame is None:
            parser.error("--previous-cells and --previous-frame must be provided together")
        previous_pred = read_prediction(args.previous_cells, args.previous_frame)
        previous_gt = read_gt_with_life(args.gt, args.previous_frame)
        previous_maps = matched_maps(previous_pred, previous_gt, args.threshold)
        previous_missing = previous_maps["missing"]
        previous_extra = previous_maps["extra"]
        print(
            f"previous f{args.previous_frame:03d} "
            f"center={'PASS' if not previous_missing and not previous_extra else 'FAIL'} "
            f"pred={len(previous_pred)} gt={len(previous_gt)} "
            f"matched={previous_maps['matched']} "
            f"missing={len(previous_missing)} extra={len(previous_extra)}",
            flush=True,
        )
    rows: list[dict[str, object]] = []

    for frame in frames:
        pred = read_prediction(args.cells, frame)
        gt = read_gt_with_life(args.gt, frame)
        maps = matched_maps(pred, gt, args.threshold)
        missing = maps["missing"]
        extra = maps["extra"]
        center_status = "PASS" if not missing and not extra else "FAIL"
        lineage_errors = audit_topology(frame, gt, maps, previous_maps)
        lineage_status = "PASS" if not lineage_errors else "FAIL"
        row = {
            "frame": frame,
            "center_status": center_status,
            "lineage_status": lineage_status,
            "pred": len(pred),
            "gt": len(gt),
            "matched": maps["matched"],
            "missing": len(missing),
            "extra": len(extra),
            "mean_distance": f"{float(maps['mean_distance']):.6f}",
            "max_distance": f"{float(maps['max_distance']):.6f}",
            "missing_labels": ",".join(missing),
            "extra_names": ",".join(extra),
            "lineage_errors": " | ".join(lineage_errors),
        }
        rows.append(row)

        print(
            f"f{frame:03d} center={center_status} lineage={lineage_status} "
            f"pred={len(pred)} gt={len(gt)} matched={maps['matched']} "
            f"missing={len(missing)} extra={len(extra)} "
            f"mean={float(maps['mean_distance']):.3f} "
            f"max={float(maps['max_distance']):.3f}",
            flush=True,
        )
        if args.details:
            if missing:
                print(f"  missing_labels={','.join(missing)}")
            if extra:
                print(f"  extra_names={','.join(extra)}")
            for error in lineage_errors[:12]:
                print(f"  lineage_error={error}")
            if len(lineage_errors) > 12:
                print(f"  lineage_error=...+{len(lineage_errors) - 12}")

        if args.summary_csv:
            write_summary(args.summary_csv, rows)
        if args.stop_on_fail and (center_status != "PASS" or lineage_status != "PASS"):
            return 1

        previous_maps = maps

    all_ok = all(row["center_status"] == "PASS" and row["lineage_status"] == "PASS" for row in rows)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
