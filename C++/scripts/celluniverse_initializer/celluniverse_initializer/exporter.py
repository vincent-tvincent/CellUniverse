from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .algorithms import (
    INTERPOLATED_DETACHED_SUPPORT_REVIEW_FRACTION,
    ROBUST_CELL_CENTER_COMPONENT_ALIGNMENT_RADIUS_FACTOR,
    ROBUST_CELL_CENTER_GEOMETRIC_FLOOR,
    ROBUST_CELL_CENTER_INTENSITY_PERCENTILES,
    ROBUST_CELL_CENTER_MAX_COMPONENT_GAP_SLICES,
    ROBUST_CELL_CENTER_METHOD,
    ROBUST_CELL_CENTER_MINIMUM_COMPONENT_ALIGNMENT_PIXELS,
    ROBUST_CELL_CENTER_MINIMUM_CONSENSUS_FRACTION,
    ROBUST_CELL_CENTER_THREE_SLICE_GROSS_MINIMUM_PIXELS,
    ROBUST_CELL_CENTER_THREE_SLICE_GROSS_RADIUS_FACTOR,
    ROBUST_CELL_CENTER_THREE_SLICE_RESIDUAL_MULTIPLIER,
    ROBUST_CELL_CENTER_WEIGHT_EXPONENT,
)
from .model import (
    FINAL_BACKGROUND_ESTIMATION_STAGE,
    EllipsoidModel,
    InitializerSession,
)


CSV_SCHEMA_VERSION = 2

CSV_COLUMNS = [
    "file",
    "name",
    "cellId",
    "x",
    "y",
    "z",
    "aRadius",
    "bRadius",
    "cRadius",
    "theta_x",
    "theta_y",
    "theta_z",
    "brightness",
    "coldBackgroundBrightness",
    "hotBackgroundBrightness",
    "isTrash",
    "isHotBackgroundRegion",
    "backgroundSoftMargin",
    "zInterpolationRatio",
    "zCoordinateSpace",
    "initializerSchemaVersion",
]


def fingerprint_file(path: Path, sample_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    stat = path.stat()
    digest.update(str(path.resolve()).encode())
    digest.update(str(stat.st_size).encode())
    digest.update(str(stat.st_mtime_ns).encode())
    with path.open("rb") as handle:
        digest.update(handle.read(sample_bytes))
    return digest.hexdigest()


def _row_for_model(
    source_name: str,
    model: EllipsoidModel,
    z_ratio: float,
    hot_region: bool,
    soft_margin: float | None,
    cold_background: float | None = None,
    hot_background: float | None = None,
) -> dict[str, Any]:
    return {
        "file": source_name,
        "name": model.name,
        "cellId": "" if model.cell_id is None else model.cell_id,
        "x": f"{model.center_xyz[0]:.9g}",
        "y": f"{model.center_xyz[1]:.9g}",
        "z": f"{model.center_xyz[2]:.9g}",
        "aRadius": f"{model.radii_abc[0]:.9g}",
        "bRadius": f"{model.radii_abc[1]:.9g}",
        "cRadius": f"{model.radii_abc[2]:.9g}",
        "theta_x": f"{model.rotation_xyz[0]:.9g}",
        "theta_y": f"{model.rotation_xyz[1]:.9g}",
        "theta_z": f"{model.rotation_xyz[2]:.9g}",
        "brightness": f"{model.brightness:.9g}",
        "coldBackgroundBrightness": (
            ""
            if cold_background is None
            else f"{float(cold_background):.9g}"
        ),
        "hotBackgroundBrightness": (
            ""
            if hot_background is None
            else f"{float(hot_background):.9g}"
        ),
        "isTrash": 0,
        "isHotBackgroundRegion": int(hot_region),
        "backgroundSoftMargin": (
            "" if soft_margin is None else f"{float(soft_margin):.9g}"
        ),
        "zInterpolationRatio": f"{float(z_ratio):.9g}",
        "zCoordinateSpace": "scaled",
        "initializerSchemaVersion": CSV_SCHEMA_VERSION,
    }


def validate_export(session: InitializerSession) -> None:
    if session.membrane_ellipsoid is None:
        raise ValueError("No confirmed membrane ellipsoid")
    if not session.cell_ellipsoids:
        raise ValueError("No fitted cell ellipsoids")
    if session.z_ratio <= 0:
        raise ValueError("Invalid Z interpolation ratio")
    if (
        session.background_estimation_stage
        != FINAL_BACKGROUND_ESTIMATION_STAGE
    ):
        raise ValueError(
            "Final background brightness is stale or missing: expected "
            f"estimation stage {FINAL_BACKGROUND_ESTIMATION_STAGE!r}, received "
            f"{session.background_estimation_stage!r}"
        )
    if not session.background_statistics:
        raise ValueError(
            "Final background statistics are missing; recompute the fitted-cell-"
            "excluded interpolated background before export"
        )
    for name, value in (
        ("cold", session.cold_background),
        ("hot", session.hot_background),
    ):
        if not np.isfinite(value) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(
                f"Final {name} background brightness must be finite and in [0, 1]"
            )
    membrane_brightness = float(session.membrane_ellipsoid.brightness)
    if (
        not np.isfinite(membrane_brightness)
        or not np.isclose(
            membrane_brightness,
            float(session.hot_background),
            rtol=0.0,
            atol=1e-9,
        )
    ):
        raise ValueError(
            "The hot background brightness must match the membrane ellipsoid "
            "brightness before export"
        )
    for model in [session.membrane_ellipsoid, *session.cell_ellipsoids]:
        values = np.concatenate(
            [model.center_xyz, model.radii_abc, model.rotation_xyz]
        )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Non-finite ellipsoid values for {model.name}")
        if np.any(model.radii_abc <= 0):
            raise ValueError(f"Non-positive ellipsoid radius for {model.name}")
        if not (
            model.radii_abc[0] >= model.radii_abc[1] >= model.radii_abc[2]
        ):
            raise ValueError(f"Radii are not ordered a>=b>=c for {model.name}")


def export_initial_csv(session: InitializerSession) -> tuple[Path, Path]:
    validate_export(session)
    output_dir = session.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "initial.csv"
    temporary = output_dir / ".initial.csv.tmp"
    source_name = session.source_path.name
    soft_margin = (
        session.membrane_params.soft_margin if session.membrane_params else 0.05
    )
    rows = [
        _row_for_model(
            source_name,
            session.membrane_ellipsoid,
            session.z_ratio,
            hot_region=True,
            soft_margin=soft_margin,
            cold_background=session.cold_background,
            hot_background=session.hot_background,
        )
    ]
    rows.extend(
        _row_for_model(
            source_name,
            model,
            session.z_ratio,
            hot_region=False,
            soft_margin=None,
        )
        for model in session.cell_ellipsoids
    )
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)

    provenance_path = output_dir / "initial.initializer.json"
    provenance_tmp = output_dir / ".initial.initializer.json.tmp"
    payload = session_to_json(session)
    with provenance_tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(provenance_tmp, provenance_path)
    return destination, provenance_path


def _center_candidates_to_json(session: InitializerSession) -> list[dict[str, Any]]:
    points = np.asarray(session.cell_slice_centers_zyx)
    label_ids = np.asarray(session.cell_slice_center_ids)
    accepted = np.asarray(session.cell_slice_center_accepted)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError(
            "Cell center candidate coordinates must have shape (N, 3) in ZYX order; "
            f"received {points.shape}"
        )
    if label_ids.ndim != 1:
        raise ValueError(
            "Cell center candidate label IDs must be one-dimensional; "
            f"received shape {label_ids.shape}"
        )
    if accepted.ndim != 1:
        raise ValueError(
            "Cell center candidate acceptance flags must be one-dimensional; "
            f"received shape {accepted.shape}"
        )
    row_count = int(points.shape[0])
    if label_ids.shape[0] != row_count or accepted.shape[0] != row_count:
        raise ValueError(
            "Cell center candidate state is inconsistent: coordinates, label IDs, "
            "and acceptance flags must have matching row counts "
            f"(received {row_count}, {label_ids.shape[0]}, and "
            f"{accepted.shape[0]})"
        )
    return [
        {
            "index": index,
            "zyx": [float(value) for value in points[index]],
            "label_id": int(label_ids[index]),
            "accepted": bool(accepted[index]),
        }
        for index in range(row_count)
    ]


def _json_compatible(value: Any) -> Any:
    """Convert structured audit metadata to JSON-native scalar containers."""
    if isinstance(value, np.ndarray):
        return [_json_compatible(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            str(key): _json_compatible(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def session_to_json(session: InitializerSession) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "initializer_version": __version__,
        "source": {
            "path": str(session.source_path),
            "fingerprint": session.source_fingerprint,
            "shape_zyx": list(session.raw.shape),
            "dtype": str(session.raw.dtype),
        },
        "output_dir": str(session.output_dir),
        "current_step": int(session.current_step),
        "accepted_steps": sorted(session.accepted_steps),
        "revision": session.revision,
        "center_candidates": _center_candidates_to_json(session),
        "cell_center_estimation": {
            "method": ROBUST_CELL_CENTER_METHOD,
            "parameters": {
                "intensity_percentiles": list(
                    ROBUST_CELL_CENTER_INTENSITY_PERCENTILES
                ),
                "weight_exponent": ROBUST_CELL_CENTER_WEIGHT_EXPONENT,
                "geometric_weight_floor": (
                    ROBUST_CELL_CENTER_GEOMETRIC_FLOOR
                ),
                "minimum_centerline_consensus_fraction": (
                    ROBUST_CELL_CENTER_MINIMUM_CONSENSUS_FRACTION
                ),
                "maximum_joined_component_gap_slices": (
                    ROBUST_CELL_CENTER_MAX_COMPONENT_GAP_SLICES
                ),
                "component_alignment_radius_factor": (
                    ROBUST_CELL_CENTER_COMPONENT_ALIGNMENT_RADIUS_FACTOR
                ),
                "minimum_component_alignment_pixels": (
                    ROBUST_CELL_CENTER_MINIMUM_COMPONENT_ALIGNMENT_PIXELS
                ),
                "three_slice_gross_minimum_pixels": (
                    ROBUST_CELL_CENTER_THREE_SLICE_GROSS_MINIMUM_PIXELS
                ),
                "three_slice_gross_radius_factor": (
                    ROBUST_CELL_CENTER_THREE_SLICE_GROSS_RADIUS_FACTOR
                ),
                "three_slice_residual_multiplier": (
                    ROBUST_CELL_CENTER_THREE_SLICE_RESIDUAL_MULTIPLIER
                ),
                "interpolated_detached_support_review_fraction": (
                    INTERPOLATED_DETACHED_SUPPORT_REVIEW_FRACTION
                ),
            },
            "source": _json_compatible(
                session.cell_center_diagnostics
            ),
            "interpolated": _json_compatible(
                session.interpolated_center_diagnostics
            ),
        },
        "z_interpolation": {
            "ratio": session.z_ratio,
            "coordinate_space": "scaled",
            "new_depth": (
                None
                if session.interpolated_raw is None
                else int(session.interpolated_raw.shape[0])
            ),
            "formula": "round((source_depth - 1) * ratio) + 1",
        },
        "background": {
            "cold": session.cold_background,
            "hot": session.hot_background,
            "estimation_stage": session.background_estimation_stage,
            "statistics": _json_compatible(session.background_statistics),
            "soft_margin": (
                None
                if session.membrane_params is None
                else session.membrane_params.soft_margin
            ),
            "ellipsoid": (
                None
                if session.membrane_ellipsoid is None
                else session.membrane_ellipsoid.to_dict()
            ),
        },
        "cell_fitting": {
            "brightness_offsets": {
                str(label_id): float(offset)
                for label_id, offset in sorted(
                    session.cell_brightness_offsets.items()
                )
            },
            "baseline_cell_ellipsoids": [
                model.to_dict() for model in session.baseline_cell_ellipsoids
            ],
        },
        "cell_ellipsoids": [model.to_dict() for model in session.cell_ellipsoids],
    }


def autosave_session(session: InitializerSession) -> Path:
    root = Path("/tmp/celluniverse-codex")
    task_root = root / f"initializer-{session.source_fingerprint[:12]}"
    task_root.mkdir(parents=True, exist_ok=True)
    destination = task_root / "session.json"
    temporary = task_root / ".session.json.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(session_to_json(session), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, destination)
    return destination
