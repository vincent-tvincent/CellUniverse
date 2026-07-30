from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np


BACKGROUND_ESTIMATION_STAGE_UNCOMPUTED = "uncomputed"
FINAL_BACKGROUND_ESTIMATION_STAGE = "interpolated_fitted_cells_excluded"


class Step(IntEnum):
    MEMBRANE = 0
    CELL_REGIONS = 1
    # Transitional alias for callers written against the first wizard draft.
    CELL_EDGES = CELL_REGIONS
    Z_CALIBRATION = 2
    CELL_FITTING = 3
    REVIEW = 4


@dataclass
class EllipsoidModel:
    center_xyz: np.ndarray
    radii_abc: np.ndarray
    rotation_xyz: np.ndarray
    brightness: float = 0.5
    cell_id: int | None = None
    name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "center_xyz": [float(v) for v in self.center_xyz],
            "radii_abc": [float(v) for v in self.radii_abc],
            "rotation_xyz": [float(v) for v in self.rotation_xyz],
            "brightness": float(self.brightness),
            "cell_id": self.cell_id,
            "name": self.name,
        }


@dataclass
class MembraneParameters:
    upper_clamp: float
    dog_sigma_low: float = 1.0
    dog_sigma_high: float = 3.0
    log_sigma: float = 2.0
    fusion_weight: float = 0.5
    edge_threshold: float = 0.30
    min_component_size: int = 500
    closing_radius: int = 2
    soft_margin: float = 0.05


@dataclass
class CellRegionParameters:
    dog_sigma_low: float = 0.8
    dog_sigma_high: float = 2.0
    log_sigma: float = 1.2
    fusion_weight: float = 0.5
    # The intensity mask and connected high-intensity cores are the primary
    # region/marker sources. The fused-edge threshold selects soft watershed
    # boundary guidance; it never removes pixels from the filled foreground.
    # Explicit normalized thresholds override the adaptive quantiles when set.
    foreground_threshold: float | None = None
    center_threshold: float | None = None
    foreground_quantile: float = 99.0
    center_quantile: float = 99.70
    edge_threshold: float = 0.70
    # ``min_component_size`` is retained for compatibility and controls
    # removal of tiny boundary fragments. ``min_chunk_area_2d`` retains its
    # historical field name, but is interpreted as connected 3D raw-voxel
    # volume so a legitimate cell is not removed slice by slice.
    min_component_size: int = 8
    closing_radius: int = 1
    min_chunk_area_2d: int = 25
    min_center_core_volume: int = 300
    center_min_distance: int = 5
    center_min_radius: float = 1.5
    center_threshold_rel: float = 0.35
    smoothing_sigma_z: float = 0.5
    smoothing_sigma_xy: float = 1.2
    edge_elevation_weight: float = 0.20


# Compatibility name used by the first wizard draft.
CellEdgeParameters = CellRegionParameters


@dataclass
class GroupingParameters:
    max_centroid_distance: float = 18.0
    max_missing_slices: int = 1
    min_iou: float = 0.01
    min_slice_area: int = 25
    min_slices_per_cell: int = 1


@dataclass
class SegmentationParameters:
    intensity_threshold: float = 0.18
    closing_radius: int = 1
    connectivity: int = 26
    roi_margin: int = 4


@dataclass
class InitializerSession:
    source_path: Path
    output_dir: Path
    raw: np.ndarray
    source_fingerprint: str
    current_step: Step = Step.MEMBRANE
    membrane_params: MembraneParameters | None = None
    cell_edge_params: CellRegionParameters = field(default_factory=CellRegionParameters)
    grouping_params: GroupingParameters = field(default_factory=GroupingParameters)
    segmentation_params: SegmentationParameters = field(default_factory=SegmentationParameters)
    membrane_mask: np.ndarray | None = None
    membrane_ellipsoid_raw: EllipsoidModel | None = None
    membrane_ellipsoid: EllipsoidModel | None = None
    cold_background: float = 0.0
    hot_background: float = 0.0
    # Source-space estimates may populate the brightness values for diagnostic
    # previews, but only the finalized interpolated, cell-excluded estimate is
    # valid for export.
    background_estimation_stage: str = BACKGROUND_ESTIMATION_STAGE_UNCOMPUTED
    background_statistics: dict[str, Any] = field(default_factory=dict)
    # Step 2 result in the original, uninterpolated Z coordinate space.
    cell_labels_raw: np.ndarray | None = None
    cell_centers_xyz: dict[int, np.ndarray] = field(default_factory=dict)
    cell_center_diagnostics: dict[int, dict[str, Any]] = field(
        default_factory=dict
    )
    cell_slice_centers_zyx: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32)
    )
    cell_slice_center_ids: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.int32)
    )
    cell_slice_center_accepted: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=bool)
    )
    # Step 3 result in the reconstructed/interpolated Z coordinate space.
    interpolated_cell_labels: np.ndarray | None = None
    interpolated_membrane_mask: np.ndarray | None = None
    interpolated_cell_edges: np.ndarray | None = None
    interpolated_centers_xyz: dict[int, np.ndarray] = field(default_factory=dict)
    interpolated_center_diagnostics: dict[int, dict[str, Any]] = field(
        default_factory=dict
    )
    # Step 4 starts from the unrefined interpolated-label fits. Brightness
    # offsets are keyed by the positive source label ID so they remain stable
    # independently of display/export list ordering.
    baseline_cell_ellipsoids: list[EllipsoidModel] = field(default_factory=list)
    cell_brightness_offsets: dict[int, float] = field(default_factory=dict)
    # Legacy draft fields remain during migration so older saved sessions and
    # call sites fail soft instead of losing user work.
    accepted_edges: np.ndarray | None = None
    reviewed_slices: np.ndarray | None = None
    z_ratio: float = 1.0
    interpolated_raw: np.ndarray | None = None
    group_labels_raw: np.ndarray | None = None
    group_seed_xyz: dict[int, np.ndarray] = field(default_factory=dict)
    cell_regions: np.ndarray | None = None
    cell_ellipsoids: list[EllipsoidModel] = field(default_factory=list)
    accepted_steps: set[int] = field(default_factory=set)
    revision: int = 0

    def invalidate_after(self, step: Step) -> None:
        self.accepted_steps = {value for value in self.accepted_steps if value <= int(step)}
        if step < Step.CELL_REGIONS:
            self.cell_labels_raw = None
            self.cell_centers_xyz.clear()
            self.cell_center_diagnostics.clear()
            self.cell_slice_centers_zyx = np.empty((0, 3), dtype=np.float32)
            self.cell_slice_center_ids = np.empty((0,), dtype=np.int32)
            self.cell_slice_center_accepted = np.empty((0,), dtype=bool)
            self.accepted_edges = None
            self.reviewed_slices = None
        if step < Step.Z_CALIBRATION:
            self.interpolated_raw = None
            self.interpolated_cell_labels = None
            self.interpolated_membrane_mask = None
            self.interpolated_cell_edges = None
            self.interpolated_centers_xyz.clear()
            self.interpolated_center_diagnostics.clear()
            self.group_labels_raw = None
            self.group_seed_xyz.clear()
            self.cell_regions = None
        if step < Step.CELL_FITTING:
            # The export-space background fit is produced alongside the cell
            # fits. Keep the raw-Z diagnostic model separately, but never carry
            # a scaled background ellipsoid across an upstream edit.
            self.membrane_ellipsoid = None
            self.baseline_cell_ellipsoids.clear()
            self.cell_brightness_offsets.clear()
            self.cell_ellipsoids.clear()
            self.background_estimation_stage = (
                BACKGROUND_ESTIMATION_STAGE_UNCOMPUTED
            )
            self.background_statistics.clear()
        self.revision += 1

    def mark_accepted(self, step: Step) -> None:
        self.accepted_steps.add(int(step))
        self.revision += 1
