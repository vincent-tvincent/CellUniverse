from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Iterable

import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import draw, feature, measure, morphology, segmentation

from .model import (
    CellEdgeParameters,
    EllipsoidModel,
    GroupingParameters,
    MembraneParameters,
    SegmentationParameters,
)


@dataclass
class EdgeResult:
    clamped: np.ndarray
    dog: np.ndarray
    log: np.ndarray
    fused: np.ndarray
    edges: np.ndarray
    regions: np.ndarray
    labels: np.ndarray
    component_count: int
    centers_xyz: dict[int, np.ndarray] = field(default_factory=dict)
    slice_centers_zyx: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32)
    )
    slice_center_ids: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.int32)
    )
    center_diagnostics: dict[int, dict[str, object]] = field(
        default_factory=dict
    )


# The result kept the original name so existing callers can migrate without a
# flag day. This semantic alias makes the new region-centric API explicit.
CellRegionResult = EdgeResult


@dataclass
class ManualSplitResult:
    labels: np.ndarray
    separator_mask: np.ndarray
    source_to_children: dict[int, tuple[int, int]]
    affected_slices: tuple[int, ...]


@dataclass
class ManualMergeResult:
    """Voxel-conservative merge of one source label into a chosen target."""

    labels: np.ndarray
    kept_label: int
    absorbed_label: int
    affected_slices: tuple[int, ...]


@dataclass
class MultiLabelMergeResult:
    """Voxel-conservative merge of an unordered set of cell labels."""

    labels: np.ndarray
    kept_label: int
    absorbed_labels: tuple[int, ...]
    affected_slices: tuple[int, ...]


@dataclass
class DetachedSupportSplitResult:
    """One-ID-per-cell repair with auditable tiny-child filtering."""

    labels: np.ndarray
    source_to_children: dict[int, tuple[int, ...]]
    affected_slices: tuple[int, ...]
    discarded_voxels_by_source: dict[int, int]
    discarded_group_count_by_source: dict[int, int]


@dataclass(frozen=True)
class RobustCellCenterEstimate:
    """Auditable chunk-derived center for one integer cell label."""

    label_id: int
    center_xyz: np.ndarray
    slice_centers_zyx: np.ndarray
    slice_weights: np.ndarray
    slice_inliers: np.ndarray
    slice_residuals_xy: np.ndarray
    residual_threshold: float
    projected_inside: bool
    center_in_inferred_gap: bool
    selected_component: int
    selected_components: tuple[int, ...]
    component_count: int
    ignored_component_voxels: int
    accepted_candidate_count: int
    method: str
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "label_id": int(self.label_id),
            "center_xyz": [
                float(value) for value in np.asarray(self.center_xyz)
            ],
            "slice_centers_zyx": np.asarray(
                self.slice_centers_zyx,
                dtype=np.float64,
            ).tolist(),
            "slice_weights": np.asarray(
                self.slice_weights,
                dtype=np.float64,
            ).tolist(),
            "slice_inliers": np.asarray(
                self.slice_inliers,
                dtype=bool,
            ).tolist(),
            "slice_residuals_xy": np.asarray(
                self.slice_residuals_xy,
                dtype=np.float64,
            ).tolist(),
            "residual_threshold": float(self.residual_threshold),
            "projected_inside": bool(self.projected_inside),
            "center_in_inferred_gap": bool(
                self.center_in_inferred_gap
            ),
            "selected_component": int(self.selected_component),
            "selected_components": [
                int(value) for value in self.selected_components
            ],
            "component_count": int(self.component_count),
            "ignored_component_voxels": int(
                self.ignored_component_voxels
            ),
            "accepted_candidate_count": int(
                self.accepted_candidate_count
            ),
            "method": self.method,
            "warnings": list(self.warnings),
        }


ROBUST_CELL_CENTER_METHOD = (
    "robust_intensity_weighted_slice_centerline_v1"
)
ROBUST_CELL_CENTER_INTENSITY_PERCENTILES = (20.0, 99.5)
ROBUST_CELL_CENTER_WEIGHT_EXPONENT = 1.5
ROBUST_CELL_CENTER_GEOMETRIC_FLOOR = 0.20
ROBUST_CELL_CENTER_MINIMUM_CONSENSUS_FRACTION = 0.60
ROBUST_CELL_CENTER_MAX_HYPOTHESIS_SLICES = 48
ROBUST_CELL_CENTER_MAX_COMPONENT_GAP_SLICES = 2
ROBUST_CELL_CENTER_COMPONENT_ALIGNMENT_RADIUS_FACTOR = 1.5
ROBUST_CELL_CENTER_MINIMUM_COMPONENT_ALIGNMENT_PIXELS = 4.0
ROBUST_CELL_CENTER_THREE_SLICE_GROSS_MINIMUM_PIXELS = 4.0
ROBUST_CELL_CENTER_THREE_SLICE_GROSS_RADIUS_FACTOR = 1.5
ROBUST_CELL_CENTER_THREE_SLICE_RESIDUAL_MULTIPLIER = 2.0
INTERPOLATED_DETACHED_SUPPORT_REVIEW_FRACTION = 0.01


_MAX_SMOOTHED_CELL_CACHE_ENTRIES = 2
_MAX_CELL_RESPONSE_CACHE_ENTRIES = 4


def load_volume(path: Path) -> np.ndarray:
    """Load one ZYX volume from a TIFF file or directory of 2D TIFF slices."""
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input frame does not exist: {path}")

    if path.is_dir():
        files = sorted(
            item
            for item in path.iterdir()
            if item.is_file() and item.suffix.lower() in {".tif", ".tiff"}
        )
        if not files:
            raise ValueError(f"No TIFF slices found in directory: {path}")
        slices = [np.squeeze(tifffile.imread(item)) for item in files]
        if any(item.ndim != 2 for item in slices):
            raise ValueError("A slice directory must contain only 2D TIFF images")
        if len({item.shape for item in slices}) != 1:
            raise ValueError("TIFF slices do not all have the same XY dimensions")
        data = np.stack(slices, axis=0)
    else:
        try:
            data = np.squeeze(tifffile.imread(path))
        except Exception as exc:  # tifffile errors vary by corrupted TIFF layout
            raise ValueError(f"Unable to read TIFF volume {path}: {exc}") from exc

    if data.ndim == 2:
        data = data[np.newaxis, ...]
    if data.ndim != 3:
        raise ValueError(
            f"Expected a single 3D ZYX stack; received shape {tuple(data.shape)}. "
            "Select one timepoint/channel before launching the initializer."
        )
    if min(data.shape) < 2:
        raise ValueError(f"Input volume is degenerate: {tuple(data.shape)}")
    if not np.issubdtype(data.dtype, np.number):
        raise ValueError(f"Unsupported TIFF dtype: {data.dtype}")
    return np.ascontiguousarray(data)


def robust_normalize(data: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
    array = np.asarray(data, dtype=np.float32)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros_like(array, dtype=np.float32)
    lo, hi = np.percentile(finite, [low, high])
    if not np.isfinite(lo):
        lo = float(np.nanmin(finite))
    if not np.isfinite(hi) or hi <= lo:
        hi = float(np.nanmax(finite))
    if hi <= lo:
        return np.zeros_like(array, dtype=np.float32)
    return np.clip((array - lo) / (hi - lo), 0.0, 1.0).astype(np.float32, copy=False)


def clamp_preview(raw: np.ndarray, upper_clamp: float) -> np.ndarray:
    """Return a clamped copy; never modify the source array."""
    return np.minimum(np.asarray(raw), upper_clamp).astype(np.float32, copy=True)


def _response_normalize(response: np.ndarray) -> np.ndarray:
    response = np.abs(np.asarray(response, dtype=np.float32))
    finite = response[np.isfinite(response)]
    if finite.size == 0:
        return np.zeros_like(response)
    scale = float(np.percentile(finite, 99.5))
    if scale <= 1e-12:
        return np.zeros_like(response)
    return np.clip(response / scale, 0.0, 1.0).astype(np.float32, copy=False)


def _effective_dog_log_sigmas(
    dog_sigma_low: float,
    dog_sigma_high: float,
    log_sigma: float,
) -> tuple[float, float, float]:
    low = max(0.1, float(dog_sigma_low))
    high = max(low + 0.1, float(dog_sigma_high))
    return low, high, max(0.1, float(log_sigma))


def _dog_log_response_components(
    image: np.ndarray,
    dog_sigma_low: float,
    dog_sigma_high: float,
    log_sigma: float,
    spatial_only: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim not in {2, 3}:
        raise ValueError("DoG/LoG input must be 2D or 3D")
    low, high, log_value = _effective_dog_log_sigmas(
        dog_sigma_low,
        dog_sigma_high,
        log_sigma,
    )
    if image.ndim == 3 and spatial_only:
        low_sigma: float | tuple[float, float, float] = (0.0, low, low)
        high_sigma: float | tuple[float, float, float] = (0.0, high, high)
        log_sigmas: float | tuple[float, float, float] = (0.0, log_value, log_value)
    else:
        low_sigma = low
        high_sigma = high
        log_sigmas = log_value

    smooth_low = ndi.gaussian_filter(image, sigma=low_sigma, mode="nearest")
    smooth_high = ndi.gaussian_filter(image, sigma=high_sigma, mode="nearest")
    dog = _response_normalize(smooth_low - smooth_high)
    log = _response_normalize(
        ndi.gaussian_laplace(image, sigma=log_sigmas, mode="nearest")
        * (log_value * log_value)
    )
    return dog, log


def _fuse_dog_log_responses(
    dog: np.ndarray,
    log: np.ndarray,
    fusion_weight: float,
) -> np.ndarray:
    weight = float(np.clip(fusion_weight, 0.0, 1.0))
    weighted = weight * dog + (1.0 - weight) * log
    fused = np.maximum(weighted, np.maximum(dog, log) * 0.75)
    return np.clip(fused, 0.0, 1.0).astype(np.float32, copy=False)


def dog_log_responses(
    image: np.ndarray,
    dog_sigma_low: float,
    dog_sigma_high: float,
    log_sigma: float,
    fusion_weight: float,
    spatial_only: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dog, log = _dog_log_response_components(
        image,
        dog_sigma_low,
        dog_sigma_high,
        log_sigma,
        spatial_only=spatial_only,
    )
    return dog, log, _fuse_dog_log_responses(dog, log, fusion_weight)


def _cell_smoothing_key(
    params: CellEdgeParameters,
) -> tuple[float, float]:
    return (
        max(0.0, float(params.smoothing_sigma_z)),
        max(0.0, float(params.smoothing_sigma_xy)),
    )


def _smooth_cell_source(
    raw: np.ndarray,
    smoothing_key: tuple[float, float],
) -> np.ndarray:
    sigma_z, sigma_xy = smoothing_key
    return ndi.gaussian_filter(
        np.asarray(raw, dtype=np.float32),
        sigma=(sigma_z, sigma_xy, sigma_xy),
        mode="nearest",
    ).astype(np.float32, copy=False)


@dataclass
class PreparedCellDetection:
    """Reusable, source-bound preprocessing for interactive cell detection.

    The cache is intentionally bound to the exact source ``ndarray`` object.
    Callers must treat that source as immutable for the lifetime of this
    object. Smoothing is keyed by its Z/XY sigma pair, while normalized DoG
    and LoG components are keyed only by their effective sigma tuple. Fusion
    and every segmentation control are still evaluated on every detection.
    Both parameter caches are small and least-recently-used so dragging a
    sigma control cannot retain an unbounded number of full-volume arrays.
    """

    source: np.ndarray = field(repr=False, compare=False)
    normalized: np.ndarray = field(init=False, repr=False, compare=False)
    _source_shape: tuple[int, ...] = field(init=False, repr=False)
    _source_dtype: np.dtype = field(init=False, repr=False)
    _smoothed_cache: dict[tuple[float, float], np.ndarray] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _response_cache: dict[
        tuple[float, float, float],
        tuple[np.ndarray, np.ndarray],
    ] = field(default_factory=dict, init=False, repr=False, compare=False)
    _lock: RLock = field(
        default_factory=RLock,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        source = np.asarray(self.source)
        if source.ndim != 3:
            raise ValueError("Prepared cell detection expects a 3D ZYX source")
        if not np.issubdtype(source.dtype, np.number):
            raise ValueError(
                f"Prepared cell detection does not support dtype {source.dtype}"
            )
        self.source = source
        self._source_shape = tuple(int(value) for value in source.shape)
        self._source_dtype = source.dtype
        normalized = robust_normalize(source)
        normalized.setflags(write=False)
        self.normalized = normalized

    def validate_source(self, raw: np.ndarray) -> None:
        """Reject a cache paired with a different or structurally changed source."""
        source = np.asarray(raw)
        if source is not self.source:
            raise ValueError(
                "Prepared cell detection belongs to a different source array"
            )
        if (
            tuple(int(value) for value in source.shape) != self._source_shape
            or source.dtype != self._source_dtype
        ):
            raise ValueError(
                "Prepared cell detection source shape or dtype has changed"
            )

    def smoothed_for(self, params: CellEdgeParameters) -> np.ndarray:
        key = _cell_smoothing_key(params)
        with self._lock:
            cached = self._smoothed_cache.pop(key, None)
            if cached is None:
                cached = _smooth_cell_source(self.source, key)
                cached.setflags(write=False)
            self._smoothed_cache[key] = cached
            while (
                len(self._smoothed_cache)
                > _MAX_SMOOTHED_CELL_CACHE_ENTRIES
            ):
                self._smoothed_cache.pop(next(iter(self._smoothed_cache)))
            return cached

    def responses_for(
        self,
        params: CellEdgeParameters,
    ) -> tuple[np.ndarray, np.ndarray]:
        key = _effective_dog_log_sigmas(
            params.dog_sigma_low,
            params.dog_sigma_high,
            params.log_sigma,
        )
        with self._lock:
            cached = self._response_cache.pop(key, None)
            if cached is None:
                dog, log = _dog_log_response_components(
                    self.normalized,
                    *key,
                    spatial_only=True,
                )
                dog.setflags(write=False)
                log.setflags(write=False)
                cached = (dog, log)
            self._response_cache[key] = cached
            while (
                len(self._response_cache)
                > _MAX_CELL_RESPONSE_CACHE_ENTRIES
            ):
                self._response_cache.pop(next(iter(self._response_cache)))
            return cached

    def prepare(self, params: CellEdgeParameters) -> PreparedCellDetection:
        """Eagerly populate the entries required by ``params`` and return self."""
        self.smoothed_for(params)
        self.responses_for(params)
        return self

    @property
    def smoothed_cache_size(self) -> int:
        with self._lock:
            return len(self._smoothed_cache)

    @property
    def response_cache_size(self) -> int:
        with self._lock:
            return len(self._response_cache)


def prepare_cell_detection(
    raw: np.ndarray,
    params: CellEdgeParameters | None = None,
) -> PreparedCellDetection:
    """Create an optional reusable preprocessing cache for cell detection."""
    prepared = PreparedCellDetection(raw)
    if params is not None:
        prepared.prepare(params)
    return prepared


def _disk_or_ball(radius: int, ndim: int) -> np.ndarray:
    radius = max(0, int(radius))
    if radius == 0:
        return np.ones((1,) * ndim, dtype=bool)
    if ndim == 2:
        return morphology.disk(radius).astype(bool)
    return morphology.ball(radius).astype(bool)


def _fill_slices(binary: np.ndarray) -> np.ndarray:
    if binary.ndim == 2:
        return ndi.binary_fill_holes(binary)
    return np.stack([ndi.binary_fill_holes(item) for item in binary], axis=0)


def _remove_objects_smaller_than(
    binary: np.ndarray,
    minimum_size: int,
    *,
    fully_connected: bool = False,
) -> np.ndarray:
    """Version-independent replacement for skimage's renamed size argument."""
    mask = np.asarray(binary, dtype=bool)
    minimum_size = max(1, int(minimum_size))
    structure = (
        np.ones((3,) * mask.ndim, dtype=bool)
        if fully_connected
        else ndi.generate_binary_structure(mask.ndim, 1)
    )
    labels, count = ndi.label(
        mask,
        structure=structure,
    )
    if count == 0:
        return np.zeros_like(mask)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= minimum_size
    keep[0] = False
    return keep[labels]


def _clean_binary(
    binary: np.ndarray,
    min_component_size: int,
    closing_radius: int,
) -> np.ndarray:
    footprint = _disk_or_ball(closing_radius, binary.ndim)
    closed = ndi.binary_closing(binary, structure=footprint)
    filled = _fill_slices(closed)
    if min_component_size > 0:
        filled = _remove_objects_smaller_than(
            filled, int(min_component_size)
        )
    return np.asarray(filled, dtype=bool)


def detect_membrane(raw: np.ndarray, params: MembraneParameters) -> EdgeResult:
    clamped = clamp_preview(raw, params.upper_clamp)
    normalized = robust_normalize(clamped)
    dog, log, fused = dog_log_responses(
        normalized,
        params.dog_sigma_low,
        params.dog_sigma_high,
        params.log_sigma,
        params.fusion_weight,
        spatial_only=True,
    )
    edges = fused >= float(params.edge_threshold)
    regions = _clean_binary(
        edges,
        params.min_component_size,
        params.closing_radius,
    )
    labels, count = ndi.label(regions, structure=np.ones((3, 3, 3), dtype=bool))
    return EdgeResult(
        clamped=clamped,
        dog=dog,
        log=log,
        fused=fused,
        edges=edges,
        regions=regions,
        labels=labels.astype(np.int32, copy=False),
        component_count=int(count),
    )


def largest_component(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.max(initial=0) <= 0:
        return np.zeros_like(labels, dtype=bool)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == int(np.argmax(counts))


def _marker_slice_observations(
    markers: np.ndarray,
    final_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one ZYX centroid per marker and raw Z slice.

    Bright center cores are 3D connected watershed markers and can therefore
    span several source slices. Fallback distance markers are single voxels.
    Each observation is assigned the final, deterministically renumbered
    watershed label covering that marker.
    """
    marker_volume = np.asarray(markers, dtype=np.int32)
    labels = np.asarray(final_labels, dtype=np.int32)
    if marker_volume.shape != labels.shape or marker_volume.ndim != 3:
        raise ValueError("Marker observations require matching 3D label volumes")

    points: list[tuple[float, float, float]] = []
    ids: list[int] = []
    for z_index, marker_slice in enumerate(marker_volume):
        for marker_id in sorted(
            int(value) for value in np.unique(marker_slice) if value > 0
        ):
            marker_mask = marker_slice == marker_id
            assigned_labels = np.unique(labels[z_index][marker_mask])
            assigned_labels = assigned_labels[assigned_labels > 0]
            if assigned_labels.size != 1:
                raise RuntimeError(
                    "A watershed marker did not map to exactly one final label"
                )
            coordinates_yx = np.argwhere(marker_mask)
            centroid_y, centroid_x = coordinates_yx.mean(axis=0)
            points.append(
                (float(z_index), float(centroid_y), float(centroid_x))
            )
            ids.append(int(assigned_labels[0]))
    return (
        np.asarray(points, dtype=np.float32).reshape((-1, 3)),
        np.asarray(ids, dtype=np.int32),
    )


def _watershed_cell_regions_per_slice(
    smoothed_intensity: np.ndarray,
    fused: np.ndarray,
    boundary_mask: np.ndarray,
    membrane_mask: np.ndarray,
    params: CellEdgeParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build solid 3D regions from intensity cores and weak edge elevation.

    Despite the transitional function name, this performs one 3D watershed.
    Connected center cores provide identities across raw Z directly. Distance
    peaks supply a marker only for a foreground chunk that has no core.
    """
    smoothed = np.asarray(smoothed_intensity, dtype=np.float32)
    fused = np.asarray(fused, dtype=np.float32)
    boundaries = np.asarray(boundary_mask, dtype=bool)
    membrane = np.asarray(membrane_mask, dtype=bool)
    if (
        fused.ndim != 3
        or smoothed.shape != fused.shape
        or boundaries.shape != fused.shape
        or membrane.shape != fused.shape
    ):
        raise ValueError("Cell region inputs must be matching 3D ZYX arrays")

    close_radius = max(0, int(params.closing_radius))
    close_footprint = _disk_or_ball(close_radius, 2)
    interior_radius = max(1, close_radius)
    interior_footprint = _disk_or_ball(interior_radius, 2)
    min_chunk_volume = max(1, int(params.min_chunk_area_2d))
    interior = np.stack(
        [
            ndi.binary_erosion(slice_mask, structure=interior_footprint)
            for slice_mask in membrane
        ],
        axis=0,
    )
    empty_interior = ~np.any(interior, axis=(1, 2))
    interior[empty_interior] = membrane[empty_interior]
    # Quantiles are defined against the complete frame so the controls remain
    # comparable even when the user adjusts the membrane model.
    intensity_values = smoothed[np.isfinite(smoothed)]
    if intensity_values.size == 0:
        return (
            np.zeros(fused.shape, dtype=np.int32),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    normalized_smoothed = robust_normalize(smoothed, high=99.9)
    if params.foreground_threshold is None:
        foreground_level = float(
            np.percentile(
                intensity_values,
                np.clip(params.foreground_quantile, 0.0, 100.0),
            )
        )
        foreground = smoothed >= foreground_level
    else:
        foreground = normalized_smoothed >= float(params.foreground_threshold)
    if params.center_threshold is None:
        center_level = float(
            np.percentile(
                intensity_values,
                np.clip(params.center_quantile, 0.0, 100.0),
            )
        )
        center_cores = smoothed >= center_level
    else:
        center_cores = normalized_smoothed >= float(params.center_threshold)

    foreground &= interior
    foreground = np.stack(
        [
            ndi.binary_closing(
                foreground_slice,
                structure=close_footprint,
            )
            for foreground_slice in foreground
        ],
        axis=0,
    )
    # A real 3D cell can have no single cross-section larger than the user's
    # cutoff. Filter the connected 3D candidate volume, not each Z slice
    # independently, so those cells are not erased before center detection.
    foreground = _remove_objects_smaller_than(
        foreground,
        min_chunk_volume,
        fully_connected=True,
    )
    if not np.any(foreground):
        return (
            np.zeros(fused.shape, dtype=np.int32),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    center_cores &= foreground
    core_labels, core_count = ndi.label(
        center_cores, structure=np.ones((3, 3, 3), dtype=bool)
    )
    if core_count:
        core_sizes = np.bincount(core_labels.ravel())
        keep_core = core_sizes >= max(1, int(params.min_center_core_volume))
        keep_core[0] = False
        center_cores = keep_core[core_labels]
    markers, marker_count = ndi.label(
        center_cores, structure=np.ones((3, 3, 3), dtype=bool)
    )

    # Every disconnected foreground chunk needs at least one marker. A
    # distance-transform peak is a robust fallback when its bright core was
    # filtered out by the user's size slider.
    chunks, chunk_count = ndi.label(
        foreground, structure=np.ones((3, 3, 3), dtype=bool)
    )
    for chunk_id, bounds in enumerate(ndi.find_objects(chunks), start=1):
        if bounds is None or chunk_id > chunk_count:
            continue
        chunk = chunks[bounds] == chunk_id
        chunk_markers = markers[bounds]
        if np.any(chunk_markers[chunk]):
            continue
        distance = ndi.distance_transform_edt(chunk).astype(np.float32)
        peak_coordinates = feature.peak_local_max(
            distance,
            min_distance=max(1, int(params.center_min_distance)),
            threshold_abs=max(0.0, float(params.center_min_radius)),
            threshold_rel=float(np.clip(params.center_threshold_rel, 0.0, 1.0)),
            labels=chunk.astype(np.uint8),
            exclude_border=False,
        )
        if peak_coordinates.size == 0:
            peak_coordinates = np.asarray(
                [np.unravel_index(int(np.argmax(distance)), distance.shape)],
                dtype=np.int64,
            )
        for peak_coordinate in peak_coordinates:
            marker_count += 1
            coordinate = tuple(int(v) for v in peak_coordinate)
            global_coordinate = tuple(
                coordinate[axis] + int(bounds[axis].start) for axis in range(3)
            )
            markers[global_coordinate] = marker_count

    edge_weight = float(np.clip(params.edge_elevation_weight, 0.0, 0.30))
    intensity_elevation = normalized_smoothed
    # The user-facing fused-edge threshold controls which responses contribute
    # boundary guidance. These pixels raise the watershed elevation but remain
    # inside its foreground mask, so tuning the threshold moves cell divisions
    # without turning the final filled regions into hollow edge masks.
    thresholded_boundary_strength = np.where(
        boundaries,
        fused,
        0.0,
    ).astype(np.float32, copy=False)
    elevation = (
        -intensity_elevation
        + edge_weight * thresholded_boundary_strength
    )
    labels = segmentation.watershed(
        elevation,
        markers=markers,
        mask=foreground,
        connectivity=np.ones((3, 3, 3), dtype=bool),
        watershed_line=False,
    ).astype(np.int32, copy=False)
    labels, _ = renumber_group_labels(labels)
    slice_centers_zyx, slice_center_ids = _marker_slice_observations(
        markers,
        labels,
    )
    return labels, slice_centers_zyx, slice_center_ids


def detect_cell_regions(
    raw: np.ndarray,
    membrane_mask: np.ndarray,
    params: CellEdgeParameters,
    grouping_params: GroupingParameters | None = None,
    prepared: PreparedCellDetection | None = None,
) -> CellRegionResult:
    """Detect solid cell regions and associate their identities through raw Z.

    ``edges`` in the returned result is the diagnostic thresholded boundary
    mask. ``regions`` and ``labels`` are the solid result; labels are stable
    cross-slice cell IDs rather than edge pixels. ``slice_centers_zyx`` holds
    the per-Z centroids of actual bright-core or fallback watershed markers,
    mapped to those final labels. Pass a source-compatible
    ``PreparedCellDetection`` to reuse preprocessing during interactive
    parameter changes; omitting it preserves the uncached behavior.
    ``centers_xyz`` is calculated from the solid 3D chunks with robust
    intensity weighting; the marker observations remain separate review
    evidence in ``slice_centers_zyx``.
    """
    if prepared is None:
        normalized = robust_normalize(raw)
        # Preserve the upper-tail ordering for the adaptive p99.x masks. The
        # normalized image intentionally remains limited to edge diagnostics.
        smoothed = _smooth_cell_source(raw, _cell_smoothing_key(params))
        dog, log, fused = dog_log_responses(
            normalized,
            params.dog_sigma_low,
            params.dog_sigma_high,
            params.log_sigma,
            params.fusion_weight,
            spatial_only=True,
        )
    else:
        prepared.validate_source(raw)
        smoothed = prepared.smoothed_for(params)
        dog, log = prepared.responses_for(params)
        fused = _fuse_dog_log_responses(dog, log, params.fusion_weight)
    inside = np.asarray(membrane_mask, dtype=bool)
    edges = (fused >= float(params.edge_threshold)) & inside
    if int(params.min_component_size) > 1:
        edges = np.stack(
            [
                _remove_objects_smaller_than(
                    edge_slice,
                    int(params.min_component_size),
                )
                for edge_slice in edges
            ],
            axis=0,
        )
    labels, slice_centers_zyx, slice_center_ids = (
        _watershed_cell_regions_per_slice(
            smoothed,
            fused,
            edges,
            inside,
            params,
        )
    )
    center_estimates = robust_cell_centers_from_labels(
        labels,
        raw,
        candidate_points_zyx=slice_centers_zyx,
        candidate_ids=slice_center_ids,
        candidate_accepted=np.ones(
            len(slice_center_ids),
            dtype=bool,
        ),
    )
    centers_xyz = {
        int(label_id): estimate.center_xyz.copy()
        for label_id, estimate in center_estimates.items()
    }
    return CellRegionResult(
        clamped=np.asarray(raw, dtype=np.float32),
        dog=dog,
        log=log,
        fused=fused,
        edges=edges,
        regions=labels > 0,
        labels=labels.astype(np.int32, copy=False),
        component_count=len(centers_xyz),
        centers_xyz=centers_xyz,
        slice_centers_zyx=slice_centers_zyx,
        slice_center_ids=slice_center_ids,
        center_diagnostics={
            int(label_id): estimate.to_dict()
            for label_id, estimate in center_estimates.items()
        },
    )


def detect_cell_edges(
    raw: np.ndarray,
    membrane_mask: np.ndarray,
    params: CellEdgeParameters,
    grouping_params: GroupingParameters | None = None,
    prepared: PreparedCellDetection | None = None,
) -> CellRegionResult:
    """Compatibility wrapper for the region-centric cell detector."""
    return detect_cell_regions(
        raw,
        membrane_mask,
        params,
        grouping_params,
        prepared,
    )


def estimate_two_backgrounds(
    raw: np.ndarray,
    membrane_mask: np.ndarray,
    occupied_mask: np.ndarray | None = None,
) -> tuple[float, float]:
    image = robust_normalize(raw)
    hot_region = np.asarray(membrane_mask, dtype=bool)
    cold_values = image[~hot_region]
    if occupied_mask is None:
        occupied = np.zeros_like(hot_region)
    else:
        occupied = np.asarray(occupied_mask, dtype=bool)
    hot_values = image[hot_region & ~occupied]
    cold = float(np.percentile(cold_values, 95.0)) if cold_values.size else 0.0
    hot = float(np.percentile(hot_values, 35.0)) if hot_values.size else cold
    if not np.isfinite(cold):
        cold = 0.0
    if not np.isfinite(hot):
        hot = cold
    return cold, hot


def interpolate_z_linear(volume: np.ndarray, ratio: float) -> np.ndarray:
    volume = np.asarray(volume)
    ratio = float(ratio)
    if ratio <= 0:
        raise ValueError("Z interpolation ratio must be positive")
    if volume.shape[0] == 1:
        return volume.copy()
    new_depth = int(round((volume.shape[0] - 1) * ratio)) + 1
    positions = np.linspace(0.0, float(volume.shape[0] - 1), new_depth, dtype=np.float32)
    lower = np.floor(positions).astype(np.int32)
    upper = np.minimum(lower + 1, volume.shape[0] - 1)
    weight = (positions - lower).astype(np.float32)
    shape = (new_depth,) + (1,) * (volume.ndim - 1)
    out = (
        np.asarray(volume[lower], dtype=np.float32) * (1.0 - weight.reshape(shape))
        + np.asarray(volume[upper], dtype=np.float32) * weight.reshape(shape)
    )
    return out.astype(np.float32, copy=False)


def signed_distance(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    return (
        ndi.distance_transform_edt(mask) - ndi.distance_transform_edt(~mask)
    ).astype(np.float32)


def interpolate_binary_signed_distance(mask: np.ndarray, ratio: float) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3:
        raise ValueError("Signed-distance Z interpolation expects a 3D mask")
    filled = _fill_slices(mask)
    distances = np.stack([signed_distance(item) for item in filled], axis=0)
    return interpolate_z_linear(distances, ratio) >= 0.0


def _scaled_z_coordinate(z: float, source_depth: int, target_depth: int) -> float:
    if source_depth <= 1 or target_depth <= 1:
        return 0.0
    return float(z) * float(target_depth - 1) / float(source_depth - 1)


@dataclass(frozen=True)
class _ComponentZGeometry:
    component_id: int
    voxel_count: int
    z_min: int
    z_max: int
    lower_center_yx: np.ndarray
    upper_center_yx: np.ndarray
    lower_radius: float
    upper_radius: float


def _component_z_geometry(
    component_labels: np.ndarray,
    component_id: int,
) -> _ComponentZGeometry:
    coordinates = np.argwhere(component_labels == int(component_id))
    z_min = int(np.min(coordinates[:, 0]))
    z_max = int(np.max(coordinates[:, 0]))
    lower_coordinates = coordinates[coordinates[:, 0] == z_min, 1:]
    upper_coordinates = coordinates[coordinates[:, 0] == z_max, 1:]
    return _ComponentZGeometry(
        component_id=int(component_id),
        voxel_count=int(coordinates.shape[0]),
        z_min=z_min,
        z_max=z_max,
        lower_center_yx=np.mean(lower_coordinates, axis=0),
        upper_center_yx=np.mean(upper_coordinates, axis=0),
        lower_radius=float(
            np.sqrt(lower_coordinates.shape[0] / np.pi)
        ),
        upper_radius=float(
            np.sqrt(upper_coordinates.shape[0] / np.pi)
        ),
    )


def _z_aligned_component_pair(
    first: _ComponentZGeometry,
    second: _ComponentZGeometry,
) -> tuple[bool, int, float]:
    """Return whether two non-overlapping Z runs plausibly form one cell."""

    if first.z_max < second.z_min:
        lower = first
        upper = second
    elif second.z_max < first.z_min:
        lower = second
        upper = first
    else:
        return False, -1, np.inf
    missing_slices = int(upper.z_min - lower.z_max - 1)
    if missing_slices > ROBUST_CELL_CENTER_MAX_COMPONENT_GAP_SLICES:
        return False, missing_slices, np.inf
    lateral_distance = float(
        np.linalg.norm(
            lower.upper_center_yx - upper.lower_center_yx
        )
    )
    alignment_limit = max(
        ROBUST_CELL_CENTER_MINIMUM_COMPONENT_ALIGNMENT_PIXELS,
        ROBUST_CELL_CENTER_COMPONENT_ALIGNMENT_RADIUS_FACTOR
        * max(lower.upper_radius, upper.lower_radius),
    )
    return (
        lateral_distance <= alignment_limit + 1e-9,
        missing_slices,
        lateral_distance,
    )


def _select_z_aligned_component_group(
    component_labels: np.ndarray,
    *,
    support_counts: np.ndarray | None = None,
    anchor_zyx: np.ndarray | None = None,
) -> tuple[int, tuple[int, ...]]:
    """Choose one anchor plus short, laterally aligned disjoint Z runs."""

    labels = np.asarray(component_labels, dtype=np.int32)
    component_count = int(labels.max(initial=0))
    if component_count <= 0:
        return 0, ()
    geometries = {
        component_id: _component_z_geometry(labels, component_id)
        for component_id in range(1, component_count + 1)
    }
    if support_counts is None:
        support = np.zeros(component_count + 1, dtype=np.int64)
    else:
        support = np.asarray(support_counts, dtype=np.int64)
        if support.shape != (component_count + 1,):
            raise ValueError(
                "Component support counts must include background plus every "
                "positive component"
            )

    anchor_component = 0
    if anchor_zyx is not None:
        anchor = np.rint(
            np.asarray(anchor_zyx, dtype=np.float64)
        ).astype(np.int64)
        if anchor.shape != (3,):
            raise ValueError("Component anchor must have shape (3,) in ZYX")
        if np.all(anchor >= 0) and np.all(
            anchor < np.asarray(labels.shape, dtype=np.int64)
        ):
            anchor_component = int(labels[tuple(anchor)])
    if anchor_component <= 0:
        anchor_component = max(
            geometries,
            key=lambda component_id: (
                int(support[component_id]),
                geometries[component_id].voxel_count,
                -int(component_id),
            ),
        )

    selected = {int(anchor_component)}
    while True:
        added = False
        candidates = sorted(
            set(geometries) - selected,
            key=lambda component_id: (
                -int(support[component_id]),
                -geometries[component_id].voxel_count,
                int(component_id),
            ),
        )
        for candidate_id in candidates:
            candidate = geometries[candidate_id]
            overlaps_selected_z = any(
                not (
                    candidate.z_max < geometries[selected_id].z_min
                    or geometries[selected_id].z_max < candidate.z_min
                )
                for selected_id in selected
            )
            if overlaps_selected_z:
                continue
            if any(
                _z_aligned_component_pair(
                    geometries[selected_id],
                    candidate,
                )[0]
                for selected_id in selected
            ):
                selected.add(candidate_id)
                added = True
                break
        if not added:
            break
    ordered = (
        int(anchor_component),
        *sorted(selected - {int(anchor_component)}),
    )
    return int(anchor_component), tuple(ordered)


def _partition_z_aligned_component_groups(
    component_labels: np.ndarray,
    component_ids: Iterable[int],
) -> tuple[tuple[int, ...], ...]:
    """Partition components into deterministic, short aligned Z runs."""

    labels = np.asarray(component_labels, dtype=np.int32)
    remaining = {
        int(component_id)
        for component_id in component_ids
        if int(component_id) > 0
        and np.any(labels == int(component_id))
    }
    geometries = {
        component_id: _component_z_geometry(labels, component_id)
        for component_id in remaining
    }
    groups: list[tuple[int, ...]] = []
    while remaining:
        anchor = max(
            remaining,
            key=lambda component_id: (
                geometries[component_id].voxel_count,
                -int(component_id),
            ),
        )
        selected = {int(anchor)}
        while True:
            added = False
            candidates = sorted(
                remaining - selected,
                key=lambda component_id: (
                    -geometries[component_id].voxel_count,
                    int(component_id),
                ),
            )
            for candidate_id in candidates:
                candidate = geometries[candidate_id]
                if any(
                    not (
                        candidate.z_max < geometries[selected_id].z_min
                        or geometries[selected_id].z_max < candidate.z_min
                    )
                    for selected_id in selected
                ):
                    continue
                if any(
                    _z_aligned_component_pair(
                        geometries[selected_id],
                        candidate,
                    )[0]
                    for selected_id in selected
                ):
                    selected.add(candidate_id)
                    added = True
                    break
            if not added:
                break
        ordered = (int(anchor), *sorted(selected - {int(anchor)}))
        groups.append(tuple(ordered))
        remaining -= selected

    def group_center(group: tuple[int, ...]) -> tuple[float, float, float]:
        coordinates = np.argwhere(np.isin(labels, group))
        return tuple(float(value) for value in np.mean(coordinates, axis=0))

    groups.sort(key=group_center)
    return tuple(groups)


def split_detached_label_support(
    labels_zyx: np.ndarray,
    selected_components_by_label: dict[int, Iterable[int]],
    *,
    minimum_child_voxels: int = 1,
) -> DetachedSupportSplitResult:
    """Give every unresolved 3D component group its own cell ID.

    The robustly selected component group keeps the source label. Remaining
    components are partitioned using the same short-gap, lateral-alignment
    rule used by robust center estimation. A detached child smaller than
    ``minimum_child_voxels`` is discarded; every retained child receives a new
    ID. The robustly selected source child is never size-filtered here.
    """

    labels = np.asarray(labels_zyx)
    if labels.ndim != 3 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(
            "Detached-support repair requires one integer 3D label volume"
        )
    output = labels.astype(np.int32, copy=True)
    minimum = int(minimum_child_voxels)
    if minimum < 1:
        raise ValueError("Detached-child minimum must be at least one voxel")
    source_to_children: dict[int, tuple[int, ...]] = {}
    discarded_voxels_by_source: dict[int, int] = {}
    discarded_group_count_by_source: dict[int, int] = {}
    affected_slices: set[int] = set()
    next_label = int(output.max(initial=0)) + 1
    object_bounds = ndi.find_objects(output)

    for label_id in sorted(int(value) for value in selected_components_by_label):
        if label_id <= 0:
            continue
        bounds = (
            object_bounds[label_id - 1]
            if label_id - 1 < len(object_bounds)
            else None
        )
        if bounds is None:
            continue
        local_mask = output[bounds] == label_id
        component_labels, component_count = ndi.label(
            local_mask,
            structure=np.ones((3, 3, 3), dtype=bool),
        )
        if component_count <= 1:
            continue
        selected = {
            int(component_id)
            for component_id in selected_components_by_label[label_id]
            if 0 < int(component_id) <= int(component_count)
        }
        if not selected:
            _, selected_group = _select_z_aligned_component_group(
                component_labels
            )
            selected = set(selected_group)
        remaining = set(range(1, int(component_count) + 1)) - selected
        if not remaining:
            continue

        local_output = output[bounds]
        children = [label_id]
        for group in _partition_z_aligned_component_groups(
            component_labels,
            remaining,
        ):
            group_mask = np.isin(component_labels, group)
            global_z = np.flatnonzero(np.any(group_mask, axis=(1, 2)))
            z_origin = int(bounds[0].start)
            affected_slices.update(
                z_origin + int(z_index) for z_index in global_z
            )
            voxel_count = int(np.count_nonzero(group_mask))
            if voxel_count < minimum:
                local_output[group_mask] = 0
                discarded_voxels_by_source[label_id] = (
                    discarded_voxels_by_source.get(label_id, 0)
                    + voxel_count
                )
                discarded_group_count_by_source[label_id] = (
                    discarded_group_count_by_source.get(label_id, 0) + 1
                )
                continue
            local_output[group_mask] = next_label
            children.append(next_label)
            next_label += 1
        if len(children) > 1:
            source_to_children[label_id] = tuple(children)

    return DetachedSupportSplitResult(
        labels=output,
        source_to_children=source_to_children,
        affected_slices=tuple(sorted(affected_slices)),
        discarded_voxels_by_source=discarded_voxels_by_source,
        discarded_group_count_by_source=discarded_group_count_by_source,
    )


def _fill_short_internal_z_gaps(mask_zyx: np.ndarray) -> np.ndarray:
    """Interpolate short bounded empty-Z runs in one selected cell mask."""

    mask = np.asarray(mask_zyx, dtype=bool)
    output = mask.copy()
    occupied = np.any(mask, axis=(1, 2))
    empty_labels, empty_count = ndi.label(~occupied)
    for gap_id in range(1, int(empty_count) + 1):
        gap_indices = np.flatnonzero(empty_labels == gap_id)
        if gap_indices.size == 0:
            continue
        first = int(gap_indices[0])
        last = int(gap_indices[-1])
        if (
            first == 0
            or last == mask.shape[0] - 1
            or gap_indices.size
            > ROBUST_CELL_CENTER_MAX_COMPONENT_GAP_SLICES
        ):
            continue
        lower = first - 1
        upper = last + 1
        lower_distance = signed_distance(mask[lower])
        upper_distance = signed_distance(mask[upper])
        span = float(upper - lower)
        for z_index in gap_indices:
            weight = float(z_index - lower) / span
            interpolated_distance = (
                (1.0 - weight) * lower_distance
                + weight * upper_distance
            )
            output[int(z_index)] = interpolated_distance >= 0.0
    return output


def interpolate_labeled_z(
    labels_raw: np.ndarray,
    ratio: float,
    centers_xyz: dict[int, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Interpolate each positive label independently through Z.

    Per-label signed-distance scores are evaluated only in that label's padded
    bounding box. A single global best-score volume resolves overlaps in favor
    of the label whose point is deepest inside its interpolated region. Thus
    memory stays O(output volume), rather than O(labels * output volume).
    ``None`` infers label centroids; an explicitly empty center mapping remains
    empty and lets disconnected candidates fall back to their largest component.
    """
    labels = np.asarray(labels_raw)
    if labels.ndim != 3:
        raise ValueError("Label-preserving Z interpolation expects a 3D label volume")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Label-preserving Z interpolation expects integer labels")
    ratio = float(ratio)
    if ratio <= 0:
        raise ValueError("Z interpolation ratio must be positive")

    source_depth, height, width = labels.shape
    target_depth = (
        1
        if source_depth == 1
        else int(round((source_depth - 1) * ratio)) + 1
    )
    target_positions = (
        np.zeros((1,), dtype=np.float32)
        if source_depth == 1
        else np.linspace(
            0.0,
            float(source_depth - 1),
            target_depth,
            dtype=np.float32,
        )
    )
    output = np.zeros((target_depth, height, width), dtype=np.int32)
    best_score = np.full(output.shape, -np.inf, dtype=np.float32)

    label_ids = sorted(int(value) for value in np.unique(labels) if value > 0)
    source_centers = (
        group_seeds_from_labels(labels)
        if centers_xyz is None
        else centers_xyz
    )
    scaled_centers: dict[int, np.ndarray] = {}
    for label_id, center_xyz in sorted(source_centers.items()):
        center = np.asarray(center_xyz, dtype=np.float64).copy()
        if center.shape != (3,):
            raise ValueError(f"Center for label {label_id} must be XYZ with shape (3,)")
        center[2] = float(
            np.clip(
                _scaled_z_coordinate(center[2], source_depth, target_depth),
                0.0,
                max(0, target_depth - 1),
            )
        )
        scaled_centers[int(label_id)] = center

    object_bounds = ndi.find_objects(labels)
    for label_id in label_ids:
        bounds = (
            object_bounds[label_id - 1]
            if label_id - 1 < len(object_bounds)
            else None
        )
        if bounds is None:
            continue
        z_bounds, y_bounds, x_bounds = bounds
        # One empty source slice on either side makes cells appear/disappear
        # smoothly. One XY pixel is enough to establish a negative exterior
        # while keeping the EDT workload proportional to cell size.
        z0 = max(0, int(z_bounds.start) - 1)
        z1 = min(source_depth, int(z_bounds.stop) + 1)
        y0 = max(0, int(y_bounds.start) - 1)
        y1 = min(height, int(y_bounds.stop) + 1)
        x0 = max(0, int(x_bounds.start) - 1)
        x1 = min(width, int(x_bounds.stop) + 1)
        local_mask = labels[z0:z1, y0:y1, x0:x1] == label_id
        source_component_labels, source_component_count = ndi.label(
            local_mask,
            structure=np.ones((3, 3, 3), dtype=bool),
        )
        if source_component_count > 1:
            source_center = source_centers.get(label_id)
            source_anchor = (
                None
                if source_center is None
                else np.array(
                    [
                        float(source_center[2]) - z0,
                        float(source_center[1]) - y0,
                        float(source_center[0]) - x0,
                    ],
                    dtype=np.float64,
                )
            )
            _, selected_source_components = (
                _select_z_aligned_component_group(
                    source_component_labels,
                    anchor_zyx=source_anchor,
                )
            )
            local_mask = np.isin(
                source_component_labels,
                selected_source_components,
            )
        local_mask = _fill_short_internal_z_gaps(local_mask)
        local_distances = np.stack(
            [signed_distance(slice_mask) for slice_mask in local_mask],
            axis=0,
        )

        target_indices = np.flatnonzero(
            (target_positions >= float(z0))
            & (target_positions <= float(z1 - 1))
        )
        if target_indices.size == 0:
            continue
        local_positions = target_positions[target_indices] - float(z0)
        lower = np.floor(local_positions).astype(np.int32)
        upper = np.minimum(lower + 1, local_distances.shape[0] - 1)
        weights = (local_positions - lower).astype(np.float32)
        local_scores = (
            local_distances[lower] * (1.0 - weights[:, None, None])
            + local_distances[upper] * weights[:, None, None]
        )
        candidate = local_scores >= 0.0
        # Detached source artifacts were already removed by the selected source
        # component group above. Retain every derived fragment of that selected
        # identity here: filtering the output to one component would again
        # delete a legitimate same-ID run when a manually blocked or
        # image-supported gap remains open.

        for local_z, target_z in enumerate(target_indices):
            score = local_scores[local_z]
            current_best = best_score[target_z, y0:y1, x0:x1]
            update = candidate[local_z] & (score > current_best)
            if not np.any(update):
                continue
            current_output = output[target_z, y0:y1, x0:x1]
            current_output[update] = label_id
            current_best[update] = score[update]
    return output, scaled_centers


def interpolated_label_boundaries(labels: np.ndarray) -> np.ndarray:
    """Return an ID-preserving inner boundary for an interpolated label volume."""
    label_volume = np.asarray(labels)
    if label_volume.ndim != 3:
        raise ValueError("Interpolated label boundaries expect a 3D label volume")
    if not np.issubdtype(label_volume.dtype, np.integer):
        raise ValueError("Interpolated label boundaries expect integer labels")
    boundaries = segmentation.find_boundaries(
        label_volume,
        connectivity=label_volume.ndim,
        mode="inner",
        background=0,
    )
    return np.where(boundaries, label_volume, 0).astype(np.int32, copy=False)


# Descriptive alias retained for callers that prefer the operation in the name.
interpolate_labels_signed_distance = interpolate_labeled_z


@dataclass
class SliceComponent:
    slice_index: int
    local_label: int
    bbox_yx: tuple[int, int, int, int]
    cropped_mask: np.ndarray
    centroid_yx: np.ndarray
    area: int


def _slice_components(edges: np.ndarray, min_size: int = 4) -> list[list[SliceComponent]]:
    all_components: list[list[SliceComponent]] = []
    for z, edge_slice in enumerate(np.asarray(edges, dtype=bool)):
        filled = ndi.binary_fill_holes(edge_slice)
        labels, count = ndi.label(filled)
        components: list[SliceComponent] = []
        for local_label, bounds in enumerate(ndi.find_objects(labels), start=1):
            if bounds is None:
                continue
            y_slice, x_slice = bounds
            cropped_mask = labels[bounds] == local_label
            area = int(cropped_mask.sum())
            if area < min_size:
                continue
            coords = np.argwhere(cropped_mask)
            offset = np.array([y_slice.start, x_slice.start], dtype=np.float64)
            components.append(
                SliceComponent(
                    slice_index=z,
                    local_label=local_label,
                    bbox_yx=(
                        int(y_slice.start),
                        int(y_slice.stop),
                        int(x_slice.start),
                        int(x_slice.stop),
                    ),
                    cropped_mask=cropped_mask,
                    centroid_yx=coords.mean(axis=0) + offset,
                    area=area,
                )
            )
        all_components.append(components)
    return all_components


def _slice_label_components(
    slice_labels: np.ndarray,
    min_size: int,
) -> list[list[SliceComponent]]:
    """Read every integer label as a separate 2D region, even when touching."""
    labels = np.asarray(slice_labels)
    if labels.ndim != 3 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Slice-label association expects a 3D integer array")
    all_components: list[list[SliceComponent]] = []
    for z, label_slice in enumerate(labels):
        components: list[SliceComponent] = []
        for local_label in sorted(int(value) for value in np.unique(label_slice) if value > 0):
            local_mask = label_slice == local_label
            # Edited label layers can contain disconnected islands with the
            # same value. Treat each island independently during association.
            islands, island_count = ndi.label(
                local_mask, structure=np.ones((3, 3), dtype=bool)
            )
            for island_id, bounds in enumerate(ndi.find_objects(islands), start=1):
                if bounds is None or island_id > island_count:
                    continue
                y_slice, x_slice = bounds
                cropped_mask = islands[bounds] == island_id
                area = int(cropped_mask.sum())
                if area < max(1, int(min_size)):
                    continue
                coords = np.argwhere(cropped_mask)
                offset = np.array([y_slice.start, x_slice.start], dtype=np.float64)
                components.append(
                    SliceComponent(
                        slice_index=z,
                        local_label=local_label,
                        bbox_yx=(
                            int(y_slice.start),
                            int(y_slice.stop),
                            int(x_slice.start),
                            int(x_slice.stop),
                        ),
                        cropped_mask=cropped_mask,
                        centroid_yx=coords.mean(axis=0) + offset,
                        area=area,
                    )
                )
        all_components.append(components)
    return all_components


def _component_cost(
    previous: SliceComponent,
    current: SliceComponent,
    max_distance: float,
    min_iou: float,
) -> float:
    distance = float(np.linalg.norm(previous.centroid_yx - current.centroid_yx))
    if distance > max_distance:
        return 1e6

    previous_y0, previous_y1, previous_x0, previous_x1 = previous.bbox_yx
    current_y0, current_y1, current_x0, current_x1 = current.bbox_yx
    overlap_y0 = max(previous_y0, current_y0)
    overlap_y1 = min(previous_y1, current_y1)
    overlap_x0 = max(previous_x0, current_x0)
    overlap_x1 = min(previous_x1, current_x1)
    if overlap_y0 < overlap_y1 and overlap_x0 < overlap_x1:
        previous_crop = previous.cropped_mask[
            overlap_y0 - previous_y0 : overlap_y1 - previous_y0,
            overlap_x0 - previous_x0 : overlap_x1 - previous_x0,
        ]
        current_crop = current.cropped_mask[
            overlap_y0 - current_y0 : overlap_y1 - current_y0,
            overlap_x0 - current_x0 : overlap_x1 - current_x0,
        ]
        intersection = int(np.logical_and(previous_crop, current_crop).sum())
    else:
        intersection = 0
    union = previous.area + current.area - intersection
    iou = intersection / union if union else 0.0
    if iou < float(min_iou):
        return 1e6
    area_change = abs(previous.area - current.area) / max(previous.area, current.area, 1)
    return distance / max(max_distance, 1e-6) + (1.0 - iou) + 0.25 * area_change


def _associate_components_across_slices(
    components_by_z: list[list[SliceComponent]],
    output_shape: tuple[int, int, int],
    params: GroupingParameters,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    output = np.zeros(output_shape, dtype=np.int32)
    next_group = 1
    active: dict[int, SliceComponent] = {}
    last_seen: dict[int, int] = {}
    group_components: dict[int, list[SliceComponent]] = {}

    for z, components in enumerate(components_by_z):
        active = {
            group_id: component
            for group_id, component in active.items()
            if z - last_seen[group_id] <= params.max_missing_slices + 1
        }
        last_seen = {
            group_id: last_seen[group_id]
            for group_id in active
        }
        eligible = [
            (group_id, component)
            for group_id, component in active.items()
        ]
        assignments: dict[int, int] = {}
        if eligible and components:
            previous_centroids = np.stack(
                [component.centroid_yx for _, component in eligible],
                axis=0,
            )
            tree = cKDTree(previous_centroids)
            maximum_gap = max(
                z - component.slice_index for _, component in eligible
            )
            search_radius = params.max_centroid_distance * max(1, maximum_gap)
            candidates: list[tuple[float, int, int]] = []
            for current_index, current in enumerate(components):
                for previous_index in tree.query_ball_point(
                    current.centroid_yx,
                    search_radius,
                ):
                    previous = eligible[previous_index][1]
                    cost = _component_cost(
                        previous,
                        current,
                        params.max_centroid_distance
                        * max(1, z - previous.slice_index),
                        params.min_iou,
                    )
                    if cost < 1e5:
                        candidates.append((cost, previous_index, current_index))

            claimed_previous: set[int] = set()
            for _, previous_index, current_index in sorted(candidates):
                if (
                    previous_index in claimed_previous
                    or current_index in assignments
                ):
                    continue
                assignments[current_index] = eligible[previous_index][0]
                claimed_previous.add(previous_index)

        for index, component in enumerate(components):
            group_id = assignments.get(index)
            if group_id is None:
                group_id = next_group
                next_group += 1
            y0, y1, x0, x1 = component.bbox_yx
            output_crop = output[z, y0:y1, x0:x1]
            output_crop[component.cropped_mask] = group_id
            active[group_id] = component
            last_seen[group_id] = z
            group_components.setdefault(group_id, []).append(component)

    # Deterministically filter and renumber by mean Z/Y/X.
    descriptors: list[tuple[tuple[float, float, float], int]] = []
    for group_id, history in group_components.items():
        if len(history) < max(1, int(params.min_slices_per_cell)):
            continue
        total_area = sum(component.area for component in history)
        mean_z = sum(
            component.slice_index * component.area for component in history
        ) / total_area
        mean_yx = sum(
            (
                component.centroid_yx * component.area
                for component in history
            ),
            start=np.zeros(2, dtype=np.float64),
        ) / total_area
        descriptors.append(((mean_z, float(mean_yx[0]), float(mean_yx[1])), group_id))
    descriptors.sort()

    lookup = np.zeros(next_group, dtype=np.int32)
    seeds: dict[int, np.ndarray] = {}
    for new_id, (_, old_id) in enumerate(descriptors, start=1):
        lookup[old_id] = new_id
        history = group_components[old_id]
        mean_yx = np.mean(
            [component.centroid_yx for component in history],
            axis=0,
        )
        seeds[new_id] = np.array(
            [
                mean_yx[1],
                mean_yx[0],
                float(np.mean([component.slice_index for component in history])),
            ],
            dtype=np.float64,
        )
    renumbered = lookup[output]
    return renumbered, seeds


def group_edges_across_slices(
    edges: np.ndarray,
    params: GroupingParameters,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Compatibility helper that fills boolean edges before association."""
    edge_array = np.asarray(edges, dtype=bool)
    if edge_array.ndim != 3:
        raise ValueError("Cross-slice edge grouping expects a 3D ZYX mask")
    components_by_z = _slice_components(
        edge_array,
        min_size=max(1, int(params.min_slice_area)),
    )
    return _associate_components_across_slices(
        components_by_z,
        edge_array.shape,
        params,
    )


def associate_slice_labels(
    slice_labels: np.ndarray,
    params: GroupingParameters,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Associate distinct integer regions through Z without boolean merging."""
    labels = np.asarray(slice_labels)
    components_by_z = _slice_label_components(
        labels,
        min_size=max(1, int(params.min_slice_area)),
    )
    return _associate_components_across_slices(
        components_by_z,
        tuple(int(value) for value in labels.shape),
        params,
    )


def slice_centers_from_labels(
    group_labels_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one ZYX centroid and repeated cross-Z ID per visible slice region."""
    labels = np.asarray(group_labels_raw, dtype=np.int32)
    if labels.ndim != 3:
        raise ValueError("Slice center extraction expects a 3D label volume")
    points: list[tuple[float, float, float]] = []
    ids: list[int] = []
    for z, label_slice in enumerate(labels):
        for label_id in sorted(int(value) for value in np.unique(label_slice) if value > 0):
            islands, _ = ndi.label(
                label_slice == label_id,
                structure=np.ones((3, 3), dtype=bool),
            )
            for region in measure.regionprops(islands):
                centroid_y, centroid_x = region.centroid
                points.append((float(z), float(centroid_y), float(centroid_x)))
                ids.append(label_id)
    return (
        np.asarray(points, dtype=np.float32).reshape((-1, 3)),
        np.asarray(ids, dtype=np.int32),
    )


def _fit_separator_plane_from_lines(
    guides: list[tuple[int, np.ndarray]],
) -> tuple[np.ndarray, float]:
    """Fit one oriented ZYX plane from per-slice separator lines.

    The plane uses the shared in-slice direction of every guide and a
    least-squares linear shift through Z. With only one guide, the line is
    extruded vertically through the stack. The returned normal is oriented so
    its positive side points toward increasing X whenever possible.
    """
    direction_projector = np.zeros((2, 2), dtype=np.float64)
    guide_centers: list[tuple[float, np.ndarray]] = []

    for z_index, line in sorted(guides, key=lambda item: item[0]):
        start_yx = np.asarray(line[0, 1:3], dtype=np.float64)
        end_yx = np.asarray(line[-1, 1:3], dtype=np.float64)
        direction_yx = end_yx - start_yx
        length = float(np.linalg.norm(direction_yx))
        if length < 0.5:
            raise ValueError(
                f"Manual separator on Z={z_index} is too short to define a plane"
            )
        unit_direction = direction_yx / length
        # The outer product makes reversed line endpoints equivalent.
        direction_projector += np.outer(unit_direction, unit_direction)
        guide_centers.append(
            (
                float(z_index),
                np.mean(np.asarray(line[:, 1:3], dtype=np.float64), axis=0),
            )
        )

    _, eigenvectors = np.linalg.eigh(direction_projector)
    shared_direction_yx = eigenvectors[:, -1]
    normal_yx = np.array(
        [-shared_direction_yx[1], shared_direction_yx[0]],
        dtype=np.float64,
    )
    normal_yx /= float(np.linalg.norm(normal_yx))

    # Positive signed distance is the right-hand (larger-X) side. A horizontal
    # separator has no X ordering, so increasing Y is the deterministic fallback.
    if abs(float(normal_yx[1])) > 1e-8:
        if normal_yx[1] < 0.0:
            normal_yx *= -1.0
    elif normal_yx[0] < 0.0:
        normal_yx *= -1.0

    z_positions = np.asarray(
        [z_index for z_index, _ in guide_centers],
        dtype=np.float64,
    )
    slice_intercepts = -np.asarray(
        [
            float(np.dot(normal_yx, center_yx))
            for _, center_yx in guide_centers
        ],
        dtype=np.float64,
    )
    if np.ptp(z_positions) > 0.0:
        design = np.column_stack(
            (z_positions, np.ones(z_positions.shape[0], dtype=np.float64))
        )
        z_slope, offset = np.linalg.lstsq(
            design,
            slice_intercepts,
            rcond=None,
        )[0]
    else:
        z_slope = 0.0
        offset = float(np.mean(slice_intercepts))

    normal_zyx = np.array(
        [float(z_slope), float(normal_yx[0]), float(normal_yx[1])],
        dtype=np.float64,
    )
    scale = float(np.linalg.norm(normal_zyx))
    return normal_zyx / scale, float(offset) / scale


def split_labels_with_line_guides(
    labels_raw: np.ndarray,
    guide_lines_zyx: Iterable[np.ndarray],
    separator_radius: int = 1,
    minimum_target_overlap_fraction: float = 0.0,
) -> ManualSplitResult:
    """Partition one 3D label into two children using per-Z line guides.

    Each guide must lie on one source Z slice. Its target is inferred from the
    positive label owning the largest share of the undilated centerline. When a
    line crosses several labels, ``minimum_target_overlap_fraction`` controls
    how dominant that target must be; zero disables this optional guard. Radius
    expansion happens only after target selection and is clipped to that target,
    so a nearby label touched only by the expanded brush cannot reject an
    otherwise valid guide.

    All guides for one target are fitted into a single separation plane. Every
    target voxel is classified directly by its signed side of that plane:
    smaller-X voxels keep the source label as the left child and larger-X
    voxels receive the new right-child label. A single line is extruded through
    Z; lines on several slices control the plane's tilt. No connectivity test,
    separator gap, or watershed propagation is required.
    """
    labels = np.asarray(labels_raw, dtype=np.int32)
    if labels.ndim != 3:
        raise ValueError("Manual split guidance expects a 3D ZYX label volume")
    radius = max(0, int(separator_radius))
    minimum_target_share = float(minimum_target_overlap_fraction)
    if not 0.0 <= minimum_target_share <= 1.0:
        raise ValueError(
            "Manual separator minimum target overlap fraction must be in [0.0, 1.0]"
        )
    separator_mask = np.zeros(labels.shape, dtype=bool)
    source_labels: set[int] = set()
    source_guides: dict[int, list[tuple[int, np.ndarray]]] = {}
    line_count = 0

    for line_index, raw_line in enumerate(guide_lines_zyx, start=1):
        line = np.asarray(raw_line, dtype=np.float64)
        if line.ndim != 2 or line.shape[0] < 2 or line.shape[1] != 3:
            raise ValueError(
                f"Manual split guide {line_index} must contain at least two "
                "ZYX vertices"
            )
        if not np.all(np.isfinite(line)):
            raise ValueError(
                f"Manual split guide {line_index} contains non-finite coordinates"
            )
        z_values = np.rint(line[:, 0]).astype(np.int64)
        if np.any(np.abs(line[:, 0] - z_values) > 0.25) or np.any(
            z_values != z_values[0]
        ):
            raise ValueError(
                f"Manual split guide {line_index} must stay on one Z slice"
            )
        z_index = int(z_values[0])
        if not 0 <= z_index < labels.shape[0]:
            raise ValueError(
                f"Manual split guide {line_index} lies outside the Z stack"
            )

        line_mask = np.zeros(labels.shape[1:], dtype=bool)
        for start, end in zip(line[:-1], line[1:]):
            rows, columns = draw.line(
                int(round(float(start[1]))),
                int(round(float(start[2]))),
                int(round(float(end[1]))),
                int(round(float(end[2]))),
            )
            valid = (
                (rows >= 0)
                & (rows < labels.shape[1])
                & (columns >= 0)
                & (columns < labels.shape[2])
            )
            line_mask[rows[valid], columns[valid]] = True

        centerline_labels = labels[z_index][line_mask]
        centerline_labels = centerline_labels[centerline_labels > 0]
        if centerline_labels.size == 0:
            raise ValueError(
                f"Manual split guide {line_index} does not cross a colored segment"
            )
        touched, counts = np.unique(centerline_labels, return_counts=True)
        largest_count = int(counts.max(initial=0))
        largest_indices = np.flatnonzero(counts == largest_count)
        target_share = largest_count / int(counts.sum())
        midpoint = np.rint(np.mean(line[:, 1:3], axis=0)).astype(np.int64)
        midpoint_label = 0
        if (
            0 <= midpoint[0] < labels.shape[1]
            and 0 <= midpoint[1] < labels.shape[2]
        ):
            midpoint_label = int(labels[z_index, midpoint[0], midpoint[1]])
        tied_labels = touched[largest_indices]
        if midpoint_label in tied_labels:
            source_label = midpoint_label
        else:
            source_label = int(tied_labels[0])
        if target_share < minimum_target_share:
            raise ValueError(
                f"Manual split guide {line_index} is ambiguous: its strongest "
                f"target owns {target_share:.1%} of colored centerline pixels, "
                f"below the required {minimum_target_share:.1%}. Redraw the "
                "line or lower 'Optional line-target guard'."
            )

        if radius:
            line_mask = morphology.dilation(
                line_mask,
                footprint=morphology.disk(radius),
            )
        separator_mask[z_index] |= line_mask & (
            labels[z_index] == source_label
        )
        source_labels.add(source_label)
        source_guides.setdefault(source_label, []).append((z_index, line.copy()))
        line_count += 1

    if line_count == 0:
        raise ValueError("Draw at least one manual separator line first")

    output = labels.copy()
    next_label = int(labels.max(initial=0)) + 1
    source_to_children: dict[int, tuple[int, int]] = {}
    affected_slices: set[int] = set()
    for source_label in sorted(source_labels):
        source_mask = labels == source_label
        cut_mask = separator_mask & source_mask
        if not np.any(cut_mask):
            raise ValueError(
                f"No separator pixels intersect segment {source_label - 1}"
            )
        normal_zyx, offset = _fit_separator_plane_from_lines(
            source_guides[source_label]
        )
        zz, yy, xx = np.ogrid[
            : labels.shape[0],
            : labels.shape[1],
            : labels.shape[2],
        ]
        signed_side = (
            normal_zyx[0] * zz
            + normal_zyx[1] * yy
            + normal_zyx[2] * xx
            + offset
        )
        # Least-squares arithmetic can put a mathematically on-plane integer
        # coordinate a few ulps above zero. Keep that boundary deterministically
        # with the left child instead of letting roundoff flip its ID.
        plane_tolerance = 1e-7
        negative_child = source_mask & (signed_side <= plane_tolerance)
        positive_child = source_mask & (signed_side > plane_tolerance)
        if not np.any(negative_child) or not np.any(positive_child):
            raise ValueError(
                f"Separator plane for segment {source_label - 1} does not pass "
                "through the target with pixels on both sides"
            )

        child_masks = [negative_child, positive_child]
        child_keys = []
        for child_mask in child_masks:
            coordinates = np.argwhere(child_mask)
            child_keys.append(
                (
                    float(np.mean(coordinates[:, 2])),
                    float(np.mean(coordinates[:, 1])),
                )
            )
        if child_keys[1] < child_keys[0]:
            child_masks.reverse()

        child_ids = (source_label, next_label)
        next_label += 1
        output[source_mask] = 0
        for child_id, child_mask in zip(child_ids, child_masks):
            output[child_mask] = child_id

        source_to_children[source_label] = child_ids
        affected_slices.update(
            int(value)
            for value in np.flatnonzero(np.any(source_mask, axis=(1, 2)))
        )

    return ManualSplitResult(
        labels=output.astype(np.int32, copy=False),
        separator_mask=separator_mask,
        source_to_children=source_to_children,
        affected_slices=tuple(sorted(affected_slices)),
    )


def merge_two_cell_labels(
    labels_zyx: np.ndarray,
    kept_label: int,
    absorbed_label: int,
) -> ManualMergeResult:
    """Relabel one complete 3D cell as another without changing foreground."""

    labels = np.asarray(labels_zyx)
    if labels.ndim != 3 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Manual merge requires one integer 3D label volume")
    kept = int(kept_label)
    absorbed = int(absorbed_label)
    if kept <= 0 or absorbed <= 0:
        raise ValueError("Manual merge requires two non-background cells")
    if kept == absorbed:
        raise ValueError("Manual merge requires two different cells")
    kept_mask = labels == kept
    absorbed_mask = labels == absorbed
    if not np.any(kept_mask):
        raise ValueError(f"Merge target cell ID {kept - 1} is no longer present")
    if not np.any(absorbed_mask):
        raise ValueError(
            f"Merge source cell ID {absorbed - 1} is no longer present"
        )

    output = labels.astype(np.int32, copy=True)
    output[absorbed_mask] = kept
    affected = np.flatnonzero(
        np.any(kept_mask | absorbed_mask, axis=(1, 2))
    )
    return ManualMergeResult(
        labels=output,
        kept_label=kept,
        absorbed_label=absorbed,
        affected_slices=tuple(int(value) for value in affected),
    )


def merge_cell_labels(
    labels_zyx: np.ndarray,
    label_ids: Iterable[int],
) -> MultiLabelMergeResult:
    """Merge two or more selected 3D cells, keeping the smallest label ID.

    The operation changes label identity only: it neither removes foreground
    voxels nor creates bridge voxels between selected regions.
    """

    labels = np.asarray(labels_zyx)
    if labels.ndim != 3 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Manual merge requires one integer 3D label volume")

    try:
        requested = tuple(label_ids)
    except TypeError as error:
        raise ValueError(
            "Manual merge requires at least two distinct cell IDs"
        ) from error
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        for value in requested
    ):
        raise ValueError("Manual merge requires integer cell IDs")

    selected = tuple(sorted({int(value) for value in requested}))
    if len(selected) < 2:
        raise ValueError("Manual merge requires at least two distinct cell IDs")
    if selected[0] <= 0:
        raise ValueError("Manual merge requires non-background cells")

    present = set(int(value) for value in np.unique(labels))
    missing = tuple(value for value in selected if value not in present)
    if missing:
        formatted = ", ".join(str(value) for value in missing)
        noun = "label is" if len(missing) == 1 else "labels are"
        raise ValueError(f"Selected cell {noun} no longer present: {formatted}")

    kept = selected[0]
    absorbed = selected[1:]
    selected_mask = np.isin(labels, selected)
    output = labels.astype(np.int32, copy=True)
    output[np.isin(labels, absorbed)] = kept
    affected = np.flatnonzero(np.any(selected_mask, axis=(1, 2)))
    return MultiLabelMergeResult(
        labels=output,
        kept_label=kept,
        absorbed_labels=absorbed,
        affected_slices=tuple(int(value) for value in affected),
    )


def _bounded_chunk_intensity_weights(
    values: np.ndarray,
    *,
    weight_exponent: float,
    geometric_floor: float,
) -> tuple[np.ndarray, bool]:
    """Return robust local brightness weights with a geometric mass floor."""

    samples = np.asarray(values, dtype=np.float64)
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return np.ones(samples.shape, dtype=np.float64), True
    low, high = np.percentile(
        finite,
        ROBUST_CELL_CENTER_INTENSITY_PERCENTILES,
    )
    spread = float(high - low)
    scale = max(1.0, abs(float(low)), abs(float(high)))
    if not np.isfinite(spread) or spread <= np.finfo(np.float64).eps * scale:
        return np.ones(samples.shape, dtype=np.float64), True
    normalized = np.clip((samples - float(low)) / spread, 0.0, 1.0)
    normalized[~np.isfinite(normalized)] = 0.0
    weights = geometric_floor + (1.0 - geometric_floor) * (
        normalized ** weight_exponent
    )
    return weights.astype(np.float64, copy=False), False


def _robust_centerline_slice_inliers(
    slice_centers_zyx: np.ndarray,
    slice_weights: np.ndarray,
    equivalent_radii: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, tuple[str, ...]]:
    """Select the dominant tilted XY centerline through ordered Z slices."""

    points = np.asarray(slice_centers_zyx, dtype=np.float64)
    masses = np.asarray(slice_weights, dtype=np.float64)
    radii = np.asarray(equivalent_radii, dtype=np.float64)
    count = int(points.shape[0])
    if count < 3:
        return (
            np.ones(count, dtype=bool),
            np.zeros(count, dtype=np.float64),
            0.0,
            (),
        )

    z_values = points[:, 0]
    xy_values = points[:, [2, 1]]
    positive_radii = radii[radii > 0.0]
    median_radius = (
        float(np.median(positive_radii))
        if positive_radii.size
        else 1.0
    )
    if count == 3:
        # Three slices cannot support the full majority/RANSAC procedure, but
        # a grossly displaced endpoint or middle slice is still identifiable
        # when the other two centers form a compact, low-motion pair. Keep the
        # rule deliberately conservative so ordinary tilt and mild curvature
        # remain untouched.
        hypotheses: list[tuple[float, float, int]] = []
        for omitted in range(3):
            retained = np.asarray(
                [index for index in range(3) if index != omitted],
                dtype=np.int64,
            )
            first, second = retained
            dz = float(z_values[second] - z_values[first])
            if abs(dz) <= 1e-12:
                continue
            slope = (xy_values[second] - xy_values[first]) / dz
            predicted = (
                xy_values[first]
                + slope * float(z_values[omitted] - z_values[first])
            )
            residual = float(
                np.linalg.norm(xy_values[omitted] - predicted)
            )
            speed = float(np.linalg.norm(slope))
            hypotheses.append(
                (speed, -residual, omitted)
            )
        if hypotheses:
            speed, negative_residual, omitted = min(hypotheses)
            residual = -float(negative_residual)
            gross_threshold = max(
                ROBUST_CELL_CENTER_THREE_SLICE_GROSS_MINIMUM_PIXELS,
                ROBUST_CELL_CENTER_THREE_SLICE_GROSS_RADIUS_FACTOR
                * median_radius,
            )
            if (
                speed <= gross_threshold
                and residual
                > ROBUST_CELL_CENTER_THREE_SLICE_RESIDUAL_MULTIPLIER
                * gross_threshold
            ):
                inliers = np.ones(3, dtype=bool)
                inliers[int(omitted)] = False
                residuals = np.zeros(3, dtype=np.float64)
                residuals[int(omitted)] = residual
                return (
                    inliers,
                    residuals,
                    float(
                        ROBUST_CELL_CENTER_THREE_SLICE_RESIDUAL_MULTIPLIER
                        * gross_threshold
                    ),
                    (
                        "excluded_1_xy_outlier_slices",
                        "three_slice_gross_outlier_rule",
                    ),
                )
        return (
            np.ones(3, dtype=bool),
            np.zeros(3, dtype=np.float64),
            float(
                max(
                    ROBUST_CELL_CENTER_THREE_SLICE_GROSS_MINIMUM_PIXELS,
                    ROBUST_CELL_CENTER_THREE_SLICE_GROSS_RADIUS_FACTOR
                    * median_radius,
                )
            ),
            (),
        )

    seed_tolerance = max(1.5, 0.25 * median_radius)
    positive_masses = masses[masses > 0.0]
    median_mass = (
        float(np.median(positive_masses))
        if positive_masses.size
        else 1.0
    )
    capped_masses = np.minimum(masses, max(1e-12, 2.0 * median_mass))

    candidates: list[tuple[np.ndarray, np.ndarray]] = [
        (
            np.zeros(2, dtype=np.float64),
            np.median(xy_values, axis=0),
        )
    ]
    # Interpolated stacks can contain hundreds of Z slices. A deterministic,
    # evenly spaced subset supplies ample two-point hypotheses without turning
    # one slider release into tens of thousands of Python-level iterations per
    # cell.
    if count <= ROBUST_CELL_CENTER_MAX_HYPOTHESIS_SLICES:
        candidate_indices = np.arange(count, dtype=np.int64)
    else:
        candidate_indices = np.unique(
            np.rint(
                np.linspace(
                    0,
                    count - 1,
                    ROBUST_CELL_CENTER_MAX_HYPOTHESIS_SLICES,
                )
            ).astype(np.int64)
        )
    for first_offset, first in enumerate(candidate_indices[:-1]):
        for second in candidate_indices[first_offset + 1 :]:
            dz = float(z_values[second] - z_values[first])
            if abs(dz) <= 1e-12:
                continue
            slope = (xy_values[second] - xy_values[first]) / dz
            intercept = xy_values[first] - slope * z_values[first]
            candidates.append((slope, intercept))

    best_inliers = np.ones(count, dtype=bool)
    best_key: tuple[float, ...] | None = None
    for slope, intercept in candidates:
        predicted = z_values[:, None] * slope[None, :] + intercept[None, :]
        residuals = np.linalg.norm(xy_values - predicted, axis=1)
        inliers = residuals <= seed_tolerance
        inlier_count = int(np.count_nonzero(inliers))
        if inlier_count < 2:
            continue
        key = (
            float(inlier_count),
            float(np.sum(capped_masses[inliers])),
            -float(np.median(residuals[inliers])),
            -float(np.linalg.norm(slope)),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_inliers = inliers

    design = np.column_stack(
        [z_values[best_inliers], np.ones(np.count_nonzero(best_inliers))]
    )
    fit_weights = np.sqrt(
        np.maximum(capped_masses[best_inliers], np.finfo(np.float64).eps)
    )
    coefficients = np.linalg.lstsq(
        design * fit_weights[:, None],
        xy_values[best_inliers] * fit_weights[:, None],
        rcond=None,
    )[0]
    predicted = (
        z_values[:, None] * coefficients[0][None, :]
        + coefficients[1][None, :]
    )
    residuals = np.linalg.norm(xy_values - predicted, axis=1)
    consensus_residuals = residuals[best_inliers]
    residual_median = float(np.median(consensus_residuals))
    residual_mad = float(
        np.median(np.abs(consensus_residuals - residual_median))
    )
    robust_threshold = (
        residual_median + 3.5 * 1.4826 * residual_mad
    )
    threshold_cap = max(seed_tolerance, 0.75 * median_radius)
    threshold = min(
        max(seed_tolerance, robust_threshold),
        threshold_cap,
    )
    inliers = residuals <= threshold + 1e-9
    if np.count_nonzero(inliers) < 2:
        inliers = best_inliers

    warnings: list[str] = []
    required_majority = max(
        3,
        int(
            np.ceil(
                ROBUST_CELL_CENTER_MINIMUM_CONSENSUS_FRACTION
                * count
            )
        ),
    )
    if np.count_nonzero(inliers) < required_majority:
        # Without a clear majority, choosing one of several plausible
        # centerlines would silently move the modeled cell. Retaining all
        # slices is the conservative, inspectable fallback.
        inliers = np.ones(count, dtype=bool)
        warnings.append("ambiguous_centerline_kept_all_slices")
    excluded = count - int(np.count_nonzero(inliers))
    if excluded:
        warnings.append(f"excluded_{excluded}_xy_outlier_slices")
    return inliers, residuals, float(threshold), tuple(warnings)


def robust_cell_centers_from_labels(
    labels_zyx: np.ndarray,
    intensity_zyx: np.ndarray,
    *,
    candidate_points_zyx: np.ndarray | None = None,
    candidate_ids: np.ndarray | None = None,
    candidate_accepted: np.ndarray | None = None,
    weight_exponent: float = ROBUST_CELL_CENTER_WEIGHT_EXPONENT,
    geometric_floor: float = ROBUST_CELL_CENTER_GEOMETRIC_FLOOR,
) -> dict[int, RobustCellCenterEstimate]:
    """Estimate one robust, chunk-derived XYZ center per positive label.

    Every occupied source slice supplies one local intensity-weighted XY
    centroid. A deterministic robust linear centerline preserves legitimate
    tilt through Z while excluding displaced slice centroids. The retained
    voxels then determine all three final coordinates, so Z is weighted by the
    actual retained cell mass instead of by the number of marker observations.

    Optional reviewed marker observations do not supply the final coordinates.
    Accepted observations select the intended connected component when one
    label contains detached islands. Rejected observations supply no support,
    but they never invalidate an otherwise solid chunk or remove a complete Z
    slice from the geometry-derived center.
    """

    labels = np.asarray(labels_zyx)
    intensity = np.asarray(intensity_zyx)
    if labels.ndim != 3 or intensity.ndim != 3:
        raise ValueError("Robust cell centers require matching 3D ZYX volumes")
    if labels.shape != intensity.shape:
        raise ValueError(
            "Robust cell-center labels and intensity must have matching shapes"
        )
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Robust cell centers require integer labels")
    candidate_values = (
        candidate_points_zyx,
        candidate_ids,
        candidate_accepted,
    )
    candidates_provided = any(value is not None for value in candidate_values)
    if candidates_provided and not all(
        value is not None for value in candidate_values
    ):
        raise ValueError(
            "Candidate points, IDs, and accepted flags must be provided together"
        )
    if candidates_provided:
        reviewed_points = np.asarray(
            candidate_points_zyx,
            dtype=np.float64,
        )
        reviewed_ids = np.asarray(candidate_ids, dtype=np.int32)
        reviewed_accepted = np.asarray(candidate_accepted, dtype=bool)
        if reviewed_points.ndim != 2 or reviewed_points.shape[1:] != (3,):
            raise ValueError(
                "Candidate points must have shape (N, 3) in ZYX order"
            )
        if (
            reviewed_ids.ndim != 1
            or reviewed_accepted.ndim != 1
            or len(reviewed_points) != len(reviewed_ids)
            or len(reviewed_points) != len(reviewed_accepted)
        ):
            raise ValueError(
                "Candidate points, IDs, and accepted flags must have matching rows"
            )
        if not np.all(np.isfinite(reviewed_points)):
            raise ValueError("Candidate points must contain only finite values")
    else:
        reviewed_points = np.empty((0, 3), dtype=np.float64)
        reviewed_ids = np.empty((0,), dtype=np.int32)
        reviewed_accepted = np.empty((0,), dtype=bool)
    exponent = float(weight_exponent)
    floor = float(geometric_floor)
    if not np.isfinite(exponent) or exponent <= 0.0:
        raise ValueError("Cell-center weight exponent must be positive")
    if not np.isfinite(floor) or floor < 0.0 or floor > 1.0:
        raise ValueError("Cell-center geometric weight floor must be in [0, 1]")

    label_ids = sorted(
        int(value) for value in np.unique(labels) if int(value) > 0
    )
    object_bounds = ndi.find_objects(labels)
    estimates: dict[int, RobustCellCenterEstimate] = {}
    for label_id in label_ids:
        bounds = (
            object_bounds[label_id - 1]
            if label_id - 1 < len(object_bounds)
            else None
        )
        if bounds is None:
            continue
        local_mask = labels[bounds] == label_id
        component_labels, component_count = ndi.label(
            local_mask,
            structure=np.ones((3, 3, 3), dtype=bool),
        )
        local_coordinates = np.argwhere(local_mask)
        if local_coordinates.size == 0:
            continue
        origin_zyx = np.asarray(
            [int(axis.start) for axis in bounds],
            dtype=np.int64,
        )
        candidate_rows = np.flatnonzero(reviewed_ids == label_id)
        candidate_components = np.zeros(
            candidate_rows.shape[0],
            dtype=np.int32,
        )
        if candidate_rows.size:
            for local_index, candidate_row in enumerate(candidate_rows):
                local_point = (
                    reviewed_points[candidate_row] - origin_zyx
                )
                rounded_point = np.rint(local_point).astype(np.int64)
                if np.all(rounded_point >= 0) and np.all(
                    rounded_point < np.asarray(local_mask.shape)
                ):
                    component_id = int(
                        component_labels[tuple(rounded_point)]
                    )
                else:
                    component_id = 0
                if component_id == 0 and reviewed_accepted[candidate_row]:
                    distances_squared = np.sum(
                        (
                            local_coordinates.astype(np.float64)
                            - local_point[None, :]
                        )
                        ** 2,
                        axis=1,
                    )
                    closest = int(np.argmin(distances_squared))
                    # A removed isolated chunk must not be reassigned to a
                    # distant surviving component merely because its old
                    # candidate retains the same label ID.
                    if float(distances_squared[closest]) <= 2.5**2:
                        component_id = int(
                            component_labels[
                                tuple(local_coordinates[closest])
                            ]
                        )
                candidate_components[local_index] = component_id

        component_sizes = np.bincount(
            component_labels.ravel(),
            minlength=component_count + 1,
        )
        component_support = np.zeros(
            component_count + 1,
            dtype=np.int64,
        )
        if candidate_rows.size:
            for candidate_row, component_id in zip(
                candidate_rows,
                candidate_components,
            ):
                if component_id > 0 and reviewed_accepted[candidate_row]:
                    component_support[component_id] += 1
        selected_component, selected_components = (
            _select_z_aligned_component_group(
                component_labels,
                support_counts=component_support,
            )
        )
        accepted_candidate_count = int(
            np.sum(component_support[list(selected_components)])
        )
        local_mask = np.isin(component_labels, selected_components)
        local_coordinates = np.argwhere(local_mask)
        global_coordinates = local_coordinates + origin_zyx[None, :]
        values = np.asarray(intensity[bounds], dtype=np.float64)[
            tuple(local_coordinates.T)
        ]
        voxel_weights, used_uniform_weights = (
            _bounded_chunk_intensity_weights(
                values,
                weight_exponent=exponent,
                geometric_floor=floor,
            )
        )

        slice_points: list[list[float]] = []
        slice_masses: list[float] = []
        equivalent_radii: list[float] = []
        multi_component_slices = 0
        for global_z in np.unique(global_coordinates[:, 0]):
            selected = global_coordinates[:, 0] == global_z
            coordinates = global_coordinates[selected]
            weights = voxel_weights[selected]
            mass = float(np.sum(weights))
            if not np.isfinite(mass) or mass <= 0.0:
                weights = np.ones(coordinates.shape[0], dtype=np.float64)
                mass = float(coordinates.shape[0])
            center_yx = np.sum(
                coordinates[:, 1:].astype(np.float64) * weights[:, None],
                axis=0,
            ) / mass
            slice_points.append(
                [float(global_z), float(center_yx[0]), float(center_yx[1])]
            )
            slice_masses.append(mass)
            equivalent_radii.append(
                float(np.sqrt(coordinates.shape[0] / np.pi))
            )
            local_z = int(global_z) - int(origin_zyx[0])
            components, slice_component_count = ndi.label(
                local_mask[local_z],
                structure=np.ones((3, 3), dtype=bool),
            )
            if slice_component_count > 1:
                sizes = np.bincount(components.ravel())[1:]
                sizes.sort()
                if sizes.size > 1 and sizes[-2] >= 0.35 * sizes[-1]:
                    multi_component_slices += 1

        points_array = np.asarray(slice_points, dtype=np.float64)
        masses_array = np.asarray(slice_masses, dtype=np.float64)
        radii_array = np.asarray(equivalent_radii, dtype=np.float64)
        inliers, residuals, threshold, centerline_warnings = (
            _robust_centerline_slice_inliers(
                points_array,
                masses_array,
                radii_array,
            )
        )
        retained_z = points_array[inliers, 0].astype(np.int64)
        retained_voxels = np.isin(global_coordinates[:, 0], retained_z)
        if not np.any(retained_voxels):
            retained_voxels = np.ones(global_coordinates.shape[0], dtype=bool)
            inliers = np.ones(points_array.shape[0], dtype=bool)
        retained_coordinates = global_coordinates[retained_voxels]
        retained_weights = voxel_weights[retained_voxels]
        retained_mass = float(np.sum(retained_weights))
        if not np.isfinite(retained_mass) or retained_mass <= 0.0:
            retained_weights = np.ones(
                retained_coordinates.shape[0],
                dtype=np.float64,
            )
            retained_mass = float(retained_coordinates.shape[0])
        center_zyx = np.sum(
            retained_coordinates.astype(np.float64)
            * retained_weights[:, None],
            axis=0,
        ) / retained_mass

        rounded = np.rint(center_zyx).astype(np.int64)
        rounded = np.clip(
            rounded,
            np.zeros(3, dtype=np.int64),
            np.asarray(labels.shape, dtype=np.int64) - 1,
        )
        rounded_local = rounded - origin_zyx
        rounded_in_bounds = np.all(rounded_local >= 0) and np.all(
            rounded_local < np.asarray(component_labels.shape)
        )
        center_in_selected_support = (
            rounded_in_bounds
            and int(component_labels[tuple(rounded_local)])
            in selected_components
        )
        inferred_support = _fill_short_internal_z_gaps(local_mask)
        center_in_inferred_gap = bool(
            rounded_in_bounds
            and not center_in_selected_support
            and inferred_support[tuple(rounded_local)]
        )
        projected_inside = not (
            center_in_selected_support or center_in_inferred_gap
        )
        if projected_inside:
            distances_squared = np.sum(
                (
                    retained_coordinates.astype(np.float64)
                    - center_zyx[None, :]
                )
                ** 2,
                axis=1,
            )
            minimum_distance = float(np.min(distances_squared))
            closest = np.flatnonzero(
                np.isclose(
                    distances_squared,
                    minimum_distance,
                    rtol=0.0,
                    atol=1e-12,
                )
            )
            if closest.size > 1:
                chosen = int(
                    closest[np.argmax(retained_weights[closest])]
                )
            else:
                chosen = int(closest[0])
            center_zyx = retained_coordinates[chosen].astype(np.float64)

        warnings = list(centerline_warnings)
        if used_uniform_weights:
            warnings.append("uniform_intensity_weight_fallback")
        ignored_component_voxels = int(
            np.sum(component_sizes[1:])
            - np.sum(component_sizes[list(selected_components)])
        )
        if len(selected_components) > 1:
            warnings.append(
                f"joined_{len(selected_components)}_z_aligned_components"
            )
        if ignored_component_voxels > 0:
            warnings.append(
                f"selected_{len(selected_components)}_of_"
                f"{component_count}_connected_components"
            )
        if multi_component_slices:
            warnings.append(
                f"{multi_component_slices}_multi_component_slices"
            )
        if projected_inside:
            warnings.append("projected_center_inside_label")
        if center_in_inferred_gap:
            warnings.append("center_inside_inferred_short_z_gap")
        method = ROBUST_CELL_CENTER_METHOD
        estimates[label_id] = RobustCellCenterEstimate(
            label_id=label_id,
            center_xyz=center_zyx[::-1].copy(),
            slice_centers_zyx=points_array.astype(np.float32),
            slice_weights=masses_array.astype(np.float32),
            slice_inliers=inliers.astype(bool),
            slice_residuals_xy=residuals.astype(np.float32),
            residual_threshold=float(threshold),
            projected_inside=projected_inside,
            center_in_inferred_gap=center_in_inferred_gap,
            selected_component=int(selected_component),
            selected_components=selected_components,
            component_count=int(component_count),
            ignored_component_voxels=ignored_component_voxels,
            accepted_candidate_count=accepted_candidate_count,
            method=method,
            warnings=tuple(warnings),
        )
    return estimates


def group_seeds_from_slice_centers(
    points_zyx: np.ndarray,
    ids: np.ndarray,
    accepted: np.ndarray | None = None,
) -> dict[int, np.ndarray]:
    """Return an XYZ mean for each positive ID among accepted observations."""
    points = np.asarray(points_zyx, dtype=np.float64)
    center_ids = np.asarray(ids, dtype=np.int32)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError("Slice-center points must have shape (N, 3) in ZYX order")
    if center_ids.ndim != 1 or center_ids.shape[0] != points.shape[0]:
        raise ValueError("Slice-center IDs must have shape (N,)")
    if not np.all(np.isfinite(points)):
        raise ValueError("Slice-center points must contain only finite values")

    if accepted is None:
        accepted_mask = np.ones(points.shape[0], dtype=bool)
    else:
        accepted_mask = np.asarray(accepted, dtype=bool)
        if accepted_mask.ndim != 1 or accepted_mask.shape[0] != points.shape[0]:
            raise ValueError("Accepted-center mask must have shape (N,)")

    seeds: dict[int, np.ndarray] = {}
    for label_id in sorted(
        int(value)
        for value in np.unique(center_ids[accepted_mask])
        if value > 0
    ):
        selected = points[accepted_mask & (center_ids == label_id)]
        mean_z, mean_y, mean_x = np.mean(selected, axis=0)
        seeds[label_id] = np.array(
            [mean_x, mean_y, mean_z],
            dtype=np.float64,
        )
    return seeds


def group_seeds_from_labels(group_labels_raw: np.ndarray) -> dict[int, np.ndarray]:
    """Compute the requested unweighted per-slice-average XYZ seed per group."""
    labels = np.asarray(group_labels_raw, dtype=np.int32)
    observations: dict[int, list[tuple[float, float, float]]] = {}
    for z, label_slice in enumerate(labels):
        for region in measure.regionprops(label_slice):
            group_id = int(region.label)
            centroid_y, centroid_x = region.centroid
            observations.setdefault(group_id, []).append(
                (float(z), float(centroid_y), float(centroid_x))
            )
    seeds: dict[int, np.ndarray] = {}
    for group_id, centers in sorted(observations.items()):
        mean_z, mean_y, mean_x = np.mean(centers, axis=0)
        seeds[group_id] = np.array(
            [mean_x, mean_y, mean_z],
            dtype=np.float64,
        )
    return seeds


def renumber_group_labels(group_labels_raw: np.ndarray) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    labels = np.asarray(group_labels_raw, dtype=np.int32)
    descriptors = [
        (tuple(float(value) for value in region.centroid), int(region.label))
        for region in measure.regionprops(labels)
    ]
    descriptors.sort()
    maximum_label = int(labels.max(initial=0))
    lookup = np.zeros(maximum_label + 1, dtype=np.int32)
    for new_id, (_, old_id) in enumerate(descriptors, start=1):
        lookup[old_id] = new_id
    output = lookup[labels]
    return output, group_seeds_from_labels(output)


def interpolate_group_support(
    group_labels_raw: np.ndarray,
    group_id: int,
    ratio: float,
) -> np.ndarray:
    return interpolate_binary_signed_distance(group_labels_raw == group_id, ratio)


def _connectivity_structure(connectivity: int) -> np.ndarray:
    if connectivity == 6:
        return ndi.generate_binary_structure(3, 1)
    if connectivity == 18:
        return ndi.generate_binary_structure(3, 2)
    return np.ones((3, 3, 3), dtype=bool)


def segment_cells(
    interpolated_raw: np.ndarray,
    group_labels_raw: np.ndarray,
    group_seed_xyz: dict[int, np.ndarray],
    z_ratio: float,
    membrane_mask_raw: np.ndarray,
    params: SegmentationParameters,
    prepared_supports: dict[int, np.ndarray] | None = None,
    prepared_membrane: np.ndarray | None = None,
) -> np.ndarray:
    image = robust_normalize(interpolated_raw)
    membrane = (
        np.asarray(prepared_membrane, dtype=bool)
        if prepared_membrane is not None
        else interpolate_binary_signed_distance(membrane_mask_raw, z_ratio)
    )
    output = np.zeros_like(image, dtype=np.int32)
    structure = _connectivity_structure(params.connectivity)

    for group_id in sorted(group_seed_xyz):
        support = (
            np.asarray(prepared_supports[group_id], dtype=bool)
            if prepared_supports is not None and group_id in prepared_supports
            else interpolate_group_support(group_labels_raw, group_id, z_ratio)
        )
        if params.roi_margin > 0:
            support = ndi.binary_dilation(
                support,
                structure=_disk_or_ball(params.roi_margin, 3),
            )
        candidate = (
            (image >= float(params.intensity_threshold))
            & support
            & membrane
        )
        if params.closing_radius > 0:
            candidate = ndi.binary_closing(
                candidate,
                structure=_disk_or_ball(params.closing_radius, 3),
            )
        labels, count = ndi.label(candidate, structure=structure)
        if count == 0:
            # Keep the interpolated edge-derived support as a visible fallback.
            selected = support & membrane
        else:
            seed_xyz = group_seed_xyz[group_id].copy()
            seed_zyx = np.array(
                [seed_xyz[2] * z_ratio, seed_xyz[1], seed_xyz[0]],
                dtype=np.float64,
            )
            seed_index = np.rint(seed_zyx).astype(int)
            seed_index = np.clip(seed_index, 0, np.array(labels.shape) - 1)
            selected_label = int(labels[tuple(seed_index)])
            if selected_label == 0:
                best_label = 0
                best_distance = np.inf
                for label_id in range(1, count + 1):
                    coords = np.argwhere(labels == label_id)
                    if coords.size == 0:
                        continue
                    distance = float(np.linalg.norm(coords.mean(axis=0) - seed_zyx))
                    if distance < best_distance:
                        best_distance = distance
                        best_label = label_id
                selected_label = best_label
            selected = labels == selected_label
        # Do not overwrite cells already accepted earlier in the deterministic order.
        output[selected & (output == 0)] = group_id
    return output


def rotation_matrix_to_euler_zyx(rotation: np.ndarray) -> np.ndarray:
    """Return (theta_x, theta_y, theta_z) for R = Rz * Ry * Rx."""
    rotation = np.asarray(rotation, dtype=np.float64)
    sy = float(np.hypot(rotation[0, 0], rotation[1, 0]))
    singular = sy < 1e-8
    if not singular:
        theta_x = np.arctan2(rotation[2, 1], rotation[2, 2])
        theta_y = np.arctan2(-rotation[2, 0], sy)
        theta_z = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        theta_x = np.arctan2(-rotation[1, 2], rotation[1, 1])
        theta_y = np.arctan2(-rotation[2, 0], sy)
        theta_z = 0.0
    return np.array([theta_x, theta_y, theta_z], dtype=np.float64)


def ellipsoid_rotation_matrix(model: EllipsoidModel) -> np.ndarray:
    """Return the model's local-to-world rotation matrix."""
    tx, ty, tz = np.asarray(model.rotation_xyz, dtype=np.float64)
    cx, sx = np.cos(tx), np.sin(tx)
    cy, sy = np.cos(ty), np.sin(ty)
    cz, sz = np.cos(tz), np.sin(tz)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def fit_ellipsoid_from_mask(
    mask: np.ndarray,
    intensity: np.ndarray | None = None,
    weight_exponent: float = 1.5,
    name: str = "",
    cell_id: int | None = None,
    intensity_is_normalized: bool = False,
) -> EllipsoidModel:
    coords_zyx = np.argwhere(np.asarray(mask, dtype=bool))
    if coords_zyx.shape[0] < 8:
        raise ValueError(f"Not enough filled voxels to fit ellipsoid {name!r}")
    coords_xyz = coords_zyx[:, [2, 1, 0]].astype(np.float64)
    if intensity is None:
        weights = np.ones(coords_xyz.shape[0], dtype=np.float64)
        brightness = 0.5
    else:
        normalized_intensity = (
            np.asarray(intensity, dtype=np.float32)
            if intensity_is_normalized
            else robust_normalize(intensity)
        )
        values = normalized_intensity[tuple(coords_zyx.T)].astype(np.float64)
        weights = np.maximum(values, 1e-6) ** float(weight_exponent)
        brightness = float(np.mean(values))
    weight_sum = float(weights.sum())
    center = np.sum(coords_xyz * weights[:, None], axis=0) / weight_sum
    centered = coords_xyz - center
    covariance = (centered * weights[:, None]).T @ centered / weight_sum
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 1e-6)
    rotation = eigenvectors[:, order]
    if np.linalg.det(rotation) < 0:
        rotation[:, 2] *= -1.0
    radii = np.sqrt(5.0 * eigenvalues)
    angles = rotation_matrix_to_euler_zyx(rotation)
    return EllipsoidModel(
        center_xyz=center,
        radii_abc=radii,
        rotation_xyz=angles,
        brightness=brightness,
        cell_id=cell_id,
        name=name,
    )


def fit_cell_ellipsoids(
    cell_regions: np.ndarray,
    interpolated_raw: np.ndarray,
    *,
    intensity_is_normalized: bool = False,
    centers_xyz: dict[int, np.ndarray] | None = None,
) -> list[EllipsoidModel]:
    """Fit cell shape/intensity, optionally replacing centers with accepted XYZ."""
    models: list[EllipsoidModel] = []
    labels = np.asarray(cell_regions, dtype=np.int32)
    label_ids = sorted(int(v) for v in np.unique(labels) if v > 0)
    counts = np.bincount(labels.ravel())
    too_small = [
        (export_id, int(counts[label_id]))
        for export_id, label_id in enumerate(label_ids)
        if int(counts[label_id]) < 8
    ]
    if too_small:
        details = ", ".join(
            f"{export_id} ({voxel_count} voxels)"
            for export_id, voxel_count in too_small
        )
        raise ValueError(
            "Cannot fit ellipsoid for cell ID(s) "
            f"{details}; each cell needs at least 8 filled voxels"
        )
    supplied_centers: dict[int, np.ndarray] | None = None
    if centers_xyz is not None:
        supplied_centers = {
            int(label_id): np.asarray(center, dtype=np.float64)
            for label_id, center in centers_xyz.items()
        }
        missing = [
            label_id
            for label_id in label_ids
            if label_id not in supplied_centers
        ]
        if missing:
            missing_text = ", ".join(str(label_id) for label_id in missing)
            raise ValueError(
                f"Missing accepted center for cell label(s): {missing_text}"
            )
        for label_id in label_ids:
            center = supplied_centers[label_id]
            if center.shape != (3,) or not np.all(np.isfinite(center)):
                raise ValueError(
                    f"Center for cell label {label_id} must be a finite XYZ vector"
                )

    for export_id, group_id in enumerate(label_ids):
        model = fit_ellipsoid_from_mask(
            labels == group_id,
            interpolated_raw,
            name=str(export_id),
            cell_id=export_id,
            intensity_is_normalized=intensity_is_normalized,
        )
        if supplied_centers is not None:
            model.center_xyz = supplied_centers[group_id].copy()
        models.append(model)
    return models


@dataclass
class CellRefinementResult:
    """Result and audit metadata for one per-cell brightness refinement."""

    model: EllipsoidModel
    source_label_id: int
    brightness_offset: float
    threshold: float
    status: str
    reason: str
    used_baseline: bool
    component_count: int
    selected_voxel_count: int
    selection_method: str
    containment_scale: float
    roi_bounds_zyx: tuple[tuple[int, int], tuple[int, int], tuple[int, int]]


def _copy_ellipsoid_model(model: EllipsoidModel) -> EllipsoidModel:
    return EllipsoidModel(
        center_xyz=np.asarray(model.center_xyz).copy(),
        radii_abc=np.asarray(model.radii_abc).copy(),
        rotation_xyz=np.asarray(model.rotation_xyz).copy(),
        brightness=float(model.brightness),
        cell_id=model.cell_id,
        name=model.name,
    )


def _ellipsoid_roi_bounds(
    shape_zyx: tuple[int, int, int],
    model: EllipsoidModel,
    radial_limit: float = 1.0,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    center_xyz = np.asarray(model.center_xyz, dtype=np.float64)
    radii = np.asarray(model.radii_abc, dtype=np.float64)
    rotation = ellipsoid_rotation_matrix(model)
    limit = float(radial_limit)
    if not np.isfinite(limit) or limit < 0.0:
        raise ValueError("Ellipsoid radial limit must be finite and nonnegative")
    # The exact axis-aligned half-extent of a rotated ellipsoid along each
    # world axis is the row norm of R @ diag(radii).
    half_extent_xyz = (
        np.sqrt((rotation * rotation) @ (radii * radii)) * limit
    )
    lower_xyz = np.floor(center_xyz - half_extent_xyz).astype(np.int64) - 1
    upper_xyz = np.ceil(center_xyz + half_extent_xyz).astype(np.int64) + 2
    shape_xyz = np.asarray(shape_zyx[::-1], dtype=np.int64)
    lower_xyz = np.clip(lower_xyz, 0, shape_xyz)
    upper_xyz = np.clip(upper_xyz, 0, shape_xyz)
    return (
        (int(lower_xyz[2]), int(upper_xyz[2])),
        (int(lower_xyz[1]), int(upper_xyz[1])),
        (int(lower_xyz[0]), int(upper_xyz[0])),
    )


def _ellipsoid_radial_squared_in_roi(
    bounds_zyx: tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    model: EllipsoidModel,
) -> np.ndarray:
    (z0, z1), (y0, y1), (x0, x1) = bounds_zyx
    z, y, x = np.ogrid[z0:z1, y0:y1, x0:x1]
    center = np.asarray(model.center_xyz, dtype=np.float64)
    rotation = ellipsoid_rotation_matrix(model)
    dx = x.astype(np.float64) - center[0]
    dy = y.astype(np.float64) - center[1]
    dz = z.astype(np.float64) - center[2]
    radii = np.asarray(model.radii_abc, dtype=np.float64)
    radial_squared = np.zeros(
        (z1 - z0, y1 - y0, x1 - x0),
        dtype=np.float64,
    )
    for axis in range(3):
        local = (
            dx * rotation[0, axis]
            + dy * rotation[1, axis]
            + dz * rotation[2, axis]
        )
        radial_squared += (local / radii[axis]) ** 2
    return radial_squared


def _nest_ellipsoid_within_baseline(
    fitted: EllipsoidModel,
    baseline: EllipsoidModel,
) -> tuple[EllipsoidModel, float]:
    """Scale ``fitted`` inside ``baseline`` at the accepted baseline center."""
    candidate = _copy_ellipsoid_model(fitted)
    candidate.center_xyz = np.asarray(baseline.center_xyz, dtype=np.float64).copy()
    baseline_rotation = ellipsoid_rotation_matrix(baseline)
    candidate_rotation = ellipsoid_rotation_matrix(candidate)
    baseline_radii = np.asarray(baseline.radii_abc, dtype=np.float64)
    candidate_radii = np.asarray(candidate.radii_abc, dtype=np.float64)
    transform = (
        np.diag(1.0 / baseline_radii)
        @ baseline_rotation.T
        @ candidate_rotation
        @ np.diag(candidate_radii)
    )
    maximum_stretch = float(np.linalg.svd(transform, compute_uv=False)[0])
    if not np.isfinite(maximum_stretch) or maximum_stretch <= 0.0:
        raise ValueError("Unable to determine refined ellipsoid containment")
    # A small inward margin protects the mathematical containment guarantee
    # from float conversion in downstream mesh generation.
    containment_scale = min(1.0, (1.0 - 1e-7) / maximum_stretch)
    candidate.radii_abc = candidate_radii * containment_scale
    return candidate, float(containment_scale)


def refine_cell_ellipsoid(
    normalized_interpolated_intensity: np.ndarray,
    interpolated_labels: np.ndarray,
    source_label_id: int,
    baseline_model: EllipsoidModel,
    brightness_offset: float,
    *,
    min_component_voxels: int = 8,
) -> CellRefinementResult:
    """Refine one cell inside its baseline ellipsoid using a bright component.

    A zero offset is an explicit identity operation. Positive offsets threshold
    the already-normalized interpolated image at
    ``baseline_model.brightness + brightness_offset``. The center-containing
    26-connected component wins; otherwise the closest component that overlaps
    the cell's interpolated label is selected. Any invalid or undersized result
    returns an exact copy of the baseline model with fallback metadata.
    """
    intensity = np.asarray(normalized_interpolated_intensity, dtype=np.float32)
    labels = np.asarray(interpolated_labels)
    if intensity.ndim != 3 or labels.ndim != 3:
        raise ValueError("Cell refinement expects 3D intensity and label volumes")
    if intensity.shape != labels.shape:
        raise ValueError("Cell refinement intensity and labels must have matching shapes")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Cell refinement expects integer labels")
    source_label_id = int(source_label_id)
    if source_label_id <= 0:
        raise ValueError("Cell refinement source label ID must be positive")
    if not np.any(labels == source_label_id):
        raise ValueError(
            f"Cell refinement label {source_label_id} is absent from the label volume"
        )
    offset = float(brightness_offset)
    if not np.isfinite(offset) or offset < 0.0:
        raise ValueError("Cell refinement brightness offset must be finite and nonnegative")
    minimum_voxels = int(min_component_voxels)
    if minimum_voxels < 8:
        raise ValueError("Cell refinement requires at least 8 component voxels")

    baseline = _copy_ellipsoid_model(baseline_model)
    center = np.asarray(baseline.center_xyz, dtype=np.float64)
    radii = np.asarray(baseline.radii_abc, dtype=np.float64)
    angles = np.asarray(baseline.rotation_xyz, dtype=np.float64)
    if (
        center.shape != (3,)
        or radii.shape != (3,)
        or angles.shape != (3,)
        or not np.all(np.isfinite(np.concatenate([center, radii, angles])))
        or np.any(radii <= 0.0)
    ):
        raise ValueError("Cell refinement baseline ellipsoid is invalid")
    baseline_brightness = float(baseline.brightness)
    if (
        not np.isfinite(baseline_brightness)
        or baseline_brightness < 0.0
        or baseline_brightness > 1.0
    ):
        raise ValueError(
            "Cell refinement baseline brightness must be normalized to [0, 1]"
        )

    threshold = baseline_brightness + offset
    bounds = _ellipsoid_roi_bounds(intensity.shape, baseline)

    def fallback(
        reason: str,
        *,
        component_count: int = 0,
        selected_voxel_count: int = 0,
        selection_method: str = "none",
    ) -> CellRefinementResult:
        return CellRefinementResult(
            model=_copy_ellipsoid_model(baseline),
            source_label_id=source_label_id,
            brightness_offset=offset,
            threshold=threshold,
            status="baseline" if reason == "zero_offset" else "fallback",
            reason=reason,
            used_baseline=True,
            component_count=int(component_count),
            selected_voxel_count=int(selected_voxel_count),
            selection_method=selection_method,
            containment_scale=1.0,
            roi_bounds_zyx=bounds,
        )

    if offset == 0.0:
        return fallback("zero_offset")

    if any(stop <= start for start, stop in bounds):
        return fallback("baseline_outside_volume")
    slices = tuple(slice(start, stop) for start, stop in bounds)
    local_intensity = intensity[slices]
    finite_values = local_intensity[np.isfinite(local_intensity)]
    if finite_values.size:
        if float(finite_values.min()) < -1e-6 or float(finite_values.max()) > 1.0 + 1e-6:
            raise ValueError("Cell refinement intensity must be normalized to [0, 1]")
    radial_squared = _ellipsoid_radial_squared_in_roi(bounds, baseline)
    inside_baseline = radial_squared <= 1.0 + 1e-7
    candidate = (
        inside_baseline
        & np.isfinite(local_intensity)
        & (local_intensity >= threshold)
    )
    structure = np.ones((3, 3, 3), dtype=bool)
    components, component_count = ndi.label(candidate, structure=structure)
    if component_count == 0:
        return fallback("no_component_above_threshold")

    (z0, _), (y0, _), (x0, _) = bounds
    center_zyx = np.rint([center[2], center[1], center[0]]).astype(np.int64)
    center_local = center_zyx - np.array([z0, y0, x0], dtype=np.int64)
    selected_component = 0
    selection_method = "none"
    if np.all(center_local >= 0) and np.all(
        center_local < np.asarray(components.shape, dtype=np.int64)
    ):
        selected_component = int(components[tuple(center_local)])
        if selected_component > 0:
            selection_method = "center"

    if selected_component == 0:
        local_label_support = labels[slices] == source_label_id
        component_ids = np.arange(1, component_count + 1, dtype=np.int32)
        overlap_counts = np.asarray(
            ndi.sum(local_label_support, components, index=component_ids),
            dtype=np.float64,
        )
        eligible_ids = component_ids[overlap_counts > 0.0]
        if eligible_ids.size == 0:
            return fallback(
                "no_component_overlaps_source_label",
                component_count=component_count,
            )
        nearest_distances = np.asarray(
            ndi.minimum(radial_squared, components, index=eligible_ids),
            dtype=np.float64,
        )
        component_sizes = np.asarray(
            ndi.sum(candidate, components, index=eligible_ids),
            dtype=np.float64,
        )
        eligible_overlaps = overlap_counts[eligible_ids - 1]
        ordering = sorted(
            range(eligible_ids.size),
            key=lambda index: (
                float(nearest_distances[index]),
                -float(eligible_overlaps[index]),
                -float(component_sizes[index]),
                int(eligible_ids[index]),
            ),
        )
        selected_component = int(eligible_ids[ordering[0]])
        selection_method = "nearest_label_overlap"

    selected = components == selected_component
    selected_voxel_count = int(np.count_nonzero(selected))
    if selected_voxel_count < minimum_voxels:
        return fallback(
            "component_too_small",
            component_count=component_count,
            selected_voxel_count=selected_voxel_count,
            selection_method=selection_method,
        )
    selected_coords = np.argwhere(selected)
    if np.any(np.ptp(selected_coords, axis=0) < 1):
        return fallback(
            "component_not_three_dimensional",
            component_count=component_count,
            selected_voxel_count=selected_voxel_count,
            selection_method=selection_method,
        )

    try:
        fitted = fit_ellipsoid_from_mask(
            selected,
            local_intensity,
            name=baseline.name,
            cell_id=baseline.cell_id,
            intensity_is_normalized=True,
        )
        # The bright component supplies PCA radii and orientation, but the
        # manually reviewed/accepted baseline center remains the final XYZ.
        fitted.center_xyz = center.copy()
        fitted.name = baseline.name
        fitted.cell_id = baseline.cell_id
        fitted.brightness = float(np.mean(local_intensity[selected]))
        nested, containment_scale = _nest_ellipsoid_within_baseline(
            fitted,
            baseline,
        )
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return fallback(
            "ellipsoid_fit_failed",
            component_count=component_count,
            selected_voxel_count=selected_voxel_count,
            selection_method=selection_method,
        )

    return CellRefinementResult(
        model=nested,
        source_label_id=source_label_id,
        brightness_offset=offset,
        threshold=threshold,
        status="refined",
        reason="",
        used_baseline=False,
        component_count=int(component_count),
        selected_voxel_count=selected_voxel_count,
        selection_method=selection_method,
        containment_scale=containment_scale,
        roi_bounds_zyx=bounds,
    )


def ellipsoid_weight_volume(
    shape_zyx: tuple[int, int, int],
    model: EllipsoidModel,
    soft_margin: float,
) -> np.ndarray:
    shape = tuple(int(value) for value in shape_zyx)
    if len(shape) != 3 or any(value < 0 for value in shape):
        raise ValueError("Ellipsoid weight volume requires a nonnegative ZYX shape")
    _validate_ellipsoid_for_rasterization(model)
    margin = max(0.0, float(soft_margin))
    support_limit = 1.0 if margin <= 1e-8 else 1.0 + margin
    bounds = _ellipsoid_roi_bounds(shape, model, support_limit)
    output = np.zeros(shape, dtype=np.float32)
    if any(lower >= upper for lower, upper in bounds):
        return output
    radial = np.sqrt(_ellipsoid_radial_squared_in_roi(bounds, model))
    if margin <= 1e-8:
        weights = (radial <= 1.0).astype(np.float32)
    else:
        u = np.clip((1.0 + margin - radial) / (2.0 * margin), 0.0, 1.0)
        weights = (u * u * (3.0 - 2.0 * u)).astype(np.float32)
    slices = tuple(slice(lower, upper) for lower, upper in bounds)
    output[slices] = weights
    return output


def _validate_ellipsoid_for_rasterization(model: EllipsoidModel) -> None:
    center = np.asarray(model.center_xyz, dtype=np.float64)
    radii = np.asarray(model.radii_abc, dtype=np.float64)
    rotation = np.asarray(model.rotation_xyz, dtype=np.float64)
    if center.shape != (3,) or radii.shape != (3,) or rotation.shape != (3,):
        raise ValueError("Ellipsoid center, radii, and rotation must be XYZ triples")
    if not np.all(np.isfinite(np.concatenate([center, radii, rotation]))):
        raise ValueError("Ellipsoid geometry must contain only finite values")
    if np.any(radii <= 0.0):
        raise ValueError("Ellipsoid radii must be positive")


def rasterize_ellipsoid_labels(
    shape_zyx: tuple[int, int, int],
    models: Iterable[EllipsoidModel],
) -> np.ndarray:
    """Rasterize ordered ellipsoids into a compact integer ZYX label volume.

    Model ``i`` receives label ``i + 1``.  Overlaps use deterministic
    first-model-wins semantics so an earlier model is never overwritten by a
    later one.  Each ellipsoid is evaluated only inside its clipped
    axis-aligned bounding box; no full-size floating-point work volume is
    allocated per model.
    """
    shape = tuple(int(value) for value in shape_zyx)
    if len(shape) != 3 or any(value < 0 for value in shape):
        raise ValueError(
            "Ellipsoid label rasterization requires a nonnegative ZYX shape"
        )

    output = np.zeros(shape, dtype=np.int32)
    for index, model in enumerate(models, start=1):
        if index > np.iinfo(output.dtype).max:
            raise ValueError("Too many ellipsoids for an int32 label volume")
        _validate_ellipsoid_for_rasterization(model)
        bounds = _ellipsoid_roi_bounds(shape, model)
        if any(lower >= upper for lower, upper in bounds):
            continue
        slices = tuple(slice(lower, upper) for lower, upper in bounds)
        target = output[slices]
        inside = _ellipsoid_radial_squared_in_roi(bounds, model) <= 1.0
        target[inside & (target == 0)] = index
    return output


@dataclass(frozen=True)
class BackgroundRegionStats:
    """Audit statistics for one final background-level estimate."""

    value: float
    sample_count: int
    candidate_voxel_count: int
    nonfinite_voxel_count: int
    estimator: str
    trim_count_per_tail: int
    minimum: float
    median: float
    maximum: float

    def to_dict(self) -> dict[str, object]:
        return {
            "value": float(self.value),
            "sample_count": int(self.sample_count),
            "candidate_voxel_count": int(self.candidate_voxel_count),
            "nonfinite_voxel_count": int(self.nonfinite_voxel_count),
            "estimator": self.estimator,
            "trim_count_per_tail": int(self.trim_count_per_tail),
            "minimum": float(self.minimum),
            "median": float(self.median),
            "maximum": float(self.maximum),
        }


@dataclass(frozen=True)
class FinalBackgroundEstimate:
    """Final two-level background values in interpolated export space."""

    cold: float
    hot: float
    occupied_voxel_count: int
    transition_voxel_count: int
    cold_stats: BackgroundRegionStats
    hot_stats: BackgroundRegionStats
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "cold": float(self.cold),
            "hot": float(self.hot),
            "occupied_voxel_count": int(self.occupied_voxel_count),
            "transition_voxel_count": int(self.transition_voxel_count),
            "cold_stats": self.cold_stats.to_dict(),
            "hot_stats": self.hot_stats.to_dict(),
            "warnings": list(self.warnings),
        }


def _hard_ellipsoid_union(
    shape_zyx: tuple[int, int, int],
    models: Iterable[EllipsoidModel],
) -> np.ndarray:
    occupied = np.zeros(shape_zyx, dtype=bool)
    for model in models:
        _validate_ellipsoid_for_rasterization(model)
        bounds = _ellipsoid_roi_bounds(shape_zyx, model)
        if any(lower >= upper for lower, upper in bounds):
            continue
        radial_squared = _ellipsoid_radial_squared_in_roi(bounds, model)
        slices = tuple(slice(lower, upper) for lower, upper in bounds)
        occupied[slices] |= radial_squared <= 1.0
    return occupied


def _background_region_statistics(
    image: np.ndarray,
    sample_mask: np.ndarray,
    region_name: str,
) -> BackgroundRegionStats:
    candidates = np.asarray(image[sample_mask], dtype=np.float64)
    finite = candidates[np.isfinite(candidates)]
    candidate_count = int(candidates.size)
    sample_count = int(finite.size)
    nonfinite_count = candidate_count - sample_count
    if sample_count < 8:
        raise ValueError(
            f"{region_name} background has only {sample_count} finite samples; "
            "at least 8 are required"
        )

    ordered = np.sort(finite)
    median = float(np.median(ordered))
    trim_count = 0
    if sample_count >= 32:
        trim_count = int(np.floor(sample_count * 0.05))
        retained = ordered[trim_count : sample_count - trim_count]
        value = float(np.mean(retained, dtype=np.float64))
        estimator = "trimmed_mean_5_percent"
    else:
        value = median
        estimator = "median"
    return BackgroundRegionStats(
        value=value,
        sample_count=sample_count,
        candidate_voxel_count=candidate_count,
        nonfinite_voxel_count=nonfinite_count,
        estimator=estimator,
        trim_count_per_tail=trim_count,
        minimum=float(ordered[0]),
        median=median,
        maximum=float(ordered[-1]),
    )


def estimate_final_backgrounds(
    normalized_interpolated_intensity: np.ndarray,
    membrane_model: EllipsoidModel,
    cell_models: Iterable[EllipsoidModel],
    soft_margin: float,
) -> FinalBackgroundEstimate:
    """Estimate final cold/hot levels around the fitted membrane ellipsoid.

    The input and every model must use the interpolated export space: the image
    is indexed as ZYX, while ellipsoid centers and rotations use XYZ. The
    membrane geometry fixes spatial identity (inside is hot); intensities only
    determine the two scalar levels. Current refined cell ellipsoids are
    excluded with hard boundaries, and the membrane's soft transition band is
    excluded from both estimates.
    """

    image = np.asarray(normalized_interpolated_intensity)
    if image.ndim != 3 or any(length <= 0 for length in image.shape):
        raise ValueError("Final background estimation requires a nonempty 3D ZYX image")
    finite_values = image[np.isfinite(image)]
    if finite_values.size and (
        float(np.min(finite_values)) < 0.0
        or float(np.max(finite_values)) > 1.0
    ):
        raise ValueError(
            "Final background intensity must be normalized to the [0, 1] range"
        )
    margin = float(soft_margin)
    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError("Background soft margin must be finite and nonnegative")
    _validate_ellipsoid_for_rasterization(membrane_model)

    shape = tuple(int(length) for length in image.shape)
    weights = ellipsoid_weight_volume(shape, membrane_model, margin)
    occupied = _hard_ellipsoid_union(shape, cell_models)
    available = ~occupied
    hot_mask = (weights >= 0.95) & available
    cold_mask = (weights <= 0.05) & available
    transition_mask = (weights > 0.05) & (weights < 0.95)

    cold_stats = _background_region_statistics(image, cold_mask, "Cold")
    hot_stats = _background_region_statistics(image, hot_mask, "Hot")
    warnings: list[str] = []
    if cold_stats.nonfinite_voxel_count:
        warnings.append(
            "Cold background ignored "
            f"{cold_stats.nonfinite_voxel_count} non-finite sample(s)."
        )
    if hot_stats.nonfinite_voxel_count:
        warnings.append(
            "Hot background ignored "
            f"{hot_stats.nonfinite_voxel_count} non-finite sample(s)."
        )
    if hot_stats.value <= cold_stats.value:
        warnings.append(
            "Estimated hot background is not brighter than the cold background; "
            "spatial labels were preserved and values were not swapped."
        )
    return FinalBackgroundEstimate(
        cold=cold_stats.value,
        hot=hot_stats.value,
        occupied_voxel_count=int(np.count_nonzero(occupied)),
        transition_voxel_count=int(np.count_nonzero(transition_mask)),
        cold_stats=cold_stats,
        hot_stats=hot_stats,
        warnings=tuple(warnings),
    )


def ellipsoid_mesh(
    model: EllipsoidModel,
    n_latitude: int = 20,
    n_longitude: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    latitudes = np.linspace(0.0, np.pi, n_latitude)
    longitudes = np.linspace(0.0, 2.0 * np.pi, n_longitude, endpoint=False)
    vertices_xyz: list[np.ndarray] = []
    tx, ty, tz = model.rotation_xyz
    cx, sx = np.cos(tx), np.sin(tx)
    cy, sy = np.cos(ty), np.sin(ty)
    cz, sz = np.cos(tz), np.sin(tz)
    rotation = (
        np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    )
    for latitude in latitudes:
        for longitude in longitudes:
            unit = np.array(
                [
                    np.sin(latitude) * np.cos(longitude),
                    np.sin(latitude) * np.sin(longitude),
                    np.cos(latitude),
                ]
            )
            vertices_xyz.append(model.center_xyz + rotation @ (model.radii_abc * unit))
    faces: list[tuple[int, int, int]] = []
    for lat in range(n_latitude - 1):
        for lon in range(n_longitude):
            nxt = (lon + 1) % n_longitude
            a = lat * n_longitude + lon
            b = lat * n_longitude + nxt
            c = (lat + 1) * n_longitude + lon
            d = (lat + 1) * n_longitude + nxt
            faces.append((a, c, b))
            faces.append((b, c, d))
    vertices_xyz_array = np.asarray(vertices_xyz, dtype=np.float32)
    vertices_zyx = vertices_xyz_array[:, [2, 1, 0]]
    return vertices_zyx, np.asarray(faces, dtype=np.int32)


def combine_ellipsoid_meshes(
    models: Iterable[EllipsoidModel],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices_parts: list[np.ndarray] = []
    face_parts: list[np.ndarray] = []
    values_parts: list[np.ndarray] = []
    offset = 0
    for index, model in enumerate(models):
        vertices, faces = ellipsoid_mesh(model)
        vertices_parts.append(vertices)
        face_parts.append(faces + offset)
        values_parts.append(np.full(vertices.shape[0], index + 1, dtype=np.float32))
        offset += vertices.shape[0]
    if not vertices_parts:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.int32),
            np.zeros((0,), dtype=np.float32),
        )
    return (
        np.concatenate(vertices_parts),
        np.concatenate(face_parts),
        np.concatenate(values_parts),
    )
