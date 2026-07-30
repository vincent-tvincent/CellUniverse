from __future__ import annotations

import copy
import traceback
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
from napari.utils.colormaps import DirectLabelColormap
from qtpy import QtCore, QtWidgets
from scipy import ndimage as ndi

from . import algorithms as algorithm_api
from . import model as model_api
from .exporter import autosave_session, export_initial_csv
from .model import InitializerSession, MembraneParameters, Step
from .workers import ComputeWorker


# CELL_EDGES remains a model alias only so old autosaves can be read; it is not
# a separate screen.
CELL_REGIONS_STEP = Step.CELL_REGIONS
MEMBRANE_STEP = Step.MEMBRANE
Z_STEP = Step.Z_CALIBRATION
CELL_FITTING_STEP = Step.CELL_FITTING
REVIEW_STEP = Step.REVIEW
WORKFLOW_STEPS = (
    MEMBRANE_STEP,
    CELL_REGIONS_STEP,
    Z_STEP,
    CELL_FITTING_STEP,
    REVIEW_STEP,
)
PAGE_FOR_STEP = {step: index for index, step in enumerate(WORKFLOW_STEPS)}

STEP_TITLES = {
    MEMBRANE_STEP: "1/5 — Detect embryo membrane",
    CELL_REGIONS_STEP: "2/5 — Detect solid cell regions and review seeds",
    Z_STEP: "3/5 — Calibrate and interpolate Z",
    CELL_FITTING_STEP: "4/5 — Refine each cell ellipsoid",
    REVIEW_STEP: "5/5 — Review the simulated environment and export",
}

STEP_INSTRUCTIONS = {
    MEMBRANE_STEP: (
        "Move the upper clamp until one membrane region is detected. The clamp "
        "is preview-only and never changes the source image."
    ),
    CELL_REGIONS_STEP: (
        "Tune the colored fills to cover the cells. Yellow points are seed "
        "observations; one numbered marker identifies each 3D cell. Reject "
        "false seeds or segments. Split a joined pair with magenta separator "
        "lines, or mark any touching cells that belong together and merge the "
        "whole marked set. The last split or merge can be undone."
    ),
    Z_STEP: (
        "Adjust the Z interpolation ratio until the geometry is correct. Every "
        "update interpolates the image, the ID-preserving cell boundaries, and "
        "recalculates robust centers from the interpolated 3D chunks. The "
        "normal preview is the 3D volume; turn "
        "it off only if this Mac needs the safe 2D renderer. No ellipsoid is "
        "fitted in this step."
    ),
    CELL_FITTING_STEP: (
        "Each slider raises that cell's brightness threshold inside its original "
        "ellipsoid and refits a smaller, reoriented ellipsoid. Click a cell in "
        "the smooth 3D view or 2D fallback to jump to its slider, then "
        "compare the translucent fit with the interpolated image. After every "
        "fit, all current cell ellipsoids are excluded from the final dim/bright "
        "background estimate."
    ),
    REVIEW_STEP: (
        "Inspect the complete simulated environment. Return to an earlier step "
        "if anything is incorrect; otherwise export initial.csv."
    ),
}

LAYER_NAMES = {
    "raw": "00 Raw source (immutable)",
    "clamped": "10 Clamped source preview",
    "membrane_dog": "11 Diagnostic — membrane DoG",
    "membrane_log": "12 Diagnostic — membrane LoG",
    "membrane_fused": "13 Diagnostic — membrane fused response",
    "membrane_labels": "14 Detected membrane region",
    "background": "15 Diagnostic — two-region background model",
    "cell_dog": "20 Diagnostic — cell DoG",
    "cell_log": "21 Diagnostic — cell LoG",
    "cell_fused": "22 Diagnostic — cell fused response",
    "cell_edges": "23 Diagnostic — thresholded cell edges",
    "cell_regions_raw": "24 Solid cell regions",
    "cell_centers_raw": "25 Accepted watershed seed observations",
    "rejected_cell_centers_raw": "26 Rejected watershed seed observations",
    "merge_selection": "27A Cells marked for merge",
    "manual_split_guides": "27B Manual per-slice split guides",
    "centerline_outliers_raw": (
        "28 Diagnostic — automatically excluded slice centroids"
    ),
    "source_cell_ids": "29 Step 2 cell IDs (one per 3D cell)",
    "interpolated_raw": "30 Interpolated source",
    "interpolated_regions": "31 Diagnostic — interpolated cell labels",
    "interpolated_edges": "32 Interpolated cell boundaries",
    "interpolated_centers": "33 Interpolated cell centers",
    "interpolated_centerline_outliers": (
        "34 Diagnostic — interpolated excluded slice centroids"
    ),
    "background_cross_sections": "38 Background ellipsoid outline (2D-safe)",
    "cell_cross_sections": "39 Cell ellipsoid cross-sections (2D-safe)",
    "background_surface": "41 Background envelope ellipsoid (3D)",
    "cell_surface": "42 Cell ellipsoids (3D foreground)",
    "cell_ids": "43 Simulated cell IDs",
}

MERGE_SELECTION_COLORMAP = DirectLabelColormap(
    color_dict={
        0: np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        1: np.array([0.0, 1.0, 1.0, 1.0], dtype=np.float32),
        None: np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    }
)

BASE_LAYER_BY_STEP = {
    MEMBRANE_STEP: "clamped",
    CELL_REGIONS_STEP: "raw",
    Z_STEP: "interpolated_raw",
    CELL_FITTING_STEP: "interpolated_raw",
    REVIEW_STEP: "interpolated_raw",
}

RESULT_LAYERS_BY_STEP = {
    MEMBRANE_STEP: {"membrane_labels"},
    CELL_REGIONS_STEP: {
        "cell_regions_raw",
        "cell_centers_raw",
        "rejected_cell_centers_raw",
        "merge_selection",
        "manual_split_guides",
        "source_cell_ids",
    },
    Z_STEP: {"interpolated_edges", "interpolated_centers"},
    CELL_FITTING_STEP: set(),
    REVIEW_STEP: set(),
}

MODEL_LAYERS_2D = {
    "background_cross_sections",
    "cell_cross_sections",
    "cell_ids",
}
MODEL_LAYERS_3D = {"background_surface", "cell_surface", "cell_ids"}

DIAGNOSTIC_LAYERS_BY_STEP = {
    MEMBRANE_STEP: {
        "membrane_dog",
        "membrane_log",
        "membrane_fused",
        "background",
    },
    CELL_REGIONS_STEP: {
        "background",
        "cell_dog",
        "cell_log",
        "cell_fused",
        "cell_edges",
        "centerline_outliers_raw",
    },
    Z_STEP: {
        "interpolated_regions",
        "interpolated_centerline_outliers",
    },
    CELL_FITTING_STEP: {
        "interpolated_regions",
        "interpolated_edges",
        "interpolated_centers",
        "interpolated_centerline_outliers",
    },
    REVIEW_STEP: {
        "interpolated_regions",
        "interpolated_edges",
        "interpolated_centers",
        "interpolated_centerline_outliers",
    },
}

COMPUTE_LABELS = {
    "membrane": "membrane-region preview",
    "cell_regions": "solid cell regions, seeds, and robust chunk centers",
    "interpolation": "interpolated image, boundaries, and robust centers",
    "cell_fitting": "per-cell brightness-offset ellipsoids",
    "manual_merge": "merged cells and robust 3D center",
}

@dataclass
class InterpolationPreview:
    ratio: float
    interpolated_raw: np.ndarray
    interpolated_labels: np.ndarray
    interpolated_membrane: np.ndarray
    interpolated_edges: np.ndarray
    scaled_centers_xyz: dict[int, np.ndarray]
    center_diagnostics: dict[int, dict[str, Any]]


@dataclass
class CellFittingPreview:
    ratio: float
    normalized_raw: np.ndarray
    membrane_model: Any
    baseline_cell_models: list[Any]
    cell_models: list[Any]
    offsets: dict[int, float]
    refinement_status: dict[int, str]
    background_vertices: np.ndarray
    background_faces: np.ndarray
    background_values: np.ndarray
    cell_vertices: np.ndarray
    cell_faces: np.ndarray
    cell_values: np.ndarray
    background_cross_sections: np.ndarray
    cell_cross_sections: np.ndarray
    cold_background: float
    hot_background: float
    background_statistics: dict[str, Any]


@dataclass
class RejectedSliceCenter:
    center_uid: int
    point_zyx: np.ndarray
    source_label_id: int
    removed_chunk_mask: np.ndarray | None = None
    removed_chunk_bounds_zyx: tuple[slice, slice, slice] | None = None


@dataclass
class RejectedSegment:
    source_label_id: int
    bounds_zyx: tuple[slice, slice, slice]
    local_mask: np.ndarray
    all_center_uids: tuple[int, ...]
    accepted_center_uids: tuple[int, ...]


@dataclass
class LabelEditSnapshot:
    operation: str
    labels: np.ndarray
    result_labels: np.ndarray | None
    center_points_zyx: np.ndarray
    center_ids: np.ndarray
    center_accepted: np.ndarray
    reviewed_slices: np.ndarray | None
    rejected_slice_centers: dict[int, RejectedSliceCenter]
    rejected_segments: list[RejectedSegment]


class FloatControl(QtWidgets.QWidget):
    valueChanged = QtCore.Signal(float)

    def __init__(
        self,
        label: str,
        minimum: float,
        maximum: float,
        value: float,
        step: float,
        decimals: int = 3,
    ) -> None:
        super().__init__()
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self._resolution = 2000
        self.label = QtWidgets.QLabel(label)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, self._resolution)
        self.spin = QtWidgets.QDoubleSpinBox()
        self.spin.setRange(self.minimum, self.maximum)
        self.spin.setSingleStep(step)
        self.spin.setDecimals(decimals)
        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label, 0, 0, 1, 2)
        layout.addWidget(self.slider, 1, 0)
        layout.addWidget(self.spin, 1, 1)
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)
        self.set_value(value, emit=False)

    def _slider_value(self, value: float) -> int:
        fraction = (float(value) - self.minimum) / max(
            self.maximum - self.minimum, 1e-12
        )
        return int(round(np.clip(fraction, 0.0, 1.0) * self._resolution))

    def _float_value(self, value: int) -> float:
        fraction = float(value) / self._resolution
        return self.minimum + fraction * (self.maximum - self.minimum)

    def _from_slider(self, value: int) -> None:
        float_value = self._float_value(value)
        self.spin.blockSignals(True)
        self.spin.setValue(float_value)
        self.spin.blockSignals(False)
        self.valueChanged.emit(float(self.spin.value()))

    def _from_spin(self, value: float) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(self._slider_value(value))
        self.slider.blockSignals(False)
        self.valueChanged.emit(float(value))

    def value(self) -> float:
        return float(self.spin.value())

    def set_value(self, value: float, emit: bool = True) -> None:
        self.spin.blockSignals(True)
        self.slider.blockSignals(True)
        self.spin.setValue(float(value))
        self.slider.setValue(self._slider_value(value))
        self.spin.blockSignals(False)
        self.slider.blockSignals(False)
        if emit:
            self.valueChanged.emit(self.value())


class IntControl(QtWidgets.QWidget):
    valueChanged = QtCore.Signal(int)

    def __init__(self, label: str, minimum: int, maximum: int, value: int) -> None:
        super().__init__()
        self.label = QtWidgets.QLabel(label)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(minimum, maximum)
        self.spin = QtWidgets.QSpinBox()
        self.spin.setRange(minimum, maximum)
        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label, 0, 0, 1, 2)
        layout.addWidget(self.slider, 1, 0)
        layout.addWidget(self.spin, 1, 1)
        self.slider.valueChanged.connect(self.spin.setValue)
        self.spin.valueChanged.connect(self.slider.setValue)
        self.spin.valueChanged.connect(self.valueChanged.emit)
        self.set_value(value, emit=False)

    def value(self) -> int:
        return int(self.spin.value())

    def set_value(self, value: int, emit: bool = True) -> None:
        self.spin.blockSignals(True)
        self.slider.blockSignals(True)
        self.spin.setValue(int(value))
        self.slider.setValue(int(value))
        self.spin.blockSignals(False)
        self.slider.blockSignals(False)
        if emit:
            self.valueChanged.emit(self.value())


class InitializerWizard(QtWidgets.QWidget):
    def __init__(self, viewer: Any, session: InitializerSession) -> None:
        super().__init__()
        self.viewer = viewer
        self.session = session
        self.thread_pool = QtCore.QThreadPool(self)
        self.thread_pool.setMaxThreadCount(1)
        self.generations: dict[str, int] = {}
        self._workers: set[ComputeWorker] = set()
        self._busy_kind: str | None = None
        self._busy_completion_revision = 0
        self._latest_membrane_result: Any | None = None
        self._latest_cell_region_result: Any | None = None
        self._latest_interpolation_ratio: float | None = None
        self._latest_fitting_ratio: float | None = None
        self._normalized_interpolated_raw: np.ndarray | None = None
        self.cell_fit_controls: dict[int, FloatControl] = {}
        self._selected_cell_fit_label_id: int | None = None
        self._cell_surface_label_ids: tuple[int, ...] = ()
        self._cell_pick_callback = self._on_viewer_cell_clicked
        # The source frame is immutable for the lifetime of the wizard, so the
        # expensive normalization and filter responses can safely be reused
        # while the user tunes Step 2 controls.
        self._prepared_cell_detection = algorithm_api.prepare_cell_detection(
            self.session.raw
        )
        self._rejected_slice_centers: dict[int, RejectedSliceCenter] = {}
        self._rejected_segments: list[RejectedSegment] = []
        self._label_edit_snapshot: LabelEditSnapshot | None = None
        self._marked_merge_label_ids: set[int] = set()
        self._result_current = {
            "membrane": False,
            "cell_regions": False,
            "interpolation": False,
            "cell_fitting": False,
        }

        self._membrane_timer = self._new_timer(self._submit_membrane_preview, 300)
        self._cell_timer = self._new_timer(self._submit_cell_region_preview, 300)
        self._interpolation_timer = self._new_timer(
            self._submit_interpolation_preview, 450
        )
        self._cell_fitting_timer = self._new_timer(
            self._submit_cell_fitting_preview, 250
        )
        self._busy_clock_timer = QtCore.QTimer(self)
        self._busy_clock_timer.setInterval(250)
        self._busy_clock_timer.timeout.connect(self._update_busy_elapsed)
        self._busy_elapsed = QtCore.QElapsedTimer()
        # Kept as inert aliases for callers that clean up timers created by the
        # six-step prototype. They do not represent visible workflow stages.
        self._grouping_timer = self._new_timer(lambda: None, 300)
        self._segmentation_timer = self._new_timer(lambda: None, 350)

        self._build_ui()
        self._add_initial_layers()
        self._connect_step_events()
        mouse_callbacks = getattr(self.viewer, "mouse_drag_callbacks", None)
        if mouse_callbacks is not None:
            mouse_callbacks.append(self._cell_pick_callback)
        initial = self.session.current_step
        if initial not in WORKFLOW_STEPS:
            initial = MEMBRANE_STEP
        self._set_step(initial, entering=True)

    def _new_timer(self, callback: Any, delay_ms: int) -> QtCore.QTimer:
        timer = QtCore.QTimer(self)
        timer.setSingleShot(True)
        timer.setInterval(delay_ms)
        timer.timeout.connect(callback)
        return timer

    def _build_ui(self) -> None:
        root = QtWidgets.QVBoxLayout(self)
        self.title_label = QtWidgets.QLabel()
        title_font = self.title_label.font()
        title_font.setBold(True)
        title_font.setPointSize(title_font.pointSize() + 2)
        self.title_label.setFont(title_font)
        self.instruction_label = QtWidgets.QLabel()
        self.instruction_label.setWordWrap(True)
        self.diagnostics_toggle = QtWidgets.QCheckBox(
            "Diagnostics (show DoG / LoG / fused and intermediate layers)"
        )
        self.diagnostics_toggle.setChecked(False)
        self.global_status = QtWidgets.QLabel()
        self.global_status.setWordWrap(True)
        root.addWidget(self.title_label)
        root.addWidget(self.instruction_label)
        root.addWidget(self.diagnostics_toggle)

        self.busy_widget = QtWidgets.QWidget()
        busy_layout = QtWidgets.QVBoxLayout(self.busy_widget)
        busy_layout.setContentsMargins(0, 4, 0, 4)
        self.busy_label = QtWidgets.QLabel()
        self.busy_progress = QtWidgets.QProgressBar()
        self.busy_progress.setRange(0, 0)
        self.busy_progress.setTextVisible(False)
        self.busy_progress.setFixedHeight(8)
        busy_layout.addWidget(self.busy_label)
        busy_layout.addWidget(self.busy_progress)
        self.busy_widget.hide()
        root.addWidget(self.global_status)

        self.pages = QtWidgets.QStackedWidget()
        self.pages.addWidget(self._build_membrane_page())
        self.pages.addWidget(self._build_cell_regions_page())
        self.pages.addWidget(self._build_z_page())
        self.pages.addWidget(self._build_cell_fitting_page())
        self.pages.addWidget(self._build_review_page())
        root.addWidget(self.pages, 1)
        # Keep recalculation feedback below the scrollable slider pages. Showing
        # or hiding it then changes only the space beneath the controls instead
        # of pushing the entire control panel down.
        root.addWidget(self.busy_widget)

        nav = QtWidgets.QHBoxLayout()
        self.back_button = QtWidgets.QPushButton("Back")
        self.save_button = QtWidgets.QPushButton("Save session")
        self.next_button = QtWidgets.QPushButton("Confirm and continue")
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        nav.addWidget(self.back_button)
        nav.addWidget(self.save_button)
        nav.addStretch(1)
        nav.addWidget(self.next_button)
        nav.addWidget(self.cancel_button)
        root.addLayout(nav)
        self.back_button.clicked.connect(self._go_back)
        self.next_button.clicked.connect(self._confirm_current_step)
        self.save_button.clicked.connect(self._save_session)
        self.cancel_button.clicked.connect(self._cancel)
        self.diagnostics_toggle.toggled.connect(self._on_diagnostics_toggled)
        self.setMinimumWidth(410)

    def _page(self) -> tuple[QtWidgets.QWidget, QtWidgets.QVBoxLayout]:
        content = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(content)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        page = QtWidgets.QScrollArea()
        page.setWidgetResizable(True)
        page.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        page.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        page.setWidget(content)
        return page, layout

    def _build_membrane_page(self) -> QtWidgets.QWidget:
        page, layout = self._page()
        finite = self.session.raw[np.isfinite(self.session.raw)]
        raw_min = float(np.min(finite))
        raw_max = float(np.percentile(finite, 99.99))
        initial_clamp = float(np.percentile(finite, 99.0))
        self.membrane_clamp = FloatControl(
            "Upper clamp (source intensity)",
            raw_min,
            max(raw_min + 1.0, raw_max),
            initial_clamp,
            max((raw_max - raw_min) / 200.0, 1.0),
            decimals=1,
        )
        self.membrane_dog_low = FloatControl(
            "DoG small sigma", 0.2, 6.0, 1.0, 0.1
        )
        self.membrane_dog_high = FloatControl(
            "DoG large sigma", 0.4, 12.0, 3.0, 0.1
        )
        self.membrane_log_sigma = FloatControl(
            "LoG sigma", 0.2, 10.0, 2.0, 0.1
        )
        self.membrane_fusion = FloatControl(
            "DoG contribution (0=LoG, 1=DoG)", 0.0, 1.0, 0.5, 0.05
        )
        self.membrane_threshold = FloatControl(
            "Fused edge threshold", 0.01, 1.0, 0.30, 0.01
        )
        self.membrane_min_size = IntControl(
            "Minimum 3D component voxels", 10, 100000, 500
        )
        self.membrane_closing = IntControl("Closing radius", 0, 8, 2)
        self.membrane_soft_margin = FloatControl(
            "Soft envelope margin", 0.0, 0.30, 0.05, 0.01
        )
        self.membrane_status = QtWidgets.QLabel("Waiting for membrane analysis…")
        self.membrane_status.setWordWrap(True)
        for control in (
            self.membrane_clamp,
            self.membrane_dog_low,
            self.membrane_dog_high,
            self.membrane_log_sigma,
            self.membrane_fusion,
            self.membrane_threshold,
            self.membrane_min_size,
            self.membrane_closing,
            self.membrane_soft_margin,
        ):
            layout.addWidget(control)
        layout.addWidget(self.membrane_status)
        return page

    def _build_cell_regions_page(self) -> QtWidgets.QWidget:
        page, layout = self._page()
        explanation = QtWidgets.QLabel(
            "Colored fills are cell regions; Diagnostics shows edges and excluded "
            "center outliers. Lower the fused edge threshold for weaker "
            "boundaries, or raise it for stronger ones. Yellow points are "
            "per-slice seeds, not final centers: select false ones and click "
            "Reject. The cell mask stays unless a seed belongs to a detached "
            "chunk. Use the colored-segment picker to remove a false cell or "
            "to mark or unmark every cell that belongs in one merged chunk, "
            "then merge the complete marked set with one button. "
            "Draw magenta lines across a joined pair to define a 3D left/right "
            "split."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        self.cell_dog_low = FloatControl("DoG small sigma", 0.2, 5.0, 0.8, 0.1)
        self.cell_dog_high = FloatControl("DoG large sigma", 0.3, 8.0, 2.0, 0.1)
        self.cell_log_sigma = FloatControl("LoG sigma", 0.2, 6.0, 1.2, 0.1)
        self.cell_fusion = FloatControl(
            "DoG contribution (0=LoG, 1=DoG)", 0.0, 1.0, 0.5, 0.05
        )
        self.cell_foreground_quantile = FloatControl(
            "Primary cell-body intensity percentile", 90.0, 99.99, 99.0, 0.05, 2
        )
        self.cell_center_quantile = FloatControl(
            "Center / seed intensity percentile",
            99.50,
            99.95,
            99.70,
            0.01,
            2,
        )
        self.cell_center_core_volume = IntControl(
            "Discard center cores smaller than (raw voxels)", 20, 5000, 300
        )
        self.cell_threshold = FloatControl(
            "Fused edge threshold (soft boundary guidance)",
            0.01,
            1.0,
            0.70,
            0.01,
        )
        self.cell_boundary_min_size = IntControl(
            "Discard diagnostic edge fragments smaller than (pixels)", 0, 1000, 8
        )
        self.cell_min_size = IntControl(
            "Discard enclosed 3D chunks smaller than (raw voxels)",
            1,
            5000,
            25,
        )
        self.cell_closing = IntControl("Cell-body closing radius", 0, 8, 1)
        self.cell_center_distance = IntControl(
            "Fallback peak separation when no bright center exists (pixels)",
            1,
            50,
            5,
        )
        self.cell_center_radius = FloatControl(
            "Fallback minimum peak radius (pixels)", 0.0, 20.0, 1.5, 0.1
        )
        for control in (
            self.cell_dog_low,
            self.cell_dog_high,
            self.cell_log_sigma,
            self.cell_fusion,
            self.cell_foreground_quantile,
            self.cell_center_quantile,
            self.cell_center_core_volume,
            self.cell_threshold,
            self.cell_boundary_min_size,
            self.cell_min_size,
            self.cell_closing,
            self.cell_center_distance,
            self.cell_center_radius,
        ):
            layout.addWidget(control)

        self.renumber_regions = QtWidgets.QPushButton(
            "Recompute robust chunk centers and renumber edited regions"
        )
        layout.addWidget(self.renumber_regions)
        center_correction_buttons = QtWidgets.QGridLayout()
        self.reject_selected_centers = QtWidgets.QPushButton(
            "Mark selected yellow center(s) wrong"
        )
        self.restore_selected_centers = QtWidgets.QPushButton(
            "Restore selected red center(s)"
        )
        center_correction_buttons.addWidget(self.reject_selected_centers, 0, 0)
        center_correction_buttons.addWidget(self.restore_selected_centers, 0, 1)
        layout.addLayout(center_correction_buttons)

        segment_correction_buttons = QtWidgets.QGridLayout()
        self.pick_wrong_segment = QtWidgets.QPushButton(
            "Pick a colored cell in the viewer"
        )
        self.reject_picked_segment = QtWidgets.QPushButton(
            "Mark picked segment wrong"
        )
        self.restore_last_segment = QtWidgets.QPushButton(
            "Restore most recently marked segment"
        )
        self.restore_last_segment.setEnabled(False)
        segment_correction_buttons.addWidget(self.pick_wrong_segment, 0, 0)
        segment_correction_buttons.addWidget(self.reject_picked_segment, 0, 1)
        segment_correction_buttons.addWidget(
            self.restore_last_segment,
            1,
            0,
            1,
            2,
        )
        layout.addLayout(segment_correction_buttons)

        manual_merge_buttons = QtWidgets.QGridLayout()
        self.toggle_picked_merge_cell = QtWidgets.QPushButton(
            "Mark/unmark picked cell for merge"
        )
        self.merge_marked_cells = QtWidgets.QPushButton(
            "Merge all marked cells"
        )
        self.merge_marked_cells.setEnabled(False)
        self.clear_marked_merge_cells = QtWidgets.QPushButton(
            "Clear marked cells"
        )
        self.clear_marked_merge_cells.setEnabled(False)
        manual_merge_buttons.addWidget(self.toggle_picked_merge_cell, 0, 0)
        manual_merge_buttons.addWidget(self.merge_marked_cells, 0, 1)
        manual_merge_buttons.addWidget(
            self.clear_marked_merge_cells,
            1,
            0,
            1,
            2,
        )
        layout.addLayout(manual_merge_buttons)
        self.manual_merge_status = QtWidgets.QLabel(
            "No cells are marked for merge."
        )
        self.manual_merge_status.setWordWrap(True)
        layout.addWidget(self.manual_merge_status)

        self.manual_split_target_share = FloatControl(
            "Optional line-target guard (0 = off)",
            0.00,
            1.00,
            0.00,
            0.01,
            2,
        )
        layout.addWidget(self.manual_split_target_share)
        manual_split_buttons = QtWidgets.QGridLayout()
        self.draw_manual_split = QtWidgets.QPushButton(
            "Draw separator line on current Z"
        )
        self.apply_manual_splits = QtWidgets.QPushButton(
            "Apply all separator lines"
        )
        self.clear_manual_splits = QtWidgets.QPushButton(
            "Clear separator lines"
        )
        self.undo_label_edit = QtWidgets.QPushButton(
            "Undo last split or merge"
        )
        self.undo_label_edit.setEnabled(False)
        # Compatibility alias for callers and tests written before manual
        # merge shared the one-level label-edit undo.
        self.undo_manual_split = self.undo_label_edit
        manual_split_buttons.addWidget(self.draw_manual_split, 0, 0)
        manual_split_buttons.addWidget(self.apply_manual_splits, 0, 1)
        manual_split_buttons.addWidget(self.clear_manual_splits, 1, 0)
        manual_split_buttons.addWidget(self.undo_label_edit, 1, 1)
        layout.addLayout(manual_split_buttons)

        review_buttons = QtWidgets.QGridLayout()
        self.previous_unreviewed = QtWidgets.QPushButton("Previous unreviewed")
        self.next_unreviewed = QtWidgets.QPushButton("Next unreviewed")
        self.accept_slice = QtWidgets.QPushButton("Accept current slice")
        self.empty_slice = QtWidgets.QPushButton("Mark current slice empty")
        self.accept_all = QtWidgets.QPushButton("Accept all remaining")
        review_buttons.addWidget(self.previous_unreviewed, 0, 0)
        review_buttons.addWidget(self.next_unreviewed, 0, 1)
        review_buttons.addWidget(self.accept_slice, 1, 0)
        review_buttons.addWidget(self.empty_slice, 1, 1)
        review_buttons.addWidget(self.accept_all, 2, 0, 1, 2)
        layout.addLayout(review_buttons)
        self.cell_review_status = QtWidgets.QLabel()
        self.cell_review_status.setWordWrap(True)
        layout.addWidget(self.cell_review_status)
        self.previous_unreviewed.clicked.connect(lambda: self._jump_unreviewed(-1))
        self.next_unreviewed.clicked.connect(lambda: self._jump_unreviewed(1))
        self.accept_slice.clicked.connect(self._accept_current_slice)
        self.empty_slice.clicked.connect(self._mark_current_slice_empty)
        self.accept_all.clicked.connect(self._accept_all_slices)
        self.renumber_regions.clicked.connect(self._renumber_edited_regions)
        self.reject_selected_centers.clicked.connect(self._reject_selected_centers)
        self.restore_selected_centers.clicked.connect(self._restore_selected_centers)
        self.pick_wrong_segment.clicked.connect(self._activate_segment_picker)
        self.reject_picked_segment.clicked.connect(self._reject_picked_segment)
        self.restore_last_segment.clicked.connect(self._restore_last_segment)
        self.toggle_picked_merge_cell.clicked.connect(
            self._toggle_picked_merge_cell
        )
        self.merge_marked_cells.clicked.connect(
            self._merge_marked_cells
        )
        self.clear_marked_merge_cells.clicked.connect(
            self._clear_marked_merge_cells
        )
        self.draw_manual_split.clicked.connect(
            self._activate_manual_split_guides
        )
        self.apply_manual_splits.clicked.connect(
            self._apply_manual_split_guides
        )
        self.clear_manual_splits.clicked.connect(
            self._clear_manual_split_guides
        )
        self.undo_label_edit.clicked.connect(self._undo_last_label_edit)
        return page

    def _build_z_page(self) -> QtWidgets.QWidget:
        page, layout = self._page()
        self.z_ratio = FloatControl(
            "Z interpolation ratio",
            1.0,
            12.0,
            self.session.z_ratio,
            0.1,
            decimals=2,
        )
        self.z_reset_four = QtWidgets.QPushButton(
            "Set ratio to dataset metadata (4.0)"
        )
        self.z_refresh = QtWidgets.QPushButton("Rebuild interpolated preview")
        self.z_view_3d = QtWidgets.QCheckBox("Show 3D volume view")
        self.z_view_3d.setChecked(True)
        self.z_rendering_note = QtWidgets.QLabel(
            "The 3D volume is the normal view. If the canvas becomes blank or "
            "reports a GL error on this Mac, uncheck this box to use the safe "
            "2D slice fallback."
        )
        self.z_rendering_note.setWordWrap(True)
        self.z_status = QtWidgets.QLabel()
        self.z_status.setWordWrap(True)
        layout.addWidget(self.z_ratio)
        layout.addWidget(self.z_reset_four)
        layout.addWidget(self.z_refresh)
        layout.addWidget(self.z_view_3d)
        layout.addWidget(self.z_rendering_note)
        layout.addWidget(self.z_status)
        self.z_reset_four.clicked.connect(lambda: self.z_ratio.set_value(4.0))
        self.z_refresh.clicked.connect(self._submit_interpolation_preview)
        self.z_view_3d.toggled.connect(
            partial(self._on_display_mode_toggled, Z_STEP)
        )
        return page

    def _build_cell_fitting_page(self) -> QtWidgets.QWidget:
        page, layout = self._page()
        self.cell_fitting_page = page
        explanation = QtWidgets.QLabel(
            "The zero-offset shape is the original fitted ellipsoid. Raising a "
            "cell's offset searches for a brighter connected component only "
            "inside that original ellipsoid, then fits a contained, smaller "
            "ellipsoid around it. The robust chunk-derived center stays fixed "
            "while the "
            "radii and orientation are refitted. Click a fitted cross-section "
            "in the 2D fallback or an ellipsoid in normal 3D to jump to that "
            "cell's slider."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        self.cell_fitting_controls_widget = QtWidgets.QWidget()
        self.cell_fitting_controls_layout = QtWidgets.QVBoxLayout(
            self.cell_fitting_controls_widget
        )
        self.cell_fitting_controls_layout.setContentsMargins(0, 0, 0, 0)
        self.cell_fitting_controls_layout.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignTop
        )
        layout.addWidget(self.cell_fitting_controls_widget)

        self.reset_cell_offsets = QtWidgets.QPushButton(
            "Reset every cell brightness offset to zero"
        )
        self.cell_fitting_view_3d = QtWidgets.QCheckBox(
            "Show 3D volume and smooth fitted ellipsoids"
        )
        self.cell_fitting_view_3d.setChecked(True)
        self.cell_fitting_rendering_note = QtWidgets.QLabel(
            "The 3D volume with smooth fitted surfaces is the normal view. If "
            "the canvas becomes blank or reports a GL error on this Mac, "
            "uncheck this box to use clickable 2D cross-sections."
        )
        self.cell_fitting_rendering_note.setWordWrap(True)
        self.cell_fitting_status = QtWidgets.QLabel(
            "Confirm Z interpolation before fitting cells."
        )
        self.cell_fitting_status.setWordWrap(True)
        layout.addWidget(self.reset_cell_offsets)
        layout.addWidget(self.cell_fitting_view_3d)
        layout.addWidget(self.cell_fitting_rendering_note)
        layout.addWidget(self.cell_fitting_status)
        self.reset_cell_offsets.clicked.connect(self._reset_all_cell_offsets)
        self.cell_fitting_view_3d.toggled.connect(
            partial(self._on_display_mode_toggled, CELL_FITTING_STEP)
        )
        return page

    def _build_review_page(self) -> QtWidgets.QWidget:
        page, layout = self._page()
        self.review_summary = QtWidgets.QLabel()
        self.review_summary.setWordWrap(True)
        self.review_3d = QtWidgets.QCheckBox(
            "Show 3D volume and smooth environment"
        )
        self.review_3d.setChecked(True)
        self.review_rendering_note = QtWidgets.QLabel(
            "The complete smooth 3D environment is the normal review. If the "
            "canvas becomes blank or reports a GL error on this Mac, uncheck "
            "this box to use the safe 2D cross-section fallback."
        )
        self.review_rendering_note.setWordWrap(True)
        rollback_form = QtWidgets.QHBoxLayout()
        self.rollback_step = QtWidgets.QComboBox()
        self.rollback_step.addItems(
            [
                "Membrane",
                "Cell regions and seed review",
                "Z calibration",
                "Cell ellipsoid fitting",
            ]
        )
        self.rollback_button = QtWidgets.QPushButton("Return to selected step")
        rollback_form.addWidget(self.rollback_step)
        rollback_form.addWidget(self.rollback_button)
        layout.addWidget(self.review_summary)
        layout.addWidget(self.review_3d)
        layout.addWidget(self.review_rendering_note)
        layout.addLayout(rollback_form)
        self.review_3d.toggled.connect(
            partial(self._on_display_mode_toggled, REVIEW_STEP)
        )
        self.rollback_button.clicked.connect(self._rollback_from_review)
        return page

    def _connect_step_events(self) -> None:
        for control in (
            self.membrane_dog_low,
            self.membrane_dog_high,
            self.membrane_log_sigma,
            self.membrane_fusion,
            self.membrane_threshold,
            self.membrane_min_size,
            self.membrane_closing,
        ):
            control.valueChanged.connect(lambda _: self._schedule_membrane_preview())
        self.membrane_clamp.valueChanged.connect(self._on_membrane_clamp_changed)
        self.membrane_soft_margin.valueChanged.connect(
            lambda _: self._update_background_preview()
        )

        for control in (
            self.cell_dog_low,
            self.cell_dog_high,
            self.cell_log_sigma,
            self.cell_fusion,
            self.cell_foreground_quantile,
            self.cell_center_quantile,
            self.cell_center_core_volume,
            self.cell_threshold,
            self.cell_boundary_min_size,
            self.cell_min_size,
            self.cell_closing,
            self.cell_center_distance,
            self.cell_center_radius,
        ):
            control.valueChanged.connect(
                lambda _: self._schedule_cell_region_preview()
            )
        self.z_ratio.valueChanged.connect(self._on_z_ratio_changed)

    def _add_initial_layers(self) -> None:
        finite = self.session.raw[np.isfinite(self.session.raw)]
        contrast_low = float(np.percentile(finite, 0.5))
        contrast_high = float(np.percentile(finite, 99.8))
        if contrast_high <= contrast_low:
            contrast_high = contrast_low + 1.0
        self._raw_contrast_limits = (contrast_low, contrast_high)
        self._clamp_contrast_limits = (
            self._raw_contrast_limits[0],
            float(self.membrane_clamp.maximum),
        )
        self.viewer.add_image(
            self.session.raw,
            name=LAYER_NAMES["raw"],
            colormap="gray",
            contrast_limits=self._raw_contrast_limits,
        )
        initial_clamped = algorithm_api.clamp_preview(
            self.session.raw, self.membrane_clamp.value()
        )
        self._set_image(
            "clamped",
            initial_clamped,
            colormap="gray",
            blending="opaque",
            contrast_limits=self._clamp_contrast_limits,
        )
        self._ensure_manual_split_layer()
        self._set_2d_view()

    def _layer(self, key: str) -> Any | None:
        try:
            return self.viewer.layers[LAYER_NAMES[key]]
        except (KeyError, ValueError):
            return None

    def _set_image(
        self,
        key: str,
        data: np.ndarray,
        *,
        colormap: str = "gray",
        opacity: float = 1.0,
        blending: str = "translucent",
        contrast_limits: tuple[float, float] | None = None,
    ) -> Any:
        layer = self._layer(key)
        if layer is None:
            kwargs: dict[str, Any] = {}
            if contrast_limits is not None:
                kwargs["contrast_limits"] = contrast_limits
            layer = self.viewer.add_image(
                data,
                name=LAYER_NAMES[key],
                colormap=colormap,
                opacity=opacity,
                blending=blending,
                **kwargs,
            )
        else:
            layer.data = data
            layer.opacity = opacity
            if contrast_limits is not None:
                layer.contrast_limits = contrast_limits
        return layer

    def _set_labels(
        self,
        key: str,
        data: np.ndarray,
        *,
        opacity: float = 0.55,
        editable: bool = False,
    ) -> Any:
        labels = np.asarray(data, dtype=np.int32)
        layer = self._layer(key)
        if layer is None:
            layer = self.viewer.add_labels(
                labels,
                name=LAYER_NAMES[key],
                opacity=opacity,
            )
        else:
            layer.data = labels
            layer.opacity = opacity
        layer.editable = bool(editable)
        return layer

    def _set_merge_selection_overlay(self, data: np.ndarray) -> Any:
        selection = np.asarray(data, dtype=np.uint8)
        layer = self._layer("merge_selection")
        if layer is None:
            layer = self.viewer.add_labels(
                selection,
                name=LAYER_NAMES["merge_selection"],
                opacity=0.48,
                colormap=MERGE_SELECTION_COLORMAP,
            )
        else:
            layer.data = selection
            layer.opacity = 0.48
            layer.colormap = MERGE_SELECTION_COLORMAP
        layer.editable = False
        return layer

    def _ensure_manual_split_layer(self) -> Any:
        layer = self._layer("manual_split_guides")
        if layer is None:
            layer = self.viewer.add_shapes(
                data=[],
                ndim=3,
                shape_type="line",
                name=LAYER_NAMES["manual_split_guides"],
                edge_color="magenta",
                edge_width=3,
            )
        return layer

    def _set_points(
        self,
        key: str,
        points_zyx: np.ndarray,
        ids: list[int] | np.ndarray,
        *,
        extra_features: dict[str, np.ndarray] | None = None,
        face_color: str = "yellow",
        border_color: str = "black",
        symbol: str = "o",
        show_text: bool = True,
    ) -> Any:
        points = np.asarray(points_zyx, dtype=np.float32).reshape((-1, 3))
        features = {"cell_id": np.asarray(ids, dtype=int)}
        if extra_features:
            features.update(
                {
                    name: np.asarray(values)
                    for name, values in extra_features.items()
                }
            )
        if any(len(values) != len(points) for values in features.values()):
            raise ValueError("Point feature rows must match the number of points")
        is_cell_id_layer = key in {"source_cell_ids", "cell_ids"}
        text = (
            {
                "string": "{cell_id}",
                "size": 18 if is_cell_id_layer else 12,
                "color": "yellow" if is_cell_id_layer else "white",
                "anchor": "center" if is_cell_id_layer else "upper_left",
                # Final IDs lie at the centers of translucent cell ellipsoids.
                # Disable depth testing for both the marker and its text so the
                # surface cannot hide the number even when the camera rotates.
                "blending": (
                    "translucent_no_depth"
                    if is_cell_id_layer
                    else "translucent"
                ),
            }
            if show_text
            else None
        )
        layer = self._layer(key)
        visible = True if layer is None else bool(layer.visible)
        # Napari 0.8 refreshes text while ``data`` and ``features`` are assigned
        # separately. If the point count changes, Vispy can briefly index the
        # new feature table with the old view indices. Recreating this small
        # overlay makes the replacement atomic and avoids that mismatch.
        if layer is not None:
            self.viewer.layers.remove(layer)
        layer = self.viewer.add_points(
            points,
            name=LAYER_NAMES[key],
            size=16 if is_cell_id_layer else 8,
            face_color="black" if is_cell_id_layer else face_color,
            border_color="white" if is_cell_id_layer else border_color,
            border_width=0.15 if is_cell_id_layer else 0.05,
            symbol=symbol,
            features=features,
            text=text,
            blending=(
                "translucent_no_depth"
                if is_cell_id_layer
                else "translucent"
            ),
            canvas_size_limits=(
                (12, 40) if is_cell_id_layer else (0, 10_000)
            ),
            visible=visible,
        )
        layer.text.visible = bool(show_text)
        if key in {"cell_centers_raw", "rejected_cell_centers_raw"}:
            layer.mode = "select"
        return layer

    def _set_surface(
        self,
        key: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        values: np.ndarray,
        *,
        opacity: float,
        colormap: str,
        blending: str,
    ) -> Any:
        data = (vertices, faces, values)
        layer = self._layer(key)
        if layer is None:
            layer = self.viewer.add_surface(
                data,
                name=LAYER_NAMES[key],
                opacity=opacity,
                colormap=colormap,
                blending=blending,
                shading="smooth",
            )
        else:
            layer.data = data
            layer.opacity = opacity
            layer.colormap = colormap
            layer.blending = blending
            layer.shading = "smooth"
        return layer

    def _set_step(self, step: Step, entering: bool = False) -> None:
        if step not in PAGE_FOR_STEP:
            step = MEMBRANE_STEP
        if step != CELL_REGIONS_STEP:
            self._clear_marked_merge_cells()
        self.session.current_step = step
        self.pages.setCurrentIndex(PAGE_FOR_STEP[step])
        self.title_label.setText(STEP_TITLES[step])
        self.instruction_label.setText(STEP_INSTRUCTIONS[step])
        self.back_button.setEnabled(step != MEMBRANE_STEP)
        self.next_button.setText(
            "Export initial.csv and exit"
            if step == REVIEW_STEP
            else "Confirm and continue"
        )
        self.next_button.setEnabled(
            self._busy_kind is None and self._current_result_is_ready()
        )
        self.global_status.setText("")
        self._apply_display_mode_for_step(step)
        if entering:
            self._enter_step(step)

    def _enter_step(self, step: Step) -> None:
        if step == MEMBRANE_STEP:
            self._schedule_membrane_preview(20)
        elif step == CELL_REGIONS_STEP:
            if self.session.reviewed_slices is None:
                self.session.reviewed_slices = np.zeros(
                    self.session.raw.shape[0], dtype=bool
                )
            source = self._source_region_labels()
            if source is not None and np.any(source):
                self._result_current["cell_regions"] = True
                self._show_source_regions(source)
                self._update_review_status()
                self._apply_layer_visibility()
            else:
                self._update_review_status()
                self._schedule_cell_region_preview(20)
        elif step == Z_STEP:
            self.z_ratio.set_value(self.session.z_ratio, emit=False)
            if (
                self._result_current["interpolation"]
                and self.session.interpolated_raw is not None
                and self._latest_interpolation_ratio is not None
                and abs(
                    self._latest_interpolation_ratio - self.session.z_ratio
                )
                <= 1e-6
            ):
                self._show_interpolation_from_session()
                self._apply_layer_visibility()
            else:
                self._on_z_ratio_changed(self.session.z_ratio)
        elif step == CELL_FITTING_STEP:
            if (
                self._result_current["cell_fitting"]
                and self.session.baseline_cell_ellipsoids
                and self.session.cell_ellipsoids
                and self.session.membrane_ellipsoid is not None
            ):
                self._rebuild_cell_fit_controls(
                    self.session.baseline_cell_ellipsoids
                )
                self._apply_layer_visibility()
            else:
                self._submit_cell_fitting_preview()
        elif step == REVIEW_STEP:
            self._prepare_review()
        self.next_button.setEnabled(
            self._busy_kind is None and self._current_result_is_ready()
        )

    def _apply_layer_visibility(self, *_: Any) -> None:
        step = self.session.current_step
        base = BASE_LAYER_BY_STEP.get(step)
        result_kind = {
            MEMBRANE_STEP: "membrane",
            CELL_REGIONS_STEP: "cell_regions",
            Z_STEP: "interpolation",
            CELL_FITTING_STEP: "cell_fitting",
            REVIEW_STEP: "cell_fitting",
        }.get(step)
        results = (
            RESULT_LAYERS_BY_STEP.get(step, set())
            if result_kind is not None and self._result_current[result_kind]
            else set()
        )
        if (
            step in {CELL_FITTING_STEP, REVIEW_STEP}
            and result_kind is not None
            and self._result_current[result_kind]
        ):
            results = results | (
                MODEL_LAYERS_3D
                if self._display_3d_requested(step)
                else MODEL_LAYERS_2D
            )
        diagnostics = (
            DIAGNOSTIC_LAYERS_BY_STEP.get(step, set())
            if (
                self.diagnostics_toggle.isChecked()
                and result_kind is not None
                and self._result_current[result_kind]
            )
            else set()
        )
        allowed = ({base} if base is not None else set()) | results | diagnostics
        for key in LAYER_NAMES:
            layer = self._layer(key)
            if layer is not None:
                visible = key in allowed
                if (
                    key == "rejected_cell_centers_raw"
                    and not self._rejected_slice_centers
                ):
                    visible = False
                if (
                    key == "merge_selection"
                    and not self._marked_merge_label_ids
                ):
                    visible = False
                if (
                    key == "manual_split_guides"
                    and len(layer.data) == 0
                ):
                    visible = False
                if (
                    key
                    in {
                        "centerline_outliers_raw",
                        "interpolated_centerline_outliers",
                    }
                    and len(layer.data) == 0
                ):
                    visible = False
                layer.visible = visible

    def _on_diagnostics_toggled(self, checked: bool) -> None:
        if checked:
            if (
                self.session.current_step == MEMBRANE_STEP
                and self._latest_membrane_result is not None
            ):
                result = self._latest_membrane_result
                self._set_image(
                    "membrane_dog",
                    result.dog,
                    colormap="cyan",
                    opacity=0.75,
                )
                self._set_image(
                    "membrane_log",
                    result.log,
                    colormap="magenta",
                    opacity=0.75,
                )
                self._set_image(
                    "membrane_fused",
                    result.fused,
                    colormap="yellow",
                    opacity=0.8,
                )
            elif (
                self.session.current_step == CELL_REGIONS_STEP
                and self._latest_cell_region_result is not None
            ):
                result = self._latest_cell_region_result
                self._set_image(
                    "cell_dog",
                    result.dog,
                    colormap="cyan",
                    opacity=0.75,
                )
                self._set_image(
                    "cell_log",
                    result.log,
                    colormap="magenta",
                    opacity=0.75,
                )
                self._set_image(
                    "cell_fused",
                    result.fused,
                    colormap="yellow",
                    opacity=0.8,
                )
                self._set_labels(
                    "cell_edges",
                    result.edges.astype(np.int32),
                    opacity=0.55,
                )
            if (
                self.session.current_step in {MEMBRANE_STEP, CELL_REGIONS_STEP}
                and self.session.membrane_ellipsoid_raw is not None
            ):
                self._update_background_preview()
        self._apply_layer_visibility()

    def _current_result_is_ready(self) -> bool:
        result_kind = {
            MEMBRANE_STEP: "membrane",
            CELL_REGIONS_STEP: "cell_regions",
            Z_STEP: "interpolation",
            CELL_FITTING_STEP: "cell_fitting",
            REVIEW_STEP: "cell_fitting",
        }.get(self.session.current_step)
        if result_kind is None or not self._result_current[result_kind]:
            return False
        if self.session.current_step == Z_STEP:
            return (
                self.session.interpolated_raw is not None
                and self.session.interpolated_cell_labels is not None
                and self.session.interpolated_membrane_mask is not None
                and self.session.interpolated_cell_edges is not None
                and self._latest_interpolation_ratio is not None
                and abs(
                    self._latest_interpolation_ratio - self.z_ratio.value()
                )
                <= 1e-6
            )
        if self.session.current_step in {CELL_FITTING_STEP, REVIEW_STEP}:
            labels = self.session.interpolated_cell_labels
            expected_cells = (
                0
                if labels is None
                else len([value for value in np.unique(labels) if value > 0])
            )
            return (
                self.session.interpolated_raw is not None
                and self.session.membrane_ellipsoid is not None
                and expected_cells > 0
                and len(self.session.baseline_cell_ellipsoids) == expected_cells
                and len(self.session.cell_ellipsoids) == expected_cells
                and self._latest_fitting_ratio is not None
                and abs(self._latest_fitting_ratio - self.session.z_ratio) <= 1e-6
                and self.session.background_estimation_stage
                == model_api.FINAL_BACKGROUND_ESTIMATION_STAGE
            )
        return True

    def _cancel_pending_computations(self) -> None:
        """Make scheduled or running results stale before changing stages."""
        for timer in (
            self._membrane_timer,
            self._cell_timer,
            self._interpolation_timer,
            self._cell_fitting_timer,
            self._grouping_timer,
            self._segmentation_timer,
        ):
            timer.stop()
        for kind in COMPUTE_LABELS:
            self.generations[kind] = self.generations.get(kind, 0) + 1
        self._busy_kind = None
        self._busy_completion_revision += 1
        self._busy_clock_timer.stop()
        self.busy_progress.hide()
        self.busy_widget.hide()

    def _current_z(self) -> int:
        current = tuple(self.viewer.dims.current_step)
        if not current:
            return 0
        return int(np.clip(round(current[0]), 0, self.session.raw.shape[0] - 1))

    def _set_current_z(self, z: int) -> None:
        current = list(self.viewer.dims.current_step)
        if not current:
            return
        current[0] = int(np.clip(z, 0, self.session.raw.shape[0] - 1))
        self.viewer.dims.current_step = tuple(current)

    def _set_2d_view(self) -> None:
        self.viewer.dims.ndisplay = 2

    def _set_3d_view(self, enabled: bool) -> None:
        self.viewer.dims.ndisplay = 3 if enabled else 2

    def _display_toggle_for_step(
        self, step: Step
    ) -> QtWidgets.QCheckBox | None:
        return {
            Z_STEP: self.z_view_3d,
            CELL_FITTING_STEP: self.cell_fitting_view_3d,
            REVIEW_STEP: self.review_3d,
        }.get(step)

    def _display_3d_requested(self, step: Step) -> bool:
        toggle = self._display_toggle_for_step(step)
        return bool(toggle is not None and toggle.isChecked())

    def _apply_display_mode_for_step(self, step: Step) -> None:
        use_3d = self._display_3d_requested(step)
        if use_3d:
            # Hide the 2D-only model volumes before switching them into a 3D
            # rendering path. The raw volume is then activated only because the
            # user explicitly opted into 3D on this page.
            self._apply_layer_visibility()
            self._set_3d_view(True)
        else:
            # Recover the reliable slice renderer before showing any layer for
            # the newly selected page. This avoids briefly drawing the new raw
            # volume through the failing 3D OpenGL path.
            self._set_2d_view()
            self._apply_layer_visibility()

    def _on_display_mode_toggled(
        self, owner_step: Step, checked: bool
    ) -> None:
        if self.session.current_step != owner_step:
            return
        if checked:
            self._apply_layer_visibility()
            self._set_3d_view(True)
        else:
            self._set_2d_view()
            self._apply_layer_visibility()

    def _submit(self, kind: str, function: Any, *args: Any) -> None:
        self._set_recalculating(kind)
        generation = self.generations.get(kind, 0) + 1
        self.generations[kind] = generation
        worker = ComputeWorker(kind, generation, function, *args)
        worker.signals.returned.connect(self._worker_returned)
        worker.signals.failed.connect(self._worker_failed)
        worker.signals.returned.connect(lambda *_: self._workers.discard(worker))
        worker.signals.failed.connect(lambda *_: self._workers.discard(worker))
        self._workers.add(worker)
        self.thread_pool.start(worker)

    @QtCore.Slot(str, int, object)
    def _worker_returned(self, kind: str, generation: int, result: Any) -> None:
        if generation != self.generations.get(kind):
            return
        expected_step = {
            "membrane": MEMBRANE_STEP,
            "cell_regions": CELL_REGIONS_STEP,
            "interpolation": Z_STEP,
            "cell_fitting": CELL_FITTING_STEP,
        }.get(kind)
        if expected_step is not None and self.session.current_step != expected_step:
            return
        try:
            if kind == "membrane":
                self._apply_membrane_result(result)
            elif kind == "cell_regions":
                self._apply_cell_region_result(result)
            elif kind == "interpolation":
                self._apply_interpolation_result(result)
            elif kind == "cell_fitting":
                self._apply_cell_fitting_result(result)
        except Exception:
            result_kind = {
                "membrane": "membrane",
                "cell_regions": "cell_regions",
                "interpolation": "interpolation",
                "cell_fitting": "cell_fitting",
            }.get(kind)
            if result_kind is not None:
                self._result_current[result_kind] = False
            self.global_status.setText(
                f"{kind} result could not be displayed. See the terminal traceback."
            )
            self._set_recalculation_failed(kind)
            self._apply_layer_visibility()
            traceback.print_exc()
            return
        self._set_recalculation_finished(kind)

    @QtCore.Slot(str, int, str)
    def _worker_failed(self, kind: str, generation: int, traceback_text: str) -> None:
        if generation != self.generations.get(kind):
            return
        expected_step = {
            "membrane": MEMBRANE_STEP,
            "cell_regions": CELL_REGIONS_STEP,
            "interpolation": Z_STEP,
            "cell_fitting": CELL_FITTING_STEP,
        }.get(kind)
        if expected_step is not None and self.session.current_step != expected_step:
            return
        result_kind = {
            "membrane": "membrane",
            "cell_regions": "cell_regions",
            "interpolation": "interpolation",
            "cell_fitting": "cell_fitting",
        }.get(kind)
        if result_kind is not None:
            self._result_current[result_kind] = False
        self.global_status.setText(
            f"{kind} computation failed. See terminal for details."
        )
        self._set_recalculation_failed(kind)
        self._apply_layer_visibility()
        print(traceback_text)

    def _set_recalculating(self, kind: str) -> None:
        self._busy_kind = kind
        self._busy_completion_revision += 1
        self._busy_elapsed.restart()
        self._busy_clock_timer.start()
        self._update_busy_elapsed()
        self.busy_progress.show()
        self.busy_widget.show()
        self.next_button.setEnabled(False)

    def _update_busy_elapsed(self) -> None:
        if self._busy_kind is None or not self._busy_elapsed.isValid():
            return
        label = COMPUTE_LABELS.get(
            self._busy_kind,
            self._busy_kind.replace("_", " "),
        )
        seconds = self._busy_elapsed.elapsed() / 1000.0
        self.busy_label.setText(f"Recalculating {label}… {seconds:.1f} s")

    def _set_recalculation_finished(self, kind: str) -> None:
        if self._busy_kind != kind:
            return
        self._busy_kind = None
        self._busy_completion_revision += 1
        self._busy_clock_timer.stop()
        revision = self._busy_completion_revision
        label = COMPUTE_LABELS.get(kind, kind.replace("_", " "))
        self.busy_label.setText(f"Result updated: {label}.")
        self.busy_progress.hide()
        self.next_button.setEnabled(self._current_result_is_ready())

        def hide_if_still_finished() -> None:
            if self._busy_kind is None and self._busy_completion_revision == revision:
                self.busy_widget.hide()

        QtCore.QTimer.singleShot(1200, hide_if_still_finished)

    def _set_recalculation_failed(self, kind: str) -> None:
        if self._busy_kind != kind:
            return
        self._busy_kind = None
        self._busy_completion_revision += 1
        self._busy_clock_timer.stop()
        label = COMPUTE_LABELS.get(kind, kind.replace("_", " "))
        self.busy_label.setText(f"Recalculation failed: {label}.")
        self.busy_progress.hide()
        self.busy_widget.show()
        self.next_button.setEnabled(False)

    def _schedule_compute(
        self,
        kind: str,
        timer: QtCore.QTimer,
        delay_ms: int | None = None,
    ) -> None:
        self.generations[kind] = self.generations.get(kind, 0) + 1
        self._set_recalculating(kind)
        if delay_ms is None:
            timer.start()
        else:
            timer.start(delay_ms)

    def _membrane_parameters(self) -> MembraneParameters:
        return MembraneParameters(
            upper_clamp=self.membrane_clamp.value(),
            dog_sigma_low=self.membrane_dog_low.value(),
            dog_sigma_high=self.membrane_dog_high.value(),
            log_sigma=self.membrane_log_sigma.value(),
            fusion_weight=self.membrane_fusion.value(),
            edge_threshold=self.membrane_threshold.value(),
            min_component_size=self.membrane_min_size.value(),
            closing_radius=self.membrane_closing.value(),
            soft_margin=self.membrane_soft_margin.value(),
        )

    def _hide_layers(self, *keys: str) -> None:
        for key in keys:
            layer = self._layer(key)
            if layer is not None:
                layer.visible = False

    def _schedule_membrane_preview(self, delay_ms: int | None = None) -> None:
        self._result_current["membrane"] = False
        self._hide_layers(
            "membrane_dog",
            "membrane_log",
            "membrane_fused",
            "membrane_labels",
            "background",
        )
        self.membrane_status.setText(
            "Parameters changed; waiting to recompute the membrane region…"
        )
        self._schedule_compute("membrane", self._membrane_timer, delay_ms)

    @QtCore.Slot(float)
    def _on_membrane_clamp_changed(self, value: float) -> None:
        preview = algorithm_api.clamp_preview(self.session.raw, float(value))
        layer = self._set_image(
            "clamped",
            preview,
            colormap="gray",
            blending="opaque",
            contrast_limits=self._clamp_contrast_limits,
        )
        layer.visible = self.session.current_step == MEMBRANE_STEP
        self._schedule_membrane_preview()

    def _submit_membrane_preview(self) -> None:
        self.membrane_status.setText("Computing membrane-region preview…")
        params = self._membrane_parameters()
        self.session.membrane_params = params
        self._submit(
            "membrane",
            algorithm_api.detect_membrane,
            self.session.raw,
            params,
        )

    def _apply_membrane_result(self, result: Any) -> None:
        self._latest_membrane_result = result
        self._result_current["membrane"] = True
        self._set_image(
            "clamped",
            result.clamped,
            colormap="gray",
            blending="opaque",
            contrast_limits=self._clamp_contrast_limits,
        )
        if self.diagnostics_toggle.isChecked():
            self._on_diagnostics_toggled(True)
        self._set_labels("membrane_labels", result.labels, opacity=0.45)
        if result.component_count == 1:
            self.membrane_status.setText(
                "Exactly one solid membrane region detected. Inspect it, then confirm."
            )
        elif result.component_count == 0:
            self.membrane_status.setText(
                "No membrane region detected. Adjust the clamp or edge threshold."
            )
        else:
            self.membrane_status.setText(
                f"{result.component_count} membrane regions detected; confirmation "
                "requires exactly one."
            )
        self._apply_layer_visibility()

    def _update_background_preview(self) -> None:
        if (
            not self.diagnostics_toggle.isChecked()
            or self.session.membrane_ellipsoid_raw is None
        ):
            return
        model = self.session.membrane_ellipsoid_raw
        weights = algorithm_api.ellipsoid_weight_volume(
            self.session.raw.shape,
            model,
            self.membrane_soft_margin.value(),
        )
        background = (
            self.session.cold_background * (1.0 - weights)
            + self.session.hot_background * weights
        )
        self._set_image(
            "background",
            background.astype(np.float32),
            colormap="magma",
            opacity=0.55,
        )
        self._apply_layer_visibility()

    def _cell_region_parameters(self) -> model_api.CellRegionParameters:
        return model_api.CellRegionParameters(
            dog_sigma_low=self.cell_dog_low.value(),
            dog_sigma_high=self.cell_dog_high.value(),
            log_sigma=self.cell_log_sigma.value(),
            fusion_weight=self.cell_fusion.value(),
            foreground_quantile=self.cell_foreground_quantile.value(),
            center_quantile=self.cell_center_quantile.value(),
            edge_threshold=self.cell_threshold.value(),
            min_component_size=self.cell_boundary_min_size.value(),
            closing_radius=self.cell_closing.value(),
            min_chunk_area_2d=self.cell_min_size.value(),
            min_center_core_volume=self.cell_center_core_volume.value(),
            center_min_distance=self.cell_center_distance.value(),
            center_min_radius=self.cell_center_radius.value(),
        )

    def _schedule_cell_region_preview(self, delay_ms: int | None = None) -> None:
        self._discard_label_edit_undo()
        self._result_current["cell_regions"] = False
        self._hide_layers(
            "cell_dog",
            "cell_log",
            "cell_fused",
            "cell_edges",
            "cell_regions_raw",
            "cell_centers_raw",
            "rejected_cell_centers_raw",
            "merge_selection",
            "centerline_outliers_raw",
            "source_cell_ids",
        )
        if self.session.reviewed_slices is not None:
            self.session.reviewed_slices[:] = False
        self.cell_review_status.setText(
            "Parameters changed; waiting to recompute solid regions and seeds…"
        )
        self._schedule_compute("cell_regions", self._cell_timer, delay_ms)

    def _submit_cell_region_preview(self) -> None:
        if self.session.membrane_mask is None:
            self.cell_review_status.setText(
                "Cell-region calculation cannot start without a confirmed membrane."
            )
            self._set_recalculation_failed("cell_regions")
            return
        params = self._cell_region_parameters()
        self.session.cell_edge_params = params
        self.cell_review_status.setText(
            "Computing enclosed cell regions, watershed seed observations, "
            "and robust chunk centers…"
        )
        self._submit(
            "cell_regions",
            algorithm_api.detect_cell_regions,
            self.session.raw,
            self.session.membrane_mask,
            params,
            None,
            self._prepared_cell_detection,
        )

    def _apply_cell_region_result(self, result: algorithm_api.EdgeResult) -> None:
        self._latest_cell_region_result = result
        self._result_current["cell_regions"] = True
        self._rejected_slice_centers.clear()
        self._rejected_segments.clear()
        self._discard_label_edit_undo()
        self.restore_last_segment.setEnabled(False)
        guide_layer = self._ensure_manual_split_layer()
        guide_layer.data = []
        guide_layer.mode = "select"
        self.session.invalidate_after(CELL_REGIONS_STEP)
        labels = np.asarray(result.labels, dtype=np.int32)
        if (
            result.slice_centers_zyx.shape[0]
            == result.slice_center_ids.shape[0]
            > 0
        ):
            points = result.slice_centers_zyx
            ids = result.slice_center_ids
        else:
            points, ids = algorithm_api.slice_centers_from_labels(labels)
        self._initialize_source_regions(
            labels,
            points,
            ids,
            precomputed_centers_xyz=(
                result.centers_xyz
                if result.center_diagnostics
                else None
            ),
            precomputed_center_diagnostics=(
                result.center_diagnostics or None
            ),
        )
        if self.session.reviewed_slices is None:
            self.session.reviewed_slices = np.zeros(labels.shape[0], dtype=bool)
        else:
            self.session.reviewed_slices[:] = False
        if self.diagnostics_toggle.isChecked():
            self._on_diagnostics_toggled(True)
        self._show_source_regions(labels)
        self._update_review_status()
        self._apply_layer_visibility()

    def _source_region_labels(self) -> np.ndarray | None:
        for name in ("cell_labels_raw", "group_labels_raw", "accepted_edges"):
            value = getattr(self.session, name, None)
            if value is not None:
                array = np.asarray(value)
                if array.ndim == 3 and np.issubdtype(array.dtype, np.integer):
                    return array.astype(np.int32, copy=False)
        return None

    def _center_candidate_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        points = np.asarray(
            self.session.cell_slice_centers_zyx,
            dtype=np.float32,
        ).reshape((-1, 3))
        ids = np.asarray(
            self.session.cell_slice_center_ids,
            dtype=np.int32,
        ).reshape((-1,))
        accepted = np.asarray(
            self.session.cell_slice_center_accepted,
            dtype=bool,
        ).reshape((-1,))
        if not (len(points) == len(ids) == len(accepted)):
            raise ValueError(
                "Center points, label IDs, and acceptance flags must have "
                "matching row counts"
            )
        return points, ids, accepted

    def _initialize_source_regions(
        self,
        labels: np.ndarray,
        slice_points_zyx: np.ndarray,
        slice_center_ids: np.ndarray,
        *,
        precomputed_centers_xyz: dict[int, np.ndarray] | None = None,
        precomputed_center_diagnostics: (
            dict[int, dict[str, Any]] | None
        ) = None,
    ) -> None:
        points = np.asarray(
            slice_points_zyx, dtype=np.float32
        ).reshape((-1, 3))
        ids = np.asarray(
            slice_center_ids, dtype=np.int32
        ).reshape((-1,))
        if len(points) != len(ids):
            raise ValueError("Center points and label IDs must have matching rows")
        self.session.cell_slice_centers_zyx = points.copy()
        self.session.cell_slice_center_ids = ids.copy()
        self.session.cell_slice_center_accepted = np.ones(len(points), dtype=bool)
        if (
            precomputed_centers_xyz is not None
            and precomputed_center_diagnostics is not None
        ):
            data = np.asarray(labels, dtype=np.int32)
            self.session.cell_labels_raw = data.copy()
            self.session.cell_centers_xyz = {
                int(label_id): np.asarray(
                    center,
                    dtype=np.float64,
                ).copy()
                for label_id, center in precomputed_centers_xyz.items()
            }
            self.session.cell_center_diagnostics = copy.deepcopy(
                precomputed_center_diagnostics
            )
            self.session.group_labels_raw = data.copy()
            self.session.group_seed_xyz = copy.deepcopy(
                self.session.cell_centers_xyz
            )
            self.session.accepted_edges = data.copy()
        else:
            self._sync_source_regions(labels)

    def _sync_source_regions(self, labels: np.ndarray) -> None:
        data = np.asarray(labels, dtype=np.int32)
        points, ids, accepted = self._center_candidate_arrays()
        estimates = algorithm_api.robust_cell_centers_from_labels(
            data,
            self.session.raw,
            candidate_points_zyx=points,
            candidate_ids=ids,
            candidate_accepted=accepted,
        )
        self.session.cell_labels_raw = data.copy()
        self.session.cell_centers_xyz = {
            int(label_id): np.asarray(
                estimate.center_xyz,
                dtype=np.float64,
            ).copy()
            for label_id, estimate in estimates.items()
        }
        self.session.cell_center_diagnostics = {
            int(label_id): estimate.to_dict()
            for label_id, estimate in estimates.items()
        }
        # Transitional mirrors keep older provenance readers usable.
        self.session.group_labels_raw = data.copy()
        self.session.group_seed_xyz = copy.deepcopy(self.session.cell_centers_xyz)
        self.session.accepted_edges = data.copy()

    def _show_source_regions(
        self,
        labels: np.ndarray,
    ) -> None:
        self._set_labels(
            "cell_regions_raw",
            labels,
            opacity=0.58,
            editable=True,
        )
        if self._layer("merge_selection") is None:
            merge_layer = self._set_merge_selection_overlay(
                np.zeros_like(np.asarray(labels), dtype=np.uint8)
            )
            merge_layer.visible = False
        points, source_ids, accepted = self._center_candidate_arrays()
        accepted_uids = np.flatnonzero(accepted).astype(np.int64, copy=False)
        accepted_ids = source_ids[accepted_uids]
        self._set_points(
            "cell_centers_raw",
            points[accepted_uids],
            accepted_ids - 1,
            extra_features={
                "source_label_id": accepted_ids,
                "center_uid": accepted_uids,
            },
            show_text=False,
        )
        rejected_uids = np.flatnonzero(~accepted).astype(np.int64, copy=False)
        rejected_source_ids = source_ids[rejected_uids]
        self._set_points(
            "rejected_cell_centers_raw",
            points[rejected_uids],
            rejected_source_ids - 1,
            extra_features={
                "source_label_id": rejected_source_ids,
                "center_uid": rejected_uids,
            },
            face_color="red",
            border_color="darkred",
            symbol="x",
            show_text=False,
        )
        outlier_points: list[np.ndarray] = []
        for diagnostic in self.session.cell_center_diagnostics.values():
            slice_points = np.asarray(
                diagnostic.get("slice_centers_zyx", []),
                dtype=np.float32,
            ).reshape((-1, 3))
            inliers = np.asarray(
                diagnostic.get("slice_inliers", []),
                dtype=bool,
            ).reshape((-1,))
            if len(slice_points) == len(inliers):
                outlier_points.extend(slice_points[~inliers])
        outlier_array = np.asarray(
            outlier_points,
            dtype=np.float32,
        ).reshape((-1, 3))
        self._set_points(
            "centerline_outliers_raw",
            outlier_array,
            np.zeros(len(outlier_array), dtype=np.int32),
            face_color="magenta",
            border_color="white",
            symbol="x",
            show_text=False,
        )
        display_points: list[np.ndarray] = []
        display_label_ids: list[int] = []
        data = np.asarray(labels, dtype=np.int32)
        for label_id in sorted(
            int(value) for value in np.unique(data) if int(value) > 0
        ):
            center_xyz = self.session.cell_centers_xyz.get(label_id)
            if center_xyz is not None:
                point_zyx = np.asarray(
                    center_xyz, dtype=np.float32
                ).reshape((3,))[[2, 1, 0]]
            else:
                # Keep an actionable ID visible even when robust-center
                # validation is the reason Step 2 cannot continue.
                coordinates = np.argwhere(data == label_id)
                if len(coordinates) == 0:
                    continue
                mean_zyx = coordinates.mean(axis=0)
                nearest = int(
                    np.argmin(
                        np.sum(
                            (coordinates - mean_zyx[None, :]) ** 2,
                            axis=1,
                        )
                    )
                )
                point_zyx = coordinates[nearest].astype(np.float32)
            display_points.append(point_zyx)
            display_label_ids.append(label_id)
        source_ids_array = np.asarray(display_label_ids, dtype=np.int32)
        self._set_points(
            "source_cell_ids",
            np.asarray(display_points, dtype=np.float32).reshape((-1, 3)),
            source_ids_array - 1,
            extra_features={"source_label_id": source_ids_array},
        )

    def _invalidate_environment_for_cell_edit(self) -> None:
        self._interpolation_timer.stop()
        self._cell_fitting_timer.stop()
        for kind in ("interpolation", "cell_fitting"):
            self.generations[kind] = self.generations.get(kind, 0) + 1
            self._result_current[kind] = False
        self._latest_interpolation_ratio = None
        self._latest_fitting_ratio = None
        self._normalized_interpolated_raw = None
        self.session.background_estimation_stage = "stale"
        self.session.background_statistics.clear()
        self._hide_layers(
            "interpolated_raw",
            "interpolated_regions",
            "interpolated_edges",
            "interpolated_centers",
            "interpolated_centerline_outliers",
            "background_cross_sections",
            "cell_cross_sections",
            "background_surface",
            "cell_surface",
            "cell_ids",
        )

    def _commit_slice_center_edit(
        self,
        labels: np.ndarray,
        affected_slices: set[int],
        *,
        preserve_label_edit_undo: bool = False,
    ) -> None:
        if not preserve_label_edit_undo:
            self._discard_label_edit_undo()
        self.session.invalidate_after(CELL_REGIONS_STEP)
        self._invalidate_environment_for_cell_edit()
        self._sync_source_regions(labels)
        if self.session.reviewed_slices is None:
            self.session.reviewed_slices = np.zeros(labels.shape[0], dtype=bool)
        for z_index in affected_slices:
            if 0 <= z_index < self.session.reviewed_slices.size:
                self.session.reviewed_slices[z_index] = False
        self._result_current["cell_regions"] = True
        self._show_source_regions(labels)
        self._update_review_status()
        self._apply_layer_visibility()

    def _discard_label_edit_undo(self) -> None:
        self._label_edit_snapshot = None
        self.undo_label_edit.setEnabled(False)
        self._clear_marked_merge_cells()

    def _activate_manual_split_guides(self) -> None:
        if self._layer("cell_regions_raw") is None:
            self.global_status.setText(
                "No filled cell segments are available to split."
            )
            return
        self._clear_marked_merge_cells()
        guide_layer = self._ensure_manual_split_layer()
        self.viewer.layers.selection.active = guide_layer
        guide_layer.visible = True
        guide_layer.mode = "add_line"
        self.global_status.setText(
            f"Manual separator active on Z={self._current_z()}. Draw a magenta "
            "line across the shared neck. Add lines on other representative Z "
            "slices to control the tilt of one shared 3D separation plane. The "
            "optional target guard is off by default."
        )

    def _clear_manual_split_guides(self) -> None:
        guide_layer = self._ensure_manual_split_layer()
        guide_layer.data = []
        guide_layer.mode = "select"
        self.global_status.setText(
            "Cleared all unapplied manual separator lines."
        )

    def _label_edit_snapshot_from_session(
        self,
        labels: np.ndarray,
        *,
        operation: str,
    ) -> LabelEditSnapshot:
        reviewed = self.session.reviewed_slices
        return LabelEditSnapshot(
            operation=str(operation),
            labels=np.asarray(labels, dtype=np.int32).copy(),
            result_labels=None,
            center_points_zyx=self.session.cell_slice_centers_zyx.copy(),
            center_ids=self.session.cell_slice_center_ids.copy(),
            center_accepted=self.session.cell_slice_center_accepted.copy(),
            reviewed_slices=(
                None if reviewed is None else np.asarray(reviewed, dtype=bool).copy()
            ),
            rejected_slice_centers=copy.deepcopy(
                self._rejected_slice_centers
            ),
            rejected_segments=copy.deepcopy(self._rejected_segments),
        )

    @staticmethod
    def _reassign_seed_observations_after_label_split(
        labels: np.ndarray,
        points_zyx: np.ndarray,
        source_ids: np.ndarray,
        accepted: np.ndarray,
        source_to_children: dict[int, tuple[int, ...]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
        """Preserve reviewed seeds and move each one to its actual child."""

        label_volume = np.asarray(labels, dtype=np.int32)
        points = np.asarray(points_zyx, dtype=np.float32).reshape((-1, 3)).copy()
        ids = np.asarray(source_ids, dtype=np.int32).reshape((-1,)).copy()
        accepted_mask = np.asarray(accepted, dtype=bool).reshape((-1,)).copy()
        if len(points) != len(ids) or len(points) != len(accepted_mask):
            raise ValueError(
                "Seed points, label IDs, and accepted flags must have matching rows"
            )

        shape = np.asarray(label_volume.shape, dtype=np.int64)
        for source_id, raw_children in sorted(source_to_children.items()):
            children = tuple(int(value) for value in raw_children)
            child_set = set(children)
            child_coordinates = np.argwhere(
                np.isin(label_volume, np.asarray(children, dtype=np.int32))
            )
            if child_coordinates.size == 0:
                continue
            for center_uid in np.flatnonzero(ids == int(source_id)):
                point = np.asarray(points[center_uid], dtype=np.float64)
                rounded = np.clip(
                    np.rint(point).astype(np.int64),
                    np.zeros(3, dtype=np.int64),
                    shape - 1,
                )
                direct_label = int(label_volume[tuple(rounded)])
                if direct_label in child_set:
                    ids[center_uid] = direct_label
                    continue
                distances_squared = np.sum(
                    (child_coordinates.astype(np.float64) - point[None, :]) ** 2,
                    axis=1,
                )
                nearest = child_coordinates[int(np.argmin(distances_squared))]
                ids[center_uid] = int(label_volume[tuple(nearest)])

        fallback_points: list[np.ndarray] = []
        fallback_ids: list[int] = []
        for children in source_to_children.values():
            for child_id in children:
                child_id = int(child_id)
                if np.any(ids == child_id):
                    continue
                coordinates = np.argwhere(label_volume == child_id)
                if coordinates.size == 0:
                    continue
                geometric_center = np.mean(coordinates, axis=0)
                distances_squared = np.sum(
                    (
                        coordinates.astype(np.float64)
                        - geometric_center[None, :]
                    )
                    ** 2,
                    axis=1,
                )
                fallback_points.append(
                    coordinates[int(np.argmin(distances_squared))].astype(
                        np.float32
                    )
                )
                fallback_ids.append(child_id)

        if fallback_points:
            points = np.concatenate(
                (
                    points,
                    np.asarray(fallback_points, dtype=np.float32).reshape(
                        (-1, 3)
                    ),
                ),
                axis=0,
            )
            ids = np.concatenate(
                (ids, np.asarray(fallback_ids, dtype=np.int32)),
                axis=0,
            )
            accepted_mask = np.concatenate(
                (
                    accepted_mask,
                    np.ones(len(fallback_ids), dtype=bool),
                ),
                axis=0,
            )
        return points, ids, accepted_mask, tuple(fallback_ids)

    def _update_rejected_seed_labels(self, source_ids: np.ndarray) -> None:
        """Keep rejected-seed restoration records aligned with current IDs."""

        ids = np.asarray(source_ids, dtype=np.int32)
        for center_uid, record in self._rejected_slice_centers.items():
            if 0 <= int(center_uid) < len(ids):
                record.source_label_id = int(ids[int(center_uid)])

    def _apply_manual_split_guides(self) -> None:
        region_layer = self._layer("cell_regions_raw")
        guide_layer = self._ensure_manual_split_layer()
        if region_layer is None:
            self.global_status.setText(
                "No filled cell segments are available to split."
            )
            return
        guides = [
            np.asarray(line, dtype=np.float64).copy()
            for line in guide_layer.data
        ]
        try:
            result = algorithm_api.split_labels_with_line_guides(
                np.asarray(region_layer.data, dtype=np.int32),
                guides,
                minimum_target_overlap_fraction=(
                    self.manual_split_target_share.value()
                ),
            )
        except ValueError as exc:
            self.global_status.setText(f"Manual split not applied: {exc}")
            return

        self._clear_marked_merge_cells()
        self._label_edit_snapshot = self._label_edit_snapshot_from_session(
            np.asarray(region_layer.data, dtype=np.int32),
            operation="manual split",
        )
        self._label_edit_snapshot.result_labels = result.labels.copy()
        old_points, old_ids, old_accepted = self._center_candidate_arrays()
        points, source_ids, accepted, fallback_ids = (
            self._reassign_seed_observations_after_label_split(
                result.labels,
                old_points,
                old_ids,
                old_accepted,
                result.source_to_children,
            )
        )
        self._update_rejected_seed_labels(source_ids)
        self.session.cell_slice_centers_zyx = points
        self.session.cell_slice_center_ids = source_ids
        self.session.cell_slice_center_accepted = accepted
        self.restore_last_segment.setEnabled(bool(self._rejected_segments))
        self.undo_label_edit.setEnabled(True)
        guide_layer.mode = "select"
        self._commit_slice_center_edit(
            result.labels,
            set(result.affected_slices),
            preserve_label_edit_undo=True,
        )
        split_ids = ", ".join(
            str(source_label - 1)
            for source_label in sorted(result.source_to_children)
        )
        self.global_status.setText(
            f"Manual split applied to cell ID(s) {split_ids}. All guide lines "
            "for each cell formed one 3D separation plane: the left side kept "
            "the source ID and the right side received a new ID on every layer. "
            "Existing seed observations and rejection decisions were preserved "
            f"and reassigned; {len(fallback_ids)} new child cell(s) needed one "
            "fallback seed. Affected slices were marked for review. "
            "Clear the guides before drawing a different split, or use "
            "'Undo last split or merge' to withdraw this split."
        )

    def _undo_last_label_edit(self) -> None:
        snapshot = self._label_edit_snapshot
        if snapshot is None:
            self.global_status.setText(
                "No recent split or merge is available to undo."
            )
            self.undo_label_edit.setEnabled(False)
            return
        self.session.cell_slice_centers_zyx = (
            snapshot.center_points_zyx.copy()
        )
        self.session.cell_slice_center_ids = snapshot.center_ids.copy()
        self.session.cell_slice_center_accepted = (
            snapshot.center_accepted.copy()
        )
        self._rejected_slice_centers = copy.deepcopy(
            snapshot.rejected_slice_centers
        )
        self._rejected_segments = copy.deepcopy(
            snapshot.rejected_segments
        )
        self.session.invalidate_after(CELL_REGIONS_STEP)
        self._invalidate_environment_for_cell_edit()
        self._sync_source_regions(snapshot.labels)
        self.session.reviewed_slices = (
            None
            if snapshot.reviewed_slices is None
            else snapshot.reviewed_slices.copy()
        )
        self.restore_last_segment.setEnabled(bool(self._rejected_segments))
        self._result_current["cell_regions"] = True
        self._show_source_regions(snapshot.labels)
        self._update_review_status()
        self._apply_layer_visibility()
        operation = snapshot.operation
        self._label_edit_snapshot = None
        self.undo_label_edit.setEnabled(False)
        self._clear_marked_merge_cells()
        self.global_status.setText(
            "Restored the exact segmentation and seed-review state from before "
            f"the last {operation}. The magenta separator guides remain "
            "available for editing or reapplication."
        )

    def _drop_center_observations_on_mask(
        self,
        discarded_mask_zyx: np.ndarray,
        source_labels_zyx: np.ndarray,
    ) -> int:
        """Drop seeds on automatically discarded support and remap their UIDs."""

        discarded = np.asarray(discarded_mask_zyx, dtype=bool)
        source_labels = np.asarray(source_labels_zyx, dtype=np.int32)
        if (
            discarded.shape != self.session.raw.shape
            or source_labels.shape != discarded.shape
        ):
            raise ValueError(
                "Discarded support, source labels, and image must match"
            )
        points, source_ids, accepted = self._center_candidate_arrays()
        if len(points) == 0 or not np.any(discarded):
            return 0
        rounded = np.rint(points).astype(np.int64)
        limits = np.asarray(discarded.shape, dtype=np.int64) - 1
        rounded = np.clip(
            rounded,
            np.zeros(3, dtype=np.int64),
            limits,
        )
        drop = np.zeros(len(points), dtype=bool)
        support_cache: dict[int, np.ndarray] = {}
        for center_uid, (point, rounded_point, source_id) in enumerate(
            zip(points, rounded, source_ids, strict=True)
        ):
            label_id = int(source_id)
            if label_id <= 0:
                continue
            location = tuple(rounded_point)
            if int(source_labels[location]) == label_id:
                drop[center_uid] = bool(discarded[location])
                continue
            support = support_cache.get(label_id)
            if support is None:
                support = np.argwhere(source_labels == label_id)
                support_cache[label_id] = support
            if support.size == 0:
                continue
            nearest = support[
                int(
                    np.argmin(
                        np.sum(
                            (support.astype(np.float64) - point[None, :]) ** 2,
                            axis=1,
                        )
                    )
                )
            ]
            drop[center_uid] = bool(discarded[tuple(nearest)])
        if not np.any(drop):
            return 0

        kept_indices = np.flatnonzero(~drop).astype(np.int64, copy=False)
        old_to_new = {
            int(old_uid): int(new_uid)
            for new_uid, old_uid in enumerate(kept_indices)
        }
        self.session.cell_slice_centers_zyx = points[kept_indices].copy()
        self.session.cell_slice_center_ids = source_ids[kept_indices].copy()
        self.session.cell_slice_center_accepted = accepted[kept_indices].copy()

        # Copy the table in one operation so records that intentionally share
        # one removed-chunk snapshot keep that object identity after UID
        # remapping. Restoration uses identity to restore the chunk once.
        copied_rejections = copy.deepcopy(self._rejected_slice_centers)
        remapped_rejections: dict[int, RejectedSliceCenter] = {}
        for old_uid, updated in copied_rejections.items():
            new_uid = old_to_new.get(int(old_uid))
            if new_uid is None:
                continue
            updated.center_uid = new_uid
            remapped_rejections[new_uid] = updated
        self._rejected_slice_centers = remapped_rejections

        remapped_segments: list[RejectedSegment] = []
        for record in self._rejected_segments:
            remapped_segments.append(
                RejectedSegment(
                    source_label_id=int(record.source_label_id),
                    bounds_zyx=record.bounds_zyx,
                    local_mask=record.local_mask.copy(),
                    all_center_uids=tuple(
                        old_to_new[int(center_uid)]
                        for center_uid in record.all_center_uids
                        if int(center_uid) in old_to_new
                    ),
                    accepted_center_uids=tuple(
                        old_to_new[int(center_uid)]
                        for center_uid in record.accepted_center_uids
                        if int(center_uid) in old_to_new
                    ),
                )
            )
        self._rejected_segments = remapped_segments
        return int(np.count_nonzero(drop))

    def _auto_split_detached_support(
        self,
        labels: np.ndarray,
    ) -> tuple[np.ndarray, str]:
        """Preserve unresolved support by assigning it independent cell IDs."""

        selected_components_by_label = {
            int(label_id): tuple(
                int(value)
                for value in diagnostic.get("selected_components", [])
            )
            for label_id, diagnostic in (
                self.session.cell_center_diagnostics.items()
            )
            if int(diagnostic.get("ignored_component_voxels", 0)) > 0
        }
        if not selected_components_by_label:
            return np.asarray(labels, dtype=np.int32), ""
        result = algorithm_api.split_detached_label_support(
            labels,
            selected_components_by_label,
            minimum_child_voxels=self.cell_min_size.value(),
        )
        if (
            not result.source_to_children
            and not result.discarded_voxels_by_source
        ):
            return np.asarray(labels, dtype=np.int32), ""

        self._label_edit_snapshot = self._label_edit_snapshot_from_session(
            labels,
            operation="automatic detached-support repair",
        )
        self._label_edit_snapshot.result_labels = result.labels.copy()
        discarded_mask = (
            (np.asarray(labels, dtype=np.int32) > 0)
            & (np.asarray(result.labels, dtype=np.int32) == 0)
        )
        removed_seed_count = self._drop_center_observations_on_mask(
            discarded_mask,
            labels,
        )
        points, source_ids, accepted = self._center_candidate_arrays()
        if result.source_to_children:
            points, source_ids, accepted, fallback_ids = (
                self._reassign_seed_observations_after_label_split(
                    result.labels,
                    points,
                    source_ids,
                    accepted,
                    result.source_to_children,
                )
            )
        else:
            fallback_ids = ()
        self.session.cell_slice_centers_zyx = points
        self.session.cell_slice_center_ids = source_ids
        self.session.cell_slice_center_accepted = accepted
        self._update_rejected_seed_labels(source_ids)
        self.session.invalidate_after(CELL_REGIONS_STEP)
        self._invalidate_environment_for_cell_edit()
        self._sync_source_regions(result.labels)
        self._result_current["cell_regions"] = True
        self._show_source_regions(result.labels)
        self._update_review_status()
        self._apply_layer_visibility()
        self.undo_label_edit.setEnabled(True)

        split_source_display_ids = ", ".join(
            str(label_id - 1)
            for label_id in sorted(result.source_to_children)
        )
        discarded_source_display_ids = ", ".join(
            str(label_id - 1)
            for label_id in sorted(result.discarded_voxels_by_source)
        )
        created_ids = sorted(
            child_id
            for children in result.source_to_children.values()
            for child_id in children[1:]
        )
        created_display_ids = ", ".join(
            str(label_id - 1) for label_id in created_ids
        )
        fallback_suffix = (
            ""
            if not fallback_ids
            else (
                f" Added one fallback seed for {len(fallback_ids)} child "
                "cell(s) that had no prior seed observation."
            )
        )
        message_parts: list[str] = []
        if created_ids:
            message_parts.append(
                "Automatically separated retained detached 3D support from "
                f"cell ID(s) {split_source_display_ids} into new cell ID(s) "
                f"{created_display_ids}; no synthetic bridge was introduced."
            )
        discarded_voxels = int(
            sum(result.discarded_voxels_by_source.values())
        )
        discarded_groups = int(
            sum(result.discarded_group_count_by_source.values())
        )
        if discarded_voxels:
            message_parts.append(
                f"Discarded {discarded_groups} detached group(s) containing "
                f"{discarded_voxels} raw voxel(s) from cell ID(s) "
                f"{discarded_source_display_ids} because each was below the "
                "current Step 2 minimum of "
                f"{self.cell_min_size.value()} raw voxels."
            )
            if removed_seed_count:
                message_parts.append(
                    f"Removed {removed_seed_count} seed observation(s) that "
                    "belonged to the discarded support."
                )
        else:
            message_parts.append(
                "No segmented voxels were deleted; existing accepted/rejected "
                "seed decisions were reassigned to the correct child."
            )
        message_parts.append(fallback_suffix.strip())
        message_parts.append(
            "Use Back, then Undo last split or merge to restore the exact previous "
            "labels and seed-review state if needed."
        )
        message = " ".join(part for part in message_parts if part)
        return np.asarray(result.labels, dtype=np.int32), message

    @staticmethod
    def _component_id_for_center(
        component_labels: np.ndarray,
        source_mask: np.ndarray,
        point_zyx: np.ndarray,
    ) -> int:
        components = np.asarray(component_labels, dtype=np.int32)
        mask = np.asarray(source_mask, dtype=bool)
        point = np.asarray(point_zyx, dtype=np.float64)
        if components.shape != mask.shape or components.ndim != 3:
            raise ValueError(
                "Center component lookup requires matching 3D arrays"
            )
        rounded = np.rint(point).astype(np.int64)
        rounded = np.clip(
            rounded,
            np.zeros(3, dtype=np.int64),
            np.asarray(mask.shape, dtype=np.int64) - 1,
        )
        direct = int(components[tuple(rounded)])
        if direct > 0 and bool(mask[tuple(rounded)]):
            return direct
        support = np.argwhere(mask)
        if support.size == 0:
            return 0
        nearest = support[
            int(np.argmin(np.sum((support - point[None, :]) ** 2, axis=1)))
        ]
        return int(components[tuple(nearest)])

    def _reject_selected_centers(self) -> None:
        center_layer = self._layer("cell_centers_raw")
        region_layer = self._layer("cell_regions_raw")
        if center_layer is None or region_layer is None:
            self.global_status.setText(
                "No watershed seed observations are available to reject."
            )
            return
        selected = sorted(int(index) for index in center_layer.selected_data)
        if not selected:
            self.global_status.setText(
                "Select one or more yellow seed points in Napari first."
            )
            return
        if "center_uid" not in center_layer.features:
            self.global_status.setText(
                "The center layer is stale. Recalculate Step 2 before rejecting."
            )
            return
        layer_uids = np.asarray(
            center_layer.features["center_uid"],
            dtype=np.int64,
        )
        selected_uids = sorted(
            {
                int(layer_uids[index])
                for index in selected
                if 0 <= index < len(layer_uids)
            }
        )
        points, source_ids, accepted = self._center_candidate_arrays()
        selected_uids = [
            uid
            for uid in selected_uids
            if 0 <= uid < len(points) and accepted[uid]
        ]
        if not selected_uids:
            self.global_status.setText(
                "The selected centers were already rejected or are no longer valid."
            )
            return
        labels = np.asarray(region_layer.data, dtype=np.int32).copy()
        affected_slices: set[int] = set()
        accepted_after = accepted.copy()
        accepted_after[selected_uids] = False
        post_rejection_estimates = (
            algorithm_api.robust_cell_centers_from_labels(
                labels,
                self.session.raw,
                candidate_points_zyx=points,
                candidate_ids=source_ids,
                candidate_accepted=accepted_after,
            )
        )
        component_info: dict[
            int,
            tuple[
                tuple[slice, slice, slice],
                np.ndarray,
                np.ndarray,
                set[int],
            ],
        ] = {}
        selected_by_component: dict[tuple[int, int], list[int]] = {}

        for center_uid in selected_uids:
            point = points[center_uid]
            z_index = int(
                np.clip(round(float(point[0])), 0, labels.shape[0] - 1)
            )
            label_id = int(source_ids[center_uid])
            self._rejected_slice_centers[center_uid] = RejectedSliceCenter(
                center_uid=center_uid,
                point_zyx=point.copy(),
                source_label_id=label_id,
            )
            affected_slices.add(z_index)
            if label_id <= 0 or not np.any(labels == label_id):
                continue
            if label_id not in component_info:
                coordinates = np.argwhere(labels == label_id)
                bounds = tuple(
                    slice(
                        int(coordinates[:, axis].min()),
                        int(coordinates[:, axis].max()) + 1,
                    )
                    for axis in range(3)
                )
                local_mask = labels[bounds] == label_id
                components, _ = ndi.label(
                    local_mask,
                    structure=np.ones((3, 3, 3), dtype=bool),
                )
                estimate = post_rejection_estimates.get(label_id)
                selected_components = (
                    set()
                    if estimate is None
                    else {
                        int(value)
                        for value in estimate.selected_components
                    }
                )
                component_info[label_id] = (
                    bounds,
                    local_mask,
                    components,
                    selected_components,
                )
            bounds, local_mask, components, _ = component_info[label_id]
            origin = np.asarray(
                [int(axis.start) for axis in bounds],
                dtype=np.float64,
            )
            component_id = self._component_id_for_center(
                components,
                local_mask,
                np.asarray(point, dtype=np.float64) - origin,
            )
            if component_id > 0:
                selected_by_component.setdefault(
                    (label_id, component_id),
                    [],
                ).append(center_uid)

        removed_chunks = 0
        for (label_id, component_id), component_uids in (
            selected_by_component.items()
        ):
            bounds, local_mask, components, selected_components = (
                component_info[label_id]
            )
            # Even with zero accepted seeds, the component selected from the
            # solid label remains a valid cell body. Whole-cell deletion is a
            # separate explicit action; seed rejection only removes support
            # detached from this selected component group.
            if component_id in selected_components:
                continue
            origin = np.asarray(
                [int(axis.start) for axis in bounds],
                dtype=np.float64,
            )
            remaining_in_component = False
            for other_uid in np.flatnonzero(
                accepted_after & (source_ids == label_id)
            ):
                other_component = self._component_id_for_center(
                    components,
                    local_mask,
                    np.asarray(points[other_uid], dtype=np.float64)
                    - origin,
                )
                if other_component == component_id:
                    remaining_in_component = True
                    break
            if remaining_in_component:
                continue

            component_coordinates = np.argwhere(
                components == component_id
            )
            if component_coordinates.size == 0:
                continue
            local_bounds = tuple(
                slice(
                    int(component_coordinates[:, axis].min()),
                    int(component_coordinates[:, axis].max()) + 1,
                )
                for axis in range(3)
            )
            global_bounds = tuple(
                slice(
                    int(bounds[axis].start)
                    + int(local_bounds[axis].start),
                    int(bounds[axis].start)
                    + int(local_bounds[axis].stop),
                )
                for axis in range(3)
            )
            saved_mask = (
                components[local_bounds] == component_id
            ).copy()
            target = labels[global_bounds]
            target[saved_mask & (target == label_id)] = 0
            local_z_indices = np.flatnonzero(
                np.any(saved_mask, axis=(1, 2))
            )
            affected_slices.update(
                int(global_bounds[0].start) + int(local_z)
                for local_z in local_z_indices
            )

            # Associate the one snapshot with every already-rejected marker
            # from this detached component. Restoring any one of them can then
            # restore the complete 3D chunk exactly once.
            related_uids = set(component_uids)
            for rejected_uid, record in self._rejected_slice_centers.items():
                if (
                    int(record.source_label_id) != label_id
                    or record.removed_chunk_mask is not None
                ):
                    continue
                rejected_component = self._component_id_for_center(
                    components,
                    local_mask,
                    np.asarray(record.point_zyx, dtype=np.float64)
                    - origin,
                )
                if rejected_component == component_id:
                    related_uids.add(int(rejected_uid))
            for rejected_uid in related_uids:
                record = self._rejected_slice_centers[rejected_uid]
                record.removed_chunk_mask = saved_mask
                record.removed_chunk_bounds_zyx = global_bounds
            removed_chunks += 1

        self.session.cell_slice_center_accepted = accepted_after
        excluded_only = len(selected_uids) - sum(
            1
            for center_uid in selected_uids
            if self._rejected_slice_centers[
                center_uid
            ].removed_chunk_mask is not None
        )
        self.global_status.setText(
            f"Rejected {len(selected_uids)} seed observation(s): removed "
            f"{removed_chunks} detached 3D chunk(s), and excluded "
            f"{excluded_only} seed(s) while preserving their segmentation. "
            "Rejected seeds no longer contribute component-selection evidence; "
            "valid solid segmentation remains in the robust center fit."
        )
        self._commit_slice_center_edit(labels, affected_slices)

    def _activate_segment_picker(self) -> None:
        region_layer = self._layer("cell_regions_raw")
        if region_layer is None:
            self.global_status.setText(
                "No filled cell segments are available to select."
            )
            return
        self.viewer.layers.selection.active = region_layer
        region_layer.selected_label = 0
        region_layer.mode = "pick"
        self.global_status.setText(
            "Segment picker active. Click a colored filled cell, then either "
            "mark it wrong or toggle it in the merge selection."
        )

    def _refresh_marked_merge_display(self) -> None:
        region_layer = self._layer("cell_regions_raw")
        labels = (
            None
            if region_layer is None
            else np.asarray(region_layer.data, dtype=np.int32)
        )
        if labels is not None:
            present_labels = {
                int(value)
                for value in np.unique(labels)
                if int(value) > 0
            }
            self._marked_merge_label_ids.intersection_update(present_labels)
            highlighted = np.isin(
                labels,
                np.asarray(
                    sorted(self._marked_merge_label_ids),
                    dtype=np.int32,
                ),
            ).astype(np.uint8, copy=False)
            if self._marked_merge_label_ids or self._layer(
                "merge_selection"
            ) is not None:
                layer = self._set_merge_selection_overlay(
                    highlighted,
                )
                layer.visible = bool(
                    self.session.current_step == CELL_REGIONS_STEP
                    and self._marked_merge_label_ids
                )

        count = len(self._marked_merge_label_ids)
        if hasattr(self, "merge_marked_cells"):
            self.merge_marked_cells.setEnabled(count >= 2)
        if hasattr(self, "clear_marked_merge_cells"):
            self.clear_marked_merge_cells.setEnabled(count > 0)
        if hasattr(self, "manual_merge_status"):
            if count == 0:
                message = "No cells are marked for merge."
            else:
                display_ids = ", ".join(
                    str(label_id - 1)
                    for label_id in sorted(self._marked_merge_label_ids)
                )
                if count == 1:
                    message = (
                        f"Marked cell ID: {display_ids}. Mark at least one "
                        "more cell before merging."
                    )
                else:
                    message = (
                        f"Marked {count} cells: {display_ids}. The lowest "
                        "current ID will be retained."
                    )
            self.manual_merge_status.setText(message)

    def _clear_marked_merge_cells(self) -> None:
        self._marked_merge_label_ids.clear()
        layer = self._layer("merge_selection")
        if layer is not None:
            layer.data = np.zeros_like(
                np.asarray(layer.data, dtype=np.uint8)
            )
            layer.visible = False
        if hasattr(self, "merge_marked_cells"):
            self.merge_marked_cells.setEnabled(False)
        if hasattr(self, "clear_marked_merge_cells"):
            self.clear_marked_merge_cells.setEnabled(False)
        if hasattr(self, "manual_merge_status"):
            self.manual_merge_status.setText(
                "No cells are marked for merge."
            )

    def _toggle_picked_merge_cell(self) -> None:
        region_layer = self._layer("cell_regions_raw")
        if region_layer is None:
            self.global_status.setText(
                "No filled cell segments are available to merge."
            )
            return
        label_id = int(getattr(region_layer, "selected_label", 0))
        labels = np.asarray(region_layer.data, dtype=np.int32)
        if label_id <= 0 or not np.any(labels == label_id):
            self.global_status.setText(
                "Activate the colored-cell picker and click a non-background "
                "cell before marking or unmarking it."
            )
            return
        if label_id in self._marked_merge_label_ids:
            self._marked_merge_label_ids.remove(label_id)
            action = "Unmarked"
        else:
            self._marked_merge_label_ids.add(label_id)
            action = "Marked"
        self._refresh_marked_merge_display()
        region_layer.selected_label = 0
        self.viewer.layers.selection.active = region_layer
        region_layer.mode = "pick"
        self.global_status.setText(
            f"{action} cell ID {label_id - 1} for the next multi-cell merge."
        )

    def _merge_marked_cells(self) -> None:
        if self._busy_kind is not None:
            self.global_status.setText(
                "Wait for the current recalculation to finish before merging "
                "the marked cells."
            )
            return
        region_layer = self._layer("cell_regions_raw")
        if region_layer is None:
            self.global_status.setText(
                "No filled cell segments are available to merge."
            )
            return
        labels = np.asarray(region_layer.data, dtype=np.int32)
        present_labels = {
            int(value)
            for value in np.unique(labels)
            if int(value) > 0
        }
        self._marked_merge_label_ids.intersection_update(present_labels)
        marked_labels = tuple(sorted(self._marked_merge_label_ids))
        self._refresh_marked_merge_display()
        if len(marked_labels) < 2:
            self.global_status.setText(
                "Mark at least two current cells before applying a merge."
            )
            return

        try:
            result = algorithm_api.merge_cell_labels(
                labels,
                marked_labels,
            )
        except ValueError as exc:
            self.global_status.setText(f"Manual merge not applied: {exc}")
            return
        kept_label = int(result.kept_label)
        absorbed_labels = tuple(
            int(value) for value in result.absorbed_labels
        )
        points, source_ids, accepted = self._center_candidate_arrays()
        merged_source_ids = source_ids.copy()
        merged_source_ids[
            np.isin(
                merged_source_ids,
                np.asarray(absorbed_labels, dtype=np.int32),
            )
        ] = kept_label

        # Preflight only the selected union, cropped to its bounding box. The
        # normal commit below recalculates the complete Step-2 center table
        # once; avoiding a second whole-volume pass keeps this UI action
        # responsive on the full PAVAK stacks.
        target_mask = result.labels == kept_label
        target_coordinates = np.argwhere(target_mask)
        crop_bounds = tuple(
            slice(
                int(target_coordinates[:, axis].min()),
                int(target_coordinates[:, axis].max()) + 1,
            )
            for axis in range(3)
        )
        crop_origin = np.asarray(
            [int(axis.start) for axis in crop_bounds],
            dtype=np.float64,
        )
        target_candidate_rows = np.flatnonzero(
            merged_source_ids == kept_label
        )
        self._set_recalculating("manual_merge")
        QtWidgets.QApplication.processEvents(
            QtCore.QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents
        )
        try:
            estimates = algorithm_api.robust_cell_centers_from_labels(
                target_mask[crop_bounds].astype(np.int32),
                self.session.raw[crop_bounds],
                candidate_points_zyx=(
                    points[target_candidate_rows] - crop_origin[None, :]
                ),
                candidate_ids=np.ones(
                    len(target_candidate_rows),
                    dtype=np.int32,
                ),
                candidate_accepted=accepted[target_candidate_rows],
            )
        except Exception as exc:
            self._set_recalculation_finished("manual_merge")
            self.global_status.setText(
                f"Manual merge center check failed: {exc}"
            )
            traceback.print_exc()
            return
        merged_estimate = estimates.get(1)
        ignored_voxels = (
            -1
            if merged_estimate is None
            else int(merged_estimate.ignored_component_voxels)
        )
        if merged_estimate is None or ignored_voxels > 0:
            self._set_recalculation_finished("manual_merge")
            display_ids = ", ".join(
                str(label_id - 1) for label_id in marked_labels
            )
            self.global_status.setText(
                f"Marked cell IDs {display_ids} were not merged because their "
                "union is still detached in 3D. The complete marked set must "
                "form one connected cell or short, aligned Z continuations; "
                "no bridge voxels were invented."
            )
            return

        previous_snapshot = self._label_edit_snapshot
        merge_snapshot = self._label_edit_snapshot_from_session(
            labels,
            operation="manual merge",
        )
        merge_snapshot.result_labels = result.labels.copy()
        try:
            self.session.cell_slice_center_ids = merged_source_ids
            self._update_rejected_seed_labels(merged_source_ids)
            for record in self._rejected_segments:
                if int(record.source_label_id) in absorbed_labels:
                    record.source_label_id = kept_label
            self._commit_slice_center_edit(
                result.labels,
                set(result.affected_slices),
                preserve_label_edit_undo=True,
            )
        except Exception as exc:
            # Keep a failed display/commit transactional: return to the exact
            # pre-merge segmentation and review state, and retain any older
            # successful split/merge as the available one-level undo.
            rollback_error: Exception | None = None
            try:
                self.session.cell_slice_centers_zyx = (
                    merge_snapshot.center_points_zyx.copy()
                )
                self.session.cell_slice_center_ids = (
                    merge_snapshot.center_ids.copy()
                )
                self.session.cell_slice_center_accepted = (
                    merge_snapshot.center_accepted.copy()
                )
                self._rejected_slice_centers = copy.deepcopy(
                    merge_snapshot.rejected_slice_centers
                )
                self._rejected_segments = copy.deepcopy(
                    merge_snapshot.rejected_segments
                )
                self.session.invalidate_after(CELL_REGIONS_STEP)
                self._invalidate_environment_for_cell_edit()
                self._sync_source_regions(merge_snapshot.labels)
                self.session.reviewed_slices = (
                    None
                    if merge_snapshot.reviewed_slices is None
                    else merge_snapshot.reviewed_slices.copy()
                )
                self._result_current["cell_regions"] = True
                self._show_source_regions(merge_snapshot.labels)
                self._update_review_status()
                self._apply_layer_visibility()
            except Exception as rollback_exc:
                rollback_error = rollback_exc
                traceback.print_exc()
            finally:
                self._label_edit_snapshot = previous_snapshot
                self.undo_label_edit.setEnabled(previous_snapshot is not None)
                self._set_recalculation_finished("manual_merge")
            suffix = (
                ""
                if rollback_error is None
                else (
                    " The label state was restored in memory, but refreshing "
                    f"the viewer also failed: {rollback_error}"
                )
            )
            self.global_status.setText(
                "Manual merge was rolled back because its result could not "
                f"be applied: {exc}.{suffix}"
            )
            traceback.print_exception(type(exc), exc, exc.__traceback__)
            return
        self._label_edit_snapshot = merge_snapshot
        self._set_recalculation_finished("manual_merge")
        self.undo_label_edit.setEnabled(True)
        absorbed_display_ids = ", ".join(
            str(label_id - 1) for label_id in absorbed_labels
        )
        marked_count = len(marked_labels)
        self._clear_marked_merge_cells()
        region_layer = self._layer("cell_regions_raw")
        if region_layer is not None:
            region_layer.selected_label = kept_label
            self.viewer.layers.selection.active = region_layer
            region_layer.mode = "pick"
        self.manual_merge_status.setText(
            f"Merged {marked_count} marked cells; absorbed ID(s) "
            f"{absorbed_display_ids} into ID {kept_label - 1}. The marked "
            "selection was cleared."
        )
        self.global_status.setText(
            f"Merged {marked_count} marked cells into cell ID "
            f"{kept_label - 1} without changing foreground voxels; recomputed "
            f"its robust 3D center and marked {len(result.affected_slices)} "
            "occupied Z slice(s) for review. Use 'Undo last split or merge' "
            "to withdraw this merge."
        )

    def _reject_picked_segment(self) -> None:
        region_layer = self._layer("cell_regions_raw")
        if region_layer is None:
            self.global_status.setText(
                "No filled cell segments are available to reject."
            )
            return
        label_id = int(getattr(region_layer, "selected_label", 0))
        if label_id <= 0:
            self.global_status.setText(
                "Pick a non-background colored segment in the viewer first."
            )
            return
        labels = np.asarray(region_layer.data, dtype=np.int32).copy()
        selected_mask = labels == label_id
        if not np.any(selected_mask):
            self.global_status.setText(
                f"Segment label {label_id} is no longer present."
            )
            region_layer.selected_label = 0
            return

        coordinates = np.nonzero(selected_mask)
        bounds = tuple(
            slice(int(axis.min()), int(axis.max()) + 1)
            for axis in coordinates
        )
        local_mask = selected_mask[bounds].copy()
        points, source_ids, accepted = self._center_candidate_arrays()
        all_center_uids = np.flatnonzero(source_ids == label_id)
        accepted_center_uids = all_center_uids[accepted[all_center_uids]]
        self._rejected_segments.append(
            RejectedSegment(
                source_label_id=label_id,
                bounds_zyx=bounds,
                local_mask=local_mask,
                all_center_uids=tuple(int(uid) for uid in all_center_uids),
                accepted_center_uids=tuple(
                    int(uid) for uid in accepted_center_uids
                ),
            )
        )
        for center_uid in accepted_center_uids:
            center_uid = int(center_uid)
            self._rejected_slice_centers[center_uid] = RejectedSliceCenter(
                center_uid=center_uid,
                point_zyx=points[center_uid].copy(),
                source_label_id=label_id,
            )
            accepted[center_uid] = False
        labels[selected_mask] = 0
        self.session.cell_slice_center_accepted = accepted
        region_layer.selected_label = 0
        self.restore_last_segment.setEnabled(True)
        affected_slices = {
            int(z_index)
            for z_index in np.unique(coordinates[0])
        }
        self.global_status.setText(
            f"Marked segment {label_id - 1} wrong: removed "
            f"{int(selected_mask.sum())} voxels across "
            f"{len(affected_slices)} Z slice(s) and rejected "
            f"{len(accepted_center_uids)} associated accepted seed(s)."
        )
        self._commit_slice_center_edit(labels, affected_slices)

    def _restore_last_segment(self) -> None:
        region_layer = self._layer("cell_regions_raw")
        if region_layer is None or not self._rejected_segments:
            self.global_status.setText(
                "No rejected segment snapshot is available to restore."
            )
            self.restore_last_segment.setEnabled(False)
            return
        record = self._rejected_segments[-1]
        labels = np.asarray(region_layer.data, dtype=np.int32).copy()
        target = labels[record.bounds_zyx]
        if np.any(target[record.local_mask] > 0):
            self.global_status.setText(
                "The rejected segment cannot be restored because part of its "
                "former region is now occupied by another label."
            )
            return
        label_id = int(record.source_label_id)
        if label_id <= 0 or np.any(labels == label_id):
            label_id = int(labels.max(initial=0)) + 1
        target[record.local_mask] = label_id

        _, source_ids, accepted = self._center_candidate_arrays()
        all_center_uids = np.asarray(record.all_center_uids, dtype=np.int64)
        accepted_center_uids = np.asarray(
            record.accepted_center_uids,
            dtype=np.int64,
        )
        source_ids[all_center_uids] = label_id
        accepted[accepted_center_uids] = True
        for center_uid in record.all_center_uids:
            rejected_center = self._rejected_slice_centers.get(center_uid)
            if rejected_center is not None:
                rejected_center.source_label_id = label_id
        for center_uid in record.accepted_center_uids:
            self._rejected_slice_centers.pop(center_uid, None)
        self.session.cell_slice_center_ids = source_ids
        self.session.cell_slice_center_accepted = accepted
        self._rejected_segments.pop()
        self.restore_last_segment.setEnabled(bool(self._rejected_segments))
        local_z = np.flatnonzero(np.any(record.local_mask, axis=(1, 2)))
        z_start = int(record.bounds_zyx[0].start)
        affected_slices = {z_start + int(value) for value in local_z}
        self.global_status.setText(
            f"Restored the most recently rejected segment as cell "
            f"{label_id - 1}, including "
            f"{len(record.accepted_center_uids)} accepted seed(s)."
        )
        self._commit_slice_center_edit(labels, affected_slices)

    def _restore_selected_centers(self) -> None:
        rejected_layer = self._layer("rejected_cell_centers_raw")
        region_layer = self._layer("cell_regions_raw")
        if rejected_layer is None or region_layer is None:
            self.global_status.setText("No rejected centers are available to restore.")
            return
        selected = sorted(int(index) for index in rejected_layer.selected_data)
        if not selected:
            self.global_status.setText(
                "Select one or more red rejected-seed points in Napari first."
            )
            return
        if "center_uid" not in rejected_layer.features:
            self.global_status.setText(
                "The rejected-center layer is stale. Recalculate Step 2."
            )
            return
        layer_uids = np.asarray(
            rejected_layer.features["center_uid"],
            dtype=np.int64,
        )
        selected_uids = sorted(
            {
                int(layer_uids[index])
                for index in selected
                if 0 <= index < len(layer_uids)
            }
        )
        segment_center_uids = {
            center_uid
            for segment in self._rejected_segments
            for center_uid in segment.all_center_uids
        }
        blocked_segment_uids = [
            center_uid
            for center_uid in selected_uids
            if center_uid in segment_center_uids
        ]
        selected_uids = [
            center_uid
            for center_uid in selected_uids
            if center_uid not in segment_center_uids
        ]
        if not selected_uids and blocked_segment_uids:
            self.global_status.setText(
                "Those centers belong to a rejected segment. Restore the segment "
                "snapshot instead so its pixels and centers return together."
            )
            return
        labels = np.asarray(region_layer.data, dtype=np.int32).copy()
        _, source_ids, accepted = self._center_candidate_arrays()
        affected_slices: set[int] = set()
        restored_count = 0
        restored_chunks = 0

        def snapshot_for(
            record: RejectedSliceCenter,
        ) -> tuple[np.ndarray, tuple[slice, slice, slice]]:
            z_index = int(round(float(record.point_zyx[0])))
            saved_mask = np.asarray(
                record.removed_chunk_mask,
                dtype=bool,
            )
            bounds = record.removed_chunk_bounds_zyx
            if bounds is None:
                # Compatibility with an in-memory snapshot created by the
                # earlier 2D implementation before this widget reloaded.
                bounds = (
                    slice(z_index, z_index + 1),
                    slice(0, labels.shape[1]),
                    slice(0, labels.shape[2]),
                )
                saved_mask = saved_mask[None, ...]
            return saved_mask, bounds

        # Exact restoration is all-or-nothing. Do not consume a saved chunk
        # or accept its seed when later label edits now occupy any old voxel.
        for center_uid in selected_uids:
            record = self._rejected_slice_centers.get(center_uid)
            if record is None or record.removed_chunk_mask is None:
                continue
            saved_mask, bounds = snapshot_for(record)
            target = labels[bounds]
            if target.shape != saved_mask.shape or np.any(
                target[saved_mask] != 0
            ):
                self.global_status.setText(
                    "The detached 3D chunk cannot be restored exactly because "
                    "part of its saved region is now occupied. Undo or erase "
                    "the overlapping edit, then restore the red seed again."
                )
                return

        for center_uid in selected_uids:
            record = self._rejected_slice_centers.get(center_uid)
            if record is None:
                continue
            z_index = int(round(float(record.point_zyx[0])))
            if record.removed_chunk_mask is not None:
                saved_mask, bounds = snapshot_for(record)
                label_id = int(record.source_label_id)
                related_records = [
                    related
                    for related in self._rejected_slice_centers.values()
                    if related.removed_chunk_mask is record.removed_chunk_mask
                ]
                if label_id <= 0:
                    label_id = int(labels.max(initial=0)) + 1
                for related in related_records:
                    related.source_label_id = label_id
                    source_ids[related.center_uid] = label_id
                target = labels[bounds]
                target[saved_mask] = label_id
                if np.any(saved_mask):
                    restored_chunks += 1
                    local_z = np.flatnonzero(
                        np.any(saved_mask, axis=(1, 2))
                    )
                    affected_slices.update(
                        int(bounds[0].start) + int(value)
                        for value in local_z
                    )
                for related in related_records:
                    related.removed_chunk_mask = None
                    related.removed_chunk_bounds_zyx = None
            accepted[center_uid] = True
            affected_slices.add(z_index)
            restored_count += 1
            del self._rejected_slice_centers[center_uid]
        if restored_count == 0:
            self.global_status.setText(
                "The selected rejected centers were already restored."
            )
            return
        self.session.cell_slice_center_ids = source_ids
        self.session.cell_slice_center_accepted = accepted
        self.global_status.setText(
            f"Restored {restored_count} seed(s) and "
            f"{restored_chunks} isolated chunk snapshot(s)."
        )
        self._commit_slice_center_edit(labels, affected_slices)

    def _renumber_edited_regions(
        self,
        preserve_label_edit_undo: bool = False,
    ) -> None:
        layer = self._layer("cell_regions_raw")
        if layer is None:
            return
        # Review-time renumbering can compact sparse label IDs. A pending
        # marked set uses the pre-compaction IDs, so clear it rather than risk
        # applying a later merge to different cells.
        self._clear_marked_merge_cells()
        source_labels = np.asarray(layer.data, dtype=np.int32)
        snapshot = self._label_edit_snapshot
        keep_label_edit_undo = bool(
            preserve_label_edit_undo
            and snapshot is not None
            and snapshot.result_labels is not None
            and np.array_equal(source_labels, snapshot.result_labels)
        )
        if not keep_label_edit_undo:
            self._discard_label_edit_undo()
        self.session.invalidate_after(CELL_REGIONS_STEP)
        self._invalidate_environment_for_cell_edit()
        labels, _ = algorithm_api.renumber_group_labels(source_labels)
        id_mapping: dict[int, int] = {}
        for old_id in np.unique(source_labels):
            old_id = int(old_id)
            if old_id <= 0:
                continue
            mapped = np.unique(labels[source_labels == old_id])
            mapped = mapped[mapped > 0]
            if len(mapped) == 1:
                id_mapping[old_id] = int(mapped[0])
        points, source_ids, accepted = self._center_candidate_arrays()
        remapped_ids = np.asarray(
            [id_mapping.get(int(label_id), 0) for label_id in source_ids],
            dtype=np.int32,
        )
        missing_active = np.flatnonzero(accepted & (remapped_ids == 0))
        for center_uid in missing_active:
            accepted[center_uid] = False
            self._rejected_slice_centers.setdefault(
                int(center_uid),
                RejectedSliceCenter(
                    center_uid=int(center_uid),
                    point_zyx=points[center_uid].copy(),
                    source_label_id=0,
                ),
            )
        for record in self._rejected_slice_centers.values():
            record.source_label_id = id_mapping.get(
                int(record.source_label_id),
                0,
            )
        for record in self._rejected_segments:
            record.source_label_id = id_mapping.get(
                int(record.source_label_id),
                0,
            )
        self.session.cell_slice_center_ids = remapped_ids
        self.session.cell_slice_center_accepted = accepted
        self._sync_source_regions(labels)
        if (
            keep_label_edit_undo
            and snapshot is self._label_edit_snapshot
        ):
            snapshot.result_labels = labels.copy()
        self._result_current["cell_regions"] = True
        self._show_source_regions(labels)
        self._update_review_status()
        self._apply_layer_visibility()

    def _update_review_status(self) -> None:
        reviewed = self.session.reviewed_slices
        labels = self._source_region_labels()
        if reviewed is None:
            return
        count = int(reviewed.sum())
        current = self._current_z()
        state = "reviewed" if reviewed[current] else "not reviewed"
        cells = (
            0
            if labels is None
            else len([value for value in np.unique(labels) if value > 0])
        )
        auto_outliers = sum(
            int(
                np.count_nonzero(
                    ~np.asarray(
                        diagnostic.get("slice_inliers", []),
                        dtype=bool,
                    )
                )
            )
            for diagnostic in self.session.cell_center_diagnostics.values()
        )
        joined_gap_cells = sum(
            len(diagnostic.get("selected_components", [])) > 1
            for diagnostic in self.session.cell_center_diagnostics.values()
        )
        unresolved_component_cells = sum(
            int(diagnostic.get("ignored_component_voxels", 0)) > 0
            for diagnostic in self.session.cell_center_diagnostics.values()
        )
        self.cell_review_status.setText(
            f"{cells} cross-Z cells; reviewed {count}/{reviewed.size} original "
            f"slices. Current Z={current}: {state}. Colored pixels are solid "
            "enclosed regions; yellow points are watershed seed observations; "
            "centers come from robust intensity-weighted 3D chunks. "
            f"{auto_outliers} magenta slice centroid(s) are automatically "
            f"excluded under Diagnostics; {joined_gap_cells} cell(s) have "
            "short aligned Z runs joined, and "
            f"{unresolved_component_cells} cell(s) still have unresolved "
            "detached support; "
            f"{len(self._rejected_slice_centers)} red seed(s) and "
            f"{len(self._rejected_segments)} whole segment(s) are rejected."
        )

    def _accept_current_slice(self) -> None:
        if self.session.reviewed_slices is None:
            return
        self._renumber_edited_regions(preserve_label_edit_undo=True)
        self.session.reviewed_slices[self._current_z()] = True
        self._update_review_status()
        self._jump_unreviewed(1)

    def _mark_current_slice_empty(self) -> None:
        layer = self._layer("cell_regions_raw")
        if layer is None or self.session.reviewed_slices is None:
            return
        labels = np.asarray(layer.data, dtype=np.int32).copy()
        z = self._current_z()
        labels[z] = 0
        layer.data = labels
        self.session.reviewed_slices[z] = True
        self._renumber_edited_regions()
        self._jump_unreviewed(1)

    def _accept_all_slices(self) -> None:
        if self.session.reviewed_slices is None:
            return
        self._renumber_edited_regions(preserve_label_edit_undo=True)
        self.session.reviewed_slices[:] = True
        self._update_review_status()

    def _jump_unreviewed(self, direction: int) -> None:
        reviewed = self.session.reviewed_slices
        if reviewed is None or reviewed.all():
            return
        current = self._current_z()
        for offset in range(1, reviewed.size + 1):
            candidate = (current + direction * offset) % reviewed.size
            if not reviewed[candidate]:
                self._set_current_z(candidate)
                self._update_review_status()
                return

    def _on_z_ratio_changed(self, value: float) -> None:
        self.session.z_ratio = float(value)
        # A ratio change invalidates only stages downstream of the confirmed
        # source-space regions. In particular, no old fitted ellipsoid may be
        # mistaken for geometry in the new interpolated coordinate space.
        self.session.invalidate_after(CELL_REGIONS_STEP)
        self.session.membrane_ellipsoid = None
        self._latest_interpolation_ratio = None
        self._latest_fitting_ratio = None
        self._normalized_interpolated_raw = None
        self._result_current["interpolation"] = False
        self._result_current["cell_fitting"] = False
        self._cell_fitting_timer.stop()
        self.generations["cell_fitting"] = (
            self.generations.get("cell_fitting", 0) + 1
        )
        predicted = int(round((self.session.raw.shape[0] - 1) * value)) + 1
        self.z_status.setText(
            f"Target interpolated depth: {predicted} slices. Waiting to rebuild "
            "the image, cell boundaries, and centers…"
        )
        self._hide_layers(
            "interpolated_raw",
            "interpolated_regions",
            "interpolated_edges",
            "interpolated_centers",
            "interpolated_centerline_outliers",
            "background_cross_sections",
            "cell_cross_sections",
            "background_surface",
            "cell_surface",
            "cell_ids",
        )
        self._schedule_compute("interpolation", self._interpolation_timer)

    def _submit_interpolation_preview(self) -> None:
        labels = self._source_region_labels()
        if labels is None or self.session.membrane_mask is None:
            self.z_status.setText(
                "Interpolation requires confirmed membrane and cell regions."
            )
            self._set_recalculation_failed("interpolation")
            return
        ratio = self.z_ratio.value()
        self.z_status.setText(
            "Interpolating the raw image, ID-preserving cell boundaries, "
            "membrane support, then recalculating robust chunk centers…"
        )
        self._submit(
            "interpolation",
            self._compute_interpolation,
            self.session.raw,
            labels,
            self.session.membrane_mask,
            ratio,
            self.session.cell_centers_xyz,
        )

    @staticmethod
    def _compute_interpolation(
        raw: np.ndarray,
        labels_raw: np.ndarray,
        membrane_raw: np.ndarray,
        ratio: float,
        centers_xyz: dict[int, np.ndarray],
    ) -> InterpolationPreview:
        interpolated_raw = algorithm_api.interpolate_z_linear(raw, ratio)
        interpolated_labels, _ = algorithm_api.interpolate_labeled_z(
            labels_raw,
            ratio,
            centers_xyz,
        )
        interpolated_membrane = algorithm_api.interpolate_binary_signed_distance(
            membrane_raw, ratio
        )
        interpolated_edges = algorithm_api.interpolated_label_boundaries(
            interpolated_labels
        )
        center_estimates = algorithm_api.robust_cell_centers_from_labels(
            interpolated_labels,
            interpolated_raw,
        )
        scaled_centers = {
            int(label_id): np.asarray(
                estimate.center_xyz,
                dtype=np.float64,
            ).copy()
            for label_id, estimate in center_estimates.items()
        }
        return InterpolationPreview(
            ratio=float(ratio),
            interpolated_raw=interpolated_raw,
            interpolated_labels=interpolated_labels,
            interpolated_membrane=interpolated_membrane,
            interpolated_edges=interpolated_edges,
            scaled_centers_xyz=scaled_centers,
            center_diagnostics={
                int(label_id): estimate.to_dict()
                for label_id, estimate in center_estimates.items()
            },
        )

    def _show_interpolation_from_session(self) -> None:
        if (
            self.session.interpolated_raw is None
            or self.session.interpolated_cell_labels is None
            or self.session.interpolated_cell_edges is None
        ):
            return
        self._set_image(
            "interpolated_raw",
            self.session.interpolated_raw,
            colormap="gray",
            contrast_limits=self._raw_contrast_limits,
        )
        self._set_labels(
            "interpolated_regions",
            self.session.interpolated_cell_labels,
            opacity=0.45,
        )
        self._set_labels(
            "interpolated_edges",
            self.session.interpolated_cell_edges,
            opacity=0.72,
        )
        label_ids = sorted(self.session.interpolated_centers_xyz)
        points = np.array(
            [
                [
                    self.session.interpolated_centers_xyz[label_id][2],
                    self.session.interpolated_centers_xyz[label_id][1],
                    self.session.interpolated_centers_xyz[label_id][0],
                ]
                for label_id in label_ids
            ],
            dtype=np.float32,
        ).reshape((-1, 3))
        export_ids = list(range(len(label_ids)))
        self._set_points("interpolated_centers", points, export_ids)
        outlier_points: list[np.ndarray] = []
        for diagnostic in (
            self.session.interpolated_center_diagnostics.values()
        ):
            slice_points = np.asarray(
                diagnostic.get("slice_centers_zyx", []),
                dtype=np.float32,
            ).reshape((-1, 3))
            inliers = np.asarray(
                diagnostic.get("slice_inliers", []),
                dtype=bool,
            ).reshape((-1,))
            if len(slice_points) == len(inliers):
                outlier_points.extend(slice_points[~inliers])
        outlier_array = np.asarray(
            outlier_points,
            dtype=np.float32,
        ).reshape((-1, 3))
        self._set_points(
            "interpolated_centerline_outliers",
            outlier_array,
            np.zeros(len(outlier_array), dtype=np.int32),
            face_color="magenta",
            border_color="white",
            symbol="x",
            show_text=False,
        )

    @staticmethod
    def _material_interpolated_detached_support(
        labels: np.ndarray,
        diagnostics: dict[int, dict[str, Any]],
    ) -> dict[int, tuple[int, int, float]]:
        """Return labels whose ignored target support can alter a cell fit."""

        label_volume = np.asarray(labels, dtype=np.int32)
        counts = np.bincount(label_volume.ravel())
        material: dict[int, tuple[int, int, float]] = {}
        for label_id, diagnostic in diagnostics.items():
            label_id = int(label_id)
            total = (
                int(counts[label_id])
                if 0 <= label_id < counts.size
                else 0
            )
            ignored = int(
                diagnostic.get("ignored_component_voxels", 0)
            )
            if total <= 0 or ignored <= 0:
                continue
            fraction = float(ignored) / float(total)
            if (
                fraction
                > algorithm_api.INTERPOLATED_DETACHED_SUPPORT_REVIEW_FRACTION
            ):
                material[label_id] = (ignored, total, fraction)
        return material

    def _apply_interpolation_result(self, result: InterpolationPreview) -> None:
        if abs(result.ratio - self.z_ratio.value()) > 1e-6:
            return
        self._latest_interpolation_ratio = result.ratio
        self._latest_fitting_ratio = None
        self._normalized_interpolated_raw = None
        self._result_current["interpolation"] = True
        self._result_current["cell_fitting"] = False
        self.session.z_ratio = result.ratio
        self.session.interpolated_raw = result.interpolated_raw
        self.session.interpolated_cell_labels = result.interpolated_labels
        self.session.interpolated_membrane_mask = result.interpolated_membrane
        self.session.interpolated_cell_edges = result.interpolated_edges
        self.session.interpolated_centers_xyz = {
            int(label_id): np.asarray(center, dtype=np.float64).copy()
            for label_id, center in result.scaled_centers_xyz.items()
        }
        self.session.interpolated_center_diagnostics = copy.deepcopy(
            result.center_diagnostics
        )
        self.session.cell_regions = result.interpolated_labels
        self.session.membrane_ellipsoid = None
        self.session.baseline_cell_ellipsoids.clear()
        self.session.cell_brightness_offsets.clear()
        self.session.cell_ellipsoids.clear()
        self.session.background_estimation_stage = "stale"
        self.session.background_statistics.clear()
        self._hide_layers(
            "background_cross_sections",
            "cell_cross_sections",
            "background_surface",
            "cell_surface",
            "cell_ids",
        )
        self._show_interpolation_from_session()
        auto_outliers = sum(
            int(
                np.count_nonzero(
                    ~np.asarray(
                        diagnostic.get("slice_inliers", []),
                        dtype=bool,
                    )
                )
            )
            for diagnostic in result.center_diagnostics.values()
        )
        material_support = self._material_interpolated_detached_support(
            result.interpolated_labels,
            result.center_diagnostics,
        )
        material_message = ""
        if material_support:
            display_ids = ", ".join(
                str(label_id - 1)
                for label_id in sorted(material_support)
            )
            material_message = (
                f" Cell ID(s) {display_ids} have material detached support "
                "after interpolation; use Back to correct or remove them in "
                "Step 2 before fitting."
            )
        self.z_status.setText(
            f"Interpolation ready at ratio {result.ratio:.3f}: "
            f"{result.interpolated_raw.shape[0]} Z slices, "
            f"{len(result.scaled_centers_xyz)} robust chunk-derived centers, and "
            f"{auto_outliers} automatically excluded slice centroid(s). "
            "ID-preserving boundaries are ready; no ellipsoid has been fitted "
            f"yet.{material_message}"
        )
        self._apply_layer_visibility()

    def _clear_cell_fit_controls(self) -> None:
        while self.cell_fitting_controls_layout.count():
            item = self.cell_fitting_controls_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self.cell_fit_controls.clear()

    def _focus_cell_fit_control(self, label_id: int) -> bool:
        label_id = int(label_id)
        control = self.cell_fit_controls.get(label_id)
        if control is None:
            return False
        previous = self.cell_fit_controls.get(
            self._selected_cell_fit_label_id
            if self._selected_cell_fit_label_id is not None
            else -1
        )
        if previous is not None and previous is not control:
            previous.label.setStyleSheet("")
        self._selected_cell_fit_label_id = label_id
        control.label.setStyleSheet(
            "font-weight: 700; color: #ffd54f; "
            "background-color: rgba(0, 0, 0, 150); padding: 3px;"
        )
        self.cell_fitting_page.ensureWidgetVisible(control, 24, 64)
        control.slider.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
        export_id = self._cell_surface_label_ids.index(label_id)
        self.cell_fitting_status.setText(
            f"Selected Cell {export_id}. Adjust its highlighted brightness "
            "offset slider."
        )
        return True

    def _on_viewer_cell_clicked(self, _viewer: Any, event: Any) -> None:
        if self.session.current_step != CELL_FITTING_STEP:
            return
        button = getattr(event, "button", 1)
        if button not in (None, 1, "left"):
            return
        position = getattr(event, "position", None)
        view_direction = getattr(event, "view_direction", None)
        dims_displayed = getattr(event, "dims_displayed", None)
        if position is None or dims_displayed is None:
            return
        displayed = list(dims_displayed)
        if len(displayed) == 2:
            cell_layer = self._layer("cell_cross_sections")
            if cell_layer is None or not cell_layer.visible:
                return
            picked = cell_layer.get_value(
                position,
                view_direction=view_direction,
                dims_displayed=displayed,
                world=True,
            )
        elif len(displayed) == 3 and view_direction is not None:
            cell_layer = self._layer("cell_surface")
            if cell_layer is None or not cell_layer.visible:
                return
            picked = cell_layer.get_value(
                position,
                view_direction=view_direction,
                dims_displayed=displayed,
                world=True,
            )
        else:
            return
        value = picked[0] if isinstance(picked, tuple) else picked
        if value is None:
            return
        value = float(value)
        mesh_value = int(round(value))
        if (
            not np.isfinite(value)
            or not np.isclose(value, mesh_value, atol=0.25)
            or mesh_value < 1
            or mesh_value > len(self._cell_surface_label_ids)
        ):
            return
        self._focus_cell_fit_control(
            self._cell_surface_label_ids[mesh_value - 1]
        )

    def _rebuild_cell_fit_controls(self, baseline_models: list[Any]) -> None:
        labels = self.session.interpolated_cell_labels
        label_ids = (
            []
            if labels is None
            else sorted(int(value) for value in np.unique(labels) if value > 0)
        )
        if len(label_ids) != len(baseline_models):
            raise ValueError(
                "Baseline cell models do not match interpolated cell labels"
            )
        self._clear_cell_fit_controls()
        current_offsets: dict[int, float] = {}
        for export_id, (label_id, model) in enumerate(
            zip(label_ids, baseline_models, strict=True)
        ):
            headroom = max(0.0, 1.0 - float(model.brightness))
            maximum = min(0.5, headroom)
            value = float(
                np.clip(
                    self.session.cell_brightness_offsets.get(label_id, 0.0),
                    0.0,
                    maximum,
                )
            )
            current_offsets[label_id] = value
            control = FloatControl(
                (
                    f"Cell {export_id} brightness offset "
                    f"(baseline {model.brightness:.3f}; threshold "
                    f"{model.brightness:.3f} + offset)"
                ),
                0.0,
                maximum,
                value,
                0.005,
                decimals=3,
            )
            control.valueChanged.connect(
                partial(self._on_cell_brightness_offset_changed, label_id)
            )
            if maximum <= 1e-9:
                control.setEnabled(False)
            self.cell_fitting_controls_layout.addWidget(control)
            self.cell_fit_controls[label_id] = control
        self.session.cell_brightness_offsets = current_offsets
        if self._selected_cell_fit_label_id not in self.cell_fit_controls:
            self._selected_cell_fit_label_id = None
        elif self._selected_cell_fit_label_id is not None:
            selected = self.cell_fit_controls[self._selected_cell_fit_label_id]
            selected.label.setStyleSheet(
                "font-weight: 700; color: #ffd54f; "
                "background-color: rgba(0, 0, 0, 150); padding: 3px;"
            )
            self.cell_fitting_page.ensureWidgetVisible(selected, 24, 64)

    def _on_cell_brightness_offset_changed(
        self,
        label_id: int,
        value: float,
    ) -> None:
        previous = self.session.cell_brightness_offsets.get(int(label_id))
        value = float(value)
        if previous is not None and abs(float(previous) - value) <= 1e-9:
            return
        self.session.cell_brightness_offsets[int(label_id)] = value
        self.session.accepted_steps.discard(int(CELL_FITTING_STEP))
        self.session.accepted_steps.discard(int(REVIEW_STEP))
        self.session.revision += 1
        self._latest_fitting_ratio = None
        self._result_current["cell_fitting"] = False
        self.session.background_estimation_stage = "stale"
        self.session.background_statistics.clear()
        self.cell_fitting_status.setText(
            f"Cell {int(label_id) - 1} offset changed; recalculating the "
            "transparent fitted ellipsoids…"
        )
        self._schedule_compute("cell_fitting", self._cell_fitting_timer)

    def _reset_all_cell_offsets(self) -> None:
        changed = False
        for label_id, control in self.cell_fit_controls.items():
            if abs(control.value()) > 1e-9:
                changed = True
            control.set_value(0.0, emit=False)
            self.session.cell_brightness_offsets[label_id] = 0.0
        if changed:
            self.session.accepted_steps.discard(int(CELL_FITTING_STEP))
            self.session.accepted_steps.discard(int(REVIEW_STEP))
            self.session.revision += 1
            self._latest_fitting_ratio = None
            self._result_current["cell_fitting"] = False
            self.session.background_estimation_stage = "stale"
            self.session.background_statistics.clear()
            self.cell_fitting_status.setText(
                "All offsets reset; restoring the baseline ellipsoids…"
            )
            self._schedule_compute("cell_fitting", self._cell_fitting_timer)

    def _submit_cell_fitting_preview(self) -> None:
        if (
            self.session.interpolated_raw is None
            or self.session.interpolated_cell_labels is None
            or self.session.interpolated_membrane_mask is None
            or not self.session.interpolated_centers_xyz
        ):
            self.cell_fitting_status.setText(
                "Cell fitting requires a confirmed interpolation result."
            )
            self._set_recalculation_failed("cell_fitting")
            return
        label_ids = sorted(
            int(value)
            for value in np.unique(self.session.interpolated_cell_labels)
            if value > 0
        )
        offsets = {
            label_id: float(
                max(
                    0.0,
                    self.session.cell_brightness_offsets.get(label_id, 0.0),
                )
            )
            for label_id in label_ids
        }
        if len(self.session.baseline_cell_ellipsoids) == len(label_ids):
            for label_id, baseline in zip(
                label_ids,
                self.session.baseline_cell_ellipsoids,
                strict=True,
            ):
                offsets[label_id] = min(
                    offsets[label_id],
                    max(0.0, 1.0 - float(baseline.brightness)),
                    0.5,
                )
        self.session.cell_brightness_offsets = offsets.copy()
        self.cell_fitting_status.setText(
            "Fitting the interpolated background and cells, applying every "
            "cell's brightness offset, then excluding the final cell ellipsoids "
            "from the dim/bright background calculation…"
        )
        soft_margin = (
            self.session.membrane_params.soft_margin
            if self.session.membrane_params is not None
            else 0.05
        )
        self._submit(
            "cell_fitting",
            self._compute_cell_fitting,
            self.session.interpolated_raw,
            self.session.interpolated_cell_labels,
            self.session.interpolated_membrane_mask,
            self.session.interpolated_centers_xyz,
            self.session.z_ratio,
            offsets,
            copy.deepcopy(self.session.baseline_cell_ellipsoids),
            copy.deepcopy(self.session.membrane_ellipsoid),
            soft_margin,
            self._normalized_interpolated_raw,
        )

    @staticmethod
    def _compute_cell_fitting(
        interpolated_raw: np.ndarray,
        interpolated_labels: np.ndarray,
        interpolated_membrane: np.ndarray,
        centers_xyz: dict[int, np.ndarray],
        ratio: float,
        offsets: dict[int, float],
        baseline_models: list[Any],
        membrane_model: Any | None,
        background_soft_margin: float,
        normalized_raw: np.ndarray | None,
    ) -> CellFittingPreview:
        normalized = (
            algorithm_api.robust_normalize(interpolated_raw)
            if normalized_raw is None
            else np.asarray(normalized_raw, dtype=np.float32)
        )
        if membrane_model is None:
            membrane_model = algorithm_api.fit_ellipsoid_from_mask(
                interpolated_membrane,
                normalized,
                name="embryo_envelope",
                intensity_is_normalized=True,
            )
        if not baseline_models:
            baseline_models = algorithm_api.fit_cell_ellipsoids(
                interpolated_labels,
                normalized,
                intensity_is_normalized=True,
                centers_xyz=centers_xyz,
            )
        label_ids = sorted(
            int(value) for value in np.unique(interpolated_labels) if value > 0
        )
        if len(label_ids) != len(baseline_models):
            raise ValueError(
                "Interpolated labels and baseline ellipsoid count do not match"
            )
        refined_models: list[Any] = []
        refinement_status: dict[int, str] = {}
        for label_id, baseline in zip(
            label_ids,
            baseline_models,
            strict=True,
        ):
            result = algorithm_api.refine_cell_ellipsoid(
                normalized,
                interpolated_labels,
                label_id,
                baseline,
                offsets.get(label_id, 0.0),
            )
            refined_models.append(result.model)
            refinement_status[label_id] = (
                result.status
                if not result.reason
                else f"{result.status}: {result.reason}"
            )
        background_estimate = algorithm_api.estimate_final_backgrounds(
            normalized,
            membrane_model,
            refined_models,
            background_soft_margin,
        )
        membrane_model.brightness = float(background_estimate.hot)
        background_vertices, background_faces, background_values = (
            algorithm_api.combine_ellipsoid_meshes([membrane_model])
        )
        cell_vertices, cell_faces, cell_values = (
            algorithm_api.combine_ellipsoid_meshes(refined_models)
        )
        shape_zyx = tuple(int(length) for length in normalized.shape)
        background_fill = (
            algorithm_api.rasterize_ellipsoid_labels(
                shape_zyx, [membrane_model]
            )
            > 0
        )
        background_eroded = ndi.binary_erosion(
            background_fill,
            structure=np.ones((1, 3, 3), dtype=bool),
            border_value=0,
        )
        background_cross_sections = (
            background_fill & ~background_eroded
        ).astype(np.uint8)
        cell_cross_sections = algorithm_api.rasterize_ellipsoid_labels(
            shape_zyx, refined_models
        )
        return CellFittingPreview(
            ratio=float(ratio),
            normalized_raw=normalized,
            membrane_model=membrane_model,
            baseline_cell_models=baseline_models,
            cell_models=refined_models,
            offsets={int(key): float(value) for key, value in offsets.items()},
            refinement_status=refinement_status,
            background_vertices=background_vertices,
            background_faces=background_faces,
            background_values=background_values,
            cell_vertices=cell_vertices,
            cell_faces=cell_faces,
            cell_values=cell_values,
            background_cross_sections=background_cross_sections,
            cell_cross_sections=cell_cross_sections,
            cold_background=float(background_estimate.cold),
            hot_background=float(background_estimate.hot),
            background_statistics=background_estimate.to_dict(),
        )

    def _apply_cell_fitting_result(self, result: CellFittingPreview) -> None:
        if abs(result.ratio - self.session.z_ratio) > 1e-6:
            return
        current_offsets = {
            int(key): float(value)
            for key, value in self.session.cell_brightness_offsets.items()
        }
        if set(current_offsets) != set(result.offsets) or any(
            abs(current_offsets[key] - result.offsets[key]) > 1e-9
            for key in current_offsets
        ):
            return
        self._latest_fitting_ratio = result.ratio
        self._normalized_interpolated_raw = result.normalized_raw
        self._result_current["cell_fitting"] = True
        self.session.membrane_ellipsoid = result.membrane_model
        self.session.baseline_cell_ellipsoids = result.baseline_cell_models
        self.session.cell_ellipsoids = result.cell_models
        self.session.cell_brightness_offsets = result.offsets.copy()
        self.session.cold_background = float(result.cold_background)
        self.session.hot_background = float(result.hot_background)
        self.session.membrane_ellipsoid.brightness = float(
            result.hot_background
        )
        self.session.background_estimation_stage = (
            model_api.FINAL_BACKGROUND_ESTIMATION_STAGE
        )
        self.session.background_statistics = copy.deepcopy(
            result.background_statistics
        )
        self._cell_surface_label_ids = tuple(sorted(result.offsets))
        if len(self._cell_surface_label_ids) != len(result.cell_models):
            raise ValueError(
                "Fitted cell surfaces do not match brightness-offset controls"
            )
        self._rebuild_cell_fit_controls(result.baseline_cell_models)

        background_cross_sections = self._set_image(
            "background_cross_sections",
            result.background_cross_sections,
            colormap="cyan",
            opacity=0.9,
            blending="additive",
            contrast_limits=(0.0, 1.0),
        )
        cell_cross_sections = self._set_labels(
            "cell_cross_sections",
            result.cell_cross_sections,
            opacity=0.34,
        )
        if self.viewer.layers.index(
            cell_cross_sections
        ) < self.viewer.layers.index(background_cross_sections):
            self.viewer.layers.move(
                self.viewer.layers.index(cell_cross_sections),
                self.viewer.layers.index(background_cross_sections) + 1,
            )
        background_layer = self._set_surface(
            "background_surface",
            result.background_vertices,
            result.background_faces,
            result.background_values,
            opacity=0.35,
            colormap="cyan",
            blending="translucent",
        )
        wireframe = getattr(background_layer, "wireframe", None)
        if wireframe is not None:
            wireframe.visible = True
            wireframe.color = "white"
            wireframe.width = 1.25

        cell_layer = self._set_surface(
            "cell_surface",
            result.cell_vertices,
            result.cell_faces,
            result.cell_values,
            opacity=0.32,
            colormap="turbo",
            blending="translucent_no_depth",
        )
        if self.viewer.layers.index(cell_layer) < self.viewer.layers.index(
            background_layer
        ):
            self.viewer.layers.move(
                self.viewer.layers.index(cell_layer),
                self.viewer.layers.index(background_layer) + 1,
            )
        points = np.array(
            [
                [model.center_xyz[2], model.center_xyz[1], model.center_xyz[0]]
                for model in result.cell_models
            ],
            dtype=np.float32,
        ).reshape((-1, 3))
        ids = [int(model.cell_id) for model in result.cell_models]
        id_layer = self._set_points("cell_ids", points, ids)
        if self.viewer.layers.index(id_layer) < self.viewer.layers.index(cell_layer):
            self.viewer.layers.move(
                self.viewer.layers.index(id_layer),
                self.viewer.layers.index(cell_layer) + 1,
            )

        fallback_count = sum(
            value.startswith("fallback")
            for value in result.refinement_status.values()
        )
        suffix = (
            ""
            if fallback_count == 0
            else (
                f" {fallback_count} cell(s) had no valid brighter component "
                "and retained their baseline shape."
            )
        )
        self.cell_fitting_status.setText(
            f"Fitted {len(result.cell_models)} cell ellipsoids at Z ratio "
            f"{result.ratio:.3f}. Smooth 3D surfaces are ready; safe 2D "
            f"cross-sections remain available as a fallback. Final "
            f"cell-excluded backgrounds: dim/cold "
            f"{result.cold_background:.4f}, bright/hot "
            f"{result.hot_background:.4f}.{suffix}"
        )
        self._apply_layer_visibility()

    def _prepare_review(self) -> None:
        if (
            self.session.interpolated_raw is None
            or not self.session.cell_ellipsoids
            or self.session.membrane_ellipsoid is None
            or self.session.background_estimation_stage
            != model_api.FINAL_BACKGROUND_ESTIMATION_STAGE
        ):
            self.review_summary.setText(
                "The simulated environment is incomplete. Return to Cell "
                "ellipsoid fitting and wait for recalculation."
            )
            self.next_button.setEnabled(False)
            return
        self.review_summary.setText(
            f"Ready to export {len(self.session.cell_ellipsoids)} cells plus one "
            "soft-margin embryo background ellipsoid. The normal view contains "
            "the interpolated 3D volume, smooth translucent cell and background "
            "surfaces, and exactly one numbered ID per cell. Uncheck the 3D "
            "box only when the safe 2D cross-section fallback is needed.\n"
            f"Final cell-excluded background brightness: dim/cold "
            f"{self.session.cold_background:.4f}, bright/hot "
            f"{self.session.hot_background:.4f}.\n"
            f"Output directory: {self.session.output_dir}"
        )
        self._apply_layer_visibility()
        autosave_session(self.session)

    def _confirm_current_step(self) -> None:
        try:
            step = self.session.current_step
            if step == MEMBRANE_STEP:
                self._confirm_membrane()
            elif step == CELL_REGIONS_STEP:
                self._confirm_cell_regions()
            elif step == Z_STEP:
                self._confirm_z()
            elif step == CELL_FITTING_STEP:
                self._confirm_cell_fitting()
            elif step == REVIEW_STEP:
                self._export_and_exit()
        except Exception as exc:
            self.global_status.setText(str(exc))
            QtWidgets.QMessageBox.warning(self, "Cannot continue", str(exc))

    def _advance(self, step: Step) -> None:
        self.session.mark_accepted(step)
        autosave_session(self.session)
        index = WORKFLOW_STEPS.index(step)
        self._set_step(WORKFLOW_STEPS[index + 1], entering=True)

    def _confirm_membrane(self) -> None:
        if not self._result_current["membrane"]:
            raise ValueError("The membrane result is still being recalculated")
        result = self._latest_membrane_result
        if result is None:
            raise ValueError("Membrane analysis has not completed")
        if result.component_count != 1:
            raise ValueError(
                f"Exactly one membrane region is required; detected "
                f"{result.component_count}"
            )
        mask = algorithm_api.largest_component(result.labels)
        if mask.sum() < 8:
            raise ValueError("The membrane region is too small")
        model = algorithm_api.fit_ellipsoid_from_mask(
            mask,
            algorithm_api.robust_normalize(self.session.raw),
            name="embryo_envelope",
        )
        self.session.membrane_mask = mask
        self.session.membrane_ellipsoid_raw = model
        # This is a source-Z diagnostic fit only. The export-space background
        # ellipsoid is deliberately deferred until after Z interpolation has
        # been confirmed.
        self.session.membrane_ellipsoid = None
        self.session.membrane_params = self._membrane_parameters()
        self.session.cold_background, self.session.hot_background = (
            algorithm_api.estimate_two_backgrounds(self.session.raw, mask)
        )
        self.session.background_estimation_stage = "source_provisional"
        self.session.background_statistics.clear()
        self._update_background_preview()
        self._advance(MEMBRANE_STEP)

    def _confirm_cell_regions(self) -> None:
        if not self._result_current["cell_regions"]:
            raise ValueError(
                "The solid cell-region result is still being recalculated"
            )
        if (
            self.session.reviewed_slices is None
            or not self.session.reviewed_slices.all()
        ):
            missing = (
                self.session.raw.shape[0]
                if self.session.reviewed_slices is None
                else int((~self.session.reviewed_slices).sum())
            )
            raise ValueError(f"{missing} original Z slices are still unreviewed")
        layer = self._layer("cell_regions_raw")
        if layer is None:
            raise ValueError("Solid cell-region layer is missing")
        self._renumber_edited_regions(preserve_label_edit_undo=True)
        labels = np.asarray(self.session.cell_labels_raw, dtype=np.int32)
        if not np.any(labels > 0):
            raise ValueError("No solid cell regions were accepted")
        label_ids = {
            int(value) for value in np.unique(labels) if int(value) > 0
        }
        missing_centers = sorted(label_ids - set(self.session.cell_centers_xyz))
        if missing_centers:
            formatted = ", ".join(str(value - 1) for value in missing_centers)
            raise ValueError(
                "A robust center could not be derived for every remaining "
                f"solid cell. Inspect or edit cell ID(s): {formatted}"
            )
        unresolved_components = {
            label_id
            for label_id, diagnostic in (
                self.session.cell_center_diagnostics.items()
            )
            if int(diagnostic.get("ignored_component_voxels", 0)) > 0
        }
        automatic_repair_message = ""
        if unresolved_components:
            labels, automatic_repair_message = (
                self._auto_split_detached_support(labels)
            )
            label_ids = {
                int(value) for value in np.unique(labels) if int(value) > 0
            }
            missing_centers = sorted(
                label_ids - set(self.session.cell_centers_xyz)
            )
            if missing_centers:
                formatted = ", ".join(
                    str(value - 1) for value in missing_centers
                )
                raise ValueError(
                    "Automatic detached-support separation could not derive "
                    f"a robust center for cell ID(s): {formatted}"
                )
            unresolved_components = {
                label_id
                for label_id, diagnostic in (
                    self.session.cell_center_diagnostics.items()
                )
                if int(diagnostic.get("ignored_component_voxels", 0)) > 0
            }
        if unresolved_components:
            formatted = ", ".join(
                str(value - 1) for value in sorted(unresolved_components)
            )
            raise ValueError(
                "Automatic detached-support separation could not resolve cell "
                f"ID(s): {formatted}. Return to Step 2 and inspect those cells."
            )
        self.session.cell_edge_params = self._cell_region_parameters()
        self.session.cold_background, self.session.hot_background = (
            algorithm_api.estimate_two_backgrounds(
                self.session.raw,
                self.session.membrane_mask,
                occupied_mask=labels > 0,
            )
        )
        self.session.background_estimation_stage = (
            "source_cell_labels_excluded_provisional"
        )
        self.session.background_statistics.clear()
        self._advance(CELL_REGIONS_STEP)
        if automatic_repair_message:
            self.global_status.setText(automatic_repair_message)

    def _confirm_z(self) -> None:
        ratio = self.z_ratio.value()
        if (
            not self._result_current["interpolation"]
            or self.session.interpolated_raw is None
            or self.session.interpolated_cell_labels is None
            or self.session.interpolated_membrane_mask is None
            or self.session.interpolated_cell_edges is None
            or self._latest_interpolation_ratio is None
            or abs(self._latest_interpolation_ratio - ratio) > 1e-6
        ):
            raise ValueError(
                "The interpolated image, boundaries, and centers for the current "
                "Z ratio are not ready"
            )
        material_support = self._material_interpolated_detached_support(
            self.session.interpolated_cell_labels,
            self.session.interpolated_center_diagnostics,
        )
        if material_support:
            details = ", ".join(
                (
                    f"{label_id - 1} "
                    f"({fraction * 100.0:.1f}% detached)"
                )
                for label_id, (_, _, fraction) in sorted(
                    material_support.items()
                )
            )
            raise ValueError(
                "Interpolation produced materially detached support for cell "
                f"ID(s) {details}. This can distort the ellipsoid fit. Use Back "
                "to Step 2 and correct, split, or remove those cell regions "
                "before continuing"
            )
        labels = np.asarray(
            self.session.interpolated_cell_labels,
            dtype=np.int32,
        )
        label_ids = sorted(
            int(value) for value in np.unique(labels) if int(value) > 0
        )
        counts = np.bincount(labels.ravel())
        too_small = [
            (display_id, int(counts[label_id]))
            for display_id, label_id in enumerate(label_ids)
            if int(counts[label_id]) < 8
        ]
        if too_small:
            details = ", ".join(
                f"{display_id} ({voxel_count} voxels)"
                for display_id, voxel_count in too_small
            )
            raise ValueError(
                "Interpolated cell ID(s) cannot support an ellipsoid fit: "
                f"{details}. Each cell needs at least 8 filled voxels. Use "
                "Back to Step 2 and remove or repair those cells."
            )
        self.session.z_ratio = ratio
        self._advance(Z_STEP)

    def _confirm_cell_fitting(self) -> None:
        labels = self.session.interpolated_cell_labels
        label_ids = (
            []
            if labels is None
            else sorted(int(value) for value in np.unique(labels) if value > 0)
        )
        if (
            not self._result_current["cell_fitting"]
            or self._latest_fitting_ratio is None
            or abs(self._latest_fitting_ratio - self.session.z_ratio) > 1e-6
            or self.session.membrane_ellipsoid is None
            or len(self.session.baseline_cell_ellipsoids) != len(label_ids)
            or len(self.session.cell_ellipsoids) != len(label_ids)
            or not label_ids
            or self.session.background_estimation_stage
            != model_api.FINAL_BACKGROUND_ESTIMATION_STAGE
        ):
            raise ValueError(
                "The per-cell ellipsoid fits for the current brightness offsets "
                "are not ready"
            )
        if set(self.session.cell_brightness_offsets) != set(label_ids):
            raise ValueError(
                "Every interpolated cell must have one brightness-offset control"
            )
        self._advance(CELL_FITTING_STEP)

    def _export_and_exit(self) -> None:
        csv_path, provenance_path = export_initial_csv(self.session)
        QtWidgets.QMessageBox.information(
            self,
            "CellUniverse initializer export complete",
            f"Exported:\n{csv_path}\n\nSession metadata:\n{provenance_path}",
        )
        print(f"Exported initial CSV: {csv_path}")
        print(f"Exported initializer metadata: {provenance_path}")
        self.viewer.close()

    def _go_back(self) -> None:
        step = self.session.current_step
        if step == MEMBRANE_STEP:
            return
        self._cancel_pending_computations()
        index = WORKFLOW_STEPS.index(step)
        target = WORKFLOW_STEPS[index - 1]
        self.session.invalidate_after(target)
        self._synchronize_result_state_after_rollback(target)
        self._set_step(target, entering=True)

    def _rollback_from_review(self) -> None:
        self._cancel_pending_computations()
        target = WORKFLOW_STEPS[self.rollback_step.currentIndex()]
        self.session.invalidate_after(target)
        self._synchronize_result_state_after_rollback(target)
        self._set_step(target, entering=True)

    def _synchronize_result_state_after_rollback(self, target: Step) -> None:
        if target < Z_STEP:
            self._result_current["interpolation"] = False
            self._latest_interpolation_ratio = None
            self._normalized_interpolated_raw = None
        else:
            self._result_current["interpolation"] = (
                self.session.interpolated_raw is not None
                and self.session.interpolated_cell_labels is not None
                and self.session.interpolated_membrane_mask is not None
                and self.session.interpolated_cell_edges is not None
                and self._latest_interpolation_ratio is not None
                and abs(
                    self._latest_interpolation_ratio - self.session.z_ratio
                )
                <= 1e-6
            )
        if target < CELL_FITTING_STEP:
            self._result_current["cell_fitting"] = False
            self._latest_fitting_ratio = None
            self.session.membrane_ellipsoid = None
        else:
            self._result_current["cell_fitting"] = (
                self.session.membrane_ellipsoid is not None
                and bool(self.session.baseline_cell_ellipsoids)
                and len(self.session.baseline_cell_ellipsoids)
                == len(self.session.cell_ellipsoids)
                and self._latest_fitting_ratio is not None
                and abs(self._latest_fitting_ratio - self.session.z_ratio) <= 1e-6
            )

    def _save_session(self) -> None:
        path = autosave_session(self.session)
        self.global_status.setText(f"Session state saved to {path}")

    def _cancel(self) -> None:
        response = QtWidgets.QMessageBox.question(
            self,
            "Cancel initializer",
            "Save the current session and close without exporting initial.csv?",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No
            | QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        if response == QtWidgets.QMessageBox.StandardButton.Cancel:
            return
        if response == QtWidgets.QMessageBox.StandardButton.Yes:
            autosave_session(self.session)
        self._cancel_pending_computations()
        self.viewer.close()

    def closeEvent(self, event: Any) -> None:
        callbacks = getattr(self.viewer, "mouse_drag_callbacks", None)
        if callbacks is not None and self._cell_pick_callback in callbacks:
            callbacks.remove(self._cell_pick_callback)
        self._cancel_pending_computations()
        super().closeEvent(event)
