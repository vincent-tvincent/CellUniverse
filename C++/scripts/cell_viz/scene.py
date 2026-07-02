"""Build napari-ready geometry arrays from frame data.

Kept free of napari imports so scene construction is unit-testable. ``render.py``
turns a :class:`Scene` into napari layers; ``mpl_render.py`` reuses the same
colours and split logic.

All coordinates are napari order ``(z_index, y, x)``.

Colour scheme (few hues, high contrast on the green synth fill):
- normal tracked cell  -> WHITE outline
- a split's daughters  -> ORANGE outline (this frame's new children)
- the split parent      -> ORANGE dashed/ghost outline from the PREVIOUS frame,
                           labelled and linked to its daughters
- trash                -> GRAY
Brightness is carried by the green synth *fill*, not the outline colour.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import geometry, lineage
from .model import FrameData

OUTLINE_NORMAL = (0.92, 0.92, 0.92, 0.95)   # white
OUTLINE_DAUGHTER = (1.0, 0.55, 0.10, 1.0)   # orange — a child of a split this frame
OUTLINE_TRASH = (0.55, 0.55, 0.55, 0.8)     # gray
GHOST_COLOR = (1.0, 0.55, 0.10, 1.0)        # orange — last-frame parent


@dataclass
class Scene:
    """napari-ready arrays for one rendered frame."""

    ring_paths: list[np.ndarray] = field(default_factory=list)
    ring_colors: list[tuple] = field(default_factory=list)
    axis_paths: list[np.ndarray] = field(default_factory=list)
    axis_colors: list[tuple] = field(default_factory=list)
    lumen_points: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    lumen_sizes: np.ndarray = field(default_factory=lambda: np.empty((0,)))
    # split parent ghosts (prev-frame outline)
    ghost_paths: list[np.ndarray] = field(default_factory=list)
    # parent->daughter connector lines: list of (2,3)
    link_paths: list[np.ndarray] = field(default_factory=list)
    # cell name labels
    label_points: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    label_texts: list[str] = field(default_factory=list)
    # ghost-parent labels (e.g. "↳1_2011 (f-1)")
    ghost_label_points: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    ghost_label_texts: list[str] = field(default_factory=list)


def build_scene(
    curr: FrameData,
    prev: FrameData | None,
    z_scale: float,
    reveal: int | None = None,
    ring_points: int = 40,
    show_labels: bool = True,
    lumen_size: float = 6.0,
) -> Scene:
    """Build a Scene. ``reveal`` limits cells drawn to the first N in fit order
    (for cell-by-cell animation); None draws all."""
    scene = Scene()

    # Committed splits first so we can mark this frame's daughters distinctly.
    splits = lineage.committed_splits(prev, curr) if prev is not None else []
    daughters = {}
    for parent_cell, d1, d2 in splits:
        daughters[d1.name] = parent_cell
        daughters[d2.name] = parent_cell

    ordered = curr.cells_in_fit_order()
    if reveal is not None:
        ordered = ordered[: max(0, reveal)]

    drawn_names = set()
    for cell in ordered:
        drawn_names.add(cell.name)
        if cell.is_trash:
            color = OUTLINE_TRASH
        elif cell.name in daughters:
            color = OUTLINE_DAUGHTER
        else:
            color = OUTLINE_NORMAL
        sil = geometry.silhouette_path_napari(
            cell.center, cell.radii, cell.angles, z_scale, n=ring_points
        )
        scene.ring_paths.append(sil)
        scene.ring_colors.append(color)
        if show_labels and not cell.is_trash:
            scene.label_points = np.vstack(
                [scene.label_points, geometry.center_napari(cell.center, z_scale)]
            )
            parts = lineage.split_name(cell.name)
            scene.label_texts.append(parts[1] if parts else cell.name)

    # CellLumen brightness centers (detector output; not fit-ordered).
    if curr.lumen_centers:
        pts = np.array(
            [geometry.center_napari(lc.center, z_scale) for lc in curr.lumen_centers]
        )
        vox = np.array([max(lc.voxels, 1.0) for lc in curr.lumen_centers])
        scene.lumen_points = pts
        scene.lumen_sizes = lumen_size * (0.6 + 0.4 * (vox / (vox.max() or 1.0)))

    # Split parent ghosts (PREVIOUS-frame outline) + label + connectors.
    for parent_cell, d1, d2 in splits:
        if d1.name not in drawn_names and d2.name not in drawn_names:
            continue  # daughters not yet revealed in this animation step
        scene.ghost_paths.append(geometry.silhouette_path_napari(
            parent_cell.center, parent_cell.radii, parent_cell.angles,
            z_scale, n=ring_points,
        ))
        pc = geometry.center_napari(parent_cell.center, z_scale)
        for d in (d1, d2):
            scene.link_paths.append(
                np.vstack([pc, geometry.center_napari(d.center, z_scale)])
            )
        if show_labels:
            scene.ghost_label_points = np.vstack([scene.ghost_label_points, pc])
            parts = lineage.split_name(parent_cell.name)
            scene.ghost_label_texts.append(
                f"↳{parts[1]} (f-1)" if parts else f"{parent_cell.name} (f-1)"
            )
    return scene
