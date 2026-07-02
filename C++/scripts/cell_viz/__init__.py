"""Live Napari cell-tracking visualizer for CellUniverse.

Read-only companion that overlays the tracker's per-frame decisions
(final ellipsoids, CellLumen brightness centers, split parent ghosts)
on the raw 3D image, live as a run progresses or in replay.

See docs/plans/2026-06-02-live-napari-cell-viz-design.md.
"""

__all__ = [
    "model",
    "lineage",
    "readers",
    "geometry",
]
