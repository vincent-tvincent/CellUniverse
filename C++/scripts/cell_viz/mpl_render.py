"""Matplotlib backend: synth-style 2D overlay (Agg, no OpenGL).

Stacks the synth on the real image and outlines each cell with a single green
silhouette (lightness = synth brightness). One hue — not a rainbow. Prefers the
actual C++ synth volume (``<frame>_synth.tif``) for the fill; falls back to a
reproduced projection when it isn't available.

Headless/CI export path (works on display-less machines) and the alignment check.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import geometry, lineage  # noqa: E402
from .model import FrameData  # noqa: E402
from .scene import (  # noqa: E402  (shared colours / split logic)
    GHOST_COLOR,
    OUTLINE_DAUGHTER,
    OUTLINE_NORMAL,
    OUTLINE_TRASH,
)
from .synth import synth_projection_2d  # noqa: E402

# Black -> green ramp for the synth fill (single hue; intensity = brightness).
_GREEN = mcolors.LinearSegmentedColormap.from_list(
    "synthgreen", [(0, 0, 0), (0.15, 0.55, 0.25), (0.4, 1.0, 0.45)]
)


def render_frame_synth(
    curr: FrameData,
    prev: FrameData | None,
    raw,
    out_path: Path,
    frame: int,
    z_scale: float = 7.0,
    reveal: int | None = None,
    show_labels: bool = False,
    synth=None,
    synth_alpha: float = 0.55,
    dpi: int = 130,
):
    """Render one frame, synth-stacked-on-real, to a PNG."""
    if raw is not None:
        bg = raw.max(axis=0) if raw.ndim == 3 else raw
        h, w = bg.shape
    else:
        bg = None
        h, w = 512, 708

    ordered = curr.cells_in_fit_order()
    if reveal is not None:
        ordered = ordered[: max(0, reveal)]
    drawn = {c.name for c in ordered}

    # committed splits -> mark this frame's daughters orange, like the napari path
    splits = lineage.committed_splits(prev, curr) if prev is not None else []
    daughter_names = set()
    for _p, d1, d2 in splits:
        daughter_names.add(d1.name)
        daughter_names.add(d2.name)

    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=dpi)
    if bg is not None:
        ax.imshow(bg, cmap="gray", origin="upper", interpolation="nearest", zorder=0)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)

    # --- synth stacked on real (prefer the actual C++ synth volume) ---
    if synth is not None:
        fill = synth.max(axis=0).astype(np.float32) if synth.ndim == 3 else synth.astype(np.float32)
        fill = fill / (fill.max() or 1.0)
    else:
        fill = synth_projection_2d(ordered, (h, w))  # 0..~1 brightness
    rgba = _GREEN(np.clip(fill, 0, 1))
    rgba[..., 3] = np.where(fill > 0.02, synth_alpha, 0.0)
    ax.imshow(rgba, origin="upper", interpolation="nearest", zorder=1)

    # --- split parent ghosts: orange dashed silhouette of the PREV-frame parent,
    #     labelled "(f-1)" and linked to its two daughters ---
    for parent_cell, d1, d2 in splits:
        if d1.name not in drawn and d2.name not in drawn:
            continue
        sil = geometry.projected_silhouette_xy(
            parent_cell.center, parent_cell.radii, parent_cell.angles
        )
        ax.plot(sil[:, 0], sil[:, 1], color=GHOST_COLOR, lw=1.6, alpha=0.6,
                ls="--", zorder=2)
        for d in (d1, d2):
            ax.plot([parent_cell.x, d.x], [parent_cell.y, d.y],
                    color=GHOST_COLOR, lw=0.8, alpha=0.7, zorder=2)
        parts = lineage.split_name(parent_cell.name)
        ax.text(parent_cell.x, parent_cell.y,
                f"↳{parts[1]} (f-1)" if parts else f"{parent_cell.name}(f-1)",
                color=GHOST_COLOR, fontsize=5, ha="center", va="bottom",
                zorder=6, fontweight="bold")

    # --- cell outlines: white normal, ORANGE if a split daughter this frame ---
    for cell in ordered:
        sil = geometry.projected_silhouette_xy(cell.center, cell.radii, cell.angles)
        if cell.is_trash:
            col = OUTLINE_TRASH
        elif cell.name in daughter_names:
            col = OUTLINE_DAUGHTER
        else:
            col = OUTLINE_NORMAL
        lw = 1.4 if cell.name in daughter_names else 1.0
        ax.plot(sil[:, 0], sil[:, 1], color=col, lw=lw, alpha=0.95, zorder=3)
        if show_labels and not cell.is_trash:
            parts = lineage.split_name(cell.name)
            ax.text(cell.x, cell.y, parts[1] if parts else cell.name,
                    color="white", fontsize=4, ha="center", va="center", zorder=4)

    # --- CellLumen brightness centers ---
    if curr.lumen_centers:
        lx = [lc.x for lc in curr.lumen_centers]
        ly = [lc.y for lc in curr.lumen_centers]
        ax.scatter(lx, ly, s=10, facecolor="cyan", edgecolor="black",
                   linewidths=0.3, alpha=0.8, zorder=5)

    n_split = 0 if prev is None else len(lineage.committed_splits(prev, curr))
    ax.set_title(f"frame {frame}  cells={len(ordered)}  lumen={len(curr.lumen_centers)}  "
                 f"splits={n_split}", fontsize=8, color="white")
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor="black")
    plt.close(fig)
    return out_path
