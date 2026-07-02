"""Synthetic image rendering — the same kind of drawing the C++ synth produces.

The C++ renderer fills each ellipsoid's voxels with the cell's ``brightness``
(Ellipsoid::drawWithRotation: ``image.at(y,x) = _brightness``). A max-projection
of that volume equals the cell's projected silhouette ellipse filled with its
brightness, max-composited across cells. We reproduce that directly in 2D —
faithful and fast (no full voxel loop).
"""

from __future__ import annotations

import numpy as np

from . import geometry
from .model import Cell


def _silhouette_matrix(cell: Cell) -> np.ndarray:
    """2x2 quadratic form S of the cell's xy projection (shadow ellipse)."""
    M = geometry.quadratic_matrix(cell.radii, cell.angles)
    m_xy = M[:2, :2]
    m_z = M[:2, 2]
    return m_xy - np.outer(m_z, m_z) / M[2, 2]


def synth_projection_2d(cells, shape_yx, only=None) -> np.ndarray:
    """Max-projected synthetic image (float, cell brightness inside silhouettes).

    ``shape_yx`` = (H, W) of the raw frame. ``only`` optionally restricts to a
    set of cell names (for fit-order animation).
    """
    h, w = shape_yx
    img = np.zeros((h, w), dtype=np.float32)
    for cell in cells:
        if only is not None and cell.name not in only:
            continue
        s = _silhouette_matrix(cell)
        # bounding box: extent along x,y = sqrt(diag(S^-1))
        try:
            s_inv = np.linalg.inv(s)
        except np.linalg.LinAlgError:
            continue
        ex = float(np.sqrt(max(s_inv[0, 0], 0.0)))
        ey = float(np.sqrt(max(s_inv[1, 1], 0.0)))
        cx, cy = cell.x, cell.y
        x0, x1 = int(np.floor(cx - ex)), int(np.ceil(cx + ex))
        y0, y1 = int(np.floor(cy - ey)), int(np.ceil(cy + ey))
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, w - 1), min(y1, h - 1)
        if x1 < x0 or y1 < y0:
            continue
        ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
        dx = xs - cx
        dy = ys - cy
        # u^T S u <= 1  inside the silhouette
        q = (s[0, 0] * dx * dx + 2 * s[0, 1] * dx * dy + s[1, 1] * dy * dy)
        mask = q <= 1.0
        sub = img[y0:y1 + 1, x0:x1 + 1]
        np.maximum(sub, np.where(mask, cell.brightness, 0.0).astype(np.float32), out=sub)
    return img
