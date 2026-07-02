"""Ellipsoid → wireframe geometry, in napari ``(z, y, x)`` coordinates.

The tracker stores cells in ``(x, y, z)`` with a forward rotation
``R = Rz(theta_z) @ Ry(theta_y) @ Rx(theta_x)`` and semi-axes
``aRadius→local x, bRadius→local y, cRadius→local z`` (see Ellipsoid.cpp).

We render a recognizable wireframe: the three principal-plane ellipse rings
(xy, yz, xz) plus the three semi-axis line segments. This reads well as an
"outline with auxiliary lines" in both napari 2D and 3D display modes.

All returned point arrays are ``(N, 3)`` float arrays in napari order
``(z_index, y, x)`` where ``z_index = tracker_z / z_scale``.
"""

from __future__ import annotations

import numpy as np


def rotation_matrix(theta_x: float, theta_y: float, theta_z: float) -> np.ndarray:
    """Forward rotation R = Rz @ Ry @ Rx (tracker x,y,z space)."""
    cx, sx = np.cos(theta_x), np.sin(theta_x)
    cy, sy = np.cos(theta_y), np.sin(theta_y)
    cz, sz = np.cos(theta_z), np.sin(theta_z)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def quadratic_matrix(radii, angles) -> np.ndarray:
    """3x3 matrix M (tracker x,y,z) with surface {u: (u)^T M (u) = 1}, u=p-center."""
    a, b, c = radii
    R = rotation_matrix(*angles)
    d = np.diag([1.0 / max(a, 1e-6) ** 2, 1.0 / max(b, 1e-6) ** 2, 1.0 / max(c, 1e-6) ** 2])
    return R @ d @ R.T


def projected_silhouette_xy(center, radii, angles, n: int = 64):
    """Orthographic-projection (along z) silhouette ellipse, as ``(N,2)`` (x,y).

    The shadow boundary is the Schur complement of M on the z row/col:
    S = M_xy - m_z m_z^T / M_zz, with {u: u^T S u = 1}.
    """
    M = quadratic_matrix(radii, angles)
    m_xy = M[:2, :2]
    m_z = M[:2, 2]
    s = m_xy - np.outer(m_z, m_z) / M[2, 2]
    # parametrize the ellipse u^T S u = 1 via eigendecomposition of S
    vals, vecs = np.linalg.eigh(s)
    vals = np.clip(vals, 1e-9, None)
    t = np.linspace(0.0, 2.0 * np.pi, n)
    unit = np.column_stack([np.cos(t), np.sin(t)])
    axes = vecs @ np.diag(1.0 / np.sqrt(vals))
    pts = (axes @ unit.T).T  # (N,2) offsets in x,y
    return pts + np.array([center[0], center[1]])


def silhouette_path_napari(center, radii, angles, z_scale: float, n: int = 64) -> np.ndarray:
    """Projected silhouette as a napari ``(N,3)`` path at the cell's center z-plane."""
    xy = projected_silhouette_xy(center, radii, angles, n)
    z_index = center[2] / z_scale
    return np.column_stack([np.full(len(xy), z_index), xy[:, 1], xy[:, 0]])


def _to_napari(points_xyz: np.ndarray, z_scale: float) -> np.ndarray:
    """Convert tracker ``(x, y, z)`` rows to napari ``(z/z_scale, y, x)`` rows."""
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    z = points_xyz[:, 2] / z_scale
    return np.column_stack([z, y, x])


def center_napari(center_xyz, z_scale: float) -> np.ndarray:
    cx, cy, cz = center_xyz
    return np.array([cz / z_scale, cy, cx])


def _ring(center, R, radii, plane: str, n: int) -> np.ndarray:
    """One principal-plane ellipse ring as tracker-space ``(N,3)`` points."""
    a, b, c = radii
    t = np.linspace(0.0, 2.0 * np.pi, n)
    zeros = np.zeros_like(t)
    if plane == "xy":
        local = np.column_stack([a * np.cos(t), b * np.sin(t), zeros])
    elif plane == "yz":
        local = np.column_stack([zeros, b * np.cos(t), c * np.sin(t)])
    else:  # "xz"
        local = np.column_stack([a * np.cos(t), zeros, c * np.sin(t)])
    world = (R @ local.T).T + np.asarray(center, dtype=float)
    return world


def _axis(center, R, radii, axis: str) -> np.ndarray:
    """A full principal semi-axis line segment (−r .. +r) in tracker space."""
    a, b, c = radii
    if axis == "a":
        v = np.array([a, 0.0, 0.0])
    elif axis == "b":
        v = np.array([0.0, b, 0.0])
    else:
        v = np.array([0.0, 0.0, c])
    p0 = (R @ (-v)) + np.asarray(center, dtype=float)
    p1 = (R @ v) + np.asarray(center, dtype=float)
    return np.vstack([p0, p1])


def ellipsoid_wireframe(
    center,
    radii,
    angles,
    z_scale: float,
    ring_points: int = 48,
    rings=("xy", "yz", "xz"),
    axes=("a", "b", "c"),
) -> dict[str, list[np.ndarray]]:
    """Return wireframe polylines for one ellipsoid in napari coords.

    Output: ``{"rings": [ (N,3), ... ], "axes": [ (2,3), ... ]}`` — each entry is
    a path suitable for a napari Shapes layer (type ``path``).
    """
    tx, ty, tz = angles
    R = rotation_matrix(tx, ty, tz)
    out_rings = [
        _to_napari(_ring(center, R, radii, plane, ring_points), z_scale)
        for plane in rings
    ]
    out_axes = [_to_napari(_axis(center, R, radii, ax), z_scale) for ax in axes]
    return {"rings": out_rings, "axes": out_axes}
