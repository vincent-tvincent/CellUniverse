"""Volume/MIP demonstration renderer (image-based, like napari) — no modeled spheres.

Renders the actual image data the way napari does: a bright max-intensity
projection of the real volume (grayscale) with the synth volume on top, colored by
**viridis** (brightness), translucent. Cells are revealed ONE BY ONE in fit order
by gating the synth with an accumulating per-cell mask; cells that split this frame
are colored with a distinct **magenta** ramp instead of viridis. A subtle oblique
parallax ('3d') or a flat top view ('2d') gives the camera; both can rotate.

Geometry (cell masks) comes from a run cells.csv or a lineage CSV. Brightness/colour
comes straight from the synth image, so it matches the napari synth-over-real view.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib
import numpy as np
import tifffile

from . import geometry, lineage
from .demo3d import load_lineage_csv  # reuse the lineage-CSV loader
from .model import FrameData
from . import readers

_VIRIDIS = matplotlib.colormaps["viridis"]
_MAGENTA = matplotlib.colors.LinearSegmentedColormap.from_list(
    "split_magenta", [(0.30, 0.0, 0.30), (0.85, 0.12, 0.70), (1.0, 0.5, 1.0)]
)

# parallax constants (from build_embryo_showcase_mp4.py)
_DEPTH_SHIFT = 90.0
_PITCH_SHIFT = 84.0


def _cmap_u8(cmap, u8):
    """Map a uint8 image through a matplotlib colormap to a BGR float image."""
    rgb = (cmap(u8.astype(np.float32) / 255.0)[..., :3] * 255.0).astype(np.float32)
    return rgb[..., ::-1]  # RGB -> BGR


def _window_u8(img, lo, hi):
    scaled = (np.clip(img, lo, hi) - lo) / max(hi - lo, 1e-6)
    return np.round(scaled * 255.0).astype(np.uint8)


def _parallax(yaw_deg, pitch_deg, shape):
    z, h, w = shape
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    sx = 1.0 - 0.18 * abs(math.sin(yaw))
    sy = 1.0 - 0.12 * abs(math.sin(pitch))
    return sx, sy, _DEPTH_SHIFT * math.sin(yaw), _PITCH_SHIFT * math.sin(pitch)


def _project_volume(vol, yaw_deg, pitch_deg, stride=2):
    if abs(yaw_deg) < 1e-6 and abs(pitch_deg) < 1e-6:
        return vol.max(axis=0).astype(np.float32)
    z, h, w = vol.shape
    sx, sy, shift, pshift = _parallax(yaw_deg, pitch_deg, vol.shape)
    cx, cy, cz = (w - 1) / 2.0, (h - 1) / 2.0, (z - 1) / 2.0
    acc = np.zeros((h, w), np.float32)
    idx = list(range(0, z, stride))
    if (z - 1) not in idx:
        idx.append(z - 1)
    for zi in idx:
        zn = (zi - cz) / max(cz, 1.0)
        M = np.float32([[sx, 0, cx - sx * cx + zn * shift],
                        [0, sy, cy - sy * cy + zn * pshift]])
        np.maximum(acc, cv2.warpAffine(vol[zi].astype(np.float32), M, (w, h)), out=acc)
    return acc


def _project_points(pts_xyz, yaw_deg, pitch_deg, shape):
    pts = np.asarray(pts_xyz, np.float32).reshape(-1, 3)
    if abs(yaw_deg) < 1e-6 and abs(pitch_deg) < 1e-6:
        return pts[:, :2].copy()
    z, h, w = shape
    sx, sy, shift, pshift = _parallax(yaw_deg, pitch_deg, shape)
    cx, cy, cz = (w - 1) / 2.0, (h - 1) / 2.0, (z - 1) / 2.0
    zn = (pts[:, 2] - cz) / max(cz, 1.0)
    out = np.empty((len(pts), 2), np.float32)
    out[:, 0] = cx + (pts[:, 0] - cx) * sx + zn * shift
    out[:, 1] = cy + (pts[:, 1] - cy) * sy + zn * pshift
    return out


def _cell_surface_points(cell, n=12):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    xs = np.outer(np.cos(u), np.sin(v)).ravel()
    ys = np.outer(np.sin(u), np.sin(v)).ravel()
    zs = np.outer(np.ones_like(u), np.cos(v)).ravel()
    a, b, c = cell.radii
    local = np.vstack([xs * a, ys * b, zs * c])
    world = geometry.rotation_matrix(*cell.angles) @ local
    return np.column_stack([world[0] + cell.x, world[1] + cell.y, world[2] + cell.z])


def render_demo_volume(run_dir, frames, out_path, lineage_csv=None, view="3d",
                       real_pattern="{n}_real.tif", synth_pattern="{n}_synth.tif",
                       fps=10, opacity=0.92, hold_frames=10, rotate_degrees=32.0,
                       rotate_period=80.0, stride=2):
    run_dir = Path(run_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lineage_frames = load_lineage_csv(Path(lineage_csv)) if lineage_csv else None

    def cells_of(n):
        if lineage_frames is not None:
            return list(lineage_frames.get(n, []))
        return readers.load_frame(run_dir, n).cells

    def vol(pattern, n):
        p = run_dir / pattern.format(n=n)
        return tifffile.imread(str(p)).astype(np.float32) if p.exists() else None

    # global windows so brightness/contrast is stable across the clip
    reals, synths = [], []
    for n in frames:
        r = vol(real_pattern, n); s = vol(synth_pattern, n)
        if r is not None:
            reals.append(np.max(r, axis=0))
        if s is not None:
            synths.append(np.max(s, axis=0))
    rlo, rhi = (np.percentile(np.concatenate([x.ravel()[::8] for x in reals]), [1.0, 99.8])
                if reals else (0.0, 255.0))
    slo, shi = (np.percentile(np.concatenate([x.ravel()[::8] for x in synths]), [50.0, 99.8])
                if synths else (0.0, 255.0))

    base_pitch = 22.0 if view == "3d" else 0.0
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8,
                                macro_block_size=1)
    n0 = frames[0]
    cap_i = 0  # global capture index -> continuous camera rotation across the clip

    def camera():
        """(yaw, pitch, roll) for the current capture. 3D = turntable yaw;
        2D = in-plane spin (roll) of the flat top view."""
        phase = rotate_degrees * math.sin(2.0 * math.pi * cap_i / max(1.0, rotate_period))
        if view == "3d":
            return phase, base_pitch, 0.0
        return 0.0, 0.0, phase

    for n in frames:
        real_vol = vol(real_pattern, n)
        synth_vol = vol(synth_pattern, n)
        if real_vol is None or synth_vol is None:
            continue
        shape = real_vol.shape
        h, w = shape[1], shape[2]
        cells = cells_of(n)
        prev_cells = cells_of(n - 1) if (n - 1) >= n0 else []
        splits = lineage.committed_splits(FrameData(n - 1, list(prev_cells)),
                                          FrameData(n, list(cells))) if prev_cells else []
        daughters = set()
        for _p, d1, d2 in splits:
            daughters.add(d1.name); daughters.add(d2.name)

        def capture(revealed, extra=""):
            nonlocal cap_i
            yaw, pitch, roll = camera()
            real_u8 = _window_u8(_project_volume(real_vol, yaw, pitch, stride), rlo, rhi)
            synth_u8 = _window_u8(_project_volume(synth_vol, yaw, pitch, stride), slo, shi)
            base = cv2.cvtColor(real_u8, cv2.COLOR_GRAY2BGR).astype(np.float32)
            synth_v = _cmap_u8(_VIRIDIS, synth_u8)
            synth_m = _cmap_u8(_MAGENTA, synth_u8)
            # solid-ish synth: floor so even mid-intensity stays clearly coloured
            synth_disp = np.clip(0.35 + synth_u8.astype(np.float32) / 255.0, 0.0, 1.0)
            mask_n = np.zeros((h, w), np.uint8)
            mask_s = np.zeros((h, w), np.uint8)
            for c in revealed:
                pts = _project_points(_cell_surface_points(c), yaw, pitch, shape)
                hull = cv2.convexHull(np.round(pts).astype(np.int32))
                cv2.fillConvexPoly(mask_s if c.name in daughters else mask_n, hull, 1)
            for mask, colored in ((mask_n, synth_v), (mask_s, synth_m)):
                a = (opacity * synth_disp * mask.astype(np.float32))[:, :, None]
                base = base * (1.0 - a) + colored * a
            img = np.ascontiguousarray(np.clip(base, 0, 255).astype(np.uint8)[:, :, ::-1])
            if abs(roll) > 1e-6:
                M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), roll, 1.0)
                img = cv2.warpAffine(img, M, (w, h))
            cv2.putText(img, f"frame {n}   {extra}", (16, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            writer.append_data(img)
            cap_i += 1

        revealed = []
        for k, cell in enumerate(cells):
            revealed.append(cell)
            tag = "  (split)" if cell.name in daughters else ""
            capture(revealed, f"cell {k+1}/{len(cells)}{tag}")
        for _ in range(hold_frames):
            capture(revealed, f"complete ({len(cells)} cells, {len(splits)} splits)")
        print(f"[volume:{view}] frame {n}: {len(cells)} cells, {len(splits)} splits", flush=True)

    writer.close()
    print(f"[volume:{view}] wrote -> {out_path}", flush=True)
    return out_path
