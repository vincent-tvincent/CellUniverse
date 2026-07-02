"""napari-quality demo via NumPy/SciPy MIP volume raycasting (headless, no GL).

napari's volume 'mip' rendering is just a maximum-intensity projection along the
view rays. We reproduce it exactly: rotate the real + synth volumes to the camera
orientation and max-project. Result has the same soft volumetric look as napari.

Per frame: real shown as a grainy MIP slab; synth cells revealed ONE BY ONE in
fit order (viridis by brightness); split cells in magenta. Turntable camera. mp4.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import imageio.v2 as imageio
import matplotlib
import numpy as np
import scipy.ndimage as ndi

from . import geometry, lineage
from .demo3d import load_lineage_csv
from .model import FrameData

# Solid hues (no colormap): normal cells one colour, split cells another. The
# synth intensity drives only the alpha, so cells stay soft/volumetric but a
# single colour each.
NON_SPLIT_RGB = np.array([0.72, 0.85, 0.33])   # yellow-green
SPLIT_RGB = np.array([1.0, 0.32, 0.74])        # magenta


def _spin(t):  # turntable about the vertical (rotates the y,x floor plane)
    c, s = math.cos(t), math.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _tilt(t):  # tilt from top-down toward a front view (rotates z,y)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _rot(elev_deg, azim_deg):
    """Fixed front tilt (elev above horizontal) + animated turntable spin."""
    return _tilt(math.radians(90.0 - elev_deg)) @ _spin(math.radians(azim_deg))


def _fit(img, out_w, out_h):
    """Resize preserving aspect, centered on a black canvas of (out_h,out_w)."""
    h, w = img.shape[:2]
    sc = min(out_w / w, out_h / h)
    nw, nh = max(1, int(w * sc)), max(1, int(h * sc))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)
    canvas = np.zeros((out_h, out_w, 3), np.uint8)
    y0, x0 = (out_h - nh) // 2, (out_w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = r
    return canvas


def _mip(vol_padded, R, center):
    off = center - R @ np.array([center, center, center])
    return ndi.affine_transform(vol_padded, R, offset=off, order=1).max(axis=0)


def _pad_cube(v, side):
    pads = [((side - d) // 2, side - d - (side - d) // 2) for d in v.shape]
    offs = [p[0] for p in pads]
    return np.pad(v, pads), offs


def _cell_mask(shape, offs, scale, cell):
    a, b, c = (max(cell.a_radius, 1.0) * scale, max(cell.b_radius, 1.0) * scale,
               max(cell.c_radius, 1.0) * scale)
    R = geometry.rotation_matrix(*cell.angles)
    cz = cell.z * scale + offs[0]; cy = cell.y * scale + offs[1]; cx = cell.x * scale + offs[2]
    Z, Y, X = shape
    rr = max(a, b)
    z0, z1 = max(0, int(cz - c - 1)), min(Z, int(cz + c + 2))
    y0, y1 = max(0, int(cy - rr - 1)), min(Y, int(cy + rr + 2))
    x0, x1 = max(0, int(cx - rr - 1)), min(X, int(cx + rr + 2))
    out = np.zeros(shape, bool)
    if z1 <= z0 or y1 <= y0 or x1 <= x0:
        return out
    zz, yy, xx = np.mgrid[z0:z1, y0:y1, x0:x1]
    dx, dy, dz = xx - cx, yy - cy, zz - cz
    lx = R[0, 0] * dx + R[1, 0] * dy + R[2, 0] * dz
    ly = R[0, 1] * dx + R[1, 1] * dy + R[2, 1] * dz
    lz = R[0, 2] * dx + R[1, 2] * dy + R[2, 2] * dz
    out[z0:z1, y0:y1, x0:x1] = (lx / a) ** 2 + (ly / b) ** 2 + (lz / c) ** 2 <= 1.0
    return out


def render_demo_mip(run_dir, frames, out_path, lineage_csv,
                    fps=12, hold_frames=12, max_dim=340, elev=22.0,
                    out_w=960, out_h=640, crop_margin=0.10,
                    rot_amp=28.0, rot_period=90.0, real_gain=0.78, synth_gain=0.98):
    run_dir = Path(run_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lf = load_lineage_csv(Path(lineage_csv))

    import tifffile
    # contrast windows from the first synth/real
    r0 = tifffile.imread(str(run_dir / f"{frames[0]}_real.tif")).astype(np.float32)
    s0 = tifffile.imread(str(run_dir / f"{frames[0]}_synth.tif")).astype(np.float32)
    rlo, rhi = np.percentile(r0, [1.0, 99.6])
    sthr, smax = np.percentile(s0, [80.0, 99.5])
    scale = max_dim / max(s0.shape)
    side = int(max(d * scale for d in s0.shape)) + 6
    center = (side - 1) / 2.0

    # Stable crop box: union of the real slab's MIP extent over the spin range
    # (constant across frames -> no jitter, embryo fills the frame).
    realp0, _ = _pad_cube(ndi.zoom(r0, scale, order=1), side)
    bx0 = by0 = side; bx1 = by1 = 0
    for az in (-rot_amp, 0.0, rot_amp):
        rm = _mip(realp0, _rot(elev, az), center)
        ys, xs = np.where(rm > (rlo + 0.12 * (rhi - rlo)))
        if xs.size:
            bx0, bx1 = min(bx0, xs.min()), max(bx1, xs.max())
            by0, by1 = min(by0, ys.min()), max(by1, ys.max())
    if bx1 <= bx0:
        bx0, by0, bx1, by1 = 0, 0, side, side
    mx = int(crop_margin * (bx1 - bx0)); my = int(crop_margin * (by1 - by0))
    cx0, cx1 = max(0, bx0 - mx), min(side, bx1 + mx)
    cy0, cy1 = max(0, by0 - my), min(side, by1 + my)

    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8,
                                macro_block_size=1)
    cap_i = 0

    def colorize(real_mip, v_mip, m_mip):
        rg = np.clip((real_mip - rlo) / max(rhi - rlo, 1e-6), 0, 1)
        out = np.stack([rg] * 3, -1) * real_gain
        for mip_img, rgb in ((v_mip, NON_SPLIT_RGB), (m_mip, SPLIT_RGB)):
            sn = np.clip((mip_img - sthr) / max(smax - sthr, 1e-6), 0, 1)
            a = (sn ** 0.6)[..., None] * synth_gain   # soft edges, solid hue
            out = out * (1 - a) + rgb[None, None, :] * a
        img = np.clip(out * 255, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(img)

    for n in frames:
        real = tifffile.imread(str(run_dir / f"{n}_real.tif")).astype(np.float32)
        synth = tifffile.imread(str(run_dir / f"{n}_synth.tif")).astype(np.float32)
        realp, _ = _pad_cube(ndi.zoom(real, scale, order=1), side)
        synthp, offs = _pad_cube(ndi.zoom(synth, scale, order=1), side)
        cells = list(lf.get(n, []))
        prev = list(lf.get(n - 1, []))
        splits = lineage.committed_splits(FrameData(n - 1, prev), FrameData(n, cells)) if prev else []
        daughters = set()
        for _p, d1, d2 in splits:
            daughters.add(d1.name); daughters.add(d2.name)

        revealed_v = np.zeros_like(synthp)
        revealed_m = np.zeros_like(synthp)

        def capture(extra=""):
            nonlocal cap_i
            azim = rot_amp * math.sin(2.0 * math.pi * cap_i / rot_period)
            R = _rot(elev, azim)
            rm = _mip(realp, R, center)
            vm = _mip(revealed_v, R, center)
            mm = _mip(revealed_m, R, center)
            img = colorize(rm, vm, mm)[cy0:cy1, cx0:cx1]
            img = _fit(img, out_w, out_h)
            cv2.putText(img, f"frame {n}   {extra}", (22, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            writer.append_data(np.ascontiguousarray(img))
            cap_i += 1

        for k, cell in enumerate(cells):
            m = _cell_mask(synthp.shape, offs, scale, cell)
            if cell.name in daughters:
                revealed_m[m] = synthp[m]
            else:
                revealed_v[m] = synthp[m]
            tag = "  (split)" if cell.name in daughters else ""
            capture(f"cell {k + 1}/{len(cells)}{tag}")
        for _ in range(hold_frames):
            capture(f"complete ({len(cells)} cells, {len(splits)} splits)")
        print(f"[mip] frame {n}: {len(cells)} cells, {len(splits)} splits", flush=True)

    writer.close()
    print(f"[mip] wrote -> {out_path}", flush=True)
    return out_path
