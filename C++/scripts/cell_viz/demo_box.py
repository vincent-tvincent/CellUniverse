"""Volume-in-a-box demo renderer (matplotlib 3D, image voxels — no modeled spheres).

Reproduces the napari volume look: the real image is shown as dense, grainy
MIP **textures on the back walls** of a rectangular bounding box; the synth cells
are clusters of the actual synth voxels coloured by **viridis** (brightness),
revealed ONE BY ONE in fit order, with split cells coloured **magenta**. The
camera is a true 3D turntable (azimuth spin at a low oblique elevation), like
demo_3d — not top-down. The back walls update as the box rotates.

Geometry (voxel->cell, split detection) comes from a run cells.csv or a lineage
CSV; colour/brightness comes from the synth voxels.
"""

from __future__ import annotations

import math
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tifffile  # noqa: E402

from . import geometry, lineage, readers  # noqa: E402
from .demo3d import load_lineage_csv  # noqa: E402
from .model import FrameData  # noqa: E402

_VIRIDIS = matplotlib.colormaps["viridis"]
_MAGENTA = matplotlib.colors.LinearSegmentedColormap.from_list(
    "split_magenta", [(0.30, 0.0, 0.30), (0.85, 0.12, 0.70), (1.0, 0.5, 1.0)]
)


def _box_edges(ax, w, h, d, color=(0.5, 0.5, 0.5), lw=0.8):
    pts = np.array([[0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0],
                    [0, 0, d], [w, 0, d], [w, h, d], [0, h, d]], float)
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]:
        ax.plot(*zip(pts[a], pts[b]), color=color, lw=lw, alpha=0.45)


def _gray_window(img, lo_pct=1.0, hi_pct=99.6):
    lo, hi = np.percentile(img, [lo_pct, hi_pct])
    g = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    return plt.cm.gray(g)


def _build_faces(real_vol, target=84):
    """Windowed gray MIP textures + coordinate grids for each box face."""
    Z, H, W = real_vol.shape
    dz = max(1, Z // target); dy = max(1, H // target); dx = max(1, W // target)
    xy = _gray_window(real_vol.max(0))[::dy, ::dx]      # (H',W') for z-faces
    xz = _gray_window(real_vol.max(1))[::dz, ::dx]      # (Z',W') for y-faces
    yz = _gray_window(real_vol.max(2))[::dz, ::dy]      # (Z',H') for x-faces
    xs = np.linspace(0, W, xy.shape[1]); ys = np.linspace(0, H, xy.shape[0])
    xsz = np.linspace(0, W, xz.shape[1]); zsz = np.linspace(0, Z, xz.shape[0])
    ysx = np.linspace(0, H, yz.shape[1]); zsx = np.linspace(0, Z, yz.shape[0])
    return {
        "xy": (xy, np.meshgrid(xs, ys)),       # floor/ceiling
        "xz": (xz, np.meshgrid(xsz, zsz)),     # y-walls
        "yz": (yz, np.meshgrid(ysx, zsx)),     # x-walls
        "dims": (W, H, Z),
    }


def _draw_back_faces(ax, faces, azim, elev, alpha=0.55):
    """Draw the box walls that face away from the camera; return their artists."""
    W, H, Z = faces["dims"]
    az, el = math.radians(azim), math.radians(elev)
    artists = []

    def surf(Xg, Yg, Zg, fc):
        artists.append(ax.plot_surface(Xg, Yg, Zg, facecolors=fc, shade=False,
                                       rcount=fc.shape[0], ccount=fc.shape[1],
                                       alpha=alpha, zorder=0))

    # floor z=0 (always back when looking down)
    xy, (Xg, Yg) = faces["xy"]
    surf(Xg, Yg, np.zeros_like(Xg), xy)
    # y-wall: far side depends on azimuth
    xz, (Xg, Zg) = faces["xz"]
    yval = 0.0 if math.sin(az) > 0 else float(H)
    surf(Xg, np.full_like(Xg, yval), Zg, xz)
    # x-wall
    yz, (Yg, Zg) = faces["yz"]
    xval = 0.0 if math.cos(az) > 0 else float(W)
    surf(np.full_like(Yg, xval), Yg, Zg, yz)
    return artists


def _cell_voxels(synth_vol, cell, thr, stride=2):
    a, b, c = (max(cell.a_radius, 1.0), max(cell.b_radius, 1.0), max(cell.c_radius, 1.0))
    R = geometry.rotation_matrix(*cell.angles)
    Z, H, W = synth_vol.shape
    rr = max(a, b)
    z0, z1 = max(0, int(cell.z - c - 1)), min(Z, int(cell.z + c + 2))
    y0, y1 = max(0, int(cell.y - rr - 1)), min(H, int(cell.y + rr + 2))
    x0, x1 = max(0, int(cell.x - rr - 1)), min(W, int(cell.x + rr + 2))
    if z1 <= z0 or y1 <= y0 or x1 <= x0:
        return (np.empty(0),) * 4
    zz, yy, xx = np.mgrid[z0:z1:stride, y0:y1:stride, x0:x1:stride]
    dx, dy, dz = xx - cell.x, yy - cell.y, zz - cell.z
    lx = R[0, 0] * dx + R[1, 0] * dy + R[2, 0] * dz
    ly = R[0, 1] * dx + R[1, 1] * dy + R[2, 1] * dz
    lz = R[0, 2] * dx + R[1, 2] * dy + R[2, 2] * dz
    vals = synth_vol[zz, yy, xx]
    keep = ((lx / a) ** 2 + (ly / b) ** 2 + (lz / c) ** 2 <= 1.0) & (vals > thr)
    return xx[keep], yy[keep], zz[keep], vals[keep]


def render_demo_box(run_dir, frames, out_path, lineage_csv=None,
                    real_pattern="{n}_real.tif", synth_pattern="{n}_synth.tif",
                    fps=10, hold_frames=10, elev=14.0, azim_step=0.7,
                    point_size=7.0, wall_alpha=0.6):
    run_dir = Path(run_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lineage_frames = load_lineage_csv(Path(lineage_csv)) if lineage_csv else None

    def cells_of(n):
        return list(lineage_frames.get(n, [])) if lineage_frames is not None \
            else readers.load_frame(run_dir, n).cells

    def vol(pat, n):
        p = run_dir / pat.format(n=n)
        return tifffile.imread(str(p)) if p.exists() else None

    samp = [np.percentile(vol(synth_pattern, n), [80, 99.5]) for n in frames
            if vol(synth_pattern, n) is not None]
    thr = float(np.mean([a for a, _ in samp])) if samp else 30.0
    norm = plt.Normalize(vmin=thr, vmax=float(np.mean([b for _, b in samp])) if samp else 150.0)

    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8,
                                macro_block_size=1)
    azim = -75.0
    for n in frames:
        real_vol = vol(real_pattern, n)
        synth_vol = vol(synth_pattern, n)
        if synth_vol is None:
            continue
        Z, H, W = synth_vol.shape
        cells = cells_of(n)
        prev = cells_of(n - 1) if (n - 1) >= frames[0] else []
        splits = lineage.committed_splits(FrameData(n - 1, list(prev)),
                                          FrameData(n, list(cells))) if prev else []
        daughters = set()
        for _p, d1, d2 in splits:
            daughters.add(d1.name); daughters.add(d2.name)

        faces = _build_faces(real_vol) if real_vol is not None else None
        fig = plt.figure(figsize=(7.2, 4.9), dpi=100)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_position([-0.04, -0.06, 1.08, 1.04])
        fig.patch.set_facecolor("black"); ax.set_facecolor("black")
        _box_edges(ax, W, H, Z)
        ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_zlim(0, Z)
        ax.set_box_aspect((W, H, Z))
        ax.set_axis_off()
        wall_artists = []

        def capture(extra=""):
            nonlocal azim, wall_artists
            for a in wall_artists:
                a.remove()
            wall_artists = _draw_back_faces(ax, faces, azim, elev, wall_alpha) if faces else []
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f"frame {n}   {extra}", color="white", fontsize=10)
            fig.canvas.draw()
            writer.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
            azim += azim_step

        for k, cell in enumerate(cells):
            x, y, z, vals = _cell_voxels(synth_vol, cell, thr)
            cmap = _MAGENTA if cell.name in daughters else _VIRIDIS
            if x.size:
                ax.scatter(x, y, z, c=cmap(norm(vals)), s=point_size,
                           alpha=0.9, linewidths=0, depthshade=True)
            tag = "  (split)" if cell.name in daughters else ""
            capture(f"cell {k+1}/{len(cells)}{tag}")
        for _ in range(hold_frames):
            capture(f"complete ({len(cells)} cells, {len(splits)} splits)")
        plt.close(fig)
        print(f"[box] frame {n}: {len(cells)} cells, {len(splits)} splits", flush=True)

    writer.close()
    print(f"[box] wrote -> {out_path}", flush=True)
    return out_path
