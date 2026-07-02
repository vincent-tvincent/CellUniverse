"""Demonstration 3D / 2D video renderer (matplotlib, headless).

Per frame: the real image is a faint co-located point cloud; cells are filled
ellipsoid masks coloured by their synth brightness under **viridis** (split cells
this frame use a different colormap). Cells appear ONE BY ONE in fit order (each
frame starts blank, then accumulates). A gentle turntable rotation conveys depth.
``view='2d'`` looks straight down (top-down, same rotation). Output is an mp4.

Cell geometry comes from either a run (cells.csv + checkpoints) or a lineage CSV
(file,name,x,y,z,majorRadius,bRadius,minorRadius,theta_*). Brightness for the
colormap is sampled from <frame>_synth.tif at each cell centre.
"""

from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tifffile  # noqa: E402

from . import geometry, lineage, readers  # noqa: E402
from .model import Cell, FrameData  # noqa: E402

# Solid colours (no colormap): normal cells one colour, split cells another.
NON_SPLIT_COLOR = (0.72, 0.85, 0.33, 0.95)   # yellow-green
SPLIT_COLOR = (1.0, 0.32, 0.74, 0.95)        # magenta/pink
_FRAME_RE = re.compile(r"(\d+)")


# ---------- lineage-CSV source ----------

def load_lineage_csv(path: Path) -> dict[int, list[Cell]]:
    """Group a FinalLineageTree-style CSV into per-frame Cell lists (no brightness)."""
    by_frame: dict[int, list[Cell]] = defaultdict(list)
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            m = _FRAME_RE.search(row.get("file", ""))
            if not m:
                continue
            f = int(m.group(1))

            def num(k, d=0.0):
                try:
                    return float(row[k])
                except (KeyError, TypeError, ValueError):
                    return d
            by_frame[f].append(Cell(
                name=row["name"],
                x=num("x"), y=num("y"), z=num("z"),
                a_radius=num("majorRadius", num("aRadius", 10.0)),
                b_radius=num("bRadius", 10.0),
                c_radius=num("minorRadius", num("cRadius", 10.0)),
                theta_x=num("theta_x"), theta_y=num("theta_y"), theta_z=num("theta_z"),
                is_trash=False,
            ))
    return dict(by_frame)


def _sample_brightness(vol, cell) -> float:
    z, y, x = int(round(cell.z)), int(round(cell.y)), int(round(cell.x))
    zz = slice(max(0, z - 1), z + 2)
    yy = slice(max(0, y - 2), y + 3)
    xx = slice(max(0, x - 2), x + 3)
    patch = vol[zz, yy, xx]
    return float(patch.max()) if patch.size else 0.0


# ---------- geometry ----------

def _ellipsoid_surface(center, radii, angles, n=12):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    xs = np.outer(np.cos(u), np.sin(v)).ravel()
    ys = np.outer(np.sin(u), np.sin(v)).ravel()
    zs = np.outer(np.ones_like(u), np.cos(v)).ravel()
    a, b, c = radii
    local = np.vstack([xs * a, ys * b, zs * c])
    world = geometry.rotation_matrix(*angles) @ local
    cx, cy, cz = center
    return ((world[0] + cx).reshape(n, n),
            (world[1] + cy).reshape(n, n),
            (world[2] + cz).reshape(n, n))


def _gray_window(img, gamma=0.7, lo_pct=1.0, hi_pct=99.6):
    lo, hi = np.percentile(img, [lo_pct, hi_pct])
    g = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1) ** gamma
    return plt.cm.gray(g)


def _wall_projections(vol, xlim, ylim, zlim, target=96):
    """Crisp MIP textures of the real cropped to the cell bbox, for floor + walls."""
    Z, H, W = vol.shape
    ix0, ix1 = max(0, int(xlim[0])), min(W, int(math.ceil(xlim[1])))
    iy0, iy1 = max(0, int(ylim[0])), min(H, int(math.ceil(ylim[1])))
    iz0, iz1 = max(0, int(zlim[0])), min(Z, int(math.ceil(zlim[1])))
    if ix1 <= ix0 or iy1 <= iy0 or iz1 <= iz0:
        return None
    crop = vol[iz0:iz1, iy0:iy1, ix0:ix1]
    dz = max(1, crop.shape[0] // target); dy = max(1, crop.shape[1] // target); dx = max(1, crop.shape[2] // target)
    xy = _gray_window(crop.max(0))[::dy, ::dx]
    xz = _gray_window(crop.max(1))[::dz, ::dx]
    yz = _gray_window(crop.max(2))[::dz, ::dy]
    xs = np.linspace(ix0, ix1, xy.shape[1]); ys = np.linspace(iy0, iy1, xy.shape[0])
    xs2 = np.linspace(ix0, ix1, xz.shape[1]); zs2 = np.linspace(iz0, iz1, xz.shape[0])
    ys3 = np.linspace(iy0, iy1, yz.shape[1]); zs3 = np.linspace(iz0, iz1, yz.shape[0])
    return {"xy": (xy, np.meshgrid(xs, ys)),
            "xz": (xz, np.meshgrid(xs2, zs2)),
            "yz": (yz, np.meshgrid(ys3, zs3)),
            "lims": (xlim, ylim, zlim)}


def _draw_walls(ax, projs, azim, alpha=0.7):
    """Floor + the two camera-facing-away walls (crisp real nuclei). Returns artists."""
    if projs is None:
        return []
    (xlim, ylim, zlim) = projs["lims"]
    az = math.radians(azim)
    arts = []

    def surf(Xg, Yg, Zg, fc):
        arts.append(ax.plot_surface(Xg, Yg, Zg, facecolors=fc, shade=False,
                                    rcount=fc.shape[0], ccount=fc.shape[1],
                                    alpha=alpha, zorder=0))
    xy, (Xg, Yg) = projs["xy"]
    surf(Xg, Yg, np.full_like(Xg, zlim[0]), xy)                     # floor
    xz, (Xg, Zg) = projs["xz"]
    yval = ylim[0] if math.sin(az) > 0 else ylim[1]
    surf(Xg, np.full_like(Xg, yval), Zg, xz)                        # y-wall
    yz, (Yg, Zg) = projs["yz"]
    xval = xlim[0] if math.cos(az) > 0 else xlim[1]
    surf(np.full_like(Yg, xval), Yg, Zg, yz)                        # x-wall
    return arts


# ---------- main ----------

def render_demo(run_dir, frames, out_path, lineage_csv=None, view="3d",
                real_pattern="{n}_real.tif", synth_pattern="{n}_synth.tif",
                fps=12, dpi=110, figsize=(7.2, 5.4), ellipsoid_n=12,
                hold_frames=6, zoom_pad=0.32):
    run_dir = Path(run_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lineage_frames = load_lineage_csv(Path(lineage_csv)) if lineage_csv else None

    def cells_of(n):
        if lineage_frames is not None:
            return list(lineage_frames.get(n, []))
        fd = readers.load_frame(run_dir, n)
        return fd.cells

    # ---- prepass: global bbox (stable zoom) ----
    bx0 = by0 = bz0 = 1e9
    bx1 = by1 = bz1 = -1e9
    frame_cache = {}
    for n in frames:
        cells = [c for c in cells_of(n) if not c.is_trash]
        frame_cache[n] = cells
        for c in cells:
            r = max(c.a_radius, c.b_radius, c.c_radius)
            bx0, bx1 = min(bx0, c.x - r), max(bx1, c.x + r)
            by0, by1 = min(by0, c.y - r), max(by1, c.y + r)
            bz0, bz1 = min(bz0, c.z - r), max(bz1, c.z + r)
    if bx1 < bx0:  # no cells
        bx0, bx1, by0, by1, bz0, bz1 = 0, 708, 0, 512, 0, 239
    padx = zoom_pad * (bx1 - bx0); pady = zoom_pad * (by1 - by0); padz = zoom_pad * (bz1 - bz0)
    xlim = (bx0 - padx, bx1 + padx)
    ylim = (by0 - pady, by1 + pady)
    zlim = (bz0 - padz, bz1 + padz)
    elev = 22.0 if view == "3d" else 88.0

    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8,
                                macro_block_size=1)
    azim = -70.0
    for n in frames:
        cells = frame_cache[n]
        prev_cells = frame_cache.get(n - 1)
        if prev_cells is None and (n - 1) in frames:
            prev_cells = [c for c in cells_of(n - 1) if not c.is_trash]
        real_path = run_dir / real_pattern.format(n=n)
        real_vol = tifffile.imread(str(real_path)) if real_path.exists() else None

        splits = []
        if prev_cells:
            splits = lineage.committed_splits(FrameData(n - 1, list(prev_cells)),
                                              FrameData(n, list(cells)))
        daughters = set()
        for _p, d1, d2 in splits:
            daughters.add(d1.name); daughters.add(d2.name)

        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")
        # Honor manual zorder so cells (zorder 2) always draw on top of the wall
        # textures (zorder 0); matplotlib's auto depth-sort otherwise lets the big
        # wall surfaces occlude the cells.
        ax.computed_zorder = False
        # Fill the canvas (matplotlib 3D leaves big margins by default) so the
        # zoomed cell cloud is large; leave a sliver at top for the title.
        ax.set_position([-0.02, -0.04, 1.04, 0.97])
        ax.set_facecolor("black"); fig.patch.set_facecolor("black")
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
        ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]))
        ax.set_axis_off()
        projs = _wall_projections(real_vol, xlim, ylim, zlim) if real_vol is not None else None
        wall_artists = []

        def capture(extra=""):
            nonlocal wall_artists
            for a in wall_artists:
                a.remove()
            wall_artists = _draw_walls(ax, projs, azim)
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(f"frame {n}   {extra}", color="white", fontsize=10)
            fig.canvas.draw()
            writer.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])

        for k, cell in enumerate(cells):
            color = SPLIT_COLOR if cell.name in daughters else NON_SPLIT_COLOR
            X, Y, Z = _ellipsoid_surface(cell.center, cell.radii, cell.angles, ellipsoid_n)
            ax.plot_surface(X, Y, Z, color=color, alpha=0.95, linewidth=0,
                            antialiased=True, shade=True, zorder=2)
            azim += 0.25
            tag = "  (split)" if cell.name in daughters else ""
            capture(f"cell {k+1}/{len(cells)}{tag}")

        for _ in range(hold_frames):
            azim += 0.25
            capture(f"complete ({len(cells)} cells, {len(splits)} splits)")
        plt.close(fig)
        print(f"[demo3d:{view}] frame {n}: {len(cells)} cells, {len(splits)} splits", flush=True)

    writer.close()
    print(f"[demo3d:{view}] wrote -> {out_path}", flush=True)
    return out_path
