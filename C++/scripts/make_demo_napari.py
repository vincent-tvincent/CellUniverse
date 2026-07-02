#!/usr/bin/env python3
"""Render the demo as a TRUE napari volume animation (run on a machine with a GPU/display).

This is the only way to get napari's exact volume look: real + synth rendered as
3D volumes (synth in viridis, split cells in magenta), revealed cell-by-cell in
fit order, with a rotating camera, exported to mp4. matplotlib cannot reproduce
napari's GPU ray-cast volume rendering, which is why the earlier mpl demos looked
like flat disks.

Run on your Mac (napari + a display):

    .venv-viz/bin/python scripts/make_demo_napari.py \
        outputs/Yiding_1~171_VISUAL_TIF \
        --lineage-csv outputs/Yiding_1~171_VISUAL_TIF/Yiding_Embryo_1~171_FinalLineageTree.csv \
        --first 1 --last 10 \
        --out outputs/Yiding_1~171_VISUAL_TIF/viz/demo_napari_f1-f10.mp4

Add --show to watch it build live in a napari window.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("XDG_CACHE_HOME", "/tmp/celluniverse_napari_cache")
os.environ.setdefault("XDG_CONFIG_HOME", "/tmp/celluniverse_napari_config")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import tifffile  # noqa: E402

from cell_viz import geometry, lineage  # noqa: E402
from cell_viz.demo3d import load_lineage_csv  # noqa: E402
from cell_viz.model import FrameData  # noqa: E402


def cell_mask(shape, cell, stride=1):
    """Boolean voxel mask of one cell's ellipsoid (z,y,x)."""
    a, b, c = (max(cell.a_radius, 1.0), max(cell.b_radius, 1.0), max(cell.c_radius, 1.0))
    R = geometry.rotation_matrix(*cell.angles)
    Z, H, W = shape
    rr = max(a, b)
    z0, z1 = max(0, int(cell.z - c - 1)), min(Z, int(cell.z + c + 2))
    y0, y1 = max(0, int(cell.y - rr - 1)), min(H, int(cell.y + rr + 2))
    x0, x1 = max(0, int(cell.x - rr - 1)), min(W, int(cell.x + rr + 2))
    out = np.zeros(shape, bool)
    if z1 <= z0 or y1 <= y0 or x1 <= x0:
        return out
    zz, yy, xx = np.mgrid[z0:z1, y0:y1, x0:x1]
    dx, dy, dz = xx - cell.x, yy - cell.y, zz - cell.z
    lx = R[0, 0] * dx + R[1, 0] * dy + R[2, 0] * dz
    ly = R[0, 1] * dx + R[1, 1] * dy + R[2, 1] * dz
    lz = R[0, 2] * dx + R[1, 2] * dy + R[2, 2] * dz
    out[z0:z1, y0:y1, x0:x1] = (lx / a) ** 2 + (ly / b) ** 2 + (lz / c) ** 2 <= 1.0
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--lineage-csv", type=Path, required=True)
    p.add_argument("--first", type=int, required=True)
    p.add_argument("--last", type=int, required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--hold", type=int, default=10, help="extra frames after each frame completes")
    p.add_argument("--deg-per-capture", type=float, default=1.2, help="camera spin per captured frame")
    p.add_argument("--synth-opacity", type=float, default=0.9)
    p.add_argument("--real-opacity", type=float, default=0.5)
    p.add_argument("--show", action="store_true", help="open an interactive window too")
    args = p.parse_args(argv)

    import imageio.v2 as imageio
    import napari

    run_dir = args.run_dir
    out = args.out or (run_dir / "viz" / f"demo_napari_f{args.first}-f{args.last}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)
    frames = list(range(args.first, args.last + 1))
    lf = load_lineage_csv(args.lineage_csv)

    # brightness window for the synth contrast limits
    sv0 = tifffile.imread(str(run_dir / f"{frames[0]}_synth.tif"))
    thr = float(np.percentile(sv0, 80))
    smax = float(np.percentile(sv0, 99.5))

    viewer = napari.Viewer(ndisplay=3, show=args.show)
    viewer.theme = "dark"
    viewer.text_overlay.visible = True
    viewer.text_overlay.color = "white"
    viewer.text_overlay.font_size = 14

    real_layer = synth_layer = split_layer = None
    writer = imageio.get_writer(out, fps=args.fps, codec="libx264", quality=8,
                                macro_block_size=1)
    angle = -60.0

    def capture(text):
        nonlocal angle
        viewer.text_overlay.text = text
        # napari 3D camera.angles = (rx, ry, rz) Euler degrees; spin about vertical
        viewer.camera.angles = (0.0, angle, 90.0)
        img = viewer.screenshot(canvas_only=True, flash=False)
        writer.append_data(img[..., :3])
        angle += args.deg_per_capture

    for n in frames:
        real = tifffile.imread(str(run_dir / f"{n}_real.tif"))
        synth = tifffile.imread(str(run_dir / f"{n}_synth.tif"))
        shape = synth.shape
        cells = list(lf.get(n, []))
        prev = list(lf.get(n - 1, []))
        splits = lineage.committed_splits(FrameData(n - 1, prev), FrameData(n, cells)) if prev else []
        daughters = set()
        for _p, d1, d2 in splits:
            daughters.add(d1.name); daughters.add(d2.name)

        revealed_v = np.zeros_like(synth)   # viridis (normal cells)
        revealed_m = np.zeros_like(synth)   # magenta (split cells)

        # real volume (grainy box) — mip like napari
        if real_layer is None:
            real_layer = viewer.add_image(real, name="real", colormap="gray",
                                          rendering="mip", blending="additive",
                                          opacity=args.real_opacity)
            synth_layer = viewer.add_image(revealed_v, name="synth", colormap="viridis",
                                           rendering="mip", blending="additive",
                                           opacity=args.synth_opacity,
                                           contrast_limits=[thr, smax])
            split_layer = viewer.add_image(revealed_m, name="split", colormap="magenta",
                                           rendering="mip", blending="additive",
                                           opacity=args.synth_opacity,
                                           contrast_limits=[thr, smax])
            viewer.reset_view()
        else:
            real_layer.data = real

        for k, cell in enumerate(cells):
            m = cell_mask(shape, cell)
            if cell.name in daughters:
                revealed_m[m] = synth[m]
                split_layer.data = revealed_m
            else:
                revealed_v[m] = synth[m]
                synth_layer.data = revealed_v
            tag = "  (split)" if cell.name in daughters else ""
            capture(f"frame {n}   cell {k + 1}/{len(cells)}{tag}")
        for _ in range(args.hold):
            capture(f"frame {n}   complete ({len(cells)} cells, {len(splits)} splits)")
        print(f"[napari] frame {n}: {len(cells)} cells, {len(splits)} splits", flush=True)

    writer.close()
    print(f"[napari] wrote -> {out}", flush=True)
    if args.show:
        napari.run()


if __name__ == "__main__":
    main()
