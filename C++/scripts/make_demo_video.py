#!/usr/bin/env python3
"""Render the demonstration 3D draw-on video from a completed run.

    .venv-viz/bin/python scripts/make_demo_video.py <run_dir> \
        --first 85 --last 86 --out <run_dir>/viz/demo3d.mp4

Needs a run with cells.csv + candidate_graph/ + checkpoints/ + <frame>_real.tif.
See scripts/cell_viz/demo3d.py.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("XDG_CACHE_HOME", "/tmp/celluniverse_napari_cache")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cell_viz.demo3d import render_demo  # noqa: E402
from cell_viz.demo_volume import render_demo_volume  # noqa: E402
from cell_viz.demo_box import render_demo_box  # noqa: E402
from cell_viz.demo_mip import render_demo_mip  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", type=Path, help="dir with <frame>_real.tif / _synth.tif")
    p.add_argument("--first", type=int, required=True)
    p.add_argument("--last", type=int, required=True)
    p.add_argument("--lineage-csv", type=Path, default=None,
                   help="per-cell positions/lineage CSV (else read run cells.csv)")
    p.add_argument("--view", choices=["3d", "2d", "both"], default="3d")
    p.add_argument("--engine", choices=["mip", "box", "volume", "ellipsoid"], default="mip",
                   help="'mip' = napari-quality MIP volume raycast (recommended); "
                        "'box' = image voxels in a box; 'volume' = flat MIP parallax; "
                        "'ellipsoid' = modeled spheres")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--opacity", type=float, default=0.75)
    p.add_argument("--ellipsoid-n", type=int, default=12)
    args = p.parse_args(argv)

    frames = list(range(args.first, args.last + 1))

    if args.engine == "mip":
        out = args.out or (args.run_dir / "viz" / f"demo_mip_f{args.first}-f{args.last}.mp4")
        render_demo_mip(args.run_dir, frames, out, args.lineage_csv, fps=args.fps)
        return

    if args.engine == "box":
        # single oblique turntable view (no top-down)
        out = args.out or (args.run_dir / "viz" / f"demo_box_f{args.first}-f{args.last}.mp4")
        render_demo_box(args.run_dir, frames, out, lineage_csv=args.lineage_csv, fps=args.fps)
        return

    views = ["3d", "2d"] if args.view == "both" else [args.view]
    for v in views:
        out = args.out if (args.out and len(views) == 1) else \
            (args.run_dir / "viz" / f"demo_{args.engine}_{v}_f{args.first}-f{args.last}.mp4")
        if args.engine == "volume":
            render_demo_volume(args.run_dir, frames, out, lineage_csv=args.lineage_csv,
                               view=v, fps=args.fps, opacity=args.opacity)
        else:
            render_demo(args.run_dir, frames, out, lineage_csv=args.lineage_csv,
                        view=v, fps=args.fps, ellipsoid_n=args.ellipsoid_n)


if __name__ == "__main__":
    main()
