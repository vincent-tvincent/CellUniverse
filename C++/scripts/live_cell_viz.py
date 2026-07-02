#!/usr/bin/env python3
"""Live Napari cell-tracking visualizer for CellUniverse.

Overlays the tracker's per-frame decisions (final ellipsoid outlines + auxiliary
axes, CellLumen brightness centers, split parent ghosts) on the raw 3D image,
live as a run progresses or in replay. Read-only: never touches the run.

Examples
--------
Live tail an in-progress run (interactive window):

    .venv-viz/bin/python scripts/live_cell_viz.py outputs/output_xxx \\
        --input-pattern data/input/embryo_data/t%03d.tif \\
        --first-frame 85 --last-frame 120

Headless replay of a finished run -> mp4 (no window needed):

    QT_QPA_PLATFORM=offscreen .venv-viz/bin/python scripts/live_cell_viz.py \\
        outputs/output_xxx --input-pattern data/input/embryo_data/t%03d.tif \\
        --headless --record settled

See docs/plans/2026-06-02-live-napari-cell-viz-design.md.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# napari writes theme/icon assets via XDG cache; keep usable on read-only homes.
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/celluniverse_napari_cache")
os.environ.setdefault("XDG_CONFIG_HOME", "/tmp/celluniverse_napari_config")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cell_viz.app import VizApp  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("output_dir", type=Path, help="CellUniverse run output dir")
    p.add_argument("--input-pattern", required=True,
                   help="printf pattern for raw frames, e.g. data/input/embryo_data/t%%03d.tif")
    p.add_argument("--first-frame", type=int, default=None)
    p.add_argument("--last-frame", type=int, default=None)
    p.add_argument("--z-scale", type=float, default=None,
                   help="tracker-z per raw plane (default: from checkpoint / config 7)")
    p.add_argument("--mode", choices=["2d", "3d"], default="3d")
    p.add_argument("--interval", type=float, default=2.0, help="live poll seconds")
    p.add_argument("--fit-step-ms", type=int, default=120,
                   help="cell-by-cell animation pacing (0 = instant)")
    p.add_argument("--record", choices=["off", "settled", "animated"], default="settled")
    p.add_argument("--no-labels", action="store_true")
    p.add_argument("--backend", choices=["napari", "mpl"], default="napari",
                   help="'napari' (live window / offscreen GL) or 'mpl' "
                        "(synth 2D, no GL — for display-less machines)")
    p.add_argument("--once", action="store_true",
                   help="render currently-completed frames then exit (replay). "
                        "Default: live-tail until the run finishes.")
    p.add_argument("--headless", action="store_true",
                   help="force no-window rendering (implies --backend mpl)")
    p.add_argument("--fps", type=int, default=4)
    args = p.parse_args(argv)

    # default frame range from checkpoints if not given
    first = args.first_frame
    last = args.last_frame
    if first is None or last is None:
        from cell_viz.watcher import OutputDirWatcher
        done = OutputDirWatcher(args.output_dir).completed_frames()
        if done:
            first = first if first is not None else done[0]
            last = last if last is not None else done[-1]

    app = VizApp(
        run_dir=args.output_dir,
        input_pattern=args.input_pattern,
        first_frame=first,
        last_frame=last,
        z_scale=args.z_scale,
        mode=args.mode,
        show_labels=not args.no_labels,
    )

    backend = "mpl" if args.headless else args.backend
    live = not args.once

    if backend == "mpl":
        if live:
            app.run_live_mpl(interval=args.interval, fps=args.fps)
        else:
            app.run_batch_mpl(fps=args.fps)
    else:  # napari
        if live:
            app.run_live(interval=args.interval, fit_step_ms=args.fit_step_ms,
                         record=args.record, fps=args.fps)
        else:
            app.run_batch(record=args.record, fps=args.fps)


if __name__ == "__main__":
    main()
