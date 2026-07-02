# Live cell-tracking visualizer (`live_cell_viz.py`)

Read-only companion that overlays CellUniverse's per-frame decisions on the raw
3D image, **synth-style**: it reproduces the C++ synth (each cell filled with its
`brightness`, stacked on the real image) and outlines each cell with a single
clean silhouette **coloured by that cell's synth brightness** — not a rainbow.
Also drawn: **CellLumen brightness centers** (dots) and, for splits, both
daughters plus the **parent's previous-frame ghost** + connector lines. Works
**live** (tail a running output dir) or as **replay**, and exports a **playback
video**.

## Baked into CellUniverse (no separate command)

Enable it in YAML and a normal run auto-launches the viewer pointed at its own
output dir (the toggle is inherited by every config via `base_config`):

```yaml
live_viz:
  enabled: true
  mode: 3d         # 2d | 3d
  backend: napari  # napari (live window, needs display) | mpl (headless video)
```
CellUniverse spawns it in the background; its log is written **inside** the run at
`<output>/viz/launch.log`, and the video at `<output>/viz/playback_*.mp4`.

Design doc: `C++/docs/plans/2026-06-02-live-napari-cell-viz-design.md`.

## Install (one-time)

```bash
cd C++
python3 -m venv .venv-viz
source .venv-viz/bin/activate
pip install -r scripts/cell_viz/requirements.txt
```

## Run

You can also run it standalone (same engine the bake-in uses).

`--once` = render currently-completed frames then exit (replay). Default = live
tail until the run's last frame lands. `--backend` picks window vs headless.

**Live interactive napari window** — needs a display (works on macOS):
```bash
.venv-viz/bin/python scripts/live_cell_viz.py outputs/<run_dir> \
  --input-pattern "data/input/embryo_data/t%03d.tif" \
  --first-frame 85 --last-frame 120 --mode 3d
```
Auto-advances as each `checkpoints/frame_NNN.txt` appears; stays open after the
run. Toggle layers / 2D↔3D with napari's controls.

**Headless video (no window, no OpenGL — `mpl` backend):**
```bash
# replay a finished run:
.venv-viz/bin/python scripts/live_cell_viz.py outputs/<run_dir> \
  --input-pattern "data/input/embryo_data/t%03d.tif" --once --backend mpl --fps 2
# -> outputs/<run_dir>/viz/playback_f<first>-f<last>.mp4 + viz/frames/*.png
```
`--backend napari --once` does an offscreen napari render instead, but needs a
GL-capable offscreen context (Linux + xvfb/mesa); not available under the bare
macOS `offscreen` Qt platform — use `mpl` there.

## How it works (data sources, all read-only)

| Artifact | Used for |
|---|---|
| `cells.csv` | final ellipsoid outlines (authoritative positions & splits) |
| `candidate_graph/frame_NN_candidates.csv` | CellLumen brightness centers, split *proposals*/scores |
| `checkpoints/frame_0NN.txt` | "frame done" signal; z-scale (`z_slices`,`maxZ`) |
| raw `t%03d.tif` | background image |

Committed splits are detected by **diffing consecutive frames** (parent present
last frame, gone this frame, two lineage daughters appear) — `candidate_graph`
`selected=1` rows are *proposals* and may not commit, so they are not trusted as
splits on their own. Lineage parent = cell name with its last suffix digit
removed (`Cell type 1_20100` → `Cell type 1_2010`). Coordinates map tracker
`(x,y,z)` → napari `(z/z_scale, y, x)` with `z_scale` = 7 (config `z_scaling`,
or `maxZ/z_slices` from the checkpoint).

## Tests

```bash
.venv-viz/bin/python -m pytest tests/cell_viz -q
```
Pure layers (readers, lineage, geometry, scene) are tested against a recorded
smoke run; no GUI required.

## Rendering

Cells are drawn the way the synth does: `synth.py` reproduces
`Ellipsoid::drawWithRotation` (fill each ellipsoid with its `brightness`), and a
max-projection of that is stacked on the real image. Each cell also gets one
silhouette outline (orthographic projection of the ellipsoid — the Schur
complement of its quadratic form) coloured by brightness via the `plasma`
colormap. Brightness comes from the checkpoint's quoted `cell` lines (`cells.csv`
has none); `1_310 ≈ 0.74` vs most cells `0.98`, so dim cells stand out.

## Status / known limits

- Pure logic + synth `mpl` render + headless **live tailing** (baked-in
  auto-launch): **verified** on f85–88 (unit tests + end-to-end run).
- Interactive napari window: works on a GL-capable display (macOS); not
  exercisable in headless CI without xvfb+mesa. The napari layers mirror the mpl
  scene (synth volume + brightness-coloured silhouettes).
- Fit-order animation uses `candidate_graph` row order as a proxy (design O1);
  a precise C++ `fit_order` emit is a deferred enhancement.
