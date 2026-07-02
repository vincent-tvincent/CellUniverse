# Live Napari Cell-Tracking Visualizer — Design

- **Date:** 2026-06-02
- **Branch:** `Yiding👑Cell-Lumen-SplitGuided-Universe_05272026`
- **Status:** Design (awaiting review → implementation plan)
- **Author:** design drafted with Claude Code

## 1. Purpose

A live, read-only visualization/debug companion that opens a napari window while
CellUniverse is running and, frame-by-frame as each frame completes, overlays the
tracker's decisions on the raw 3D image:

- the **final fitted cells** drawn as ellipsoid **outlines with auxiliary lines** (not solid blobs);
- the **CellLumen brightness centers** drawn as **dots**;
- cells drawn **one-by-one in the order they were picked to fit** (animated within a frame);
- for **splits**, both daughters drawn as above **plus the parent's previous-frame position** as a dim hollow ghost;
- **live** updates — pops up and waits for each frame, stays open until CellUniverse exits;
- **frame scrubbing** — jump back to inspect earlier frames at any time;
- **video export** at the end for playback.

The tool is for demonstration and debugging. It must never alter a run or block the C++ process.

## 2. Goals / Non-goals

**Goals**
- Zero (or near-zero) coupling to the C++ binary for v1: read artifacts already written to the output dir.
- Work both **live** (tail a running output dir) and as **replay** (point at a finished run).
- Faithfully show the CellLumen split-guidance story: candidate centers → chosen split pair → final daughters vs. parent ghost.

**Non-goals (v1)**
- No editing/correcting tracks from the UI.
- No re-running or re-fitting from the UI.
- Not a replacement for `scripts/napari_embryo_review.py` (GT review) — this is a *process/decision* viewer.

## 3. Build / run sanity (already verified 2026-06-02)

Smoke run on this branch confirmed the pipeline and the data this tool consumes:

```
./build/celluniverse 85 88 "data/input/embryo_data/t%03d.tif" \
  outputs/output_smoketest_f85-88_<ts> \
  config/config_embryo_CellLumenFusion_VERIFIED_F085-120_temporalRepair_noTif_20260531.yaml \
  config/embryo/initial_files/00_core_start_points/initial_embryo_85resume_from84.csv
```

Result: frames 85→88, exit 0, ~52 s/frame, cell count 61→71→75→76 (CellLumen splits
accepted, `bestLabel=cell_lumen_primary`). Per-frame artifacts written: `cells.csv`,
`checkpoints/frame_0NN.txt`, `candidate_graph/frame_NN_candidates.csv`.

## 4. Data sources (what's already on disk)

All paths relative to a run's output dir unless noted. The viz is a **consumer** of these.

### 4.1 `cells.csv` — final fitted ellipsoids (appended per frame)
Header:
```
file,name,x,y,z,aRadius,bRadius,cRadius,theta_x,theta_y,theta_z,isTrash
```
- `file` = `t085.tif` … (the frame key).
- `name` = e.g. `Cell type 1_20100`. The lineage code is the suffix after `1_`
  (`20100`). **Parent name = strip the last digit of the suffix** (`20100` → `2010`).
  Root id is the integer before `_` (here `1`). `trash_*` names mark trash cells.
- `x,y,z` center; `aRadius,bRadius,cRadius` ellipsoid semi-axes; `theta_x,theta_y,theta_z`
  orientation (Euler angles, radians — exact convention to be confirmed against
  `Ellipsoid.cpp` during implementation); `isTrash` 0/1.

This is the source of truth for **final positions** to outline.

### 4.2 `candidate_graph/frame_NN_candidates.csv` — per-frame decision log
Header:
```
frame,kind,source,parent,candidate_a,candidate_b,selected,score,raw_score,image_gain,
overlap_cost,bridge_metric,sep,min_sep,max_sep,midpoint_dist,parent_shape,
parent_persistence_penalty,neighbor_claim_penalty,parent_dist_near,parent_dist_far,
parent_dist_balance,x1,y1,z1,x2,y2,z2,vox_a,vox_b,signal_a,signal_b,note
```
Row `kind` values observed: `continuation`, `lumen_center`, `split_pair`, `perturb`.
`source` values: `current_cell_state`, `cell_lumen_center_candidate`,
`cell_lumen_high_recall`, `cell_lumen_split_prior`, `random_local_search`,
`prepass_pca`, `snapshot_seed_large_shift`, `split_reject_compensation`.

Usage by the viz:
- `kind=lumen_center` → **brightness-center dots** (`x1,y1,z1`); `note` carries
  `lumen_candidate_id`, `lumen_signal`, `lumen_voxels`, `lumen_distance`.
- `kind=split_pair`, `selected=1` → an **accepted split** for `parent`; daughter seeds
  at `(x1,y1,z1)` and `(x2,y2,z2)`, with `score`, `sep`, etc. for tooltips.
- `kind=split_pair`, `selected=0` → rejected candidate (optional faint overlay / on-demand).
- `kind=continuation` → cell carried forward (position in `x1,y1,z1`).
- **Row order ≈ processing order.** v1 uses file row order as the *fit-order* proxy for
  the cell-by-cell animation. (See §9 Open Question O1 and §10 optional C++ enhancement.)

> ⚠️ Column-offset care: `x1` is column 23 (1-based). Always parse by **header name**, never by fixed index. A quick check during smoke confirmed `parent_dist_balance` (col 22) sits right before `x1`, so off-by-one is an easy bug.

### 4.3 `checkpoints/frame_0NN.txt` — frame-complete signal + full state
Written atomically when a frame finishes (`[Checkpoint] saved frame N`). Its **appearance
is the "frame N is done, safe to read" trigger**. Also a fallback source for cell state
(`cell <name> x y z aR bR cR tx ty tz brightness isTrash` lines) if a richer per-cell
record is needed than `cells.csv`.

### 4.4 Raw input image — `data/input/embryo_data/t%03d.tif`
3D z-stack (~35 z-slices for this dataset, full z extent ~238 after scaling). Passed to the
viz via `--input-pattern` (the same printf pattern handed to the binary). This is the
background the overlays are drawn on. Note the config uses `noTif` (no rendered
`tiff/real`,`tiff/synth`), so the viz reads the **raw** input directly rather than relying on
rendered stacks.

### 4.5 Coordinate-system note (must resolve in impl)
`cells.csv`/`candidate_graph` use the tracker's `(x,y,z)`. napari indexes arrays as
`(z,y,x)` (plane,row,col). The reader must map tracker `(x,y,z)` → napari `(z,y,x)` and
account for the config's **z scaling** (the raw TIFF has fewer z planes than the tracker's z
range). The z-scale factor will be read from the run's config YAML (`z_scaling` /
interpolation settings) or inferred from `maxZ`/`z_slices` in the checkpoint header
(`z_slices 35`, `maxZ 238`). Resolving this mapping is the first implementation task and is
gated by a visual check against a known frame.

## 5. Architecture

A single standalone script, evolving the existing `scripts/live_monitor_napari.py`
(which already does napari + QTimer polling + layered `LayerSpec`s). New file so we don't
regress the existing monitor:

```
scripts/live_cell_viz.py            # entry point (CLI)
scripts/cell_viz/                   # package (keeps units small & testable)
  __init__.py
  watcher.py        # OutputDirWatcher: detect newly-completed frames (checkpoint poll)
  readers.py        # parse cells.csv, candidate_graph CSV, raw TIFF, config z-scale
  model.py          # FrameModel dataclasses: Cell, LumenCenter, SplitEvent, FrameData
  geometry.py       # ellipsoid → wireframe + auxiliary-axis line segments (3D & 2D)
  lineage.py        # parent-of(name); match daughters↔parent prev-frame position
  render.py         # napari layer construction/update; 3D & 2D modes; fit-order animation
  recorder.py       # frame capture → mp4/gif via napari screenshot + imageio
  app.py            # wires watcher→readers→model→render; QTimer loop; scrub controls
tests/cell_viz/     # pytest unit tests (readers, lineage, geometry) on fixture CSVs
```

**Data flow (per tick):**
```
QTimer (poll every --interval s)
  → watcher: any checkpoints/frame_K.txt with K not yet shown?
      → readers: load cells.csv rows for tK, candidate_graph/frame_K_candidates.csv, raw tK.tif
      → model: build FrameData (cells, lumen_centers, split_events, fit_order)
      → lineage: for each split daughter, look up parent's tK-1 position (from prev FrameData)
      → render: enqueue frame K; if live & newest, run cell-by-cell animation then settle
  → recorder: capture each settled frame for the final video
```

The watcher only reads files that already exist and only **after** the checkpoint marker for
that frame is present, so it never races a half-written `cells.csv`.

## 6. Rendering spec

Chosen mode: **3D + 2D toggle** (napari supports both; a key/buttons switch `viewer.dims.ndisplay` 2↔3).

**Layers (napari):**
| Layer | Type | Content | Style |
|---|---|---|---|
| `raw` | Image | raw z-stack `tK.tif` | gray, base |
| `cells` | Shapes/Surface | ellipsoid outlines | per-lineage color, thin outline |
| `cell_axes` | Vectors/Shapes | auxiliary axis lines through each ellipsoid (the 3 semi-axes) | same color, faint — makes orientation/extent readable |
| `lumen_centers` | Points | CellLumen brightness centers | dot markers (e.g. yellow), size ∝ `lumen_voxels` (optional) |
| `parent_ghosts` | Shapes | split parents' **previous-frame** ellipsoid | dim, **hollow**, dashed/low-opacity |
| `labels` | Text (on cells layer) | cell name + fit index | small, toggleable |
| `rejected` (opt) | Shapes/Points | non-selected split candidates | very faint, off by default |

**Outline + auxiliary lines:** ellipsoids are rendered as **wireframes**, not filled — an
outline plus its 3 principal semi-axis line segments (`geometry.py` produces the polylines /
vectors from center + radii + Euler angles). In 2D mode the ellipsoid becomes its
projected/sliced ellipse outline at the current z; the auxiliary lines become the projected
axes. This satisfies "outline with auxiliary line so you can see the outline better."

**Color convention (proposal):** color by lineage root or by lineage depth so daughters of a
split are visually related; trash cells a muted gray; selected-split daughters briefly
highlighted when they appear.

**Fit-order animation (within a frame):** when a new live frame arrives, cells are added
incrementally in fit order (candidate_graph row order proxy), one every `--fit-step-ms`
(default ~120 ms, configurable; 0 = draw all at once). Each newly added cell flashes briefly.
After the last cell, the frame "settles" (all overlays at final opacity) and is captured for
video. Scrubbing to an old frame shows it fully settled (no re-animation) unless the user
hits "replay fit order".

**Split presentation:** for each `split_pair selected=1`, draw both daughter outlines (from
`cells.csv`, matched by daughter name) and the **parent ghost** = the parent's ellipse from
the **previous** FrameData (looked up via `lineage.parent_of`). A faint connector from ghost
center to each daughter center makes the split legible.

## 7. Live behavior, scrubbing, lifecycle

- **Live tail:** default mode. Watcher polls; when a higher frame number completes, the
  viewer auto-advances to it (unless the user has "pinned" an earlier frame for inspection —
  then new frames are buffered and a "N new frames" indicator shows).
- **Scrubbing:** a frame slider (napari dims slider repurposed, or a custom Qt slider) lets
  the user jump to any already-loaded frame. All loaded `FrameData` is kept in memory
  (small: a few hundred ellipsoids/frame); raw TIFFs are lazy-loaded/cached with an LRU cap.
- **Lifecycle / "until CellUniverse closed":** the tool runs independently. It detects the
  run finished by either (a) a sentinel/`Finished` marker in the run log / a `DONE` file, or
  (b) `--last-frame` reached, or (c) no new checkpoint for `--idle-timeout`. On finish it
  stops polling, finalizes the video, and stays open for inspection until the user closes it.
  (The viz does not monitor the C++ PID in v1; it watches the filesystem. Optional PID watch
  is a possible enhancement.)

## 8. Video export

- During the run, after each frame **settles**, `recorder.py` grabs `viewer.screenshot(canvas_only=True)`
  and appends to an `imageio` writer (`.mp4` via ffmpeg, `.gif` fallback).
- Output: `<output_dir>/viz/playback_f{first}-f{last}.mp4` plus the per-frame PNGs in
  `<output_dir>/viz/frames/` for reproducibility.
- `--record {off,settled,animated}`: `settled` (one frame per tracker frame, default),
  `animated` (capture the fit-order animation too — bigger file), `off`.
- Replay mode can regenerate the video from a finished run without re-running the tracker.

## 9. Error handling & robustness

- **Half-written files:** never read `cells.csv`/candidate CSV for frame K until
  `checkpoints/frame_0K.txt` exists; re-read on parse error after a short backoff.
- **Missing candidate_graph** (older runs / disabled): degrade gracefully — draw cells from
  `cells.csv` only; skip lumen dots & split ghosts with a one-line warning banner.
- **Missing raw TIFF for a frame:** draw overlays on a blank canvas sized from prior frame; warn.
- **Coordinate/z-scale mismatch:** a `--debug-overlay` mode prints the mapped vs raw extents
  and draws a bounding box so misalignment is caught visually on frame 1.
- **napari/Qt not installed:** detect on startup and print the exact install command (§11).
- Tool failures must be **isolated** — a render exception logs and skips that frame, never
  crashes the loop or touches the run.

## 10. Optional C++ enhancement (deferred, not v1)

If the candidate_graph row order turns out to *not* match the true per-cell optimization
order, add a tiny append-only emit in the optimizer: a `fit_order` integer column (or a
`viz/fit_order_frame_NN.csv` with `name,order`) written in the exact sequence cells are
selected to fit. This is additive, behind a config flag, and doesn't change tracking results.
Decision deferred until O1 is checked.

## 11. Dependencies & environment

napari is **not currently installed** in the environment (`python3` = 3.11.5). Add a venv +
requirements rather than touching system Python:

```
python3 -m venv .venv-viz
source .venv-viz/bin/activate
pip install "napari[all]" tifffile imageio imageio-ffmpeg numpy
```
The existing `live_monitor_napari.py` already sets `XDG_CACHE_HOME`/`XDG_CONFIG_HOME` to
`/tmp` for read-only homes (ICS openlab); reuse that. A `scripts/cell_viz/requirements.txt`
will pin versions. On headless boxes, document `napari` needs a display (X11/VNC) or use the
`--record`-only batch path (offscreen) for video generation without an interactive window.

## 12. Testing strategy

- **Unit (pytest, no GUI):** `readers.py` against the smoke-run fixtures
  (`outputs/output_smoketest_f85-88_*`), `lineage.parent_of` truth table
  (`20100`→`2010`, root/trash handling), `geometry.py` ellipsoid→polyline (verify a known
  axis-aligned ellipsoid yields expected extent points).
- **Integration (offscreen napari):** build `FrameData` for f85–f88, render with
  `--record settled --headless`, assert the mp4/PNGs are produced and frame count matches.
- **Manual visual gate:** run live against a fresh `f85-90` run; confirm (1) outlines sit on
  cells, (2) lumen dots land on bright centers, (3) a known split (e.g. parent `1_20010` →
  `1_200100`/`1_200101` seen in smoke logs) shows two daughters + parent ghost, (4) scrubbing
  back works, (5) video plays.

## 13. CLI sketch

```
scripts/live_cell_viz.py <output_dir> \
  --input-pattern data/input/embryo_data/t%03d.tif \
  --first-frame 85 --last-frame 120 \
  --mode 3d|2d            # initial display, toggle in-app
  --interval 2.0          # poll seconds
  --fit-step-ms 120       # cell-by-cell animation pacing (0 = instant)
  --record settled        # off|settled|animated
  --show-rejected         # overlay rejected split candidates (default off)
  --headless              # offscreen render for video-only (no window)
```

## 14. Implementation phases (for the plan step)

1. **Readers + model + lineage** (pure, unit-tested on smoke fixtures) — incl. coordinate/z-scale mapping resolved & visually gated.
2. **Static single-frame render** (3D + 2D) — outlines, aux lines, lumen dots, parent ghosts; verify on f88.
3. **Watcher + live loop + auto-advance + scrubbing**.
4. **Fit-order animation + split highlight**.
5. **Recorder / video export** (settled, then animated).
6. **Polish:** labels, color scheme, rejected overlay, headless batch, requirements pinning, README.
7. **(Conditional) C++ `fit_order` emit** if O1 fails.

## 15. Open questions

- **O1 — fit order fidelity:** Is `candidate_graph` row order a faithful proxy for the
  optimizer's per-cell pick order, or do we need the explicit C++ emit (§10)? Check during phase 1.
- **O2 — theta convention:** exact Euler-angle order/axes from `Ellipsoid.cpp` for correct
  3D wireframe orientation. Resolve in phase 1/2.
- **O3 — run-finished signal:** is there a reliable end marker (log line / DONE file), or do
  we rely on `--last-frame` + idle timeout? Confirm preferred mechanism.
- **O4 — color semantics:** color by lineage root, lineage depth, or split-event recency?
  Pick during phase 2 visual gate.
```
