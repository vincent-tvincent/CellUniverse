# CellUniverse Pipeline (Current)

**Last updated:** 2026-07-28 — added the startup / initial-CSV stage, the **CellLumen fusion detector + fusion split priors** (bottom-up, integrated 2026-06), frame-intensity normalization, and the formatted output (`real/`+`synth/` subdirs, `tracked_cells.csv`). The PCA-fit / simulated-annealing / phased-split **core below is unchanged from the 2026-04-16 rewrite and remains accurate.**

This is the authoritative end-to-end pipeline. Per-fix rationale lives in `docs/changelogs/changelogv8.md` (Changes 16–21) and `changelogv7.md` (the core).

**Where the newer stages plug in:** Startup (§0) runs once before the first frame. Frame-intensity normalization is the first step of "Load raw image". The **CellLumen fusion detector** runs each frame around the split loop and feeds it via the `cell_lumen_primary` seed path (see the "CellLumen fusion" section after the frame flow). Output (real/synth TIFF + `cells.csv` + `tracked_cells.csv`) is written at END OF FRAME.

---

## 0. Startup: initial CSV → founder cells (once, before the first frame)

Invoked as `celluniverse <firstFrame> <lastFrame> <inputPattern> <outDir> <config.yaml> <initial.csv>`.

`CsvHandler` reads `initial.csv` and builds one **founder cell** per row. The CSV format decides the **name**:

- **`cellType` column** (e.g. `initial_embryo.csv`): `cellName = cellType + "_" + rowNumber` → `cell_1, cell_2, …` (`CsvHandler.cpp:210`). Split daughters later append `0/1` (`parent + "0"/"1"`), so `cell_1` → `cell_10`, `cell_11`. (This is why the seed's `cellType` string controls the whole naming scheme.)
- **`name` column** (hex-UUID seeds): the name is used verbatim.

Each row also gives the **position** (`x,y,z`) and *optionally* the shape (`aRadius/bRadius/cRadius`), rotation (`theta_*`), and `isTrash`. Missing radii → a default is assigned and the per-frame PCA fit takes over. Z is in native slice units, scaled by `z_scaling` at load. `CellFactory` instantiates each as a triaxial `Ellipsoid`; the set is handed to `CellUniverse`, which owns all cells across the sequence. Founders are the only cells at the first frame; every other cell is born by division.

---

## Frame-level flow

```text
================================================================
                        FRAME N START
================================================================
                             |
                             v
  Load raw image, preprocess (ImageHandler, iterative contrast)
                             |
                             v
  [Frame 2+] Adaptive background from frame N-1 cells:
    background = prevBg × (thisFrameMeanBright / prevFrameMeanBright)
    frame.setBackgroundColor(background)
                             |
                             v
  Set bbox cost mode BEFORE regenerate:
    bboxActiveThisFrame = use_bbox_cost && (frameIndex > 0)
    frame.setUseBboxCost(active, bbox_margin_scale)
                             |
                             v
  frame.regenerateSynthFrame()
    - generateSynthFrame() renders all cells on all z-slices
      (with z-skip: skip cells where |z - cell.z| > cell.maxR)
    - refreshFullCostCache() only if NOT bbox mode
                             |
                             v
  Compute mean cell brightness → frame._meanCellBrightness
  (for brightness-proportional overlap in perturbCell)
                             |
                             v
  Precompute P(split) linear ramp per cell:
    t = clamp((snapElong - 1.0) / (2.0 - 1.0), 0, 1)
    P(split) = P_split_base + t × (P_split_max - P_split_base)
                             |
                             v
  Seed expectedDaughters[name] for every cell with a snap:
    D1_seed = snap.center - 0.5 × splitAxisLength × splitAxisDir
    D2_seed = snap.center + 0.5 × splitAxisLength × splitAxisDir
                             |
                             v
================================================================
  STAGE 1 — POSITION CALIBRATION  (frame 1 only)
================================================================
  For each cell:
    Step 1: calibrateCellPositionViaCentroid (weighted mean)
    Step 2: perturbCell refinement × split_calibration_iterations_per_cell
  Frame 2+ skipped — snap + Stage 0a do position work.
                             |
                             v
================================================================
  STAGE 2 — PCA SHAPE FIT  (per cell, all cells)
                 [OpenMP-parallelized across cells]
================================================================
  Snapshot all cell positions into CellPosSnapshot[] BEFORE
  the parallel region (avoids data race: threads write radii/
  rotation while others read positions).
                             |
                             v
  FOR EACH CELL (parallel):
    Build claim set from position snapshot (immutable)
    Look up birth radii from cellShapeBirth[name] (captured
    once at first appearance, NEVER updated)
    Call calibrateCellShapeViaPca with birth radii as mask basis
                             |
    NOTE: an OPTIONAL scale-invariant variant exists behind
    cell.pcaShapeFgAnchored (default false; see changelogv8
    Changes 40-42). When ON (and a valid per-frame ForegroundModel
    is installed), the gather (gatherForegroundComponent, in its own
    TU src/ForegroundComponent.cpp) flood-fills the cell's blob at a
    PER-CELL half-max level bg + 0.5*(peak-bg) — bg = _fgModel.muBg,
    peak = 99th-pct real intensity in a core box off the FIXED entry
    radius — instead of the maskScale box; bypasses the ellipsoid
    mask pre-filter and uses the uncapped weighted centroid for
    position. The half-max level is anchored to fit-independent image
    facts, so the region is stable frame-to-frame (no bloat feedback;
    the earlier global pFg>0.5 boundary drifted with muFg and bloated).
    The fg model is estimated in optimize() (CellUniverse ~5652) from
    HARD ellipsoid membership of the current cells, so those cells
    MUST be in interpolated-z working coords: seed from
    initial_embryo_sized.csv (Change 41 fixes its z-space). Recovers
    a STABLE ~30-42px nucleus across frames. Default OFF = legacy box
    fit (NOTE: flag OFF is not bit-identical to the pre-Change-42
    binary — an -O3+LTO FP-scheduling artifact, same code path; see
    Change 42).
                             v
   ┌────────── INSIDE calibrateCellShapeViaPca ──────────┐
   │                                                     │
   │ STAGE 0a: POSITION REFINEMENT                       │
   │   birthMaxR = max of birth radii                    │
   │   Gather Voronoi-filtered pixels in 1.5 × birthMaxR │
   │   around current center (soft ownership weights)    │
   │   Compute weighted peak centroid using weight³      │
   │     → bright peak dominates over dim gaps           │
   │     → pre-split bimodal cells stay on ONE daughter  │
   │       (pre-pass still finds both for split detect)  │
   │   shift = peak - center                             │
   │   cap = 0.10 × birthMaxR                            │
   │   Apply: scale = min(1, cap/|shift|) if |shift|>0.5 │
   │   cell.position += shift × scale                    │
   │   log: [Pos Refine] cell=... peak=... shift=...     │
   │                                                     │
   │ STAGE 0b: ADAPTIVE MASK SIZING (gradient-based)     │
   │   Starting scale = pcaShapeMaskScale (1.2)          │
   │   For up to 5 iterations:                           │
   │     tryR = scale × birthMaxR                        │
   │     gather pixels in tryR sphere                    │
   │     count bright pixels in:                         │
   │       inner shell [0.70R, 0.85R]                    │
   │       outer shell [0.85R, R]                        │
   │     normalize each by its volume fraction           │
   │       (inner=0.271, outer=0.386 of R³)              │
   │     ratio = outerDensity / innerDensity             │
   │                                                     │
   │     ratio > 0.50 → expand (outer still bright →     │
   │                    cell extends further)            │
   │     ratio < 0.15 → shrink (outer dim →              │
   │                    mask overshoots cell)            │
   │     else → converged (brightness trailing off)      │
   │   Clamped to [0.8, 2.5]                             │
   │   log: [Adaptive Mask] cell=... scale=... ratio=...│
   │                                                     │
   │ PCA ITERATION LOOP (up to pcaShapeMaxIters = 15):   │
   │   1. Gather bright pixels (cached by center):       │
   │      - Sphere of adaptiveMaskScale × birthMaxR      │
   │      - Ellipsoid mask (per-axis birth radii × scale)│
   │      - Brightness cutoff: max(0.05, bg + 0.02)      │
   │      - Soft Voronoi: weight ×=                      │
   │          otherD² / (selfD² + otherD²)               │
   │        (hard cutoff if otherD < 0.25 × selfD)       │
   │   2. Cache pixelWeights[] = pow(w, exp) per pixel   │
   │   3. Adaptive exponent (if pcaShapeAdaptiveExponent):│
   │      pCore = fraction(w > 1.5 × meanW)              │
   │      exp ramp: dim (1.3) ← pCore → bright (1.15)    │
   │      radInfl ramp: 1.0 ← pCore → 1.15               │
   │   4. Weighted centroid (reuse cached weights)       │
   │   5. Weighted covariance (reuse cached weights)     │
   │   6. cv::eigen → λ0 ≥ λ1 ≥ λ2, eigenvectors         │
   │   7. If λ0/λ2 < 1.1: skip rotation (degenerate)     │
   │   8. Rank assignment (a←λ0, b←λ1, c←λ2)             │
   │      Sign-align eigenvecs with current slot dirs    │
   │   9. Variance-based radii:                          │
   │      r_i = radInfl × pcaShapeRadiusScale × √λ_i     │
   │   10. Shift center toward centroid                  │
   │       (capped at maxPosShiftFraction × maxR;        │
   │        pcaShapeUpdatePosition=false for mature cells│
   │        — only daughter refit enables this)          │
   │   11. Converged when max|dR| < convRadius AND       │
   │       max axis angle < convAngleDeg                 │
   │                                                     │
   └─────────────────────────────────────────────────────┘
                             |
                             v
  After all cells fit (serial merge of per-cell logs):
    FIT-SIDE GROWTH CAP (10%/frame):
      for each cell with existing ref:
        cell.radii[i] = min(fitted[i], ref[i] × 1.10)

    REFERENCE UPDATE (5%/frame bounded growth):
      new cells: cellShapeReference[name] = current fit
      existing: ref[i] = clamp(fit[i], ref[i]×0.95, ref[i]×1.05)

    BIRTH CAPTURE (once, never updated):
      new cells: cellShapeBirth[name] = current fit
                             |
                             v
================================================================
  STAGE 3 — SNAP-ANCHORED BBOX INSTALL  (frame 2+ only)
================================================================
  For each cell with valid snapshot:
    center = snap.position
    radius = snap.maxRadius
    bbox.half-extent = max(40, marginScale × radius)  ← floor 40px
    frame.setSnapBbox(name, bbox)
                             |
                             v
================================================================
  STAGE 4 — IMAGE-GROUNDED DAUGHTER PRE-PASS  (all cells)
                 [OpenMP-parallelized across cells]
================================================================
  For every cell with valid snapshot:
    Gather bright pixels in 3 × snap.maxR box
    (soft Voronoi weighting vs neighbors)
    LOCAL-FRAME PCA on pixels:
      localPos = R⁻¹ × (worldPos - center)
      localPos /= (a, b, c)
      Principal eigenvector → world space (un-normalize, rotate)
    D1_exp, D2_exp = centroids of two halves
    (split at median projection onto eigenvector)
                             |
                             v
================================================================
  STAGE 5 — UNIFIED PERTURB + SPLIT LOOP
================================================================
  phaseNames = set of all cell names at frame start
  totalIters = cells.size() × iterations_per_cell
  Loop totalIters:
    Pick random eligible cell index
    if rand < P(split) and not in splitBlacklist:
        attempt split (see Split Attempt below)
        if accept:
          phaseNames.insert(daughter0_name, daughter1_name)
          recompute frame._meanCellBrightness
          rebuildEligible()
    else:
        perturbCell (position + rotation only; radii fixed
                     by Stage 2 PCA)

  perturbCell cost evaluation:
    Radius-proportional sigma:
      posScale = maxR / perturbSigmaReferenceRadius (25)
      getPerturbedCell(dir, posScale) scales x/y/z offsets

    Brightness-proportional overlap:
      bRatio = cellBrightness / meanBrightness
      effOverlapWeight = overlapWeight × bRatio²

    Frame 1 or no snap-bbox: legacy full-image L2 incremental
    Frame 2+ with snap-bbox:
      asymmetric L2 over bbox voxels (no exclusion mask)
      per voxel: d = synth - real
        d² × asymK if d > asymThreshold (0.03), else d² × 1
      (threshold eliminates double-boundary artifacts)
                             |
                             v
================================================================
                        END OF FRAME N
================================================================
                             |
                             v
  Brightness EMA update (per-cell, from real image)
                             |
                             v
  Freeze snapshot for frame N+1:
    shapeElongation = max(a,b,c) / min(a,b,c)
    splitAxisDir = cell.worldSplitAxis()  (SHORTEST axis)
    splitAxisLength = shortest radius
                             |
                             v
  Save output (see "Output" section below):
    checkpoint + cells.csv + tracked_cells.csv + name_map.csv
    + real/{N}_real.tif + synth/{N}_synth.tif (if export_frame_tiff)
```

---

## Frame-intensity normalization (first step of image load)

Before fitting, each loaded frame is normalized to `[0,1]` by stretching between two references (`normalizeStackToFrameIntensity`, `CellUniverse.cpp`):

- `low_ref` = percentile `frame_intensity_scale_low_percentile`; `high_ref` = percentile `frame_intensity_scale_high_percentile` (excluding zeros).
- **Robust low reference (Change 20):** with `frame_intensity_low_signal_floor_fraction > 0`, voxels below `fraction × high_ref` are excluded from the low percentile, so a near-zero border/pedestal can't drag `low_ref` to ~0 and wash the tissue to white. Default `0` = off (fine for embryo_data, pedestal ~36); set ~`0.25` for high-pedestal raw crops (original_data / Pavak 422×517, pedestal ~107).
- Z is upsampled by `z_scaling` (7). `preprocess_mode` may add contrast steps (`none` currently).

`generateOutputFrame()` renders `_realFrame × 255`, so this normalization is exactly what the exported `real/` TIFFs show — mis-tuned, it washes them near-white.

---

## CellLumen fusion detector + split priors (bottom-up; integrated 2026-06)

Alongside the top-down PCA fit, **CellLumen** reads the raw signal directly and proposes divisions. Per frame (around the split loop, Stage 5):

1. **Seeded watershed detection** over a short frame window (`fusionSplitPriorWindowSize: 3`) finds bright blobs. It can heavily **over-segment** (`base_cells≈9 → watershed_cells≈24`) — the current source of over-splitting.
2. **Fusion split priors** — for each tracked cell, ranked "this is really two" hypotheses scored from daughter separation, size balance, and window persistence. Key knobs (`config.yaml`, `prob:`):
   - `fusionSplitPriorWindowBalancedDaughterBonus: 25.0` — a big +25 bonus when a balanced daughter pair persists across the window (**main over-split driver**).
   - `fusionSplitPriorMaxScore: 22.0` — accept threshold (lower score = stronger).
   - `fusionSplitPriorMinSeparation: 8.0`, `fusionSplitPriorMinVoxels: 500`, `fusionSplitPriorRoundParentMaxShape: 1.25` (round parents up to this elongation may still be split).
3. **Injection into the split loop:** a strong prior supplies the daughter seeds (`bestLabel = cell_lumen_primary`) and can **force** the split with a nominal `costDiff = -1`, bypassing the cost gate. It still must pass the shape/image/bio gates and the **snapshot refit-drift** check (`fusionSplitPriorSnapshotSeedMaxRefitDrift: 12` — daughters drifting > ~12px on refit are rejected). That drift gate is what blocks all splits on the coarse 422×517 data, where drift is 14–45px.

So a division can be triggered by EITHER the top-down elongation / PCA-bridge path OR a CellLumen fusion prior; both feed the same gate gauntlet in the Split Attempt below.

---

## Output (END OF FRAME)

- `checkpoints/frame_NNN.txt` — full state, enables resume.
- `cells.csv` — internal per-cell rows (raw names, `aRadius/bRadius/cRadius`, …). Read by `CsvHandler` and the `--lineage-tree` step, so **not** renamed.
- `tracked_cells.csv` + `name_map.csv` (Change 19) — human-readable, output-only: `frame,cell,x,y,z,semi_axis_a/b/c,rot_*,is_marker` with `cell_N` names.
- `real/{N}_real.tif` + `synth/{N}_synth.tif` (when `export_frame_tiff: true`) — the normalized image vs the model's synthetic render, for visual comparison.
- After the last frame, the parent-child **lineage tree** is reconstructed from the name suffixes (`cell_3` → `cell_30/cell_31`) and rendered.

---

## Split attempt (inside Frame::trySplitCellPhased)

For a cell that rolled a split attempt:

1. **Install snapshot-state parent**

   - Build Ellipsoid at (snap.position, snap.radii, snap.rotation)
   - Compare live vs snap baseline using FRESH bbox cost over
     a temp union bbox (not stale `_currentCost` in bbox mode)
   - Pick baseline = min(liveCost, snapCost)
2. **Gather bright pixels** — Voronoi-filtered (soft weights) in
   3 × snap.maxR box. `selfClaim` from pre-pass D1_exp, D2_exp.
3. **Build primary axes**

   - Shortest snapshot local axis
   - imgPca direction (from Stage 4 pre-pass) if distinct
   - Typically 2 primaries × 2 midpoints × 5 variants = up to 20
4. **Split bbox install (bbox mode)**

   - splitBbox = union of parent + all daughter seeds × margin
   - `_snapBboxes[parentName+"0"] = splitBbox; ["1"] = splitBbox`
   - No Voronoi exclusion mask (`_sharedMasks` removed)
5. **Candidate loop** — for each candidate:

   - Build daughters at seed positions
     (radii = volumeScale × snap radii, volumeScale = 0.7937)
   - Partial z-slice render (z-skip: only cells near the slice)
   - Burn-in: split_candidate_burn_in_iterations × perturbCell
     (alternate daughters, tight position sigmas)
   - candTotal = image_cost + overlap_penalty

   **Two-pass pre-filter** (before cost ranking):

   - Edge brightness at d1/d2 positions >= bio_bridge_min_edge_brightness_absolute (0.04)
   - Quick valley on d1→d2 axis: gap/max(edges) < bio_bridge_max_valley_ratio (0.85)
   - Candidates failing either are excluded from competition

   Track best candidate that passed pre-filter by lowest candTotal
6. **Refine** — split_final_refine_iterations × perturbCell on winner
7. **Daughter refit** (per daughter)

   - Short PCA shape fit with updatePosition=true
   - Radii clamped to [0.85, 1.05] × built_radii per axis
   - (Ceiling 1.05: ensures max single-daughter volume fraction
     = (1.05 × 0.7937)³ = 0.579 < bio gate 0.65)
8. **Bridge gate (Pass 2, final check)**

   - Sample real-image brightness along d1→d2 axis
   - Zones: edge1 (in d1, far from gap), gap (middle ≥30% of halfLen),
     edge2 (in d2, far from gap)
   - valleyFromBright = gapBright / max(edge1, edge2)
   - **Reject edge_too_dim**: min(edges) < 0.04
   - **Reject bridge_flat**: valleyFromBright > 0.85
9. **Bio gates**

   - size_ratio: max(r1,r2)/min(r1,r2) ≤ 1.5
   - combined_volume: 0.6 ≤ (v1+v2)/refParentVol ≤ 1.3
   - max_single_daughter: max(v1,v2)/refParentVol ≤ 0.65
   - d1/d2 not buried in any non-sibling cell
   - d1/d2 not buried in sibling
   - **NEIGHBOR-BRIDGING** (anti-false-positive):
     for each neighbor: if |neighbor - d1| < 0.5 × |d2 - d1| OR
     |neighbor - d2| < 0.5 × |d2 - d1| → reject
     (catches splits where a cell stretches to a neighbor's blob)
10. **Cost gate**
    adaptiveThreshold = max(split_cost, split_cost_fraction × baseline)
    accept if costDiff < -adaptiveThreshold
11. **Cleanup** — erase `_snapBboxes` entries; `restoreLiveParent` on reject

---

## Cost (bbox mode, frame 2+)

```
cost = Σ voxels in bbox: (synth - real)² × multiplier
where multiplier = asymK if (synth - real) > asymThreshold, else 1
```

- **bbox**: snap-anchored per cell, half-extent = max(40, marginScale × snap.maxR)
- **No Voronoi exclusion mask** — all voxels in bbox contribute.
  Neighbors' synth is constant during this cell's perturbation → cancels in cost delta.
- **asymK threshold (0.03)**: small boundary rendering artifacts (d ≈ 0.01-0.05)
  are penalized at 1x. Genuine overshoot (cell covering dark region, d ≈ 0.1+)
  gets the full asymK. Eliminates double-boundary bias: two daughters' boundary
  artifacts no longer structurally cost more than one parent's.

---

## Shape reference system

Each cell tracks THREE persistent (a, b, c) state vectors:

### `cellShapeBirth` (frozen, NEVER updated)

- Captured once at first appearance (frame 1 for initial cells, post-refit
  for daughters)
- Used as the reference for:
  - Adaptive mask sizing (`scale × birthMaxR` pixel gathering)
  - Position refinement cap (10% of birthMaxR per frame)
- Decoupled from all feedback loops — frozen.

### `cellShapeReference` (bounded-growth, ±5%/frame)

- Per-frame update: `ref[i] = clamp(fit[i], ref[i]×0.95, ref[i]×1.05)`
- Used ONLY for the fit-side growth cap (10%/frame):
  `cell.radii[i] = min(fitted[i], ref[i] × 1.10)`

### `cells[i].radii` (live fit)

- Updated by PCA shape fit each frame
- Input to perturbCell, rendering, etc.

Combined: adaptive mask (image-grounded) + bounded ref (slow change) + fit cap (no sudden bloat).

---

## Tiered Voronoi pixel ownership

`gatherBrightPixelsVoronoi` uses a three-tier ownership model:

```
selfBest = min distance² from pixel to self claim points
otherBest = min distance² from pixel to any other cell's claim points

if otherBest < 0.25 × selfBest:
    REJECT (clearly owned by another cell)
elif otherBest < selfBest:
    CONTESTED — ownership = otherBest / (selfBest + otherBest)
    pixel.weight = (brightness - background) × ownership
else:  # otherBest ≥ selfBest
    FULL WEIGHT — self is closer than any other cell
    pixel.weight = (brightness - background) × 1.0
```

Three-tier design rationale:

- **Reject tier** (other 2x closer): pixel genuinely belongs to another cell
- **Contested tier** (self and other comparable distance): proportional split
  prevents zero-sum competition where one cell starves neighbors
- **Full-weight tier** (self is nearest): preserves variance-based radius accuracy

The full-weight tier is the critical distinction vs earlier designs. Without
it, isolated boundary pixels (far from all neighbors) got ownership 0.7-0.9
just because `otherBest` was finite, shrinking the weighted variance across
the entire cell. Only genuinely contested pixels now see soft weighting.

---

## Config (active values)

```yaml
cell:
  pcaShapeMaxIters: 15
  pcaShapeRadiusScale: 2.236          # sqrt(5)
  pcaShapeMinPixels: 50
  pcaShapeMaskScale: 1.2              # STARTING scale for adaptive mask
  pcaShapeConvergeRadius: 0.3
  pcaShapeConvergeAngleDeg: 2.0
  pcaShapeUpdatePosition: false       # snap + Stage 0a handle position
  pcaShapeMaxPosShiftFraction: 0.5
  pcaShapeWeightExponent: 1.3
  pcaShapeAdaptiveExponent: true
  pcaShapeWeightExponentBright: 1.15
  pcaShapeCoreBrightnessThreshold: 0.15  # bypassed; relative pCore active
  pcaShapeCoreFractionLow: 0.25
  pcaShapeCoreFractionHigh: 0.35
  pcaShapeRadiusInflationBright: 1.15
  pcaShapeRadiusPercentile: 0.90      # parsed but unused (variance active)
  perturbSigmaReferenceRadius: 25.0

simulation:
  asymmetric_cost_weight: 8.0
  asymmetric_cost_threshold: 0.03     # asymK only fires above this

prob:
  P_split_base: 0.03
  P_split_max: 0.5
  split_cost: 2000
  split_cost_fraction: 0.03           # adaptive: max(fixed, fraction × baseline)
  overlap_penalty_weight: 30000.0
  use_bbox_cost: true
  bbox_margin_scale: 2.5
  bio_daughter_size_ratio_max: 1.5
  bio_combined_volume_min_fraction: 0.6
  bio_combined_volume_max_fraction: 1.3
  bio_max_single_daughter_volume_fraction: 0.65
  bio_bridge_max_valley_ratio: 0.85
  bio_bridge_min_edge_brightness_absolute: 0.04
  split_daughter_refit_iterations: 3
  split_daughter_refit_min_radius_fraction: 0.85
  split_daughter_refit_max_radius_fraction: 1.05
  split_daughter_volume_scale: 0.7937   # cbrt(1/2), volume-preserving
  split_candidates_per_attempt: 30
  split_candidate_burn_in_iterations: 50
  split_final_refine_iterations: 30
```

Hardcoded constants (not in YAML):

- Refit ceiling per axis: 1.05 (in config, but conceptually hardcoded by math)
- Bbox minimum half-extent: 40 px
- Adaptive mask gradient thresholds: 0.15 (shrink), 0.50 (expand) — outer/inner ratio
- Adaptive mask scale range: [0.8, 2.5]
- Position refinement cap: 10% × birthMaxR per frame
- Position refinement weight exponent: 3 (weight³ for peak dominance)
- Bounded ref growth cap: ±5%/frame
- Fit-side growth cap: +10%/frame vs ref
- Neighbor-bridging rejection: d_to_neighbor < 0.5 × d_to_sibling
- Soft Voronoi reject threshold: otherBest < 0.25 × selfBest

---

## Key invariants

| Invariant                          | Active mechanism                                                     |
| ---------------------------------- | -------------------------------------------------------------------- |
| Cells divide along SHORTEST axis   | `Ellipsoid::worldSplitAxis()` returns min semi-axis direction      |
| Mask is image-grounded, not static | Adaptive mask sizing (shell density feedback)                        |
| Position tracks real blob          | Stage 0a (weight³ peak centroid, 10% capped shift)                  |
| No Voronoi starvation              | Soft Voronoi weighting (proportional ownership)                      |
| No double-boundary bias            | asymK threshold (0.03) — boundary artifacts at 1x                   |
| No neighbor-bridging splits        | Bio gate: daughter can't be closer to neighbor than 50% sibling dist |
| Radii correctly sized              | Variance formula`pcaShapeRadiusScale × sqrt(λ)` + adaptive mask  |
| Position anchor works              | Snap bbox (floor 40) + brightness² × overlap weight                |
| Daughters get perturbed            | `phaseNames.insert(d0, d1)` on split accept                        |
| No data race in parallel PCA       | Position snapshot before`#pragma omp parallel for`                 |
| Refit ceiling < bio gate           | 1.05 × 0.7937 = 0.833 per axis → max vol fraction 0.579 < 0.65     |
| No full-image cost on bbox frames  | `setUseBboxCost` before `regenerateSynthFrame` → skip cache     |

---

## Diagnostic log tags

- `[Adaptive Background] frame N base=... ratio=... background=...`
- `[Optimize] frame N (M cells, K iterations) useBboxCost=B bboxMarginScale=X`
- `[P(split)] cell=... shapeElong=... P(split)=...`
- `[Calibration] cell=...` — frame 1 only
- `[Pos Refine] cell=... peak=... shift=... capped=... newPos=...` — Stage 0a
- `[Adaptive Mask] cell=... scale=... density=... iters=...` — Stage 0b
- `[PCA Shape Exp] cell=... pCore=... exp=... radInfl=...` — adaptive exponent
- `[PCA Shape] cell=... iter=i n=... R=(a,b,c) dR=... axisAng=... posShift=...`
- `[PCA Shape Cache]` — gather cache hit/miss
- `[Fit Growth Cap] frame N clamped=X` — fit-side cap fired
- `[Shape Reference] frame N births=... refCaptured=... refUpdated=...`
- `[Snap Bbox] frame N installed=X total=Y`
- `[Pre-Pass] round=0 cell=... shift1=... shift2=...`
- `[Split Snapshot Parent] cell=... livePos=... snapPos=... baseline=... (bbox)`
- `[Split Attempt] cell=...`
- `[Split Seeds] cell=... splitAxisDir=...`
- `[Split Dirs] selected=[axX, imgPca]`
- `[Split Bbox Init] cell=... bboxXYZ=...`
- `[Split Baseline] cell=... imageCost=... overlap=... threshold=...`
- `[Split Cand] idx=i label=... final1=... final2=... total=...`
- `[Split Cand PreFilter] EDGE_DIM / NO_VALLEY` — Pass 1 rejection
- `[Split Winner] bestIdx=... label=...`
- `[Split Daughter Refit] d1/d2 built=... floor=... ceil=... post=...`
- `[Split Refine] iters=... accepts=... preTotal=... postTotal=...`
- `[Split Bridge] axisLen=... valleyFromBright=... gapDensity=...`
- `[Split Reject bio] reason=... (bridge_flat / edge_too_dim / volume_fraction / size_ratio / buried / sibling_buried / d1_bridging_to_... / d2_bridging_to_...)`
- `[Split Reject cost] diff=... threshold=...`
- `[Split Accepted] costDiff=... bestLabel=... d1=... d2=...`
- `[Optimize Done] frame N perturb_accepted=... split_attempts=... split_accepted=... final_cells=...`

---

## Related docs

- `docs/changelogs/changelogv8.md` — **active** change log (Changes 16–21: config flatten, dead-knob audit, formatted output, normalization fix, ILP removal)
- `docs/changelogs/changelogv7.md` — the PCA/SA/split core changes (closed)
- `.claude/rules/gotchas.md`, `architecture.md`, `dataset.md`, `configs.md`, `workflow.md` — invariants, module roles, dataset/GT, config, workflow
- `docs/details.md` — deep reference (lower sections predate the triaxial rewrite; trust its "Update Pointer" headers)
