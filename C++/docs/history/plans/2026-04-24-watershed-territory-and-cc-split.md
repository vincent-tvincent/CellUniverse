# Plan: Watershed territory + connected-component split trigger

**Date:** 2026-04-24
**Branch:** `jl_voronoi_bleed_penalty_04212026`
**Target:** fix case 1 (missed split where parent spans two daughter blobs) and case 2 (missed split where a basin contains two biological blobs, synth stays at N cells vs biology's N+1) reported on the IMG_BDE7FDCB90AC sketch.
**Prior diagnostics:** 2026-04-24 session log.

## Root cause, one line

The territory partition is a function of neighbor cell positions (Voronoi midlines), so any asymmetry (bloated neighbor, drifted cell, dense cluster) distorts each cell's pixel ownership, and split detection based on distorted pixel sets misses topological signals (two disjoint bright components inside one cell's footprint).

## Fix, one line

Replace center-distance Voronoi with seed-corrected watershed flooding so territory boundaries follow the image gradient ridges; detect splits by counting connected bright components per basin (if K ≥ 2, force split along the vector between the two largest components' centroids).

## What stays the same

- `_voronoiMap` data structure (`std::vector<cv::Mat>` of `CV_8U` per-slice maps, value = cell index). All downstream readers (`calculateBboxCost`, `computeVoronoiBleedVoxels`, shape-fit claim filter) unchanged.
- `rebuildVoronoiMap` entry point. Internally it now dispatches: nearest-neighbor (legacy) or watershed (new), based on a config flag.
- All other penalties (OOB, bleed, overlap, bloat cap).
- Split-cost gate, bio gates, bridge gate, cost-callback plumbing.
- Existing kmeans2 / curImgPca / imgPca / axC axes — they remain as fallbacks when CC trigger doesn't fire.

## What changes

### Stage 1 — Config + Frame members

`C++/includes/ConfigTypes.hpp` SimulationConfig:

```cpp
bool  use_watershed_territory = false;          // stage-1 default off
float watershed_bright_threshold = 0.15f;       // voxels above this participate in flooding
float watershed_seed_correction_cap_ratio = 0.5f;  // seed snap-to-nearest-bright cap = cap_ratio × min(birth radii)
bool  use_cc_split_trigger = false;             // stage-4 default off
float cc_split_bright_threshold = 0.15f;        // voxels above this participate in CC labeling
int   cc_split_min_component_volume = 200;      // components smaller than this ignored
```

`C++/includes/Frame.hpp` Frame members:

```cpp
bool  _useWatershedTerritory = false;
float _watershedBrightThreshold = 0.15f;
float _watershedSeedCorrectionCap = 0.5f;
bool  _useCCSplitTrigger = false;
float _ccSplitBrightThreshold = 0.15f;
int   _ccSplitMinComponentVolume = 200;
```

Public setters for each. Called once per frame by `CellUniverse::optimize`.

### Stage 2 — Watershed flooding

New private method `Frame::fillTerritoryMapWatershed()`:

1. **Seed correction.** For each `_voronoiAnchors[i]`:
   - Look up `_cellBirthRadii[name]`; cap = `_watershedSeedCorrectionCap × min(birth.a, birth.b, birth.c)`.
   - In a (2·cap)³ window around the anchor, find the voxel with the highest brightness.
   - If that voxel's brightness ≥ `_watershedBrightThreshold` and it's within cap of the anchor, replace anchor with that voxel's coordinates.
   - Else leave anchor alone (handled naturally — its basin may have negligible bright mass).
2. **Initialize labels.** `_voronoiMap[z]` all set to 255 (sentinel = unclaimed). For each corrected seed, set its voxel to the cell index.
3. **Priority-queue flooding (Meyer).** Min-heap keyed on voxel brightness (brightest floods first). Push seed neighbors; pop until empty; for each popped voxel that's still unclaimed, assign it the label of its neighbor that popped it, then push its unclaimed neighbors. Below-threshold voxels never get pushed → remain label 255.
4. **Post-fill.** For any voxel that's still 255 after flooding but sits inside a cell's live ellipsoid (dim voxel inside an ellipsoid with no bright flood reaching it), assign it the nearest-seed label via a quick Euclidean fallback. Prevents empty zones inside ellipsoids that confuse bleed/bbox-cost.

Dispatch in `rebuildVoronoiMap`:

```cpp
if (_useWatershedTerritory) fillTerritoryMapWatershed();
else                         fillTerritoryMapNearestNeighbor();   // refactored existing body
```

### Stage 3 — No new call sites

All downstream code (`calculateBboxCost`, bleed, shape-fit claim sets via `buildShapeClaimSet`) reads `_voronoiMap` — no changes required. Flag is fully encapsulated.

### Stage 4 — CC-based split trigger

In `Frame::trySplitCellPhased`, AFTER the bright-pixel gather block (before the primaryDirs construction), if `_useCCSplitTrigger`:

1. Collect the set of voxels within the parent cell's bbox that: (a) have label == parent cell index in `_voronoiMap`, (b) have brightness ≥ `_ccSplitBrightThreshold`.
2. Run 3D connected-component labeling (6-connectivity) on this voxel set via union-find. Label each voxel with its component ID.
3. Count per-component sizes; drop components smaller than `_ccSplitMinComponentVolume`.
4. If remaining K ≥ 2:
   - Sort by size descending; take top-2.
   - Compute each component's brightness-weighted centroid (in world coordinates).
   - Daughter-axis direction = unit vector from centroid1 to centroid2.
   - Store centroids for the AxisPlacement loop (same override path as current kmeans2 centroid-seeds).
   - Insert direction at FRONT of `primaryDirs` with name `"cc2"`.
   - Log: `[Split CC] cell=X K=K nLargest1=n1 nLargest2=n2 sep=sep`
5. If K < 2: no-op, fall through to existing kmeans2 / curImgPca / axC axes.

The CC direction gets candidate-cap priority over kmeans2 because it's topologically stronger (disconnected components is a harder signal than pixel-distribution bimodality).

### Stage 5 — Config + changelog + run

`C++/config/config.yaml`:
- `simulation.use_watershed_territory: true`
- `simulation.watershed_bright_threshold: 0.15`
- `simulation.watershed_seed_correction_cap_ratio: 0.5`
- `simulation.use_cc_split_trigger: true`
- `simulation.cc_split_bright_threshold: 0.15`
- `simulation.cc_split_min_component_volume: 200`

`C++/config/user_input_configurations.ini`:
- `[jihang] last_frame=45`
- `output_name_rule=output_watershed_cc_f1-f45_{timestamp}`

`C++/docs/changelogs/changelogv8.md`:
- Change 24: watershed territory map + seed correction (Stages 1–3)
- Change 25: CC-based split trigger (Stage 4)

## f45 validation criteria

Compare new run against `output_clean_gt_f1-f70_kmeans2_20260424_024113` (stopped at f41):

| Metric | kmeans2 (baseline) | Target (watershed+CC) |
|---|---|---|
| f20–f27 splits | all correct | all correct (no regression) |
| f28 1f89abf4:0 split | clean (drifts 0.3) | clean |
| f39 seven splits | all seven correct | all seven correct |
| f40 e9077:10 split | **MISSED** (23rd cell) | **ACCEPTED** (26 cells by f41) |
| f41 cell count | 25 | 26 |
| f45 final cell count | — (stopped) | run to f45; expect ~30 per GT |

Specifically track e9077:10 size recovery: its aRadius at f35–f39 in kmeans2 was 33–35 vs perfect_45's 45–52. Watershed should restore e9077:10's territory so its shape-fit PCA sees the full bright mass; aRadius should return to the 40+ range by f38.

If f1–f38 shows any regression vs kmeans2 baseline (cell drifts > 10 vx, missed splits, new FPs), halt and diagnose — the watershed seed correction or threshold may need tuning.

## Rollback

Set `use_watershed_territory: false` and `use_cc_split_trigger: false` in config.yaml — pipeline reverts to kmeans2 behavior exactly.

## Notes on edge cases

- **Nearly-merged daughter blobs (connected above threshold):** CC labeling sees them as K=1 → kmeans2 fallback handles. No regression.
- **Parent seed mis-placed (snap is in dim valley):** seed correction pulls it onto the nearest bright lobe within cap. The OTHER lobe becomes unclaimed bright region — the CC labeling on the parent's (one-lobe) Voronoi territory only sees 1 component → misses this. Mitigation (deferred to Stage 6): if any unclaimed bright region adjacent to a cell has volume ≥ threshold, spawn a forced split candidate there. For f45 validation, not required because the observed failure modes have parent's snap inside one of the daughter blobs.
- **Very dim cells:** `watershed_bright_threshold = 0.15` may exclude legitimately dim cells. If f1–f38 regresses on dim cells, lower to 0.10.
