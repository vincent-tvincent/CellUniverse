# Signal Center And Pooling Updates — 2026-04-20

This note summarizes the changes made in the current working session around:

- adaptive cube pooling
- pooled-frame cleanup
- signal-center detection
- signal-center cleanup and merging
- debug exports
- related YAML/config wiring

It is intended as a compact reference for the current behavior in the codebase.

---

## 1. Adaptive Cube Pooling Changes

### 1.1 Strong-peak override for sparse dotted cells

Problem observed:

- Increasing `adaptive_cube_pooling_cube_size_scale` caused some sparse, dotted cells to disappear.
- This happened because large cubes with mostly zero voxels were falling into the mean-pooling branch and their few bright dots were averaged away.

Change made:

- Added a strong-peak override in `src/ImageHandler.cpp`.
- If a cube's `maxValue` is above `adaptive_cube_pooling_strong_peak_threshold`, that cube is forced to use `max` pooling even when the zero-fraction rule would otherwise pick mean pooling.

Current YAML knob:

- `adaptive_cube_pooling_strong_peak_threshold`

Logging added:

- `strongPeakOverrideCubes`

### 1.2 Cleanup changed from single-cube isolation to chunk-size filtering

Problem observed:

- Removing only isolated bright cubes was too brittle and did not reflect cell-scale structure.

Change made:

- Replaced single-cube cleanup with face-connected chunk cleanup.
- Bright pooled cubes are grouped by 6-neighbor connectivity.
- Chunks smaller than a minimum expected size are removed.

Minimum chunk size is derived from:

- configured minimum ellipsoid volume:
  `4/3 * pi * minARadius * minBRadius * minCRadius`
- divided by pooled cube volume:
  `cubeSize^3`
- then scaled by:
  `adaptive_cube_pooling_min_chunk_size_scale`

Effective threshold:

- `minChunkCubeCount = ceil(scale * minCellVolume / cubeVolume)`

Current YAML knobs:

- `adaptive_cube_pooling_remove_isolated_bright_cubes`
  This remains the on/off switch for the cleanup stage even though the logic is now chunk-based.
- `adaptive_cube_pooling_isolated_bright_cube_threshold`
  Used as the cleanup brightness threshold.
- `adaptive_cube_pooling_min_chunk_size_scale`

Logging added:

- `minChunkSizeScale`
- `minChunkCubeCount`
- `cleanupCandidateCubes`
- `removedSmallChunks`
- `removedSmallChunkCubes`

---

## 2. Signal-Center Detection Moved To Preprocessing

### 2.1 Precompute once after pooling

Old behavior:

- Signal centers were localized inside `CellUniverse::optimize()` by analyzing the frame again at runtime.

New behavior:

- Signal centers are now computed once during preprocessing, after interpolation and adaptive cube pooling.
- The result is cached onto each `Frame` during construction.
- `optimize()` reuses `frame.getSignalCenters()` instead of recomputing them.

Files affected:

- `src/ImageHandler.cpp`
- `includes/ImageHandler.hpp`
- `src/CellUniverse.cpp`

### 2.2 Signal-center analysis uses cube-based pooled boxes

New behavior:

- Signal-center detection now works on a cube grid built from the already preprocessed stack.
- The signal-center cube size reuses the adaptive pooling cube size and can be scaled.

Current YAML knobs:

- `signal_center_pooling_cube_scale`
  `1.0` means the same cube size as adaptive pooling.
  Larger values group multiple small pooled cubes per side into a larger signal-center cube.
- `signal_center_pooling_mode`
  Controls whether cube brightness for signal-center detection is measured by:
  - `max`
  - `mean`
- `signal_center_min_cube_brightness_delta`
  Minimum cube brightness above background for a signal-center cube to be kept.

---

## 3. Signal-Center Grouping Logic

### 3.1 Face-adjacency graph

Old behavior:

- The earlier signal-center grouping used looser adjacency and then experimented with chunk-growth rules.

Current behavior:

- Saved bright cubes are treated as a graph.
- Two cubes are neighbors only if they share one full face.
- Edge-only and corner-only touching are not accepted as neighbors.

Initial grouping:

- Plain BFS/connected components on the 6-neighbor cube graph.

### 3.2 Recursive refinement by brightest subgraph

Requested behavior:

- After initial grouping, recursively try to split each group using the brightest subset of cubes.

Current behavior:

- For each initial group:
  1. sort cubes by brightness
  2. select the top configured percentile
  3. run the same 6-neighbor grouping on that subset
  4. if the subset splits, use those split components as seeds to repartition the full parent group
  5. recurse until:
     - max recursion depth is reached, or
     - the top-percentile subset no longer changes

Current YAML knobs:

- `signal_center_recursive_top_percentile`
- `signal_center_recursive_max_depth`

---

## 4. Signal-Center Cleanup And Merge

### 4.1 Finalized-center cleanup now uses surrounding signal-center cubes

Clarified requirement:

- Cleanup should happen after center points are finalized, not on intermediate groups.

Current behavior:

- Each finalized center is mapped back to its signal-center cube coordinate.
- The 26 surrounding neighboring signal-center cubes are inspected.
- A center is removed unless it has enough surrounding bright cubes.

Brightness cutoff for a surrounding cube to count as bright:

- `max(signal_center_min_cube_brightness_delta * 0.5, adaptive_cube_pooling_zero_threshold)`

Required count:

- `signal_center_min_bright_surrounding_cubes`

Current YAML knob:

- `signal_center_min_bright_surrounding_cubes`

Logging added:

- `surroundingZeroThreshold`
- `minBrightSurroundingCubes`
- `removedDarkSurroundCenters`

### 4.2 Merge close finalized centers

Requested behavior:

- If finalized centers are packed closer than the minimum radius, merge them back into a single center.

Current behavior:

- After finalized-center cleanup, centers are merged when Euclidean distance is less than:
  `min(minARadius, minBRadius, minCRadius)`

Merged-center fields:

- `position` = brightness-weighted average position
- `brightness` = average brightness across merged centers
- `boxes` = sum of merged box counts

Logging added:

- `mergedCenterGroups`

---

## 5. Debug Export Added

New behavior:

- Added a signal-center debug export similar to the other debug stacks.
- It writes a dark 3D stack where only signal-center locations are marked as bright cubes.

Output directory:

- `output/signal_centers/<frame>/`

Current YAML knob:

- `export_signal_center_images`

---

## 6. YAML Reorganization

Signal-center-related configuration was moved closer to the preprocessing section so it sits near:

- preprocessing controls
- adaptive cube pooling controls
- debug export toggles

This was done to make the whole preprocess -> pooling -> signal-center flow easier to reason about in one place.

---

## 7. Current Relevant YAML Knobs

### Adaptive pooling

- `adaptive_cube_pooling_enabled`
- `adaptive_cube_pooling_cost_comparison_enabled`
- `adaptive_cube_pooling_cube_size_scale`
- `adaptive_cube_pooling_zero_threshold`
- `adaptive_cube_pooling_majority_threshold`
- `adaptive_cube_pooling_strong_peak_threshold`
- `adaptive_cube_pooling_remove_isolated_bright_cubes`
- `adaptive_cube_pooling_isolated_bright_cube_threshold`
- `adaptive_cube_pooling_min_chunk_size_scale`

### Signal-center preprocessing and grouping

- `export_signal_center_images`
- `signal_center_pooling_cube_scale`
- `signal_center_pooling_mode`
- `signal_center_min_cube_brightness_delta`
- `signal_center_min_bright_surrounding_cubes`
- `signal_center_recursive_top_percentile`
- `signal_center_recursive_max_depth`

### Signal-guided perturbation runtime knobs still used later

- `signal_guided_position_enabled`
- `signal_guided_iterations_per_cell`
- `random_iterations_per_cell`
- `signal_guided_min_sigma_scale`
- `signal_guided_sigma_range_multiplier`

---

## 8. Current High-Level Flow

The current relevant pipeline is:

1. load raw frame
2. normalize to shared global scale
3. run preprocessing
4. interpolate z
5. run adaptive cube pooling
6. run cached signal-center detection on the pooled stack
7. optionally export:
   - preprocessed stack
   - signal-center stack
8. build `Frame`
9. during `optimize()`, reuse cached signal centers for optional signal-guided perturbation

---

## 9. Build Status

After these changes, the project was rebuilt successfully with:

```bash
cmake --build build -j $(nproc)
```

