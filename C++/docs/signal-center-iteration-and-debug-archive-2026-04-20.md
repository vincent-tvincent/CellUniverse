# Signal Center Iteration And Debug Archive — 2026-04-20

This document archives the later changes from the same development session after the first summary note. It focuses on:

- post-pooling signal-center behavior
- recursive signal-center grouping
- finalized-center cleanup and merging
- adaptive pooling cleanup changes
- extra post-max pooling
- restored post-localization export
- tuning observations from the current YAML

Related earlier summary:

- [signal-center-and-pooling-updates-2026-04-20.md](./signal-center-and-pooling-updates-2026-04-20.md)

---

## 1. Signal-Center Detection: Current Structure

Signal-center detection now runs once during preprocessing and is cached onto each `Frame`.

Current high-level order:

1. preprocessing
2. z interpolation
3. adaptive cube pooling
4. optional extra post-max pooling
5. signal-center detection on the resulting pooled stack
6. optional debug export of signal centers
7. `Frame` construction with cached centers

The cached centers are later reused by `CellUniverse::optimize()` when signal-guided perturbation is enabled.

---

## 2. Recursive Signal-Center Grouping

### 2.1 Face-neighbor graph

Signal-center grouping is no longer using loose 3D adjacency or edge/corner touches.

Instead:

- cubes are nodes in a graph
- two cubes are neighbors only if they share one face
- initial groups are plain BFS connected components on that face-neighbor graph

### 2.2 Recursive brightest-subset regrouping

After initial grouping, each group is recursively refined:

1. sort cubes in the current group by brightness
2. select the top `signal_center_recursive_top_percentile`
3. run face-connected grouping on that selected subset
4. if that subset splits into multiple components, use those components as seeds to repartition the whole parent group
5. recurse into the resulting child groups

### 2.3 Recursion terminate conditions

Recursion stops when any of these happens:

- the current group has size `<= 1`
- recursion depth reaches `signal_center_recursive_max_depth`
- the newly selected brightest subset is exactly the same as the previous subset for that branch
- the brightest subset does not split into at least 2 face-connected components

Important note:

- The recursive stage does **not** re-pool the image at each level.
- Pooling happens once.
- Recursion only re-groups the already-created bright cubes.

---

## 3. Why Higher `signal_center_min_cube_brightness_delta` Can Create More Centers

Observed behavior:

- raising `signal_center_min_cube_brightness_delta` sometimes increased the number of centers instead of decreasing it

Reason:

- this threshold removes dim cubes first
- dim cubes are often the bridges between stronger bright lobes
- removing those bridge cubes can split one connected component into several disconnected components
- more disconnected components usually means more centers

So in the current logic, `signal_center_min_cube_brightness_delta` acts partly like a bridge-pruning parameter, not just a noise filter.

This is especially pronounced when:

- `signal_center_pooling_mode: max`
- cube size is not large enough to merge local peaks
- recursive splitting is enabled

---

## 4. Finalized-Center Cleanup And Merge

### 4.1 Cleanup now happens on finalized centers

Clarification requested during the session:

- cleanup should happen after center points are finalized, not on intermediate cube groups

Current behavior:

- group-level cleanup was removed
- centers are first finalized from groups
- each finalized center is mapped back to its signal-center cube coordinate
- the 26 neighboring signal-center cubes around that finalized center are checked

### 4.2 Current survival rule

A finalized center is removed unless it has enough surrounding bright signal-center cubes.

Brightness threshold for a surrounding cube to count as bright:

- `surroundingZeroThreshold = max(signal_center_min_cube_brightness_delta * 0.5, adaptive_cube_pooling_zero_threshold)`

Required number of bright surrounding cubes:

- `signal_center_min_bright_surrounding_cubes`

This changed the earlier raw-voxel neighborhood check to a surrounding **signal-center cube** check.

### 4.3 Close-center merge

After finalized-center cleanup:

- centers are merged if Euclidean distance is less than the minimum configured cell radius:
  `min(minARadius, minBRadius, minCRadius)`

Merged center values:

- `position`: brightness-weighted average
- `brightness`: average brightness of merged centers
- `boxes`: sum of merged `boxes`

---

## 5. Adaptive Pooling Updates Added Later

### 5.1 Strong-peak override

To prevent sparse dotted cells from disappearing when cube size increased:

- a cube now forces `max` pooling if its internal max value exceeds `adaptive_cube_pooling_strong_peak_threshold`
- this prevents some mostly-empty but real-signal cubes from being averaged away

### 5.2 Chunk-based cleanup

The old isolated-bright-cube cleanup was replaced with chunk-based cleanup:

- bright pooled cubes are grouped by face connectivity
- chunks smaller than a minimum expected cell-sized chunk are removed

The size threshold is derived from:

- minimum configured ellipsoid volume
- divided by cube volume
- scaled by `adaptive_cube_pooling_min_chunk_size_scale`

This means the cleanup is now cell-scale-based, not just single-cube-based.

### 5.3 Extra post-max pooling pass

An additional max-pooling pass was added after adaptive pooling.

Current behavior:

- adaptive pooling runs first
- then an optional second cube-wise max pooling pass runs if `adaptive_cube_pooling_post_max_scale > 1`

Its cube size is:

- `round(adaptive_cube_size * adaptive_cube_pooling_post_max_scale)`

This extra pass modifies the stack before signal-center detection.

### 5.4 Signal-center cube size now follows the post-max size

Originally, signal-center detection still used the original adaptive cube size as its base.

That was changed so signal-center detection now uses the **effective pooled cube size**, which includes the post-max scaling.

Current base signal-center size:

- `effective pooled cube size = round(adaptive cube size * adaptive_cube_pooling_post_max_scale)`

Then signal-center analysis applies its own ratio:

- `signal center cube size = round(effective pooled cube size * signal_center_pooling_cube_scale)`

---

## 6. Debug Exports

### 6.1 Signal-center debug export

Added:

- a debug stack export that writes a dark 3D stack with bright cubes placed at signal-center locations only

Output location:

- `output/signal_centers/<frame>/`

YAML switch:

- `export_signal_center_images`

### 6.2 Post-localization debug export restored

During this session it was discovered that:

- `export_post_localization_images` still existed in config parsing
- but the runtime export block had been lost from `CellUniverse.cpp`

This export was restored.

Current behavior:

- it runs after localization/calibration
- before PCA shape fitting

Output location:

- `output/post_localization/<frame>/`

YAML switch:

- `export_post_localization_images`

---

## 7. Current Relevant YAML Knobs

### Adaptive pooling

- `adaptive_cube_pooling_enabled`
- `adaptive_cube_pooling_cube_size_scale`
- `adaptive_cube_pooling_zero_threshold`
- `adaptive_cube_pooling_majority_threshold`
- `adaptive_cube_pooling_strong_peak_threshold`
- `adaptive_cube_pooling_remove_isolated_bright_cubes`
- `adaptive_cube_pooling_isolated_bright_cube_threshold`
- `adaptive_cube_pooling_min_chunk_size_scale`
- `adaptive_cube_pooling_post_max_scale`

### Signal-center preprocessing and grouping

- `export_signal_center_images`
- `signal_center_pooling_cube_scale`
- `signal_center_pooling_mode`
- `signal_center_min_cube_brightness_delta`
- `signal_center_min_bright_surrounding_cubes`
- `signal_center_recursive_top_percentile`
- `signal_center_recursive_max_depth`

### Post-localization export

- `export_post_localization_images`

---

## 8. Current Diagnostic Log Fields

### Adaptive pooling logs

Current adaptive pooling logs can include:

- `cubeSize`
- `zeroThreshold`
- `majorityThreshold`
- `strongPeakThreshold`
- `meanCubes`
- `maxCubes`
- `strongPeakOverrideCubes`
- `cleanupThreshold`
- `minChunkSizeScale`
- `minChunkCubeCount`
- `cleanupCandidateCubes`
- `removedSmallChunks`
- `removedSmallChunkCubes`

### Signal-center logs

Current signal-center logs can include:

- `cubeSize`
- `scale`
- `pooling`
- `grid`
- `minDelta`
- `recursiveTopPercentile`
- `recursiveMaxDepth`
- `keptBoxes`
- `initialGroups`
- `surroundingZeroThreshold`
- `minBrightSurroundingCubes`
- `removedDarkSurroundCenters`
- `mergedCenterGroups`
- `clusters`

---

## 9. Notes On Current Tuning Behavior

### 9.1 Why some visually small pooled chunks can survive

The chunk cleanup threshold is based on cube count, not raw visual appearance.

So a small-looking cluster can survive when:

- the effective `minChunkCubeCount` is low
- the chunk cube count is equal to, not less than, the threshold

The current comparison removes:

- `chunk_size < minChunkCubeCount`

not:

- `chunk_size <= minChunkCubeCount`

### 9.2 Why noisy centers can appear even when background is near zero

Key causes identified during this session:

- `signal_center_pooling_mode: max`
- low `signal_center_min_cube_brightness_delta`
- small or only moderately large signal-center cubes
- recursive splitting enabled

This combination preserves local peaks and can fragment broader bright structure into multiple centers.

---

## 10. Build Status

The project was rebuilt successfully after the final changes in this archive with:

```bash
cmake --build build -j $(nproc)
```

