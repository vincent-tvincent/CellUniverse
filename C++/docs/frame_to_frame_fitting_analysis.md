# Frame-to-Frame Cell Fitting Analysis

## Context
This note summarizes the analysis from the current discussion about how cells are propagated from frame `t-1` to frame `t`, and where the actual adjustment/fitting happens.

## Key Conclusion
- `copyCellsForward()` is only initialization (warm start).
- The actual update to match the current frame image happens in `optimize()` (primarily Phase 1 perturbation).

## Main Execution Flow
From `main()`:

1. `lineage.optimize(frame);`
2. `lineage.copyCellsForward(frame + 1);`
3. save outputs

Reference:
- `src/main.cpp:221-246`

## Function That Copies Cells
The copy is done here:

- Function: `Lineage::copyCellsForward(int to)`
- File: `src/Lineage.cpp`
- Core line: `frames[to].cells = frames[to - 1].cells;`

Reference:
- `src/Lineage.cpp:1117-1125`

## What `copyCellsForward()` Does and Does Not Do
What it does:
- Copies fitted cell objects from previous frame into next frame's `cells` container.

What it does not do:
- Does not evaluate cost against the new frame.
- Does not perturb/update position, radius, or rotation.
- Does not do split detection.

## Where Cells Are Actually Updated to Current Frame
This happens in `Lineage::optimize(int frameIndex)`:

1. Regenerates synthetic frame from current `frame.cells` seed.
2. Runs Phase 1 perturbation optimization loop.
3. Runs split candidate evaluation and accepted split application.
4. Optional brightness recovery and volume cap.
5. Final synth regeneration and brightness update.

Reference:
- `src/Lineage.cpp:953-984`

### Core Per-Cell Update Mechanism
`Frame::perturb()` performs one Monte Carlo move:

1. Randomly pick one cell.
2. Randomly perturb its parameters.
3. Check validity/overlap constraints.
4. Re-render affected region.
5. Compute cost delta.
6. Accept only if cost decreases (`newCost - oldCost < 0`), otherwise revert.

Reference:
- `src/Frame.cpp:226-274`

## Interpretation
The system uses temporal continuity by warm-starting frame `t` from fitted state at `t-1`, then locally re-optimizing against frame `t` image data. There is no separate explicit motion model in the analyzed path; continuity mainly comes from initialization plus greedy cost-based perturbation updates.

## Per-Iteration Update Behavior (Phase 1)
In Phase 1, each perturbation iteration updates exactly one randomly selected cell while all others are fixed for that iteration.

Reference:
- `src/Frame.cpp:226-274`

Important nuance:
- Selection is random with replacement, so there is no hard guarantee every cell is selected at least once in a frame.
- Total iterations are `num_cells * iterations_per_cell`, so expected picks per cell are `iterations_per_cell`.

## Split Timing Relative to Phase 1
Phase 1 does not assume a split yet. It first fits the current parent-only set.  
Split handling is a separate Phase 2 step in the same frame:

1. Propose a `parent -> (child1, child2)` split.
2. Evaluate split quality against parent baseline.
3. Accept/reject using configured gates.
4. If accepted, apply both daughters and run extra post-split fitting.

Reference:
- `src/Lineage.cpp:971-975`
- `src/Frame.cpp:285-438`

## Parent-to-Daughter Replacement Semantics
When split is accepted, the parent is replaced by both daughters together (not mapped to only one daughter first).

Reference:
- `src/Lineage.cpp` split application helpers used by `optimize()`
- In `Frame::trySplitCell`, trial state also removes parent and inserts both daughters before cost comparison.

## How `child1` and `child2` Are Generated
Implemented in:
- `Spheroid::getSplitCells(...)` in `src/Cells/Spheroid.cpp:364-773`

High-level steps:
1. Collect bright voxels (`rawPoints`) in a local 3D search box using adaptive thresholding and neighbor exclusion.
2. Run PCA on those points.
3. Use PCA principal axis as split direction.
4. Split points by projection sign onto that axis.
5. Use each side's centroid as daughter center.
6. Set daughter radii with volume-halving scale `cbrt(0.5)`.
7. Construct daughters (`name+"0"`, `name+"1"`), inherit parent rotation and brightness history.

## How Daughter Positions Are Computed
Normal case (two non-empty point groups):
- `new_position1 = centroid(group_projection>=0)`
- `new_position2 = centroid(group_projection<0)`

Fallback case (all points on one side):
- use symmetric offsets along split axis around `pcaCenter`.

Reference:
- `src/Cells/Spheroid.cpp:686-750`

## How `split_axis` Is Defined
`split_axis` is the first PCA eigenvector (largest variance direction) of bright voxels in local 3D image space.

Reference:
- `src/Cells/Spheroid.cpp:605-629`

Fallback:
- If PCA is degenerate or support is too weak, a random unit direction is used.
- `src/Cells/Spheroid.cpp:631-655`

Clarification:
- This is data-driven from bright voxels, not automatically the cell's shortest-radius direction.

## How Cell Orientation ("Facing") Is Set
Orientation is represented by `(theta_x, theta_y, theta_z)`.

Behavior:
1. Initialized from input/constructor parameters.
2. During perturbation fitting, theta values are randomly perturbed and accepted/rejected by cost.
3. During split creation, daughters inherit parent theta values directly.
4. After split, daughters can further rotate via perturbation optimization.

References:
- theta perturbation: `src/Cells/Spheroid.cpp:303-313`
- split inheritance: `src/Cells/Spheroid.cpp:756-761`
- rotation-aware drawing: `src/Cells/Spheroid.cpp:226-260`
