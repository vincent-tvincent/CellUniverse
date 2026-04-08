# Current Split Constraints and Why the Early False Small Spheres Disappeared

This note summarizes the split-decision logic on the current pushed branch state, with emphasis on why the obvious false small spheres seen in earlier runs disappeared before frame 21.

Scope:
- This document describes the pushed behavior corresponding to the current branch history up to the recent false-split filtering work.
- It does not include newer local experimental rules that have not been pushed yet.

## Short answer

The early false small spheres were not removed by a later cleanup step. They disappeared because the false splits that would have created them were blocked earlier in the split-decision pipeline.

In other words, the algorithm is not doing:
- create a tiny fake daughter
- detect that it is too small
- delete it

Instead, the current pushed logic is doing:
- build a cleaner split candidate from the image
- reject suspicious split geometry before burn-in
- reject marginal or unstable splits after burn-in

That is why the early chain of fake small spheres was greatly reduced.

## Important background

Every cell in the current CellUniverse model is represented as a `Spheroid`. This does not only mean a flattened pre-split cell. It means every tracked cell object, including:
- a rounder normal cell
- a flattened cell before division
- each daughter cell after a split

So when the code talks about distances between 2 spheroids, it means distances between 2 current cell models.

## Why the old small spheres appeared

In the older behavior, a false split could happen early, especially when a single real cell contained uneven internal bright structure. Once that parent cell was incorrectly split into 2 daughters, one daughter could later shrink toward the minimum allowed size and visually become a small attached sphere.

This means many of the old small spheres were not independent noise objects. They were downstream products of an earlier false split.

That is the key reason why reducing false splits also reduced the number of visible small spheres.

## The current pushed split pipeline

The current pushed branch blocks false small spheres through 3 main stages.

### 1. Cleaner split-candidate construction inside `Spheroid::getSplitCells(...)`

The split candidate is built from bright voxels in the real image, but the current code no longer lets PCA see any bright voxel too freely.

The important protections are:

1. A brightness threshold is computed inside the parent spheroid:

```cpp
float pixelThreshold = meanBrightness + 0.5f * stddevBrightness;
```

2. Bright voxels are only kept if they are inside an expanded ellipsoidal gate around the parent region.

3. Bright voxels are rejected if they are closer to a neighboring cell than to the current parent candidate.

4. If the collected bright-voxel centroid drifts too far away from the parent center, the code recenters once and recollects the point cloud.

This stage matters because many old false splits came from dirty PCA input. If PCA sees neighboring bright structure, far-field bright noise, or strongly biased internal structure, it can invent a fake 2-lobed split candidate. Cleaning this input already removes many bad proposals before any accept/reject decision is made.

Relevant code:
- `Spheroid::getSplitCells(...)` in `C++/src/Spheroid.cpp`
- `SplitDiagnostics` in `C++/includes/Spheroid.hpp`

## 2. Pre-burn-in heuristic rejection inside `Frame::trySplitCell(...)`

After `getSplitCells(...)` returns 2 daughter candidates and split diagnostics, the current pushed code rejects several suspicious split patterns before burn-in.

These rules live in:
- `shouldRejectSplitPreBurnIn(...)` in `C++/src/Frame.cpp`

The pushed rules are:

### A. `overlapping_daughters`

Reject if:

```cpp
diag.separationOverDaughterMajor < 1.0f
```

Meaning:
- the 2 proposed daughters still overlap too strongly
- this is a classic false-split pattern where 2 overlapping daughters overfit 1 large dim cell

### B. `weak_geometry`

Reject if:

```cpp
diag.elongationRatio < splitElongationThreshold + 0.20f
&& diag.separationOverDaughterMajor < 1.10f
```

Meaning:
- the current-frame split geometry is too weak
- the candidate only barely looks elongated
- the daughters are not clearly separated enough

### C. `z_axis_internal_structure`

Reject if:

```cpp
diag.axisAbsZ > 0.92f
&& diag.separationOverDaughterMajor < 1.30f
&& diag.driftOverParentMajor > 0.40f
```

Meaning:
- the PCA axis is almost aligned with the z direction
- the bright centroid has drifted a lot from the parent center
- this often looks more like internal bright structure or chromosome artifact than a true biological split

These pre-burn-in rules are the main reason the early “big cell becomes 2 fake daughters” case was blocked.

## 3. Post-burn-in heuristic rejection inside `Frame::trySplitCell(...)`

Even if a split survives the earlier stage and finishes burn-in, the current pushed code still has one more rejection rule before the split is finally allowed to compete on total cost.

This rule lives in:
- `shouldRejectSplitPostBurnIn(...)` in `C++/src/Frame.cpp`

The pushed rule is:

### D. `large_recenter_marginal_gain`

Reject if:

```cpp
diag.driftOverParentMajor > 0.85f && costDiff > -40.0
```

Meaning:
- the bright-voxel centroid had already drifted very far away from the parent center
- but even after burn-in, the split only gives a relatively limited cost improvement

This is meant to block cases where the algorithm is overfitting local internal structure rather than detecting a true split.

## Why this also reduced the false small spheres

This is the most important practical point.

The code does not currently contain a special “delete tiny daughters” rule in the pushed branch.

In fact, the basic geometry constraint still allows small but valid cells as long as they stay within the configured size bounds:

```cpp
return (cellConfig.minMajorRadius <= _major_radius) &&
       (_major_radius <= cellConfig.maxMajorRadius) &&
       (cellConfig.minMinorRadius <= _minor_radius) &&
       (_minor_radius <= cellConfig.maxMinorRadius);
```

And in `config.yaml`, the lower bounds are currently:
- `minMajorRadius: 10`
- `minMinorRadius: 5`

So once a fake daughter already exists, it often shrinks toward that minimum-size range rather than disappearing completely.

That means the reason the early small spheres went away is not that they were filtered after appearing. The reason is that the upstream false split that would have produced them was rejected earlier.

In short:
- old behavior: false split happens first, tiny sphere appears later
- current pushed behavior: false split is blocked, so the tiny sphere is never created

## Why frame 21 was much cleaner

The visible improvement up to frame 21 came from the combination of:

1. cleaner PCA input around the parent cell
2. pre-burn-in rejection of obviously suspicious split geometry
3. post-burn-in rejection of unstable, drift-driven, weakly beneficial splits

This combination was strong enough to stop the earlier false-split cascade that used to begin near frame 3 and then create more fake daughters in later frames.

## What this does not solve yet

The current pushed branch still does not fully solve every false-split pattern.

In particular, later false small spheres can still appear from other modes, such as:
- highly unbalanced splits
- small attached satellite daughters near a larger cell

Those later remaining cases are different from the earlier “2 overlapping daughters inside 1 dim parent” pattern. That is why the current pushed logic already improved the early frames a lot, but did not fully eliminate all later tiny-sphere artifacts.

## Main takeaway

The early false small spheres disappeared because the algorithm now rejects the split events that used to create them. The current pushed branch does this mainly by cleaning the split PCA input and by adding explicit false-split rejection rules before a split is finally accepted.
