# Interpolation and Split-Detection Change Summary

## Scope
This note summarizes the major code changes and design reasoning around:
- z-slice interpolation in frame loading
- math used for split detection (PCA, thresholds, geometry gates)
- why certain approaches were added, then reverted

Sources reviewed:
- `docs/changelogv1.md`
- `docs/changelogv2.md`
- current implementation in `src/Frame.cpp`, `src/Lineage.cpp`, `src/Cells/Spheroid.cpp`

---

## Current Behavior (as of latest code)

### 1) Z interpolation
- Implemented in `interpolateSlices()` (`src/Frame.cpp`).
- Called from `loadFrame()` (`src/Lineage.cpp`) for TIFF stacks.
- Method: linear interpolation (LERP) between adjacent grayscale slices:
  - `interp = (1 - t) * slice1 + t * slice2`
  - `t = i / (numInterpolations + 1)`
- Target stack size:
  - `numSynthSlices = z_scaling * (numTiffSlices - 1) + 1`

### 2) Preprocessing after interpolation
- Normalize slices to `CV_32F`.
- Gaussian blur (`blur_sigma`).
- Sigmoid intensity mapping:
  - `1 / (1 + exp(-k * (x - c)))`

### 3) Split math (current)
- Split candidates come from `Spheroid::getSplitCells(...)`.
- Bright pixels are collected in an expanded local box (`splitSearchRadius = 3 * maxR`).
- Neighbor exclusion is applied to reduce contamination from nearby cells.
- PCA is run on collected bright-pixel coordinates.
- Split axis comes from the principal eigenvector (after normalization strategy updates in changelog history).
- Daughter placement is centroid-based by projection onto split axis.
- Split acceptance is cost-driven (real vs synthetic frame L2 difference), with gating checks to skip weak or invalid candidates.

---

## Change Timeline and Reasoning

## 2026-02-17: Core split/PCA bug-fix cycle

### What changed
- Fixed PCA pipeline bugs where:
  - PCA was fed incorrect points (too broad / noisy point sets).
  - PCA result was computed but not effectively used for split direction.
  - scaling choices caused axis bias.
- Added and adjusted split validation: constraints, overlap handling, burn-in, cost thresholds.
- Added two-phase optimization structure:
  - Phase 1 perturbation settle
  - Phase 2 split attempts

### Why
- Initial split attempts were frequently rejected or unstable.
- Root issues were:
  - wrong signal for PCA (background/neighbor contamination),
  - axis bias from scaling,
  - evaluating raw, unoptimized daughters against an already-optimized parent.

### Outcome
- Split logic became data-driven (real-image bright-pixel structure + PCA + cost evaluation), instead of random or geometry-only behavior.

---

## 2026-02-21 to 2026-02-23: Stability tuning and reverts

### What changed
- Increased split burn-in iterations.
- Reworked split evaluation order so candidates are assessed more consistently.
- Added neighbor filtering and debug instrumentation.
- Tried wider/minimum PCA search radii and alternate boundary rules, then reverted some as unnecessary or harmful.
- Added pre-optimization radius preservation for split search boundaries.

### Why
- Needed to reduce false negatives (missed true splits) while avoiding false positives (spurious splits).
- Some protective heuristics over-constrained detection; others expanded noise capture too much.

### Outcome
- Kept improvements that increased robustness (pre-opt boundary preservation, better filtering, stronger diagnostics).
- Reverted heuristics that did not improve real behavior.

---

## 2026-02-26 to 2026-02-27: Daughter geometry and gating cleanup

### What changed
- Fixed pre-opt radii side effects that inflated daughter sizes.
- Removed/relaxed hard daughter-daughter overlap rejection when it blocked valid splits.
- Removed inconsistent ellipsoidal inclusion gating in PCA collection and increased search radius to `3x`.

### Why
- Hard geometric gates were preventing plausible splits before cost function evaluation.
- Inconsistent use of current vs pre-opt radii caused contradictory behavior.

### Outcome
- More decisions moved to the objective function (cost comparison), while geometric checks were kept focused on obvious invalid cases.

---

## 2026-03-02: PCA center and normalization refinements

### What changed
- Added iterative centroid recentering for PCA center when drift is detected.
- Added use of pre-optimization position in split detection flow.
- Updated PCA normalization strategy again (per changelog notes) to better preserve split signal.
- Added split elongation threshold gating to skip low-information candidates early.

### Why
- Phase-1 drift could bias PCA center toward one blob, suppressing elongation signal.
- Normalization choice strongly affects whether true split structure appears in PCA eigenvalues.

### Outcome
- Better split candidate quality and runtime efficiency (fewer expensive burn-ins for near-spherical/non-splitting cases).

---

## Key Design Thoughts That Emerged

1. Interpolation is simple by design.
- Linear z interpolation is intentionally minimal and stable.
- Most complexity moved to split detection and optimization, not interpolation.

2. Cost function is the final judge.
- Hard geometric rules are useful as sanity checks.
- But strict early gates often blocked valid splits.
- The L2 real-vs-synth cost is the most reliable acceptance criterion.

3. PCA quality depends more on point selection/normalization than PCA itself.
- Neighbor contamination, center bias, and axis scaling can dominate outcomes.
- Repeated revisions mainly improved input conditioning around PCA.

4. Pre-opt context is valuable but must be scoped.
- Pre-opt radii/position help preserve split evidence after Phase 1 collapse/drift.
- Using pre-opt values in the wrong places (e.g., daughter sizing) introduced regressions.

---

## Practical Takeaway

- Interpolation math: linear LERP between adjacent TIFF slices.
- Split math: PCA on filtered bright pixels + centroid-based daughter placement + cost-based acceptance.
- Most historical churn came from balancing signal quality (true split evidence) against over-constrained rules that suppressed valid splits.
