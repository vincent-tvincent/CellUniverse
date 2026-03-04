# Split Tracking + Brightness Progress Summary (2026-03-03)

## Scope
This document summarizes the debugging and tuning work done in this session for:
- split false positives / false negatives after rebase
- frame-to-frame tracking instability (especially frame 4 -> 5)
- in-progress per-cell brightness tracking feature

It includes:
- what was diagnosed
- what was changed
- what was verified
- what is still TODO

---

## 1) Root-Cause Investigation Timeline

### 1.1 Rebase and history audit
- Confirmed branch rebase event in reflog:
  - rebased `yp_handle_brightness_problem_02282026` onto `jh-split-improvements-02272026` (`d5c77b5` base).
- Compared commit history, docs, and current code behavior.
- Found mismatches between docs and runtime behavior:
  - docs describe deterministic independent split evaluation
  - code had greedy/randomized variations at different points

### 1.2 Key findings from logs and diffs
- `split_elongation_threshold: 4` was over-strict and blocked valid candidates.
- Some local working-tree edits had previously weakened/disabled parts of split validation.
- Large split search windows and overly permissive pixel thresholds in recenter stages can inflate false-positive risk.
- A frame-4 false positive in `output_jihang_20260303_025309` was reproduced and traced.

---

## 2) Split/Tracking Changes Applied

## 2.1 Config threshold normalization
- Changed:
  - `examples/config.yaml`
  - `prob.split_elongation_threshold: 4 -> 1.3`
- Purpose:
  - restore expected sensitivity for real split candidates.

## 2.2 Recenter filter uses frame background (instead of `pixel > 0.0`)
- Wired per-frame background into split logic path:
  - `Frame::trySplitCell(...)` passes `simulationConfig.background_color` into `Spheroid::getSplitCells(...)`.
- Recenter inclusion changed from:
  - `pixel > 0.0`
  to:
  - `pixel > max(pixelThreshold, frameBackground + 0.015f)`
- Purpose:
  - avoid flooding PCA with near-background voxels.

## 2.3 Split search radius
- Restored split search multiplier to `3.0x` in `Spheroid::getSplitCells(...)`.
- Purpose:
  - reduce contamination from unrelated structures.

## 2.4 Phase-2 split evaluation behavior
- Returned to deterministic independent evaluation:
  - evaluate all original cells from same baseline
  - revert each trial
  - apply accepted splits together
- Purpose:
  - prevent cascading daughter re-splitting artifacts.

## 2.5 Added skip diagnostics
- Added `[Split Skip] ...` logs with gate status:
  - `absGate`, `relGate`, `diff`, `split_cost`, `rel_gain`, `min_rel_gain`
- Purpose:
  - make rejection causes explicit in debug logs.

## 2.6 Dynamic relative-gain gating (targeted anti-false-positive behavior)
- Added Phase-1 acceptance-rate-aware gating in `Lineage::optimize(...)`:
  - base `min_rel_gain = 0.01`
  - stricter `min_rel_gain = 0.02` when Phase-1 accepted-perturbation rate is very low (`< 2%`)
- Purpose:
  - suppress split acceptance in stalled optimization states that often produce false positives.

---

## 3) Verification Runs and Observations

## 3.1 `input_2` (frames 1..3) verification
- Achieved target behavior in tested run:
  - frame 2: no false split
  - frame 3: two accepted splits

## 3.2 `input/original_data` (`jihang`) verification
- Problem run analyzed: `examples/output_jihang_20260303_025309`
  - frame 4 had one false positive
  - frame 5 continued with extra splits
- New run with current anti-stall dual gate:
  - `examples/output_jihang_dualgate_20260303`
  - frame 1: 6 cells
  - frame 2: 6 cells
  - frame 3: 8 cells (2 accepted splits)
  - frame 4: 8 cells (no new split accepted)
  - frame 5: 8 cells (no new split accepted)

Note:
- This is validated for the tested ranges/config; broader dataset validation is still required.

---

## 4) In-Progress Feature: Per-Cell Brightness Tracking

This section describes the requested feature work:
- initialize brightness from frame-1 ground truth
- carry brightness through tracking
- update brightness each frame using current real data
- later add retry loop for shrink events

## 4.1 DONE (implemented)

### A) Brightness state added to cell model
- Added per-cell fields in `Spheroid`:
  - current brightness
  - min brightness history
  - max brightness history
  - initialized flag
- Added `brightness` to `SpheroidParams`.

### B) Rendering now uses per-cell brightness
- `Spheroid::draw(...)` now paints with the cell brightness when initialized.
- Falls back to `simulation.cell_color` only if needed.

### C) Brightness propagation across lifecycle operations
- Preserved brightness state through:
  - perturbation-generated cells
  - parameterized cells
  - split daughter creation (daughters inherit parent brightness/history state)

### D) Measurement API
- Added `Spheroid::estimateBrightnessFromFrame(const std::vector<cv::Mat>&)`
  - estimates mean voxel intensity inside current spheroid volume.
- Added `Spheroid::setBrightness(...)` with clamping + history update.

### E) Lineage integration
- Added `Frame::getRealFrame()` accessor.
- Added `Lineage` helpers:
  - `initializeBrightnessFromGroundTruth()`
  - `updateFrameCellBrightness(frameIndex)`
- Constructor now initializes frame-1 cells from real frame.
- `optimize()` now updates brightness for cells at end of frame optimization.

### F) Smoke validation
- Build passes.
- 1..2 frame smoke run confirms logs:
  - `[Brightness Init] ...`
  - `[Brightness Update] frame ... cell=... old=... new=... min=... max=...`

### G) Shrink-triggered brightness recovery (new)
- Added bounded post-optimization recovery in `Lineage::optimize(...)`:
  - detects significant non-split shrink vs previous-frame volume
  - detects split-daughter over-shrink vs expected half-parent volume
  - lowers per-cell brightness in small steps and reruns local perturbation search
  - stops on success, max attempts, or brightness floor near frame background
- Added detailed logs:
  - `[Brightness Recovery] ... status=success|stopped_floor|max_attempts`
  - includes reason (`non_split_shrink` or `split_shrink`), attempt, brightness delta, and volume delta.

### H) Configurable recovery + split gates (new)
- Added configurable knobs in `prob`:
  - `min_relative_split_gain_base`
  - `min_relative_split_gain_strict`
  - `phase1_accept_rate_threshold`
  - `enable_brightness_recovery`
  - `nonsplit_shrink_trigger_ratio`
  - `nonsplit_recover_target_ratio`
  - `split_shrink_trigger_ratio`
  - `split_recover_target_ratio`
  - `brightness_retry_step`
  - `brightness_retry_max_attempts`
  - `brightness_retry_iters_per_attempt`
  - `brightness_retry_background_margin`
- Added `Frame::getBackgroundColor()` to anchor retry floor to per-frame calibrated background.

## 4.2 TBD (not implemented yet)

The following requested behavior is still pending:

1. Optional output persistence
- If desired, extend `cells.csv` to include:
  - current brightness
  - brightness min/max history

2. Configurability cleanup
- Several currently hardcoded values should be promoted to config once behavior stabilizes:
  - recenter brightness margin
  - split search radius multiplier
  - any remaining recovery heuristics that are still fixed in code

---

## 5) Current Status

- Split debugging and instrumentation: active, with meaningful improvements in tested runs.
- Brightness tracking phase (state + init + per-frame update): implemented and running.
- Brightness-based retry-on-shrink logic: implemented with config knobs; dataset-level tuning/validation still in progress.
