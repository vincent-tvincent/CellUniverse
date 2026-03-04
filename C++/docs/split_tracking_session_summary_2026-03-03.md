# Split Tracking Session Summary (2026-03-03)

## Scope
This summarizes the debugging and implementation work completed in this session for:
- missed splits vs false positives after rebase
- dim-cell oversize / undersize behavior
- per-cell brightness tracking and recovery logic
- frame-specific validation on `jihang` sample outputs

---

## 1) Main Problems Investigated

1. Missed true split for cell `1f89abf484c94c498a23cad71ebee0cb` at frame 8 in:
- `examples/output_jihang_20260303_063856`

2. False positives / unstable split acceptance caused by threshold and gate interactions.

3. Dim-cell tracking instability:
- cells growing too large after brightness-retry loops
- cells becoming too small when capped too aggressively by frame-1 reference volume

4. Frame-3 split consistency across runs (one run had only 1 split at frame 3; prior run had 2).

---

## 2) Code Changes Implemented

## 2.1 Split gate and diagnostics

- Kept deterministic split evaluation (trial-revert-apply accepted candidates).
- Added detailed skip diagnostics including:
  - `absGate`, `relGate`, `strongAbsOverride`, `diff`, `split_cost`, `rel_gain`, `min_rel_gain`, aspect info.

### Strong-absolute split override (targeted)
In `Lineage::optimize(...)`:
- Added override that can pass relative gate when split cost improvement is very strong:
  - `costDiff < -(split_cost * strong_split_cost_multiplier)`
- Tightened with parent aspect-ratio condition:
  - requires `parent_aspect_ratio >= strong_split_min_aspect_ratio`
- Purpose:
  - allow packed-double-cell cases to split (strong evidence)
  - avoid globally lowering relative gain threshold.

---

## 2.2 Brightness-recovery logic (bounded retry)

Implemented in `Lineage::optimize(...)`:
- Trigger recovery when:
  - non-split cell shrinks too much vs previous frame
  - split daughter shrinks too much vs expected daughter volume
- Recovery loop:
  - reduce brightness by step
  - rerun local perturb optimization
  - stop on success, brightness floor, or max attempts
- Added `[Brightness Recovery]` logs per attempt and final status.

---

## 2.3 EMA brightness update + history

In `updateFrameCellBrightness(...)`:
- Brightness update switched from direct assignment to EMA:
  - `blended = (1-alpha)*old + alpha*measured`
- `alpha` configurable (`brightness_update_ema_alpha`).
- Min/max brightness history remains in each `Spheroid` object and continues updating.

---

## 2.4 Frame-1 reference-volume anchoring + growth cap

Added lineage state:
- `referenceVolumes` map (initialized from frame-1 ground-truth cell volumes).
- On split acceptance, daughters inherit half of parent reference volume.

Added growth cap per cell:
- clamp current volume to:
  - `reference_volume * max_volume_growth_from_initial`
- implemented by scaling radii down when above cap.
- emits `[Volume Cap]` logs.

---

## 2.5 API/config extensions

### Added config fields under `prob`
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
- `strong_split_cost_multiplier`
- `strong_split_min_aspect_ratio`
- `brightness_update_ema_alpha`
- `max_volume_growth_from_initial`

### Frame accessor
- Added `Frame::getBackgroundColor()` for per-frame calibrated floor handling.

---

## 3) Verification and Run Findings

## 3.1 Missed frame-8 split root cause (older run)
In `output_jihang_20260303_063856`, frame 8 for `1f89...`:
- split candidate had strong absolute improvement
- rejected only by relative gate (`rel_gain` slightly below `min_rel_gain`).

## 3.2 Latest run status
Latest verified output folder:
- `examples/output_jihang_20260303_152145`

Observed:
- Frame 3: two splits accepted
  - `12345679342524354354502034985234 -> ...340 + ...341`
  - `e9077677575842b1a2925729fbcfb3a5 -> ...50 + ...51`
- Frame 8: `1f89...` split accepted
- Frame 8 saved 9 cells.

---

## 4) Important Side Effects Seen During Debugging

1. Recovery overshoot can happen:
- previous recovery logic accepted first `volume >= target` without upper-bound control.
- dim cells can jump above intended size.

2. Over-aggressive frame-1 volume cap can underfit blobs:
- example cell near `(x,y) ~ (111,253)` (`e3d034...`) repeatedly hit `[Volume Cap]`.
- indicates cap may be too tight for some cells if initial volume is under-estimated.

3. Tuning response applied:
- raised `max_volume_growth_from_initial` in config to reduce chronic under-sizing pressure.

---

## 5) Current Config Snapshot (examples/config.yaml)

Key values currently set:
- `min_relative_split_gain_base: 0.01`
- `strong_split_cost_multiplier: 30.0`
- `strong_split_min_aspect_ratio: 1.4`
- `brightness_update_ema_alpha: 0.35`
- `max_volume_growth_from_initial: 2.0`
- `enable_brightness_recovery: true`

Note:
- class default for `max_volume_growth_from_initial` remains `1.45` in `ConfigTypes.hpp`; runtime uses YAML override (`2.0`) from `examples/config.yaml`.

---

## 6) Files Touched in This Session

- `includes/ConfigTypes.hpp`
- `includes/Frame.hpp`
- `includes/Lineage.hpp`
- `src/Lineage.cpp`
- `examples/config.yaml`
- `docs/split_tracking_and_brightness_progress_2026-03-03.md` (updated earlier)
- this file: `docs/split_tracking_session_summary_2026-03-03.md`

---

## 7) Remaining Risks / TBD

1. Recovery overshoot control is still heuristic.
- Consider best-attempt rollback (closest-to-target volume) rather than first crossing target.

2. Reference volume is anchored from frame 1.
- Consider slow upward adaptation for cells persistently cap-limited against clear blob evidence.

3. Split acceptance still highly cost-driven.
- Consider adding daughter-evidence quality checks (cluster balance / separation confidence) as secondary gates.

