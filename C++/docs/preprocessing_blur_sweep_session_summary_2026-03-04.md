# Preprocessing Blur-Sweep + Split-Gate Session Summary (2026-03-04)

## Scope
This note summarizes work completed in this session for:
- understanding current image preprocessing logic,
- implementing blur-sweep fusion preprocessing,
- debugging shape instability and split behavior,
- adding small split-gate tolerance,
- validating with yuancen-style runs on frames 1..4.

---

## 1) Baseline Pipeline Analysis (before edits)

### 1.1 Current preprocessing path in code
Reviewed and traced in:
- `src/main.cpp`
- `src/Lineage.cpp`
- `includes/ConfigTypes.hpp`
- `docs/interpolation_and_split_change_summary.md`
- `examples/config.yaml`

Key baseline behavior:
1. Load TIFF stack (`imreadmulti`) and convert slices to grayscale.
2. Interpolate z stack using `z_scaling`.
3. Normalize each slice to `CV_32F` with robust scale from `computeAverageSliceMax(...)`.
4. Apply Gaussian blur (`blur_sigma`) and sigmoid transform.
5. Use calibration zone to estimate background/sigmoid center.
6. Apply frame-to-frame mean alignment (`rescaleStack`) before sigmoid.
7. Recompute processed background after sigmoid.

### 1.2 Historical context checked
From changelog/session docs, prior failed ideas were mostly per-cell brightness/normalization strategies. This session’s preprocessing change remained global frame preprocessing (not per-cell normalization), which is safer with prior history.

---

## 2) Implemented Changes

## 2.1 Added blur-sweep fusion preprocessing

### What was added
- Configurable blur sweep range and step:
  - new `simulation.blur_sweep_step`.
- New modular helpers in `Lineage.cpp`:
  - `buildBlurFactors(...)`
  - `applyBlurFactor(...)`
  - `applySigmoidToStack(...)`
  - `fuseBlurSweepSigmoidStack(...)`
  - `preprocessLinearNoBlur(...)`

### Blur-sweep behavior
For each frame stack:
1. Build blur factors from `[1 .. blur_sigma]` with step `blur_sweep_step`.
2. For each factor:
   - blur each slice (factor 1 means no blur),
   - apply sigmoid.
3. Sum all outputs.
4. Normalize back by averaging (`sum / n`) and clamp to `[0,1]`.
5. Continue downstream flow.

### Wiring change
- Replaced direct single sigmoid pass with fused multi-blur pass in `Lineage::Lineage(...)` preprocessing flow.

Files updated:
- `includes/ConfigTypes.hpp`
- `src/Lineage.cpp`
- `examples/config.yaml`

---

## 2.2 Stabilization patch after first failures

Observed issue after initial blur-sweep integration:
- severe shape/radius instability,
- suspiciously low processed background (around `~0.033`),
- behavior consistent with distorted optimization landscape.

### Applied stabilization
1. Disabled brightness-recovery loop:
   - `prob.enable_brightness_recovery: false`.
2. Kept fused average normalization (`sum / n`).
3. Stopped applying post-sigmoid `processed_background` back into `frameBackground`.
   - processed background still logged for diagnostics.
   - pre-sigmoid calibrated background retained for optimization.

Files updated:
- `src/Lineage.cpp`
- `examples/config.yaml`

---

## 2.3 Relaxed split gate and tolerance

### Gate relaxation
- `prob.split_elongation_threshold` lowered:
  - `1.3 -> 1.15`.

### Added small numerical tolerance
- New config: `prob.split_gate_tolerance` (default and set to `0.001`).
- Applied tolerance to split decision comparisons:
  - elongation threshold check,
  - absolute split-cost gate,
  - relative gain gate,
  - strong-absolute override checks.

Files updated:
- `includes/ConfigTypes.hpp`
- `includes/Frame.hpp`
- `src/Frame.cpp`
- `src/Lineage.cpp`
- `examples/config.yaml`

---

## 3) Runs Performed (Local/Out-of-Sandbox)

All requested validation runs were executed locally with escalated permissions and yuancen-style input path.

### 3.1 Early runs (debug/stopped)
- `examples/output_yuancen_local_f1to4_20260304_075235`
- `examples/output_yuancen_local_f1to4_escalated_20260304_075459`

These exposed instability and were intentionally stopped for diagnosis.

### 3.2 Yuancen output location runs
- `/Volumes/vincent/celluniverse/outputs/output_yuancen_20260304_081130`
- `/Volumes/vincent/celluniverse/outputs/output_yuancen_relaxedgate_20260304_082206`
- `/Volumes/vincent/celluniverse/outputs/output_yuancen_tol_20260304_083103`

Final completed run with tolerance:
- Output: `/Volumes/vincent/celluniverse/outputs/output_yuancen_tol_20260304_083103`
- Log: `/Volumes/vincent/celluniverse/outputs/output_yuancen_tol_20260304_083103/run.log`
- Time: `568.098 seconds`
- Frames run: `1..4`

---

## 4) Key Findings from Split Analysis

## 4.1 Frame 3 accepted splits (final run)
Accepted:
- `12345679342524354354502034985234`
- `e9077677575842b1a2925729fbcfb3a5`

## 4.2 Why similar candidates can differ
Main remaining discriminator is usually **relative gain gate** (`min_relative_split_gain_base`), not elongation.

Even with good elongation and negative cost diff, many candidates are still skipped when:
- `rel_gain << min_rel_gain`.

This explained accepted-vs-skipped differences in logs.

## 4.3 User-confirmed correctness
User confirmed that two skipped cells in discussion were correct skips.

---

## 5) Files Changed in This Session

- `includes/ConfigTypes.hpp`
- `includes/Frame.hpp`
- `src/Frame.cpp`
- `src/Lineage.cpp`
- `examples/config.yaml`
- `docs/preprocessing_blur_sweep_session_summary_2026-03-04.md` (this file)

---

## 6) Current Config-Relevant State

Notable active settings in `examples/config.yaml` at end of session:
- `simulation.blur_sigma: 10.0`
- `simulation.blur_sweep_step: 1.0`
- `prob.enable_brightness_recovery: false`
- `prob.split_elongation_threshold: 1.15`
- `prob.split_gate_tolerance: 0.001`

---

## 7) Practical Conclusion

- Blur-sweep fusion is integrated and modular.
- Keeping fused normalization (`sum / n`) is necessary; removing normalization is not recommended.
- Post-sigmoid processed background should not be blindly fed back as frame background in this setup.
- Small numeric tolerance helps borderline gate behavior, but major split acceptance still depends on relative-gain thresholds.

