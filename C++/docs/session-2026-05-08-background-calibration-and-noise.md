# 2026-05-08 Background Calibration, Signal Map, and Noise Updates

This note records the implementation changes made while debugging auto
calibration, runtime background initialization, signal-map cost, synthetic
background noise, and PCA body-pixel selection.

## Motivation

The original auto-calibration path coupled two behaviors:

- measuring frame-0 background/cell intensity anchors, and
- remapping raw/preprocessed intensities into a clamped `[0, 1]` range.

That made it hard to use the useful part of calibration, the background
measurement, without also changing the image dynamic range. In practice, that
could make the entire frame look white or shift downstream thresholds. The
desired behavior is:

- keep auto-calibration measurement available,
- keep dynamic-range remap optional,
- allow runtime `_backgroundValue` to be initialized separately,
- avoid computing signal maps that will never be used,
- let synthetic background noise match the real background when requested, and
- prevent PCA shape fitting from treating noisy background as cell body.

## Config Knobs

Added or made explicit in `simulation:`:

```yaml
calibrated_preprocess_dynamic_range_enabled: false
auto_calibrate_runtime_background_enabled: true

pca_shape_use_local_bg_estimation: true
pca_shape_bg_shell_inner_scale: 1.2
pca_shape_bg_shell_outer_scale: 1.5
pca_shape_bg_percentile: 0.50
pca_shape_bg_sigma_k: 2.0

synth_background_noise_enabled: true
synth_background_noise_mode: random_gaussian
synth_background_noise_cell_mask_expand_factor: 1.2
synth_background_noise_scale: 1.0
synth_background_brightness_factor: 1.1
synth_background_noise_seed: 12345

global_thread_cap: 10
opencv_thread_cap: 1

pca_bridge_shape_fit_shortcut_enabled: true
```

Defaults in `SimulationConfig` stay conservative:

- `calibrated_preprocess_dynamic_range_enabled = false`
- `auto_calibrate_runtime_background_enabled = false`
- `synth_background_noise_enabled = false`
- `synth_background_brightness_factor = 1.0`
- `pca_shape_bg_sigma_k = 2.0`
- `global_thread_cap = 0`
- `opencv_thread_cap = 1`
- `pca_bridge_shape_fit_shortcut_enabled = false`

The active YAML currently enables runtime background initialization and
empirical synth background noise, but leaves calibrated dynamic-range remapping
disabled. It also caps OpenMP at 10 threads and OpenCV at 1 thread to avoid
nested OpenMP/OpenCV overscheduling on the current 10-core / 20-logical-core
machine.

## Auto-Calibration Split

`auto_calibrate_brightness_enabled` still controls whether frame-0 brightness
anchors are measured from the initial CSV ellipsoids.

`calibrated_preprocess_dynamic_range_enabled` now separately controls whether
those anchors are used to remap intensities:

```text
raw -> (raw - bg_anchor) / (cell_anchor - bg_anchor), clamped to [0, 1]
```

When the dynamic-range knob is off, calibration may still measure and log
anchors, but preprocessing follows the legacy/global path and preserves the
current intensity scale.

Important behavior:

- percentile frame-intensity normalization is skipped only when calibrated
  dynamic-range remap is actually active,
- `ImageHandler::preprocessLoadedFrame` dispatches to the calibrated path only
  when the dynamic-range knob is on and calibration is valid,
- `ImageHandler::applyCalibratedPreprocess` preserves input intensities when
  the dynamic-range knob is off.

## Runtime Background Initialization

`auto_calibrate_runtime_background_enabled` is independent from dynamic-range
remapping. When true, frame 0 initializes the runtime synth/PCA background from
the preprocessed frame's non-cell background pixels before `loadImageStacks()`
regenerates the synth frame.

Implementation details:

- `estimateAdaptiveBackgroundFromStack()` estimates background from pixels
  outside expanded initial cell masks.
- `shouldInitializeRuntimeBackground()` gates frame-0 initialization.
- The initialization runs in both `prepareFrame()` and
  `preprocessAllFramesAlignedToMinimumBackground()` when frames are loaded into
  memory.
- The log line is:

```text
[Runtime Background Init] frame=<absolute_frame> background=<value> source=preprocessed_frame_background
```

This means frame-0 `_backgroundValue` no longer has to remain `0.0` just
because dynamic-range remapping is off.

## Signal Map Gating

The signal map in this code path is the blur-based perturb-guidance map, not
the separate cell-center guidance data.

Signal-map stack construction is now skipped unless it can affect optimization:

```cpp
simulation.signal_map_enabled &&
simulation.signal_map_perturb_guidance_enabled
```

This avoids cloning and blurring real frames when the map will not be consumed
later. Signal-center debug/export behavior remains separate.

## Empirical Synthetic Background Noise

Synthetic frames can now start from a deterministic noisy background instead
of a flat `_backgroundValue`.

When `synth_background_noise_enabled` is true and
`synth_background_noise_mode` is `empirical_inpaint`:

1. Real background pixels are sampled outside expanded cell masks.
2. The real background mean is subtracted to form a residual texture.
3. Cell-masked pixels are filled deterministically from sampled residuals.
4. Synth rendering starts from `_backgroundValue + residual`.
5. The result is clamped to `[0, 1]`.

This affects both full synth rendering and fast partial synth rendering, so
optimization cost and synth export see the same background model.

The model is deterministic via `synth_background_noise_seed`.

`synth_background_noise_mode: random_gaussian` is also supported. It samples
real non-cell background pixels, estimates their robust noise level via MAD,
then fills synth background pixels with deterministic zero-mean Gaussian noise:

```text
sigma = 1.4826 * median(abs(background_pixel - background_median))
noise = gaussian(0, sigma * synth_background_noise_scale)
```

This mode does not copy the real residual texture; it only matches the real
background noise level statistically in synth space. Current YAML uses
`random_gaussian`.

If `synth_background_noise_enabled` is false, synth frames fall back to a flat
background. This does not disable PCA's real-image noise-aware cutoff, because
that cutoff is measured from the real preprocessed frame.

`synth_background_brightness_factor` multiplies the runtime synth background
before either flat-background rendering or empirical residual rendering:

```text
synth_background = clamp(_backgroundValue * factor, 0, 1)
```

This lets the synthetic background be intentionally brighter than the measured
real/runtime background while preserving the optional empirical residual
texture. Current YAML uses `1.1`.

## PCA Body-Pixel Selection

PCA shape fitting was selecting too much background as cell body when local
background was noisy. The old adaptive cutoff was effectively:

```text
local_background_percentile + 0.02
```

With `pca_shape_bg_percentile = 0.50`, this meant median background plus a
small margin, which could admit random noisy background voxels.

The cutoff is now noise-aware:

```text
max(
  local_background_percentile + 0.02,
  local_background_median + pca_shape_bg_sigma_k * robust_background_sigma + 0.02
)
```

`robust_background_sigma` is estimated from MAD:

```text
1.4826 * median(abs(value - local_background_median))
```

The PCA log now includes:

```text
median=<...> robust_sigma=<...> sigma_k=<...> adaptiveCutoff=<...>
```

Tuning guidance:

- Increase `pca_shape_bg_sigma_k` first if cells absorb noisy background.
- Increase `pca_shape_bg_percentile` if the local background shell still has
  too many bright non-cell pixels.
- Disable `pca_shape_use_local_bg_estimation` only to fall back to the older
  global `_backgroundValue + 0.02` behavior.

## Export Bit Depth Note

Current behavior observed during this session:

- real output export respects `simulation.export_bit_depth`,
- synth output still converts float synth images to `CV_8U` with scale `255`,
- with the current YAML `export_bit_depth: 8`, real and synth exports both end
  up 8-bit,
- if `export_bit_depth` is changed to `16`, real export becomes 16-bit but
  synth export remains 8-bit unless that path is updated separately.

## Global Thread Cap

Preprocessing uses OpenMP loops over slices/cells, and individual OpenCV calls
such as `GaussianBlur` may also use an internal thread pool. Without caps, a
20-logical-core machine can overschedule by running OpenMP workers and nested
OpenCV workers at the same time.

The YAML now exposes:

```yaml
global_thread_cap: 10
opencv_thread_cap: 1
```

`main.cpp` applies these caps immediately after YAML load and CLI resume
overrides, before any frame probing, auto-geometry, auto-calibration, or
preprocessing begins.

Behavior:

- `global_thread_cap > 0` calls `omp_set_dynamic(0)` and
  `omp_set_num_threads(cap)`.
- `global_thread_cap: 0` leaves OpenMP at its default.
- `opencv_thread_cap > 0` calls `cv::setNumThreads(cap)`.
- `opencv_thread_cap: 0` leaves OpenCV at its current/default thread setting.

Recommended for the current i9-10850K system:

```yaml
global_thread_cap: 10
opencv_thread_cap: 1
```

Use `global_thread_cap: 20` only when maximizing throughput is more important
than leaving CPU headroom for the desktop or other processes.

## PCA Bridge Shape-Fit Shortcut

Inspection showed that the black PCA bridge logic still existed as
`Frame::discoverPcaBridgeProposal()`, but the old shape-fit-time shortcut had
been removed. The end-of-frame PCA shape-fit block explicitly noted that bridge
discovery had moved to the normal split pre-pass before the perturb/split loop.

The shortcut is restored behind:

```yaml
pca_bridge_shape_fit_shortcut_enabled: true
```

Behavior:

- runs immediately after end-of-frame PCA shape fitting,
- scans newly fitted elongated cells for a black/dark bridge,
- creates a bridge proposal using the same `discoverPcaBridgeProposal()` path,
- injects that proposal into `trySplitCellPhased()`,
- still requires the normal burn-in, bio, overlap, bridge, and image-cost
  validation gates to accept the split.

The intent is to prevent a false-long PCA fit with a real dark bridge from
being captured into the next frame's snapshot/reference state before a split is
attempted.

## Tests Updated

Coverage added or adjusted for:

- calibrated preprocessing remap enabled/disabled behavior,
- legacy preprocessing dispatch when dynamic-range remap is disabled,
- default config values for the new knobs,
- default config values for thread-cap knobs,
- default config values for synth-background factor and bridge shortcut knobs,
- synth background brightness factor render behavior,
- random Gaussian synth background noise behavior,
- empirical synth background noise in full and fast synth rendering,
- calibration test stability with explicit ellipsoid bounds.

Verified with:

```bash
cmake --build build -j 20
ctest --test-dir build -j 20 --output-on-failure
```

Result: all 55 tests passed.

## Main Files Touched

- `config/config.yaml`
- `includes/ConfigTypes.hpp`
- `includes/Frame.hpp`
- `src/CellUniverse.cpp`
- `src/Frame.cpp`
- `src/ImageHandler.cpp`
- `tests/calibration_test.cc`
- `tests/config_types_test.cc`
- `tests/frame_test.cc`
- `tests/process_image_test.cc`
