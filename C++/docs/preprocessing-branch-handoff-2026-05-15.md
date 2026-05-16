# Preprocessing Branch Handoff - 2026-05-15

Audience: another Codex/LLM coding agent.

Primary task: replace the C++ preprocessing path with the current Python
range-filtered preprocessing logic from:

```text
/run/media/blue-lobster/vincent/pre-processing experiment/range_filtered_preprocessing_pipeline.py
/run/media/blue-lobster/vincent/pre-processing experiment/tiff_stack_loader.py
```

Target C++ project:

```text
/home/blue-lobster/p2/UCI/CS295p/CellUniverse/C++
```

This document is intentionally optimized for LLM agents. Follow it as an
implementation checklist. Do not treat old C++ preprocessing behavior as the
source of truth unless this document explicitly says to preserve it.

## Current Source Of Truth

The Python source of truth at the time this document was written is:

```python
DEFAULT_RANGE_COUNT = 100
DEFAULT_RANGE_PERCENTILE = 99.0
DEFAULT_OCCUPANCY_THRESHOLD_PERCENT = 0.2
DEFAULT_THRESHOLD_PERCENTILE = 90.0
DEFAULT_SIGMA = 6.0
DEFAULT_REAL_RATIO = 0.0
DEFAULT_BIT_DEPTH = 16
DEFAULT_MAX_BRIGHTNESS = 65536
```

Important: these values have been tuned repeatedly. Before coding, re-open
`range_filtered_preprocessing_pipeline.py` and verify the current defaults. If
they differ, implement the code so the defaults come from config and update the
examples in this document.

Current canonical Python pipeline:

1. Load one multi-page TIFF as a 3D grayscale stack shaped `(z, y, x)`.
2. Convert to `float32`, clamp to `[0, max_brightness]`, divide by
   `max_brightness`. Default `max_brightness = 65536`.
3. Compute `range_upper_bound = percentile(nonzero normalized pixels, 99.0)`.
4. Split `[0.0, range_upper_bound]` into `range_count` equal ranges.
5. For each range copy:
   - Lower edge is inclusive.
   - Upper edge is exclusive, except the final range includes the upper edge.
   - Pixels inside the range preserve original normalized brightness.
   - Pixels outside the range become zero.
6. Compute global whole-stack nonzero percent for that range copy:

   ```text
   global_percent = count_nonzero(range_stack) / total_voxels * 100
   ```

7. Exclude the entire range copy if:

   ```text
   global_percent >= occupancy_threshold_percent
   ```

8. Sum all non-excluded range copies into one normalized stack.
9. Apply the current blur blend:

   ```python
   blurred = GaussianBlur2D(each_z_slice, sigma)
   output = combined * real_ratio + blurred * (1.0 - real_ratio)
   ```

   Notes:
   - The blur is 2D per z-slice.
   - No blur is applied across z.
   - Current default `real_ratio = 0.0`, so the output is pure Gaussian blur.
   - Do not copy the older C++ formula that uses `realRatio * rawValue *
     backgroundFactor`; that is not the current Python formula.

10. Compute `final_threshold = percentile(nonzero preprocessed pixels,
    final_threshold_percentile)`.
11. Set pixels `< final_threshold` to zero. Preserve pixels `>= threshold`.
12. Export as 8-bit or 16-bit TIFF by multiplying normalized floats by
    `export_max_brightness` defaulting to `65536`, rounding, and clipping to
    the chosen integer type.

The internal stack consumed by downstream C++ should remain `CV_32F` normalized
values after step 11. Do not rescale the selected signal to `[0, 1]` unless the
user explicitly asks for a non-Python-parity mode.

## Important Behavioral Warnings

### Warning 1: Output brightness is intentionally low in normalized scale

Recent Python outputs contain small integer values after multiplying by 65536.
Examples from the current batch run:

```text
t000: uint8 max=41,  uint16 max=41,  nonzero=1268685
t149: uint8 max=107, uint16 max=107, nonzero=1049609
t249: uint8 max=56,  uint16 max=56,  nonzero=1195486
```

That means the normalized values consumed by C++ are approximately:

```text
41 / 65536  = 0.0006256
107 / 65536 = 0.0016327
56 / 65536  = 0.0008545
```

Existing C++ downstream code often assumes preprocessed real images are closer
to `[0, 1]` with cell bodies much brighter than background. If the new pipeline
is connected directly, downstream thresholds and synthetic brightness scale may
need follow-up tuning. Do not hide this by silently rescaling the new output;
that would no longer match the Python reference.

### Warning 2: Do not run old C++ blackoff after the new final threshold

The Python pipeline already performs its final nonzero percentile threshold.
The current C++ path also has post-preprocess operations:

```text
edge brightness alignment
fixed black threshold
black percentile
chunk blackoff
tiny-particle removal
final blur
calibrated dynamic range path
```

For range mode, skip those old post-processing steps when the new range
pipeline is enabled, unless the user explicitly requests a hybrid. Z
interpolation is not treated as old post-processing in this C++ branch: keep it
configurable with `range_preprocess_interpolate_z` so CellUniverse can still
analyze in interpolated z-space.

### Warning 3: Do not use old C++ load-time blur

`ImageHandler::processImage` currently applies `config.simulation.blur_sigma`
during raw load. The Python reference does not do this. For exact parity:

```text
new range pipeline input must be raw grayscale CV_32F, unblurred, source scale
```

Either set `blur_sigma: 0` in the parity config or refactor `loadRawFrame` /
`processImage` to allow bypassing load-time blur for range preprocessing.

### Warning 4: Percentile semantics must match NumPy

Python uses `np.percentile` default behavior, which is linear interpolation.
The existing C++ helper `computePercentileFromValues` uses a floor index. That
is not equivalent. Implement a new helper for this pipeline:

```text
rank = (n - 1) * percentile / 100
lo = floor(rank)
hi = ceil(rank)
t = rank - lo
value = values[lo] * (1 - t) + values[hi] * t
```

Use this for both the 99th percentile range upper bound and the final nonzero
threshold percentile.

## C++ Files To Inspect First

Read these before editing:

```text
includes/ImageHandler.hpp
src/ImageHandler.cpp
includes/ConfigTypes.hpp
src/CellUniverse.cpp
src/main.cpp
tests/process_image_test.cc
tests/config_types_test.cc
CMakeLists.txt
docs/preprocessing-branch-handoff-2026-05-15.md
docs/pipeline.md
```

Relevant current C++ behavior:

```text
ImageHandler::loadRawFrame(...)
  uses cv::imreadmulti for TIFF
  converts color to grayscale
  converts to CV_32F
  may apply load-time blur_sigma

CellUniverse::prepareFrame(...)
  loadRawFrame
  normalizeStackToFrameIntensity unless calibrated dynamic range is active
  ImageHandler::preprocessLoadedFrame
  old post-processing
  Frame::loadImageStacks

CellUniverse::preprocessAllFramesAlignedToMinimumBackground(...)
  same conceptual load/preprocess path for eager/preprocess-only mode

ImageHandler::preprocessLoadedFrame(...)
  dispatches calibrated/simple/legacy preprocessing
  can z-interpolate and cube-pool

ConfigTypes.hpp
  owns SimulationConfig defaults, YAML parsing, and printConfig output
```

## Recommended Architecture

Add a small isolated range preprocessing module. Prefer one of these layouts:

Option A, easiest:

```text
includes/ImageHandler.hpp
src/ImageHandler.cpp
  add ImageHandler::applyRangeFilteredPreprocess(...)
  add private/static helper functions near existing preprocessing helpers
```

Option B, cleaner:

```text
includes/RangePreprocessor.hpp
src/RangePreprocessor.cpp
CMakeLists.txt
tests/range_preprocessor_test.cc
```

If you choose Option B, keep `ImageHandler` as the IO owner and use
`RangePreprocessor` for pure stack transforms.

Suggested public C++ API:

```cpp
struct RangePreprocessStats {
    float rangeUpperBound = 0.0f;
    int ranges = 0;
    int activeRanges = 0;
    int excludedRanges = 0;
    float finalThreshold = 0.0f;
    std::size_t outputNonzero = 0;
};

static ImageStack applyRangeFilteredPreprocess(
    const ImageStack &rawSlices,
    const BaseConfig &config,
    RangePreprocessStats *stats,
    std::ostream *logSink);
```

Contract:

```text
input:
  rawSlices is CV_32F source-scale grayscale stack
  shape is z slices, each rows x cols

output:
  CV_32F stack with same y/x shape
  values are normalized float brightness after thresholding
  z depth is interpolated when range_preprocess_interpolate_z is true
  no frame deletion
  no old post-blackoff
```

## Add Config Fields

In `includes/ConfigTypes.hpp`, add fields to `SimulationConfig`.

Recommended names:

```cpp
bool range_preprocess_enabled = false;
int range_preprocess_range_count = 100;
float range_preprocess_range_percentile = 99.0f;
float range_preprocess_occupancy_threshold_percent = 0.2f;
float range_preprocess_final_threshold_percentile = 90.0f;
float range_preprocess_sigma = 6.0f;
float range_preprocess_real_ratio = 0.0f;
bool range_preprocess_blur_enabled = true;
bool range_preprocess_interpolate_z = true;
float range_preprocess_max_brightness = 65536.0f;
float range_preprocess_export_max_brightness = 65536.0f;
bool range_preprocess_skip_legacy_postprocess = true;
bool range_preprocess_bypass_load_blur = true;
bool range_preprocess_debug_stats = true;
```

Then update all three places:

1. `SimulationConfig` field declarations/defaults.
2. YAML parsing in `SimulationConfig::explodeConfig`.
3. `SimulationConfig::printConfig`.

Add tests in `tests/config_types_test.cc` proving YAML values parse and print
defaults remain stable.

Suggested YAML block:

```yaml
range_preprocess_enabled: true
range_preprocess_range_count: 100
range_preprocess_range_percentile: 99.0
range_preprocess_occupancy_threshold_percent: 0.2
range_preprocess_final_threshold_percentile: 90.0
range_preprocess_blur_enabled: true
range_preprocess_interpolate_z: true
range_preprocess_sigma: 6.0
range_preprocess_real_ratio: 0.0
range_preprocess_max_brightness: 65536.0
range_preprocess_export_max_brightness: 65536.0
range_preprocess_skip_legacy_postprocess: true
range_preprocess_bypass_load_blur: true
global_thread_cap: 8
opencv_thread_cap: 1
```

For exact Python parity also ensure legacy knobs are disabled or bypassed:

```yaml
blur_sigma: 0.0
frame_intensity_normalization_enabled: false
auto_calibrate_brightness_enabled: false
calibrated_preprocess_dynamic_range_enabled: false
simple_preprocess_enabled: false
simple_preprocess_interpolate_z: false
cube_pooling_enabled: false
edge_brightness_alignment_enabled: false
post_alignment_black_threshold: 0.0
post_alignment_black_percentile: 0.0
post_alignment_chunk_blackoff_enabled: false
post_alignment_tiny_particle_removal_enabled: false
post_alignment_final_blur_sigma: 0.0
post_process_final_blur_sigma: 0.0
```

Do not rely only on YAML disabling. Add code-path guards so the new pipeline
does not accidentally run old post-processing if config defaults change later.

## Implement The Transform Efficiently

Do not literally allocate 100 full range stacks. The Python implementation does
that conceptually, but the ranges are disjoint. Equivalent C++ implementation:

### Step 1: Validate input

```text
- stack not empty
- every slice non-empty
- every slice CV_32F
- all slices same rows/cols
- range_count > 0
- max_brightness > 0
- occupancy threshold >= 0
- percentile values in [0, 100]
- real_ratio in [0, 1]
- sigma >= 0
```

If there are no nonzero pixels after normalization, Python raises an error from
`nonzero_percentile`. C++ should throw `std::runtime_error` with filename/stage
context, or return all zeros only if the user explicitly wants permissive mode.

### Step 2: Clamp and normalize

For every voxel:

```cpp
float v = raw;
if (!std::isfinite(v) || v < 0.0f) v = 0.0f;
v = std::min(v, maxBrightness);
normalized = v / maxBrightness;
```

Preserve stack shape `(z, y, x)`.

### Step 3: Collect nonzero values and compute p99

Collect values where `normalized > 0.0f`.

Use NumPy-compatible linear percentile:

```cpp
float rangeUpperBound = percentileLinear(nonzeroValues, rangePercentile);
```

If `rangeUpperBound <= 0`, throw.

### Step 4: Count occupancy per range

Let:

```cpp
float step = rangeUpperBound / rangeCount;
std::vector<std::size_t> binCounts(rangeCount, 0);
```

For every normalized voxel:

```text
if value <= 0: ignore for count
if value > rangeUpperBound: ignore; it is outside all Python ranges
bin = floor(value / step)
if bin >= rangeCount: bin = rangeCount - 1   // handles value == upper bound
binCounts[bin] += 1
```

Then:

```cpp
const double totalVoxels = depth * rows * cols;
excluded = binCounts[i] > 0 &&
           (binCounts[i] / totalVoxels * 100.0) >= occupancyThresholdPercent;
active = binCounts[i] > 0 && !excluded;
```

Note that Python increments `excluded_ranges` even when a range count is high.
It increments `active_ranges` only when `np.any(range_stack)` and not excluded.

### Step 5: Build the combined stack

Second pass over normalized voxels:

```text
if value <= 0: output = 0
else if value > rangeUpperBound: output = 0
else if bin is active: output = value
else output = 0
```

This is exactly equivalent to summing all selected range copies, because each
nonzero value belongs to at most one range.

### Step 6: Apply blur blend

For each z slice independently:

```cpp
if sigma > 0:
    cv::GaussianBlur(combinedSlice, blurredSlice, cv::Size(0, 0), sigma, sigma);
else:
    blurredSlice = combinedSlice;

preprocessed = combinedSlice * realRatio + blurredSlice * (1.0f - realRatio);
```

Use OpenCV default border behavior to match `cv2.GaussianBlur(..., (0,0),
sigma)` as closely as possible.

### Step 7: Final nonzero percentile threshold

Collect nonzero values from the preprocessed stack.

```cpp
float finalThreshold = percentileLinear(nonzeroPreprocessedValues,
                                        finalThresholdPercentile);
```

Then:

```text
if value < finalThreshold: value = 0
else preserve value
```

Use strict `<`, not `<=`.

### Step 8: Stats and logging

Log one line per frame:

```text
[RangePreprocess] file=t149.tif shape=(35,512,708)
  range_upper_bound=0.0028533935546875
  ranges=100 active_ranges=50 excluded_ranges=44
  occupancy_threshold_percent=0.2
  sigma=6 real_ratio=0
  final_threshold_percentile=90 final_threshold=0.00042002325
  output_nonzero=1049609
```

Keep logs parseable and stable. Future agents will compare these values against
Python reference runs.

## Integrate Into C++ Load/Prepare Flow

Patch both frame preparation paths. Do not update only one.

### Path 1: `CellUniverse::prepareFrame`

Current path:

```text
loadRawFrame
normalizeStackToFrameIntensity
ImageHandler::preprocessLoadedFrame
legacy post-processing
Frame::loadImageStacks
```

Required branch:

```cpp
std::vector<cv::Mat> real_frame = ImageHandler::loadRawFrame(...);

if (config.simulation.range_preprocess_enabled) {
    // Do not call normalizeStackToFrameIntensity.
    // Do not call ImageHandler::preprocessLoadedFrame.
    RangePreprocessStats stats;
    real_frame = ImageHandler::applyRangeFilteredPreprocess(
        real_frame, config, &stats, &preprocessLog);

    if (!config.simulation.range_preprocess_skip_legacy_postprocess) {
        // Optional hybrid mode only. Default should skip.
        runLegacyPostProcessing(real_frame);
    }
} else {
    // Existing behavior unchanged.
}
```

After this branch, keep downstream shared steps:

```text
build signal map if configured
export preprocessed images if configured
Frame::loadImageStacks(real_frame)
runtime background initialization if configured
prepareSignalCentersForFrame
```

But avoid old blackoff/final blur in default range mode.

### Path 2: `CellUniverse::preprocessAllFramesAlignedToMinimumBackground`

Apply the same branch in the eager/preprocess-only path. This function has two
sub-paths depending on edge brightness alignment. For Python parity, when
`range_preprocess_enabled && range_preprocess_skip_legacy_postprocess`, bypass:

```text
edgeBrightnessAlignment
blackThresholdStackAfterAlignment
blackPercentileStackAfterAlignment
adaptBlackPercentileToChunkCount
removeTinyIsolatedParticles
applyFinalPreprocessingBlur
minimum background alignment
```

Still allow:

```text
export_preprocessed_images
loadIntoFrames -> frame.loadImageStacks
signal map building if needed
runtime background initialization if needed
```

### Path 3: M2 probe in `CellUniverse` constructor

The constructor has a probe that loads and preprocesses the first frame to see
post-preprocess z size. Add the same range branch there. The Python reference
does not z-interpolate, so output depth should equal input TIFF page count.

### Path 4: `ImageHandler::loadFrame`

This is a convenience method and currently does not know calibration. Either:

```text
leave it as legacy and do not use it for range preprocess
```

or add a range-preprocess-aware overload. Do not accidentally make it apply
range preprocessing after old normalization unless that is clearly documented.

## Handling Load-Time Blur

Because `ImageHandler::loadRawFrame` calls `processImage`, and `processImage`
may apply `config.simulation.blur_sigma`, exact parity requires one of these:

Preferred code change:

```cpp
Image ImageHandler::processImage(const Image &image,
                                 const BaseConfig &config,
                                 bool applyLoadBlur);

std::vector<cv::Mat> ImageHandler::loadRawFrame(..., bool applyLoadBlur = true);
```

Then range mode calls:

```cpp
loadRawFrame(path, config, &log, !config.simulation.range_preprocess_bypass_load_blur);
```

Simpler config-only fallback:

```yaml
blur_sigma: 0.0
```

Do not rely on the fallback in tests. Add at least one test proving range mode
does not apply load-time blur when `range_preprocess_bypass_load_blur=true`.

## Export Semantics

Current C++ export helpers in `CellUniverse.cpp` use `65535.0f` for 16-bit
float export. Python uses `65536`. This matters for parity.

Recommended implementation: add a range-preprocess export helper or parameter:

```cpp
float exportMax = config.simulation.range_preprocess_export_max_brightness;
converted = round(clamp(value, 0, 1) * exportMax)
clip to uint8 or uint16 max
```

For existing non-range exports, avoid changing behavior globally unless tests
cover it. A one-LSB difference may be acceptable for visual inspection, but the
parity tests should know which rule they use.

Implementation note for this branch: preprocessed-stack export now accepts a
range-specific normalized export scale and passes
`range_preprocess_export_max_brightness` when `range_preprocess_enabled=true`.

When exporting both 8-bit and 16-bit from the same processed stack, compute the
preprocessed float stack once and write both integer versions. Do not rerun the
preprocess for each bit depth.

## Config Migration Strategy

Add a new config preset or profile instead of mutating every existing profile.
For example:

```text
config/config_range_preprocess.yaml
```

Start from the active YAML, then add:

```yaml
range_preprocess_enabled: true
range_preprocess_range_count: 100
range_preprocess_range_percentile: 99.0
range_preprocess_occupancy_threshold_percent: 0.2
range_preprocess_final_threshold_percentile: 90.0
range_preprocess_blur_enabled: true
range_preprocess_interpolate_z: true
range_preprocess_sigma: 6.0
range_preprocess_real_ratio: 0.0
range_preprocess_max_brightness: 65536.0
range_preprocess_export_max_brightness: 65536.0
range_preprocess_skip_legacy_postprocess: true
range_preprocess_bypass_load_blur: true
export_preprocessed_images: true
export_frame_tiff: true
export_bit_depth: 16
quit_after_preprocessing: true
global_thread_cap: 8
opencv_thread_cap: 1
```

Then explicitly disable old preprocessing and blackoff knobs as listed earlier.

## Tests To Add

Add a focused unit-test file, preferably:

```text
tests/range_preprocessor_test.cc
```

Update `tests/CMakeLists.txt` if needed.

Minimum tests:

1. `PercentileLinearMatchesNumpyStyle`
   - Values `[0, 10, 20, 30]`.
   - P50 should be 15, not 10.
   - P25 should be 7.5.
   - P100 should be 30.

2. `RangeBoundarySemantics`
   - Use a tiny stack with values exactly on range edges.
   - Verify lower-inclusive, upper-exclusive behavior.
   - Verify final range includes the upper bound.

3. `OccupancyExcludesWholeRange`
   - Construct values where one range has global percent equal to threshold.
   - Verify `>= threshold` excludes it.
   - Verify another lower-occupancy range survives.

4. `PreservesOriginalBrightnessInsideSelectedRanges`
   - Input nonzero values survive unchanged before blur/threshold.
   - This guards against accidental binarization.

5. `FinalThresholdUsesStrictLessThan`
   - Value equal to threshold survives.
   - Value below threshold becomes zero.

6. `NoShapeChange`
   - Input stack depth/height/width equals output stack depth/height/width.
   - No all-zero frame deletion.

7. `AllZeroStackThrows`
   - Match Python behavior.

8. `NoLoadTimeBlurInRangeMode`
   - If you refactor load blur, prove range mode bypasses it.

9. `ExportScalingPythonBrightness`
   - For a float value `107/65536`, 8-bit export writes 107 and 16-bit export
     writes 107 when export max is 65536.
   - This captures the "8-bit and 16-bit values can be numerically identical
     when max < 255" behavior observed in Python.

## Python Parity Smoke Tests

Use these current Python reference outputs if they still exist:

```text
/run/media/blue-lobster/vincent/pre-processing experiment/fluo_01_range_filtered_pipeline_8bit
/run/media/blue-lobster/vincent/pre-processing experiment/fluo_01_range_filtered_pipeline_16bit
```

Reference input folder:

```text
/run/media/blue-lobster/vincent/celluniverse/fluo-n3dh-ce/data/Fluo-N3DH-CE/01
```

Representative reference stats from current defaults:

```text
t000.tif: shape=(35,512,708), uint8 max=41,  uint16 max=41,  nonzero=1268685
t149.tif: shape=(35,512,708), uint8 max=107, uint16 max=107, nonzero=1049609
t249.tif: shape=(35,512,708), uint8 max=56,  uint16 max=56,  nonzero=1195486
```

If Python defaults change, regenerate these reference outputs before comparing.

Recommended C++ preprocess-only run:

```bash
cd /home/blue-lobster/p2/UCI/CS295p/CellUniverse/C++
cmake --build build -j "$(nproc)"
./build/celluniverse \
  0 249 \
  /run/media/blue-lobster/vincent/celluniverse/fluo-n3dh-ce/data/Fluo-N3DH-CE/01 \
  output/range_preprocess_smoke \
  config/config_range_preprocess.yaml \
  config/initial.csv
```

Adjust binary path and initial CSV to match the local build/profile. The code's
CLI signature is:

```text
celluniverse <firstFrame> <lastFrame> <input_pattern_or_dir_or_file> <output_dir> <config.yaml> <initial.csv>
```

Run with:

```yaml
quit_after_preprocessing: true
export_preprocessed_images: true
export_frame_tiff: true
```

Then compare TIFF stats for `t000`, `t149`, `t249`:

```text
shape
dtype
min
max
nonzero count
mean of all pixels
mean of nonzero pixels
selected active/excluded range counts from logs
final threshold from logs
```

Acceptable parity target:

```text
exact match preferred for counts/max after integer export
small floating tolerance acceptable before export
document any <= 1 LSB export difference if using 65535 instead of 65536
```

## Build And Run Guidance

This project uses CMake and OpenCV.

Use all hardware for compilation:

```bash
cd /home/blue-lobster/p2/UCI/CS295p/CellUniverse/C++
cmake --build build -j "$(nproc)"
```

Avoid runtime oversubscription during preprocessing:

```yaml
global_thread_cap: 8
opencv_thread_cap: 1
```

Reason: range preprocessing can use OpenMP over slices/voxels, and OpenCV
GaussianBlur may create its own thread pool unless capped.

## Suggested Implementation Order

1. Create/checkout a working branch.
2. Re-read Python files and update default values in this document if needed.
3. Add config fields, YAML parsing, printConfig, and config tests.
4. Implement `percentileLinear`.
5. Implement pure stack transform helpers:
   - validate stack
   - normalize raw stack
   - collect nonzero values
   - count range occupancy
   - build combined stack
   - blur blend
   - final threshold
   - stats/logging
6. Add unit tests for helpers.
7. Refactor load-time blur bypass for range mode.
8. Integrate range branch into `prepareFrame`.
9. Integrate the same range branch into `preprocessAllFramesAlignedToMinimumBackground`.
10. Integrate the same range branch into the M2 probe.
11. Add preprocess-only config profile.
12. Build and run unit tests.
13. Run preprocess-only smoke on 3 frames: `t000`, `t149`, `t249`.
14. Compare against Python reference stats.
15. Run full preprocess-only batch only after the 3-frame parity passes.
16. Document any deliberate deviations.

## Acceptance Criteria

The implementation is acceptable only if:

```text
- range preprocessing can be enabled/disabled by config
- legacy preprocessing path still works when disabled
- C++ output stack shape matches input TIFF shape
- no frames are deleted
- inside-range pixels preserve brightness, not binarized
- range exclusion uses global whole-stack nonzero percent
- exclusion rule uses >= threshold
- percentile behavior matches NumPy linear percentile
- old post-blackoff/cube/z-interp does not run in default range mode
- 8-bit and 16-bit export can be produced from the same processed stack
- thread caps are respected
- tests cover helper semantics
- smoke stats for t000/t149/t249 are recorded in docs or logs
```

## Rollback Plan

Keep the new path behind:

```yaml
range_preprocess_enabled: false
```

Do not delete legacy preprocessing during the first implementation. If the new
path destabilizes optimization:

1. Set `range_preprocess_enabled: false`.
2. Restore the previous config profile.
3. Re-run the old preprocess-only smoke.
4. Use the range path only for exported TIFF generation until downstream
   optimizer thresholds are retuned.

## Common Failure Modes

1. Accidentally normalizing twice:
   - Symptom: output almost all zero or far dimmer than Python.
   - Cause: `normalizeStackToFrameIntensity` ran before range normalization.

2. Accidentally using the old C++ simple blend:
   - Symptom: active ranges match but output brightness differs.
   - Cause: copied `rawValue * (realRatio * rawValue * backgroundFactor)`.
   - Fix: use direct blend `raw * real_ratio + blur * (1 - real_ratio)`.

3. Accidentally binarizing ranges:
   - Symptom: selected pixels are all 1 or all dtype max.
   - Fix: preserve original normalized brightness inside selected ranges.

4. Percentile mismatch:
   - Symptom: thresholds are close but not equal to Python.
   - Cause: floor-index percentile instead of NumPy linear interpolation.

5. Old blackoff runs after new threshold:
   - Symptom: nonzero count much lower than Python.
   - Fix: skip old postprocessing in range mode.

6. Load-time blur still active:
   - Symptom: p99/range counts differ before the new blur stage.
   - Fix: bypass `blur_sigma` in range mode or set it to 0.

7. Napari display confusion:
   - Symptom: 8-bit and 16-bit look different even with same pixel values.
   - Cause: different contrast limits.
   - Fix: inspect pixel stats and set contrast limits manually.

## Final Note For The Next Agent

The goal is not to make the C++ image look nicer than the Python output. The
first goal is parity with the current Python range-filtered pipeline. After
parity is proven, downstream C++ optimizer brightness thresholds can be tuned
as a separate task.
