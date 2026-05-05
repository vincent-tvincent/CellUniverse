# Changelog v9 — Auto-calibration era

**Status:** ACTIVE (opened 2026-05-04)
**Predecessor:** [`changelogv8.md`](./changelogv8.md) — closed at Change 13b (prolate-shape gate port)
**Branch baseline:** `jl_auto_calibrate_05042026` off `jl_bridge_demote_05042026`

This volume opens with the auto-calibration-from-frame-zero work. Goal: collapse ~35 dataset-specific iterative-preprocessing tunables to zero by inferring the brightness scale from frame-0 cell positions. Same code now runs on multiple datasets without retuning.

---

## 2026-05-04: Auto-calibration from frame zero (Change 14) — **status: ACTIVE**

### Problem / Motivation

The legacy iterative preprocessing pipeline (`processPreparedSequence` + `Frame Intensity Scale` + `IterPreprocess`) had ~35 dataset-specific tunables that needed re-tuning for every new microscopy dataset. Symptoms reported by the project supervisor: cells sometimes disappeared after preprocessing on new datasets, and the per-dataset tuning effort was the main blocker for using CellUniverse as a generic tracker.

The plan: anchor the brightness scale to actual biology by measuring median intensity inside vs outside the initial-CSV cell ellipsoids on frame 0. All bio thresholds (`bio_bridge_min_edge_brightness_absolute`, `pca_bridge_black_threshold`, etc.) are already in fractional [0, 1] form; they become portable when "1.0" consistently means "median cell brightness" across all datasets. Plan file: [`docs/plans/2026-05-04-auto-calibration-from-frame-zero.md`](../plans/2026-05-04-auto-calibration-from-frame-zero.md).

### Files changed

- `C++/includes/CalibrationTypes.hpp` (new)
- `C++/includes/CellUniverse.hpp`
- `C++/includes/ConfigTypes.hpp`
- `C++/includes/ImageHandler.hpp`
- `C++/src/CellUniverse.cpp`
- `C++/src/ImageHandler.cpp`
- `C++/src/main.cpp`
- `C++/config/config.yaml`
- `C++/CMakeLists.txt` (re-enable disabled test target)
- `C++/tests/calibration_test.cc` (new)
- `C++/tests/process_image_test.cc` (added 5 tests + fixed a stale assertion that masked an unrelated regression)
- `C++/tests/frame_test.cc` (purged stale field references)
- `C++/tests/config_types_test.cc` (purged stale field references + 1 new auto-calibration test)
- `C++/tests/spheroid_rotation_test.cc` (purged stale field references)
- `C++/tests/spheroid_test.cc` (purged stale field references)
- `C++/tests/CMakeLists.txt` (registered new test file)

### Code changes

#### 1. `BrightnessCalibration` value type — `C++/includes/CalibrationTypes.hpp`

New header. Plain struct + free `calibrationDescribe()` helper.

```cpp
enum class BrightnessCalibrationSource {
    Uninitialized,
    AutoFrameZero,
    ManualOverride,
};

struct BrightnessCalibration {
    float background_intensity = 0.0f;
    float cell_intensity = 0.0f;
    std::size_t cell_pixel_count = 0;
    std::size_t background_pixel_count = 0;
    BrightnessCalibrationSource source = BrightnessCalibrationSource::Uninitialized;

    bool isValid() const {
        // Span-only validity: a default-constructed calibration has
        // cell=bg=0 and is invalid. Source is metadata, not an invariant.
        return (cell_intensity - background_intensity) > 1e-3f;
    }

    float normalize(float raw) const {
        if (!isValid()) return raw;
        const float span = cell_intensity - background_intensity;
        const float t = (raw - background_intensity) / span;
        return std::clamp(t, 0.0f, 1.0f);
    }
};

inline std::string calibrationDescribe(const BrightnessCalibration &cal);  // body in header
```

#### 2. Config fields — `C++/includes/ConfigTypes.hpp`

Added 5 new `SimulationConfig` fields:

```cpp
bool auto_calibrate_brightness_enabled = true;     // master switch
float manual_background_intensity = -1.0f;          // sentinel: -1 = auto
float manual_cell_intensity = -1.0f;
float calibration_cell_inner_fraction = 0.7f;       // sample inside 70% of each radius
float calibration_pixel_trim_percent = 0.10f;       // drop top+bottom 10% before median
```

YAML reads + cout dumps added in the same `explodeConfig` / `printConfig` blocks. Three deprecation comment headers placed above the now-bypassed `iterative_*`, `contrast_*`, and `frame_intensity_scale_*` field groups (fields stay parsed for backward compatibility; values ignored when calibrated path is active).

#### 3. `computeAutoCalibration` — `C++/src/CellUniverse.cpp`

Pure static function, ~60 lines. Pseudocode:

```
if manual_*_intensity > 0 for both:
    return ManualOverride{manual_bg, manual_cell}
if cells empty or rawFrame empty:
    return Uninitialized
zScale = z_scaling   # cell positions live in interpolated z; raw frame is not interpolated
for each (x, y, z) voxel in rawFrame:
    insideAny = ANY cell contains (x, y, z*zScale) at scale=1.0
    insideInner = ANY non-trash cell contains (x, y, z*zScale) at scale=0.7
    if insideInner: cellPixels.push(value)
    if not insideAny: bgPixels.push(value)
if cellPixels empty or bgPixels empty:
    return Uninitialized
return AutoFrameZero{
    background_intensity = trimmedMedian(bgPixels, 10%),
    cell_intensity        = trimmedMedian(cellPixels, 10%)
}
```

Two helper functions in an anonymous namespace:
- `trimmedMedian(values, trim_percent)` — sorts the vector and returns median of the inner [trim, 1-trim] window.
- `pointInsideScaledEllipsoid(cell, scale, x, y, z, zScale)` — applies cell's inverse rotation to (x, y, z*zScale) and tests the unit-sphere inequality with scaled radii. The `zScale` argument was added during integration after a smoke-test bug: cell z is in interpolated space, raw frame z is not, so a multiplier is required.

Backed by 3 unit tests in `calibration_test.cc`:
- `RecoversBackgroundAndCellMediansFromSyntheticFrame` — 100×100×50 synthetic, 50 vs 800, recovers within ±5.
- `ManualOverrideBypassesAutoDetection` — manual values pass through.
- `EmptyCellsReturnsInvalidCalibration` — empty inputs → not valid.

#### 4. `applyCalibratedPreprocess` — `C++/src/ImageHandler.cpp`

New static function. Replaces the iterative gradient walk with three cheap dataset-agnostic steps:

```
linearScaled[i] = clamp((rawFrame[i] - bg) / (cell - bg), 0, 1)   # per voxel
if blur_sigma > 0: gaussian blur each slice
if z_scaling > 1: linearly interpolate z slices
return result
```

Returns `normalizedSlices` unchanged when calibration is invalid (graceful fallback). Logs `[Calibrated Preprocess] background=X cell=Y span=Z bg_pixels=N cell_pixels=M source=auto`.

#### 5. Dispatch overload of `preprocessLoadedFrame` — `C++/src/ImageHandler.cpp`

New 5-argument overload:

```cpp
static std::vector<cv::Mat> preprocessLoadedFrame(
    const std::vector<cv::Mat> &normalizedSlices,
    const std::string &imageFile,
    const BaseConfig &config,
    const BrightnessCalibration *calibration,
    std::ostream *logSink = nullptr);
```

Dispatcher logic:

```
calibratedPath = auto_calibrate_brightness_enabled
                 && calibration != nullptr
                 && calibration->isValid()
if calibratedPath:
    log "[Preprocess Dispatch] file=... path=calibrated"
    return applyCalibratedPreprocess(slices, *calibration, config, log)
log "[Preprocess Dispatch] file=... path=legacy_iterative"
return preprocessLoadedFrame(slices, imageFile, config, log)   # existing 4-arg overload
```

The existing 4-arg `preprocessLoadedFrame` stays in place untouched and is called for the legacy fallback.

#### 6. main.cpp install — `C++/src/main.cpp`

Between `cellFactory.createCells(...)` and `CellUniverse lineage = CellUniverse(...)`:

```cpp
BrightnessCalibration calibration;
if (config.simulation.auto_calibrate_brightness_enabled) {
    if (cells.empty() || imageFilePaths.empty()) {
        std::cerr << "[Auto Calibration] cells or imageFilePaths empty; "
                     "skipping calibration ...\n";
    } else {
        const auto &firstFrameCells = cells.begin()->second;
        std::vector<cv::Mat> rawFrame0 = ImageHandler::loadRawFrame(
            imageFilePaths.front().string(), config);
        calibration = CellUniverse::computeAutoCalibration(
            firstFrameCells, rawFrame0, config);
        std::cout << "[Auto Calibration] " << calibrationDescribe(calibration)
                  << std::endl;
    }
}
// ... CellUniverse construction ...
lineage.setBrightnessCalibration(calibration);
```

#### 7. Plumb calibration through 3 preprocess call sites — `C++/src/CellUniverse.cpp`

The constructor's M2 z_slices probe + the two real-frame preprocess paths (in `prepareFrame` and `preprocessAllFramesAlignedToMinimumBackground`) all updated to pass `&brightnessCalibration` as the new 4th argument.

#### 8. Skip `normalizeStackToFrameIntensity` when calibrated path is active — `C++/src/CellUniverse.cpp`

This was the second integration bug found during the smoke test. The legacy code calls `normalizeStackToFrameIntensity(rawFrame, ...)` BEFORE `preprocessLoadedFrame`, scaling raw to [0, 1] using percentile reference points. But our calibration anchors were measured against the RAW frame (not the percentile-normalized frame). If normalize runs first, `applyCalibratedPreprocess` would treat the [0, 1] input as if it were raw 16-bit, mapping everything to ~0 via the (24, 87)-anchored linear scale.

Fix at both real-frame preprocess sites (CellUniverse.cpp lines ~1933 and ~2191):

```cpp
const bool calibratedActive = config.simulation.auto_calibrate_brightness_enabled
    && brightnessCalibration.isValid();
if (config.simulation.frame_intensity_normalization_enabled && !calibratedActive) {
    normalizeStackToFrameIntensity(...);  // legacy path only
}
```

The constructor M2 probe was deliberately NOT gated — at construction time the calibration is still Uninitialized (it's installed by main.cpp AFTER the constructor returns), so the dispatcher correctly falls back to legacy and the probe runs through the unchanged legacy pipeline. This gives the right z_slices count regardless of which path subsequent frames will use.

#### 9. Test target restoration — `C++/CMakeLists.txt` + 5 test files

`add_subdirectory(tests)` had been disabled since 2026-04-08 because `frame_test.cc`, `config_types_test.cc`, `spheroid_rotation_test.cc`, `spheroid_test.cc` referenced removed `SimulationConfig::background_color` / `cell_color` fields and removed `EllipsoidConfig::flatCellRotationRefineEnabled`.

Repaired in this changelog as Task 0 of the auto-calibration plan:
- 26 stale field references replaced with literal `1.0f`/`0.0f` constants or `Frame::setBackgroundColor()` calls (per the gotchas-rule that these former config fields are now sigmoid-pipeline invariants baked in as literals).
- 2 obsolete `flatCellRotationRefineEnabled` test blocks deleted.
- 1 unrelated stale test (`OutputValuesStayWithinUnitRange`) updated — `ImageHandler::processImage` no longer normalizes to [0, 1] (it preserves source intensity scale per its current docstring), so the test was renamed to `OutputPreservesSourceIntensityScale` and assertions updated. Without this fix the target wouldn't link.
- `add_subdirectory(tests)` re-enabled with a `Tests re-enabled 2026-05-04` comment replacing the disabled-with-explanation comment.

### Effect

**Before (legacy preprocessing per frame):**

```
loadRawFrame → normalizeStackToFrameIntensity → IterPreprocess (100 iter × radius gradient walks)
  → PostAlignmentBlackPercentile/ChunkBlackoff (~6 percentile passes)
  → PostAlignmentTinyParticleRemoval → PostAlignmentFinalBlur
  → z-interpolate
```

~35 dataset-specific tunables. ~110 sec/frame at 26 cells (measured on the bridge-demote baseline). Cells "disappeared" on new datasets where the iterative refinement was tuned wrong.

**After (calibrated preprocessing per frame):**

```
loadRawFrame → applyCalibratedPreprocess (linear scale + blur + z-interpolate)
```

Zero dataset-specific tunables in the active path. ~10–15 sec/frame estimated (validation TODO). One-shot calibration at startup against frame 0 + initial CSV.

**Validation (smoke tests, both datasets, no config retuning between them):**

| Dataset | bg_intensity | cell_intensity | span | First splits caught |
|---|---|---|---|---|
| Fluo (`output_calibrated_smoke3_20260504_235756`) | 24.48 | 86.93 | 62.45 | cell_1 + cell_2 at f4 ✓ |
| Embryo (`output_calibrated_orig_smoke_20260505_001230`) | 108.55 | 121.38 | 12.84 | 12345 + e9077 at f3 ✓ |

Cell counts at every checked frame match GT. Zero false splits on `e3d03` or any other cell. Same code, same config, completely different intensity profiles, both datasets correctly tracked.

### Diagnostic surface

New log tags:
- `[Auto Calibration] background=X cell=Y span=Z bg_pixels=N cell_pixels=M source=auto|manual|uninitialized` — once at startup
- `[Frame Intensity Scale] frame=... enabled=skipped_for_calibration mean=...` — per frame, replaces the legacy frame-intensity log when calibrated path is active
- `[Preprocess Dispatch] file=... path=calibrated|legacy_iterative` — per frame, confirms which path took
- `[Calibrated Preprocess] background=X cell=Y span=Z bg_pixels=N cell_pixels=M source=auto input_slices=I output_slices=O` — per frame when calibrated path runs

Removed log tags from the calibrated path (still appear if you set `auto_calibrate_brightness_enabled: false`):
- `[IterPreprocess] ...` — every gradient-walk iteration
- `[PostAlignmentBlackPercentile] ...`, `[PostAlignmentChunkBlackoff] ...` — percentile passes
- `[PreprocessScores] ...`

### Test results

After all changes: **37 unit tests passing, 0 failing**. New test cases (10 total across the auto-calibration work):

```
BrightnessCalibrationTest.ScaleMapsBackgroundToZeroAndCellToOne
BrightnessCalibrationTest.ClipsValuesOutsideAnchorRange
BrightnessCalibrationTest.IsValidRequiresPositiveSpan
BrightnessCalibrationTest.DescribeIncludesAllFields
ConfigTypesTest.AutoCalibrationFieldsHaveExpectedDefaults
InitializeBrightnessCalibrationTest.RecoversBackgroundAndCellMediansFromSyntheticFrame
InitializeBrightnessCalibrationTest.ManualOverrideBypassesAutoDetection
InitializeBrightnessCalibrationTest.EmptyCellsReturnsInvalidCalibration
CellUniverseCalibrationStorageTest.SetterAndGetterRoundtrip
ApplyCalibratedPreprocessTest.MapsBackgroundToZeroAndCellToOne
ApplyCalibratedPreprocessTest.ClipsValuesAboveCellAndBelowBackground
ApplyCalibratedPreprocessTest.InterpolatesZSlicesWhenZScalingGreaterThanOne
PreprocessLoadedFrameWithCalibrationTest.UsesCalibratedPathWhenCalibrationValid
PreprocessLoadedFrameWithCalibrationTest.FallsBackToLegacyWhenCalibrationNullOrInvalid
ManualOverrideEndToEndTest.ManualValuesPassThroughPreprocessing
```

### Open follow-ups

- **Full f0–f50 fluo validation run on Linux WSL** (Task 11 of the plan). Smoke tests cover f0–f5; the long run validates against the full bridge-demote baseline at `output_fluo_0-50_20260504_005706`.
- **Spatial illumination correction (vignetting).** Currently a single `background_intensity` constant. Some datasets need a per-pixel low-pass background field. Out of scope for this changelog; potential Change 15.
- **Per-channel calibration for multi-channel datasets.** Single channel only today.
- **Eventual removal of deprecated `iterative_*` / `contrast_*` / `frame_intensity_scale_*` config fields** once auto-calibration is validated on 3+ datasets. Removing these will collapse `SimulationConfig` from ~70 fields to ~35.
- **Tighten `calibration_cell_inner_fraction`** if the embryo dataset's narrow span (12.84) proves too low-contrast for late frames. The smoke test passed f0–f6 cleanly so this is observation only.
- **Bypass the constructor's M2 probe through the legacy preprocess path** when calibrated path is active. Currently the probe always runs ~110 sec of legacy iterative preprocessing just to determine `z_slices`, which can be computed analytically from `rawFrame.size()` and `z_scaling`. Cosmetic perf win.

### References

- Plan: [`docs/plans/2026-05-04-auto-calibration-from-frame-zero.md`](../plans/2026-05-04-auto-calibration-from-frame-zero.md)
- Bridge-demote regression baseline: `outputs/output_fluo_0-50_20260504_005706/` (49,491-line debug log; cells.csv through t050.tif)
- Smoke test artifacts:
  - Fluo: `outputs/output_calibrated_smoke3_20260504_235756/`
  - Embryo: `outputs/output_calibrated_orig_smoke_20260505_001230/`
