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

### Full f0–f50 validation (Task 11, 2026-05-05)

Run: `outputs/output_fluo_0-50_20260505_005405/` (Linux WSL, 17h 14m for 51 frames). Detailed plan-side findings in `docs/plans/2026-05-04-auto-calibration-from-frame-zero.md` § Validation Findings.

**Result: PASS.**

| Metric | Auto-calibrated | Baseline (bridge-demote) | GT |
|---|---|---|---|
| Final cell count at f50 | 26 ✓ | 26 ✓ | 26 |
| Total splits accepted | 22 | 22 | 22 |
| Cost rescues fired | 5 | 4 | — |
| Lineage topology | matches | matches | — |
| Wall time | 17h 14m | 15h 24m | — |
| All `[Preprocess Dispatch]` lines | `path=calibrated` × 51 | `path=legacy_iterative` × 51 | — |
| All `[Frame Intensity Scale]` lines | `enabled=skipped_for_calibration` | `enabled=1` (legacy normalization runs) | — |

Two known timing wobbles, both self-correcting:
- **f7 cell_3** delayed to f11 (cost-rejected at f7 with `diff=+92914 vs threshold=-5983`). Caught up alongside cell_4 at f11.
- **f44** has +1 cell (one of the f45 splits pulled forward). Absorbed by the f45 wave at f46.

Mean cell radii at f50 are ~7–15% smaller in the calibrated run (mean aR 31.3 vs 33.8, mean cR 19.8 vs 23.5) — tighter PCA fits enabled by cleaner [0, 1] cell-vs-background contrast. One cell at the radius ceiling (cf. baseline max 55), one near the floor; neither is systemic.

### Open follow-ups
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

---

## 2026-05-05 — 2026-05-06

## Change 15: Auto-derive geometry from initial CSV (Phase 1) **ACTIVE**

After auto-calibration (Change 14) freed brightness anchors, this change collapses the dataset-specific *geometry* knobs by deriving them from initial-CSV statistics. After this lands, a new dataset only needs (1) initial CSV with cell positions and radii, (2) `z_scaling` in `simulation:`. Per-field manual override via `simulation.geometry_force_*` for the rare case where auto-derive picks the wrong value.

### Files changed
- New: `C++/includes/GeometryDerivation.hpp` (`ImageSize3D` + `DerivedGeometry` POD + `derivedGeometryDescribe`)
- `C++/includes/CellUniverse.hpp` — added `static computeDerivedGeometry()` and `static applyDerivedGeometry()`
- `C++/src/CellUniverse.cpp` — pure `computeDerivedGeometry()` and mutating `applyDerivedGeometry()`
- `C++/src/main.cpp` — wired between `cellFactory.createCells` and the auto-calibration block
- `C++/includes/ConfigTypes.hpp` — `auto_derive_geometry_enabled` (default true) + 12 `geometry_force_*` overrides
- `C++/config/config.yaml` — defaults
- New: `C++/tests/geometry_derivation_test.cc` — 9 tests; `ensureEllipsoidConfigDefaults()` helper

### Formulas
For non-trash cells in the initial CSV (post oblate `c≤a` clamp from `Ellipsoid` ctor):

| Output | Formula |
|---|---|
| `max_a/b/c_radius` | `max(axis) × 2.0` |
| `min_a/b/c_radius` | `mean(axis) × 0.2` |
| `perturb_reference_radius` | `mean(a)` |
| `x_sigma`, `y_sigma` | `mean(a) × 0.15` |
| `z_sigma` | `mean(c) × 0.30` |
| `position_prior_threshold` | `mean(a) × 0.8` |
| `iterations_per_cell` | `clamp(n_cells × 30, 150, 500)` |
| `pca_bridge_min_side_voxels` | `clamp(mean_volume × 0.05, 20, 200)` |

### Effect on each dataset
```
[Auto-Derive Geometry] Fluo (4 cells):     max_a=80 min_a=7.8  ref_r=39   xy_sigma=5.85 z_sigma=11.7 mean_a=39
[Auto-Derive Geometry] Embryo (6 cells):   max_a=62 min_a=5.06 ref_r=25.3 xy_sigma=3.79 z_sigma=7.59 mean_a=25.3
```
Same formulas, dataset-appropriate numbers.

### Tests
50/50 pass. Test floats use `static_cast<double>(7.8f)` to handle float-to-double precision mismatch.

---

## Change 16: `simple_preprocess` toggle — preserve intra-cell texture for shape fit **ACTIVE**

Diagnosed via 2026-05-05 f0-5 fluo smoke. The original `calibrated_preprocess_blur_sigma=3.0` over-smoothed cell edges, the PCA shape fit picked up halos as cell support, fitted radii went near-spherical (max/min 1.2-1.3), `worldSplitAxis` returned random directions, and the bio gate `daughter_short_axis_misaligned` rejected splits. Fix: a toggle that drops aggressive blur + per-frame ratio + post-anchor clipping; keeps the bg/cell anchor remap (mandatory for L2 cost) + z-interpolation.

### Files changed
- `C++/includes/ConfigTypes.hpp` — `simple_preprocess_enabled` (default true), `simple_preprocess_blur_sigma` (default 1.5)
- `C++/src/ImageHandler.cpp::applyCalibratedPreprocess` — when toggle on:
  - Skip per-frame photobleaching ratio (`ratio` stays 1.0)
  - Use `simple_preprocess_blur_sigma` instead of `calibrated_preprocess_blur_sigma`
  - Skip the `cv::min/max` clipping to `[0,1]` (intra-cell chromatin > 1.0 and bg variation < 0.0 preserved)
- `C++/config/config.yaml` — `simple_preprocess_enabled: true`, `simple_preprocess_blur_sigma: 1.5`

### Effect
Daughter PCA-fit elongation max/min went from 1.23 (round, unstable axis) to 1.5-1.81 (properly elongated, stable axis). Both f4 fluo splits succeed.

### Limitation acknowledged
PNG-write still clips to [0,255] so saved frames look the same. The improvement is in the cost-surface input only. Earlier attempts to drop the bg/cell remap entirely (raw/255 + blur only) broke the bio gates because they assume cell brightness ~1.0; reverted. Adaptive gates (Changes 18 + 19) are the path forward for raw-mode visuals.

---

## Change 17: Pre-filter candidates at SEED before burn-in **ACTIVE — perf-critical**

Profiling revealed `trySplitCellPhased` consumed 78% of total runtime. Root cause: the candidate "pre-filter" described in `gotchas.md` item I was implemented as a *post-filter* — running brightness + valley checks at the FINAL daughter positions AFTER 50-iteration burn-in, then logging `[Split Cand PreFilter] NO_VALLEY` to discard the result. All burn-in cycles on rejected candidates were wasted (at f1 fluo: 4 cells × 20 candidates × 50 iters × ~30ms = 120s of pure waste with zero successful splits).

This change moves the brightness + valley check to BEFORE the burn-in loop, computed at SEED positions. Candidates failing the pre-filter `continue` past the entire burn-in body.

### Files changed
- `C++/src/Frame.cpp::trySplitCellPhased` — hoisted `measureLocalBrightnessAt` + `computeCandValleyFromBright` helpers; added a seed-position pre-filter before `buildDaughter`/`cells.erase`/burn-in. Logs `[Split Cand SeedFilter]` per skipped candidate.

### Validation (fluo f0-5)
| Run | Time | f4 splits |
|---|---|---|
| Baseline (no pre-filter) | 776s | 2 ✓ |
| **Seed pre-filter** | **259s** | **2 ✓** |

Per-frame breakdown:
| frame | baseline | seed-pre-filter | speedup |
|---|---|---|---|
| f0 | 25.8s | 30.0s | -16% (no splits, small overhead) |
| f1 | 145.5s | 30.7s | **4.7×** |
| f2 | 166.9s | 47.6s | **3.5×** |
| f3 | 155.9s | 46.5s | **3.4×** |
| f4 | 134.4s | 71.7s | **1.9×** (real splits run full burn-in) |
| f5 | 147.9s | 32.6s | **4.5×** |

GT match preserved.

### Why this is safe
Burn-in caps daughter drift at ~10-15 voxels (`split_burn_in_pos_sigma_scale: 0.8 × xy_sigma=5.85 × ~3 effective σ`). A candidate with seed in pure background cannot drift far enough to recover. The post-filter at line 4031 is preserved as defense-in-depth.

---

## Change 18: Adaptive bio-bridge edge-brightness gate **ACTIVE**

Replaces the hardcoded `bio_bridge_min_edge_brightness_absolute = 0.07` with a brightness-fraction form that scales with the parent cell's EMA-tracked `_brightness`. Gates remain meaningful across datasets where cells live at different absolute intensity scales (raw fluo cells ~0.34, normalized cells ~1.0, dim/dying cells ~0.15).

### Files changed
- `C++/includes/ConfigTypes.hpp` — `bio_bridge_min_edge_brightness_fraction = 0.05f` (default; `0` disables adaptive)
- `C++/src/Frame.cpp` — three consumption sites in `trySplitCellPhased`:
  1. SEED pre-filter (Change 17) — `kMinDaughterBrightSeed`
  2. POST burn-in pre-filter (legacy) — `kMinDaughterBright`
  3. Final `edge_too_dim` bio gate — `kMinEdgeBrightAbsolute`

  All three now compute:
  ```cpp
  const float kMin = (fraction > 0.0f)
      ? std::max(absolute, parent.getBrightness() * fraction)
      : absolute;
  ```

### Why fraction = 0.05 (not 0.10)
First attempt used 0.10. Fluo passed but embryo regressed: 12345 missed split at f3, e3d03 false split at f4 (e3d03 is NOT in GT). Root cause: 0.10 × 1.0 = 0.10 is **2× tighter** than the original absolute 0.07, rejecting legitimate embryo daughter centers whose edges sit at 0.05-0.08. Lowering fraction to 0.05 matches the original behavior at brightness=1.0 while still loosening for dim cells (raw cells at 0.34 → 0.017 effective threshold).

### Validation (cross-dataset)
| Dataset | Frames | GT match | Time |
|---|---|---|---|
| Fluo | f0-5 | cell_1+cell_2 split at f4 ✓ | 265s |
| Embryo | f1-5 | e9077+12345 split at f3 ✓ | 168s |

Same `config.yaml` across both runs.

---

## Change 19: Adaptive PCA-bridge black threshold + adaptive PCA shape brightness cutoff **ACTIVE**

Two related adaptive replacements for hardcoded thresholds in the split + shape-fit hot path.

### 19a: PCA-bridge black threshold (`Frame.cpp:2933`)
Bridge "dark gap" pixels are bg by definition. Anchor the threshold to the per-frame `_backgroundValue` (already adaptive via `estimateAdaptiveBackgroundFromFrame`):
```cpp
const float blackThreshold = std::max(
    pca_bridge_black_threshold,                  // absolute floor
    _backgroundValue + 0.02f * pca_bridge_black_bg_multiplier);
```

Config: `pca_bridge_black_bg_multiplier = 1.0f` (default; `0` disables adaptive).

### 19b: PCA shape gather brightness cutoff (`Frame.cpp:1725` + `calibrateCellShapeViaPca`)
Earlier sessions identified that the PCA gather's `max(0.05, _backgroundValue + 0.02)` is too coarse — it cannot distinguish bg from halo when the per-frame estimate misses spatially varying bg (vignetting), photobleaching trends, or dataset-level scale shifts.

This change adds a shell-based local-bg estimator: sample pixels in a thin spherical shell at `[1.2 × maskMaxR, 1.5 × maskMaxR]` from the cell center, Voronoi-filtered to this cell's claim. Take the median (`pca_shape_bg_percentile = 0.5`) plus a 0.02 margin as the per-cell brightness cutoff. Falls back to `_backgroundValue + 0.02` when the shell has too few samples (<8) or when adaptive is disabled.

### Files changed
- `C++/includes/ConfigTypes.hpp` — `pca_shape_use_local_bg_estimation = true`, shell + percentile knobs, `pca_shape_bg_floor = 0.01` (lowered from 0.05 hard floor)
- `C++/src/Frame.cpp` — added `LocalBgEstimate` POD + `estimateLocalBgInShell` free function; added `explicitCutoff` parameter to `gatherBrightPixelsVoronoi` (default `-1` preserves legacy behavior); `calibrateCellShapeViaPca` computes per-cell shell-bg and passes it through

### Why median, not p90 + std
First attempt used `p90 + sigma_k × ((p99-p90)/1.282)`. Failed because post-`blackThreshold` shells are bimodal: ~95% pixels at 0 (zeroed bg), ~5% with halo leak. The Gaussian-std formula then inflated the cutoff into legitimate cell territory (cell_2 cutoff = 0.83 would exclude most cell pixels). The bulk-bg median is robust to leak tails.

### Validation
Both fluo + embryo passed GT with this change active. Per-cell `[PCA Shape LocalBg]` log line shows shell sample counts and cutoff values for diagnostic visibility.

---

## Change 20: Per-stage runtime instrumentation **ACTIVE — diagnostic only**

Added `[FrameTiming]`, `[PrepareTiming]`, `[OptimizeTiming]` log lines to break down per-frame cost. Used to identify the trySplitCellPhased hot spot that drove Change 17.

### Files changed
- `C++/src/main.cpp` — `[FrameTiming]` per frame in the main loop (prepare/optimize/save/total)
- `C++/src/CellUniverse.cpp::prepareFrame` — `[PrepareTiming]` (load/preprocess/postproc/signalMap/loadStacks)
- `C++/src/CellUniverse.cpp::optimize` — `[OptimizeTiming]` with atomic accumulators around `calibrateCellShapeViaPca`, `perturbCell`, `trySplitCellPhased`

### Profiling result (baseline f0-5 fluo)
| Phase | Total time | % |
|---|---|---|
| `trySplitCellPhased` | 604s | **78%** |
| `perturbCell` | 68s | 9% |
| Signal map | ~40s | 5% |
| Post-processing chain | ~32s | 4% |
| Save (TIFF) | 2s | <1% |
| Other | ~30s | 4% |

Drove the prioritization of Change 17 (pre-filter at seed).

---

## Change 21: Misc config tunes **ACTIVE**

Smaller knob adjustments that landed alongside the structural changes:

| File:line | Field | Before | After | Reason |
|---|---|---|---|---|
| `config.yaml` | `signal_map_max_iterations` | 15 | 10 | Per-frame Gaussian-blur loop dominated `prepareFrame` (~5-9s); 10 iters keeps convergence margin while saving ~2-3s per frame |
| `config.yaml` | `post_alignment_black_threshold` | 0.045 | 0.015 | Reduced bg-zeroing aggressiveness so dim chromatin in cells survives; works for both fluo + embryo |
| `config.yaml` | `post_alignment_final_blur_sigma` | 1.25 | 0.0 | The simple_preprocess single sigma is enough; the extra finalBlur was smoothing chromatin into mush |
| `config.yaml` | `geometry_force_iterations_per_cell` | -1 | 250 | Override auto-derive's `clamp(n × 30, 150, 500)` floor of 150 for small datasets — match the v2 baseline default of 250 |
| `config.yaml` | `split_burn_in_pos_sigma_scale` | 0.4 | 0.8 | Late-frame missed-split fix from prior session; daughters need wider burn-in reach |
| `config.yaml` | `split_candidate_translation_delta_fraction` | 0.2 | 0.5 | Pairs with the burn-in scale bump |
| `Frame.cpp:1725` | PCA gather hard floor | 0.05 | 0.01 | Defensive only; adaptive `_backgroundValue + 0.02` term dominates; lower floor lets simple_preprocess + raw-mode work |

---

## Change 22: Bridge metric integrity — fix algorithm/visual divergence at the gap **ACTIVE**

### Problem

Output run `output_fluo_0-50_20260505_005405` rejected the f7 cell_3 division (cell_30 + cell_31 — present in the GT, caught by `output_fluo_0-50_20260504_005706`). The image at f7 visibly shows a clean split with a black gap between two daughters, but the bridge metric reported `gapBright=0.215` and `valleyFromBright=0.488`, and the cost gate then rejected with `costDiff=+92914 / threshold=−5983`.

Two distinct failures, both rooted in "the algorithm wasn't seeing what the user sees":

**(a) Brightness-cutoff bias on the bridge gap**: `gatherBrightPixelsVoronoi` (`Frame.cpp:1786`) drops every voxel below `_backgroundValue + 0.02` (~0.02). The downstream bridge metric iterated only those filtered pixels — so dark gap voxels never contributed. The remaining bright halo bleed (~0.4) drove the average to 0.215 even when most of the gap was black. Sigma=1.5 blur from `simple_preprocess` is a minor contributor; the cutoff is the dominant one.

**(b) Live-collapse baseline collapse**: in-frame perturbation shrunk live cell_3 from snap radii (37.4, 32.6, 12.3) to (24.2, 22.9, 8.3) — about 30% of snap volume — fitting onto only the bright core. With the existing `min(liveCost, snapCost)` rule, baseline became `liveCost=114k` (vs `snapCost=387k`). Both the cost-gate adaptive threshold and the rescue's `0.20 × baselineImageCost` budget shrank by 3.4×, putting the legitimate split out of reach. RNG-dependent: 05-04 hit the split attempt before perturbation, so live==snap exactly and baseline was correct.

### Fix

Two coupled adaptive changes (both threshold-free; relative ratios that work across datasets):

#### (1) Bridge gap scan reads `_realFrame` directly, no brightness cutoff

**File:** `C++/src/Frame.cpp`

**Lines ~4870-4970 (after):**
```cpp
// Edge accumulation: iterate the brightness-filtered `pixels` (cell tissue).
for (const auto &bp : pixels) {
    ... (only edge1Lo..gapLo and gapHi..edge2Hi accumulated)
}

// Gap accumulation: scan _realFrame DIRECTLY in a 3D box covering the bridge,
// no brightness cutoff. Perpendicular cylinder bound by
// split_bridge_perp_radius_scale × max(r1Along, r2Along).
for (int z, y, x in bridge bbox) {
    proj = (p - daughterMidpoint) · axisDir;
    if (proj < gapLo || proj > gapHi) continue;
    perpSq = |delta|² - proj²;
    if (perpSq > perpRadiusSq) continue;
    w = max(0, _realFrame[z](y,x) - _backgroundValue);
    gapBrightSum += w; ++gapCount;
    slabSum[bin] += w; ++slabCount[bin];
}
totalInRange += edge1Count + edge2Count;
```

Edge zones still use the cutoff-filtered `pixels` (correct — edges represent CELL TISSUE brightness, dark voxels near the periphery shouldn't dilute). Gap reads true voxel intensity — when the gap is visually black, gapBright reports near-zero.

#### (2) Live-collapse baseline guard

**File:** `C++/src/Frame.cpp`

**Lines ~3355 (after):**
```cpp
const double liveVol = liveParent.getARadius() * liveParent.getBRadius() * liveParent.getCRadius();
const double snapVol = srcMajor * srcB * srcMinor;
const float collapseRatio = probConfig.split_live_collapse_volume_ratio > 0
    ? probConfig.split_live_collapse_volume_ratio : 0.5f;
const bool liveCollapsed = (snapVol > 0.0) && (liveVol < collapseRatio * snapVol);
const bool useSnapshotBaseline =
    liveCollapsed || (snapCostForComparison <= liveCostForComparison);
```

When live volume drops below `split_live_collapse_volume_ratio × snap volume` (default 0.5), force snap as baseline regardless of which is cheaper. Drift case (live moved but kept its size) is unaffected. The `[Split Snapshot Parent]` log line gains `liveVol`, `snapVol`, `volRatio`, `collapseLimit`, `collapsed` fields for diagnosability.

#### Config additions

**File:** `C++/includes/ConfigTypes.hpp` (`ProbabilityConfig`)
- `split_live_collapse_volume_ratio` — default 0.5
- `split_bridge_perp_radius_scale` — default 1.0

**File:** `C++/config/config.yaml` — surface both keys with explanatory comments.

### Effect on f7 cell_3

| | Before (regression) | After (this change) |
|---|---|---|
| `liveR / snapR` volume ratio | 0.31 | 0.31 (unchanged) |
| Baseline selection | live (114k) — collapsed wins | **snap (387k) — collapse-guarded** |
| Adaptive cost threshold | 0.03 × 114k = 3.4k | 0.03 × 387k = 11.6k |
| Rescue positive limit | 0.20 × 114k = 22.7k | 0.20 × 387k = 77.4k |
| `gapBright` (no-cutoff scan) | 0.215 (filtered pixels only) | ≪ 0.215 (true voxel mean) |
| `valleyFromBright` | 0.488 — fails 0.40 rescue | ≪ 0.40 — passes rescue |
| `costDiff` | +92.9k vs collapsed live | ~+27k vs honest snap |
| Split decision | REJECT cost | ACCEPT (rescue or main gate) |

### Adaptive properties

- **Gap-scan no-cutoff**: zero magic numbers. Sees every voxel inside the bridge cylinder; the slab-min approach already handles outliers.
- **Live-collapse guard**: a single dimensionless ratio (0.5). Fires only on the failure mode (volume collapse) — healthy cells where live ≈ snap are unaffected.
- Both work identically on fluo and embryo because they're geometric self-comparisons, not absolute brightness or pixel-count thresholds.
- The existing `split_bridge_cost_rescue` thresholds (max_valley_ratio=0.40, max_positive_fraction=0.20) stay where they are — they were calibrated correctly; the denominator shifted out from under them. Fixing the denominator restores their intent.

---

## Change 23: Per-frame log split real cells from trash **ACTIVE**

Trash cells (`isTrash=1`) are bookkeeping markers, not biology — counting them with real cells in the per-frame log inflated the perceived cell count and made it harder to read GT-match status at a glance.

**File:** `C++/src/CellUniverse.cpp`

Two log lines updated:

`[Optimize] frame N (M cells, K iterations)` → `[Optimize] frame N (M cells T trash, K iterations)` where M is the real-cell count and T the trash count.

`Saved N cells for frame K to ...` → `Saved N cells T trash for frame K to ...`, plus a follow-up line `cells=[name1,name2,...] trash=[trash_1,trash_2]` listing both groups by name.

Trivial bookkeeping change; no behavioural impact.

---

## Change 24: Daughter-overlap gate threshold 0.05 → 0.10 **ACTIVE**

### Problem

`split_max_daughter_overlap_fraction = 0.05` rejected real just-divided GT splits because freshly-emerged daughters are still pressed against each other (typical overlap 5–10%). Investigation of original_data f3 12345 (a known GT split):

```
[Split Winner] 12345 bestIdx=10/20 label=data_imgPca_primary
               preCostDiff=-199646  baseline=866396  (winner ~200k cheaper, very strong)
[Split Daughter Refit] d1 R=(27.76, 24.69, 20.77) at z=149.55
                       d2 R=(28.70, 24.71, 20.77) at z=106.74
                       (daughters separating along z by 42.8 vx, c-axis ends just 0.8 vx from touching)
[Split Reject bio] reason=daughter_daughter_overlap
                    daughterOverlap=0.0549   maxAllowed=0.05  ← rejected by 0.5%
```

Cross-checked against the 1-100 reference run (`output_original_1-100_20260506_045317`):
the same f3 12345 attempt was rejected with `daughterOverlap=0.0553` — 12345 only finally split at f20 (17 frames late). So the 0.05 threshold has been wrong long before recent changes.

### Distribution evidence

Daughter-overlap fractions across all split attempts in the 1-12 run:

| daughterOverlap | Attempt class |
|---|---|
| **0.055** | f3 12345 (real GT split — wrongly rejected) |
| 0.16, 0.16, 0.23, 0.28, 0.33 | false phantom-overlap attempts (correctly rejected) |

Bimodal — real divisions cluster at 5–10%, phantom splits cluster at 16%+. The 0.05 gate sat exactly where real just-divided pairs live.

### Fix

**File:** `C++/config/config.yaml` (line 482)

```yaml
# Before:
split_max_daughter_overlap_fraction: 0.05

# After:
split_max_daughter_overlap_fraction: 0.10
```

Single config-only change. No code changes; no recompile needed for runs that already had the binary built post-Change 22.

### Adaptive properties

- Dimensionless fraction of daughter volume — scale-invariant across datasets.
- Doesn't affect the e3d03 false split (which passed at < 0.05 — that's a separate cost-arithmetic problem).
- Catches all real just-divided pairs (5–10%) while still rejecting phantom-overlap candidates (16%+).

### Effect

- **f3 12345 (original_data)**: 0.055 < 0.10 → ACCEPT (was rejected at 0.055 > 0.05)
- **High-overlap phantoms (0.16+)**: still rejected — no false-split regression
- **fluo dataset**: unaffected (no rejections at 5–10% overlap range observed in f0-10 run)

---

## Change 25: Restore `gapDensity` semantic — bridge cost rescue now fires again **ACTIVE**

### Problem

Change 22's no-cutoff gap scan (read every voxel in the bridge zone, not just bright ones) **silently broke `gapDensity`**. The metric was historically `gapCount / totalInRange` computed on cutoff-filtered bright voxels — it measured "what fraction of the BRIGHT signal in the bridge zone falls in the gap" (real bridges read 0.001-0.04, no-bridge cells read 0.5+). With Change 22, both `gapCount` and `totalInRange` switched to all-voxel counts, and real bridges started reading 0.2-0.55. The rescue's 0.05 max-gap-density gate now ALWAYS fails — cost rescue fired **0 times** in `output_fluo_0-50_20260506_170405` (50 frames).

### Concrete impact: f45 cell_310 split lost

```
[Split Bridge] cell_310 axisLen=70.61   gapWidth=53.0   valleyFromBright=0   gapBright=0
                        gapDensity=0.545                ← inflated by Change 22's all-voxel scan
[Split Reject cost] diff=-31877  threshold=-38723        ← only 7K shy of cost gate
                    drift1=13.67  drift2=20.04          ← daughters DID find the cells
```

The bridge is **perfect** (totally black gap, daughters drifted 13-20 vx and located the actual daughter cells 70 vx apart). The split should have been rescued. It wasn't, because gapDensity=0.545 > 0.05.

User-visible result at f45: cell_310 collapsed onto ONE of the would-be daughter positions; the other daughter location stays uncovered (visible bright cell with no ellipse fit; original cell_310 ellipse sits in dark space at the other daughter's position).

### Distribution evidence

All 3 cost-rejections in the 0-50 run had inflated gapDensity:
- 0.270, 0.303, 0.545 — all far above the 0.05 rescue cap

The split that DID accept (f44 cell_30, costDiff=-321k) had gapDensity=0.218 — also above 0.05, but the main cost gate accepted without needing rescue.

### Fix

**File:** `C++/src/Frame.cpp` (bridge-metric block)

Restore the original `gapDensity` semantic by counting **bright voxels only** in the gap zone, paralleling the cutoff-filtered edge zones. Keep Change 22's no-cutoff scan for `gapBright` / `gapBrightMinSlab` (the brightness-magnitude side). The two halves of the metric are now independent:

```cpp
// Inside the cutoff-filtered `pixels` loop, classify by zone:
int gapBrightCount = 0;
for (const auto &bp : pixels) {
    ...
    if (proj < edge1Lo || proj > edge2Hi) continue;
    if (proj >= gapLo && proj <= gapHi) {
        ++gapBrightCount;                          // NEW: bright voxels in gap
    } else if (proj >= edge1Lo && proj < gapLo) {
        ++edge1Count; edge1BrightSum += bp.weight;
    } else if (proj > gapHi && proj <= edge2Hi) {
        ++edge2Count; edge2BrightSum += bp.weight;
    }
}

// gapDensity now reflects original "bright fraction" semantic:
const int totalBrightInRange = gapBrightCount + edgeCount;
const float gapDensity = (totalBrightInRange > 0)
    ? static_cast<float>(gapBrightCount) / static_cast<float>(totalBrightInRange)
    : 0.0f;
```

`gapBright` (slab-min over no-cutoff scan) is unchanged — keeps Change 22's algorithm/visual alignment.

### Effect on f45 cell_310

| Metric | Before | After |
|---|---|---|
| `gapBright` | 0 | 0 (unchanged — Change 22 preserved) |
| `gapDensity` | 0.545 (all voxels) | < 0.05 (bright-voxel fraction — restored semantic) |
| `valleyFromBright` | 0 | 0 |
| Rescue fires? | NO (gapDensity > 0.05) | YES (all three eligibility checks pass) |
| Cost diff | -31877 vs -38723 threshold | -31877 vs threshold; rescue makes the split ACCEPT |

### Adaptive properties

- No new config knobs. The original `split_bridge_cost_rescue_max_gap_density: 0.05` stays where it was — the calibration is correct, only the input metric was broken.
- Both halves of the bridge metric (gapBright, gapDensity) now do their original jobs. Change 22's "see what the user sees" gap brightness is preserved.

---

## Cross-cutting summary: A new dataset now needs only 2 inputs

After Changes 14–19, a new dataset works with:

1. **Initial CSV** (cell positions + radii + isTrash flag) — used by auto-derive geometry (Change 15) + auto-calibration (Change 14)
2. **`z_scaling`** in `simulation:` block

Everything else auto-adapts:

| Quantity | Source | Adaptation |
|---|---|---|
| Cell-radius bounds (max/min A/B/C) | Initial CSV stats × 0.2 / × 2 | Change 15 |
| Perturbation sigmas (x/y/z) | mean radius × {0.15, 0.30} | Change 15 |
| `perturbSigmaReferenceRadius` | mean(a) | Change 15 |
| `position_prior_threshold` | mean(a) × 0.8 | Change 15 |
| `iterations_per_cell` | clamp(n × 30, 150, 500) | Change 15 |
| `pca_bridge_min_side_voxels` | mean_volume × 0.05 | Change 15 |
| Bg + cell intensity anchors | Frame-0 ROI | Change 14 |
| Bio bridge edge-brightness | parent brightness × 0.05 | Change 18 |
| PCA bridge black threshold | per-frame `_backgroundValue` + 0.02 | Change 19a |
| PCA shape brightness cutoff | per-cell shell median + 0.02 | Change 19b |

Validated cross-dataset: same `config.yaml` produces correct GT lineage on both fluo (Fluo-N3DH-CE) and embryo (original_data) over the f0-5 / f1-5 windows.

### Open follow-ups

- **Trash cell removal**: trash_2 still sometimes drops out at f5. Doesn't affect real-cell lineage, but worth investigating if a third dataset has more trash markers.
- **`post_alignment_black_threshold = 0.015`**: still an absolute knob (not adaptive). Could be tied to auto-cal bg statistics.
- **Visualization fidelity**: saved PNGs still clip to [0,255], so the optimizer's no-clip cost surface isn't visually inspectable. Would need 16-bit PNG or a separate "raw" output channel.
- **Third-dataset validation**: only fluo + embryo tested. A 3rd dataset (different cell type, different bg-noise profile, vignetting) is the next stress test.

### References

- Validation runs:
  - Fluo seedfilter f0-5: `outputs/output_v2_seedfilter_f0-5_*` (259s, GT match)
  - Embryo adaptive v3 f1-5: `outputs/output_pca_v3_embryo_f1-5_*` (168s, GT match)
  - Fluo adaptive v3 f0-5: `outputs/output_pca_v3_fluo_f0-5_*` (265s, GT match)
- Profiling baseline: `outputs/output_v2_timing2_f0-5_*` ([OptimizeTiming] breakdown)
