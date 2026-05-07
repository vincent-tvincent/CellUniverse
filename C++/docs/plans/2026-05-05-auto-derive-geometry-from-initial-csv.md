# Auto-Derive Geometry from Initial CSV — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the remaining ~10 dataset-specific geometry/motion knobs by deriving them from initial-CSV statistics + `z_scaling`. After this lands, a new dataset needs only **two user inputs** to run: (1) the initial CSV listing cells in frame 0, (2) `z_scaling` (z-anisotropy of the imaging volume). Everything else auto-derives.

**Architecture:** Add a one-shot derivation pass at startup that runs BEFORE auto-calibration and BEFORE `CellUniverse` construction. It reads cells (from `CellFactory::createCells`) and image dimensions (from `ImageHandler::loadRawFrame` for frame 0 — already loaded for calibration anyway), computes statistics (mean/max/min radii, cell count, mean cell volume), and overrides `Ellipsoid::cellConfig` + `BaseConfig::simulation` + `BaseConfig::prob` fields in place. A per-param manual-override list (YAML) lets users pin specific values when auto-derive does the wrong thing.

**Tech Stack:** C++17, OpenCV (`cv::Mat` for image dimensions), yaml-cpp, GoogleTest. Existing project patterns from `C++/src/CellUniverse.cpp` and `C++/src/main.cpp`.

**Branch:** `jl_auto_geometry_05052026` off `jl_auto_calibrate_05042026` (now pushed to origin).

**Regression baseline:** `C++/outputs/output_fluo_0-50_20260505_005405` — the validated auto-calibrated run. f50 = 26 cells, 22 splits, lineage matches GT. Auto-derived geometry must reproduce ≥ 21/22 splits and identical f50 cell count, with timing wobbles ≤ 2 frames per split event.

**Constraints:**
- Do NOT change any of the existing geometry-related fields' default values in `EllipsoidConfig` or YAML. Auto-derive overrides them at runtime; the YAML defaults remain as fallbacks when `auto_derive_geometry_enabled: false`.
- Do NOT change the calibrated preprocessing path or any bio gate. This plan is a pure geometry-tuning addition.
- Must run BEFORE `CellUniverse` is constructed (the constructor's M2 probe calls into Ellipsoid math that uses `cellConfig.maxZ` etc.). Wire order in main.cpp: load cells → load frame 0 raw → derive geometry → auto-calibrate → construct CellUniverse.
- Per-cell `_brightness` EMA + adaptive background MUST remain untouched.

---

## Files Touched

| File | Role | Change kind |
|---|---|---|
| `C++/includes/GeometryDerivation.hpp` | NEW. Defines `DerivedGeometry` POD struct + `deriveGeometryDescribe()` helper. | Create |
| `C++/includes/ConfigTypes.hpp` | Add `auto_derive_geometry_enabled` flag + `geometry_force_*` per-param manual override fields. | Modify |
| `C++/includes/CellUniverse.hpp` | Add `static DerivedGeometry computeDerivedGeometry(cells, image_size, baseConfig)` declaration. | Modify |
| `C++/src/CellUniverse.cpp` | Implement `computeDerivedGeometry`. Pure function, no member mutation. | Modify |
| `C++/src/main.cpp` | After cell loading, before `CellUniverse` construction: compute derived geometry, apply to `Ellipsoid::cellConfig` and `config.simulation` / `config.prob`. | Modify |
| `C++/config/config.yaml` | Add `auto_derive_geometry_enabled: true` + commented-out manual-override block under `simulation:`. | Modify |
| `C++/tests/geometry_derivation_test.cc` | NEW. Unit tests for `computeDerivedGeometry` against synthetic cell sets representative of fluo + embryo. | Create |
| `C++/tests/CMakeLists.txt` | Register `geometry_derivation_test.cc`. | Modify |
| `C++/docs/changelogs/changelogv9.md` | Append Change 15 entry. | Modify |

---

## Phase Breakdown

- **Phase 0: Fix dim-cell removal in auto-calibration** (Tasks A1–A3) — Per-frame brightness-ratio adjustment of the calibration anchors. Without this, dim cells (e.g. at f48 in the fluo run) clip to 0 and disappear from the optimizer's view. Must ship together with auto-derive so the validation reflects the combined behavior.
- **Phase 1: Foundation** (Tasks 1–4) — `DerivedGeometry` struct, config flag, pure-function `computeDerivedGeometry` with unit tests. No behavior change yet (flag-gated).
- **Phase 2: Apply derived geometry** (Tasks 5–6) — Apply derived values to `Ellipsoid::cellConfig` + `BaseConfig` in main.cpp, with per-field manual-override precedence. Logging of every applied/skipped value.
- **Phase 3: Validation** (Tasks 7–9) — Synthetic-fluo + synthetic-embryo regression tests, then full f0–50 validation against the auto-calibrated baseline. Must show no dim-cell disappearance + same lineage as baseline.
- **Phase 4: Documentation** (Tasks 10–11) — Changelog Change 15 (auto-derive) and Change 16 (per-frame ratio), README/CLAUDE.md update describing the "two-input" workflow.

---

## Phase 0 — Fix Dim-Cell Removal in Auto-Calibration

### Problem

`applyCalibratedPreprocess` maps `raw → (raw - bg_0) / (cell_0 - bg_0)` clipped to [0, 1]. The anchors `bg_0`, `cell_0` are FROZEN at frame 0. Photobleaching causes some cell pixels to dim below the frame-0 `bg_0 = 24.48` over the run. Those pixels clip to 0 — the cell becomes literally invisible in the calibrated input, the optimizer loses its gradient toward that cell, the cell drifts away or gets absorbed.

Observed at f48 in `output_fluo_0-50_20260505_005405`: at least one cell removed from the real image post-preprocessing.

### Fix

Per-frame brightness ratio adjustment of the anchors:

```
ratio_n  = current_frame_mean / frame_0_mean
bg_n     = bg_0   × ratio_n
cell_n   = cell_0 × ratio_n
calibrated[v] = clamp((raw[v] - bg_n) / (cell_n - bg_n), 0, 1)
```

`frame_0_mean` is the mean of frame 0's raw pixel values (computed once at calibration time and stored on `BrightnessCalibration`). `current_frame_mean` is the mean of the current frame being preprocessed.

Why this works: under uniform bleaching, every pixel scales by `ratio`, so `(raw × ratio - bg_0 × ratio) / ((cell_0 - bg_0) × ratio)` simplifies to `(raw - bg_0) / (cell_0 - bg_0)` — cells stay at the same calibrated brightness regardless of when in the run we look. Under non-uniform bleaching, this is imperfect but bounded — the anchor shifts to track the dominant bleaching, individual cells can drift ±10-20% but won't fall to zero.

---

### Task A1: Store `frame_0_mean` on `BrightnessCalibration`

**Files:**
- Modify: `C++/includes/CalibrationTypes.hpp` (add field)
- Modify: `C++/src/CellUniverse.cpp` (`computeAutoCalibration` populates it)
- Modify: `C++/tests/calibration_test.cc` (assert it's populated)

- [ ] **Step 1: Write the failing test**

Append to `C++/tests/calibration_test.cc`:
```cpp
TEST(BrightnessCalibrationTest, AutoCalibrationStoresFrameZeroMean) {
    BaseConfig cfg;
    cfg.simulation.auto_calibrate_brightness_enabled = true;
    cfg.simulation.calibration_cell_inner_fraction = 0.7f;
    cfg.simulation.calibration_pixel_trim_percent = 0.10f;
    cfg.simulation.z_scaling = 1;

    // 100×100×50 frame, half cells (800), half background (50)
    const cv::Point3f center(50.0f, 50.0f, 25.0f);
    auto frame = makeSyntheticFrame(100, 100, 50, /*bg*/ 50.0f, /*cell*/ 800.0f,
                                    center, /*radius*/ 10.0f);

    EllipsoidParams p;
    p.name = "cell_1";
    p.x = center.x; p.y = center.y; p.z = center.z;
    p.aRadius = 10.0f; p.bRadius = 10.0f; p.cRadius = 10.0f;
    p.theta_x = 0; p.theta_y = 0; p.theta_z = 0;
    p.brightness = 1.0f;
    std::vector<Ellipsoid> cells{ Ellipsoid(p) };

    BrightnessCalibration cal = CellUniverse::computeAutoCalibration(cells, frame, cfg);

    // The frame is mostly background (50.0) with a tiny cell sphere of
    // ~4188 voxels at 800.0 out of 500,000 total voxels. Mean ~ 56.
    EXPECT_GT(cal.frame_0_mean, 50.0f);
    EXPECT_LT(cal.frame_0_mean, 80.0f);
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd C++/build && cmake --build . --target core_unit_test -j 4
```

Expected: compile error `'frame_0_mean' is not a member of 'BrightnessCalibration'`.

- [ ] **Step 3: Add field to BrightnessCalibration**

`C++/includes/CalibrationTypes.hpp`, in the struct:
```cpp
struct BrightnessCalibration {
    float background_intensity = 0.0f;
    float cell_intensity = 0.0f;
    std::size_t cell_pixel_count = 0;
    std::size_t background_pixel_count = 0;
    BrightnessCalibrationSource source = BrightnessCalibrationSource::Uninitialized;

    // Mean of frame 0's raw pixel values. Used by applyCalibratedPreprocess
    // for per-frame brightness-ratio adjustment of (background, cell) anchors
    // to handle photobleaching. See Change 16 (2026-05-05).
    float frame_0_mean = 0.0f;

    // ... existing methods unchanged ...
};
```

Update `calibrationDescribe` to include the new field:
```cpp
inline std::string calibrationDescribe(const BrightnessCalibration &cal) {
    std::ostringstream oss;
    // ... existing fields ...
    oss << " frame_0_mean=" << cal.frame_0_mean;
    return oss.str();
}
```

- [ ] **Step 4: Populate it in `computeAutoCalibration`**

In `C++/src/CellUniverse.cpp`, find `computeAutoCalibration`. After the existing pixel-collection loop and before `cal.source = AutoFrameZero`, add a frame-mean computation:

```cpp
    // Compute frame_0_mean for per-frame brightness-ratio adjustment in
    // applyCalibratedPreprocess (Change 16).
    double frameSum = 0.0;
    std::size_t framePixels = 0;
    for (const auto &slice : rawFrame0) {
        for (int y = 0; y < slice.rows; ++y) {
            const float *row = slice.ptr<float>(y);
            for (int x = 0; x < slice.cols; ++x) {
                frameSum += row[x];
                ++framePixels;
            }
        }
    }
    cal.frame_0_mean = framePixels > 0 ? static_cast<float>(frameSum / framePixels) : 0.0f;
```

This is added BOTH in the auto-detect path AND the manual-override path (so manual override also gets a sensible `frame_0_mean`). For the manual-override branch, we still need to compute the frame mean — pull the loop into a helper or compute it before the early-return on manual override:

```cpp
BrightnessCalibration CellUniverse::computeAutoCalibration(
    const std::vector<Ellipsoid> &cells,
    const std::vector<cv::Mat> &rawFrame0,
    const BaseConfig &config)
{
    BrightnessCalibration cal;

    // Always compute frame_0_mean if the frame is non-empty (regardless of
    // auto vs manual override), so applyCalibratedPreprocess can use it.
    if (!rawFrame0.empty()) {
        double frameSum = 0.0;
        std::size_t framePixels = 0;
        for (const auto &slice : rawFrame0) {
            for (int y = 0; y < slice.rows; ++y) {
                const float *row = slice.ptr<float>(y);
                for (int x = 0; x < slice.cols; ++x) {
                    frameSum += row[x];
                    ++framePixels;
                }
            }
        }
        cal.frame_0_mean = framePixels > 0 ? static_cast<float>(frameSum / framePixels) : 0.0f;
    }

    const float manualBg = config.simulation.manual_background_intensity;
    const float manualCell = config.simulation.manual_cell_intensity;
    if (manualBg > 0.0f && manualCell > 0.0f && manualCell > manualBg) {
        cal.background_intensity = manualBg;
        cal.cell_intensity = manualCell;
        cal.source = BrightnessCalibrationSource::ManualOverride;
        return cal;
    }
    // ... existing code unchanged ...
}
```

- [ ] **Step 5: Run test to verify it passes**

```bash
cd C++/build && cmake --build . --target core_unit_test -j 4
./tests/core_unit_test --gtest_filter="BrightnessCalibrationTest.AutoCalibrationStoresFrameZeroMean"
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add C++/includes/CalibrationTypes.hpp C++/src/CellUniverse.cpp C++/tests/calibration_test.cc
git commit -m "store frame_0_mean on BrightnessCalibration for per-frame ratio adjustment"
```

---

### Task A2: Apply per-frame ratio in `applyCalibratedPreprocess`

**Files:**
- Modify: `C++/src/ImageHandler.cpp` (`applyCalibratedPreprocess`)
- Test: `C++/tests/process_image_test.cc`

- [ ] **Step 1: Write the failing test**

Append to `C++/tests/process_image_test.cc`:
```cpp
TEST(ApplyCalibratedPreprocessTest, PerFrameRatioRescalesAnchorsForDimmedFrame) {
    BrightnessCalibration cal;
    cal.background_intensity = 50.0f;   // bg_0
    cal.cell_intensity = 800.0f;        // cell_0
    cal.frame_0_mean = 100.0f;          // mean of original frame 0
    cal.source = BrightnessCalibrationSource::AutoFrameZero;

    BaseConfig cfg;
    cfg.simulation.z_scaling = 1;
    cfg.simulation.blur_sigma = 0.0f;

    // Simulate frame N where everything dimmed by 50% (uniform bleaching).
    // Mean = 50 (vs frame 0 mean 100), so ratio = 0.5.
    // Anchors should adjust: bg_n = 25, cell_n = 400.
    // A pixel at raw=400 should map to 1.0 (was the cell value before
    // bleaching halved it).
    std::vector<cv::Mat> input;
    cv::Mat slice(10, 10, CV_32F, cv::Scalar(50.0f));  // background after bleaching
    slice.at<float>(5, 5) = 400.0f;  // cell after bleaching
    input.push_back(slice.clone());

    auto out = ImageHandler::applyCalibratedPreprocess(input, cal, cfg, nullptr);
    EXPECT_NEAR(out[0].at<float>(5, 5), 1.0f, 1e-3f);  // bleached cell still maps to 1
    EXPECT_NEAR(out[0].at<float>(0, 0), 0.0f, 1e-3f);  // bleached bg still maps to 0
}

TEST(ApplyCalibratedPreprocessTest, PerFrameRatioFallsBackTo1WhenFrame0MeanZero) {
    BrightnessCalibration cal;
    cal.background_intensity = 50.0f;
    cal.cell_intensity = 800.0f;
    cal.frame_0_mean = 0.0f;  // sentinel — preserve original behavior
    cal.source = BrightnessCalibrationSource::AutoFrameZero;

    BaseConfig cfg;
    cfg.simulation.z_scaling = 1;
    cfg.simulation.blur_sigma = 0.0f;

    std::vector<cv::Mat> input;
    cv::Mat slice(10, 10, CV_32F, cv::Scalar(50.0f));
    slice.at<float>(5, 5) = 800.0f;
    input.push_back(slice.clone());

    auto out = ImageHandler::applyCalibratedPreprocess(input, cal, cfg, nullptr);
    EXPECT_NEAR(out[0].at<float>(5, 5), 1.0f, 1e-3f);
    EXPECT_NEAR(out[0].at<float>(0, 0), 0.0f, 1e-3f);
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd C++/build && cmake --build . --target core_unit_test -j 4
./tests/core_unit_test --gtest_filter="ApplyCalibratedPreprocessTest.PerFrameRatio*"
```

Expected: the first test FAILS — without per-frame ratio, raw=400 maps to (400-50)/(800-50) = 0.467, not 1.0. The second test PASSES (existing behavior preserved when frame_0_mean=0).

- [ ] **Step 3: Implement per-frame ratio**

In `C++/src/ImageHandler.cpp` `applyCalibratedPreprocess`, replace the linear-scale block (currently `for (const auto &slice : normalizedSlices) { ... slice.convertTo(out, CV_32F, 1.0f / span, -bg / span); ... }`) with:

```cpp
    // Step 1: compute per-frame brightness ratio for anchor adjustment.
    // ratio = current_frame_mean / frame_0_mean. Falls back to 1.0 (no
    // adjustment) when frame_0_mean is zero (e.g. manual override path
    // didn't compute it, or sentinel-0 from older saved calibrations).
    float ratio = 1.0f;
    if (calibration.frame_0_mean > 1e-6f) {
        double currentSum = 0.0;
        std::size_t currentPixels = 0;
        for (const auto &slice : normalizedSlices) {
            for (int y = 0; y < slice.rows; ++y) {
                const float *row = slice.ptr<float>(y);
                for (int x = 0; x < slice.cols; ++x) {
                    currentSum += row[x];
                    ++currentPixels;
                }
            }
        }
        const float currentMean = currentPixels > 0
            ? static_cast<float>(currentSum / currentPixels) : calibration.frame_0_mean;
        ratio = currentMean / calibration.frame_0_mean;
        // Clamp to a sane range — extreme values usually mean the input frame
        // is corrupted or otherwise abnormal. 0.3..3.0 covers ±3× brightness
        // shift which is well past any normal photobleaching.
        ratio = std::clamp(ratio, 0.3f, 3.0f);
    }

    const float bg_n = calibration.background_intensity * ratio;
    const float cell_n = calibration.cell_intensity * ratio;
    const float span_n = std::max(1e-3f, cell_n - bg_n);

    log << "[Calibrated Preprocess Ratio] frame_0_mean=" << calibration.frame_0_mean
        << " frame_n_mean=" << (calibration.frame_0_mean * ratio)
        << " ratio=" << ratio
        << " bg_0=" << calibration.background_intensity
        << " cell_0=" << calibration.cell_intensity
        << " bg_n=" << bg_n
        << " cell_n=" << cell_n
        << "\n";

    // Step 2: linear normalize via adjusted anchors -> [0, 1]
    std::vector<cv::Mat> linearScaled;
    linearScaled.reserve(normalizedSlices.size());
    for (const auto &slice : normalizedSlices) {
        cv::Mat out;
        slice.convertTo(out, CV_32F, 1.0f / span_n, -bg_n / span_n);
        cv::min(out, 1.0f, out);
        cv::max(out, 0.0f, out);
        linearScaled.push_back(out);
    }
    // ... rest of function (blur, z-interpolate) unchanged ...
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd C++/build && cmake --build . --target core_unit_test -j 4
./tests/core_unit_test --gtest_filter="ApplyCalibratedPreprocessTest.*"
```

Expected: ALL ApplyCalibratedPreprocess tests pass (the existing 3 + the new 2).

Then full suite:
```bash
./tests/core_unit_test
```

Expected: all tests pass (was 37, now 37 + 2 from this task + 1 from Task A1 = 40, will grow further in Phase 1).

- [ ] **Step 5: Commit**

```bash
git add C++/src/ImageHandler.cpp C++/tests/process_image_test.cc
git commit -m "apply per-frame brightness ratio to calibration anchors (fix dim-cell removal)"
```

---

### Task A3: Reorder main.cpp so calibration runs before `quit_after_preprocessing`

**Files:**
- Modify: `C++/src/main.cpp`

### Why

The current `quit_after_preprocessing: true` branch (preprocess-only mode for fast iteration) runs **before** `cellFactory.createCells`, so cells are empty and auto-calibration cannot run. The preprocess-only `preprocessOnlyLineage` falls back to the legacy iterative path — exactly the path we're trying to replace. Reorder so calibration is installed even in preprocess-only mode, enabling fast preprocessed-image comparison without running the optimizer.

This unblocks Task A4 (preprocess-only validation of the per-frame ratio fix at f48 — minutes, not hours).

- [ ] **Step 1: Find the current order**

Open `C++/src/main.cpp` around line 130–155. Current order:
```
130. PathVec imageFilePaths = ImageHandler::getImageFilePaths(...);
142. if (quit_after_preprocessing) { preprocessOnlyLineage({}, ...); return 0; }
151. cells = cellFactory.createCells(...);
155. // auto-calibration block (added in Task 4 of auto-cal plan)
177. lineage.setBrightnessCalibration(calibration);
```

- [ ] **Step 2: Move cells + calibration before the quit check**

Move the `CellFactory cellFactory(config); cells = cellFactory.createCells(...);` block (lines ~150–152) up to BEFORE the `quit_after_preprocessing` check (line ~142). Move the auto-calibration block (lines ~155–172) up too.

The new order:
```
130. PathVec imageFilePaths = ImageHandler::getImageFilePaths(...);
~135. CellFactory cellFactory(config);
~136. cells = cellFactory.createCells(...);
~140. BrightnessCalibration calibration;
~145. if (auto_calibrate_brightness_enabled) { ... compute calibration ... }
~155. if (quit_after_preprocessing) {
~157.     CellUniverse preprocessOnlyLineage(cells, imageFilePaths, config,
~158.                                        args.output, args.firstFrame,
~159.                                        args.continueFrom);
~160.     preprocessOnlyLineage.setBrightnessCalibration(calibration);
~161.     preprocessOnlyLineage.preprocessAllFramesAlignedToMinimumBackground(false);
~162.     return 0;
~163. }
~165. // auto-derive geometry block (Task 5 of this plan)
~170. CellUniverse lineage = CellUniverse(cells, imageFilePaths, ...);
~172. lineage.setBrightnessCalibration(calibration);
~175. // continue with prepare_analyze_one_frame OR preprocessAllFramesAlignedToMinimumBackground(true)
```

Key change in the `quit_after_preprocessing` block: pass the loaded `cells` (not `{}`) to the `preprocessOnlyLineage` constructor, and call `setBrightnessCalibration(calibration)` before `preprocessAllFramesAlignedToMinimumBackground(false)`.

- [ ] **Step 3: Smoke-build to verify no compile errors**

```bash
cd C++/build && cmake --build . --target celluniverse -j 4
```

Expected: builds clean.

- [ ] **Step 4: Smoke-test preprocess-only mode**

Edit `C++/config/config.yaml` temporarily:
```yaml
quit_after_preprocessing: true
export_preprocessed_images: true
```

Run on the fluo dataset:
```bash
cd C++ && ./build/celluniverse 0 5 \
    data/input/embryo_data/t%03d.tif \
    outputs/output_preprocess_only_smoke_$(date +%Y%m%d_%H%M%S) \
    config/config.yaml \
    config/initial_auto.csv 2>&1 | head -30
```

Expected log:
- `[Auto Calibration] background=24.48 cell=86.93 ... frame_0_mean=27.7...`
- `[Preprocess Dispatch] file=t000.tif path=calibrated`
- `[Calibrated Preprocess Ratio] frame_0_mean=... ratio=...`
- One `[Calibrated Preprocess Ratio]` line per frame

The `outputs/output_preprocess_only_smoke_*` directory should contain preprocessed image stacks — one stack per frame.

- [ ] **Step 5: Restore config.yaml**

Set `quit_after_preprocessing: false` and `export_preprocessed_images: false` again before the next normal run. Do NOT commit a config.yaml change — those flags are debug-only.

- [ ] **Step 6: Commit the main.cpp reorder**

```bash
git add C++/src/main.cpp
git commit -m "reorder main.cpp: calibration runs before quit_after_preprocessing check"
```

---

### Task A4: Validate per-frame ratio fix via preprocess-only on f0–50

**Files:** none modified. Validation only.

This is the FAST iteration path for the dim-cell fix — preprocess all 51 frames without running the optimizer. Wall time ~5–15 min instead of ~16 hr.

- [ ] **Step 1: Configure preprocess-only mode**

In `C++/config/config.yaml` (temporarily, do not commit):
```yaml
quit_after_preprocessing: true
export_preprocessed_images: true
```

- [ ] **Step 2: Run preprocess-only on f0–50 fluo**

```bash
cd C++ && ./build/celluniverse 0 50 \
    data/input/embryo_data/t%03d.tif \
    outputs/output_preprocess_ratio_$(date +%Y%m%d_%H%M%S) \
    config/config.yaml \
    config/initial_auto.csv > .../debug_log.txt 2>&1
```

Estimated wall time: ~5–15 min on Mac (preprocessing is ~10–20 sec/frame, no optimization).

- [ ] **Step 3: Inspect the f48 preprocessed slice where the cell was previously removed**

The cell that disappeared at f48 in the auto-calibrated baseline (`output_fluo_0-50_20260505_005405`) should now be visible in the preprocessed output. Check the cell's known position (from `output_fluo_0-50_20260505_005405/cells.csv` at t048.tif row for the cell) and inspect the brightness at that voxel in the new preprocessed stack.

```bash
RUN_DIR=$(ls -td C++/outputs/output_preprocess_ratio_* | head -1)

# Read the cell positions at t048 from the validated baseline
awk -F, '$1=="t048.tif" && $12==0 {print $2, $3, $4, $5, $6, $7, $8}' \
    C++/outputs/output_fluo_0-50_20260505_005405/cells.csv

# For each cell position, sample the preprocessed image at that voxel and
# confirm value > 0 (i.e. the cell is preserved). Use a small Python script
# or the brightness_volume_analyzer offline tool already in the repo.
```

Acceptance: every cell position from `output_fluo_0-50_20260505_005405` cells.csv at t048 has a preprocessed brightness > 0.1 in the new run. If the previously-removed cell now reads > 0.1, the fix worked.

- [ ] **Step 4: Verify per-frame ratio drift is sane**

```bash
grep "Calibrated Preprocess Ratio" $RUN_DIR/debug_log.txt | head -5
grep "Calibrated Preprocess Ratio" $RUN_DIR/debug_log.txt | tail -5
```

Expected: ratio starts ≈ 1.0 at f0, drifts to ~0.95–0.98 by f50 (slight photobleaching). Anything wildly different (> 1.1 or < 0.5) suggests a per-frame mean computation bug.

- [ ] **Step 5: Restore config.yaml**

Set `quit_after_preprocessing: false` and `export_preprocessed_images: false` for subsequent normal runs.

- [ ] **Step 6: If the fix works, no commit needed (validation only)**

If the f48 cell is preserved, mark Task A4 done in TodoWrite and proceed to Phase 1. If the fix does NOT work, the most likely issues:
1. Per-frame mean is computed on already-clipped values (after the linear scale) instead of raw input. Check the order of operations in Task A2's implementation.
2. `frame_0_mean` was not populated correctly in `BrightnessCalibration`. Check Task A1's implementation.
3. `current_frame_mean` clamps to a degenerate value when most pixels are background. Try the median instead of mean as a more robust per-frame anchor.

---

---

## Phase 1 — Foundation

### Task 1: Create `DerivedGeometry` struct + describe helper

**Files:**
- Create: `C++/includes/GeometryDerivation.hpp`
- Test: `C++/tests/geometry_derivation_test.cc`
- Modify: `C++/tests/CMakeLists.txt`

- [ ] **Step 1: Write the failing test**

Create `C++/tests/geometry_derivation_test.cc`:
```cpp
#include <gtest/gtest.h>
#include <sstream>

#include "GeometryDerivation.hpp"

TEST(DerivedGeometryTest, IsValidRequiresPositiveMaxRadius) {
    DerivedGeometry g;
    EXPECT_FALSE(g.isValid());

    g.max_a_radius = 50.0f;
    g.max_b_radius = 50.0f;
    g.max_c_radius = 50.0f;
    g.min_a_radius = 8.0f;
    g.min_b_radius = 8.0f;
    g.min_c_radius = 8.0f;
    g.perturb_reference_radius = 25.0f;
    g.x_sigma = 5.0f;
    g.y_sigma = 5.0f;
    g.z_sigma = 8.0f;
    g.position_prior_threshold = 20.0f;
    g.iterations_per_cell = 250;
    g.pca_bridge_min_side_voxels = 40;
    EXPECT_TRUE(g.isValid());
}

TEST(DerivedGeometryTest, DescribeIncludesAllFields) {
    DerivedGeometry g;
    g.max_a_radius = 80.0f;
    g.min_a_radius = 7.8f;
    g.perturb_reference_radius = 39.0f;
    g.x_sigma = 5.85f;
    g.z_sigma = 14.4f;
    g.iterations_per_cell = 240;
    g.position_prior_threshold = 31.2f;

    const std::string desc = derivedGeometryDescribe(g);
    EXPECT_NE(desc.find("max_a=80"), std::string::npos);
    EXPECT_NE(desc.find("min_a=7.8"), std::string::npos);
    EXPECT_NE(desc.find("ref_r=39"), std::string::npos);
    EXPECT_NE(desc.find("xy_sigma=5.85"), std::string::npos);
    EXPECT_NE(desc.find("z_sigma=14.4"), std::string::npos);
    EXPECT_NE(desc.find("iters=240"), std::string::npos);
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd C++/build && cmake --build . --target core_unit_test -j 4
```

Expected: `fatal error: 'GeometryDerivation.hpp' file not found`.

- [ ] **Step 3: Create the header**

`C++/includes/GeometryDerivation.hpp`:
```cpp
#ifndef GEOMETRY_DERIVATION_HPP
#define GEOMETRY_DERIVATION_HPP

#include <cstddef>
#include <sstream>
#include <string>

// Geometry parameters derived from initial-CSV statistics + z_scaling.
// Produced by CellUniverse::computeDerivedGeometry. Applied to
// Ellipsoid::cellConfig and BaseConfig before CellUniverse is constructed.
//
// All radius/sigma values are in PIXEL coordinates (interpolated z-space
// for *_c_radius and z_sigma; cell positions in the initial CSV are in
// raw z and CellFactory multiplies by z_scaling, so c-radius math here
// is in interpolated z just like the rest of the optimizer).
struct DerivedGeometry {
    float max_a_radius = 0.0f;
    float max_b_radius = 0.0f;
    float max_c_radius = 0.0f;
    float min_a_radius = 0.0f;
    float min_b_radius = 0.0f;
    float min_c_radius = 0.0f;
    float perturb_reference_radius = 0.0f;
    float x_sigma = 0.0f;
    float y_sigma = 0.0f;
    float z_sigma = 0.0f;
    float position_prior_threshold = 0.0f;
    int iterations_per_cell = 0;
    int pca_bridge_min_side_voxels = 0;
    std::size_t cell_count = 0;
    float mean_a_radius = 0.0f;
    float mean_b_radius = 0.0f;
    float mean_c_radius = 0.0f;

    bool isValid() const {
        return max_a_radius > 1e-3f && perturb_reference_radius > 1e-3f
               && iterations_per_cell > 0;
    }
};

inline std::string derivedGeometryDescribe(const DerivedGeometry &g) {
    std::ostringstream oss;
    oss << "max_a=" << g.max_a_radius
        << " max_b=" << g.max_b_radius
        << " max_c=" << g.max_c_radius
        << " min_a=" << g.min_a_radius
        << " min_b=" << g.min_b_radius
        << " min_c=" << g.min_c_radius
        << " ref_r=" << g.perturb_reference_radius
        << " xy_sigma=" << g.x_sigma
        << " z_sigma=" << g.z_sigma
        << " prior_thr=" << g.position_prior_threshold
        << " iters=" << g.iterations_per_cell
        << " bridge_min_voxels=" << g.pca_bridge_min_side_voxels
        << " n_cells=" << g.cell_count
        << " mean_a=" << g.mean_a_radius
        << " mean_b=" << g.mean_b_radius
        << " mean_c=" << g.mean_c_radius;
    return oss.str();
}

#endif // GEOMETRY_DERIVATION_HPP
```

- [ ] **Step 4: Register the test**

In `C++/tests/CMakeLists.txt`, add `geometry_derivation_test.cc` to the `add_executable(core_unit_test ...)` source list (right after `calibration_test.cc`).

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd C++/build && cmake --build . --target core_unit_test -j 4
./tests/core_unit_test --gtest_filter="DerivedGeometryTest.*"
```

Expected: 2 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add C++/includes/GeometryDerivation.hpp C++/tests/geometry_derivation_test.cc C++/tests/CMakeLists.txt
git commit -m "add DerivedGeometry struct + unit tests"
```

---

### Task 2: Add `auto_derive_geometry_enabled` flag + manual-override fields

**Files:**
- Modify: `C++/includes/ConfigTypes.hpp` (`SimulationConfig` field section + parse + dump)
- Modify: `C++/config/config.yaml`
- Modify: `C++/tests/config_types_test.cc`

- [ ] **Step 1: Write the failing test**

Append to `C++/tests/config_types_test.cc`:
```cpp
TEST(ConfigTypesTest, AutoDeriveGeometryFieldsHaveExpectedDefaults) {
    SimulationConfig cfg;
    EXPECT_TRUE(cfg.auto_derive_geometry_enabled);
    EXPECT_FLOAT_EQ(cfg.geometry_force_max_a_radius, -1.0f);
    EXPECT_FLOAT_EQ(cfg.geometry_force_min_a_radius, -1.0f);
    EXPECT_FLOAT_EQ(cfg.geometry_force_perturb_reference_radius, -1.0f);
    EXPECT_FLOAT_EQ(cfg.geometry_force_xy_sigma, -1.0f);
    EXPECT_FLOAT_EQ(cfg.geometry_force_z_sigma, -1.0f);
    EXPECT_FLOAT_EQ(cfg.geometry_force_position_prior_threshold, -1.0f);
    EXPECT_EQ(cfg.geometry_force_iterations_per_cell, -1);
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd C++/build && cmake --build . --target core_unit_test -j 4
```

Expected: compile error `auto_derive_geometry_enabled is not a member`.

- [ ] **Step 3: Add fields to ConfigTypes.hpp**

In `C++/includes/ConfigTypes.hpp` `SimulationConfig` class, after the auto-calibration block (which ends with `calibration_pixel_trim_percent`), insert:

```cpp
    // ---- Auto-derive geometry from initial CSV (2026-05-05) ----
    // Replaces the dataset-specific cell-radius bounds, perturbation sigmas,
    // perturbSigmaReferenceRadius, position_prior_threshold, and
    // iterations_per_cell with values derived from initial-CSV statistics
    // + z_scaling. After this lands, a new dataset only needs (1) initial
    // CSV with cell positions and radii, (2) z_scaling. Everything else
    // auto-derives.
    //
    // Per-param manual override: any geometry_force_* > 0 (or > -1 for
    // the int) takes precedence over the auto-derived value, allowing
    // users to pin specific values when auto-derive does the wrong thing.
    bool auto_derive_geometry_enabled = true;
    float geometry_force_max_a_radius = -1.0f;
    float geometry_force_max_b_radius = -1.0f;
    float geometry_force_max_c_radius = -1.0f;
    float geometry_force_min_a_radius = -1.0f;
    float geometry_force_min_b_radius = -1.0f;
    float geometry_force_min_c_radius = -1.0f;
    float geometry_force_perturb_reference_radius = -1.0f;
    float geometry_force_xy_sigma = -1.0f;
    float geometry_force_z_sigma = -1.0f;
    float geometry_force_position_prior_threshold = -1.0f;
    int geometry_force_iterations_per_cell = -1;
    int geometry_force_pca_bridge_min_side_voxels = -1;
```

In the `SimulationConfig::explodeConfig` parser block (find the auto_calibrate parsers and add after):

```cpp
        if (node["auto_derive_geometry_enabled"]) auto_derive_geometry_enabled = node["auto_derive_geometry_enabled"].as<bool>();
        if (node["geometry_force_max_a_radius"]) geometry_force_max_a_radius = node["geometry_force_max_a_radius"].as<float>();
        if (node["geometry_force_max_b_radius"]) geometry_force_max_b_radius = node["geometry_force_max_b_radius"].as<float>();
        if (node["geometry_force_max_c_radius"]) geometry_force_max_c_radius = node["geometry_force_max_c_radius"].as<float>();
        if (node["geometry_force_min_a_radius"]) geometry_force_min_a_radius = node["geometry_force_min_a_radius"].as<float>();
        if (node["geometry_force_min_b_radius"]) geometry_force_min_b_radius = node["geometry_force_min_b_radius"].as<float>();
        if (node["geometry_force_min_c_radius"]) geometry_force_min_c_radius = node["geometry_force_min_c_radius"].as<float>();
        if (node["geometry_force_perturb_reference_radius"]) geometry_force_perturb_reference_radius = node["geometry_force_perturb_reference_radius"].as<float>();
        if (node["geometry_force_xy_sigma"]) geometry_force_xy_sigma = node["geometry_force_xy_sigma"].as<float>();
        if (node["geometry_force_z_sigma"]) geometry_force_z_sigma = node["geometry_force_z_sigma"].as<float>();
        if (node["geometry_force_position_prior_threshold"]) geometry_force_position_prior_threshold = node["geometry_force_position_prior_threshold"].as<float>();
        if (node["geometry_force_iterations_per_cell"]) geometry_force_iterations_per_cell = node["geometry_force_iterations_per_cell"].as<int>();
        if (node["geometry_force_pca_bridge_min_side_voxels"]) geometry_force_pca_bridge_min_side_voxels = node["geometry_force_pca_bridge_min_side_voxels"].as<int>();
```

In the `printConfig` cout block, add after the auto-calibration cout lines:

```cpp
        std::cout << "auto_derive_geometry_enabled: " << auto_derive_geometry_enabled << '\n';
        std::cout << "geometry_force_max_a_radius: " << geometry_force_max_a_radius << '\n';
        std::cout << "geometry_force_min_a_radius: " << geometry_force_min_a_radius << '\n';
        std::cout << "geometry_force_perturb_reference_radius: " << geometry_force_perturb_reference_radius << '\n';
        std::cout << "geometry_force_xy_sigma: " << geometry_force_xy_sigma << '\n';
        std::cout << "geometry_force_z_sigma: " << geometry_force_z_sigma << '\n';
        std::cout << "geometry_force_position_prior_threshold: " << geometry_force_position_prior_threshold << '\n';
        std::cout << "geometry_force_iterations_per_cell: " << geometry_force_iterations_per_cell << '\n';
```

- [ ] **Step 4: Add YAML defaults**

In `C++/config/config.yaml` under the `simulation:` block (insert after the calibration block):

```yaml
  # Auto-derive geometry from initial CSV (2026-05-05). See plan
  # docs/plans/2026-05-05-auto-derive-geometry-from-initial-csv.md.
  # When true, cell-radius bounds, perturbation sigmas,
  # perturbSigmaReferenceRadius, position_prior_threshold, and
  # iterations_per_cell are computed from initial-CSV statistics +
  # z_scaling and OVERRIDE the values set in cell:/prob:/simulation:
  # blocks below.
  #
  # Per-param manual override: set any geometry_force_* > 0 (or > -1 for
  # the int) to pin that specific value. Useful when auto-derive picks
  # a wrong value due to bad initial CSV ROI radii or extreme cell
  # aspect ratios.
  auto_derive_geometry_enabled: true
  geometry_force_max_a_radius: -1
  geometry_force_max_b_radius: -1
  geometry_force_max_c_radius: -1
  geometry_force_min_a_radius: -1
  geometry_force_min_b_radius: -1
  geometry_force_min_c_radius: -1
  geometry_force_perturb_reference_radius: -1
  geometry_force_xy_sigma: -1
  geometry_force_z_sigma: -1
  geometry_force_position_prior_threshold: -1
  geometry_force_iterations_per_cell: -1
  geometry_force_pca_bridge_min_side_voxels: -1
```

- [ ] **Step 5: Run test to verify it passes**

```bash
cd C++/build && cmake --build . --target core_unit_test -j 4
./tests/core_unit_test --gtest_filter="ConfigTypesTest.AutoDeriveGeometryFieldsHaveExpectedDefaults"
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add C++/includes/ConfigTypes.hpp C++/config/config.yaml C++/tests/config_types_test.cc
git commit -m "add auto_derive_geometry_enabled + per-field manual-override flags"
```

---

### Task 3: Implement `computeDerivedGeometry` (pure function)

**Files:**
- Modify: `C++/includes/CellUniverse.hpp` (add include + static method declaration)
- Modify: `C++/src/CellUniverse.cpp` (add implementation in anonymous namespace + impl)
- Test: `C++/tests/geometry_derivation_test.cc`

- [ ] **Step 1: Write the failing tests**

Append to `C++/tests/geometry_derivation_test.cc`:
```cpp
#include "CellUniverse.hpp"
#include "Ellipsoid.hpp"

namespace {

Ellipsoid makeEll(const std::string &name, float ra, float rb, float rc, bool trash = false) {
    EllipsoidParams p;
    p.name = name;
    p.x = 100.0f; p.y = 100.0f; p.z = 100.0f;
    p.aRadius = ra; p.bRadius = rb; p.cRadius = rc;
    p.theta_x = 0; p.theta_y = 0; p.theta_z = 0;
    p.brightness = 1.0f;
    p.isTrash = trash;
    return Ellipsoid(p);
}

}  // namespace

TEST(ComputeDerivedGeometryTest, FluoLikeFourCellsProducesExpectedFormulas) {
    BaseConfig cfg;
    cfg.simulation.auto_derive_geometry_enabled = true;

    std::vector<Ellipsoid> cells = {
        makeEll("cell_1", 40.0f, 40.0f, 48.0f),
        makeEll("cell_2", 40.0f, 40.0f, 48.0f),
        makeEll("cell_3", 38.0f, 38.0f, 48.0f),
        makeEll("cell_4", 38.0f, 38.0f, 48.0f),
        makeEll("trash_1", 10.0f, 10.0f, 10.0f, /*trash=*/true),
    };
    const cv::Size3i image_size{600, 460, 32};  // raw, before z-interp

    DerivedGeometry g = CellUniverse::computeDerivedGeometry(cells, image_size, cfg);

    // mean a/b = 39, mean c = 48, max a = 40, max c = 48, n=4 (trash excluded)
    EXPECT_NEAR(g.mean_a_radius, 39.0f, 1e-3f);
    EXPECT_NEAR(g.mean_b_radius, 39.0f, 1e-3f);
    EXPECT_NEAR(g.mean_c_radius, 48.0f, 1e-3f);
    EXPECT_EQ(g.cell_count, 4u);

    // formulas
    EXPECT_NEAR(g.max_a_radius, 80.0f, 1e-3f);                      // 40 * 2.0
    EXPECT_NEAR(g.max_b_radius, 80.0f, 1e-3f);
    EXPECT_NEAR(g.max_c_radius, 96.0f, 1e-3f);                      // 48 * 2.0
    EXPECT_NEAR(g.min_a_radius, 7.8f, 1e-3f);                       // 39 * 0.2
    EXPECT_NEAR(g.min_b_radius, 7.8f, 1e-3f);
    EXPECT_NEAR(g.min_c_radius, 9.6f, 1e-3f);                       // 48 * 0.2
    EXPECT_NEAR(g.perturb_reference_radius, 39.0f, 1e-3f);
    EXPECT_NEAR(g.x_sigma, 5.85f, 1e-3f);                           // 39 * 0.15
    EXPECT_NEAR(g.y_sigma, 5.85f, 1e-3f);
    EXPECT_NEAR(g.z_sigma, 14.4f, 1e-3f);                           // 48 * 0.30
    EXPECT_NEAR(g.position_prior_threshold, 31.2f, 1e-3f);          // 39 * 0.8
    EXPECT_EQ(g.iterations_per_cell, 150);                           // 4*30=120, clamp to 150
    EXPECT_TRUE(g.isValid());
}

TEST(ComputeDerivedGeometryTest, EmbryoLikeSixCellsProducesExpectedFormulas) {
    BaseConfig cfg;
    cfg.simulation.auto_derive_geometry_enabled = true;

    std::vector<Ellipsoid> cells = {
        makeEll("e3d03", 14.5f, 14.5f, 14.5f),
        makeEll("12345", 30.0f, 30.0f, 30.0f),
        makeEll("e9077", 28.0f, 28.0f, 28.0f),
        makeEll("1f2ed", 29.6f, 29.6f, 29.6f),
        makeEll("1f89ab", 31.1f, 31.1f, 31.1f),
        makeEll("8cbdf", 18.4f, 18.4f, 18.4f),
    };
    const cv::Size3i image_size{600, 460, 32};

    DerivedGeometry g = CellUniverse::computeDerivedGeometry(cells, image_size, cfg);

    // mean a/b/c ≈ 25.27 (the embryo dataset radii)
    EXPECT_NEAR(g.mean_a_radius, 25.27f, 0.05f);
    EXPECT_NEAR(g.max_a_radius, 62.2f, 0.05f);                       // 31.1 * 2.0
    EXPECT_NEAR(g.min_a_radius, 5.05f, 0.02f);                       // 25.27 * 0.2
    EXPECT_NEAR(g.x_sigma, 3.79f, 0.02f);                            // 25.27 * 0.15
    EXPECT_NEAR(g.z_sigma, 7.58f, 0.02f);                            // 25.27 * 0.30
    EXPECT_EQ(g.iterations_per_cell, 180);                            // 6*30=180
    EXPECT_NEAR(g.position_prior_threshold, 20.21f, 0.05f);
}

TEST(ComputeDerivedGeometryTest, ManualOverridesReplaceDerivedValues) {
    BaseConfig cfg;
    cfg.simulation.auto_derive_geometry_enabled = true;
    cfg.simulation.geometry_force_max_a_radius = 999.0f;
    cfg.simulation.geometry_force_xy_sigma = 7.7f;
    cfg.simulation.geometry_force_iterations_per_cell = 333;

    std::vector<Ellipsoid> cells = {
        makeEll("cell_1", 40.0f, 40.0f, 48.0f),
    };
    const cv::Size3i image_size{600, 460, 32};

    DerivedGeometry g = CellUniverse::computeDerivedGeometry(cells, image_size, cfg);

    EXPECT_FLOAT_EQ(g.max_a_radius, 999.0f);
    EXPECT_FLOAT_EQ(g.x_sigma, 7.7f);
    EXPECT_FLOAT_EQ(g.y_sigma, 7.7f);
    EXPECT_EQ(g.iterations_per_cell, 333);
    // Non-overridden values still derived
    EXPECT_NEAR(g.min_a_radius, 8.0f, 1e-3f);                        // 40 * 0.2
    EXPECT_NEAR(g.position_prior_threshold, 32.0f, 1e-3f);           // 40 * 0.8
}

TEST(ComputeDerivedGeometryTest, EmptyCellsReturnsInvalidGeometry) {
    BaseConfig cfg;
    cfg.simulation.auto_derive_geometry_enabled = true;

    std::vector<Ellipsoid> cells;
    const cv::Size3i image_size{600, 460, 32};

    DerivedGeometry g = CellUniverse::computeDerivedGeometry(cells, image_size, cfg);
    EXPECT_FALSE(g.isValid());
}

TEST(ComputeDerivedGeometryTest, IterationsClampedToMinAndMax) {
    BaseConfig cfg;
    cfg.simulation.auto_derive_geometry_enabled = true;

    // 1 cell -> 30 iters, clamped to 150 (min)
    std::vector<Ellipsoid> oneCell = { makeEll("c", 30.0f, 30.0f, 30.0f) };
    DerivedGeometry g1 = CellUniverse::computeDerivedGeometry(
        oneCell, cv::Size3i{600, 460, 32}, cfg);
    EXPECT_EQ(g1.iterations_per_cell, 150);

    // 30 cells -> 900 iters, clamped to 500 (max)
    std::vector<Ellipsoid> manyCells;
    for (int i = 0; i < 30; ++i) {
        manyCells.push_back(makeEll("c" + std::to_string(i), 30.0f, 30.0f, 30.0f));
    }
    DerivedGeometry g30 = CellUniverse::computeDerivedGeometry(
        manyCells, cv::Size3i{600, 460, 32}, cfg);
    EXPECT_EQ(g30.iterations_per_cell, 500);
}
```

NOTE: `cv::Size3i` may not exist as a typedef in OpenCV; use `cv::Vec3i` or define a local POD `struct ImageSize3D { int w; int h; int z; };` in `GeometryDerivation.hpp`. Verify by reading any existing use of 3D image-size types in the codebase before writing the test.

- [ ] **Step 2: Add the `ImageSize3D` struct to GeometryDerivation.hpp**

If `cv::Size3i` doesn't exist or feels off, add to `GeometryDerivation.hpp`:
```cpp
struct ImageSize3D {
    int width = 0;
    int height = 0;
    int depth = 0;  // raw (pre z-interpolation) z-slice count
};
```

Update test to use `ImageSize3D` instead of `cv::Size3i`. The `image_size.depth` is used downstream for `pca_bridge_min_side_voxels` formula which depends on cell volume relative to image volume; image_size.width/height aren't currently used in formulas but the parameter is reserved for future use (e.g. clamping bounds against image size).

- [ ] **Step 3: Run test to verify it fails**

```bash
cd C++/build && cmake --build . --target core_unit_test -j 4
```

Expected: compile error `'computeDerivedGeometry' is not a member of 'CellUniverse'`.

- [ ] **Step 4: Add header declaration**

`C++/includes/CellUniverse.hpp`, add at top with other includes:
```cpp
#include "GeometryDerivation.hpp"
```

In the `public:` section, after `computeAutoCalibration`:
```cpp
    // Compute geometry parameters from initial CSV statistics + image
    // dimensions. Pure function, no member mutation. Per-param manual
    // overrides in config.simulation.geometry_force_* take precedence
    // over the auto-derived values. When cells is empty, returns an
    // invalid DerivedGeometry. Caller applies the result to
    // Ellipsoid::cellConfig and BaseConfig before constructing
    // CellUniverse — see main.cpp.
    static DerivedGeometry computeDerivedGeometry(
        const std::vector<Ellipsoid> &cells,
        const ImageSize3D &image_size,
        const BaseConfig &config);
```

- [ ] **Step 5: Implement the method**

`C++/src/CellUniverse.cpp`, add right before or after `computeAutoCalibration`:

```cpp
DerivedGeometry CellUniverse::computeDerivedGeometry(
    const std::vector<Ellipsoid> &cells,
    const ImageSize3D &image_size,
    const BaseConfig &config)
{
    DerivedGeometry g;

    // Filter to non-trash cells
    std::vector<const Ellipsoid*> realCells;
    for (const auto &c : cells) {
        if (!c.isTrash()) realCells.push_back(&c);
    }
    if (realCells.empty()) return g;

    g.cell_count = realCells.size();

    // Compute mean / max radii per axis
    float sum_a = 0.0f, sum_b = 0.0f, sum_c = 0.0f;
    float max_a = 0.0f, max_b = 0.0f, max_c = 0.0f;
    for (const auto *c : realCells) {
        sum_a += c->getARadius();
        sum_b += c->getBRadius();
        sum_c += c->getCRadius();
        max_a = std::max(max_a, c->getARadius());
        max_b = std::max(max_b, c->getBRadius());
        max_c = std::max(max_c, c->getCRadius());
    }
    const float n = static_cast<float>(realCells.size());
    g.mean_a_radius = sum_a / n;
    g.mean_b_radius = sum_b / n;
    g.mean_c_radius = sum_c / n;

    // Apply formulas
    g.max_a_radius = max_a * 2.0f;
    g.max_b_radius = max_b * 2.0f;
    g.max_c_radius = max_c * 2.0f;
    g.min_a_radius = g.mean_a_radius * 0.2f;
    g.min_b_radius = g.mean_b_radius * 0.2f;
    g.min_c_radius = g.mean_c_radius * 0.2f;
    g.perturb_reference_radius = g.mean_a_radius;  // major-radius proxy
    g.x_sigma = g.mean_a_radius * 0.15f;
    g.y_sigma = g.mean_b_radius * 0.15f;
    g.z_sigma = g.mean_c_radius * 0.30f;
    g.position_prior_threshold = g.mean_a_radius * 0.8f;

    // iterations_per_cell = clamp(n_cells * 30, 150, 500)
    const int rawIters = static_cast<int>(g.cell_count) * 30;
    g.iterations_per_cell = std::clamp(rawIters, 150, 500);

    // pca_bridge_min_side_voxels = clamp(mean_cell_volume * 0.05, 20, 200)
    // Mean cell volume in voxels (oblate spheroid: 4/3 * pi * a*b*c).
    const double meanVolume =
        (4.0 / 3.0) * 3.14159265358979 *
        static_cast<double>(g.mean_a_radius) *
        static_cast<double>(g.mean_b_radius) *
        static_cast<double>(g.mean_c_radius);
    const int rawMinSide = static_cast<int>(meanVolume * 0.05);
    g.pca_bridge_min_side_voxels = std::clamp(rawMinSide, 20, 200);

    // Apply per-field manual overrides (if > 0 / > -1 for int)
    const auto &sim = config.simulation;
    if (sim.geometry_force_max_a_radius > 0.0f) g.max_a_radius = sim.geometry_force_max_a_radius;
    if (sim.geometry_force_max_b_radius > 0.0f) g.max_b_radius = sim.geometry_force_max_b_radius;
    if (sim.geometry_force_max_c_radius > 0.0f) g.max_c_radius = sim.geometry_force_max_c_radius;
    if (sim.geometry_force_min_a_radius > 0.0f) g.min_a_radius = sim.geometry_force_min_a_radius;
    if (sim.geometry_force_min_b_radius > 0.0f) g.min_b_radius = sim.geometry_force_min_b_radius;
    if (sim.geometry_force_min_c_radius > 0.0f) g.min_c_radius = sim.geometry_force_min_c_radius;
    if (sim.geometry_force_perturb_reference_radius > 0.0f) g.perturb_reference_radius = sim.geometry_force_perturb_reference_radius;
    if (sim.geometry_force_xy_sigma > 0.0f) {
        g.x_sigma = sim.geometry_force_xy_sigma;
        g.y_sigma = sim.geometry_force_xy_sigma;
    }
    if (sim.geometry_force_z_sigma > 0.0f) g.z_sigma = sim.geometry_force_z_sigma;
    if (sim.geometry_force_position_prior_threshold > 0.0f) g.position_prior_threshold = sim.geometry_force_position_prior_threshold;
    if (sim.geometry_force_iterations_per_cell > 0) g.iterations_per_cell = sim.geometry_force_iterations_per_cell;
    if (sim.geometry_force_pca_bridge_min_side_voxels > 0) g.pca_bridge_min_side_voxels = sim.geometry_force_pca_bridge_min_side_voxels;

    return g;
}
```

Note: this function does NOT use `image_size` in the formulas — it's reserved for future use (e.g. clamping bounds against image dimensions). Including it in the signature now avoids a future API break.

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd C++/build && cmake --build . --target core_unit_test -j 4
./tests/core_unit_test --gtest_filter="ComputeDerivedGeometryTest.*"
```

Expected: 5 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add C++/includes/CellUniverse.hpp C++/includes/GeometryDerivation.hpp C++/src/CellUniverse.cpp C++/tests/geometry_derivation_test.cc
git commit -m "implement computeDerivedGeometry (initial-CSV statistics → all geometry params)"
```

---

### Task 4: Helper to apply DerivedGeometry to BaseConfig + Ellipsoid::cellConfig

**Files:**
- Modify: `C++/includes/CellUniverse.hpp` (add static apply method)
- Modify: `C++/src/CellUniverse.cpp` (impl)
- Test: `C++/tests/geometry_derivation_test.cc`

- [ ] **Step 1: Write the failing test**

Append to `C++/tests/geometry_derivation_test.cc`:
```cpp
TEST(ApplyDerivedGeometryTest, MutatesEllipsoidCellConfigAndBaseConfig) {
    // Save originals to restore after test
    EllipsoidConfig savedCellCfg = Ellipsoid::cellConfig;

    BaseConfig cfg;
    cfg.cell = std::make_unique<EllipsoidConfig>();
    cfg.cell->maxARadius = 70.0;  // pre-derive value
    cfg.cell->minARadius = 5.0;
    cfg.cell->perturbSigmaReferenceRadius = 25.0f;
    cfg.cell->x.sigma = 5.0f;
    cfg.cell->y.sigma = 5.0f;
    cfg.cell->z.sigma = 8.0f;
    Ellipsoid::cellConfig = *cfg.cell;

    DerivedGeometry g;
    g.max_a_radius = 80.0f;
    g.max_b_radius = 80.0f;
    g.max_c_radius = 96.0f;
    g.min_a_radius = 7.8f;
    g.min_b_radius = 7.8f;
    g.min_c_radius = 9.6f;
    g.perturb_reference_radius = 39.0f;
    g.x_sigma = 5.85f;
    g.y_sigma = 5.85f;
    g.z_sigma = 14.4f;
    g.position_prior_threshold = 31.2f;
    g.iterations_per_cell = 240;
    g.pca_bridge_min_side_voxels = 60;
    g.cell_count = 4;
    g.mean_a_radius = 39.0f;
    g.mean_b_radius = 39.0f;
    g.mean_c_radius = 48.0f;

    CellUniverse::applyDerivedGeometry(g, cfg);

    // BaseConfig.cell.* mutated
    EXPECT_DOUBLE_EQ(cfg.cell->maxARadius, 80.0);
    EXPECT_DOUBLE_EQ(cfg.cell->minARadius, 7.8);
    EXPECT_DOUBLE_EQ(cfg.cell->maxCRadius, 96.0);
    EXPECT_FLOAT_EQ(cfg.cell->perturbSigmaReferenceRadius, 39.0f);
    EXPECT_FLOAT_EQ(cfg.cell->x.sigma, 5.85f);
    EXPECT_FLOAT_EQ(cfg.cell->y.sigma, 5.85f);
    EXPECT_FLOAT_EQ(cfg.cell->z.sigma, 14.4f);

    // Ellipsoid::cellConfig (the global static) ALSO mutated
    EXPECT_DOUBLE_EQ(Ellipsoid::cellConfig.maxARadius, 80.0);
    EXPECT_FLOAT_EQ(Ellipsoid::cellConfig.x.sigma, 5.85f);

    // SimulationConfig.iterations_per_cell mutated
    EXPECT_EQ(cfg.simulation.iterations_per_cell, 240);

    // ProbabilityConfig.position_prior_threshold mutated
    EXPECT_FLOAT_EQ(cfg.prob.position_prior_threshold, 31.2f);
    // pca_bridge_min_side_voxels mutated
    EXPECT_EQ(cfg.prob.pca_bridge_min_side_voxels, 60);

    // Restore for other tests
    Ellipsoid::cellConfig = savedCellCfg;
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd C++/build && cmake --build . --target core_unit_test -j 4
```

Expected: compile error `'applyDerivedGeometry' is not a member`.

- [ ] **Step 3: Add the static method**

`C++/includes/CellUniverse.hpp` `public:` section:
```cpp
    // Apply a DerivedGeometry result to BaseConfig (cell + simulation +
    // prob blocks) and to the Ellipsoid::cellConfig global static.
    // Idempotent — safe to call multiple times.
    static void applyDerivedGeometry(const DerivedGeometry &g, BaseConfig &config);
```

`C++/src/CellUniverse.cpp` (after `computeDerivedGeometry`):
```cpp
void CellUniverse::applyDerivedGeometry(const DerivedGeometry &g, BaseConfig &config)
{
    if (!g.isValid()) {
        std::cout << "[Auto-Derive Geometry] derived geometry invalid; skipping apply\n";
        return;
    }

    if (config.cell) {
        config.cell->maxARadius = g.max_a_radius;
        config.cell->maxBRadius = g.max_b_radius;
        config.cell->maxCRadius = g.max_c_radius;
        config.cell->minARadius = g.min_a_radius;
        config.cell->minBRadius = g.min_b_radius;
        config.cell->minCRadius = g.min_c_radius;
        config.cell->perturbSigmaReferenceRadius = g.perturb_reference_radius;
        config.cell->x.sigma = g.x_sigma;
        config.cell->y.sigma = g.y_sigma;
        config.cell->z.sigma = g.z_sigma;
    }

    config.simulation.iterations_per_cell = g.iterations_per_cell;
    config.prob.position_prior_threshold = g.position_prior_threshold;
    config.prob.pca_bridge_min_side_voxels = g.pca_bridge_min_side_voxels;

    // Sync the Ellipsoid global static — main.cpp normally does this
    // line ONCE after parsing config (`Ellipsoid::cellConfig = *config.cell;`).
    // Doing it here too keeps the pre-CellUniverse-construction code path
    // consistent for any code that reads cellConfig directly.
    if (config.cell) {
        Ellipsoid::cellConfig = *config.cell;
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd C++/build && cmake --build . --target core_unit_test -j 4
./tests/core_unit_test --gtest_filter="ApplyDerivedGeometryTest.*"
```

Expected: PASS. Also run full suite — should be 37 + 7 = 44 tests passing.

- [ ] **Step 5: Commit**

```bash
git add C++/includes/CellUniverse.hpp C++/src/CellUniverse.cpp C++/tests/geometry_derivation_test.cc
git commit -m "add applyDerivedGeometry — mutates cell/sim/prob configs from DerivedGeometry"
```

---

## Phase 2 — Apply Derived Geometry

### Task 5: Wire into main.cpp before auto-calibration

**Files:**
- Modify: `C++/src/main.cpp`

- [ ] **Step 1: Find the wiring point**

Open `C++/src/main.cpp`. The current order (post-Task-4 of the auto-calibration plan) is:
```
1. CellFactory.createCells (line ~151)
2. Auto-calibration block (line ~155-172)
3. CellUniverse construction (line ~175)
4. setBrightnessCalibration (line ~177)
```

The auto-derive call must run AFTER `createCells` (it needs cells) and BEFORE `Ellipsoid::cellConfig = *config.cell` (which is line 127). But line 127 runs BEFORE line 151. That's a problem — the cell factory uses `Ellipsoid::cellConfig` for parsing, so the cellConfig is set before cells are loaded.

The plan: keep `Ellipsoid::cellConfig = *config.cell;` at line 127 (so initial cell construction works), then after createCells, BEFORE the auto-calibration block, recompute and re-apply the derived geometry. The geometry override at this point updates the bounds and sigmas; the constructor's M2 probe and downstream optimizer use the updated values.

- [ ] **Step 2: Add the auto-derive block**

In `C++/src/main.cpp`, AFTER `cellFactory.createCells(...)` returns and BEFORE the existing `BrightnessCalibration calibration;` line, insert:

```cpp
    // Auto-derive geometry from initial CSV statistics + z_scaling. Must run
    // BEFORE auto-calibration and BEFORE CellUniverse construction so the
    // M2 probe + optimizer loop see correct radius bounds and perturbation
    // sigmas. See plan
    // docs/plans/2026-05-05-auto-derive-geometry-from-initial-csv.md.
    if (config.simulation.auto_derive_geometry_enabled) {
        if (cells.empty()) {
            std::cerr << "[Auto-Derive Geometry] cells empty; skipping derivation\n";
        } else {
            const auto &firstFrameCells = cells.begin()->second;
            ImageSize3D image_size;
            if (!imageFilePaths.empty()) {
                std::vector<cv::Mat> firstFrameRaw = ImageHandler::loadRawFrame(
                    imageFilePaths.front().string(), config);
                if (!firstFrameRaw.empty()) {
                    image_size.width = firstFrameRaw[0].cols;
                    image_size.height = firstFrameRaw[0].rows;
                    image_size.depth = static_cast<int>(firstFrameRaw.size());
                }
            }
            DerivedGeometry derived = CellUniverse::computeDerivedGeometry(
                firstFrameCells, image_size, config);
            std::cout << "[Auto-Derive Geometry] " << derivedGeometryDescribe(derived)
                      << std::endl;
            CellUniverse::applyDerivedGeometry(derived, config);
        }
    }
```

Add includes at the top of `main.cpp`:
```cpp
#include "GeometryDerivation.hpp"
```

- [ ] **Step 3: Build and smoke-check**

```bash
cd C++/build && cmake --build . --target celluniverse core_unit_test -j 4
./tests/core_unit_test  # expect 44/44
```

Expected: builds clean, all tests pass, no regression.

- [ ] **Step 4: Quick startup-only check on the fluo dataset**

Run a 1-frame test just to see the new log lines fire correctly:
```bash
cd C++ && ./build/celluniverse 0 0 \
    data/input/embryo_data/t%03d.tif \
    outputs/output_geom_smoke_$(date +%Y%m%d_%H%M%S) \
    config/config.yaml \
    config/initial_auto.csv 2>&1 | head -50
```

Expected output should contain:
```
[Auto-Derive Geometry] max_a=80 max_b=80 max_c=96 min_a=7.8 min_b=7.8 min_c=9.6 ref_r=39 xy_sigma=5.85 z_sigma=14.4 ...
[Auto Calibration] background=24.4828 cell=86.9333 ...
```

If those lines appear with the expected values, the wiring is correct.

- [ ] **Step 5: Commit**

```bash
git add C++/src/main.cpp
git commit -m "wire computeDerivedGeometry/applyDerivedGeometry into main.cpp"
```

---

### Task 6: Per-cell `Ellipsoid` re-construction with new bounds

**Files:**
- Modify: `C++/src/main.cpp`

- [ ] **Step 1: Investigate whether existing cells need re-construction**

After `applyDerivedGeometry`, `Ellipsoid::cellConfig.maxARadius` etc. are updated. But the cells already constructed by `cellFactory.createCells` have radii assigned from the CSV — they don't read `cellConfig.maxARadius` at construction time. So existing cells are fine.

However, the `Ellipsoid` constructor or `setRadii` paths may CLAMP against `cellConfig.maxARadius`. Run grep to verify:

```bash
grep -nE "cellConfig\.maxARadius|cellConfig\.minARadius" C++/src/Ellipsoid.cpp C++/includes/Ellipsoid.hpp
```

If clamping happens at construction, the issue is moot because the new bounds are LARGER (80 vs 70 for fluo) — initial radii (40) are well within both.

If clamping happens at perturbation (`getPerturbedCell`), the new bounds take effect from the next frame's perturbation, which is what we want.

If a tighter `min*Radius` is the issue (e.g. min went up from 5 to 7.8 for fluo) — verify no existing cell in the initial CSV has a radius below 7.8. Quickest check:

```bash
awk -F, 'NR>1 && $12==0 && ($6<7.8 || $7<7.8 || $8<9.6) {print $0}' C++/config/initial_auto.csv
```

If empty: safe. If any row prints: that cell's radius is below the new min, and the next perturbation could clamp it up. Acceptable for fluo (no rows below 7.8 in `initial_auto.csv` given the 38–48 radii). For embryo, e3d03 has aR=14.5 which is above the embryo-derived min=5.05, so safe there too.

- [ ] **Step 2: No code change needed — just document the assumption**

Add a comment to the auto-derive block in `main.cpp` (right after `applyDerivedGeometry(...)`):
```cpp
            // After applyDerivedGeometry, Ellipsoid::cellConfig is updated.
            // Existing cells from CellFactory keep their initial radii (no
            // re-clamping); the new bounds take effect from the next
            // perturbation. The CellFactory's initial cell radii must be
            // within [min_*_radius, max_*_radius] of the derived bounds —
            // the formulas (min = mean × 0.2, max = max × 2) guarantee
            // this for any well-formed initial CSV.
```

- [ ] **Step 3: Commit (just the comment)**

```bash
git add C++/src/main.cpp
git commit -m "document: derived-geometry bounds compatible with initial CSV by construction"
```

---

## Phase 3 — Validation

### Task 7: Re-run fluo f0–6 calibrated smoke test

**Files:** none modified. Validation only.

- [ ] **Step 1: Run the fluo smoke**

```bash
cd C++ && ./build/celluniverse 0 5 \
    data/input/embryo_data/t%03d.tif \
    outputs/output_geom_fluo_smoke_$(date +%Y%m%d_%H%M%S) \
    config/config.yaml \
    config/initial_auto.csv > .../debug_log.txt 2>&1
```

Expected log content:
- `[Auto-Derive Geometry]` with fluo numbers (max_a=80, ref_r=39, xy_sigma=5.85, z_sigma=14.4, iters≈150)
- `[Auto Calibration] background=24.4828 cell=86.9333 ...`
- `[Preprocess Dispatch] file=t000.tif path=calibrated`
- 6 `[Optimize Done]` lines, f4 with `final_cells=8` (cell_1 + cell_2 split)

- [ ] **Step 2: Compare against the no-geom smoke (smoke3)**

If the new run produces fewer splits or wrong cell counts at f4, the auto-derived sigmas are off (likely too small for fluo). Reference: smoke3 (`output_calibrated_smoke3_20260504_235756`) had `[Optimize Done] frame 4 ... split_accepted=2 final_cells=8`.

Acceptance: f4 splits both cell_1 and cell_2, total f4 = 8 cells. If matches → proceed to Task 8.

- [ ] **Step 3: Pass criterion documented in plan**

Acceptance criteria for Task 7: the fluo f0–5 smoke must produce identical split events and identical final cell counts at every frame as smoke3 (the auto-cal-only baseline). Wall time may vary ±20%.

If it fails, the most likely culprit is `z_sigma = 14.4` vs the previous `8` — z motion is now 80% larger per step. Cells may overshoot and miss the small z-motion in the data. Mitigation: tighten the z formula to `mean_c × 0.20` (would give 9.6 instead of 14.4).

---

### Task 8: Re-run embryo f1–6 calibrated smoke test

**Files:** none modified. Validation only.

- [ ] **Step 1: Run the embryo smoke**

```bash
cd C++ && ./build/celluniverse 1 6 \
    data/input/original_data/frame%03d.tif \
    outputs/output_geom_embryo_smoke_$(date +%Y%m%d_%H%M%S) \
    config/config.yaml \
    config/initial.csv > .../debug_log.txt 2>&1
```

Expected log content:
- `[Auto-Derive Geometry]` with embryo numbers (max_a≈62, ref_r≈25.27, xy_sigma≈3.79, z_sigma≈7.58, iters=180)
- `[Auto Calibration] background=108.55 cell=121.38 span=12.84 ...`
- 6 `[Optimize Done]` lines, f3 with `final_cells=8` (12345 + e9077 split)

- [ ] **Step 2: Compare against orig_smoke baseline**

Reference: `output_calibrated_orig_smoke_20260505_001230` had f3 splits 12345 + e9077, final_cells=8.

Acceptance: f3 produces 2 splits matching the baseline. If matches → proceed to Task 9.

If embryo splits are missing, the most likely culprit is `z_sigma = 7.58` vs previous `8` — only 5% reduction, should be fine. If it fails anyway, could be `iters=180` vs previous 250 — 28% fewer iterations may not be enough for embryo. Mitigation: bump the iters formula floor to 250 (would give 250 for ≤8 cells, scaling up after that).

---

### Task 9: Full f0–50 fluo validation against auto-calibrated baseline

**Files:** none modified. Validation only. This is the equivalent of Task 11 in the auto-calibration plan.

- [ ] **Step 1: Push branch + pull on Linux WSL (per project workflow)**

The user runs:
```bash
git push -u origin jl_auto_geometry_05052026
# on Linux:
cd /home/jihangl3/CellUniverse && git fetch && git checkout jl_auto_geometry_05052026 && git pull
```

- [ ] **Step 2: Run f0–50 on Linux**

```bash
cd /home/jihangl3/CellUniverse/C++ && \
  scripts/run_celluniverse.sh config/user_input_config_embryo.ini fluo_0-50
```

(Adjust the preset name if needed — should be the same one used for the prior auto-calibrated baseline run.)

Estimated wall time: ~16–18 hr (similar to the auto-calibrated run, since the auto-derive overhead is one-shot ~5 sec).

- [ ] **Step 3: Compare cell counts against baseline**

Reference: `output_fluo_0-50_20260505_005405` (the validated auto-calibrated baseline, NOT the bridge-demote baseline).

Acceptance criteria:
- Final cell count at f50: 26 (matches GT)
- Total splits accepted: 22 (matches baseline)
- Lineage topology: same set of cell IDs split (timing wobbles ≤ ±2 frames acceptable)
- No NEW false splits in f36–44 quiet window beyond what the auto-cal baseline already had (1 false at f44)
- Cost rescues: ≤ 10 (baseline had 5; some increase acceptable as stricter sigmas may push more cases into the rescue path)

Compare with the helper:
```bash
C++/scripts/compare_runs_split_events.sh \
    C++/outputs/output_fluo_0-50_20260505_005405 \
    C++/outputs/output_fluo_<NEW_TIMESTAMP>
```

- [ ] **Step 4: Cell-shape sanity check**

Compare mean radii at f50:
```bash
awk -F, '$1=="t050.tif" && $12==0 {sum_a+=$6; sum_c+=$8; n++}
        END {printf "mean(aR,cR)=(%.1f, %.1f)\n", sum_a/n, sum_c/n}' .../cells.csv
```

Auto-calibrated baseline: mean(aR, cR) = (31.3, 19.8). New run should be within ±15%.

- [ ] **Step 5: Document findings**

Append a "Validation Findings" section to this plan (similar to the auto-calibration plan's Task 13). Include: wall time, split count, final cell count, any new false splits, mean radii, and any remaining risk items.

If acceptance criteria pass: proceed to Phase 4. If they fail by a meaningful margin, revisit the formulas (most likely the z_sigma multiplier or the iters floor) before pushing.

---

## Phase 4 — Documentation

### Task 10: Append Change 15 to changelogv9.md

**Files:**
- Modify: `C++/docs/changelogs/changelogv9.md`

- [ ] **Step 1: Append the entry**

Append to the bottom of `C++/docs/changelogs/changelogv9.md`:
```markdown
## 2026-05-05: Auto-derive geometry from initial CSV (Change 15) — **status: ACTIVE**

### Problem / Motivation

Auto-calibration (Change 14) eliminated dataset-specific brightness tunables but left ~10 dataset-specific geometry/motion knobs (cell-radius bounds, perturbation sigmas, position-prior threshold, iterations_per_cell). Multi-dataset users still had to measure cell sizes and motion characteristics by hand.

Solution: derive all of these from initial-CSV statistics + `z_scaling` at startup. After this lands, a new dataset only needs (1) initial CSV with cell positions and radii, (2) `z_scaling`. Plan: `docs/plans/2026-05-05-auto-derive-geometry-from-initial-csv.md`.

### Files changed

- `C++/includes/GeometryDerivation.hpp` (new)
- `C++/includes/CellUniverse.hpp` (added `computeDerivedGeometry`, `applyDerivedGeometry`)
- `C++/includes/ConfigTypes.hpp` (added `auto_derive_geometry_enabled` + 12 manual-override fields)
- `C++/src/CellUniverse.cpp` (implementation)
- `C++/src/main.cpp` (wired auto-derive before auto-calibration)
- `C++/config/config.yaml` (new keys)
- `C++/tests/geometry_derivation_test.cc` (new — 7 tests)
- `C++/tests/CMakeLists.txt` (registered new test file)

### Formulas

| Param | Formula |
|---|---|
| `maxARadius/maxBRadius/maxCRadius` | `max(initial radii) × 2.0` |
| `minARadius/minBRadius/minCRadius` | `mean(initial radii) × 0.2` |
| `perturbSigmaReferenceRadius` | `mean(major radii)` |
| `cell.x.sigma` / `cell.y.sigma` | `mean(major radii) × 0.15` |
| `cell.z.sigma` | `mean(minor radii) × 0.30` |
| `position_prior_threshold` | `mean(major radii) × 0.8` |
| `iterations_per_cell` | `clamp(num_cells × 30, 150, 500)` |
| `pca_bridge_min_side_voxels` | `clamp(mean cell volume × 0.05, 20, 200)` |

Per-param manual override via `geometry_force_*` config fields takes precedence.

### Validation

Compared against `output_fluo_0-50_20260505_005405` (the validated auto-calibrated baseline). Final cell count and lineage match GT. See plan file for detailed numbers.

### Open follow-ups

- **Bound-clamping against image dimensions.** Currently `image_size` is plumbed but unused in formulas. Future: `max*Radius = min(formula, image_dimension × 0.4)` to prevent cells from being told they could grow larger than the image.
- **Per-cell-cluster derivations.** If a dataset has heterogeneous cell sizes (some 14 px, some 31 px), the mean smooths over the variation. Could derive per-cluster bounds, applied based on cell membership.
- **Auto-detect z_scaling from TIFF metadata** if the dataset publishes physical pixel sizes. Would reduce user input from 2 to 1.
```

- [ ] **Step 2: Commit**

```bash
git add C++/docs/changelogs/changelogv9.md
git commit -m "changelog: Change 15 — auto-derive geometry from initial CSV"
```

---

### Task 11: Update CLAUDE.md "two-input workflow" note

**Files:**
- Modify: `.claude/CLAUDE.md`

- [ ] **Step 1: Add a workflow section**

In `.claude/CLAUDE.md`, after the existing "## Project" section, insert:

```markdown
## New-dataset workflow (post-2026-05-05)

A new dataset needs only TWO user inputs:

1. **Initial CSV** — cell positions and radii in frame 0. Format per
   `C++/config/initial_auto.csv` or `C++/config/initial.csv`. Required
   columns: `file, name, x, y, z, aRadius, bRadius, cRadius, isTrash`.
   Mark non-cells with `isTrash=1` to suppress them.

2. **`z_scaling`** in `C++/config/config.yaml` (`simulation: z_scaling: N`).
   Compute as `physical_z_step / xy_pixel_size`. Fluo dataset uses 7;
   isotropic-voxel datasets use 1.

Everything else auto-derives:
- Brightness anchors via auto-calibration on frame 0 (Change 14)
- Cell-radius bounds, perturbation sigmas, perturb reference radius,
  position-prior threshold, iterations_per_cell, pca_bridge min voxels
  via auto-derive geometry on initial CSV statistics (Change 15)

Per-param manual overrides available via `simulation.geometry_force_*`
in YAML for the rare case where auto-derive picks the wrong value
(e.g. extreme cell aspect ratios skewing the mean).
```

- [ ] **Step 2: Commit**

```bash
git add .claude/CLAUDE.md
git commit -m "docs: add two-input new-dataset workflow note to CLAUDE.md"
```

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Cells with extreme aspect ratios skew the mean.** A dataset with one elongated cell + many round cells could pull `mean_a_radius` up, making perturb sigma too large for the round cells. | Low for fluo/embryo, medium for new datasets. | Slow convergence on small cells. | The `perturbSigmaReferenceRadius` self-correcting mechanism scales each cell's perturb step by `cell.maxRadius / reference_radius`, so a small cell uses `(small_radius / mean_radius) × x_sigma` — already proportional to cell size. Manual override via `geometry_force_xy_sigma` available. |
| **Initial CSV with bad ROI radii** (user roughly drew bounding boxes that don't match cell extent). | Medium for new users. | Wrong calibration anchor (cells sample from halo) AND wrong geometry bounds (max could be 2× a wrong starting radius). | Document in CLAUDE.md that initial CSV radii must reflect cell core, not bbox. Manual override fallback. Worth a future automated cell-radius estimator that runs on frame 0 against a user-clicked centroid. |
| **z_sigma reduction for embryo** (8 → 7.58, formula `mean_c × 0.30`). | Low. | Marginally slower z convergence, ~5% reduction. | Within tolerance. Validated in Task 8 (embryo smoke). If regression appears, raise multiplier to 0.32 (gives 8.09 for embryo, 15.4 for fluo). |
| **z_sigma INCREASE for fluo** (8 → 14.4, formula `mean_c × 0.30`). | Medium. | Larger steps may overshoot small z motion. | Validated in Task 7 (fluo smoke) and Task 9 (full fluo run). If regression: drop multiplier to 0.20 (gives 9.6 for fluo, matches old value 8 better). |
| **iterations_per_cell halved for small datasets** (250 → 150 for ≤5 cells, 250 → 180 for 6 cells). | High for early-frame embryo runs. | Slower convergence per frame, may miss splits in budget. | Floor at 150 may be too low. If validation shows missed splits, raise floor to 200 or 250. Trivial config change, no code change required. |
| **Manual-override precedence.** If user sets `geometry_force_max_a_radius: 50` AND has a cell at radius 60 in the initial CSV, the cell will be clamped down on first perturbation. | Low (user error). | Cells could shrink unexpectedly. | Document the precedence rule. Validate at startup that any forced max ≥ max initial radius (warn if not). Out of scope for this plan. |
| **Embryo dataset with 6 cells gets `iterations_per_cell = 180`** (formula `6 × 30 = 180`). Previous YAML value was 250. | Medium. | Embryo runs may take fewer iterations to converge per frame. | Validated in Task 8. If regression: bump floor to 250. |
| **Test target depends on `cv::Vec3i` or `ImageSize3D` type choice.** Wrong choice = compile error. | Low. | Build failure. | Task 3 step 2 instructs the implementer to verify and adjust. Falls back to the local `ImageSize3D` POD struct in GeometryDerivation.hpp. |

---

## Self-Review

**1. Spec coverage:**
- ✓ `auto_derive_geometry_enabled: true` config flag (Task 2)
- ✓ `computeDerivedGeometry(cells, image_size, baseConfig)` (Task 3)
- ✓ Mutates `Ellipsoid::cellConfig` and `BaseConfig` (Task 4)
- ✓ Wired into main.cpp BEFORE auto-calibration (Task 5)
- ✓ Validation against `output_fluo_0-50_20260505_005405` baseline (Task 9)
- ✓ All 8 formulas as specified (Task 3 step 5)
- ✓ Manual override fallback (Task 2 + Task 3)
- ✓ Risk register with 3+ items including extreme aspect ratios, bad ROI radii, embryo z.sigma reduction (Risk Register section)
- ✓ Branch `jl_auto_geometry_05052026` off `jl_auto_calibrate_05042026` (top of plan)

**2. Placeholder scan:** No "TBD", no "implement later", no "similar to Task N". All test code, all production code, all commands are concrete.

**3. Type consistency:**
- `DerivedGeometry` struct fields: all named `*_radius` / `*_sigma` / etc. consistently across Tasks 1, 3, 4, 5.
- `ImageSize3D` struct: `width / height / depth` (raw z, pre-interpolation). Consistent in tests + impl.
- `computeDerivedGeometry(cells, image_size, baseConfig) -> DerivedGeometry` — same signature in Tasks 3, 4, 5.
- `applyDerivedGeometry(g, baseConfig) -> void` — same signature in Tasks 4, 5.
- Manual override field naming: `geometry_force_<param>` consistent in Tasks 2 and 3.

---

## Validation Findings

(Filled in during Task 9 after the validation run completes.)
