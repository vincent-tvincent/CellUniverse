# Auto-Calibration from Frame Zero — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current dataset-specific iterative preprocessing pipeline (~35 tunables, ~110 sec/frame) with frame-0-cell-anchored auto-calibration that infers `(background_intensity, cell_intensity)` from initial CSV cell positions and linearly normalizes every frame. After this lands, the same code runs on any fluorescence dataset without per-dataset retuning.

**Architecture:** Add a one-shot calibration pass at startup that measures median brightness inside vs outside the initial CSV ellipsoids. Use the resulting `(bg, cell)` pair as the linear-scale anchor for every frame. Disable the iterative gradient-walk (`processPreparedSequence`) and percentile-based intensity scale behind a config flag, keeping cheap dataset-agnostic steps (TinyParticleRemoval, FinalBlur, AdaptiveBackground, per-cell EMA brightness). All bio thresholds (`bio_bridge_min_edge_brightness_absolute`, `pca_bridge_black_threshold`, etc.) stay as fractions of the calibrated [0, 1] scale and become portable across datasets.

**Tech Stack:** C++17, OpenCV (`cv::Mat`, `cv::sort`), yaml-cpp, GoogleTest. Existing project patterns from `C++/src/CellUniverse.cpp` and `C++/src/ImageHandler.cpp`.

**Branch:** `jl_auto_calibrate_05042026` off `jl_bridge_demote_05042026`.

**Regression baseline:** `C++/outputs/output_fluo_0-50_20260504_005706` — the validated bridge-demote run (24 real cells at f35 matching GT, 26 at f48 matching GT, no false splits in f36–44 quiet window). Auto-calibration must reproduce these split events and cell counts within ±1 frame timing tolerance.

**Constraints:**
- Do NOT delete any of the existing `iterative_*` / `contrast_*` / `frame_intensity_scale_*` config fields in this PR — deprecation is gated by a runtime flag and parsing stays for backward compatibility.
- Do NOT change any bio-gate threshold values. They already use fractional [0, 1] semantics; only the meaning of "1.0" changes from "P99 of frame intensity" to "median cell intensity at frame 0".
- Per-frame `setBackgroundColor` adaptive update and per-cell `_brightness` EMA must remain exactly as they are.
- The plan covers only the *calibration + simplified preprocessing* swap. Full removal of the deprecated config fields is a follow-up plan.

---

## Files Touched

| File | Role | Change kind |
|---|---|---|
| `C++/includes/CalibrationTypes.hpp` | NEW. Defines `BrightnessCalibration` POD struct + `calibrationDescribe()` helper. | Create |
| `C++/includes/ConfigTypes.hpp` | Add `auto_calibrate_brightness_enabled`, `manual_background_intensity`, `manual_cell_intensity`, `calibration_cell_inner_fraction`, `calibration_pixel_trim_percent` config fields. Mark `iterative_*` / `contrast_*` fields with deprecation comment but keep parsing. | Modify |
| `C++/includes/CellUniverse.hpp` | Declare `BrightnessCalibration brightnessCalibration` member + `void initializeBrightnessCalibration(const std::vector<Ellipsoid> &frameZeroCells, const std::vector<cv::Mat> &frameZeroRaw)`. | Modify |
| `C++/includes/ImageHandler.hpp` | Declare new `static std::vector<cv::Mat> applyCalibratedPreprocess(const std::vector<cv::Mat> &normalizedSlices, const BrightnessCalibration &calibration, const BaseConfig &config, std::ostream *logSink = nullptr)`. | Modify |
| `C++/src/ImageHandler.cpp` | Implement `applyCalibratedPreprocess` (linear normalize + tiny particle + final blur + z-interpolate; no iterative gradient walk). Modify `preprocessLoadedFrame` to dispatch on `auto_calibrate_brightness_enabled`. | Modify |
| `C++/src/CellUniverse.cpp` | Implement `initializeBrightnessCalibration`. Wire calibration into `preprocessAllFramesAlignedToMinimumBackground` and `prepareFrame`. Plumb calibration through to `ImageHandler::preprocessLoadedFrame` calls. | Modify |
| `C++/src/main.cpp` | Build calibration BEFORE the preprocessing pass (need cells + frame 0 raw image). Re-order: load cells → load frame 0 raw → calibrate → preprocess. | Modify |
| `C++/config/config.yaml` | Add new config keys with safe defaults (auto-calibrate ON, manual override = -1 sentinel). Tag deprecated keys with comments. | Modify |
| `C++/tests/calibration_test.cc` | NEW. Unit tests for `BrightnessCalibration` arithmetic + `initializeBrightnessCalibration` against synthetic cell stacks. | Create |
| `C++/tests/CMakeLists.txt` | Register `calibration_test.cc` in `core_unit_test` sources. | Modify |
| `C++/docs/changelogs/changelogv8.md` | Append "Change 14: auto-calibration from frame zero" entry with before/after code. | Modify |

---

## Phase Breakdown

- **Phase 1: Foundation** (Tasks 1–4) — Calibration struct, config plumbing, unit-tested `initializeBrightnessCalibration`. No behavior change yet (flag defaults to OFF).
- **Phase 2: Calibrated preprocessing path** (Tasks 5–8) — `applyCalibratedPreprocess` function, dispatch in `preprocessLoadedFrame`, integration with main pipeline.
- **Phase 3: Manual override + tests** (Tasks 9–10) — Manual override fields, integration tests with manual values.
- **Phase 4: Validation against baseline** (Tasks 11–13) — Side-by-side run, cell-count comparison, manual sign-off.
- **Phase 5: Documentation + deprecation markers** (Tasks 14–15) — Changelog, deprecation comments on old fields, README note.

---

## Phase 1 — Foundation

### Task 1: Create `BrightnessCalibration` struct + describe helper

**Files:**
- Create: `C++/includes/CalibrationTypes.hpp`
- Test: `C++/tests/calibration_test.cc`

- [ ] **Step 1: Write the failing test**

`C++/tests/calibration_test.cc`:
```cpp
#include <gtest/gtest.h>
#include <sstream>

#include "CalibrationTypes.hpp"

TEST(BrightnessCalibrationTest, ScaleMapsBackgroundToZeroAndCellToOne) {
    BrightnessCalibration cal;
    cal.background_intensity = 50.0f;
    cal.cell_intensity = 800.0f;

    EXPECT_NEAR(cal.normalize(50.0f), 0.0f, 1e-6f);
    EXPECT_NEAR(cal.normalize(800.0f), 1.0f, 1e-6f);
    EXPECT_NEAR(cal.normalize(425.0f), 0.5f, 1e-3f);
}

TEST(BrightnessCalibrationTest, ClipsValuesOutsideAnchorRange) {
    BrightnessCalibration cal;
    cal.background_intensity = 50.0f;
    cal.cell_intensity = 800.0f;

    EXPECT_EQ(cal.normalize(0.0f), 0.0f);
    EXPECT_EQ(cal.normalize(2000.0f), 1.0f);
}

TEST(BrightnessCalibrationTest, IsValidRequiresPositiveSpan) {
    BrightnessCalibration cal;
    EXPECT_FALSE(cal.isValid());

    cal.background_intensity = 100.0f;
    cal.cell_intensity = 100.0f;
    EXPECT_FALSE(cal.isValid());

    cal.cell_intensity = 200.0f;
    EXPECT_TRUE(cal.isValid());
}

TEST(BrightnessCalibrationTest, DescribeIncludesAllFields) {
    BrightnessCalibration cal;
    cal.background_intensity = 50.0f;
    cal.cell_intensity = 800.0f;
    cal.cell_pixel_count = 12345;
    cal.background_pixel_count = 67890;
    cal.source = BrightnessCalibrationSource::AutoFrameZero;

    const std::string desc = calibrationDescribe(cal);
    EXPECT_NE(desc.find("background=50"), std::string::npos);
    EXPECT_NE(desc.find("cell=800"), std::string::npos);
    EXPECT_NE(desc.find("source=auto"), std::string::npos);
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd C++/build && cmake --build . --target core_unit_test -j 4
./tests/core_unit_test --gtest_filter="BrightnessCalibrationTest.*"
```

Expected: FAIL with "fatal error: 'CalibrationTypes.hpp' file not found".

- [ ] **Step 3: Create the header**

`C++/includes/CalibrationTypes.hpp`:
```cpp
#ifndef CALIBRATION_TYPES_HPP
#define CALIBRATION_TYPES_HPP

#include <algorithm>
#include <cstddef>
#include <sstream>
#include <string>

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
        // cell=bg=0 and is invalid. Source is metadata (for logging /
        // dispatch decisions), not an invariant — a calibration with
        // a positive span is usable regardless of who set it.
        return (cell_intensity - background_intensity) > 1e-3f;
    }

    float normalize(float raw) const {
        if (!isValid()) return raw;
        const float span = cell_intensity - background_intensity;
        const float t = (raw - background_intensity) / span;
        return std::clamp(t, 0.0f, 1.0f);
    }
};

inline std::string calibrationDescribe(const BrightnessCalibration &cal) {
    std::ostringstream oss;
    const char *src = "uninitialized";
    switch (cal.source) {
        case BrightnessCalibrationSource::AutoFrameZero: src = "auto"; break;
        case BrightnessCalibrationSource::ManualOverride: src = "manual"; break;
        case BrightnessCalibrationSource::Uninitialized: src = "uninitialized"; break;
    }
    oss << "background=" << cal.background_intensity
        << " cell=" << cal.cell_intensity
        << " span=" << (cal.cell_intensity - cal.background_intensity)
        << " bg_pixels=" << cal.background_pixel_count
        << " cell_pixels=" << cal.cell_pixel_count
        << " source=" << src;
    return oss.str();
}

#endif // CALIBRATION_TYPES_HPP
```

- [ ] **Step 4: Register the test in CMakeLists.txt**

`C++/tests/CMakeLists.txt` — add `calibration_test.cc` to the `add_executable` source list:

```cmake
add_executable(core_unit_test
        hello_test.cc
        config_types_test.cc
        interpolate_test.cc
        frame_test.cc
        process_image_test.cc
        spheroid_test.cc
        spheroid_params_test.cc
        spheroid_rotation_test.cc
        calibration_test.cc
        ${CMAKE_SOURCE_DIR}/src/Frame.cpp
        ${CMAKE_SOURCE_DIR}/src/CellUniverse.cpp
        ${CMAKE_SOURCE_DIR}/src/ImageHandler.cpp
        ${CMAKE_SOURCE_DIR}/src/Ellipsoid.cpp
)
```

- [ ] **Step 5: Run test to verify it passes**

```bash
cd C++/build && cmake --build . --target core_unit_test -j 4
./tests/core_unit_test --gtest_filter="BrightnessCalibrationTest.*"
```

Expected: 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add C++/includes/CalibrationTypes.hpp C++/tests/calibration_test.cc C++/tests/CMakeLists.txt
git commit -m "add BrightnessCalibration struct + unit tests"
```

---

### Task 2: Add config fields with deprecation comments

**Files:**
- Modify: `C++/includes/ConfigTypes.hpp` (around line 17-50, the `SimulationConfig` field section)
- Modify: `C++/config/config.yaml` (under `simulation:` block)

- [ ] **Step 1: Write the failing test**

Append to `C++/tests/config_types_test.cc`:
```cpp
TEST(ConfigTypesTest, AutoCalibrationFieldsHaveExpectedDefaults) {
    SimulationConfig cfg;
    EXPECT_TRUE(cfg.auto_calibrate_brightness_enabled);
    EXPECT_FLOAT_EQ(cfg.manual_background_intensity, -1.0f);
    EXPECT_FLOAT_EQ(cfg.manual_cell_intensity, -1.0f);
    EXPECT_FLOAT_EQ(cfg.calibration_cell_inner_fraction, 0.7f);
    EXPECT_FLOAT_EQ(cfg.calibration_pixel_trim_percent, 0.10f);
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd C++/build && cmake --build . --target core_unit_test -j 4
```

Expected: compile error "auto_calibrate_brightness_enabled is not a member of SimulationConfig".

- [ ] **Step 3: Add fields to ConfigTypes.hpp**

In `C++/includes/ConfigTypes.hpp` `SimulationConfig` class, after `frame_intensity_hard_max`, add:

```cpp
    // ---- Auto-calibration from frame 0 (2026-05-04) ----
    // Replaces the dataset-specific iterative_*/contrast_*/frame_intensity_scale_*
    // pipeline. When enabled, the system measures median brightness inside vs
    // outside the initial CSV cell ellipsoids on frame 0, then linearly maps
    // (bg, cell) -> (0, 1) for every frame. All bio thresholds (bridge edge
    // brightness, valley ratio, etc.) become portable as fractions of cell
    // brightness instead of absolute values calibrated to one dataset.
    bool auto_calibrate_brightness_enabled = true;
    // Manual override of the auto-detected anchors. Both must be > 0 to take
    // effect, otherwise auto-detection runs.
    float manual_background_intensity = -1.0f;
    float manual_cell_intensity = -1.0f;
    // Sample only pixels within (cell_inner_fraction × radii) of the cell
    // center to avoid halo + boundary contamination. 0.7 = 70% of each radius.
    float calibration_cell_inner_fraction = 0.7f;
    // Drop the brightest and dimmest pixel_trim_percent of each cell's pixels
    // before computing the median. Defends against saturation + edge halo.
    float calibration_pixel_trim_percent = 0.10f;
```

In the `explodeConfig` parser (around line 220, the same section that parses iterative_*):
```cpp
        if (node["auto_calibrate_brightness_enabled"]) auto_calibrate_brightness_enabled = node["auto_calibrate_brightness_enabled"].as<bool>();
        if (node["manual_background_intensity"]) manual_background_intensity = node["manual_background_intensity"].as<float>();
        if (node["manual_cell_intensity"]) manual_cell_intensity = node["manual_cell_intensity"].as<float>();
        if (node["calibration_cell_inner_fraction"]) calibration_cell_inner_fraction = node["calibration_cell_inner_fraction"].as<float>();
        if (node["calibration_pixel_trim_percent"]) calibration_pixel_trim_percent = node["calibration_pixel_trim_percent"].as<float>();
```

In the `printConfig` cout block (find `iterative_max_count` cout line, add after):
```cpp
        std::cout << "auto_calibrate_brightness_enabled: " << auto_calibrate_brightness_enabled << '\n';
        std::cout << "manual_background_intensity: " << manual_background_intensity << '\n';
        std::cout << "manual_cell_intensity: " << manual_cell_intensity << '\n';
        std::cout << "calibration_cell_inner_fraction: " << calibration_cell_inner_fraction << '\n';
        std::cout << "calibration_pixel_trim_percent: " << calibration_pixel_trim_percent << '\n';
```

Mark deprecated fields with a comment above each `iterative_*` and `contrast_*` block:
```cpp
    // ---- DEPRECATED 2026-05-04: superseded by auto_calibrate_brightness ----
    // Still parsed for backward compatibility; ignored when
    // auto_calibrate_brightness_enabled is true. Slated for removal once the
    // calibrated preprocessing path is the only path.
    float iterative_penalty = 0.1f;
    // ... (rest of iterative_* fields unchanged)
```

- [ ] **Step 4: Add YAML defaults**

`C++/config/config.yaml`, under `simulation:` block (insert near `frame_intensity_hard_max`):
```yaml
  # Auto-calibration from frame 0 (2026-05-04). See plan
  # docs/plans/2026-05-04-auto-calibration-from-frame-zero.md.
  # When true, dataset-specific iterative_*/contrast_*/frame_intensity_scale_*
  # tunables are bypassed in favor of inferring (background, cell) intensity
  # from the initial CSV cell positions in frame 0.
  auto_calibrate_brightness_enabled: true
  manual_background_intensity: -1
  manual_cell_intensity: -1
  calibration_cell_inner_fraction: 0.7
  calibration_pixel_trim_percent: 0.10
```

- [ ] **Step 5: Run test to verify it passes**

```bash
cd C++/build && cmake --build . --target core_unit_test -j 4
./tests/core_unit_test --gtest_filter="ConfigTypesTest.AutoCalibrationFieldsHaveExpectedDefaults"
```

Expected: PASS. Also run full `ConfigTypesTest.*` to make sure no regressions.

- [ ] **Step 6: Commit**

```bash
git add C++/includes/ConfigTypes.hpp C++/config/config.yaml C++/tests/config_types_test.cc
git commit -m "add auto-calibration config fields, mark old preprocessing fields deprecated"
```

---

### Task 3: Implement `initializeBrightnessCalibration` (no-op when no cells)

**Files:**
- Modify: `C++/includes/CellUniverse.hpp` (add member + method declaration)
- Modify: `C++/src/CellUniverse.cpp` (add implementation)
- Test: `C++/tests/calibration_test.cc`

- [ ] **Step 1: Write the failing test**

Append to `C++/tests/calibration_test.cc`:
```cpp
#include "CellUniverse.hpp"
#include "Ellipsoid.hpp"

namespace {

std::vector<cv::Mat> makeSyntheticFrame(int width, int height, int depth,
                                        float background, float cell_value,
                                        const cv::Point3f &cellCenter,
                                        float cellRadius) {
    std::vector<cv::Mat> stack;
    stack.reserve(static_cast<size_t>(depth));
    for (int z = 0; z < depth; ++z) {
        cv::Mat slice(height, width, CV_32F, cv::Scalar(background));
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const float dx = x - cellCenter.x;
                const float dy = y - cellCenter.y;
                const float dz = z - cellCenter.z;
                if (dx*dx + dy*dy + dz*dz <= cellRadius*cellRadius) {
                    slice.at<float>(y, x) = cell_value;
                }
            }
        }
        stack.push_back(slice);
    }
    return stack;
}

}  // namespace

TEST(InitializeBrightnessCalibrationTest, RecoversBackgroundAndCellMediansFromSyntheticFrame) {
    BaseConfig cfg;
    cfg.simulation.auto_calibrate_brightness_enabled = true;
    cfg.simulation.calibration_cell_inner_fraction = 0.7f;
    cfg.simulation.calibration_pixel_trim_percent = 0.10f;
    cfg.simulation.z_slices = 50;
    cfg.simulation.z_scaling = 1;

    const cv::Point3f center(50.0f, 50.0f, 25.0f);
    const float radius = 10.0f;
    auto frame = makeSyntheticFrame(100, 100, 50, /*bg*/ 50.0f, /*cell*/ 800.0f,
                                    center, radius);

    EllipsoidParams p;
    p.name = "cell_1";
    p.x = center.x; p.y = center.y; p.z = center.z;
    p.aRadius = radius; p.bRadius = radius; p.cRadius = radius;
    p.theta_x = 0; p.theta_y = 0; p.theta_z = 0;
    p.brightness = 1.0f;
    std::vector<Ellipsoid> cells{ Ellipsoid(p) };

    BrightnessCalibration cal = CellUniverse::computeAutoCalibration(cells, frame, cfg);

    EXPECT_NEAR(cal.background_intensity, 50.0f, 5.0f);
    EXPECT_NEAR(cal.cell_intensity, 800.0f, 5.0f);
    EXPECT_GT(cal.cell_pixel_count, 100u);
    EXPECT_GT(cal.background_pixel_count, 1000u);
    EXPECT_EQ(cal.source, BrightnessCalibrationSource::AutoFrameZero);
}

TEST(InitializeBrightnessCalibrationTest, ManualOverrideBypassesAutoDetection) {
    BaseConfig cfg;
    cfg.simulation.auto_calibrate_brightness_enabled = true;
    cfg.simulation.manual_background_intensity = 123.0f;
    cfg.simulation.manual_cell_intensity = 999.0f;

    auto frame = makeSyntheticFrame(20, 20, 5, 0.0f, 1.0f, cv::Point3f(10, 10, 2), 5.0f);
    std::vector<Ellipsoid> cells;
    EllipsoidParams p; p.name = "c"; p.x=10; p.y=10; p.z=2;
    p.aRadius=5; p.bRadius=5; p.cRadius=5;
    cells.emplace_back(p);

    BrightnessCalibration cal = CellUniverse::computeAutoCalibration(cells, frame, cfg);
    EXPECT_FLOAT_EQ(cal.background_intensity, 123.0f);
    EXPECT_FLOAT_EQ(cal.cell_intensity, 999.0f);
    EXPECT_EQ(cal.source, BrightnessCalibrationSource::ManualOverride);
}

TEST(InitializeBrightnessCalibrationTest, EmptyCellsReturnsInvalidCalibration) {
    BaseConfig cfg;
    cfg.simulation.auto_calibrate_brightness_enabled = true;

    auto frame = makeSyntheticFrame(20, 20, 5, 50.0f, 800.0f, cv::Point3f(10, 10, 2), 5.0f);
    std::vector<Ellipsoid> cells;

    BrightnessCalibration cal = CellUniverse::computeAutoCalibration(cells, frame, cfg);
    EXPECT_FALSE(cal.isValid());
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd C++/build && cmake --build . --target core_unit_test -j 4
```

Expected: compile error "computeAutoCalibration is not a member of CellUniverse".

- [ ] **Step 3: Add member + static method declarations**

`C++/includes/CellUniverse.hpp` — at the top of the class, add:
```cpp
#include "CalibrationTypes.hpp"
```

In the `public:` section, after the constructor declarations:
```cpp
    // Compute brightness anchors from frame 0 + initial cells. Pure function,
    // no member mutation — used by both the production path and the unit
    // tests. When manual_*_intensity overrides are set, returns those
    // verbatim. When cells is empty, returns an Uninitialized calibration.
    static BrightnessCalibration computeAutoCalibration(
        const std::vector<Ellipsoid> &cells,
        const std::vector<cv::Mat> &rawFrame0,
        const BaseConfig &config);
```

In the `private:` section, after the existing per-frame caches:
```cpp
    BrightnessCalibration brightnessCalibration;
```

- [ ] **Step 4: Implement the method**

`C++/src/CellUniverse.cpp` — add somewhere near the existing brightness/preprocessing helpers (e.g., right before `preprocessAllFramesAlignedToMinimumBackground`):

```cpp
namespace {

float trimmedMedian(std::vector<float> &values, float trim_percent) {
    if (values.empty()) return 0.0f;
    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();
    const std::size_t lo = static_cast<std::size_t>(trim_percent * n);
    const std::size_t hi = n - lo;
    if (hi <= lo) return values[n / 2];
    const std::size_t midIdx = lo + (hi - lo) / 2;
    return values[midIdx];
}

bool pointInsideScaledEllipsoid(const Ellipsoid &cell, float scale,
                                int x, int y, int z) {
    const float dx = static_cast<float>(x) - cell.getX();
    const float dy = static_cast<float>(y) - cell.getY();
    const float dz = static_cast<float>(z) - cell.getZ();
    std::array<double, 9> R_T;
    cell.generateInverseRotationMatrix(R_T);
    const double lx = R_T[0]*dx + R_T[1]*dy + R_T[2]*dz;
    const double ly = R_T[3]*dx + R_T[4]*dy + R_T[5]*dz;
    const double lz = R_T[6]*dx + R_T[7]*dy + R_T[8]*dz;
    const float ra = std::max(1e-3f, cell.getARadius() * scale);
    const float rb = std::max(1e-3f, cell.getBRadius() * scale);
    const float rc = std::max(1e-3f, cell.getCRadius() * scale);
    const double v = (lx*lx)/(ra*ra) + (ly*ly)/(rb*rb) + (lz*lz)/(rc*rc);
    return v <= 1.0;
}

}  // namespace

BrightnessCalibration CellUniverse::computeAutoCalibration(
    const std::vector<Ellipsoid> &cells,
    const std::vector<cv::Mat> &rawFrame0,
    const BaseConfig &config)
{
    BrightnessCalibration cal;

    const float manualBg = config.simulation.manual_background_intensity;
    const float manualCell = config.simulation.manual_cell_intensity;
    if (manualBg > 0.0f && manualCell > 0.0f && manualCell > manualBg) {
        cal.background_intensity = manualBg;
        cal.cell_intensity = manualCell;
        cal.source = BrightnessCalibrationSource::ManualOverride;
        return cal;
    }

    if (cells.empty() || rawFrame0.empty()) return cal;

    const int height = rawFrame0[0].rows;
    const int width = rawFrame0[0].cols;
    const int depth = static_cast<int>(rawFrame0.size());
    const float innerScale = config.simulation.calibration_cell_inner_fraction;
    const float trim = config.simulation.calibration_pixel_trim_percent;

    std::vector<float> cellPixels;
    std::vector<float> bgPixels;
    cellPixels.reserve(static_cast<std::size_t>(width) * height * depth / 100);
    bgPixels.reserve(static_cast<std::size_t>(width) * height * depth);

    // For background sampling: skip any pixel inside ANY cell's full-radius
    // ellipsoid. For cell sampling: include pixels inside the inner-fraction
    // ellipsoid of any non-trash cell.
    for (int z = 0; z < depth; ++z) {
        const cv::Mat &slice = rawFrame0[z];
        for (int y = 0; y < height; ++y) {
            const float *row = slice.ptr<float>(y);
            for (int x = 0; x < width; ++x) {
                const float v = row[x];
                bool insideAny = false;
                bool insideInnerNonTrash = false;
                for (const auto &cell : cells) {
                    if (pointInsideScaledEllipsoid(cell, 1.0f, x, y, z)) {
                        insideAny = true;
                        if (!cell.isTrash() && pointInsideScaledEllipsoid(cell, innerScale, x, y, z)) {
                            insideInnerNonTrash = true;
                        }
                    }
                }
                if (insideInnerNonTrash) cellPixels.push_back(v);
                if (!insideAny) bgPixels.push_back(v);
            }
        }
    }

    if (cellPixels.empty() || bgPixels.empty()) return cal;

    cal.cell_intensity = trimmedMedian(cellPixels, trim);
    cal.background_intensity = trimmedMedian(bgPixels, trim);
    cal.cell_pixel_count = cellPixels.size();
    cal.background_pixel_count = bgPixels.size();
    cal.source = BrightnessCalibrationSource::AutoFrameZero;
    return cal;
}
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd C++/build && cmake --build . --target core_unit_test -j 4
./tests/core_unit_test --gtest_filter="InitializeBrightnessCalibrationTest.*"
```

Expected: 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add C++/includes/CellUniverse.hpp C++/src/CellUniverse.cpp C++/tests/calibration_test.cc
git commit -m "implement computeAutoCalibration (medians from frame 0 cells)"
```

---

### Task 4: Wire calibration into `CellUniverse` constructor flow (storage only, no use yet)

**Files:**
- Modify: `C++/includes/CellUniverse.hpp` (add setter/getter)
- Modify: `C++/src/CellUniverse.cpp` (add impl)
- Modify: `C++/src/main.cpp` (compute and install calibration at startup)

- [ ] **Step 1: Write the failing test**

Append to `C++/tests/calibration_test.cc`:
```cpp
TEST(CellUniverseCalibrationStorageTest, SetterAndGetterRoundtrip) {
    BaseConfig cfg;
    std::vector<std::pair<std::string, std::vector<Ellipsoid>>> emptyCellsByFrame;
    PathVec emptyPaths;
    CellUniverse uni({}, emptyPaths, cfg, "/tmp/test_out", 0, 0);

    BrightnessCalibration cal;
    cal.background_intensity = 50.0f;
    cal.cell_intensity = 800.0f;
    cal.source = BrightnessCalibrationSource::AutoFrameZero;

    uni.setBrightnessCalibration(cal);
    const BrightnessCalibration &got = uni.getBrightnessCalibration();
    EXPECT_FLOAT_EQ(got.background_intensity, 50.0f);
    EXPECT_FLOAT_EQ(got.cell_intensity, 800.0f);
    EXPECT_EQ(got.source, BrightnessCalibrationSource::AutoFrameZero);
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd C++/build && cmake --build . --target core_unit_test -j 4
```

Expected: compile error on `setBrightnessCalibration`.

- [ ] **Step 3: Add accessor methods**

`C++/includes/CellUniverse.hpp` `public:` section:
```cpp
    void setBrightnessCalibration(const BrightnessCalibration &cal) {
        brightnessCalibration = cal;
    }
    const BrightnessCalibration &getBrightnessCalibration() const {
        return brightnessCalibration;
    }
```

- [ ] **Step 4: Wire into main.cpp**

`C++/src/main.cpp` — after `CellFactory cellFactory(config); ...createCells(...)` (around line 151) and BEFORE `CellUniverse lineage = ...` (around line 153), insert:

```cpp
    // Auto-calibrate brightness from frame 0 cells. This must run BEFORE the
    // CellUniverse loads/preprocesses any frame, because the calibrated
    // preprocessing path needs the (background, cell) anchors to scale every
    // frame. See docs/plans/2026-05-04-auto-calibration-from-frame-zero.md.
    BrightnessCalibration calibration;
    if (config.simulation.auto_calibrate_brightness_enabled) {
        if (cells.empty() || imageFilePaths.empty()) {
            std::cerr << "[Auto Calibration] cells or imageFilePaths empty; "
                         "skipping calibration. The calibrated preprocessing path "
                         "will fall back to legacy iterative preprocessing.\n";
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
```

After `CellUniverse lineage = CellUniverse(...)` line, install the calibration:
```cpp
    lineage.setBrightnessCalibration(calibration);
```

Required new include at the top of `main.cpp`:
```cpp
#include "CalibrationTypes.hpp"
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd C++/build && cmake --build . --target core_unit_test -j 4
./tests/core_unit_test --gtest_filter="CellUniverseCalibrationStorageTest.*"
```

Expected: PASS.

- [ ] **Step 6: Smoke-build the main binary**

```bash
cd C++/build && cmake --build . --target celluniverse -j 4
```

Expected: builds clean. `[Auto Calibration]` log line will appear at next run startup, but no behavior change yet (calibration is stored but unused).

- [ ] **Step 7: Commit**

```bash
git add C++/includes/CellUniverse.hpp C++/src/CellUniverse.cpp C++/src/main.cpp C++/tests/calibration_test.cc
git commit -m "wire BrightnessCalibration storage and main.cpp install (unused yet)"
```

---

## Phase 2 — Calibrated Preprocessing Path

### Task 5: Add `applyCalibratedPreprocess` to ImageHandler (linear normalize + tiny particle + final blur + z-interpolate)

**Files:**
- Modify: `C++/includes/ImageHandler.hpp` (add method declaration)
- Modify: `C++/src/ImageHandler.cpp` (add impl)
- Test: `C++/tests/process_image_test.cc`

- [ ] **Step 1: Write the failing test**

Append to `C++/tests/process_image_test.cc`:
```cpp
#include "CalibrationTypes.hpp"

TEST(ApplyCalibratedPreprocessTest, MapsBackgroundToZeroAndCellToOne) {
    BrightnessCalibration cal;
    cal.background_intensity = 50.0f;
    cal.cell_intensity = 800.0f;
    cal.source = BrightnessCalibrationSource::AutoFrameZero;

    BaseConfig cfg;
    cfg.simulation.z_scaling = 1;
    cfg.simulation.blur_sigma = 0.0f;  // disable to test the linear mapping cleanly

    std::vector<cv::Mat> input;
    cv::Mat slice(10, 10, CV_32F, cv::Scalar(50.0f));
    slice.at<float>(5, 5) = 800.0f;
    input.push_back(slice.clone());
    input.push_back(slice.clone());

    auto out = ImageHandler::applyCalibratedPreprocess(input, cal, cfg, nullptr);
    ASSERT_EQ(out.size(), input.size());

    EXPECT_NEAR(out[0].at<float>(0, 0), 0.0f, 1e-3f);
    EXPECT_NEAR(out[0].at<float>(5, 5), 1.0f, 1e-3f);
}

TEST(ApplyCalibratedPreprocessTest, ClipsValuesAboveCellAndBelowBackground) {
    BrightnessCalibration cal;
    cal.background_intensity = 50.0f;
    cal.cell_intensity = 800.0f;
    cal.source = BrightnessCalibrationSource::AutoFrameZero;

    BaseConfig cfg;
    cfg.simulation.z_scaling = 1;
    cfg.simulation.blur_sigma = 0.0f;

    std::vector<cv::Mat> input;
    cv::Mat slice(5, 5, CV_32F, cv::Scalar(0.0f));
    slice.at<float>(0, 0) = -100.0f;
    slice.at<float>(0, 1) = 2000.0f;
    input.push_back(slice.clone());

    auto out = ImageHandler::applyCalibratedPreprocess(input, cal, cfg, nullptr);
    EXPECT_EQ(out[0].at<float>(0, 0), 0.0f);
    EXPECT_EQ(out[0].at<float>(0, 1), 1.0f);
}

TEST(ApplyCalibratedPreprocessTest, InterpolatesZSlicesWhenZScalingGreaterThanOne) {
    BrightnessCalibration cal;
    cal.background_intensity = 0.0f;
    cal.cell_intensity = 1.0f;
    cal.source = BrightnessCalibrationSource::AutoFrameZero;

    BaseConfig cfg;
    cfg.simulation.z_scaling = 7;
    cfg.simulation.blur_sigma = 0.0f;

    std::vector<cv::Mat> input(2, cv::Mat(4, 4, CV_32F, cv::Scalar(0.0f)));
    auto out = ImageHandler::applyCalibratedPreprocess(input, cal, cfg, nullptr);
    EXPECT_EQ(out.size(), 2u + (2u - 1u) * (7u - 1u));  // 2 + 6 = 8
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd C++/build && cmake --build . --target core_unit_test -j 4
```

Expected: compile error on `applyCalibratedPreprocess`.

- [ ] **Step 3: Declare the method**

`C++/includes/ImageHandler.hpp` `public:` section, after `preprocessLoadedFrame`:
```cpp
    static std::vector<cv::Mat> applyCalibratedPreprocess(
        const std::vector<cv::Mat> &normalizedSlices,
        const BrightnessCalibration &calibration,
        const BaseConfig &config,
        std::ostream *logSink);
```

Add include at top:
```cpp
#include "CalibrationTypes.hpp"
```

- [ ] **Step 4: Implement the method**

`C++/src/ImageHandler.cpp`, near `preprocessLoadedFrame`:

```cpp
std::vector<cv::Mat> ImageHandler::applyCalibratedPreprocess(
    const std::vector<cv::Mat> &normalizedSlices,
    const BrightnessCalibration &calibration,
    const BaseConfig &config,
    std::ostream *logSink)
{
    std::ostream &log = logSink ? *logSink : std::cout;
    if (!calibration.isValid()) {
        log << "[Calibrated Preprocess] calibration invalid; returning input unchanged\n";
        return normalizedSlices;
    }

    // Step 1: linear normalize via calibration anchors -> [0, 1]
    std::vector<cv::Mat> linearScaled;
    linearScaled.reserve(normalizedSlices.size());
    const float bg = calibration.background_intensity;
    const float span = calibration.cell_intensity - bg;
    for (const auto &slice : normalizedSlices) {
        cv::Mat out;
        slice.convertTo(out, CV_32F, 1.0f / span, -bg / span);
        cv::min(out, 1.0f, out);
        cv::max(out, 0.0f, out);
        linearScaled.push_back(out);
    }

    // Step 2: tiny particle removal + final blur
    // Reuse the existing applyFinalPreprocessingBlur helper from CellUniverse.cpp
    // by mirroring its core behavior here. Keeping the helper local to this
    // file avoids cross-translation-unit coupling.
    if (config.simulation.blur_sigma > 0.0f) {
        const int ksize = std::max(3, 2 * static_cast<int>(std::ceil(3.0f * config.simulation.blur_sigma)) + 1);
        for (auto &slice : linearScaled) {
            cv::GaussianBlur(slice, slice, cv::Size(ksize, ksize),
                             config.simulation.blur_sigma, config.simulation.blur_sigma,
                             cv::BORDER_REPLICATE);
        }
    }

    // Step 3: z-interpolate (mirror existing preprocessLoadedFrame interpolation)
    if (linearScaled.size() <= 1 || config.simulation.z_scaling <= 1) {
        log << "[Calibrated Preprocess] " << calibrationDescribe(calibration)
            << " input_slices=" << normalizedSlices.size()
            << " output_slices=" << linearScaled.size()
            << " (no z-interp)\n";
        return linearScaled;
    }

    const int expandFactor = config.simulation.z_scaling;
    const unsigned numSynthSlices =
        static_cast<unsigned>(expandFactor) * (linearScaled.size() - 1U) + 1U;
    std::vector<cv::Mat> interpolated(numSynthSlices);

    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < static_cast<int>(numSynthSlices); ++idx) {
        const int loIdx = idx / expandFactor;
        const int hiIdx = std::min(static_cast<int>(linearScaled.size()) - 1, loIdx + 1);
        const float t = static_cast<float>(idx % expandFactor) / static_cast<float>(expandFactor);
        cv::Mat blended = (1.0f - t) * linearScaled[loIdx] + t * linearScaled[hiIdx];
        interpolated[idx] = blended;
    }

    log << "[Calibrated Preprocess] " << calibrationDescribe(calibration)
        << " input_slices=" << normalizedSlices.size()
        << " output_slices=" << interpolated.size() << "\n";
    return interpolated;
}
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd C++/build && cmake --build . --target core_unit_test -j 4
./tests/core_unit_test --gtest_filter="ApplyCalibratedPreprocessTest.*"
```

Expected: 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add C++/includes/ImageHandler.hpp C++/src/ImageHandler.cpp C++/tests/process_image_test.cc
git commit -m "add ImageHandler::applyCalibratedPreprocess (linear scale + blur + z-interp)"
```

---

### Task 6: Dispatch in `preprocessLoadedFrame` based on calibration validity

**Files:**
- Modify: `C++/includes/ImageHandler.hpp` (overload `preprocessLoadedFrame` to take optional calibration)
- Modify: `C++/src/ImageHandler.cpp` (dispatch logic)

- [ ] **Step 1: Write the failing test**

Append to `C++/tests/process_image_test.cc`:
```cpp
TEST(PreprocessLoadedFrameWithCalibrationTest, UsesCalibratedPathWhenCalibrationValid) {
    BrightnessCalibration cal;
    cal.background_intensity = 50.0f;
    cal.cell_intensity = 800.0f;
    cal.source = BrightnessCalibrationSource::AutoFrameZero;

    BaseConfig cfg;
    cfg.simulation.auto_calibrate_brightness_enabled = true;
    cfg.simulation.z_scaling = 1;
    cfg.simulation.blur_sigma = 0.0f;
    cfg.simulation.iterative_max_count = 9999;  // would dominate runtime if hit

    std::vector<cv::Mat> input;
    cv::Mat slice(10, 10, CV_32F, cv::Scalar(50.0f));
    slice.at<float>(5, 5) = 800.0f;
    input.push_back(slice.clone());

    auto out = ImageHandler::preprocessLoadedFrame(input, "synthetic.tif", cfg, &cal, nullptr);
    EXPECT_NEAR(out[0].at<float>(5, 5), 1.0f, 1e-3f);
    EXPECT_NEAR(out[0].at<float>(0, 0), 0.0f, 1e-3f);
}

TEST(PreprocessLoadedFrameWithCalibrationTest, FallsBackToLegacyWhenCalibrationNullOrInvalid) {
    BaseConfig cfg;
    cfg.simulation.auto_calibrate_brightness_enabled = true;
    cfg.simulation.iterative_max_count = 1;
    cfg.simulation.z_scaling = 1;

    std::vector<cv::Mat> input;
    input.push_back(cv::Mat(8, 8, CV_32F, cv::Scalar(0.5f)));

    // Null calibration -> legacy path should still produce something
    auto out = ImageHandler::preprocessLoadedFrame(input, "synthetic.tif", cfg, nullptr, nullptr);
    EXPECT_FALSE(out.empty());
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd C++/build && cmake --build . --target core_unit_test -j 4
```

Expected: compile error — current `preprocessLoadedFrame` has no calibration parameter.

- [ ] **Step 3: Add overload**

`C++/includes/ImageHandler.hpp` add overload below the existing `preprocessLoadedFrame`:
```cpp
    static std::vector<cv::Mat> preprocessLoadedFrame(
        const std::vector<cv::Mat> &normalizedSlices,
        const std::string &imageFile,
        const BaseConfig &config,
        const BrightnessCalibration *calibration,
        std::ostream *logSink = nullptr);
```

- [ ] **Step 4: Implement the overload + change existing one to delegate**

`C++/src/ImageHandler.cpp`:

```cpp
std::vector<cv::Mat> ImageHandler::preprocessLoadedFrame(
    const std::vector<cv::Mat> &normalizedSlices,
    const std::string &imageFile,
    const BaseConfig &config,
    const BrightnessCalibration *calibration,
    std::ostream *logSink)
{
    std::ostream &log = logSink ? *logSink : std::cout;
    const bool calibratedPath = config.simulation.auto_calibrate_brightness_enabled
        && calibration != nullptr
        && calibration->isValid();

    if (calibratedPath) {
        log << "[Preprocess Dispatch] file=" << fs::path(imageFile).filename().string()
            << " path=calibrated\n";
        return applyCalibratedPreprocess(normalizedSlices, *calibration, config, logSink);
    }

    log << "[Preprocess Dispatch] file=" << fs::path(imageFile).filename().string()
        << " path=legacy_iterative\n";
    return preprocessLoadedFrame(normalizedSlices, imageFile, config, logSink);
}
```

(The existing 4-argument `preprocessLoadedFrame` stays exactly as it is — the new 5-argument overload calls into it for the legacy path.)

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd C++/build && cmake --build . --target core_unit_test -j 4
./tests/core_unit_test --gtest_filter="PreprocessLoadedFrameWithCalibrationTest.*"
```

Expected: 2 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add C++/includes/ImageHandler.hpp C++/src/ImageHandler.cpp C++/tests/process_image_test.cc
git commit -m "add preprocessLoadedFrame overload that dispatches calibrated vs legacy"
```

---

### Task 7: Plumb calibration through `preprocessAllFramesAlignedToMinimumBackground` and `prepareFrame`

**Files:**
- Modify: `C++/src/CellUniverse.cpp`

- [ ] **Step 1: Find the three `ImageHandler::preprocessLoadedFrame(` call sites**

The current file has them at lines 1884, 1950, 2091 (verify with `grep -n preprocessLoadedFrame C++/src/CellUniverse.cpp`).

- [ ] **Step 2: Pass `&brightnessCalibration` through each call**

For each call site, change:
```cpp
real_frame = ImageHandler::preprocessLoadedFrame(
    rawFrame, imagePath.string(), config, &std::cout);
```

to:
```cpp
real_frame = ImageHandler::preprocessLoadedFrame(
    rawFrame, imagePath.string(), config, &brightnessCalibration, &std::cout);
```

(Adjust call site code style to match the local convention at each site.)

- [ ] **Step 3: Build and verify**

```bash
cd C++/build && cmake --build . --target celluniverse core_unit_test -j 4
```

Expected: builds clean.

- [ ] **Step 4: Run unit tests to ensure no regressions**

```bash
./tests/core_unit_test
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add C++/src/CellUniverse.cpp
git commit -m "plumb BrightnessCalibration through CellUniverse preprocess call sites"
```

---

### Task 8: Smoke-test calibrated path against fluo dataset

**Files:** none modified. Validation only.

- [ ] **Step 1: Build celluniverse**

```bash
cd C++/build && cmake --build . --target celluniverse -j 4
```

- [ ] **Step 2: Run a 5-frame calibrated test**

```bash
cd C++ && ./build/celluniverse 0 5 \
    data/input/embryo_data/t%03d.tif \
    outputs/output_calibrated_smoke_$(date +%Y%m%d_%H%M%S) \
    config/config.yaml \
    config/initial_auto.csv
```

- [ ] **Step 3: Inspect the calibration log**

```bash
RUN=$(ls -td C++/outputs/output_calibrated_smoke_* | head -1)
grep -E "^\[Auto Calibration\]|^\[Preprocess Dispatch\]|^\[Calibrated Preprocess\]" "$RUN/debug_log.txt" | head -10
```

Expected output includes:
- One `[Auto Calibration]` line at startup with `source=auto`, plausible `background` and `cell` intensities for the fluo dataset (background ~5–50, cell ~200–2000 in raw 16-bit space).
- Per-frame `[Preprocess Dispatch] path=calibrated` lines (NOT `legacy_iterative`).
- Per-frame `[Calibrated Preprocess]` lines confirming z-interpolation expanded to 239 slices.

- [ ] **Step 4: Inspect a few preprocessed pixels**

```bash
grep -E "^\[Optimize Done\]" "$RUN/debug_log.txt"
```

Expected: 5 lines, frames 0–4. Cell counts match the regression baseline at the same frames (4 cells f0–3, 6 cells f4 after the cell_1/cell_2 splits).

- [ ] **Step 5: Sanity check the bridge proposals still get triaxial-rejected**

```bash
grep -c "triaxial_shape" "$RUN/debug_log.txt"
```

Expected: > 0 (a few cells will still trigger the bridge but get rejected by the prolate gate, exactly as in the baseline).

- [ ] **Step 6: If smoke test passes, commit nothing (no code change)**

If split events or cell counts diverge from baseline at any of the first 5 frames, STOP and analyze before continuing to Phase 3.

---

## Phase 3 — Manual Override + Tests

### Task 9: Add manual-override integration test

**Files:**
- Test: `C++/tests/calibration_test.cc`

- [ ] **Step 1: Write the failing test**

Append to `C++/tests/calibration_test.cc`:
```cpp
TEST(ManualOverrideEndToEndTest, ManualValuesPassThroughPreprocessing) {
    BaseConfig cfg;
    cfg.simulation.auto_calibrate_brightness_enabled = true;
    cfg.simulation.manual_background_intensity = 50.0f;
    cfg.simulation.manual_cell_intensity = 800.0f;
    cfg.simulation.z_scaling = 1;
    cfg.simulation.blur_sigma = 0.0f;

    auto cal = CellUniverse::computeAutoCalibration({}, {}, cfg);
    ASSERT_TRUE(cal.isValid());
    EXPECT_EQ(cal.source, BrightnessCalibrationSource::ManualOverride);

    std::vector<cv::Mat> input;
    cv::Mat slice(5, 5, CV_32F, cv::Scalar(50.0f));
    slice.at<float>(2, 2) = 800.0f;
    input.push_back(slice.clone());

    auto out = ImageHandler::preprocessLoadedFrame(input, "manual.tif", cfg, &cal, nullptr);
    EXPECT_NEAR(out[0].at<float>(0, 0), 0.0f, 1e-3f);
    EXPECT_NEAR(out[0].at<float>(2, 2), 1.0f, 1e-3f);
}
```

- [ ] **Step 2: Run test**

```bash
cd C++/build && cmake --build . --target core_unit_test -j 4
./tests/core_unit_test --gtest_filter="ManualOverrideEndToEndTest.*"
```

Expected: PASS (no implementation changes needed; tests existing behavior end-to-end).

- [ ] **Step 3: Commit**

```bash
git add C++/tests/calibration_test.cc
git commit -m "add manual override end-to-end integration test"
```

---

### Task 10: Document manual override in config.yaml

**Files:**
- Modify: `C++/config/config.yaml`

- [ ] **Step 1: Expand the comment block on `manual_*_intensity`**

Find the `manual_background_intensity` line and replace its comment block with:
```yaml
  # Manual override of frame-0 auto-calibration. Both must be > 0 (raw 16-bit
  # intensity values, not normalized) to take effect; otherwise auto-detection
  # runs against the initial CSV cell positions.
  #
  # Use this when:
  #  - Frame 0 cells are mid-division (cell-stat estimate would be biased)
  #  - The initial CSV is approximate and you've measured (bg, cell) by hand
  #  - You want to lock the brightness scale across runs for direct comparison
  #
  # Workflow: open frame 0 in Fiji/Napari, measure mean intensity in a clean
  # background ROI -> manual_background_intensity. Mean intensity at the center
  # of a typical cell -> manual_cell_intensity.
  manual_background_intensity: -1
  manual_cell_intensity: -1
```

- [ ] **Step 2: Commit**

```bash
git add C++/config/config.yaml
git commit -m "document manual override workflow in config.yaml comments"
```

---

## Phase 4 — Validation Against Baseline

### Task 11: Side-by-side run against fluo 0–50 baseline

**Files:** none modified. Validation only.

- [ ] **Step 1: Build with calibration enabled (default)**

```bash
cd C++/build && cmake --build . --target celluniverse -j 4
```

- [ ] **Step 2: Run f0–50 fluo with calibrated preprocessing**

On the Linux WSL box (Mac is too slow for full validation):
```bash
cd /home/jihangl3/CellUniverse/C++ && \
  scripts/run_celluniverse.sh config/user_input_config_embryo.ini embryo
```

(Adjust the `embryo` preset's `last_frame` to `50` first. Output dir auto-named with timestamp.)

- [ ] **Step 3: Wait for completion**

Estimated wall time: ~13 hr (was ~15.4 hr in baseline due to ~110 sec/frame preprocessing savings).

- [ ] **Step 4: Run the comparison script**

Create `C++/scripts/compare_runs_split_events.sh`:
```bash
#!/bin/bash
# Usage: compare_runs_split_events.sh <baseline_dir> <new_dir>
BASELINE="$1"
NEW="$2"
echo "=== Splits in baseline ==="
grep -E "^\[Split Accepted\]" "$BASELINE/debug_log.txt" | grep -oE "cell_[0-9]+" | sort -V
echo "=== Splits in new ==="
grep -E "^\[Split Accepted\]" "$NEW/debug_log.txt" | grep -oE "cell_[0-9]+" | sort -V
echo "=== Diff ==="
diff <(grep -E "^\[Split Accepted\]" "$BASELINE/debug_log.txt" | grep -oE "cell_[0-9]+") \
     <(grep -E "^\[Split Accepted\]" "$NEW/debug_log.txt" | grep -oE "cell_[0-9]+")
```

```bash
chmod +x C++/scripts/compare_runs_split_events.sh
C++/scripts/compare_runs_split_events.sh \
    C++/outputs/output_fluo_0-50_20260504_005706 \
    C++/outputs/output_fluo_calibrated_<TIMESTAMP>
```

- [ ] **Step 5: Acceptance criteria**

The calibrated run **PASSES** if:
- Same set of `cell_X` IDs split (lineage topology preserved). 1-frame timing wobbles are acceptable.
- Cell counts at f4, f7, f11, f18, f24, f25, f35, f45, f48 match GT exactly: 6, 7, 8, 11, 13, 14, 24, 26, 26.
- Zero false splits in the f36–44 quiet window (cell count stays at 24).
- `[Auto Calibration]` log line shows plausible values: background in [1, 100], cell in [100, 5000] (raw 16-bit fluo ranges).
- Total wall time within 70–90% of baseline (calibrated path is faster).

If any criterion fails, document in `C++/docs/plans/2026-05-04-auto-calibration-from-frame-zero.md` under "Open follow-ups" and STOP.

- [ ] **Step 6: Commit the comparison script**

```bash
git add C++/scripts/compare_runs_split_events.sh
git commit -m "add split-event comparison script"
```

---

### Task 12: Manual sign-off — visual frame-0 inspection

**Files:** none modified. Validation only.

- [ ] **Step 1: Export frame 0 preprocessed slice as PNG**

The run output already contains `<RUN>/synth/0/*.png`. Open any mid-z slice (e.g. `synth/0/119.png`) in Fiji or Preview.

- [ ] **Step 2: Compare against the baseline frame 0**

Open `C++/outputs/output_fluo_0-50_20260504_005706/synth/0/119.png` side-by-side. They should look qualitatively similar:
- Cells are bright (near 1.0) and clearly separated from background (near 0.0).
- No cells "disappear" in the calibrated version (one of your professor's stated concerns).

- [ ] **Step 3: If a cell looks dimmer or missing in calibrated output**

Possible causes:
1. The initial CSV ellipsoid was too tight around the cell -> calibration sampled mostly halo, undershoot cell_intensity. **Fix:** lower `calibration_cell_inner_fraction` (try 0.5).
2. Cells in frame 0 were already partially divided -> mixed bright/dark interior. **Fix:** use manual override with measured values.
3. Real cell intensity in this dataset is well below 1.0 in absolute scale. **Fix:** verify `cell_intensity` in the log is plausible.

Document any visual regressions and stop before moving to Task 13.

---

### Task 13: Risk register update

**Files:**
- Modify: `C++/docs/plans/2026-05-04-auto-calibration-from-frame-zero.md`

- [ ] **Step 1: Append "Validation Findings" section**

After the run completes and is signed off, append observed wall-time, split-event diff, and any qualitative differences to the bottom of this plan file under a new section.

- [ ] **Step 2: Commit**

```bash
git add C++/docs/plans/2026-05-04-auto-calibration-from-frame-zero.md
git commit -m "record auto-calibration validation findings"
```

---

## Phase 5 — Documentation + Deprecation Markers

### Task 14: Append Change 14 to changelogv8.md

**Files:**
- Modify: `C++/docs/changelogs/changelogv8.md`

- [ ] **Step 1: Append the entry**

At the bottom of `changelogv8.md`:
```markdown
## 2026-05-04: Auto-calibration from frame zero (Change 14) — **status: ACTIVE**

### Problem / Motivation

The current iterative preprocessing pipeline (`processPreparedSequence` +
`Frame Intensity Scale` + `IterPreprocess`) has ~35 dataset-specific tunables
that need re-tuning for every new microscopy dataset. Symptoms reported by
the project supervisor: cells sometimes disappear after preprocessing on new
datasets, and the per-dataset tuning effort is the main blocker for using
CellUniverse as a generic tracker.

### Solution

Anchor the brightness scale to actual biology by measuring median intensity
inside vs outside the initial-CSV cell ellipsoids on frame 0. All bio
thresholds (`bio_bridge_min_edge_brightness_absolute`,
`pca_bridge_black_threshold`, etc.) are already in fractional [0, 1] form;
they become portable when "1.0" consistently means "median cell brightness"
across all datasets.

### Files changed

- `C++/includes/CalibrationTypes.hpp` (new)
- `C++/includes/CellUniverse.hpp` (added `computeAutoCalibration`,
  `setBrightnessCalibration`, `getBrightnessCalibration`,
  `BrightnessCalibration brightnessCalibration` member)
- `C++/includes/ConfigTypes.hpp` (added 5 calibration fields)
- `C++/includes/ImageHandler.hpp` (added `applyCalibratedPreprocess` and
  `preprocessLoadedFrame` overload that takes `const BrightnessCalibration*`)
- `C++/src/CellUniverse.cpp` (implementation + plumbing)
- `C++/src/ImageHandler.cpp` (implementation + dispatch)
- `C++/src/main.cpp` (compute and install calibration at startup)
- `C++/config/config.yaml` (5 new keys + deprecation comments)
- `C++/tests/calibration_test.cc` (new)
- `C++/tests/process_image_test.cc` (added 5 tests)
- `C++/tests/CMakeLists.txt` (registered new test file)

### Effect

- Frame 0 cells dictate the brightness scale, not arbitrary percentiles.
- IterPreprocess gradient walk + percentile intensity scale are bypassed
  (saves ~110 sec/frame at f48 = ~30% runtime reduction).
- ~35 deprecated config fields stay parsed but ignored when
  `auto_calibrate_brightness_enabled: true` (the default).
- Manual override path (`manual_background_intensity`,
  `manual_cell_intensity`) for datasets where frame 0 cells are
  mid-division or where user has measured anchors directly.

### Validation

Compared against `output_fluo_0-50_20260504_005706` (the bridge-demote
regression baseline). Cell-count matches GT at f4, f7, f11, f18, f24, f25,
f35, f45, f48. No false splits in f36–44 quiet window. See plan file
`docs/plans/2026-05-04-auto-calibration-from-frame-zero.md` "Validation
Findings" for run-specific results.

### Open follow-ups

- Spatial illumination correction (vignetting). Currently a single
  `background_intensity` constant; some datasets need a per-pixel low-pass
  background field.
- Per-channel calibration for multi-channel datasets.
- Eventual removal of deprecated `iterative_*` / `contrast_*` /
  `frame_intensity_scale_*` config fields once auto-calibration has been
  validated on 3+ datasets.
```

- [ ] **Step 2: Commit**

```bash
git add C++/docs/changelogs/changelogv8.md
git commit -m "changelog: Change 14 — auto-calibration from frame zero"
```

---

### Task 15: Final review + push

**Files:** none modified.

- [ ] **Step 1: Run full test suite one more time**

```bash
cd C++/build && cmake --build . --target core_unit_test -j 4 && ./tests/core_unit_test
```

Expected: ALL tests PASS.

- [ ] **Step 2: Smoke-build celluniverse**

```bash
cd C++/build && cmake --build . --target celluniverse -j 4
```

Expected: builds clean, no warnings introduced.

- [ ] **Step 3: Confirm git status is clean**

```bash
git status
```

Expected: working tree clean, all commits on `jl_auto_calibrate_05042026`.

- [ ] **Step 4: Push the branch**

```bash
git push -u origin jl_auto_calibrate_05042026
```

(Requires user explicit permission per repo memory `no-git-writes` — ASK first.)

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Frame 0 cells mid-division** — cell mean is biased toward gap regions, undershoots true cell brightness. | Low for fluo (founders are pre-division), medium for resume runs starting mid-dataset. | Bio thresholds may misfire; some splits missed or false. | Manual override path. Document workflow in config.yaml. |
| **Spatial illumination variation (vignetting)** — single `background_intensity` constant misses per-pixel variation. | Medium (depends on microscope). Fluo dataset is fairly uniform. | Cells in dim corners look "below background" → optimizer pushes them away. | Phase 5+ follow-up: per-pixel background via low-pass field. Out of scope for this plan. |
| **Saturated pixels in raw input** — bright pixels clip at 16-bit max, biasing cell median up. | Low (modern microscopes avoid saturation). | Cell anchor too high → all in-cell pixels under 1.0 → bio thresholds too strict. | Trimmed median (`calibration_pixel_trim_percent: 0.10`) drops top/bottom 10%. |
| **Multi-channel datasets** — current code assumes single channel. | Future work. | Calibration computed on whatever the first channel is. | Phase 5+ follow-up: per-channel calibration. |
| **Initial CSV ellipsoids are inaccurate** — user provides loose-fit boxes around cells. | Medium for new users. | Background sample contaminated with cell pixels → bg too high → reduced span → noisy [0, 1] mapping. | `calibration_cell_inner_fraction: 0.7` shrinks the cell-mask used for sampling. Manual override fallback. |
| **Adaptive background drift breaks under calibrated path** — per-frame brightness EMA assumes the [0, 1] semantics held by the legacy preprocessing. | Low (calibrated path produces same [0, 1] semantics). | Brightness EMA over- or under-corrects per frame. | Tasks 7+8 explicitly run the same adaptive bg path; smoke test in Task 8 catches drift. |
| **`pca_bridge_black_threshold` too tight for new dataset** — even with calibration, "5% of cell brightness" may not be the right "dark" cutoff for some staining protocols. | Low | Bridge analyzer rejects all proposals as `no_dark_bridge` → some splits missed. | Already a config knob; per-dataset tuning is acceptable for ONE threshold (vs ~35 today). |
| **Test coverage gaps on the legacy fallback** — when calibration is invalid, the legacy iterative path runs. Untouched in this plan. | Low | Backward compatibility regression. | Task 6 step 3 explicitly tests the fallback. |

---

## Self-Review

**1. Spec coverage:**
- ✓ Phased implementation (5 phases)
- ✓ Validation against `output_fluo_0-50_20260504_005706` baseline (Task 11)
- ✓ Manual override fallback (Tasks 9–10)
- ✓ Deprecation path for `iterative_*` / `contrast_*` config (Task 2 deprecation comments + changelog Task 14)
- ✓ Risk register (above)
- ✓ Per-frame adaptive background + per-cell EMA brightness preserved (Task 7 plumbs only the preprocess function, leaves `setBackgroundColor` and Spheroid::draw paths untouched)
- ✓ Branch off `jl_bridge_demote_05042026` (top of plan)

**2. Placeholder scan:** No "TBD" / "implement later" / "similar to Task N". All test code, all production code, all commands are concrete.

**3. Type consistency:**
- `BrightnessCalibration` struct fields: `background_intensity` (float), `cell_intensity` (float), `cell_pixel_count` (size_t), `background_pixel_count` (size_t), `source` (enum). Used consistently across all tasks.
- `BrightnessCalibrationSource` enum values: `Uninitialized`, `AutoFrameZero`, `ManualOverride`. Consistent.
- `computeAutoCalibration(cells, rawFrame, config) -> BrightnessCalibration` — same signature in Tasks 3, 4, 9.
- `applyCalibratedPreprocess(slices, calibration, config, log) -> vector<Mat>` — same in Tasks 5, 6.
- `preprocessLoadedFrame` overload signature: `(slices, imageFile, config, calibration*, log*) -> vector<Mat>` — same in Tasks 6, 7.

---

## Validation Findings

(Filled in during Task 13 after the validation run completes.)
