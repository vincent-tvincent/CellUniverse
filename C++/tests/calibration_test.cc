#include <gtest/gtest.h>
#include <sstream>
#include <vector>

#include <opencv2/opencv.hpp>

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

#include "CellUniverse.hpp"
#include "Ellipsoid.hpp"
#include "ImageHandler.hpp"

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
    EllipsoidParams p;
    p.name = "c";
    p.x = 10; p.y = 10; p.z = 2;
    p.aRadius = 5; p.bRadius = 5; p.cRadius = 5;
    p.theta_x = 0; p.theta_y = 0; p.theta_z = 0;
    p.brightness = 1.0f;
    std::vector<Ellipsoid> cells{ Ellipsoid(p) };

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

TEST(CellUniverseCalibrationStorageTest, SetterAndGetterRoundtrip) {
    BaseConfig cfg;
    // Skip the M2 z_slices probe in the constructor — it would dereference
    // imagePaths.front() on our empty PathVec. The probe is gated by
    // !quit_after_preprocessing, so flipping this flag short-circuits it.
    // Calibration storage is independent of preprocessing-only mode.
    cfg.simulation.quit_after_preprocessing = true;
    PathVec emptyPaths;
    std::map<Path, std::vector<Ellipsoid>> emptyCellsByPath;
    CellUniverse uni(emptyCellsByPath, emptyPaths, cfg, "/tmp/test_out", 0, -1);

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

TEST(ManualOverrideEndToEndTest, ManualValuesPassThroughPreprocessing) {
    BaseConfig cfg;
    cfg.simulation.auto_calibrate_brightness_enabled = true;
    cfg.simulation.manual_background_intensity = 50.0f;
    cfg.simulation.manual_cell_intensity = 800.0f;
    cfg.simulation.z_scaling = 1;
    cfg.simulation.blur_sigma = 0.0f;
    cfg.simulation.calibrated_preprocess_blur_sigma = 0.0f;

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

TEST(BrightnessCalibrationTest, AutoCalibrationStoresFrameZeroMean) {
    BaseConfig cfg;
    cfg.simulation.auto_calibrate_brightness_enabled = true;
    cfg.simulation.calibration_cell_inner_fraction = 0.7f;
    cfg.simulation.calibration_pixel_trim_percent = 0.10f;
    cfg.simulation.z_scaling = 1;

    // 100x100x50 frame, mostly background (50.0) with a small cell (radius 10) at value 800.
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

    // Expected: frame mean = (n_cell_voxels * 800 + n_bg_voxels * 50) / total_voxels
    // Cell sphere radius 10 has volume (4/3)*pi*1000 ~= 4188 voxels at 800.
    // Total = 100*100*50 = 500,000 voxels.
    // n_bg = 500000 - 4188 = 495812 at 50.
    // mean = (4188*800 + 495812*50) / 500000 ~= (3,350,400 + 24,790,600) / 500,000 ~= 56.28
    EXPECT_GT(cal.frame_0_mean, 50.0f);
    EXPECT_LT(cal.frame_0_mean, 80.0f);
}
