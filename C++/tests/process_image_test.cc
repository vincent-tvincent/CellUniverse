#include <gtest/gtest.h>

#include "CalibrationTypes.hpp"
#include "ImageHandler.hpp"

namespace {
BaseConfig MakeConfig() {
    BaseConfig cfg;
    cfg.simulation.z_scaling = 2.0f;
    return cfg;
}
}

TEST(ProcessImageTest, ConvertsRgbInputToSingleChannelFloatImage) {
    BaseConfig cfg = MakeConfig();
    cv::Mat rgb(5, 7, CV_8UC3, cv::Scalar(10, 20, 30));

    cv::Mat out = ImageHandler::processImage(rgb, cfg);

    EXPECT_EQ(out.rows, 5);
    EXPECT_EQ(out.cols, 7);
    EXPECT_EQ(out.channels(), 1);
    EXPECT_EQ(out.type(), CV_32F);
}

TEST(ProcessImageTest, OutputPreservesSourceIntensityScale) {
    // processImage no longer normalizes to [0,1] — that step was moved to a
    // later run-wide percentile scale. processImage now only converts dtype to
    // CV_32F (and optionally blurs). So an 8-bit input with values in [0, 255]
    // should round-trip into a float image with values in [0, 255].
    BaseConfig cfg = MakeConfig();
    cv::Mat gray = cv::Mat::zeros(7, 7, CV_8U);
    gray.at<unsigned char>(3, 3) = 255;

    cv::Mat out = ImageHandler::processImage(gray, cfg);

    double minVal = 0.0;
    double maxVal = 0.0;
    cv::minMaxLoc(out, &minVal, &maxVal);

    EXPECT_EQ(out.type(), CV_32F);
    EXPECT_GE(minVal, 0.0);
    EXPECT_LE(maxVal, 255.0);
    EXPECT_NEAR(maxVal, 255.0, 1e-3);
}

TEST(LoadFrameTest, MissingNonTiffFileReturnsEmptyVector) {
    BaseConfig cfg = MakeConfig();

    std::vector<cv::Mat> out = ImageHandler::loadFrame("/path/that/does/not/exist.png", cfg);

    EXPECT_TRUE(out.empty());
}

TEST(ApplyCalibratedPreprocessTest, MapsBackgroundToZeroAndCellToOne) {
    BrightnessCalibration cal;
    cal.background_intensity = 50.0f;
    cal.cell_intensity = 800.0f;
    cal.source = BrightnessCalibrationSource::AutoFrameZero;

    BaseConfig cfg;
    cfg.simulation.z_scaling = 1;
    cfg.simulation.blur_sigma = 0.0f;
    cfg.simulation.calibrated_preprocess_blur_sigma = 0.0f;  // disable to test the linear mapping cleanly

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
    cfg.simulation.calibrated_preprocess_blur_sigma = 0.0f;

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
    cfg.simulation.calibrated_preprocess_blur_sigma = 0.0f;

    std::vector<cv::Mat> input;
    input.push_back(cv::Mat(4, 4, CV_32F, cv::Scalar(0.0f)));
    input.push_back(cv::Mat(4, 4, CV_32F, cv::Scalar(0.0f)));
    auto out = ImageHandler::applyCalibratedPreprocess(input, cal, cfg, nullptr);
    EXPECT_EQ(out.size(), 2u + (2u - 1u) * (7u - 1u));  // 2 + 6 = 8
}

TEST(PreprocessLoadedFrameWithCalibrationTest, UsesCalibratedPathWhenCalibrationValid) {
    BrightnessCalibration cal;
    cal.background_intensity = 50.0f;
    cal.cell_intensity = 800.0f;
    cal.source = BrightnessCalibrationSource::AutoFrameZero;

    BaseConfig cfg;
    cfg.simulation.auto_calibrate_brightness_enabled = true;
    cfg.simulation.z_scaling = 1;
    cfg.simulation.blur_sigma = 0.0f;
    cfg.simulation.calibrated_preprocess_blur_sigma = 0.0f;
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

TEST(SimplePreprocessTest, RealRatioChangesWeightedBlend) {
    BaseConfig blurOnly;
    blurOnly.simulation.auto_calibrate_brightness_enabled = false;
    blurOnly.simulation.simple_preprocess_enabled = true;
    blurOnly.simulation.simple_preprocess_blur_sigma = 2.0f;
    blurOnly.simulation.simple_preprocess_real_ratio = 0.0f;
    blurOnly.simulation.simple_preprocess_background_factor = 1.0f;
    blurOnly.simulation.simple_preprocess_interpolate_z = false;
    blurOnly.simulation.z_scaling = 1;
    blurOnly.simulation.cube_pooling_enabled = false;

    BaseConfig rawWeighted = blurOnly;
    rawWeighted.simulation.simple_preprocess_real_ratio = 1.0f;

    std::vector<cv::Mat> input;
    cv::Mat slice(31, 31, CV_32F, cv::Scalar(0.0f));
    slice.at<float>(15, 15) = 1.0f;
    input.push_back(slice);

    auto blurOnlyOut = ImageHandler::preprocessLoadedFrame(
        input, "ratio_zero.tif", blurOnly, nullptr, nullptr);
    auto rawWeightedOut = ImageHandler::preprocessLoadedFrame(
        input, "ratio_one.tif", rawWeighted, nullptr, nullptr);

    ASSERT_EQ(blurOnlyOut.size(), 1u);
    ASSERT_EQ(rawWeightedOut.size(), 1u);
    EXPECT_LT(blurOnlyOut[0].at<float>(15, 15), 0.1f);
    EXPECT_GT(rawWeightedOut[0].at<float>(15, 15), 0.9f);
}

TEST(ApplyCalibratedPreprocessTest, PerFrameRatioRescalesAnchorsForDimmedFrame) {
    BrightnessCalibration cal;
    cal.background_intensity = 50.0f;   // bg_0
    cal.cell_intensity = 800.0f;        // cell_0
    // Original frame_0_mean: dataset is mostly bg=50 with a few cell pixels
    // at 800. For a 100x100 slice with 1 cell pixel out of 10000:
    // mean = (9999*50 + 800)/10000 = 50.075. Round to 50.
    cal.frame_0_mean = 50.0f;
    cal.source = BrightnessCalibrationSource::AutoFrameZero;

    BaseConfig cfg;
    cfg.simulation.z_scaling = 1;
    cfg.simulation.blur_sigma = 0.0f;
    cfg.simulation.calibrated_preprocess_blur_sigma = 0.0f;

    // Simulate frame N after uniform 50% photobleaching: cells at 400 (was
    // 800), bg at 25 (was 50). Frame mean ≈ 25, so ratio ≈ 0.5. Anchors
    // adjust to bg_n = 25, cell_n = 400. The bleached cell pixel at raw=400
    // maps to ~1.0; the bleached bg at raw=25 maps to ~0.0 — proving that
    // dim cells aren't lost to clipping when bleaching shifts them below
    // the static frame-0 anchors.
    std::vector<cv::Mat> input;
    cv::Mat slice(100, 100, CV_32F, cv::Scalar(25.0f));  // background after bleaching
    slice.at<float>(50, 50) = 400.0f;                    // cell after bleaching
    input.push_back(slice.clone());

    auto out = ImageHandler::applyCalibratedPreprocess(input, cal, cfg, nullptr);
    ASSERT_FALSE(out.empty());
    EXPECT_NEAR(out[0].at<float>(50, 50), 1.0f, 2e-2f);  // bleached cell still maps to 1
    EXPECT_NEAR(out[0].at<float>(0, 0), 0.0f, 2e-2f);    // bleached bg still maps to 0
}

TEST(ApplyCalibratedPreprocessTest, PerFrameRatioFallsBackTo1WhenFrame0MeanZero) {
    BrightnessCalibration cal;
    cal.background_intensity = 50.0f;
    cal.cell_intensity = 800.0f;
    cal.frame_0_mean = 0.0f;            // sentinel — preserve original behavior
    cal.source = BrightnessCalibrationSource::AutoFrameZero;

    BaseConfig cfg;
    cfg.simulation.z_scaling = 1;
    cfg.simulation.blur_sigma = 0.0f;
    cfg.simulation.calibrated_preprocess_blur_sigma = 0.0f;

    std::vector<cv::Mat> input;
    cv::Mat slice(10, 10, CV_32F, cv::Scalar(50.0f));
    slice.at<float>(5, 5) = 800.0f;
    input.push_back(slice.clone());

    auto out = ImageHandler::applyCalibratedPreprocess(input, cal, cfg, nullptr);
    ASSERT_FALSE(out.empty());
    EXPECT_NEAR(out[0].at<float>(5, 5), 1.0f, 1e-3f);  // pre-bleach raw=800 maps to 1
    EXPECT_NEAR(out[0].at<float>(0, 0), 0.0f, 1e-3f);  // pre-bleach raw=50 maps to 0
}
