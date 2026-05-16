#include <gtest/gtest.h>

#include "Frame.hpp"

#include <stdexcept>
#include <vector>

namespace {
SimulationConfig MakeFrameConfig(int zSlices) {
    SimulationConfig cfg;
    cfg.z_slices = zSlices;
    cfg.z_scaling = 1.0f;
    return cfg;
}

void ConfigureFrameTestEllipsoidBounds() {
    Ellipsoid::cellConfig.minARadius = 0.01;
    Ellipsoid::cellConfig.maxARadius = 10.0;
    Ellipsoid::cellConfig.minBRadius = 0.01;
    Ellipsoid::cellConfig.maxBRadius = 10.0;
    Ellipsoid::cellConfig.minCRadius = 0.01;
    Ellipsoid::cellConfig.maxCRadius = 10.0;
    Ellipsoid::cellConfig.minBrightness = 0.0;
    Ellipsoid::cellConfig.maxBrightness = 1.0;
}
}

TEST(FrameTest, CalculateCostReturnsZeroForIdenticalStacks) {
    const SimulationConfig cfg = MakeFrameConfig(1);
    std::vector<cv::Mat> real = {cv::Mat::zeros(2, 2, CV_32F)};
    Frame frame(real, cfg, {}, "", "f0");

    EXPECT_DOUBLE_EQ(frame.calculateCost(real), 0.0);
}

TEST(FrameTest, CalculateCostSumsL2NormAcrossSlices) {
    const SimulationConfig cfg = MakeFrameConfig(2);
    std::vector<cv::Mat> real = {
        cv::Mat::zeros(2, 2, CV_32F),
        cv::Mat::zeros(2, 2, CV_32F)
    };
    Frame frame(real, cfg, {}, "", "f1");

    std::vector<cv::Mat> synth = {
        cv::Mat::ones(2, 2, CV_32F),
        cv::Mat(2, 2, CV_32F, cv::Scalar(2.0f))
    };

    EXPECT_NEAR(frame.calculateCost(synth), 6.0, 1e-9);
}

TEST(FrameTest, CalculateCostThrowsOnMismatchedStackSize) {
    const SimulationConfig cfg = MakeFrameConfig(2);
    std::vector<cv::Mat> real = {
        cv::Mat::zeros(2, 2, CV_32F),
        cv::Mat::zeros(2, 2, CV_32F)
    };
    Frame frame(real, cfg, {}, "", "f2");

    std::vector<cv::Mat> synth = {cv::Mat::zeros(2, 2, CV_32F)};
    EXPECT_THROW(frame.calculateCost(synth), std::runtime_error);
}

TEST(FrameTest, GenerateOutputSynthFrameConvertsFloatTo8Bit) {
    SimulationConfig cfg = MakeFrameConfig(1);
    cfg.export_bit_depth = 8;
    std::vector<cv::Mat> real = {cv::Mat::zeros(3, 3, CV_32F)};
    Frame frame(real, cfg, {}, "", "f3");
    frame.setBackgroundColor(0.5f);
    frame.regenerateSynthFrame();

    std::vector<cv::Mat> out = frame.generateOutputSynthFrame();

    ASSERT_EQ(out.size(), 1U);
    EXPECT_EQ(out[0].type(), CV_8U);
    EXPECT_NEAR(static_cast<double>(out[0].at<unsigned char>(0, 0)), 128.0, 1.0);
}

TEST(FrameTest, GenerateOutputFrameUsesCustomRangeExportScale) {
    SimulationConfig cfg = MakeFrameConfig(1);
    cfg.export_bit_depth = 8;
    std::vector<cv::Mat> real = {cv::Mat(1, 1, CV_32F, cv::Scalar(0.00144958f))};
    Frame frame(real, cfg, {}, "", "range_real_export_scale");

    const std::vector<cv::Mat> defaultOut = frame.generateOutputFrame();
    const std::vector<cv::Mat> rangeScaledOut = frame.generateOutputFrame(65536.0f);

    ASSERT_EQ(defaultOut.size(), 1U);
    ASSERT_EQ(rangeScaledOut.size(), 1U);
    EXPECT_EQ(defaultOut[0].type(), CV_8U);
    EXPECT_EQ(rangeScaledOut[0].type(), CV_8U);
    EXPECT_EQ(defaultOut[0].at<unsigned char>(0, 0), 0);
    EXPECT_GT(rangeScaledOut[0].at<unsigned char>(0, 0), 0);
}

TEST(FrameTest, GenerateOutputSynthFrameUsesCustomRangeExportScale) {
    SimulationConfig cfg = MakeFrameConfig(1);
    cfg.export_bit_depth = 8;
    std::vector<cv::Mat> real = {cv::Mat::zeros(3, 3, CV_32F)};
    Frame frame(real, cfg, {}, "", "range_synth_export_scale");
    frame.setBackgroundColor(0.00144958f);
    frame.regenerateSynthFrame();

    const std::vector<cv::Mat> defaultOut = frame.generateOutputSynthFrame();
    const std::vector<cv::Mat> rangeScaledOut = frame.generateOutputSynthFrame(65536.0f);

    ASSERT_EQ(defaultOut.size(), 1U);
    ASSERT_EQ(rangeScaledOut.size(), 1U);
    EXPECT_EQ(defaultOut[0].type(), CV_8U);
    EXPECT_EQ(rangeScaledOut[0].type(), CV_8U);
    EXPECT_EQ(defaultOut[0].at<unsigned char>(0, 0), 0);
    EXPECT_GT(rangeScaledOut[0].at<unsigned char>(0, 0), 0);
}

TEST(FrameTest, GenerateSynthFrameAppliesBackgroundBrightnessFactor) {
    SimulationConfig cfg = MakeFrameConfig(1);
    cfg.synth_background_brightness_factor = 1.2f;
    std::vector<cv::Mat> real = {cv::Mat::zeros(3, 3, CV_32F)};
    Frame frame(real, cfg, {}, "", "synth_background_factor");
    frame.setBackgroundColor(0.5f);

    auto synth = frame.generateSynthFrame();

    ASSERT_EQ(synth.size(), 1u);
    EXPECT_NEAR(synth[0].at<float>(0, 0), 0.6f, 1e-6f);
}

TEST(FrameTest, BurnBboxLocalBackgroundUpdatesOnlyBackgroundSynthPixels) {
    ConfigureFrameTestEllipsoidBounds();
    SimulationConfig cfg = MakeFrameConfig(1);
    cfg.bbox_local_background_enabled = true;
    cfg.bbox_local_background_burn_into_synth = true;
    cfg.bbox_local_background_percentile = 0.5f;
    cfg.bbox_local_background_blend_with_frame = 0.0f;
    cfg.bbox_local_background_min_samples = 1;
    cfg.bbox_local_background_min_delta = -1.0f;
    cfg.bbox_local_background_max_delta = 1.0f;
    cfg.bbox_local_background_burn_cell_exclusion_scale = 1.0f;
    cfg.bbox_local_background_burn_feather_radius = 0.0f;
    cfg.bbox_local_background_burn_margin = 1.0f;
    cfg.bbox_local_background_burn_global_fallback_weight = 0.0f;

    cv::Mat slice(7, 7, CV_32F, cv::Scalar(0.2f));
    slice.at<float>(3, 3) = 1.0f;
    std::vector<cv::Mat> real = {slice};

    EllipsoidParams p("cell", 3.0f, 3.0f, 0.0f, 1.0f, 1.0f,
                      0.0f, 0.0f, 0.0f, 0.8f);
    Frame frame(real, cfg, {Ellipsoid(p)}, "", "bbox_burn");
    frame.setBackgroundColor(0.1f);
    frame.regenerateSynthFrame();

    frame.burnBboxLocalBackgroundIntoSynth();
    const auto synth = frame.getSynthFrame();

    ASSERT_EQ(synth.size(), 1u);
    EXPECT_NEAR(synth[0].at<float>(0, 0), 0.2f, 1e-6f);
    EXPECT_GT(synth[0].at<float>(3, 3), 0.2f);
}

TEST(FrameTest, GenerateSynthFrameUsesEmpiricalBackgroundNoise) {
    SimulationConfig cfg = MakeFrameConfig(1);
    cfg.synth_background_noise_enabled = true;
    cfg.synth_background_noise_scale = 1.0f;
    cfg.synth_background_noise_cell_mask_expand_factor = 1.0f;

    cv::Mat slice(1, 2, CV_32F);
    slice.at<float>(0, 0) = 0.4f;
    slice.at<float>(0, 1) = 0.6f;
    std::vector<cv::Mat> real = {slice};
    Frame frame(real, cfg, {}, "", "noise_full");
    frame.setBackgroundColor(0.5f);

    auto synth = frame.generateSynthFrame();

    ASSERT_EQ(synth.size(), 1u);
    EXPECT_NEAR(synth[0].at<float>(0, 0), 0.4f, 1e-6f);
    EXPECT_NEAR(synth[0].at<float>(0, 1), 0.6f, 1e-6f);
}

TEST(FrameTest, GenerateSynthFrameFastUsesEmpiricalBackgroundNoise) {
    ConfigureFrameTestEllipsoidBounds();
    SimulationConfig cfg = MakeFrameConfig(1);
    cfg.synth_background_noise_enabled = true;
    cfg.synth_background_noise_scale = 1.0f;
    cfg.synth_background_noise_cell_mask_expand_factor = 1.0f;

    cv::Mat slice(2, 2, CV_32F, cv::Scalar(0.5f));
    slice.at<float>(0, 0) = 0.4f;
    slice.at<float>(0, 1) = 0.6f;
    std::vector<cv::Mat> real = {slice};

    EllipsoidParams p("cell", 1.0f, 1.0f, 0.0f, 0.2f, 0.2f,
                      0.0f, 0.0f, 0.0f, 0.9f);
    Ellipsoid oldCell(p);
    std::vector<Ellipsoid> cells = {oldCell};
    Frame frame(real, cfg, cells, "", "noise_fast");
    frame.setBackgroundColor(0.5f);
    frame.regenerateSynthFrame();

    EllipsoidParams moved = p;
    moved.x = 1.0f;
    moved.y = 1.0f;
    Ellipsoid newCell(moved);
    auto synth = frame.generateSynthFrameFast(oldCell, newCell);

    ASSERT_EQ(synth.size(), 1u);
    EXPECT_NEAR(synth[0].at<float>(0, 0), 0.4f, 1e-6f);
    EXPECT_NEAR(synth[0].at<float>(0, 1), 0.6f, 1e-6f);
}

TEST(FrameTest, GenerateSynthFrameUsesRandomGaussianBackgroundNoise) {
    SimulationConfig cfg = MakeFrameConfig(1);
    cfg.synth_background_noise_enabled = true;
    cfg.synth_background_noise_mode = "random_gaussian";
    cfg.synth_background_noise_scale = 1.0f;
    cfg.synth_background_noise_seed = 7;

    cv::Mat slice(1, 9, CV_32F);
    for (int x = 0; x < slice.cols; ++x) {
        slice.at<float>(0, x) = 0.5f + 0.01f * static_cast<float>(x - 4);
    }
    std::vector<cv::Mat> real = {slice};
    Frame frame(real, cfg, {}, "", "noise_random_gaussian");
    frame.setBackgroundColor(0.5f);

    auto synth = frame.generateSynthFrame();

    ASSERT_EQ(synth.size(), 1u);
    bool anyDifferentFromBackground = false;
    for (int x = 0; x < synth[0].cols; ++x) {
        const float v = synth[0].at<float>(0, x);
        EXPECT_GE(v, 0.0f);
        EXPECT_LE(v, 1.0f);
        anyDifferentFromBackground =
            anyDifferentFromBackground || std::abs(v - 0.5f) > 1e-6f;
    }
    EXPECT_TRUE(anyDifferentFromBackground);
}
