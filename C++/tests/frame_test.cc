#include <gtest/gtest.h>

#include "Frame.hpp"

#include <stdexcept>
#include <vector>

namespace {
SimulationConfig MakeFrameConfig(int zSlices, float background) {
    SimulationConfig cfg;
    cfg.z_slices = zSlices;
    cfg.z_scaling = 1.0f;
    cfg.comparison_blur_sigma = 0.0f;
    return cfg;
}
}

TEST(FrameTest, CalculateCostReturnsZeroForIdenticalStacks) {
    const SimulationConfig cfg = MakeFrameConfig(1, 0.0f);
    std::vector<cv::Mat> real = {cv::Mat::zeros(2, 2, CV_32F)};
    Frame frame(real, cfg, {}, "", "f0");

    EXPECT_DOUBLE_EQ(frame.calculateCost(real), 0.0);
}

TEST(FrameTest, CalculateCostSumsL2NormAcrossSlices) {
    const SimulationConfig cfg = MakeFrameConfig(2, 0.0f);
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

TEST(FrameTest, CalculateCostCanBlurSyntheticSliceBeforeComparison) {
    SimulationConfig cfg = MakeFrameConfig(1, 0.0f);
    cfg.comparison_blur_sigma = 1.0f;

    cv::Mat real = cv::Mat::zeros(7, 7, CV_32F);
    cv::Mat synth = cv::Mat::zeros(7, 7, CV_32F);
    synth.at<float>(3, 3) = 1.0f;

    Frame frame({real}, cfg, {}, "", "blurred-cost");

    cv::Mat blurredSynth;
    cv::GaussianBlur(synth, blurredSynth, cv::Size(0, 0), cfg.comparison_blur_sigma, cfg.comparison_blur_sigma);

    EXPECT_NEAR(frame.calculateCost({synth}), cv::norm(real, blurredSynth, cv::NORM_L2), 1e-6);
}

TEST(FrameTest, CalculateCostThrowsOnMismatchedStackSize) {
    const SimulationConfig cfg = MakeFrameConfig(2, 0.0f);
    std::vector<cv::Mat> real = {
        cv::Mat::zeros(2, 2, CV_32F),
        cv::Mat::zeros(2, 2, CV_32F)
    };
    Frame frame(real, cfg, {}, "", "f2");

    std::vector<cv::Mat> synth = {cv::Mat::zeros(2, 2, CV_32F)};
    EXPECT_THROW(frame.calculateCost(synth), std::runtime_error);
}

TEST(FrameTest, GenerateOutputSynthFrameConvertsFloatTo8Bit) {
    const SimulationConfig cfg = MakeFrameConfig(1, 0.5f);
    std::vector<cv::Mat> real = {cv::Mat::zeros(3, 3, CV_32F)};
    Frame frame(real, cfg, {}, "", "f3");

    std::vector<cv::Mat> out = frame.generateOutputSynthFrame();

    ASSERT_EQ(out.size(), 1U);
    EXPECT_EQ(out[0].type(), CV_8U);
    EXPECT_NEAR(static_cast<double>(out[0].at<unsigned char>(0, 0)), 128.0, 1.0);
}
