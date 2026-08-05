#include <gtest/gtest.h>

#include "Frame.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace {
SimulationConfig MakeFrameConfig(int zSlices, float background) {
    SimulationConfig cfg;
    cfg.z_slices = zSlices;
    cfg.z_scaling = 1.0f;
    (void)background;
    return cfg;
}

Ellipsoid MakeTestEllipsoid(const std::string &name,
                            float brightness,
                            float a,
                            float b,
                            float c) {
    EllipsoidParams params(name, 1.0f, 1.0f, 1.0f,
                           a, c, 0.0f, 0.0f, 0.0f, brightness);
    params.bRadius = b;
    return Ellipsoid(params);
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
    frame.setBackgroundColor(0.5f);

    std::vector<cv::Mat> out = frame.generateOutputSynthFrame();

    ASSERT_EQ(out.size(), 1U);
    EXPECT_EQ(out[0].type(), CV_8U);
    EXPECT_NEAR(static_cast<double>(out[0].at<unsigned char>(0, 0)), 128.0, 1.0);
}

TEST(FrameTest, PsfBrightnessMomentsUseEllipsoidVolumeWeights) {
    const Ellipsoid small = MakeTestEllipsoid("a", 0.2f, 1.0f, 1.0f, 1.0f);
    const Ellipsoid large = MakeTestEllipsoid("b", 0.8f, 2.0f, 2.0f, 2.0f);

    const PsfBrightnessMoments moments =
        Frame::computePsfBrightnessMoments({&small, &large});

    ASSERT_TRUE(moments.valid);
    const double expectedMean = (0.2 + 8.0 * 0.8) / 9.0;
    const double expectedVariance =
        (std::pow(0.2 - expectedMean, 2.0) +
         8.0 * std::pow(0.8 - expectedMean, 2.0)) / 9.0;
    EXPECT_NEAR(moments.mean, expectedMean, 1e-6);
    EXPECT_NEAR(moments.spread, std::sqrt(expectedVariance), 1e-6);
}

TEST(FrameTest, PsfBrightnessCostUsesSymmetricImageTermAndPriorPenalty) {
    SimulationConfig cfg = MakeFrameConfig(1, 0.0f);
    cfg.asymmetric_cost_weight = 8.0f;
    cfg.asymmetric_cost_threshold = 0.0f;
    cfg.psf_brightness_cost_enabled = true;
    cfg.psf_brightness_cost_sigma = 0.0f;
    cfg.psf_brightness_prior_weight = 0.1f;
    cfg.psf_brightness_prior_tolerance = 0.0f;
    cfg.psf_brightness_contrast_floor = 0.05f;

    std::vector<cv::Mat> real = {cv::Mat::zeros(1, 1, CV_32F)};
    Frame frame(real, cfg, {}, "", "psf_cost");
    std::vector<cv::Mat> synth = {
        cv::Mat(1, 1, CV_32F, cv::Scalar(1.0f))
    };
    BoundingBox3D bbox;
    bbox.xMin = bbox.xMax = 0;
    bbox.yMin = bbox.yMax = 0;
    bbox.zMin = bbox.zMax = 0;
    PsfBrightnessCostContext context;
    context.candidate = {0.8, 0.0, true};
    context.prior = {1.0, 0.0, true};
    context.background = 0.0;
    context.valid = true;

    // Symmetric image term = 1. Legacy asymmetric L2 would be 8.
    // Brightness term = 1 voxel * 0.1 * ((0.8 - 1.0) / 1.0)^2 = 0.004.
    EXPECT_NEAR(frame.calculateBboxCost(bbox, synth, {}, -1, &context),
                1.004, 1e-6);
}

TEST(FrameTest, DisabledPsfBrightnessCostPreservesLegacyAsymmetricL2) {
    SimulationConfig cfg = MakeFrameConfig(1, 0.0f);
    cfg.asymmetric_cost_weight = 8.0f;
    cfg.asymmetric_cost_threshold = 0.0f;
    cfg.psf_brightness_cost_enabled = false;
    std::vector<cv::Mat> real = {cv::Mat::zeros(1, 1, CV_32F)};
    Frame frame(real, cfg, {}, "", "legacy_cost");
    std::vector<cv::Mat> synth = {
        cv::Mat(1, 1, CV_32F, cv::Scalar(1.0f))
    };
    BoundingBox3D bbox;
    bbox.xMin = bbox.xMax = 0;
    bbox.yMin = bbox.yMax = 0;
    bbox.zMin = bbox.zMax = 0;

    EXPECT_DOUBLE_EQ(frame.calculateBboxCost(bbox, synth, {}), 8.0);
}

TEST(FrameTest, PsfBrightnessCostMatchesSeparableGaussianGolden) {
    constexpr int side = 11;
    constexpr int center = side / 2;
    constexpr double sigma = 1.0;
    constexpr int radius = 4;
    std::vector<double> kernel(2 * radius + 1);
    double kernelSum = 0.0;
    for (int d = -radius; d <= radius; ++d) {
        const double weight = std::exp(-0.5 * d * d / (sigma * sigma));
        kernel[static_cast<size_t>(d + radius)] = weight;
        kernelSum += weight;
    }
    for (double &weight : kernel) weight /= kernelSum;

    std::vector<cv::Mat> real(
        side, cv::Mat::zeros(side, side, CV_32F));
    std::vector<cv::Mat> synth(
        side, cv::Mat::zeros(side, side, CV_32F));
    synth[center].at<float>(center, center) = 1.0f;
    for (int dz = -radius; dz <= radius; ++dz) {
        for (int dy = -radius; dy <= radius; ++dy) {
            float *row = real[center + dz].ptr<float>(center + dy);
            for (int dx = -radius; dx <= radius; ++dx) {
                row[center + dx] = static_cast<float>(
                    kernel[static_cast<size_t>(dz + radius)] *
                    kernel[static_cast<size_t>(dy + radius)] *
                    kernel[static_cast<size_t>(dx + radius)]);
            }
        }
    }

    SimulationConfig cfg = MakeFrameConfig(side, 0.0f);
    cfg.psf_brightness_cost_enabled = true;
    cfg.psf_brightness_cost_sigma = static_cast<float>(sigma);
    cfg.psf_brightness_prior_weight = 0.0f;
    Frame frame(real, cfg, {}, "", "psf_golden");
    BoundingBox3D bbox;
    bbox.xMin = bbox.yMin = bbox.zMin = 0;
    bbox.xMax = bbox.yMax = bbox.zMax = side - 1;
    PsfBrightnessCostContext context;
    context.candidate = {1.0, 0.0, true};
    context.prior = {1.0, 0.0, true};
    context.background = 0.0;
    context.valid = true;

    BboxCostBreakdown breakdown;
    const double cost = frame.calculateBboxCost(
        bbox, synth, {}, -1, &context, &breakdown);
    EXPECT_NEAR(cost, 0.0, 1e-10);
    EXPECT_NEAR(breakdown.imageCost, 0.0, 1e-10);
    EXPECT_DOUBLE_EQ(breakdown.brightnessPriorCost, 0.0);
    EXPECT_EQ(breakdown.evaluatedVoxelCount,
              static_cast<size_t>(side * side * side));
}

TEST(FrameTest, EnabledPsfBrightnessCostRejectsMissingContext) {
    SimulationConfig cfg = MakeFrameConfig(1, 0.0f);
    cfg.psf_brightness_cost_enabled = true;
    std::vector<cv::Mat> real = {cv::Mat::zeros(1, 1, CV_32F)};
    Frame frame(real, cfg, {}, "", "missing_context");
    BoundingBox3D bbox;
    bbox.xMin = bbox.xMax = 0;
    bbox.yMin = bbox.yMax = 0;
    bbox.zMin = bbox.zMax = 0;

    EXPECT_THROW(frame.calculateBboxCost(bbox, real, {}), std::runtime_error);
}

TEST(FrameTest, FrozenBrightnessPriorIsInheritedByNewbornLineage) {
    SimulationConfig cfg = MakeFrameConfig(1, 0.1f);
    cfg.psf_brightness_cost_enabled = true;
    const Ellipsoid parent =
        MakeTestEllipsoid("2", 0.8f, 2.0f, 2.0f, 2.0f);
    std::vector<cv::Mat> real = {cv::Mat::zeros(3, 3, CV_32F)};
    Frame frame(real, cfg, {parent}, "", "prior");
    frame.setBackgroundColor(0.1f);
    frame.freezePsfBrightnessPriors();

    const Ellipsoid daughter =
        MakeTestEllipsoid("20", 0.7f, 1.5f, 1.5f, 1.5f);
    const PsfBrightnessCostContext context =
        frame.makePsfBrightnessCostContext({&daughter}, daughter.getName());

    ASSERT_TRUE(context.valid);
    EXPECT_NEAR(context.candidate.mean, 0.7, 1e-6);
    EXPECT_NEAR(context.prior.mean, 0.8, 1e-6);
    EXPECT_NEAR(context.background, 0.1, 1e-6);
}
