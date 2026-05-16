#include <gtest/gtest.h>

#include "RangePreprocessor.hpp"

#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

BaseConfig makeRangeConfig()
{
    BaseConfig cfg;
    cfg.simulation.range_preprocess_enabled = true;
    cfg.simulation.range_preprocess_range_count = 4;
    cfg.simulation.range_preprocess_range_percentile = 100.0f;
    cfg.simulation.range_preprocess_occupancy_threshold_percent = 101.0f;
    cfg.simulation.range_preprocess_final_threshold_percentile = 0.0f;
    cfg.simulation.range_preprocess_bright_boost_fraction = 0.30f;
    cfg.simulation.range_preprocess_bright_boost_factor = 1.0f;
    cfg.simulation.range_preprocess_blur_enabled = false;
    cfg.simulation.range_preprocess_sigma = 0.0f;
    cfg.simulation.range_preprocess_real_ratio = 0.0f;
    cfg.simulation.range_preprocess_max_brightness = 100.0f;
    cfg.simulation.range_preprocess_export_max_brightness = 100.0f;
    cfg.simulation.range_preprocess_debug_stats = false;
    return cfg;
}

ImageStack singleSlice(int rows, int cols, const std::vector<float> &values)
{
    if (static_cast<int>(values.size()) != rows * cols) {
        throw std::runtime_error("test fixture values do not match requested shape");
    }

    cv::Mat slice(rows, cols, CV_32F);
    for (int y = 0; y < rows; ++y) {
        float *row = slice.ptr<float>(y);
        for (int x = 0; x < cols; ++x) {
            row[x] = values[static_cast<std::size_t>(y * cols + x)];
        }
    }
    return ImageStack{slice};
}

cv::Mat onePixelSlice(float value)
{
    cv::Mat slice(1, 1, CV_32F);
    slice.at<float>(0, 0) = value;
    return slice;
}

} // namespace

TEST(RangePreprocessorTest, PercentileLinearMatchesNumpyInterpolation)
{
    EXPECT_FLOAT_EQ(
        RangePreprocessor::percentileLinear({0.0f, 10.0f, 20.0f, 30.0f}, 0.0f),
        0.0f);
    EXPECT_FLOAT_EQ(
        RangePreprocessor::percentileLinear({0.0f, 10.0f, 20.0f, 30.0f}, 25.0f),
        7.5f);
    EXPECT_FLOAT_EQ(
        RangePreprocessor::percentileLinear({0.0f, 10.0f, 20.0f, 30.0f}, 50.0f),
        15.0f);
    EXPECT_FLOAT_EQ(
        RangePreprocessor::percentileLinear({0.0f, 10.0f, 20.0f, 30.0f}, 75.0f),
        22.5f);
    EXPECT_FLOAT_EQ(
        RangePreprocessor::percentileLinear({0.0f, 10.0f, 20.0f, 30.0f}, 100.0f),
        30.0f);
}

TEST(RangePreprocessorTest, PreservesNormalizedBrightnessAtRangeBoundaries)
{
    BaseConfig cfg = makeRangeConfig();
    const ImageStack input = singleSlice(1, 5, {0.0f, 25.0f, 50.0f, 75.0f, 100.0f});
    RangePreprocessStats stats;
    std::ostringstream log;

    const ImageStack out = RangePreprocessor::apply(input, cfg, "synthetic.tif", &stats, &log);

    ASSERT_EQ(out.size(), 1u);
    EXPECT_FLOAT_EQ(out[0].at<float>(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(out[0].at<float>(0, 1), 0.25f);
    EXPECT_FLOAT_EQ(out[0].at<float>(0, 2), 0.5f);
    EXPECT_FLOAT_EQ(out[0].at<float>(0, 3), 0.75f);
    EXPECT_FLOAT_EQ(out[0].at<float>(0, 4), 1.0f);
    EXPECT_EQ(stats.ranges, 4);
    EXPECT_EQ(stats.excludedRanges, 0);
}

TEST(RangePreprocessorTest, ExcludesRangesAtOrAboveGlobalOccupancyThreshold)
{
    BaseConfig cfg = makeRangeConfig();
    cfg.simulation.range_preprocess_occupancy_threshold_percent = 60.0f;
    const ImageStack input = singleSlice(
        2, 5,
        {10.0f, 10.0f, 10.0f, 10.0f, 10.0f,
         10.0f, 50.0f, 100.0f, 0.0f, 0.0f});
    RangePreprocessStats stats;
    std::ostringstream log;

    const ImageStack out = RangePreprocessor::apply(input, cfg, "synthetic.tif", &stats, &log);

    ASSERT_EQ(out.size(), 1u);
    for (int x = 0; x < 5; ++x) {
        EXPECT_FLOAT_EQ(out[0].at<float>(0, x), 0.0f);
    }
    EXPECT_FLOAT_EQ(out[0].at<float>(1, 0), 0.0f);
    EXPECT_FLOAT_EQ(out[0].at<float>(1, 1), 0.5f);
    EXPECT_FLOAT_EQ(out[0].at<float>(1, 2), 1.0f);
    EXPECT_FLOAT_EQ(out[0].at<float>(1, 3), 0.0f);
    EXPECT_FLOAT_EQ(out[0].at<float>(1, 4), 0.0f);
    EXPECT_EQ(stats.excludedRanges, 1);
    EXPECT_EQ(stats.activeRanges, 2);
    EXPECT_EQ(stats.outputNonzero, 2u);
}

TEST(RangePreprocessorTest, FinalThresholdZerosOnlyValuesStrictlyBelowThreshold)
{
    BaseConfig cfg = makeRangeConfig();
    cfg.simulation.range_preprocess_range_count = 1;
    cfg.simulation.range_preprocess_final_threshold_percentile = 50.0f;
    const ImageStack input = singleSlice(1, 3, {10.0f, 20.0f, 30.0f});
    RangePreprocessStats stats;
    std::ostringstream log;

    const ImageStack out = RangePreprocessor::apply(input, cfg, "synthetic.tif", &stats, &log);

    ASSERT_EQ(out.size(), 1u);
    EXPECT_NEAR(stats.finalThreshold, 0.2f, 1e-6f);
    EXPECT_FLOAT_EQ(out[0].at<float>(0, 0), 0.0f);
    EXPECT_FLOAT_EQ(out[0].at<float>(0, 1), 0.2f);
    EXPECT_FLOAT_EQ(out[0].at<float>(0, 2), 0.3f);
    EXPECT_EQ(stats.outputNonzero, 2u);
}

TEST(RangePreprocessorTest, BoostsBrightestNonzeroValuesBeforeInterpolation)
{
    BaseConfig cfg = makeRangeConfig();
    cfg.simulation.range_preprocess_range_count = 1;
    cfg.simulation.range_preprocess_bright_boost_fraction = 0.50f;
    cfg.simulation.range_preprocess_bright_boost_factor = 5.0f;
    const ImageStack input = singleSlice(1, 4, {10.0f, 20.0f, 30.0f, 40.0f});
    RangePreprocessStats stats;
    std::ostringstream log;

    const ImageStack out = RangePreprocessor::apply(input, cfg, "synthetic.tif", &stats, &log);

    ASSERT_EQ(out.size(), 1u);
    EXPECT_FLOAT_EQ(out[0].at<float>(0, 0), 0.1f);
    EXPECT_FLOAT_EQ(out[0].at<float>(0, 1), 0.2f);
    EXPECT_FLOAT_EQ(out[0].at<float>(0, 2), 0.75f);
    EXPECT_FLOAT_EQ(out[0].at<float>(0, 3), 1.0f);
    EXPECT_NEAR(stats.brightBoostThreshold, 0.25f, 1e-6f);
    EXPECT_FLOAT_EQ(stats.brightBoostEffectiveFactor, 2.5f);
    EXPECT_EQ(stats.brightBoostedVoxels, 2u);
    EXPECT_EQ(stats.outputNonzero, 4u);
}

TEST(RangePreprocessorTest, HalvesBrightBoostFactorUntilTopValuesDoNotSaturate)
{
    BaseConfig cfg = makeRangeConfig();
    cfg.simulation.range_preprocess_range_count = 1;
    cfg.simulation.range_preprocess_bright_boost_fraction = 0.50f;
    cfg.simulation.range_preprocess_bright_boost_factor = 8.0f;
    const ImageStack input = singleSlice(1, 4, {10.0f, 20.0f, 30.0f, 40.0f});
    RangePreprocessStats stats;
    std::ostringstream log;

    const ImageStack out = RangePreprocessor::apply(input, cfg, "synthetic.tif", &stats, &log);

    ASSERT_EQ(out.size(), 1u);
    EXPECT_FLOAT_EQ(out[0].at<float>(0, 0), 0.1f);
    EXPECT_FLOAT_EQ(out[0].at<float>(0, 1), 0.2f);
    EXPECT_FLOAT_EQ(out[0].at<float>(0, 2), 0.6f);
    EXPECT_FLOAT_EQ(out[0].at<float>(0, 3), 0.8f);
    EXPECT_NEAR(stats.brightBoostThreshold, 0.25f, 1e-6f);
    EXPECT_FLOAT_EQ(stats.brightBoostEffectiveFactor, 2.0f);
    EXPECT_EQ(stats.brightBoostedVoxels, 2u);
    EXPECT_EQ(stats.outputNonzero, 4u);
}

TEST(RangePreprocessorTest, InterpolatesZForRangeOutputWhenEnabled)
{
    BaseConfig cfg = makeRangeConfig();
    cfg.simulation.range_preprocess_range_count = 1;
    cfg.simulation.range_preprocess_interpolate_z = true;
    cfg.simulation.z_scaling = 3.0f;
    const ImageStack input = {onePixelSlice(30.0f), onePixelSlice(90.0f)};
    RangePreprocessStats stats;
    std::ostringstream log;

    const ImageStack out = RangePreprocessor::apply(input, cfg, "synthetic.tif", &stats, &log);

    ASSERT_EQ(out.size(), 4u);
    EXPECT_FLOAT_EQ(out[0].at<float>(0, 0), 0.3f);
    EXPECT_FLOAT_EQ(out[1].at<float>(0, 0), 0.5f);
    EXPECT_FLOAT_EQ(out[2].at<float>(0, 0), 0.7f);
    EXPECT_FLOAT_EQ(out[3].at<float>(0, 0), 0.9f);
    EXPECT_EQ(stats.outputNonzero, 4u);
}

TEST(RangePreprocessorTest, PreservesRawZDepthWhenRangeInterpolationDisabled)
{
    BaseConfig cfg = makeRangeConfig();
    cfg.simulation.range_preprocess_range_count = 1;
    cfg.simulation.range_preprocess_interpolate_z = false;
    cfg.simulation.z_scaling = 3.0f;
    const ImageStack input = {onePixelSlice(30.0f), onePixelSlice(90.0f)};
    std::ostringstream log;

    const ImageStack out = RangePreprocessor::apply(input, cfg, "synthetic.tif", nullptr, &log);

    ASSERT_EQ(out.size(), 2u);
    EXPECT_FLOAT_EQ(out[0].at<float>(0, 0), 0.3f);
    EXPECT_FLOAT_EQ(out[1].at<float>(0, 0), 0.9f);
}

TEST(RangePreprocessorTest, AllZeroStackThrows)
{
    BaseConfig cfg = makeRangeConfig();
    const ImageStack input = singleSlice(2, 2, {0.0f, 0.0f, 0.0f, 0.0f});
    std::ostringstream log;

    EXPECT_THROW(
        RangePreprocessor::apply(input, cfg, "empty.tif", nullptr, &log),
        std::runtime_error);
}
