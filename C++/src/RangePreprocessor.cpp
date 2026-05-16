#include "../includes/RangePreprocessor.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

struct StackShape {
    int depth = 0;
    int rows = 0;
    int cols = 0;
};

StackShape validateStack(const ImageStack &stack)
{
    if (stack.empty()) {
        throw std::runtime_error("range preprocess: input stack is empty");
    }
    if (stack.front().empty()) {
        throw std::runtime_error("range preprocess: first input slice is empty");
    }
    if (stack.front().type() != CV_32F) {
        throw std::runtime_error("range preprocess: input slices must be CV_32F");
    }

    StackShape shape;
    shape.depth = static_cast<int>(stack.size());
    shape.rows = stack.front().rows;
    shape.cols = stack.front().cols;
    if (shape.rows <= 0 || shape.cols <= 0) {
        throw std::runtime_error("range preprocess: invalid input slice shape");
    }

    for (std::size_t z = 0; z < stack.size(); ++z) {
        const cv::Mat &slice = stack[z];
        if (slice.empty()) {
            throw std::runtime_error("range preprocess: input contains an empty slice");
        }
        if (slice.type() != CV_32F) {
            throw std::runtime_error("range preprocess: all input slices must be CV_32F");
        }
        if (slice.rows != shape.rows || slice.cols != shape.cols) {
            throw std::runtime_error("range preprocess: all input slices must share one shape");
        }
    }
    return shape;
}

void validateConfig(const SimulationConfig &simulation)
{
    if (simulation.range_preprocess_range_count <= 0) {
        throw std::runtime_error("range preprocess: range_preprocess_range_count must be positive");
    }
    if (simulation.range_preprocess_max_brightness <= 0.0f) {
        throw std::runtime_error("range preprocess: range_preprocess_max_brightness must be positive");
    }
    if (simulation.range_preprocess_export_max_brightness <= 0.0f) {
        throw std::runtime_error("range preprocess: range_preprocess_export_max_brightness must be positive");
    }
    const auto requirePercentile = [](float value, const char *name) {
        if (value < 0.0f || value > 100.0f) {
            throw std::runtime_error(std::string("range preprocess: ") + name +
                                     " must be in [0, 100]");
        }
    };
    requirePercentile(simulation.range_preprocess_range_percentile,
                      "range_preprocess_range_percentile");
    requirePercentile(simulation.range_preprocess_final_threshold_percentile,
                      "range_preprocess_final_threshold_percentile");
    if (simulation.range_preprocess_occupancy_threshold_percent < 0.0f) {
        throw std::runtime_error(
            "range preprocess: range_preprocess_occupancy_threshold_percent must be non-negative");
    }
    if (simulation.range_preprocess_sigma < 0.0f) {
        throw std::runtime_error("range preprocess: range_preprocess_sigma must be non-negative");
    }
    if (simulation.range_preprocess_real_ratio < 0.0f ||
        simulation.range_preprocess_real_ratio > 1.0f) {
        throw std::runtime_error("range preprocess: range_preprocess_real_ratio must be in [0, 1]");
    }
    if (!std::isfinite(simulation.range_preprocess_bright_boost_fraction) ||
        simulation.range_preprocess_bright_boost_fraction < 0.0f ||
        simulation.range_preprocess_bright_boost_fraction > 1.0f) {
        throw std::runtime_error(
            "range preprocess: range_preprocess_bright_boost_fraction must be in [0, 1]");
    }
    if (!std::isfinite(simulation.range_preprocess_bright_boost_factor) ||
        simulation.range_preprocess_bright_boost_factor <= 0.0f) {
        throw std::runtime_error(
            "range preprocess: range_preprocess_bright_boost_factor must be positive");
    }
}

ImageStack makeEmptyStack(const StackShape &shape)
{
    ImageStack out;
    out.reserve(static_cast<std::size_t>(shape.depth));
    for (int z = 0; z < shape.depth; ++z) {
        out.emplace_back(cv::Mat::zeros(shape.rows, shape.cols, CV_32F));
    }
    return out;
}

int binForValue(float value, float step, int rangeCount)
{
    int bin = static_cast<int>(std::floor(value / step));
    if (bin < 0) bin = 0;
    if (bin >= rangeCount) bin = rangeCount - 1;
    return bin;
}

std::vector<float> collectNonzeroValues(const ImageStack &stack)
{
    std::vector<std::vector<float>> perSlice(stack.size());

    #pragma omp parallel for schedule(static)
    for (int zi = 0; zi < static_cast<int>(stack.size()); ++zi) {
        const cv::Mat &slice = stack[static_cast<std::size_t>(zi)];
        std::vector<float> values;
        values.reserve(slice.total() / 16U + 1U);
        for (int y = 0; y < slice.rows; ++y) {
            const float *row = slice.ptr<float>(y);
            for (int x = 0; x < slice.cols; ++x) {
                const float value = row[x];
                if (std::isfinite(value) && value > 0.0f) {
                    values.push_back(value);
                }
            }
        }
        perSlice[static_cast<std::size_t>(zi)] = std::move(values);
    }

    std::size_t total = 0;
    for (const auto &values : perSlice) total += values.size();

    std::vector<float> merged;
    merged.reserve(total);
    for (auto &values : perSlice) {
        merged.insert(merged.end(),
                      std::make_move_iterator(values.begin()),
                      std::make_move_iterator(values.end()));
    }
    return merged;
}

std::size_t countNonzeroFinite(const ImageStack &stack)
{
    std::size_t count = 0;
    #pragma omp parallel for schedule(static) reduction(+:count)
    for (int z = 0; z < static_cast<int>(stack.size()); ++z) {
        const cv::Mat &slice = stack[static_cast<std::size_t>(z)];
        for (int y = 0; y < slice.rows; ++y) {
            const float *row = slice.ptr<float>(y);
            for (int x = 0; x < slice.cols; ++x) {
                if (std::isfinite(row[x]) && row[x] > 0.0f) {
                    ++count;
                }
            }
        }
    }
    return count;
}

ImageStack interpolateZStack(const ImageStack &stack, const SimulationConfig &simulation)
{
    const int expandFactor = static_cast<int>(simulation.z_scaling);
    if (!simulation.range_preprocess_interpolate_z ||
        stack.size() <= 1U ||
        expandFactor <= 1) {
        return stack;
    }

    const unsigned numSynthSlices =
        static_cast<unsigned>(expandFactor) * (stack.size() - 1U) + 1U;
    ImageStack interpolated(numSynthSlices);

    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < static_cast<int>(numSynthSlices); ++idx) {
        const int loIdx = idx / expandFactor;
        const int hiIdx = std::min(static_cast<int>(stack.size()) - 1, loIdx + 1);
        const float t = static_cast<float>(idx % expandFactor) /
                        static_cast<float>(expandFactor);
        if (idx % expandFactor == 0) {
            stack[static_cast<std::size_t>(loIdx)].copyTo(
                interpolated[static_cast<std::size_t>(idx)]);
        } else {
            interpolated[static_cast<std::size_t>(idx)] =
                (1.0f - t) * stack[static_cast<std::size_t>(loIdx)] +
                t * stack[static_cast<std::size_t>(hiIdx)];
        }
    }

    return interpolated;
}

} // namespace

float RangePreprocessor::percentileLinear(std::vector<float> values, float percentile)
{
    if (values.empty()) {
        throw std::runtime_error("range preprocess: cannot compute percentile of empty values");
    }
    const float clamped = std::clamp(percentile, 0.0f, 100.0f);
    const double rank =
        (static_cast<double>(values.size()) - 1.0) * static_cast<double>(clamped) / 100.0;
    const std::size_t lo = static_cast<std::size_t>(std::floor(rank));
    const std::size_t hi = static_cast<std::size_t>(std::ceil(rank));
    const double t = rank - static_cast<double>(lo);

    std::nth_element(values.begin(),
                     values.begin() + static_cast<std::ptrdiff_t>(lo),
                     values.end());
    const float loValue = values[lo];
    if (hi == lo) {
        return loValue;
    }

    std::nth_element(values.begin(),
                     values.begin() + static_cast<std::ptrdiff_t>(hi),
                     values.end());
    const float hiValue = values[hi];
    return static_cast<float>(
        static_cast<double>(loValue) * (1.0 - t) +
        static_cast<double>(hiValue) * t);
}

ImageStack RangePreprocessor::apply(const ImageStack &rawSlices,
                                    const BaseConfig &config,
                                    const std::string &imageFile,
                                    RangePreprocessStats *stats,
                                    std::ostream *logSink)
{
    std::ostream &log = logSink ? *logSink : std::cout;
    const SimulationConfig &simulation = config.simulation;
    validateConfig(simulation);
    const StackShape shape = validateStack(rawSlices);

    const int rangeCount = simulation.range_preprocess_range_count;
    const float maxBrightness = simulation.range_preprocess_max_brightness;
    const std::size_t totalVoxels =
        static_cast<std::size_t>(shape.depth) *
        static_cast<std::size_t>(shape.rows) *
        static_cast<std::size_t>(shape.cols);

    ImageStack normalized = makeEmptyStack(shape);
    #pragma omp parallel for schedule(static)
    for (int z = 0; z < shape.depth; ++z) {
        const cv::Mat &src = rawSlices[static_cast<std::size_t>(z)];
        cv::Mat &dst = normalized[static_cast<std::size_t>(z)];
        for (int y = 0; y < shape.rows; ++y) {
            const float *srcRow = src.ptr<float>(y);
            float *dstRow = dst.ptr<float>(y);
            for (int x = 0; x < shape.cols; ++x) {
                float value = srcRow[x];
                if (!std::isfinite(value) || value < 0.0f) {
                    value = 0.0f;
                }
                value = std::min(value, maxBrightness);
                dstRow[x] = value / maxBrightness;
            }
        }
    }

    std::vector<float> nonzeroNormalized = collectNonzeroValues(normalized);
    if (nonzeroNormalized.empty()) {
        throw std::runtime_error("range preprocess: normalized stack has no nonzero pixels");
    }
    const float rangeUpperBound =
        percentileLinear(std::move(nonzeroNormalized),
                         simulation.range_preprocess_range_percentile);
    if (rangeUpperBound <= 0.0f) {
        throw std::runtime_error("range preprocess: range upper bound is not positive");
    }

    const float step = rangeUpperBound / static_cast<float>(rangeCount);
    if (step <= 0.0f) {
        throw std::runtime_error("range preprocess: range step is not positive");
    }

    std::vector<std::vector<std::size_t>> perSliceCounts(
        static_cast<std::size_t>(shape.depth),
        std::vector<std::size_t>(static_cast<std::size_t>(rangeCount), 0U));
    #pragma omp parallel for schedule(static)
    for (int z = 0; z < shape.depth; ++z) {
        const cv::Mat &slice = normalized[static_cast<std::size_t>(z)];
        auto &counts = perSliceCounts[static_cast<std::size_t>(z)];
        for (int y = 0; y < shape.rows; ++y) {
            const float *row = slice.ptr<float>(y);
            for (int x = 0; x < shape.cols; ++x) {
                const float value = row[x];
                if (value <= 0.0f || value > rangeUpperBound) {
                    continue;
                }
                ++counts[static_cast<std::size_t>(binForValue(value, step, rangeCount))];
            }
        }
    }

    std::vector<std::size_t> binCounts(static_cast<std::size_t>(rangeCount), 0U);
    for (const auto &counts : perSliceCounts) {
        for (int bin = 0; bin < rangeCount; ++bin) {
            binCounts[static_cast<std::size_t>(bin)] += counts[static_cast<std::size_t>(bin)];
        }
    }

    std::vector<uint8_t> activeBins(static_cast<std::size_t>(rangeCount), 0U);
    int activeRanges = 0;
    int excludedRanges = 0;
    for (int bin = 0; bin < rangeCount; ++bin) {
        const double percent =
            static_cast<double>(binCounts[static_cast<std::size_t>(bin)]) /
            static_cast<double>(std::max<std::size_t>(1U, totalVoxels)) *
            100.0;
        const bool excluded =
            percent >= static_cast<double>(simulation.range_preprocess_occupancy_threshold_percent);
        if (excluded) {
            ++excludedRanges;
        } else if (binCounts[static_cast<std::size_t>(bin)] > 0U) {
            activeBins[static_cast<std::size_t>(bin)] = 1U;
            ++activeRanges;
        }
    }

    ImageStack combined = makeEmptyStack(shape);
    #pragma omp parallel for schedule(static)
    for (int z = 0; z < shape.depth; ++z) {
        const cv::Mat &src = normalized[static_cast<std::size_t>(z)];
        cv::Mat &dst = combined[static_cast<std::size_t>(z)];
        for (int y = 0; y < shape.rows; ++y) {
            const float *srcRow = src.ptr<float>(y);
            float *dstRow = dst.ptr<float>(y);
            for (int x = 0; x < shape.cols; ++x) {
                const float value = srcRow[x];
                if (value <= 0.0f || value > rangeUpperBound) {
                    dstRow[x] = 0.0f;
                    continue;
                }
                const int bin = binForValue(value, step, rangeCount);
                dstRow[x] = activeBins[static_cast<std::size_t>(bin)] ? value : 0.0f;
            }
        }
    }

    ImageStack preprocessed = makeEmptyStack(shape);
    const float sigma = simulation.range_preprocess_sigma;
    const float realRatio = simulation.range_preprocess_real_ratio;
    if (simulation.range_preprocess_blur_enabled) {
        #pragma omp parallel for schedule(static)
        for (int z = 0; z < shape.depth; ++z) {
            const cv::Mat &src = combined[static_cast<std::size_t>(z)];
            cv::Mat blurred;
            if (sigma > 0.0f) {
                cv::GaussianBlur(src, blurred, cv::Size(0, 0), sigma, sigma);
            } else {
                blurred = src;
            }
            preprocessed[static_cast<std::size_t>(z)] =
                src * realRatio + blurred * (1.0f - realRatio);
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (int z = 0; z < shape.depth; ++z) {
            combined[static_cast<std::size_t>(z)].copyTo(
                preprocessed[static_cast<std::size_t>(z)]);
        }
    }

    std::vector<float> nonzeroPreprocessed = collectNonzeroValues(preprocessed);
    if (nonzeroPreprocessed.empty()) {
        throw std::runtime_error("range preprocess: preprocessed stack has no nonzero pixels");
    }
    const float finalThreshold =
        percentileLinear(std::move(nonzeroPreprocessed),
                         simulation.range_preprocess_final_threshold_percentile);

    std::size_t thresholdedNonzero = 0;
    #pragma omp parallel for schedule(static) reduction(+:thresholdedNonzero)
    for (int z = 0; z < shape.depth; ++z) {
        cv::Mat &slice = preprocessed[static_cast<std::size_t>(z)];
        for (int y = 0; y < shape.rows; ++y) {
            float *row = slice.ptr<float>(y);
            for (int x = 0; x < shape.cols; ++x) {
                if (!std::isfinite(row[x]) || row[x] < finalThreshold) {
                    row[x] = 0.0f;
                } else if (row[x] > 0.0f) {
                    ++thresholdedNonzero;
                }
            }
        }
    }

    float brightBoostThreshold = 0.0f;
    float brightBoostEffectiveFactor = 1.0f;
    std::size_t brightBoostedVoxels = 0;
    const float brightBoostFraction = simulation.range_preprocess_bright_boost_fraction;
    const float brightBoostFactor = simulation.range_preprocess_bright_boost_factor;
    if (brightBoostFraction > 0.0f &&
        std::abs(brightBoostFactor - 1.0f) > 1e-6f &&
        thresholdedNonzero > 0U) {
        std::vector<float> nonzeroThresholded = collectNonzeroValues(preprocessed);
        const float boostPercentile =
            std::clamp((1.0f - brightBoostFraction) * 100.0f, 0.0f, 100.0f);
        brightBoostThreshold =
            percentileLinear(nonzeroThresholded, boostPercentile);

        constexpr float kNormalizedSaturation = 1.0f;
        float maxBoostInput = 0.0f;
        for (float value : nonzeroThresholded) {
            if (std::isfinite(value) && value > 0.0f &&
                value >= brightBoostThreshold) {
                maxBoostInput = std::max(maxBoostInput, value);
            }
        }
        brightBoostEffectiveFactor = brightBoostFactor;
        while (maxBoostInput > 0.0f &&
               maxBoostInput * brightBoostEffectiveFactor > kNormalizedSaturation) {
            brightBoostEffectiveFactor *= 0.5f;
        }

        #pragma omp parallel for schedule(static) reduction(+:brightBoostedVoxels)
        for (int z = 0; z < shape.depth; ++z) {
            cv::Mat &slice = preprocessed[static_cast<std::size_t>(z)];
            for (int y = 0; y < shape.rows; ++y) {
                float *row = slice.ptr<float>(y);
                for (int x = 0; x < shape.cols; ++x) {
                    if (std::isfinite(row[x]) && row[x] > 0.0f &&
                        row[x] >= brightBoostThreshold) {
                        row[x] *= brightBoostEffectiveFactor;
                        ++brightBoostedVoxels;
                    }
                }
            }
        }
    }

    const std::size_t thresholdedSlices = preprocessed.size();
    preprocessed = interpolateZStack(preprocessed, simulation);
    const std::size_t outputNonzero =
        preprocessed.size() == thresholdedSlices
            ? thresholdedNonzero
            : countNonzeroFinite(preprocessed);

    if (stats) {
        stats->rangeUpperBound = rangeUpperBound;
        stats->ranges = rangeCount;
        stats->activeRanges = activeRanges;
        stats->excludedRanges = excludedRanges;
        stats->finalThreshold = finalThreshold;
        stats->brightBoostThreshold = brightBoostThreshold;
        stats->brightBoostEffectiveFactor = brightBoostEffectiveFactor;
        stats->brightBoostedVoxels = brightBoostedVoxels;
        stats->outputNonzero = outputNonzero;
    }

    if (simulation.range_preprocess_debug_stats) {
        log << "[RangePreprocess]"
            << " file=" << std::filesystem::path(imageFile).filename().string()
            << " shape=(" << shape.depth << "," << shape.rows << "," << shape.cols << ")"
            << " range_upper_bound=" << rangeUpperBound
            << " ranges=" << rangeCount
            << " active_ranges=" << activeRanges
            << " excluded_ranges=" << excludedRanges
            << " occupancy_threshold_percent="
            << simulation.range_preprocess_occupancy_threshold_percent
            << " blur_enabled=" << simulation.range_preprocess_blur_enabled
            << " interpolate_z=" << simulation.range_preprocess_interpolate_z
            << " auto_dynamic_range=" << simulation.range_preprocess_auto_dynamic_range
            << " max_brightness=" << simulation.range_preprocess_max_brightness
            << " export_max_brightness=" << simulation.range_preprocess_export_max_brightness
            << " sigma=" << sigma
            << " real_ratio=" << realRatio
            << " final_threshold_percentile="
            << simulation.range_preprocess_final_threshold_percentile
            << " final_threshold=" << finalThreshold
            << " bright_boost_fraction=" << brightBoostFraction
            << " bright_boost_factor=" << brightBoostFactor
            << " bright_boost_effective_factor=" << brightBoostEffectiveFactor
            << " bright_boost_threshold=" << brightBoostThreshold
            << " bright_boosted_voxels=" << brightBoostedVoxels
            << " output_slices=" << preprocessed.size()
            << " output_nonzero=" << outputNonzero
            << std::endl;
    }

    return preprocessed;
}
