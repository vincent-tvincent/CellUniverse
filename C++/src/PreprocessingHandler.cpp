#include "../includes/PreprocessingHandler.hpp"
#include "../includes/ImageHandler.hpp"

#ifndef CELLUNIVERSE_HAS_N2V2_PREPROCESS
#define CELLUNIVERSE_HAS_N2V2_PREPROCESS 0
#endif

#if CELLUNIVERSE_HAS_N2V2_PREPROCESS
#include "../prototype/n2v2_preprocess/include/N2V2Preprocessor.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

struct N2V2Runtime
{
#if CELLUNIVERSE_HAS_N2V2_PREPROCESS
    explicit N2V2Runtime(n2v2::PreprocessConfig config)
        : preprocessor(std::move(config))
    {
    }

    n2v2::N2V2Preprocessor preprocessor;
#else
    N2V2Runtime() = default;
#endif
};

#ifdef _OPENMP
#include <omp.h>
#else
static int omp_get_max_threads()
{
    return 1;
}

static int omp_in_parallel()
{
    return 0;
}
#endif

namespace
{
static float computeStackMean(const std::vector<cv::Mat> &stack)
{
    double sum = 0.0;
    double count = 0.0;
    for (const auto &slice : stack) {
        if (slice.empty()) {
            continue;
        }
        sum += cv::sum(slice)[0];
        count += static_cast<double>(slice.total());
    }
    return (count > 0.0) ? static_cast<float>(sum / count) : 0.0f;
}

static cv::Mat makeNapariFriendlyTiffSlice(const cv::Mat &slice)
{
    if (slice.empty()) {
        return {};
    }

    cv::Mat gray;
    if (slice.channels() == 1) {
        gray = slice;
    } else if (slice.channels() == 3) {
        cv::cvtColor(slice, gray, cv::COLOR_BGR2GRAY);
    } else if (slice.channels() == 4) {
        cv::cvtColor(slice, gray, cv::COLOR_BGRA2GRAY);
    } else {
        throw std::runtime_error("Unsupported channel count for TIFF export: " +
                                 std::to_string(slice.channels()));
    }

    cv::Mat output;
    if (gray.depth() == CV_8U) {
        output = gray.clone();
    } else {
        cv::Mat clipped = gray.clone();
        cv::patchNaNs(clipped, 0.0);
        cv::min(clipped, 1.0f, clipped);
        cv::max(clipped, 0.0f, clipped);
        clipped.convertTo(output, CV_8U, 255.0);
    }
    return output;
}

static std::vector<cv::Mat> makeNapariFriendlyTiffStack(const std::vector<cv::Mat> &stack)
{
    std::vector<cv::Mat> output;
    output.reserve(stack.size());

    cv::Size expectedSize;
    for (const auto &slice : stack) {
        cv::Mat converted = makeNapariFriendlyTiffSlice(slice);
        if (converted.empty()) {
            continue;
        }
        if (expectedSize.empty()) {
            expectedSize = converted.size();
        } else if (converted.size() != expectedSize) {
            throw std::runtime_error("TIFF export requires all slices to have the same size");
        }
        output.push_back(std::move(converted));
    }

    if (output.empty()) {
        throw std::runtime_error("TIFF export received an empty image stack");
    }
    return output;
}

static void writeNapariFriendlyTiffStack(const std::string &path,
                                         const std::vector<cv::Mat> &stack)
{
    std::vector<cv::Mat> output = makeNapariFriendlyTiffStack(stack);
    const std::vector<int> params = {
        cv::IMWRITE_TIFF_COMPRESSION, 1
    };

    if (!cv::imwritemulti(path, output, params)) {
        throw std::runtime_error("Failed to write TIFF stack: " + path);
    }
}

static float computeStackPercentile(const std::vector<cv::Mat> &stack,
                                    float percentile,
                                    bool excludeZeros = false)
{
    std::vector<float> values;
    size_t totalCount = 0;
    for (const auto &slice : stack) {
        if (!slice.empty()) {
            totalCount += static_cast<size_t>(slice.total());
        }
    }
    values.reserve(totalCount);
    for (const auto &slice : stack) {
        if (slice.empty()) {
            continue;
        }
        for (int y = 0; y < slice.rows; ++y) {
            const float *row = slice.ptr<float>(y);
            for (int x = 0; x < slice.cols; ++x) {
                const float value = row[x];
                if (!std::isfinite(value)) {
                    continue;
                }
                if (excludeZeros && value == 0.0f) {
                    continue;
                }
                values.push_back(value);
            }
        }
    }
    if (values.empty()) {
        return 0.0f;
    }
    const float clamped = std::clamp(percentile, 0.0f, 1.0f);
    const size_t index = static_cast<size_t>(
        std::floor(clamped * static_cast<float>(values.size() - 1)));
    std::nth_element(values.begin(),
                     values.begin() + static_cast<std::ptrdiff_t>(index),
                     values.end());
    return values[index];
}

static float computeStackMax(const std::vector<cv::Mat> &stack)
{
    float maxValue = 0.0f;
    bool foundValue = false;
    for (const auto &slice : stack) {
        if (slice.empty()) {
            continue;
        }
        double sliceMin = 0.0;
        double sliceMax = 0.0;
        cv::minMaxLoc(slice, &sliceMin, &sliceMax);
        if (!foundValue || static_cast<float>(sliceMax) > maxValue) {
            maxValue = static_cast<float>(sliceMax);
            foundValue = true;
        }
    }
    return foundValue ? maxValue : 0.0f;
}

static void normalizeStackToFrameScale(std::vector<cv::Mat> &stack,
                                       float lowReference,
                                       float highReference,
                                       float hardMax)
{
    if (hardMax > 0.0f) {
        highReference = std::min(highReference, hardMax);
    }
    const float denominator = highReference - lowReference;
    if (denominator <= 1e-6f) {
        for (auto &slice : stack) {
            if (!slice.empty()) {
                slice.setTo(0.0f);
            }
        }
        return;
    }
    for (auto &slice : stack) {
        if (slice.empty()) {
            continue;
        }
        if (hardMax > 0.0f) {
            cv::min(slice, hardMax, slice);
        }
        slice -= lowReference;
        slice *= (1.0f / denominator);
        cv::min(slice, 1.0f, slice);
        cv::max(slice, 0.0f, slice);
    }
}

static std::pair<float, float> normalizeStackToFrameIntensity(std::vector<cv::Mat> &stack,
                                                              const SimulationConfig &config)
{
    float lowReference = computeStackPercentile(
        stack,
        config.frame_intensity_scale_low_percentile,
        config.frame_intensity_percentile_exclude_zeros);
    float highReference = computeStackPercentile(
        stack,
        config.frame_intensity_scale_high_percentile,
        config.frame_intensity_percentile_exclude_zeros);
    if (config.frame_intensity_hard_max > 0.0f &&
        highReference > config.frame_intensity_hard_max) {
        highReference = config.frame_intensity_hard_max;
    }

    if (highReference <= lowReference + 1e-6f) {
        const float fallback = computeStackMax(stack);
        if (fallback > lowReference + 1e-6f) {
            highReference = fallback;
            if (config.frame_intensity_hard_max > 0.0f) {
                highReference = std::min(highReference, config.frame_intensity_hard_max);
            }
        } else {
            lowReference = 0.0f;
            highReference = 1.0f;
            if (config.frame_intensity_hard_max > 0.0f) {
                highReference = std::min(highReference, config.frame_intensity_hard_max);
            }
        }
    }

    normalizeStackToFrameScale(stack,
                               lowReference,
                               highReference,
                               config.frame_intensity_hard_max);
    return {lowReference, highReference};
}

static int clampEdgeLimit(int requestedLimit, int axisLength)
{
    if (axisLength <= 0) {
        return 0;
    }
    return std::clamp(requestedLimit, 0, axisLength);
}

static float computeEdgeBrightnessMean(const std::vector<cv::Mat> &stack,
                                       const SimulationConfig &config)
{
    double sum = 0.0;
    double count = 0.0;

    for (const auto &slice : stack) {
        if (slice.empty()) {
            continue;
        }
        CV_Assert(slice.type() == CV_32F);

        const int margin = std::max(1, config.edge_brightness_alignment_xy_margin);
        const int leftOffset = std::max(0, config.edge_brightness_alignment_left_offset);
        const int rightOffset = std::max(0, config.edge_brightness_alignment_right_offset);
        const int topOffset = std::max(0, config.edge_brightness_alignment_top_offset);
        const int bottomOffset = std::max(0, config.edge_brightness_alignment_bottom_offset);
        const int xInnerStart = clampEdgeLimit(leftOffset, slice.cols);
        const int xInnerEnd = clampEdgeLimit(leftOffset + margin, slice.cols);
        const int xOuterStart = clampEdgeLimit(slice.cols - rightOffset - margin, slice.cols);
        const int xOuterEnd = clampEdgeLimit(slice.cols - rightOffset, slice.cols);
        const int yInnerStart = clampEdgeLimit(topOffset, slice.rows);
        const int yInnerEnd = clampEdgeLimit(topOffset + margin, slice.rows);
        const int yOuterStart = clampEdgeLimit(slice.rows - bottomOffset - margin, slice.rows);
        const int yOuterEnd = clampEdgeLimit(slice.rows - bottomOffset, slice.rows);

        for (int y = 0; y < slice.rows; ++y) {
            const float *row = slice.ptr<float>(y);
            const bool inYBand =
                (y >= yInnerStart && y < yInnerEnd) ||
                (y >= yOuterStart && y < yOuterEnd);
            for (int x = 0; x < slice.cols; ++x) {
                const bool inXBand =
                    (x >= xInnerStart && x < xInnerEnd) ||
                    (x >= xOuterStart && x < xOuterEnd);
                if (!inYBand && !inXBand) {
                    continue;
                }
                const float value = row[x];
                if (!std::isfinite(value)) {
                    continue;
                }
                sum += value;
                count += 1.0;
            }
        }
    }

    return count > 0.0 ? static_cast<float>(sum / count) : 0.0f;
}

static void alignStackToEdgeBrightness(std::vector<cv::Mat> &stack,
                                       const SimulationConfig &config,
                                       float targetEdgeMean,
                                       const fs::path &framePath,
                                       std::ostream &log)
{
    if (!config.edge_brightness_alignment_enabled || stack.empty()) {
        return;
    }

    const float edgeMean = computeEdgeBrightnessMean(stack, config);
    const float maxShift = std::max(0.0f, config.edge_brightness_alignment_max_shift);
    const float shift = std::clamp(edgeMean - targetEdgeMean, -maxShift, maxShift);
    if (std::abs(shift) <= 1e-6f) {
        log << "[EdgeBrightnessAlignment] frame=" << framePath.filename().string()
            << " edge_mean=" << edgeMean
            << " target=" << targetEdgeMean
            << " shift=0"
            << " xy_margin=" << std::max(1, config.edge_brightness_alignment_xy_margin)
            << " offsets=("
            << std::max(0, config.edge_brightness_alignment_left_offset) << ","
            << std::max(0, config.edge_brightness_alignment_right_offset) << ","
            << std::max(0, config.edge_brightness_alignment_top_offset) << ","
            << std::max(0, config.edge_brightness_alignment_bottom_offset) << ")"
            << std::endl;
        return;
    }

    #pragma omp parallel for schedule(static)
    for (int sliceIndex = 0; sliceIndex < static_cast<int>(stack.size()); ++sliceIndex) {
        auto &slice = stack[static_cast<size_t>(sliceIndex)];
        if (slice.empty()) {
            continue;
        }
        slice -= shift;
        cv::min(slice, 1.0f, slice);
        cv::max(slice, 0.0f, slice);
    }

    const float alignedEdgeMean = computeEdgeBrightnessMean(stack, config);
    log << "[EdgeBrightnessAlignment] frame=" << framePath.filename().string()
        << " edge_mean=" << edgeMean
        << " target=" << targetEdgeMean
        << " shift=" << shift
        << " aligned_edge_mean=" << alignedEdgeMean
        << " xy_margin=" << std::max(1, config.edge_brightness_alignment_xy_margin)
        << " offsets=("
        << std::max(0, config.edge_brightness_alignment_left_offset) << ","
        << std::max(0, config.edge_brightness_alignment_right_offset) << ","
        << std::max(0, config.edge_brightness_alignment_top_offset) << ","
        << std::max(0, config.edge_brightness_alignment_bottom_offset) << ")"
        << " max_shift=" << maxShift
        << std::endl;
}

static void blackThresholdStackAfterAlignment(std::vector<cv::Mat> &stack,
                                              const SimulationConfig &config,
                                              const fs::path &framePath,
                                              std::ostream &log)
{
    const float threshold = std::max(0.0f, config.post_alignment_black_threshold);
    if (threshold <= 0.0f || stack.empty()) {
        return;
    }

    std::size_t changedCount = 0;
    std::size_t totalCount = 0;
    #pragma omp parallel for schedule(static) reduction(+:changedCount,totalCount)
    for (int sliceIndex = 0; sliceIndex < static_cast<int>(stack.size()); ++sliceIndex) {
        auto &slice = stack[static_cast<size_t>(sliceIndex)];
        if (slice.empty()) {
            continue;
        }
        CV_Assert(slice.type() == CV_32F);
        for (int y = 0; y < slice.rows; ++y) {
            float *row = slice.ptr<float>(y);
            for (int x = 0; x < slice.cols; ++x) {
                ++totalCount;
                if (row[x] < threshold) {
                    if (row[x] != 0.0f) {
                        ++changedCount;
                    }
                    row[x] = 0.0f;
                }
            }
        }
    }

    const double changedFraction = totalCount > 0
        ? static_cast<double>(changedCount) / static_cast<double>(totalCount)
        : 0.0;
    log << "[PostAlignmentBlackThreshold] frame=" << framePath.filename().string()
        << " threshold=" << threshold
        << " changed_fraction=" << changedFraction
        << std::endl;
}

static std::vector<cv::Mat> cloneMatStack(const std::vector<cv::Mat> &stack)
{
    std::vector<cv::Mat> cloned(stack.size());
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(stack.size()); ++i) {
        cloned[static_cast<std::size_t>(i)] = stack[static_cast<std::size_t>(i)].clone();
    }
    return cloned;
}

static float percentileValue(std::vector<float> values, float percentile)
{
    if (values.empty()) {
        return 0.0f;
    }
    percentile = std::clamp(percentile, 0.0f, 100.0f);
    const float rank = (percentile / 100.0f) * static_cast<float>(values.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(rank));
    const size_t hi = static_cast<size_t>(std::ceil(rank));
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(lo), values.end());
    const float loVal = values[lo];
    if (hi == lo) {
        return loVal;
    }
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(hi), values.end());
    const float hiVal = values[hi];
    return loVal + (hiVal - loVal) * (rank - static_cast<float>(lo));
}

static std::vector<cv::Mat> gaussianBlurStack3D(const std::vector<cv::Mat> &stack,
                                                float sigma)
{
    if (stack.empty() || sigma <= 0.0f) {
        return cloneMatStack(stack);
    }

    std::vector<cv::Mat> xyBlurred(stack.size());
    #pragma omp parallel for schedule(static)
    for (int z = 0; z < static_cast<int>(stack.size()); ++z) {
        cv::GaussianBlur(stack[static_cast<size_t>(z)],
                         xyBlurred[static_cast<size_t>(z)],
                         cv::Size(0, 0),
                         sigma,
                         sigma,
                         cv::BORDER_REPLICATE);
    }

    const int radius = std::max(1, static_cast<int>(std::ceil(3.0f * sigma)));
    std::vector<float> kernel(static_cast<size_t>(2 * radius + 1));
    float kernelSum = 0.0f;
    for (int dz = -radius; dz <= radius; ++dz) {
        const float w = std::exp(-0.5f * static_cast<float>(dz * dz) / (sigma * sigma));
        kernel[static_cast<size_t>(dz + radius)] = w;
        kernelSum += w;
    }
    for (float &w : kernel) {
        w /= std::max(kernelSum, 1e-12f);
    }

    std::vector<cv::Mat> blurred(stack.size());
    #pragma omp parallel for schedule(static)
    for (int z = 0; z < static_cast<int>(xyBlurred.size()); ++z) {
        blurred[static_cast<size_t>(z)] = cv::Mat::zeros(xyBlurred[static_cast<size_t>(z)].size(), CV_32F);
        for (int dz = -radius; dz <= radius; ++dz) {
            const int zz = std::clamp(z + dz, 0, static_cast<int>(xyBlurred.size()) - 1);
            blurred[static_cast<size_t>(z)] +=
                kernel[static_cast<size_t>(dz + radius)] * xyBlurred[static_cast<size_t>(zz)];
        }
    }
    return blurred;
}

static std::vector<cv::Mat> buildSignalMapStack(const std::vector<cv::Mat> &realFrame,
                                                const SimulationConfig &simulation,
                                                const fs::path &framePath,
                                                std::ostream &log)
{
    std::vector<cv::Mat> signalMap = cloneMatStack(realFrame);
    if (!simulation.signal_map_enabled || signalMap.empty()) {
        return {};
    }

    std::vector<float> nonzeroValues;
    for (const auto &slice : realFrame) {
        for (int y = 0; y < slice.rows; ++y) {
            const float *row = slice.ptr<float>(y);
            for (int x = 0; x < slice.cols; ++x) {
                if (row[x] > 0.0f) {
                    nonzeroValues.push_back(row[x]);
                }
            }
        }
    }

    const float threshold = nonzeroValues.empty()
        ? 0.0f
        : percentileValue(std::move(nonzeroValues), simulation.signal_map_bright_center_percentile);
    double targetSum = 0.0;
    std::size_t targetCount = 0;
    for (const auto &slice : realFrame) {
        for (int y = 0; y < slice.rows; ++y) {
            const float *row = slice.ptr<float>(y);
            for (int x = 0; x < slice.cols; ++x) {
                if (row[x] >= threshold) {
                    targetSum += row[x];
                    ++targetCount;
                }
            }
        }
    }

    const float eps = std::max(simulation.signal_map_epsilon, 1e-12f);
    const double targetMean = targetCount > 0
        ? targetSum / static_cast<double>(targetCount)
        : 0.0;
    const int iterations = std::max(0, simulation.signal_map_max_iterations);
    for (int iter = 0; iter < iterations; ++iter) {
        signalMap = gaussianBlurStack3D(signalMap, simulation.signal_map_blur_sigma);
        double blurredSum = 0.0;
        std::size_t blurredCount = 0;
        for (size_t z = 0; z < signalMap.size(); ++z) {
            const auto &slice = signalMap[z];
            const auto &referenceSlice = realFrame[z];
            for (int y = 0; y < slice.rows; ++y) {
                const float *row = slice.ptr<float>(y);
                const float *referenceRow = referenceSlice.ptr<float>(y);
                for (int x = 0; x < slice.cols; ++x) {
                    if (referenceRow[x] >= threshold) {
                        blurredSum += row[x];
                        ++blurredCount;
                    }
                }
            }
        }
        const double blurredMean = blurredCount > 0
            ? blurredSum / static_cast<double>(blurredCount)
            : 0.0;
        const float recoveryFactor = targetMean > eps
            ? static_cast<float>(targetMean / std::max(blurredMean, static_cast<double>(eps)))
            : 1.0f;
        #pragma omp parallel for schedule(static)
        for (int z = 0; z < static_cast<int>(signalMap.size()); ++z) {
            signalMap[static_cast<size_t>(z)] *= recoveryFactor;
        }
    }

    log << "[Signal Map] frame=" << framePath.filename().string()
        << " enabled=1"
        << " sigma=" << simulation.signal_map_blur_sigma
        << " iterations=" << iterations
        << " bright_center_percentile=" << simulation.signal_map_bright_center_percentile
        << " target_mean=" << targetMean
        << " threshold=" << threshold
        << std::endl;
    return signalMap;
}

struct BlackPercentileResult
{
    bool applied = false;
    float percentile = 0.0f;
    float cutoff = 0.0f;
    std::size_t nonzeroSampleCount = 0;
    double changedFraction = 0.0;
};

static BlackPercentileResult blackPercentileStackAfterAlignment(std::vector<cv::Mat> &stack,
                                                                const SimulationConfig &config,
                                                                const fs::path &framePath,
                                                                std::ostream &log,
                                                                float percentileOverride = -1.0f)
{
    BlackPercentileResult result;
    const float requestedPercentile = percentileOverride >= 0.0f
        ? percentileOverride
        : config.post_alignment_black_percentile;
    const float percentile = std::clamp(requestedPercentile, 0.0f, 1.0f);
    result.percentile = percentile;
    if (percentile <= 0.0f || stack.empty()) {
        return result;
    }

    std::vector<float> values;
    std::size_t totalCount = 0;
    for (const auto &slice : stack) {
        if (slice.empty()) {
            continue;
        }
        CV_Assert(slice.type() == CV_32F);
        totalCount += slice.total();
    }
    values.reserve(totalCount);

    for (const auto &slice : stack) {
        if (slice.empty()) {
            continue;
        }
        for (int y = 0; y < slice.rows; ++y) {
            const float *row = slice.ptr<float>(y);
            for (int x = 0; x < slice.cols; ++x) {
                const float value = row[x];
                if (std::isfinite(value) && value > 0.0f) {
                    values.push_back(value);
                }
            }
        }
    }

    if (values.empty()) {
        return result;
    }

    const std::size_t cutoffIndex = static_cast<std::size_t>(
        std::floor(percentile * static_cast<float>(values.size() - 1)));
    std::nth_element(values.begin(),
                     values.begin() + static_cast<std::ptrdiff_t>(cutoffIndex),
                     values.end());
    const float cutoff = values[cutoffIndex];
    result.cutoff = cutoff;
    result.nonzeroSampleCount = values.size();

    std::size_t changedCount = 0;
    std::size_t finiteCount = 0;
    #pragma omp parallel for schedule(static) reduction(+:changedCount,finiteCount)
    for (int sliceIndex = 0; sliceIndex < static_cast<int>(stack.size()); ++sliceIndex) {
        auto &slice = stack[static_cast<size_t>(sliceIndex)];
        if (slice.empty()) {
            continue;
        }
        CV_Assert(slice.type() == CV_32F);
        for (int y = 0; y < slice.rows; ++y) {
            float *row = slice.ptr<float>(y);
            for (int x = 0; x < slice.cols; ++x) {
                if (!std::isfinite(row[x])) {
                    continue;
                }
                ++finiteCount;
                if (row[x] <= cutoff) {
                    if (row[x] != 0.0f) {
                        ++changedCount;
                    }
                    row[x] = 0.0f;
                }
            }
        }
    }

    const double changedFraction = finiteCount > 0
        ? static_cast<double>(changedCount) / static_cast<double>(finiteCount)
        : 0.0;
    result.applied = true;
    result.changedFraction = changedFraction;
    log << "[PostAlignmentBlackPercentile] frame=" << framePath.filename().string()
        << " percentile=" << percentile
        << " cutoff=" << cutoff
        << " nonzero_sample_count=" << values.size()
        << " changed_fraction=" << changedFraction
        << std::endl;
    return result;
}

static int countSeparatedChunksInSizeRange(const std::vector<cv::Mat> &stack,
                                           const SimulationConfig &config,
                                           int stopAfterCount = -1)
{
    if (stack.empty()) {
        return 0;
    }

    const int depth = static_cast<int>(stack.size());
    const int rows = stack.front().rows;
    const int cols = stack.front().cols;
    if (depth <= 0 || rows <= 0 || cols <= 0) {
        return 0;
    }
    for (const auto &slice : stack) {
        if (slice.empty() || slice.rows != rows || slice.cols != cols) {
            return 0;
        }
        CV_Assert(slice.type() == CV_32F);
    }

    const int minSize = std::max(1, config.post_alignment_chunk_min_size);
    const int maxSize = std::max(minSize, config.post_alignment_chunk_max_size);
    const std::size_t planeSize = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    const std::size_t voxelCount = static_cast<std::size_t>(depth) * planeSize;
    std::vector<std::uint8_t> foreground(voxelCount, 0U);
    std::vector<std::uint8_t> visited(voxelCount, 0U);
    std::vector<std::size_t> pending;
    int matchingChunkCount = 0;

    auto indexOf = [&](int z, int y, int x) {
        return static_cast<std::size_t>(z) * planeSize +
               static_cast<std::size_t>(y) * static_cast<std::size_t>(cols) +
               static_cast<std::size_t>(x);
    };

    int detectorThreads = std::max(1, config.parallel_threads);
#ifdef _OPENMP
    detectorThreads = std::min(detectorThreads, std::max(1, omp_get_max_threads()));
    if (omp_in_parallel()) {
        detectorThreads = 1;
    }
#endif

    #pragma omp parallel for schedule(static) num_threads(detectorThreads)
    for (int z = 0; z < depth; ++z) {
        const cv::Mat &slice = stack[static_cast<std::size_t>(z)];
        for (int y = 0; y < rows; ++y) {
            const float *row = slice.ptr<float>(y);
            for (int x = 0; x < cols; ++x) {
                const float value = row[x];
                foreground[indexOf(z, y, x)] =
                    (std::isfinite(value) && value > 0.0f) ? 1U : 0U;
            }
        }
    }

    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                const std::size_t startIndex = indexOf(z, y, x);
                if (visited[startIndex] || !foreground[startIndex]) {
                    visited[startIndex] = 1U;
                    continue;
                }

                int minZ = z;
                int maxZ = z;
                int minY = y;
                int maxY = y;
                int minX = x;
                int maxX = x;
                visited[startIndex] = 1U;
                pending.clear();
                pending.push_back(startIndex);

                while (!pending.empty()) {
                    const std::size_t currentIndex = pending.back();
                    pending.pop_back();
                    const int currentZ = static_cast<int>(currentIndex / planeSize);
                    const std::size_t inPlane = currentIndex % planeSize;
                    const int currentY = static_cast<int>(inPlane / static_cast<std::size_t>(cols));
                    const int currentX = static_cast<int>(inPlane % static_cast<std::size_t>(cols));

                    minZ = std::min(minZ, currentZ);
                    maxZ = std::max(maxZ, currentZ);
                    minY = std::min(minY, currentY);
                    maxY = std::max(maxY, currentY);
                    minX = std::min(minX, currentX);
                    maxX = std::max(maxX, currentX);

                    for (int dz = -1; dz <= 1; ++dz) {
                        const int nz = currentZ + dz;
                        if (nz < 0 || nz >= depth) {
                            continue;
                        }
                        for (int dy = -1; dy <= 1; ++dy) {
                            const int ny = currentY + dy;
                            if (ny < 0 || ny >= rows) {
                                continue;
                            }
                            for (int dx = -1; dx <= 1; ++dx) {
                                if (dz == 0 && dy == 0 && dx == 0) {
                                    continue;
                                }
                                const int nx = currentX + dx;
                                if (nx < 0 || nx >= cols) {
                                    continue;
                                }

                                const std::size_t neighborIndex = indexOf(nz, ny, nx);
                                if (visited[neighborIndex]) {
                                    continue;
                                }
                                visited[neighborIndex] = 1U;
                                if (foreground[neighborIndex]) {
                                    pending.push_back(neighborIndex);
                                }
                            }
                        }
                    }
                }

                const int zSize = maxZ - minZ + 1;
                const int ySize = maxY - minY + 1;
                const int xSize = maxX - minX + 1;
                if (zSize >= minSize && zSize <= maxSize &&
                    ySize >= minSize && ySize <= maxSize &&
                    xSize >= minSize && xSize <= maxSize) {
                    ++matchingChunkCount;
                    if (stopAfterCount > 0 && matchingChunkCount >= stopAfterCount) {
                        return matchingChunkCount;
                    }
                }
            }
        }
    }

    return matchingChunkCount;
}

static void removeTinyIsolatedParticles(std::vector<cv::Mat> &stack,
                                        const BaseConfig &config,
                                        const fs::path &framePath,
                                        std::ostream &log)
{
    if (!config.simulation.post_alignment_tiny_particle_removal_enabled || stack.empty()) {
        return;
    }

    const int depth = static_cast<int>(stack.size());
    const int rows = stack.front().rows;
    const int cols = stack.front().cols;
    if (depth <= 0 || rows <= 0 || cols <= 0) {
        return;
    }
    for (const auto &slice : stack) {
        if (slice.empty() || slice.rows != rows || slice.cols != cols) {
            return;
        }
        CV_Assert(slice.type() == CV_32F);
    }

    const double minARadius = config.cell ? config.cell->minARadius : 5.0;
    const double minBRadius = config.cell
        ? (config.cell->minBRadius > 0.0 ? config.cell->minBRadius : config.cell->minARadius)
        : 5.0;
    const double minCRadius = config.cell ? config.cell->minCRadius : 5.0;
    const int minXSize = std::max(1, static_cast<int>(std::ceil(2.0 * minARadius)));
    const int minYSize = std::max(1, static_cast<int>(std::ceil(2.0 * minBRadius)));
    const int minZSize = std::max(1, static_cast<int>(std::ceil(2.0 * minCRadius)));

    const std::size_t planeSize = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    const std::size_t voxelCount = static_cast<std::size_t>(depth) * planeSize;
    std::vector<std::uint8_t> foreground(voxelCount, 0U);
    std::vector<std::uint8_t> visited(voxelCount, 0U);
    std::vector<std::size_t> pending;
    std::vector<std::size_t> componentVoxels;

    auto indexOf = [&](int z, int y, int x) {
        return static_cast<std::size_t>(z) * planeSize +
               static_cast<std::size_t>(y) * static_cast<std::size_t>(cols) +
               static_cast<std::size_t>(x);
    };

    int detectorThreads = std::max(1, config.simulation.parallel_threads);
#ifdef _OPENMP
    detectorThreads = std::min(detectorThreads, std::max(1, omp_get_max_threads()));
    if (omp_in_parallel()) {
        detectorThreads = 1;
    }
#endif

    #pragma omp parallel for schedule(static) num_threads(detectorThreads)
    for (int z = 0; z < depth; ++z) {
        const cv::Mat &slice = stack[static_cast<std::size_t>(z)];
        for (int y = 0; y < rows; ++y) {
            const float *row = slice.ptr<float>(y);
            for (int x = 0; x < cols; ++x) {
                const float value = row[x];
                foreground[indexOf(z, y, x)] =
                    (std::isfinite(value) && value > 0.0f) ? 1U : 0U;
            }
        }
    }

    int removedComponentCount = 0;
    std::size_t removedVoxelCount = 0;
    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < rows; ++y) {
            for (int x = 0; x < cols; ++x) {
                const std::size_t startIndex = indexOf(z, y, x);
                if (visited[startIndex] || !foreground[startIndex]) {
                    visited[startIndex] = 1U;
                    continue;
                }

                int minZ = z;
                int maxZ = z;
                int minY = y;
                int maxY = y;
                int minX = x;
                int maxX = x;
                visited[startIndex] = 1U;
                pending.clear();
                componentVoxels.clear();
                pending.push_back(startIndex);

                while (!pending.empty()) {
                    const std::size_t currentIndex = pending.back();
                    pending.pop_back();
                    componentVoxels.push_back(currentIndex);
                    const int currentZ = static_cast<int>(currentIndex / planeSize);
                    const std::size_t inPlane = currentIndex % planeSize;
                    const int currentY = static_cast<int>(inPlane / static_cast<std::size_t>(cols));
                    const int currentX = static_cast<int>(inPlane % static_cast<std::size_t>(cols));

                    minZ = std::min(minZ, currentZ);
                    maxZ = std::max(maxZ, currentZ);
                    minY = std::min(minY, currentY);
                    maxY = std::max(maxY, currentY);
                    minX = std::min(minX, currentX);
                    maxX = std::max(maxX, currentX);

                    for (int dz = -1; dz <= 1; ++dz) {
                        const int nz = currentZ + dz;
                        if (nz < 0 || nz >= depth) {
                            continue;
                        }
                        for (int dy = -1; dy <= 1; ++dy) {
                            const int ny = currentY + dy;
                            if (ny < 0 || ny >= rows) {
                                continue;
                            }
                            for (int dx = -1; dx <= 1; ++dx) {
                                if (dz == 0 && dy == 0 && dx == 0) {
                                    continue;
                                }
                                const int nx = currentX + dx;
                                if (nx < 0 || nx >= cols) {
                                    continue;
                                }

                                const std::size_t neighborIndex = indexOf(nz, ny, nx);
                                if (visited[neighborIndex]) {
                                    continue;
                                }
                                visited[neighborIndex] = 1U;
                                if (foreground[neighborIndex]) {
                                    pending.push_back(neighborIndex);
                                }
                            }
                        }
                    }
                }

                const int zSize = maxZ - minZ + 1;
                const int ySize = maxY - minY + 1;
                const int xSize = maxX - minX + 1;
                if (zSize < minZSize && ySize < minYSize && xSize < minXSize) {
                    ++removedComponentCount;
                    removedVoxelCount += componentVoxels.size();
                    for (const std::size_t voxelIndex : componentVoxels) {
                        const int voxelZ = static_cast<int>(voxelIndex / planeSize);
                        const std::size_t inPlane = voxelIndex % planeSize;
                        const int voxelY = static_cast<int>(inPlane / static_cast<std::size_t>(cols));
                        const int voxelX = static_cast<int>(inPlane % static_cast<std::size_t>(cols));
                        stack[static_cast<std::size_t>(voxelZ)].ptr<float>(voxelY)[voxelX] = 0.0f;
                    }
                }
            }
        }
    }

    log << "[PostAlignmentTinyParticleRemoval] frame=" << framePath.filename().string()
        << " removed_components=" << removedComponentCount
        << " removed_voxels=" << removedVoxelCount
        << " min_bbox=(" << minZSize << "," << minYSize << "," << minXSize << ")"
        << " detector_threads=" << detectorThreads
        << std::endl;
}

static void adaptBlackPercentileToChunkCount(std::vector<cv::Mat> &stack,
                                             const std::vector<cv::Mat> &unblackoffedStack,
                                             const SimulationConfig &config,
                                             const fs::path &framePath,
                                             std::ostream &log)
{
    if (!config.post_alignment_chunk_blackoff_enabled || stack.empty()) {
        return;
    }

    const int targetCount = std::max(0, config.post_alignment_chunk_target_count);
    const float percentileStep = std::max(0.0f, config.post_alignment_chunk_percentile_step);
    const float maxPercentile = std::clamp(config.post_alignment_chunk_max_percentile, 0.0f, 1.0f);
    const int nonImprovementPatience = std::max(0, config.post_alignment_chunk_non_improvement_patience);
    const int disableBelowCount = std::max(0, config.post_alignment_chunk_disable_below_count);
    float currentPercentile = std::clamp(config.post_alignment_black_percentile, 0.0f, maxPercentile);
    int chunkCount = countSeparatedChunksInSizeRange(stack, config);
    int bestChunkCount = chunkCount;
    int nonImprovementCount = 0;
    int iteration = 0;

    log << "[PostAlignmentChunkBlackoff] frame=" << framePath.filename().string()
        << " iteration=" << iteration
        << " percentile=" << currentPercentile
        << " chunk_count=" << chunkCount
        << " target_count=" << targetCount
        << " min_size=" << std::max(1, config.post_alignment_chunk_min_size)
        << " max_size=" << std::max(std::max(1, config.post_alignment_chunk_min_size),
                                    config.post_alignment_chunk_max_size)
        << " non_improvement_patience=" << nonImprovementPatience
        << " disable_below_count=" << disableBelowCount
        << " detector_threads=" << std::max(1, config.parallel_threads)
        << std::endl;

    if (chunkCount < disableBelowCount) {
        log << "[PostAlignmentChunkBlackoff] frame=" << framePath.filename().string()
            << " disabled_reason=chunk_count_below_configured_threshold"
            << " chunk_count=" << chunkCount
            << " disable_below_count=" << disableBelowCount
            << std::endl;
        return;
    }

    while (chunkCount > targetCount &&
           percentileStep > 0.0f &&
           currentPercentile + 1e-6f < maxPercentile &&
           nonImprovementCount < nonImprovementPatience) {
        currentPercentile = std::min(maxPercentile, currentPercentile + percentileStep);
        ++iteration;
        stack = cloneMatStack(unblackoffedStack);
        const BlackPercentileResult result =
            blackPercentileStackAfterAlignment(stack, config, framePath, log, currentPercentile);
        if (!result.applied) {
            break;
        }
        chunkCount = countSeparatedChunksInSizeRange(stack, config);
        if (chunkCount < bestChunkCount) {
            bestChunkCount = chunkCount;
            nonImprovementCount = 0;
        } else {
            ++nonImprovementCount;
        }
        log << "[PostAlignmentChunkBlackoff] frame=" << framePath.filename().string()
            << " iteration=" << iteration
            << " percentile=" << currentPercentile
            << " chunk_count=" << chunkCount
            << " target_count=" << targetCount
            << " best_chunk_count=" << bestChunkCount
            << " non_improvement_count=" << nonImprovementCount
            << " non_improvement_patience=" << nonImprovementPatience
            << std::endl;
    }
}

static void applyFinalPreprocessingBlur(std::vector<cv::Mat> &stack,
                                        const SimulationConfig &config,
                                        const fs::path &framePath,
                                        std::ostream &log)
{
    const float sigma = std::max(0.0f, config.post_alignment_final_blur_sigma);
    if (sigma <= 0.0f || stack.empty()) {
        return;
    }

    #pragma omp parallel for schedule(static)
    for (int sliceIndex = 0; sliceIndex < static_cast<int>(stack.size()); ++sliceIndex) {
        auto &slice = stack[static_cast<size_t>(sliceIndex)];
        if (slice.empty()) {
            continue;
        }

        cv::GaussianBlur(slice, slice, cv::Size(0, 0), sigma, sigma);
        cv::patchNaNs(slice, 0.0);
        cv::min(slice, 1.0f, slice);
        cv::max(slice, 0.0f, slice);
    }

    log << "[PostAlignmentFinalBlur] frame=" << framePath.filename().string()
        << " sigma=" << sigma
        << " slices=" << stack.size()
        << std::endl;
}

static void exportStackToSubdir(const std::vector<cv::Mat> &stack,
                                const fs::path &baseOutputDir,
                                const fs::path &subdir,
                                const fs::path &framePath,
                                bool exportPng,
                                bool exportTiff)
{
    const fs::path outputDir = baseOutputDir / subdir;

    if (exportPng) {
        const fs::path frameOutputDir = outputDir / framePath.stem();
        fs::create_directories(frameOutputDir);

        for (size_t i = 0; i < stack.size(); ++i) {
            if (stack[i].empty()) {
                continue;
            }

            cv::Mat outputImage;
            if (stack[i].depth() != CV_8U) {
                cv::Mat clipped = stack[i].clone();
                cv::patchNaNs(clipped, 0.0);
                cv::min(clipped, 1.0f, clipped);
                cv::max(clipped, 0.0f, clipped);
                clipped.convertTo(outputImage, CV_8U, 255.0);
            } else {
                outputImage = stack[i].clone();
            }

            const fs::path outputFile = frameOutputDir / (std::to_string(i) + ".png");
            cv::imwrite(outputFile.string(), outputImage);
        }
    }

    if (exportTiff) {
        fs::create_directories(outputDir);
        const fs::path outputFile = outputDir / (framePath.stem().string() + ".tif");
        writeNapariFriendlyTiffStack(outputFile.string(), stack);
    }
}

static void exportPreprocessedStack(const std::vector<cv::Mat> &stack,
                                    const fs::path &baseOutputDir,
                                    const fs::path &framePath,
                                    bool exportPng,
                                    bool exportTiff)
{
    exportStackToSubdir(stack, baseOutputDir, "preprocessed", framePath,
                        exportPng, exportTiff);
}

static bool useN2V2Preprocessing(const BaseConfig &config)
{
    return config.simulation.preprocessing_pipeline == "n2v2";
}

#if CELLUNIVERSE_HAS_N2V2_PREPROCESS
static n2v2::OutputDType parseN2V2OutputDType(const std::string &value)
{
    if (value == "uint8") {
        return n2v2::OutputDType::UInt8;
    }
    if (value == "uint16") {
        return n2v2::OutputDType::UInt16;
    }
    if (value == "float32") {
        return n2v2::OutputDType::Float32;
    }
    return n2v2::OutputDType::Preserve;
}

static n2v2::ContrastLimitMode parseN2V2ContrastLimitMode(const std::string &value)
{
    return value == "absolute"
        ? n2v2::ContrastLimitMode::Absolute
        : n2v2::ContrastLimitMode::Percentile;
}

static n2v2::ContrastScope parseN2V2ContrastScope(const std::string &value)
{
    return value == "slice"
        ? n2v2::ContrastScope::Slice
        : n2v2::ContrastScope::Stack;
}

static n2v2::PreprocessConfig makeN2V2Config(const SimulationConfig &simulation)
{
    n2v2::PreprocessConfig config;
    config.enableNetwork = simulation.n2v2_enable_network;
    config.modelPath = simulation.n2v2_model_path;
    config.device = simulation.n2v2_device;
    config.inferenceBatchSize = simulation.n2v2_inference_batch_size;
    config.tileSize = simulation.n2v2_tile_size;
    config.tileOverlap = simulation.n2v2_tile_overlap;
    config.scalePercentile = simulation.n2v2_scale_percentile;
    config.useNonzeroPixels = simulation.n2v2_use_nonzero_pixels;
    config.fallbackScale = simulation.n2v2_fallback_scale;
    config.careamicsMean = simulation.n2v2_careamics_mean;
    config.careamicsStd = simulation.n2v2_careamics_std;
    config.backgroundSubtraction.enabled = simulation.n2v2_background_subtraction_enabled;
    config.backgroundSubtraction.percentile = simulation.n2v2_background_subtraction_percentile;
    config.backgroundSubtraction.excludeZero = simulation.n2v2_background_subtraction_exclude_zero;
    config.backgroundSubtraction.clipMin = simulation.n2v2_background_subtraction_clip_min;
    config.contrast.enabled = simulation.n2v2_contrast_enabled;
    config.contrast.limitMode = parseN2V2ContrastLimitMode(simulation.n2v2_contrast_limit_mode);
    config.contrast.lowLimit = simulation.n2v2_contrast_low_limit;
    config.contrast.hasHighLimit = simulation.n2v2_contrast_has_high_limit;
    config.contrast.highLimit = simulation.n2v2_contrast_high_limit;
    config.contrast.lowPercentile = simulation.n2v2_contrast_low_percentile;
    config.contrast.highPercentile = simulation.n2v2_contrast_high_percentile;
    config.contrast.excludeZero = simulation.n2v2_contrast_exclude_zero;
    config.contrast.scope = parseN2V2ContrastScope(simulation.n2v2_contrast_scope);
    config.contrast.gamma = simulation.n2v2_contrast_gamma;
    config.contrast.preserveZeroPixels = simulation.n2v2_contrast_preserve_zero_pixels;
    config.output.dtype = parseN2V2OutputDType(simulation.n2v2_output_dtype);
    config.output.writeIntermediate = simulation.n2v2_output_write_intermediate;
    config.output.quantizeBeforeContrast = simulation.n2v2_output_quantize_before_contrast;
    return config;
}
#endif

static std::vector<cv::Mat> normalizeExternalOutputForRuntime(const std::vector<cv::Mat> &stack,
                                                              const fs::path &imagePath,
                                                              std::ostream &log)
{
    std::vector<cv::Mat> normalized(stack.size());
    double maxBeforeClamp = 0.0;
    for (std::size_t i = 0; i < stack.size(); ++i) {
        const cv::Mat &slice = stack[i];
        const int depth = slice.depth();
        double scale = 1.0;
        if (depth == CV_8U) {
            scale = 1.0 / 255.0;
        } else if (depth == CV_16U) {
            scale = 1.0 / 65535.0;
        }
        slice.convertTo(normalized[i], CV_32F, scale);
        double sliceMin = 0.0;
        double sliceMax = 0.0;
        cv::minMaxLoc(normalized[i], &sliceMin, &sliceMax);
        maxBeforeClamp = std::max(maxBeforeClamp, sliceMax);
        cv::patchNaNs(normalized[i], 0.0);
        cv::min(normalized[i], 1.0f, normalized[i]);
        cv::max(normalized[i], 0.0f, normalized[i]);
    }

    log << "[PreprocessingPipeline] frame=" << imagePath.filename().string()
        << " pipeline=n2v2"
        << " runtime_contract=CV_32F_0_1"
        << " max_before_clamp=" << maxBeforeClamp
        << std::endl;
    return normalized;
}

static std::vector<cv::Mat> runN2V2Preprocessing(const BaseConfig &config,
                                                 const std::string &outputPath,
                                                 const fs::path &imagePath,
                                                 N2V2Runtime &runtime,
                                                 std::ostream &log)
{
#if CELLUNIVERSE_HAS_N2V2_PREPROCESS
    log << "[PreprocessingPipeline] frame=" << imagePath.filename().string()
        << " pipeline=n2v2"
        << " legacy_pipeline=off"
        << " model=" << config.simulation.n2v2_model_path
        << " network_enabled=" << config.simulation.n2v2_enable_network
        << std::endl;

    std::vector<cv::Mat> rawStack = n2v2::loadTiffStack(imagePath);
    n2v2::PreprocessResult result =
        runtime.preprocessor.processStack(rawStack, imagePath, log);
    if (config.simulation.n2v2_output_write_intermediate) {
        exportStackToSubdir(result.intermediateStack,
                            fs::path(outputPath),
                            "n2v2_intermediate",
                            imagePath,
                            config.simulation.export_frame_png,
                            config.simulation.export_frame_tiff);
    }

    std::vector<cv::Mat> normalized =
        normalizeExternalOutputForRuntime(result.stack, imagePath, log);
    return ImageHandler::finalizePreprocessedStack(
        normalized,
        imagePath.string(),
        config,
        &log);
#else
    (void)outputPath;
    (void)runtime;
    log << "[PreprocessingPipeline] frame=" << imagePath.filename().string()
        << " pipeline=n2v2"
        << " legacy_pipeline=off"
        << " model=" << config.simulation.n2v2_model_path
        << " network_enabled=" << config.simulation.n2v2_enable_network
        << std::endl;
    throw std::runtime_error(
        "simulation.preprocessing_pipeline is n2v2, but this binary was built without LibTorch/N2V2 support. "
        "Install standalone LibTorch and reconfigure with -DCMAKE_PREFIX_PATH=/path/to/libtorch.");
#endif
}

static std::vector<cv::Mat> loadNormalizeAndPreprocess(const BaseConfig &config,
                                                       const std::string &outputPath,
                                                       const fs::path &imagePath,
                                                       N2V2Runtime *n2v2Runtime,
                                                       std::ostream &log)
{
    if (useN2V2Preprocessing(config)) {
        if (n2v2Runtime == nullptr) {
            throw std::runtime_error("Internal error: missing N2V2 runtime for n2v2 preprocessing");
        }
        return runN2V2Preprocessing(config, outputPath, imagePath, *n2v2Runtime, log);
    }

    log << "[PreprocessingPipeline] frame=" << imagePath.filename().string()
        << " pipeline=legacy"
        << " n2v2_pipeline=off"
        << std::endl;

    std::vector<cv::Mat> stack =
        ImageHandler::loadRawFrame(imagePath.string(), config, &log);

    if (config.simulation.frame_intensity_normalization_enabled) {
        const auto [lowReference, highReference] =
            normalizeStackToFrameIntensity(stack, config.simulation);
        log << "[Frame Intensity Scale] frame="
            << imagePath.filename().string()
            << " mean=" << computeStackMean(stack)
            << " low_ref=" << lowReference
            << " high_ref=" << highReference
            << " hard_max=" << config.simulation.frame_intensity_hard_max << '\n';
    } else {
        log << "[Frame Intensity Scale] frame="
            << imagePath.filename().string()
            << " enabled=0"
            << " mean=" << computeStackMean(stack)
            << '\n';
    }

    return ImageHandler::preprocessLoadedFrame(
        stack,
        imagePath.string(),
        config,
        &log);
}

static void runPostAlignmentCleanup(std::vector<cv::Mat> &stack,
                                    const BaseConfig &config,
                                    const fs::path &imagePath,
                                    std::ostream &log)
{
    blackThresholdStackAfterAlignment(stack, config.simulation, imagePath, log);
    const std::vector<cv::Mat> unblackoffedFrame = cloneMatStack(stack);
    blackPercentileStackAfterAlignment(stack, config.simulation, imagePath, log);
    adaptBlackPercentileToChunkCount(stack,
                                     unblackoffedFrame,
                                     config.simulation,
                                     imagePath,
                                     log);
    removeTinyIsolatedParticles(stack, config, imagePath, log);
    applyFinalPreprocessingBlur(stack, config.simulation, imagePath, log);
}

static void maybeExportPreprocessed(const std::vector<cv::Mat> &stack,
                                    const BaseConfig &config,
                                    const std::string &outputPath,
                                    const fs::path &imagePath)
{
    if (!config.simulation.export_preprocessed_images) {
        return;
    }
    exportPreprocessedStack(stack,
                            fs::path(outputPath),
                            imagePath,
                            config.simulation.export_frame_png,
                            config.simulation.export_frame_tiff);
}
} // namespace

PreprocessingHandler::PreprocessingHandler(const BaseConfig &config, std::string outputPath)
    : config_(config), outputPath_(std::move(outputPath))
{
}

PreprocessingHandler::~PreprocessingHandler() = default;

N2V2Runtime &PreprocessingHandler::n2v2Runtime(std::ostream &log) const
{
    if (!n2v2Runtime_) {
        log << "[N2V2] load_runtime"
            << " model=" << config_.simulation.n2v2_model_path
            << " device=" << config_.simulation.n2v2_device
            << " network_enabled=" << config_.simulation.n2v2_enable_network
            << std::endl;
#if CELLUNIVERSE_HAS_N2V2_PREPROCESS
        n2v2Runtime_ = std::make_unique<N2V2Runtime>(makeN2V2Config(config_.simulation));
#else
        n2v2Runtime_ = std::make_unique<N2V2Runtime>();
#endif
    }
    return *n2v2Runtime_;
}

std::vector<cv::Mat> PreprocessingHandler::probePreprocessedStack(const fs::path &imagePath,
                                                                  std::ostream &log) const
{
    N2V2Runtime *runtime = useN2V2Preprocessing(config_) ? &n2v2Runtime(log) : nullptr;
    return loadNormalizeAndPreprocess(config_, outputPath_, imagePath, runtime, log);
}

PreprocessedFrame PreprocessingHandler::preprocessFrame(const fs::path &imagePath,
                                                        std::optional<float> edgeTarget,
                                                        std::ostream &log) const
{
    PreprocessedFrame frame;
    N2V2Runtime *runtime = useN2V2Preprocessing(config_) ? &n2v2Runtime(log) : nullptr;
    frame.stack = loadNormalizeAndPreprocess(config_, outputPath_, imagePath, runtime, log);

    if (config_.simulation.edge_brightness_alignment_enabled) {
        frame.sampledBackground = computeEdgeBrightnessMean(frame.stack, config_.simulation);
        if (!edgeTarget.has_value()) {
            edgeTarget = frame.sampledBackground;
            log << "[EdgeBrightnessAlignment] target_initialized_from="
                << imagePath.filename().string()
                << " target=" << *edgeTarget << '\n';
        }
        alignStackToEdgeBrightness(frame.stack,
                                   config_.simulation,
                                   *edgeTarget,
                                   imagePath,
                                   log);
    }

    runPostAlignmentCleanup(frame.stack, config_, imagePath, log);
    frame.signalMap = buildSignalMapStack(
        frame.stack,
        config_.simulation,
        imagePath,
        log);
    maybeExportPreprocessed(frame.stack, config_, outputPath_, imagePath);
    return frame;
}

std::vector<PreprocessedFrame> PreprocessingHandler::preprocessBatch(const PathVec &imagePaths,
                                                                     bool retainStacks,
                                                                     std::ostream &log,
                                                                     float *resolvedEdgeTarget) const
{
    std::vector<PreprocessedFrame> frames(imagePaths.size());
    float minimumSampledBackground = std::numeric_limits<float>::infinity();
    const bool alignmentEnabled = config_.simulation.edge_brightness_alignment_enabled;
    const bool keepResults = retainStacks || config_.simulation.export_signal_debug_images;

    for (int frameIndex = 0; frameIndex < static_cast<int>(imagePaths.size()); ++frameIndex) {
        const fs::path &imagePath = imagePaths[static_cast<size_t>(frameIndex)];
        PreprocessedFrame item;
        N2V2Runtime *runtime = useN2V2Preprocessing(config_) ? &n2v2Runtime(log) : nullptr;
        item.stack = loadNormalizeAndPreprocess(config_, outputPath_, imagePath, runtime, log);

        if (!alignmentEnabled) {
            runPostAlignmentCleanup(item.stack, config_, imagePath, log);
            item.signalMap = buildSignalMapStack(
                item.stack,
                config_.simulation,
                imagePath,
                log);
            maybeExportPreprocessed(item.stack, config_, outputPath_, imagePath);

            if (!keepResults) {
                item.stack.clear();
                item.signalMap.clear();
            }
            frames[static_cast<size_t>(frameIndex)] = std::move(item);
            continue;
        }

        item.sampledBackground = computeEdgeBrightnessMean(item.stack, config_.simulation);
        minimumSampledBackground = std::min(minimumSampledBackground, item.sampledBackground);
        frames[static_cast<size_t>(frameIndex)] = std::move(item);

        log << "[EdgeBrightnessAlignment] frame="
            << imagePath.filename().string()
            << " sampled_background=" << frames[static_cast<size_t>(frameIndex)].sampledBackground
            << " pending_minimum_background=1"
            << std::endl;
    }

    if (!alignmentEnabled) {
        return frames;
    }

    if (!std::isfinite(minimumSampledBackground)) {
        minimumSampledBackground = 0.0f;
    }
    if (resolvedEdgeTarget) {
        *resolvedEdgeTarget = minimumSampledBackground;
    }

    log << "[EdgeBrightnessAlignment] batch_target=min_sampled_background"
        << " target=" << minimumSampledBackground
        << " frame_count=" << frames.size()
        << std::endl;

    for (int frameIndex = 0; frameIndex < static_cast<int>(frames.size()); ++frameIndex) {
        const fs::path &imagePath = imagePaths[static_cast<size_t>(frameIndex)];
        auto &item = frames[static_cast<size_t>(frameIndex)];

        alignStackToEdgeBrightness(item.stack,
                                   config_.simulation,
                                   minimumSampledBackground,
                                   imagePath,
                                   log);
        runPostAlignmentCleanup(item.stack, config_, imagePath, log);
        item.signalMap = buildSignalMapStack(
            item.stack,
            config_.simulation,
            imagePath,
            log);
        maybeExportPreprocessed(item.stack, config_, outputPath_, imagePath);

        if (!keepResults) {
            item.stack.clear();
            item.signalMap.clear();
        }
    }

    return frames;
}
