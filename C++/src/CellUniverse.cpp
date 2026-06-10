#include "../includes/CellUniverse.hpp"
#include "../includes/ImageHandler.hpp"
#include "../includes/CellLumen.hpp"
#include <set>
#include <cmath>
#include <algorithm>
#include <array>
#include <chrono>
#include <limits>
#include <numeric>
#include <utility>
#include <sstream>
#include <cstdint>
#include <queue>
#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <system_error>
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

namespace utils
{
    template <typename T>
    void printMat(const cv::Mat &mat)
    {
        // Check if the matrix is empty
        if (mat.empty())
        {
            std::cout << "The matrix is empty." << '\n';
            return;
        }

        // Iterate over matrix rows
        for (int i = 0; i < mat.rows; ++i)
        {
            for (int j = 0; j < mat.cols; ++j)
            {
                // Print each element. Use mat.at<T>(i,j) to access the element at [i,j] and cast it to the type T.
                // The type T should match the type of the elements in the matrix.
                std::cout << mat.at<T>(i, j) << " ";
            }
            std::cout << '\n'; // Newline for each row
        }
    }
}

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

static float squaredDistance(const cv::Point3f &a, const cv::Point3f &b)
{
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    const float dz = a.z - b.z;
    return dx * dx + dy * dy + dz * dz;
}

struct CandidateGraphRow {
    int frame = 0;
    std::string kind;
    std::string source;
    std::string parent;
    std::string candidateA;
    std::string candidateB;
    int selected = 0;
    double score = 0.0;
    double rawScore = 0.0;
    double imageGain = 0.0;
    double overlapCost = 0.0;
    double bridgeMetric = 0.0;
    float sep = 0.0f;
    float minSep = 0.0f;
    float maxSep = 0.0f;
    float midpointDist = 0.0f;
    float parentShape = 0.0f;
    float parentPersistencePenalty = 0.0f;
    float neighborClaimPenalty = 0.0f;
    float parentDistNear = 0.0f;
    float parentDistFar = 0.0f;
    float parentDistBalance = 0.0f;
    cv::Point3f d1;
    cv::Point3f d2;
    int voxA = 0;
    int voxB = 0;
    float signalA = 0.0f;
    float signalB = 0.0f;
    std::string note;
};

static std::string csvEscape(const std::string &value)
{
    const bool needsQuotes =
        value.find_first_of(",\"\n\r") != std::string::npos;
    if (!needsQuotes) {
        return value;
    }

    std::string escaped;
    escaped.reserve(value.size() + 2);
    escaped.push_back('"');
    for (char ch : value) {
        if (ch == '"') {
            escaped.push_back('"');
        }
        escaped.push_back(ch);
    }
    escaped.push_back('"');
    return escaped;
}

static std::ofstream openCandidateGraphLog(const std::string &outputPath,
                                           int absoluteFrame)
{
    const fs::path graphDir = fs::path(outputPath) / "candidate_graph";
    std::error_code ec;
    fs::create_directories(graphDir, ec);
    if (ec) {
        return {};
    }

    const fs::path csvPath =
        graphDir / ("frame_" + std::to_string(absoluteFrame) + "_candidates.csv");
    bool writeHeader = true;
    if (fs::exists(csvPath, ec) && !ec) {
        const auto size = fs::file_size(csvPath, ec);
        writeHeader = ec || size == 0;
    }

    std::ofstream out(csvPath, std::ios::app);
    if (out && writeHeader) {
        out << "frame,kind,source,parent,candidate_a,candidate_b,selected,"
            << "score,raw_score,image_gain,overlap_cost,bridge_metric,"
            << "sep,min_sep,max_sep,midpoint_dist,parent_shape,"
            << "parent_persistence_penalty,neighbor_claim_penalty,parent_dist_near,"
            << "parent_dist_far,parent_dist_balance,x1,y1,z1,x2,y2,z2,vox_a,vox_b,"
            << "signal_a,signal_b,note\n";
    }
    return out;
}

static void writeCandidateGraphRow(std::ostream &out,
                                   const CandidateGraphRow &row)
{
    if (!out) {
        return;
    }
    out << row.frame << ','
        << csvEscape(row.kind) << ','
        << csvEscape(row.source) << ','
        << csvEscape(row.parent) << ','
        << csvEscape(row.candidateA) << ','
        << csvEscape(row.candidateB) << ','
        << row.selected << ','
        << row.score << ','
        << row.rawScore << ','
        << row.imageGain << ','
        << row.overlapCost << ','
        << row.bridgeMetric << ','
        << row.sep << ','
        << row.minSep << ','
        << row.maxSep << ','
        << row.midpointDist << ','
        << row.parentShape << ','
        << row.parentPersistencePenalty << ','
        << row.neighborClaimPenalty << ','
        << row.parentDistNear << ','
        << row.parentDistFar << ','
        << row.parentDistBalance << ','
        << row.d1.x << ','
        << row.d1.y << ','
        << row.d1.z << ','
        << row.d2.x << ','
        << row.d2.y << ','
        << row.d2.z << ','
        << row.voxA << ','
        << row.voxB << ','
        << row.signalA << ','
        << row.signalB << ','
        << csvEscape(row.note) << '\n';
}

class ScopedStageTimer {
public:
    ScopedStageTimer(int frame, std::string stage)
        : frame(frame),
          stage(std::move(stage)),
          start(std::chrono::steady_clock::now())
    {
    }

    ~ScopedStageTimer()
    {
        finish();
    }

    double finish()
    {
        if (!active) {
            return elapsedSeconds;
        }
        const auto end = std::chrono::steady_clock::now();
        elapsedSeconds = std::chrono::duration<double>(end - start).count();
        active = false;
        std::cout << "[Stage Timing] frame " << frame
                  << " stage=" << stage
                  << " seconds=" << elapsedSeconds
                  << std::endl;
        return elapsedSeconds;
    }

private:
    int frame = 0;
    std::string stage;
    std::chrono::steady_clock::time_point start;
    bool active = true;
    double elapsedSeconds = 0.0;
};

static float computeMeanOfTopFraction(std::vector<float> values, float topFraction)
{
    if (values.empty()) {
        return 0.0f;
    }

    const float clampedFraction = std::clamp(topFraction, 0.0f, 1.0f);
    if (clampedFraction <= 0.0f) {
        return 0.0f;
    }

    const size_t selectedCount = std::max<size_t>(
        1, static_cast<size_t>(std::ceil(clampedFraction * static_cast<float>(values.size()))));
    const size_t thresholdIndex = values.size() - selectedCount;

    std::nth_element(values.begin(),
                     values.begin() + static_cast<std::ptrdiff_t>(thresholdIndex),
                     values.end());

    double sum = 0.0;
    for (size_t i = thresholdIndex; i < values.size(); ++i) {
        sum += values[i];
    }

    return static_cast<float>(sum / selectedCount);
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
        cv::IMWRITE_TIFF_COMPRESSION, 1 // COMPRESSION_NONE: easiest for TIFF readers.
    };

    if (!cv::imwritemulti(path, output, params)) {
        throw std::runtime_error("Failed to write TIFF stack: " + path);
    }
}

static void scaleStackBrightness(std::vector<cv::Mat> &stack, float scale)
{
    if (std::abs(scale - 1.0f) <= 1e-6f) {
        return;
    }

    for (auto &slice : stack) {
        if (slice.empty()) {
            continue;
        }
        slice *= scale;
        cv::min(slice, 1.0f, slice);
        cv::max(slice, 0.0f, slice);
    }
}

// === Helpers ported from yp_fix_mask_04172026 (commits 25c5923, b575246) ===
// Used by the per-frame percentile normalization step.

static float computeStackPercentile(const std::vector<cv::Mat> &stack,
                                    float percentile,
                                    bool excludeZeros = false)
{
    std::vector<float> values;
    size_t totalCount = 0;
    for (const auto &slice : stack) {
        if (!slice.empty()) totalCount += static_cast<size_t>(slice.total());
    }
    values.reserve(totalCount);
    for (const auto &slice : stack) {
        if (slice.empty()) continue;
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
    if (values.empty()) return 0.0f;
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
        if (slice.empty()) continue;
        double sliceMin = 0.0, sliceMax = 0.0;
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
    if (hardMax > 0.0f) highReference = std::min(highReference, hardMax);
    const float denominator = highReference - lowReference;
    if (denominator <= 1e-6f) {
        for (auto &slice : stack) {
            if (!slice.empty()) slice.setTo(0.0f);
        }
        return;
    }
    for (auto &slice : stack) {
        if (slice.empty()) continue;
        if (hardMax > 0.0f) cv::min(slice, hardMax, slice);
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
        if (slice.empty()) continue;
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

    const float edgeMean = computeEdgeBrightnessMean(
        stack, config);
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
        if (slice.empty()) continue;
        slice -= shift;
        cv::min(slice, 1.0f, slice);
        cv::max(slice, 0.0f, slice);
    }

    const float alignedEdgeMean = computeEdgeBrightnessMean(
        stack, config);
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
        if (slice.empty()) continue;
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
        if (slice.empty()) continue;
        CV_Assert(slice.type() == CV_32F);
        totalCount += slice.total();
    }
    values.reserve(totalCount);

    for (const auto &slice : stack) {
        if (slice.empty()) continue;
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
        if (slice.empty()) continue;
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

    int detectorThreads = std::max(1, config.post_alignment_chunk_detector_threads);
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
                        if (nz < 0 || nz >= depth) continue;
                        for (int dy = -1; dy <= 1; ++dy) {
                            const int ny = currentY + dy;
                            if (ny < 0 || ny >= rows) continue;
                            for (int dx = -1; dx <= 1; ++dx) {
                                if (dz == 0 && dy == 0 && dx == 0) continue;
                                const int nx = currentX + dx;
                                if (nx < 0 || nx >= cols) continue;

                                const std::size_t neighborIndex = indexOf(nz, ny, nx);
                                if (visited[neighborIndex]) continue;
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

    int detectorThreads = std::max(1, config.simulation.post_alignment_chunk_detector_threads);
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
                        if (nz < 0 || nz >= depth) continue;
                        for (int dy = -1; dy <= 1; ++dy) {
                            const int ny = currentY + dy;
                            if (ny < 0 || ny >= rows) continue;
                            for (int dx = -1; dx <= 1; ++dx) {
                                if (dz == 0 && dy == 0 && dx == 0) continue;
                                const int nx = currentX + dx;
                                if (nx < 0 || nx >= cols) continue;

                                const std::size_t neighborIndex = indexOf(nz, ny, nx);
                                if (visited[neighborIndex]) continue;
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
        << " detector_threads=" << std::max(1, config.post_alignment_chunk_detector_threads)
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

static std::vector<cv::Mat> makeEmptyDebugStackLike(const std::vector<cv::Mat> &realFrame)
{
    std::vector<cv::Mat> stack;
    stack.reserve(realFrame.size());
    for (const auto &slice : realFrame) {
        stack.emplace_back(cv::Mat::zeros(slice.size(), CV_32F));
    }
    return stack;
}

static void accumulateDebugCellPlacement(std::vector<cv::Mat> &stack,
                                         const Ellipsoid &cell,
                                         const SimulationConfig &simulationConfig,
                                         float brightness)
{
    if (stack.empty()) {
        return;
    }

    Ellipsoid debugCell = cell;
    debugCell.setBrightness(std::max(0.0f, brightness));
    const float maxR = std::max({debugCell.getARadius(),
                                 debugCell.getBRadius(),
                                 debugCell.getCRadius()});
    const int zMin = std::max(0, static_cast<int>(std::floor(debugCell.getZ() - maxR)));
    const int zMax = std::min(static_cast<int>(stack.size()) - 1,
                              static_cast<int>(std::ceil(debugCell.getZ() + maxR)));
    for (int z = zMin; z <= zMax; ++z) {
        cv::Mat temp = cv::Mat::zeros(stack[static_cast<size_t>(z)].size(), CV_32F);
        debugCell.draw(temp, simulationConfig, static_cast<float>(z));
        stack[static_cast<size_t>(z)] += temp;
    }
}

static std::vector<cv::Mat> makeCellCenterLabelDebugStack(
    const std::vector<cv::Mat> &realFrame,
    const std::vector<Ellipsoid> &cells,
    int cubeRadius)
{
    std::vector<cv::Mat> stack = makeEmptyDebugStackLike(realFrame);
    if (stack.empty()) {
        return stack;
    }

    const int radius = std::max(1, cubeRadius);
    const int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    const double fontScale = 0.45;
    const int thickness = 1;
    const float value = 1.0f;

    for (const auto &cell : cells) {
        const int cx = static_cast<int>(std::round(cell.getX()));
        const int cy = static_cast<int>(std::round(cell.getY()));
        const int cz = static_cast<int>(std::round(cell.getZ()));
        if (cz < 0 || cz >= static_cast<int>(stack.size())) {
            continue;
        }

        const int zMin = std::max(0, cz - radius);
        const int zMax = std::min(static_cast<int>(stack.size()) - 1, cz + radius);
        for (int z = zMin; z <= zMax; ++z) {
            cv::Mat &slice = stack[static_cast<size_t>(z)];
            if (slice.empty()) {
                continue;
            }
            const int xMin = std::max(0, cx - radius);
            const int xMax = std::min(slice.cols - 1, cx + radius);
            const int yMin = std::max(0, cy - radius);
            const int yMax = std::min(slice.rows - 1, cy + radius);
            for (int y = yMin; y <= yMax; ++y) {
                float *row = slice.ptr<float>(y);
                for (int x = xMin; x <= xMax; ++x) {
                    row[x] = value;
                }
            }
        }

        cv::Mat &centerSlice = stack[static_cast<size_t>(cz)];
        if (centerSlice.empty()) {
            continue;
        }
        int baseline = 0;
        const cv::Size textSize = cv::getTextSize(
            cell.getName(), fontFace, fontScale, thickness, &baseline);
        int textX = cx + radius + 4;
        if (textX + textSize.width >= centerSlice.cols) {
            textX = cx - radius - 4 - textSize.width;
        }
        textX = std::clamp(textX, 0, std::max(0, centerSlice.cols - textSize.width - 1));
        int textY = std::clamp(cy - radius - 4, textSize.height + 1,
                               std::max(textSize.height + 1, centerSlice.rows - 2));
        cv::putText(centerSlice, cell.getName(), cv::Point(textX, textY),
                    fontFace, fontScale, cv::Scalar(value), thickness, cv::LINE_AA);
    }

    return stack;
}

// === Signal-guided perturbation helpers (yp ffc1917) ===
//
// Detect bright clusters in the real frame and use them as teleport
// targets for cells that get stuck on local optima. Per-frame init only;
// the actual cell teleport happens in Frame::perturbCell when
// useSignalGuidance=true.

struct BrightBox {
    int ix = 0;
    int iy = 0;
    int iz = 0;
    cv::Point3f center{0.0f, 0.0f, 0.0f};
    float brightness = 0.0f;
    int voxels = 0;
};

static int chooseNearestDivisorSize(int axisLength, float targetSize)
{
    if (axisLength <= 1) return 1;
    const float clampedTarget = std::clamp(targetSize, 1.0f, static_cast<float>(axisLength));
    int bestDivisor = 1;
    float bestDistance = std::abs(static_cast<float>(bestDivisor) - clampedTarget);
    for (int d = 1; d <= axisLength; ++d) {
        if (axisLength % d != 0) continue;
        const float distance = std::abs(static_cast<float>(d) - clampedTarget);
        if (distance < bestDistance ||
            (std::abs(distance - bestDistance) <= 1e-6f && d > bestDivisor)) {
            bestDivisor = d;
            bestDistance = distance;
        }
    }
    return bestDivisor;
}

static constexpr std::array<std::array<int, 3>, 6> kSignalFaceNeighbors{{
    {{ 1,  0,  0}}, {{-1,  0,  0}},
    {{ 0,  1,  0}}, {{ 0, -1,  0}},
    {{ 0,  0,  1}}, {{ 0,  0, -1}}
}};

static std::vector<size_t> keepBrightestBoxFraction(const std::vector<BrightBox> &boxes,
                                                    std::vector<size_t> members,
                                                    float fraction)
{
    if (members.empty()) {
        return members;
    }

    const float clampedFraction = std::clamp(fraction, 0.0f, 1.0f);
    if (clampedFraction >= 1.0f - 1e-6f) {
        return members;
    }
    if (clampedFraction <= 0.0f) {
        return {};
    }

    std::sort(members.begin(), members.end(),
              [&](size_t lhs, size_t rhs) {
                  return boxes[lhs].brightness > boxes[rhs].brightness;
              });
    const size_t keepCount = std::max<size_t>(
        1, static_cast<size_t>(std::ceil(clampedFraction * static_cast<float>(members.size()))));
    members.resize(std::min(keepCount, members.size()));
    return members;
}

static std::vector<std::vector<size_t>> connectedSignalBoxComponents(
    const std::vector<BrightBox> &boxes,
    const std::vector<size_t> &members)
{
    std::map<std::tuple<int, int, int>, size_t> memberIndex;
    for (size_t local = 0; local < members.size(); ++local) {
        const BrightBox &box = boxes[members[local]];
        memberIndex[{box.ix, box.iy, box.iz}] = local;
    }

    std::vector<std::vector<size_t>> components;
    std::vector<char> visited(members.size(), 0);
    for (size_t seedLocal = 0; seedLocal < members.size(); ++seedLocal) {
        if (visited[seedLocal]) continue;

        std::vector<size_t> component;
        std::queue<size_t> queue;
        queue.push(seedLocal);
        visited[seedLocal] = 1;
        while (!queue.empty()) {
            const size_t local = queue.front();
            queue.pop();
            const size_t boxIdx = members[local];
            component.push_back(boxIdx);
            const BrightBox &box = boxes[boxIdx];

            for (const auto &offset : kSignalFaceNeighbors) {
                auto it = memberIndex.find({
                    box.ix + offset[0],
                    box.iy + offset[1],
                    box.iz + offset[2]});
                if (it == memberIndex.end()) continue;
                const size_t neighborLocal = it->second;
                if (visited[neighborLocal]) continue;
                visited[neighborLocal] = 1;
                queue.push(neighborLocal);
            }
        }
        components.push_back(std::move(component));
    }
    return components;
}

static Frame::SignalCenter makeSignalCenterFromBoxes(const std::vector<BrightBox> &boxes,
                                                     const std::vector<size_t> &members,
                                                     float backgroundValue)
{
    double weightSum = 0.0;
    double xSum = 0.0;
    double ySum = 0.0;
    double zSum = 0.0;
    double brightnessSum = 0.0;
    float minX = std::numeric_limits<float>::max();
    float minY = std::numeric_limits<float>::max();
    float minZ = std::numeric_limits<float>::max();
    float maxX = std::numeric_limits<float>::lowest();
    float maxY = std::numeric_limits<float>::lowest();
    float maxZ = std::numeric_limits<float>::lowest();
    int totalVoxels = 0;
    for (size_t boxIdx : members) {
        const BrightBox &box = boxes[boxIdx];
        const float weight = std::max(1e-6f, box.brightness - backgroundValue);
        weightSum += weight;
        xSum += weight * static_cast<double>(box.center.x);
        ySum += weight * static_cast<double>(box.center.y);
        zSum += weight * static_cast<double>(box.center.z);
        brightnessSum += static_cast<double>(box.brightness);
        minX = std::min(minX, box.center.x);
        minY = std::min(minY, box.center.y);
        minZ = std::min(minZ, box.center.z);
        maxX = std::max(maxX, box.center.x);
        maxY = std::max(maxY, box.center.y);
        maxZ = std::max(maxZ, box.center.z);
        totalVoxels += std::max(0, box.voxels);
    }

    Frame::SignalCenter center;
    if (!members.empty() && weightSum > 0.0) {
        center.position = cv::Point3f(
            static_cast<float>(xSum / weightSum),
            static_cast<float>(ySum / weightSum),
            static_cast<float>(zSum / weightSum));
        center.brightness = static_cast<float>(brightnessSum / static_cast<double>(members.size()));
        center.boxes = static_cast<int>(members.size());
        center.sizeVoxels = static_cast<float>(std::max(totalVoxels, center.boxes));
        const float boxSide = std::cbrt(
            center.sizeVoxels / static_cast<float>(std::max(1, center.boxes)));
        const float rx = 0.5f * (maxX - minX) + 0.5f * boxSide;
        const float ry = 0.5f * (maxY - minY) + 0.5f * boxSide;
        const float rz = 0.5f * (maxZ - minZ) + 0.5f * boxSide;
        const float minR = std::max(1e-3f, std::min({rx, ry, rz}));
        const float maxR = std::max({rx, ry, rz});
        center.shapeElongation = maxR / minR;
    }
    return center;
}

static void splitSignalClusterRecursive(const std::vector<BrightBox> &boxes,
                                        const std::vector<size_t> &members,
                                        float backgroundValue,
                                        float recursiveBrightFraction,
                                        int depth,
                                        int maxDepth,
                                        std::vector<Frame::SignalCenter> &outCenters)
{
    if (members.empty()) {
        return;
    }

    if (depth >= maxDepth || members.size() < 2) {
        outCenters.push_back(makeSignalCenterFromBoxes(boxes, members, backgroundValue));
        return;
    }

    std::vector<size_t> splitMembers =
        keepBrightestBoxFraction(boxes, members, recursiveBrightFraction);
    if (splitMembers.size() < 2) {
        outCenters.push_back(makeSignalCenterFromBoxes(boxes, members, backgroundValue));
        return;
    }

    std::vector<std::vector<size_t>> components =
        connectedSignalBoxComponents(boxes, splitMembers);
    if (components.size() < 2) {
        outCenters.push_back(makeSignalCenterFromBoxes(boxes, members, backgroundValue));
        return;
    }

    std::sort(components.begin(), components.end(),
              [&](const std::vector<size_t> &lhs, const std::vector<size_t> &rhs) {
                  const Frame::SignalCenter a = makeSignalCenterFromBoxes(boxes, lhs, backgroundValue);
                  const Frame::SignalCenter b = makeSignalCenterFromBoxes(boxes, rhs, backgroundValue);
                  return a.brightness > b.brightness;
              });

    for (const auto &component : components) {
        splitSignalClusterRecursive(boxes, component, backgroundValue,
                                    recursiveBrightFraction, depth + 1,
                                    maxDepth, outCenters);
    }
}

static std::vector<Frame::SignalCenter> localizeSignalCentersForFrame(
    const Frame &frame,
    const BaseConfig &config,
    int displayFrame,
    bool verbose = true)
{
    std::vector<Frame::SignalCenter> centers;
    const auto &realFrame = frame.getRealFrame();
    if (realFrame.empty() || !config.cell) return centers;

    const int sizeX = realFrame[0].cols;
    const int sizeY = realFrame[0].rows;
    const int sizeZ = static_cast<int>(realFrame.size());
    if (sizeX <= 0 || sizeY <= 0 || sizeZ <= 0) return centers;

    const float targetBoxSize =
        static_cast<float>(std::max(1, config.simulation.signal_guided_box_side_length));

    const int boxSizeX = chooseNearestDivisorSize(sizeX, targetBoxSize);
    const int boxSizeY = chooseNearestDivisorSize(sizeY, targetBoxSize);
    const int boxSizeZ = chooseNearestDivisorSize(sizeZ, targetBoxSize);
    const int gridX = std::max(1, sizeX / boxSizeX);
    const int gridY = std::max(1, sizeY / boxSizeY);
    const int gridZ = std::max(1, sizeZ / boxSizeZ);

    const float backgroundValue = frame.getBackgroundValue();
    const float minDelta = std::max(0.0f, config.simulation.signal_guided_min_box_brightness_delta);

    std::vector<BrightBox> boxes;
    boxes.reserve(static_cast<size_t>(gridX) * gridY * gridZ);
    for (int iz = 0; iz < gridZ; ++iz) {
        const int z0 = iz * boxSizeZ;
        const int z1 = z0 + boxSizeZ;
        for (int iy = 0; iy < gridY; ++iy) {
            const int y0 = iy * boxSizeY;
            const int y1 = y0 + boxSizeY;
            for (int ix = 0; ix < gridX; ++ix) {
                const int x0 = ix * boxSizeX;
                const int x1 = x0 + boxSizeX;
                double sum = 0.0;
                int voxels = 0;
                for (int z = z0; z < z1; ++z) {
                    for (int y = y0; y < y1; ++y) {
                        const float *row = realFrame[z].ptr<float>(y);
                        for (int x = x0; x < x1; ++x) {
                            sum += row[x];
                            ++voxels;
                        }
                    }
                }
                if (voxels <= 0) continue;
                const float meanBrightness = static_cast<float>(sum / static_cast<double>(voxels));
                if (meanBrightness <= backgroundValue + minDelta) continue;
                boxes.push_back({
                    ix, iy, iz,
                    cv::Point3f(
                        static_cast<float>(x0 + x1 - 1) * 0.5f,
                        static_cast<float>(y0 + y1 - 1) * 0.5f,
                        static_cast<float>(z0 + z1 - 1) * 0.5f),
                    meanBrightness, voxels
                });
            }
        }
    }

    std::sort(boxes.begin(), boxes.end(),
              [](const BrightBox &a, const BrightBox &b) { return a.brightness > b.brightness; });

    if (boxes.empty()) {
        if (!verbose) {
            return centers;
        }
        std::cout << "[Signal Centers] frame " << displayFrame
                  << " enabled=" << config.simulation.signal_guided_position_enabled
                  << " boxSize=(" << boxSizeX << "," << boxSizeY << "," << boxSizeZ << ")"
                  << " grid=(" << gridX << "," << gridY << "," << gridZ << ")"
                  << " background=" << backgroundValue
                  << " minDelta=" << minDelta
                  << " keptBoxes=0 clusters=0"
                  << " reason=no_boxes_above_background_delta"
                  << '\n';
        return centers;
    }

    std::vector<size_t> initialMembers(boxes.size());
    std::iota(initialMembers.begin(), initialMembers.end(), 0);
    const float initialBrightFraction =
        std::clamp(config.simulation.signal_guided_initial_bright_fraction, 0.0f, 1.0f);
    const float recursiveBrightFraction =
        std::clamp(config.simulation.signal_guided_recursive_bright_fraction, 0.0f, 1.0f);
    const int maxRecursiveDepth =
        std::max(0, config.simulation.signal_guided_max_recursive_depth);
    initialMembers = keepBrightestBoxFraction(boxes, std::move(initialMembers),
                                              initialBrightFraction);

    std::vector<std::vector<size_t>> initialComponents =
        connectedSignalBoxComponents(boxes, initialMembers);
    float maxCenterBrightness = backgroundValue;
    for (const auto &component : initialComponents) {
        splitSignalClusterRecursive(boxes, component, backgroundValue,
                                    recursiveBrightFraction, 0,
                                    maxRecursiveDepth, centers);
    }

    for (const auto &center : centers) {
        maxCenterBrightness = std::max(maxCenterBrightness, center.brightness);
    }

    for (auto &center : centers) {
        const float normalized = (maxCenterBrightness > backgroundValue + 1e-6f)
            ? std::clamp((center.brightness - backgroundValue) /
                         (maxCenterBrightness - backgroundValue), 0.0f, 1.0f)
            : 0.0f;
        center.sigmaScale = std::max(
            config.simulation.signal_guided_min_sigma_scale,
            1.0f - normalized * (1.0f - config.simulation.signal_guided_min_sigma_scale));
    }

    std::sort(centers.begin(), centers.end(),
              [](const Frame::SignalCenter &a, const Frame::SignalCenter &b) {
                  return a.brightness > b.brightness;
              });

    if (verbose) {
        std::cout << "[Signal Centers] frame " << displayFrame
                  << " enabled=" << config.simulation.signal_guided_position_enabled
                  << " boxSize=(" << boxSizeX << "," << boxSizeY << "," << boxSizeZ << ")"
                  << " grid=(" << gridX << "," << gridY << "," << gridZ << ")"
                  << " background=" << backgroundValue
                  << " minDelta=" << minDelta
                  << " keptBoxes=" << boxes.size()
                  << " initialBrightFraction=" << initialBrightFraction
                  << " initialBoxes=" << initialMembers.size()
                  << " recursiveBrightFraction=" << recursiveBrightFraction
                  << " maxRecursiveDepth=" << maxRecursiveDepth
                  << " clusters=" << centers.size() << '\n';
        for (size_t i = 0; i < centers.size(); ++i) {
            const auto &c = centers[i];
            std::cout << "  [Signal Center] idx=" << i
                      << " pos=(" << c.position.x << "," << c.position.y << "," << c.position.z << ")"
                      << " brightness=" << c.brightness
                      << " sigmaScale=" << c.sigmaScale
                      << " boxes=" << c.boxes << '\n';
        }
    }
    return centers;
}

struct SignalCenterRescueCandidate {
    bool valid = false;
    size_t centerIndex = 0;
    cv::Point3f position{0.0f, 0.0f, 0.0f};
    float score = std::numeric_limits<float>::max();
    float distance = 0.0f;
    float distanceUnits = 0.0f;
    float sizeRatio = 1.0f;
    float shapeRatio = 1.0f;
    float cellSizeVoxels = 0.0f;
    float centerSizeVoxels = 0.0f;
    float probeScore = std::numeric_limits<float>::max();
    float probeMean = 0.0f;
    float probeStddev = 0.0f;
    float probeMove = 0.0f;
};

static float safeRatio(float a, float b)
{
    a = std::max(a, 1e-3f);
    b = std::max(b, 1e-3f);
    return std::max(a, b) / std::min(a, b);
}

static float ellipsoidVolumeVoxels(const Ellipsoid &cell)
{
    return 4.18879020479f *
           std::max(1e-3f, cell.getARadius()) *
           std::max(1e-3f, cell.getBRadius()) *
           std::max(1e-3f, cell.getCRadius());
}

static std::vector<SignalCenterRescueCandidate> findSignalCenterRescueCandidates(
    const Ellipsoid &cell,
    const std::vector<Frame::SignalCenter> &centers,
    const std::vector<char> &claimedCenters,
    const SimulationConfig &simulation,
    const cv::Point3f *siblingHalfspaceMidpoint = nullptr,
    const cv::Point3f *siblingHalfspaceDirection = nullptr,
    int *rejectedBySiblingHalfspace = nullptr,
    float minDistanceUnitsOverride = -1.0f,
    float maxDistanceUnitsOverride = -1.0f)
{
    std::vector<SignalCenterRescueCandidate> candidates;
    const cv::Point3f oldPos(cell.getX(), cell.getY(), cell.getZ());
    const float bodyUnit = std::max(
        1e-3f, std::max({cell.getARadius(), cell.getBRadius(), cell.getCRadius()}));
    const float configuredMinDistanceUnits =
        std::max(0.0f, simulation.signal_center_rescue_min_distance_body_units);
    const float minDistanceUnits = minDistanceUnitsOverride >= 0.0f
        ? std::max(0.0f, minDistanceUnitsOverride)
        : configuredMinDistanceUnits;
    const float configuredMaxDistanceUnits =
        std::max(configuredMinDistanceUnits, simulation.signal_center_rescue_max_distance_body_units);
    const float maxDistanceUnits = maxDistanceUnitsOverride >= 0.0f
        ? std::max(minDistanceUnits, maxDistanceUnitsOverride)
        : configuredMaxDistanceUnits;
    const float maxSizeRatio =
        std::max(1.0f, simulation.signal_center_rescue_max_size_ratio);
    const float maxShapeRatio =
        std::max(1.0f, simulation.signal_center_rescue_max_shape_ratio);
    const float sizeWeight =
        std::max(0.0f, simulation.signal_center_rescue_size_weight);
    const float shapeWeight =
        std::max(0.0f, simulation.signal_center_rescue_shape_weight);
    const float maxHeuristicScore =
        std::max(0.0f, simulation.signal_center_rescue_max_heuristic_score);
    const float cellSize = ellipsoidVolumeVoxels(cell);
    const float cellShape = std::max(1.0f, cell.shapeElongation());

    for (size_t i = 0; i < centers.size(); ++i) {
        if (i < claimedCenters.size() && claimedCenters[i]) {
            continue;
        }
        const auto &center = centers[i];
        if (siblingHalfspaceMidpoint != nullptr &&
            siblingHalfspaceDirection != nullptr) {
            const float side =
                (center.position - *siblingHalfspaceMidpoint)
                    .dot(*siblingHalfspaceDirection);
            if (side < 0.0f) {
                if (rejectedBySiblingHalfspace != nullptr) {
                    ++(*rejectedBySiblingHalfspace);
                }
                continue;
            }
        }

        const float distance = cv::norm(center.position - oldPos);
        const float distanceUnits = distance / bodyUnit;
        if (distanceUnits < minDistanceUnits || distanceUnits > maxDistanceUnits) {
            continue;
        }

        const float centerSize = center.sizeVoxels > 0.0f
            ? center.sizeVoxels
            : static_cast<float>(std::max(1, center.boxes));
        const float sizeRatio = safeRatio(cellSize, centerSize);
        if (sizeRatio > maxSizeRatio) {
            continue;
        }

        const float centerShape = std::max(1.0f, center.shapeElongation);
        const float shapeRatio = safeRatio(cellShape, centerShape);
        if (shapeRatio > maxShapeRatio) {
            continue;
        }

        const float score =
            distanceUnits +
            sizeWeight * std::log(sizeRatio) +
            shapeWeight * std::log(shapeRatio);
        if (maxHeuristicScore > 0.0f && score > maxHeuristicScore) {
            continue;
        }
        SignalCenterRescueCandidate candidate;
        candidate.valid = true;
        candidate.centerIndex = i;
        candidate.position = center.position;
        candidate.score = score;
        candidate.distance = distance;
        candidate.distanceUnits = distanceUnits;
        candidate.sizeRatio = sizeRatio;
        candidate.shapeRatio = shapeRatio;
        candidate.cellSizeVoxels = cellSize;
        candidate.centerSizeVoxels = centerSize;
        candidates.push_back(candidate);
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const SignalCenterRescueCandidate &a,
                 const SignalCenterRescueCandidate &b) {
                  return a.score < b.score;
              });
    return candidates;
}

static SignalCenterRescueCandidate findSignalCenterRescueCandidate(
    const Ellipsoid &cell,
    const std::vector<Frame::SignalCenter> &centers,
    const std::vector<char> &claimedCenters,
    const SimulationConfig &simulation,
    const cv::Point3f *siblingHalfspaceMidpoint = nullptr,
    const cv::Point3f *siblingHalfspaceDirection = nullptr,
    int *rejectedBySiblingHalfspace = nullptr)
{
    auto candidates = findSignalCenterRescueCandidates(
        cell, centers, claimedCenters, simulation, siblingHalfspaceMidpoint,
        siblingHalfspaceDirection, rejectedBySiblingHalfspace);
    return candidates.empty() ? SignalCenterRescueCandidate{} : candidates.front();
}

static bool isSplitDaughterLineageName(const std::string &name)
{
    constexpr size_t kCellPrefixLength = 5; // "cell_"
    if (name.rfind("cell_", 0) != 0 || name.size() <= kCellPrefixLength + 1) {
        return false;
    }
    const char last = name.back();
    return last == '0' || last == '1';
}

static std::string splitDaughterSiblingName(const std::string &name)
{
    if (!isSplitDaughterLineageName(name)) {
        return {};
    }

    std::string sibling = name;
    sibling.back() = (name.back() == '0') ? '1' : '0';
    return sibling;
}

static int applySignalCenterRescueBeforePca(
    Frame &frame,
    const BaseConfig &config,
    const std::map<std::string, PreviousFrameSnapshot> &previousSnapshots,
    const std::unordered_map<std::string, int> &cellFirstSeenFrame,
    int displayFrame)
{
    const auto &simulation = config.simulation;
    if (!simulation.signal_center_rescue_enabled || frame.getRealFrame().empty()) {
        return 0;
    }

    const std::vector<Frame::SignalCenter> &centers = frame.getSignalCenters();
    if (centers.empty()) {
        std::cout << "[Signal Center Rescue] frame " << displayFrame
                  << " enabled=1 centers=0 action=skip\n";
        return 0;
    }

    const float background = frame.getBackgroundValue();
    const float bgDelta =
        std::max(0.0f, simulation.signal_center_rescue_background_delta);
    const float maxStddev =
        std::max(0.0f, simulation.signal_center_rescue_background_stddev_max);
    struct RescueProbe {
        size_t cellIndex = 0;
        std::string name;
        bool isTrash = false;
        std::pair<float, float> stats{0.0f, 0.0f};
        cv::Point3f oldPos{0.0f, 0.0f, 0.0f};
        std::string siblingName;
        bool siblingHalfspace = false;
        int rejectedBySiblingHalfspace = 0;
        bool recentSplitDaughterProbe = false;
        int splitDaughterAge = -1;
        SignalCenterRescueCandidate match;
    };

    std::vector<RescueProbe> probes;
    const bool globalAssignment =
        simulation.signal_center_rescue_global_assignment_enabled;

    for (size_t ci = 0; ci < frame.cells.size(); ++ci) {
        Ellipsoid &cell = frame.cells[ci];
        const std::string &name = cell.getName();
        auto snapIt = previousSnapshots.find(name);
        if (snapIt == previousSnapshots.end() || !snapIt->second.valid) {
            continue;
        }

        const auto stats = cell.measureBrightnessStats(frame.getRealFrame());
        const bool bodyLooksBackground =
            stats.first <= background + bgDelta &&
            stats.second <= maxStddev;

        const cv::Point3f oldPos(cell.getX(), cell.getY(), cell.getZ());
        std::string siblingName;
        cv::Point3f siblingHalfspaceMidpoint;
        cv::Point3f siblingHalfspaceDirection;
        const cv::Point3f *halfspaceMidpointPtr = nullptr;
        const cv::Point3f *halfspaceDirectionPtr = nullptr;
        int rejectedBySiblingHalfspace = 0;
        if (config.simulation.celluniverse2_enabled && !cell.isTrash()) {
            siblingName = splitDaughterSiblingName(name);
            auto siblingSnapIt = previousSnapshots.find(siblingName);
            if (!siblingName.empty() &&
                siblingSnapIt != previousSnapshots.end() &&
                siblingSnapIt->second.valid) {
                const cv::Point3f thisPreviousPos = snapIt->second.position;
                const cv::Point3f siblingPreviousPos =
                    siblingSnapIt->second.position;
                siblingHalfspaceDirection =
                    thisPreviousPos - siblingPreviousPos;
                if (cv::norm(siblingHalfspaceDirection) > 1e-3f) {
                    siblingHalfspaceMidpoint =
                        0.5f * (thisPreviousPos + siblingPreviousPos);
                    halfspaceMidpointPtr = &siblingHalfspaceMidpoint;
                    halfspaceDirectionPtr = &siblingHalfspaceDirection;
                }
            }
        }

        bool recentSplitDaughterProbe = false;
        int splitDaughterAge = -1;
        if (config.simulation.celluniverse2_enabled &&
            config.simulation.signal_center_rescue_split_daughter_probe_enabled &&
            !cell.isTrash() &&
            halfspaceMidpointPtr != nullptr &&
            halfspaceDirectionPtr != nullptr) {
            auto firstSeenIt = cellFirstSeenFrame.find(name);
            if (firstSeenIt != cellFirstSeenFrame.end()) {
                splitDaughterAge = displayFrame - firstSeenIt->second;
                recentSplitDaughterProbe =
                    splitDaughterAge >= 1 &&
                    splitDaughterAge <= std::max(
                        1,
                        config.simulation
                            .signal_center_rescue_split_daughter_probe_frames);
            }
        }

        if (!bodyLooksBackground && !recentSplitDaughterProbe) {
            continue;
        }

        const float minDistanceOverride =
            recentSplitDaughterProbe ? 0.0f : -1.0f;
        const float maxDistanceOverride =
            recentSplitDaughterProbe
                ? std::max(
                      0.0f,
                      config.simulation
                          .signal_center_rescue_split_daughter_max_distance_body_units)
                : -1.0f;

        std::vector<SignalCenterRescueCandidate> candidates =
            findSignalCenterRescueCandidates(cell,
                                             centers,
                                             std::vector<char>(centers.size(), 0),
                                             simulation,
                                             halfspaceMidpointPtr,
                                             halfspaceDirectionPtr,
                                             &rejectedBySiblingHalfspace,
                                             minDistanceOverride,
                                             maxDistanceOverride);
        if (candidates.empty()) {
            std::cout << "[Signal Center Rescue] frame " << displayFrame
                      << " cell=" << name
                      << " trigger="
                      << (recentSplitDaughterProbe ? "recent_split_daughter"
                                                   : "background_body")
                      << " splitDaughterAge=" << splitDaughterAge
                      << " bodyMean=" << stats.first
                      << " bodyStd=" << stats.second
                      << " background=" << background
                      << " sibling=" << siblingName
                      << " siblingHalfspace="
                      << (halfspaceMidpointPtr != nullptr ? 1 : 0)
                      << " rejectedBySiblingHalfspace="
                      << rejectedBySiblingHalfspace
                      << " action=no_compatible_center\n";
            continue;
        }

        const Ellipsoid originalCell = cell;
        auto buildRescueClaimSet = [&]() -> Frame::ClaimSet {
            Frame::ClaimSet others;
            for (const auto &other : frame.cells) {
                if (other.getName() == name) continue;
                others[other.getName()].push_back(cv::Point3f(
                    other.getX(), other.getY(), other.getZ()));
            }
            return others;
        };
        Frame::ClaimSet rescueOthers = buildRescueClaimSet();
        const int probeIters = std::min(
            4, std::max(0, config.cell ? config.cell->pcaShapeMaxIters : 0));
        const size_t maxProbeCandidates = std::min<size_t>(5, candidates.size());
        const float probeDistanceWeight =
            recentSplitDaughterProbe
                ? std::max(
                      0.0f,
                      config.simulation
                          .signal_center_rescue_split_daughter_probe_distance_weight)
                : 0.03f;
        const float oldBodyUnit = std::max(
            1e-3f,
            std::max({originalCell.getARadius(),
                      originalCell.getBRadius(),
                      originalCell.getCRadius()}));
        auto centerOccupiedByOtherCell = [&](const cv::Point3f &pos,
                                             std::string &occupantName,
                                             float &occupantDistance,
                                             float &occupancyLimit) -> bool {
            occupantDistance = std::numeric_limits<float>::infinity();
            occupancyLimit = 0.0f;
            occupantName.clear();
            for (size_t oi = 0; oi < frame.cells.size(); ++oi) {
                if (oi == ci || frame.cells[oi].isTrash()) {
                    continue;
                }
                const Ellipsoid &other = frame.cells[oi];
                const cv::Point3f otherPos(other.getX(), other.getY(), other.getZ());
                const float distance = static_cast<float>(cv::norm(pos - otherPos));
                const float otherBodyUnit = std::max(
                    1.0f,
                    std::max({other.getARadius(), other.getBRadius(), other.getCRadius()}));
                const float limit = std::max(4.0f, 0.25f * otherBodyUnit);
                if (distance < occupantDistance) {
                    occupantDistance = distance;
                    occupancyLimit = limit;
                    occupantName = other.getName();
                }
                if (distance <= limit) {
                    return true;
                }
            }
            return false;
        };

        SignalCenterRescueCandidate match;
        for (size_t probeIdx = 0; probeIdx < maxProbeCandidates; ++probeIdx) {
            auto candidate = candidates[probeIdx];
            std::string occupantName;
            float occupantDistance = 0.0f;
            float occupancyLimit = 0.0f;
            if (centerOccupiedByOtherCell(candidate.position,
                                          occupantName,
                                          occupantDistance,
                                          occupancyLimit)) {
                std::cout << "[Signal Center Rescue Candidate Reject] frame "
                          << displayFrame
                          << " cell=" << name
                          << " rank=" << probeIdx
                          << " centerIdx=" << candidate.centerIndex
                          << " reason=center_occupied_by_live_cell"
                          << " occupant=" << occupantName
                          << " distance=" << occupantDistance
                          << " limit=" << occupancyLimit
                          << std::endl;
                continue;
            }
            cell = originalCell;
            cell.setPosition(candidate.position.x,
                             candidate.position.y,
                             candidate.position.z);
            if (probeIters > 0 && config.cell) {
                std::ostringstream probeLog;
                frame.calibrateCellShapeViaPca(
                    ci,
                    rescueOthers,
                    probeIters,
                    config.cell->pcaShapeRadiusScale,
                    config.cell->pcaShapeMinPixels,
                    config.cell->pcaShapeMaskScale,
                    config.cell->pcaShapeConvergeRadius,
                    config.cell->pcaShapeConvergeAngleDeg,
                    config.cell->pcaShapeUpdatePosition,
                    config.cell->pcaShapeMaxPosShiftFraction,
                    originalCell.getARadius(),
                    originalCell.getBRadius(),
                    originalCell.getCRadius(),
                    &probeLog);
            }
            const auto probeStats = cell.measureBrightnessStats(frame.getRealFrame());
            const cv::Point3f probePos(cell.getX(), cell.getY(), cell.getZ());
            if (recentSplitDaughterProbe &&
                halfspaceMidpointPtr != nullptr &&
                halfspaceDirectionPtr != nullptr) {
                const float probeSide =
                    (probePos - *halfspaceMidpointPtr)
                        .dot(*halfspaceDirectionPtr);
                if (probeSide < 0.0f) {
                    std::cout << "[Signal Center Rescue Candidate Reject] frame "
                              << displayFrame
                              << " cell=" << name
                              << " rank=" << probeIdx
                              << " centerIdx=" << candidate.centerIndex
                              << " reason=probe_crossed_sibling_halfspace"
                              << " splitDaughterAge=" << splitDaughterAge
                              << " probeSide=" << probeSide
                              << std::endl;
                    continue;
                }
            }
            candidate.probeMean = probeStats.first;
            candidate.probeStddev = probeStats.second;
            candidate.probeMove = static_cast<float>(
                cv::norm(probePos - candidate.position));
            candidate.probeScore =
                -candidate.probeMean +
                0.20f * candidate.probeStddev +
                0.04f * candidate.probeMove +
                probeDistanceWeight * candidate.score;
            if (recentSplitDaughterProbe) {
                const float probeMoveUnits = candidate.probeMove / oldBodyUnit;
                candidate.probeScore +=
                    0.08f * probeMoveUnits +
                    probeDistanceWeight * candidate.distanceUnits;
            }
            std::cout << "[Signal Center Rescue Candidate] frame "
                      << displayFrame
                      << " cell=" << name
                      << " trigger="
                      << (recentSplitDaughterProbe ? "recent_split_daughter"
                                                   : "background_body")
                      << " splitDaughterAge=" << splitDaughterAge
                      << " rank=" << probeIdx
                      << " centerIdx=" << candidate.centerIndex
                      << " center=(" << candidate.position.x << ","
                      << candidate.position.y << ","
                      << candidate.position.z << ")"
                      << " heuristicScore=" << candidate.score
                      << " probeMean=" << candidate.probeMean
                      << " probeStd=" << candidate.probeStddev
                      << " probeMove=" << candidate.probeMove
                      << " probeScore=" << candidate.probeScore
                      << std::endl;
            if (!match.valid || candidate.probeScore < match.probeScore) {
                match = candidate;
            }
        }
        cell = originalCell;

        if (!match.valid) {
            std::cout << "[Signal Center Rescue] frame " << displayFrame
                      << " cell=" << name
                      << " bodyMean=" << stats.first
                      << " bodyStd=" << stats.second
                      << " background=" << background
                      << " sibling=" << siblingName
                      << " siblingHalfspace="
                      << (halfspaceMidpointPtr != nullptr ? 1 : 0)
                      << " rejectedBySiblingHalfspace="
                      << rejectedBySiblingHalfspace
                      << " action=no_probe_winner\n";
            continue;
        }

        RescueProbe probe;
        probe.cellIndex = ci;
        probe.name = name;
        probe.isTrash = cell.isTrash();
        probe.stats = stats;
        probe.oldPos = oldPos;
        probe.siblingName = siblingName;
        probe.siblingHalfspace = (halfspaceMidpointPtr != nullptr);
        probe.rejectedBySiblingHalfspace = rejectedBySiblingHalfspace;
        probe.recentSplitDaughterProbe = recentSplitDaughterProbe;
        probe.splitDaughterAge = splitDaughterAge;
        probe.match = match;
        probes.push_back(probe);
    }

    if (globalAssignment) {
        std::sort(probes.begin(), probes.end(),
                  [](const RescueProbe &a, const RescueProbe &b) {
                      if (a.match.centerIndex == b.match.centerIndex &&
                          a.isTrash != b.isTrash) {
                          return a.isTrash;
                      }
                      if (a.match.probeScore != b.match.probeScore) {
                          return a.match.probeScore < b.match.probeScore;
                      }
                      return a.match.score < b.match.score;
                  });
    }

    std::vector<char> trashPreferredCenters(centers.size(), 0);
    if (globalAssignment) {
        for (const RescueProbe &probe : probes) {
            if (probe.isTrash && probe.match.centerIndex < trashPreferredCenters.size()) {
                trashPreferredCenters[probe.match.centerIndex] = 1;
            }
        }
    }

    std::vector<char> claimedCenters(centers.size(), 0);
    std::vector<char> claimedCells(frame.cells.size(), 0);
    int rescued = 0;
    for (const RescueProbe &probe : probes) {
        if (probe.cellIndex >= frame.cells.size()) {
            continue;
        }
        if (probe.cellIndex < claimedCells.size() && claimedCells[probe.cellIndex]) {
            continue;
        }
        if (!probe.isTrash &&
            probe.match.centerIndex < trashPreferredCenters.size() &&
            trashPreferredCenters[probe.match.centerIndex]) {
            std::cout << "[Signal Center Rescue] frame " << displayFrame
                      << " cell=" << probe.name
                      << " centerIdx=" << probe.match.centerIndex
                      << " probeScore=" << probe.match.probeScore
                      << " action=center_reserved_for_trash_probe\n";
            continue;
        }
        if (probe.match.centerIndex < claimedCenters.size() &&
            claimedCenters[probe.match.centerIndex]) {
            std::cout << "[Signal Center Rescue] frame " << displayFrame
                      << " cell=" << probe.name
                      << " centerIdx=" << probe.match.centerIndex
                      << " probeScore=" << probe.match.probeScore
                      << " action=center_claimed_by_better_probe\n";
            continue;
        }

        claimedCells[probe.cellIndex] = 1;
        if (probe.match.centerIndex < claimedCenters.size()) {
            claimedCenters[probe.match.centerIndex] = 1;
        }
        Ellipsoid &cell = frame.cells[probe.cellIndex];
        cell.setPosition(probe.match.position.x,
                         probe.match.position.y,
                         probe.match.position.z);
        ++rescued;
        std::cout << "[Signal Center Rescue] frame " << displayFrame
                  << " cell=" << probe.name
                  << " kind=" << (probe.isTrash ? "trash" : "cell")
                  << " trigger="
                  << (probe.recentSplitDaughterProbe
                          ? "recent_split_daughter"
                          : "background_body")
                  << " splitDaughterAge=" << probe.splitDaughterAge
                  << " bodyMean=" << probe.stats.first
                  << " bodyStd=" << probe.stats.second
                  << " background=" << background
                  << " from=(" << probe.oldPos.x << "," << probe.oldPos.y << "," << probe.oldPos.z << ")"
                  << " to=(" << probe.match.position.x << "," << probe.match.position.y << "," << probe.match.position.z << ")"
                  << " centerIdx=" << probe.match.centerIndex
                  << " distance=" << probe.match.distance
                  << " bodyUnits=" << probe.match.distanceUnits
                  << " sizeRatio=" << probe.match.sizeRatio
                  << " shapeRatio=" << probe.match.shapeRatio
                  << " score=" << probe.match.score
                  << " probeScore=" << probe.match.probeScore
                  << " probeMean=" << probe.match.probeMean
                  << " probeStd=" << probe.match.probeStddev
                  << " probeMove=" << probe.match.probeMove
                  << " sibling=" << probe.siblingName
                  << " siblingHalfspace="
                  << (probe.siblingHalfspace ? 1 : 0)
                  << " rejectedBySiblingHalfspace="
                  << probe.rejectedBySiblingHalfspace
                  << " globalAssignment=" << (globalAssignment ? 1 : 0)
                  << " action=move_before_pca\n";
    }

    if (rescued > 0) {
        std::cout << "[Signal Center Rescue] frame " << displayFrame
                  << " rescued=" << rescued
                  << " centers=" << centers.size() << '\n';
    }
    return rescued;
}

struct SignalCenterSplitProposalResult {
    bool valid = false;
    BridgeSplitProposal proposal;
    size_t centerIndexA = 0;
    size_t centerIndexB = 0;
    int centersInside = 0;
    float parentShape = 1.0f;
    float separation = 0.0f;
    float separationRatio = 0.0f;
    float midpointDistance = 0.0f;
    float axisAlignment = 0.0f;
    int candidatePairs = 0;
    int rejectedBySeparation = 0;
    int rejectedByMidpoint = 0;
    int rejectedByAxis = 0;
    float bestRejectedAxisAlignment = 0.0f;
    float bestRejectedSeparationRatio = 0.0f;
    float bestRejectedMidpointDistance = 0.0f;
    float score = std::numeric_limits<float>::max();
    std::string rejectReason = "not_evaluated";
};

static bool pointInsideBbox(const cv::Point3f &point, const BoundingBox3D &bbox)
{
    return point.x >= static_cast<float>(bbox.xMin) &&
           point.x <= static_cast<float>(bbox.xMax) &&
           point.y >= static_cast<float>(bbox.yMin) &&
           point.y <= static_cast<float>(bbox.yMax) &&
           point.z >= static_cast<float>(bbox.zMin) &&
           point.z <= static_cast<float>(bbox.zMax);
}

struct RodTipSphereSplitProposalResult {
    bool valid = false;
    BridgeSplitProposal proposal;
    float parentShape = 1.0f;
    float longRadius = 0.0f;
    float midRadius = 0.0f;
    float shortRadius = 0.0f;
    float sphereRadius = 0.0f;
    float tipDistance = 0.0f;
    float longMidRatio = 0.0f;
    float midShortRatio = 0.0f;
    std::string rejectReason = "not_evaluated";
};

static RodTipSphereSplitProposalResult makeRodTipSphereSplitProposal(
    const Ellipsoid &parent,
    const PreviousFrameSnapshot &snapshot,
    const BaseConfig &config)
{
    RodTipSphereSplitProposalResult result;
    if (parent.isTrash() || !snapshot.valid) {
        result.rejectReason = "bad_parent_or_snapshot";
        return result;
    }

    const float radii[3] = {
        parent.getARadius(), parent.getBRadius(), parent.getCRadius()
    };
    int longIdx = 0;
    int shortIdx = 0;
    for (int i = 1; i < 3; ++i) {
        if (radii[i] > radii[longIdx]) longIdx = i;
        if (radii[i] < radii[shortIdx]) shortIdx = i;
    }
    int midIdx = 0;
    for (int i = 0; i < 3; ++i) {
        if (i != longIdx && i != shortIdx) {
            midIdx = i;
            break;
        }
    }

    const float longR = std::max(1e-3f, radii[longIdx]);
    const float midR = std::max(1e-3f, radii[midIdx]);
    const float shortR = std::max(1e-3f, radii[shortIdx]);
    const float elong = longR / shortR;
    result.parentShape = elong;
    result.longRadius = longR;
    result.midRadius = midR;
    result.shortRadius = shortR;
    if (elong < config.prob.pca_bridge_elongation_ratio) {
        result.rejectReason = "not_rod_like";
        return result;
    }

    const float minLongMidRatio = std::max(0.0f, config.prob.pca_bridge_min_long_mid_ratio);
    const float maxMidShortRatio = std::max(0.0f, config.prob.pca_bridge_max_mid_short_ratio);
    result.longMidRatio = longR / midR;
    result.midShortRatio = midR / shortR;
    const bool longAxisDistinct =
        minLongMidRatio <= 0.0f || result.longMidRatio >= minLongMidRatio;
    const bool shortAxesSimilar =
        maxMidShortRatio <= 0.0f || result.midShortRatio <= maxMidShortRatio;
    if (!longAxisDistinct || !shortAxesSimilar) {
        result.rejectReason = "triaxial_shape";
        return result;
    }

    std::array<double, 9> R_T;
    parent.generateInverseRotationMatrix(R_T);
    const int base = 3 * longIdx;
    cv::Point3f longAxis(
        static_cast<float>(R_T[base]),
        static_cast<float>(R_T[base + 1]),
        static_cast<float>(R_T[base + 2]));
    const float axisNorm = static_cast<float>(cv::norm(longAxis));
    if (axisNorm <= 1e-6f) {
        result.rejectReason = "bad_long_axis";
        return result;
    }
    longAxis *= (1.0f / axisNorm);

    cv::Point3f referenceAxis = snapshot.splitAxisDir;
    const float refAxisNorm = static_cast<float>(cv::norm(referenceAxis));
    if (refAxisNorm > 1e-6f) {
        referenceAxis *= (1.0f / refAxisNorm);
        if (longAxis.dot(referenceAxis) < 0.0f) {
            longAxis *= -1.0f;
        }
    }

    const cv::Point3f center(parent.getX(), parent.getY(), parent.getZ());
    const float sphereRadius = 0.5f * (midR + shortR);
    if (sphereRadius <= 1e-3f) {
        result.rejectReason = "bad_sphere_radius";
        return result;
    }

    // Place one daughter at the rod center and one same-radius sphere just
    // outside one rod tip. Try both tip sides because no two-peak signal was
    // available to identify which side contains the off-rod daughter.
    const float tipDistance = longR + sphereRadius;
    result.proposal.d1Pos = center;
    result.proposal.d2Pos = center + tipDistance * longAxis;
    result.proposal.hasAlternateSeedPair = true;
    result.proposal.altD1Pos = center;
    result.proposal.altD2Pos = center - tipDistance * longAxis;
    result.proposal.elongation = elong;
    result.proposal.parentShapeElongation = elong;
    result.proposal.elongatedParentRescued = true;
    result.proposal.gapStartBin = -8;
    result.proposal.gapEndBin = -8;
    result.proposal.leftPixelCount = 1;
    result.proposal.rightPixelCount = 1;
    result.proposal.daughterSphereRadius = sphereRadius;
    result.sphereRadius = sphereRadius;
    result.tipDistance = tipDistance;
    result.valid = true;
    result.rejectReason = "ok";
    return result;
}

static SignalCenterSplitProposalResult findSignalCenterSplitProposal(
    const Frame &frame,
    size_t cellIndex,
    const PreviousFrameSnapshot &snapshot,
    const BaseConfig &config)
{
    SignalCenterSplitProposalResult best;
    if (!config.simulation.celluniverse2_enabled ||
        !config.prob.signal_center_split_enabled ||
        cellIndex >= frame.cells.size()) {
        return best;
    }

    const Ellipsoid &parent = frame.cells[cellIndex];
    if (parent.isTrash() || !snapshot.valid) {
        return best;
    }

    const std::vector<Frame::SignalCenter> &centers = frame.getSignalCenters();
    if (centers.size() < 2) {
        best.rejectReason = "too_few_global_centers";
        return best;
    }

    const float parentShape =
        std::max(parent.shapeElongation(), snapshot.shapeElongation);
    best.parentShape = parentShape;
    if (parentShape < config.prob.signal_center_split_min_parent_elongation) {
        best.rejectReason = "not_flat_enough";
        return best;
    }

    cv::Point3f splitAxis = snapshot.splitAxisDir;
    float splitAxisNorm = static_cast<float>(cv::norm(splitAxis));
    if (splitAxisNorm <= 1e-6f) {
        float splitAxisLength = 0.0f;
        parent.worldSplitAxis(splitAxis, splitAxisLength);
        splitAxisNorm = static_cast<float>(cv::norm(splitAxis));
    }
    if (splitAxisNorm <= 1e-6f) {
        best.rejectReason = "bad_split_axis";
        return best;
    }
    splitAxis *= (1.0f / splitAxisNorm);

    const float parentMaxR = std::max({snapshot.aRadius, snapshot.bRadius, snapshot.cRadius,
                                       parent.getARadius(), parent.getBRadius(), parent.getCRadius()});
    if (parentMaxR <= 1e-3f) {
        best.rejectReason = "bad_parent_radius";
        return best;
    }

    const BoundingBox3D bbox = frame.computeBboxAtPoint(
        snapshot.position,
        parentMaxR,
        config.prob.bbox_margin_scale);
    if (!bbox.isValid()) {
        best.rejectReason = "bad_bbox";
        return best;
    }

    std::vector<size_t> inside;
    inside.reserve(centers.size());
    for (size_t i = 0; i < centers.size(); ++i) {
        if (pointInsideBbox(centers[i].position, bbox)) {
            inside.push_back(i);
        }
    }
    best.centersInside = static_cast<int>(inside.size());
    if (inside.size() < 2) {
        best.rejectReason = "too_few_inside_centers";
        return best;
    }

    const float minSep = std::max(0.0f, config.prob.signal_center_split_min_separation_radius_scale) *
                         parentMaxR;
    const float maxSep = std::max(config.prob.signal_center_split_min_separation_radius_scale,
                                  config.prob.signal_center_split_max_separation_radius_scale) *
                         parentMaxR;
    const float maxMidpointDistance =
        std::max(0.0f, config.prob.signal_center_split_max_midpoint_radius_scale) *
        parentMaxR;
    const float minAxisAlignment =
        std::clamp(config.prob.signal_center_split_min_axis_alignment, 0.0f, 1.0f);
    const cv::Point3f parentPos(parent.getX(), parent.getY(), parent.getZ());
    const cv::Point3f referencePos = snapshot.valid ? snapshot.position : parentPos;

    for (size_t ai = 0; ai < inside.size(); ++ai) {
        for (size_t bi = ai + 1; bi < inside.size(); ++bi) {
            const size_t centerAIdx = inside[ai];
            const size_t centerBIdx = inside[bi];
            const auto &centerA = centers[centerAIdx];
            const auto &centerB = centers[centerBIdx];
            cv::Point3f delta = centerB.position - centerA.position;
            const float separation = static_cast<float>(cv::norm(delta));
            ++best.candidatePairs;
            const float separationRatio = separation / parentMaxR;
            const cv::Point3f midpoint = 0.5f * (centerA.position + centerB.position);
            const float midpointDistance =
                static_cast<float>(cv::norm(midpoint - referencePos));
            const float minPairBrightness = std::min(centerA.brightness, centerB.brightness);
            const int minPairBoxes = std::min(centerA.boxes, centerB.boxes);
            const float centeredRescueMaxSep = std::max(maxSep, 2.75f * parentMaxR);
            const float centeredRescueMaxMidpoint =
                std::min(maxMidpointDistance, std::max(8.0f, 0.55f * parentMaxR));
            const bool centeredBrightWidePair =
                separation >= minSep &&
                separation <= centeredRescueMaxSep &&
                midpointDistance <= centeredRescueMaxMidpoint &&
                minPairBoxes >= 96 &&
                minPairBrightness >= 0.12f;
            if (separation < minSep ||
                (separation > maxSep && !centeredBrightWidePair)) {
                ++best.rejectedBySeparation;
                continue;
            }
            if (midpointDistance > maxMidpointDistance) {
                ++best.rejectedByMidpoint;
                continue;
            }
            const cv::Point3f axisDir = delta * (1.0f / std::max(1e-6f, separation));
            const float axisAlignment = std::abs(axisDir.dot(splitAxis));
            const float relaxedAxisAlignment =
                std::max(0.0f, minAxisAlignment * 0.80f);
            if (axisAlignment < relaxedAxisAlignment && !centeredBrightWidePair) {
                ++best.rejectedByAxis;
                if (axisAlignment > best.bestRejectedAxisAlignment) {
                    best.bestRejectedAxisAlignment = axisAlignment;
                    best.bestRejectedSeparationRatio = separationRatio;
                    best.bestRejectedMidpointDistance = midpointDistance;
                }
                continue;
            }

            const float brightness = centerA.brightness + centerB.brightness;
            const float effectiveAxisAlignment =
                centeredBrightWidePair
                    ? std::max(axisAlignment, relaxedAxisAlignment)
                    : axisAlignment;
            const float softAxisPenalty =
                std::max(0.0f, minAxisAlignment - effectiveAxisAlignment) * 2.0f;
            const float centeredRescuePenalty = centeredBrightWidePair ? 0.35f : 0.0f;
            const float score =
                midpointDistance / parentMaxR +
                0.50f * std::abs(separationRatio - 1.0f) +
                0.75f * (1.0f - effectiveAxisAlignment) -
                0.10f * brightness +
                softAxisPenalty +
                centeredRescuePenalty;
            if (score >= best.score) {
                continue;
            }

            best.valid = true;
            best.centerIndexA = centerAIdx;
            best.centerIndexB = centerBIdx;
            best.proposal.d1Pos = centerA.position;
            best.proposal.d2Pos = centerB.position;
            if ((best.proposal.d2Pos - best.proposal.d1Pos).dot(splitAxis) < 0.0f) {
                std::swap(best.proposal.d1Pos, best.proposal.d2Pos);
                std::swap(best.centerIndexA, best.centerIndexB);
            }
            best.proposal.candidateIdA = static_cast<int>(best.centerIndexA);
            best.proposal.candidateIdB = static_cast<int>(best.centerIndexB);
            best.proposal.elongation = score;
            best.proposal.parentShapeElongation = parentShape;
            best.proposal.elongatedParentRescued = true;
            best.proposal.gapStartBin = -4;
            best.proposal.gapEndBin = -4;
            best.proposal.leftPixelCount = std::max(1, centerA.boxes);
            best.proposal.rightPixelCount = std::max(1, centerB.boxes);
            best.separation = separation;
            best.separationRatio = separationRatio;
            best.midpointDistance = midpointDistance;
            best.axisAlignment = axisAlignment;
            best.score = score;
        }
    }

    if (!best.valid) {
        if (best.candidatePairs <= 0) {
            best.rejectReason = "no_candidate_pairs";
        } else if (best.rejectedBySeparation == best.candidatePairs) {
            best.rejectReason = "separation_gate";
        } else if (best.rejectedByMidpoint > 0 &&
                   best.rejectedBySeparation + best.rejectedByMidpoint == best.candidatePairs) {
            best.rejectReason = "midpoint_gate";
        } else if (best.rejectedByAxis > 0) {
            best.rejectReason = "axis_alignment_gate";
        } else {
            best.rejectReason = "no_scoring_candidate";
        }
    }

    return best;
}

static std::vector<cv::Mat> buildSignalProbabilityStack(
    const std::vector<cv::Mat> &realFrame,
    const std::vector<Frame::SignalCenter> &centers,
    const BaseConfig &config)
{
    std::vector<cv::Mat> probabilityDebug;
    probabilityDebug.reserve(realFrame.size());
    for (const auto &slice : realFrame) {
        probabilityDebug.emplace_back(cv::Mat::zeros(slice.size(), CV_32F));
    }

    if (realFrame.empty() || centers.empty()) {
        return probabilityDebug;
    }

    std::vector<cv::Mat> probability;
    probability.reserve(realFrame.size());
    for (const auto &slice : realFrame) {
        probability.emplace_back(cv::Mat::zeros(slice.size(), CV_32F));
    }

    const float minSigmaScale =
        std::max(0.0f, config.simulation.signal_guided_min_sigma_scale);
    const float sigmaRangeMultiplier =
        std::max(1e-3f, config.simulation.signal_guided_sigma_range_multiplier);

    float maxProb = 0.0f;
    #pragma omp parallel for schedule(static) reduction(max:maxProb)
    for (int z = 0; z < static_cast<int>(probability.size()); ++z) {
        cv::Mat &probSlice = probability[static_cast<size_t>(z)];
        for (int y = 0; y < probSlice.rows; ++y) {
            float *probRow = probSlice.ptr<float>(y);
            for (int x = 0; x < probSlice.cols; ++x) {
                float bestProb = 0.0f;
                for (const auto &center : centers) {
                    const float sigmaScale =
                        std::max(minSigmaScale, center.sigmaScale);
                    const float sx = std::max(
                        1e-3f, Ellipsoid::cellConfig.x.sigma * sigmaScale * sigmaRangeMultiplier);
                    const float sy = std::max(
                        1e-3f, Ellipsoid::cellConfig.y.sigma * sigmaScale * sigmaRangeMultiplier);
                    const float sz = std::max(
                        1e-3f, Ellipsoid::cellConfig.z.sigma * sigmaScale * sigmaRangeMultiplier);
                    const float dx = (static_cast<float>(x) - center.position.x) / sx;
                    const float dy = (static_cast<float>(y) - center.position.y) / sy;
                    const float dz = (static_cast<float>(z) - center.position.z) / sz;
                    const float p = std::exp(-0.5f * (dx * dx + dy * dy + dz * dz));
                    bestProb = std::max(bestProb, p);
                }
                probRow[x] = bestProb;
                maxProb = std::max(maxProb, bestProb);
            }
        }
    }

    if (maxProb > 1e-6f) {
        #pragma omp parallel for schedule(static)
        for (int z = 0; z < static_cast<int>(probabilityDebug.size()); ++z) {
            cv::Mat &debugSlice = probabilityDebug[static_cast<size_t>(z)];
            const cv::Mat &probSlice = probability[static_cast<size_t>(z)];
            for (int y = 0; y < debugSlice.rows; ++y) {
                float *debugRow = debugSlice.ptr<float>(y);
                const float *probRow = probSlice.ptr<float>(y);
                for (int x = 0; x < debugSlice.cols; ++x) {
                    debugRow[x] = std::clamp(probRow[x] / maxProb, 0.0f, 1.0f);
                }
            }
        }
    }

    return probabilityDebug;
}

static void exportSignalDebugStacks(const std::vector<cv::Mat> &realFrame,
                                    const std::vector<Frame::SignalCenter> &centers,
                                    const std::vector<cv::Mat> &probabilityDebug,
                                    const BaseConfig &config,
                                    const fs::path &baseOutputDir,
                                    const fs::path &framePath)
{
    if (!config.simulation.export_signal_debug_images || realFrame.empty()) {
        return;
    }

    std::vector<cv::Mat> centerCubes;
    centerCubes.reserve(realFrame.size());
    for (const auto &slice : realFrame) {
        centerCubes.emplace_back(cv::Mat::zeros(slice.size(), CV_32F));
    }

    constexpr int kCubeHalfSize = 2;
    const int zCount = static_cast<int>(realFrame.size());
    for (const auto &center : centers) {
        const int cx = static_cast<int>(std::round(center.position.x));
        const int cy = static_cast<int>(std::round(center.position.y));
        const int cz = static_cast<int>(std::round(center.position.z));
        const int z0 = std::max(0, cz - kCubeHalfSize);
        const int z1 = std::min(zCount - 1, cz + kCubeHalfSize);
        for (int z = z0; z <= z1; ++z) {
            cv::Mat &slice = centerCubes[static_cast<size_t>(z)];
            const int x0 = std::max(0, cx - kCubeHalfSize);
            const int x1 = std::min(slice.cols - 1, cx + kCubeHalfSize);
            const int y0 = std::max(0, cy - kCubeHalfSize);
            const int y1 = std::min(slice.rows - 1, cy + kCubeHalfSize);
            for (int y = y0; y <= y1; ++y) {
                float *row = slice.ptr<float>(y);
                for (int x = x0; x <= x1; ++x) {
                    row[x] = 1.0f;
                }
            }
        }
    }

    double probabilityMin = 0.0;
    double probabilityMax = 0.0;
    int nonzeroVoxels = 0;
    bool statsInitialized = false;
    for (const auto &slice : probabilityDebug) {
        if (slice.empty()) continue;
        double sliceMin = 0.0;
        double sliceMax = 0.0;
        cv::minMaxLoc(slice, &sliceMin, &sliceMax);
        if (!statsInitialized) {
            probabilityMin = sliceMin;
            probabilityMax = sliceMax;
            statsInitialized = true;
        } else {
            probabilityMin = std::min(probabilityMin, sliceMin);
            probabilityMax = std::max(probabilityMax, sliceMax);
        }
        nonzeroVoxels += cv::countNonZero(slice > 0.0f);
    }

    std::cout << "[Signal Debug Export] frame=" << framePath.filename().string()
              << " centers=" << centers.size()
              << " probability_min=" << probabilityMin
              << " probability_max=" << probabilityMax
              << " probability_nonzero_voxels=" << nonzeroVoxels
              << " sigma=(" << Ellipsoid::cellConfig.x.sigma
              << "," << Ellipsoid::cellConfig.y.sigma
              << "," << Ellipsoid::cellConfig.z.sigma << ")"
              << " sigmaRangeMultiplier="
              << config.simulation.signal_guided_sigma_range_multiplier
              << " output_dir=" << (baseOutputDir / "signal_debug").string()
              << '\n';

    exportStackToSubdir(centerCubes, baseOutputDir, "signal_debug/centers",
                        framePath,
                        config.simulation.export_frame_png,
                        config.simulation.export_frame_tiff);
    exportStackToSubdir(probabilityDebug, baseOutputDir, "signal_debug/perturb_probability",
                        framePath,
                        config.simulation.export_frame_png,
                        config.simulation.export_frame_tiff);
}

static float estimateAdaptiveBackgroundFromFrame(const Frame &frame,
                                                 const SimulationConfig &simulationConfig)
{
    const auto &realFrame = frame.getRealFrame();
    if (realFrame.empty()) {
        return 0.0f;  // post-sigmoid background invariant
    }

    const float expandFactor = std::max(1.0f, simulationConfig.adaptive_background_expand_factor);
    std::vector<cv::Mat> exclusionMask;
    exclusionMask.reserve(realFrame.size());
    for (const auto &slice : realFrame) {
        exclusionMask.emplace_back(cv::Mat::zeros(slice.size(), CV_32F));
    }

    for (const auto &cell : frame.cells) {
        auto params = cell.getCellParams();
        params.aRadius *= expandFactor;
        params.bRadius     *= expandFactor;
        params.cRadius *= expandFactor;
        Ellipsoid expandedCell(params);
        for (size_t z = 0; z < exclusionMask.size(); ++z) {
            expandedCell.draw(exclusionMask[z], simulationConfig, static_cast<float>(z));
        }
    }

    std::vector<float> backgroundCandidates;
    for (size_t z = 0; z < realFrame.size(); ++z) {
        const cv::Mat &slice = realFrame[z];
        const cv::Mat &mask = exclusionMask[z];
        for (int y = 0; y < slice.rows; ++y) {
            const float *sliceRow = slice.ptr<float>(y);
            const float *maskRow = mask.ptr<float>(y);
            for (int x = 0; x < slice.cols; ++x) {
                if (maskRow[x] <= 0.0f) {
                    backgroundCandidates.push_back(sliceRow[x]);
                }
            }
        }
    }

    if (backgroundCandidates.empty()) {
        return 0.0f;  // post-sigmoid background invariant
    }

    return computeMeanOfTopFraction(backgroundCandidates, simulationConfig.adaptive_background_top_fraction);
}

static float estimateAdaptiveBackgroundFromLocalCellBboxes(
    const Frame &frame,
    const SimulationConfig &simulationConfig,
    int displayFrame)
{
    const auto &realFrame = frame.getRealFrame();
    if (realFrame.empty()) {
        return 0.0f;
    }

    const float expandFactor =
        std::max(1.0f, simulationConfig.adaptive_background_expand_factor);
    std::vector<cv::Mat> cellVolumeMask;
    std::vector<cv::Mat> sampledMask;
    cellVolumeMask.reserve(realFrame.size());
    sampledMask.reserve(realFrame.size());
    for (const auto &slice : realFrame) {
        cellVolumeMask.emplace_back(cv::Mat::zeros(slice.size(), CV_32F));
        sampledMask.emplace_back(cv::Mat::zeros(slice.size(), CV_8U));
    }

    for (const auto &cell : frame.cells) {
        EllipsoidParams params = cell.getCellParams();
        params.brightness = 1.0f;
        Ellipsoid maskCell(params);
        for (size_t z = 0; z < cellVolumeMask.size(); ++z) {
            maskCell.draw(cellVolumeMask[z], simulationConfig, static_cast<float>(z));
        }
    }

    std::vector<float> backgroundCandidates;
    std::size_t bboxVoxelVisits = 0;
    std::size_t duplicateSkipped = 0;
    std::size_t cellVolumeSkipped = 0;
    for (const auto &cell : frame.cells) {
        const float maxR = std::max({
            cell.getARadius(), cell.getBRadius(), cell.getCRadius()});
        const float boxR = std::max(1.0f, maxR * expandFactor);
        const int minX = std::max(
            0, static_cast<int>(std::floor(cell.getX() - boxR)));
        const int maxX = std::min(
            realFrame.front().cols - 1,
            static_cast<int>(std::ceil(cell.getX() + boxR)));
        const int minY = std::max(
            0, static_cast<int>(std::floor(cell.getY() - boxR)));
        const int maxY = std::min(
            realFrame.front().rows - 1,
            static_cast<int>(std::ceil(cell.getY() + boxR)));
        const int minZ = std::max(
            0, static_cast<int>(std::floor(cell.getZ() - boxR)));
        const int maxZ = std::min(
            static_cast<int>(realFrame.size()) - 1,
            static_cast<int>(std::ceil(cell.getZ() + boxR)));

        if (minX > maxX || minY > maxY || minZ > maxZ) {
            continue;
        }

        for (int z = minZ; z <= maxZ; ++z) {
            const cv::Mat &slice = realFrame[static_cast<size_t>(z)];
            const cv::Mat &mask = cellVolumeMask[static_cast<size_t>(z)];
            cv::Mat &sampled = sampledMask[static_cast<size_t>(z)];
            for (int y = minY; y <= maxY; ++y) {
                const float *sliceRow = slice.ptr<float>(y);
                const float *maskRow = mask.ptr<float>(y);
                uint8_t *sampledRow = sampled.ptr<uint8_t>(y);
                for (int x = minX; x <= maxX; ++x) {
                    ++bboxVoxelVisits;
                    if (sampledRow[x]) {
                        ++duplicateSkipped;
                        continue;
                    }
                    sampledRow[x] = 1;
                    if (maskRow[x] > 0.0f) {
                        ++cellVolumeSkipped;
                        continue;
                    }
                    backgroundCandidates.push_back(sliceRow[x]);
                }
            }
        }
    }

    if (backgroundCandidates.empty()) {
        std::cout << "[Adaptive Background] frame " << displayFrame
                  << " mode=celluniverse2_local_bbox"
                  << " candidates=0"
                  << " fallback=global_background\n";
        return estimateAdaptiveBackgroundFromFrame(frame, simulationConfig);
    }

    const float localPercentile = std::clamp(
        simulationConfig.adaptive_background_local_bbox_percentile, 0.0f, 100.0f);
    const float background = percentileValue(backgroundCandidates, localPercentile);
    std::cout << "[Adaptive Background] frame " << displayFrame
              << " mode=celluniverse2_local_bbox"
              << " cells=" << frame.cells.size()
              << " bbox_voxel_visits=" << bboxVoxelVisits
              << " duplicate_skipped=" << duplicateSkipped
              << " cell_volume_skipped=" << cellVolumeSkipped
              << " candidates=" << backgroundCandidates.size()
              << " percentile=" << localPercentile
              << " background=" << background << '\n';
    return background;
}

static std::size_t countBackgroundVoxelsAfterCells(const Frame &frame,
                                                   const SimulationConfig &simulationConfig)
{
    const auto &realFrame = frame.getRealFrame();
    if (realFrame.empty()) {
        return 0;
    }

    std::size_t backgroundVoxels = 0;
    for (size_t z = 0; z < realFrame.size(); ++z) {
        cv::Mat mask = cv::Mat::zeros(realFrame[z].size(), CV_32F);
        for (const auto &cell : frame.cells) {
            EllipsoidParams params = cell.getCellParams();
            params.brightness = 1.0f;
            Ellipsoid maskCell(params);
            maskCell.draw(mask, simulationConfig, static_cast<float>(z));
        }

        for (int y = 0; y < mask.rows; ++y) {
            const float *row = mask.ptr<float>(y);
            for (int x = 0; x < mask.cols; ++x) {
                if (row[x] <= 0.0f) {
                    ++backgroundVoxels;
                }
            }
        }
    }

    return backgroundVoxels;
}

static float blackVoxelFractionInsideCell(const Frame &frame,
                                          const Ellipsoid &cell,
                                          const SimulationConfig &simulationConfig)
{
    const auto &realFrame = frame.getRealFrame();
    if (realFrame.empty()) {
        return 0.0f;
    }

    EllipsoidParams params = cell.getCellParams();
    params.brightness = 1.0f;
    Ellipsoid maskCell(params);

    std::size_t inside = 0;
    std::size_t black = 0;
    const float blackThreshold = std::max(0.0f, simulationConfig.post_alignment_black_threshold);
    for (size_t z = 0; z < realFrame.size(); ++z) {
        cv::Mat mask = cv::Mat::zeros(realFrame[z].size(), CV_32F);
        maskCell.draw(mask, simulationConfig, static_cast<float>(z));

        for (int y = 0; y < mask.rows; ++y) {
            const float *maskRow = mask.ptr<float>(y);
            const float *realRow = realFrame[z].ptr<float>(y);
            for (int x = 0; x < mask.cols; ++x) {
                if (maskRow[x] <= 0.0f) {
                    continue;
                }
                ++inside;
                if (realRow[x] <= blackThreshold) {
                    ++black;
                }
            }
        }
    }

    if (inside == 0) {
        return 0.0f;
    }
    return static_cast<float>(black) / static_cast<float>(inside);
}

static bool usesIterativePostPreprocessing(const SimulationConfig &config)
{
    (void)config;
    return false;
}

// Preprocessing moved to ImageHandler. Single preprocessed stack via
// ImageHandler::loadFrame.
CellUniverse::CellUniverse(std::map<std::string, std::vector<Ellipsoid>> initialCells,
                           PathVec imagePaths,
                           BaseConfig &config,
                           std::string outputPath,
                           int firstFrame,
                           int continueFrom)
: config(config), outputPath(outputPath), firstFrame(firstFrame),
  imagePaths(imagePaths), initialCells(initialCells), continueFrom(continueFrom)
{
    std::cout << "[Frame Intensity Scale] enabled="
              << config.simulation.frame_intensity_normalization_enabled
              << " low_percentile="
              << config.simulation.frame_intensity_scale_low_percentile
              << " high_percentile="
              << config.simulation.frame_intensity_scale_high_percentile
              << " exclude_zeros="
              << config.simulation.frame_intensity_percentile_exclude_zeros
              << " hard_max=" << config.simulation.frame_intensity_hard_max
              << '\n';

    if (config.simulation.edge_brightness_alignment_enabled) {
        std::cout << "[EdgeBrightnessAlignment] enabled=1"
                  << " target=pending_all_frames_min_background"
                  << " xy_margin=" << std::max(1, config.simulation.edge_brightness_alignment_xy_margin)
                  << " offsets=("
                  << std::max(0, config.simulation.edge_brightness_alignment_left_offset) << ","
                  << std::max(0, config.simulation.edge_brightness_alignment_right_offset) << ","
                  << std::max(0, config.simulation.edge_brightness_alignment_top_offset) << ","
                  << std::max(0, config.simulation.edge_brightness_alignment_bottom_offset) << ")"
                  << " max_shift=" << std::max(0.0f, config.simulation.edge_brightness_alignment_max_shift)
                  << '\n';
    } else {
        std::cout << "[EdgeBrightnessAlignment] enabled=0\n";
    }

    // Determine post-interpolation z_slices depth. getImageFilePaths() records
    // the raw TIFF slice count in config.simulation.z_slices, but optimizer
    // cells live in the interpolated z space. Keep Ellipsoid::cellConfig.maxZ
    // in that same coordinate system, otherwise newly built split daughters get
    // clamped back to the raw stack depth.
    if (!config.simulation.quit_after_preprocessing &&
        config.simulation.z_slices > 0) {
        const int rawSlices = config.simulation.z_slices;
        const int zScale = std::max(1, static_cast<int>(config.simulation.z_scaling));
        const int runtimeSlices = (rawSlices > 1)
            ? zScale * (rawSlices - 1) + 1
            : rawSlices;
        config.simulation.z_slices = runtimeSlices;
        Ellipsoid::cellConfig.maxZ =
            static_cast<float>(config.simulation.z_slices) - 1.0f;
        std::cout << "[M2 Probe] computed_from_raw_z_slices="
                  << rawSlices
                  << " z_scaling=" << zScale
                  << " runtime_z_slices=" << config.simulation.z_slices
                  << " maxZ=" << Ellipsoid::cellConfig.maxZ << '\n';
    } else if (!config.simulation.quit_after_preprocessing) {
        std::ostringstream probeLog;
        std::vector<cv::Mat> probe = ImageHandler::loadFrame(
            imagePaths.front().string(), config, &probeLog);
        config.simulation.z_slices = static_cast<int>(probe.size());
        Ellipsoid::cellConfig.maxZ = static_cast<float>(probe.size()) - 1.0f;
        std::cout << "[M2 Probe] post-preprocess z_slices=" << probe.size()
                  << " maxZ=" << Ellipsoid::cellConfig.maxZ << '\n';
    } else {
        std::cout << "[M2 Probe] skipped for preprocess-only mode\n";
    }

    // Allocate lazy-load Frame placeholders. Each holds only cells + config;
    // image stacks populated by `prepareFrame(i)` just before optimize.
    frames.reserve(imagePaths.size());
    for (size_t i = 0; i < imagePaths.size(); ++i)
    {
        fs::path path(imagePaths[i]);
        std::string file_name = path.filename();
        const bool hasInitial =
            (continueFrom == -1 || static_cast<int>(i) < continueFrom) &&
            initialCells.find(file_name) != initialCells.end();
        const std::vector<Ellipsoid> &cells =
            hasInitial ? initialCells.at(file_name) : std::vector<Ellipsoid>{};
        frames.emplace_back(config.simulation, cells, outputPath, file_name);
        if (config.cell) {
            frames.back().setBackgroundColor(config.cell->backgroundColor);
        }
    }
    perFrameAdaptiveBackground.assign(imagePaths.size(), 0.0f);
    perFrameMeanBrightness.assign(imagePaths.size(), 0.0f);

    for (size_t i = 0; i < frames.size(); ++i) {
        const size_t nonTrashCount = countNonTrashCellsInFrame(static_cast<int>(i));
        if (nonTrashCount > 0) {
            initialNonTrashCellCount = static_cast<int>(nonTrashCount);
            break;
        }
    }
    if (config.simulation.n2v2_contrast_gamma_split_decay_enabled) {
        std::cout << "[Gamma Split Decay] enabled=1"
                  << " base=" << config.simulation.n2v2_contrast_gamma
                  << " target=" << config.simulation.n2v2_contrast_gamma_split_decay_target
                  << " splits_to_target="
                  << config.simulation.n2v2_contrast_gamma_split_decay_splits_to_target
                  << " initial_nontrash=" << initialNonTrashCellCount
                  << '\n';
    }
}

size_t CellUniverse::countNonTrashCellsInFrame(int frameIndex) const
{
    if (frameIndex < 0 || static_cast<size_t>(frameIndex) >= frames.size()) {
        return 0;
    }
    size_t count = 0;
    for (const auto &cell : frames[static_cast<size_t>(frameIndex)].cells) {
        if (!cell.isTrash()) {
            ++count;
        }
    }
    return count;
}

double CellUniverse::effectiveN2V2ContrastGamma() const
{
    const SimulationConfig &simulation = config.simulation;
    const double baseGamma = simulation.n2v2_contrast_gamma;
    if (!simulation.n2v2_contrast_gamma_split_decay_enabled) {
        return baseGamma;
    }

    const double targetGamma = simulation.n2v2_contrast_gamma_split_decay_target;
    const int splitsToTarget =
        std::max(1, simulation.n2v2_contrast_gamma_split_decay_splits_to_target);
    const double fraction = std::clamp(
        static_cast<double>(std::max(0, cumulativeAcceptedSplits)) /
            static_cast<double>(splitsToTarget),
        0.0,
        1.0);
    return baseGamma + (targetGamma - baseGamma) * fraction;
}

BaseConfig CellUniverse::configForPreprocessing(int frameIndex) const
{
    (void)frameIndex;
    BaseConfig effectiveConfig = config;
    if (config.simulation.n2v2_contrast_gamma_split_decay_enabled) {
        effectiveConfig.simulation.n2v2_contrast_gamma = effectiveN2V2ContrastGamma();
    }
    return effectiveConfig;
}

void CellUniverse::invalidatePreparedFramesAfter(int frameIndex)
{
    int removed = 0;
    for (auto it = preparedFrameStacks.begin(); it != preparedFrameStacks.end();) {
        if (it->first > frameIndex) {
            it = preparedFrameStacks.erase(it);
            ++removed;
        } else {
            ++it;
        }
    }
    if (removed > 0) {
        std::cout << "[Gamma Split Decay] invalidated_future_prepared_frames="
                  << removed
                  << " after_frame=" << (firstFrame + frameIndex)
                  << '\n';
    }
}

void CellUniverse::applyCellLumenRescue(int frameIndex,
                                        const std::vector<cv::Mat> &preparedFrame)
{
    const CellLumenConfig &lumenConfig = config.cellLumen;
    if (!lumenConfig.enabled || !lumenConfig.fusionEnabled) {
        return;
    }
    if (frameIndex < 0 || static_cast<size_t>(frameIndex) >= frames.size()) {
        throw std::invalid_argument("applyCellLumenRescue: invalid frame index");
    }

    const int absoluteFrame = firstFrame + frameIndex;
    if (absoluteFrame < lumenConfig.fusionStartFrame) {
        return;
    }
    const int everyN = std::max(1, lumenConfig.fusionEveryNFrames);
    if (((absoluteFrame - lumenConfig.fusionStartFrame) % everyN) != 0) {
        return;
    }
    if (lumenConfig.fusionMaxAddedPerFrame == 0 &&
        !lumenConfig.fusionCenterPriorEnabled &&
        !lumenConfig.fusionRepairCloseCellsEnabled &&
        !lumenConfig.fusionSplitPriorEnabled) {
        return;
    }

    Frame &frame = frames[static_cast<size_t>(frameIndex)];
    const fs::path framePath = imagePaths[static_cast<size_t>(frameIndex)];
    auto &splitPriorsForFrame = cellLumenSplitPriors[frameIndex];
    splitPriorsForFrame.clear();
    auto &badSplitPriorParentsForFrame = cellLumenSplitPriorRejectedBadParents[frameIndex];
    badSplitPriorParentsForFrame.clear();
    auto &centerCandidatesForFrame = cellLumenCenterCandidates[frameIndex];
    centerCandidatesForFrame.clear();

    EllipsoidConfig savedCellConfig = Ellipsoid::cellConfig;
    std::vector<CellLumen::DetectedCell> candidates;
    try {
        CellLumen lumen(config, fs::path(outputPath) / "cell_lumen_fusion");
        candidates = lumen.detectCellsInPreparedStack(preparedFrame, framePath, false);
    } catch (const std::exception &ex) {
        Ellipsoid::cellConfig = savedCellConfig;
        std::cout << "[CellLumen Fusion] frame=" << absoluteFrame
                  << " status=skipped error=\"" << ex.what() << "\""
                  << std::endl;
        return;
    }
    Ellipsoid::cellConfig = savedCellConfig;

    std::sort(candidates.begin(), candidates.end(),
              [](const CellLumen::DetectedCell &a, const CellLumen::DetectedCell &b) {
                  if (a.top10MinusShell != b.top10MinusShell) {
                      return a.top10MinusShell > b.top10MinusShell;
                  }
                  return a.voxelCount > b.voxelCount;
              });

    std::ofstream candidateGraphLog = openCandidateGraphLog(outputPath, absoluteFrame);
    if (candidateGraphLog) {
        for (const auto &cell : frame.cells) {
            CandidateGraphRow row;
            row.frame = absoluteFrame;
            row.kind = "continuation";
            row.source = "current_cell_state";
            row.parent = cell.getName();
            row.selected = 1;
            row.parentShape = cell.shapeElongation();
            row.d1 = cv::Point3f(cell.getX(), cell.getY(), cell.getZ());
            row.note = "state_before_cell_lumen_rescue";
            writeCandidateGraphRow(candidateGraphLog, row);
        }
        for (size_t candIdx = 0; candIdx < candidates.size(); ++candIdx) {
            const auto &candidate = candidates[candIdx];
            CandidateGraphRow row;
            row.frame = absoluteFrame;
            row.kind = "lumen_center";
            row.source = "cell_lumen_high_recall";
            row.candidateA = std::to_string(candIdx);
            row.d1 = candidate.centerScaled;
            row.voxA = candidate.voxelCount;
            row.signalA = candidate.top10MinusShell;
            row.note = "cell_lumen_high_recall_center";
            writeCandidateGraphRow(candidateGraphLog, row);
        }
    }

    const float minDistance = std::max(0.0f, lumenConfig.fusionMinDistanceToExisting);
    const float minDistanceSq = minDistance * minDistance;
    const int maxAdded = lumenConfig.fusionMaxAddedPerFrame < 0
                             ? std::numeric_limits<int>::max()
                             : lumenConfig.fusionMaxAddedPerFrame;
    const float radiusScale = std::max(0.05f, lumenConfig.fusionRadiusScale);
    const float brightness = (lumenConfig.fusionBrightness >= 0.0f)
                                 ? lumenConfig.fusionBrightness
                                 : (config.cell ? config.cell->initialBrightness : 0.5f);

    const float centerCandidateMaxDistance = std::max(
        0.0f,
        lumenConfig.fusionCenterPriorMaxDistance > 0.0f
            ? lumenConfig.fusionCenterPriorMaxDistance
            : lumenConfig.fusionMinDistanceToExisting);
    const float centerCandidateMaxDistanceSq =
        centerCandidateMaxDistance * centerCandidateMaxDistance;
    for (size_t candIdx = 0; candIdx < candidates.size(); ++candIdx) {
        const auto &candidate = candidates[candIdx];
        if (candidate.voxelCount < lumenConfig.fusionMinVoxels ||
            candidate.top10MinusShell < lumenConfig.fusionMinTop10MinusShell) {
            continue;
        }

        size_t nearestIndex = std::numeric_limits<size_t>::max();
        float nearestDistanceSq = std::numeric_limits<float>::max();
        for (size_t existingIdx = 0; existingIdx < frame.cells.size(); ++existingIdx) {
            const cv::Point3f existingCenter(frame.cells[existingIdx].getX(),
                                             frame.cells[existingIdx].getY(),
                                             frame.cells[existingIdx].getZ());
            const float distSq = squaredDistance(candidate.centerScaled, existingCenter);
            if (distSq < nearestDistanceSq) {
                nearestDistanceSq = distSq;
                nearestIndex = existingIdx;
            }
        }
        if (nearestIndex == std::numeric_limits<size_t>::max() ||
            nearestDistanceSq > centerCandidateMaxDistanceSq) {
            continue;
        }

        const std::string parentName = frame.cells[nearestIndex].getName();
        const float distance = std::sqrt(nearestDistanceSq);
        auto existing = centerCandidatesForFrame.find(parentName);
        const bool replaceExisting =
            existing == centerCandidatesForFrame.end() ||
            distance < existing->second.distance - 1e-3f ||
            (std::abs(distance - existing->second.distance) <= 1e-3f &&
             candidate.top10MinusShell > existing->second.signal);
        if (!replaceExisting) {
            continue;
        }

        CellLumenCenterCandidate stored;
        stored.position = candidate.centerScaled;
        stored.distance = distance;
        stored.voxelCount = candidate.voxelCount;
        stored.signal = candidate.top10MinusShell;
        stored.candidateId = static_cast<int>(candIdx);
        centerCandidatesForFrame[parentName] = stored;
    }
    std::unordered_map<int, std::string> centerCandidateOwnerById;
    std::unordered_map<int, float> centerCandidateOwnerDistanceById;
    for (const auto &entry : centerCandidatesForFrame) {
        const auto &candidate = entry.second;
        if (candidate.candidateId < 0) {
            continue;
        }
        centerCandidateOwnerById[candidate.candidateId] = entry.first;
        centerCandidateOwnerDistanceById[candidate.candidateId] =
            candidate.distance;
    }

    struct FusionCandidateRef {
        int candidateId = -1;
        cv::Point3f center;
        int voxelCount = 0;
        float signal = 0.0f;
        float majorRadius = 0.0f;
        float bRadius = 0.0f;
        float minorRadius = 0.0f;
    };

    const int splitPriorMinVoxels =
        lumenConfig.fusionSplitPriorMinVoxels >= 0
            ? lumenConfig.fusionSplitPriorMinVoxels
            : lumenConfig.fusionMinVoxels;
    const float splitPriorMinTop10MinusShell =
        lumenConfig.fusionSplitPriorMinTop10MinusShell >= 0.0f
            ? lumenConfig.fusionSplitPriorMinTop10MinusShell
            : lumenConfig.fusionMinTop10MinusShell;
    auto passesFusionSplitPriorSignalGate = [&](const CellLumen::DetectedCell &candidate) {
        return candidate.voxelCount >= splitPriorMinVoxels &&
               candidate.top10MinusShell >= splitPriorMinTop10MinusShell;
    };

    int splitPriorCandidateAssignments = 0;
    int splitPriorCandidatesPassedGate = 0;
    int splitPriorRejectedParentDistance = 0;
    int splitPriorRejectedSeparation = 0;
    int splitPriorRejectedMidpoint = 0;
    int splitPriorRejectedScore = 0;
    int splitPriorRejectedNeighborClaim = 0;
    int splitPriorRejectedContinuationOwner = 0;
    int splitPriorRejectedSignal = 0;
    if (lumenConfig.fusionSplitPriorEnabled && frame.cells.size() > 0) {
        std::unordered_map<size_t, std::vector<FusionCandidateRef>> candidatesByParent;
        struct RankedSplitPrior {
            size_t parentIdx = 0;
            BridgeSplitProposal proposal;
            double score = 0.0;
            double rawScore = 0.0;
            float sep = 0.0f;
            float minSep = 0.0f;
            float maxSep = 0.0f;
            float midpointDist = 0.0f;
            int candidateA = -1;
            int candidateB = -1;
            int voxA = 0;
            int voxB = 0;
            float signalA = 0.0f;
            float signalB = 0.0f;
            float nearParentDist = 0.0f;
            float farParentDist = 0.0f;
            float parentDistanceBalance = 1.0f;
            float parentShapeElongation = 1.0f;
            float parentPersistencePenalty = 0.0f;
            float neighborClaimPenalty = 0.0f;
            float rankingSoftPenalty = 0.0f;
            float windowSupportScore = 0.0f;
            int windowBothDaughtersSupported = 0;
            int windowMissingDaughterCount = 0;
            int windowParentPersists = 0;
            float continuationClaimSoftPenalty = 0.0f;
            float balancedWindowBonus = 0.0f;
            std::vector<size_t> continuationClaimBlockers;
            std::string continuationClaimBlockerNames;
            bool conflictReplacementEligible = false;
            bool elongatedParentRescued = false;
            bool parentAnchored = false;
        };
        std::vector<RankedSplitPrior> rankedPriors;
        struct WindowSupportResult {
            float score = 0.0f;
            int bothDaughtersSupported = 0;
            int missingDaughterCount = 0;
            int parentPersists = 0;
        };
        std::map<int, const std::vector<CellLumenLookaheadCandidate> *> windowCandidatesByOffset;
        const bool windowEnabled =
            lumenConfig.fusionSplitPriorWindowEnabled &&
            lumenConfig.fusionSplitPriorWindowSize > 1;
        if (windowEnabled) {
            const int windowSize = std::clamp(
                lumenConfig.fusionSplitPriorWindowSize,
                2,
                5);
            for (int offset = 1; offset < windowSize; ++offset) {
                const int lookaheadFrameIndex = frameIndex + offset;
                if (lookaheadFrameIndex < 0 ||
                    static_cast<size_t>(lookaheadFrameIndex) >= frames.size()) {
                    break;
                }
                const auto &futureCandidates =
                    getCellLumenLookaheadCandidates(lookaheadFrameIndex);
                windowCandidatesByOffset[offset] = &futureCandidates;
            }
            std::cout << "[CellLumen Window] frame=" << absoluteFrame
                      << " enabled=1"
                      << " window_size=" << windowSize
                      << " loaded_offsets=" << windowCandidatesByOffset.size()
                      << " match_distance=" << lumenConfig.fusionSplitPriorWindowMatchDistance
                      << " match_distance_per_frame="
                      << lumenConfig.fusionSplitPriorWindowMatchDistancePerFrame
                      << std::endl;
        }

        auto nearestWindowCandidateDistance =
            [](const std::vector<CellLumenLookaheadCandidate> &futureCandidates,
               const cv::Point3f &point) -> float {
                float best = std::numeric_limits<float>::infinity();
                for (const auto &candidate : futureCandidates) {
                    best = std::min(
                        best,
                        static_cast<float>(cv::norm(candidate.position - point)));
                }
                return best;
            };

        auto computeWindowSupport = [&](const cv::Point3f &parentCenter,
                                        const cv::Point3f &d1,
                                        const cv::Point3f &d2) {
            WindowSupportResult result;
            if (!windowEnabled || windowCandidatesByOffset.empty()) {
                return result;
            }
            for (const auto &entry : windowCandidatesByOffset) {
                const int offset = entry.first;
                const auto *futureCandidates = entry.second;
                if (futureCandidates == nullptr || futureCandidates->empty()) {
                    result.missingDaughterCount += 2;
                    continue;
                }
                const float matchDistance =
                    std::max(0.0f, lumenConfig.fusionSplitPriorWindowMatchDistance) +
                    std::max(0.0f, lumenConfig.fusionSplitPriorWindowMatchDistancePerFrame) *
                        static_cast<float>(std::max(0, offset - 1));
                const float d1Dist = nearestWindowCandidateDistance(*futureCandidates, d1);
                const float d2Dist = nearestWindowCandidateDistance(*futureCandidates, d2);
                const float parentDist =
                    nearestWindowCandidateDistance(*futureCandidates, parentCenter);
                const bool d1Supported = d1Dist <= matchDistance;
                const bool d2Supported = d2Dist <= matchDistance;
                const bool parentPersists = parentDist <= matchDistance;
                const float offsetWeight = 1.0f / static_cast<float>(offset);
                if (d1Supported && d2Supported) {
                    ++result.bothDaughtersSupported;
                    result.score -=
                        std::max(0.0f, lumenConfig.fusionSplitPriorWindowDaughterSupportBonus) *
                        offsetWeight;
                } else {
                    const int missingCount =
                        (d1Supported ? 0 : 1) + (d2Supported ? 0 : 1);
                    result.missingDaughterCount += missingCount;
                    result.score +=
                        std::max(0.0f, lumenConfig.fusionSplitPriorWindowMissingDaughterPenalty) *
                        static_cast<float>(missingCount) *
                        0.5f *
                        offsetWeight;
                }
                if (parentPersists && !(d1Supported && d2Supported)) {
                    ++result.parentPersists;
                    result.score +=
                        std::max(0.0f, lumenConfig.fusionSplitPriorWindowParentPersistencePenalty) *
                        offsetWeight;
                }
            }
            return result;
        };

        for (size_t candIdx = 0; candIdx < candidates.size(); ++candIdx) {
            const auto &candidate = candidates[candIdx];
            if (!passesFusionSplitPriorSignalGate(candidate)) {
                ++splitPriorRejectedSignal;
                continue;
            }
            ++splitPriorCandidatesPassedGate;
            bool assignedToAnyParent = false;
            for (size_t ci = 0; ci < frame.cells.size(); ++ci) {
                const auto &cell = frame.cells[ci];
                const float dx = cell.getX() - candidate.centerScaled.x;
                const float dy = cell.getY() - candidate.centerScaled.y;
                const float dz = cell.getZ() - candidate.centerScaled.z;
                const float distSq = dx * dx + dy * dy + dz * dz;

                const Ellipsoid &parent = frame.cells[ci];
                const float parentMaxR = std::max({parent.getARadius(),
                                                   parent.getBRadius(),
                                                   parent.getCRadius()});
                const float parentMinR = std::max(
                    1e-5f,
                    std::min({parent.getARadius(),
                              parent.getBRadius(),
                              parent.getCRadius()}));
                const float parentShapeElongation = parentMaxR / parentMinR;
                const float absoluteCatch = std::max(0.0f, lumenConfig.fusionSplitPriorMaxParentDistance);
                const float scaledCatch = std::max(0.0f, parentMaxR *
                                                         lumenConfig.fusionSplitPriorParentRadiusScale);
                const float catchRadius = (absoluteCatch > 0.0f && scaledCatch > 0.0f)
                                              ? std::min(absoluteCatch, scaledCatch)
                                              : std::max(absoluteCatch, scaledCatch);
                if (catchRadius <= 0.0f ||
                    distSq > catchRadius * catchRadius) {
                    continue;
                }
                if (lumenConfig.fusionSplitPriorContinuationClaimGuardEnabled) {
                    const int candidateId = static_cast<int>(candIdx);
                    const auto ownerIt = centerCandidateOwnerById.find(candidateId);
                    if (ownerIt != centerCandidateOwnerById.end() &&
                        ownerIt->second != cell.getName()) {
                        const auto ownerDistanceIt =
                            centerCandidateOwnerDistanceById.find(candidateId);
                        const float ownerDistance =
                            ownerDistanceIt != centerCandidateOwnerDistanceById.end()
                                ? ownerDistanceIt->second
                                : std::numeric_limits<float>::infinity();
                        const float parentDistance = std::sqrt(distSq);
                        const float tieMargin = std::max(
                            0.0f,
                            lumenConfig
                                .fusionSplitPriorContinuationClaimTieMargin);
                        if (std::isfinite(ownerDistance) &&
                            ownerDistance + tieMargin < parentDistance) {
                            const bool elongatedParentCanCarrySoftConflict =
                                parentShapeElongation >=
                                    std::max(
                                        0.0f,
                                        lumenConfig
                                            .fusionSplitPriorElongatedParentMinShape) &&
                                parentDistance <= catchRadius;
                            if (!elongatedParentCanCarrySoftConflict) {
                                ++splitPriorRejectedContinuationOwner;
                                continue;
                            }
                        }
                    }
                }

                candidatesByParent[ci].push_back({
                    static_cast<int>(candIdx),
                    candidate.centerScaled,
                    candidate.voxelCount,
                    candidate.top10MinusShell,
                    candidate.majorRadius,
                    candidate.bRadius,
                    candidate.minorRadius});
                ++splitPriorCandidateAssignments;
                assignedToAnyParent = true;
            }
            if (!assignedToAnyParent) {
                ++splitPriorRejectedParentDistance;
            }
        }

        int splitPriorParentsWithTwoCandidates = 0;
        for (const auto &entry : candidatesByParent) {
            const size_t parentIdx = entry.first;
            if (parentIdx >= frame.cells.size()) {
                continue;
            }
            std::vector<FusionCandidateRef> list = entry.second;
            const Ellipsoid &parent = frame.cells[parentIdx];
            const cv::Point3f parentCenter(parent.getX(), parent.getY(), parent.getZ());
            const float parentMaxR = std::max({parent.getARadius(),
                                               parent.getBRadius(),
                                               parent.getCRadius()});
            const float parentMinR = std::max(
                1e-5f,
                std::min({parent.getARadius(), parent.getBRadius(), parent.getCRadius()}));
            const float parentShapeElongation = parentMaxR / parentMinR;
            const float absoluteCatch = std::max(0.0f, lumenConfig.fusionSplitPriorMaxParentDistance);
            const float scaledCatch = std::max(0.0f, parentMaxR *
                                                         lumenConfig.fusionSplitPriorParentRadiusScale);
            const float catchRadius = (absoluteCatch > 0.0f && scaledCatch > 0.0f)
                                          ? std::min(absoluteCatch, scaledCatch)
                                          : std::max(absoluteCatch, scaledCatch);
            const int parentAnchorCandidateId =
                2000000000 - static_cast<int>(std::min<size_t>(parentIdx, 1000000));

            if (windowEnabled &&
                list.size() < 2 &&
                catchRadius > 0.0f &&
                parentShapeElongation >=
                    std::max(2.0f, lumenConfig.fusionSplitPriorElongatedParentMinShape)) {
                struct LookaheadRef {
                    FusionCandidateRef ref;
                    float distance = 0.0f;
                };
                std::vector<LookaheadRef> lookaheadRefs;
                for (const auto &windowEntry : windowCandidatesByOffset) {
                    const int offset = windowEntry.first;
                    const auto *futureCandidates = windowEntry.second;
                    if (futureCandidates == nullptr || futureCandidates->empty()) {
                        continue;
                    }
                    const float futureCatchRadius =
                        catchRadius +
                        std::max(0.0f,
                                 lumenConfig.fusionSplitPriorWindowMatchDistancePerFrame) *
                            static_cast<float>(std::max(0, offset - 1));
                    for (const auto &future : *futureCandidates) {
                        const float distance =
                            static_cast<float>(cv::norm(future.position - parentCenter));
                        if (distance > futureCatchRadius) {
                            continue;
                        }
                        bool duplicatesCurrentCandidate = false;
                        for (const auto &current : list) {
                            if (cv::norm(current.center - future.position) < 2.0f) {
                                duplicatesCurrentCandidate = true;
                                break;
                            }
                        }
                        if (duplicatesCurrentCandidate) {
                            continue;
                        }
                        FusionCandidateRef ref;
                        ref.candidateId =
                            1000000 + offset * 100000 + std::max(0, future.candidateId);
                        ref.center = future.position;
                        ref.voxelCount = future.voxelCount;
                        ref.signal = future.signal;
                        lookaheadRefs.push_back({ref, distance});
                    }
                }
                std::sort(lookaheadRefs.begin(), lookaheadRefs.end(),
                          [](const LookaheadRef &a, const LookaheadRef &b) {
                              if (std::abs(a.distance - b.distance) > 1e-4f) {
                                  return a.distance < b.distance;
                              }
                              if (std::abs(a.ref.signal - b.ref.signal) > 1e-4f) {
                                  return a.ref.signal > b.ref.signal;
                              }
                              return a.ref.candidateId < b.ref.candidateId;
                          });
                const size_t maxLookaheadRefs = std::min<size_t>(3, lookaheadRefs.size());
                for (size_t idx = 0; idx < maxLookaheadRefs; ++idx) {
                    list.push_back(lookaheadRefs[idx].ref);
                }
                if (maxLookaheadRefs > 0) {
                    std::cout << "[CellLumen Fusion Lookahead Inject] frame="
                              << absoluteFrame
                              << " parent=" << parent.getName()
                              << " added=" << maxLookaheadRefs
                              << " parentShapeElong=" << parentShapeElongation
                              << " catchRadius=" << catchRadius
                              << std::endl;
                }
            }

            if (windowEnabled &&
                catchRadius > 0.0f &&
                !list.empty() &&
                parentShapeElongation >= 1.45f) {
                bool hasParentCenterCandidate = false;
                for (const auto &current : list) {
                    if (cv::norm(current.center - parentCenter) < 2.0f) {
                        hasParentCenterCandidate = true;
                        break;
                    }
                }
                if (!hasParentCenterCandidate) {
                    FusionCandidateRef parentAnchor;
                    parentAnchor.candidateId = parentAnchorCandidateId;
                    parentAnchor.center = parentCenter;
                    parentAnchor.voxelCount = 0;
                    parentAnchor.signal = 0.0f;
                    list.push_back(parentAnchor);
                    std::cout << "[CellLumen Fusion Parent Anchor] frame="
                              << absoluteFrame
                              << " parent=" << parent.getName()
                              << " parentShapeElong=" << parentShapeElongation
                              << " candidates=" << (list.size() - 1)
                              << std::endl;
                }
            }

            if (list.size() < 2) {
                continue;
            }
            ++splitPriorParentsWithTwoCandidates;
            const float minSep = std::max(
                lumenConfig.fusionSplitPriorMinSeparation,
                parentMaxR * lumenConfig.fusionSplitPriorMinSeparationRadiusScale);
            const float maxSepByScale = parentMaxR * lumenConfig.fusionSplitPriorMaxSeparationRadiusScale;
            const float maxSep = lumenConfig.fusionSplitPriorMaxSeparation > 0.0f
                                     ? std::min(lumenConfig.fusionSplitPriorMaxSeparation, maxSepByScale)
                                     : maxSepByScale;
            const float targetSep = std::max(minSep, parentMaxR * 1.55f);
            const bool rankingSoftGateEnabled =
                lumenConfig.fusionSplitPriorRankingSoftGateEnabled;
            struct NeighborClaim {
                size_t index = std::numeric_limits<size_t>::max();
                float distance = std::numeric_limits<float>::infinity();
                float maxRadius = 0.0f;
            };
            auto nearestOtherCellClaim = [&](const cv::Point3f &point) {
                NeighborClaim best;
                for (size_t otherIdx = 0; otherIdx < frame.cells.size(); ++otherIdx) {
                    if (otherIdx == parentIdx) {
                        continue;
                    }
                    const auto &other = frame.cells[otherIdx];
                    const cv::Point3f otherCenter(other.getX(), other.getY(), other.getZ());
                    const float distance = static_cast<float>(cv::norm(point - otherCenter));
                    if (distance < best.distance) {
                        best.index = otherIdx;
                        best.distance = distance;
                        best.maxRadius = std::max({other.getARadius(),
                                                   other.getBRadius(),
                                                   other.getCRadius()});
                    }
                }
                return best;
            };
            auto appendUniqueBlockerName = [&](std::vector<size_t> &blockers,
                                               std::string &names,
                                               size_t blockerIdx) {
                if (blockerIdx >= frame.cells.size()) {
                    return;
                }
                if (std::find(blockers.begin(), blockers.end(), blockerIdx) != blockers.end()) {
                    return;
                }
                blockers.push_back(blockerIdx);
                if (!names.empty()) {
                    names += '|';
                }
                names += frame.cells[blockerIdx].getName();
            };
            auto areImmediateLineageSiblings = [](const std::string &a,
                                                  const std::string &b) {
                if (a.size() != b.size() || a.size() < 2 || a == b) {
                    return false;
                }
                const char aLast = a.back();
                const char bLast = b.back();
                if ((aLast != '0' && aLast != '1') ||
                    (bLast != '0' && bLast != '1') ||
                    aLast == bLast) {
                    return false;
                }
                return a.compare(0, a.size() - 1, b, 0, b.size() - 1) == 0;
            };

            for (size_t i = 0; i < list.size(); ++i) {
                for (size_t j = i + 1; j < list.size(); ++j) {
                    double rankingSoftPenalty = 0.0;
                    const bool parentAnchoredPair =
                        list[i].candidateId == parentAnchorCandidateId ||
                        list[j].candidateId == parentAnchorCandidateId;
                    const float sep = static_cast<float>(cv::norm(list[i].center - list[j].center));
                    if (sep < minSep || sep > maxSep) {
                        const float softMinSep = minSep * 0.70f;
                        const float softMaxSep = maxSep * 1.30f;
                        const bool hardSeparationReject =
                            !rankingSoftGateEnabled ||
                            sep < softMinSep ||
                            sep > softMaxSep;
                        if (hardSeparationReject) {
                            ++splitPriorRejectedSeparation;
                            continue;
                        }
                        const float excess =
                            sep < minSep ? (minSep - sep) : (sep - maxSep);
                        rankingSoftPenalty +=
                            static_cast<double>(excess) *
                            static_cast<double>(
                                std::max(0.0f,
                                         lumenConfig
                                             .fusionSplitPriorRankingSoftSeparationPenalty));
                    }
                    if (lumenConfig.fusionSplitPriorRoundParentMinSeparationRadiusScale > 0.0f &&
                        parentShapeElongation <= lumenConfig.fusionSplitPriorRoundParentMaxShape) {
                        const float roundParentMinSep =
                            parentMaxR *
                            lumenConfig.fusionSplitPriorRoundParentMinSeparationRadiusScale;
                        if (sep < roundParentMinSep) {
                            const bool hardRoundSeparationReject =
                                !rankingSoftGateEnabled ||
                                sep < roundParentMinSep * 0.80f;
                            if (hardRoundSeparationReject) {
                                ++splitPriorRejectedSeparation;
                                continue;
                            }
                            rankingSoftPenalty +=
                                static_cast<double>(roundParentMinSep - sep) *
                                static_cast<double>(
                                    std::max(0.0f,
                                             lumenConfig
                                                 .fusionSplitPriorRankingSoftSeparationPenalty));
                        }
                    }
                    const cv::Point3f midpoint = 0.5f * (list[i].center + list[j].center);
                    const float midpointDist = static_cast<float>(cv::norm(midpoint - parentCenter));
                    const float maxMidpointDist = std::max(
                        lumenConfig.fusionSplitPriorMaxMidpointDistance,
                        parentMaxR * lumenConfig.fusionSplitPriorMaxMidpointRadiusScale);
                    if (maxMidpointDist > 0.0f && midpointDist > maxMidpointDist) {
                        const bool hardMidpointReject =
                            !rankingSoftGateEnabled ||
                            midpointDist > maxMidpointDist * 1.60f;
                        if (hardMidpointReject) {
                            ++splitPriorRejectedMidpoint;
                            continue;
                        }
                        rankingSoftPenalty +=
                            static_cast<double>(midpointDist - maxMidpointDist) *
                            static_cast<double>(
                                std::max(0.0f,
                                         lumenConfig
                                             .fusionSplitPriorRankingSoftMidpointPenalty));
                    }
                    const float parentDistA = static_cast<float>(cv::norm(list[i].center - parentCenter));
                    const float parentDistB = static_cast<float>(cv::norm(list[j].center - parentCenter));
                    const float nearParentDist = std::min(parentDistA, parentDistB);
                    const float farParentDist = std::max(parentDistA, parentDistB);
                    const float parentDistanceBalance =
                        parentAnchoredPair
                            ? 1.0f
                            : (farParentDist > 1e-5f ? nearParentDist / farParentDist : 1.0f);
                    const float minDaughterParentDist = std::max(
                        lumenConfig.fusionSplitPriorMinDaughterParentDistance,
                        parentMaxR * lumenConfig.fusionSplitPriorMinDaughterParentDistanceRadiusScale);
                    double parentPersistencePenalty = 0.0;
                    if (nearParentDist < minDaughterParentDist) {
                        parentPersistencePenalty +=
                            static_cast<double>(minDaughterParentDist - nearParentDist) *
                            static_cast<double>(lumenConfig.fusionSplitPriorParentPersistencePenalty);
                    }
                    if (parentDistanceBalance < lumenConfig.fusionSplitPriorMinParentDistanceBalance) {
                        parentPersistencePenalty +=
                            static_cast<double>(lumenConfig.fusionSplitPriorMinParentDistanceBalance -
                                                parentDistanceBalance) *
                            static_cast<double>(lumenConfig.fusionSplitPriorParentPersistencePenalty);
                    }
                    if (parentAnchoredPair) {
                        parentPersistencePenalty = 0.0;
                    }
                    double neighborClaimPenalty = 0.0;
                    const NeighborClaim nearestOtherA = nearestOtherCellClaim(list[i].center);
                    const NeighborClaim nearestOtherB = nearestOtherCellClaim(list[j].center);
                    const float neighborMargin = std::max(0.0f, lumenConfig.fusionSplitPriorNeighborClaimMargin);
                    if (std::isfinite(nearestOtherA.distance) &&
                        nearestOtherA.distance + neighborMargin < parentDistA) {
                        neighborClaimPenalty +=
                            static_cast<double>(parentDistA - nearestOtherA.distance - neighborMargin) *
                            static_cast<double>(lumenConfig.fusionSplitPriorNeighborClaimPenalty);
                    }
                    if (std::isfinite(nearestOtherB.distance) &&
                        nearestOtherB.distance + neighborMargin < parentDistB) {
                        neighborClaimPenalty +=
                            static_cast<double>(parentDistB - nearestOtherB.distance - neighborMargin) *
                            static_cast<double>(lumenConfig.fusionSplitPriorNeighborClaimPenalty);
                    }
                    std::vector<size_t> continuationClaimBlockers;
                    std::string continuationClaimBlockerNames;
                    bool siblingContinuationBlocked = false;
                    float closestContinuationClaimDistance =
                        std::numeric_limits<float>::infinity();
                    if (lumenConfig.fusionSplitPriorContinuationClaimGuardEnabled) {
                        const float claimRadiusScale = std::max(
                            0.0f,
                            lumenConfig.fusionSplitPriorContinuationClaimRadiusScale);
                        const float tieMargin = std::max(
                            0.0f,
                            lumenConfig.fusionSplitPriorContinuationClaimTieMargin);
                        auto maybeAddClaimBlocker =
                            [&](const NeighborClaim &claim, float parentDist) {
                                if (claim.index >= frame.cells.size() ||
                                    !std::isfinite(claim.distance) ||
                                    claim.maxRadius <= 0.0f) {
                                    return;
                                }
                                const bool insideNeighborContinuation =
                                    claim.distance <= claim.maxRadius * claimRadiusScale;
                                const bool neighborIsCompetitive =
                                    claim.distance <= parentDist + tieMargin;
                                if (insideNeighborContinuation && neighborIsCompetitive) {
                                    appendUniqueBlockerName(continuationClaimBlockers,
                                                            continuationClaimBlockerNames,
                                                            claim.index);
                                    siblingContinuationBlocked =
                                        siblingContinuationBlocked ||
                                        areImmediateLineageSiblings(
                                            parent.getName(),
                                            frame.cells[claim.index].getName());
                                    closestContinuationClaimDistance =
                                        std::min(closestContinuationClaimDistance,
                                                 claim.distance);
	                                }
	                            };
                        maybeAddClaimBlocker(nearestOtherA, parentDistA);
                        maybeAddClaimBlocker(nearestOtherB, parentDistB);
                    }
                    const bool conflictReplacementEligible =
                        lumenConfig.fusionSplitPriorConflictReplacementEnabled &&
                        nearParentDist <= parentMaxR *
                                              lumenConfig.fusionSplitPriorConflictCloseParentRadiusScale &&
                        sep >= parentMaxR *
                                   lumenConfig.fusionSplitPriorConflictMinNewSeparationRadiusScale;
                    if (conflictReplacementEligible &&
                        lumenConfig.fusionSplitPriorSuppressConflictPenaltiesForAsymmetricClose) {
                        parentPersistencePenalty = 0.0;
                        neighborClaimPenalty = 0.0;
                    }
                    if (lumenConfig.fusionSplitPriorMaxNeighborClaimPenalty >= 0.0f &&
                        neighborClaimPenalty >
                            static_cast<double>(lumenConfig.fusionSplitPriorMaxNeighborClaimPenalty)) {
                        const double excess =
                            neighborClaimPenalty -
                            static_cast<double>(
                                lumenConfig.fusionSplitPriorMaxNeighborClaimPenalty);
                        const bool hardNeighborReject =
                            !rankingSoftGateEnabled ||
                            excess > 24.0;
                        if (hardNeighborReject) {
                            ++splitPriorRejectedNeighborClaim;
                            continue;
                        }
                        rankingSoftPenalty +=
                            excess *
                            static_cast<double>(
                                std::max(0.0f,
                                         lumenConfig
                                             .fusionSplitPriorRankingSoftNeighborPenalty));
                    }
                    const float sepPenalty = std::abs(sep - targetSep);
                    const float signalBonus =
                        lumenConfig.fusionSplitPriorSignalBonusWeight *
                        (list[i].signal + list[j].signal);
                    const WindowSupportResult windowSupport =
                        computeWindowSupport(parentCenter,
                                             list[i].center,
                                             list[j].center);
                    double continuationClaimSoftPenalty = 0.0;
	                    if (lumenConfig.fusionSplitPriorContinuationClaimGuardEnabled &&
	                        !continuationClaimBlockers.empty()) {
	                        const float closeParentRadius =
	                            parentMaxR *
	                            std::max(
                                0.0f,
                                lumenConfig
                                    .fusionSplitPriorContinuationClaimCloseParentRadiusScale);
                        if (closeParentRadius > 0.0f && nearParentDist < closeParentRadius) {
                            continuationClaimSoftPenalty +=
                                static_cast<double>(closeParentRadius - nearParentDist) *
	                                static_cast<double>(
	                                    std::max(
	                                        0.0f,
	                                        lumenConfig
	                                            .fusionSplitPriorContinuationClaimCloseParentPenalty));
	                        }
	                        const float tightContinuationRadius =
	                            parentMaxR *
	                            std::max(
	                                0.85f,
	                                lumenConfig
	                                    .fusionSplitPriorContinuationClaimCloseParentRadiusScale);
                        if (std::isfinite(closestContinuationClaimDistance) &&
                            tightContinuationRadius > 0.0f &&
                            closestContinuationClaimDistance < tightContinuationRadius) {
	                            continuationClaimSoftPenalty +=
	                                static_cast<double>(
	                                    tightContinuationRadius -
	                                    closestContinuationClaimDistance) *
	                                static_cast<double>(
	                                    std::max(
	                                        0.0f,
	                                        lumenConfig
                                            .fusionSplitPriorContinuationClaimCloseParentPenalty));
                        }
                        if (siblingContinuationBlocked) {
                            continuationClaimSoftPenalty += 100.0;
                        }
                    }
                    const bool cleanStrongWindowSupported =
                        windowSupport.bothDaughtersSupported >= 2 &&
                        windowSupport.missingDaughterCount == 0 &&
                        windowSupport.parentPersists == 0;
                    const bool tolerableContinuationClaim =
                        continuationClaimBlockers.empty() ||
                        (continuationClaimBlockers.size() == 1 &&
                         parentDistanceBalance >=
                             std::max(
                                 0.0f,
                                 lumenConfig
                                     .fusionSplitPriorMinParentDistanceBalance) &&
                         nearParentDist >=
                             std::max(
                                 6.0f,
                                 lumenConfig
                                     .fusionSplitPriorMinDaughterParentDistance));
                    double balancedWindowBonus = 0.0;
                    // Future support is not strong evidence if a daughter is
                    // also claimed by a neighboring cell's continuation.
                    const bool balancedWindowSupported =
                        cleanStrongWindowSupported &&
                        continuationClaimBlockers.empty() &&
                        parentDistanceBalance >=
                            lumenConfig
                                .fusionSplitPriorWindowBalancedMinParentDistanceBalance &&
                        nearParentDist >=
                            parentMaxR *
                                std::max(
                                    0.0f,
                                    lumenConfig
                                        .fusionSplitPriorWindowBalancedMinNearParentRadiusScale);
                    if (balancedWindowSupported) {
                        balancedWindowBonus =
                            static_cast<double>(
                                std::max(
                                    0.0f,
                                    lumenConfig
                                        .fusionSplitPriorWindowBalancedDaughterBonus));
                    }
                    double parentAnchorBonus = 0.0;
                    const bool parentAnchorWindowSupported =
                        parentAnchoredPair &&
                        cleanStrongWindowSupported &&
                        continuationClaimBlockers.empty() &&
                        rankingSoftPenalty <= 1e-5 &&
                        neighborClaimPenalty <= 1e-5 &&
                        farParentDist >= std::max(minDaughterParentDist, minSep * 1.75f) &&
                        parentShapeElongation >= 1.45f;
                    if (parentAnchorWindowSupported) {
                        parentAnchorBonus = std::min(
                            8.0,
                            static_cast<double>(std::max(
                                0.0f,
                                lumenConfig.fusionSplitPriorWindowBalancedDaughterBonus)) *
                                0.30);
                    }
                    double rawScore =
                        static_cast<double>(lumenConfig.fusionSplitPriorMidpointWeight) *
                            static_cast<double>(midpointDist) +
                        static_cast<double>(lumenConfig.fusionSplitPriorSeparationPenaltyWeight) *
                            static_cast<double>(sepPenalty) -
                                         static_cast<double>(signalBonus) +
                                         parentPersistencePenalty +
                                         neighborClaimPenalty +
                                         rankingSoftPenalty +
                                         continuationClaimSoftPenalty +
                                         static_cast<double>(windowSupport.score) -
                                         balancedWindowBonus -
                                         parentAnchorBonus;
                    const bool elongatedRescueNeighborClaimOk =
                        lumenConfig.fusionSplitPriorElongatedParentMaxNeighborClaimPenalty < 0.0f ||
                        neighborClaimPenalty <=
                            static_cast<double>(
                                lumenConfig
                                    .fusionSplitPriorElongatedParentMaxNeighborClaimPenalty);
                    const bool elongatedParentRescued =
                        lumenConfig.fusionSplitPriorElongatedParentRescueEnabled &&
                        parentShapeElongation >= lumenConfig.fusionSplitPriorElongatedParentMinShape &&
                        (lumenConfig.fusionSplitPriorElongatedParentMaxScore <= 0.0f ||
                         rawScore <= static_cast<double>(lumenConfig.fusionSplitPriorElongatedParentMaxScore)) &&
                        elongatedRescueNeighborClaimOk;
                    if (lumenConfig.fusionSplitPriorMaxScore > 0.0f &&
                        rawScore > static_cast<double>(lumenConfig.fusionSplitPriorMaxScore) &&
                        !elongatedParentRescued) {
                        const double excess =
                            rawScore -
                            static_cast<double>(lumenConfig.fusionSplitPriorMaxScore);
                        const bool hardScoreReject =
                            !rankingSoftGateEnabled ||
                            rawScore >
                                static_cast<double>(
                                    lumenConfig.fusionSplitPriorMaxScore) *
                                    2.0 +
                                20.0;
                        if (hardScoreReject) {
                            ++splitPriorRejectedScore;
                            continue;
                        }
                        rankingSoftPenalty +=
                            excess *
                            static_cast<double>(
                                std::max(0.0f,
                                         lumenConfig
                                             .fusionSplitPriorRankingSoftScorePenalty));
                        rawScore +=
                            excess *
                            static_cast<double>(
                                std::max(0.0f,
                                         lumenConfig
                                             .fusionSplitPriorRankingSoftScorePenalty));
                    }
                    const double score =
                        rawScore -
                        (elongatedParentRescued
                             ? static_cast<double>(lumenConfig.fusionSplitPriorElongatedParentScoreBonus)
                             : 0.0);

                    BridgeSplitProposal proposal;
                    proposal.d1Pos = list[i].center;
                    proposal.d2Pos = list[j].center;
                    proposal.elongation = static_cast<float>(score);
                    proposal.parentShapeElongation = parentShapeElongation;
                    proposal.elongatedParentRescued = elongatedParentRescued;
                    proposal.candidateIdA = list[i].candidateId;
                    proposal.candidateIdB = list[j].candidateId;
                    proposal.windowBothDaughtersSupported =
                        windowSupport.bothDaughtersSupported;
                    proposal.windowMissingDaughterCount =
                        windowSupport.missingDaughterCount;
                    proposal.windowParentPersists =
                        windowSupport.parentPersists;
                    proposal.windowSupportScore = windowSupport.score;
                    proposal.balancedWindowBonus =
                        static_cast<float>(balancedWindowBonus);
                    proposal.parentDistanceBalance = parentDistanceBalance;
                    proposal.parentPersistencePenalty =
                        static_cast<float>(parentPersistencePenalty);
                    proposal.neighborClaimPenalty =
                        static_cast<float>(neighborClaimPenalty);
                    proposal.continuationClaimSoftPenalty =
                        static_cast<float>(continuationClaimSoftPenalty);
                    proposal.continuationClaimBlockerNames =
                        continuationClaimBlockerNames;
                    proposal.parentAnchored = parentAnchoredPair;
                    if (cleanStrongWindowSupported &&
                        tolerableContinuationClaim &&
                        lumenConfig
                                .fusionSplitPriorWindowHighConfidenceMaxOverlapCostFraction >=
                            0.0f) {
                        proposal.maxOverlapCostFractionOverride =
                            lumenConfig
                                .fusionSplitPriorWindowHighConfidenceMaxOverlapCostFraction;
                    }
                    proposal.leftPixelCount = list[i].voxelCount;
                    proposal.rightPixelCount = list[j].voxelCount;
                    rankedPriors.push_back({
                        parentIdx,
                        proposal,
                        score,
                        rawScore,
                        sep,
                        minSep,
                        maxSep,
                        midpointDist,
                        list[i].candidateId,
                        list[j].candidateId,
                        list[i].voxelCount,
                        list[j].voxelCount,
                        list[i].signal,
                        list[j].signal,
                        nearParentDist,
                        farParentDist,
                        parentDistanceBalance,
                        parentShapeElongation,
                        static_cast<float>(parentPersistencePenalty),
                        static_cast<float>(neighborClaimPenalty),
                        static_cast<float>(rankingSoftPenalty),
                        windowSupport.score,
                        windowSupport.bothDaughtersSupported,
                        windowSupport.missingDaughterCount,
                        windowSupport.parentPersists,
                        static_cast<float>(continuationClaimSoftPenalty),
                        static_cast<float>(balancedWindowBonus),
                        continuationClaimBlockers,
                        continuationClaimBlockerNames,
                        conflictReplacementEligible,
                        elongatedParentRescued,
                        parentAnchoredPair});
                }
            }
        }
        std::sort(rankedPriors.begin(), rankedPriors.end(),
                  [](const RankedSplitPrior &a, const RankedSplitPrior &b) {
                      return a.score < b.score;
                  });
        if (lumenConfig.fusionSplitPriorPrepassFallbackRejectBadLumenParent) {
            std::set<std::string> parentsSeenWithBestPair;
            for (const auto &ranked : rankedPriors) {
                if (ranked.parentIdx >= frame.cells.size()) {
                    continue;
                }
                const std::string parentName = frame.cells[ranked.parentIdx].getName();
                if (!parentsSeenWithBestPair.insert(parentName).second) {
                    continue;
                }
                const bool scoreTooHigh =
                    lumenConfig.fusionSplitPriorPrepassFallbackBadLumenMaxScore >= 0.0f &&
                    ranked.score >
                        static_cast<double>(
                            lumenConfig.fusionSplitPriorPrepassFallbackBadLumenMaxScore);
                const bool neighborClaimTooHigh =
                    lumenConfig.fusionSplitPriorPrepassFallbackBadLumenMaxNeighborClaimPenalty >= 0.0f &&
                    ranked.neighborClaimPenalty >
                        lumenConfig.fusionSplitPriorPrepassFallbackBadLumenMaxNeighborClaimPenalty;
                if (scoreTooHigh || neighborClaimTooHigh) {
                    badSplitPriorParentsForFrame.insert(parentName);
                }
            }
            if (!badSplitPriorParentsForFrame.empty()) {
                std::cout << "[CellLumen Fusion SplitPrior BadParent Summary] frame="
                          << absoluteFrame
                          << " parents=" << badSplitPriorParentsForFrame.size()
                          << " max_score="
                          << lumenConfig.fusionSplitPriorPrepassFallbackBadLumenMaxScore
                          << " max_neighbor_claim_penalty="
                          << lumenConfig
                                 .fusionSplitPriorPrepassFallbackBadLumenMaxNeighborClaimPenalty
                          << std::endl;
            }
        }
        const int debugTopN = std::max(0, lumenConfig.fusionSplitPriorDebugTopN);
        for (int debugIdx = 0;
             debugIdx < static_cast<int>(rankedPriors.size()) && debugIdx < debugTopN;
             ++debugIdx) {
            const auto &ranked = rankedPriors[static_cast<size_t>(debugIdx)];
            if (ranked.parentIdx >= frame.cells.size()) {
                continue;
            }
            const Ellipsoid &parent = frame.cells[ranked.parentIdx];
            std::cout << "[CellLumen Fusion SplitPrior Ranked] frame=" << absoluteFrame
                      << " rank=" << debugIdx
                      << " parent=" << parent.getName()
                      << " score=" << ranked.score
                      << " sep=" << ranked.sep
                      << " midpointDist=" << ranked.midpointDist
                      << " parentDistNearFar=(" << ranked.nearParentDist
                      << "," << ranked.farParentDist << ")"
                      << " parentDistBalance=" << ranked.parentDistanceBalance
                      << " parentShapeElong=" << ranked.parentShapeElongation
                      << " parentPersistencePenalty=" << ranked.parentPersistencePenalty
                      << " neighborClaimPenalty=" << ranked.neighborClaimPenalty
                      << " rankingSoftPenalty=" << ranked.rankingSoftPenalty
                      << " windowScore=" << ranked.windowSupportScore
                      << " windowBoth=" << ranked.windowBothDaughtersSupported
                      << " windowMissing=" << ranked.windowMissingDaughterCount
                      << " windowParentPersists=" << ranked.windowParentPersists
                      << " continuationClaimSoftPenalty="
                      << ranked.continuationClaimSoftPenalty
                      << " balancedWindowBonus=" << ranked.balancedWindowBonus
                      << " continuationClaimBlockers="
                      << (ranked.continuationClaimBlockerNames.empty()
                              ? "none"
                              : ranked.continuationClaimBlockerNames)
                      << " elongatedParentRescued=" << ranked.elongatedParentRescued
                      << " parentAnchored=" << ranked.parentAnchored
                      << " candidateIds=(" << ranked.candidateA << "," << ranked.candidateB << ")"
                      << std::endl;
        }
        const int maxPriors =
            lumenConfig.fusionSplitPriorMaxPriorsPerFrame < 0
                ? std::numeric_limits<int>::max()
                : lumenConfig.fusionSplitPriorMaxPriorsPerFrame;
        auto logStoredSplitPrior = [&](const RankedSplitPrior &ranked,
                                       const std::string &parentName) {
            std::cout << "[CellLumen Fusion SplitPrior] frame=" << absoluteFrame
                      << " parent=" << parentName
                      << " d1=(" << ranked.proposal.d1Pos.x << "," << ranked.proposal.d1Pos.y
                      << "," << ranked.proposal.d1Pos.z << ")"
                      << " d2=(" << ranked.proposal.d2Pos.x << "," << ranked.proposal.d2Pos.y
                      << "," << ranked.proposal.d2Pos.z << ")"
                      << " sep=" << ranked.sep
                      << " minSep=" << ranked.minSep
                      << " maxSep=" << ranked.maxSep
                      << " midpointDist=" << ranked.midpointDist
                      << " parentDistNearFar=(" << ranked.nearParentDist
                      << "," << ranked.farParentDist << ")"
                      << " parentDistBalance=" << ranked.parentDistanceBalance
                      << " parentShapeElong=" << ranked.parentShapeElongation
                      << " parentPersistencePenalty=" << ranked.parentPersistencePenalty
                      << " neighborClaimPenalty=" << ranked.neighborClaimPenalty
                      << " rankingSoftPenalty=" << ranked.rankingSoftPenalty
                      << " windowScore=" << ranked.windowSupportScore
                      << " windowBoth=" << ranked.windowBothDaughtersSupported
                      << " windowMissing=" << ranked.windowMissingDaughterCount
                      << " windowParentPersists=" << ranked.windowParentPersists
                      << " continuationClaimSoftPenalty="
                      << ranked.continuationClaimSoftPenalty
                      << " balancedWindowBonus=" << ranked.balancedWindowBonus
                      << " continuationClaimBlockers="
                      << (ranked.continuationClaimBlockerNames.empty()
                              ? "none"
                              : ranked.continuationClaimBlockerNames)
                      << " conflictReplacementEligible=" << ranked.conflictReplacementEligible
                      << " elongatedParentRescued=" << ranked.elongatedParentRescued
                      << " parentAnchored=" << ranked.parentAnchored
                      << " score=" << ranked.score
                      << " candidateIds=(" << ranked.candidateA << "," << ranked.candidateB << ")"
                      << " vox=(" << ranked.voxA << "," << ranked.voxB << ")"
                      << " signal=(" << ranked.signalA << "," << ranked.signalB << ")"
                      << std::endl;
        };
        struct ParentPriorGroup {
            std::string parentName;
            std::vector<size_t> rankedIndexes;
            double bestScore = 0.0;
        };

        std::map<std::string, ParentPriorGroup> groupedByParent;
        for (size_t rankedIdx = 0; rankedIdx < rankedPriors.size(); ++rankedIdx) {
            const auto &ranked = rankedPriors[rankedIdx];
            if (ranked.parentIdx >= frame.cells.size()) {
                continue;
            }
            if (ranked.candidateA < 0 || ranked.candidateB < 0 ||
                ranked.candidateA == ranked.candidateB) {
                continue;
            }
            const std::string parentName = frame.cells[ranked.parentIdx].getName();
            auto &group = groupedByParent[parentName];
            group.parentName = parentName;
            group.rankedIndexes.push_back(rankedIdx);
        }

        std::vector<ParentPriorGroup> parentGroups;
        parentGroups.reserve(groupedByParent.size());
        for (auto &entry : groupedByParent) {
            auto &group = entry.second;
            std::sort(group.rankedIndexes.begin(), group.rankedIndexes.end(),
                      [&](size_t a, size_t b) {
                          return rankedPriors[a].score < rankedPriors[b].score;
                      });
            group.bestScore = group.rankedIndexes.empty()
                                  ? std::numeric_limits<double>::infinity()
                                  : rankedPriors[group.rankedIndexes.front()].score;
            parentGroups.push_back(group);
        }
        std::sort(parentGroups.begin(), parentGroups.end(),
                  [](const ParentPriorGroup &a, const ParentPriorGroup &b) {
                      if (std::abs(a.bestScore - b.bestScore) > 1e-9) {
                          return a.bestScore < b.bestScore;
                      }
                      return a.parentName < b.parentName;
                  });

        std::set<size_t> selectedRankedPriorIndexes;
        size_t searchNodes = 0;
        bool searchTruncated = false;
        const double maxSelectableCost =
            static_cast<double>(lumenConfig.fusionSplitPriorGlobalSelectMaxCost);
        auto hasCleanWindowSupport = [](const RankedSplitPrior &ranked) {
            return ranked.windowBothDaughtersSupported > 0 &&
                   ranked.windowMissingDaughterCount == 0 &&
                   ranked.windowParentPersists == 0;
        };
        auto isFutureContinuationConflictRescue =
            [&](const RankedSplitPrior &ranked) {
                if (ranked.parentAnchored || !hasCleanWindowSupport(ranked)) {
                    return false;
                }
                return ranked.continuationClaimBlockers.size() == 1 &&
                       ranked.neighborClaimPenalty <= 1e-5f &&
                       ranked.parentPersistencePenalty <= 1e-5f &&
                       ranked.continuationClaimSoftPenalty >= 15.0f &&
                       ranked.rankingSoftPenalty >= 4.0f &&
                       ranked.parentDistanceBalance >=
                           std::max(
                               0.45f,
                               lumenConfig
                                   .fusionSplitPriorMinParentDistanceBalance) &&
                       ranked.nearParentDist >=
                           std::max(
                               8.0f,
                               lumenConfig
                                   .fusionSplitPriorMinDaughterParentDistance) &&
                       ranked.farParentDist >=
                           std::max(ranked.minSep * 1.45f, 16.0f) &&
                       ranked.midpointDist <=
                           std::max(ranked.minSep * 0.55f, 7.0f) &&
                       ranked.sep >= ranked.minSep * 2.0f &&
                       ranked.score <= maxSelectableCost;
            };
        if (maxPriors > 0 && !parentGroups.empty()) {
            std::set<int> activeCandidateIds;
            std::set<size_t> activeParentIndexes;
            std::vector<size_t> activeSelection;
            std::vector<size_t> bestSelection;
            int bestCount = 0;
            double bestScore = 0.0;
            const int finiteMaxPriors =
                maxPriors == std::numeric_limits<int>::max()
                    ? static_cast<int>(parentGroups.size())
                    : std::min<int>(maxPriors, static_cast<int>(parentGroups.size()));
            std::vector<std::vector<size_t>> viableRankedIndexesByGroup(parentGroups.size());
            auto hasTolerableContinuationClaim = [&](const RankedSplitPrior &ranked) {
                if (ranked.continuationClaimBlockers.empty()) {
                    return true;
                }
                return ranked.continuationClaimBlockers.size() == 1 &&
                       ranked.parentDistanceBalance >=
                           std::max(0.0f,
                                    lumenConfig
                                        .fusionSplitPriorMinParentDistanceBalance) &&
                       ranked.nearParentDist >=
                           std::max(6.0f,
                                    lumenConfig
                                        .fusionSplitPriorMinDaughterParentDistance);
            };
            auto isWindowBackedConflictResolution = [&](const RankedSplitPrior &ranked) {
                if (!hasCleanWindowSupport(ranked)) {
                    return false;
                }
                const double hardScoreCeiling =
                    static_cast<double>(std::max(
                        lumenConfig.fusionSplitPriorMaxScore * 2.0f + 20.0f,
                        0.0f));
                return ranked.neighborClaimPenalty >
                           std::max(0.0f, lumenConfig.fusionSplitPriorMaxNeighborClaimPenalty) &&
                       ranked.parentDistanceBalance >=
                           std::max(0.0f, lumenConfig.fusionSplitPriorMinParentDistanceBalance) &&
                       ranked.nearParentDist >= 10.0f &&
                       ranked.score <= hardScoreCeiling;
            };
            auto isWeakAsymmetricElongatedRescue = [&](const RankedSplitPrior &ranked) {
                const bool cleanAsymmetricWindowPair =
                    !ranked.parentAnchored &&
                    hasCleanWindowSupport(ranked) &&
                    ranked.continuationClaimBlockers.empty() &&
                    ranked.neighborClaimPenalty <= 1e-5f &&
                    ranked.parentShapeElongation >=
                        std::max(
                            1.50f,
                            lumenConfig.fusionSplitPriorElongatedParentMinShape) &&
                    ranked.parentDistanceBalance >= 0.40f &&
                    ranked.nearParentDist >=
                        std::max(
                            6.0f,
                            lumenConfig.fusionSplitPriorMinDaughterParentDistance) &&
                    ranked.score <= maxSelectableCost;
                if (cleanAsymmetricWindowPair) {
                    return false;
                }
                return !ranked.parentAnchored &&
                       ranked.parentDistanceBalance <
                           std::max(
                               0.60f,
                               lumenConfig
                                   .fusionSplitPriorWindowBalancedMinParentDistanceBalance) &&
                       ranked.rawScore > 0.0 &&
                       ranked.score <= 0.0;
            };
            auto isStrongParentAnchoredOneSided = [&](const RankedSplitPrior &ranked) {
                if (!ranked.parentAnchored || !hasCleanWindowSupport(ranked)) {
                    return false;
                }
                return ranked.continuationClaimBlockers.empty() &&
                       ranked.rankingSoftPenalty <= 1e-5f &&
                       ranked.neighborClaimPenalty <= 1e-5f &&
                       ranked.parentShapeElongation >= 1.45f &&
                       ranked.farParentDist >=
                           std::max(ranked.minSep * 1.75f,
                                    std::max(6.0f,
                                             lumenConfig
                                                 .fusionSplitPriorMinDaughterParentDistance)) &&
                       ranked.score <= maxSelectableCost;
            };
            auto isCompactParentAnchoredWindowRescue = [&](const RankedSplitPrior &ranked) {
                (void) ranked;
                return false;
            };
	            auto isPartialParentAnchoredWindowRescue = [&](const RankedSplitPrior &ranked) {
	                (void) ranked;
	                return false;
	            };
	            auto isRealLumenCandidateId = [](int candidateId) {
	                return candidateId >= 0 && candidateId < 1000000000;
	            };
	            auto isCleanTwoRealWindowPair = [&](const RankedSplitPrior &ranked) {
	                return !ranked.parentAnchored &&
	                       isRealLumenCandidateId(ranked.candidateA) &&
	                       isRealLumenCandidateId(ranked.candidateB) &&
	                       ranked.windowBothDaughtersSupported >= 2 &&
	                       ranked.windowMissingDaughterCount == 0 &&
	                       ranked.windowParentPersists == 0 &&
	                       ranked.continuationClaimSoftPenalty <= 1e-5f &&
	                       ranked.parentDistanceBalance >= 0.75f &&
	                       ranked.nearParentDist >=
	                           std::max(10.0f, ranked.minSep * 1.35f) &&
	                       ranked.score <= std::min(8.0, maxSelectableCost);
	            };
            auto isSelectableRankedPrior = [&](const RankedSplitPrior &ranked) {
                if (ranked.parentAnchored) {
                    return isStrongParentAnchoredOneSided(ranked) ||
                           isCompactParentAnchoredWindowRescue(ranked) ||
                           isPartialParentAnchoredWindowRescue(ranked);
                }
                if (isWeakAsymmetricElongatedRescue(ranked)) {
                    return false;
                }
                if (ranked.score <= 0.0) {
                    return true;
                }
	                if (isFutureContinuationConflictRescue(ranked)) {
	                    return true;
	                }
	                if (isCleanTwoRealWindowPair(ranked)) {
	                    return true;
	                }
	                if (hasCleanWindowSupport(ranked) &&
	                    hasTolerableContinuationClaim(ranked) &&
	                    ranked.score <= maxSelectableCost) {
                    return true;
                }
                return isWindowBackedConflictResolution(ranked);
            };
            auto selectionCost = [&](const RankedSplitPrior &ranked) {
                if (ranked.parentAnchored) {
                    if (isStrongParentAnchoredOneSided(ranked)) {
                        return ranked.score;
                    }
                    if (isCompactParentAnchoredWindowRescue(ranked)) {
                        return ranked.score -
                               std::min(
                                   40.0,
                                   std::max(0.0, maxSelectableCost * 0.80));
                    }
                    if (isPartialParentAnchoredWindowRescue(ranked)) {
                        return ranked.score - 55.0;
                    }
                    return 0.0;
                }
                if (ranked.score <= 0.0) {
                    return ranked.score;
                }
	                if (isFutureContinuationConflictRescue(ranked)) {
	                    return ranked.score -
	                           std::min(
	                               60.0,
	                               static_cast<double>(
	                                   ranked.continuationClaimSoftPenalty) +
	                                   35.0);
	                }
	                if (isCleanTwoRealWindowPair(ranked)) {
	                    return ranked.score -
	                           std::min(18.0, std::max(0.0, maxSelectableCost * 0.36));
	                }
	                const bool elongatedCleanWindowRescue =
                    hasCleanWindowSupport(ranked) &&
                    hasTolerableContinuationClaim(ranked) &&
                    ranked.parentShapeElongation >=
                        std::max(
                            0.0f,
                            lumenConfig.fusionSplitPriorElongatedParentMinShape) &&
                    ranked.parentDistanceBalance >=
                        std::max(
                            0.0f,
                            lumenConfig
                                .fusionSplitPriorWindowBalancedMinParentDistanceBalance) &&
                    ranked.score <= maxSelectableCost;
                if (elongatedCleanWindowRescue) {
                    return ranked.score -
                           std::min(30.0, std::max(0.0, maxSelectableCost * 0.60));
                }
                if (hasCleanWindowSupport(ranked) &&
                    hasTolerableContinuationClaim(ranked)) {
                    return ranked.score;
                }
                return 0.0;
            };
            for (size_t groupIdx = 0; groupIdx < parentGroups.size(); ++groupIdx) {
                for (const size_t rankedIdx : parentGroups[groupIdx].rankedIndexes) {
                    const RankedSplitPrior &ranked = rankedPriors[rankedIdx];
                    if (isSelectableRankedPrior(ranked)) {
                        viableRankedIndexesByGroup[groupIdx].push_back(rankedIdx);
                    }
                }
            }
            std::vector<double> suffixBestPossible(parentGroups.size() + 1, 0.0);
            for (int groupIdx = static_cast<int>(parentGroups.size()) - 1; groupIdx >= 0; --groupIdx) {
                double bestGroupScore = 0.0;
                for (const size_t rankedIdx : viableRankedIndexesByGroup[static_cast<size_t>(groupIdx)]) {
                    bestGroupScore = std::min(
                        bestGroupScore,
                        selectionCost(rankedPriors[rankedIdx]));
                }
                suffixBestPossible[static_cast<size_t>(groupIdx)] =
                    suffixBestPossible[static_cast<size_t>(groupIdx + 1)] + bestGroupScore;
            }
            const size_t maxSearchNodes = 1000000;

            auto updateBest = [&](int count, double score) {
                if (score < bestScore - 1e-9 ||
                    (std::abs(score - bestScore) <= 1e-9 && count > bestCount)) {
                    bestCount = count;
                    bestScore = score;
                    bestSelection = activeSelection;
                }
            };

            std::function<void(size_t, int, double)> search =
                [&](size_t groupIdx, int count, double score) {
                    if (searchTruncated) {
                        return;
                    }
                    ++searchNodes;
                    if (searchNodes > maxSearchNodes) {
                        searchTruncated = true;
                        return;
                    }
                    if (score + suffixBestPossible[groupIdx] > bestScore + 1e-9) {
                        return;
                    }
                    if (count >= finiteMaxPriors || groupIdx >= parentGroups.size()) {
                        updateBest(count, score);
                        return;
                    }

                    for (const size_t rankedIdx : viableRankedIndexesByGroup[groupIdx]) {
                        const RankedSplitPrior &ranked = rankedPriors[rankedIdx];
                        if (activeCandidateIds.count(ranked.candidateA) > 0 ||
                            activeCandidateIds.count(ranked.candidateB) > 0) {
                            continue;
                        }
                        activeCandidateIds.insert(ranked.candidateA);
                        activeCandidateIds.insert(ranked.candidateB);
                        activeParentIndexes.insert(ranked.parentIdx);
                        activeSelection.push_back(rankedIdx);
                        search(groupIdx + 1, count + 1, score + selectionCost(ranked));
                        activeSelection.pop_back();
                        activeParentIndexes.erase(ranked.parentIdx);
                        activeCandidateIds.erase(ranked.candidateA);
                        activeCandidateIds.erase(ranked.candidateB);
                    }

                    search(groupIdx + 1, count, score);
                };

            search(0, 0, 0.0);
            selectedRankedPriorIndexes.insert(bestSelection.begin(), bestSelection.end());
        }

        for (const size_t rankedIdx : selectedRankedPriorIndexes) {
            const RankedSplitPrior &ranked = rankedPriors[rankedIdx];
            if (ranked.parentIdx >= frame.cells.size()) {
                continue;
            }
            const std::string parentName = frame.cells[ranked.parentIdx].getName();
            BridgeSplitProposal proposal = ranked.proposal;
            proposal.futureContinuationConflictRescued =
                isFutureContinuationConflictRescue(ranked);
            proposal.continuationClaimBlockerNames =
                ranked.continuationClaimBlockerNames;
            splitPriorsForFrame[parentName] = proposal;
            logStoredSplitPrior(ranked, parentName);
        }

        double selectedScore = 0.0;
        for (const size_t rankedIdx : selectedRankedPriorIndexes) {
            selectedScore += rankedPriors[rankedIdx].score;
        }
        std::cout << "[CellLumen Fusion SplitPrior GlobalSelect] frame=" << absoluteFrame
                  << " ranked=" << rankedPriors.size()
                  << " parent_groups=" << parentGroups.size()
                  << " selected=" << selectedRankedPriorIndexes.size()
                  << " max_priors=" << maxPriors
                  << " objective=min_cost_skip_zero"
                  << " max_selectable_cost=" << maxSelectableCost
                  << " selected_score=" << selectedScore
                  << " search_nodes=" << searchNodes
                  << " search_truncated=" << (searchTruncated ? 1 : 0)
                  << std::endl;

        if (candidateGraphLog) {
            for (size_t rankedIdx = 0; rankedIdx < rankedPriors.size(); ++rankedIdx) {
                const RankedSplitPrior &ranked = rankedPriors[rankedIdx];
                if (ranked.parentIdx >= frame.cells.size()) {
                    continue;
                }
                CandidateGraphRow row;
                row.frame = absoluteFrame;
                row.kind = "split_pair";
                row.source = "cell_lumen_split_prior";
                row.parent = frame.cells[ranked.parentIdx].getName();
                row.candidateA = std::to_string(ranked.candidateA);
                row.candidateB = std::to_string(ranked.candidateB);
                row.selected = selectedRankedPriorIndexes.count(rankedIdx) > 0 ? 1 : 0;
                row.score = ranked.score;
                row.rawScore = ranked.rawScore;
                row.bridgeMetric = ranked.elongatedParentRescued ? 1.0 : 0.0;
                row.sep = ranked.sep;
                row.minSep = ranked.minSep;
                row.maxSep = ranked.maxSep;
                row.midpointDist = ranked.midpointDist;
                row.parentShape = ranked.parentShapeElongation;
                row.parentPersistencePenalty = ranked.parentPersistencePenalty;
                row.neighborClaimPenalty = ranked.neighborClaimPenalty;
                row.parentDistNear = ranked.nearParentDist;
                row.parentDistFar = ranked.farParentDist;
                row.parentDistBalance = ranked.parentDistanceBalance;
                row.d1 = ranked.proposal.d1Pos;
                row.d2 = ranked.proposal.d2Pos;
                row.voxA = ranked.voxA;
                row.voxB = ranked.voxB;
                row.signalA = ranked.signalA;
                row.signalB = ranked.signalB;
                std::ostringstream note;
                if (ranked.conflictReplacementEligible) {
                    note << "conflict_replacement_eligible;";
                }
                if (ranked.parentAnchored) {
                    note << "parent_anchored;";
                }
                note << "ranking_soft_penalty=" << ranked.rankingSoftPenalty
                     << ";window_score=" << ranked.windowSupportScore
                     << ";window_both=" << ranked.windowBothDaughtersSupported
                     << ";window_missing=" << ranked.windowMissingDaughterCount
                     << ";window_parent_persists=" << ranked.windowParentPersists
                     << ";continuation_claim_soft_penalty="
                     << ranked.continuationClaimSoftPenalty
                     << ";balanced_window_bonus=" << ranked.balancedWindowBonus
                     << ";continuation_claim_blockers="
                     << (ranked.continuationClaimBlockerNames.empty()
                             ? "none"
                             : ranked.continuationClaimBlockerNames);
                row.note = note.str();
                writeCandidateGraphRow(candidateGraphLog, row);
            }
        }
        std::cout << "[CellLumen Fusion SplitPrior Summary] frame=" << absoluteFrame
                  << " candidates=" << candidates.size()
                  << " passed_gate=" << splitPriorCandidatesPassedGate
                  << " assigned_candidates=" << splitPriorCandidateAssignments
                  << " parents_with_2plus=" << splitPriorParentsWithTwoCandidates
                  << " ranked_pairs=" << rankedPriors.size()
                  << " priors=" << splitPriorsForFrame.size()
                  << " max_priors=" << maxPriors
                  << " rejected_parent_distance=" << splitPriorRejectedParentDistance
                  << " rejected_separation_pairs=" << splitPriorRejectedSeparation
                  << " rejected_midpoint_pairs=" << splitPriorRejectedMidpoint
                  << " rejected_score_pairs=" << splitPriorRejectedScore
                  << " rejected_neighbor_claim_pairs=" << splitPriorRejectedNeighborClaim
                  << " rejected_continuation_owner_assignments="
                  << splitPriorRejectedContinuationOwner
                  << " rejected_signal_candidates=" << splitPriorRejectedSignal
                  << " window_enabled=" << (windowEnabled ? 1 : 0)
                  << " window_offsets=" << windowCandidatesByOffset.size()
                  << " ranking_soft_gate="
                  << (lumenConfig.fusionSplitPriorRankingSoftGateEnabled ? 1 : 0)
                  << " split_min_voxels=" << splitPriorMinVoxels
                  << " split_min_top10_minus_shell=" << splitPriorMinTop10MinusShell
                  << std::endl;
    }

    int rejectedSize = 0;
    int rejectedSignal = 0;
    int rejectedClose = 0;
    int repairedClose = 0;
    int centerPriorGuided = 0;
    int added = 0;
    std::vector<bool> centerPriorUsed(frame.cells.size(), false);

    for (const auto &candidate : candidates) {
        if (added >= maxAdded) {
            break;
        }
        if (candidate.voxelCount < lumenConfig.fusionMinVoxels) {
            ++rejectedSize;
            continue;
        }
        if (candidate.top10MinusShell < lumenConfig.fusionMinTop10MinusShell) {
            ++rejectedSignal;
            continue;
        }

        const float candidateA = std::max(1.0f, candidate.majorRadius * radiusScale);
        const float candidateB = std::max(1.0f, candidate.bRadius * radiusScale);
        const float candidateC = std::max(1.0f, candidate.minorRadius * radiusScale);

        bool tooClose = false;
        size_t nearestIndex = std::numeric_limits<size_t>::max();
        float nearestDistanceSq = std::numeric_limits<float>::max();
        for (size_t existingIdx = 0; existingIdx < frame.cells.size(); ++existingIdx) {
            const cv::Point3f existingCenter(frame.cells[existingIdx].getX(),
                                             frame.cells[existingIdx].getY(),
                                             frame.cells[existingIdx].getZ());
            const float d2 = squaredDistance(candidate.centerScaled, existingCenter);
            if (d2 < nearestDistanceSq) {
                nearestDistanceSq = d2;
                nearestIndex = existingIdx;
            }
            if (d2 < minDistanceSq) {
                tooClose = true;
            }
        }
        if (tooClose) {
            const float centerPriorMaxDistance =
                std::max(0.0f, lumenConfig.fusionCenterPriorMaxDistance);
            const float centerPriorMaxDistanceSq =
                centerPriorMaxDistance * centerPriorMaxDistance;
            if (lumenConfig.fusionCenterPriorEnabled &&
                nearestIndex != std::numeric_limits<size_t>::max() &&
                nearestIndex < centerPriorUsed.size() &&
                !centerPriorUsed[nearestIndex] &&
                nearestDistanceSq <= centerPriorMaxDistanceSq) {
                Ellipsoid &target = frame.cells[nearestIndex];
                const float posBlend = std::clamp(lumenConfig.fusionCenterPriorPositionBlend,
                                                  0.0f, 1.0f);
                const float radiusBlend = std::clamp(lumenConfig.fusionCenterPriorRadiusBlend,
                                                     0.0f, 1.0f);
                const float oldX = target.getX();
                const float oldY = target.getY();
                const float oldZ = target.getZ();
                const float oldA = target.getARadius();
                const float oldB = target.getBRadius();
                const float oldC = target.getCRadius();

                target.setPosition(oldX * (1.0f - posBlend) +
                                       candidate.centerScaled.x * posBlend,
                                   oldY * (1.0f - posBlend) +
                                       candidate.centerScaled.y * posBlend,
                                   oldZ * (1.0f - posBlend) +
                                       candidate.centerScaled.z * posBlend);
                target.setRadii(std::max(1.0f, oldA * (1.0f - radiusBlend) +
                                                    candidateA * radiusBlend),
                                std::max(1.0f, oldB * (1.0f - radiusBlend) +
                                                    candidateB * radiusBlend),
                                std::max(1.0f, oldC * (1.0f - radiusBlend) +
                                                    candidateC * radiusBlend));

                const std::string &targetName = target.getName();
                const std::array<float, 3> guidedRadii{
                    target.getARadius(), target.getBRadius(), target.getCRadius()};
                cellShapeBirth[targetName] = guidedRadii;
                cellShapeReference[targetName] = guidedRadii;
                centerPriorUsed[nearestIndex] = true;
                ++centerPriorGuided;
                std::cout << "[CellLumen Fusion CenterPrior] frame=" << absoluteFrame
                          << " target=" << targetName
                          << " candidate_center=(" << candidate.centerScaled.x << ","
                          << candidate.centerScaled.y << "," << candidate.centerScaled.z << ")"
                          << " old_center=(" << oldX << "," << oldY << "," << oldZ << ")"
                          << " new_center=(" << target.getX() << ","
                          << target.getY() << "," << target.getZ() << ")"
                          << " old_radii=(" << oldA << "," << oldB << "," << oldC << ")"
                          << " new_radii=(" << target.getARadius() << ","
                          << target.getBRadius() << "," << target.getCRadius() << ")"
                          << " distance=" << std::sqrt(nearestDistanceSq)
                          << " vox=" << candidate.voxelCount
                          << " top10MinusShell=" << candidate.top10MinusShell
                          << std::endl;
            }

            const float repairMaxDistance =
                std::max(0.0f, lumenConfig.fusionRepairMaxDistance);
            const float repairMaxDistanceSq = repairMaxDistance * repairMaxDistance;
            if (lumenConfig.fusionRepairCloseCellsEnabled &&
                nearestIndex != std::numeric_limits<size_t>::max() &&
                nearestDistanceSq <= repairMaxDistanceSq) {
                Ellipsoid &target = frame.cells[nearestIndex];
                const float minCandidateRadius = std::min({candidateA, candidateB, candidateC});
                const float repairRatio = std::clamp(lumenConfig.fusionRepairRadiusRatio,
                                                     0.0f, 1.5f);
                const bool radiusLooksCollapsed =
                    target.getMinorRadius() < minCandidateRadius * repairRatio ||
                    target.getARadius() < candidateA * repairRatio ||
                    target.getBRadius() < candidateB * repairRatio ||
                    target.getCRadius() < candidateC * repairRatio;
                if (radiusLooksCollapsed) {
                    const float posBlend = std::clamp(lumenConfig.fusionRepairPositionBlend,
                                                      0.0f, 1.0f);
                    const float radiusBlend = std::clamp(lumenConfig.fusionRepairRadiusBlend,
                                                         0.0f, 1.0f);
                    const float oldX = target.getX();
                    const float oldY = target.getY();
                    const float oldZ = target.getZ();
                    target.setPosition(oldX * (1.0f - posBlend) +
                                           candidate.centerScaled.x * posBlend,
                                       oldY * (1.0f - posBlend) +
                                           candidate.centerScaled.y * posBlend,
                                       oldZ * (1.0f - posBlend) +
                                           candidate.centerScaled.z * posBlend);
                    const float newA = std::max(target.getARadius(),
                                                target.getARadius() * (1.0f - radiusBlend) +
                                                    candidateA * radiusBlend);
                    const float newB = std::max(target.getBRadius(),
                                                target.getBRadius() * (1.0f - radiusBlend) +
                                                    candidateB * radiusBlend);
                    const float newC = std::max(target.getCRadius(),
                                                target.getCRadius() * (1.0f - radiusBlend) +
                                                    candidateC * radiusBlend);
                    target.setRadii(newA, newB, newC);

                    const std::string &targetName = target.getName();
                    const std::array<float, 3> repairedRadii{
                        target.getARadius(), target.getBRadius(), target.getCRadius()};
                    cellShapeBirth[targetName] = repairedRadii;
                    cellShapeReference[targetName] = repairedRadii;
                    ++repairedClose;
                    std::cout << "[CellLumen Fusion Repair] frame=" << absoluteFrame
                              << " target=" << targetName
                              << " candidate_center=(" << candidate.centerScaled.x << ","
                              << candidate.centerScaled.y << "," << candidate.centerScaled.z << ")"
                              << " new_center=(" << target.getX() << ","
                              << target.getY() << "," << target.getZ() << ")"
                              << " new_radii=(" << target.getARadius() << ","
                              << target.getBRadius() << "," << target.getCRadius() << ")"
                              << " distance=" << std::sqrt(nearestDistanceSq)
                              << " vox=" << candidate.voxelCount
                              << " top10MinusShell=" << candidate.top10MinusShell
                              << std::endl;
                }
            }
            ++rejectedClose;
            continue;
        }

        const std::string name = lumenConfig.fusionNamePrefix + "_" +
                                 std::to_string(absoluteFrame) + "_" +
                                 std::to_string(added + 1);
        EllipsoidParams params(name,
                               candidate.centerScaled.x,
                               candidate.centerScaled.y,
                               candidate.centerScaled.z,
                               candidateA,
                               candidateC,
                               0.0f,
                               0.0f,
                               0.0f,
                               brightness);
        params.bRadius = candidateB;

        Ellipsoid rescuedCell(params);
        frame.cells.push_back(rescuedCell);
        centerPriorUsed.push_back(true);
        cellShapeBirth[name] = {rescuedCell.getARadius(),
                                rescuedCell.getBRadius(),
                                rescuedCell.getCRadius()};
        cellShapeReference[name] = cellShapeBirth[name];
        ++added;

        std::cout << "[CellLumen Fusion Add] frame=" << absoluteFrame
                  << " name=" << name
                  << " center=(" << candidate.centerScaled.x << ","
                  << candidate.centerScaled.y << "," << candidate.centerScaled.z << ")"
                  << " radii=(" << rescuedCell.getARadius() << ","
                  << rescuedCell.getBRadius() << "," << rescuedCell.getCRadius() << ")"
                  << " vox=" << candidate.voxelCount
                  << " top10MinusShell=" << candidate.top10MinusShell
                  << std::endl;
    }

    std::cout << "[CellLumen Fusion Summary] frame=" << absoluteFrame
              << " candidates=" << candidates.size()
              << " added=" << added
              << " rejected_size=" << rejectedSize
              << " rejected_signal=" << rejectedSignal
              << " rejected_close=" << rejectedClose
              << " center_prior_guided=" << centerPriorGuided
              << " repaired_close=" << repairedClose
              << " existing_after=" << frame.cells.size()
              << std::endl;
}

int CellUniverse::rollingPreprocessWindowSize() const
{
    if (config.simulation.celluniverse2_enabled &&
        config.prob.pca_bridge_future_window_enabled &&
        config.prob.pca_bridge_future_window_size > 1)
    {
        return std::clamp(config.prob.pca_bridge_future_window_size, 2, 5);
    }

    const CellLumenConfig &lumenConfig = config.cellLumen;
    if (lumenConfig.enabled &&
        lumenConfig.fusionEnabled &&
        lumenConfig.fusionSplitPriorWindowEnabled &&
        lumenConfig.fusionSplitPriorWindowSize > 1)
    {
        return std::clamp(lumenConfig.fusionSplitPriorWindowSize, 2, 5);
    }
    return 1;
}

CellUniverse::PreparedFrameStack CellUniverse::loadPreparedFrameStack(int frameIndex)
{
    if (frameIndex < 0 || static_cast<size_t>(frameIndex) >= frames.size())
    {
        throw std::invalid_argument("loadPreparedFrameStack: invalid frame index");
    }

    std::ostringstream prepareLog;
    PreparedFrameStack prepared;
    BaseConfig preprocessConfig = configForPreprocessing(frameIndex);
    if (config.simulation.n2v2_contrast_gamma_split_decay_enabled) {
        std::cout << "[Gamma Split Decay] frame=" << (firstFrame + frameIndex)
                  << " accepted_splits_total=" << cumulativeAcceptedSplits
                  << " base=" << config.simulation.n2v2_contrast_gamma
                  << " effective="
                  << preprocessConfig.simulation.n2v2_contrast_gamma
                  << " target="
                  << config.simulation.n2v2_contrast_gamma_split_decay_target
                  << " splits_to_target="
                  << config.simulation.n2v2_contrast_gamma_split_decay_splits_to_target
                  << '\n';
    }
    prepared.realFrame =
        ImageHandler::loadFrame(imagePaths[static_cast<size_t>(frameIndex)].string(),
                                preprocessConfig,
                                &prepareLog);
    std::cout << prepareLog.str();
    if (usesIterativePostPreprocessing(preprocessConfig.simulation)) {
        if (preprocessConfig.simulation.edge_brightness_alignment_enabled &&
            !edgeBrightnessAlignmentTargetInitialized) {
            edgeBrightnessAlignmentTarget =
                computeEdgeBrightnessMean(prepared.realFrame, preprocessConfig.simulation);
            edgeBrightnessAlignmentTargetInitialized = true;
            std::cout << "[EdgeBrightnessAlignment] target_initialized_from="
                      << imagePaths[static_cast<size_t>(frameIndex)].filename().string()
                      << " target=" << edgeBrightnessAlignmentTarget << '\n';
        }
        alignStackToEdgeBrightness(prepared.realFrame,
                                   preprocessConfig.simulation,
                                   edgeBrightnessAlignmentTarget,
                                   imagePaths[static_cast<size_t>(frameIndex)],
                                   std::cout);
        blackThresholdStackAfterAlignment(prepared.realFrame,
                                          preprocessConfig.simulation,
                                          imagePaths[static_cast<size_t>(frameIndex)],
                                          std::cout);
        const std::vector<cv::Mat> unblackoffedFrame = cloneMatStack(prepared.realFrame);
        blackPercentileStackAfterAlignment(prepared.realFrame,
                                           preprocessConfig.simulation,
                                           imagePaths[static_cast<size_t>(frameIndex)],
                                           std::cout);
        adaptBlackPercentileToChunkCount(prepared.realFrame,
                                         unblackoffedFrame,
                                         preprocessConfig.simulation,
                                         imagePaths[static_cast<size_t>(frameIndex)],
                                         std::cout);
        removeTinyIsolatedParticles(prepared.realFrame,
                                    preprocessConfig,
                                    imagePaths[static_cast<size_t>(frameIndex)],
                                    std::cout);
        applyFinalPreprocessingBlur(prepared.realFrame,
                                    preprocessConfig.simulation,
                                    imagePaths[static_cast<size_t>(frameIndex)],
                                    std::cout);
    } else {
        std::cout << "[PostPreprocess] frame="
                  << imagePaths[static_cast<size_t>(frameIndex)].filename().string()
                  << " skipped_for_mode=" << preprocessConfig.simulation.preprocess_mode
                  << '\n';
    }
    prepared.signalMap = buildSignalMapStack(
        prepared.realFrame,
        preprocessConfig.simulation,
        imagePaths[static_cast<size_t>(frameIndex)],
        std::cout);

    if (preprocessConfig.simulation.export_preprocessed_images) {
        exportPreprocessedStack(prepared.realFrame, fs::path(outputPath),
                                imagePaths[static_cast<size_t>(frameIndex)],
                                preprocessConfig.simulation.export_frame_png,
                                preprocessConfig.simulation.export_frame_tiff);
    }

    return prepared;
}

void CellUniverse::cachePreparedFrameStack(int frameIndex)
{
    if (frameIndex < 0 || static_cast<size_t>(frameIndex) >= frames.size())
    {
        return;
    }
    if (frames[frameIndex].hasImageStacks() ||
        preparedFrameStacks.find(frameIndex) != preparedFrameStacks.end())
    {
        return;
    }

    std::cout << "[RollingPreprocess] frame=" << (firstFrame + frameIndex)
              << " source=" << imagePaths[static_cast<size_t>(frameIndex)].filename().string()
              << " action=load" << std::endl;
    preparedFrameStacks.emplace(frameIndex, loadPreparedFrameStack(frameIndex));
}

void CellUniverse::prepareFrameWindow(int frameIndex)
{
    if (frameIndex < 0 || static_cast<size_t>(frameIndex) >= frames.size())
    {
        throw std::invalid_argument("prepareFrameWindow: invalid frame index");
    }

    const int windowSize = rollingPreprocessWindowSize();
    for (int offset = 0; offset < windowSize; ++offset)
    {
        const int targetFrame = frameIndex + offset;
        if (static_cast<size_t>(targetFrame) >= frames.size())
        {
            break;
        }
        cachePreparedFrameStack(targetFrame);
    }
}

void CellUniverse::prepareFrame(int frameIndex)
{
    if (frameIndex < 0 || static_cast<size_t>(frameIndex) >= frames.size())
    {
        throw std::invalid_argument("prepareFrame: invalid frame index");
    }
    if (frames[frameIndex].hasImageStacks()) {
        return;  // already loaded
    }

    PreparedFrameStack prepared;
    auto cached = preparedFrameStacks.find(frameIndex);
    if (cached != preparedFrameStacks.end())
    {
        prepared = std::move(cached->second);
        preparedFrameStacks.erase(cached);
        std::cout << "[RollingPreprocess] frame=" << (firstFrame + frameIndex)
                  << " action=consume_cached" << std::endl;
    }
    else
    {
        prepared = loadPreparedFrameStack(frameIndex);
    }

    {
        ScopedStageTimer timer(firstFrame + frameIndex,
                               "cell_lumen_fusion_prepare");
        applyCellLumenRescue(frameIndex, prepared.realFrame);
    }

    // loadImageStacks already generates _synthFrame + refreshes the full-image
    // cost cache. The previous `if (config.cell) regenerateSynthFrame()` was
    // redundant work (2x render + 2x cost cache per frame).
    frames[frameIndex].loadImageStacks(prepared.realFrame);
    frames[frameIndex].setSignalMap(std::move(prepared.signalMap));
    prepareSignalCentersForFrame(frameIndex, prepared.realFrame, true);
}

void CellUniverse::prepareSignalCentersForFrame(int frameIndex,
                                                const std::vector<cv::Mat> &realFrame,
                                                bool keepLoaded)
{
    if (frameIndex < 0 || static_cast<size_t>(frameIndex) >= frames.size())
    {
        throw std::invalid_argument("prepareSignalCentersForFrame: invalid frame index");
    }

    Frame &frame = frames[static_cast<size_t>(frameIndex)];
    if (!frame.hasImageStacks()) {
        frame.loadImageStacks(realFrame);
    }

    if (config.simulation.signal_guided_position_enabled ||
        config.simulation.signal_center_rescue_enabled ||
        config.simulation.export_signal_debug_images) {
        std::vector<Frame::SignalCenter> centers = localizeSignalCentersForFrame(
            frame, config, firstFrame + frameIndex);
        std::vector<cv::Mat> probability;
        if (config.simulation.signal_guided_position_enabled ||
            config.simulation.export_signal_debug_images) {
            probability = buildSignalProbabilityStack(realFrame, centers, config);
            exportSignalDebugStacks(realFrame, centers, probability, config,
                                    fs::path(outputPath),
                                    imagePaths[static_cast<size_t>(frameIndex)]);
        }
        if (config.simulation.signal_guided_position_enabled ||
            config.simulation.signal_center_rescue_enabled) {
            frame.setSignalCenters(std::move(centers));
            if (config.simulation.signal_guided_position_enabled) {
                frame.setSignalProbability(std::move(probability));
            } else {
                frame.setSignalProbability({});
            }
        } else {
            frame.setSignalCenters({});
            frame.setSignalProbability({});
        }
    } else {
        frame.setSignalCenters({});
        frame.setSignalProbability({});
    }

    if (!keepLoaded) {
        frame.releaseImageStacks();
    }
}

void CellUniverse::preprocessAllFramesAlignedToMinimumBackground(bool loadIntoFrames)
{
    struct PreprocessedFrame
    {
        std::vector<cv::Mat> stack;
        float sampledBackground = 0.0f;
    };

    std::vector<PreprocessedFrame> preprocessedFrames(imagePaths.size());
    float minimumSampledBackground = std::numeric_limits<float>::infinity();
    const int frameCount = static_cast<int>(imagePaths.size());
    const bool heavyPostPreprocessing = usesIterativePostPreprocessing(config.simulation);
    const bool alignmentEnabled =
        heavyPostPreprocessing && config.simulation.edge_brightness_alignment_enabled;

    for (int frameIndex = 0; frameIndex < frameCount; ++frameIndex)
    {
        std::ostringstream prepareLog;
        std::vector<cv::Mat> realFrame =
            ImageHandler::loadFrame(imagePaths[static_cast<size_t>(frameIndex)].string(),
                                    config,
                                    &prepareLog);
        std::cout << prepareLog.str();

        if (!alignmentEnabled) {
            if (heavyPostPreprocessing) {
                blackThresholdStackAfterAlignment(realFrame,
                                                  config.simulation,
                                                  imagePaths[static_cast<size_t>(frameIndex)],
                                                  std::cout);
                const std::vector<cv::Mat> unblackoffedFrame = cloneMatStack(realFrame);
                blackPercentileStackAfterAlignment(realFrame,
                                                   config.simulation,
                                                   imagePaths[static_cast<size_t>(frameIndex)],
                                                   std::cout);
                adaptBlackPercentileToChunkCount(realFrame,
                                                 unblackoffedFrame,
                                                 config.simulation,
                                                 imagePaths[static_cast<size_t>(frameIndex)],
                                                 std::cout);
                removeTinyIsolatedParticles(realFrame,
                                            config,
                                            imagePaths[static_cast<size_t>(frameIndex)],
                                            std::cout);
                applyFinalPreprocessingBlur(realFrame,
                                            config.simulation,
                                            imagePaths[static_cast<size_t>(frameIndex)],
                                            std::cout);
            } else {
                std::cout << "[PostPreprocess] frame="
                          << imagePaths[static_cast<size_t>(frameIndex)].filename().string()
                          << " skipped_for_mode=" << config.simulation.preprocess_mode
                          << '\n';
            }
            std::vector<cv::Mat> signalMap = buildSignalMapStack(
                realFrame,
                config.simulation,
                imagePaths[static_cast<size_t>(frameIndex)],
                std::cout);

            if (config.simulation.export_preprocessed_images) {
                exportPreprocessedStack(realFrame, fs::path(outputPath),
                                        imagePaths[static_cast<size_t>(frameIndex)],
                                        config.simulation.export_frame_png,
                                        config.simulation.export_frame_tiff);
            }

            if (loadIntoFrames) {
                frames[static_cast<size_t>(frameIndex)].loadImageStacks(realFrame);
                frames[static_cast<size_t>(frameIndex)].setSignalMap(std::move(signalMap));
                prepareSignalCentersForFrame(frameIndex, realFrame, true);
            } else if (config.simulation.export_signal_debug_images) {
                prepareSignalCentersForFrame(frameIndex, realFrame, false);
            }
            continue;
        }

        const float sampledBackground =
            computeEdgeBrightnessMean(realFrame, config.simulation);
        minimumSampledBackground =
            std::min(minimumSampledBackground, sampledBackground);
        preprocessedFrames[static_cast<size_t>(frameIndex)] = {
            std::move(realFrame),
            sampledBackground
        };

        std::cout << "[EdgeBrightnessAlignment] frame="
                  << imagePaths[static_cast<size_t>(frameIndex)].filename().string()
                  << " sampled_background=" << sampledBackground
                  << " pending_minimum_background=1"
                  << std::endl;
    }

    if (!alignmentEnabled) {
        return;
    }

    if (!std::isfinite(minimumSampledBackground))
    {
        minimumSampledBackground = 0.0f;
    }

    edgeBrightnessAlignmentTarget = minimumSampledBackground;
    edgeBrightnessAlignmentTargetInitialized = true;
    std::cout << "[EdgeBrightnessAlignment] batch_target=min_sampled_background"
              << " target=" << edgeBrightnessAlignmentTarget
              << " frame_count=" << preprocessedFrames.size()
              << std::endl;

    for (int frameIndex = 0; frameIndex < static_cast<int>(preprocessedFrames.size()); ++frameIndex)
    {
        auto &item = preprocessedFrames[static_cast<size_t>(frameIndex)];
        alignStackToEdgeBrightness(item.stack,
                                   config.simulation,
                                   edgeBrightnessAlignmentTarget,
                                   imagePaths[static_cast<size_t>(frameIndex)],
                                   std::cout);
        blackThresholdStackAfterAlignment(item.stack,
                                          config.simulation,
                                          imagePaths[static_cast<size_t>(frameIndex)],
                                          std::cout);
        const std::vector<cv::Mat> unblackoffedFrame = cloneMatStack(item.stack);
        blackPercentileStackAfterAlignment(item.stack,
                                           config.simulation,
                                           imagePaths[static_cast<size_t>(frameIndex)],
                                           std::cout);
        adaptBlackPercentileToChunkCount(item.stack,
                                         unblackoffedFrame,
                                         config.simulation,
                                         imagePaths[static_cast<size_t>(frameIndex)],
                                         std::cout);
        removeTinyIsolatedParticles(item.stack,
                                    config,
                                    imagePaths[static_cast<size_t>(frameIndex)],
                                    std::cout);
        applyFinalPreprocessingBlur(item.stack,
                                    config.simulation,
                                    imagePaths[static_cast<size_t>(frameIndex)],
                                    std::cout);
        std::vector<cv::Mat> signalMap = buildSignalMapStack(
            item.stack,
            config.simulation,
            imagePaths[static_cast<size_t>(frameIndex)],
            std::cout);

        if (config.simulation.export_preprocessed_images) {
            exportPreprocessedStack(item.stack, fs::path(outputPath),
                                    imagePaths[static_cast<size_t>(frameIndex)],
                                    config.simulation.export_frame_png,
                                    config.simulation.export_frame_tiff);
        }

        if (loadIntoFrames) {
            frames[static_cast<size_t>(frameIndex)].loadImageStacks(item.stack);
            frames[static_cast<size_t>(frameIndex)].setSignalMap(std::move(signalMap));
            prepareSignalCentersForFrame(frameIndex, item.stack, true);
        } else if (config.simulation.export_signal_debug_images) {
            prepareSignalCentersForFrame(frameIndex, item.stack, false);
        }
    }
}

const std::vector<CellUniverse::CellLumenLookaheadCandidate> &
CellUniverse::getCellLumenLookaheadCandidates(int frameIndex)
{
    static const std::vector<CellLumenLookaheadCandidate> empty;
    if (frameIndex < 0 || static_cast<size_t>(frameIndex) >= imagePaths.size()) {
        return empty;
    }

    const auto cached = cellLumenLookaheadCandidates.find(frameIndex);
    if (cached != cellLumenLookaheadCandidates.end()) {
        return cached->second;
    }

    auto &stored = cellLumenLookaheadCandidates[frameIndex];
    const CellLumenConfig &lumenConfig = config.cellLumen;
    const int minVoxels =
        lumenConfig.fusionSplitPriorMinVoxels >= 0
            ? lumenConfig.fusionSplitPriorMinVoxels
            : lumenConfig.fusionMinVoxels;
    const float minSignal =
        lumenConfig.fusionSplitPriorMinTop10MinusShell >= 0.0f
            ? lumenConfig.fusionSplitPriorMinTop10MinusShell
            : lumenConfig.fusionMinTop10MinusShell;

    EllipsoidConfig savedCellConfig = Ellipsoid::cellConfig;
    try {
        CellLumen lumen(config, fs::path(outputPath) / "cell_lumen_fusion_window");
        std::vector<CellLumen::DetectedCell> detected;
        std::string source = "direct_load";
        const auto prepared = preparedFrameStacks.find(frameIndex);
        if (prepared != preparedFrameStacks.end() &&
            !prepared->second.realFrame.empty())
        {
            detected = lumen.detectCellsInPreparedStack(
                prepared->second.realFrame,
                imagePaths[static_cast<size_t>(frameIndex)],
                false);
            source = "rolling_prepared_cache";
        }
        else if (static_cast<size_t>(frameIndex) < frames.size() &&
                 frames[frameIndex].hasImageStacks())
        {
            detected = lumen.detectCellsInPreparedStack(
                frames[frameIndex].getRealFrame(),
                imagePaths[static_cast<size_t>(frameIndex)],
                false);
            source = "prepared_frame";
        }
        else
        {
            detected = lumen.detectCellsForFrame(imagePaths[static_cast<size_t>(frameIndex)],
                                                 false,
                                                 false);
        }
        int candidateId = 0;
        for (const auto &candidate : detected) {
            if (candidate.voxelCount < minVoxels ||
                candidate.top10MinusShell < minSignal) {
                ++candidateId;
                continue;
            }
            stored.push_back({
                candidate.centerScaled,
                candidate.voxelCount,
                candidate.top10MinusShell,
                candidateId
            });
            ++candidateId;
        }
        std::sort(stored.begin(), stored.end(),
                  [](const CellLumenLookaheadCandidate &a,
                     const CellLumenLookaheadCandidate &b) {
                      if (a.signal != b.signal) {
                          return a.signal > b.signal;
                      }
                      return a.voxelCount > b.voxelCount;
                  });
        std::cout << "[CellLumen Window Candidates] frame=" << (firstFrame + frameIndex)
                  << " candidates=" << stored.size()
                  << " source=" << source
                  << " min_voxels=" << minVoxels
                  << " min_signal=" << minSignal
                  << std::endl;
    } catch (const std::exception &ex) {
        std::cout << "[CellLumen Window Candidates] frame=" << (firstFrame + frameIndex)
                  << " status=skipped error=\"" << ex.what() << "\""
                  << std::endl;
    }
    Ellipsoid::cellConfig = savedCellConfig;
    return stored;
}

void CellUniverse::optimize(int frameIndex)
{
    if (frameIndex < 0 || static_cast<size_t>(frameIndex) >= frames.size())
    {
        throw std::invalid_argument("Invalid frame index");
    }

    Frame &frame = frames[frameIndex];
    const int absoluteFrame = firstFrame + frameIndex;

    const bool hasPreviousFrameSummary =
        frameIndex > 0 ||
        (frameIndex == 0 && resumePreviousFrameSummaryValid);
    if (hasPreviousFrameSummary) {
        // Prev-frame summaries cached at end of optimize(frameIndex-1) so we
        // don't need frames[frameIndex-1]'s image stacks. Falls back to 0 if
        // not cached (shouldn't happen if prepareFrame + optimize sequence is
        // respected).
        const float previousBackground = (frameIndex > 0)
            ? perFrameAdaptiveBackground[frameIndex - 1]
            : resumePreviousAdaptiveBackground;
        const float previousMeanBrightness = (frameIndex > 0)
            ? perFrameMeanBrightness[frameIndex - 1]
            : resumePreviousMeanBrightness;
        const float currentMeanBrightness = computeStackMean(frame.getRealFrame());
        const float brightnessScale =
            (previousMeanBrightness > 1e-6f) ? (currentMeanBrightness / previousMeanBrightness) : 1.0f;
        const float updatedBackground = previousBackground * brightnessScale;

        frame.setBackgroundColor(updatedBackground);
        std::cout << "[Adaptive Background] frame " << absoluteFrame
                  << " base=" << previousBackground
                  << " ratio=" << brightnessScale
                  << " background=" << updatedBackground << '\n';
    }

    // Propagate the universal bbox cost flag to the frame BEFORE
    // regenerateSynthFrame so the regenerate can skip the full-image L2
    // cache (refreshFullCostCache) when bbox cost is active — saves ~32M
    // pixel ops per frame for frames 2+.
    //
    // Frame 1 exception: there are no previous-frame snapshots, so bbox
    // mode cannot be snap-anchored (Option A fallback would make every
    // cell use a follow-the-cell bbox and lose the position anchor that's
    // the whole point of bbox cost). Use full-image L2 for frame 1 instead —
    // it's slower but gives the correct global anchoring while cells settle
    // into their initial fit. Frames 2+ use bbox with snap-anchoring.
    const bool bboxActiveThisFrame =
        config.prob.use_bbox_cost &&
        absoluteFrame > 0 &&
        (frameIndex > 0 || !previousSnapshots.empty());
    frame.setUseBboxCost(bboxActiveThisFrame,
                         config.prob.bbox_margin_scale);

    frame.regenerateSynthFrame();

    // Signal-guided perturbation activation (yp ffc1917 — full per-frame design).
    // When enabled and enough signal centers are detected (>= previous frame
    // cell count), the prepared frame uses signal-guided perturbation with
    // signal_guided_iterations_per_cell. When disabled or not enough centers,
    // fall back to random perturbation with random_iterations_per_cell.
    bool useSignalGuidanceThisFrame = false;
    if (config.simulation.signal_guided_position_enabled) {
        const auto &centers = frame.getSignalCenters();
        useSignalGuidanceThisFrame = true;
        const bool hasPreviousCellCount =
            frameIndex > 0 ||
            (frameIndex == 0 && resumePreviousFrameSummaryValid);
        if (hasPreviousCellCount) {
            const size_t previousCellCount = (frameIndex > 0)
                ? frames[frameIndex - 1].cells.size()
                : resumePreviousCellCount;
            if (centers.size() < previousCellCount) {
                useSignalGuidanceThisFrame = false;
                std::cout << "[Signal Guidance Fallback] frame " << absoluteFrame
                          << " centers=" << centers.size()
                          << " previousCells=" << previousCellCount
                          << " mode=random\n";
            }
        }
    }

    const int guidedPerCellIters =
        (config.simulation.signal_guided_iterations_per_cell >= 0)
            ? config.simulation.signal_guided_iterations_per_cell
            : config.simulation.iterations_per_cell;
    const int randomPerCellIters =
        (config.simulation.random_iterations_per_cell >= 0)
            ? config.simulation.random_iterations_per_cell
            : config.simulation.iterations_per_cell;
    const auto lumenSplitPriorForFrameIt = cellLumenSplitPriors.find(frameIndex);
    const bool fusionSplitPriorPreparedForFrame =
        config.cellLumen.fusionSplitPriorEnabled &&
        lumenSplitPriorForFrameIt != cellLumenSplitPriors.end();
    const bool fusionSplitPriorForFrame =
        fusionSplitPriorPreparedForFrame &&
        !lumenSplitPriorForFrameIt->second.empty();
    int effectiveGuidedPerCellIters = guidedPerCellIters;
    int effectiveRandomPerCellIters = randomPerCellIters;
    const bool cellUniverse2Mode = config.simulation.celluniverse2_enabled;
    const bool reducedFusionPerturb =
        fusionSplitPriorPreparedForFrame &&
        config.cellLumen.fusionReducePostSplitPerturbEnabled;
    if (reducedFusionPerturb) {
        const int fusionPerturbIters =
            std::max(0, config.cellLumen.fusionPostSplitPerturbItersPerCell);
        effectiveGuidedPerCellIters = std::min(effectiveGuidedPerCellIters, fusionPerturbIters);
        effectiveRandomPerCellIters = std::min(effectiveRandomPerCellIters, fusionPerturbIters);
    }
    if (cellUniverse2Mode) {
        effectiveGuidedPerCellIters = 0;
        effectiveRandomPerCellIters = 0;
    }
    const size_t activePerCellIters = static_cast<size_t>(std::max(
        0, useSignalGuidanceThisFrame ? effectiveGuidedPerCellIters : effectiveRandomPerCellIters));
    size_t totalIterations = frame.length() * activePerCellIters;
    int displayFrame = firstFrame + frameIndex;
    ScopedStageTimer optimizeTimer(displayFrame, "optimize_total");
    const float randomPerturbRadiusRatio =
        config.cell ? config.cell->randomPerturbRadiusRatio : 0.5f;
    std::ofstream candidateGraphLog = openCandidateGraphLog(outputPath, displayFrame);

    std::cout << "[Optimize] frame " << displayFrame
              << " (" << frame.cells.size() << " cells, " << totalIterations << " iterations)"
              << " perturbMode=" << (useSignalGuidanceThisFrame ? "signal_guided" : "random")
              << " guidedItersPerCell=" << guidedPerCellIters
              << " randomItersPerCell=" << randomPerCellIters
              << " effectiveGuidedItersPerCell=" << effectiveGuidedPerCellIters
              << " effectiveRandomItersPerCell=" << effectiveRandomPerCellIters
              << " fusionPriorPrepared=" << (fusionSplitPriorPreparedForFrame ? 1 : 0)
              << " fusionPriorCount="
              << (fusionSplitPriorPreparedForFrame
                      ? static_cast<int>(lumenSplitPriorForFrameIt->second.size())
                      : 0)
              << " fusionPerturbReduced=" << (reducedFusionPerturb ? 1 : 0)
              << " celluniverse2=" << (cellUniverse2Mode ? 1 : 0)
              << " randomPerturbRadiusRatio=" << randomPerturbRadiusRatio
              << " useBboxCost=" << (bboxActiveThisFrame ? 1 : 0)
              << " bboxMarginScale=" << config.prob.bbox_margin_scale
              << (config.prob.use_bbox_cost && frameIndex == 0
                  ? " (frame-1 forced full-image)" : "")
              << std::endl;

    // Compute mean cell brightness for brightness-proportional overlap
    // scaling in perturbCell. Dim cells get lower overlap weight so the
    // penalty can't overwhelm their lower image cost signal.
    {
        float bSum = 0.0f;
        for (const auto &c : frame.cells) bSum += c.getBrightness();
        frame.setMeanCellBrightness(
            frame.cells.empty() ? 0.0f : bSum / static_cast<float>(frame.cells.size()));
    }

    auto logPerturbCandidate = [&](const std::string &source,
                                   const std::string &cellName,
                                   const Ellipsoid &cell,
                                   double costDiff,
                                   bool accepted,
                                   float radiusRatio,
                                   const std::string &note) {
        if (!candidateGraphLog) {
            return;
        }
        CandidateGraphRow row;
        row.frame = displayFrame;
        row.kind = "perturb";
        row.source = source;
        row.parent = cellName;
        row.selected = accepted ? 1 : 0;
        row.score = costDiff;
        row.imageGain = -costDiff;
        row.sep = radiusRatio;
        row.parentShape = cell.shapeElongation();
        row.d1 = cv::Point3f(cell.getX(), cell.getY(), cell.getZ());
        row.note = note;
        writeCandidateGraphRow(candidateGraphLog, row);
    };

    if (frame.cells.size() <= 24)
    {
        std::cout << "[FrameState Before] frame " << displayFrame << std::endl;
        for (const auto &cell : frame.cells)
        {
            auto p = cell.getCellParams();
            std::cout << "  " << p.name
                      << " pos=(" << p.x << "," << p.y << "," << p.z << ")"
                      << " R=(" << p.aRadius << "," << p.cRadius << ")"
                      << " theta=(" << p.theta_x << "," << p.theta_y << "," << p.theta_z << ")"
                      << " brightness=" << p.brightness
                      << std::endl;
        }
    }

    std::mt19937 &gen = cellUniverseRandomGenerator();
    std::uniform_real_distribution<float> uniform01(0.0f, 1.0f);

    const float overlapWeight = config.prob.overlap_penalty_weight;
    const float baseSplitProb = config.prob.P_split_base;

    // Resume CSVs are usually the validated state from the previous frame
    // copied onto the first requested image. Treat those cells as snapshots so
    // the first resumed frame can split. A fresh run from frame 0 still starts
    // with no split attempts because no previous biological time step exists.
    if (frameIndex == 0 && firstFrame > 0 && previousSnapshots.empty()) {
        std::cout << "[Snapshot Bootstrap] frame " << displayFrame
                  << " source=initial_cells cells=" << frame.cells.size()
                  << std::endl;
        for (const auto &cell : frame.cells) {
            auto p = cell.getCellParams();
            cv::Point3f fitSplitAxisDir;
            float fitSplitAxisLength = 0.0f;
            cell.worldSplitAxis(fitSplitAxisDir, fitSplitAxisLength);

            PreviousFrameSnapshot snap;
            snap.valid = true;
            snap.shapeElongation = cell.shapeElongation();
            snap.splitAxisDir = fitSplitAxisDir;
            snap.splitAxisLength = fitSplitAxisLength;
            snap.position = cv::Point3f(p.x, p.y, p.z);
            snap.aRadius = p.aRadius;
            snap.bRadius = p.bRadius;
            snap.cRadius = p.cRadius;
            snap.thetaX = p.theta_x;
            snap.thetaY = p.theta_y;
            snap.thetaZ = p.theta_z;
            snap.brightness = p.brightness;
            previousSnapshots[p.name] = snap;
        }
    }

    const bool frameCanAttemptSplits =
        frameIndex > 0 ||
        (frameIndex == 0 && firstFrame > 0 && !previousSnapshots.empty());
    const bool allowStandardSplits =
        !cellUniverse2Mode &&
        !config.prob.disable_split_trials &&
        frameCanAttemptSplits;
    const bool allowPcaBridgeSplitCut =
        cellUniverse2Mode &&
        config.prob.pca_bridge_split_enabled &&
        frameCanAttemptSplits;
    const bool allowSignalCenterSplit =
        cellUniverse2Mode &&
        config.prob.signal_center_split_enabled &&
        frameCanAttemptSplits;
    const bool allowSplits =
        allowStandardSplits || allowPcaBridgeSplitCut || allowSignalCenterSplit;

    // Cells that already failed a burn-in this frame — skip all further split attempts.
    std::set<std::string> splitBlacklist;
    auto cellUniverse2SplitCoolingDown = [&](const std::string &cellName) {
        if (!cellUniverse2Mode) return false;
        auto firstSeenIt = cellFirstSeenFrame.find(cellName);
        if (firstSeenIt == cellFirstSeenFrame.end()) return false;
        return displayFrame - firstSeenIt->second <= 1;
    };

    // P(split) is a LINEAR ramp on snapshot elongation from P_split_base
    // at elong=1.0 to P_split_max at elong=2.0. Clamped to [P_split_base,
    // P_split_max]. No classification threshold — every cell gets a split
    // probability proportional to how elongated its snapshot was.
    //
    //   elong = 1.0 → P(split) = P_split_base (e.g., 0.03)
    //   elong = 2.0 → P(split) = P_split_max  (e.g., 0.5)
    //   elong > 2.0 → P(split) = P_split_max  (clamped)
    //
    // Elongated cells (likely to be dividing) get tried often; round cells
    // still get a small base chance because snapshot elongation isn't a
    // perfect signal (first-gen splits often have near-round snaps).
    constexpr float kElongRefLow  = 1.0f;
    constexpr float kElongRefHigh = 2.0f;
    const float pMax = config.prob.P_split_max;
    std::map<std::string, float> splitProbabilities;

    std::cout << "[P(split)] frame " << displayFrame
              << (allowStandardSplits ? " (linear elong ramp)"
                                      : ((allowPcaBridgeSplitCut || allowSignalCenterSplit)
                                             ? " (celluniverse2 deterministic split probes)"
                                             : " (splits disabled)"))
              << " celluniverse2=" << (cellUniverse2Mode ? 1 : 0)
              << " disable_split_trials=" << (config.prob.disable_split_trials ? 1 : 0)
              << " pca_bridge_cut=" << (allowPcaBridgeSplitCut ? 1 : 0)
              << " signal_center_split=" << (allowSignalCenterSplit ? 1 : 0)
              << std::endl;
    for (const auto &cell : frame.cells) {
        auto p = cell.getCellParams();
        float snapElong = 1.0f;
        auto snapIt = previousSnapshots.find(p.name);
        if (snapIt != previousSnapshots.end() && snapIt->second.valid) {
            snapElong = snapIt->second.shapeElongation;
        }
        float ps = 0.0f;
        if (allowStandardSplits && !cell.isTrash()) {
            const float t = std::clamp(
                (snapElong - kElongRefLow) / (kElongRefHigh - kElongRefLow),
                0.0f, 1.0f);
            ps = baseSplitProb + t * (pMax - baseSplitProb);
        }
        splitProbabilities[p.name] = ps;
        std::cout << "  " << p.name << " shapeElong=" << snapElong
                  << " P(split)=" << ps << std::endl;
    }

    double residSum = 0;
    double absResidSum = 0;
    double residCount = 0;
    int splitAccepted = 0;
    int splitAttempted = 0;
    int perturbAccepted = 0;

    // ===============================================================
    // Unified pipeline (2026-04-14 — classification removed):
    //   1. Per-cell position calibration.
    //   2. Pre-pass for every cell: image-grounded D1/D2 centroids of
    //      bright pixels. Pre-pass direction + midpoint become candidates
    //      for the split attempt, alongside parent-rotation axes and the
    //      true snap center. Cost + bio + bridge gates decide acceptance.
    //   3. Unified perturb + split loop over all cells. Split attempts
    //      are gated only by P(split) which scales linearly with elong.
    //      Cells that aren't really splitting get filtered by:
    //        - bridge gate (valleyRatio ≥ 0.95 → no valley, reject)
    //        - bio gates (volume fraction, daughter size ratio, buried)
    //        - cost gate (must improve by ≥ split_cost)
    //   4. PCA shape fit with snap-mask at the final post-perturb positions.
    // ===============================================================

    // ---- Per-cell position calibration pass (frame 1 only) ----
    //
    // Calibration uses brightness centroid + perturbation refinement to
    // place each cell at its true image position. It is meaningful only
    // when no authoritative position exists from the previous frame — i.e.
    // frame 1, which has only initial CSV positions.
    //
    // From frame 2 onward, every cell has a snap from end-of-previous-frame
    // optimization. Snap is the correct parent center. Running calibration
    // on a snap-valid cell whose bright pixels span two emerging daughter
    // blobs (any cell about to divide, or any elongated cell with
    // asymmetric brightness) drifts the centroid toward the brighter blob,
    // mis-anchoring the parent for the upcoming split attempt. Empirical
    // analysis (see changelog 2026-04-14 evening "Snap-anchored parent")
    // showed snap elongation cannot reliably gate this — e9077 split at
    // snap elong 1.28 was missed by the threshold approach. Pure skip is
    // the only safe rule.
    //
    // For surviving frame-2+ cells, position adjustments for genuine
    // motion happen in the unified perturb+split loop, where each move
    // is gated by `costDiff < 0`.
    const int calibrationIters = std::max(0, config.prob.split_calibration_iterations_per_cell);
    const bool runCalibration =
        !cellUniverse2Mode &&
        (frameIndex == 0) &&
        previousSnapshots.empty() &&
        (calibrationIters > 0) &&
        !frame.cells.empty();
    if (runCalibration) {
        const float posScale = std::max(0.0f, config.prob.split_burn_in_pos_sigma_scale);

        PerturbParams savedCalX = Ellipsoid::cellConfig.x;
        PerturbParams savedCalY = Ellipsoid::cellConfig.y;
        PerturbParams savedCalZ = Ellipsoid::cellConfig.z;

        Ellipsoid::cellConfig.x.sigma = savedCalX.sigma * posScale;
        Ellipsoid::cellConfig.y.sigma = savedCalY.sigma * posScale;
        Ellipsoid::cellConfig.z.sigma = savedCalZ.sigma * posScale;

        std::cout << "[Calibration] frame " << displayFrame
                  << " cells=" << frame.cells.size()
                  << " itersPerCell=" << calibrationIters
                  << " posScale=" << posScale
                  << " (frame-1-only path)"
                  << std::endl;

        // No previous snapshots exist on frame 1; fall back to live cell
        // positions for Voronoi claim points.
        auto buildCalibrationClaimSet = [&](const std::string &selfName) -> Frame::ClaimSet {
            Frame::ClaimSet others;
            for (const auto &other : frame.cells) {
                const std::string otherName = other.getName();
                if (otherName == selfName) continue;
                others[otherName].push_back(cv::Point3f(
                    other.getX(), other.getY(), other.getZ()));
            }
            return others;
        };

        for (size_t ci = 0; ci < frame.cells.size(); ++ci) {
            const std::string calName = frame.cells[ci].getName();
            const cv::Point3f preCalPos(
                frame.cells[ci].getX(),
                frame.cells[ci].getY(),
                frame.cells[ci].getZ());

            // Step 1: analytic centroid calibration. Frame-1 cells have no
            // snap; build a synthetic snapshot from initial CSV state so
            // calibrateCellPositionViaCentroid has the position+radii
            // anchor it expects.
            PreviousFrameSnapshot bootSnap;
            bootSnap.valid = true;
            bootSnap.position = preCalPos;
            bootSnap.aRadius = frame.cells[ci].getARadius();
            bootSnap.bRadius = frame.cells[ci].getBRadius();
            bootSnap.cRadius = frame.cells[ci].getCRadius();
            bootSnap.thetaX = frame.cells[ci].getThetaX();
            bootSnap.thetaY = frame.cells[ci].getThetaY();
            bootSnap.thetaZ = frame.cells[ci].getThetaZ();

            Frame::ClaimSet calOthers = buildCalibrationClaimSet(calName);
            frame.calibrateCellPositionViaCentroid(ci, bootSnap, calOthers);

            const cv::Point3f postCentroidPos(
                frame.cells[ci].getX(),
                frame.cells[ci].getY(),
                frame.cells[ci].getZ());

            // Step 2: perturbation refinement
            int calAccepts = 0;
            for (int it = 0; it < calibrationIters; ++it) {
                auto calResult = frame.perturbCell(
                    ci, overlapWeight, useSignalGuidanceThisFrame);
                const bool calAccept = calResult.first < 0.0;
                logPerturbCandidate("calibration_refine",
                                    calName,
                                    frame.cells[ci],
                                    calResult.first,
                                    calAccept,
                                    randomPerturbRadiusRatio,
                                    "score_is_total_cost_diff");
                if (calAccept) ++calAccepts;
                calResult.second(calAccept);
            }
            const cv::Point3f postCalPos(
                frame.cells[ci].getX(),
                frame.cells[ci].getY(),
                frame.cells[ci].getZ());
            const float centroidShift = static_cast<float>(cv::norm(postCentroidPos - preCalPos));
            const float perturbDrift = static_cast<float>(cv::norm(postCalPos - postCentroidPos));
            const float totalDrift = static_cast<float>(cv::norm(postCalPos - preCalPos));
            std::cout << "  [Calibration] cell=" << calName
                      << " iters=" << calibrationIters
                      << " accepts=" << calAccepts
                      << " pre=(" << preCalPos.x << "," << preCalPos.y << "," << preCalPos.z << ")"
                      << " postCentroid=(" << postCentroidPos.x << "," << postCentroidPos.y << "," << postCentroidPos.z << ")"
                      << " post=(" << postCalPos.x << "," << postCalPos.y << "," << postCalPos.z << ")"
                      << " centroidShift=" << centroidShift
                      << " perturbDrift=" << perturbDrift
                      << " totalDrift=" << totalDrift
                      << std::endl;
        }

        Ellipsoid::cellConfig.x = savedCalX;
        Ellipsoid::cellConfig.y = savedCalY;
        Ellipsoid::cellConfig.z = savedCalZ;
    }

    std::set<std::string> pcaPositionLockedCells;
    std::unordered_map<std::string, std::string> pcaPositionLockReasons;
    std::set<std::string> pcaPositionGuardRelaxedCells;
    std::unordered_map<std::string, std::string> pcaPositionGuardRelaxReasons;
    auto lockPcaPosition = [&](const std::string &cellName,
                               const std::string &reason) {
        pcaPositionLockedCells.insert(cellName);
        auto reasonIt = pcaPositionLockReasons.find(cellName);
        if (reasonIt == pcaPositionLockReasons.end()) {
            pcaPositionLockReasons[cellName] = reason;
        } else if (reasonIt->second.find(reason) == std::string::npos) {
            reasonIt->second += "|";
            reasonIt->second += reason;
        }
    };
    auto relaxPcaPositionGuard = [&](const std::string &cellName,
                                     const std::string &reason) {
        if (cellName.empty()) {
            return;
        }
        pcaPositionGuardRelaxedCells.insert(cellName);
        auto reasonIt = pcaPositionGuardRelaxReasons.find(cellName);
        if (reasonIt == pcaPositionGuardRelaxReasons.end()) {
            pcaPositionGuardRelaxReasons[cellName] = reason;
        } else if (reasonIt->second.find(reason) == std::string::npos) {
            reasonIt->second += "|";
            reasonIt->second += reason;
        }
    };

    const float snapPositionGuardMinDrift =
        std::max(0.0f, config.prob.snap_position_guard_min_drift);
    const float snapPositionGuardMaxDrift =
        config.prob.snap_position_guard_max_drift;
    const float snapPositionGuardRadiusFraction =
        std::max(0.0f, config.prob.snap_position_guard_radius_fraction);
    const float snapPositionGuardRelaxedMaxDrift =
        config.prob.snap_position_guard_relaxed_max_drift;
    const float snapPositionGuardRelaxedRadiusFraction =
        std::max(0.0f, config.prob.snap_position_guard_relaxed_radius_fraction);
    const bool snapPositionGuardCrowdingEnabled =
        config.prob.snap_position_guard_crowding_enabled;
    const float snapPositionGuardCrowdingFraction =
        std::max(0.0f, config.prob.snap_position_guard_crowding_fraction);
    float snapPositionGuardCrowdingAverageDistance = -1.0f;
    float snapPositionGuardCrowdingLimit = -1.0f;
    if (snapPositionGuardCrowdingEnabled &&
        snapPositionGuardCrowdingFraction > 0.0f &&
        frame.cells.size() >= 2) {
        std::vector<cv::Point3f> nonTrashCenters;
        nonTrashCenters.reserve(frame.cells.size());
        for (const auto &cell : frame.cells) {
            if (cell.isTrash()) {
                continue;
            }
            nonTrashCenters.emplace_back(cell.getX(), cell.getY(), cell.getZ());
        }

        if (nonTrashCenters.size() >= 2) {
            double nearestDistanceSum = 0.0;
            for (size_t i = 0; i < nonTrashCenters.size(); ++i) {
                float nearestDistanceSq = std::numeric_limits<float>::max();
                for (size_t j = 0; j < nonTrashCenters.size(); ++j) {
                    if (i == j) {
                        continue;
                    }
                    nearestDistanceSq = std::min(
                        nearestDistanceSq,
                        squaredDistance(nonTrashCenters[i], nonTrashCenters[j]));
                }
                nearestDistanceSum += std::sqrt(nearestDistanceSq);
            }
            snapPositionGuardCrowdingAverageDistance = static_cast<float>(
                nearestDistanceSum / static_cast<double>(nonTrashCenters.size()));
            snapPositionGuardCrowdingLimit =
                snapPositionGuardCrowdingAverageDistance *
                snapPositionGuardCrowdingFraction;
            std::cout << "[Snap Position Guard Crowding] frame " << displayFrame
                      << " nonTrashCells=" << nonTrashCenters.size()
                      << " avgNearestDistance="
                      << snapPositionGuardCrowdingAverageDistance
                      << " fraction=" << snapPositionGuardCrowdingFraction
                      << " adaptiveLimit=" << snapPositionGuardCrowdingLimit
                      << std::endl;
        }
    }
    auto snapPositionGuardLimit = [&](float snapMaxR, bool relaxed) -> float {
        const float maxDrift =
            (relaxed && snapPositionGuardRelaxedMaxDrift > 0.0f)
                ? snapPositionGuardRelaxedMaxDrift
                : snapPositionGuardMaxDrift;
        if (maxDrift <= 0.0f) {
            return -1.0f;
        }
        const float radiusFraction =
            relaxed ? snapPositionGuardRelaxedRadiusFraction
                    : snapPositionGuardRadiusFraction;
        const float configuredLimit =
            std::max(snapPositionGuardMinDrift,
                     std::min(maxDrift, snapMaxR * radiusFraction));
        if (snapPositionGuardCrowdingLimit <= 0.0f) {
            return configuredLimit;
        }
        const float adaptiveLimit =
            std::max(snapPositionGuardMinDrift,
                     snapPositionGuardCrowdingLimit);
        return std::min(configuredLimit, adaptiveLimit);
    };

    auto runPcaShapeFit = [&]() {
    // ---- Per-cell iterative PCA shape fit ----
    // Runs after the current-frame perturb/split loop so the final frame
    // shape is fitted at the post-perturb positions before snapshot/export.
        const int pcaMaxIters = config.cell ? config.cell->pcaShapeMaxIters : 0;
        if (pcaMaxIters > 0 && !frame.cells.empty()) {
        const float pcaScale    = config.cell->pcaShapeRadiusScale;
        const int   pcaMin      = config.cell->pcaShapeMinPixels;
        const float maskScale   = config.cell->pcaShapeMaskScale;
        const float convR       = config.cell->pcaShapeConvergeRadius;
        const float convAng     = config.cell->pcaShapeConvergeAngleDeg;
        const bool  updatePos   = config.cell->pcaShapeUpdatePosition;
        const float posShiftCap = config.cell->pcaShapeMaxPosShiftFraction;

        std::cout << "[PCA Shape] frame " << displayFrame
                  << " cells=" << frame.cells.size()
                  << " maxIters=" << pcaMaxIters
                  << " scale=" << pcaScale
                  << " minPixels=" << pcaMin
                  << " maskScale=" << maskScale
                  << " updatePos=" << updatePos
                  << std::endl;

        // Snapshot cell positions BEFORE the parallel region to avoid a
        // data race: each OMP thread writes to frame.cells[ci] (radii,
        // rotation, and optionally position), while buildShapeClaimSet
        // reads all cells' positions. Without the snapshot, thread A
        // could read a partially-written Ellipsoid from thread B.
        // The snapshot is read-only during the parallel region.
        struct CellPosSnapshot { std::string name; float x, y, z; };
        std::vector<CellPosSnapshot> cellPosSnap;
        cellPosSnap.reserve(frame.cells.size());
        for (const auto &c : frame.cells) {
            cellPosSnap.push_back({c.getName(), c.getX(), c.getY(), c.getZ()});
        }

        auto buildShapeClaimSet = [&](const std::string &selfName) -> Frame::ClaimSet {
            Frame::ClaimSet others;
            for (const auto &snap : cellPosSnap) {
                if (snap.name == selfName) continue;
                others[snap.name].push_back(cv::Point3f(snap.x, snap.y, snap.z));
            }
            return others;
        };

        // Snap-mask shape fit: pass snapshot radii as the FIXED mask.
        // The mask stays constant across iterations, so the fit cannot
        // collapse onto one emerging daughter (no mask-feedback loop).
        // Radii are free to reach true image extent (shrink or grow).
        // Newly-born cells with no snapshot fall back to their current
        // live radii inside calibrateCellShapeViaPca.
        //
        // Parallel: cells are independent (_realFrame is read-only, each
        // thread writes a unique cells[ci], claim sets read from the
        // pre-snapshot). Per-cell logs are accumulated into per-cell
        // ostringstreams via the optional logSink parameter, then emitted
        // in cell-index order during the serial merge below — log
        // ordering is deterministic regardless of thread count.
        const bool trashPcaShapeFitEnabled =
            config.cell && config.cell->trashPcaShapeFitEnabled;
        const bool trashPcaShapeUpdatePosition =
            config.cell && config.cell->trashPcaShapeUpdatePosition;
        const float trashMaxOriginalRadiusFactor = std::max(
            1.0f,
            config.cell ? config.cell->trashPcaShapeMaxOriginalRadiusFactor : 2.0f);
        const float trashMaxElongation =
            config.cell ? config.cell->trashPcaShapeMaxElongation : 0.0f;
        if (trashPcaShapeFitEnabled) {
            for (const auto &cell : frame.cells) {
                if (!cell.isTrash()) continue;
                const std::string &name = cell.getName();
                if (cellShapeBirth.find(name) == cellShapeBirth.end()) {
                    cellShapeBirth[name] = {
                        cell.getARadius(), cell.getBRadius(), cell.getCRadius()};
                }
            }
        }
        applySignalCenterRescueBeforePca(frame,
                                         config,
                                         previousSnapshots,
                                         cellFirstSeenFrame,
                                         absoluteFrame);
        const int nCells = static_cast<int>(frame.cells.size());
        std::unordered_map<std::string, Ellipsoid> prePcaCellsByName;
        if (config.simulation.celluniverse2_enabled) {
            prePcaCellsByName.reserve(frame.cells.size());
            for (const auto &cell : frame.cells) {
                prePcaCellsByName[cell.getName()] = cell;
            }
        }
        std::vector<std::ostringstream> shapeLogs(nCells);
        #pragma omp parallel for schedule(dynamic)
        for (int ci = 0; ci < nCells; ++ci) {
            const std::string sname = frame.cells[ci].getName();
            const bool isTrashCell = frame.cells[ci].isTrash();
            const cv::Point3f prePcaPos(frame.cells[ci].getX(),
                                        frame.cells[ci].getY(),
                                        frame.cells[ci].getZ());
            const float prePcaMaxR = std::max({frame.cells[ci].getARadius(),
                                               frame.cells[ci].getBRadius(),
                                               frame.cells[ci].getCRadius()});
            if (isTrashCell && !trashPcaShapeFitEnabled) {
                shapeLogs[ci] << "  [PCA Shape] cell=" << sname
                              << " skipped=trash_fixed_size" << std::endl;
                continue;
            }
            Frame::ClaimSet others = buildShapeClaimSet(sname);

            // Mask radii: prefer the FROZEN per-cell reference (set at cell
            // birth, never updated) over the snap radii (which reflect the
            // previous frame's fit and propagate bloat). The reference is
            // captured post-fit at frame 1 for initial cells and post-refit
            // at split-accept for daughters. Falling back to snap covers
            // cells whose reference wasn't registered (shouldn't happen in
            // Mask radii: use BIRTH values (captured once, never updated).
            // The mask defines the pixel-gathering window — it should
            // always be bigger than the cell so PCA sees the full picture.
            // Birth-based mask is decoupled from the bounded ref, so it
            // can't participate in feedback loops (upward bloat or
            // downward thinning). With maskScale=3.0, mask ≈ 3× birth
            // — generous enough that the PCA fit is determined entirely
            // by the pixel distribution, not the mask boundary.
            // Bounded ref (cellShapeReference) is used ONLY for the
            // fit-side growth cap downstream.
            float maskA = 0.0f, maskB = 0.0f, maskC = 0.0f;
            auto birthIt = cellShapeBirth.find(sname);
            if (birthIt != cellShapeBirth.end()) {
                maskA = birthIt->second[0];
                maskB = birthIt->second[1] > 1e-3f ? birthIt->second[1] : maskA;
                maskC = birthIt->second[2];
            } else {
                // First appearance (frame 1 or just-born daughter):
                // no birth ref yet. Fall back to snap, then live radii.
                auto snapIt = previousSnapshots.find(sname);
                if (snapIt != previousSnapshots.end() && snapIt->second.valid) {
                    maskA = snapIt->second.aRadius;
                    maskB = snapIt->second.bRadius > 1e-3f
                            ? snapIt->second.bRadius : maskA;
                    maskC = snapIt->second.cRadius;
                }
            }

            const bool pcaPositionLocked =
                pcaPositionLockedCells.count(sname) > 0;
            const bool cellUpdatePos =
                isTrashCell ? trashPcaShapeUpdatePosition : updatePos;
            const bool updatePosForCell = cellUpdatePos && !pcaPositionLocked;
            if (pcaPositionLocked && cellUpdatePos) {
                const auto reasonIt = pcaPositionLockReasons.find(sname);
                shapeLogs[ci] << "  [PCA Shape Position Lock] cell=" << sname
                              << " reason="
                              << (reasonIt != pcaPositionLockReasons.end()
                                      ? reasonIt->second
                                      : std::string("unknown"))
                              << std::endl;
            }

            frame.calibrateCellShapeViaPca(ci, others,
                                           pcaMaxIters, pcaScale, pcaMin,
                                           maskScale, convR, convAng,
                                           updatePosForCell, posShiftCap,
                                           maskA, maskB, maskC,
                                           &shapeLogs[ci]);
            const bool hasPreviousSnapshot =
                previousSnapshots.find(sname) != previousSnapshots.end();
            if (updatePosForCell && !isTrashCell && hasPreviousSnapshot) {
                const cv::Point3f postPcaPos(frame.cells[ci].getX(),
                                             frame.cells[ci].getY(),
                                             frame.cells[ci].getZ());
                const float cumulativeMove =
                    static_cast<float>(cv::norm(postPcaPos - prePcaPos));
                const bool pcaPositionGuardRelaxed =
                    pcaPositionGuardRelaxedCells.count(sname) > 0;
                const float normalCumulativeMoveLimit =
                    snapPositionGuardLimit(prePcaMaxR, false);
                const float cumulativeMoveLimit =
                    pcaPositionGuardRelaxed
                        ? snapPositionGuardLimit(prePcaMaxR, true)
                        : normalCumulativeMoveLimit;
                if (cumulativeMoveLimit > 0.0f &&
                    cumulativeMove > cumulativeMoveLimit) {
                    frame.cells[ci].setPosition(prePcaPos.x, prePcaPos.y, prePcaPos.z);
                    shapeLogs[ci] << "  [PCA Shape Position Guard] cell=" << sname
                                  << " cumulativeMove=" << cumulativeMove
                                  << " limit=" << cumulativeMoveLimit
                                  << " action=revert_position_keep_shape"
                                  << std::endl;
                } else if (pcaPositionGuardRelaxed) {
                    const auto reasonIt =
                        pcaPositionGuardRelaxReasons.find(sname);
                    shapeLogs[ci] << "  [PCA Shape Position Guard] cell=" << sname
                                  << " cumulativeMove=" << cumulativeMove
                                  << " normalLimit="
                                  << normalCumulativeMoveLimit
                                  << " relaxedLimit=" << cumulativeMoveLimit
                                  << " action=keep_relaxed_position"
                                  << " reason="
                                  << (reasonIt !=
                                              pcaPositionGuardRelaxReasons.end()
                                          ? reasonIt->second
                                          : std::string("unknown"))
                                  << std::endl;
                }
            }
            if (isTrashCell) {
                auto originalIt = cellShapeBirth.find(sname);
                if (originalIt != cellShapeBirth.end()) {
                    const auto &original = originalIt->second;
                    const float preA = frame.cells[ci].getARadius();
                    const float preB = frame.cells[ci].getBRadius();
                    const float preC = frame.cells[ci].getCRadius();
                    const float capA = original[0] * trashMaxOriginalRadiusFactor;
                    const float capB = original[1] * trashMaxOriginalRadiusFactor;
                    const float capC = original[2] * trashMaxOriginalRadiusFactor;
                    const float newA = std::min(preA, capA);
                    const float newB = std::min(preB, capB);
                    const float newC = std::min(preC, capC);
                    if (newA != preA || newB != preB || newC != preC) {
                        frame.cells[ci].setRadii(newA, newB, newC);
                        shapeLogs[ci] << "  [Trash PCA Radius Cap] cell=" << sname
                                      << " factor=" << trashMaxOriginalRadiusFactor
                                      << " original=(" << original[0] << "," << original[1] << "," << original[2] << ")"
                                      << " pre=(" << preA << "," << preB << "," << preC << ")"
                                      << " post=(" << newA << "," << newB << "," << newC << ")"
                                      << std::endl;
                    }
                }
                const float maxTrashElong = config.cell
                    ? std::max(0.0f, config.cell->trashPcaShapeMaxElongation)
                    : 0.0f;
                if (maxTrashElong > 0.0f) {
                    const float preA = frame.cells[ci].getARadius();
                    const float preB = frame.cells[ci].getBRadius();
                    const float preC = frame.cells[ci].getCRadius();
                    const float shortR = std::max(1e-3f, std::min({preA, preB, preC}));
                    const float maxAllowed = shortR * maxTrashElong;
                    const float newA = std::min(preA, maxAllowed);
                    const float newB = std::min(preB, maxAllowed);
                    const float newC = std::min(preC, maxAllowed);
                    if (newA != preA || newB != preB || newC != preC) {
                        frame.cells[ci].setRadii(newA, newB, newC);
                        shapeLogs[ci] << "  [Trash PCA Elongation Cap] cell=" << sname
                                      << " maxElong=" << maxTrashElong
                                      << " pre=(" << preA << "," << preB << "," << preC << ")"
                                      << " post=(" << newA << "," << newB << "," << newC << ")"
                                      << std::endl;
                    }
                }
            }
        }
        // Serial merge: emit per-cell log blocks in cell-index order.
        for (int ci = 0; ci < nCells; ++ci) {
            const std::string buf = shapeLogs[ci].str();
            if (!buf.empty()) std::cout << buf;
        }

        // PCA-bridge discovery moved out of this end-of-frame lambda — it
        // now runs BEFORE the main split loop (see "PCA Bridge Propose"
        // block earlier in optimize()), so its (left, right) daughter
        // centroids can be injected as a "bridge_primary" candidate into
        // trySplitCellPhased and validated by the main path's full gate
        // stack instead of an independent accepting path.

        // ---- Fit-side growth cap (anti-bloat within mask) ----
        //
        // Bounded-growth on the REFERENCE (below) caps ref at ±5%/frame,
        // but the fit can still jump within the mask (mask = ref × 1.6
        // allows fit to grow up to 60% above ref in a single frame).
        // Observed in run 091510: e9077..a50 fit jumped aRadius 41.5 →
        // 65.2 at f20 (+57%) and hit the ceiling, cascading chaos at
        // f21/f22.
        //
        // Fit-side growth cap: two-sided bound at ref × (1 ± fitGrowthCap)
        // per frame.
        //   Up:   prevents single-frame bloat (PCA seeing halo, outputting
        //         R=60 when ref=40).
        //   Down: prevents single-frame collapse (PCA outputting a
        //         spuriously small radius due to Voronoi cut or neighbor
        //         overlap, causing e.g. 12345..0 c to crash 22→14 in one
        //         frame of run 082209, then compounding through splits
        //         to hit the 10-px floor). 10% downward still allows
        //         biological pre-split pinching over several frames.
        const bool fitGrowthCapEnabled =
            config.cell && config.cell->pcaShapeFitGrowthCapEnabled;
        const float fitGrowthCap = std::clamp(
            config.cell ? config.cell->pcaShapeFitGrowthCap : 0.10f,
            0.0f, 1.0f);
        const float fitUpFactor = 1.0f + fitGrowthCap;
        const float fitDownFactor = 1.0f - fitGrowthCap;
        int fitsClamped = 0;
        if (fitGrowthCapEnabled) {
            for (size_t ci = 0; ci < frame.cells.size(); ++ci) {
                const std::string &name = frame.cells[ci].getName();
                if (frame.cells[ci].isTrash()) continue;
                auto it = cellShapeReference.find(name);
                if (it == cellShapeReference.end()) continue;
                const auto &ref = it->second;
                const float fA = frame.cells[ci].getARadius();
                const float fB = frame.cells[ci].getBRadius();
                const float fC = frame.cells[ci].getCRadius();
                const float capUpA   = ref[0] * fitUpFactor;
                const float capUpB   = ref[1] * fitUpFactor;
                const float capUpC   = ref[2] * fitUpFactor;
                const float capDownA = ref[0] * fitDownFactor;
                const float capDownB = ref[1] * fitDownFactor;
                const float capDownC = ref[2] * fitDownFactor;
                const float newA = std::clamp(fA, capDownA, capUpA);
                const float newB = std::clamp(fB, capDownB, capUpB);
                const float newC = std::clamp(fC, capDownC, capUpC);
                if (newA != fA || newB != fB || newC != fC) {
                    frame.cells[ci].setRadii(newA, newB, newC);
                    ++fitsClamped;
                }
            }
        }
        if (fitsClamped > 0) {
            std::cout << "[Fit Growth Cap] frame " << displayFrame
                      << " clamped=" << fitsClamped
                      << " maxUp=" << fitGrowthCap << std::endl;
            // Re-render with clamped radii so downstream cost eval sees
            // the capped synth, not the pre-clamp one.
            frame.regenerateSynthFrame();
        } else {
            frame.regenerateSynthFrame();
        }

        // ---- Birth-relative growth cap (anti-bloat over long horizons) ----
        //
        // Caps each radius at `birthR × birth_growth_cap_factor`. Unlike the
        // per-frame fit growth cap above (which compounds over time since
        // cellShapeReference updates every frame), this is a cumulative
        // ceiling tied to the frozen birth radii. A cell that has existed
        // for many frames and slowly bloated (observed: a511 grew a-axis
        // from 28.6 at birth to 58.6 after 25 frames) gets caught here.
        //
        // When a bloated cell wants to grow but hits the cap, the only way
        // to reduce its image cost is to split — which is exactly the
        // biologically correct behavior.
        const float birthCapFactor = config.prob.birth_growth_cap_factor;
        const float birthCapElongThresh = config.prob.birth_growth_cap_elong_threshold;
        if (birthCapFactor > 0.0f) {
            int birthCapped = 0;
            for (size_t ci = 0; ci < frame.cells.size(); ++ci) {
                const std::string &name = frame.cells[ci].getName();
                if (frame.cells[ci].isTrash()) continue;
                auto birthIt = cellShapeBirth.find(name);
                if (birthIt == cellShapeBirth.end()) continue;  // not yet captured
                const auto &birth = birthIt->second;
                const float capA = birth[0] * birthCapFactor;
                const float capB = birth[1] * birthCapFactor;
                const float capC = birth[2] * birthCapFactor;
                const float fA = frame.cells[ci].getARadius();
                const float fB = frame.cells[ci].getBRadius();
                const float fC = frame.cells[ci].getCRadius();
                const bool radiusOver = (fA > capA || fB > capB || fC > capC);
                if (!radiusOver) continue;
                // Gate on elongation too — only classic bloat (big + elongated)
                // gets clamped. Healthy growing cells that exceed birth×factor
                // without elongation pass through unchanged.
                const float elong = frame.cells[ci].shapeElongation();
                if (elong < birthCapElongThresh) continue;
                frame.cells[ci].setRadii(std::min(fA, capA),
                                         std::min(fB, capB),
                                         std::min(fC, capC));
                ++birthCapped;
            }
            if (birthCapped > 0) {
                std::cout << "[Birth Growth Cap] frame " << displayFrame
                          << " clamped=" << birthCapped
                          << " factor=" << birthCapFactor
                          << " elongThresh=" << birthCapElongThresh << std::endl;
                frame.regenerateSynthFrame();
            }
        }

        if (config.simulation.celluniverse2_enabled) {
            auto findCellIndexByName = [&](const std::string &name) -> int {
                for (size_t ci = 0; ci < frame.cells.size(); ++ci) {
                    if (frame.cells[ci].getName() == name) {
                        return static_cast<int>(ci);
                    }
                }
                return -1;
            };

            std::set<std::string> restoredNames;
            int hardOverlapRestores = 0;
            int hardOverlapPasses = 0;
            constexpr int maxHardOverlapPasses = 64;
            std::string overlapA;
            std::string overlapB;
            float aInB = 0.0f;
            float bInA = 0.0f;
            const auto isSiblingDaughterName = [](const std::string &a,
                                                  const std::string &b) {
                if (a.size() != b.size() || a.empty()) return false;
                const char aLast = a.back();
                const char bLast = b.back();
                if (!((aLast == '0' && bLast == '1') ||
                      (aLast == '1' && bLast == '0'))) {
                    return false;
                }
                return a.substr(0, a.size() - 1) == b.substr(0, b.size() - 1);
            };
            while (hardOverlapPasses < maxHardOverlapPasses &&
                   frame.findAnyCellBodyOverlap(/*includeTrash=*/false,
                                                /*scale=*/1.0f,
                                                &overlapA, &overlapB,
                                                &aInB, &bInA)) {
                const float toleratedSiblingOverlap = std::clamp(
                    config.prob.pca_bridge_soft_daughter_overlap_max, 0.0f, 1.0f);
                if (isSiblingDaughterName(overlapA, overlapB) &&
                    std::max(aInB, bInA) <= toleratedSiblingOverlap) {
                    std::cout << "[CellUniverse2 Tolerated Sibling Overlap] frame "
                              << displayFrame
                              << " stage=post_pca"
                              << " cellA=" << overlapA
                              << " cellB=" << overlapB
                              << " aInB=" << aInB
                              << " bInA=" << bInA
                              << " tolerance=" << toleratedSiblingOverlap
                              << std::endl;
                    break;
                }
                ++hardOverlapPasses;
                bool changed = false;
                for (const std::string &name : {overlapA, overlapB}) {
                    if (restoredNames.count(name) > 0) continue;
                    auto preIt = prePcaCellsByName.find(name);
                    const int idx = findCellIndexByName(name);
                    if (preIt != prePcaCellsByName.end() && idx >= 0) {
                        frame.cells[static_cast<size_t>(idx)] = preIt->second;
                        restoredNames.insert(name);
                        ++hardOverlapRestores;
                        changed = true;
                    }
                }

                if (changed) {
                    frame.regenerateSynthFrame();
                    std::cout << "[CellUniverse2 Hard Overlap] frame "
                              << displayFrame
                              << " stage=post_pca"
                              << " pass=" << hardOverlapPasses
                              << " cellA=" << overlapA
                              << " cellB=" << overlapB
                              << " aInB=" << aInB
                              << " bInA=" << bInA
                              << " restored=" << hardOverlapRestores
                              << std::endl;
                } else {
                    std::cout << "[CellUniverse2 Hard Overlap Warning] frame "
                              << displayFrame
                              << " stage=post_pca"
                              << " unresolved_without_resize=1"
                              << " cellA=" << overlapA
                              << " cellB=" << overlapB
                              << " aInB=" << aInB
                              << " bInA=" << bInA
                              << std::endl;
                    break;
                }
            }

            if (hardOverlapPasses >= maxHardOverlapPasses) {
                std::cout << "[CellUniverse2 Hard Overlap Warning] frame "
                          << displayFrame
                          << " unresolved_after_passes="
                          << maxHardOverlapPasses
                          << std::endl;
            } else if (hardOverlapRestores > 0) {
                std::cout << "[CellUniverse2 Hard Overlap Resolved] frame "
                          << displayFrame
                          << " passes=" << hardOverlapPasses
                          << " restored=" << hardOverlapRestores
                          << std::endl;
            }
        }

        // ---- Reference-side growth cap (anti-compounding mask basis) ----
        //
        // Each frame the reference tracks toward the observed fit but is
        // capped at ±refGrowthCap per frame:
        //   ref_new[i] = clamp(fit[i], ref_old[i] * (1-cap), ref_old[i] * (1+cap))
        // This allows legitimate multi-frame growth (daughter cells growing
        // toward adult size at ~5%/frame passes through unchanged) while
        // denying instantaneous bloat (20%/frame compounding hits the cap,
        // mask can't follow, fit can't run away).
        //
        // New cells (frame 1 seeds, newly-born daughters): reference is
        // captured directly from the current fit — they don't have a
        // "previous" reference to bound against yet.
        constexpr float refGrowthCap = 0.05f;   // 5%/frame max Δ, up or down
        const float refLo = 1.0f - refGrowthCap;
        const float refHi = 1.0f + refGrowthCap;
        int refsCaptured = 0;
        int refsUpdated = 0;
        int birthsCaptured = 0;
        for (const auto &cell : frame.cells) {
            const std::string &name = cell.getName();
            const float fA = cell.getARadius();
            const float fB = cell.getBRadius();
            const float fC = cell.getCRadius();

            // Birth capture (once, never updated). Used for mask basis.
            if (cellShapeBirth.find(name) == cellShapeBirth.end()) {
                cellShapeBirth[name] = {fA, fB, fC};
                ++birthsCaptured;
            }

            auto it = cellShapeReference.find(name);
            if (it == cellShapeReference.end()) {
                cellShapeReference[name] = {fA, fB, fC};
                ++refsCaptured;
            } else {
                auto &ref = it->second;
                ref[0] = std::clamp(fA, ref[0] * refLo, ref[0] * refHi);
                ref[1] = std::clamp(fB, ref[1] * refLo, ref[1] * refHi);
                ref[2] = std::clamp(fC, ref[2] * refLo, ref[2] * refHi);
                ++refsUpdated;
            }
        }
        if (refsCaptured > 0 || refsUpdated > 0 || birthsCaptured > 0) {
            std::cout << "[Shape Reference] frame " << displayFrame
                      << " births=" << birthsCaptured
                      << " refCaptured=" << refsCaptured
                      << " refUpdated=" << refsUpdated
                      << " totalRef=" << cellShapeReference.size()
                      << " totalBirth=" << cellShapeBirth.size()
                      << " growthCap=" << refGrowthCap << std::endl;
        }
        }
    };

    // ---- Install snap-anchored per-cell bboxes (Option A) ----
    // Each cell with a valid PreviousFrameSnapshot gets a bbox fixed at
    // the snap position with half-extent = bbox_margin_scale × snapMaxR.
    // perturbCell then uses this fixed bbox for every iteration in this
    // frame — so drifting away from snap incurs an undershoot cost at the
    // abandoned snap voxels. Cells without a snap (frame 1, or newborn
    // daughters after a split accepts mid-frame) have no entry and fall
    // back to the legacy live pre/post-union bbox in perturbCell. Always
    // cleared first so stale names don't leak across frames.
    frame.clearSnapBboxes();
    frame.clearSnapPositions();
    frame.setPositionPriorWeight(config.prob.position_prior_weight);
    frame.setPositionPriorThreshold(config.prob.position_prior_threshold);
    frame.setMaxPerturbDriftXY(config.prob.max_perturb_drift_xy);
    frame.setMaxPerturbDriftZ(config.prob.max_perturb_drift_z);
    // Voronoi map is needed for EITHER the (legacy) cost-replacement filter
    // or the additive bleed penalty — enable map build when either is on.
    const bool voronoiMapNeeded =
        config.simulation.voronoi_cost_enabled ||
        (config.simulation.voronoi_bleed_penalty_enabled &&
         config.simulation.voronoi_bleed_penalty_weight > 0.0f);
    frame.enableVoronoiCost(voronoiMapNeeded);
    frame.setVoronoiBleedWeight(
        config.simulation.voronoi_bleed_penalty_enabled
            ? config.simulation.voronoi_bleed_penalty_weight
            : 0.0f);
    if (bboxActiveThisFrame) {
        int installed = 0;
        for (const auto &cell : frame.cells) {
            const std::string &name = cell.getName();
            auto snapIt = previousSnapshots.find(name);
            if (snapIt == previousSnapshots.end() || !snapIt->second.valid) continue;
            const auto &snap = snapIt->second;
            const float snapMaxR = std::max({snap.aRadius, snap.bRadius, snap.cRadius});
            if (snapMaxR <= 1e-3f) continue;
            BoundingBox3D snapBbox = frame.computeBboxAtPoint(
                snap.position, snapMaxR, config.prob.bbox_margin_scale);
            if (!snapBbox.isValid()) continue;
            frame.setSnapBbox(name, snapBbox);
            frame.setSnapPosition(name, snap.position);
            ++installed;
        }
        std::cout << "[Snap Bbox] frame " << displayFrame
                  << " installed=" << installed
                  << " total=" << frame.cells.size()
                  << " priorWeight=" << config.prob.position_prior_weight
                  << std::endl;
    }

    // Build the static-Voronoi cost territory map for this frame. Anchors
    // are snap positions (for cells with a valid snap) or live centers
    // (frame 1, newborn daughters). Pixels in each cell's bbox that are
    // closer to another anchor contribute zero to this cell's image cost
    // during perturbation — prevents bloat-induced territory annexation.
    // Rebuilt after each split accept below.
    if (voronoiMapNeeded) {
        const auto vorT0 = std::chrono::steady_clock::now();
        frame.rebuildVoronoiMap();
        const auto vorT1 = std::chrono::steady_clock::now();
        const double vorMs = std::chrono::duration<double, std::milli>(
            vorT1 - vorT0).count();
        std::cout << "[Voronoi Map] frame " << displayFrame
                  << " anchors=" << frame.cells.size()
                  << " bleed_w=" << frame.getVoronoiBleedWeight()
                  << " build_ms=" << vorMs << std::endl;
    }

    // Seed expected-daughter positions for EVERY cell with a valid
    // snapshot. The current-frame PCA (pre-pass below) needs a starting
    // point for Voronoi claims — using the snapshot's split axis is the
    // natural seed. Classification no longer gates this: non-classified
    // cells also need image-grounded midpoints when they attempt splits,
    // because snap axis is unreliable for near-round cells.
    std::map<std::string, std::pair<cv::Point3f, cv::Point3f>> expectedDaughters;
    std::map<std::string, std::pair<cv::Point3f, cv::Point3f>> snapshotSeedDaughters;
    std::map<std::string, float> expectedDaughterMaxShift;
    std::map<std::string, int> expectedDaughterKeptPixels;
    if (allowStandardSplits) {
        for (const auto &cell : frame.cells) {
            const std::string &name = cell.getName();
            auto snapIt = previousSnapshots.find(name);
            if (snapIt == previousSnapshots.end() || !snapIt->second.valid) continue;
            const auto &snap = snapIt->second;
            const float half = 0.5f * snap.splitAxisLength;
            const cv::Point3f seed1(
                snap.position.x - half * snap.splitAxisDir.x,
                snap.position.y - half * snap.splitAxisDir.y,
                snap.position.z - half * snap.splitAxisDir.z);
            const cv::Point3f seed2(
                snap.position.x + half * snap.splitAxisDir.x,
                snap.position.y + half * snap.splitAxisDir.y,
                snap.position.z + half * snap.splitAxisDir.z);
            expectedDaughters[name] = {seed1, seed2};
            snapshotSeedDaughters[name] = {seed1, seed2};
            expectedDaughterMaxShift[name] = 0.0f;
        }

        // ---- Pre-pass: image-ground the seeds for ALL cells ----
        // Runs PCA on current-frame bright pixels around each cell to find the
        // midpoint + direction connecting the two bright blobs (if any). This
        // image-grounded direction is the primary signal when snap axis is
        // unreliable (near-round snap = arbitrary axC direction). For single-
        // blob cells, PCA returns the blob's principal direction — harmless,
        // cost gate still rejects because there's no valley.
        const int prePassRounds = std::max(1, config.prob.expected_daughter_pre_pass_iterations);
        ScopedStageTimer prePassTimer(displayFrame, "pre_pass_image_grounded_pca");
        if (!frame.cells.empty()) {
            std::cout << "[Pre-Pass] frame " << displayFrame
                      << " rounds=" << prePassRounds
                      << " allCells=" << frame.cells.size()
                      << " mode=image_grounded_pca" << std::endl;
        }
        // Per-cell pre-pass result, accumulated in parallel then merged serially.
        // Parallelization is safe because:
        //  - imageGroundExpectedDaughters is const on Frame (read-only)
        //  - claim set built from `roundSnapshot` (frozen at start of round) and
        //    `frame.cells` (read-only during the parallel region)
        //  - results[ci] is uniquely written per thread
        //  - logs written into thread-local ostringstream; emitted in order
        //    after the parallel region for deterministic log order
        struct PrePassResult {
            bool present = false;
            bool ok = false;
            cv::Point3f groundedD1{0, 0, 0};
            cv::Point3f groundedD2{0, 0, 0};
            cv::Point3f oldD1{0, 0, 0};
            cv::Point3f oldD2{0, 0, 0};
            int keptPixels = 0;
            std::string claimsLog;
            std::string resultLog;
        };

    for (int round = 0; round < prePassRounds; ++round) {
        // Snapshot expectedDaughters at start of round so all parallel
        // readers see consistent neighbor seed positions.
        const auto roundSnapshot = expectedDaughters;
        std::vector<PrePassResult> results(frame.cells.size());

        const int nCells = static_cast<int>(frame.cells.size());
        #pragma omp parallel for schedule(dynamic)
        for (int ci = 0; ci < nCells; ++ci) {
            const std::string name = frame.cells[ci].getName();
            auto roundIt = roundSnapshot.find(name);
            if (roundIt == roundSnapshot.end()) continue;

            auto snapIt = previousSnapshots.find(name);
            if (snapIt == previousSnapshots.end()) continue;

            results[ci].present = true;
            results[ci].oldD1 = roundIt->second.first;
            results[ci].oldD2 = roundIt->second.second;

            const auto &snap = snapIt->second;

            // Build claim-set from the round snapshot (frozen) and live cells.
            // Every other cell contributes BOTH its D1/D2 seed when present
            // (dumbbell-aware Voronoi for dividing neighbors).
            Frame::ClaimSet others;
            for (const auto &other : frame.cells) {
                const std::string otherName = other.getName();
                if (otherName == name) continue;
                auto itD = roundSnapshot.find(otherName);
                if (itD != roundSnapshot.end()) {
                    others[otherName].push_back(itD->second.first);
                    others[otherName].push_back(itD->second.second);
                } else {
                    others[otherName].push_back(cv::Point3f(
                        other.getX(), other.getY(), other.getZ()));
                }
            }

            {
                size_t claimPointCount = 0;
                for (const auto &kv : others) {
                    claimPointCount += kv.second.size();
                }
                constexpr bool verbosePrePassClaimNames = false;
                std::ostringstream claimLog;
                if (verbosePrePassClaimNames) {
                    for (const auto &kv : others) {
                        claimLog << " " << kv.first.substr(0, 8)
                                 << "[" << kv.second.size() << "]";
                    }
                }
                std::ostringstream line;
                line << "  [Pre-Pass Claims] " << name.substr(0, 8)
                     << " nNeighbors=" << others.size()
                     << " claim_points=" << claimPointCount;
                if (verbosePrePassClaimNames) {
                    line << " claims:" << claimLog.str();
                }
                line << "\n";
                results[ci].claimsLog = line.str();
            }

            cv::Point3f groundedD1, groundedD2;
            int keptPixels = 0;
            const bool ok = frame.imageGroundExpectedDaughters(
                static_cast<size_t>(ci), snap, others, groundedD1, groundedD2, &keptPixels);

            results[ci].ok = ok;
            results[ci].groundedD1 = groundedD1;
            results[ci].groundedD2 = groundedD2;
            results[ci].keptPixels = keptPixels;

            std::ostringstream rlog;
            if (ok) {
                const cv::Point3f delta1 = groundedD1 - results[ci].oldD1;
                const cv::Point3f delta2 = groundedD2 - results[ci].oldD2;
                const float shift1 = static_cast<float>(cv::norm(delta1));
                const float shift2 = static_cast<float>(cv::norm(delta2));
                rlog << "  [Pre-Pass] round=" << round
                     << " cell=" << name
                     << " shapeElong=" << snap.shapeElongation
                     << " kept=" << keptPixels
                     << " snapPos=(" << snap.position.x << "," << snap.position.y << "," << snap.position.z << ")"
                     << " seedD1=(" << results[ci].oldD1.x << "," << results[ci].oldD1.y << "," << results[ci].oldD1.z << ")"
                     << " expD1=(" << groundedD1.x << "," << groundedD1.y << "," << groundedD1.z << ")"
                     << " shift1=" << shift1
                     << " seedD2=(" << results[ci].oldD2.x << "," << results[ci].oldD2.y << "," << results[ci].oldD2.z << ")"
                     << " expD2=(" << groundedD2.x << "," << groundedD2.y << "," << groundedD2.z << ")"
                     << " shift2=" << shift2
                     << "\n";
            } else {
                rlog << "  [Pre-Pass] round=" << round
                     << " cell=" << name
                     << " shapeElong=" << snap.shapeElongation
                     << " kept=" << keptPixels
                     << " result=pca_failed_keep_snapshot_seeds"
                     << " seedD1=(" << results[ci].oldD1.x << "," << results[ci].oldD1.y << "," << results[ci].oldD1.z << ")"
                     << " seedD2=(" << results[ci].oldD2.x << "," << results[ci].oldD2.y << "," << results[ci].oldD2.z << ")"
                     << "\n";
            }
            results[ci].resultLog = rlog.str();
        }

        // Serial merge: apply results to expectedDaughters and emit logs
        // in cell-index order for deterministic output.
        for (size_t ci = 0; ci < frame.cells.size(); ++ci) {
            if (!results[ci].present) continue;
            if (results[ci].ok) {
                const std::string &name = frame.cells[ci].getName();
                const float shift1 = static_cast<float>(
                    cv::norm(results[ci].groundedD1 - results[ci].oldD1));
                const float shift2 = static_cast<float>(
                    cv::norm(results[ci].groundedD2 - results[ci].oldD2));
                expectedDaughters[name] = {results[ci].groundedD1, results[ci].groundedD2};
                expectedDaughterKeptPixels[name] = results[ci].keptPixels;
                expectedDaughterMaxShift[name] = std::max(shift1, shift2);
            }
            std::cout << results[ci].claimsLog << results[ci].resultLog;
        }
    }
        prePassTimer.finish();
    }

    std::set<std::string> forcedSplitNames;
    if (allowStandardSplits && config.prob.expected_daughter_force_split_enabled) {
        const float configuredMinFraction =
            config.prob.expected_daughter_force_min_separation_parent_fraction;
        const float minSepFraction = configuredMinFraction > 0.0f
            ? configuredMinFraction
            : config.prob.bio_min_daughter_separation_parent_fraction;
        const int minKeptPixels =
            std::max(0, config.prob.expected_daughter_force_min_kept_pixels);

        int candidatesChecked = 0;
        for (const auto &cell : frame.cells) {
            const std::string &name = cell.getName();
            if (cell.isTrash()) continue;

            auto snapIt = previousSnapshots.find(name);
            auto dIt = expectedDaughters.find(name);
            auto keptIt = expectedDaughterKeptPixels.find(name);
            if (snapIt == previousSnapshots.end() || !snapIt->second.valid ||
                dIt == expectedDaughters.end() || keptIt == expectedDaughterKeptPixels.end()) {
                continue;
            }
            ++candidatesChecked;
            if (keptIt->second < minKeptPixels) continue;

            const auto &snap = snapIt->second;
            const cv::Point3f delta = dIt->second.second - dIt->second.first;
            const float separation = static_cast<float>(cv::norm(delta));
            const float parentMaxRadius = std::max({snap.aRadius, snap.bRadius, snap.cRadius});
            if (parentMaxRadius <= 1e-3f) continue;

            const float minSeparation = parentMaxRadius * minSepFraction;
            if (separation < minSeparation) continue;

            forcedSplitNames.insert(name);
            const float ratio = separation / parentMaxRadius;
            const float pSplit = splitProbabilities.count(name) > 0
                ? splitProbabilities[name]
                : 0.0f;
            std::cout << "[Pre-Pass Force Split] frame " << displayFrame
                      << " cell=" << name
                      << " separation=" << separation
                      << " parentMaxR=" << parentMaxRadius
                      << " ratio=" << ratio
                      << " minRatio=" << minSepFraction
                      << " kept=" << keptIt->second
                      << " P(split)=" << pSplit
                      << std::endl;
        }

        std::cout << "[Pre-Pass Force Split] frame " << displayFrame
                  << " checked=" << candidatesChecked
                  << " forced=" << forcedSplitNames.size()
                  << " minRatio=" << minSepFraction
                  << " minKept=" << minKeptPixels
                  << std::endl;
    }

    // ---- Helper: build claim-set for other cells during a split attempt ----
    auto buildOtherClaimSet = [&](const std::string &selfName,
                                  const std::set<std::string> &alreadySplitInB,
                                  const std::set<std::string> &splitRejectedInB,
                                  bool phaseB) -> Frame::ClaimSet {
        Frame::ClaimSet others;
        for (const auto &other : frame.cells) {
            const std::string otherName = other.getName();
            if (otherName == selfName) continue;

            // If this neighbor already split-accepted this frame, its
            // daughters are the relevant Voronoi claims.
            if (alreadySplitInB.count(otherName) > 0) {
                const std::string d1n = otherName + "0";
                const std::string d2n = otherName + "1";
                for (const auto &c : frame.cells) {
                    if (c.getName() == d1n || c.getName() == d2n) {
                        others[c.getName()].push_back(cv::Point3f(
                            c.getX(), c.getY(), c.getZ()));
                    }
                }
                continue;
            }

            // If this neighbor's split was rejected this frame OR it has no
            // D1/D2 seed, use its (calibrated) live position.
            if (splitRejectedInB.count(otherName) > 0) {
                others[otherName].push_back(cv::Point3f(
                    other.getX(), other.getY(), other.getZ()));
                continue;
            }

            // Default: use pre-pass D1/D2 if available (covers dumbbell
            // neighbors), else live position.
            auto itD = expectedDaughters.find(otherName);
            if (itD != expectedDaughters.end()) {
                others[otherName].push_back(itD->second.first);
                others[otherName].push_back(itD->second.second);
            } else {
                others[otherName].push_back(cv::Point3f(
                    other.getX(), other.getY(), other.getZ()));
            }
        }
        return others;
    };

    std::vector<cv::Mat> movementPerturbDebugPlacements;
    std::vector<cv::Mat> splitPerturbDebugPlacements;
    int movementPerturbDebugPlacementCount = 0;
    int splitPerturbDebugPlacementCount = 0;
    const bool exportPerturbDebug =
        config.simulation.export_perturb_debug_images &&
        !frame.getRealFrame().empty() &&
        (config.simulation.export_frame_png || config.simulation.export_frame_tiff);
    const bool exportPerturbCenterDebug =
        config.simulation.export_perturb_cell_center_debug_images &&
        !frame.getRealFrame().empty() &&
        (config.simulation.export_frame_png || config.simulation.export_frame_tiff);
    if (exportPerturbDebug) {
        movementPerturbDebugPlacements = makeEmptyDebugStackLike(frame.getRealFrame());
        splitPerturbDebugPlacements = makeEmptyDebugStackLike(frame.getRealFrame());
    }

    auto attachCellUniverse2FutureWindowSupport =
        [&](const std::string &source,
            const std::string &parentName,
            const PreviousFrameSnapshot &snapshot,
            BridgeSplitProposal &proposal) {
            if (!config.simulation.celluniverse2_enabled ||
                !config.prob.pca_bridge_future_window_enabled ||
                !snapshot.valid) {
                return;
            }
            const float parentMaxR = std::max({
                snapshot.aRadius, snapshot.bRadius, snapshot.cRadius});
            if (parentMaxR <= 1e-3f) return;

            const BoundingBox3D bbox = frame.computeBboxAtPoint(
                snapshot.position, parentMaxR, config.prob.bbox_margin_scale);
            if (!bbox.isValid()) return;

            const int windowSize = std::clamp(
                config.prob.pca_bridge_future_window_size, 2, 5);
            const float minSeparation = std::max(
                6.0f,
                parentMaxR *
                    std::max(
                        0.0f,
                        config.prob
                            .pca_bridge_future_window_min_separation_parent_fraction));
            const cv::Point3f proposalMidpoint =
                0.5f * (proposal.d1Pos + proposal.d2Pos);
            const float proposalSep =
                static_cast<float>(cv::norm(proposal.d2Pos - proposal.d1Pos));
            const float proposalMidpointDistance =
                static_cast<float>(cv::norm(proposalMidpoint - snapshot.position));
            const bool motionExpandedFutureMatch =
                (source == "signal_center_split" ||
                 source == "rod_tip_split_fallback" ||
                 proposal.immediateFutureCenterBacked) &&
                proposalSep >= minSeparation &&
                proposalMidpointDistance <= 1.35f * parentMaxR;

            for (int offset = 1; offset < windowSize; ++offset) {
                const int futureFrameIndex = frameIndex + offset;
                if (futureFrameIndex < 0 ||
                    static_cast<size_t>(futureFrameIndex) >= frames.size()) {
                    break;
                }

                std::vector<cv::Mat> futureRealFrame;
                auto cached = preparedFrameStacks.find(futureFrameIndex);
                if (cached != preparedFrameStacks.end() &&
                    !cached->second.realFrame.empty()) {
                    futureRealFrame = cached->second.realFrame;
                } else if (frames[futureFrameIndex].hasImageStacks()) {
                    futureRealFrame = frames[futureFrameIndex].getRealFrame();
                } else {
                    futureRealFrame = loadPreparedFrameStack(futureFrameIndex).realFrame;
                }
                if (futureRealFrame.empty()) continue;

                Frame futureFrame(config.simulation,
                                  std::vector<Ellipsoid>{},
                                  Path(outputPath),
                                  imagePaths[static_cast<size_t>(futureFrameIndex)]
                                      .filename()
                                      .string());
                futureFrame.loadImageStacks(futureRealFrame);
                std::vector<Frame::SignalCenter> centers =
                    localizeSignalCentersForFrame(
                        futureFrame, config, firstFrame + futureFrameIndex,
                        /*verbose=*/false);

                std::vector<Frame::SignalCenter> inBbox;
                inBbox.reserve(centers.size());
                for (const auto &center : centers) {
                    if (pointInsideBbox(center.position, bbox)) {
                        inBbox.push_back(center);
                    }
                }

                const float matchDistance =
                    std::max(
                        std::max(
                            0.0f,
                            config.prob.pca_bridge_future_window_match_distance) +
                            std::max(
                                0.0f,
                                config.prob
                                    .pca_bridge_future_window_match_distance_per_frame) *
                                static_cast<float>(std::max(0, offset - 1)),
                        motionExpandedFutureMatch ? 1.5f * parentMaxR : 0.0f);

                bool bothSupported = false;
                float bestMatchedSep = 0.0f;
                float bestMatchedPairDistance =
                    std::numeric_limits<float>::infinity();
                float bestMatchedMinBrightness = 0.0f;
                for (size_t i = 0; i < inBbox.size(); ++i) {
                    for (size_t j = i + 1; j < inBbox.size(); ++j) {
                        const cv::Point3f &posI = inBbox[i].position;
                        const cv::Point3f &posJ = inBbox[j].position;
                        const float sep =
                            static_cast<float>(cv::norm(posI - posJ));
                        if (sep < minSeparation) continue;

                        const float directA =
                            static_cast<float>(cv::norm(posI - proposal.d1Pos));
                        const float directB =
                            static_cast<float>(cv::norm(posJ - proposal.d2Pos));
                        const float swappedA =
                            static_cast<float>(cv::norm(posI - proposal.d2Pos));
                        const float swappedB =
                            static_cast<float>(cv::norm(posJ - proposal.d1Pos));
                        const float directMax = std::max(directA, directB);
                        const float swappedMax = std::max(swappedA, swappedB);
                        const float pairDistance =
                            std::min(directA + directB, swappedA + swappedB);
                        const float pairMaxDistance =
                            std::min(directMax, swappedMax);
                        if (pairMaxDistance > matchDistance) continue;
                        if (pairDistance >= bestMatchedPairDistance) continue;

                        bothSupported = true;
                        bestMatchedSep = sep;
                        bestMatchedPairDistance = pairDistance;
                        bestMatchedMinBrightness =
                            std::min(inBbox[i].brightness, inBbox[j].brightness);
                    }
                }

                float parentDist = std::numeric_limits<float>::infinity();
                for (const auto &center : inBbox) {
                    parentDist = std::min(
                        parentDist,
                        static_cast<float>(
                            cv::norm(center.position - snapshot.position)));
                }
                const bool parentPersists =
                    !bothSupported &&
                    parentDist <= matchDistance &&
                    inBbox.size() <= 1;

                if (bothSupported) {
                    ++proposal.windowBothDaughtersSupported;
                    if (offset == 1) {
                        proposal.windowImmediateBothDaughtersSupported = 1;
                    }
                } else {
                    proposal.windowMissingDaughterCount += 2;
                }
                if (parentPersists) {
                    ++proposal.windowParentPersists;
                }

                std::cout << "[CellUniverse2 Future Window] frame "
                          << (firstFrame + frameIndex)
                          << " source=" << source
                          << " parent=" << parentName
                          << " futureFrame="
                          << (firstFrame + futureFrameIndex)
                          << " offset=" << offset
                          << " centersInBbox=" << inBbox.size()
                          << " bothSupported=" << (bothSupported ? 1 : 0)
                          << " bestMatchedSep=" << bestMatchedSep
                          << " bestMatchedPairDistance=" << bestMatchedPairDistance
                          << " bestMatchedMinBrightness=" << bestMatchedMinBrightness
                          << " minSeparation=" << minSeparation
                          << " matchDistance=" << matchDistance
                          << " motionExpanded="
                          << (motionExpandedFutureMatch ? 1 : 0)
                          << " parentDist=" << parentDist
                          << " parentPersists="
                          << (parentPersists ? 1 : 0)
                          << " bbox=(" << bbox.xMin << "-" << bbox.xMax
                          << "," << bbox.yMin << "-" << bbox.yMax
                          << "," << bbox.zMin << "-" << bbox.zMax
                          << ")"
                          << std::endl;
            }
        };

    auto snapCellUniverse2ProposalToAlignedSignalCenters =
        [&](const std::string &source,
            const Ellipsoid &parent,
            const PreviousFrameSnapshot &snapshot,
            BridgeSplitProposal &proposal) -> bool {
            if (!config.simulation.celluniverse2_enabled || !snapshot.valid) {
                return true;
            }

            const std::vector<Frame::SignalCenter> &centers =
                frame.getSignalCenters();
            if (centers.size() < 2) {
                std::cout << "[CellUniverse2 Split Proposal Center Snap Reject] frame "
                          << displayFrame
                          << " source=" << source
                          << " parent=" << parent.getName()
                          << " reason=too_few_signal_centers"
                          << " centers=" << centers.size()
                          << std::endl;
                return false;
            }

            const float parentMaxR = std::max({
                snapshot.aRadius, snapshot.bRadius, snapshot.cRadius,
                parent.getARadius(), parent.getBRadius(), parent.getCRadius()});
            if (parentMaxR <= 1e-3f) {
                std::cout << "[CellUniverse2 Split Proposal Center Snap Reject] frame "
                          << displayFrame
                          << " source=" << source
                          << " parent=" << parent.getName()
                          << " reason=bad_parent_radius"
                          << std::endl;
                return false;
            }

            const BoundingBox3D bbox = frame.computeBboxAtPoint(
                snapshot.position, parentMaxR, config.prob.bbox_margin_scale);
            if (!bbox.isValid()) {
                std::cout << "[CellUniverse2 Split Proposal Center Snap Reject] frame "
                          << displayFrame
                          << " source=" << source
                          << " parent=" << parent.getName()
                          << " reason=bad_bbox"
                          << std::endl;
                return false;
            }

            const float maxSnapDistance = std::max(
                std::max(0.0f, config.prob.pca_bridge_future_window_match_distance),
                1.5f * parentMaxR);
            // Signal-center localization can over-fire in crowded frames. Keep
            // snaps close to the bridge/signal seed; distant centers still have
            // to win through the raw PCA proposal and downstream image gates.
            const float maxSeedSnapDistance = std::min(
                maxSnapDistance,
                std::max(
                    std::max(0.0f, config.prob.pca_bridge_future_window_match_distance),
                    0.75f * parentMaxR));
            const float minAlignment = std::clamp(
                config.prob.signal_center_split_min_axis_alignment, 0.0f, 1.0f);
            const float sideTolerance = 0.05f * parentMaxR;

            struct Pick {
                bool valid = false;
                size_t index = 0;
                cv::Point3f position{0.0f, 0.0f, 0.0f};
                float score = std::numeric_limits<float>::max();
                float distance = 0.0f;
                float alignment = 0.0f;
                float axisProjection = 0.0f;
            };
            struct PairPick {
                bool valid = false;
                Pick d1;
                Pick d2;
                cv::Point3f seed1{0.0f, 0.0f, 0.0f};
                cv::Point3f seed2{0.0f, 0.0f, 0.0f};
                float score = std::numeric_limits<float>::max();
                bool usedAlignedPairFallback = false;
            };

            const std::vector<Frame::SignalCenter> *activeCenters = &centers;

            auto chooseForSeed =
                [&](const cv::Point3f &seed,
                    const cv::Point3f &midpoint,
                    const cv::Point3f &axis,
                    int side,
                    int forbiddenIndex) -> Pick {
                    Pick best;
                    const cv::Point3f seedVector = seed - midpoint;
                    const float seedNorm = static_cast<float>(cv::norm(seedVector));
                    const cv::Point3f seedDir =
                        seedNorm > 1e-6f
                            ? seedVector * (1.0f / seedNorm)
                            : (side < 0 ? -axis : axis);

                    for (size_t i = 0; i < activeCenters->size(); ++i) {
                        if (static_cast<int>(i) == forbiddenIndex) continue;
                        const auto &center = (*activeCenters)[i];
                        if (!pointInsideBbox(center.position, bbox)) continue;

                        const cv::Point3f centerVector =
                            center.position - midpoint;
                        const float axisProjection = centerVector.dot(axis);
                        if (side < 0 && axisProjection > sideTolerance) continue;
                        if (side > 0 && axisProjection < -sideTolerance) continue;

                        const float centerNorm =
                            static_cast<float>(cv::norm(centerVector));
                        if (centerNorm <= 1e-6f) continue;
                        const cv::Point3f centerDir =
                            centerVector * (1.0f / centerNorm);
                        const float alignment = centerDir.dot(seedDir);
                        if (alignment < minAlignment) continue;

                        const float distance =
                            static_cast<float>(cv::norm(center.position - seed));
                        if (distance > maxSeedSnapDistance) continue;

                        const cv::Point3f axisVector =
                            axisProjection * axis;
                        const float perpendicularDistance =
                            static_cast<float>(
                                cv::norm(centerVector - axisVector));
                        const float score =
                            distance / std::max(1e-3f, maxSnapDistance) +
                            0.35f * perpendicularDistance /
                                std::max(1e-3f, parentMaxR) +
                            0.50f * (1.0f - alignment) -
                            0.05f * center.brightness;
                        if (score >= best.score) continue;

                        best.valid = true;
                        best.index = i;
                        best.position = center.position;
                        best.score = score;
                        best.distance = distance;
                        best.alignment = alignment;
                        best.axisProjection = axisProjection;
                    }
                    return best;
                };

            auto evaluateSeedPair =
                [&](const cv::Point3f &seed1,
                    const cv::Point3f &seed2) -> PairPick {
                    PairPick result;
                    result.seed1 = seed1;
                    result.seed2 = seed2;
                    const cv::Point3f midpoint = snapshot.position;
                    cv::Point3f axis = seed2 - seed1;
                    const float axisNorm = static_cast<float>(cv::norm(axis));
                    if (axisNorm <= 1e-6f) {
                        return result;
                    }
                    axis *= (1.0f / axisNorm);

                    const Pick d1 = chooseForSeed(
                        seed1, midpoint, axis, -1, -1);
                    if (!d1.valid) {
                        return result;
                    }
                    const Pick d2 = chooseForSeed(
                        seed2, midpoint, axis, 1, static_cast<int>(d1.index));
                    if (!d2.valid) {
                        return result;
                    }

                    result.valid = true;
                    result.d1 = d1;
                    result.d2 = d2;
                    result.score = d1.score + d2.score;
                    return result;
                };

            auto evaluateAlignedPairInActiveCenters = [&]() -> PairPick {
                PairPick result;
                cv::Point3f splitAxis = snapshot.splitAxisDir;
                float splitAxisNorm = static_cast<float>(cv::norm(splitAxis));
                if (splitAxisNorm <= 1e-6f) {
                    float splitAxisLength = 0.0f;
                    parent.worldSplitAxis(splitAxis, splitAxisLength);
                    splitAxisNorm = static_cast<float>(cv::norm(splitAxis));
                }
                if (splitAxisNorm <= 1e-6f) {
                    return result;
                }
                splitAxis *= (1.0f / splitAxisNorm);

                const float minSep =
                    std::max(0.0f,
                             config.prob
                                 .signal_center_split_min_separation_radius_scale) *
                    parentMaxR;
                const float maxSep =
                    std::max(
                        config.prob.signal_center_split_min_separation_radius_scale,
                        config.prob.signal_center_split_max_separation_radius_scale) *
                    parentMaxR;
                const float maxMidpointDistance =
                    std::max(
                        0.0f,
                        config.prob
                            .signal_center_split_max_midpoint_radius_scale) *
                    parentMaxR;

                for (size_t ai = 0; ai < activeCenters->size(); ++ai) {
                    const auto &centerA = (*activeCenters)[ai];
                    if (!pointInsideBbox(centerA.position, bbox)) continue;
                    for (size_t bi = ai + 1; bi < activeCenters->size(); ++bi) {
                        const auto &centerB = (*activeCenters)[bi];
                        if (!pointInsideBbox(centerB.position, bbox)) continue;

                        cv::Point3f delta = centerB.position - centerA.position;
                        const float sep = static_cast<float>(cv::norm(delta));
                        if (sep < minSep || sep > maxSep) continue;
                        const cv::Point3f pairAxis =
                            delta * (1.0f / std::max(1e-6f, sep));
                        const float axisAlignment =
                            std::abs(pairAxis.dot(splitAxis));
                        if (axisAlignment < minAlignment) continue;

                        const cv::Point3f midpoint =
                            0.5f * (centerA.position + centerB.position);
                        const float midpointDistance =
                            static_cast<float>(
                                cv::norm(midpoint - snapshot.position));
                        if (midpointDistance > maxMidpointDistance) continue;

                        const float sepRatio = sep / parentMaxR;
                        const float brightness =
                            centerA.brightness + centerB.brightness;
                        const float score =
                            midpointDistance / std::max(1e-3f, parentMaxR) +
                            0.50f * std::abs(sepRatio - 1.0f) +
                            0.75f * (1.0f - axisAlignment) -
                            0.05f * brightness;
                        if (score >= result.score) continue;

                        result.valid = true;
                        result.d1.valid = true;
                        result.d2.valid = true;
                        result.d1.index = ai;
                        result.d2.index = bi;
                        result.d1.position = centerA.position;
                        result.d2.position = centerB.position;
                        if ((result.d2.position - result.d1.position)
                                .dot(splitAxis) < 0.0f) {
                            std::swap(result.d1, result.d2);
                        }
                        result.d1.distance = static_cast<float>(
                            cv::norm(result.d1.position - proposal.d1Pos));
                        result.d2.distance = static_cast<float>(
                            cv::norm(result.d2.position - proposal.d2Pos));
                        result.d1.alignment = axisAlignment;
                        result.d2.alignment = axisAlignment;
                        result.seed1 = proposal.d1Pos;
                        result.seed2 = proposal.d2Pos;
                        result.score = score;
                        result.usedAlignedPairFallback = true;
                    }
                }
                return result;
            };

            PairPick best = evaluateSeedPair(proposal.d1Pos, proposal.d2Pos);
            bool usedImmediateFutureCenters = false;
            if (proposal.hasAlternateSeedPair) {
                PairPick alternate =
                    evaluateSeedPair(proposal.altD1Pos, proposal.altD2Pos);
                if (alternate.valid && (!best.valid || alternate.score < best.score)) {
                    best = alternate;
                }
            }

            if (!best.valid &&
                config.prob.pca_bridge_future_window_enabled &&
                frameIndex + 1 >= 0 &&
                static_cast<size_t>(frameIndex + 1) < frames.size()) {
                std::vector<cv::Mat> futureRealFrame;
                auto cached = preparedFrameStacks.find(frameIndex + 1);
                if (cached != preparedFrameStacks.end() &&
                    !cached->second.realFrame.empty()) {
                    futureRealFrame = cached->second.realFrame;
                } else if (frames[frameIndex + 1].hasImageStacks()) {
                    futureRealFrame = frames[frameIndex + 1].getRealFrame();
                } else {
                    futureRealFrame =
                        loadPreparedFrameStack(frameIndex + 1).realFrame;
                }

                if (!futureRealFrame.empty()) {
                    Frame futureFrame(config.simulation,
                                      std::vector<Ellipsoid>{},
                                      Path(outputPath),
                                      imagePaths[static_cast<size_t>(frameIndex + 1)]
                                          .filename()
                                          .string());
                    futureFrame.loadImageStacks(futureRealFrame);
                    std::vector<Frame::SignalCenter> futureCenters =
                        localizeSignalCentersForFrame(
                            futureFrame,
                            config,
                            firstFrame + frameIndex + 1,
                            /*verbose=*/false);
                    activeCenters = &futureCenters;
                    PairPick futureBest =
                        evaluateSeedPair(proposal.d1Pos, proposal.d2Pos);
                    if (proposal.hasAlternateSeedPair) {
                        PairPick alternate =
                            evaluateSeedPair(proposal.altD1Pos,
                                             proposal.altD2Pos);
                        if (alternate.valid &&
                            (!futureBest.valid ||
                             alternate.score < futureBest.score)) {
                            futureBest = alternate;
                        }
                    }
                    if (!futureBest.valid) {
                        futureBest = evaluateAlignedPairInActiveCenters();
                    }
                    if (futureBest.valid) {
                        best = futureBest;
                        usedImmediateFutureCenters = true;
                    }
                    activeCenters = &centers;
                }
            }

            if (!best.valid) {
                std::cout << "[CellUniverse2 Split Proposal Center Snap Reject] frame "
                          << displayFrame
                          << " source=" << source
                          << " parent=" << parent.getName()
                          << " reason=no_aligned_center_pair"
                          << " maxSnapDistance=" << maxSnapDistance
                          << " maxSeedSnapDistance=" << maxSeedSnapDistance
                          << " minAlignment=" << minAlignment
                          << " bbox=(" << bbox.xMin << "-" << bbox.xMax
                          << "," << bbox.yMin << "-" << bbox.yMax
                          << "," << bbox.zMin << "-" << bbox.zMax
                          << ")"
                          << std::endl;
                return false;
            }

            const cv::Point3f oldD1 = proposal.d1Pos;
            const cv::Point3f oldD2 = proposal.d2Pos;
            proposal.d1Pos = best.d1.position;
            proposal.d2Pos = best.d2.position;
            proposal.candidateIdA =
                usedImmediateFutureCenters ? -1 : static_cast<int>(best.d1.index);
            proposal.candidateIdB =
                usedImmediateFutureCenters ? -1 : static_cast<int>(best.d2.index);
            proposal.immediateFutureCenterBacked = usedImmediateFutureCenters;
            proposal.centerSnapMaxSeedDistance = std::max(best.d1.distance, best.d2.distance);
            proposal.centerSnapScore = best.score;
            proposal.centerSnapUsedAlignedPairFallback = best.usedAlignedPairFallback;
            proposal.hasAlternateSeedPair = false;

            std::cout << "[CellUniverse2 Split Proposal Center Snap] frame "
                      << displayFrame
                      << " source=" << source
                      << " parent=" << parent.getName()
                      << " centerSource="
                      << (usedImmediateFutureCenters ? "immediate_future"
                                                     : "current")
                      << " oldD1=(" << oldD1.x << "," << oldD1.y << ","
                      << oldD1.z << ")"
                      << " oldD2=(" << oldD2.x << "," << oldD2.y << ","
                      << oldD2.z << ")"
                      << " seedD1=(" << best.seed1.x << "," << best.seed1.y
                      << "," << best.seed1.z << ")"
                      << " seedD2=(" << best.seed2.x << "," << best.seed2.y
                      << "," << best.seed2.z << ")"
                      << " newD1=(" << proposal.d1Pos.x << ","
                      << proposal.d1Pos.y << "," << proposal.d1Pos.z << ")"
                      << " newD2=(" << proposal.d2Pos.x << ","
                      << proposal.d2Pos.y << "," << proposal.d2Pos.z << ")"
                      << " centerA=" << best.d1.index
                      << " centerB=" << best.d2.index
                      << " d1Distance=" << best.d1.distance
                      << " d2Distance=" << best.d2.distance
                      << " maxSeedDistance=" << proposal.centerSnapMaxSeedDistance
                      << " snapScore=" << proposal.centerSnapScore
                      << " alignedPairFallback="
                      << (proposal.centerSnapUsedAlignedPairFallback ? 1 : 0)
                      << " d1Alignment=" << best.d1.alignment
                      << " d2Alignment=" << best.d2.alignment
                      << std::endl;
            return true;
        };

    auto validateCellUniverse2SplitProposal =
        [&](const std::string &source,
            const Ellipsoid &parent,
            const PreviousFrameSnapshot &snapshot,
            BridgeSplitProposal &proposal) -> bool {
            if (!config.simulation.celluniverse2_enabled || !snapshot.valid) {
                return true;
            }

            attachCellUniverse2FutureWindowSupport(
                source, parent.getName(), snapshot, proposal);

            const cv::Point3f parentPos = snapshot.position;
            const float parentMaxR = std::max({
                snapshot.aRadius, snapshot.bRadius, snapshot.cRadius,
                parent.getARadius(), parent.getBRadius(), parent.getCRadius()});
            const cv::Point3f splitDelta = proposal.d2Pos - proposal.d1Pos;
            const float daughterSep = static_cast<float>(cv::norm(splitDelta));
            const float minBioSep =
                std::max(0.0f,
                         config.prob.bio_min_daughter_separation_parent_fraction) *
                parentMaxR;

            auto rejectProposal = [&](const std::string &reason,
                                      const std::string &detail) -> bool {
                std::cout << "[CellUniverse2 Split Proposal Reject] frame "
                          << displayFrame
                          << " source=" << source
                          << " parent=" << parent.getName()
                          << " reason=" << reason
                          << " sep=" << daughterSep
                          << " minBioSep=" << minBioSep
                          << " parentDistBalance="
                          << proposal.parentDistanceBalance
                          << " futureBoth="
                          << proposal.windowBothDaughtersSupported
                          << " futureMissing="
                          << proposal.windowMissingDaughterCount
                          << " parentPersists="
                          << proposal.windowParentPersists
                          << " " << detail
                          << std::endl;
                return false;
            };

            if (parentMaxR <= 1e-3f || daughterSep <= 1e-3f) {
                return rejectProposal("bad_geometry", "");
            }

            const float d1ParentDist =
                static_cast<float>(cv::norm(proposal.d1Pos - parentPos));
            const float d2ParentDist =
                static_cast<float>(cv::norm(proposal.d2Pos - parentPos));
            proposal.parentDistanceBalance =
                std::min(d1ParentDist, d2ParentDist) /
                std::max(1e-3f, std::max(d1ParentDist, d2ParentDist));
            const int configuredMinFutureBoth =
                std::max(0,
                         config.prob.pca_bridge_future_window_min_both_daughter_support);
            const int futureWindowSize = std::clamp(
                config.prob.pca_bridge_future_window_size, 2, 5);
            const int availableFutureFrames = std::max(
                0,
                std::min(futureWindowSize - 1,
                         static_cast<int>(frames.size()) - frameIndex - 1));
            const bool futureWindowSupportRequired =
                config.prob.pca_bridge_future_window_enabled &&
                configuredMinFutureBoth > 0;
            const bool futureWindowUnavailable =
                futureWindowSupportRequired && availableFutureFrames <= 0;
            const int minFutureBoth = futureWindowSupportRequired
                                          ? std::max(1,
                                                     std::min(configuredMinFutureBoth,
                                                              availableFutureFrames))
                                          : std::min(configuredMinFutureBoth,
                                                     availableFutureFrames);
            const int maxFutureMissing =
                std::max(0,
                         config.prob.pca_bridge_future_window_max_missing_daughters);
            const int maxParentPersists =
                std::max(0,
                         config.prob.pca_bridge_future_window_max_parent_persists);
            const bool currentFrameSignalCenterProposal =
                source == "signal_center_split" &&
                !proposal.immediateFutureCenterBacked;
            // Signal centers over-detect, so one-frame future support is
            // allowed only when current-frame geometry is already strong.
            // This preserves early bootstrap splits and late crowded-wave
            // splits, while avoiding dim/near-miss proposals like frame-18
            // false positives.
            const bool strongCurrentFrameSignalCenterProposal =
                currentFrameSignalCenterProposal &&
                daughterSep >= minBioSep &&
                proposal.parentDistanceBalance >= 0.70f;
            const int activeMinFutureBoth =
                strongCurrentFrameSignalCenterProposal
                    ? std::min(minFutureBoth, 1)
                    : minFutureBoth;
            const int activeMaxFutureMissing =
                strongCurrentFrameSignalCenterProposal
                    ? std::max(maxFutureMissing, 2)
                    : maxFutureMissing;
            const float cleanSnapDistanceLimit = std::max(
                std::max(0.0f, config.prob.pca_bridge_future_window_match_distance),
                0.75f * parentMaxR);
            const bool cleanCenterSnap =
                !proposal.immediateFutureCenterBacked ||
                (!proposal.centerSnapUsedAlignedPairFallback &&
                 proposal.centerSnapMaxSeedDistance <= cleanSnapDistanceLimit);
            const bool cleanFutureDaughterSupport =
                config.prob.pca_bridge_future_window_enabled &&
                cleanCenterSnap &&
                proposal.windowBothDaughtersSupported >= activeMinFutureBoth &&
                proposal.windowImmediateBothDaughtersSupported > 0 &&
                proposal.windowMissingDaughterCount <= activeMaxFutureMissing &&
                proposal.windowParentPersists <= maxParentPersists;
            proposal.futureWindowSplitRescue =
                proposal.immediateFutureCenterBacked && cleanFutureDaughterSupport;
            const bool futureSeparationRescue =
                proposal.futureWindowSplitRescue &&
                config.prob.pca_bridge_future_window_min_separation_parent_fraction > 0.0f;
            const float activeMinBioSep =
                futureSeparationRescue
                    ? std::min(
                          minBioSep,
                          config.prob
                                  .pca_bridge_future_window_min_separation_parent_fraction *
                              parentMaxR)
                    : minBioSep;
            if (activeMinBioSep > 0.0f && daughterSep < activeMinBioSep) {
                const float softMinFraction = std::clamp(
                    config.prob.bio_separation_soft_min_fraction, 0.0f, 1.0f);
                const float hardMinBioSep = activeMinBioSep * softMinFraction;
                const bool futureSupportOk =
                    !config.prob.bio_separation_soft_require_future_support ||
                    cleanFutureDaughterSupport;
                const bool softEligible =
                    config.prob.bio_separation_soft_gate_enabled &&
                    futureSupportOk &&
                    daughterSep >= hardMinBioSep;
                if (!softEligible) {
                    std::ostringstream detail;
                    detail << "activeMinBioSep=" << activeMinBioSep
                           << " hardMinBioSep=" << hardMinBioSep
                           << " futureSeparationRescue="
                           << (futureSeparationRescue ? 1 : 0)
                           << " softEligible=0"
                           << " futureSupportOk=" << (futureSupportOk ? 1 : 0);
                    return rejectProposal("bio_separation_gate", detail.str());
                }
                proposal.bioSeparationSoftRescued = true;
                proposal.bioSeparationObserved = daughterSep;
                proposal.bioSeparationRequired = activeMinBioSep;
                proposal.bioSeparationSoftNormalizedExcess =
                    (activeMinBioSep - daughterSep) /
                    std::max(1.0f, activeMinBioSep);
                proposal.bioSeparationSoftPenaltyFraction =
                    std::max(0.0f,
                             config.prob.bio_separation_soft_penalty_fraction);
                std::cout << "[CellUniverse2 Split Proposal Soft Gate] frame "
                          << displayFrame
                          << " source=" << source
                          << " parent=" << parent.getName()
                          << " reason=bio_separation_near_miss"
                          << " sep=" << daughterSep
                          << " activeMinBioSep=" << activeMinBioSep
                          << " hardMinBioSep=" << hardMinBioSep
                          << " normalizedExcess="
                          << proposal.bioSeparationSoftNormalizedExcess
                          << " penaltyFraction="
                          << proposal.bioSeparationSoftPenaltyFraction
                          << " futureBoth="
                          << proposal.windowBothDaughtersSupported
                          << " futureImmediate="
                          << proposal.windowImmediateBothDaughtersSupported
                          << " futureMissing="
                          << proposal.windowMissingDaughterCount
                          << " parentPersists="
                          << proposal.windowParentPersists
                          << std::endl;
            }
            const float minParentBalance =
                std::max(0.0f,
                         config.cellLumen.fusionSplitPriorMinParentDistanceBalance);
            const float relaxedFutureBalance =
                std::clamp(
                    config.prob.pca_bridge_future_window_parent_balance_rescue_min,
                    0.0f,
                    minParentBalance);
            const float activeParentBalance =
                cleanFutureDaughterSupport ? relaxedFutureBalance : minParentBalance;
            if (proposal.parentDistanceBalance < activeParentBalance) {
                std::ostringstream detail;
                detail << "activeParentBalance=" << activeParentBalance
                       << " normalParentBalance=" << minParentBalance
                       << " futureBalanceRescue="
                       << (cleanFutureDaughterSupport ? 1 : 0);
                return rejectProposal("parent_distance_balance_gate",
                                      detail.str());
            }

            if (config.prob.pca_bridge_future_window_enabled &&
                (futureWindowUnavailable ||
                 proposal.windowBothDaughtersSupported < activeMinFutureBoth ||
                 proposal.windowMissingDaughterCount > activeMaxFutureMissing ||
                 proposal.windowParentPersists > maxParentPersists)) {
                std::ostringstream detail;
                detail << "minFutureBoth=" << activeMinFutureBoth
                       << " configuredMinFutureBoth="
                       << configuredMinFutureBoth
                       << " availableFutureFrames="
                       << availableFutureFrames
                       << " futureWindowUnavailable="
                       << (futureWindowUnavailable ? 1 : 0)
                       << " maxFutureMissing=" << activeMaxFutureMissing
                       << " maxParentPersists=" << maxParentPersists
                       << " cleanCenterSnap=" << (cleanCenterSnap ? 1 : 0)
                       << " centerSnapMaxSeedDistance="
                       << proposal.centerSnapMaxSeedDistance
                       << " cleanSnapDistanceLimit=" << cleanSnapDistanceLimit
                       << " alignedPairFallback="
                       << (proposal.centerSnapUsedAlignedPairFallback ? 1 : 0);
                return rejectProposal(
                    futureWindowUnavailable ? "future_window_unavailable"
                                            : "future_window_gate",
                    detail.str());
            }

            const float tieMargin =
                std::max(0.0f,
                         config.cellLumen.fusionSplitPriorContinuationClaimTieMargin);
            const float claimRadiusScale =
                std::max(0.0f,
                         config.cellLumen.fusionSplitPriorContinuationClaimRadiusScale);
            const auto ownedByNeighbor =
                [&](const cv::Point3f &daughterPos, std::string &blocker) -> bool {
                    const float parentDist =
                        static_cast<float>(cv::norm(daughterPos - parentPos));
                    for (const auto &other : frame.cells) {
                        const std::string otherName = other.getName();
                        if (other.isTrash() || otherName == parent.getName()) {
                            continue;
                        }
                        auto otherSnapIt = previousSnapshots.find(otherName);
                        const cv::Point3f otherPos =
                            (otherSnapIt != previousSnapshots.end() &&
                             otherSnapIt->second.valid)
                                ? otherSnapIt->second.position
                                : cv::Point3f(other.getX(), other.getY(), other.getZ());
                        const float otherDist =
                            static_cast<float>(cv::norm(daughterPos - otherPos));
                        const float otherMaxR =
                            (otherSnapIt != previousSnapshots.end() &&
                             otherSnapIt->second.valid)
                                ? std::max({otherSnapIt->second.aRadius,
                                            otherSnapIt->second.bRadius,
                                            otherSnapIt->second.cRadius})
                                : std::max({other.getARadius(),
                                            other.getBRadius(),
                                            other.getCRadius()});
                        const bool closeToOther =
                            claimRadiusScale > 0.0f &&
                            otherDist <= claimRadiusScale * otherMaxR;
                        const bool closerToOther =
                            otherDist + tieMargin < parentDist;
                        if (closeToOther && closerToOther) {
                            std::ostringstream ss;
                            ss << otherName
                               << ":parentDist=" << parentDist
                               << ":otherDist=" << otherDist
                               << ":otherRadius=" << otherMaxR;
                            blocker = ss.str();
                            return true;
                        }
                    }
                    return false;
            };

            std::string blocker;
            if (!cleanFutureDaughterSupport &&
                (ownedByNeighbor(proposal.d1Pos, blocker) ||
                 ownedByNeighbor(proposal.d2Pos, blocker))) {
                proposal.continuationClaimBlockerNames = blocker;
                proposal.neighborClaimPenalty = 1.0f;
                return rejectProposal("neighbor_continuation_claim_gate",
                                      "blocker=" + blocker);
            }
            if (cleanFutureDaughterSupport &&
                (ownedByNeighbor(proposal.d1Pos, blocker) ||
                 ownedByNeighbor(proposal.d2Pos, blocker))) {
                std::cout << "[CellUniverse2 Split Proposal Claim Bypass] frame "
                          << displayFrame
                          << " source=" << source
                          << " parent=" << parent.getName()
                          << " reason=clean_future_daughter_support"
                          << " blocker=" << blocker
                          << std::endl;
            }

            std::cout << "[CellUniverse2 Split Proposal Acceptable] frame "
                      << displayFrame
                      << " source=" << source
                      << " parent=" << parent.getName()
                      << " sep=" << daughterSep
                      << " minBioSep=" << minBioSep
                      << " parentDistBalance="
                      << proposal.parentDistanceBalance
                      << " futureBoth="
                      << proposal.windowBothDaughtersSupported
                      << " futureMissing="
                      << proposal.windowMissingDaughterCount
                      << " parentPersists="
                      << proposal.windowParentPersists
                      << " futureBalanceRescue="
                      << (cleanFutureDaughterSupport ? 1 : 0)
                      << " cleanCenterSnap=" << (cleanCenterSnap ? 1 : 0)
                      << " centerSnapMaxSeedDistance="
                      << proposal.centerSnapMaxSeedDistance
                      << " cleanSnapDistanceLimit=" << cleanSnapDistanceLimit
                      << " alignedPairFallback="
                      << (proposal.centerSnapUsedAlignedPairFallback ? 1 : 0)
                      << " minFutureBoth=" << activeMinFutureBoth
                      << " configuredMinFutureBoth="
                      << configuredMinFutureBoth
                      << " maxFutureMissing=" << activeMaxFutureMissing
                      << " currentFrameSignalCenter="
                      << (currentFrameSignalCenterProposal ? 1 : 0)
                      << " availableFutureFrames="
                      << availableFutureFrames
                      << std::endl;
            return true;
        };

    // ---- CellUniverse2 signal-center split proposal pre-pass ----
    //
    // If frame N's copied-forward parent is already flat, frame N+1 may show
    // two separate signal centers before the normal PCA fit has a chance to
    // collapse the parent onto one blob. Use the parent snap bbox from frame N
    // to find two distinctive centers, then send that pair through the same
    // split validation path as bridge cuts. A rejection leaves the parent
    // untouched, so the default CellUniverse2 final PCA fit still runs.
    std::unordered_map<std::string, BridgeSplitProposal> signalCenterSplitProposals;
    std::unordered_map<std::string, BridgeSplitProposal> rodTipSplitFallbackProposals;
    if (allowSignalCenterSplit && !frame.getSignalCenters().empty()) {
        int signalSplitConsidered = 0;
        int signalSplitEligibleFlat = 0;
        int signalSplitTooFewCenters = 0;
        float maxSignalSplitElong = 0.0f;
        for (size_t ci = 0; ci < frame.cells.size(); ++ci) {
            const Ellipsoid &cell = frame.cells[ci];
            if (cell.isTrash()) continue;
            const std::string &cellName = cell.getName();
            if (cellUniverse2SplitCoolingDown(cellName)) continue;
            auto snapIt = previousSnapshots.find(cellName);
            if (snapIt == previousSnapshots.end() || !snapIt->second.valid) {
                continue;
            }
            ++signalSplitConsidered;
            const float parentShape =
                std::max(cell.shapeElongation(), snapIt->second.shapeElongation);
            maxSignalSplitElong = std::max(maxSignalSplitElong, parentShape);
            if (parentShape >= config.prob.signal_center_split_min_parent_elongation) {
                ++signalSplitEligibleFlat;
            }

            SignalCenterSplitProposalResult signalProposal =
                findSignalCenterSplitProposal(frame, ci, snapIt->second, config);
            if (parentShape >= config.prob.signal_center_split_min_parent_elongation &&
                signalProposal.centersInside < 2) {
                ++signalSplitTooFewCenters;
            }
            if (!signalProposal.valid) {
                if (parentShape >= config.prob.signal_center_split_min_parent_elongation) {
                    std::cout << "[Signal Center Split Reject] frame "
                              << (firstFrame + frameIndex)
                              << " cell=" << cellName
                              << " reason=" << signalProposal.rejectReason
                              << " parentShape=" << signalProposal.parentShape
                              << " centersInside=" << signalProposal.centersInside
                              << " candidatePairs=" << signalProposal.candidatePairs
                              << " rejectedSep=" << signalProposal.rejectedBySeparation
                              << " rejectedMid=" << signalProposal.rejectedByMidpoint
                              << " rejectedAxis=" << signalProposal.rejectedByAxis
                              << " bestRejectedAxis="
                              << signalProposal.bestRejectedAxisAlignment
                              << " bestRejectedSepRatio="
                              << signalProposal.bestRejectedSeparationRatio
                              << " bestRejectedMidpointDist="
                              << signalProposal.bestRejectedMidpointDistance
                              << std::endl;
                }

                const bool allowRodTipFallback =
                    signalProposal.rejectReason != "not_flat_enough";
                RodTipSphereSplitProposalResult tipProposal;
                if (allowRodTipFallback) {
                    tipProposal = makeRodTipSphereSplitProposal(cell, snapIt->second, config);
                } else {
                    tipProposal.rejectReason = "parent_not_flat_enough";
                    tipProposal.parentShape = parentShape;
                }
                if (tipProposal.valid) {
                    const bool snappedToCurrentCenter =
                        snapCellUniverse2ProposalToAlignedSignalCenters(
                            "rod_tip_split_fallback",
                            cell,
                            snapIt->second,
                            tipProposal.proposal);
                    if (!snappedToCurrentCenter) {
                        std::cout << "[CellUniverse2 Split Proposal Center Snap Bypass] frame "
                                  << displayFrame
                                  << " source=rod_tip_split_fallback"
                                  << " parent=" << cellName
                                  << " reason=use_raw_seed_with_future_window_gate"
                                  << std::endl;
                    }
                }
                if (tipProposal.valid &&
                    validateCellUniverse2SplitProposal(
                        "rod_tip_split_fallback",
                        cell,
                        snapIt->second,
                        tipProposal.proposal)) {
                    rodTipSplitFallbackProposals[cellName] = tipProposal.proposal;
                    std::cout << "[Rod Tip Split Fallback Propose] frame "
                              << (firstFrame + frameIndex)
                              << " cell=" << cellName
                              << " signalReject=" << signalProposal.rejectReason
                              << " parentShape=" << tipProposal.parentShape
                              << " longR=" << tipProposal.longRadius
                              << " midR=" << tipProposal.midRadius
                              << " shortR=" << tipProposal.shortRadius
                              << " longMidRatio=" << tipProposal.longMidRatio
                              << " midShortRatio=" << tipProposal.midShortRatio
                              << " sphereRadius=" << tipProposal.sphereRadius
                              << " tipDistance=" << tipProposal.tipDistance
                              << " centerDaughter=(" << tipProposal.proposal.d1Pos.x
                              << "," << tipProposal.proposal.d1Pos.y
                              << "," << tipProposal.proposal.d1Pos.z << ")"
                              << " tipDaughter=(" << tipProposal.proposal.d2Pos.x
                              << "," << tipProposal.proposal.d2Pos.y
                              << "," << tipProposal.proposal.d2Pos.z << ")"
                              << " altTipDaughter=(" << tipProposal.proposal.altD2Pos.x
                              << "," << tipProposal.proposal.altD2Pos.y
                              << "," << tipProposal.proposal.altD2Pos.z << ")"
                              << std::endl;
                    continue;
                }

                if (parentShape >= config.prob.signal_center_split_min_parent_elongation) {
                    std::cout << "[Rod Tip Split Fallback Reject] frame "
                              << (firstFrame + frameIndex)
                              << " cell=" << cellName
                              << " reason=" << tipProposal.rejectReason
                              << " parentShape=" << tipProposal.parentShape
                              << " longR=" << tipProposal.longRadius
                              << " midR=" << tipProposal.midRadius
                              << " shortR=" << tipProposal.shortRadius
                              << " longMidRatio=" << tipProposal.longMidRatio
                              << " midShortRatio=" << tipProposal.midShortRatio
                              << std::endl;
                }
                continue;
            }

            if (!validateCellUniverse2SplitProposal(
                    "signal_center_split",
                    cell,
                    snapIt->second,
                    signalProposal.proposal)) {
                continue;
            }

            signalCenterSplitProposals[cellName] = signalProposal.proposal;
            std::cout << "[Signal Center Split Propose] frame "
                      << (firstFrame + frameIndex)
                      << " cell=" << cellName
                      << " parentShape=" << signalProposal.parentShape
                      << " centersInside=" << signalProposal.centersInside
                      << " centerA=" << signalProposal.centerIndexA
                      << " centerB=" << signalProposal.centerIndexB
                      << " sep=" << signalProposal.separation
                      << " sepRatio=" << signalProposal.separationRatio
                      << " midpointDist=" << signalProposal.midpointDistance
                      << " axisAlign=" << signalProposal.axisAlignment
                      << " score=" << signalProposal.score
                      << " d1=(" << signalProposal.proposal.d1Pos.x
                      << "," << signalProposal.proposal.d1Pos.y
                      << "," << signalProposal.proposal.d1Pos.z << ")"
                      << " d2=(" << signalProposal.proposal.d2Pos.x
                      << "," << signalProposal.proposal.d2Pos.y
                      << "," << signalProposal.proposal.d2Pos.z << ")"
                      << std::endl;
        }
        std::cout << "[Signal Center Split Propose] frame "
                  << (firstFrame + frameIndex)
                  << " considered=" << signalSplitConsidered
                  << " eligibleFlat=" << signalSplitEligibleFlat
                  << " proposalsFound=" << signalCenterSplitProposals.size()
                  << " rodTipFallbacks="
                  << rodTipSplitFallbackProposals.size()
                  << " tooFewCenters=" << signalSplitTooFewCenters
                  << " maxElong=" << maxSignalSplitElong
                  << " threshold="
                  << config.prob.signal_center_split_min_parent_elongation
                  << std::endl;
    }

    if (signalCenterSplitProposals.size() > 1) {
        std::vector<std::pair<std::string, BridgeSplitProposal>> rankedSignalProposals;
        rankedSignalProposals.reserve(signalCenterSplitProposals.size());
        for (const auto &entry : signalCenterSplitProposals) {
            rankedSignalProposals.push_back(entry);
        }
        std::sort(rankedSignalProposals.begin(),
                  rankedSignalProposals.end(),
                  [](const auto &a, const auto &b) {
                      if (std::abs(a.second.elongation - b.second.elongation) > 1e-6f) {
                          return a.second.elongation < b.second.elongation;
                      }
                      return a.first < b.first;
                  });

        std::unordered_set<int> usedSignalSplitCenterIds;
        std::unordered_map<int, std::string> signalSplitCenterOwner;
        std::unordered_set<std::string> keptSignalSplitParents;
        for (const auto &entry : rankedSignalProposals) {
            const BridgeSplitProposal &proposal = entry.second;
            const int candidateIds[2] = {
                proposal.candidateIdA,
                proposal.candidateIdB
            };
            std::string blocker;
            for (const int candidateId : candidateIds) {
                if (candidateId < 0) continue;
                auto ownerIt = signalSplitCenterOwner.find(candidateId);
                if (ownerIt != signalSplitCenterOwner.end()) {
                    blocker = ownerIt->second;
                    break;
                }
            }
            if (!blocker.empty()) {
                std::cout << "[CellUniverse2 Split Proposal Reject] frame "
                          << displayFrame
                          << " source=signal_center_split"
                          << " parent=" << entry.first
                          << " reason=split_local_center_claim_gate"
                          << " blocker=" << blocker
                          << " candidateA=" << proposal.candidateIdA
                          << " candidateB=" << proposal.candidateIdB
                          << std::endl;
                continue;
            }
            keptSignalSplitParents.insert(entry.first);
            for (const int candidateId : candidateIds) {
                if (candidateId < 0) continue;
                usedSignalSplitCenterIds.insert(candidateId);
                signalSplitCenterOwner[candidateId] = entry.first;
            }
        }

        for (auto it = signalCenterSplitProposals.begin();
             it != signalCenterSplitProposals.end();) {
            if (keptSignalSplitParents.count(it->first) == 0) {
                it = signalCenterSplitProposals.erase(it);
            } else {
                ++it;
            }
        }
        std::cout << "[CellUniverse2 Split Proposal Local Center Claims] frame "
                  << displayFrame
                  << " kept=" << signalCenterSplitProposals.size()
                  << " centersReserved="
                  << usedSignalSplitCenterIds.size()
                  << std::endl;
    }

    // ---- PCA-bridge proposal pre-pass (split-gate-overlap-analysis.md) ----
    //
    // The PCA bridge no longer accepts splits on its own. For each elongated
    // cell with a valid dark-bridge gap, we discover a (left, right)
    // daughter-centroid proposal here and store it by cell name. The
    // proposal is then injected as a "bridge_primary" candidate inside the
    // main split loop's trySplitCellPhased call below — the standard
    // burn-in + bio + daughter-overlap + bridge + asymmetric L2 + adaptive
    // cost gates decide acceptance.
    //
    // Cells use the previous frame's PCA-fitted shape at this point (the
    // current frame's PCA fit happens in runPcaShapeFit at end of frame),
    // but the bright-pixel input is the current frame's image — same input
    // the original end-of-frame bridge had access to via the next frame's
    // start. Daughter centroids go through 50-iter burn-in inside
    // trySplitCellPhased, so any seed-position drift is corrected.
    std::unordered_map<std::string, BridgeSplitProposal> bridgeProposals;
    if (allowSplits && config.prob.pca_bridge_split_enabled) {
        float maxBridgeElong = 0.0f;
        int bridgeEligible = 0;
        for (const auto &cell : frame.cells) {
            if (cell.isTrash()) continue;
            const float elong = cell.shapeElongation();
            maxBridgeElong = std::max(maxBridgeElong, elong);
            if (elong >= config.prob.pca_bridge_elongation_ratio) {
                ++bridgeEligible;
            }
        }
        for (size_t ci = 0; ci < frame.cells.size(); ++ci) {
            const std::string parentName = frame.cells[ci].getName();
            if (cellUniverse2SplitCoolingDown(parentName)) continue;
            BridgeSplitProposal proposal;
            if (frame.discoverPcaBridgeProposal(ci, config.prob, proposal)) {
                auto snapIt = previousSnapshots.find(parentName);
                if (snapIt == previousSnapshots.end() || !snapIt->second.valid) {
                    std::cout << "[CellUniverse2 Split Proposal Reject] frame "
                              << displayFrame
                              << " source=pca_bridge_cut"
                              << " parent=" << parentName
                              << " reason=no_valid_snapshot"
                              << std::endl;
                    continue;
                }
                const bool snappedToCurrentCenter =
                    snapCellUniverse2ProposalToAlignedSignalCenters(
                        "pca_bridge_cut",
                        frame.cells[ci],
                        snapIt->second,
                        proposal);
                if (!snappedToCurrentCenter) {
                    std::cout << "[CellUniverse2 Split Proposal Center Snap Bypass] frame "
                              << displayFrame
                              << " source=pca_bridge_cut"
                              << " parent=" << parentName
                              << " reason=use_raw_seed_with_future_window_gate"
                              << std::endl;
                }
                if (!validateCellUniverse2SplitProposal(
                        "pca_bridge_cut",
                        frame.cells[ci],
                        snapIt->second,
                        proposal)) {
                    continue;
                }
                bridgeProposals[parentName] = proposal;
            }
        }
        std::cout << "[PCA Bridge Propose] frame " << (firstFrame + frameIndex)
                  << " scanned=" << frame.cells.size()
                  << " eligible=" << bridgeEligible
                  << " proposalsFound=" << bridgeProposals.size()
                  << " maxElong=" << maxBridgeElong
                  << " threshold=" << config.prob.pca_bridge_elongation_ratio
                  << std::endl;
    }

    std::unordered_map<std::string, BridgeSplitProposal> combinedLumenSplitPriors;
    const std::unordered_map<std::string, BridgeSplitProposal> *lumenSplitPriorsForFrame = nullptr;
    auto lumenPriorIt = cellLumenSplitPriors.find(frameIndex);
    const bool lumenSplitPriorPrepared =
        config.cellLumen.fusionSplitPriorEnabled &&
        lumenPriorIt != cellLumenSplitPriors.end();
    if (lumenSplitPriorPrepared && !lumenPriorIt->second.empty()) {
        combinedLumenSplitPriors = lumenPriorIt->second;
    }

    if (config.cellLumen.fusionSplitPriorEnabled &&
        config.cellLumen.fusionSplitPriorPrepassFallbackEnabled &&
        config.cellLumen.fusionSplitPriorPrepassFallbackMaxPriors > 0) {
        ScopedStageTimer fallbackTimer(displayFrame,
                                       "cell_lumen_prepass_fallback");
        struct PrepassFallbackCandidate {
            std::string parentName;
            BridgeSplitProposal proposal;
            float score = 0.0f;
            float separationRatio = 0.0f;
            float midpointDistance = 0.0f;
            float parentShape = 1.0f;
            float d1ParentMargin = 0.0f;
            float d2ParentMargin = 0.0f;
            float sourceMaxShift = 0.0f;
            int keptPixels = 0;
            std::string source = "prepass_pca";
        };

        std::vector<PrepassFallbackCandidate> fallbackCandidates;
        int fallbackConsidered = 0;
        int fallbackRejectedExistingPrior = 0;
        int fallbackRejectedMissingData = 0;
        int fallbackRejectedKept = 0;
        int fallbackRejectedShape = 0;
        int fallbackRejectedSeparation = 0;
        int fallbackRejectedClaim = 0;
        int fallbackRejectedScore = 0;
        int fallbackRejectedBadLumenParent = 0;

        const auto badParentIt = cellLumenSplitPriorRejectedBadParents.find(frameIndex);
        const std::set<std::string> *badLumenParentsForFrame =
            badParentIt != cellLumenSplitPriorRejectedBadParents.end()
                ? &badParentIt->second
                : nullptr;

        const auto nearestOtherDistance = [&](const std::string &selfName,
                                              const cv::Point3f &point) -> float {
            float best = std::numeric_limits<float>::max();
            for (const auto &other : frame.cells) {
                if (other.isTrash() || other.getName() == selfName) continue;
                const cv::Point3f otherPos(other.getX(), other.getY(), other.getZ());
                best = std::min(best, static_cast<float>(cv::norm(point - otherPos)));
            }
            return best;
        };

        for (const auto &cell : frame.cells) {
            if (cell.isTrash()) continue;
            const std::string parentName = cell.getName();
            ++fallbackConsidered;

            if (combinedLumenSplitPriors.count(parentName) > 0) {
                ++fallbackRejectedExistingPrior;
                continue;
            }
            if (config.cellLumen.fusionSplitPriorPrepassFallbackRejectBadLumenParent &&
                badLumenParentsForFrame != nullptr &&
                badLumenParentsForFrame->count(parentName) > 0) {
                ++fallbackRejectedBadLumenParent;
                continue;
            }

            auto snapIt = previousSnapshots.find(parentName);
            auto daughterIt = expectedDaughters.find(parentName);
            auto seedIt = snapshotSeedDaughters.find(parentName);
            auto shiftIt = expectedDaughterMaxShift.find(parentName);
            auto keptIt = expectedDaughterKeptPixels.find(parentName);
            if (snapIt == previousSnapshots.end() || !snapIt->second.valid ||
                daughterIt == expectedDaughters.end() || seedIt == snapshotSeedDaughters.end() ||
                keptIt == expectedDaughterKeptPixels.end()) {
                ++fallbackRejectedMissingData;
                continue;
            }

            const int keptPixels = keptIt->second;
            if (keptPixels < config.cellLumen.fusionSplitPriorPrepassFallbackMinKeptPixels) {
                ++fallbackRejectedKept;
                continue;
            }

            const auto &snap = snapIt->second;
            const float parentShape = std::max(cell.shapeElongation(), snap.shapeElongation);
            if (parentShape < config.cellLumen.fusionSplitPriorPrepassFallbackMinShape) {
                ++fallbackRejectedShape;
                continue;
            }

            const float parentMaxRadius = std::max({snap.aRadius, snap.bRadius, snap.cRadius});
            const float sourceMaxShift = shiftIt != expectedDaughterMaxShift.end()
                ? shiftIt->second
                : 0.0f;
            cv::Point3f d1 = daughterIt->second.first;
            cv::Point3f d2 = daughterIt->second.second;
            std::string source = "prepass_pca";
            if (config.cellLumen.fusionSplitPriorPrepassFallbackUseSnapshotSeedOnLargeDrift &&
                sourceMaxShift > config.cellLumen.fusionSplitPriorPrepassFallbackSeedMaxShift &&
                parentShape >= config.cellLumen.fusionSplitPriorPrepassFallbackSeedMinShape &&
                parentMaxRadius > 1e-3f) {
                const cv::Point3f seedDelta = seedIt->second.second - seedIt->second.first;
                const float seedSeparation = static_cast<float>(cv::norm(seedDelta));
                const float seedRatio = seedSeparation / parentMaxRadius;
                if (seedRatio >= config.cellLumen.fusionSplitPriorPrepassFallbackSeedMinSeparationRadiusScale &&
                    seedRatio <= config.cellLumen.fusionSplitPriorPrepassFallbackSeedMaxSeparationRadiusScale) {
                    d1 = seedIt->second.first;
                    d2 = seedIt->second.second;
                    source = "snapshot_seed_large_shift";
                }
            }
            const cv::Point3f delta = d2 - d1;
            const float separation = static_cast<float>(cv::norm(delta));
            if (parentMaxRadius <= 1e-3f || separation <= 1e-3f) {
                ++fallbackRejectedSeparation;
                continue;
            }

            const float separationRatio = separation / parentMaxRadius;
            const float minSeparationRatio = source == "snapshot_seed_large_shift"
                ? config.cellLumen.fusionSplitPriorPrepassFallbackSeedMinSeparationRadiusScale
                : config.cellLumen.fusionSplitPriorPrepassFallbackMinSeparationRadiusScale;
            const float maxSeparationRatio = source == "snapshot_seed_large_shift"
                ? config.cellLumen.fusionSplitPriorPrepassFallbackSeedMaxSeparationRadiusScale
                : config.cellLumen.fusionSplitPriorPrepassFallbackMaxSeparationRadiusScale;
            if (separationRatio < minSeparationRatio || separationRatio > maxSeparationRatio) {
                ++fallbackRejectedSeparation;
                continue;
            }

            const cv::Point3f parentPos(cell.getX(), cell.getY(), cell.getZ());
            const cv::Point3f midpoint = 0.5f * (d1 + d2);
            const float d1ParentDist = static_cast<float>(cv::norm(d1 - parentPos));
            const float d2ParentDist = static_cast<float>(cv::norm(d2 - parentPos));
            const float d1NearestOther = nearestOtherDistance(parentName, d1);
            const float d2NearestOther = nearestOtherDistance(parentName, d2);
            const float d1ParentMargin = d1NearestOther - d1ParentDist;
            const float d2ParentMargin = d2NearestOther - d2ParentDist;
            if (d1ParentMargin < config.cellLumen.fusionSplitPriorPrepassFallbackParentClaimMargin ||
                d2ParentMargin < config.cellLumen.fusionSplitPriorPrepassFallbackParentClaimMargin) {
                ++fallbackRejectedClaim;
                continue;
            }

            const float midpointDistance = static_cast<float>(cv::norm(midpoint - parentPos));
            const float score =
                midpointDistance +
                std::abs(separationRatio - 1.0f) * 25.0f -
                std::max(0.0f, parentShape - config.cellLumen.fusionSplitPriorPrepassFallbackMinShape) * 5.0f;
            const float maxScoreForSource = source == "snapshot_seed_large_shift"
                ? config.cellLumen.fusionSplitPriorPrepassFallbackSeedMaxScore
                : config.cellLumen.fusionSplitPriorPrepassFallbackMaxScore;
            if (score > maxScoreForSource) {
                ++fallbackRejectedScore;
                continue;
            }

            BridgeSplitProposal proposal;
            proposal.d1Pos = d1;
            proposal.d2Pos = d2;
            proposal.elongation = score;
            proposal.parentShapeElongation = parentShape;
            proposal.elongatedParentRescued =
                parentShape >= config.cellLumen.fusionSplitPriorPositiveGateElongatedParentMinShape;
            // -2 marks a normal pre-pass fallback proposal.  -3 is a stricter
            // marker for the snapshot-seed fallback path, where the PCA
            // daughter centroid drifted far away and the original split seeds
            // are more trustworthy than the contaminated PCA centroids.
            proposal.gapStartBin = (source == "snapshot_seed_large_shift") ? -3 : -2;
            proposal.gapEndBin = proposal.gapStartBin;
            proposal.leftPixelCount = keptPixels / 2;
            proposal.rightPixelCount = keptPixels - proposal.leftPixelCount;

            fallbackCandidates.push_back(PrepassFallbackCandidate{
                parentName,
                proposal,
                score,
                separationRatio,
                midpointDistance,
                parentShape,
                d1ParentMargin,
                d2ParentMargin,
                sourceMaxShift,
                keptPixels,
                source
            });
        }

        std::sort(fallbackCandidates.begin(), fallbackCandidates.end(),
                  [](const PrepassFallbackCandidate &a, const PrepassFallbackCandidate &b) {
                      if (std::abs(a.score - b.score) > 1e-6f) return a.score < b.score;
                      if (std::abs(a.parentShape - b.parentShape) > 1e-6f) {
                          return a.parentShape > b.parentShape;
                      }
                      const bool aSeed = a.source == "snapshot_seed_large_shift";
                      const bool bSeed = b.source == "snapshot_seed_large_shift";
                      if (aSeed != bSeed) return !aSeed;
                      return a.parentName < b.parentName;
                  });

        const int maxFallbackPriors =
            std::max(0, config.cellLumen.fusionSplitPriorPrepassFallbackMaxPriors);
        int fallbackAdded = 0;
        std::set<std::string> selectedFallbackParents;
        for (const auto &candidate : fallbackCandidates) {
            if (fallbackAdded >= maxFallbackPriors) break;
            if (combinedLumenSplitPriors.count(candidate.parentName) > 0) continue;
            combinedLumenSplitPriors[candidate.parentName] = candidate.proposal;
            ++fallbackAdded;
            selectedFallbackParents.insert(candidate.parentName);
            std::cout << "[CellLumen Prepass Fallback Prior] frame " << (firstFrame + frameIndex)
                      << " parent=" << candidate.parentName
                      << " score=" << candidate.score
                      << " sepRatio=" << candidate.separationRatio
                      << " midpointDist=" << candidate.midpointDistance
                      << " parentShape=" << candidate.parentShape
                      << " kept=" << candidate.keptPixels
                      << " d1ParentMargin=" << candidate.d1ParentMargin
                      << " d2ParentMargin=" << candidate.d2ParentMargin
                      << " source=" << candidate.source
                      << " sourceMaxShift=" << candidate.sourceMaxShift
                      << " d1=(" << candidate.proposal.d1Pos.x << "," << candidate.proposal.d1Pos.y << "," << candidate.proposal.d1Pos.z << ")"
                      << " d2=(" << candidate.proposal.d2Pos.x << "," << candidate.proposal.d2Pos.y << "," << candidate.proposal.d2Pos.z << ")"
                      << std::endl;
        }

        if (candidateGraphLog) {
            for (const auto &candidate : fallbackCandidates) {
                CandidateGraphRow row;
                row.frame = displayFrame;
                row.kind = "split_pair";
                row.source = candidate.source;
                row.parent = candidate.parentName;
                row.selected = selectedFallbackParents.count(candidate.parentName) > 0 ? 1 : 0;
                row.score = candidate.score;
                row.sep = candidate.separationRatio;
                row.midpointDist = candidate.midpointDistance;
                row.parentShape = candidate.parentShape;
                row.parentDistNear = candidate.d1ParentMargin;
                row.parentDistFar = candidate.d2ParentMargin;
                row.bridgeMetric = candidate.sourceMaxShift;
                row.d1 = candidate.proposal.d1Pos;
                row.d2 = candidate.proposal.d2Pos;
                row.voxA = candidate.proposal.leftPixelCount;
                row.voxB = candidate.proposal.rightPixelCount;
                row.note = "prepass_fallback_sep_column_is_radius_ratio";
                writeCandidateGraphRow(candidateGraphLog, row);
            }
        }

        std::cout << "[CellLumen Prepass Fallback Summary] frame " << (firstFrame + frameIndex)
                  << " considered=" << fallbackConsidered
                  << " ranked=" << fallbackCandidates.size()
                  << " added=" << fallbackAdded
                  << " rejected_existing_prior=" << fallbackRejectedExistingPrior
                  << " rejected_missing_data=" << fallbackRejectedMissingData
                  << " rejected_kept=" << fallbackRejectedKept
                  << " rejected_shape=" << fallbackRejectedShape
                  << " rejected_separation=" << fallbackRejectedSeparation
                  << " rejected_claim=" << fallbackRejectedClaim
                  << " rejected_score=" << fallbackRejectedScore
                  << " rejected_bad_lumen_parent=" << fallbackRejectedBadLumenParent
                  << std::endl;
    }

    if (!combinedLumenSplitPriors.empty()) {
        lumenSplitPriorsForFrame = &combinedLumenSplitPriors;
        std::cout << "[CellLumen SplitPrior Ready] frame " << (firstFrame + frameIndex)
                  << " priors=" << lumenSplitPriorsForFrame->size()
                  << " guided_only=" << config.cellLumen.fusionSplitPriorGuidedOnly
                  << " skip_random=" << config.cellLumen.fusionSplitPriorSkipRandomSplits
                  << std::endl;
    }
    const bool lumenSplitPriorActive =
        config.cellLumen.fusionSplitPriorEnabled &&
        lumenSplitPriorsForFrame != nullptr &&
        !lumenSplitPriorsForFrame->empty();
    const bool lumenSplitRandomDisabled =
        lumenSplitPriorPrepared &&
        config.cellLumen.fusionSplitPriorSkipRandomSplits;
    if (lumenSplitRandomDisabled) {
        std::cout << "[CellLumen SplitPrior Mode] frame " << (firstFrame + frameIndex)
                  << " random_splits_disabled=1"
                  << " priors=" << (lumenSplitPriorsForFrame ? lumenSplitPriorsForFrame->size() : 0)
                  << " reason=deterministic_cell_lumen_split_config"
                  << std::endl;
        if (!lumenSplitPriorActive) {
            std::cout << "[CellLumen SplitPrior Mode] frame " << (firstFrame + frameIndex)
                      << " no_cell_lumen_split_priors=1"
                      << " action=skip_classic_random_split"
                      << " reason=cell_lumen_high_recall_found_no_ranked_parent_pair"
                      << std::endl;
        }
    }

    const auto lumenCenterCandidatesIt = cellLumenCenterCandidates.find(frameIndex);
    const std::unordered_map<std::string, CellLumenCenterCandidate> *lumenCenterCandidatesForFrame =
        lumenCenterCandidatesIt != cellLumenCenterCandidates.end()
            ? &lumenCenterCandidatesIt->second
            : nullptr;
    if (lumenCenterCandidatesForFrame != nullptr && !lumenCenterCandidatesForFrame->empty()) {
        std::cout << "[CellLumen CenterCandidate Ready] frame " << displayFrame
                  << " candidates=" << lumenCenterCandidatesForFrame->size()
                  << std::endl;
    }

    std::set<int> reservedLumenSplitCandidateIds;
    std::unordered_map<int, std::string> reservedLumenSplitCandidateOwners;
    if (lumenSplitPriorsForFrame != nullptr) {
        for (const auto &entry : *lumenSplitPriorsForFrame) {
            const auto reserveCandidate = [&](int candidateId) {
                if (candidateId < 0) {
                    return;
                }
                reservedLumenSplitCandidateIds.insert(candidateId);
                reservedLumenSplitCandidateOwners[candidateId] = entry.first;
            };
            reserveCandidate(entry.second.candidateIdA);
            reserveCandidate(entry.second.candidateIdB);
        }
    }
    if (!reservedLumenSplitCandidateIds.empty()) {
        std::cout << "[CellLumen SplitPrior Reserved Candidates] frame "
                  << displayFrame
                  << " count=" << reservedLumenSplitCandidateIds.size()
                  << std::endl;
    }
    if (lumenSplitPriorsForFrame != nullptr) {
        int relaxedBlockers = 0;
        for (const auto &entry : *lumenSplitPriorsForFrame) {
            const BridgeSplitProposal &proposal = entry.second;
            if (!proposal.futureContinuationConflictRescued ||
                proposal.continuationClaimBlockerNames.empty()) {
                continue;
            }
            std::stringstream blockers(proposal.continuationClaimBlockerNames);
            std::string blockerName;
            while (std::getline(blockers, blockerName, '|')) {
                const bool wasAlreadyRelaxed =
                    pcaPositionGuardRelaxedCells.count(blockerName) > 0;
                relaxPcaPositionGuard(
                    blockerName,
                    "future_window_split_continuation_conflict");
                if (!wasAlreadyRelaxed) {
                    ++relaxedBlockers;
                }
            }
        }
        if (relaxedBlockers > 0) {
            std::cout << "[PCA Shape Position Guard Relax] frame "
                      << displayFrame
                      << " cells=" << relaxedBlockers
                      << " reason=future_window_split_continuation_conflict"
                      << std::endl;
        }
    }
    if (lumenCenterCandidatesForFrame != nullptr &&
        !reservedLumenSplitCandidateIds.empty()) {
        for (const auto &entry : *lumenCenterCandidatesForFrame) {
            const auto &centerCandidate = entry.second;
            if (centerCandidate.candidateId >= 0 &&
                reservedLumenSplitCandidateIds.count(centerCandidate.candidateId) > 0) {
                lockPcaPosition(entry.first, "reserved_lumen_split_candidate");
            }
        }
        if (!pcaPositionLockedCells.empty()) {
            std::cout << "[CellLumen SplitPrior Continuation Locks] frame "
                      << displayFrame
                      << " cells=" << pcaPositionLockedCells.size()
                      << " reason=reserved_lumen_split_candidate"
                      << std::endl;
        }
    }
    if (lumenCenterCandidatesForFrame != nullptr) {
        int elongatedCenterAnchorLocks = 0;
        for (const auto &entry : *lumenCenterCandidatesForFrame) {
            auto cellIt = std::find_if(
                frame.cells.begin(),
                frame.cells.end(),
                [&](const Ellipsoid &cell) {
                    return cell.getName() == entry.first;
                });
            if (cellIt == frame.cells.end()) {
                continue;
            }
            const auto &centerCandidate = entry.second;
            const float maxR = std::max({cellIt->getARadius(),
                                         cellIt->getBRadius(),
                                         cellIt->getCRadius()});
            const float lockDistance = std::min(
                std::max(0.0f, config.cellLumen.fusionCenterPriorMaxDistance),
                std::max(8.0f, maxR * 0.75f));
            const float minSignal = std::max(
                100.0f,
                std::max(0.0f, config.cellLumen.fusionMinTop10MinusShell));
            const bool highConfidenceCenter =
                centerCandidate.distance <= lockDistance &&
                centerCandidate.signal >= minSignal &&
                centerCandidate.voxelCount >=
                    std::max(0, config.cellLumen.fusionMinVoxels);
            const bool elongatedCell =
                cellIt->shapeElongation() >=
                std::max(1.85f,
                         config.cellLumen.fusionSplitPriorElongatedParentMinShape + 0.35f);
            if (!highConfidenceCenter || !elongatedCell) {
                continue;
            }
            const bool wasAlreadyLocked =
                pcaPositionLockedCells.count(entry.first) > 0;
            lockPcaPosition(entry.first, "elongated_cell_lumen_center_anchor");
            if (!wasAlreadyLocked) {
                ++elongatedCenterAnchorLocks;
            }
        }
        if (elongatedCenterAnchorLocks > 0) {
            std::cout << "[CellLumen Center Anchor PCA Locks] frame "
                      << displayFrame
                      << " cells=" << elongatedCenterAnchorLocks
                      << " reason=elongated_cell_lumen_center_anchor"
                      << std::endl;
        }
    }
    if (!pcaPositionLockedCells.empty()) {
        std::cout << "[PCA Shape Position Locks] frame "
                  << displayFrame
                  << " cells=" << pcaPositionLockedCells.size()
                  << std::endl;
    }

    struct AcceptedLumenSplitRecord {
        std::string parentName;
        BridgeSplitProposal proposal;
    };
    std::vector<AcceptedLumenSplitRecord> acceptedLumenSplits;

    // ---- Single-phase iteration helper ----
    auto runPhase = [&](std::set<std::string> &phaseNames,
                        bool phaseB,
                        std::set<std::string> &splitAcceptedInPhase,
                        std::set<std::string> &splitRejectedInPhase) {
        if (phaseNames.empty()) return;
        ScopedStageTimer phaseTimer(displayFrame, "phase_split_and_perturb");

        const size_t perCellIters = activePerCellIters;
        const size_t totalPhaseIters = perCellIters * phaseNames.size();

        // Maintain eligibility incrementally. Only changes when a split
        // accepts (parent replaced by two daughters) — no cell is ever
        // removed without replacement, and daughters inherit eligibility.
        // Splits that fail blacklist the cell for future SPLIT attempts but
        // leave it eligible for perturbation, so no index maintenance is
        // needed on rejection.
        std::vector<size_t> eligible;
        eligible.reserve(frame.cells.size());
        const auto rebuildEligible = [&]() {
            eligible.clear();
            for (size_t ci = 0; ci < frame.cells.size(); ++ci) {
                const std::string &cname = frame.cells[ci].getName();
                if (phaseNames.count(cname) == 0) continue;
                eligible.push_back(ci);
            }
        };
        rebuildEligible();

        struct PerturbOscillationState {
            size_t attempts = 0;
            bool hasLastCost = false;
            double lastCost = 0.0;
            std::deque<int> trendSigns;
            float multiplier = 1.0f;
        };

        std::unordered_map<std::string, PerturbOscillationState> perturbOscillation;
        const float oscillationWarmupFraction = std::clamp(
            config.simulation.perturb_oscillation_warmup_fraction, 0.0f, 1.0f);
        const size_t oscillationWarmupAttempts = static_cast<size_t>(std::ceil(
            static_cast<double>(perCellIters) * oscillationWarmupFraction));
        const size_t oscillationWindow = static_cast<size_t>(std::max(
            2, config.simulation.perturb_oscillation_window));
        const size_t oscillationMinSamples = static_cast<size_t>(std::max(
            2, config.simulation.perturb_oscillation_min_samples));
        const float oscillationSignChangeRatio = std::clamp(
            config.simulation.perturb_oscillation_sign_change_ratio, 0.0f, 1.0f);
        const float oscillationBoostRatio = std::max(
            1.0f, config.simulation.perturb_oscillation_boost_ratio);
        const float oscillationMaxMultiplier = std::max(
            1.0f, config.simulation.perturb_oscillation_max_multiplier);
        const float oscillationSmallStepProbability = std::clamp(
            config.simulation.perturb_oscillation_small_step_probability, 0.0f, 1.0f);
        const float oscillationSmallStepMultiplier = std::clamp(
            config.simulation.perturb_oscillation_small_step_multiplier, 0.0f, 1.0f);
        const bool oscillationBlackPixelBoostEnabled =
            config.simulation.perturb_oscillation_black_pixel_boost_enabled;
        const float oscillationBlackPixelBoostWeight = std::max(
            0.0f, config.simulation.perturb_oscillation_black_pixel_boost_weight);

        auto oscillationEnabledFor = [&](const Ellipsoid &cell) {
            return cell.isTrash()
                ? config.simulation.perturb_oscillation_boost_trash_enabled
                : config.simulation.perturb_oscillation_boost_cells_enabled;
        };

        auto perturbRatioFor = [&](const std::string &name) {
            auto it = perturbOscillation.find(name);
            const float mult = (it != perturbOscillation.end()) ? it->second.multiplier : 1.0f;
            if (mult > 1.0f && oscillationSmallStepProbability > 0.0f) {
                std::bernoulli_distribution smallStepDist(oscillationSmallStepProbability);
                if (smallStepDist(gen)) {
                    return randomPerturbRadiusRatio * oscillationSmallStepMultiplier;
                }
            }
            return randomPerturbRadiusRatio * mult;
        };

        auto recordPerturbLoss = [&](const std::string &name,
                                     const Ellipsoid &cell,
                                     double costDiff,
                                     bool accepted) {
            if (!oscillationEnabledFor(cell) || perCellIters == 0 ||
                oscillationWarmupAttempts == 0) {
                return;
            }

            auto &state = perturbOscillation[name];
            ++state.attempts;
            if (accepted && config.simulation.perturb_oscillation_reset_on_accept) {
                if (state.multiplier > 1.0f) {
                    const float oldMultiplier = state.multiplier;
                    state.multiplier = std::max(1.0f, state.multiplier / oscillationBoostRatio);
                    std::cout << "[Perturb Oscillation Boost Step Down] frame " << displayFrame
                              << " cell=" << name
                              << " attempts=" << state.attempts
                              << " multiplier=" << oldMultiplier << "->" << state.multiplier
                              << std::endl;
                } else {
                    state.multiplier = 1.0f;
                }
                state.trendSigns.clear();
                state.hasLastCost = false;
                return;
            }

            if (state.attempts > oscillationWarmupAttempts) {
                state.hasLastCost = true;
                state.lastCost = costDiff;
                return;
            }

            if (state.hasLastCost) {
                constexpr double kLossEpsilon = 1e-9;
                const double delta = costDiff - state.lastCost;
                int sign = 0;
                if (delta > kLossEpsilon) sign = 1;
                if (delta < -kLossEpsilon) sign = -1;
                if (sign != 0) {
                    state.trendSigns.push_back(sign);
                    while (state.trendSigns.size() > oscillationWindow) {
                        state.trendSigns.pop_front();
                    }
                }
            }
            state.hasLastCost = true;
            state.lastCost = costDiff;

            if (state.trendSigns.size() < oscillationMinSamples ||
                state.multiplier >= oscillationMaxMultiplier) {
                return;
            }

            int signChanges = 0;
            for (size_t si = 1; si < state.trendSigns.size(); ++si) {
                if (state.trendSigns[si] != state.trendSigns[si - 1]) {
                    ++signChanges;
                }
            }
            const float ratio = static_cast<float>(signChanges) /
                                static_cast<float>(std::max<size_t>(1, state.trendSigns.size() - 1));
            if (ratio < oscillationSignChangeRatio) {
                return;
            }

            float blackFraction = 0.0f;
            float blackBoostFactor = 1.0f;
            if (oscillationBlackPixelBoostEnabled && oscillationBlackPixelBoostWeight > 0.0f) {
                blackFraction = blackVoxelFractionInsideCell(frame, cell, config.simulation);
                blackBoostFactor += blackFraction * oscillationBlackPixelBoostWeight;
            }

            const float oldMultiplier = state.multiplier;
            const float effectiveBoostRatio = oscillationBoostRatio * blackBoostFactor;
            state.multiplier = std::min(oscillationMaxMultiplier,
                                        state.multiplier * effectiveBoostRatio);
            state.trendSigns.clear();
            std::cout << "[Perturb Oscillation Boost] frame " << displayFrame
                      << " cell=" << name
                      << " attempts=" << state.attempts
                      << " sign_change_ratio=" << ratio
                      << " black_fraction=" << blackFraction
                      << " black_boost_factor=" << blackBoostFactor
                      << " effective_boost_ratio=" << effectiveBoostRatio
                      << " multiplier=" << oldMultiplier << "->" << state.multiplier
                      << " base_radius_ratio=" << randomPerturbRadiusRatio
                      << std::endl;
        };

    std::set<std::string> lumenClassicFallbackNames;

    auto attemptSplitAtIndex = [&](size_t cellIdx,
                                   const std::string &cellName,
                                   const std::string &scheduleReason,
                                       bool useLumenPrior) {
            ScopedStageTimer splitTimer(displayFrame,
                                        "split_attempt_" + scheduleReason);
            ++splitAttempted;
            std::cout << "[Split Schedule] frame " << displayFrame
                      << " cell=" << cellName
                      << " reason=" << scheduleReason
                      << std::endl;

            auto snapIt = previousSnapshots.find(cellName);
            if (snapIt == previousSnapshots.end() || !snapIt->second.valid) {
                splitBlacklist.insert(cellName);
                std::cout << "[Split Schedule Reject] frame " << displayFrame
                          << " cell=" << cellName
                          << " reason=no_valid_snapshot"
                          << std::endl;
                return;
            }

            // useSnapDir=true adds snapshot-axis candidates alongside
            // image-PCA candidates inside trySplitCellPhased — cost picks
            // the winner. Always true now so every split attempt tries
            // both direction sources.
            const bool useSnapDir = true;

            Frame::ClaimSet others = buildOtherClaimSet(
                cellName,
                splitAcceptedInPhase,
                splitRejectedInPhase,
                phaseB);

            // Pass pre-pass direction + length via splitSnapshot so
            // trySplitCellPhased can add it as an extra primary axis
            // (labeled "imgPca"). We KEEP splitSnapshot.position at
            // the snap center so the split attempt tries BOTH midpoints:
            //   - snap_* candidates: midpoint = true snap center
            //   - data_* candidates: midpoint = pixel-projection centroid
            //     (which for the imgPca axis equals the pre-pass midpoint)
            // Snap radii preserved for daughter sizing.
            PreviousFrameSnapshot splitSnapshot = snapIt->second;
            const BridgeSplitProposal *lumenForCell = nullptr;
            if (useLumenPrior && lumenSplitPriorsForFrame != nullptr) {
                auto lIt = lumenSplitPriorsForFrame->find(cellName);
                if (lIt != lumenSplitPriorsForFrame->end()) {
                    lumenForCell = &lIt->second;
                }
            }
            if (lumenForCell != nullptr) {
                const cv::Point3f delta = lumenForCell->d2Pos - lumenForCell->d1Pos;
                const float len = static_cast<float>(cv::norm(delta));
                if (len > 1e-3f) {
                    splitSnapshot.splitAxisDir = cv::Point3f(
                        delta.x / len, delta.y / len, delta.z / len);
                    splitSnapshot.splitAxisLength = len;
                    splitSnapshot.position = 0.5f * (lumenForCell->d1Pos + lumenForCell->d2Pos);
                }
            } else if (auto itD = expectedDaughters.find(cellName); itD != expectedDaughters.end()) {
                const cv::Point3f &gd1 = itD->second.first;
                const cv::Point3f &gd2 = itD->second.second;
                const cv::Point3f delta = gd2 - gd1;
                const float len = static_cast<float>(cv::norm(delta));
                if (len > 1e-3f) {
                    splitSnapshot.splitAxisDir = cv::Point3f(
                        delta.x / len, delta.y / len, delta.z / len);
                    splitSnapshot.splitAxisLength = len;
                }
            }

            // If the post-PCA pass produced a bridge proposal for this cell,
            // hand it to trySplitCellPhased. CellUniverse 2 pca_bridge_cut
            // uses the bridge proposal as the only candidate; legacy split
            // scheduling still lets it compete with data_/snap_ candidates.
            const BridgeSplitProposal *bridgeForCell = nullptr;
            if (scheduleReason == "signal_center_split") {
                auto sIt = signalCenterSplitProposals.find(cellName);
                if (sIt != signalCenterSplitProposals.end()) {
                    bridgeForCell = &sIt->second;
                }
            } else if (scheduleReason == "rod_tip_split_fallback") {
                auto rIt = rodTipSplitFallbackProposals.find(cellName);
                if (rIt != rodTipSplitFallbackProposals.end()) {
                    bridgeForCell = &rIt->second;
                }
            } else if (!bridgeProposals.empty()) {
                auto bIt = bridgeProposals.find(cellName);
                if (bIt != bridgeProposals.end()) {
                    bridgeForCell = &bIt->second;
                }
            }
            const bool deterministicSingleProposal =
                scheduleReason == "pca_bridge_cut" ||
                scheduleReason == "signal_center_split" ||
                scheduleReason == "rod_tip_split_fallback";

            auto result = frame.trySplitCellPhased(
                cellIdx, splitSnapshot, others, useSnapDir, config.prob,
                exportPerturbDebug ? &splitPerturbDebugPlacements : nullptr,
                exportPerturbDebug ? &splitPerturbDebugPlacementCount : nullptr,
                config.simulation.perturb_debug_cell_brightness,
                bridgeForCell,
                deterministicSingleProposal,
                lumenForCell,
                lumenForCell != nullptr && config.cellLumen.fusionSplitPriorGuidedOnly,
                config.cellLumen.fusionSplitPriorBurnInIterations,
                config.cellLumen.fusionSplitPriorRefineIterations,
                config.cellLumen.fusionSplitPriorUseDedicatedCostGate,
                config.cellLumen.fusionSplitPriorUseImageCostGate,
                config.cellLumen.fusionSplitPriorCost,
                config.cellLumen.fusionSplitPriorCostFraction,
                config.cellLumen.fusionSplitPriorMaxPositiveCostFraction,
                config.cellLumen.fusionSplitPriorPositiveGateMinImageGain,
                config.cellLumen.fusionSplitPriorPositiveGateMinImageGainPenaltyRatio,
                config.cellLumen.fusionSplitPriorPositiveGateElongatedParentMinShape,
                config.cellLumen.fusionSplitPriorPositiveGateElongatedMaxRawWorsening,
                config.cellLumen.fusionSplitPriorPositiveGateElongatedMaxSoftPenaltyFraction,
                config.cellLumen.fusionSplitPriorPositiveGateElongatedMaxScore,
                config.cellLumen.fusionSplitPriorMaxOverlapCostFraction,
                config.cellLumen.fusionSplitPriorHighConfidenceMaxScore,
                config.cellLumen.fusionSplitPriorHighConfidenceMaxOverlapCostFraction,
                config.cellLumen.fusionSplitPriorHighConfidenceAxisAlignmentDegrees,
                config.cellLumen.fusionSplitPriorDaughterVolumeScale,
                config.cellLumen.fusionSplitPriorPrefilterMaxValleyRatio,
                config.cellLumen.fusionSplitPriorBridgeMaxValleyRatio,
                config.cellLumen.fusionSplitPriorMinBridgeGapWidth,
                config.cellLumen.fusionSplitPriorMinEdgeBrightness,
                config.cellLumen.fusionSplitPriorMaxDaughterOverlapFraction,
                config.cellLumen.fusionSplitPriorSoftGateEnabled,
                config.cellLumen.fusionSplitPriorSoftAxisPenaltyFraction,
                config.cellLumen.fusionSplitPriorSoftDaughterOverlapPenaltyFraction,
                config.cellLumen.fusionSplitPriorSoftValleyPenaltyFraction,
                config.cellLumen.fusionSplitPriorSoftBridgeGapPenaltyFraction,
                config.cellLumen.fusionSplitPriorSoftOverlapCostPenaltyWeight,
                config.cellLumen.fusionSplitPriorBridgeEvidenceWaivesOverlapSoftPenalty,
                config.cellLumen.fusionSplitPriorHardMaxDaughterOverlapFraction,
                config.cellLumen.fusionSplitPriorHardMaxValleyRatio,
                config.cellLumen.fusionSplitPriorHardMaxOverlapCostFraction,
                config.cellLumen.fusionSplitPriorMinPostRefitLateralSeparation,
                config.cellLumen.fusionSplitPriorMinPostRefitLateralSeparationRadiusScale,
                config.cellLumen.fusionSplitPriorMaxZDominanceForLowLateralSeparation,
                config.cellLumen.fusionSplitPriorDynamicOverlapEnabled,
                config.cellLumen.fusionSplitPriorLocalDensityRadiusScale,
                config.cellLumen.fusionSplitPriorLocalDensityOverlapBonus,
                config.cellLumen.fusionSplitPriorMaxDynamicDaughterOverlapFraction,
                config.cellLumen.fusionSplitPriorSnapshotSeedMaxRefitDrift,
                config.cellLumen.fusionSplitPriorSkipExistingCellBuriedCheck,
                config.cellLumen.fusionSplitPriorSkipNeighborBridgeCheck);

            double costDiff = result.first;
            auto callback = result.second;

            const bool accept = (costDiff < 0.0);
            callback(accept);
            if (accept) {
                ++splitAccepted;
                splitAcceptedInPhase.insert(cellName);
                previousSnapshots.erase(cellName);
                if (lumenForCell != nullptr) {
                    acceptedLumenSplits.push_back({cellName, *lumenForCell});
                }
                // Add daughter names to phaseNames so they become eligible
                // for perturbation during the rest of this frame. Without
                // this, rebuildEligible skips daughters (their names weren't
                // in phaseNames at frame start) and they get zero perturbation
                // iterations — only the split burn-in + refine positioning.
                phaseNames.insert(cellName + "0");
                phaseNames.insert(cellName + "1");
                // Rebuild eligible: the parent cell at cellIdx was replaced
                // by two daughters appended to the cells vector.
                rebuildEligible();
                // Recompute mean brightness — cell count changed.
                {
                    float bSum = 0.0f;
                    for (const auto &c : frame.cells) bSum += c.getBrightness();
                    frame.setMeanCellBrightness(
                        bSum / static_cast<float>(frame.cells.size()));
                }
                // Rebuild Voronoi territory — cell count and positions
                // changed (parent replaced, two daughters appended).
                // Daughters don't have a snap entry yet, so their live
                // positions become their Voronoi anchors going forward.
                if (voronoiMapNeeded) {
                    frame.rebuildVoronoiMap();
                }
                return;
            }

            const bool allowBridgeRetryAfterSignalReject =
                config.simulation.celluniverse2_enabled &&
                scheduleReason == "signal_center_split" &&
                bridgeProposals.find(cellName) != bridgeProposals.end();
            if (allowBridgeRetryAfterSignalReject) {
                std::cout << "[Split Signal Reject Bridge Retry] frame "
                          << displayFrame
                          << " cell=" << cellName
                          << " reason=defer_blacklist_for_pca_bridge_cut"
                          << std::endl;
                return;
            }

            const bool wasCellLumenPriorAttempt = (scheduleReason == "cell_lumen_prior");
            const bool allowClassicFallback =
                wasCellLumenPriorAttempt &&
                config.cellLumen.fusionSplitPriorFallbackToClassicOnReject;
            if (!allowClassicFallback) {
                splitBlacklist.insert(cellName);
                splitRejectedInPhase.insert(cellName);
            } else {
                lumenClassicFallbackNames.insert(cellName);
                std::cout << "[CellLumen SplitPrior Fallback] frame " << displayFrame
                          << " cell=" << cellName
                          << " reason=prior_rejected_schedule_classic_once"
                          << std::endl;
            }

            // Compensation perturb. The revert left cells[cellIdx] as the
            // parent, so no find_if is needed as long as cellIdx still points
            // at the same parent after the rejected split callback.
            auto compResult = frame.perturbCell(
                cellIdx, overlapWeight, useSignalGuidanceThisFrame,
                randomPerturbRadiusRatio,
                /*pcaRefitWellFilledMove=*/false,
                /*useSignalMapGuidance=*/false);
            const bool compAccept = compResult.first < 0.0;
            logPerturbCandidate("split_reject_compensation",
                                cellName,
                                frame.cells[cellIdx],
                                compResult.first,
                                compAccept,
                                randomPerturbRadiusRatio,
                                "score_is_total_cost_diff");
            if (exportPerturbDebug) {
                accumulateDebugCellPlacement(
                    splitPerturbDebugPlacements,
                    frame.cells[cellIdx],
                    config.simulation,
                    config.simulation.perturb_debug_cell_brightness);
                ++splitPerturbDebugPlacementCount;
            }
            compResult.second(compAccept);
            if (compAccept) {
                ++perturbAccepted;
                residSum += compResult.first;
                absResidSum += std::abs(compResult.first);
                ++residCount;
            }
        };

        // CellUniverse2 signal-center proposals are the earliest split probe:
        // if accepted, the parent is replaced by two daughters before the
        // normal final PCA fit. If rejected, the parent remains and the older
        // bridge-cut/default PCA paths still run.
        if (allowSignalCenterSplit && !signalCenterSplitProposals.empty()) {
            for (const auto &entry : signalCenterSplitProposals) {
                const std::string &signalSplitName = entry.first;
                if (phaseNames.count(signalSplitName) == 0 ||
                    cellUniverse2SplitCoolingDown(signalSplitName) ||
                    splitBlacklist.count(signalSplitName) > 0) {
                    continue;
                }
                auto it = std::find_if(
                    frame.cells.begin(), frame.cells.end(),
                    [&](const Ellipsoid &cell) {
                        return cell.getName() == signalSplitName;
                    });
                if (it == frame.cells.end()) continue;
                const size_t cellIdx = static_cast<size_t>(
                    std::distance(frame.cells.begin(), it));
                attemptSplitAtIndex(cellIdx, signalSplitName, "signal_center_split",
                                    /*useLumenPrior=*/false);
            }
            rebuildEligible();
        }

        // Image-grounded bridge proposal candidates are tried once up front
        // so correctness does not depend on random scheduling. Acceptance
        // still uses the same gates as every normal random split attempt.
        if (allowPcaBridgeSplitCut && !bridgeProposals.empty()) {
            for (const auto &entry : bridgeProposals) {
                const std::string &bridgeName = entry.first;
                if (phaseNames.count(bridgeName) == 0 ||
                    cellUniverse2SplitCoolingDown(bridgeName) ||
                    splitBlacklist.count(bridgeName) > 0) {
                    continue;
                }
                auto it = std::find_if(
                    frame.cells.begin(), frame.cells.end(),
                    [&](const Ellipsoid &cell) {
                        return cell.getName() == bridgeName;
                    });
                if (it == frame.cells.end()) continue;
                const size_t cellIdx = static_cast<size_t>(
                    std::distance(frame.cells.begin(), it));
                attemptSplitAtIndex(cellIdx, bridgeName, "pca_bridge_cut",
                                    /*useLumenPrior=*/false);
            }
            rebuildEligible();
        }

        if (allowSignalCenterSplit && !rodTipSplitFallbackProposals.empty()) {
            for (const auto &entry : rodTipSplitFallbackProposals) {
                const std::string &fallbackName = entry.first;
                if (phaseNames.count(fallbackName) == 0 ||
                    cellUniverse2SplitCoolingDown(fallbackName) ||
                    splitBlacklist.count(fallbackName) > 0) {
                    continue;
                }
                auto it = std::find_if(
                    frame.cells.begin(), frame.cells.end(),
                    [&](const Ellipsoid &cell) {
                        return cell.getName() == fallbackName;
                    });
                if (it == frame.cells.end()) continue;
                const size_t cellIdx = static_cast<size_t>(
                    std::distance(frame.cells.begin(), it));
                attemptSplitAtIndex(cellIdx,
                                    fallbackName,
                                    "rod_tip_split_fallback",
                                    /*useLumenPrior=*/false);
            }
            rebuildEligible();
        }

        if (allowStandardSplits && !forcedSplitNames.empty() &&
            !lumenSplitRandomDisabled) {
            for (const auto &forcedName : forcedSplitNames) {
                if (phaseNames.count(forcedName) == 0 ||
                    splitBlacklist.count(forcedName) > 0) {
                    continue;
                }
                auto it = std::find_if(
                    frame.cells.begin(), frame.cells.end(),
                    [&](const Ellipsoid &cell) {
                        return cell.getName() == forcedName;
                    });
                if (it == frame.cells.end()) continue;
                const size_t cellIdx = static_cast<size_t>(
                    std::distance(frame.cells.begin(), it));
                attemptSplitAtIndex(cellIdx, forcedName, "prepass_force",
                                    /*useLumenPrior=*/false);
            }
            rebuildEligible();
        }

        if (allowStandardSplits &&
            lumenSplitPriorActive &&
            config.cellLumen.fusionSplitPriorForceSchedule) {
            for (const auto &entry : *lumenSplitPriorsForFrame) {
                const std::string &forcedName = entry.first;
                if (phaseNames.count(forcedName) == 0 ||
                    splitBlacklist.count(forcedName) > 0) {
                    continue;
                }
                auto it = std::find_if(
                    frame.cells.begin(), frame.cells.end(),
                    [&](const Ellipsoid &cell) {
                        return cell.getName() == forcedName;
                    });
                if (it == frame.cells.end()) continue;
                const size_t cellIdx = static_cast<size_t>(
                    std::distance(frame.cells.begin(), it));
                attemptSplitAtIndex(cellIdx, forcedName, "cell_lumen_prior",
                                    /*useLumenPrior=*/true);
            }
            rebuildEligible();
        }

        if (allowStandardSplits && !lumenClassicFallbackNames.empty()) {
            const std::vector<std::string> fallbackNames(
                lumenClassicFallbackNames.begin(), lumenClassicFallbackNames.end());
            lumenClassicFallbackNames.clear();
            for (const auto &fallbackName : fallbackNames) {
                if (phaseNames.count(fallbackName) == 0 ||
                    splitBlacklist.count(fallbackName) > 0) {
                    continue;
                }
                auto it = std::find_if(
                    frame.cells.begin(), frame.cells.end(),
                    [&](const Ellipsoid &cell) {
                        return cell.getName() == fallbackName;
                    });
                if (it == frame.cells.end()) continue;
                const size_t cellIdx = static_cast<size_t>(
                    std::distance(frame.cells.begin(), it));
                attemptSplitAtIndex(cellIdx, fallbackName,
                                    "cell_lumen_classic_fallback",
                                    /*useLumenPrior=*/false);
            }
            rebuildEligible();
        }

        ScopedStageTimer perturbLoopTimer(displayFrame, "perturb_iteration_loop");
        for (size_t i = 0; i < totalPhaseIters; ++i) {
            if (eligible.empty()) break;

            std::uniform_int_distribution<size_t> cellDist(0, eligible.size() - 1);
            const size_t cellIdx = eligible[cellDist(gen)];
            // VALUE COPY — must not be a reference. trySplitCellPhased and
            // perturbCell mutate frame.cells (erase/push_back/full reassign
            // via savedCells), invalidating any reference to the original
            // ellipsoid's _name string. A dangling cellName silently caused
            // splitBlacklist to be inserted with garbage content, defeating
            // the "max one split attempt per cell per frame" invariant —
            // the same cell would re-attempt because its real name wasn't
            // found in the blacklist (the blacklist had a corrupted entry).
            // Observed as the "8cbdf86d gets 2 split attempts" symptom.
            const std::string cellName = frame.cells[cellIdx].getName();

            float pSplit = 0.0f;
            const bool randomSplitsAllowed =
                allowStandardSplits &&
                !lumenSplitRandomDisabled;
            if (randomSplitsAllowed) {
                auto pIt = splitProbabilities.find(cellName);
                if (pIt != splitProbabilities.end()) pSplit = pIt->second;
            }
            const bool canSplit = pSplit > 0.0f
                               && splitBlacklist.count(cellName) == 0;

            if (canSplit && uniform01(gen) < pSplit) {
                attemptSplitAtIndex(cellIdx, cellName, "random",
                                    /*useLumenPrior=*/false);
            } else {
                // --- Perturbation ---
                const float effectivePerturbRadiusRatio = perturbRatioFor(cellName);
                bool perturbHandled = false;
                if (lumenCenterCandidatesForFrame != nullptr) {
                    auto centerIt = lumenCenterCandidatesForFrame->find(cellName);
                    if (centerIt != lumenCenterCandidatesForFrame->end()) {
                        const auto &centerCandidate = centerIt->second;
                        const bool reservedBySplit =
                            centerCandidate.candidateId >= 0 &&
                            reservedLumenSplitCandidateIds.count(
                                centerCandidate.candidateId) > 0;
                        if (reservedBySplit) {
                            auto ownerIt =
                                reservedLumenSplitCandidateOwners.find(
                                    centerCandidate.candidateId);
                            std::ostringstream note;
                            note << "score_is_total_cost_diff"
                                 << ";lumen_candidate_id="
                                 << centerCandidate.candidateId
                                 << ";lumen_distance="
                                 << centerCandidate.distance
                                 << ";lumen_signal="
                                 << centerCandidate.signal
                                 << ";lumen_voxels="
                                 << centerCandidate.voxelCount
                                 << ";reserved_by_split_parent="
                                 << (ownerIt !=
                                             reservedLumenSplitCandidateOwners.end()
                                         ? ownerIt->second
                                         : std::string("unknown"));
                            logPerturbCandidate(
                                "cell_lumen_center_candidate_reserved_by_split",
                                cellName,
                                frame.cells[cellIdx],
                                0.0,
                                false,
                                effectivePerturbRadiusRatio,
                                note.str());
                            lockPcaPosition(cellName,
                                            "reserved_lumen_split_candidate");
                            perturbHandled = true;
                        } else {
                        const cv::Point3f oldPos(frame.cells[cellIdx].getX(),
                                                 frame.cells[cellIdx].getY(),
                                                 frame.cells[cellIdx].getZ());
                        const float blend = std::clamp(
                            config.cellLumen.fusionCenterPriorPositionBlend,
                            0.0f, 1.0f);
                        const cv::Point3f target =
                            oldPos * (1.0f - blend) + centerCandidate.position * blend;
                        auto centerResult = frame.perturbCell(
                            cellIdx, overlapWeight, useSignalGuidanceThisFrame,
                            effectivePerturbRadiusRatio,
                            /*pcaRefitWellFilledMove=*/true,
                            /*useSignalMapGuidance=*/false,
                            &target);
                        double centerCostDiff = centerResult.first;
                        auto centerCallback = centerResult.second;
                        const bool centerAccept = centerCostDiff < 0;
                        std::ostringstream note;
                        note << "score_is_total_cost_diff"
                             << ";lumen_candidate_id=" << centerCandidate.candidateId
                             << ";lumen_distance=" << centerCandidate.distance
                             << ";lumen_signal=" << centerCandidate.signal
                             << ";lumen_voxels=" << centerCandidate.voxelCount
                             << ";blend=" << blend;
                        logPerturbCandidate("cell_lumen_center_candidate",
                                            cellName,
                                            frame.cells[cellIdx],
                                            centerCostDiff,
                                            centerAccept,
                                            effectivePerturbRadiusRatio,
                                            note.str());
                        if (exportPerturbDebug) {
                            accumulateDebugCellPlacement(
                                movementPerturbDebugPlacements,
                                frame.cells[cellIdx],
                                config.simulation,
                                config.simulation.perturb_debug_cell_brightness);
                            ++movementPerturbDebugPlacementCount;
                        }
                        recordPerturbLoss(cellName, frame.cells[cellIdx],
                                          centerCostDiff, centerAccept);
                        if (centerAccept) {
                            centerCallback(true);
                            perturbAccepted++;
                            residSum += centerCostDiff;
                            absResidSum += std::abs(centerCostDiff);
                            residCount++;
                            perturbHandled = true;
                        } else {
                            centerCallback(false);
                        }
                        }
                    }
                }

                if (!perturbHandled) {
                    auto result = frame.perturbCell(
                        cellIdx, overlapWeight, useSignalGuidanceThisFrame,
                        effectivePerturbRadiusRatio,
                        /*pcaRefitWellFilledMove=*/true);
                    double costDiff = result.first;
                    auto callback = result.second;
                    bool perturbAccept = costDiff < 0;
                    std::ostringstream perturbNote;
                    perturbNote << "score_is_total_cost_diff";
                    if (perturbAccept && !useSignalGuidanceThisFrame) {
                        auto snapIt = previousSnapshots.find(cellName);
                        if (snapIt != previousSnapshots.end() && snapIt->second.valid) {
                            const PreviousFrameSnapshot &snap = snapIt->second;
                            const cv::Point3f candidatePos(frame.cells[cellIdx].getX(),
                                                           frame.cells[cellIdx].getY(),
                                                           frame.cells[cellIdx].getZ());
                            const float snapMaxR =
                                std::max({snap.aRadius, snap.bRadius, snap.cRadius});
                            const float driftFromSnapshot =
                                static_cast<float>(cv::norm(candidatePos - snap.position));
                            const float driftLimit =
                                snapPositionGuardLimit(snapMaxR, false);
                            bool hasNearbyLumenEvidence = false;
                            if (lumenCenterCandidatesForFrame != nullptr) {
                                auto centerIt =
                                    lumenCenterCandidatesForFrame->find(cellName);
                                if (centerIt != lumenCenterCandidatesForFrame->end()) {
                                    const auto &centerCandidate = centerIt->second;
                                    const float lumenDist =
                                        static_cast<float>(cv::norm(
                                            candidatePos - centerCandidate.position));
                                    hasNearbyLumenEvidence =
                                        lumenDist <= std::max(8.0f, snapMaxR * 0.45f) &&
                                        centerCandidate.signal >= 100.0f;
                                }
                            }
                            if (driftLimit > 0.0f &&
                                driftFromSnapshot > driftLimit &&
                                !hasNearbyLumenEvidence) {
                                perturbAccept = false;
                                perturbNote << ";large_random_drift_rejected=1"
                                            << ";drift_from_snapshot="
                                            << driftFromSnapshot
                                            << ";drift_limit=" << driftLimit;
                            }
                        }
                    }
                    logPerturbCandidate(useSignalGuidanceThisFrame
                                            ? "signal_guided_local_search"
                                            : "random_local_search",
                                        cellName,
                                        frame.cells[cellIdx],
                                        costDiff,
                                        perturbAccept,
                                        effectivePerturbRadiusRatio,
                                        perturbNote.str());
                    if (exportPerturbDebug) {
                        accumulateDebugCellPlacement(
                            movementPerturbDebugPlacements,
                            frame.cells[cellIdx],
                            config.simulation,
                            config.simulation.perturb_debug_cell_brightness);
                        ++movementPerturbDebugPlacementCount;
                    }
                    recordPerturbLoss(cellName, frame.cells[cellIdx], costDiff, perturbAccept);
                    if (perturbAccept) {
                        callback(true);
                        perturbAccepted++;
                        residSum += costDiff;
                        absResidSum += std::abs(costDiff);
                        residCount++;
                    } else {
                        callback(false);
                    }
                }
            }

            if ((i + 1) % 500 == 0) {
                std::cout << "Frame " << displayFrame
                          << " iter=" << i
                          << " perturb_accepted=" << perturbAccepted
                          << " split_attempts=" << splitAttempted
                          << " split_accepted=" << splitAccepted
                          << " cells=" << frame.cells.size() << std::endl;
            }
        }
        perturbLoopTimer.finish();
    };

    // Unified loop over ALL cells. Classification removed; all cells share
    // the same perturb+split iteration budget, and the split attempt gets
    // both snap-axis and image-PCA candidates regardless of snap elongation.
    std::set<std::string> allNames;
    for (const auto &cell : frame.cells) {
        allNames.insert(cell.getName());
    }
    std::set<std::string> splitAcceptedInLoop;
    std::set<std::string> splitRejectedInLoop;

    runPhase(allNames, /* phaseB */ true, splitAcceptedInLoop, splitRejectedInLoop);

    if (!acceptedLumenSplits.empty()) {
        auto isRealCandidateId = [](int candidateId) {
            return candidateId >= 0 && candidateId < 1000000000;
        };
        auto findCellIndexByName = [&](const std::string &name) -> int {
            for (size_t ci = 0; ci < frame.cells.size(); ++ci) {
                if (frame.cells[ci].getName() == name) {
                    return static_cast<int>(ci);
                }
            }
            return -1;
        };
        auto maxRadiusOf = [](const Ellipsoid &cell) {
            return std::max({cell.getARadius(), cell.getBRadius(), cell.getCRadius()});
        };
        auto centerOf = [](const Ellipsoid &cell) {
            return cv::Point3f(cell.getX(), cell.getY(), cell.getZ());
        };

        int continuationMergeCount = 0;
        std::set<std::string> mergedContinuations;
        for (const AcceptedLumenSplitRecord &record : acceptedLumenSplits) {
            const BridgeSplitProposal &proposal = record.proposal;
            const bool realA = isRealCandidateId(proposal.candidateIdA);
            const bool realB = isRealCandidateId(proposal.candidateIdB);
            const bool parentAnchorOneReal =
                proposal.parentAnchored && (realA != realB);
            const bool cleanWindow =
                proposal.windowBothDaughtersSupported >= 2 &&
                proposal.windowMissingDaughterCount == 0 &&
                proposal.windowParentPersists == 0;
            const bool parentAnchorMergeMode =
                parentAnchorOneReal &&
                cleanWindow &&
                proposal.parentShapeElongation >= 1.75f &&
                proposal.neighborClaimPenalty <= 1e-5f &&
                proposal.parentPersistencePenalty <= 1e-5f;
            const bool weakNeighborHandoffMode =
                !proposal.parentAnchored &&
                cleanWindow &&
                proposal.elongatedParentRescued &&
                proposal.parentDistanceBalance < 0.68f &&
                proposal.neighborClaimPenalty <= 1e-5f &&
                proposal.continuationClaimSoftPenalty <= 1e-5f &&
                proposal.parentPersistencePenalty <= 1e-5f;
            if (!parentAnchorMergeMode && !weakNeighborHandoffMode) {
                continue;
            }

            std::vector<std::string> daughterNames;
            if (parentAnchorMergeMode) {
                daughterNames.push_back(realA ? record.parentName + "1"
                                              : record.parentName + "0");
            } else {
                daughterNames.push_back(record.parentName + "0");
                daughterNames.push_back(record.parentName + "1");
            }

            bool mergedThisSplit = false;
            for (const std::string &daughterName : daughterNames) {
                const int daughterIdx = findCellIndexByName(daughterName);
                if (daughterIdx < 0) {
                    continue;
                }
                const Ellipsoid &daughter = frame.cells[static_cast<size_t>(daughterIdx)];
                const cv::Point3f dPos = centerOf(daughter);
                const float dMaxR = maxRadiusOf(daughter);

                int bestOtherIdx = -1;
                float bestNorm = std::numeric_limits<float>::infinity();
                float bestDist = std::numeric_limits<float>::infinity();
                for (size_t oi = 0; oi < frame.cells.size(); ++oi) {
                    if (static_cast<int>(oi) == daughterIdx) continue;
                    const Ellipsoid &other = frame.cells[oi];
                    const std::string &otherName = other.getName();
                    if (other.isTrash()) continue;
                    if (otherName == record.parentName) continue;
                    if (otherName == record.parentName + "0" ||
                        otherName == record.parentName + "1") {
                        continue;
                    }
                    if (allNames.count(otherName) == 0) continue;
                    if (splitAcceptedInLoop.count(otherName) > 0) continue;
                    if (mergedContinuations.count(otherName) > 0) continue;

                    const float oMaxR = maxRadiusOf(other);
                    const float dist = static_cast<float>(cv::norm(dPos - centerOf(other)));
                    const float norm = dist / std::max(1.0f, dMaxR + oMaxR);
                    const float normLimit = parentAnchorMergeMode ? 0.72f : 0.68f;
                    const float distLimit = parentAnchorMergeMode ? 30.0f : 32.0f;
                    if (norm > normLimit || dist > distLimit) {
                        continue;
                    }
                    if (norm < bestNorm) {
                        bestNorm = norm;
                        bestDist = dist;
                        bestOtherIdx = static_cast<int>(oi);
                    }
                }

                if (bestOtherIdx < 0) {
                    continue;
                }

                const std::string continuationName =
                    frame.cells[static_cast<size_t>(bestOtherIdx)].getName();
                EllipsoidParams mergedParams = daughter.getCellParams();
                mergedParams.name = continuationName;
                frame.cells[static_cast<size_t>(bestOtherIdx)] = Ellipsoid(mergedParams);
                frame.cells.erase(frame.cells.begin() + daughterIdx);
                previousSnapshots.erase(daughterName);
                cellShapeReference.erase(daughterName);
                cellShapeBirth.erase(daughterName);
                mergedContinuations.insert(continuationName);
                ++continuationMergeCount;
                mergedThisSplit = true;
                std::cout << "[Split Continuation Merge] frame " << displayFrame
                          << " parent=" << record.parentName
                          << " daughter=" << daughterName
                          << " continuation=" << continuationName
                          << " dist=" << bestDist
                          << " normalized=" << bestNorm
                          << " mode="
                          << (parentAnchorMergeMode
                                  ? "parent_anchor_one_real"
                                  : "weak_neighbor_handoff")
                          << std::endl;
                break;
            }

            (void)mergedThisSplit;
        }

        if (continuationMergeCount > 0) {
            frame.regenerateSynthFrame();
            if (voronoiMapNeeded) {
                frame.rebuildVoronoiMap();
            }
            std::cout << "[Split Continuation Merge Summary] frame "
                      << displayFrame
                      << " merged=" << continuationMergeCount
                      << " cells=" << frame.cells.size()
                      << std::endl;
        }
    }

    if (exportPerturbDebug) {
        const fs::path framePath = (frameIndex >= 0 && static_cast<size_t>(frameIndex) < imagePaths.size())
            ? imagePaths[static_cast<size_t>(frameIndex)]
            : fs::path(frame.getImageName());
        std::cout << "[Perturb Debug Export] frame " << displayFrame
                  << " movement_placements=" << movementPerturbDebugPlacementCount
                  << " split_placements=" << splitPerturbDebugPlacementCount
                  << " brightness=" << config.simulation.perturb_debug_cell_brightness
                  << " output_dir=" << (fs::path(outputPath) / "perturb_debug").string()
                  << std::endl;
        exportStackToSubdir(movementPerturbDebugPlacements,
                            fs::path(outputPath),
                            "perturb_debug/movement_placements",
                            framePath,
                            config.simulation.export_frame_png,
                            config.simulation.export_frame_tiff);
        exportStackToSubdir(splitPerturbDebugPlacements,
                            fs::path(outputPath),
                            "perturb_debug/split_placements",
                            framePath,
                            config.simulation.export_frame_png,
                            config.simulation.export_frame_tiff);
    }

    if (exportPerturbCenterDebug) {
        const fs::path framePath = (frameIndex >= 0 && static_cast<size_t>(frameIndex) < imagePaths.size())
            ? imagePaths[static_cast<size_t>(frameIndex)]
            : fs::path(frame.getImageName());
        std::vector<cv::Mat> cellCenterLabels = makeCellCenterLabelDebugStack(
            frame.getRealFrame(), frame.cells,
            config.simulation.perturb_debug_center_cube_radius);
        std::cout << "[Perturb Debug Center Export] frame " << displayFrame
                  << " cells=" << frame.cells.size()
                  << " cube_radius=" << config.simulation.perturb_debug_center_cube_radius
                  << " output_dir=" << (fs::path(outputPath) / "perturb_debug/cell_centers").string()
                  << std::endl;
        exportStackToSubdir(cellCenterLabels,
                            fs::path(outputPath),
                            "perturb_debug/cell_centers",
                            framePath,
                            config.simulation.export_frame_png,
                            config.simulation.export_frame_tiff);
    }

    {
        ScopedStageTimer pcaTimer(displayFrame, "final_pca_shape");
        runPcaShapeFit();
    }

    // End of frame: build PreviousFrameSnapshot for each cell, combining the
    // in-frame running max (from the periodic sampling above) with the
    // end-of-frame PCA measurement. The snapshot is the authoritative source
    // for next-frame split decisions (PR-B trigger, PR-C placement, PR-D peaks).
    // PR-E: previousElongations removed — snapshots are the sole data store.
    //
    // Don't clear previousSnapshots — entries for cells that disappeared this
    // frame (e.g., parents replaced by daughters) become stale but are
    // overwritten below for surviving cells. Daughters get fresh snapshots.
    // We do erase entries that don't exist anymore to prevent unbounded growth.
    {
        std::set<std::string> liveCells;
        for (const auto &cell : frame.cells) {
            liveCells.insert(cell.getName());
        }
        for (auto it = previousSnapshots.begin(); it != previousSnapshots.end();) {
            if (liveCells.find(it->first) == liveCells.end()) {
                it = previousSnapshots.erase(it);
            } else {
                ++it;
            }
        }
        for (auto it = cellFirstSeenFrame.begin(); it != cellFirstSeenFrame.end();) {
            if (liveCells.find(it->first) == liveCells.end()) {
                it = cellFirstSeenFrame.erase(it);
            } else {
                ++it;
            }
        }
    }
    std::cout << "[Snapshot] frame " << displayFrame << std::endl;
    for (size_t ci = 0; ci < frame.cells.size(); ++ci) {
        auto p = frame.cells[ci].getCellParams();
        if (cellFirstSeenFrame.find(p.name) == cellFirstSeenFrame.end()) {
            cellFirstSeenFrame[p.name] = displayFrame;
        }

        // Triaxial fitted-shape elongation is the classification signal.
        // max(a,b,c)/min(a,b,c) from the fit, plus the world-space direction
        // and length of the longest axis. No image-PCA anymore.
        const float fitShapeElong = frame.cells[ci].shapeElongation();
        cv::Point3f fitSplitAxisDir;
        float fitSplitAxisLength = 0.0f;
        frame.cells[ci].worldSplitAxis(fitSplitAxisDir, fitSplitAxisLength);

        PreviousFrameSnapshot snap;
        snap.valid = true;
        snap.shapeElongation = fitShapeElong;
        snap.splitAxisDir = fitSplitAxisDir;
        snap.splitAxisLength = fitSplitAxisLength;

        snap.position = cv::Point3f(p.x, p.y, p.z);
        snap.aRadius = p.aRadius;
        snap.bRadius     = p.bRadius;
        snap.cRadius = p.cRadius;
        snap.thetaX = p.theta_x;
        snap.thetaY = p.theta_y;
        snap.thetaZ = p.theta_z;
        snap.brightness = p.brightness;

        previousSnapshots[p.name] = snap;

        std::cout << "  " << p.name
                  << " shapeElong=" << snap.shapeElongation
                  << " splitAxisLen=" << snap.splitAxisLength
                  << " pos=(" << snap.position.x
                  << "," << snap.position.y
                  << "," << snap.position.z << ")"
                  << std::endl;
    }

    const float brightnessBlend = std::clamp(config.cell ? config.cell->brightnessUpdateBlend : 0.0f, 0.0f, 1.0f);
    if (brightnessBlend > 0.0f && config.cell) {
        const auto &realFrame = frame.getRealFrame();
        const float brightnessAmplification = std::max(0.0f, config.cell->brightnessMeanAmplification);
        const float brightnessMeasurementTopPercentile =
            std::clamp(config.cell->brightnessMeasurementTopPercentile, 0.0f, 1.0f);
        for (auto &cell : frame.cells) {
            const float observedBrightness =
                cell.measureMeanBrightness(realFrame, brightnessMeasurementTopPercentile);
            const float amplifiedObservedBrightness = observedBrightness * brightnessAmplification;
            const float updatedBrightness =
                cell.getBrightness() * (1.0f - brightnessBlend) + amplifiedObservedBrightness * brightnessBlend;
            cell.setBrightness(updatedBrightness);
        }
        frame.regenerateSynthFrame();
    }

    bool trashRemovalUpdatedBackground = false;
    if (config.cell && config.cell->trashRemovalEnabled) {
        const float threshold = config.cell->trashRemovalBrightnessThreshold;
        const int consecutiveDimFrames =
            std::max(1, config.cell->trashRemovalConsecutiveDimFrames);
        float removedBrightnessSum = 0.0f;
        std::vector<std::string> removedNames;

        auto &cells = frame.cells;
        for (auto it = cells.begin(); it != cells.end();) {
            if (!it->isTrash()) {
                ++it;
                continue;
            }

            const std::string trashName = it->getName();
            const bool dim = it->getBrightness() < threshold;
            const int dimFrames = dim
                ? ++trashDimFrameCounts[trashName]
                : 0;
            if (!dim) {
                trashDimFrameCounts.erase(trashName);
            }

            if (dim && dimFrames >= consecutiveDimFrames) {
                removedBrightnessSum += it->getBrightness();
                removedNames.push_back(trashName);
                trashDimFrameCounts.erase(trashName);
                it = cells.erase(it);
                continue;
            }
            ++it;
        }

        if (!removedNames.empty()) {
            for (const std::string &name : removedNames) {
                previousSnapshots.erase(name);
                cellShapeReference.erase(name);
                cellShapeBirth.erase(name);
                trashDimFrameCounts.erase(name);
            }

            std::size_t backgroundVoxels =
                countBackgroundVoxelsAfterCells(frame, config.simulation);
            if (backgroundVoxels == 0 && !frame.getRealFrame().empty()) {
                const auto &realFrame = frame.getRealFrame();
                backgroundVoxels = static_cast<std::size_t>(realFrame.front().rows) *
                                   static_cast<std::size_t>(realFrame.front().cols) *
                                   realFrame.size();
            }
            backgroundVoxels = std::max<std::size_t>(backgroundVoxels, 1);

            const float backgroundDelta =
                removedBrightnessSum / static_cast<float>(backgroundVoxels);
            const float updatedBackground = std::clamp(
                frame.getBackgroundValue() + backgroundDelta, 0.0f, 1.0f);
            frame.setBackgroundColor(updatedBackground);
            frame.regenerateSynthFrame();
            trashRemovalUpdatedBackground = true;

            std::cout << "[Trash Removal] frame " << displayFrame
                      << " removed=" << removedNames.size()
                      << " consecutiveDimFrames=" << consecutiveDimFrames
                      << " brightnessSum=" << removedBrightnessSum
                      << " backgroundVoxels=" << backgroundVoxels
                      << " backgroundDelta=" << backgroundDelta
                      << " background=" << updatedBackground
                      << std::endl;
        }
    }

    std::cout << "[Optimize Done] frame " << displayFrame
              << " perturb_accepted=" << perturbAccepted
              << " split_attempts=" << splitAttempted
              << " split_accepted=" << splitAccepted
              << " final_cells=" << frame.cells.size() << std::endl;

    if (config.simulation.n2v2_contrast_gamma_split_decay_enabled &&
        splitAccepted > 0) {
        cumulativeAcceptedSplits += splitAccepted;
        invalidatePreparedFramesAfter(frameIndex);
        std::cout << "[Gamma Split Decay] frame=" << displayFrame
                  << " accepted_splits_this_frame=" << splitAccepted
                  << " accepted_splits_total=" << cumulativeAcceptedSplits
                  << " next_effective_gamma=" << effectiveN2V2ContrastGamma()
                  << '\n';
    }

    // M1/M2 cache per-frame summaries so optimize(frameIndex+1) doesn't need
    // frames[frameIndex]'s image stacks.
    perFrameAdaptiveBackground[frameIndex] = trashRemovalUpdatedBackground
        ? frame.getBackgroundValue()
        : (config.simulation.celluniverse2_enabled
            ? estimateAdaptiveBackgroundFromLocalCellBboxes(
                frame, config.simulation, firstFrame + frameIndex)
            : estimateAdaptiveBackgroundFromFrame(frame, config.simulation));
    perFrameMeanBrightness[frameIndex] = computeStackMean(frame.getRealFrame());
}

void CellUniverse::releaseFrameImages(int frameIndex)
{
    if (frameIndex < 0 || static_cast<size_t>(frameIndex) >= frames.size())
    {
        return;
    }
    frames[frameIndex].releaseImageStacks();
}

void CellUniverse::saveImages(int frameIndex, const std::string &stage)
{
    if (frameIndex < 0 || static_cast<size_t>(frameIndex) >= frames.size())
    {
        throw std::invalid_argument("Invalid frame index");
    }

    std::vector<Image> realImages = frames[frameIndex].generateOutputFrame();
    std::vector<Image> synthImages = frames[frameIndex].generateOutputSynthFrame();
    int displayFrame = firstFrame + frameIndex;
    std::cout << "Saving images for frame " << displayFrame << "..." << '\n';
    std::cout << "Real Image Type: " << realImages[0].type() << '\n';
    std::cout << "Synth Image Type: " << synthImages[0].type() << '\n';
    const std::string exportRoot = stage.empty() ? outputPath : (outputPath + "/" + stage);

    if (config.simulation.export_frame_png)
    {
        std::string realOutputPath = exportRoot + "/png/real/" + std::to_string(displayFrame);
        if (!std::filesystem::exists(realOutputPath))
        {
            std::filesystem::create_directories(realOutputPath);
        }
        std::string synthOutputPath = exportRoot + "/png/synth/" + std::to_string(displayFrame);
        if (!std::filesystem::exists(synthOutputPath))
        {
            std::filesystem::create_directories(synthOutputPath);
        }

        // Parallelize PNG encoding/write: independent across slices and stacks.
        const int nSlices = static_cast<int>(std::max(realImages.size(), synthImages.size()));
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < nSlices; ++i)
        {
            if (static_cast<size_t>(i) < realImages.size()) {
                cv::imwrite(realOutputPath + "/" + std::to_string(i) + ".png", realImages[i]);
            }
            if (static_cast<size_t>(i) < synthImages.size()) {
                cv::imwrite(synthOutputPath + "/" + std::to_string(i) + ".png", synthImages[i]);
            }
        }
    }

    if (config.simulation.export_frame_tiff)
    {
        const std::string realTiffOutputPath = exportRoot + "/tiff/real";
        const std::string synthTiffOutputPath = exportRoot + "/tiff/synth";
        std::filesystem::create_directories(realTiffOutputPath);
        std::filesystem::create_directories(synthTiffOutputPath);

        writeNapariFriendlyTiffStack(realTiffOutputPath + "/" + std::to_string(displayFrame) + ".tif",
                                     realImages);
        writeNapariFriendlyTiffStack(synthTiffOutputPath + "/" + std::to_string(displayFrame) + ".tif",
                                     synthImages);
    }

    std::cout << "Done" << '\n';
}

void CellUniverse::saveCells(int frameIndex)
{
    std::string cellsPath = outputPath + "/cells.csv";
    bool fileExists = std::filesystem::exists(cellsPath);

    // Append mode: each frame adds its rows as it finishes optimizing
    std::ofstream file(cellsPath, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << cellsPath << " for writing" << '\n';
        return;
    }

    // Write header only for the first frame
    if (!fileExists || frameIndex == 0) {
        // Truncate if frame 0 (fresh run)
        if (frameIndex == 0) {
            file.close();
            file.open(cellsPath, std::ios::trunc);
        }
        file << "file,name,x,y,z,aRadius,bRadius,cRadius,theta_x,theta_y,theta_z,isTrash" << '\n';
    }

    Frame &frame = frames[frameIndex];
    std::string imageName = frame.getImageName();

    for (const auto &cell : frame.cells) {
        EllipsoidParams params = cell.getCellParams();
        cell.printCellInfo();
        file << imageName << ","
             << params.name << ","
             << params.x << ","
             << params.y << ","
             << params.z << ","
             << params.aRadius << ","
             << params.bRadius << ","
             << params.cRadius << ","
             << params.theta_x << ","
             << params.theta_y << ","
             << params.theta_z << ","
             << (params.isTrash ? 1 : 0)
             << '\n';
    }

    std::cout << "Saved " << frame.cells.size() << " cells for frame " << (firstFrame + frameIndex)
              << " to " << cellsPath << std::endl;
}

// ---------------------------------------------------------------------------
// Checkpoint save / load (Approach 2).
// ---------------------------------------------------------------------------
//
// Format: simple line-oriented text. One directive per line, space-separated
// fields. Each checkpoint file holds the full state needed to resume an
// optimization run at frameIndex+1.

void CellUniverse::saveCheckpoint(int frameIndex)
{
    if (frameIndex < 0 || static_cast<size_t>(frameIndex) >= frames.size()) return;
    const std::string dir = outputPath + "/checkpoints";
    fs::create_directories(dir);
    char buf[64];
    const int absoluteFrame = firstFrame + frameIndex;
    std::snprintf(buf, sizeof(buf), "frame_%03d.txt", absoluteFrame);
    const std::string path = dir + "/" + buf;

    std::ofstream out(path);
    if (!out.is_open()) {
        std::cerr << "[Checkpoint] failed to open " << path << '\n';
        return;
    }

    // Scalar header
    out << "frame " << absoluteFrame << '\n';
    out << "z_slices " << config.simulation.z_slices << '\n';
    out << "maxZ " << Ellipsoid::cellConfig.maxZ << '\n';

    // Per-frame summaries (indexed by frameIndex, since we cache AFTER
    // optimize(frameIndex) completes).
    const float adaptiveBg = (static_cast<size_t>(frameIndex) < perFrameAdaptiveBackground.size())
        ? perFrameAdaptiveBackground[frameIndex] : 0.0f;
    const float meanBright = (static_cast<size_t>(frameIndex) < perFrameMeanBrightness.size())
        ? perFrameMeanBrightness[frameIndex] : 0.0f;
    out << "perFrameAdaptiveBackground " << adaptiveBg << '\n';
    out << "perFrameMeanBrightness " << meanBright << '\n';

    // Next frame's background value (what optimize(frameIndex+1) will use
    // as initial bg for the frame being optimized).
    if (static_cast<size_t>(frameIndex + 1) < frames.size()) {
        out << "nextFrameBackgroundValue " << frames[frameIndex + 1].getBackgroundValue() << '\n';
    }

    // Cells of frameIndex + 1 (already copied forward, ready for optimize).
    // If frame+1 doesn't exist (last frame), save frame N's cells instead.
    const size_t cellsFrameIdx = (static_cast<size_t>(frameIndex + 1) < frames.size())
        ? static_cast<size_t>(frameIndex + 1) : static_cast<size_t>(frameIndex);
    const auto &cells = frames[cellsFrameIdx].cells;
    for (const auto &cell : cells) {
        auto p = cell.getCellParams();
        out << "cell " << p.name
            << " " << p.x << " " << p.y << " " << p.z
            << " " << p.aRadius << " " << p.bRadius << " " << p.cRadius
            << " " << p.theta_x << " " << p.theta_y << " " << p.theta_z
            << " " << p.brightness
            << " " << (p.isTrash ? 1 : 0) << '\n';
    }

    // Previous-frame snapshots (per-cell PCA fit state for split detection).
    for (const auto &kv : previousSnapshots) {
        const auto &s = kv.second;
        out << "snap " << kv.first
            << " " << (s.valid ? 1 : 0)
            << " " << s.shapeElongation
            << " " << s.splitAxisDir.x << " " << s.splitAxisDir.y << " " << s.splitAxisDir.z
            << " " << s.splitAxisLength
            << " " << s.position.x << " " << s.position.y << " " << s.position.z
            << " " << s.aRadius << " " << s.bRadius << " " << s.cRadius
            << " " << s.thetaX << " " << s.thetaY << " " << s.thetaZ
            << " " << s.brightness << '\n';
    }

    // Shape reference (bounded growth ref for fit-cap).
    for (const auto &kv : cellShapeReference) {
        out << "ref " << kv.first
            << " " << kv.second[0] << " " << kv.second[1] << " " << kv.second[2] << '\n';
    }

    // Shape birth (frozen birth radii for mask + birth cap).
    for (const auto &kv : cellShapeBirth) {
        out << "birth " << kv.first
            << " " << kv.second[0] << " " << kv.second[1] << " " << kv.second[2] << '\n';
    }

    // First-seen absolute frame. Used by CellUniverse2's young daughter
    // localization probe; omitted in older checkpoints, which remain valid.
    for (const auto &kv : cellFirstSeenFrame) {
        out << "first_seen " << kv.first << " " << kv.second << '\n';
    }

    out.close();
    std::cout << "[Checkpoint] saved frame " << absoluteFrame << " → " << path << std::endl;
}

bool CellUniverse::loadCheckpoint(int frameIndex, const std::string &checkpointPath)
{
    return loadCheckpoint(frameIndex, frameIndex + 1, checkpointPath);
}

bool CellUniverse::loadCheckpoint(int checkpointFrameIndex,
                                  int targetFrameIndex,
                                  const std::string &checkpointPath)
{
    std::ifstream in(checkpointPath);
    if (!in.is_open()) {
        std::cerr << "[Checkpoint] cannot open " << checkpointPath << '\n';
        return false;
    }

    // Clear maps that will be repopulated.
    previousSnapshots.clear();
    cellShapeReference.clear();
    cellShapeBirth.clear();
    cellFirstSeenFrame.clear();
    resumePreviousFrameSummaryValid = false;
    resumePreviousAdaptiveBackground = 0.0f;
    resumePreviousMeanBrightness = 0.0f;
    resumePreviousCellCount = 0;

    if (targetFrameIndex < 0 || static_cast<size_t>(targetFrameIndex) >= frames.size()) {
        std::cerr << "[Checkpoint] target frame out of range: " << targetFrameIndex << '\n';
        return false;
    }
    const size_t cellsFrameIdx = static_cast<size_t>(targetFrameIndex);
    frames[cellsFrameIdx].cells.clear();

    const int targetAbsoluteFrame = firstFrame + targetFrameIndex;
    const bool checkpointMatchesLocal =
        checkpointFrameIndex == (targetFrameIndex - 1);
    const int expectedCheckpointFrame = targetAbsoluteFrame - 1;
    if (checkpointFrameIndex != expectedCheckpointFrame && !checkpointMatchesLocal) {
        std::cerr << "[Checkpoint] caller checkpoint frame " << checkpointFrameIndex
                  << " does not match expected absolute " << expectedCheckpointFrame
                  << " for target local frame " << targetFrameIndex << '\n';
    }

    std::string line;
    int loadedCells = 0, loadedSnaps = 0, loadedRefs = 0, loadedBirths = 0;
    int loadedFirstSeen = 0;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string tag;
        ss >> tag;
        if (tag == "frame") {
            int f; ss >> f;
            // Accept both the new absolute frame tag and legacy local tags
            // from older first_frame=0 runs.
            if (f != checkpointFrameIndex && f != targetFrameIndex - 1) {
                std::cerr << "[Checkpoint] frame mismatch: file=" << f
                          << " expected=" << checkpointFrameIndex << '\n';
            }
        } else if (tag == "z_slices") {
            int v; ss >> v;
            config.simulation.z_slices = v;
        } else if (tag == "maxZ") {
            float v; ss >> v;
            Ellipsoid::cellConfig.maxZ = v;
        } else if (tag == "perFrameAdaptiveBackground") {
            float v; ss >> v;
            resumePreviousAdaptiveBackground = v;
            resumePreviousFrameSummaryValid = true;
            if (targetFrameIndex > 0) {
                perFrameAdaptiveBackground[targetFrameIndex - 1] = v;
            }
        } else if (tag == "perFrameMeanBrightness") {
            float v; ss >> v;
            resumePreviousMeanBrightness = v;
            resumePreviousFrameSummaryValid = true;
            if (targetFrameIndex > 0) {
                perFrameMeanBrightness[targetFrameIndex - 1] = v;
            }
        } else if (tag == "nextFrameBackgroundValue") {
            float v; ss >> v;
            frames[cellsFrameIdx].setBackgroundColor(v);
        } else if (tag == "cell") {
            EllipsoidParams p;
            ss >> p.name >> p.x >> p.y >> p.z
               >> p.aRadius >> p.bRadius >> p.cRadius
               >> p.theta_x >> p.theta_y >> p.theta_z
               >> p.brightness;
            int isTrashInt = 0;
            if (ss >> isTrashInt) {
                p.isTrash = (isTrashInt != 0);
            }
            frames[cellsFrameIdx].cells.emplace_back(p);
            ++loadedCells;
        } else if (tag == "snap") {
            PreviousFrameSnapshot s;
            std::string name;
            int validInt = 0;
            ss >> name >> validInt >> s.shapeElongation
               >> s.splitAxisDir.x >> s.splitAxisDir.y >> s.splitAxisDir.z
               >> s.splitAxisLength
               >> s.position.x >> s.position.y >> s.position.z
               >> s.aRadius >> s.bRadius >> s.cRadius
               >> s.thetaX >> s.thetaY >> s.thetaZ
               >> s.brightness;
            s.valid = (validInt != 0);
            previousSnapshots[name] = s;
            ++loadedSnaps;
        } else if (tag == "ref") {
            std::string name; std::array<float, 3> r;
            ss >> name >> r[0] >> r[1] >> r[2];
            cellShapeReference[name] = r;
            ++loadedRefs;
        } else if (tag == "birth") {
            std::string name; std::array<float, 3> r;
            ss >> name >> r[0] >> r[1] >> r[2];
            cellShapeBirth[name] = r;
            ++loadedBirths;
        } else if (tag == "first_seen") {
            std::string name;
            int firstSeenFrame = checkpointFrameIndex;
            ss >> name >> firstSeenFrame;
            cellFirstSeenFrame[name] = firstSeenFrame;
            ++loadedFirstSeen;
        }
    }
    in.close();
    if (loadedFirstSeen == 0) {
        const int matureFallbackFirstSeen = checkpointFrameIndex - 1000;
        for (const Ellipsoid &cell : frames[cellsFrameIdx].cells) {
            cellFirstSeenFrame[cell.getName()] = matureFallbackFirstSeen;
        }
        loadedFirstSeen = static_cast<int>(cellFirstSeenFrame.size());
        std::cout << "[Checkpoint] first_seen absent; treating loaded cells as mature"
                  << " fallbackFirstSeen=" << matureFallbackFirstSeen
                  << " cells=" << loadedFirstSeen << std::endl;
    }
    resumePreviousCellCount = frames[cellsFrameIdx].cells.size();
    if (config.simulation.n2v2_contrast_gamma_split_decay_enabled &&
        initialNonTrashCellCount > 0) {
        const int inferredAcceptedSplits = std::max(
            0,
            static_cast<int>(countNonTrashCellsInFrame(targetFrameIndex)) -
                initialNonTrashCellCount);
        cumulativeAcceptedSplits =
            std::max(cumulativeAcceptedSplits, inferredAcceptedSplits);
        std::cout << "[Gamma Split Decay] checkpoint_inferred_splits="
                  << inferredAcceptedSplits
                  << " accepted_splits_total=" << cumulativeAcceptedSplits
                  << " next_effective_gamma=" << effectiveN2V2ContrastGamma()
                  << '\n';
    }

    std::cout << "[Checkpoint] loaded frame " << checkpointFrameIndex
              << " into local frame " << targetFrameIndex
              << " (absolute " << targetAbsoluteFrame << ")"
              << " cells=" << loadedCells
              << " snaps=" << loadedSnaps
              << " refs=" << loadedRefs
              << " births=" << loadedBirths
              << " firstSeen=" << loadedFirstSeen << std::endl;
    return true;
}

void CellUniverse::copyCellsForward(size_t to)
{
    if (to >= frames.size())
    {
        return;
    }
    // assumes cells have deepcopy copy constructors
    frames[to].cells = frames[to - 1].cells;
    if (config.cell) {
        for (auto &cell : frames[to].cells) {
            cell.blendAdaptivePerturbProbabilitiesWithConfig(
                config.cell->brightnessProbabilityTrust,
                config.cell->aRadiusProbabilityTrust,
                config.cell->cRadiusProbabilityTrust,
                config.cell->bRadiusProbabilityTrust);
        }
    }
}

unsigned int CellUniverse::length()
{
    return frames.size();
}

const std::vector<Ellipsoid> &CellUniverse::getCells(int frameIndex) const
{
    if (frameIndex < 0 || static_cast<size_t>(frameIndex) >= frames.size())
    {
        throw std::invalid_argument("CellUniverse::getCells - invalid frameIndex");
    }
    return frames[frameIndex].cells;
}

std::vector<std::string> CellUniverse::getCellNames(int frameIndex) const
{
    const auto &cells = getCells(frameIndex);
    std::vector<std::string> names;
    names.reserve(cells.size());
    for (const auto &c : cells)
    {
        names.push_back(c.getCellParams().name);
    }
    return names;
}
