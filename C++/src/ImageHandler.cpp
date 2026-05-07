#include "../includes/ImageHandler.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>
#include <limits>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>

namespace
{
int makeOddAtLeast(int value, int minimum = 3)
{
    int adjusted = std::max(value, minimum);
    if (adjusted % 2 == 0)
    {
        ++adjusted;
    }
    return adjusted;
}

bool shouldIgnoreImagePath(const fs::path &file)
{
    const std::string name = file.filename().string();
    return name.empty() || name[0] == '.' || name.rfind("._", 0) == 0;
}

void updateTiffConfigIfNeeded(const fs::path &file, BaseConfig &config)
{
    if (shouldIgnoreImagePath(file) || !(file.extension() == ".tif" || file.extension() == ".tiff"))
    {
        return;
    }

    std::vector<cv::Mat> images;
    cv::imreadmulti(file.string(), images, cv::IMREAD_UNCHANGED);
    config.simulation.z_slices = static_cast<int>(images.size());
}

struct StackStats
{
    double minValue = 0.0;
    double maxValue = 0.0;
    double mean = 0.0;
    double stddev = 0.0;
    double saturatedFraction = 0.0;
    std::size_t count = 0;
    std::size_t nonFiniteCount = 0;
};

StackStats computeStackStats(const ImageStack &stack)
{
    StackStats stats;
    double sum = 0.0;
    double sumSq = 0.0;
    double minValue = std::numeric_limits<double>::infinity();
    double maxValue = -std::numeric_limits<double>::infinity();
    std::size_t count = 0;
    std::size_t saturatedCount = 0;
    std::size_t nonFiniteCount = 0;

    #pragma omp parallel for schedule(static) reduction(+:sum,sumSq,count,saturatedCount,nonFiniteCount) reduction(min:minValue) reduction(max:maxValue)
    for (int i = 0; i < static_cast<int>(stack.size()); ++i)
    {
        const auto &slice = stack[static_cast<std::size_t>(i)];
        CV_Assert(slice.type() == CV_32F);

        for (int y = 0; y < slice.rows; ++y)
        {
            const float *row = slice.ptr<float>(y);
            for (int x = 0; x < slice.cols; ++x)
            {
                const double value = row[x];
                if (!std::isfinite(value))
                {
                    ++nonFiniteCount;
                    ++count;
                    continue;
                }
                minValue = std::min(minValue, value);
                maxValue = std::max(maxValue, value);
                if (value >= 1.0 - 1e-6 && value <= 1.0 + 1e-6)
                {
                    ++saturatedCount;
                }
                sum += value;
                sumSq += value * value;
                ++count;
            }
        }
    }

    stats.count = count;
    if (stats.count == 0)
    {
        return stats;
    }

    stats.minValue = minValue;
    stats.maxValue = maxValue;
    stats.mean = sum / static_cast<double>(stats.count);
    stats.saturatedFraction =
        static_cast<double>(saturatedCount) / static_cast<double>(stats.count);
    stats.nonFiniteCount = nonFiniteCount;
    const double variance =
        std::max(0.0, sumSq / static_cast<double>(stats.count) - stats.mean * stats.mean);
    stats.stddev = std::sqrt(variance);
    return stats;
}

void printStackStats(std::ostream &log,
                     const std::string &stage,
                     const std::string &imageFile,
                     const ImageStack &stack)
{
    const StackStats stats = computeStackStats(stack);
    log << "[Preprocess] file=" << fs::path(imageFile).filename().string()
        << " stage=" << stage
        << " slices=" << stack.size()
        << " voxels=" << stats.count
        << " min=" << stats.minValue
        << " max=" << stats.maxValue
        << " mean=" << stats.mean
        << " stddev=" << stats.stddev
        << " saturated_fraction=" << stats.saturatedFraction
        << " nonfinite=" << stats.nonFiniteCount
        << std::endl;
}

ImageStack cloneStack(const ImageStack &sequence)
{
    ImageStack cloned(sequence.size());

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(sequence.size()); ++i)
    {
        cloned[static_cast<std::size_t>(i)] = sequence[static_cast<std::size_t>(i)].clone();
    }
    return cloned;
}

ImageStack applyCubePooling(const ImageStack &stack,
                            const BaseConfig &config,
                            std::ostream &log)
{
    if (stack.empty() || !config.simulation.cube_pooling_enabled)
    {
        return cloneStack(stack);
    }

    const int depth = static_cast<int>(stack.size());
    const int rows = stack[0].rows;
    const int cols = stack[0].cols;
    if (depth <= 0 || rows <= 0 || cols <= 0)
    {
        return cloneStack(stack);
    }

    const int cubeSize = std::max(1, config.simulation.cube_pooling_cube_size);
    std::string poolingMode = config.simulation.cube_pooling_mode;
    std::transform(poolingMode.begin(), poolingMode.end(), poolingMode.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    if (poolingMode != "mean" && poolingMode != "max" &&
        poolingMode != "min" && poolingMode != "median" &&
        poolingMode != "top_percentile" && poolingMode != "low_percentile")
    {
        log << "[CubePooling] unsupported mode=" << config.simulation.cube_pooling_mode
            << "; using mean" << std::endl;
        poolingMode = "mean";
    }
    const float topFraction = std::clamp(config.simulation.cube_pooling_top_fraction, 0.0f, 1.0f);
    const float lowFraction = std::clamp(config.simulation.cube_pooling_low_fraction, 0.0f, 1.0f);
    const int gridZ = (depth + cubeSize - 1) / cubeSize;
    const int gridY = (rows + cubeSize - 1) / cubeSize;
    const int gridX = (cols + cubeSize - 1) / cubeSize;

    ImageStack pooled(depth);
    for (int z = 0; z < depth; ++z)
    {
        pooled[static_cast<size_t>(z)] = cv::Mat::zeros(rows, cols, CV_32F);
    }

    #pragma omp parallel for collapse(3) schedule(static)
    for (int gz = 0; gz < gridZ; ++gz)
    {
        for (int gy = 0; gy < gridY; ++gy)
        {
            for (int gx = 0; gx < gridX; ++gx)
            {
                const int z0 = gz * cubeSize;
                const int z1 = std::min(depth, z0 + cubeSize);
                const int y0 = gy * cubeSize;
                const int y1 = std::min(rows, y0 + cubeSize);
                const int x0 = gx * cubeSize;
                const int x1 = std::min(cols, x0 + cubeSize);

                double sum = 0.0;
                float extrema = poolingMode == "min"
                    ? std::numeric_limits<float>::infinity()
                    : -std::numeric_limits<float>::infinity();
                int voxelCount = 0;
                std::vector<float> pooledValues;
                if (poolingMode == "median" ||
                    poolingMode == "top_percentile" ||
                    poolingMode == "low_percentile")
                {
                    pooledValues.reserve(
                        static_cast<std::size_t>((z1 - z0) * (y1 - y0) * (x1 - x0)));
                }
                for (int z = z0; z < z1; ++z)
                {
                    const cv::Mat &slice = stack[static_cast<size_t>(z)];
                    for (int y = y0; y < y1; ++y)
                    {
                        const float *row = slice.ptr<float>(y);
                        for (int x = x0; x < x1; ++x)
                        {
                            const float value = row[x];
                            sum += value;
                            if (poolingMode == "median" ||
                                poolingMode == "top_percentile" ||
                                poolingMode == "low_percentile")
                            {
                                pooledValues.push_back(value);
                            }
                            else if (poolingMode == "min")
                            {
                                extrema = std::min(extrema, value);
                            }
                            else if (poolingMode == "max")
                            {
                                extrema = std::max(extrema, value);
                            }
                            ++voxelCount;
                        }
                    }
                }
                float pooledValue = 0.0f;
                if (voxelCount > 0)
                {
                    if (poolingMode == "mean")
                    {
                        pooledValue = static_cast<float>(sum / static_cast<double>(voxelCount));
                    }
                    else if (poolingMode == "median")
                    {
                        const std::size_t medianIndex = pooledValues.size() / 2U;
                        std::nth_element(pooledValues.begin(),
                                         pooledValues.begin() + static_cast<std::ptrdiff_t>(medianIndex),
                                         pooledValues.end());
                        pooledValue = pooledValues[medianIndex];
                    }
                    else if (poolingMode == "top_percentile")
                    {
                        const std::size_t selectedCount = std::max<std::size_t>(
                            1U,
                            static_cast<std::size_t>(
                                std::ceil(std::max(topFraction, 1e-6f) *
                                          static_cast<float>(pooledValues.size()))));
                        const std::size_t thresholdIndex = pooledValues.size() - selectedCount;
                        std::nth_element(pooledValues.begin(),
                                         pooledValues.begin() + static_cast<std::ptrdiff_t>(thresholdIndex),
                                         pooledValues.end());
                        double topSum = 0.0;
                        for (std::size_t i = thresholdIndex; i < pooledValues.size(); ++i)
                        {
                            topSum += pooledValues[i];
                        }
                        pooledValue = static_cast<float>(
                            topSum / static_cast<double>(selectedCount));
                    }
                    else if (poolingMode == "low_percentile")
                    {
                        const std::size_t selectedCount = std::max<std::size_t>(
                            1U,
                            static_cast<std::size_t>(
                                std::ceil(std::max(lowFraction, 1e-6f) *
                                          static_cast<float>(pooledValues.size()))));
                        const std::size_t lastSelectedIndex = selectedCount - 1U;
                        std::nth_element(pooledValues.begin(),
                                         pooledValues.begin() + static_cast<std::ptrdiff_t>(lastSelectedIndex),
                                         pooledValues.end());
                        double lowSum = 0.0;
                        for (std::size_t i = 0; i < selectedCount; ++i)
                        {
                            lowSum += pooledValues[i];
                        }
                        pooledValue = static_cast<float>(
                            lowSum / static_cast<double>(selectedCount));
                    }
                    else
                    {
                        pooledValue = extrema;
                    }
                }

                for (int z = z0; z < z1; ++z)
                {
                    cv::Mat &slice = pooled[static_cast<size_t>(z)];
                    for (int y = y0; y < y1; ++y)
                    {
                        float *row = slice.ptr<float>(y);
                        for (int x = x0; x < x1; ++x)
                        {
                            row[x] = pooledValue;
                        }
                    }
                }
            }
        }
    }

    log << "[CubePooling]"
        << " cubeSize=" << cubeSize
        << " grid=" << gridX << "x" << gridY << "x" << gridZ
        << " mode=" << poolingMode
        << " top_fraction=" << topFraction
        << " low_fraction=" << lowFraction
        << std::endl;

    return pooled;
}

struct BrightBox
{
    int ix = 0;
    int iy = 0;
    int iz = 0;
    cv::Point3f center{0.0f, 0.0f, 0.0f};
    float brightness = 0.0f;
    int voxels = 0;
};

int chooseNearestDivisorSize(int axisLength, float targetSize)
{
    if (axisLength <= 1)
    {
        return 1;
    }

    const float clampedTarget = std::clamp(targetSize, 1.0f, static_cast<float>(axisLength));
    int bestDivisor = 1;
    float bestDistance = std::abs(static_cast<float>(bestDivisor) - clampedTarget);
    for (int d = 1; d <= axisLength; ++d)
    {
        if (axisLength % d != 0)
        {
            continue;
        }
        const float distance = std::abs(static_cast<float>(d) - clampedTarget);
        if (distance < bestDistance ||
            (std::abs(distance - bestDistance) <= 1e-6f && d > bestDivisor))
        {
            bestDivisor = d;
            bestDistance = distance;
        }
    }
    return bestDivisor;
}

std::vector<Frame::SignalCenter> localizeSignalCentersInStack(const ImageStack &stack,
                                                              const BaseConfig &config)
{
    std::vector<Frame::SignalCenter> centers;
    if (stack.empty() || !config.cell)
    {
        return centers;
    }

    const int sizeX = stack[0].cols;
    const int sizeY = stack[0].rows;
    const int sizeZ = static_cast<int>(stack.size());
    if (sizeX <= 0 || sizeY <= 0 || sizeZ <= 0)
    {
        return centers;
    }

    const float targetBoxSize =
        static_cast<float>(std::max(1, config.simulation.signal_guided_box_side_length));

    const int boxSizeX = chooseNearestDivisorSize(sizeX, targetBoxSize);
    const int boxSizeY = chooseNearestDivisorSize(sizeY, targetBoxSize);
    const int boxSizeZ = chooseNearestDivisorSize(sizeZ, targetBoxSize);
    const int gridX = std::max(1, sizeX / boxSizeX);
    const int gridY = std::max(1, sizeY / boxSizeY);
    const int gridZ = std::max(1, sizeZ / boxSizeZ);

    const float backgroundValue = 0.0f;
    const float minDelta = std::max(0.0f, config.simulation.signal_guided_min_box_brightness_delta);

    std::vector<BrightBox> boxes;
    boxes.reserve(static_cast<size_t>(gridX) * gridY * gridZ);
    for (int iz = 0; iz < gridZ; ++iz)
    {
        const int z0 = iz * boxSizeZ;
        const int z1 = z0 + boxSizeZ;
        for (int iy = 0; iy < gridY; ++iy)
        {
            const int y0 = iy * boxSizeY;
            const int y1 = y0 + boxSizeY;
            for (int ix = 0; ix < gridX; ++ix)
            {
                const int x0 = ix * boxSizeX;
                const int x1 = x0 + boxSizeX;

                double sum = 0.0;
                int voxels = 0;
                for (int z = z0; z < z1; ++z)
                {
                    for (int y = y0; y < y1; ++y)
                    {
                        const float *row = stack[static_cast<size_t>(z)].ptr<float>(y);
                        for (int x = x0; x < x1; ++x)
                        {
                            sum += row[x];
                            ++voxels;
                        }
                    }
                }
                if (voxels <= 0)
                {
                    continue;
                }

                const float meanBrightness = static_cast<float>(sum / static_cast<double>(voxels));
                if (meanBrightness <= backgroundValue + minDelta)
                {
                    continue;
                }

                boxes.push_back({
                    ix, iy, iz,
                    cv::Point3f(
                        static_cast<float>(x0 + x1 - 1) * 0.5f,
                        static_cast<float>(y0 + y1 - 1) * 0.5f,
                        static_cast<float>(z0 + z1 - 1) * 0.5f),
                    meanBrightness,
                    voxels
                });
            }
        }
    }

    std::sort(boxes.begin(), boxes.end(),
              [](const BrightBox &a, const BrightBox &b)
              {
                  return a.brightness > b.brightness;
              });

    std::map<std::tuple<int, int, int>, size_t> boxIndex;
    for (size_t i = 0; i < boxes.size(); ++i)
    {
        boxIndex[{boxes[i].ix, boxes[i].iy, boxes[i].iz}] = i;
    }

    std::vector<char> visited(boxes.size(), 0);
    float maxCenterBrightness = backgroundValue;
    for (size_t seed = 0; seed < boxes.size(); ++seed)
    {
        if (visited[seed])
        {
            continue;
        }

        std::queue<size_t> queue;
        queue.push(seed);
        visited[seed] = 1;
        double weightSum = 0.0;
        double xSum = 0.0;
        double ySum = 0.0;
        double zSum = 0.0;
        double brightnessSum = 0.0;
        int clusterBoxes = 0;

        while (!queue.empty())
        {
            const size_t current = queue.front();
            queue.pop();
            const BrightBox &box = boxes[current];
            const float weight = std::max(1e-6f, box.brightness - backgroundValue);
            weightSum += weight;
            xSum += weight * static_cast<double>(box.center.x);
            ySum += weight * static_cast<double>(box.center.y);
            zSum += weight * static_cast<double>(box.center.z);
            brightnessSum += static_cast<double>(box.brightness);
            ++clusterBoxes;

            static constexpr std::array<std::array<int, 3>, 6> kFaceNeighbors{{
                {{ 1,  0,  0}}, {{-1,  0,  0}},
                {{ 0,  1,  0}}, {{ 0, -1,  0}},
                {{ 0,  0,  1}}, {{ 0,  0, -1}}
            }};
            for (const auto &offset : kFaceNeighbors)
            {
                auto it = boxIndex.find({
                    box.ix + offset[0],
                    box.iy + offset[1],
                    box.iz + offset[2]});
                if (it == boxIndex.end())
                {
                    continue;
                }
                const size_t neighbor = it->second;
                if (visited[neighbor])
                {
                    continue;
                }
                visited[neighbor] = 1;
                queue.push(neighbor);
            }
        }

        if (clusterBoxes <= 0 || weightSum <= 0.0)
        {
            continue;
        }

        Frame::SignalCenter center;
        center.position = cv::Point3f(
            static_cast<float>(xSum / weightSum),
            static_cast<float>(ySum / weightSum),
            static_cast<float>(zSum / weightSum));
        center.brightness = static_cast<float>(brightnessSum / static_cast<double>(clusterBoxes));
        center.boxes = clusterBoxes;
        centers.push_back(center);
        maxCenterBrightness = std::max(maxCenterBrightness, center.brightness);
    }

    for (auto &center : centers)
    {
        const float normalized = (maxCenterBrightness > backgroundValue + 1e-6f)
            ? std::clamp((center.brightness - backgroundValue) /
                         (maxCenterBrightness - backgroundValue), 0.0f, 1.0f)
            : 0.0f;
        center.sigmaScale = std::max(
            config.simulation.signal_guided_min_sigma_scale,
            1.0f - normalized * (1.0f - config.simulation.signal_guided_min_sigma_scale));
    }

    std::sort(centers.begin(), centers.end(),
              [](const Frame::SignalCenter &a, const Frame::SignalCenter &b)
              {
                  return a.brightness > b.brightness;
              });

    return centers;
}

void clipStack(ImageStack &sequence)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(sequence.size()); ++i)
    {
        auto &slice = sequence[static_cast<std::size_t>(i)];
        for (int y = 0; y < slice.rows; ++y)
        {
            float *row = slice.ptr<float>(y);
            for (int x = 0; x < slice.cols; ++x)
            {
                if (!std::isfinite(row[x]) || row[x] < 0.0f)
                {
                    row[x] = 0.0f;
                }
                else if (row[x] > 1.0f)
                {
                    row[x] = 1.0f;
                }
            }
        }
    }
}

ImageStack applySimpleBlurBlendPreprocess(const ImageStack &sequence,
                                          const BaseConfig &config,
                                          std::ostream &log,
                                          const std::string &imageFile)
{
    ImageStack processed = cloneStack(sequence);
    if (processed.empty())
    {
        return processed;
    }

    const float sigma = std::max(0.0f, config.simulation.simple_preprocess_blur_sigma);
    const float realRatio = config.simulation.simple_preprocess_real_ratio;
    const float backgroundFactor = config.simulation.simple_preprocess_background_factor;

    log << "[SimplePreprocess] file=" << fs::path(imageFile).filename().string()
        << " blur_sigma=" << sigma
        << " real_ratio=" << realRatio
        << " background_factor=" << backgroundFactor
        << std::endl;

    double rawSum = 0.0;
    double blurredSum = 0.0;
    double blendSum = 0.0;
    double maxBlend = -std::numeric_limits<double>::infinity();
    double outputMinusBlurAbsSum = 0.0;
    double outputMinusRawAbsSum = 0.0;
    std::size_t voxelCount = 0;

    #pragma omp parallel for schedule(static) reduction(+:rawSum,blurredSum,blendSum,outputMinusBlurAbsSum,outputMinusRawAbsSum,voxelCount) reduction(max:maxBlend)
    for (int sliceIndex = 0; sliceIndex < static_cast<int>(processed.size()); ++sliceIndex)
    {
        const cv::Mat rawSlice = sequence[static_cast<std::size_t>(sliceIndex)];
        cv::Mat blurredSlice;
        if (sigma > 0.0f)
        {
            cv::GaussianBlur(rawSlice,
                             blurredSlice,
                             cv::Size(0, 0),
                             sigma,
                             sigma);
        }
        else
        {
            blurredSlice = rawSlice;
        }

        cv::Mat &outSlice = processed[static_cast<std::size_t>(sliceIndex)];
        for (int y = 0; y < rawSlice.rows; ++y)
        {
            const float *rawRow = rawSlice.ptr<float>(y);
            const float *blurredRow = blurredSlice.ptr<float>(y);
            float *outRow = outSlice.ptr<float>(y);
            for (int x = 0; x < rawSlice.cols; ++x)
            {
                const float rawValue = rawRow[x];
                const float blend = realRatio * rawValue * backgroundFactor;
                const float outputValue = rawValue * blend + blurredRow[x] * (1.0f - blend);
                outRow[x] = outputValue;

                rawSum += rawValue;
                blurredSum += blurredRow[x];
                blendSum += blend;
                maxBlend = std::max(maxBlend, static_cast<double>(blend));
                outputMinusBlurAbsSum += std::abs(outputValue - blurredRow[x]);
                outputMinusRawAbsSum += std::abs(outputValue - rawValue);
                ++voxelCount;
            }
        }
    }

    clipStack(processed);
    if (voxelCount > 0)
    {
        const double invCount = 1.0 / static_cast<double>(voxelCount);
        log << "[SimplePreprocess Stats] file=" << fs::path(imageFile).filename().string()
            << " raw_mean=" << rawSum * invCount
            << " blurred_mean=" << blurredSum * invCount
            << " blend_mean=" << blendSum * invCount
            << " blend_max=" << maxBlend
            << " mean_abs_output_minus_blur=" << outputMinusBlurAbsSum * invCount
            << " mean_abs_output_minus_raw=" << outputMinusRawAbsSum * invCount
            << std::endl;
    }
    return processed;
}
} // namespace

Image ImageHandler::processImage(const Image &image, const BaseConfig &config)
{
    Image processedImage;

    if (image.channels() == 3)
    {
        cv::cvtColor(image, processedImage, cv::COLOR_RGB2GRAY);
    }
    else
    {
        processedImage = image.clone();
    }

    // Preserve the source intensity scale here. Frames are normalized later
    // using one shared percentile-based scale across the selected run.
    processedImage.convertTo(processedImage, CV_32F);

    if (config.simulation.blur_sigma > 0.0f)
    {
        cv::GaussianBlur(processedImage,
                         processedImage,
                         cv::Size(0, 0),
                         config.simulation.blur_sigma,
                         config.simulation.blur_sigma);
    }

    return processedImage;
}

float ImageHandler::computePercentileFromValues(std::vector<float> values, float percentileFraction)
{
    if (values.empty())
    {
        return 0.0f;
    }

    const float clampedPercentile = std::clamp(percentileFraction, 0.0f, 1.0f);
    const std::size_t index = static_cast<std::size_t>(
        std::floor(clampedPercentile * static_cast<float>(values.size() - 1)));

    std::nth_element(values.begin(),
                     values.begin() + static_cast<std::ptrdiff_t>(index),
                     values.end());
    return values[index];
}

float computeMeanOfTopFraction(std::vector<float> values, float topFraction)
{
    if (values.empty())
    {
        return 0.0f;
    }

    const float clampedFraction = std::clamp(topFraction, 0.0f, 1.0f);
    if (clampedFraction <= 0.0f)
    {
        return 0.0f;
    }

    const std::size_t selectedCount = std::max<std::size_t>(
        1,
        static_cast<std::size_t>(
            std::ceil(clampedFraction * static_cast<float>(values.size()))));
    const std::size_t thresholdIndex = values.size() - selectedCount;

    std::nth_element(values.begin(),
                     values.begin() + static_cast<std::ptrdiff_t>(thresholdIndex),
                     values.end());

    double sum = 0.0;
    for (std::size_t i = thresholdIndex; i < values.size(); ++i)
    {
        sum += values[i];
    }

    return static_cast<float>(sum / static_cast<double>(selectedCount));
}

float ImageHandler::computePercentileFromSlice(const cv::Mat &slice, float percentileFraction)
{
    CV_Assert(slice.type() == CV_32F);

    std::vector<float> values;
    values.reserve(static_cast<std::size_t>(slice.total()));
    for (int y = 0; y < slice.rows; ++y)
    {
        const float *row = slice.ptr<float>(y);
        values.insert(values.end(), row, row + slice.cols);
    }
    return computePercentileFromValues(std::move(values), percentileFraction);
}

cv::Mat ImageHandler::boxMean(const cv::Mat &image, int windowSize)
{
    if (windowSize < 1 || windowSize % 2 == 0)
    {
        throw std::invalid_argument("windowSize must be a positive odd integer");
    }

    cv::Mat output;
    cv::blur(image, output, cv::Size(windowSize, windowSize), cv::Point(-1, -1), cv::BORDER_REPLICATE);
    return output;
}

float ImageHandler::evaluateSequenceContrastScore(const ImageStack &sequence, const BaseConfig &config)
{
    const float aRadius =
        (config.cell ? static_cast<float>(config.cell->maxARadius) : 40.0f);
    const float cRadius =
        (config.cell ? static_cast<float>(config.cell->maxCRadius) : 35.0f);
    const float maxRadius = std::max(1.0f, std::max(aRadius, cRadius));
    return evaluateSequenceContrastScoreForRadius(sequence, config, maxRadius);
}

float ImageHandler::evaluateBestWindowContrastScore(const ImageStack &sequence, const BaseConfig &config)
{
    const float defaultMinRadius = 5.0f;
    const float defaultMaxRadius = 65.0f;
    const float minRadius = config.cell
        ? std::max(1.0f, static_cast<float>(std::min({
              config.cell->minARadius,
              config.cell->minBRadius,
              config.cell->minCRadius})))
        : defaultMinRadius;
    const float maxRadius = config.cell
        ? std::max(minRadius, static_cast<float>(std::max({
              config.cell->maxARadius,
              config.cell->maxBRadius > 0.0 ? config.cell->maxBRadius : config.cell->maxARadius,
              config.cell->maxCRadius})))
        : defaultMaxRadius;
    const float radiusStep = std::max(1.0f, config.simulation.contrast_window_radius_step);
    const float radiusStart = 0.5f * (minRadius + maxRadius);
    const float penaltyRadius = std::clamp(
        minRadius * std::max(1.0f, config.simulation.contrast_penalty_min_radius_scale),
        minRadius,
        maxRadius);
    const float rewardWeight = config.simulation.contrast_reward_weight;
    const float penaltyWeight = config.simulation.contrast_penalty_weight;

    std::vector<float> scoringRadii;
    for (float radius = radiusStart; radius < maxRadius - 1e-6f; radius += radiusStep) {
        scoringRadii.push_back(radius);
    }
    if (scoringRadii.empty() || std::abs(scoringRadii.back() - maxRadius) > 1e-6f) {
        scoringRadii.push_back(maxRadius);
    }

    const float penaltyScore = evaluateSequenceContrastScoreForRadius(sequence, config, penaltyRadius);
    std::vector<float> scores(scoringRadii.size(), 0.0f);
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < static_cast<int>(scoringRadii.size()); ++i) {
        const float rawScore = evaluateSequenceContrastScoreForRadius(
            sequence, config, scoringRadii[static_cast<std::size_t>(i)]);
        scores[static_cast<std::size_t>(i)] = rewardWeight * rawScore - penaltyWeight * penaltyScore;
    }

    return *std::max_element(scores.begin(), scores.end());
}

float ImageHandler::evaluateSequenceContrastScoreForRadius(const ImageStack &sequence,
                                                           const BaseConfig &config,
                                                           float radiusAtScale)
{
    const int innerWindow = makeOddAtLeast(
        static_cast<int>(std::lround(radiusAtScale * 2.0f + 1.0f)));
    const int outerWindow = makeOddAtLeast(
        static_cast<int>(std::lround(radiusAtScale * 4.0f + 1.0f)),
        innerWindow + 2);

    std::vector<float> sliceScores(sequence.size(), 0.0f);

    #pragma omp parallel for schedule(dynamic)
    for (int sliceIndex = 0; sliceIndex < static_cast<int>(sequence.size()); ++sliceIndex)
    {
        const auto &slice = sequence[static_cast<std::size_t>(sliceIndex)];
        CV_Assert(slice.type() == CV_32F);

        const cv::Mat innerMean = boxMean(slice, innerWindow);
        const cv::Mat outerMean = boxMean(slice, outerWindow);

        std::vector<float> contrastValues;
        std::vector<float> brightInteriorValues;
        contrastValues.reserve(static_cast<std::size_t>(slice.total()));
        brightInteriorValues.reserve(static_cast<std::size_t>(slice.total() / 8));

        for (int y = 0; y < slice.rows; ++y)
        {
            const float *sliceRow = slice.ptr<float>(y);
            const float *innerRow = innerMean.ptr<float>(y);
            const float *outerRow = outerMean.ptr<float>(y);
            for (int x = 0; x < slice.cols; ++x)
            {
                const float localDifference = innerRow[x] - outerRow[x];
                if (localDifference < config.simulation.contrast_structure_threshold)
                {
                    continue;
                }

                const float stableBackground =
                    std::max(outerRow[x], config.simulation.contrast_background_floor);
                const float localContrast =
                    localDifference / (stableBackground + config.simulation.contrast_eps);
                contrastValues.push_back(localContrast);

                // Reward bright filled interiors in regions that already
                // exhibit positive local contrast.
                brightInteriorValues.push_back(sliceRow[x]);
            }
        }

        if (contrastValues.empty())
        {
            sliceScores[static_cast<std::size_t>(sliceIndex)] = 0.0f;
            continue;
        }

        const float contrastScore = computePercentileFromValues(
            std::move(contrastValues),
            config.simulation.iterative_score_percentile);
        const float brightInteriorScore = computeMeanOfTopFraction(
            std::move(brightInteriorValues),
            config.simulation.contrast_bright_fraction);

        sliceScores[static_cast<std::size_t>(sliceIndex)] =
            contrastScore +
            config.simulation.contrast_brightness_weight * brightInteriorScore;
    }

    if (sliceScores.empty())
    {
        return 0.0f;
    }

    return computePercentileFromValues(std::move(sliceScores), 0.5f);
}

ImageStack ImageHandler::processPreparedSequence(const ImageStack &sequence,
                                                const BaseConfig &config,
                                                std::ostream &log,
                                                const std::string &imageFile)
{
    return applySimpleBlurBlendPreprocess(sequence, config, log, imageFile);
}

std::vector<cv::Mat> ImageHandler::loadRawFrame(const std::string &imageFile,
                                                BaseConfig &config,
                                                std::ostream *logSink)
{
    std::ostream &log = logSink ? *logSink : std::cout;
    std::vector<cv::Mat> normalizedSlices;

    const std::string extension = imageFile.substr(imageFile.find_last_of('.') + 1);
    if (extension == "tiff" || extension == "tif")
    {
        std::vector<cv::Mat> tiffImage;
        cv::imreadmulti(imageFile, tiffImage, cv::IMREAD_ANYDEPTH | cv::IMREAD_COLOR);

        const auto numTiffSlices = tiffImage.size();
        if (numTiffSlices == 0)
        {
            throw std::runtime_error("TIFF has 0 slices: " + imageFile);
        }

        const cv::Mat &firstSlice = tiffImage.front();
        if (firstSlice.empty())
        {
            std::cout << "Error: Could not read the TIFF image" << '\n';
            return normalizedSlices;
        }

        log << "[LoadFrame] file=" << fs::path(imageFile).filename().string()
            << " rawSlices=" << numTiffSlices
            << " rawType=" << firstSlice.type()
            << " rawDepth=" << firstSlice.depth()
            << " rawChannels=" << firstSlice.channels()
            << " rawRows=" << firstSlice.rows
            << " rawCols=" << firstSlice.cols
            << std::endl;

        if (config.simulation.frame_intensity_auto_hard_max &&
            config.simulation.frame_intensity_fixed_hard_max_scale)
        {
            if (firstSlice.depth() == CV_8U)
            {
                config.simulation.frame_intensity_hard_max = 255.0f;
            }
            else if (firstSlice.depth() == CV_16U)
            {
                config.simulation.frame_intensity_hard_max = 65536.0f;
            }
            log << "[Frame Intensity Auto Hard Max] file="
                << fs::path(imageFile).filename().string()
                << " raw_depth=" << firstSlice.depth()
                << " selected_hard_max="
                << config.simulation.frame_intensity_hard_max
                << std::endl;
        }

        normalizedSlices.resize(numTiffSlices);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < static_cast<int>(numTiffSlices); ++i)
        {
            cv::Mat slice = tiffImage[static_cast<std::size_t>(i)];
            if (slice.channels() == 3)
            {
                cv::cvtColor(slice, slice, cv::COLOR_BGR2GRAY);
            }
            normalizedSlices[static_cast<std::size_t>(i)] = processImage(slice, config);
        }
    }
    else
    {
        cv::Mat image = cv::imread(imageFile);
        if (image.empty())
        {
            std::cout << "Error: Could not read the image" << '\n';
            return normalizedSlices;
        }

        if (image.channels() == 3)
        {
            cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
        }

        if (config.simulation.frame_intensity_auto_hard_max &&
            config.simulation.frame_intensity_fixed_hard_max_scale)
        {
            config.simulation.frame_intensity_hard_max =
                image.depth() == CV_16U ? 65536.0f : 255.0f;
            log << "[Frame Intensity Auto Hard Max] file="
                << fs::path(imageFile).filename().string()
                << " raw_depth=" << image.depth()
                << " selected_hard_max="
                << config.simulation.frame_intensity_hard_max
                << std::endl;
        }

        normalizedSlices.push_back(processImage(image, config));
    }

    printStackStats(log, "normalized_input", imageFile, normalizedSlices);
    return normalizedSlices;
}

std::vector<cv::Mat> ImageHandler::preprocessLoadedFrame(const std::vector<cv::Mat> &normalizedSlices,
                                                         const std::string &imageFile,
                                                         const BaseConfig &config,
                                                         std::ostream *logSink)
{
    std::ostream &log = logSink ? *logSink : std::cout;
    std::vector<cv::Mat> processedZSlices = cloneStack(normalizedSlices);
    std::vector<cv::Mat> interpolatedZSlices;

    if (config.simulation.simple_preprocess_enabled)
    {
        processedZSlices = processPreparedSequence(processedZSlices, config, log, imageFile);
    }
    else
    {
        log << "[SimplePreprocess] file=" << fs::path(imageFile).filename().string()
            << " enabled=0; weighted_blend_skipped=1"
            << std::endl;
    }

    const float localScore = evaluateBestWindowContrastScore(processedZSlices, config);
    log << "[PreprocessScores] file=" << fs::path(imageFile).filename().string()
        << " local=" << localScore
        << std::endl;

    printStackStats(log, "processed_sequence", imageFile, processedZSlices);

    if (processedZSlices.empty())
    {
        return interpolatedZSlices;
    }

    if (!config.simulation.simple_preprocess_interpolate_z || processedZSlices.size() == 1)
    {
        interpolatedZSlices = processedZSlices;
    }
    else
    {
        const int expandFactor = config.simulation.z_scaling;
        const unsigned numSynthSlices =
            static_cast<unsigned>(expandFactor) * (processedZSlices.size() - 1U) + 1U;

        interpolatedZSlices.resize(numSynthSlices);
        #pragma omp parallel for schedule(static)
        for (int synthSliceIndex = 0; synthSliceIndex < static_cast<int>(numSynthSlices); ++synthSliceIndex)
        {
            const unsigned synthSlice = static_cast<unsigned>(synthSliceIndex);
            const int sourceSlice = static_cast<int>(synthSlice / expandFactor);
            if (synthSlice % expandFactor == 0)
            {
                interpolatedZSlices[static_cast<std::size_t>(synthSlice)] =
                    processedZSlices[static_cast<std::size_t>(sourceSlice)];
            }
            else
            {
                const double t = static_cast<double>(synthSlice % expandFactor) /
                                 static_cast<double>(expandFactor);
                interpolatedZSlices[static_cast<std::size_t>(synthSlice)] =
                    (1.0 - t) * processedZSlices[static_cast<std::size_t>(sourceSlice)] +
                    t * processedZSlices[static_cast<std::size_t>(sourceSlice + 1)];
            }
        }

        if (interpolatedZSlices.size() != numSynthSlices)
        {
            throw std::runtime_error(
                "interpolatedZSlices must have exactly " + std::to_string(numSynthSlices) +
                " slices, but has " + std::to_string(interpolatedZSlices.size()) + " slices");
        }
    }

    printStackStats(log, "post_interpolation", imageFile, interpolatedZSlices);
    ImageStack preCubePoolingSlices = cloneStack(interpolatedZSlices);
    interpolatedZSlices = applyCubePooling(interpolatedZSlices, config, log);
    clipStack(interpolatedZSlices);
    const StackStats cubeStats = computeStackStats(interpolatedZSlices);
    const float candidateMaxMean = std::max(0.0f, config.simulation.iterative_candidate_max_mean);
    const float candidateMaxSaturatedFraction = std::clamp(
        config.simulation.iterative_candidate_max_saturated_fraction, 0.0f, 1.0f);
    const bool cubeMeanOk =
        candidateMaxMean <= 0.0f || cubeStats.mean <= candidateMaxMean;
    const bool cubeSaturationOk =
        candidateMaxSaturatedFraction <= 0.0f ||
        cubeStats.saturatedFraction <= candidateMaxSaturatedFraction;
    if (!cubeMeanOk || !cubeSaturationOk || cubeStats.nonFiniteCount > 0)
    {
        log << "[CubePooling] reject=range"
            << " mean=" << cubeStats.mean
            << " saturated_fraction=" << cubeStats.saturatedFraction
            << " nonfinite=" << cubeStats.nonFiniteCount
            << " max_mean=" << candidateMaxMean
            << " max_saturated_fraction=" << candidateMaxSaturatedFraction
            << "; using_pre_cube_pooling_sequence=1"
            << std::endl;
        interpolatedZSlices = std::move(preCubePoolingSlices);
    }
    printStackStats(log, "post_cube_pooling", imageFile, interpolatedZSlices);
    log << std::to_string(interpolatedZSlices.size()) << "slices built successfully" << std::endl;
    return interpolatedZSlices;
}

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

    // Step 1a: compute per-frame brightness ratio for anchor adjustment.
    // ratio = current_frame_mean / frame_0_mean. Falls back to 1.0 (no
    // adjustment) when frame_0_mean is zero (sentinel for "not populated"
    // in older calibrations or manual-override paths that didn't compute it).
    float ratio = 1.0f;
    const bool simplePreprocess = config.simulation.simple_preprocess_enabled;
    if (calibration.frame_0_mean > 1e-6f) {
        double currentSum = 0.0;
        std::size_t currentPixels = 0;
        for (const auto &slice : normalizedSlices) {
            for (int y = 0; y < slice.rows; ++y) {
                const float *row = slice.ptr<float>(y);
                for (int x = 0; x < slice.cols; ++x) {
                    currentSum += row[x];
                    ++currentPixels;
                }
            }
        }
        const float currentMean = currentPixels > 0
            ? static_cast<float>(currentSum / currentPixels)
            : calibration.frame_0_mean;
        ratio = currentMean / calibration.frame_0_mean;
        // Clamp to a sane range — extreme values usually mean the input
        // frame is corrupted or otherwise abnormal. 0.3..3.0 covers
        // ±3× brightness shift which is well past any normal photobleaching.
        ratio = std::clamp(ratio, 0.3f, 3.0f);
    }

    const float bg_n = calibration.background_intensity * ratio;
    const float cell_n = calibration.cell_intensity * ratio;
    const float span_n = std::max(1e-3f, cell_n - bg_n);

    log << "[Calibrated Preprocess Ratio] frame_0_mean=" << calibration.frame_0_mean
        << " ratio=" << ratio
        << " bg_0=" << calibration.background_intensity
        << " cell_0=" << calibration.cell_intensity
        << " bg_n=" << bg_n
        << " cell_n=" << cell_n
        << "\n";

    // Step 1b: linear normalize via bg/cell anchors → [0, 1].
    // bg → 0, cell → 1. Mandatory for the L2 cost (synth cells render with
    // `_brightness` clamped to [0.1, 0.98]) and for the bio gates that
    // expect cell-edge brightness in absolute terms (e.g., 0.07 floor).
    //
    // Clip to [0, 1] after calibration. If simple preprocessing is also
    // enabled, the weighted raw/blurred blend below expects normalized input,
    // just like the fixed-hard-max branch pipeline.
    std::vector<cv::Mat> linearScaled;
    linearScaled.reserve(normalizedSlices.size());
    for (const auto &slice : normalizedSlices) {
        cv::Mat out;
        slice.convertTo(out, CV_32F, 1.0f / span_n, -bg_n / span_n);
        cv::min(out, 1.0f, out);
        cv::max(out, 0.0f, out);
        linearScaled.push_back(out);
    }

    // Step 2: optional filter. With simple preprocessing enabled, compose the
    // calibrated intensity remap with the branch's weighted raw/blurred blend.
    // Otherwise use the calibrated Gaussian blur path.
    if (simplePreprocess) {
        linearScaled = applySimpleBlurBlendPreprocess(
            linearScaled, config, log, "calibrated");
    } else if (config.simulation.calibrated_preprocess_blur_sigma > 0.0f) {
        const float blurSigma = config.simulation.calibrated_preprocess_blur_sigma;
        const int ksize = std::max(3, 2 * static_cast<int>(std::ceil(3.0f * blurSigma)) + 1);
        for (auto &slice : linearScaled) {
            cv::GaussianBlur(slice, slice, cv::Size(ksize, ksize),
                             blurSigma, blurSigma,
                             cv::BORDER_REPLICATE);
        }
    }

    // Step 3: z-interpolate unless the simple branch explicitly requests the
    // Python-reference behavior of preserving original z pages.
    const int expandFactor = (simplePreprocess && !config.simulation.simple_preprocess_interpolate_z)
        ? 1
        : static_cast<int>(config.simulation.z_scaling);
    if (linearScaled.size() <= 1 || expandFactor <= 1) {
        log << "[Calibrated Preprocess] " << calibrationDescribe(calibration)
            << " input_slices=" << normalizedSlices.size()
            << " output_slices=" << linearScaled.size()
            << " (no z-interp)\n";
        return linearScaled;
    }

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
            << " path="
            << (config.simulation.simple_preprocess_enabled
                    ? "calibrated+simple_blend" : "calibrated")
            << "\n";
        return applyCalibratedPreprocess(normalizedSlices, *calibration, config, logSink);
    }

    log << "[Preprocess Dispatch] file=" << fs::path(imageFile).filename().string()
        << " path="
        << (config.simulation.simple_preprocess_enabled ? "simple_blend" : "unweighted_fallback")
        << " auto_requested=" << config.simulation.auto_calibrate_brightness_enabled
        << " calibration_valid=" << (calibration != nullptr && calibration->isValid())
        << "\n";
    return preprocessLoadedFrame(normalizedSlices, imageFile, config, logSink);
}

std::vector<cv::Mat> ImageHandler::loadFrame(const std::string &imageFile,
                                             BaseConfig &config,
                                             std::ostream *logSink)
{
    const std::vector<cv::Mat> normalizedSlices = loadRawFrame(imageFile, config, logSink);
    return preprocessLoadedFrame(normalizedSlices, imageFile, config, logSink);
}

PathVec ImageHandler::getImageFilePaths(const std::string &input, int firstFrame, int lastFrame, BaseConfig &config)
{
    PathVec imagePaths;

    if (input.find('%') != std::string::npos)
    {
        for (int i = firstFrame; lastFrame == -1 || i <= lastFrame; ++i)
        {
            char buffer[1024];
            std::snprintf(buffer, sizeof(buffer), input.c_str(), i);
            fs::path file(buffer);

            if (fs::exists(file) && fs::is_regular_file(file))
            {
                imagePaths.push_back(file);
                continue;
            }

            std::cerr << "Input file not found \"" << file << "\"" << '\n';
            throw std::runtime_error("Input file not found");
        }
    }
    else if (fs::is_directory(input))
    {
        PathVec allFiles;
        for (const auto &entry : fs::directory_iterator(input))
        {
            if (!entry.is_regular_file())
            {
                continue;
            }

            const fs::path &path = entry.path();
            if (shouldIgnoreImagePath(path))
            {
                continue;
            }

            if (path.extension() == ".tif" || path.extension() == ".tiff")
            {
                allFiles.push_back(path);
            }
        }

        if (allFiles.empty())
        {
            throw std::runtime_error("No .tif/.tiff files found in directory: " + input);
        }

        std::sort(allFiles.begin(), allFiles.end());

        if (firstFrame < 0)
        {
            throw std::runtime_error("firstFrame must be >= 0 for directory input");
        }

        const int start = firstFrame;
        const int end = (lastFrame < 0) ? static_cast<int>(allFiles.size()) - 1
                                        : std::min(lastFrame, static_cast<int>(allFiles.size()) - 1);

        if (start >= static_cast<int>(allFiles.size()))
        {
            throw std::runtime_error("firstFrame is out of range for directory input");
        }
        if (start > end)
        {
            throw std::runtime_error("Invalid frame range for directory input");
        }

        for (int i = start; i <= end; ++i)
        {
            imagePaths.push_back(allFiles[i]);
        }
    }
    else if (fs::exists(input) && fs::is_regular_file(input))
    {
        imagePaths.push_back(input);
    }
    else
    {
        throw std::runtime_error("Input is neither a pattern, directory, nor file: " + input);
    }

    if (!imagePaths.empty())
    {
        updateTiffConfigIfNeeded(imagePaths.front(), config);
    }

    for (const auto &path : imagePaths)
    {
        std::cout << path << '\n';
    }

    return imagePaths;
}
