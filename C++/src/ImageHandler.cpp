#include "../includes/ImageHandler.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
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
    std::size_t count = 0;
};

StackStats computeStackStats(const ImageStack &stack)
{
    StackStats stats;
    double sum = 0.0;
    double sumSq = 0.0;
    bool firstValue = true;

    for (const auto &slice : stack)
    {
        CV_Assert(slice.type() == CV_32F);

        double sliceMin = 0.0;
        double sliceMax = 0.0;
        cv::minMaxLoc(slice, &sliceMin, &sliceMax);
        if (firstValue)
        {
            stats.minValue = sliceMin;
            stats.maxValue = sliceMax;
            firstValue = false;
        }
        else
        {
            stats.minValue = std::min(stats.minValue, sliceMin);
            stats.maxValue = std::max(stats.maxValue, sliceMax);
        }

        for (int y = 0; y < slice.rows; ++y)
        {
            const float *row = slice.ptr<float>(y);
            for (int x = 0; x < slice.cols; ++x)
            {
                const double value = row[x];
                sum += value;
                sumSq += value * value;
                ++stats.count;
            }
        }
    }

    if (stats.count == 0)
    {
        return stats;
    }

    stats.mean = sum / static_cast<double>(stats.count);
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
        << std::endl;
}

ImageStack cloneStack(const ImageStack &sequence)
{
    ImageStack cloned;
    cloned.reserve(sequence.size());
    for (const auto &slice : sequence)
    {
        cloned.push_back(slice.clone());
    }
    return cloned;
}

int computeAdaptiveCubeSize(const BaseConfig &config)
{
    if (!config.cell)
    {
        return 1;
    }

    const float minRadius = std::max(
        1.0f,
        static_cast<float>(std::min({config.cell->minARadius,
                                     config.cell->minBRadius,
                                     config.cell->minCRadius})));
    const float cubeSizeTarget =
        std::max(1.0f, config.simulation.adaptive_cube_pooling_cube_size_scale * minRadius);
    return std::max(1, static_cast<int>(std::lround(cubeSizeTarget)));
}

ImageStack applyAdaptiveCubePooling(const ImageStack &stack,
                                    const BaseConfig &config,
                                    std::ostream &log)
{
    if (stack.empty() || !config.cell || !config.simulation.adaptive_cube_pooling_enabled)
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

    const int cubeSize = computeAdaptiveCubeSize(config);
    const float zeroThreshold = std::max(0.0f, config.simulation.adaptive_cube_pooling_zero_threshold);
    const float majorityThreshold =
        std::clamp(config.simulation.adaptive_cube_pooling_majority_threshold, 0.0f, 1.0f);
    const float strongPeakThreshold =
        std::max(0.0f, config.simulation.adaptive_cube_pooling_strong_peak_threshold);

    const int gridZ = (depth + cubeSize - 1) / cubeSize;
    const int gridY = (rows + cubeSize - 1) / cubeSize;
    const int gridX = (cols + cubeSize - 1) / cubeSize;

    struct CubeStats
    {
        float mean = 0.0f;
        float maxValue = 0.0f;
        float zeroFraction = 0.0f;
    };

    std::vector<CubeStats> cubeStats(static_cast<size_t>(gridZ) * gridY * gridX);
    auto cubeIndex = [gridX, gridY](int gz, int gy, int gx) -> size_t {
        return static_cast<size_t>((gz * gridY + gy) * gridX + gx);
    };

    // Parallelize cube-stats computation. Each cube writes only to its
    // own cubeStats[idx] → no races. collapse(2) gives threads gz/gy tiles
    // big enough to amortize scheduling overhead.
    #pragma omp parallel for schedule(static)
    for (int gz = 0; gz < gridZ; ++gz)
    {
        const int z0 = gz * cubeSize;
        const int z1 = std::min(depth, z0 + cubeSize);
        for (int gy = 0; gy < gridY; ++gy)
        {
            const int y0 = gy * cubeSize;
            const int y1 = std::min(rows, y0 + cubeSize);
            for (int gx = 0; gx < gridX; ++gx)
            {
                const int x0 = gx * cubeSize;
                const int x1 = std::min(cols, x0 + cubeSize);

                double sum = 0.0;
                float maxValue = 0.0f;
                int zeroCount = 0;
                int voxelCount = 0;
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
                            maxValue = std::max(maxValue, value);
                            zeroCount += (value <= zeroThreshold) ? 1 : 0;
                            ++voxelCount;
                        }
                    }
                }

                CubeStats &stats = cubeStats[cubeIndex(gz, gy, gx)];
                if (voxelCount > 0)
                {
                    stats.mean = static_cast<float>(sum / static_cast<double>(voxelCount));
                    stats.maxValue = maxValue;
                    stats.zeroFraction = static_cast<float>(zeroCount) /
                                         static_cast<float>(voxelCount);
                }
            }
        }
    }

    ImageStack pooled(depth);
    for (int z = 0; z < depth; ++z)
    {
        pooled[static_cast<size_t>(z)] = cv::Mat::zeros(rows, cols, CV_32F);
    }
    std::vector<float> pooledCubeValues(static_cast<size_t>(gridZ) * gridY * gridX, 0.0f);

    int meanPooledCubes = 0;
    int maxPooledCubes = 0;
    int strongPeakOverrideCubes = 0;
    // Parallelize cube-reweighting + voxel fill. Each iteration reads from
    // cubeStats (read-only) and writes to its own cube voxel range in pooled
    // (disjoint per-tuple) and pooledCubeValues[cubeIndex(gz,gy,gx)] (disjoint).
    // Counters updated via reduction.
    #pragma omp parallel for schedule(static) reduction(+:meanPooledCubes, maxPooledCubes, strongPeakOverrideCubes)
    for (int gz = 0; gz < gridZ; ++gz)
    {
        const int z0 = gz * cubeSize;
        const int z1 = std::min(depth, z0 + cubeSize);
        for (int gy = 0; gy < gridY; ++gy)
        {
            const int y0 = gy * cubeSize;
            const int y1 = std::min(rows, y0 + cubeSize);
            for (int gx = 0; gx < gridX; ++gx)
            {
                const int x0 = gx * cubeSize;
                const int x1 = std::min(cols, x0 + cubeSize);
                const CubeStats &stats = cubeStats[cubeIndex(gz, gy, gx)];

                float neighborMean = 0.0f;
                int neighborCount = 0;
                for (int nz = std::max(0, gz - 1); nz <= std::min(gridZ - 1, gz + 1); ++nz)
                {
                    for (int ny = std::max(0, gy - 1); ny <= std::min(gridY - 1, gy + 1); ++ny)
                    {
                        for (int nx = std::max(0, gx - 1); nx <= std::min(gridX - 1, gx + 1); ++nx)
                        {
                            if (nz == gz && ny == gy && nx == gx)
                            {
                                continue;
                            }
                            neighborMean += cubeStats[cubeIndex(nz, ny, nx)].mean;
                            ++neighborCount;
                        }
                    }
                }
                if (neighborCount > 0)
                {
                    neighborMean /= static_cast<float>(neighborCount);
                }

                const bool hasStrongPeak = stats.maxValue >= strongPeakThreshold;
                const bool useMeanPooling =
                    stats.zeroFraction >= majorityThreshold &&
                    neighborMean <= zeroThreshold &&
                    !hasStrongPeak;
                const float pooledValue = useMeanPooling ? stats.mean : stats.maxValue;
                pooledCubeValues[cubeIndex(gz, gy, gx)] = pooledValue;
                if (useMeanPooling)
                {
                    ++meanPooledCubes;
                }
                else
                {
                    ++maxPooledCubes;
                    if (hasStrongPeak)
                    {
                        ++strongPeakOverrideCubes;
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

    int removedSmallChunkCubes = 0;
    int smallChunkCandidateCubes = 0;
    int removedSmallChunks = 0;
    if (config.simulation.adaptive_cube_pooling_remove_isolated_bright_cubes)
    {
        const float cleanupThreshold =
            std::max(zeroThreshold,
                     config.simulation.adaptive_cube_pooling_isolated_bright_cube_threshold);
        const double minCellVolume =
            (4.0 / 3.0) * M_PI *
            static_cast<double>(std::max(1.0, config.cell->minARadius)) *
            static_cast<double>(std::max(1.0, config.cell->minBRadius)) *
            static_cast<double>(std::max(1.0, config.cell->minCRadius));
        const double cubeVolume = static_cast<double>(cubeSize) * cubeSize * cubeSize;
        const double minChunkSizeTarget =
            std::max(0.0f, config.simulation.adaptive_cube_pooling_min_chunk_size_scale) *
            (cubeVolume > 0.0 ? (minCellVolume / cubeVolume) : 0.0);
        const int minChunkCubeCount =
            std::max(1, static_cast<int>(std::ceil(minChunkSizeTarget)));

        std::vector<char> clearCube(static_cast<size_t>(gridZ) * gridY * gridX, 0);
        const std::array<cv::Point3i, 6> faceOffsets = {
            cv::Point3i(1, 0, 0), cv::Point3i(-1, 0, 0),
            cv::Point3i(0, 1, 0), cv::Point3i(0, -1, 0),
            cv::Point3i(0, 0, 1), cv::Point3i(0, 0, -1)
        };
        auto inBounds = [&](int gz, int gy, int gx) {
            return gz >= 0 && gz < gridZ && gy >= 0 && gy < gridY && gx >= 0 && gx < gridX;
        };

        std::vector<char> visited(static_cast<size_t>(gridZ) * gridY * gridX, 0);
        for (int gz = 0; gz < gridZ; ++gz)
        {
            for (int gy = 0; gy < gridY; ++gy)
            {
                for (int gx = 0; gx < gridX; ++gx)
                {
                    const size_t idx = cubeIndex(gz, gy, gx);
                    if (visited[idx] || pooledCubeValues[idx] <= cleanupThreshold)
                    {
                        continue;
                    }

                    std::vector<size_t> chunk;
                    std::vector<cv::Point3i> queue{cv::Point3i(gx, gy, gz)};
                    visited[idx] = 1;
                    size_t cursor = 0;
                    while (cursor < queue.size())
                    {
                        const cv::Point3i current = queue[cursor++];
                        const size_t currentIdx = cubeIndex(current.z, current.y, current.x);
                        chunk.push_back(currentIdx);
                        for (const auto &offset : faceOffsets)
                        {
                            const cv::Point3i neighbor(
                                current.x + offset.x,
                                current.y + offset.y,
                                current.z + offset.z);
                            if (!inBounds(neighbor.z, neighbor.y, neighbor.x))
                            {
                                continue;
                            }
                            const size_t neighborIdx =
                                cubeIndex(neighbor.z, neighbor.y, neighbor.x);
                            if (visited[neighborIdx] ||
                                pooledCubeValues[neighborIdx] <= cleanupThreshold)
                            {
                                continue;
                            }
                            visited[neighborIdx] = 1;
                            queue.push_back(neighbor);
                        }
                    }

                    smallChunkCandidateCubes += static_cast<int>(chunk.size());
                    if (static_cast<int>(chunk.size()) < minChunkCubeCount)
                    {
                        ++removedSmallChunks;
                        for (const size_t chunkIdx : chunk)
                        {
                            clearCube[chunkIdx] = 1;
                        }
                    }
                }
            }
        }

        // Parallel voxel-zero pass. Each cube writes to its own disjoint
        // voxel range in pooled. Counter via reduction.
        #pragma omp parallel for schedule(static) reduction(+:removedSmallChunkCubes)
        for (int gz = 0; gz < gridZ; ++gz)
        {
            const int z0 = gz * cubeSize;
            const int z1 = std::min(depth, z0 + cubeSize);
            for (int gy = 0; gy < gridY; ++gy)
            {
                const int y0 = gy * cubeSize;
                const int y1 = std::min(rows, y0 + cubeSize);
                for (int gx = 0; gx < gridX; ++gx)
                {
                    if (!clearCube[cubeIndex(gz, gy, gx)])
                    {
                        continue;
                    }

                    ++removedSmallChunkCubes;
                    const int x0 = gx * cubeSize;
                    const int x1 = std::min(cols, x0 + cubeSize);
                    for (int z = z0; z < z1; ++z)
                    {
                        cv::Mat &slice = pooled[static_cast<size_t>(z)];
                        for (int y = y0; y < y1; ++y)
                        {
                            float *row = slice.ptr<float>(y);
                            for (int x = x0; x < x1; ++x)
                            {
                                row[x] = 0.0f;
                            }
                        }
                    }
                }
            }
        }
    }

    log << "[AdaptiveCubePooling]"
        << " cubeSize=" << cubeSize
        << " grid=" << gridX << "x" << gridY << "x" << gridZ
        << " zeroThreshold=" << zeroThreshold
        << " majorityThreshold=" << majorityThreshold
        << " strongPeakThreshold=" << strongPeakThreshold
        << " meanCubes=" << meanPooledCubes
        << " maxCubes=" << maxPooledCubes
        << " strongPeakOverrideCubes=" << strongPeakOverrideCubes
        << " cleanupThreshold="
        << (config.simulation.adaptive_cube_pooling_remove_isolated_bright_cubes
                ? std::max(zeroThreshold,
                           config.simulation.adaptive_cube_pooling_isolated_bright_cube_threshold)
                : 0.0f)
        << " minChunkSizeScale="
        << config.simulation.adaptive_cube_pooling_min_chunk_size_scale
        << " minChunkCubeCount="
        << (config.simulation.adaptive_cube_pooling_remove_isolated_bright_cubes
                ? std::max(1, static_cast<int>(std::ceil(
                      std::max(0.0f, config.simulation.adaptive_cube_pooling_min_chunk_size_scale) *
                      (((4.0 / 3.0) * M_PI *
                        static_cast<double>(std::max(1.0, config.cell->minARadius)) *
                        static_cast<double>(std::max(1.0, config.cell->minBRadius)) *
                        static_cast<double>(std::max(1.0, config.cell->minCRadius))) /
                       static_cast<double>(cubeSize * cubeSize * cubeSize)))))
                : 0)
        << " cleanupCandidateCubes=" << smallChunkCandidateCubes
        << " removedSmallChunks=" << removedSmallChunks
        << " removedSmallChunkCubes=" << removedSmallChunkCubes
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

std::string normalizePoolingMode(std::string mode)
{
    std::transform(mode.begin(), mode.end(), mode.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (mode != "mean" && mode != "max")
    {
        mode = "max";
    }
    return mode;
}

ImageHandler::SignalCenterDetectionResult detectSignalCenters(
    const ImageStack &stack,
    const BaseConfig &config,
    std::ostream &log)
{
    ImageHandler::SignalCenterDetectionResult result;
    if (stack.empty() || !config.cell)
    {
        return result;
    }

    const int sizeX = stack[0].cols;
    const int sizeY = stack[0].rows;
    const int sizeZ = static_cast<int>(stack.size());
    if (sizeX <= 0 || sizeY <= 0 || sizeZ <= 0)
    {
        return result;
    }

    const int baseCubeSize = computeAdaptiveCubeSize(config);
    const float cubeScale = std::max(1.0f, config.simulation.signal_center_pooling_cube_scale);
    const int cubeSize = std::max(
        1, static_cast<int>(std::lround(static_cast<float>(baseCubeSize) * cubeScale)));
    const std::string poolingMode =
        normalizePoolingMode(config.simulation.signal_center_pooling_mode);
    const int gridX = (sizeX + cubeSize - 1) / cubeSize;
    const int gridY = (sizeY + cubeSize - 1) / cubeSize;
    const int gridZ = (sizeZ + cubeSize - 1) / cubeSize;

    const float backgroundValue = 0.0f;
    const float minDelta =
        std::max(0.0f, config.simulation.signal_center_min_cube_brightness_delta);
    const float recursiveTopPercentile =
        std::clamp(config.simulation.signal_center_recursive_top_percentile, 0.0f, 1.0f);
    const int recursiveMaxDepth =
        std::max(0, config.simulation.signal_center_recursive_max_depth);

    std::vector<BrightBox> boxes;
    boxes.reserve(static_cast<size_t>(gridX) * gridY * gridZ);
    for (int iz = 0; iz < gridZ; ++iz)
    {
        const int z0 = iz * cubeSize;
        const int z1 = std::min(sizeZ, z0 + cubeSize);
        for (int iy = 0; iy < gridY; ++iy)
        {
            const int y0 = iy * cubeSize;
            const int y1 = std::min(sizeY, y0 + cubeSize);
            for (int ix = 0; ix < gridX; ++ix)
            {
                const int x0 = ix * cubeSize;
                const int x1 = std::min(sizeX, x0 + cubeSize);

                double sum = 0.0;
                float maxValue = 0.0f;
                int voxels = 0;
                for (int z = z0; z < z1; ++z)
                {
                    for (int y = y0; y < y1; ++y)
                    {
                        const float *row = stack[static_cast<size_t>(z)].ptr<float>(y);
                        for (int x = x0; x < x1; ++x)
                        {
                            const float value = row[x];
                            sum += value;
                            maxValue = std::max(maxValue, value);
                            ++voxels;
                        }
                    }
                }
                if (voxels <= 0)
                {
                    continue;
                }

                const float meanBrightness = static_cast<float>(sum / static_cast<double>(voxels));
                const float pooledBrightness =
                    (poolingMode == "mean") ? meanBrightness : maxValue;
                if (pooledBrightness <= backgroundValue + minDelta)
                {
                    continue;
                }

                boxes.push_back({
                    ix, iy, iz,
                    cv::Point3f(
                        static_cast<float>(x0 + x1 - 1) * 0.5f,
                        static_cast<float>(y0 + y1 - 1) * 0.5f,
                        static_cast<float>(z0 + z1 - 1) * 0.5f),
                    pooledBrightness,
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

    const std::array<cv::Point3i, 6> faceOffsets = {
        cv::Point3i(1, 0, 0), cv::Point3i(-1, 0, 0),
        cv::Point3i(0, 1, 0), cv::Point3i(0, -1, 0),
        cv::Point3i(0, 0, 1), cv::Point3i(0, 0, -1)
    };

    std::vector<std::vector<size_t>> adjacency(boxes.size());
    for (size_t i = 0; i < boxes.size(); ++i)
    {
        const BrightBox &box = boxes[i];
        for (const auto &offset : faceOffsets)
        {
            auto it = boxIndex.find({box.ix + offset.x, box.iy + offset.y, box.iz + offset.z});
            if (it != boxIndex.end())
            {
                adjacency[i].push_back(it->second);
            }
        }
    }

    auto sortIndicesByBrightness = [&](std::vector<size_t> &indices) {
        std::sort(indices.begin(), indices.end(),
                  [&](size_t lhs, size_t rhs) {
                      if (boxes[lhs].brightness != boxes[rhs].brightness)
                      {
                          return boxes[lhs].brightness > boxes[rhs].brightness;
                      }
                      return lhs < rhs;
                  });
    };

    auto connectedComponentsForNodes =
        [&](const std::vector<size_t> &nodes) -> std::vector<std::vector<size_t>>
    {
        std::vector<std::vector<size_t>> components;
        if (nodes.empty())
        {
            return components;
        }

        std::vector<char> allowed(boxes.size(), 0);
        std::vector<char> visited(boxes.size(), 0);
        for (const size_t node : nodes)
        {
            allowed[node] = 1;
        }

        for (const size_t seed : nodes)
        {
            if (visited[seed])
            {
                continue;
            }
            std::vector<size_t> component;
            std::vector<size_t> queue{seed};
            visited[seed] = 1;
            size_t cursor = 0;
            while (cursor < queue.size())
            {
                const size_t current = queue[cursor++];
                component.push_back(current);
                for (const size_t neighbor : adjacency[current])
                {
                    if (!allowed[neighbor] || visited[neighbor])
                    {
                        continue;
                    }
                    visited[neighbor] = 1;
                    queue.push_back(neighbor);
                }
            }
            sortIndicesByBrightness(component);
            components.push_back(std::move(component));
        }

        std::sort(components.begin(), components.end(),
                  [&](const std::vector<size_t> &lhs, const std::vector<size_t> &rhs) {
                      const float lhsBrightness = boxes[lhs.front()].brightness;
                      const float rhsBrightness = boxes[rhs.front()].brightness;
                      if (lhsBrightness != rhsBrightness)
                      {
                          return lhsBrightness > rhsBrightness;
                      }
                      return lhs.front() < rhs.front();
                  });
        return components;
    };

    std::function<std::vector<std::vector<size_t>>(
        const std::vector<size_t> &,
        int,
        const std::vector<size_t> &)> refineGroup;
    refineGroup =
        [&](const std::vector<size_t> &group,
            int depth,
            const std::vector<size_t> &previousSelected) -> std::vector<std::vector<size_t>>
    {
        if (group.size() <= 1 || depth >= recursiveMaxDepth)
        {
            return {group};
        }

        std::vector<size_t> sortedGroup = group;
        sortIndicesByBrightness(sortedGroup);

        const size_t selectedCount = std::max<size_t>(
            1, static_cast<size_t>(std::ceil(
                   recursiveTopPercentile * static_cast<float>(sortedGroup.size()))));
        std::vector<size_t> selected(sortedGroup.begin(),
                                     sortedGroup.begin() + static_cast<std::ptrdiff_t>(
                                         std::min(selectedCount, sortedGroup.size())));
        std::sort(selected.begin(), selected.end());
        if (!previousSelected.empty() && selected == previousSelected)
        {
            return {group};
        }

        std::vector<std::vector<size_t>> selectedComponents =
            connectedComponentsForNodes(selected);
        if (selectedComponents.size() < 2)
        {
            return {group};
        }

        std::vector<int> labels(boxes.size(), -1);
        std::vector<char> inGroup(boxes.size(), 0);
        std::vector<size_t> queue;
        queue.reserve(group.size());
        for (const size_t node : group)
        {
            inGroup[node] = 1;
        }
        for (size_t compIdx = 0; compIdx < selectedComponents.size(); ++compIdx)
        {
            for (const size_t node : selectedComponents[compIdx])
            {
                labels[node] = static_cast<int>(compIdx);
                queue.push_back(node);
            }
        }

        size_t cursor = 0;
        while (cursor < queue.size())
        {
            const size_t current = queue[cursor++];
            for (const size_t neighbor : adjacency[current])
            {
                if (!inGroup[neighbor] || labels[neighbor] != -1)
                {
                    continue;
                }
                labels[neighbor] = labels[current];
                queue.push_back(neighbor);
            }
        }

        std::vector<std::vector<size_t>> partitioned(selectedComponents.size());
        for (const size_t node : group)
        {
            const int label = labels[node];
            if (label >= 0)
            {
                partitioned[static_cast<size_t>(label)].push_back(node);
            }
        }

        std::vector<std::vector<size_t>> refinedGroups;
        for (size_t compIdx = 0; compIdx < partitioned.size(); ++compIdx)
        {
            if (partitioned[compIdx].empty())
            {
                continue;
            }
            sortIndicesByBrightness(partitioned[compIdx]);
            std::vector<size_t> previousCore = selectedComponents[compIdx];
            std::sort(previousCore.begin(), previousCore.end());
            std::vector<std::vector<size_t>> childGroups =
                refineGroup(partitioned[compIdx], depth + 1, previousCore);
            refinedGroups.insert(refinedGroups.end(),
                                 childGroups.begin(),
                                 childGroups.end());
        }
        return refinedGroups.empty() ? std::vector<std::vector<size_t>>{group}
                                     : refinedGroups;
    };

    std::vector<std::vector<size_t>> initialGroups = connectedComponentsForNodes(
        [&]() {
            std::vector<size_t> nodes(boxes.size());
            std::iota(nodes.begin(), nodes.end(), 0);
            return nodes;
        }());

    std::vector<std::vector<size_t>> finalGroups;
    for (const auto &group : initialGroups)
    {
        std::vector<std::vector<size_t>> refined = refineGroup(group, 0, {});
        finalGroups.insert(finalGroups.end(), refined.begin(), refined.end());
    }

    float maxCenterBrightness = backgroundValue;
    std::vector<Frame::SignalCenter> rawCenters;
    rawCenters.reserve(finalGroups.size());
    for (const auto &group : finalGroups)
    {
        if (group.empty())
        {
            continue;
        }
        double weightSum = 0.0;
        double xSum = 0.0;
        double ySum = 0.0;
        double zSum = 0.0;
        double brightnessSum = 0.0;
        for (const size_t node : group)
        {
            const BrightBox &box = boxes[node];
            const float weight = std::max(1e-6f, box.brightness - backgroundValue);
            weightSum += weight;
            xSum += weight * static_cast<double>(box.center.x);
            ySum += weight * static_cast<double>(box.center.y);
            zSum += weight * static_cast<double>(box.center.z);
            brightnessSum += static_cast<double>(box.brightness);
        }
        if (weightSum <= 0.0)
        {
            continue;
        }

        Frame::SignalCenter center;
        center.position = cv::Point3f(
            static_cast<float>(xSum / weightSum),
            static_cast<float>(ySum / weightSum),
            static_cast<float>(zSum / weightSum));
        center.brightness = static_cast<float>(
            brightnessSum / static_cast<double>(group.size()));
        center.boxes = static_cast<int>(group.size());
        rawCenters.push_back(center);
        maxCenterBrightness = std::max(maxCenterBrightness, center.brightness);
    }

    const float surroundingZeroThreshold = std::max(
        backgroundValue + 1e-6f,
        std::max(minDelta * 0.5f, config.simulation.adaptive_cube_pooling_zero_threshold));
    const int minBrightSurroundingCubes =
        std::max(1, config.simulation.signal_center_min_bright_surrounding_cubes);
    std::vector<Frame::SignalCenter> filteredCenters;
    filteredCenters.reserve(rawCenters.size());
    int removedDarkSurroundCenters = 0;
    for (const auto &center : rawCenters)
    {
        const int cgx = std::clamp(
            static_cast<int>(std::floor(center.position.x / static_cast<float>(cubeSize))),
            0, std::max(0, gridX - 1));
        const int cgy = std::clamp(
            static_cast<int>(std::floor(center.position.y / static_cast<float>(cubeSize))),
            0, std::max(0, gridY - 1));
        const int cgz = std::clamp(
            static_cast<int>(std::floor(center.position.z / static_cast<float>(cubeSize))),
            0, std::max(0, gridZ - 1));

        int brightSurroundCount = 0;
        for (int dz = -1; dz <= 1; ++dz)
        {
            for (int dy = -1; dy <= 1; ++dy)
            {
                for (int dx = -1; dx <= 1; ++dx)
                {
                    if (dx == 0 && dy == 0 && dz == 0)
                    {
                        continue;
                    }

                    auto it = boxIndex.find({cgx + dx, cgy + dy, cgz + dz});
                    if (it == boxIndex.end())
                    {
                        continue;
                    }
                    if (boxes[it->second].brightness > surroundingZeroThreshold)
                    {
                        ++brightSurroundCount;
                    }
                }
            }
        }

        if (brightSurroundCount < minBrightSurroundingCubes)
        {
            ++removedDarkSurroundCenters;
            continue;
        }
        filteredCenters.push_back(center);
    }

    const float minRadius = std::max(
        1.0f,
        static_cast<float>(std::min({config.cell->minARadius,
                                     config.cell->minBRadius,
                                     config.cell->minCRadius})));
    std::vector<char> centerVisited(filteredCenters.size(), 0);
    int mergedCenterGroups = 0;
    for (size_t seed = 0; seed < filteredCenters.size(); ++seed)
    {
        if (centerVisited[seed])
        {
            continue;
        }

        std::vector<size_t> component{seed};
        centerVisited[seed] = 1;
        size_t cursor = 0;
        while (cursor < component.size())
        {
            const size_t current = component[cursor++];
            for (size_t other = 0; other < filteredCenters.size(); ++other)
            {
                if (centerVisited[other])
                {
                    continue;
                }
                const float dist = static_cast<float>(
                    cv::norm(filteredCenters[current].position - filteredCenters[other].position));
                if (dist < minRadius)
                {
                    centerVisited[other] = 1;
                    component.push_back(other);
                }
            }
        }

        if (component.size() == 1)
        {
            result.centers.push_back(filteredCenters[seed]);
            continue;
        }

        ++mergedCenterGroups;
        double weightSum = 0.0;
        double xSum = 0.0;
        double ySum = 0.0;
        double zSum = 0.0;
        double brightnessSum = 0.0;
        int totalBoxes = 0;
        for (const size_t idx : component)
        {
            const Frame::SignalCenter &center = filteredCenters[idx];
            const double weight = std::max(1e-6, static_cast<double>(center.brightness));
            weightSum += weight;
            xSum += weight * static_cast<double>(center.position.x);
            ySum += weight * static_cast<double>(center.position.y);
            zSum += weight * static_cast<double>(center.position.z);
            brightnessSum += static_cast<double>(center.brightness);
            totalBoxes += center.boxes;
        }

        Frame::SignalCenter merged;
        merged.position = cv::Point3f(
            static_cast<float>(xSum / weightSum),
            static_cast<float>(ySum / weightSum),
            static_cast<float>(zSum / weightSum));
        merged.brightness = static_cast<float>(
            brightnessSum / static_cast<double>(component.size()));
        merged.boxes = totalBoxes;
        result.centers.push_back(merged);
    }

    for (auto &center : result.centers)
    {
        const float normalized = (maxCenterBrightness > backgroundValue + 1e-6f)
            ? std::clamp((center.brightness - backgroundValue) /
                         (maxCenterBrightness - backgroundValue), 0.0f, 1.0f)
            : 0.0f;
        center.sigmaScale = std::max(
            config.simulation.signal_guided_min_sigma_scale,
            1.0f - normalized * (1.0f - config.simulation.signal_guided_min_sigma_scale));
    }

    std::sort(result.centers.begin(), result.centers.end(),
              [](const Frame::SignalCenter &a, const Frame::SignalCenter &b)
              {
                  return a.brightness > b.brightness;
              });

    result.cubeSize = cubeSize;
    result.gridX = gridX;
    result.gridY = gridY;
    result.gridZ = gridZ;
    result.keptBoxes = static_cast<int>(boxes.size());
    result.poolingMode = poolingMode;

    log << "[Signal Centers]"
        << " cubeSize=" << cubeSize
        << " scale=" << cubeScale
        << " pooling=" << poolingMode
        << " grid=(" << gridX << "," << gridY << "," << gridZ << ")"
        << " minDelta=" << minDelta
        << " recursiveTopPercentile=" << recursiveTopPercentile
        << " recursiveMaxDepth=" << recursiveMaxDepth
        << " keptBoxes=" << boxes.size()
        << " initialGroups=" << initialGroups.size()
        << " surroundingZeroThreshold=" << surroundingZeroThreshold
        << " minBrightSurroundingCubes=" << minBrightSurroundingCubes
        << " removedDarkSurroundCenters=" << removedDarkSurroundCenters
        << " mergedCenterGroups=" << mergedCenterGroups
        << " clusters=" << result.centers.size()
        << std::endl;
    for (size_t i = 0; i < result.centers.size(); ++i)
    {
        const auto &center = result.centers[i];
        log << "  [Signal Center] idx=" << i
            << " pos=(" << center.position.x << "," << center.position.y << "," << center.position.z << ")"
            << " brightness=" << center.brightness
            << " sigmaScale=" << center.sigmaScale
            << " boxes=" << center.boxes
            << std::endl;
    }

    return result;
}

void clipStack(ImageStack &sequence)
{
    for (auto &slice : sequence)
    {
        cv::min(slice, 1.0f, slice);
        cv::max(slice, 0.0f, slice);
    }
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
    const float minRadius = std::max(1.0f, std::min(aRadius, cRadius));
    const float maxRadius = std::max(1.0f, std::max(aRadius, cRadius));
    const float midRadius = 0.5f * (minRadius + maxRadius);
    const std::array<float, 2> radii = {
        midRadius,
        maxRadius
    };

    std::vector<float> scaleScores;
    scaleScores.reserve(radii.size());

    for (const float radiusAtScale : radii)
    {
        const int innerWindow = makeOddAtLeast(
            static_cast<int>(std::lround(radiusAtScale * 2.0f + 1.0f)));
        const int outerWindow = makeOddAtLeast(
            static_cast<int>(std::lround(radiusAtScale * 4.0f + 1.0f)),
            innerWindow + 2);

        std::vector<float> sliceScores;
        sliceScores.reserve(sequence.size());

        for (const auto &slice : sequence)
        {
            CV_Assert(slice.type() == CV_32F);

            const cv::Mat innerMean = boxMean(slice, innerWindow);
            const cv::Mat outerMean = boxMean(slice, outerWindow);

            std::vector<float> contrastValues;
            std::vector<float> brightInteriorValues;
            std::vector<float> hollowPenaltyValues;
            contrastValues.reserve(static_cast<std::size_t>(slice.total()));
            brightInteriorValues.reserve(static_cast<std::size_t>(slice.total() / 8));
            hollowPenaltyValues.reserve(static_cast<std::size_t>(slice.total() / 8));

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

                    const float hollowDifference = outerRow[x] - innerRow[x];
                    if (hollowDifference > config.simulation.contrast_structure_threshold)
                    {
                        hollowPenaltyValues.push_back(
                            hollowDifference / (stableBackground + config.simulation.contrast_eps));
                    }
                }
            }

            if (contrastValues.empty())
            {
                sliceScores.push_back(0.0f);
                continue;
            }

            const float contrastScore = computePercentileFromValues(
                std::move(contrastValues),
                config.simulation.iterative_score_percentile);
            const float brightInteriorScore = computeMeanOfTopFraction(
                std::move(brightInteriorValues),
                config.simulation.contrast_bright_fraction);
            const float hollowPenaltyScore = computeMeanOfTopFraction(
                std::move(hollowPenaltyValues),
                config.simulation.contrast_bright_fraction);

            sliceScores.push_back(
                contrastScore +
                config.simulation.contrast_brightness_weight * brightInteriorScore -
                config.simulation.contrast_hollow_penalty_weight * hollowPenaltyScore);
        }

        if (sliceScores.empty())
        {
            scaleScores.push_back(0.0f);
            continue;
        }

        scaleScores.push_back(computePercentileFromValues(std::move(sliceScores), 0.5f));
    }

    if (scaleScores.empty())
    {
        return 0.0f;
    }

    float sumScores = 0.0f;
    for (float score : scaleScores)
    {
        sumScores += score;
    }
    return sumScores / static_cast<float>(scaleScores.size());
}

ImageStack ImageHandler::processPreparedSequence(const ImageStack &sequence,
                                                const BaseConfig &config,
                                                std::ostream &log)
{
    ImageStack current = cloneStack(sequence);
    const int maxIterations = std::max(1, config.simulation.iterative_max_count);

    ImageStack bestSequence = cloneStack(current);
    float bestScore = -std::numeric_limits<float>::infinity();
    float previousScore = 0.0f;
    bool hasPreviousScore = false;

    int count = 0;
    float currentPenalty = config.simulation.iterative_penalty;
    bool restoreBestBeforeReward = false;
    float scorePercentile = config.simulation.iterative_score_percentile;
    float rewardGate = config.simulation.iterative_reward_gate;

    while (count < maxIterations)
    {
        if (restoreBestBeforeReward)
        {
            current = cloneStack(bestSequence);
            restoreBestBeforeReward = false;
        }

        for (auto &slice : current)
        {
            for (int y = 0; y < slice.rows; ++y)
            {
                float *row = slice.ptr<float>(y);
                for (int x = 0; x < slice.cols; ++x)
                {
                    if (row[x] < config.simulation.iterative_penalty_range)
                    {
                        const float normalizedDistanceToKeep =
                            (config.simulation.iterative_penalty_range > 1e-6f)
                                ? ((config.simulation.iterative_penalty_range - row[x]) /
                                   config.simulation.iterative_penalty_range)
                                : 1.0f;
                        const float penaltyStrength =
                            std::clamp(normalizedDistanceToKeep, 0.0f, 1.0f);
                        row[x] -= currentPenalty * penaltyStrength;
                        if (row[x] < 0.0f)
                        {
                            row[x] = 0.0f;
                        }
                    }
                }
            }
        }

        BaseConfig penaltyScoringConfig = config;
        penaltyScoringConfig.simulation.iterative_score_percentile = scorePercentile;
        const float penaltyScore = evaluateSequenceContrastScore(current, penaltyScoringConfig);

        if (hasPreviousScore &&
            previousScore - penaltyScore >=
                config.simulation.iterative_penalty_score_drop_stop_threshold)
        {
            restoreBestBeforeReward = true;
            currentPenalty = std::max(
                config.simulation.iterative_min_penalty,
                currentPenalty * config.simulation.iterative_collapse_backoff);
            previousScore = penaltyScore;
            hasPreviousScore = true;
            ++count;
            continue;
        }

        for (auto &slice : current)
        {
            for (int y = 0; y < slice.rows; ++y)
            {
                float *row = slice.ptr<float>(y);
                for (int x = 0; x < slice.cols; ++x)
                {
                    if (row[x] > rewardGate)
                    {
                        row[x] += config.simulation.iterative_reward;
                        if (row[x] > 1.0f)
                        {
                            row[x] = 1.0f;
                        }
                    }
                }
            }
        }

        scorePercentile = std::min(
            scorePercentile + config.simulation.iterative_score_percentile_increment,
            config.simulation.iterative_score_percentile_max);
        rewardGate = std::max(
            config.simulation.iterative_reward_gate_min,
            rewardGate -
                config.simulation.iterative_reward_gate_decrement);

        BaseConfig rewardScoringConfig = config;
        rewardScoringConfig.simulation.iterative_score_percentile = scorePercentile;
        const float score = evaluateSequenceContrastScore(current, rewardScoringConfig);

        if (hasPreviousScore &&
            previousScore - score >= config.simulation.iterative_score_drop_stop_threshold)
        {
            restoreBestBeforeReward = true;
            currentPenalty = std::max(
                config.simulation.iterative_min_penalty,
                currentPenalty * config.simulation.iterative_collapse_backoff);
            previousScore = score;
            hasPreviousScore = true;
            ++count;
            continue;
        }

        if (score > bestScore + config.simulation.iterative_improvement_tolerance)
        {
            bestScore = score;
            bestSequence = cloneStack(current);
        }

        if (count % 50 == 0)
        {
            log << "[IterPreprocess] round=" << count
                << " score=" << score << std::endl;
        }

        previousScore = score;
        hasPreviousScore = true;
        ++count;

        if (score >= config.simulation.iterative_score_max)
        {
            break;
        }

        if (score == 0.0f)
        {
            currentPenalty = std::max(
                config.simulation.iterative_min_penalty,
                currentPenalty * config.simulation.iterative_collapse_backoff);
            current = cloneStack(bestSequence);
            continue;
        }

    }

    log << "[IterPreprocess] best_score=" << bestScore << std::endl;

    for (auto &slice : bestSequence)
    {
        if (config.simulation.post_process_blur_sigma > 0.0f)
        {
            cv::GaussianBlur(slice,
                             slice,
                             cv::Size(0, 0),
                             config.simulation.post_process_blur_sigma,
                             config.simulation.post_process_blur_sigma);
        }

        for (int y = 0; y < slice.rows; ++y)
        {
            float *row = slice.ptr<float>(y);
            for (int x = 0; x < slice.cols; ++x)
            {
                if (row[x] < config.simulation.post_process_black_percentile)
                {
                    row[x] = 0.0f;
                }
                else if (row[x] < config.simulation.post_process_white_percentile)
                {
                    row[x] *= config.simulation.post_process_amplification;
                }
            }
        }

    }

    if (config.simulation.post_process_final_blur_sigma > 0.0f)
    {
        const float directWeight = std::clamp(
            config.simulation.post_process_final_direct_weight,
            0.0f,
            1.0f);
        const float blurredWeight = 1.0f - directWeight;
        const float directAmplification =
            std::max(0.0f, config.simulation.post_process_final_direct_amplification);
        const float blurredAmplification =
            std::max(0.0f, config.simulation.post_process_final_blurred_amplification);

        for (auto &slice : bestSequence)
        {
            cv::Mat directSlice = slice.clone();
            cv::Mat blurredSlice;
            cv::GaussianBlur(slice,
                             blurredSlice,
                             cv::Size(0, 0),
                             config.simulation.post_process_final_blur_sigma,
                             config.simulation.post_process_final_blur_sigma);

            if (directAmplification != 1.0f)
            {
                directSlice *= directAmplification;
            }
            if (blurredAmplification != 1.0f)
            {
                blurredSlice *= blurredAmplification;
            }

            cv::addWeighted(directSlice,
                            directWeight,
                            blurredSlice,
                            blurredWeight,
                            0.0,
                            slice);
        }
    }

    clipStack(bestSequence);
    return bestSequence;
}

std::vector<cv::Mat> ImageHandler::loadRawFrame(const std::string &imageFile,
                                                const BaseConfig &config,
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
            << " rawChannels=" << firstSlice.channels()
            << " rawRows=" << firstSlice.rows
            << " rawCols=" << firstSlice.cols
            << std::endl;

        normalizedSlices.reserve(numTiffSlices);
        for (const auto &rawSlice : tiffImage)
        {
            cv::Mat slice = rawSlice;
            if (slice.channels() == 3)
            {
                cv::cvtColor(slice, slice, cv::COLOR_BGR2GRAY);
            }
            normalizedSlices.push_back(processImage(slice, config));
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

    processedZSlices = processPreparedSequence(processedZSlices, config, log);

    const float localScore = evaluateSequenceContrastScore(processedZSlices, config);
    log << "[PreprocessScores] file=" << fs::path(imageFile).filename().string()
        << " local=" << localScore
        << std::endl;

    printStackStats(log, "processed_sequence", imageFile, processedZSlices);

    if (processedZSlices.empty())
    {
        return interpolatedZSlices;
    }

    if (processedZSlices.size() == 1)
    {
        interpolatedZSlices = processedZSlices;
    }
    else
    {
        const int expandFactor = config.simulation.z_scaling;
        const unsigned numSynthSlices =
            static_cast<unsigned>(expandFactor) * (processedZSlices.size() - 1U) + 1U;

        for (unsigned synthSlice = 0; synthSlice < numSynthSlices; ++synthSlice)
        {
            const int sourceSlice = static_cast<int>(synthSlice / expandFactor);
            if (synthSlice % expandFactor == 0)
            {
                interpolatedZSlices.push_back(processedZSlices[sourceSlice]);
            }
            else if (synthSlice % expandFactor == 1)
            {
                interpolateSlices(processedZSlices[sourceSlice],
                                  processedZSlices[sourceSlice + 1],
                                  interpolatedZSlices,
                                  expandFactor - 1);
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
    interpolatedZSlices = applyAdaptiveCubePooling(interpolatedZSlices, config, log);
    printStackStats(log, "post_cube_pooling", imageFile, interpolatedZSlices);
    log << std::to_string(interpolatedZSlices.size()) << "slices built successfully" << std::endl;
    return interpolatedZSlices;
}

ImageHandler::SignalCenterDetectionResult ImageHandler::detectSignalCentersInStack(
    const ImageStack &stack,
    const BaseConfig &config,
    std::ostream *logSink)
{
    std::ostream &log = logSink ? *logSink : std::cout;
    return detectSignalCenters(stack, config, log);
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
