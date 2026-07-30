#include "../includes/Frame.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <sstream>

namespace {
template <typename Fn>
void forEachSliceIndex(const SimulationConfig &config, int count, const Fn &fn)
{
    if (count <= 0) {
        return;
    }

    const bool useParallel = config.parallel_threads > 1 &&
                            count >= config.parallel_min_slices;
    if (!useParallel) {
        for (int i = 0; i < count; ++i) {
            fn(i);
        }
        return;
    }

#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < count; ++i) {
        fn(i);
    }
#else
    cv::parallel_for_(cv::Range(0, count), [&](const cv::Range &range) {
        for (int i = range.start; i < range.end; ++i) {
            fn(i);
        }
    });
#endif
}

// OpenMP pragmas are parsed by the compiler when -fopenmp is passed;
// no header include needed unless runtime functions (omp_get_thread_num, etc.)
// are used. The #pragma directives below become no-ops without -fopenmp.
} // namespace

// Asymmetric-L2 per-slice cost (Fix E).
// Returns sqrt(sum(w_i * (synth_i - real_i)^2)) where w_i = k when synth>real
// and w_i = 1 when synth<=real. With k=1.0 this is identical to
// cv::norm(real, synth, NORM_L2). Penalizes a cell covering a dark image
// region more heavily than a cell undershooting a bright image region —
// forces split-vs-no-split cost comparison to prefer daughters that cover
// only bright blobs over a parent that also covers the inter-daughter valley.
//
// Uses SIMD-optimized OpenCV primitives (subtract, multiply, compare, sum)
// to stay close to cv::norm performance. Decomposition:
//   sumSq       = sum(diff^2)              (all pixels)
//   posSumSq    = sum(diff^2 where diff>0) (overshoot pixels only)
//   asymSumSq   = sumSq + (k-1) * posSumSq
static double asymmetricL2Slice(const cv::Mat &real, const cv::Mat &synth, float k)
{
    if (k <= 1.0f + 1e-6f) {
        return cv::norm(real, synth, cv::NORM_L2);
    }
    CV_Assert(real.type() == CV_32F && synth.type() == CV_32F);
    CV_Assert(real.size() == synth.size());

    cv::Mat diff;
    cv::subtract(synth, real, diff);
    cv::Mat diffSq;
    cv::multiply(diff, diff, diffSq);
    const double sumSq = cv::sum(diffSq)[0];

    // Mask of pixels where diff > 0 (synth overshoots real).
    cv::Mat posMask;
    cv::compare(diff, 0.0f, posMask, cv::CMP_GT);   // 8U mask, 255 or 0

    // Copy squared diffs only at overshoot pixels; sum those.
    cv::Mat posSq = cv::Mat::zeros(diffSq.size(), diffSq.type());
    diffSq.copyTo(posSq, posMask);
    const double posSumSq = cv::sum(posSq)[0];

    const double asymSumSq = sumSq + static_cast<double>(k - 1.0f) * posSumSq;
    return std::sqrt(std::max(0.0, asymSumSq));
}

struct BrightPixel
{
    cv::Point3f pos;   // world coordinates (x, y, z in interpolated-z space)
    float weight;      // pixel intensity above background
};

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

static float sampleSignalProbability(const std::vector<cv::Mat> &probability,
                                     float x,
                                     float y,
                                     float z)
{
    if (probability.empty()) {
        return 0.0f;
    }

    const int maxZ = static_cast<int>(probability.size()) - 1;
    const int maxY = probability[0].rows - 1;
    const int maxX = probability[0].cols - 1;
    if (maxX < 0 || maxY < 0 || maxZ < 0) {
        return 0.0f;
    }

    x = std::clamp(x, 0.0f, static_cast<float>(maxX));
    y = std::clamp(y, 0.0f, static_cast<float>(maxY));
    z = std::clamp(z, 0.0f, static_cast<float>(maxZ));

    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int z0 = static_cast<int>(std::floor(z));
    const int x1 = std::min(maxX, x0 + 1);
    const int y1 = std::min(maxY, y0 + 1);
    const int z1 = std::min(maxZ, z0 + 1);
    const float tx = x - static_cast<float>(x0);
    const float ty = y - static_cast<float>(y0);
    const float tz = z - static_cast<float>(z0);

    auto at = [&](int zz, int yy, int xx) -> float {
        const cv::Mat &slice = probability[static_cast<size_t>(zz)];
        if (slice.empty() || slice.type() != CV_32F) {
            return 0.0f;
        }
        return slice.ptr<float>(yy)[xx];
    };

    const float c000 = at(z0, y0, x0);
    const float c100 = at(z0, y0, x1);
    const float c010 = at(z0, y1, x0);
    const float c110 = at(z0, y1, x1);
    const float c001 = at(z1, y0, x0);
    const float c101 = at(z1, y0, x1);
    const float c011 = at(z1, y1, x0);
    const float c111 = at(z1, y1, x1);

    const float c00 = c000 * (1.0f - tx) + c100 * tx;
    const float c10 = c010 * (1.0f - tx) + c110 * tx;
    const float c01 = c001 * (1.0f - tx) + c101 * tx;
    const float c11 = c011 * (1.0f - tx) + c111 * tx;
    const float c0 = c00 * (1.0f - ty) + c10 * ty;
    const float c1 = c01 * (1.0f - ty) + c11 * ty;
    return std::clamp(c0 * (1.0f - tz) + c1 * tz, 0.0f, 1.0f);
}

static cv::Point3f signalProbabilityGradientAt(const std::vector<cv::Mat> &probability,
                                               const cv::Point3f &pos)
{
    return cv::Point3f(
        0.5f * (sampleSignalProbability(probability, pos.x + 1.0f, pos.y, pos.z) -
                sampleSignalProbability(probability, pos.x - 1.0f, pos.y, pos.z)),
        0.5f * (sampleSignalProbability(probability, pos.x, pos.y + 1.0f, pos.z) -
                sampleSignalProbability(probability, pos.x, pos.y - 1.0f, pos.z)),
        0.5f * (sampleSignalProbability(probability, pos.x, pos.y, pos.z + 1.0f) -
                sampleSignalProbability(probability, pos.x, pos.y, pos.z - 1.0f)));
}

static bool sampleSignalProbabilityCandidate(const std::vector<cv::Mat> &probability,
                                             const cv::Point3f &center,
                                             float radius,
                                             float minProbability,
                                             cv::Point3f &candidate,
                                             float &candidateProbability)
{
    if (probability.empty() || probability[0].empty() || radius <= 0.0f) {
        return false;
    }

    const int maxZ = static_cast<int>(probability.size()) - 1;
    const int maxY = probability[0].rows - 1;
    const int maxX = probability[0].cols - 1;
    if (maxX < 0 || maxY < 0 || maxZ < 0) {
        return false;
    }

    const float r = std::max(1.0f, radius);
    const float invR2 = 1.0f / (r * r);
    const int x0 = std::max(0, static_cast<int>(std::floor(center.x - r)));
    const int x1 = std::min(maxX, static_cast<int>(std::ceil(center.x + r)));
    const int y0 = std::max(0, static_cast<int>(std::floor(center.y - r)));
    const int y1 = std::min(maxY, static_cast<int>(std::ceil(center.y + r)));
    const int z0 = std::max(0, static_cast<int>(std::floor(center.z - r)));
    const int z1 = std::min(maxZ, static_cast<int>(std::ceil(center.z + r)));
    if (x0 > x1 || y0 > y1 || z0 > z1) {
        return false;
    }

    std::mt19937 &gen = cellUniverseRandomGenerator();
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    double totalWeight = 0.0;
    bool found = false;
    cv::Point3f chosen = center;
    float chosenProbability = 0.0f;
    const float minP = std::max(0.0f, minProbability);
    for (int z = z0; z <= z1; ++z) {
        const cv::Mat &slice = probability[static_cast<size_t>(z)];
        if (slice.empty() || slice.type() != CV_32F) {
            continue;
        }
        const float dz = static_cast<float>(z) - center.z;
        for (int y = y0; y <= y1; ++y) {
            const float dy = static_cast<float>(y) - center.y;
            const float *row = slice.ptr<float>(y);
            for (int x = x0; x <= x1; ++x) {
                const float dx = static_cast<float>(x) - center.x;
                if ((dx * dx + dy * dy + dz * dz) * invR2 > 1.0f) {
                    continue;
                }
                const float p = row[x];
                if (!std::isfinite(p) || p <= minP) {
                    continue;
                }
                totalWeight += static_cast<double>(p);
                if (unit(gen) * totalWeight <= static_cast<double>(p)) {
                    chosen = cv::Point3f(static_cast<float>(x),
                                         static_cast<float>(y),
                                         static_cast<float>(z));
                    chosenProbability = p;
                    found = true;
                }
            }
        }
    }

    if (!found) {
        return false;
    }
    candidate = chosen;
    candidateProbability = chosenProbability;
    return true;
}

static bool findMaxSignalMapGradientInCell(const std::vector<cv::Mat> &signalMap,
                                           const Ellipsoid &cell,
                                           float radiusScale,
                                           float minGradientNorm,
                                           cv::Point3f &direction)
{
    if (signalMap.empty() || signalMap[0].empty() || radiusScale <= 0.0f) {
        return false;
    }

    Ellipsoid analysisCell = cell;
    analysisCell.setRadii(cell.getARadius() * radiusScale,
                          cell.getBRadius() * radiusScale,
                          cell.getCRadius() * radiusScale);

    const float maxR = std::max({analysisCell.getARadius(),
                                 analysisCell.getBRadius(),
                                 analysisCell.getCRadius()});
    const int maxZIndex = static_cast<int>(signalMap.size()) - 1;
    const int minX = std::max(1, static_cast<int>(std::floor(analysisCell.getX() - maxR)));
    const int maxX = std::min(signalMap[0].cols - 2,
                              static_cast<int>(std::ceil(analysisCell.getX() + maxR)));
    const int minY = std::max(1, static_cast<int>(std::floor(analysisCell.getY() - maxR)));
    const int maxY = std::min(signalMap[0].rows - 2,
                              static_cast<int>(std::ceil(analysisCell.getY() + maxR)));
    const int minZ = std::max(1, static_cast<int>(std::floor(analysisCell.getZ() - maxR)));
    const int maxZ = std::min(maxZIndex - 1,
                              static_cast<int>(std::ceil(analysisCell.getZ() + maxR)));
    if (minX > maxX || minY > maxY || minZ > maxZ) {
        return false;
    }

    float bestNorm = 0.0f;
    cv::Point3f bestGradient(0.0f, 0.0f, 0.0f);
    for (int z = minZ; z <= maxZ; ++z) {
        if (signalMap[static_cast<size_t>(z)].type() != CV_32F) {
            return false;
        }
        const cv::Mat &prevSlice = signalMap[static_cast<size_t>(z - 1)];
        const cv::Mat &slice = signalMap[static_cast<size_t>(z)];
        const cv::Mat &nextSlice = signalMap[static_cast<size_t>(z + 1)];
        for (int y = minY; y <= maxY; ++y) {
            const float *row = slice.ptr<float>(y);
            const float *rowYm = slice.ptr<float>(y - 1);
            const float *rowYp = slice.ptr<float>(y + 1);
            const float *prevRow = prevSlice.ptr<float>(y);
            const float *nextRow = nextSlice.ptr<float>(y);
            for (int x = minX; x <= maxX; ++x) {
                const cv::Point3f point(static_cast<float>(x),
                                        static_cast<float>(y),
                                        static_cast<float>(z));
                if (!analysisCell.isPointInsideEllipsoid(point)) {
                    continue;
                }
                const cv::Point3f gradient(
                    0.5f * (row[x + 1] - row[x - 1]),
                    0.5f * (rowYp[x] - rowYm[x]),
                    0.5f * (nextRow[x] - prevRow[x]));
                const float norm = static_cast<float>(cv::norm(gradient));
                if (norm > bestNorm) {
                    bestNorm = norm;
                    bestGradient = gradient;
                }
            }
        }
    }

    if (bestNorm <= std::max(0.0f, minGradientNorm)) {
        return false;
    }

    direction = bestGradient * (1.0f / bestNorm);
    return true;
}

static bool estimateBrightCoreCentroidInCell(const std::vector<cv::Mat> &realFrame,
                                             const Ellipsoid &cell,
                                             const EllipsoidConfig &cellConfig,
                                             cv::Point3f &centroid,
                                             float &trust)
{
    if (realFrame.empty() || realFrame[0].empty()) {
        return false;
    }

    const int maxZIndex = static_cast<int>(realFrame.size()) - 1;
    const float maxR = std::max({cell.getARadius(), cell.getBRadius(), cell.getCRadius()});
    const int minX = std::max(0, static_cast<int>(std::floor(cell.getX() - maxR)));
    const int maxX = std::min(realFrame[0].cols - 1, static_cast<int>(std::ceil(cell.getX() + maxR)));
    const int minY = std::max(0, static_cast<int>(std::floor(cell.getY() - maxR)));
    const int maxY = std::min(realFrame[0].rows - 1, static_cast<int>(std::ceil(cell.getY() + maxR)));
    const int minZ = std::max(0, static_cast<int>(std::floor(cell.getZ() - maxR)));
    const int maxZ = std::min(maxZIndex, static_cast<int>(std::ceil(cell.getZ() + maxR)));
    if (minX > maxX || minY > maxY || minZ > maxZ) {
        return false;
    }

    constexpr float kBlackEpsilon = 1e-6f;
    int insideCount = 0;
    int positiveCount = 0;
    float maxPixel = 0.0f;
    for (int z = minZ; z <= maxZ; ++z) {
        if (realFrame[static_cast<size_t>(z)].type() != CV_32F) {
            return false;
        }
        const cv::Mat &slice = realFrame[static_cast<size_t>(z)];
        for (int y = minY; y <= maxY; ++y) {
            const float *row = slice.ptr<float>(y);
            for (int x = minX; x <= maxX; ++x) {
                if (!cell.isPointInsideEllipsoid(cv::Point3f(
                        static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)))) {
                    continue;
                }
                ++insideCount;
                const float pixel = row[x];
                if (pixel > kBlackEpsilon) {
                    ++positiveCount;
                    maxPixel = std::max(maxPixel, pixel);
                }
            }
        }
    }
    if (insideCount == 0 || positiveCount == 0 || maxPixel <= kBlackEpsilon) {
        return false;
    }

    const float thresholdFraction = std::clamp(
        cellConfig.randomPerturbBrightCoreThresholdFraction, 0.0f, 1.0f);
    const float weightExponent = std::max(
        0.0f, cellConfig.randomPerturbBrightCoreWeightExponent);
    const float brightCoreThreshold = std::max(kBlackEpsilon, thresholdFraction * maxPixel);
    double weightSum = 0.0;
    cv::Point3d weightedSum(0.0, 0.0, 0.0);
    for (int z = minZ; z <= maxZ; ++z) {
        const cv::Mat &slice = realFrame[static_cast<size_t>(z)];
        for (int y = minY; y <= maxY; ++y) {
            const float *row = slice.ptr<float>(y);
            for (int x = minX; x <= maxX; ++x) {
                const cv::Point3f point(static_cast<float>(x),
                                        static_cast<float>(y),
                                        static_cast<float>(z));
                if (!cell.isPointInsideEllipsoid(point)) {
                    continue;
                }
                const float pixel = row[x];
                if (pixel < brightCoreThreshold) {
                    continue;
                }
                double weight = 1.0;
                if (std::abs(weightExponent - 1.0f) <= 1e-6f) {
                    weight = static_cast<double>(pixel);
                } else if (std::abs(weightExponent - 2.0f) <= 1e-6f) {
                    weight = static_cast<double>(pixel) * static_cast<double>(pixel);
                } else if (weightExponent > 0.0f) {
                    weight = std::pow(static_cast<double>(pixel), static_cast<double>(weightExponent));
                }
                weightSum += weight;
                weightedSum.x += weight * x;
                weightedSum.y += weight * y;
                weightedSum.z += weight * z;
            }
        }
    }
    if (weightSum <= 1e-12) {
        return false;
    }

    centroid = cv::Point3f(static_cast<float>(weightedSum.x / weightSum),
                           static_cast<float>(weightedSum.y / weightSum),
                           static_cast<float>(weightedSum.z / weightSum));
    const float brightFraction = static_cast<float>(positiveCount) /
                                 static_cast<float>(std::max(1, insideCount));
    const float blackAndBrightTrust = 1.0f - brightFraction;
    const float baseTrust = std::max(0.0f, cellConfig.randomPerturbBrightCoreBaseTrust);
    const float blackTrust = std::max(0.0f, cellConfig.randomPerturbBrightCoreBlackTrust);
    const float brightnessTrust = std::max(0.0f, cellConfig.randomPerturbBrightCoreBrightnessTrust);
    trust = std::clamp(baseTrust +
                           blackTrust * blackAndBrightTrust +
                           brightnessTrust * std::clamp(maxPixel, 0.0f, 1.0f),
                       0.0f, 1.0f);
    return true;
}

static float positiveVoxelFractionInsideCell(const std::vector<cv::Mat> &realFrame,
                                             const Ellipsoid &cell)
{
    if (realFrame.empty() || realFrame[0].empty()) {
        return 0.0f;
    }

    const int maxZIndex = static_cast<int>(realFrame.size()) - 1;
    const float maxR = std::max({cell.getARadius(), cell.getBRadius(), cell.getCRadius()});
    const int minX = std::max(0, static_cast<int>(std::floor(cell.getX() - maxR)));
    const int maxX = std::min(realFrame[0].cols - 1, static_cast<int>(std::ceil(cell.getX() + maxR)));
    const int minY = std::max(0, static_cast<int>(std::floor(cell.getY() - maxR)));
    const int maxY = std::min(realFrame[0].rows - 1, static_cast<int>(std::ceil(cell.getY() + maxR)));
    const int minZ = std::max(0, static_cast<int>(std::floor(cell.getZ() - maxR)));
    const int maxZ = std::min(maxZIndex, static_cast<int>(std::ceil(cell.getZ() + maxR)));
    if (minX > maxX || minY > maxY || minZ > maxZ) {
        return 0.0f;
    }

    std::array<double, 9> R_T;
    cell.generateInverseRotationMatrix(R_T);
    const double invA2 = 1.0 / std::max(1e-12, static_cast<double>(cell.getARadius()) * cell.getARadius());
    const double invB2 = 1.0 / std::max(1e-12, static_cast<double>(cell.getBRadius()) * cell.getBRadius());
    const double invC2 = 1.0 / std::max(1e-12, static_cast<double>(cell.getCRadius()) * cell.getCRadius());

    int insideCount = 0;
    int positiveCount = 0;
    for (int z = minZ; z <= maxZ; ++z) {
        const cv::Mat &slice = realFrame[static_cast<size_t>(z)];
        if (slice.empty()) {
            continue;
        }
        for (int y = minY; y <= maxY; ++y) {
            for (int x = minX; x <= maxX; ++x) {
                const double dx = static_cast<double>(x) - cell.getX();
                const double dy = static_cast<double>(y) - cell.getY();
                const double dz = static_cast<double>(z) - cell.getZ();
                const double lx = R_T[0] * dx + R_T[1] * dy + R_T[2] * dz;
                const double ly = R_T[3] * dx + R_T[4] * dy + R_T[5] * dz;
                const double lz = R_T[6] * dx + R_T[7] * dy + R_T[8] * dz;
                if (lx * lx * invA2 + ly * ly * invB2 + lz * lz * invC2 > 1.0) {
                    continue;
                }

                ++insideCount;
                double voxel = 0.0;
                switch (slice.depth()) {
                case CV_8U:  voxel = slice.ptr<uint8_t>(y)[x]; break;
                case CV_16U: voxel = slice.ptr<uint16_t>(y)[x]; break;
                case CV_16S: voxel = slice.ptr<int16_t>(y)[x]; break;
                case CV_32S: voxel = slice.ptr<int32_t>(y)[x]; break;
                case CV_32F: voxel = slice.ptr<float>(y)[x]; break;
                case CV_64F: voxel = slice.ptr<double>(y)[x]; break;
                default: break;
                }
                if (voxel > 0.0) {
                    ++positiveCount;
                }
            }
        }
    }

    if (insideCount == 0) {
        return 0.0f;
    }
    return static_cast<float>(positiveCount) / static_cast<float>(insideCount);
}

static float ellipsoidOverlapFractionOfFirst(const std::vector<cv::Mat> &frameShape,
                                             const Ellipsoid &body,
                                             const Ellipsoid &other,
                                             float bodyScale,
                                             float otherScale)
{
    if (frameShape.empty() || frameShape[0].empty()) {
        return 0.0f;
    }

    const float maxR = std::max({body.getARadius(), body.getBRadius(), body.getCRadius()}) *
                       std::max(1e-3f, bodyScale);
    const int maxZIndex = static_cast<int>(frameShape.size()) - 1;
    const int minX = std::max(0, static_cast<int>(std::floor(body.getX() - maxR)));
    const int maxX = std::min(frameShape[0].cols - 1, static_cast<int>(std::ceil(body.getX() + maxR)));
    const int minY = std::max(0, static_cast<int>(std::floor(body.getY() - maxR)));
    const int maxY = std::min(frameShape[0].rows - 1, static_cast<int>(std::ceil(body.getY() + maxR)));
    const int minZ = std::max(0, static_cast<int>(std::floor(body.getZ() - maxR)));
    const int maxZ = std::min(maxZIndex, static_cast<int>(std::ceil(body.getZ() + maxR)));
    if (minX > maxX || minY > maxY || minZ > maxZ) {
        return 0.0f;
    }

    int bodyCount = 0;
    int overlapCount = 0;
    for (int z = minZ; z <= maxZ; ++z) {
        for (int y = minY; y <= maxY; ++y) {
            for (int x = minX; x <= maxX; ++x) {
                const cv::Point3f p(static_cast<float>(x),
                                    static_cast<float>(y),
                                    static_cast<float>(z));
                if (!body.isPointInsideEllipsoid(p, bodyScale)) {
                    continue;
                }
                ++bodyCount;
                if (other.isPointInsideEllipsoid(p, otherScale)) {
                    ++overlapCount;
                }
            }
        }
    }

    if (bodyCount == 0) {
        return 0.0f;
    }
    return static_cast<float>(overlapCount) / static_cast<float>(bodyCount);
}

static inline float ellipsoidBoundingRadius(const Ellipsoid &cell, float scale)
{
    return std::max({cell.getARadius(), cell.getBRadius(), cell.getCRadius()}) *
           std::max(1e-3f, scale);
}

static inline bool ellipsoidBoundingSpheresCanOverlap(const Ellipsoid &a,
                                                      const Ellipsoid &b,
                                                      float aScale,
                                                      float bScale)
{
    const cv::Point3f aPos(a.getX(), a.getY(), a.getZ());
    const cv::Point3f bPos(b.getX(), b.getY(), b.getZ());
    const float maxReach = ellipsoidBoundingRadius(a, aScale) +
                           ellipsoidBoundingRadius(b, bScale);
    return cv::norm(aPos - bPos) <= maxReach;
}

static bool findAnyCellBodyOverlapIn(const std::vector<cv::Mat> &frameShape,
                                     const std::vector<Ellipsoid> &cellList,
                                     bool includeTrash,
                                     float scale,
                                     std::string *firstName,
                                     std::string *secondName,
                                     float *firstInSecond,
                                     float *secondInFirst)
{
    const float bodyScale = std::max(1e-3f, scale);
    for (size_t i = 0; i < cellList.size(); ++i) {
        if (!includeTrash && cellList[i].isTrash()) continue;
        for (size_t j = i + 1; j < cellList.size(); ++j) {
            if (!includeTrash && cellList[j].isTrash()) continue;
            if (!ellipsoidBoundingSpheresCanOverlap(
                    cellList[i], cellList[j], bodyScale, bodyScale)) {
                continue;
            }
            const float ij = ellipsoidOverlapFractionOfFirst(
                frameShape, cellList[i], cellList[j], bodyScale, bodyScale);
            const float ji = ellipsoidOverlapFractionOfFirst(
                frameShape, cellList[j], cellList[i], bodyScale, bodyScale);
            if (ij > 0.0f || ji > 0.0f) {
                if (firstName) *firstName = cellList[i].getName();
                if (secondName) *secondName = cellList[j].getName();
                if (firstInSecond) *firstInSecond = ij;
                if (secondInFirst) *secondInFirst = ji;
                return true;
            }
        }
    }
    return false;
}

static bool findNewOrWorseCellBodyOverlapIn(const std::vector<cv::Mat> &frameShape,
                                            const std::vector<Ellipsoid> &candidateCells,
                                            const std::vector<Ellipsoid> &baselineCells,
                                            bool includeTrash,
                                            float scale,
                                            const std::string &acceptedSiblingA,
                                            const std::string &acceptedSiblingB,
                                            float tolerance,
                                            std::string *firstName,
                                            std::string *secondName,
                                            float *firstInSecond,
                                            float *secondInFirst)
{
    const float bodyScale = std::max(1e-3f, scale);
    const float overlapTolerance = std::max(0.0f, tolerance);

    auto isAcceptedSiblingPair = [&](const std::string &a, const std::string &b) {
        return !acceptedSiblingA.empty() &&
               ((a == acceptedSiblingA && b == acceptedSiblingB) ||
                (a == acceptedSiblingB && b == acceptedSiblingA));
    };

    for (size_t i = 0; i < candidateCells.size(); ++i) {
        if (!includeTrash && candidateCells[i].isTrash()) continue;
        for (size_t j = i + 1; j < candidateCells.size(); ++j) {
            if (!includeTrash && candidateCells[j].isTrash()) continue;
            if (!ellipsoidBoundingSpheresCanOverlap(
                    candidateCells[i], candidateCells[j], bodyScale, bodyScale)) {
                continue;
            }

            const float ij = ellipsoidOverlapFractionOfFirst(
                frameShape, candidateCells[i], candidateCells[j], bodyScale, bodyScale);
            const float ji = ellipsoidOverlapFractionOfFirst(
                frameShape, candidateCells[j], candidateCells[i], bodyScale, bodyScale);
            if (ij <= 0.0f && ji <= 0.0f) {
                continue;
            }

            const std::string candidateA = candidateCells[i].getName();
            const std::string candidateB = candidateCells[j].getName();
            if (isAcceptedSiblingPair(candidateA, candidateB)) {
                continue;
            }
            if (ij <= overlapTolerance && ji <= overlapTolerance) {
                continue;
            }

            bool existed = false;
            float baselineAInB = 0.0f;
            float baselineBInA = 0.0f;
            for (size_t bi = 0; bi < baselineCells.size() && !existed; ++bi) {
                if (!includeTrash && baselineCells[bi].isTrash()) continue;
                for (size_t bj = bi + 1; bj < baselineCells.size(); ++bj) {
                    if (!includeTrash && baselineCells[bj].isTrash()) continue;
                    const std::string baselineA = baselineCells[bi].getName();
                    const std::string baselineB = baselineCells[bj].getName();
                    if (baselineA == candidateA && baselineB == candidateB) {
                        if (!ellipsoidBoundingSpheresCanOverlap(
                                baselineCells[bi], baselineCells[bj],
                                bodyScale, bodyScale)) {
                            existed = true;
                            break;
                        }
                        baselineAInB = ellipsoidOverlapFractionOfFirst(
                            frameShape, baselineCells[bi], baselineCells[bj],
                            bodyScale, bodyScale);
                        baselineBInA = ellipsoidOverlapFractionOfFirst(
                            frameShape, baselineCells[bj], baselineCells[bi],
                            bodyScale, bodyScale);
                        existed = true;
                        break;
                    }
                    if (baselineA == candidateB && baselineB == candidateA) {
                        if (!ellipsoidBoundingSpheresCanOverlap(
                                baselineCells[bj], baselineCells[bi],
                                bodyScale, bodyScale)) {
                            existed = true;
                            break;
                        }
                        baselineAInB = ellipsoidOverlapFractionOfFirst(
                            frameShape, baselineCells[bj], baselineCells[bi],
                            bodyScale, bodyScale);
                        baselineBInA = ellipsoidOverlapFractionOfFirst(
                            frameShape, baselineCells[bi], baselineCells[bj],
                            bodyScale, bodyScale);
                        existed = true;
                        break;
                    }
                }
            }

            if (existed &&
                ij <= baselineAInB + overlapTolerance &&
                ji <= baselineBInA + overlapTolerance) {
                continue;
            }

            if (firstName) *firstName = candidateA;
            if (secondName) *secondName = candidateB;
            if (firstInSecond) *firstInSecond = ij;
            if (secondInFirst) *secondInFirst = ji;
            return true;
        }
    }

    return false;
}

// Function to interpolate between two slices
void interpolateSlices(const cv::Mat& slice1, const cv::Mat& slice2, 
                       std::vector<cv::Mat>& processedSlices, int numInterpolations) {
    // Ensure the two slices have the same size and type
    if (slice1.size() != slice2.size() || slice1.type() != slice2.type()) {
        throw std::invalid_argument("Slices must have the same size and type for interpolation!");
    }

    // Perform interpolation
    for (int i = 1; i <= numInterpolations; ++i) {
        double t = static_cast<double>(i) / (numInterpolations + 1);
        cv::Mat interpolatedSlice = (1.0 - t) * slice1 + t * slice2;
        processedSlices.push_back(interpolatedSlice);
    }
}

Frame::Frame(const std::vector<cv::Mat> &realFrame, const SimulationConfig &simulationConfig, const std::vector<Ellipsoid> &cells,
             const Path &outputPath, const std::string &imageName)
    : cells(cells),
      simulationConfig(simulationConfig),
      outputPath(outputPath),
      imageName(imageName),
      _realFrame(realFrame)
{
    if (!_realFrame.empty()) {
        Ellipsoid::cellConfig.maxZ = static_cast<float>(_realFrame.size()) - 1.0f;
    }
    // Calculate z_slices
    for (int i = 0; i < simulationConfig.z_slices; ++i)
    {
        double zValue = i;
        z_slices.push_back(zValue);
    }
    _synthFrame = generateSynthFrame();
    refreshFullCostCache();
}

Frame::Frame(const SimulationConfig &simulationConfig,
             const std::vector<Ellipsoid> &cells,
             const Path &outputPath, const std::string &imageName)
    : cells(cells),
      simulationConfig(simulationConfig),
      outputPath(outputPath),
      imageName(imageName)
{
    // Lazy-load placeholder. `_realFrame` and `_synthFrame` are empty;
    // call `loadImageStacks()` with the preprocessed stack before use.
    // z_slices still derived from config so size-queries are valid.
    for (int i = 0; i < simulationConfig.z_slices; ++i)
    {
        z_slices.push_back(static_cast<double>(i));
    }
}

void Frame::loadImageStacks(const std::vector<cv::Mat> &realFrame)
{
    _realFrame = realFrame;
    if (!_realFrame.empty()) {
        Ellipsoid::cellConfig.maxZ = static_cast<float>(_realFrame.size()) - 1.0f;
        simulationConfig.z_slices = static_cast<int>(_realFrame.size());
    }
    // Refresh z_slices in case the stack depth differs from config
    // (shouldn't, but guard against drift).
    if (static_cast<int>(realFrame.size()) != static_cast<int>(z_slices.size())) {
        z_slices.clear();
        for (size_t i = 0; i < realFrame.size(); ++i) {
            z_slices.push_back(static_cast<double>(i));
        }
    }
    if (!_backgroundFrame.empty() &&
        _backgroundFrame.size() != _realFrame.size()) {
        _backgroundFrame.clear();
        _backgroundFrameOffset = 0.0f;
        _backgroundFrameOffsetUpdates.clear();
    }
    _synthFrame = generateSynthFrame();
    refreshFullCostCache();
}

void Frame::setBackgroundFrame(std::vector<cv::Mat> backgroundFrame)
{
    _backgroundFrameOffset = 0.0f;
    _backgroundFrameOffsetUpdates.clear();
    if (!backgroundFrame.empty() && !_realFrame.empty() &&
        backgroundFrame.size() != _realFrame.size()) {
        std::cout << "[Background Frame] ignored size_mismatch expected="
                  << _realFrame.size()
                  << " got=" << backgroundFrame.size()
                  << std::endl;
        _backgroundFrame.clear();
        return;
    }
    for (cv::Mat &slice : backgroundFrame) {
        if (slice.empty()) {
            continue;
        }
        if (slice.type() != CV_32F) {
            slice.convertTo(slice, CV_32F);
        }
    }
    _backgroundFrame = std::move(backgroundFrame);
}

float Frame::backgroundAt(int z, int y, int x) const
{
    if (z < 0 || static_cast<size_t>(z) >= _backgroundFrame.size()) {
        return _backgroundValue;
    }
    const cv::Mat &background = _backgroundFrame[static_cast<size_t>(z)];
    if (background.empty() || background.type() != CV_32F ||
        y < 0 || y >= background.rows ||
        x < 0 || x >= background.cols) {
        return _backgroundValue;
    }
    const float value = background.ptr<float>(y)[x];
    return std::isfinite(value) ? value : _backgroundValue;
}

float Frame::backgroundAt(const cv::Point3f &point) const
{
    if (!std::isfinite(point.x) ||
        !std::isfinite(point.y) ||
        !std::isfinite(point.z)) {
        return _backgroundValue;
    }
    return backgroundAt(
        cvRound(point.z),
        cvRound(point.y),
        cvRound(point.x));
}

void Frame::addBackgroundOffset(float backgroundDelta)
{
    if (!std::isfinite(backgroundDelta) ||
        std::abs(backgroundDelta) <= 1e-9f) {
        return;
    }

    _backgroundFrameOffset += backgroundDelta;
    _backgroundFrameOffsetUpdates.push_back(backgroundDelta);
    _backgroundValue = std::clamp(_backgroundValue + backgroundDelta, 0.0f, 1.0f);
    for (cv::Mat &background : _backgroundFrame) {
        if (background.empty() || background.type() != CV_32F) {
            continue;
        }
        for (int y = 0; y < background.rows; ++y) {
            float *row = background.ptr<float>(y);
            for (int x = 0; x < background.cols; ++x) {
                row[x] = std::clamp(row[x] + backgroundDelta, 0.0f, 1.0f);
            }
        }
    }
}

cv::Mat Frame::makeSynthBackgroundSlice(cv::Size shape, int sliceIndex) const
{
    if (sliceIndex >= 0 &&
        static_cast<size_t>(sliceIndex) < _backgroundFrame.size()) {
        const cv::Mat &background =
            _backgroundFrame[static_cast<size_t>(sliceIndex)];
        if (!background.empty() &&
            background.size() == shape &&
            background.type() == CV_32F) {
            return background.clone();
        }
    }
    return cv::Mat(shape, CV_32F, cv::Scalar(_backgroundValue));
}

void Frame::refreshFullCostCache()
{
    if (_realFrame.size() != _synthFrame.size())
    {
        throw std::runtime_error("Mismatch in image stack sizes");
    }

    _currentCostPerSlice.assign(_realFrame.size(), 0.0);
    const float asymK = simulationConfig.asymmetric_cost_weight;
    const int nSlices = static_cast<int>(_realFrame.size());
    double totalCost = 0.0;
    #pragma omp parallel for reduction(+:totalCost) schedule(static)
    for (int i = 0; i < nSlices; ++i)
    {
        const double sliceCost = asymmetricL2Slice(_realFrame[static_cast<size_t>(i)], _synthFrame[static_cast<size_t>(i)], asymK);
        _currentCostPerSlice[static_cast<size_t>(i)] = sliceCost;
        totalCost += sliceCost;
    }
    _currentCost = totalCost;
}

double Frame::calculateIncrementalCost(const std::vector<cv::Mat> &newSynthFrame,
                                       int affectedZMin, int affectedZMax,
                                       std::vector<double> &outNewPerSlice) const
{
    if (_realFrame.size() != newSynthFrame.size())
    {
        throw std::runtime_error("Mismatch in image stack sizes");
    }
    if (_currentCostPerSlice.size() != _realFrame.size())
    {
        throw std::runtime_error("Per-slice cost cache not initialized");
    }

    outNewPerSlice = _currentCostPerSlice;

    if (affectedZMin >= 0 && affectedZMax >= 0)
    {
        const int nSlices = static_cast<int>(_realFrame.size());
        const int zMin = std::max(0, affectedZMin);
        const int zMax = std::min(nSlices - 1, affectedZMax);
        const int affectedCount = zMax - zMin + 1;
        const float asymK = simulationConfig.asymmetric_cost_weight;
        forEachSliceIndex(simulationConfig, affectedCount, [&](int localIndex) {
            const int i = zMin + localIndex;
            outNewPerSlice[static_cast<size_t>(i)] =
                asymmetricL2Slice(_realFrame[static_cast<size_t>(i)],
                                  newSynthFrame[static_cast<size_t>(i)],
                                  asymK);
        });
    }

    // Always sum in slice-index order so the result is bit-identical to
    // what calculateCost(newSynthFrame) would return (same operands, same
    // summation order).
    double totalCost = 0.0;
    for (size_t i = 0; i < outNewPerSlice.size(); ++i)
    {
        totalCost += outNewPerSlice[i];
    }
    return totalCost;
}

std::vector<cv::Mat> Frame::generateSynthFrame()
{
    cv::Size shape = getImageShape();
    std::vector<cv::Mat> frame(z_slices.size());

    // Pre-compute per-cell rotation matrix + inv-radii-squared once per
    // render pass (A4 optimization; mirrors generateSynthFrameFast).
    const size_t nCells = cells.size();
    std::vector<std::array<double, 9>> cellRotations(nCells);
    std::vector<std::array<double, 3>> cellInvR2(nCells);
    std::vector<float> cellMaxR(nCells);
    std::vector<float> cellZ(nCells);
    for (size_t c = 0; c < nCells; ++c) {
        cells[c].generateInverseRotationMatrix(cellRotations[c]);
        const double aR = cells[c].getARadius();
        const double bR = cells[c].getBRadius();
        const double cR = cells[c].getCRadius();
        cellInvR2[c][0] = 1.0 / (aR * aR);
        cellInvR2[c][1] = 1.0 / (bR * bR);
        cellInvR2[c][2] = 1.0 / (cR * cR);
        cellMaxR[c] = static_cast<float>(std::max({aR, bR, cR}));
        cellZ[c] = cells[c].getZ();
    }

    forEachSliceIndex(simulationConfig, static_cast<int>(z_slices.size()), [&](int i) {
        const double z = z_slices[static_cast<size_t>(i)];
        Image synthImage = makeSynthBackgroundSlice(
            shape, static_cast<int>(i));
        const float zf = static_cast<float>(z);
        for (size_t c = 0; c < nCells; ++c)
        {
            if (std::abs(zf - cellZ[c]) > cellMaxR[c]) continue;
            cells[c].drawWithRotation(synthImage, simulationConfig,
                                      cellRotations[c],
                                      cellInvR2[c][0], cellInvR2[c][1], cellInvR2[c][2],
                                      zf);
        }
        frame[static_cast<size_t>(i)] = synthImage;
    });
    return frame;
}

cv::Size Frame::getImageShape()
{
    if (_realFrame.empty())
    {
        throw std::runtime_error("Real image stack is empty");
    }
    return _realFrame[0].size(); // Returns the size of the first image in the stack
}

Cost Frame::calculateCost(const std::vector<cv::Mat> &synthFrame)
{
    if (_realFrame.size() != synthFrame.size())
    {
        throw std::runtime_error("Mismatch in image stack sizes");
    }

    const float asymK = simulationConfig.asymmetric_cost_weight;
    const int nSlices = static_cast<int>(_realFrame.size());
    double totalCost = 0.0;
    #pragma omp parallel for reduction(+:totalCost) schedule(static)
    for (int i = 0; i < nSlices; ++i)
    {
        totalCost += asymmetricL2Slice(_realFrame[i], synthFrame[i], asymK);
    }
    return totalCost;
}

// =============================================================================
// Bounding-box cost infrastructure
// =============================================================================
//
// Per-cell bbox cost is computed as asymmetric-L2 over voxels inside a 3D
// box around the cell, with voxels claimed by other cells (Voronoi) excluded.
// This concentrates the cost signal on the cell's own territory and makes
// split / perturbation decisions independent of unrelated image regions.

BoundingBox3D Frame::computeCellBbox(size_t cellIdx, float marginScale) const
{
    BoundingBox3D bbox;
    if (cellIdx >= cells.size() || _realFrame.empty()) return bbox;
    const Ellipsoid &cell = cells[cellIdx];
    const float maxR = std::max({cell.getARadius(), cell.getBRadius(), cell.getCRadius()});
    const float r = marginScale * maxR;
    const int cols = _realFrame[0].cols;
    const int rows = _realFrame[0].rows;
    const int slices = static_cast<int>(_realFrame.size());
    bbox.xMin = std::max(0,        static_cast<int>(std::floor(cell.getX() - r)));
    bbox.xMax = std::min(cols - 1, static_cast<int>(std::ceil (cell.getX() + r)));
    bbox.yMin = std::max(0,        static_cast<int>(std::floor(cell.getY() - r)));
    bbox.yMax = std::min(rows - 1, static_cast<int>(std::ceil (cell.getY() + r)));
    bbox.zMin = std::max(0,          static_cast<int>(std::floor(cell.getZ() - r)));
    bbox.zMax = std::min(slices - 1, static_cast<int>(std::ceil (cell.getZ() + r)));
    return bbox;
}

BoundingBox3D Frame::computeBboxAtPoint(const cv::Point3f &center,
                                         float radius,
                                         float marginScale) const
{
    BoundingBox3D bbox;
    if (_realFrame.empty() || radius <= 0.0f) return bbox;
    // Floor on absolute half-extent: small cells (R=10, margin=2.5 → 25 px)
    // get at least 40 px of bbox regardless of their radius, ensuring the
    // position anchor has enough voxels to function.
    constexpr float kMinBboxHalfExtent = 40.0f;
    const float r = std::max(kMinBboxHalfExtent, marginScale * radius);
    const int cols = _realFrame[0].cols;
    const int rows = _realFrame[0].rows;
    const int slices = static_cast<int>(_realFrame.size());
    bbox.xMin = std::max(0,        static_cast<int>(std::floor(center.x - r)));
    bbox.xMax = std::min(cols - 1, static_cast<int>(std::ceil (center.x + r)));
    bbox.yMin = std::max(0,        static_cast<int>(std::floor(center.y - r)));
    bbox.yMax = std::min(rows - 1, static_cast<int>(std::ceil (center.y + r)));
    bbox.zMin = std::max(0,          static_cast<int>(std::floor(center.z - r)));
    bbox.zMax = std::min(slices - 1, static_cast<int>(std::ceil (center.z + r)));
    return bbox;
}

BoundingBox3D Frame::computeUnionBbox(const std::vector<size_t> &cellIndices,
                                       float marginScale) const
{
    BoundingBox3D result;
    bool first = true;
    for (size_t idx : cellIndices) {
        BoundingBox3D b = computeCellBbox(idx, marginScale);
        if (!b.isValid()) continue;
        if (first) { result = b; first = false; continue; }
        result.xMin = std::min(result.xMin, b.xMin);
        result.xMax = std::max(result.xMax, b.xMax);
        result.yMin = std::min(result.yMin, b.yMin);
        result.yMax = std::max(result.yMax, b.yMax);
        result.zMin = std::min(result.zMin, b.zMin);
        result.zMax = std::max(result.zMax, b.zMax);
    }
    return result;
}

BoundingBox3D Frame::computeUnionBboxWithPoints(
    const std::vector<size_t> &cellIndices,
    float marginScale,
    const std::vector<cv::Point3f> &extraPoints,
    float pointRadius) const
{
    BoundingBox3D result = computeUnionBbox(cellIndices, marginScale);
    if (_realFrame.empty()) return result;
    const int cols = _realFrame[0].cols;
    const int rows = _realFrame[0].rows;
    const int slices = static_cast<int>(_realFrame.size());
    bool first = !result.isValid();
    for (const auto &p : extraPoints) {
        const int px0 = std::max(0,        static_cast<int>(std::floor(p.x - pointRadius)));
        const int px1 = std::min(cols - 1, static_cast<int>(std::ceil (p.x + pointRadius)));
        const int py0 = std::max(0,        static_cast<int>(std::floor(p.y - pointRadius)));
        const int py1 = std::min(rows - 1, static_cast<int>(std::ceil (p.y + pointRadius)));
        const int pz0 = std::max(0,          static_cast<int>(std::floor(p.z - pointRadius)));
        const int pz1 = std::min(slices - 1, static_cast<int>(std::ceil (p.z + pointRadius)));
        if (px0 > px1 || py0 > py1 || pz0 > pz1) continue;
        if (first) {
            result.xMin = px0; result.xMax = px1;
            result.yMin = py0; result.yMax = py1;
            result.zMin = pz0; result.zMax = pz1;
            first = false;
            continue;
        }
        result.xMin = std::min(result.xMin, px0);
        result.xMax = std::max(result.xMax, px1);
        result.yMin = std::min(result.yMin, py0);
        result.yMax = std::max(result.yMax, py1);
        result.zMin = std::min(result.zMin, pz0);
        result.zMax = std::max(result.zMax, pz1);
    }
    return result;
}

std::size_t Frame::computeVoronoiBleedVoxels(const Ellipsoid &cell,
                                             int cellIdx) const
{
    // No map or gate off → zero cost, no penalty.
    if (!_voronoiEnabled || _voronoiMap.empty() || cellIdx < 0 ||
        _realFrame.empty()) {
        return 0;
    }
    const int Z = static_cast<int>(_realFrame.size());
    if (static_cast<int>(_voronoiMap.size()) != Z) return 0;
    const int W = _realFrame[0].cols;
    const int H = _realFrame[0].rows;

    // Bounding box around the cell (conservative: use max radius so we
    // never miss a voxel inside the ellipsoid). Per-voxel inside-test
    // uses Ellipsoid::isPointInsideEllipsoid so the count is exact for
    // the rotated triaxial shape.
    const float maxR = std::max({cell.getARadius(),
                                  cell.getBRadius(),
                                  cell.getCRadius()});
    const int xMin = std::max(0,     static_cast<int>(std::floor(cell.getX() - maxR)));
    const int xMax = std::min(W - 1, static_cast<int>(std::ceil (cell.getX() + maxR)));
    const int yMin = std::max(0,     static_cast<int>(std::floor(cell.getY() - maxR)));
    const int yMax = std::min(H - 1, static_cast<int>(std::ceil (cell.getY() + maxR)));
    const int zMin = std::max(0,     static_cast<int>(std::floor(cell.getZ() - maxR)));
    const int zMax = std::min(Z - 1, static_cast<int>(std::ceil (cell.getZ() + maxR)));

    long long bleed = 0;
    #pragma omp parallel for reduction(+:bleed) schedule(static)
    for (int z = zMin; z <= zMax; ++z) {
        const cv::Mat &vSlice = _voronoiMap[z];
        if (vSlice.empty() || vSlice.type() != CV_32S) continue;
        long long sliceBleed = 0;
        for (int y = yMin; y <= yMax; ++y) {
            const int *vorRow = vSlice.ptr<int>(y);
            for (int x = xMin; x <= xMax; ++x) {
                if (vorRow[x] == cellIdx) continue;  // own territory → no bleed
                // Only count voxels actually inside the ellipsoid. Most
                // pixels in the bbox are in own territory (cell sits near
                // its Voronoi centroid) so the early-skip above handles
                // the bulk of the work; the inside test runs only on
                // pixels already flagged as being in a neighbor's claim.
                if (cell.isPointInsideEllipsoid(
                        cv::Point3f(static_cast<float>(x),
                                    static_cast<float>(y),
                                    static_cast<float>(z)))) {
                    ++sliceBleed;
                }
            }
        }
        bleed += sliceBleed;
    }
    return static_cast<std::size_t>(bleed);
}

void Frame::rebuildVoronoiMap()
{
    if (!_voronoiEnabled || _realFrame.empty() || cells.empty()) {
        _voronoiMap.clear();
        _voronoiAnchors.clear();
        return;
    }
    const int Z = static_cast<int>(_realFrame.size());
    const int H = _realFrame[0].rows;
    const int W = _realFrame[0].cols;

    // Anchor = snap position when available, else the cell's live center.
    // Snap anchors are installed by CellUniverse at frame start from
    // previousSnapshots and do not move during the frame. Newborn daughters
    // (post-split-accept) have no snap, so they use their live position —
    // but rebuildVoronoiMap is called immediately after the split so the
    // daughters' live positions ARE their Voronoi anchors going forward.
    _voronoiAnchors.clear();
    _voronoiAnchors.reserve(cells.size());
    for (const auto &cell : cells) {
        const std::string &name = cell.getName();
        auto it = _snapPositions.find(name);
        if (it != _snapPositions.end()) {
            _voronoiAnchors.push_back(it->second);
        } else {
            _voronoiAnchors.push_back(
                cv::Point3f(cell.getX(), cell.getY(), cell.getZ()));
        }
    }

    _voronoiMap.resize(Z);
    for (int z = 0; z < Z; ++z) {
        if (_voronoiMap[z].empty() || _voronoiMap[z].rows != H
            || _voronoiMap[z].cols != W
            || _voronoiMap[z].type() != CV_32S) {
            _voronoiMap[z] = cv::Mat(H, W, CV_32S);
        }
    }

    const int nCells = static_cast<int>(_voronoiAnchors.size());
    // Snapshot to contiguous arrays so the inner loop reads from tight
    // memory (better cache behavior than cv::Point3f vector in hot loop).
    std::vector<float> ax(nCells), ay(nCells), az(nCells);
    for (int i = 0; i < nCells; ++i) {
        ax[i] = _voronoiAnchors[i].x;
        ay[i] = _voronoiAnchors[i].y;
        az[i] = _voronoiAnchors[i].z;
    }

    #pragma omp parallel for schedule(static)
    for (int z = 0; z < Z; ++z) {
        cv::Mat &slice = _voronoiMap[z];
        for (int y = 0; y < H; ++y) {
            int *row = slice.ptr<int>(y);
            for (int x = 0; x < W; ++x) {
                float bestD2 = std::numeric_limits<float>::infinity();
                int bestIdx = -1;
                for (int i = 0; i < nCells; ++i) {
                    const float dx = x - ax[i];
                    const float dy = y - ay[i];
                    const float dz = z - az[i];
                    const float d2 = dx*dx + dy*dy + dz*dz;
                    if (d2 < bestD2) { bestD2 = d2; bestIdx = i; }
                }
                row[x] = bestIdx;
            }
        }
    }
}

double Frame::calculateBboxCost(
    const BoundingBox3D &bbox,
    const std::vector<cv::Mat> &synthFrame,
    const std::vector<uint8_t> &mask,
    int voronoiCellIdx) const
{
    if (!bbox.isValid()) return 0.0;
    if (synthFrame.size() != _realFrame.size()) {
        throw std::runtime_error("bbox cost: synth/real stack size mismatch");
    }
    // Voronoi cost filter is active only when the caller passes a valid
    // cell index that matches the current anchor list — an out-of-range
    // index would otherwise silently skip every pixel and return 0.0,
    // masking the bug. Bounds check added 2026-04-21 together with the
    // split-attempt disable-guard.
    const bool useVoronoi = (_voronoiEnabled
                             && voronoiCellIdx >= 0
                             && !_voronoiMap.empty()
                             && static_cast<int>(_voronoiMap.size()) == static_cast<int>(_realFrame.size())
                             && voronoiCellIdx < static_cast<int>(_voronoiAnchors.size()));
    const float asymK = simulationConfig.asymmetric_cost_weight;
    // Threshold: only apply asymK when overshoot exceeds this value.
    // Below threshold, boundary rendering artifacts (d ≈ 0.01-0.05) are
    // penalized at 1x (symmetric). Above, genuine overshoot (cell covering
    // dark background, d ≈ 0.1+) gets the full asymK penalty.
    // This eliminates the double-boundary bias: two daughters have 2×
    // boundary artifacts but the same valley-coverage signal as one parent.
    // Without the threshold, asymK amplifies boundary artifacts 8x,
    // making two daughters structurally more expensive than one parent.
    const float asymThreshold = simulationConfig.asymmetric_cost_threshold;
    const int nx = bbox.nx();
    const int ny = bbox.ny();
    const bool useAsym = asymK > 1.0f + 1e-6f;
    // When mask is empty, ALL voxels in the bbox contribute to cost —
    // no Voronoi neighbor exclusion. This is intentional: during any
    // single perturbCell call, neighbors' synth is constant between old
    // and new → cancels out in the cost delta. Exclusion was actively
    // HARMFUL because the Voronoi boundary shifts with the perturbed
    // cell, hiding the cost of abandoning voxels at the snap position
    // and weakening the position anchor. Without exclusion, every voxel
    // in the snap-anchored bbox contributes to the delta → drift from
    // snap costs reliably.
    const bool useMask = !mask.empty();
    double totalCost = 0.0;

    // Two loop variants: with-mask (rare, includes the per-voxel mask check
    // that kills auto-vectorization) and no-mask (default in current code,
    // straight-line inner body that the compiler can SIMD-vectorize). The
    // useAsym flag is loop-invariant so the compiler hoists it; the asym
    // multiply itself is per-voxel data-dependent and stays in the loop.
    if (!useMask) {
        #pragma omp parallel for reduction(+:totalCost) schedule(static)
        for (int z = bbox.zMin; z <= bbox.zMax; ++z) {
            const cv::Mat &realSlice  = _realFrame[z];
            const cv::Mat &synthSlice = synthFrame[z];
            if (realSlice.type() != CV_32F || synthSlice.type() != CV_32F) continue;
            const int *vorRow = nullptr;  // overridden per-y when useVoronoi
            const cv::Mat *vorSlice = useVoronoi ? &_voronoiMap[z] : nullptr;
            double sliceCost = 0.0;
            for (int y = bbox.yMin; y <= bbox.yMax; ++y) {
                const float *rr = realSlice.ptr<float>(y);
                const float *ss = synthSlice.ptr<float>(y);
                if (useVoronoi) vorRow = vorSlice->ptr<int>(y);
                if (useAsym) {
                    for (int x = bbox.xMin; x <= bbox.xMax; ++x) {
                        if (useVoronoi && vorRow[x] != voronoiCellIdx) continue;
                        const float d = ss[x] - rr[x];
                        const float d2 = d * d;
                        const float mul = (d > asymThreshold) ? asymK : 1.0f;
                        sliceCost += d2 * mul;
                    }
                } else {
                    for (int x = bbox.xMin; x <= bbox.xMax; ++x) {
                        if (useVoronoi && vorRow[x] != voronoiCellIdx) continue;
                        const float d = ss[x] - rr[x];
                        sliceCost += d * d;
                    }
                }
            }
            totalCost += sliceCost;
        }
    } else {
        #pragma omp parallel for reduction(+:totalCost) schedule(static)
        for (int z = bbox.zMin; z <= bbox.zMax; ++z) {
            const cv::Mat &realSlice  = _realFrame[z];
            const cv::Mat &synthSlice = synthFrame[z];
            if (realSlice.type() != CV_32F || synthSlice.type() != CV_32F) continue;
            const int zOff = (z - bbox.zMin) * nx * ny;
            double sliceCost = 0.0;
            for (int y = bbox.yMin; y <= bbox.yMax; ++y) {
                const float *rr = realSlice.ptr<float>(y);
                const float *ss = synthSlice.ptr<float>(y);
                const int yOff = zOff + (y - bbox.yMin) * nx;
                for (int x = bbox.xMin; x <= bbox.xMax; ++x) {
                    if (!mask[yOff + (x - bbox.xMin)]) continue;
                    const float d = ss[x] - rr[x];
                    float d2 = d * d;
                    if (useAsym && d > asymThreshold) d2 *= asymK;
                    sliceCost += d2;
                }
            }
            totalCost += sliceCost;
        }
    }
    return totalCost;
}

std::vector<cv::Mat> Frame::generateSynthFrameFast(Ellipsoid &oldCell, Ellipsoid &newCell,
                                                   int *outAffectedZMin, int *outAffectedZMax)
{
    if (cells.empty())
    {
        std::cerr << "Cells are not set\n";
    }

    cv::Size shape = getImageShape(); // Assuming getImageShape() returns a cv::Size
    std::vector<cv::Mat> synthFrame = _synthFrame;

    // Calculate the smallest box that contains both the old and new cell
    MinBox minBox = oldCell.calculateMinimumBox(newCell);
    Corner &minCorner = minBox.first;
    Corner &maxCorner = minBox.second;

    // Track which slices were actually re-rendered so callers can drive
    // incremental cost updates without recomputing cv::norm on unchanged
    // slices. -1/-1 means nothing was re-rendered (move entirely outside
    // the cached z range).
    int affectedMin = -1;
    int affectedMax = -1;

    // Optimization (A4): pre-compute per-cell rotation matrix + inv radii.
    // These are constant across z-slices for a given render pass, so we
    // hoist them out of the inner z-loop — saves 6 trig calls per
    // (cell × z-slice) pair (~21k calls/frame at f20+).
    const size_t nCells = cells.size();
    std::vector<std::array<double, 9>> cellRotations(nCells);
    std::vector<std::array<double, 3>> cellInvR2(nCells);  // [invA2, invB2, invC2]
    std::vector<float> cellMaxR(nCells);
    std::vector<float> cellZ(nCells);
    for (size_t c = 0; c < nCells; ++c) {
        cells[c].generateInverseRotationMatrix(cellRotations[c]);
        const double aR = cells[c].getARadius();
        const double bR = cells[c].getBRadius();
        const double cR = cells[c].getCRadius();
        cellInvR2[c][0] = 1.0 / (aR * aR);
        cellInvR2[c][1] = 1.0 / (bR * bR);
        cellInvR2[c][2] = 1.0 / (cR * cR);
        cellMaxR[c] = static_cast<float>(std::max({aR, bR, cR}));
        cellZ[c] = cells[c].getZ();
    }

    // If this candidate touches a contiguous z region, rerender only
    // that range and keep the untouched synthetic slices from the cache.
    for (size_t i = 0; i < z_slices.size(); ++i)
    {
        const double z = z_slices[i];
        if (z < minCorner[2] || z > maxCorner[2]) {
            continue;
        }
        if (affectedMin < 0) affectedMin = static_cast<int>(i);
        affectedMax = static_cast<int>(i);
    }

    if (affectedMin >= 0 && affectedMax >= affectedMin)
    {
        const int affectedCount = affectedMax - affectedMin + 1;
        forEachSliceIndex(simulationConfig, affectedCount, [&](int localIndex) {
            const int sliceIndex = affectedMin + localIndex;
            const double z = z_slices[static_cast<size_t>(sliceIndex)];
            cv::Mat synthImage = makeSynthBackgroundSlice(shape, sliceIndex);

            for (size_t c = 0; c < nCells; ++c)
            {
                // Skip cells that can't contribute to this z-slice.
                // A cell at z=100 with maxR=25 only affects slices 75-125.
                // Without this check, ALL cells are drawn on every affected
                // slice — 80%+ of draw calls produce zero pixels.
                if (std::abs(static_cast<float>(z) - cellZ[c]) > cellMaxR[c]) continue;
                cells[c].drawWithRotation(synthImage, simulationConfig,
                                          cellRotations[c],
                                          cellInvR2[c][0], cellInvR2[c][1], cellInvR2[c][2],
                                          static_cast<float>(z));
            }

            synthFrame[static_cast<size_t>(sliceIndex)] = synthImage;
        });
    }

    if (outAffectedZMin) *outAffectedZMin = affectedMin;
    if (outAffectedZMax) *outAffectedZMax = affectedMax;

    return synthFrame;
}

std::vector<cv::Mat> Frame::generateOutputFrame()
{
    std::vector<cv::Mat> realFrameWithOutlines(_realFrame.size());

    forEachSliceIndex(simulationConfig, static_cast<int>(_realFrame.size()), [&](int i) {
        const cv::Mat &realImage = _realFrame[static_cast<size_t>(i)];
        const double z = z_slices[static_cast<size_t>(i)];
        const float outlineIntensity = std::min(1.0f, _backgroundValue * 1.6f);

        cv::Mat outputFrame = realImage.clone();

        // Draw outlines for each cell
        for (const auto &cell : cells)
        {
            cell.drawOutline(outputFrame, outlineIntensity, z);
        }

        // Convert to 8-bit image if necessary
        if (outputFrame.depth() != CV_8U)
        {
            outputFrame.convertTo(outputFrame, CV_8U, 255.0);
        }

        realFrameWithOutlines[static_cast<size_t>(i)] = outputFrame;
    });

    return realFrameWithOutlines;
}

std::vector<cv::Mat> Frame::generateOutputSynthFrame()
{
    std::vector<cv::Mat> outputSynthFrame(_synthFrame.size());

    forEachSliceIndex(simulationConfig, static_cast<int>(_synthFrame.size()), [&](int i) {
        const auto &synthImage = _synthFrame[static_cast<size_t>(i)];
        cv::Mat outputImage;
        if (synthImage.depth() != CV_8U)
        {
            // Convert to 8-bit image if necessary, scaling pixel values by 255
            synthImage.convertTo(outputImage, CV_8U, 255.0);
        }
        else
        {
            outputImage = synthImage.clone();
        }

        outputSynthFrame[static_cast<size_t>(i)] = outputImage;
    });

    return outputSynthFrame;
}

size_t Frame::length() const
{
    return cells.size();
}

CostCallbackPair Frame::perturbCell(size_t index, float overlapWeight,
                                    bool useSignalGuidance,
                                    float randomPerturbRadiusRatio,
                                    bool pcaRefitWellFilledMove,
                                    bool useSignalMapGuidance,
                                    const cv::Point3f *forcedPosition)
{
    if (index >= cells.size()) {
        return {0.0, [](bool) {}};
    }

    Ellipsoid oldCell = cells[index];
    PerturbDirections perturbDirections;

    // Brightness-proportional overlap weight: scale the penalty by
    // (cellBrightness / meanBrightness)² so dim cells get lower overlap
    // weight (their image cost is lower → overlap shouldn't dominate).
    // Prevents dim cells from being pushed out of position by overlap
    // with brighter neighbors. Disabled when _meanCellBrightness <= 0.
    float effectiveOverlapWeight = overlapWeight;
    if (_meanCellBrightness > 1e-6f) {
        const float bRatio = oldCell.getBrightness() / _meanCellBrightness;
        effectiveOverlapWeight = overlapWeight * bRatio * bRatio;
    }

    // O(n) overlap for just this cell before perturbation
    double oldOverlapCell = computeOverlapForCell(index, effectiveOverlapWeight);

    // Radius-proportional sigma: large cells take bigger position steps,
    // small cells take smaller steps. Scale = maxR / referenceR.
    // When referenceR <= 0, scaling is disabled (posScale = 1.0).
    const float refR = Ellipsoid::cellConfig.perturbSigmaReferenceRadius;
    float posScale = 1.0f;
    if (refR > 1e-3f) {
        const float maxR = std::max({oldCell.getARadius(),
                                     oldCell.getBRadius(),
                                     oldCell.getCRadius()});
        const float radiusRatio = std::max(0.0f, randomPerturbRadiusRatio);
        posScale = (maxR * radiusRatio) / refR;
    }
    const bool useForcedPosition = forcedPosition != nullptr;
    cv::Point3f signalMapDirection(0.0f, 0.0f, 0.0f);
    const bool hasSignalMapDirection =
        !useForcedPosition &&
        useSignalMapGuidance &&
        !useSignalGuidance &&
        simulationConfig.signal_map_enabled &&
        simulationConfig.signal_map_perturb_guidance_enabled &&
        !_signalMap.empty() &&
        findMaxSignalMapGradientInCell(
            _signalMap,
            oldCell,
            simulationConfig.signal_map_cell_radius_scale,
            simulationConfig.signal_map_min_gradient_norm,
            signalMapDirection);

    const float signalMapProbabilityBoost = hasSignalMapDirection
        ? simulationConfig.signal_map_direction_probability_boost
        : 0.0f;
    if (useForcedPosition) {
        cells[index] = oldCell;
        cv::Point3f candidate = *forcedPosition;
        if (!_realFrame.empty()) {
            const float maxX = static_cast<float>(_realFrame[0].cols - 1);
            const float maxY = static_cast<float>(_realFrame[0].rows - 1);
            const float maxZ = static_cast<float>(_realFrame.size() - 1);
            candidate.x = std::clamp(candidate.x, 0.0f, maxX);
            candidate.y = std::clamp(candidate.y, 0.0f, maxY);
            candidate.z = std::clamp(candidate.z, 0.0f, maxZ);
        }
        cells[index].setPosition(candidate.x, candidate.y, candidate.z);
    } else {
        cells[index] = cells[index].getPerturbedCell(&perturbDirections,
                                                     posScale,
                                                     signalMapDirection,
                                                     signalMapProbabilityBoost);
    }

    if (hasSignalMapDirection && !_realFrame.empty()) {
        const cv::Point3f oldPos(oldCell.getX(), oldCell.getY(), oldCell.getZ());
        cv::Point3f candidate(cells[index].getX(), cells[index].getY(), cells[index].getZ());
        const cv::Point3f randomDelta = candidate - oldPos;
        const float uphillDot = randomDelta.x * signalMapDirection.x +
                                randomDelta.y * signalMapDirection.y +
                                randomDelta.z * signalMapDirection.z;
        const float damping = std::clamp(
            simulationConfig.signal_map_opposing_move_damping, 0.0f, 1.0f);
        if (uphillDot < 0.0f && damping > 0.0f) {
            candidate -= signalMapDirection * (uphillDot * damping);
        }

        const float maxSigma = std::max({std::abs(Ellipsoid::cellConfig.x.sigma),
                                         std::abs(Ellipsoid::cellConfig.y.sigma),
                                         std::abs(Ellipsoid::cellConfig.z.sigma)});
        const float stepBudget = std::max(0.0f, maxSigma * posScale);
        const float guideStrength = std::max(0.0f, simulationConfig.signal_map_guide_strength);
        candidate += signalMapDirection * (guideStrength * stepBudget);

        const float maxX = static_cast<float>(_realFrame[0].cols - 1);
        const float maxY = static_cast<float>(_realFrame[0].rows - 1);
        const float maxZ = static_cast<float>(_realFrame.size() - 1);
        cells[index].setPosition(
            std::clamp(candidate.x, 0.0f, maxX),
            std::clamp(candidate.y, 0.0f, maxY),
            std::clamp(candidate.z, 0.0f, maxZ));
    }

    if (!useForcedPosition &&
        !useSignalGuidance &&
        Ellipsoid::cellConfig.randomPerturbBrightCoreGuidanceEnabled &&
        !_realFrame.empty()) {
        const cv::Point3f oldPos(oldCell.getX(), oldCell.getY(), oldCell.getZ());
        cv::Point3f brightCore;
        float brightCoreTrust = 0.0f;
        if (estimateBrightCoreCentroidInCell(
                _realFrame, oldCell, Ellipsoid::cellConfig, brightCore, brightCoreTrust)) {
            cv::Point3f candidate(cells[index].getX(), cells[index].getY(), cells[index].getZ());
            const cv::Point3f towardBrightCore = brightCore - oldPos;
            const float brightCoreDistance = static_cast<float>(cv::norm(towardBrightCore));
            if (brightCoreDistance > 1e-3f) {
                const cv::Point3f uphill = towardBrightCore * (1.0f / brightCoreDistance);
                const cv::Point3f randomDelta = candidate - oldPos;
                const float uphillDot = randomDelta.x * uphill.x +
                                        randomDelta.y * uphill.y +
                                        randomDelta.z * uphill.z;
                if (uphillDot < 0.0f) {
                    candidate -= uphill * (uphillDot * brightCoreTrust);
                }

                const float maxSigma = std::max({std::abs(Ellipsoid::cellConfig.x.sigma),
                                                 std::abs(Ellipsoid::cellConfig.y.sigma),
                                                 std::abs(Ellipsoid::cellConfig.z.sigma)});
                const float stepBudget = std::max(0.0f, maxSigma * posScale);
                const float guideStrength = std::max(
                    0.0f, Ellipsoid::cellConfig.randomPerturbBrightCoreGuideStrength);
                candidate += uphill * std::min(brightCoreDistance,
                                               guideStrength * stepBudget * brightCoreTrust);

                const float maxX = static_cast<float>(_realFrame[0].cols - 1);
                const float maxY = static_cast<float>(_realFrame[0].rows - 1);
                const float maxZ = static_cast<float>(_realFrame.size() - 1);
                cells[index].setPosition(
                    std::clamp(candidate.x, 0.0f, maxX),
                    std::clamp(candidate.y, 0.0f, maxY),
                    std::clamp(candidate.z, 0.0f, maxZ));
            }
        }
    }

    bool usedCellUniverse3WindowProbabilitySample = false;
    if (!useForcedPosition &&
        useSignalGuidance &&
        simulationConfig.celluniverse3_enabled &&
        simulationConfig.celluniverse3_window_guided_position_enabled &&
        !_signalProbability.empty() &&
        !_realFrame.empty()) {
        const cv::Point3f oldPos(oldCell.getX(), oldCell.getY(), oldCell.getZ());
        const float maxR = std::max({oldCell.getARadius(),
                                     oldCell.getBRadius(),
                                     oldCell.getCRadius(),
                                     1.0f});
        const float maxSigma = std::max({std::abs(Ellipsoid::cellConfig.x.sigma),
                                         std::abs(Ellipsoid::cellConfig.y.sigma),
                                         std::abs(Ellipsoid::cellConfig.z.sigma),
                                         1.0f});
        const float localRadius =
            std::max(maxSigma * std::max(1.0f, posScale),
                     maxR * std::max(0.0f, randomPerturbRadiusRatio)) *
            std::max(0.0f,
                     simulationConfig.celluniverse3_window_guided_sample_radius_scale);
        cv::Point3f sampledPosition;
        float sampledProbability = 0.0f;
        if (sampleSignalProbabilityCandidate(
                _signalProbability,
                oldPos,
                localRadius,
                simulationConfig.celluniverse3_window_guided_min_probability,
                sampledPosition,
                sampledProbability)) {
            const float maxX = static_cast<float>(_realFrame[0].cols - 1);
            const float maxY = static_cast<float>(_realFrame[0].rows - 1);
            const float maxZ = static_cast<float>(_realFrame.size() - 1);
            cells[index].setPosition(
                std::clamp(sampledPosition.x, 0.0f, maxX),
                std::clamp(sampledPosition.y, 0.0f, maxY),
                std::clamp(sampledPosition.z, 0.0f, maxZ));
            usedCellUniverse3WindowProbabilitySample = true;
        }
    }

    // Signal-guided perturbation: keep the random candidate rooted at the
    // original cell location, then locally bias it uphill in the precomputed
    // probability field. The field already encodes brightness and distance,
    // so nearby signal has more influence than a far bright center.
    if (!useForcedPosition &&
        !usedCellUniverse3WindowProbabilitySample &&
        useSignalGuidance && !_signalProbability.empty() && !_realFrame.empty()) {
        const cv::Point3f oldPos(oldCell.getX(), oldCell.getY(), oldCell.getZ());
        cv::Point3f candidate(cells[index].getX(), cells[index].getY(), cells[index].getZ());
        const cv::Point3f gradient = signalProbabilityGradientAt(_signalProbability, oldPos);
        const float gradientNorm = static_cast<float>(cv::norm(gradient));
        if (gradientNorm > 1e-6f) {
            const cv::Point3f uphill = gradient * (1.0f / gradientNorm);
            const float localTrust = sampleSignalProbability(
                _signalProbability, oldPos.x, oldPos.y, oldPos.z);
            const cv::Point3f randomDelta = candidate - oldPos;
            const float uphillDot = randomDelta.x * uphill.x +
                                    randomDelta.y * uphill.y +
                                    randomDelta.z * uphill.z;
            if (uphillDot < 0.0f) {
                candidate -= uphill * (uphillDot * localTrust);
            }

            const float maxSigma = std::max({Ellipsoid::cellConfig.x.sigma,
                                             Ellipsoid::cellConfig.y.sigma,
                                             Ellipsoid::cellConfig.z.sigma});
            candidate += uphill * (0.5f * std::max(0.0f, maxSigma) * localTrust);

            const float maxX = static_cast<float>(_realFrame[0].cols - 1);
            const float maxY = static_cast<float>(_realFrame[0].rows - 1);
            const float maxZ = static_cast<float>(_realFrame.size() - 1);
            cells[index].setPosition(
                std::clamp(candidate.x, 0.0f, maxX),
                std::clamp(candidate.y, 0.0f, maxY),
                std::clamp(candidate.z, 0.0f, maxZ));
        }
    }

    if (pcaRefitWellFilledMove &&
        !cells[index].isTrash() &&
        !_realFrame.empty() &&
        Ellipsoid::cellConfig.pcaShapeMaxIters > 0) {
        constexpr float kPcaRefitFillFraction = 0.30f;
        const float fillFraction = positiveVoxelFractionInsideCell(_realFrame, cells[index]);
        if (fillFraction >= kPcaRefitFillFraction) {
            ClaimSet otherClaims;
            for (size_t otherIdx = 0; otherIdx < cells.size(); ++otherIdx) {
                if (otherIdx == index) {
                    continue;
                }
                const Ellipsoid &other = cells[otherIdx];
                otherClaims[other.getName()].push_back(cv::Point3f(
                    other.getX(), other.getY(), other.getZ()));
            }

            std::ostringstream pcaLogSink;
            calibrateCellShapeViaPca(
                index,
                otherClaims,
                Ellipsoid::cellConfig.pcaShapeMaxIters,
                Ellipsoid::cellConfig.pcaShapeRadiusScale,
                Ellipsoid::cellConfig.pcaShapeMinPixels,
                Ellipsoid::cellConfig.pcaShapeMaskScale,
                Ellipsoid::cellConfig.pcaShapeConvergeRadius,
                Ellipsoid::cellConfig.pcaShapeConvergeAngleDeg,
                Ellipsoid::cellConfig.pcaShapeUpdatePosition,
                Ellipsoid::cellConfig.pcaShapeMaxPosShiftFraction,
                cells[index].getARadius(),
                cells[index].getBRadius(),
                cells[index].getCRadius(),
                &pcaLogSink);
        }
    }

    // Per-frame velocity cap: reject any perturbation that moves the cell
    // further than _maxPerturbDrift{XY,Z} from its snap position. Guards
    // against Monte Carlo random-walk drift that the soft position_prior
    // penalty fails to stop in time (observed bug: cell a510 drifted +18
    // then +23 in z over two consecutive frames, colliding with a
    // stationary neighbor at the image ceiling).
    {
        auto snapPosIt = _snapPositions.find(cells[index].getName());
        if (snapPosIt != _snapPositions.end()) {
            const cv::Point3f &snapPos = snapPosIt->second;
            const float dx = cells[index].getX() - snapPos.x;
            const float dy = cells[index].getY() - snapPos.y;
            const float dz = cells[index].getZ() - snapPos.z;
            const float driftXY = std::sqrt(dx * dx + dy * dy);
            const float driftZ  = std::abs(dz);
            const bool xyOver = (_maxPerturbDriftXY > 0.0f) && (driftXY > _maxPerturbDriftXY);
            const bool zOver  = (_maxPerturbDriftZ  > 0.0f) && (driftZ  > _maxPerturbDriftZ);
            if (xyOver || zOver) {
                cells[index] = oldCell;
                return {0.0, [](bool) {}};
            }
        }
    }

    // Min-radius hard clamp (2026-04-09): prevent cells from ratcheting down to
    // minimum radius bounds via unconstrained perturbation. The Ellipsoid ctor
    // silently clamps a/b/c radii to their configured minima,
    // so a decrease-biased perturbation sequence parks cells at the floor where
    // the L2 cost rewards the tiny footprint (see 12345...3400 at (10,5) in
    // run 074740 f22 — the degenerate crumpled ellipse visible in the f8
    // screenshot of run 101212 was the same failure mode on 12345...341).
    // Revert any proposal that would take either radius FROM above the floor
    // TO the floor; proposals that were already at the floor are still allowed
    // (those cells are already parked there and need a different recovery
    // path — see the deferred volume recovery work in config).
    {
        const float newAR = cells[index].getARadius();
        const float newBR = cells[index].getBRadius();
        const float newCR = cells[index].getCRadius();
        const float oldAR = oldCell.getARadius();
        const float oldBR = oldCell.getBRadius();
        const float oldCR = oldCell.getCRadius();
        const float newMinorR = cells[index].getMinorRadius();
        const float oldMinorR = oldCell.getMinorRadius();
        const float minAR = static_cast<float>(Ellipsoid::cellConfig.minARadius);
        const float minBR = static_cast<float>(
            Ellipsoid::cellConfig.maxBRadius > 0.0 ? Ellipsoid::cellConfig.minBRadius
                                                   : Ellipsoid::cellConfig.minARadius);
        const float minCR = static_cast<float>(Ellipsoid::cellConfig.minCRadius);
        const float minMinorR = std::min({minAR, minBR, minCR});
        constexpr float kClampEpsilon = 1e-3f;
        const bool hitAFloor = (newAR <= minAR + kClampEpsilon) &&
                               (oldAR >  minAR + kClampEpsilon);
        const bool hitBFloor = (newBR <= minBR + kClampEpsilon) &&
                               (oldBR >  minBR + kClampEpsilon);
        const bool hitCFloor = (newCR <= minCR + kClampEpsilon) &&
                               (oldCR >  minCR + kClampEpsilon);
        const bool hitMinorFloor = (newMinorR <= minMinorR + kClampEpsilon) &&
                                   (oldMinorR >  minMinorR + kClampEpsilon);
        if (hitAFloor || hitBFloor || hitCFloor || hitMinorFloor) {
            cells[index] = oldCell;
            return {0.0, [](bool) {}};
        }
    }

    // O(n) overlap for this cell after perturbation
    double newOverlapCell = computeOverlapForCell(index, effectiveOverlapWeight);

    // Render only the affected z-slice range; generateSynthFrameFast writes
    // [affectedMin, affectedMax] (inclusive) for us to drive incremental
    // cost. Unchanged slices alias _synthFrame[i] — same pixel buffer, so
    // the cached per-slice L2 for those slices is bit-exact.
    int affectedMin = -1;
    int affectedMax = -1;
    auto newSynthFrame = generateSynthFrameFast(oldCell, cells[index],
                                                &affectedMin, &affectedMax);

    double newImageCost = 0.0;
    double oldImageCost = 0.0;
    std::vector<double> newCostPerSlice;

    if (_useBboxCost) {
        // Bbox cost path: measure asymmetric L2 over a fixed 3D bbox, with
        // Voronoi exclusion of voxels claimed by any other cell.
        //
        // Option A — snap-anchored bbox: if this cell has a snap bbox
        // installed (CellUniverse::optimize populates once per frame from
        // PreviousFrameSnapshot.position + maxRadius), use it unchanged for
        // every perturbation of this cell during the frame. That way, voxels
        // at the snap position are always scored — if the cell drifts away
        // from snap, synth at the snap position is empty while real is
        // bright, producing an undershoot penalty that anchors the cell to
        // its real-cell location. Without the anchor, the bbox follows the
        // cell and the abandoned snap voxels drop out of scope entirely.
        //
        // Cells without a snap (frame 1, daughters just created by a split
        // this frame) fall back to the legacy live pre/post-union bbox.
        BoundingBox3D bboxUnion;
        const std::string &cellName = cells[index].getName();
        auto snapIt = _snapBboxes.find(cellName);
        const bool haveSnapBbox = (snapIt != _snapBboxes.end()
                                   && snapIt->second.isValid());
        if (haveSnapBbox) {
            bboxUnion = snapIt->second;
        } else {
            BoundingBox3D bboxPre = computeCellBbox(index, _bboxMarginScale);
            bboxUnion = bboxPre;
            // Old-cell bbox derived inline (cell is currently post-perturb).
            const float maxRold = std::max({oldCell.getARadius(),
                                            oldCell.getBRadius(),
                                            oldCell.getCRadius()});
            const float r = _bboxMarginScale * maxRold;
            const int cols = _realFrame[0].cols;
            const int rows = _realFrame[0].rows;
            const int slices = static_cast<int>(_realFrame.size());
            BoundingBox3D b;
            b.xMin = std::max(0,        static_cast<int>(std::floor(oldCell.getX() - r)));
            b.xMax = std::min(cols - 1, static_cast<int>(std::ceil (oldCell.getX() + r)));
            b.yMin = std::max(0,        static_cast<int>(std::floor(oldCell.getY() - r)));
            b.yMax = std::min(rows - 1, static_cast<int>(std::ceil (oldCell.getY() + r)));
            b.zMin = std::max(0,          static_cast<int>(std::floor(oldCell.getZ() - r)));
            b.zMax = std::min(slices - 1, static_cast<int>(std::ceil (oldCell.getZ() + r)));
            if (!bboxUnion.isValid()) {
                bboxUnion = b;
            } else {
                bboxUnion.xMin = std::min(bboxUnion.xMin, b.xMin);
                bboxUnion.xMax = std::max(bboxUnion.xMax, b.xMax);
                bboxUnion.yMin = std::min(bboxUnion.yMin, b.yMin);
                bboxUnion.yMax = std::max(bboxUnion.yMax, b.yMax);
                bboxUnion.zMin = std::min(bboxUnion.zMin, b.zMin);
                bboxUnion.zMax = std::max(bboxUnion.zMax, b.zMax);
            }
        }

        // No Voronoi exclusion mask for bbox cost. Neighbors' synth is
        // constant between old and new (only the perturbed cell changed)
        // → their contribution cancels in the cost delta. Building a
        // Voronoi mask was actively harmful: the mask shifts with the
        // cell's new position, hiding the cost of abandoning voxels at
        // the snap position (the anchor mechanism that prevents drift).
        // Without exclusion, every voxel in the snap-anchored bbox
        // contributes to the delta → drift from snap costs reliably.
        const std::vector<uint8_t> noMask;  // empty = all voxels count
        // Voronoi cell index = this cell's position in cells[]. When the
        // Voronoi cost is disabled (empty _voronoiMap) calculateBboxCost
        // ignores voronoiCellIdx and behaves identically to the legacy path.
        const int vorIdx = static_cast<int>(index);
        oldImageCost = calculateBboxCost(bboxUnion, _synthFrame, noMask, vorIdx);
        newImageCost = calculateBboxCost(bboxUnion, newSynthFrame, noMask, vorIdx);
    } else {
        // Legacy full-image path.
        newImageCost = calculateIncrementalCost(newSynthFrame,
                                                affectedMin, affectedMax,
                                                newCostPerSlice);
        oldImageCost = _currentCost;
    }

    // Position prior penalty (2026-04-18): quadratic penalty on distance
    // from snap beyond a free-motion threshold. Prevents the drift
    // escape-hatch where the snap bbox undershoot penalty saturates once
    // the cell fully exits its bbox, leaving a flat cost landscape
    // outside. With the prior, drift past threshold is quadratically
    // penalized, independent of image evidence.
    //
    // Formula:
    //   d = ||cell.pos - snap.pos||
    //   penalty = weight × max(0, d - threshold)²
    //
    // Evaluated for both old and new cell positions; only the delta
    // matters for the accept/reject decision.
    double oldPositionPrior = 0.0;
    double newPositionPrior = 0.0;
    if (_positionPriorWeight > 0.0f) {
        auto snapPosIt = _snapPositions.find(cells[index].getName());
        if (snapPosIt != _snapPositions.end()) {
            const cv::Point3f &snapPos = snapPosIt->second;
            auto priorOf = [&](const Ellipsoid &c) -> double {
                const float dx = c.getX() - snapPos.x;
                const float dy = c.getY() - snapPos.y;
                const float dz = c.getZ() - snapPos.z;
                const float d = std::sqrt(dx * dx + dy * dy + dz * dz);
                const float over = std::max(0.0f, d - _positionPriorThreshold);
                return static_cast<double>(_positionPriorWeight) *
                       static_cast<double>(over) * static_cast<double>(over);
            };
            oldPositionPrior = priorOf(oldCell);
            newPositionPrior = priorOf(cells[index]);
        }
    }

    // Additive Voronoi bleed penalty: cost grows with the count of
    // voxels inside this cell's ellipsoid that sit in another cell's
    // snap-anchored Voronoi territory. Prevents bloat-induced neighbor-
    // particle capture without distorting the cell's image-cost gradient
    // (that's why it is additive, not a replacement mask on the L2 sum).
    // Evaluated only when weight > 0 AND the Voronoi map is live
    // (computeVoronoiBleedVoxels returns 0 otherwise, making the term a
    // no-op and preserving legacy behavior).
    double oldBleedPenalty = 0.0;
    double newBleedPenalty = 0.0;
    if (_voronoiEnabled && _voronoiBleedWeight > 0.0f) {
        const int vorIdx = static_cast<int>(index);
        oldBleedPenalty = static_cast<double>(_voronoiBleedWeight) *
                          static_cast<double>(
                              computeVoronoiBleedVoxels(oldCell, vorIdx));
        newBleedPenalty = static_cast<double>(_voronoiBleedWeight) *
                          static_cast<double>(
                              computeVoronoiBleedVoxels(cells[index], vorIdx));
    }

    double costDiff = (newImageCost + newOverlapCell + newPositionPrior + newBleedPenalty)
                    - (oldImageCost + oldOverlapCell + oldPositionPrior + oldBleedPenalty);

    const bool useBboxLocal = _useBboxCost;
    CallBackFunc callback = [this,
                             newSynthFrame = std::move(newSynthFrame),
                             newCostPerSlice = std::move(newCostPerSlice),
                             oldCell, index, newImageCost, perturbDirections,
                             useBboxLocal](bool accept) mutable
    {
        const float brightnessStep = std::max(0.0f, Ellipsoid::cellConfig.brightnessProbabilityStep);
        const float aRadiusStep = std::max(0.0f, Ellipsoid::cellConfig.aRadiusProbabilityStep);
        const float bRadiusStep = std::max(0.0f, Ellipsoid::cellConfig.bRadiusProbabilityStep);
        const float cRadiusStep = std::max(0.0f, Ellipsoid::cellConfig.cRadiusProbabilityStep);
        if (accept) {
            if (perturbDirections.brightness != 0) this->cells[index].adjustBrightnessPerturbProbability(perturbDirections.brightness, brightnessStep);
            if (perturbDirections.aRadius != 0) this->cells[index].adjustARadiusPerturbProbability(perturbDirections.aRadius, aRadiusStep);
            if (perturbDirections.bRadius != 0) this->cells[index].adjustBRadiusPerturbProbability(perturbDirections.bRadius, bRadiusStep);
            if (perturbDirections.cRadius != 0) this->cells[index].adjustCRadiusPerturbProbability(perturbDirections.cRadius, cRadiusStep);
            this->_synthFrame = std::move(newSynthFrame);
            if (!useBboxLocal) {
                this->_currentCost = newImageCost;
                this->_currentCostPerSlice = std::move(newCostPerSlice);
            }
            // Under bbox cost, _currentCost/_currentCostPerSlice are stale
            // and unused for decisions. They remain populated from the
            // initial refreshFullCostCache for diagnostic logging only.
        } else {
            Ellipsoid revertedCell = oldCell;
            if (perturbDirections.brightness != 0) revertedCell.adjustBrightnessPerturbProbability(perturbDirections.brightness, -brightnessStep);
            if (perturbDirections.aRadius != 0) revertedCell.adjustARadiusPerturbProbability(perturbDirections.aRadius, -aRadiusStep);
            if (perturbDirections.bRadius != 0) revertedCell.adjustBRadiusPerturbProbability(perturbDirections.bRadius, -bRadiusStep);
            if (perturbDirections.cRadius != 0) revertedCell.adjustCRadiusPerturbProbability(perturbDirections.cRadius, -cRadiusStep);
            this->cells[index] = revertedCell;
        }
    };
    return {costDiff, callback};
}

namespace {
// Bounding-sphere radius for a triaxial ellipsoid. Conservative overlap
// detection: treats the cell as the smallest enclosing sphere so no pair of
// touching cells is missed. Trades some over-penalization of non-overlapping
// but elongated cells for correctness. Runtime cost is one std::max.
inline float boundingSphereRadius(const Ellipsoid &cell)
{
    return std::max({cell.getARadius(), cell.getBRadius(), cell.getCRadius()});
}
}

// Non-saturating overlap penalty (barrier form). Diverges as cells approach
// full coincidence, so any finite image-cost gain cannot overcome stacking.
//   penalty = weight × ratio² / (1 − ratio + epsilon)
// Properties:
//   ratio=0.0  → 0                (no overlap, no penalty)
//   ratio=0.3  → ~1.4× old        (barely stronger for light overlap)
//   ratio=0.5  → ~2× old
//   ratio=0.85 → ~5.6× old        (serious overlap, strong pushback)
//   ratio=0.95 → ~17× old
//   ratio=0.99 → ~100× old
//   ratio=1.0  → huge (bounded by 1/epsilon)    (full coincidence forbidden)
// EPS keeps numerical stability; choose small enough that ratio=0.95 barrier
// is already effective but not so small the near-coincidence penalty overflows.
static inline double nonSaturatingOverlap(float overlapRatio, float weight)
{
    constexpr float EPS = 0.01f;
    const float denom = std::max(EPS, 1.0f - overlapRatio + EPS);
    return static_cast<double>(weight) *
           static_cast<double>(overlapRatio) *
           static_cast<double>(overlapRatio) /
           static_cast<double>(denom);
}

double Frame::computeOverlapPenalty(float weight) const
{
    double totalPenalty = 0.0;
    for (size_t i = 0; i < cells.size(); ++i) {
        float ri = boundingSphereRadius(cells[i]);
        float xi = cells[i].getX();
        float yi = cells[i].getY();
        float zi = cells[i].getZ();
        for (size_t j = i + 1; j < cells.size(); ++j) {
            float rj = boundingSphereRadius(cells[j]);
            float dx = xi - cells[j].getX();
            float dy = yi - cells[j].getY();
            float dz = zi - cells[j].getZ();
            float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
            float combinedR = ri + rj;
            if (dist < combinedR) {
                float overlapRatio = (combinedR - dist) / combinedR;
                totalPenalty += nonSaturatingOverlap(overlapRatio, weight);
            }
        }
    }
    return totalPenalty;
}

double Frame::computeOverlapForCell(size_t cellIdx, float weight) const
{
    double penalty = 0.0;
    float ri = boundingSphereRadius(cells[cellIdx]);
    float xi = cells[cellIdx].getX();
    float yi = cells[cellIdx].getY();
    float zi = cells[cellIdx].getZ();
    for (size_t j = 0; j < cells.size(); ++j) {
        if (j == cellIdx) continue;
        float rj = boundingSphereRadius(cells[j]);
        float dx = xi - cells[j].getX();
        float dy = yi - cells[j].getY();
        float dz = zi - cells[j].getZ();
        float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
        float combinedR = ri + rj;
        if (dist < combinedR) {
            float overlapRatio = (combinedR - dist) / combinedR;
            penalty += nonSaturatingOverlap(overlapRatio, weight);
        }
    }
    return penalty;
}

bool Frame::findAnyCellBodyOverlap(bool includeTrash,
                                   float scale,
                                   std::string *firstName,
                                   std::string *secondName,
                                   float *firstInSecond,
                                   float *secondInFirst) const
{
    return findAnyCellBodyOverlapIn(_realFrame, cells, includeTrash, scale,
                                    firstName, secondName,
                                    firstInSecond, secondInFirst);
}


// ---------------------------------------------------------------------------
// Triaxial split pipeline — Phase A / Phase B helper (2026-04-11 redesign).
// ---------------------------------------------------------------------------

namespace {

// Diagnostic counts for the voxel filter pipeline.
struct GatherStats
{
    int boxVoxels = 0;          // voxels in the axis-aligned bounding box
    int inSphere = 0;           // voxels inside the spherical box
    int aboveBrightness = 0;    // voxels above the brightness cutoff
    int voronoiRejected = 0;    // rejected because a non-self claim point was closer
    int voronoiKept = 0;        // final kept (== returned.size())
};

// Gather bright pixels inside a 3D bounding box around `center`, keeping
// only those whose nearest claim point across ALL cells belongs to the
// splitting cell (`selfClaimPoints`). Returns the kept pixels plus their
// raw intensities above background.
std::vector<BrightPixel> gatherBrightPixelsVoronoi(
    const std::vector<cv::Mat> &realFrame,
    const Frame &backgroundProvider,
    const cv::Point3f &center,
    float radius,
    const std::vector<cv::Point3f> &selfClaimPoints,
    const Frame::ClaimSet &otherClaimSets,
    GatherStats *stats = nullptr)
{
    std::vector<BrightPixel> kept;
    if (realFrame.empty() || radius <= 0.0f || selfClaimPoints.empty()) {
        return kept;
    }

    const int rows = realFrame[0].rows;
    const int cols = realFrame[0].cols;
    const int slices = static_cast<int>(realFrame.size());

    const int minX = std::max(0, static_cast<int>(std::floor(center.x - radius)));
    const int maxX = std::min(cols - 1, static_cast<int>(std::ceil(center.x + radius)));
    const int minY = std::max(0, static_cast<int>(std::floor(center.y - radius)));
    const int maxY = std::min(rows - 1, static_cast<int>(std::ceil(center.y + radius)));
    const int minZ = std::max(0, static_cast<int>(std::floor(center.z - radius)));
    const int maxZ = std::min(slices - 1, static_cast<int>(std::ceil(center.z + radius)));

    const float radiusSq = radius * radius;

    if (stats) {
        stats->boxVoxels = std::max(0, (maxX - minX + 1)) *
                           std::max(0, (maxY - minY + 1)) *
                           std::max(0, (maxZ - minZ + 1));
    }

    const auto distSq = [](const cv::Point3f &a, const cv::Point3f &b) {
        const float dx = a.x - b.x;
        const float dy = a.y - b.y;
        const float dz = a.z - b.z;
        return dx * dx + dy * dy + dz * dz;
    };

    for (int z = minZ; z <= maxZ; ++z) {
        const cv::Mat &slice = realFrame[z];
        if (slice.type() != CV_32F || slice.empty()) continue;

        const float dz = static_cast<float>(z) - center.z;
        const float dzSq = dz * dz;

        for (int y = minY; y <= maxY; ++y) {
            const float *row = slice.ptr<float>(y);
            const float dy = static_cast<float>(y) - center.y;
            const float dySq = dy * dy;
            for (int x = minX; x <= maxX; ++x) {
                const float dx = static_cast<float>(x) - center.x;
                const float r2 = dx * dx + dySq + dzSq;
                if (r2 > radiusSq) continue;
                if (stats) ++stats->inSphere;

                const float v = row[x];
                const float backgroundValue =
                    backgroundProvider.backgroundAt(z, y, x);
                // Brightness cutoff: pixels above their local background plus
                // a small sensor-noise margin. With no spatial background
                // installed, backgroundAt() returns the legacy scalar value.
                const float brightnessCutoff =
                    std::max(0.05f, backgroundValue + 0.02f);
                if (v <= brightnessCutoff) continue;
                if (stats) ++stats->aboveBrightness;

                const cv::Point3f p{
                    static_cast<float>(x),
                    static_cast<float>(y),
                    static_cast<float>(z)};

                // Hard Voronoi exclusion: keep pixel only if self is the
                // nearest claim across all cells.
                float selfBest = std::numeric_limits<float>::infinity();
                for (const auto &sp : selfClaimPoints) {
                    const float d2 = distSq(p, sp);
                    if (d2 < selfBest) selfBest = d2;
                }

                float otherBest = std::numeric_limits<float>::infinity();
                for (const auto &kv : otherClaimSets) {
                    for (const auto &op : kv.second) {
                        const float d2 = distSq(p, op);
                        if (d2 < otherBest) otherBest = d2;
                    }
                }

                // Hard Voronoi exclusion: keep pixel only if self is the
                // nearest claim across ALL cells. Simple, proven in best22.
                bool keep = (otherBest >= selfBest);
                if (keep) {
                    kept.push_back({p, v - backgroundValue});
                } else if (stats) {
                    ++stats->voronoiRejected;
                }
            }
        }
    }

    if (stats) stats->voronoiKept = static_cast<int>(kept.size());
    return kept;
}

// PCA on weighted 3D points in the CELL'S LOCAL FRAME. Transforms world-
// space pixel positions into the cell's local coordinate system (inverse
// rotation + normalization by radii a, b, c) before computing PCA. This
// makes the analysis invariant to cell orientation and shape — the PCA sees
// brightness distribution relative to the cell, not absolute image geometry.
// The resulting eigenvector is rotated back to world space. D1/D2 centroids
// are returned in world space (computed from the original world positions).
//
// When cellCenter is null or radii are zero, falls back to world-frame PCA
// with isotropic normalization (divide all axes by maxR).
bool pca3DWithCentroids(
    const std::vector<BrightPixel> &points,
    cv::Point3f &eigvec1Out,
    cv::Point3f &d1Out,
    cv::Point3f &d2Out,
    const cv::Point3f *cellCenter = nullptr,
    const std::array<double, 9> *invRotation = nullptr,
    float radiusA = 0.0f, float radiusB = 0.0f, float radiusC = 0.0f)
{
    if (points.size() < 8) return false;

    const bool useLocalFrame = cellCenter && invRotation
        && radiusA > 1e-3f && radiusB > 1e-3f && radiusC > 1e-3f;

    // Weighted mean in world space.
    cv::Point3f mean{0.0f, 0.0f, 0.0f};
    double wsum = 0.0;
    for (const auto &bp : points) {
        mean.x += bp.pos.x * bp.weight;
        mean.y += bp.pos.y * bp.weight;
        mean.z += bp.pos.z * bp.weight;
        wsum += bp.weight;
    }
    if (wsum < 1e-6) return false;
    mean.x /= static_cast<float>(wsum);
    mean.y /= static_cast<float>(wsum);
    mean.z /= static_cast<float>(wsum);

    // Normalization factors.
    const double invA = useLocalFrame ? (1.0 / radiusA) : 1.0;
    const double invB = useLocalFrame ? (1.0 / radiusB) : 1.0;
    const double invC = useLocalFrame ? (1.0 / radiusC) : 1.0;
    const auto &R = invRotation; // R_T: inverse rotation (world→local)

    // Helper: transform a world-space displacement into the normalized local
    // frame. If not using local frame, just returns the displacement as-is.
    auto toLocal = [&](double wx, double wy, double wz,
                       double &lx, double &ly, double &lz) {
        if (useLocalFrame) {
            // Inverse rotation: world → local
            const auto &M = *R;
            const double rx = M[0] * wx + M[1] * wy + M[2] * wz;
            const double ry = M[3] * wx + M[4] * wy + M[5] * wz;
            const double rz = M[6] * wx + M[7] * wy + M[8] * wz;
            // Normalize by radii
            lx = rx * invA;
            ly = ry * invB;
            lz = rz * invC;
        } else {
            lx = wx; ly = wy; lz = wz;
        }
    };

    // Weighted covariance in local (or world) frame.
    double cxx = 0, cxy = 0, cxz = 0, cyy = 0, cyz = 0, czz = 0;
    for (const auto &bp : points) {
        double lx, ly, lz;
        toLocal(bp.pos.x - mean.x, bp.pos.y - mean.y, bp.pos.z - mean.z,
                lx, ly, lz);
        const double w = bp.weight;
        cxx += w * lx * lx;
        cxy += w * lx * ly;
        cxz += w * lx * lz;
        cyy += w * ly * ly;
        cyz += w * ly * lz;
        czz += w * lz * lz;
    }
    cxx /= wsum; cxy /= wsum; cxz /= wsum;
    cyy /= wsum; cyz /= wsum; czz /= wsum;

    cv::Matx33d cov(
        cxx, cxy, cxz,
        cxy, cyy, cyz,
        cxz, cyz, czz);

    cv::Matx33d eigvecs;
    cv::Vec3d eigvals;
    cv::eigen(cov, eigvals, eigvecs);
    // cv::eigen returns eigenvectors as rows of `eigvecs`, sorted by
    // descending eigenvalue. Row 0 is the principal direction.
    // This eigenvector is in LOCAL frame (or world if not using local).
    cv::Point3f ev1Local(
        static_cast<float>(eigvecs(0, 0)),
        static_cast<float>(eigvecs(0, 1)),
        static_cast<float>(eigvecs(0, 2)));

    // Rotate eigenvector back to world space if using local frame.
    // Forward rotation R maps local→world. R_T is R^T stored row-major,
    // so column i of R = (R_T[i], R_T[3+i], R_T[6+i]).
    // R * v = (col0*vx + col1*vy + col2*vz).
    cv::Point3f ev1World;
    if (useLocalFrame) {
        const auto &M = *R;
        // Un-normalize: scale eigenvector by radii to undo the 1/r normalization
        // before rotating back. This ensures the world-space direction reflects
        // actual spatial extent, not the normalized unit-sphere.
        const double sx = ev1Local.x * radiusA;
        const double sy = ev1Local.y * radiusB;
        const double sz = ev1Local.z * radiusC;
        // Forward rotation (local→world): columns of R = rows of R^T transposed
        // R * v = (R_T[0]*sx + R_T[1]*sy + R_T[2]*sz, ...)
        // Wait — R_T is R^T row-major. R = (R_T)^T. So R[i][j] = R_T[j][i].
        // R * v:  row i of R = column i of R_T = (R_T[i], R_T[i+3], R_T[i+6])? No.
        // R_T stored row-major: R_T[row*3+col]. R_T[row][col] = R^T[row][col] = R[col][row].
        // So R[i][j] = R_T[j*3+i].
        // R*v: (R*v)_i = sum_j R[i][j]*v_j = sum_j R_T[j*3+i]*v_j
        //             = R_T[0*3+i]*vx + R_T[1*3+i]*vy + R_T[2*3+i]*vz
        //             = R_T[i]*vx + R_T[3+i]*vy + R_T[6+i]*vz
        ev1World.x = static_cast<float>(M[0]*sx + M[3]*sy + M[6]*sz);
        ev1World.y = static_cast<float>(M[1]*sx + M[4]*sy + M[7]*sz);
        ev1World.z = static_cast<float>(M[2]*sx + M[5]*sy + M[8]*sz);
    } else {
        ev1World = ev1Local;
    }

    const double n = std::sqrt(ev1World.x * ev1World.x +
                               ev1World.y * ev1World.y +
                               ev1World.z * ev1World.z);
    if (n < 1e-9) return false;
    ev1World.x = static_cast<float>(ev1World.x / n);
    ev1World.y = static_cast<float>(ev1World.y / n);
    ev1World.z = static_cast<float>(ev1World.z / n);
    eigvec1Out = ev1World;

    // Project every pixel onto the WORLD-SPACE eigenvector, find the median,
    // partition into two groups, compute weighted centroid of each group.
    // Centroids are returned in world space.
    std::vector<float> projections;
    projections.reserve(points.size());
    for (const auto &bp : points) {
        const float px = bp.pos.x - mean.x;
        const float py = bp.pos.y - mean.y;
        const float pz = bp.pos.z - mean.z;
        projections.push_back(px * ev1World.x + py * ev1World.y + pz * ev1World.z);
    }

    std::vector<float> sortedProj = projections;
    std::nth_element(
        sortedProj.begin(),
        sortedProj.begin() + sortedProj.size() / 2,
        sortedProj.end());
    const float median = sortedProj[sortedProj.size() / 2];

    cv::Point3f sumLo{0, 0, 0}, sumHi{0, 0, 0};
    double wLo = 0.0, wHi = 0.0;
    for (size_t i = 0; i < points.size(); ++i) {
        const auto &bp = points[i];
        if (projections[i] < median) {
            sumLo.x += bp.pos.x * bp.weight;
            sumLo.y += bp.pos.y * bp.weight;
            sumLo.z += bp.pos.z * bp.weight;
            wLo += bp.weight;
        } else {
            sumHi.x += bp.pos.x * bp.weight;
            sumHi.y += bp.pos.y * bp.weight;
            sumHi.z += bp.pos.z * bp.weight;
            wHi += bp.weight;
        }
    }
    if (wLo < 1e-6 || wHi < 1e-6) return false;

    d1Out = cv::Point3f(
        sumLo.x / static_cast<float>(wLo),
        sumLo.y / static_cast<float>(wLo),
        sumLo.z / static_cast<float>(wLo));
    d2Out = cv::Point3f(
        sumHi.x / static_cast<float>(wHi),
        sumHi.y / static_cast<float>(wHi),
        sumHi.z / static_cast<float>(wHi));
    return true;
}

// Project bright pixels onto a given direction, split at the median, and
// return the weighted centroid of each half. This gives a data-driven
// midpoint and separation for any candidate split axis without relying on
// PCA to choose the direction.
bool centroidsAlongAxis(
    const std::vector<BrightPixel> &points,
    const cv::Point3f &axis,
    cv::Point3f &d1Out,
    cv::Point3f &d2Out)
{
    if (points.size() < 8) return false;

    // Weighted mean for centering.
    cv::Point3f mean{0, 0, 0};
    double wsum = 0.0;
    for (const auto &bp : points) {
        mean.x += bp.pos.x * bp.weight;
        mean.y += bp.pos.y * bp.weight;
        mean.z += bp.pos.z * bp.weight;
        wsum += bp.weight;
    }
    if (wsum < 1e-6) return false;
    mean.x /= static_cast<float>(wsum);
    mean.y /= static_cast<float>(wsum);
    mean.z /= static_cast<float>(wsum);

    // Project onto axis.
    std::vector<float> proj;
    proj.reserve(points.size());
    for (const auto &bp : points) {
        proj.push_back(
            (bp.pos.x - mean.x) * axis.x +
            (bp.pos.y - mean.y) * axis.y +
            (bp.pos.z - mean.z) * axis.z);
    }

    // Median split.
    std::vector<float> sorted = proj;
    std::nth_element(sorted.begin(), sorted.begin() + sorted.size() / 2, sorted.end());
    const float median = sorted[sorted.size() / 2];

    cv::Point3f sumLo{0, 0, 0}, sumHi{0, 0, 0};
    double wLo = 0.0, wHi = 0.0;
    for (size_t i = 0; i < points.size(); ++i) {
        const auto &bp = points[i];
        if (proj[i] < median) {
            sumLo += cv::Point3f(bp.pos.x * bp.weight, bp.pos.y * bp.weight, bp.pos.z * bp.weight);
            wLo += bp.weight;
        } else {
            sumHi += cv::Point3f(bp.pos.x * bp.weight, bp.pos.y * bp.weight, bp.pos.z * bp.weight);
            wHi += bp.weight;
        }
    }
    if (wLo < 1e-6 || wHi < 1e-6) return false;

    d1Out = sumLo * (1.0f / static_cast<float>(wLo));
    d2Out = sumHi * (1.0f / static_cast<float>(wHi));
    return true;
}

// Rotate a unit vector by `angleRad` around `axis` (Rodrigues' formula).
cv::Point3f rotateAroundAxis(const cv::Point3f &v, const cv::Point3f &axis, float angleRad)
{
    const float c = std::cos(angleRad);
    const float s = std::sin(angleRad);
    const float oneMinusC = 1.0f - c;
    const float ax = axis.x, ay = axis.y, az = axis.z;
    const float dot = v.x * ax + v.y * ay + v.z * az;
    cv::Point3f out;
    out.x = v.x * c + (ay * v.z - az * v.y) * s + ax * dot * oneMinusC;
    out.y = v.y * c + (az * v.x - ax * v.z) * s + ay * dot * oneMinusC;
    out.z = v.z * c + (ax * v.y - ay * v.x) * s + az * dot * oneMinusC;
    return out;
}

// Build an arbitrary orthonormal frame whose +z aligns with `primary`. Used
// for generating rotation-candidate axes perpendicular to the primary.
void orthonormalFrame(const cv::Point3f &primary, cv::Point3f &u, cv::Point3f &v)
{
    cv::Point3f seed = (std::abs(primary.x) < 0.9f)
        ? cv::Point3f(1, 0, 0) : cv::Point3f(0, 1, 0);
    u.x = primary.y * seed.z - primary.z * seed.y;
    u.y = primary.z * seed.x - primary.x * seed.z;
    u.z = primary.x * seed.y - primary.y * seed.x;
    const float un = std::sqrt(u.x * u.x + u.y * u.y + u.z * u.z);
    if (un > 1e-6f) { u.x /= un; u.y /= un; u.z /= un; }
    v.x = primary.y * u.z - primary.z * u.y;
    v.y = primary.z * u.x - primary.x * u.z;
    v.z = primary.x * u.y - primary.y * u.x;
}

// Bio check — generic sphere-like geometric sanity. Called before cost check.
// Returns true if the split is geometrically plausible, false to reject.
bool bioCheckDaughters(
    const Ellipsoid &daughter1,
    const Ellipsoid &daughter2,
    double refParentVolume,
    float refParentMaxRadius,
    const std::vector<Ellipsoid> &allCells,
    size_t d1Idx,
    size_t d2Idx,
    const ProbabilityConfig &probConfig,
    std::string &reasonOut,
    float existingCellBuriedScale = 1.0f,
    bool skipExistingCellBuriedCheck = false,
    bool skipNeighborBridgeCheck = false,
    bool ignoreTrashNeighbors = false,
    bool ignoreSuspiciousRodBlockers = false)
{
    const auto cellVolume = [](const Ellipsoid &c) {
        return static_cast<double>(c.getARadius()) *
               static_cast<double>(c.getBRadius()) *
               static_cast<double>(c.getCRadius());
    };
    const auto isSuspiciousRodBlocker = [&](const Ellipsoid &c) {
        const float maxR = std::max({c.getARadius(), c.getBRadius(), c.getCRadius()});
        const float minR = std::max(
            1e-3f,
            std::min({c.getARadius(), c.getBRadius(), c.getCRadius()}));
        return maxR / minR >= probConfig.split_overlap_suspicious_rod_shape_ratio;
    };

    const float d1R = std::max({daughter1.getARadius(),
                                 daughter1.getBRadius(),
                                 daughter1.getCRadius()});
    const float d2R = std::max({daughter2.getARadius(),
                                 daughter2.getBRadius(),
                                 daughter2.getCRadius()});

    // 1. Size ratio check (catches shrink-grow degenerate splits).
    if (d1R < 1e-3f || d2R < 1e-3f) {
        reasonOut = "degenerate_radii";
        return false;
    }
    const float sizeRatio = std::max(d1R, d2R) / std::min(d1R, d2R);
    if (sizeRatio > probConfig.bio_daughter_size_ratio_max) {
        reasonOut = "size_ratio_" + std::to_string(sizeRatio);
        return false;
    }

    // 2. Combined volume fraction. The reference volume is supplied by the
    // caller — typically computed from the snapshot (last-frame) parent radii,
    // not from the live parent, so Phase B shrinkage doesn't distort the
    // ratio.
    const double d1Vol = cellVolume(daughter1);
    const double d2Vol = cellVolume(daughter2);
    const double combinedVol = d1Vol + d2Vol;
    if (refParentVolume < 1e-6) {
        reasonOut = "parent_zero_volume";
        return false;
    }
    const double volFraction = combinedVol / refParentVolume;
    if (volFraction < probConfig.bio_combined_volume_min_fraction ||
        volFraction > probConfig.bio_combined_volume_max_fraction) {
        reasonOut = "volume_fraction_" + std::to_string(volFraction);
        return false;
    }

    // 2b. Single-daughter volume gate. For a real division each daughter is
    // ~0.5 * parent vol. If one daughter > bio_max_single_daughter_volume_fraction
    // of parent, it's inheriting the parent ("asymmetric mimic" pattern).
    const double maxDaughterFraction =
        std::max(d1Vol, d2Vol) / refParentVolume;
    if (maxDaughterFraction > probConfig.bio_max_single_daughter_volume_fraction) {
        reasonOut = "single_daughter_volume_" + std::to_string(maxDaughterFraction);
        return false;
    }

    // 3. Daughters not buried in each other or in any other cell.
    if (!skipExistingCellBuriedCheck) {
        for (size_t i = 0; i < allCells.size(); ++i) {
            if (i == d1Idx || i == d2Idx) continue;
            const Ellipsoid &other = allCells[i];
            if (ignoreTrashNeighbors && other.isTrash()) continue;
            if (ignoreSuspiciousRodBlockers && isSuspiciousRodBlocker(other)) continue;
            if (other.isPointInsideEllipsoid(cv::Point3f(
                    daughter1.getX(), daughter1.getY(), daughter1.getZ()),
                    existingCellBuriedScale)) {
                reasonOut = "d1_buried_in_" + other.getName();
                return false;
            }
            if (other.isPointInsideEllipsoid(cv::Point3f(
                    daughter2.getX(), daughter2.getY(), daughter2.getZ()),
                    existingCellBuriedScale)) {
                reasonOut = "d2_buried_in_" + other.getName();
                return false;
            }
        }
    }
    // Sibling buried check — treat the other daughter as "another cell".
    if (daughter2.isPointInsideEllipsoid(cv::Point3f(
            daughter1.getX(), daughter1.getY(), daughter1.getZ()), 1.0f)) {
        reasonOut = "d1_buried_in_sibling";
        return false;
    }
    if (daughter1.isPointInsideEllipsoid(cv::Point3f(
            daughter2.getX(), daughter2.getY(), daughter2.getZ()), 1.0f)) {
        reasonOut = "d2_buried_in_sibling";
        return false;
    }

    // 4. Neighbor-bridging check: reject if either daughter's center is
    // closer to an existing neighbor than to its sibling. This catches
    // false splits where the cell stretches to a neighbor's bright blob
    // and the bridge gate sees a valley between them — the "split" is
    // really the cell covering two separate cells, not dividing.
    const cv::Point3f d1Pos(daughter1.getX(), daughter1.getY(), daughter1.getZ());
    const cv::Point3f d2Pos(daughter2.getX(), daughter2.getY(), daughter2.getZ());
    const float siblingDist = static_cast<float>(cv::norm(d2Pos - d1Pos));
    const float minSeparationFraction =
        std::max(0.0f, probConfig.bio_min_daughter_separation_parent_fraction);
    if (minSeparationFraction > 0.0f && refParentMaxRadius > 1e-3f) {
        const float minSeparation = minSeparationFraction * refParentMaxRadius;
        if (siblingDist < minSeparation) {
            reasonOut = "daughter_separation_" + std::to_string(siblingDist) +
                        "_min_" + std::to_string(minSeparation);
            return false;
        }
    }

    if (!skipNeighborBridgeCheck) {
        for (size_t i = 0; i < allCells.size(); ++i) {
            if (i == d1Idx || i == d2Idx) continue;
            const Ellipsoid &other = allCells[i];
            if (ignoreTrashNeighbors && other.isTrash()) continue;
            if (ignoreSuspiciousRodBlockers && isSuspiciousRodBlocker(other)) continue;
            const cv::Point3f oPos(other.getX(), other.getY(), other.getZ());
            const float d1ToOther = static_cast<float>(cv::norm(oPos - d1Pos));
            const float d2ToOther = static_cast<float>(cv::norm(oPos - d2Pos));
            if (d1ToOther < siblingDist * 0.5f) {
                reasonOut = "d1_bridging_to_" + other.getName();
                return false;
            }
            if (d2ToOther < siblingDist * 0.5f) {
                reasonOut = "d2_bridging_to_" + other.getName();
                return false;
            }
        }
    }

    return true;
}

// Build a daughter Ellipsoid at position `center` with radii scaled from the
// given source radii by `volumeScale` and clamped to the config bounds.
// `srcMajor/srcB/srcMinor` are the reference dimensions the daughter should
// inherit from — typically the parent's last-frame snapshot values, so that
// Phase B perturbations that shrink the live parent do not also shrink the
// daughters below what the image actually supports.
Ellipsoid buildDaughter(
    const std::string &name,
    const cv::Point3f &center,
    const Ellipsoid &parent,
    float volumeScale,
    float srcMajor,
    float srcB,
    float srcMinor)
{
    const auto &cfg = Ellipsoid::cellConfig;
    const float dMajor = std::clamp(
        srcMajor * volumeScale,
        static_cast<float>(cfg.minARadius),
        static_cast<float>(cfg.maxARadius));
    const float dB = std::clamp(
        srcB * volumeScale,
        static_cast<float>(cfg.maxBRadius > 0.0 ? cfg.minBRadius : cfg.minARadius),
        static_cast<float>(cfg.maxBRadius > 0.0 ? cfg.maxBRadius : cfg.maxARadius));
    const float dMinor = std::clamp(
        srcMinor * volumeScale,
        static_cast<float>(cfg.minCRadius),
        static_cast<float>(cfg.maxCRadius));

    EllipsoidParams dp(
        name,
        center.x, center.y, center.z,
        dMajor, dMinor,
        parent.getCellParams().theta_x,
        parent.getCellParams().theta_y,
        parent.getCellParams().theta_z,
        parent.getBrightness());
    dp.bRadius = dB;
    return Ellipsoid(dp);
}

} // namespace

// Frame-start pre-pass: PCA-ground the expected daughter positions.
// See plan 2026-04-10-triaxial-pipeline-redesign.md "PRE-PASS" block.
// Shares the gather + PCA helpers with trySplitCellPhased above (both
// defined in the anonymous namespace of this TU) — this is the "cheap
// PCA-only path, no candidate burn-in" version.
bool Frame::imageGroundExpectedDaughters(
    size_t cellIndex,
    const PreviousFrameSnapshot &snapshot,
    const ClaimSet &otherCellsClaimSets,
    cv::Point3f &outD1,
    cv::Point3f &outD2,
    int *outKeptPixels) const
{
    if (outKeptPixels) *outKeptPixels = 0;
    if (cellIndex >= cells.size()) return false;
    if (!snapshot.valid) return false;

    // Self claim points: same two-seed pattern as trySplitCellPhased uses.
    //   D1_seed = snapshot.center - 0.5 * splitAxisLength * splitAxisDir
    //   D2_seed = snapshot.center + 0.5 * splitAxisLength * splitAxisDir
    std::vector<cv::Point3f> selfClaim;
    if (snapshot.splitAxisLength > 1e-3f) {
        const float half = 0.5f * snapshot.splitAxisLength;
        selfClaim.push_back(cv::Point3f(
            snapshot.position.x - half * snapshot.splitAxisDir.x,
            snapshot.position.y - half * snapshot.splitAxisDir.y,
            snapshot.position.z - half * snapshot.splitAxisDir.z));
        selfClaim.push_back(cv::Point3f(
            snapshot.position.x + half * snapshot.splitAxisDir.x,
            snapshot.position.y + half * snapshot.splitAxisDir.y,
            snapshot.position.z + half * snapshot.splitAxisDir.z));
    } else {
        selfClaim.push_back(snapshot.position);
    }

    // Bounding box radius: 3x the largest snapshot semi-axis, same as the
    // split path's box radius. Uses snapshot (not live) radii so the
    // pre-pass sees the same region of space regardless of how Phase A
    // has since moved the parent.
    const float srcMajor = snapshot.aRadius;
    const float srcB = (snapshot.bRadius > 1e-3f) ? snapshot.bRadius : snapshot.aRadius;
    const float srcMinor = snapshot.cRadius;
    const float boxRadius = 3.0f * std::max({srcMajor, srcB, srcMinor});

    GatherStats gstats;
    auto pixels = gatherBrightPixelsVoronoi(
        _realFrame,
        *this,
        snapshot.position,
        boxRadius,
        selfClaim,
        otherCellsClaimSets,
        &gstats);

    if (outKeptPixels) *outKeptPixels = gstats.voronoiKept;
    if (pixels.size() < 20) return false;

    // PCA in cell's local frame using snapshot rotation + radii.
    const Ellipsoid &cell = cells[cellIndex];
    std::array<double, 9> R_T;
    cell.generateInverseRotationMatrix(R_T);
    const cv::Point3f center(cell.getX(), cell.getY(), cell.getZ());

    cv::Point3f dirPca;
    if (!pca3DWithCentroids(pixels, dirPca, outD1, outD2,
                            &center, &R_T,
                            cell.getARadius(), cell.getBRadius(), cell.getCRadius()))
        return false;
    return true;
}

bool Frame::evaluateSplitSeedPairByPca(
    size_t cellIndex,
    const PreviousFrameSnapshot &snapshot,
    const ClaimSet &otherCellsClaimSets,
    const cv::Point3f &seedD1,
    const cv::Point3f &seedD2,
    float gatherRadiusScale,
    int minPixels,
    float minAxisAlignment,
    float maxSeedDistanceScale,
    cv::Point3f &outD1,
    cv::Point3f &outD2,
    float *outAxisAlignment,
    float *outMaxSeedDistance,
    int *outKeptPixels,
    float *outMeanWeight) const
{
    if (outAxisAlignment) *outAxisAlignment = 0.0f;
    if (outMaxSeedDistance) *outMaxSeedDistance = std::numeric_limits<float>::infinity();
    if (outKeptPixels) *outKeptPixels = 0;
    if (outMeanWeight) *outMeanWeight = 0.0f;
    if (cellIndex >= cells.size() || !snapshot.valid || _realFrame.empty()) {
        return false;
    }

    const cv::Point3f seedDelta = seedD2 - seedD1;
    const float seedSep = static_cast<float>(cv::norm(seedDelta));
    if (seedSep <= 1e-6f) return false;
    const cv::Point3f seedAxis = seedDelta * (1.0f / seedSep);

    const Ellipsoid &cell = cells[cellIndex];
    const float srcMajor = std::max({snapshot.aRadius, cell.getARadius(), 1.0f});
    const float srcB = std::max({snapshot.bRadius, cell.getBRadius(), 1.0f});
    const float srcMinor = std::max({snapshot.cRadius, cell.getCRadius(), 1.0f});
    const float parentMaxR = std::max({srcMajor, srcB, srcMinor, 1.0f});
    const cv::Point3f gatherCenter = 0.5f * (seedD1 + seedD2);
    const float gatherRadius = std::max(
        std::max(0.5f * seedSep + parentMaxR,
                 std::max(0.0f, gatherRadiusScale) * parentMaxR),
        parentMaxR);

    const std::vector<cv::Point3f> selfClaim{seedD1, seedD2};
    GatherStats gstats;
    const auto pixels = gatherBrightPixelsVoronoi(
        _realFrame,
        *this,
        gatherCenter,
        gatherRadius,
        selfClaim,
        otherCellsClaimSets,
        &gstats);
    if (outKeptPixels) *outKeptPixels = gstats.voronoiKept;
    if (outMeanWeight && !pixels.empty()) {
        float sum = 0.0f;
        for (const auto &bp : pixels) sum += bp.weight;
        *outMeanWeight = sum / static_cast<float>(pixels.size());
    }
    if (static_cast<int>(pixels.size()) < std::max(1, minPixels)) {
        return false;
    }

    std::array<double, 9> R_T;
    cell.generateInverseRotationMatrix(R_T);
    const cv::Point3f pcaCenter = snapshot.position;
    cv::Point3f pcaAxis;
    cv::Point3f pcaD1;
    cv::Point3f pcaD2;
    if (!pca3DWithCentroids(pixels,
                            pcaAxis,
                            pcaD1,
                            pcaD2,
                            &pcaCenter,
                            &R_T,
                            srcMajor,
                            srcB,
                            srcMinor)) {
        return false;
    }

    const float directDistance =
        static_cast<float>(cv::norm(pcaD1 - seedD1) + cv::norm(pcaD2 - seedD2));
    const float swappedDistance =
        static_cast<float>(cv::norm(pcaD2 - seedD1) + cv::norm(pcaD1 - seedD2));
    if (swappedDistance < directDistance) {
        std::swap(pcaD1, pcaD2);
    }

    const cv::Point3f pcaDelta = pcaD2 - pcaD1;
    const float pcaSep = static_cast<float>(cv::norm(pcaDelta));
    if (pcaSep <= 1e-6f) return false;
    const cv::Point3f pcaSeedOrderAxis = pcaDelta * (1.0f / pcaSep);
    const float axisAlignment =
        std::abs(seedAxis.x * pcaSeedOrderAxis.x +
                 seedAxis.y * pcaSeedOrderAxis.y +
                 seedAxis.z * pcaSeedOrderAxis.z);
    const float maxSeedDistance = std::max(
        static_cast<float>(cv::norm(pcaD1 - seedD1)),
        static_cast<float>(cv::norm(pcaD2 - seedD2)));

    if (outAxisAlignment) *outAxisAlignment = axisAlignment;
    if (outMaxSeedDistance) *outMaxSeedDistance = maxSeedDistance;

    if (axisAlignment < std::clamp(minAxisAlignment, 0.0f, 1.0f)) {
        return false;
    }
    if (maxSeedDistanceScale > 0.0f &&
        maxSeedDistance > maxSeedDistanceScale * parentMaxR) {
        return false;
    }

    outD1 = pcaD1;
    outD2 = pcaD2;
    return true;
}

// Position-only calibration via weighted bright-pixel centroid.
// The centroid is the analytic midpoint-between-daughters for a dividing
// cell (pixels split into two clusters contribute equally to the mean)
// and the true center for a non-dividing cell. This is the same quantity
// pca3DWithCentroids computes internally before the eigenvector split.
bool Frame::calibrateCellPositionViaCentroid(
    size_t cellIndex,
    const PreviousFrameSnapshot &snapshot,
    const ClaimSet &otherCellsClaimSets)
{
    if (cellIndex >= cells.size()) return false;
    if (!snapshot.valid) return false;

    // Self claim points — same two-seed pattern as the split path.
    std::vector<cv::Point3f> selfClaim;
    if (snapshot.splitAxisLength > 1e-3f) {
        const float half = 0.5f * snapshot.splitAxisLength;
        selfClaim.push_back(cv::Point3f(
            snapshot.position.x - half * snapshot.splitAxisDir.x,
            snapshot.position.y - half * snapshot.splitAxisDir.y,
            snapshot.position.z - half * snapshot.splitAxisDir.z));
        selfClaim.push_back(cv::Point3f(
            snapshot.position.x + half * snapshot.splitAxisDir.x,
            snapshot.position.y + half * snapshot.splitAxisDir.y,
            snapshot.position.z + half * snapshot.splitAxisDir.z));
    } else {
        selfClaim.push_back(snapshot.position);
    }

    const float srcMajor = snapshot.aRadius;
    const float srcB = (snapshot.bRadius > 1e-3f) ? snapshot.bRadius : snapshot.aRadius;
    const float srcMinor = snapshot.cRadius;
    const float boxRadius = 3.0f * std::max({srcMajor, srcB, srcMinor});

    GatherStats gstats;
    auto pixels = gatherBrightPixelsVoronoi(
        _realFrame,
        *this,
        snapshot.position,
        boxRadius,
        selfClaim,
        otherCellsClaimSets,
        &gstats);

    if (pixels.size() < 20) {
        std::cout << "  [Centroid Calibration] cell=" << cells[cellIndex].getName()
                  << " skipped (too_few_pixels=" << pixels.size() << ")" << std::endl;
        return false;
    }

    // Weighted mean of bright pixel positions — the analytic midpoint.
    double sumWx = 0.0, sumWy = 0.0, sumWz = 0.0, sumW = 0.0;
    for (const auto &bp : pixels) {
        sumWx += static_cast<double>(bp.pos.x) * bp.weight;
        sumWy += static_cast<double>(bp.pos.y) * bp.weight;
        sumWz += static_cast<double>(bp.pos.z) * bp.weight;
        sumW  += bp.weight;
    }
    if (sumW < 1e-6) {
        std::cout << "  [Centroid Calibration] cell=" << cells[cellIndex].getName()
                  << " skipped (sum_weight=0)" << std::endl;
        return false;
    }
    const cv::Point3f centroid(
        static_cast<float>(sumWx / sumW),
        static_cast<float>(sumWy / sumW),
        static_cast<float>(sumWz / sumW));

    const Ellipsoid savedCell = cells[cellIndex];
    const cv::Point3f currentPos(savedCell.getX(), savedCell.getY(), savedCell.getZ());
    const float moveDist = static_cast<float>(cv::norm(centroid - currentPos));

    // Skip the cost comparison if the centroid is essentially where the
    // cell already is — no point moving.
    if (moveDist < 0.5f) {
        std::cout << "  [Centroid Calibration] cell=" << savedCell.getName()
                  << " centroid=(" << centroid.x << "," << centroid.y << "," << centroid.z << ")"
                  << " currentPos=(" << currentPos.x << "," << currentPos.y << "," << currentPos.z << ")"
                  << " dist=" << moveDist << " kept=" << gstats.voronoiKept
                  << " result=no_move" << std::endl;
        return false;
    }

    // Build candidate at centroid position — radii/rotation/brightness
    // inherited from the current cell (only position changes).
    EllipsoidParams cp = savedCell.getCellParams();
    cp.x = centroid.x;
    cp.y = centroid.y;
    cp.z = centroid.z;
    Ellipsoid candidate(cp);

    const double savedCost = _currentCost;
    const std::vector<cv::Mat> savedSynth = _synthFrame;
    const std::vector<double> savedPerSlice = _currentCostPerSlice;

    // Install candidate and compute incremental cost
    cells[cellIndex] = candidate;
    int affMin = -1, affMax = -1;
    Ellipsoid savedMutable = savedCell;
    Ellipsoid candidateMutable = candidate;
    auto newSynth = generateSynthFrameFast(savedMutable, candidateMutable, &affMin, &affMax);
    std::vector<double> newPerSlice;
    const double newCost = calculateIncrementalCost(newSynth, affMin, affMax, newPerSlice);

    if (newCost < savedCost) {
        _synthFrame = newSynth;
        _currentCost = newCost;
        _currentCostPerSlice = newPerSlice;
        std::cout << "  [Centroid Calibration] cell=" << savedCell.getName()
                  << " ACCEPTED centroid=(" << centroid.x << "," << centroid.y << "," << centroid.z << ")"
                  << " currentPos=(" << currentPos.x << "," << currentPos.y << "," << currentPos.z << ")"
                  << " dist=" << moveDist
                  << " costDelta=" << (newCost - savedCost)
                  << " kept=" << gstats.voronoiKept
                  << std::endl;
        return true;
    }

    // Revert — keep current position (which may or may not be the raw
    // snapshot; we preserve whatever state was live at entry).
    cells[cellIndex] = savedCell;
    _synthFrame = savedSynth;
    _currentCost = savedCost;
    _currentCostPerSlice = savedPerSlice;
    std::cout << "  [Centroid Calibration] cell=" << savedCell.getName()
              << " REJECTED centroid=(" << centroid.x << "," << centroid.y << "," << centroid.z << ")"
              << " currentPos=(" << currentPos.x << "," << currentPos.y << "," << currentPos.z << ")"
              << " dist=" << moveDist
              << " costDelta=" << (newCost - savedCost)
              << " kept=" << gstats.voronoiKept
              << std::endl;
    return false;
}

// ----- PCA shape-from-image helpers -----
//
// Decompose a proper-rotation matrix R (world <- local columns) into Euler
// angles with convention R = Rz(tz) * Ry(ty) * Rx(tx). Matches
// Ellipsoid::generateInverseRotationMatrix. Handles gimbal lock.
static void rotationMatrixToEulerZYX(const cv::Matx33d &R,
                                     double &tx, double &ty, double &tz)
{
    const double s = std::clamp(-R(2, 0), -1.0, 1.0);
    ty = std::asin(s);
    const double c = std::cos(ty);
    if (std::abs(c) > 1e-6) {
        tx = std::atan2(R(2, 1), R(2, 2));
        tz = std::atan2(R(1, 0), R(0, 0));
    } else {
        tx = std::atan2(-R(1, 2), R(1, 1));
        tz = 0.0;
    }
}

static cv::Point3f normalizedOr(const cv::Point3f &v,
                                const cv::Point3f &fallback)
{
    const double n = cv::norm(v);
    if (n < 1e-9) return fallback;
    return cv::Point3f(static_cast<float>(v.x / n),
                       static_cast<float>(v.y / n),
                       static_cast<float>(v.z / n));
}

static bool estimateFlattenedPlaneRotation(
    const std::vector<BrightPixel> &pixels,
    const Ellipsoid &cell,
    float minPlaneRatio,
    float minLongAxisRatio,
    double &targetTx,
    double &targetTy,
    double &targetTz,
    float &planeRatio,
    float &longAxisRatio,
    cv::Point3f &planeNormal,
    cv::Point3f &planeLongAxis,
    std::string &mode)
{
    if (pixels.size() < 3) return false;

    double sx = 0.0, sy = 0.0, sz = 0.0, wsum = 0.0;
    for (const auto &bp : pixels) {
        const double w = static_cast<double>(bp.weight) * bp.weight;
        sx += bp.pos.x * w;
        sy += bp.pos.y * w;
        sz += bp.pos.z * w;
        wsum += w;
    }
    if (wsum < 1e-9) return false;

    const double mx = sx / wsum;
    const double my = sy / wsum;
    const double mz = sz / wsum;
    double cxx = 0.0, cxy = 0.0, cxz = 0.0;
    double cyy = 0.0, cyz = 0.0, czz = 0.0;
    for (const auto &bp : pixels) {
        const double w = static_cast<double>(bp.weight) * bp.weight;
        const double dx = bp.pos.x - mx;
        const double dy = bp.pos.y - my;
        const double dz = bp.pos.z - mz;
        cxx += w * dx * dx; cxy += w * dx * dy; cxz += w * dx * dz;
        cyy += w * dy * dy; cyz += w * dy * dz;
        czz += w * dz * dz;
    }
    cxx /= wsum; cxy /= wsum; cxz /= wsum;
    cyy /= wsum; cyz /= wsum; czz /= wsum;

    cv::Matx33d cov(cxx, cxy, cxz, cxy, cyy, cyz, cxz, cyz, czz);
    cv::Matx33d eigvecs;
    cv::Vec3d eigvals;
    cv::eigen(cov, eigvals, eigvecs);
    const double l0 = std::max(0.0, eigvals[0]);
    const double l1 = std::max(0.0, eigvals[1]);
    const double l2 = std::max(0.0, eigvals[2]);
    if (l1 <= 1e-9 || l2 <= 1e-9) return false;

    planeRatio = static_cast<float>(l1 / std::max(l2, 1e-9));
    longAxisRatio = static_cast<float>(l0 / std::max(l1, 1e-9));
    if (planeRatio < minPlaneRatio) return false;

    std::array<double, 9> R_T;
    cell.generateInverseRotationMatrix(R_T);
    cv::Point3f curAxis[3];
    for (int i = 0; i < 3; ++i) {
        const int base = 3 * i;
        curAxis[i] = normalizedOr(
            cv::Point3f(static_cast<float>(R_T[base]),
                        static_cast<float>(R_T[base + 1]),
                        static_cast<float>(R_T[base + 2])),
            cv::Point3f(i == 0 ? 1.0f : 0.0f,
                        i == 1 ? 1.0f : 0.0f,
                        i == 2 ? 1.0f : 0.0f));
    }

    cv::Point3f pcaAxis[3];
    for (int i = 0; i < 3; ++i) {
        pcaAxis[i] = normalizedOr(
            cv::Point3f(static_cast<float>(eigvecs(i, 0)),
                        static_cast<float>(eigvecs(i, 1)),
                        static_cast<float>(eigvecs(i, 2))),
            curAxis[i]);
    }

    cv::Point3f axisA;
    cv::Point3f axisB;
    cv::Point3f axisC = pcaAxis[2];
    if (axisC.dot(curAxis[2]) < 0.0f) axisC *= -1.0f;
    planeNormal = axisC;

    if (longAxisRatio >= minLongAxisRatio) {
        axisA = pcaAxis[0];
        if (axisA.dot(curAxis[0]) < 0.0f) axisA *= -1.0f;
        axisA = normalizedOr(axisA - axisC * axisA.dot(axisC), curAxis[0]);
        axisB = axisC.cross(axisA);
        mode = "plane_and_long_axis";
    } else {
        axisA = curAxis[0] - axisC * curAxis[0].dot(axisC);
        if (cv::norm(axisA) < 1e-6) {
            axisA = pcaAxis[0] - axisC * pcaAxis[0].dot(axisC);
        }
        axisA = normalizedOr(axisA, pcaAxis[0]);
        axisB = axisC.cross(axisA);
        mode = "plane_only";
    }
    axisB = normalizedOr(axisB, pcaAxis[1]);
    planeLongAxis = axisA;

    cv::Matx33d R(axisA.x, axisB.x, axisC.x,
                  axisA.y, axisB.y, axisC.y,
                  axisA.z, axisB.z, axisC.z);
    if (cv::determinant(R) < 0.0) {
        axisB *= -1.0f;
        R = cv::Matx33d(axisA.x, axisB.x, axisC.x,
                        axisA.y, axisB.y, axisC.y,
                        axisA.z, axisB.z, axisC.z);
    }

    rotationMatrixToEulerZYX(R, targetTx, targetTy, targetTz);
    return true;
}

static bool rotationFromConstrainedPlaneNormal(
    const Ellipsoid &cell,
    const cv::Point3f &normal,
    const cv::Point3f &preferredLongAxis,
    double &targetTx,
    double &targetTy,
    double &targetTz)
{
    const cv::Point3f axisC =
        normalizedOr(normal, cv::Point3f(0.0f, 0.0f, 1.0f));
    std::array<double, 9> R_T;
    cell.generateInverseRotationMatrix(R_T);
    const cv::Point3f curA =
        normalizedOr(cv::Point3f(static_cast<float>(R_T[0]),
                                 static_cast<float>(R_T[1]),
                                 static_cast<float>(R_T[2])),
                     cv::Point3f(1.0f, 0.0f, 0.0f));

    cv::Point3f axisA =
        preferredLongAxis - axisC * preferredLongAxis.dot(axisC);
    if (cv::norm(axisA) < 1e-6) {
        axisA = curA - axisC * curA.dot(axisC);
    }
    axisA = normalizedOr(axisA, curA);
    cv::Point3f axisB = normalizedOr(axisC.cross(axisA),
                                     cv::Point3f(0.0f, 1.0f, 0.0f));

    cv::Matx33d R(axisA.x, axisB.x, axisC.x,
                  axisA.y, axisB.y, axisC.y,
                  axisA.z, axisB.z, axisC.z);
    if (cv::determinant(R) < 0.0) {
        axisB *= -1.0f;
        R = cv::Matx33d(axisA.x, axisB.x, axisC.x,
                        axisA.y, axisB.y, axisC.y,
                        axisA.z, axisB.z, axisC.z);
    }

    rotationMatrixToEulerZYX(R, targetTx, targetTy, targetTz);
    return true;
}


static int shortestAxisSlotFor(const Ellipsoid &cell)
{
    int slot = 0;
    float value = cell.getARadius();
    if (cell.getBRadius() < value) {
        slot = 1;
        value = cell.getBRadius();
    }
    if (cell.getCRadius() < value) {
        slot = 2;
    }
    return slot;
}

static bool rotationAligningShortestAxisToDirection(
    const Ellipsoid &cell,
    const cv::Point3f &targetDirection,
    double &targetTx,
    double &targetTy,
    double &targetTz,
    int *shortSlotOut = nullptr,
    float *angleBeforeDegOut = nullptr)
{
    const cv::Point3f target = normalizedOr(targetDirection,
                                            cv::Point3f(0.0f, 0.0f, 1.0f));
    if (cv::norm(targetDirection) < 1e-6) {
        return false;
    }

    std::array<double, 9> R_T;
    cell.generateInverseRotationMatrix(R_T);
    cv::Point3f curAxis[3];
    for (int i = 0; i < 3; ++i) {
        const int base = 3 * i;
        curAxis[i] = normalizedOr(
            cv::Point3f(static_cast<float>(R_T[base]),
                        static_cast<float>(R_T[base + 1]),
                        static_cast<float>(R_T[base + 2])),
            cv::Point3f(i == 0 ? 1.0f : 0.0f,
                        i == 1 ? 1.0f : 0.0f,
                        i == 2 ? 1.0f : 0.0f));
    }

    const int shortSlot = shortestAxisSlotFor(cell);
    if (shortSlotOut) {
        *shortSlotOut = shortSlot;
    }
    if (angleBeforeDegOut) {
        const double dot = std::clamp(
            std::abs(static_cast<double>(curAxis[shortSlot].dot(target))),
            0.0,
            1.0);
        *angleBeforeDegOut = static_cast<float>(std::acos(dot) * 180.0 / M_PI);
    }

    const float radii[3] = {
        cell.getARadius(),
        cell.getBRadius(),
        cell.getCRadius()
    };
    int refSlot = -1;
    float refRadius = -1.0f;
    for (int i = 0; i < 3; ++i) {
        if (i == shortSlot) continue;
        if (radii[i] > refRadius) {
            refRadius = radii[i];
            refSlot = i;
        }
    }
    if (refSlot < 0) {
        return false;
    }
    const int remainingSlot = 3 - shortSlot - refSlot;

    cv::Point3f axis[3];
    axis[shortSlot] = target;
    axis[refSlot] = curAxis[refSlot] - target * curAxis[refSlot].dot(target);
    if (cv::norm(axis[refSlot]) < 1e-6) {
        cv::Point3f u, v;
        orthonormalFrame(target, u, v);
        axis[refSlot] = u;
    }
    axis[refSlot] = normalizedOr(axis[refSlot], cv::Point3f(1.0f, 0.0f, 0.0f));
    axis[remainingSlot] = normalizedOr(axis[shortSlot].cross(axis[refSlot]),
                                       cv::Point3f(0.0f, 1.0f, 0.0f));

    cv::Matx33d R(axis[0].x, axis[1].x, axis[2].x,
                  axis[0].y, axis[1].y, axis[2].y,
                  axis[0].z, axis[1].z, axis[2].z);
    if (cv::determinant(R) < 0.0) {
        axis[remainingSlot] *= -1.0f;
        R = cv::Matx33d(axis[0].x, axis[1].x, axis[2].x,
                        axis[0].y, axis[1].y, axis[2].y,
                        axis[0].z, axis[1].z, axis[2].z);
    }

    rotationMatrixToEulerZYX(R, targetTx, targetTy, targetTz);
    return true;
}


bool Frame::calibrateCellShapeViaPca(
    size_t cellIndex,
    const ClaimSet &otherCellsClaimSets,
    int maxIters,
    float radiusScale,
    int minPixels,
    float maskScale,
    float convergeRadius,
    float convergeAngleDeg,
    bool  updatePosition,
    float maxPosShiftFraction,
    float maskA,
    float maskB,
    float maskC,
    std::ostream *logSink)
{
    if (cellIndex >= cells.size()) return false;
    if (maxIters <= 0) return false;
    Ellipsoid &cell = cells[cellIndex];

    // Route per-iter logs to the optional sink (per-thread accumulator)
    // when provided, else to std::cout. Lets the parallelized caller emit
    // deterministic per-cell log blocks in cell-index order.
    std::ostream &log = logSink ? *logSink : std::cout;

    const float convergeAngleRad =
        convergeAngleDeg * static_cast<float>(M_PI) / 180.0f;

    // Fixed mask radii (birth radii, or fallback to live). Mask stays
    // constant across iterations so the pixel-collection scope does not
    // collapse as the fitted radii shrink — prevents the mask-feedback
    // collapse where a shrinking cell latches onto one emerging daughter.
    const float effMaskA = (maskA > 0.0f) ? maskA : cell.getARadius();
    const float effMaskB = (maskB > 0.0f) ? maskB : cell.getBRadius();
    const float effMaskC = (maskC > 0.0f) ? maskC : cell.getCRadius();
    const float maskMaxR = std::max({effMaskA, effMaskB, effMaskC});
    const float sphereR = maskScale * maskMaxR;
    const double invA2Fixed = 1.0 / (maskScale * effMaskA * maskScale * effMaskA);
    const double invB2Fixed = 1.0 / (maskScale * effMaskB * maskScale * effMaskB);
    const double invC2Fixed = 1.0 / (maskScale * effMaskC * maskScale * effMaskC);

    bool anyUpdate = false;

    // P4 — bright-pixel gather cache.
    //
    // gatherBrightPixelsVoronoi scans a 3D box of `sphereR` around `center`,
    // applies brightness cutoff, then Voronoi-filters against neighbor
    // claim points. The expensive step is the box scan + Voronoi test —
    // O(boxVolume * nNeighbors). Per shape-fit iteration the gather inputs
    // are mostly invariant: `_realFrame`, the installed background,
    // `sphereR`, and `otherCellsClaimSets` never change; only `center`
    // can change, and only when `updatePosition=true`. Cache the raw pixel
    // set keyed on `center`; re-gather only when the centroid moves.
    cv::Point3f cachedCenter{std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::quiet_NaN()};
    std::vector<BrightPixel> cachedRaw;
    int cachedHits = 0;
    int cachedMisses = 0;

    // Adaptive exponent: bright core-dominated cells get a lower exponent
    // (more halo-inclusive fit, counters core-only shrinkage); dim/uniform
    // cells keep the default exponent. Recomputed on each cache miss (same
    // scope as the raw gather — invariant across iterations otherwise).
    const bool   adaptiveExp = Ellipsoid::cellConfig.pcaShapeAdaptiveExponent;
    const float  expDim      = std::max(0.0f, Ellipsoid::cellConfig.pcaShapeWeightExponent);
    const float  expBright   = std::max(0.0f, Ellipsoid::cellConfig.pcaShapeWeightExponentBright);
    const float  coreT       = Ellipsoid::cellConfig.pcaShapeCoreBrightnessThreshold;
    const float  coreLo      = Ellipsoid::cellConfig.pcaShapeCoreFractionLow;
    const float  coreHi      = std::max(coreLo + 1e-6f,
                                        Ellipsoid::cellConfig.pcaShapeCoreFractionHigh);
    const float  radInflBright = std::max(1.0f, Ellipsoid::cellConfig.pcaShapeRadiusInflationBright);
    float cellWeightExponent = expDim;
    // Per-cell radius inflation multiplier. 1.0 for uniform/dim cells
    // (PCA variance formula is analytically exact for them); ramps up to
    // radInflBright for peaked/bright-core cells whose visible halo
    // extends beyond the 97% containment radius. Driven by the same
    // pCore metric as adaptive exponent.
    float cellRadiusInflation = 1.0f;

    for (int iter = 0; iter < maxIters; ++iter) {
        const cv::Point3f center(cell.getX(), cell.getY(), cell.getZ());
        const float curA = cell.getARadius();
        const float curB = cell.getBRadius();
        const float curC = cell.getCRadius();
        const float maxR = std::max({curA, curB, curC});
        if (maxR <= 1e-3f) break;

        // Cache hit when center hasn't moved since last gather. With
        // updatePosition=false (recommended), this hits on every iter
        // after the first → 14× speedup on the gather phase.
        const bool cacheHit = (cachedRaw.size() > 0)
            && (std::abs(center.x - cachedCenter.x) < 1e-4f)
            && (std::abs(center.y - cachedCenter.y) < 1e-4f)
            && (std::abs(center.z - cachedCenter.z) < 1e-4f);
        if (!cacheHit) {
            std::vector<cv::Point3f> selfClaim{center};
            GatherStats gstats;
            cachedRaw = gatherBrightPixelsVoronoi(
                _realFrame, *this, center, sphereR,
                selfClaim, otherCellsClaimSets, &gstats);
            cachedCenter = center;
            ++cachedMisses;

            // Adaptive exponent: measure core-dominance on the freshly
            // gathered raw pixels. Dim cells stay at expDim; bright cells
            // ramp down toward expBright to give halo a fairer vote in
            // the weighted moments.
            if (adaptiveExp && !cachedRaw.empty()) {
                // Relative pCore: fraction of pixels whose weight exceeds
                // 1.5× the cell's own mean weight. Scale-invariant — a
                // peaked cell (bright core, dim halo) has many pixels
                // above 1.5×mean; a uniform cell has few. Doesn't depend
                // on absolute brightness scale or preprocessing pipeline.
                //
                // Replaces the previous absolute-threshold metric
                // (pcaShapeCoreBrightnessThreshold) which failed for this
                // dataset: dim cells got HIGH pCore (many pixels barely
                // above threshold) while bright cells got LOW pCore
                // (pixels spread across a wide range). The relative
                // metric correctly classifies both.
                float meanW = 0.0f;
                for (const auto &bp : cachedRaw) meanW += bp.weight;
                meanW /= static_cast<float>(cachedRaw.size());

                int coreCount = 0;
                const float relativeThreshold = 1.5f * meanW;
                for (const auto &bp : cachedRaw) {
                    if (bp.weight > relativeThreshold) ++coreCount;
                }
                const float pCore = static_cast<float>(coreCount) /
                                    static_cast<float>(cachedRaw.size());
                const float t = std::clamp(
                    (pCore - coreLo) / (coreHi - coreLo), 0.0f, 1.0f);
                cellWeightExponent = expDim + t * (expBright - expDim);
                // Same pCore ramp drives radius inflation: 1.0 for uniform
                // (t=0), radInflBright for peaked (t=1).
                cellRadiusInflation = 1.0f + t * (radInflBright - 1.0f);
                log << "  [PCA Shape Exp] cell=" << cell.getName()
                    << " pCore=" << pCore
                    << " meanW=" << meanW
                    << " relThresh=" << relativeThreshold
                    << " exp=" << cellWeightExponent
                    << " radInfl=" << cellRadiusInflation
                    << std::endl;
            }
        } else {
            ++cachedHits;
        }
        const std::vector<BrightPixel> &raw = cachedRaw;

        // Ellipsoid mask uses FIXED radii (snap), not live, so it cannot
        // tighten between iterations — prevents mask-feedback collapse.
        // Rotation still uses the cell's CURRENT rotation so the mask
        // orientation follows the fit as it converges.
        std::array<double, 9> R_T;
        cell.generateInverseRotationMatrix(R_T);
        const double invA2 = invA2Fixed;
        const double invB2 = invB2Fixed;
        const double invC2 = invC2Fixed;

        std::vector<BrightPixel> pixels;
        pixels.reserve(raw.size());
        for (const auto &bp : raw) {
            const double dx = bp.pos.x - center.x;
            const double dy = bp.pos.y - center.y;
            const double dz = bp.pos.z - center.z;
            const double lx = R_T[0] * dx + R_T[1] * dy + R_T[2] * dz;
            const double ly = R_T[3] * dx + R_T[4] * dy + R_T[5] * dz;
            const double lz = R_T[6] * dx + R_T[7] * dy + R_T[8] * dz;
            const double val = lx * lx * invA2 + ly * ly * invB2 + lz * lz * invC2;
            if (val <= 1.0) pixels.push_back(bp);
        }

        if (static_cast<int>(pixels.size()) < minPixels) {
            log << "  [PCA Shape] cell=" << cell.getName()
                      << " iter=" << iter
                      << " stop_too_few pixels=" << pixels.size()
                      << " min=" << minPixels << std::endl;
            break;
        }

        // Intensity-weighted centroid + covariance with configurable exponent.
        //   exp=1.0: linear (historical behavior, halo can bloat radii).
        //   exp=2.0: quadratic (current default; suppresses halo ~25× vs core).
        //   exp=higher: stronger core emphasis.
        // Cached per-pixel weight is `weight_eff = pow(bp.weight, exp)`.
        // Fast paths for exp∈{1, 2} avoid the std::pow call.
        // `cellWeightExponent` is per-cell adaptive when enabled (set above
        // on cache miss), else equals `pcaShapeWeightExponent`.
        const float weightExponent = cellWeightExponent;
        const bool exp1 = std::abs(weightExponent - 1.0f) < 1e-6f;
        const bool exp2 = std::abs(weightExponent - 2.0f) < 1e-6f;

        auto effectiveWeight = [&](float w) -> double {
            if (exp1) return static_cast<double>(w);
            if (exp2) return static_cast<double>(w) * w;
            return std::pow(static_cast<double>(w), static_cast<double>(weightExponent));
        };

        // Cache per-pixel effective weights to avoid recomputing pow()
        // in the covariance loop below (saves N pow() calls where N is
        // typically 500-5000 pixels per PCA iteration).
        std::vector<double> pixelWeights(pixels.size());
        double sx = 0, sy = 0, sz = 0, wsum = 0;
        for (size_t pi = 0; pi < pixels.size(); ++pi) {
            const double we = effectiveWeight(pixels[pi].weight);
            pixelWeights[pi] = we;
            sx += pixels[pi].pos.x * we;
            sy += pixels[pi].pos.y * we;
            sz += pixels[pi].pos.z * we;
            wsum += we;
        }
        if (wsum < 1e-6) break;
        const double mx = sx / wsum, my = sy / wsum, mz = sz / wsum;

        double cxx = 0, cxy = 0, cxz = 0, cyy = 0, cyz = 0, czz = 0;
        for (size_t pi = 0; pi < pixels.size(); ++pi) {
            const double dx = pixels[pi].pos.x - mx;
            const double dy = pixels[pi].pos.y - my;
            const double dz = pixels[pi].pos.z - mz;
            const double we = pixelWeights[pi];
            cxx += we * dx * dx; cxy += we * dx * dy; cxz += we * dx * dz;
            cyy += we * dy * dy; cyz += we * dy * dz;
            czz += we * dz * dz;
        }
        cxx /= wsum; cxy /= wsum; cxz /= wsum;
        cyy /= wsum; cyz /= wsum; czz /= wsum;

        cv::Matx33d cov(cxx, cxy, cxz, cxy, cyy, cyz, cxz, cyz, czz);
        cv::Matx33d eigvecs;
        cv::Vec3d eigvals;
        cv::eigen(cov, eigvals, eigvecs);

        cv::Point3f pcaAxis[3];
        double pcaVariance[3];
        for (int i = 0; i < 3; ++i) {
            pcaAxis[i] = cv::Point3f(
                static_cast<float>(eigvecs(i, 0)),
                static_cast<float>(eigvecs(i, 1)),
                static_cast<float>(eigvecs(i, 2)));
            pcaVariance[i] = std::max(0.0, eigvals[i]);
        }

        // Eigenvalue degeneracy: cell is near-spherical. PCA rotation is
        // noisy. Skip rotation update and only refresh radii/position.
        const double maxVar = std::max({pcaVariance[0], pcaVariance[1], pcaVariance[2]});
        const double minVar = std::min({pcaVariance[0], pcaVariance[1], pcaVariance[2]});
        const bool degenerate = (minVar <= 1e-6) || (maxVar / minVar < 1.1);

        // Current cell axes (columns of R) in world frame.
        // R_T is R^T stored row-major (R_T[r*3+c] = R[c,r]); column i of R
        // = (R_T[3i], R_T[3i+1], R_T[3i+2]). See worldSplitAxis comment for
        // the 2026-04-19 indexing bug fix history.
        cv::Point3f curAxis[3];
        for (int i = 0; i < 3; ++i) {
            const int base = 3 * i;
            curAxis[i] = cv::Point3f(
                static_cast<float>(R_T[base]),
                static_cast<float>(R_T[base + 1]),
                static_cast<float>(R_T[base + 2]));
        }

        // Strict eigenvalue-rank assignment: a = largest variance, b = middle,
        // c = smallest. Greedy |dot| matching oscillated when eigenvalue
        // ordering differed from current slot labeling, because matched
        // variances cycled between slots each iteration. Rank assignment is
        // stable — physical variance ranks are invariant to the label rotation.
        cv::Point3f matchedAxis[3];
        double matchedVariance[3];
        double maxAxisAngle = 0.0;
        for (int ci = 0; ci < 3; ++ci) {
            cv::Point3f v = pcaAxis[ci];
            // Sign-align with current slot direction for rotation continuity.
            const double dot = curAxis[ci].x * v.x + curAxis[ci].y * v.y + curAxis[ci].z * v.z;
            if (dot < 0.0) { v.x = -v.x; v.y = -v.y; v.z = -v.z; }
            matchedAxis[ci] = v;
            matchedVariance[ci] = pcaVariance[ci];
            const double ang = std::acos(std::clamp(std::abs(dot), 0.0, 1.0));
            if (ang > maxAxisAngle) maxAxisAngle = ang;
        }

        double targetTx = cell.getThetaX();
        double targetTy = cell.getThetaY();
        double targetTz = cell.getThetaZ();
        if (!degenerate) {
            cv::Matx33d R(
                matchedAxis[0].x, matchedAxis[1].x, matchedAxis[2].x,
                matchedAxis[0].y, matchedAxis[1].y, matchedAxis[2].y,
                matchedAxis[0].z, matchedAxis[1].z, matchedAxis[2].z);
            if (cv::determinant(R) < 0.0) {
                R(0, 2) = -R(0, 2); R(1, 2) = -R(1, 2); R(2, 2) = -R(2, 2);
            }
            rotationMatrixToEulerZYX(R, targetTx, targetTy, targetTz);
        } else {
            maxAxisAngle = 0.0;
        }

        // Variance-based radii: radius = radiusScale × sqrt(weighted variance).
        //
        // The weighted variance along each axis is ALREADY computed by the
        // PCA covariance above (matchedVariance[i] = eigenvalue for axis i).
        // radiusScale = sqrt(5) ≈ 2.236 is the analytically correct
        // containment radius for a uniformly-filled ellipsoid.
        //
        // Why variance over percentile: the weight exponent (1.3) correctly
        // de-emphasizes halo for PCA rotation, but that same weighting
        // makes the weighted variance genuinely smaller along thin axes
        // (fewer bright pixels → lower weighted variance). This preserves
        // the elongation signal: e.g. 12345 at f2 gets c-axis variance
        // much smaller than a/b → elongation ~1.4 → correct split detection.
        //
        // Unweighted percentile lost this signal (halo extends equally on
        // all axes → cell looks round, elong=1.07). Weighted percentile
        // over-corrected (radii 40-50% too small, cells hit min floor).
        // Variance + radiusScale is the calibrated middle ground that the
        // best22 reference run (8/8 GT splits) validated.
        //
        // Per-cell adaptive inflation (cellRadiusInflation, from the pCore
        // ramp above) compensates for peaked cells where sqrt(5) × sqrt(var)
        // under-estimates the visible halo.
        //
        // Birth-based mask (introduced after the original variance formula)
        // prevents the mask-feedback loops that motivated the percentile
        // experiment. The mask is frozen at birth radii and never participates
        // in the fit → no compounding bloat.
        const float targetA = cellRadiusInflation * radiusScale *
                              std::sqrt(static_cast<float>(matchedVariance[0]));
        const float targetB = cellRadiusInflation * radiusScale *
                              std::sqrt(static_cast<float>(matchedVariance[1]));
        const float targetC = cellRadiusInflation * radiusScale *
                              std::sqrt(static_cast<float>(matchedVariance[2]));
        // No floor. Collapse prevented by birth-based mask (above).

        // Position update: centroid, capped.
        cv::Point3f newCenter = center;
        if (updatePosition) {
            const float dx = static_cast<float>(mx) - center.x;
            const float dy = static_cast<float>(my) - center.y;
            const float dz = static_cast<float>(mz) - center.z;
            const float shift = std::sqrt(dx * dx + dy * dy + dz * dz);
            const float cap = maxPosShiftFraction * maxR;
            if (shift > cap && shift > 1e-6f) {
                const float s = cap / shift;
                newCenter = cv::Point3f(center.x + s * dx,
                                        center.y + s * dy,
                                        center.z + s * dz);
            } else {
                newCenter = cv::Point3f(static_cast<float>(mx),
                                        static_cast<float>(my),
                                        static_cast<float>(mz));
            }
        }

        // Apply.
        cell.setRadii(targetA, targetB, targetC);
        cell.setRotation(static_cast<float>(targetTx),
                         static_cast<float>(targetTy),
                         static_cast<float>(targetTz));
        if (updatePosition) {
            EllipsoidParams p = cell.getCellParams();
            p.x = newCenter.x; p.y = newCenter.y; p.z = newCenter.z;
            cell = Ellipsoid(p);
        }
        anyUpdate = true;

        // Convergence check.
        const float dA = std::abs(targetA - curA);
        const float dB = std::abs(targetB - curB);
        const float dC = std::abs(targetC - curC);
        const float maxDR = std::max({dA, dB, dC});

        log << "  [PCA Shape] cell=" << cell.getName()
                  << " iter=" << iter
                  << " n=" << pixels.size()
                  << " degen=" << degenerate
                  << " R=(" << targetA << "," << targetB << "," << targetC << ")"
                  << " dR=" << maxDR
                  << " axisAng=" << (maxAxisAngle * 180.0 / M_PI)
                  << " posShift=" << cv::norm(newCenter - center)
                  << std::endl;

        if (maxDR < convergeRadius &&
            maxAxisAngle < convergeAngleRad) {
            break;
        }
    }

    if (cachedHits + cachedMisses > 0) {
        log << "  [PCA Shape Cache] cell=" << cell.getName()
                  << " hits=" << cachedHits
                  << " misses=" << cachedMisses
                  << " hitRate="
                  << (static_cast<float>(cachedHits) /
                      static_cast<float>(cachedHits + cachedMisses))
                  << std::endl;
    }

    return anyUpdate;
}

bool Frame::refineCellShapeViaEdgeSticks(size_t cellIndex,
                                         std::ostream *logSink)
{
    if (!simulationConfig.celluniverse3_enabled ||
        !simulationConfig.celluniverse3_edge_axis_fit_enabled ||
        cellIndex >= cells.size() ||
        _realFrame.empty()) {
        return false;
    }

    Ellipsoid &cell = cells[cellIndex];
    if (cell.isTrash()) {
        return false;
    }

    const int maxIters =
        std::max(0, simulationConfig.celluniverse3_edge_axis_fit_iterations);
    if (maxIters <= 0) {
        return false;
    }

    std::ostream &log = logSink ? *logSink : std::cout;
    bool anyChange = false;
    std::ostringstream detail;

    for (int iter = 0; iter < maxIters; ++iter) {
        const std::array<float, 3> oldR = {
            cell.getARadius(), cell.getBRadius(), cell.getCRadius()
        };
        const float minR = std::max(1e-3f, std::min({oldR[0], oldR[1], oldR[2]}));
        const float maxR = std::max({oldR[0], oldR[1], oldR[2]});
        const float shape = maxR / minR;
        int longestAxis = 0;
        if (oldR[1] > oldR[longestAxis]) longestAxis = 1;
        if (oldR[2] > oldR[longestAxis]) longestAxis = 2;
        if (shape < simulationConfig.celluniverse3_edge_axis_fit_min_shape) {
            break;
        }

        std::array<double, 9> R_T;
        cell.generateInverseRotationMatrix(R_T);
        std::array<cv::Point3f, 3> axis = {
            normalizedOr(cv::Point3f(static_cast<float>(R_T[0]),
                                     static_cast<float>(R_T[1]),
                                     static_cast<float>(R_T[2])),
                         cv::Point3f(1.0f, 0.0f, 0.0f)),
            normalizedOr(cv::Point3f(static_cast<float>(R_T[3]),
                                     static_cast<float>(R_T[4]),
                                     static_cast<float>(R_T[5])),
                         cv::Point3f(0.0f, 1.0f, 0.0f)),
            normalizedOr(cv::Point3f(static_cast<float>(R_T[6]),
                                     static_cast<float>(R_T[7]),
                                     static_cast<float>(R_T[8])),
                         cv::Point3f(0.0f, 0.0f, 1.0f))
        };

        std::array<float, 3> newR = oldR;
        std::array<int, 3> validHalfCounts = {0, 0, 0};
        std::array<float, 3> rawTargets = {oldR[0], oldR[1], oldR[2]};
        const cv::Point3f center(cell.getX(), cell.getY(), cell.getZ());
        const float band =
            simulationConfig.celluniverse3_edge_axis_fit_search_band_fraction;
        const int samples =
            std::max(3, simulationConfig.celluniverse3_edge_axis_fit_samples_per_half_axis);
        const float minGradient =
            simulationConfig.celluniverse3_edge_axis_fit_min_falling_gradient;
        const float minContrast =
            simulationConfig.celluniverse3_edge_axis_fit_min_inner_outer_contrast;
        const float stepFraction =
            simulationConfig.celluniverse3_edge_axis_fit_radius_step_fraction;
        const float maxChangeFraction =
            simulationConfig.celluniverse3_edge_axis_fit_max_radius_change_fraction;

        for (int ai = 0; ai < 3; ++ai) {
            float signedDeltaWeightSum = 0.0f;
            float edgeWeightSum = 0.0f;
            int targetCount = 0;
            std::array<float, 2> halfDelta = {0.0f, 0.0f};
            std::array<float, 2> halfScore = {0.0f, 0.0f};
            std::array<float, 2> halfEdge = {oldR[ai], oldR[ai]};
            std::array<bool, 2> halfValid = {false, false};
            cv::Point3f u, v;
            orthonormalFrame(axis[ai], u, v);
            const float other0 = oldR[(ai + 1) % 3];
            const float other1 = oldR[(ai + 2) % 3];
            const float cylR = std::max(
                simulationConfig.celluniverse3_edge_axis_fit_min_cylinder_radius,
                std::min(other0, other1) *
                    simulationConfig.celluniverse3_edge_axis_fit_cylinder_radius_scale);
            const std::array<cv::Point3f, 5> offsets = {
                cv::Point3f(0.0f, 0.0f, 0.0f),
                u * cylR,
                u * -cylR,
                v * cylR,
                v * -cylR
            };

            for (int sign = -1; sign <= 1; sign += 2) {
                const int halfIdx = (sign < 0) ? 0 : 1;
                const cv::Point3f dir = axis[ai] * static_cast<float>(sign);
                const float tMin = std::max(1.0f, oldR[ai] * (1.0f - band));
                const float tMax = std::max(tMin + 1.0f, oldR[ai] * (1.0f + band));
                float bestT = oldR[ai];
                float bestScore = 0.0f;
                float bestFalling = 0.0f;
                float bestContrast = 0.0f;

                for (int si = 0; si < samples; ++si) {
                    const float alpha = (samples == 1)
                        ? 0.0f
                        : static_cast<float>(si) / static_cast<float>(samples - 1);
                    const float t = tMin + (tMax - tMin) * alpha;
                    float fallingSum = 0.0f;
                    float contrastSum = 0.0f;
                    int support = 0;
                    for (const cv::Point3f &offset : offsets) {
                        const cv::Point3f p = center + dir * t + offset;
                        const cv::Point3f grad = signalProbabilityGradientAt(_realFrame, p);
                        const float falling = std::max(0.0f, -grad.dot(dir));
                        const float inner = sampleSignalProbability(
                            _realFrame, p.x - dir.x * 1.5f,
                            p.y - dir.y * 1.5f, p.z - dir.z * 1.5f);
                        const float outer = sampleSignalProbability(
                            _realFrame, p.x + dir.x * 1.5f,
                            p.y + dir.y * 1.5f, p.z + dir.z * 1.5f);
                        const float contrast = std::max(0.0f, inner - outer);
                        fallingSum += falling;
                        contrastSum += contrast;
                        ++support;
                    }
                    if (support <= 0) {
                        continue;
                    }
                    const float fallingMean =
                        fallingSum / static_cast<float>(support);
                    const float contrastMean =
                        contrastSum / static_cast<float>(support);
                    if (fallingMean < minGradient || contrastMean < minContrast) {
                        continue;
                    }
                    const float score = fallingMean * contrastMean;
                    if (score > bestScore) {
                        bestScore = score;
                        bestT = t;
                        bestFalling = fallingMean;
                        bestContrast = contrastMean;
                    }
                }

                if (bestScore > 0.0f) {
                    const float deltaToEdge = bestT - oldR[ai];
                    halfValid[halfIdx] = true;
                    halfDelta[halfIdx] = deltaToEdge;
                    halfScore[halfIdx] = bestScore;
                    halfEdge[halfIdx] = bestT;
                    signedDeltaWeightSum += deltaToEdge * bestScore;
                    edgeWeightSum += bestScore;
                    ++targetCount;
                    detail << " axis=" << ai
                           << " half=" << (sign < 0 ? "-" : "+")
                           << " tip=" << oldR[ai]
                           << " edge=" << bestT
                           << " halfDelta=" << deltaToEdge
                           << " fall=" << bestFalling
                           << " contrast=" << bestContrast;
                }
            }

            if (targetCount > 0) {
                validHalfCounts[ai] = targetCount;
                float signedEdgeDelta =
                    (edgeWeightSum > 1e-9f) ? signedDeltaWeightSum / edgeWeightSum
                                             : 0.0f;
                float confidence = (targetCount >= 2) ? 1.0f : 0.30f;
                float activeStepFraction = stepFraction;
                float activeMaxChangeFraction = maxChangeFraction;
                std::string mergeMode = "weighted";
                if (simulationConfig.celluniverse3_edge_axis_fit_rod_conflict_enabled &&
                    ai == longestAxis &&
                    shape >= simulationConfig.celluniverse3_edge_axis_fit_rod_conflict_min_shape &&
                    targetCount >= 2) {
                    const bool bothOutward =
                        halfValid[0] && halfValid[1] &&
                        halfDelta[0] > 0.0f && halfDelta[1] > 0.0f;
                    const int shrinkIdx =
                        (!halfValid[0] ||
                         (halfValid[1] && halfDelta[1] < halfDelta[0])) ? 1 : 0;
                    const float bestScore = std::max(halfScore[0], halfScore[1]);
                    const bool shrinkSupported =
                        halfValid[shrinkIdx] &&
                        halfDelta[shrinkIdx] < 0.0f &&
                        halfScore[shrinkIdx] >=
                            simulationConfig
                                .celluniverse3_edge_axis_fit_rod_conflict_shrink_score_ratio *
                                bestScore;
                    if (!bothOutward && shrinkSupported) {
                        signedEdgeDelta = halfDelta[shrinkIdx];
                        confidence = 1.0f;
                        activeStepFraction =
                            simulationConfig
                                .celluniverse3_edge_axis_fit_rod_conflict_step_fraction;
                        activeMaxChangeFraction =
                            simulationConfig
                                .celluniverse3_edge_axis_fit_rod_conflict_max_radius_change_fraction;
                        mergeMode = "rod_nearer_edge";
                    }
                }
                rawTargets[ai] = oldR[ai] + signedEdgeDelta * confidence;
                const float maxDelta = oldR[ai] * activeMaxChangeFraction;
                const float delta = std::clamp(signedEdgeDelta * confidence,
                                               -maxDelta, maxDelta) *
                                    activeStepFraction;
                newR[ai] = oldR[ai] + delta;
                detail << " axis=" << ai
                       << " merge=" << mergeMode
                       << " targetDelta=" << signedEdgeDelta * confidence
                       << " step=" << activeStepFraction
                       << " maxChange=" << activeMaxChangeFraction;
            }
        }

        const std::array<float, 3> minAxisR = {
            std::max(static_cast<float>(Ellipsoid::cellConfig.minARadius),
                     Ellipsoid::cellConfig.minAnyRadiusEnabled
                         ? static_cast<float>(Ellipsoid::cellConfig.minAnyRadius)
                         : 0.0f),
            std::max(static_cast<float>(Ellipsoid::cellConfig.minBRadius),
                     Ellipsoid::cellConfig.minAnyRadiusEnabled
                         ? static_cast<float>(Ellipsoid::cellConfig.minAnyRadius)
                         : 0.0f),
            std::max(static_cast<float>(Ellipsoid::cellConfig.minCRadius),
                     Ellipsoid::cellConfig.minAnyRadiusEnabled
                         ? static_cast<float>(Ellipsoid::cellConfig.minAnyRadius)
                         : 0.0f)
        };
        const std::array<float, 3> maxAxisR = {
            static_cast<float>(Ellipsoid::cellConfig.maxARadius),
            static_cast<float>(Ellipsoid::cellConfig.maxBRadius),
            static_cast<float>(Ellipsoid::cellConfig.maxCRadius)
        };
        for (int ai = 0; ai < 3; ++ai) {
            const float upper = maxAxisR[ai] > minAxisR[ai]
                ? maxAxisR[ai]
                : std::numeric_limits<float>::max();
            newR[ai] = std::clamp(newR[ai], minAxisR[ai], upper);
        }

        const float maxAbsDelta = std::max({
            std::abs(newR[0] - oldR[0]),
            std::abs(newR[1] - oldR[1]),
            std::abs(newR[2] - oldR[2])
        });
        if (maxAbsDelta < 0.05f) {
            break;
        }

        const float oldBrightness = cell.getBrightness();
        const float oldRadiusSum = oldR[0] + oldR[1] + oldR[2];
        const float newRadiusSum = newR[0] + newR[1] + newR[2];
        const float shrinkFraction = oldRadiusSum > 1e-6f
            ? std::clamp((oldRadiusSum - newRadiusSum) / oldRadiusSum, 0.0f, 1.0f)
            : 0.0f;
        float newBrightness = oldBrightness;
        bool coreBrightnessReset = false;
        if (simulationConfig.celluniverse3_edge_axis_fit_core_brightness_enabled &&
            shrinkFraction >=
                simulationConfig
                    .celluniverse3_edge_axis_fit_core_brightness_min_shrink_fraction) {
            EllipsoidParams coreParams = cell.getCellParams();
            const float coreScale = std::clamp(
                simulationConfig
                    .celluniverse3_edge_axis_fit_core_brightness_radius_scale,
                0.05f, 1.0f);
            coreParams.aRadius = std::max(1.0f, newR[0] * coreScale);
            coreParams.bRadius = std::max(1.0f, newR[1] * coreScale);
            coreParams.cRadius = std::max(1.0f, newR[2] * coreScale);
            coreParams.brightness = oldBrightness;
            Ellipsoid coreCell(coreParams);
            const float sampledCoreBrightness = coreCell.measureMeanBrightness(
                _realFrame,
                simulationConfig
                    .celluniverse3_edge_axis_fit_core_brightness_top_fraction);
            if (std::isfinite(sampledCoreBrightness) &&
                sampledCoreBrightness > 0.0f) {
                const float maxChange =
                    std::max(0.0f,
                             simulationConfig
                                 .celluniverse3_edge_axis_fit_core_brightness_max_change_fraction) *
                    std::max(1e-6f, oldBrightness);
                const float clampedCoreBrightness =
                    std::clamp(sampledCoreBrightness,
                               std::max(0.0f, oldBrightness - maxChange),
                               std::min(1.0f, oldBrightness + maxChange));
                const float blend = std::clamp(
                    simulationConfig
                        .celluniverse3_edge_axis_fit_core_brightness_blend,
                    0.0f, 1.0f);
                newBrightness = oldBrightness * (1.0f - blend) +
                                clampedCoreBrightness * blend;
                coreBrightnessReset =
                    std::abs(newBrightness - oldBrightness) > 1e-6f;
                detail << " coreBrightnessSample=" << sampledCoreBrightness
                       << " coreBrightnessClamped=" << clampedCoreBrightness
                       << " coreBrightnessOld=" << oldBrightness
                       << " coreBrightnessNew=" << newBrightness;
            }
        }
        cell.setRadii(newR[0], newR[1], newR[2]);
        if (coreBrightnessReset) {
            cell.setBrightness(newBrightness);
        }
        anyChange = true;
        log << "  [CellUniverse3 Edge Axis Fit] cell=" << cell.getName()
            << " iter=" << iter
            << " shape=" << shape
            << " old=(" << oldR[0] << "," << oldR[1] << "," << oldR[2] << ")"
            << " target=(" << rawTargets[0] << "," << rawTargets[1]
            << "," << rawTargets[2] << ")"
            << " new=(" << newR[0] << "," << newR[1] << "," << newR[2] << ")"
            << " validHalves=(" << validHalfCounts[0] << ","
            << validHalfCounts[1] << "," << validHalfCounts[2] << ")"
            << detail.str()
            << std::endl;
        detail.str("");
        detail.clear();
    }

    return anyChange;
}

// Discover-only PCA bridge: runs the dark-gap bin analysis and returns the
// (left, right) weighted centroids as a daughter-pair PROPOSAL. Does NOT
// mutate cells, _synthFrame, or any cost cache. Caller passes the proposal
// to trySplitCellPhased so the main split path's full validation stack —
// candidate burn-in, daughter-overlap gate, bridge gate, asymmetric L2 cost,
// adaptive cost gate — decides acceptance. This replaces the old standalone
// accepting bridge path that bypassed those gates and produced false splits
// (e.g. f43 cell_310 in resume33 run, costDiff=-9554 with overlap rising
// from 0 to 17392). See split-gate-overlap-analysis.md.
bool Frame::discoverPcaBridgeProposal(size_t cellIndex,
                                      const ProbabilityConfig &probConfig,
                                      BridgeSplitProposal &outProposal,
                                      std::ostream *logSink) const
{
    if (!probConfig.pca_bridge_split_enabled) return false;
    if (cellIndex >= cells.size() || _realFrame.empty()) return false;

    std::ostream &log = logSink ? *logSink : std::cout;
    const Ellipsoid &parent = cells[cellIndex];
    if (parent.isTrash()) return false;

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
    const float longR = radii[longIdx];
    const float midR = std::max(1e-3f, radii[midIdx]);
    const float shortR = std::max(1e-3f, radii[shortIdx]);
    const float elong = longR / shortR;
    if (elong < probConfig.pca_bridge_elongation_ratio) return false;

    // Prolate-shape gate (yp_opt_speed 1ec91e7). Reject bridge candidates
    // whose elongation comes from one collapsed axis rather than a clean
    // rod shape. Example: R=(43,30,12) has long/short=3.58 (passes elong
    // gate) but mid/short=2.5 — the dark-bridge bin scan along the long
    // axis would be misled by the wedge geometry, not biology.
    const float longMidRatio = longR / midR;
    const float midShortRatio = midR / shortR;
    const float minLongMidRatio = std::max(0.0f, probConfig.pca_bridge_min_long_mid_ratio);
    const float maxMidShortRatio = std::max(0.0f, probConfig.pca_bridge_max_mid_short_ratio);
    const bool longAxisDistinct = minLongMidRatio <= 0.0f || longMidRatio >= minLongMidRatio;
    const bool shortAxesSimilar = maxMidShortRatio <= 0.0f || midShortRatio <= maxMidShortRatio;
    if (!longAxisDistinct || !shortAxesSimilar) {
        log << "  [PCA Bridge Propose] cell=" << parent.getName()
            << " elong=" << elong
            << " rejected=triaxial_shape"
            << " longR=" << longR
            << " midR=" << midR
            << " shortR=" << shortR
            << " longMidRatio=" << longMidRatio
            << " minLongMidRatio=" << minLongMidRatio
            << " midShortRatio=" << midShortRatio
            << " maxMidShortRatio=" << maxMidShortRatio
            << std::endl;
        return false;
    }

    std::array<double, 9> R_T;
    parent.generateInverseRotationMatrix(R_T);
    const int base = 3 * longIdx;
    const cv::Point3f longAxis(
        static_cast<float>(R_T[base]),
        static_cast<float>(R_T[base + 1]),
        static_cast<float>(R_T[base + 2]));

    const int bins = std::max(5, probConfig.pca_bridge_profile_bins);
    std::vector<int> binCount(bins, 0);
    std::vector<int> binBlack(bins, 0);
    std::vector<std::pair<BrightPixel, float>> nonblackWithProj;

    const float blackThreshold = std::max(0.0f, probConfig.pca_bridge_black_threshold);
    const int xMin = std::max(0, static_cast<int>(std::floor(parent.getX() - longR)));
    const int xMax = std::min(_realFrame[0].cols - 1, static_cast<int>(std::ceil(parent.getX() + longR)));
    const int yMin = std::max(0, static_cast<int>(std::floor(parent.getY() - longR)));
    const int yMax = std::min(_realFrame[0].rows - 1, static_cast<int>(std::ceil(parent.getY() + longR)));
    const int zMin = std::max(0, static_cast<int>(std::floor(parent.getZ() - longR)));
    const int zMax = std::min(static_cast<int>(_realFrame.size()) - 1,
                              static_cast<int>(std::ceil(parent.getZ() + longR)));
    const double invA2 = 1.0 / std::max(1e-6f, radii[0] * radii[0]);
    const double invB2 = 1.0 / std::max(1e-6f, radii[1] * radii[1]);
    const double invC2 = 1.0 / std::max(1e-6f, radii[2] * radii[2]);

    for (int z = zMin; z <= zMax; ++z) {
        const cv::Mat &slice = _realFrame[z];
        for (int y = yMin; y <= yMax; ++y) {
            const float *row = slice.ptr<float>(y);
            for (int x = xMin; x <= xMax; ++x) {
                const float dx = static_cast<float>(x) - parent.getX();
                const float dy = static_cast<float>(y) - parent.getY();
                const float dz = static_cast<float>(z) - parent.getZ();
                const double lx = R_T[0] * dx + R_T[1] * dy + R_T[2] * dz;
                const double ly = R_T[3] * dx + R_T[4] * dy + R_T[5] * dz;
                const double lz = R_T[6] * dx + R_T[7] * dy + R_T[8] * dz;
                const double ell = lx * lx * invA2 + ly * ly * invB2 + lz * lz * invC2;
                if (ell > 1.0) continue;

                const float brightness = row[x];
                const float proj = dx * longAxis.x + dy * longAxis.y + dz * longAxis.z;
                const float t = std::clamp((proj / longR + 1.0f) * 0.5f, 0.0f, 0.999999f);
                const int bin = std::clamp(static_cast<int>(t * bins), 0, bins - 1);
                ++binCount[bin];
                if (brightness <= blackThreshold) {
                    ++binBlack[bin];
                } else {
                    nonblackWithProj.push_back({
                        BrightPixel{cv::Point3f(static_cast<float>(x),
                                                static_cast<float>(y),
                                                static_cast<float>(z)),
                                    std::max(1e-6f, brightness - blackThreshold)},
                        proj});
                }
            }
        }
    }

    if (nonblackWithProj.empty()) return false;

    const float centerFrac = std::clamp(probConfig.pca_bridge_gap_center_fraction, 0.05f, 1.0f);
    const float minBlackFrac = std::clamp(probConfig.pca_bridge_min_black_fraction, 0.0f, 1.0f);
    const int minGapBins = std::max(1, probConfig.pca_bridge_min_gap_bins);
    int bestStart = -1;
    int bestEnd = -1;
    int runStart = -1;
    for (int bi = 0; bi < bins; ++bi) {
        const float centerT = ((static_cast<float>(bi) + 0.5f) / static_cast<float>(bins)) * 2.0f - 1.0f;
        const bool inCenter = std::abs(centerT) <= centerFrac;
        const float blackFrac = (binCount[bi] > 0)
            ? static_cast<float>(binBlack[bi]) / static_cast<float>(binCount[bi])
            : 1.0f;
        const bool darkBin = inCenter && blackFrac >= minBlackFrac;
        if (darkBin) {
            if (runStart < 0) runStart = bi;
        } else if (runStart >= 0) {
            if (bi - runStart >= minGapBins &&
                (bestStart < 0 || bi - runStart > bestEnd - bestStart + 1)) {
                bestStart = runStart;
                bestEnd = bi - 1;
            }
            runStart = -1;
        }
    }
    if (runStart >= 0 && bins - runStart >= minGapBins &&
        (bestStart < 0 || bins - runStart > bestEnd - bestStart + 1)) {
        bestStart = runStart;
        bestEnd = bins - 1;
    }
    const bool darkBridgeFound = bestStart >= 0;
    if (!darkBridgeFound && probConfig.pca_bridge_require_dark_bridge) {
        log << "  [PCA Bridge Propose] cell=" << parent.getName()
            << " elong=" << elong
            << " rejected=no_dark_bridge" << std::endl;
        return false;
    }
    if (!darkBridgeFound) {
        bestStart = bins / 2;
        bestEnd = bestStart;
    }

    const float splitBinCenter = 0.5f * (static_cast<float>(bestStart + bestEnd + 1));
    const float gapSplitProj = ((splitBinCenter / static_cast<float>(bins)) * 2.0f - 1.0f) * longR;
    const bool middleCutCentroids = probConfig.pca_bridge_middle_cut_centroids;
    const float splitProj = middleCutCentroids ? 0.0f : gapSplitProj;

    int leftCount = 0;
    int rightCount = 0;
    double leftSx = 0.0, leftSy = 0.0, leftSz = 0.0, leftSw = 0.0;
    double rightSx = 0.0, rightSy = 0.0, rightSz = 0.0, rightSw = 0.0;
    for (const auto &item : nonblackWithProj) {
        const BrightPixel &bp = item.first;
        const double w = std::max(1e-6f, bp.weight);
        if (item.second < splitProj) {
            ++leftCount;
            leftSx += bp.pos.x * w; leftSy += bp.pos.y * w; leftSz += bp.pos.z * w; leftSw += w;
        } else {
            ++rightCount;
            rightSx += bp.pos.x * w; rightSy += bp.pos.y * w; rightSz += bp.pos.z * w; rightSw += w;
        }
    }

    const int minSide = std::max(1, probConfig.pca_bridge_min_side_voxels);
    if (leftCount < minSide || rightCount < minSide) {
        log << "  [PCA Bridge Propose] cell=" << parent.getName()
            << " elong=" << elong
            << " rejected=too_few_side_voxels"
            << " left=" << leftCount
            << " right=" << rightCount
            << " min=" << minSide << std::endl;
        return false;
    }

    outProposal.d1Pos = cv::Point3f(
        static_cast<float>(leftSx / leftSw),
        static_cast<float>(leftSy / leftSw),
        static_cast<float>(leftSz / leftSw));
    outProposal.d2Pos = cv::Point3f(
        static_cast<float>(rightSx / rightSw),
        static_cast<float>(rightSy / rightSw),
        static_cast<float>(rightSz / rightSw));
    outProposal.elongation = elong;
    outProposal.parentShapeElongation = elong;
    outProposal.gapStartBin = bestStart;
    outProposal.gapEndBin = bestEnd;
    outProposal.leftPixelCount = leftCount;
    outProposal.rightPixelCount = rightCount;
    outProposal.pcaBridgeHasDarkBridge = darkBridgeFound;

    log << "  [PCA Bridge Propose] cell=" << parent.getName()
        << " elong=" << elong
        << " gapBins=" << bestStart << "-" << bestEnd
        << " darkBridge=" << (darkBridgeFound ? 1 : 0)
        << " splitProj=" << splitProj
        << " gapSplitProj=" << gapSplitProj
        << " centroidSplit=" << (middleCutCentroids ? "middle" : "gap")
        << " left=" << leftCount
        << " right=" << rightCount
        << " d1=(" << outProposal.d1Pos.x << "," << outProposal.d1Pos.y << "," << outProposal.d1Pos.z << ")"
        << " d2=(" << outProposal.d2Pos.x << "," << outProposal.d2Pos.y << "," << outProposal.d2Pos.z << ")"
        << std::endl;
    return true;
}

// Triaxial split attempt with candidate refinement + bio/cost gates.
// Implements the Phase A/B split-attempt flow from the plan. Caller supplies
// the Voronoi claim-sets for all OTHER cells (not this one) and whether to
// trust the snapshot direction as a candidate primary.
CostCallbackPair Frame::trySplitCellPhased(
    size_t cellIndex,
    const PreviousFrameSnapshot &snapshot,
    const ClaimSet &otherCellsClaimSets,
    bool useSnapshotDirection,
    const ProbabilityConfig &probConfig,
    std::vector<cv::Mat> *splitPerturbDebugPlacements,
    int *splitPerturbDebugPlacementCount,
    float splitPerturbDebugBrightness,
    const BridgeSplitProposal *bridgeProposal,
    bool bridgeProposalOnly,
    const BridgeSplitProposal *lumenProposal,
    bool lumenProposalOnly,
    int lumenBurnInIterations,
    int lumenRefineIterations,
    bool lumenUseDedicatedCostGate,
    bool lumenUseImageCostGate,
    float lumenSplitCost,
    float lumenSplitCostFraction,
    float lumenMaxPositiveCostFraction,
    float lumenPositiveGateMinImageGain,
    float lumenPositiveGateMinImageGainPenaltyRatio,
    float lumenPositiveGateElongatedParentMinShape,
    float lumenPositiveGateElongatedMaxRawWorsening,
    float lumenPositiveGateElongatedMaxSoftPenaltyFraction,
    float lumenPositiveGateElongatedMaxScore,
    float lumenMaxOverlapCostFraction,
    float lumenHighConfidenceMaxScore,
    float lumenHighConfidenceMaxOverlapCostFraction,
    float lumenHighConfidenceAxisAlignmentDegrees,
    float lumenDaughterVolumeScale,
    float lumenPrefilterMaxValleyRatio,
    float lumenBridgeMaxValleyRatio,
    float lumenMinBridgeGapWidth,
    float lumenMinEdgeBrightness,
    float lumenMaxDaughterOverlapFraction,
    bool lumenSoftGateEnabled,
    float lumenSoftAxisPenaltyFraction,
    float lumenSoftDaughterOverlapPenaltyFraction,
    float lumenSoftValleyPenaltyFraction,
    float lumenSoftBridgeGapPenaltyFraction,
    float lumenSoftOverlapCostPenaltyWeight,
    bool lumenBridgeEvidenceWaivesOverlapSoftPenalty,
    float lumenHardMaxDaughterOverlapFraction,
    float lumenHardMaxValleyRatio,
    float lumenHardMaxOverlapCostFraction,
    float lumenMinPostRefitLateralSeparation,
    float lumenMinPostRefitLateralSeparationRadiusScale,
    float lumenMaxZDominanceForLowLateralSeparation,
    bool lumenDynamicOverlapEnabled,
    float lumenLocalDensityRadiusScale,
    float lumenLocalDensityOverlapBonus,
    float lumenMaxDynamicDaughterOverlapFraction,
    float lumenSnapshotSeedMaxRefitDrift,
    bool lumenSkipExistingCellBuriedCheck,
    bool lumenSkipNeighborBridgeCheck)
{
    const auto noop = [](bool) {};
    if (cellIndex >= cells.size()) return {0.0, noop};

    // Voronoi cost-territory is disabled for the entire split attempt.
    // Reason: trySplitCellPhased mutates cells[] in-place (erase parent,
    // push_back two daughters) but does not rebuild _voronoiMap. Any
    // subsequent perturbCell(daughterIdx) inside burn-in / refine would
    // query the STALE map with indices that either (a) point to an
    // unrelated cell's pre-split claim region for d1Idx = N-1, or
    // (b) are out of range for d2Idx = N, silently zero-ing the cost
    // and killing the position gradient. Disabling Voronoi here routes
    // burn-in through the legacy all-voxels cost, which is what the
    // split-candidate geometry gates already expect. The outer
    // CellUniverse::optimize rebuilds the map after accept; on reject,
    // cells[] is restored so the pre-split map remains valid. The guard
    // struct restores the flag on every exit path.
    struct VoronoiDisableGuard {
        bool *flagPtr;
        bool saved;
        VoronoiDisableGuard(bool *p) : flagPtr(p), saved(*p) { *p = false; }
        ~VoronoiDisableGuard() { *flagPtr = saved; }
    };
    VoronoiDisableGuard vorGuard(&_voronoiEnabled);

    // --- 0. Save live parent and install snapshot-state parent ---
    //
    // The split attempt must compare daughters against a FULL-SIZE parent
    // at the snapshot position/rotation, not against the Phase A/B-drifted
    // live parent. Otherwise the cost delta is dominated by "daughters vs
    // already-collapsed parent" which is small even for real divisions.
    //
    // Strategy: save the live parent, construct a snapshot-state Ellipsoid,
    // install it at cells[cellIndex], update _synthFrame + _currentCost
    // incrementally via generateSynthFrameFast. Everything downstream
    // (parent local, baseline cost, savedCells, candidate loop) then sees
    // the snapshot-state parent. On rejection we restore the live parent
    // to keep Phase B's perturbation progress.
    const Ellipsoid liveParent = cells[cellIndex];
    const std::string parentName = liveParent.getName();
    const bool bridgeTunnelConstraintActive =
        simulationConfig.celluniverse3_enabled &&
        bridgeProposalOnly &&
        bridgeProposal != nullptr &&
        bridgeProposal->cellUniverse3MapTunnelConstraintAvailable;
    auto bridgeTunnelContainsPoint = [&](const cv::Point3f &pos) {
        if (!bridgeTunnelConstraintActive) {
            return true;
        }
        if (!std::isfinite(pos.x) ||
            !std::isfinite(pos.y) ||
            !std::isfinite(pos.z) ||
            bridgeProposal->cellUniverse3MapTunnelFlatIndices.empty() ||
            bridgeProposal->cellUniverse3MapTunnelGridX <= 0 ||
            bridgeProposal->cellUniverse3MapTunnelGridY <= 0 ||
            bridgeProposal->cellUniverse3MapTunnelGridZ <= 0) {
            return false;
        }
        const BoundingBox3D &bbox =
            bridgeProposal->cellUniverse3MapTunnelBbox;
        if (bbox.isValid() &&
            (pos.x < static_cast<float>(bbox.xMin) ||
             pos.x > static_cast<float>(bbox.xMax) ||
             pos.y < static_cast<float>(bbox.yMin) ||
             pos.y > static_cast<float>(bbox.yMax) ||
             pos.z < static_cast<float>(bbox.zMin) ||
             pos.z > static_cast<float>(bbox.zMax))) {
            return false;
        }
        auto gridForPoint = [&](const cv::Point3f &point) {
            const int ix = std::clamp(
                static_cast<int>(std::floor(
                    point.x / std::max(1, bridgeProposal->cellUniverse3MapTunnelBoxSizeX))),
                0,
                std::max(0, bridgeProposal->cellUniverse3MapTunnelGridX - 1));
            const int iy = std::clamp(
                static_cast<int>(std::floor(
                    point.y / std::max(1, bridgeProposal->cellUniverse3MapTunnelBoxSizeY))),
                0,
                std::max(0, bridgeProposal->cellUniverse3MapTunnelGridY - 1));
            const int iz = std::clamp(
                static_cast<int>(std::floor(
                    point.z / std::max(1, bridgeProposal->cellUniverse3MapTunnelBoxSizeZ))),
                0,
                std::max(0, bridgeProposal->cellUniverse3MapTunnelGridZ - 1));
            return std::make_tuple(ix, iy, iz);
        };
        auto flatIndex = [&](int ix, int iy, int iz) {
            return (iz * std::max(1, bridgeProposal->cellUniverse3MapTunnelGridY) + iy) *
                       std::max(1, bridgeProposal->cellUniverse3MapTunnelGridX) +
                   ix;
        };
        const auto [baseIx, baseIy, baseIz] = gridForPoint(pos);
        const int neighborBoxes = std::clamp(
            bridgeProposal->cellUniverse3MapTunnelNeighborBoxes,
            0,
            2);
        const auto &flatIndices =
            bridgeProposal->cellUniverse3MapTunnelFlatIndices;
        for (int dz = -neighborBoxes; dz <= neighborBoxes; ++dz) {
            const int iz = baseIz + dz;
            if (iz < 0 || iz >= bridgeProposal->cellUniverse3MapTunnelGridZ) continue;
            for (int dy = -neighborBoxes; dy <= neighborBoxes; ++dy) {
                const int iy = baseIy + dy;
                if (iy < 0 || iy >= bridgeProposal->cellUniverse3MapTunnelGridY) continue;
                for (int dx = -neighborBoxes; dx <= neighborBoxes; ++dx) {
                    const int ix = baseIx + dx;
                    if (ix < 0 || ix >= bridgeProposal->cellUniverse3MapTunnelGridX) continue;
                    if (std::binary_search(flatIndices.begin(),
                                           flatIndices.end(),
                                           flatIndex(ix, iy, iz))) {
                        return true;
                    }
                }
            }
        }
        return false;
    };

    const bool snapshotValid = snapshot.valid &&
        snapshot.aRadius > 1e-3f &&
        snapshot.cRadius > 1e-3f;
    const float srcMajor = snapshotValid ? snapshot.aRadius : liveParent.getARadius();
    const float srcB     = (snapshotValid && snapshot.bRadius > 1e-3f)
        ? snapshot.bRadius : liveParent.getBRadius();
    const float srcMinor = snapshotValid ? snapshot.cRadius : liveParent.getCRadius();

    // Build the snapshot-state parent: position, radii, rotation, and
    // brightness all come from the snapshot (falling back to live values
    // when snapshot is missing a field).
    Ellipsoid snapshotParent = liveParent;
    if (snapshotValid) {
        EllipsoidParams snapParams(parentName,
                                  snapshot.position.x, snapshot.position.y, snapshot.position.z,
                                  srcMajor, srcMinor,
                                  snapshot.thetaX, snapshot.thetaY, snapshot.thetaZ,
                                  snapshot.brightness > 0.0f ? snapshot.brightness
                                                             : liveParent.getBrightness());
        snapParams.bRadius = srcB;
        snapshotParent = Ellipsoid(snapParams);

        // Swap in snapshot parent, render the affected z-range, and
        // compare costs to decide which baseline (live vs snap) is better.
        cells[cellIndex] = snapshotParent;
        int affMinS = -1, affMaxS = -1;
        Ellipsoid liveMutable = liveParent;
        Ellipsoid snapshotMutable = snapshotParent;
        auto swappedSynth = generateSynthFrameFast(liveMutable, snapshotMutable,
                                                    &affMinS, &affMaxS);

        // Compare live vs snap using the CORRECT cost mode for this frame.
        // In bbox mode, _currentCost is stale (perturbCell doesn't update it),
        // so we compute a fresh bbox cost over a temporary bbox covering
        // both live and snap positions. In legacy mode, _currentCost is
        // maintained by perturbCell and calculateIncrementalCost is exact.
        double liveCostForComparison = 0.0;
        double snapCostForComparison = 0.0;
        std::vector<double> swappedPerSlice;
        double swappedImageCost = 0.0;
        if (_useBboxCost) {
            // Build a temporary bbox covering both live and snap positions.
            const float maxR = std::max({srcMajor, srcB, srcMinor,
                                         liveParent.getARadius(),
                                         liveParent.getBRadius(),
                                         liveParent.getCRadius()});
            // Union of live-parent bbox and snap-parent bbox.
            BoundingBox3D liveBbox = computeBboxAtPoint(
                cv::Point3f(liveParent.getX(), liveParent.getY(), liveParent.getZ()),
                maxR, _bboxMarginScale);
            BoundingBox3D snapBbox = computeBboxAtPoint(
                snapshot.position, maxR, _bboxMarginScale);
            BoundingBox3D cmpBbox;
            cmpBbox.xMin = std::min(liveBbox.xMin, snapBbox.xMin);
            cmpBbox.xMax = std::max(liveBbox.xMax, snapBbox.xMax);
            cmpBbox.yMin = std::min(liveBbox.yMin, snapBbox.yMin);
            cmpBbox.yMax = std::max(liveBbox.yMax, snapBbox.yMax);
            cmpBbox.zMin = std::min(liveBbox.zMin, snapBbox.zMin);
            cmpBbox.zMax = std::max(liveBbox.zMax, snapBbox.zMax);
            const std::vector<uint8_t> noMask;
            liveCostForComparison = calculateBboxCost(cmpBbox, _synthFrame, noMask);
            snapCostForComparison = calculateBboxCost(cmpBbox, swappedSynth, noMask);
        } else {
            liveCostForComparison = _currentCost;
            swappedImageCost = calculateIncrementalCost(swappedSynth,
                                                        affMinS, affMaxS,
                                                        swappedPerSlice);
            snapCostForComparison = swappedImageCost;
        }

        // Use min(liveCost, snapCost) as baseline. If the snapshot parent
        // is much worse than live (cell drifted between frames), keeping
        // the inflated snapshot baseline would make ANY daughter placement
        // look like an improvement — causing false splits. Using the
        // minimum ensures the split must beat the TIGHTER of the two fits.
        const bool useSnapshotBaseline = (snapCostForComparison <= liveCostForComparison);
        if (useSnapshotBaseline) {
            _synthFrame = swappedSynth;
            if (!_useBboxCost) {
                _currentCost = swappedImageCost;
                _currentCostPerSlice = swappedPerSlice;
            }
        } else {
            // Revert: live parent was a better fit, keep it as baseline.
            cells[cellIndex] = liveParent;
            // _synthFrame already reflects the live parent (never changed).
        }

        std::cout << "  [Split Snapshot Parent] " << parentName
                  << " livePos=(" << liveParent.getX() << "," << liveParent.getY() << "," << liveParent.getZ() << ")"
                  << " snapPos=(" << snapshot.position.x << "," << snapshot.position.y << "," << snapshot.position.z << ")"
                  << " liveR=(" << liveParent.getARadius() << "," << liveParent.getBRadius() << "," << liveParent.getCRadius() << ")"
                  << " snapR=(" << srcMajor << "," << srcB << "," << srcMinor << ")"
                  << " liveCost=" << liveCostForComparison
                  << " snapCost=" << snapCostForComparison
                  << " baseline=" << (useSnapshotBaseline ? "snapshot" : "live")
                  << (_useBboxCost ? " (bbox)" : " (full)")
                  << std::endl;

        // Update snapshotParent to match what was actually installed so
        // restoreLiveParent works correctly on rejection paths.
        if (!useSnapshotBaseline) {
            snapshotParent = liveParent;
        }
    }

    // Geometric reference for split candidate generation: ALWAYS use the
    // snapshot-state parent (per 2026-04-19 design rule: split candidates
    // use SNAP or PCA-derived data only, never LIVE).
    //
    // Rationale: live position/radii drift during in-frame perturbation. If
    // axis directions, radii, or fallback midpoints were derived from the
    // live cell, that drift would cascade into different candidate seeds
    // each iteration, making the split decision sensitive to perturbation
    // history (frame-3 12345 / e9077 regression in run 084534).
    //
    // cells[cellIndex] still holds whatever the cost comparison above chose
    // (snapshot or live) — that is the RENDERING baseline for the cost
    // delta, intentionally separate from the GEOMETRIC reference here.
    //
    // Falls back to the live cell only when no snapshot exists (frame 1,
    // newborn daughter post-split — but those are filtered earlier).
    Ellipsoid parent = snapshotValid ? snapshotParent : cells[cellIndex];

    // Restore-live-parent helper. Used on every rejection path (early
    // returns inside this function AND the callback's reject branch) to
    // undo the snapshot-state install so Phase B's live state isn't
    // lost. No-op if snapshot wasn't valid (no install happened).
    auto restoreLiveParent = [&]() {
        // Cleanup snap-bboxes + shared masks installed for daughter
        // candidates — safe to erase unconditionally (erase of absent key
        // is a no-op). Covers every reject path through restoreLiveParent.
        _snapBboxes.erase(parentName + "0");
        _snapBboxes.erase(parentName + "1");
        // _sharedMasks removed — no longer used (cost uses empty mask)
        if (!snapshotValid) return;
        if (cellIndex >= cells.size()) return;
        cells[cellIndex] = liveParent;
        int affMinR = -1, affMaxR = -1;
        Ellipsoid snapshotMutable = snapshotParent;
        Ellipsoid liveMutable = liveParent;
        auto revertedSynth = generateSynthFrameFast(snapshotMutable, liveMutable,
                                                     &affMinR, &affMaxR);
        _synthFrame = revertedSynth;
        // In bbox mode the full-image cache is never read for decisions and
        // is intentionally left stale (Change 1). Skip the incremental
        // recompute and the stale-seeded write entirely.
        if (!_useBboxCost) {
            std::vector<double> revertedPerSlice;
            const double revertedCost = calculateIncrementalCost(revertedSynth,
                                                                    affMinR, affMaxR,
                                                                    revertedPerSlice);
            _currentCost = revertedCost;
            _currentCostPerSlice = revertedPerSlice;
        }
    };

    // --- 1. Gather bright pixels in a snapshot-centered bounding box ---

    const float parentMajor = std::max(srcMajor, parent.getARadius());
    const float parentB     = std::max(srcB,     parent.getBRadius());
    const float parentMinor = std::max(srcMinor, parent.getCRadius());
    const float boxRadius = 3.0f * std::max({parentMajor, parentB, parentMinor});

    // Reference parent volume used by the bio volume-fraction check and by
    // the drift gate. Uses source (snapshot when available) radii so a
    // shrunken live parent doesn't skew either ratio.
    const double refParentVolume =
        static_cast<double>(srcMajor) *
        static_cast<double>(srcB) *
        static_cast<double>(srcMinor);
    const float srcMaxR = std::max({srcMajor, srcB, srcMinor});

    std::cout << "[Split Attempt] " << parentName
              << " useSnapshotDir=" << (useSnapshotDirection ? 1 : 0)
              << " snapValid=" << (snapshotValid ? 1 : 0)
              << " snapElong=" << snapshot.shapeElongation
              << " snapLongLen=" << snapshot.splitAxisLength
              << " src=(" << srcMajor << "," << srcB << "," << srcMinor << ")"
              << " liveR=(" << liveParent.getARadius() << "," << liveParent.getBRadius() << "," << liveParent.getCRadius() << ")"
              << " livePos=(" << liveParent.getX() << "," << liveParent.getY() << "," << liveParent.getZ() << ")"
              << " parentNow=(" << parent.getARadius() << "," << parent.getBRadius() << "," << parent.getCRadius() << ")"
              << " parentPos=(" << parent.getX() << "," << parent.getY() << "," << parent.getZ() << ")"
              << std::endl;

    // Self claim points for Voronoi test: the two expected-daughter seeds
    // along the snapshot long axis (if we have one) or just the snapshot
    // center (for round cells).
    //
    //   D1_seed = snapshot.center - 0.5 * splitAxisLength * splitAxisDir
    //   D2_seed = snapshot.center + 0.5 * splitAxisLength * splitAxisDir
    std::vector<cv::Point3f> selfClaim;
    if (snapshot.splitAxisLength > 1e-3f) {
        const float half = 0.5f * snapshot.splitAxisLength;
        selfClaim.push_back(cv::Point3f(
            snapshot.position.x - half * snapshot.splitAxisDir.x,
            snapshot.position.y - half * snapshot.splitAxisDir.y,
            snapshot.position.z - half * snapshot.splitAxisDir.z));
        selfClaim.push_back(cv::Point3f(
            snapshot.position.x + half * snapshot.splitAxisDir.x,
            snapshot.position.y + half * snapshot.splitAxisDir.y,
            snapshot.position.z + half * snapshot.splitAxisDir.z));
        std::cout << "  [Split Seeds] " << parentName
                  << " snapCenter=(" << snapshot.position.x << "," << snapshot.position.y << "," << snapshot.position.z << ")"
                  << " splitAxisDir=(" << snapshot.splitAxisDir.x << "," << snapshot.splitAxisDir.y << "," << snapshot.splitAxisDir.z << ")"
                  << " splitAxisLen=" << snapshot.splitAxisLength
                  << " D1_seed=(" << selfClaim[0].x << "," << selfClaim[0].y << "," << selfClaim[0].z << ")"
                  << " D2_seed=(" << selfClaim[1].x << "," << selfClaim[1].y << "," << selfClaim[1].z << ")"
                  << " boxRadius=" << boxRadius
                  << std::endl;
    } else {
        selfClaim.push_back(snapshot.position);
        std::cout << "  [Split Seeds] " << parentName
                  << " snapCenter=(" << snapshot.position.x << "," << snapshot.position.y << "," << snapshot.position.z << ")"
                  << " splitAxisLen=0 (round cell, single seed = snapCenter)"
                  << " boxRadius=" << boxRadius
                  << std::endl;
    }

    // Voronoi exclusion diagnostic — summarize the other-cell claim set.
    {
        size_t otherCellCount = 0;
        size_t otherPointCount = 0;
        constexpr bool verboseSplitVoronoiClaimNames = false;
        std::ostringstream oc;
        for (const auto &kv : otherCellsClaimSets) {
            ++otherCellCount;
            otherPointCount += kv.second.size();
            if (verboseSplitVoronoiClaimNames) {
                oc << " " << kv.first << "[" << kv.second.size() << "]";
            }
        }
        std::cout << "  [Voronoi In] " << parentName
                  << " otherCells=" << otherCellCount
                  << " otherPoints=" << otherPointCount
                  << " selfPoints=" << selfClaim.size();
        if (verboseSplitVoronoiClaimNames) {
            std::cout << " others=" << oc.str();
        }
        std::cout << std::endl;
    }

    GatherStats gstats;
    std::vector<BrightPixel> pixels = gatherBrightPixelsVoronoi(
        _realFrame,
        *this,
        snapshot.position,
        boxRadius,
        selfClaim,
        otherCellsClaimSets,
        &gstats);

    std::cout << "  [Voronoi Out] " << parentName
              << " box=" << gstats.boxVoxels
              << " inSphere=" << gstats.inSphere
              << " aboveBright=" << gstats.aboveBrightness
              << " voronoiRejected=" << gstats.voronoiRejected
              << " kept=" << gstats.voronoiKept
              << std::endl;

    if (pixels.size() < 20) {
        std::cout << "[Split Reject] " << parentName
                  << " too_few_bright_pixels=" << pixels.size() << std::endl;
        restoreLiveParent();
        return {0.0, noop};
    }

    // --- 2. Local-axis directions + data-driven daughter placement ---
    //   Instead of PCA (whose direction is dominated by the z-brightness
    //   gradient for flat cells), try ALL THREE cell-local axes (a, b, c)
    //   rotated to world frame. For each axis, project bright pixels onto
    //   that direction and compute the centroid of each half — this gives a
    //   data-driven midpoint and separation. Cost picks the winning axis.
    std::array<double, 9> parentR_T;
    parent.generateInverseRotationMatrix(parentR_T);
    const cv::Point3f parentCenter(parent.getX(), parent.getY(), parent.getZ());

    // Extract all three local axes in world frame from the inverse rotation
    // matrix. R_T is R^T stored row-major (R_T[r*3+c] = R[c,r]); the world
    // direction of local axis i is column i of R = (R_T[3i], R_T[3i+1],
    // R_T[3i+2]). See worldSplitAxis comment for the 2026-04-19 indexing
    // bug fix history.
    auto extractWorldAxis = [&](int axisIdx) -> cv::Point3f {
        const int base = 3 * axisIdx;
        const double dx = parentR_T[base];
        const double dy = parentR_T[base + 1];
        const double dz = parentR_T[base + 2];
        const double norm = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (norm < 1e-9) return {0.0f, 0.0f, 1.0f};
        return cv::Point3f(
            static_cast<float>(dx / norm),
            static_cast<float>(dy / norm),
            static_cast<float>(dz / norm));
    };
    const cv::Point3f axisA = extractWorldAxis(0); // local x (a-axis) in world
    const cv::Point3f axisB = extractWorldAxis(1); // local y (b-axis) in world
    const cv::Point3f axisC = extractWorldAxis(2); // local z (c-axis) in world

    // Only try axes that are close to the shortest radius. Cells split
    // through their thin dimension. Axes much longer than the shortest
    // create false splits by placing daughters end-to-end along a long
    // direction, matching the z-brightness gradient instead of a real
    // division (e3d03 false splits at f2 via axA and axC).
    //
    // Include ONLY the single shortest local axis as a primary direction.
    //
    // Data analysis from run 002144 (196 split winners across burn-in,
    // 6 accepted splits):
    //   Axis   | Wins | Accepts
    //   -------|------|--------
    //   axC    | 100  |   1
    //   imgPca |  75  |   5
    //   axB    |  19  |   0
    //   axA    |   2  |   0
    //
    // ZERO accepts came from axB or axA — they only competed in near-
    // round cells (where the 1.2× threshold admitted them) and never
    // beat axC or imgPca in the final accept. Biologically, cells
    // divide along their SHORTEST axis; axA and axB were candidate-set
    // bloat. Dropping them reduces near-round-cell candidates from 3
    // axes × 10 variants = 30 to 2 axes × 10 = 20 (~33% saving), with
    // zero accuracy loss.
    const float rA = parent.getARadius();
    const float rB = parent.getBRadius();
    const float rC = parent.getCRadius();
    const float radii3[] = {rA, rB, rC};
    const cv::Point3f axes3[] = {axisA, axisB, axisC};
    const std::string names3[] = {"axA", "axB", "axC"};

    int shortIdx = 0;
    for (int i = 1; i < 3; ++i) {
        if (radii3[i] < radii3[shortIdx]) shortIdx = i;
    }
    std::vector<cv::Point3f> primaryDirs{axes3[shortIdx]};
    std::vector<std::string> primaryNames{names3[shortIdx]};

    // Add the image-PCA direction (from pre-pass, supplied via
    // snapshot.splitAxisDir) as an additional primary axis. For near-round
    // snap cells, parent-rotation axes are arbitrary — pre-pass finds the
    // real direction connecting the two bright blobs in the current frame.
    // Only add if it's sufficiently different from the existing axes
    // (|dot| < 0.95 against all).
    if (useSnapshotDirection && snapshotValid) {
        const cv::Point3f pcaDir = snapshot.splitAxisDir;
        const double pcaNorm = cv::norm(pcaDir);
        if (pcaNorm > 1e-3) {
            const cv::Point3f pcaUnit(
                static_cast<float>(pcaDir.x / pcaNorm),
                static_cast<float>(pcaDir.y / pcaNorm),
                static_cast<float>(pcaDir.z / pcaNorm));
            bool duplicate = false;
            for (const auto &existing : primaryDirs) {
                const double dot = std::abs(
                    static_cast<double>(existing.x) * pcaUnit.x +
                    static_cast<double>(existing.y) * pcaUnit.y +
                    static_cast<double>(existing.z) * pcaUnit.z);
                if (dot > 0.95) { duplicate = true; break; }
            }
            if (!duplicate) {
                primaryDirs.push_back(pcaUnit);
                primaryNames.push_back("imgPca");
            }
        }
    }

    {
        std::ostringstream selAxes;
        for (size_t i = 0; i < primaryNames.size(); ++i) {
            if (i > 0) selAxes << ",";
            selAxes << primaryNames[i];
            selAxes << "=(" << primaryDirs[i].x << "," << primaryDirs[i].y
                    << "," << primaryDirs[i].z << ")";
        }
        std::cout << "  [Split Dirs] " << parentName
                  << " mode=shortest_local+imgPca"
                  << " radii=(" << rA << "," << rB << "," << rC << ")"
                  << " shortestAxis=" << names3[shortIdx]
                  << " shortestR=" << radii3[shortIdx]
                  << " selected=[" << selAxes.str() << "]"
                  << " nPrimaries=" << primaryDirs.size()
                  << " (expect 2 midpoints × 5 variants each = "
                  << (primaryDirs.size() * 10) << " candidates before cap)"
                  << " nPixels=" << pixels.size()
                  << std::endl;
    }

    // --- 3. Generate K candidate placements around each (midpoint, direction) pair ---
    //
    // For each axis, centroidsAlongAxis projects bright pixels onto that
    // direction and splits at the median to get two daughter centroids.
    // This gives a data-driven midpoint and separation per axis. Snapshot
    // center is also tried as an alternative midpoint if it differs.
    struct Candidate {
        cv::Point3f d1Pos;
        cv::Point3f d2Pos;
        std::string label;
        float sphereRadiusOverride = -1.0f;
    };

    // Daughter built radii = volumeScale × snapshot parent radii. Default
    // ∛(0.5) ≈ 0.7937 preserves total cell volume across the split. Tunable
    // via probConfig.split_daughter_volume_scale — raise toward 1.0 to make
    // daughters cover more material on first cost eval (helps when parent
    // PCA fit is tight and undercounts the real cell extent), lower toward
    // 0.5 to start tight and rely on per-daughter PCA refit to grow back.
    const float volumeScale = std::max(0.1f, probConfig.split_daughter_volume_scale);
    const float daughterR = std::max(0.5f * parentMajor, 5.0f);
    const float rotDeltaRad =
        probConfig.split_candidate_rotation_delta_degrees * static_cast<float>(M_PI) / 180.0f;
    const float transDelta =
        probConfig.split_candidate_translation_delta_fraction * daughterR;

    // For each axis direction, project bright pixels onto that axis and
    // compute centroids of the two halves. This gives a data-driven midpoint
    // and separation tuned to where the brightness actually is along each
    // axis, rather than using a fixed radius as the initial separation.
    struct AxisPlacement {
        cv::Point3f d1, d2;     // data-driven daughter centroids
        cv::Point3f midpoint;
        float separation;
        bool valid;
    };
    const size_t nDirs = primaryDirs.size();
    std::vector<AxisPlacement> axisPlace(nDirs);
    // Fallback separation: use the minimum radius as a conservative default.
    const float fallbackSep = std::min({rA, rB, rC});
    for (size_t i = 0; i < nDirs; ++i) {
        axisPlace[i].valid = centroidsAlongAxis(
            pixels, primaryDirs[i], axisPlace[i].d1, axisPlace[i].d2);
        if (axisPlace[i].valid) {
            axisPlace[i].separation = static_cast<float>(
                cv::norm(axisPlace[i].d1 - axisPlace[i].d2));
            axisPlace[i].midpoint = 0.5f * (axisPlace[i].d1 + axisPlace[i].d2);
        } else {
            axisPlace[i].separation = fallbackSep;
            axisPlace[i].midpoint = parentCenter;
        }
        std::cout << "  [Split AxisPlace] " << parentName
                  << " axis=" << primaryNames[i]
                  << " sep=" << axisPlace[i].separation
                  << " mid=(" << axisPlace[i].midpoint.x << "," << axisPlace[i].midpoint.y << "," << axisPlace[i].midpoint.z << ")"
                  << " valid=" << axisPlace[i].valid
                  << std::endl;
    }

    std::vector<Candidate> candidates;
    for (size_t di = 0; di < nDirs; ++di) {
        const auto &dir0 = primaryDirs[di];
        const std::string &axLabel = primaryNames[di];
        cv::Point3f perpU, perpV;
        orthonormalFrame(dir0, perpU, perpV);

        // Two midpoint options for this axis:
        // 1. Data-driven: centroid midpoint from projecting pixels onto this axis
        // 2. Snapshot center (if available and different)
        struct AxisMidOption {
            cv::Point3f center;
            float separation;
            std::string label;
        };
        std::vector<AxisMidOption> axisMids;
        const auto &ap = axisPlace[di];
        axisMids.push_back({ap.midpoint, ap.separation, "data_" + axLabel});

        if (snapshotValid && snapshot.splitAxisLength > 1e-3f) {
            const float dist = static_cast<float>(cv::norm(ap.midpoint - snapshot.position));
            if (dist > 0.5f) {
                axisMids.push_back({snapshot.position, ap.separation, "snap_" + axLabel});
            }
        }

        for (const auto &mp : axisMids) {
            const cv::Point3f midpoint = mp.center;
            const float sep = mp.separation;
            const float half = 0.5f * sep;
            const std::string baseLabel = mp.label;

            // Primary candidate for this (midpoint, direction) pair.
            cv::Point3f d1(midpoint.x - half * dir0.x,
                           midpoint.y - half * dir0.y,
                           midpoint.z - half * dir0.z);
            cv::Point3f d2(midpoint.x + half * dir0.x,
                           midpoint.y + half * dir0.y,
                           midpoint.z + half * dir0.z);
            candidates.push_back({d1, d2, baseLabel + "_primary"});

            // Rotation variants around an axis perpendicular to dir0.
            for (float sign : {-1.0f, 1.0f}) {
                const float angle = sign * rotDeltaRad;
                const cv::Point3f rDir = rotateAroundAxis(dir0, perpU, angle);
                candidates.push_back({
                    cv::Point3f(midpoint.x - half * rDir.x,
                                midpoint.y - half * rDir.y,
                                midpoint.z - half * rDir.z),
                    cv::Point3f(midpoint.x + half * rDir.x,
                                midpoint.y + half * rDir.y,
                                midpoint.z + half * rDir.z),
                    baseLabel + (sign < 0 ? "_rot-" : "_rot+")
                });
            }

            // Translation variants along dir0.
            for (float sign : {-1.0f, 1.0f}) {
                const float t = sign * transDelta;
                candidates.push_back({
                    cv::Point3f(d1.x + t * dir0.x, d1.y + t * dir0.y, d1.z + t * dir0.z),
                    cv::Point3f(d2.x + t * dir0.x, d2.y + t * dir0.y, d2.z + t * dir0.z),
                    baseLabel + (sign < 0 ? "_trans-" : "_trans+")
                });
            }
        }
    }

    // Inject the optional PCA-bridge proposal as a single extra candidate
    // at the FRONT of the list (so it survives the Kmax truncation below).
    // In CellUniverse 2 bridge-cut mode, bridgeProposalOnly clears all other
    // generated candidates so the black-bridge proposal is the actual split.
    bool cleanSignalCandidateGateActive = false;
    cv::Point3f cleanSignalCandidateBaseD1(0.0f, 0.0f, 0.0f);
    cv::Point3f cleanSignalCandidateBaseD2(0.0f, 0.0f, 0.0f);
    if (bridgeProposal != nullptr) {
        Candidate bridgeCand;
        bridgeCand.d1Pos = bridgeProposal->d1Pos;
        bridgeCand.d2Pos = bridgeProposal->d2Pos;
        bridgeCand.label = "bridge_primary";
        bridgeCand.sphereRadiusOverride = bridgeProposal->daughterSphereRadius;
        cleanSignalCandidateGateActive =
            simulationConfig.celluniverse3_enabled &&
            probConfig.celluniverse3_clean_signal_candidate_spacing_gate_enabled &&
            bridgeProposalOnly &&
            bridgeProposal->signalCenterScore >= 0.0f &&
            bridgeProposal->centerSnapApplied &&
            !bridgeProposal->centerSnapUsedAlignedPairFallback &&
            bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
            bridgeProposal->windowBothDaughtersSupported >=
                std::max(
                    1,
                    probConfig
                        .celluniverse3_signal_center_future_position_lock_min_future_both) &&
            bridgeProposal->windowMissingDaughterCount <=
                std::max(
                    0,
                    probConfig
                        .celluniverse3_signal_center_future_position_lock_max_missing) &&
            bridgeProposal->windowParentPersists == 0 &&
            bridgeProposal->windowBestMatchedMinBrightness >=
                probConfig
                    .celluniverse3_signal_center_future_position_lock_min_future_brightness;
        cleanSignalCandidateBaseD1 = bridgeCand.d1Pos;
        cleanSignalCandidateBaseD2 = bridgeCand.d2Pos;
        std::vector<Candidate> bridgeCandidates;
        int bridgeCenterSlideCount = 0;
        int bridgeEvidenceMidpointCount = 0;
        const auto validPoint = [](const cv::Point3f &p) {
            return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
        };
        const auto duplicateBridgeCandidate = [&](const Candidate &cand) {
            for (const auto &existing : bridgeCandidates) {
                const float direct =
                    static_cast<float>(cv::norm(existing.d1Pos - cand.d1Pos) +
                                       cv::norm(existing.d2Pos - cand.d2Pos));
                const float swapped =
                    static_cast<float>(cv::norm(existing.d1Pos - cand.d2Pos) +
                                       cv::norm(existing.d2Pos - cand.d1Pos));
                if (std::min(direct, swapped) < 1.0f) {
                    return true;
                }
            }
            return false;
        };
        const auto appendBridgeCandidate = [&](const Candidate &baseCand,
                                               bool allowCenterSlides,
                                               bool countsAsEvidence) {
            if (!validPoint(baseCand.d1Pos) || !validPoint(baseCand.d2Pos)) {
                return false;
            }
            const cv::Point3f axisVec = baseCand.d2Pos - baseCand.d1Pos;
            const float sep = static_cast<float>(cv::norm(axisVec));
            if (sep <= 1e-3f || duplicateBridgeCandidate(baseCand)) {
                return false;
            }
            bridgeCandidates.push_back(baseCand);
            if (countsAsEvidence) {
                ++bridgeEvidenceMidpointCount;
            }
            if (!bridgeProposalOnly ||
                !simulationConfig.celluniverse3_enabled ||
                !probConfig.celluniverse3_injected_split_center_slide_enabled ||
                !allowCenterSlides) {
                return true;
            }
            const cv::Point3f axis(axisVec.x / sep, axisVec.y / sep, axisVec.z / sep);
            const float slideRange =
                std::max(0.0f,
                         probConfig.celluniverse3_injected_split_center_slide_range_scale) *
                std::max(srcMaxR, 1.0f);
            const float slideStep = std::max(
                std::max(0.1f,
                         probConfig.celluniverse3_injected_split_center_slide_min_step),
                std::max(0.0f,
                         probConfig.celluniverse3_injected_split_center_slide_step_scale) *
                    std::max(srcMaxR, 1.0f));
            if (slideRange < slideStep) return true;
            const int nSteps = static_cast<int>(std::floor(slideRange / slideStep));
            for (int stepIdx = 1; stepIdx <= nSteps; ++stepIdx) {
                for (float sign : {-1.0f, 1.0f}) {
                    const float offset = sign * slideStep * static_cast<float>(stepIdx);
                    if (std::abs(offset) > slideRange + 1e-3f) continue;
                    Candidate slid = baseCand;
                    slid.d1Pos = cv::Point3f(baseCand.d1Pos.x + offset * axis.x,
                                             baseCand.d1Pos.y + offset * axis.y,
                                             baseCand.d1Pos.z + offset * axis.z);
                    slid.d2Pos = cv::Point3f(baseCand.d2Pos.x + offset * axis.x,
                                             baseCand.d2Pos.y + offset * axis.y,
                                             baseCand.d2Pos.z + offset * axis.z);
                    if (!duplicateBridgeCandidate(slid)) {
                        bridgeCandidates.push_back(slid);
                        ++bridgeCenterSlideCount;
                    }
                }
            }
            return true;
        };
        const bool suppressFutureRodTipCenterSlides =
            simulationConfig.celluniverse2_enabled &&
            bridgeProposalOnly &&
            bridgeProposal != nullptr &&
            bridgeProposal->daughterSphereRadius > 0.0f &&
            probConfig.pca_bridge_future_window_enabled &&
            bridgeProposal->centerSnapUsedAlignedPairFallback &&
            bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
            bridgeProposal->windowBothDaughtersSupported >= 1 &&
            bridgeProposal->windowMissingDaughterCount <= 2 &&
            bridgeProposal->windowParentPersists == 0 &&
            bridgeProposal->windowBestMatchedMinBrightness >=
                std::max(0.0f,
                         probConfig
                             .pca_bridge_future_window_rod_tip_balance_min_brightness);
        appendBridgeCandidate(bridgeCand, !suppressFutureRodTipCenterSlides, false);
        if (suppressFutureRodTipCenterSlides) {
            std::cout << "  [Split Bridge Center Slides Suppressed] "
                      << parentName
                      << " reason=future_supported_rod_tip"
                      << " futureBoth="
                      << bridgeProposal->windowBothDaughtersSupported
                      << " futureMissing="
                      << bridgeProposal->windowMissingDaughterCount
                      << " futureBrightness="
                      << bridgeProposal->windowBestMatchedMinBrightness
                      << std::endl;
        }

        if (bridgeProposalOnly &&
            simulationConfig.celluniverse3_enabled &&
            probConfig.celluniverse3_injected_split_evidence_midpoints_enabled) {
            const cv::Point3f baseAxisVec = bridgeCand.d2Pos - bridgeCand.d1Pos;
            const float baseSep = static_cast<float>(cv::norm(baseAxisVec));
            if (baseSep > 1e-3f) {
                const cv::Point3f baseAxis =
                    baseAxisVec * (1.0f / baseSep);
                const cv::Point3f baseMid =
                    0.5f * (bridgeCand.d1Pos + bridgeCand.d2Pos);
                const float maxMidShift =
                    std::max(0.0f,
                             probConfig
                                 .celluniverse3_injected_split_evidence_midpoint_max_shift_scale) *
                    std::max(srcMaxR, 1.0f);
                const float maxPairSep =
                    std::max(0.0f,
                             probConfig
                                 .celluniverse3_injected_split_evidence_pair_max_sep_scale) *
                    std::max(srcMaxR, 1.0f);
                const auto appendMidpointCandidate =
                    [&](const cv::Point3f &mid,
                        const cv::Point3f &axis,
                        const char *,
                        bool allowSlides) {
                        if (!validPoint(mid) || !validPoint(axis)) return;
                        const float axisNorm = static_cast<float>(cv::norm(axis));
                        if (axisNorm <= 1e-3f) return;
                        if (maxMidShift > 0.0f &&
                            cv::norm(mid - baseMid) > maxMidShift) {
                            return;
                        }
                        const cv::Point3f unitAxis = axis * (1.0f / axisNorm);
                        Candidate cand = bridgeCand;
                        cand.d1Pos = mid - 0.5f * baseSep * unitAxis;
                        cand.d2Pos = mid + 0.5f * baseSep * unitAxis;
                        cand.label = "bridge_primary";
                        appendBridgeCandidate(cand, allowSlides, true);
                    };
                const auto appendPairCandidate =
                    [&](cv::Point3f d1,
                        cv::Point3f d2,
                        const char *) {
                        if (!validPoint(d1) || !validPoint(d2)) return;
                        const cv::Point3f pairVec = d2 - d1;
                        const float pairSep = static_cast<float>(cv::norm(pairVec));
                        if (pairSep <= 1e-3f) return;
                        if (maxPairSep > 0.0f && pairSep > maxPairSep) return;
                        const cv::Point3f mid = 0.5f * (d1 + d2);
                        if (maxMidShift > 0.0f &&
                            cv::norm(mid - baseMid) > maxMidShift) {
                            return;
                        }
                        const float align =
                            pairVec.x * baseAxis.x +
                            pairVec.y * baseAxis.y +
                            pairVec.z * baseAxis.z;
                        if (align < 0.0f) {
                            std::swap(d1, d2);
                        }
                        Candidate cand = bridgeCand;
                        cand.d1Pos = d1;
                        cand.d2Pos = d2;
                        cand.label = "bridge_primary";
                        appendBridgeCandidate(cand, false, true);
                    };

                if (!axisPlace.empty() && axisPlace.front().valid) {
                    appendPairCandidate(axisPlace.front().d1,
                                        axisPlace.front().d2,
                                        "current_axis_place_pair");
                    appendMidpointCandidate(axisPlace.front().midpoint,
                                            baseAxis,
                                            "current_axis_place_mid",
                                            false);
                }
                const bool asymmetricUTunnelFuturePairEvidence =
                    simulationConfig.celluniverse3_enabled &&
                    probConfig.celluniverse3_window_map_asymmetric_u_tunnel_rescue_enabled &&
                    bridgeProposal->cellUniverse3MapPriorEvaluated &&
                    ((bridgeProposal->cellUniverse3MapD1InsideTunnel ? 1 : 0) +
                     (bridgeProposal->cellUniverse3MapD2InsideTunnel ? 1 : 0)) == 1 &&
                    std::max(bridgeProposal->cellUniverse3MapUSupportD1,
                             bridgeProposal->cellUniverse3MapUSupportD2) >=
                        probConfig.celluniverse3_window_map_primary_support_min_u_support &&
                    std::min(bridgeProposal->cellUniverse3MapUSupportD1,
                             bridgeProposal->cellUniverse3MapUSupportD2) <=
                        probConfig.celluniverse3_window_map_asymmetric_u_tunnel_rescue_max_weak_u_support &&
                    bridgeProposal->windowBothDaughtersSupported >=
                        std::max(1,
                                 probConfig.celluniverse3_window_map_asymmetric_u_tunnel_rescue_min_future_both) &&
                    bridgeProposal->windowMissingDaughterCount <=
                        std::max(0,
                                 probConfig.celluniverse3_window_map_asymmetric_u_tunnel_rescue_max_missing) &&
                    bridgeProposal->windowParentPersists == 0 &&
                    bridgeProposal->windowBestMatchedMinBrightness >=
                        probConfig.celluniverse3_window_map_asymmetric_u_tunnel_rescue_min_future_brightness;
                if (bridgeProposal->windowBothDaughtersSupported >= 2 ||
                    asymmetricUTunnelFuturePairEvidence) {
                    appendPairCandidate(bridgeProposal->windowBestMatchedD1Pos,
                                        bridgeProposal->windowBestMatchedD2Pos,
                                        "future_best_pair");
                    appendMidpointCandidate(
                        0.5f * (bridgeProposal->windowBestMatchedD1Pos +
                                bridgeProposal->windowBestMatchedD2Pos),
                        baseAxis,
                        "future_best_mid",
                        false);
                }
                if (bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
                    bridgeProposal->windowBothDaughtersSupported >= 2) {
                    appendPairCandidate(
                        bridgeProposal->windowImmediateMatchedD1Pos,
                        bridgeProposal->windowImmediateMatchedD2Pos,
                        "future_immediate_pair");
                    appendMidpointCandidate(
                        0.5f * (bridgeProposal->windowImmediateMatchedD1Pos +
                                bridgeProposal->windowImmediateMatchedD2Pos),
                        baseAxis,
                        "future_immediate_mid",
                        false);
                }
                if (bridgeProposal->cellUniverse3MapPriorEvaluated) {
                    const cv::Point3f mapAxis =
                        (cv::norm(bridgeProposal->cellUniverse3MapAxis) > 1e-3f)
                            ? bridgeProposal->cellUniverse3MapAxis
                            : baseAxis;
                    appendMidpointCandidate(
                        bridgeProposal->cellUniverse3MapOCenter,
                        baseAxis,
                        "map_o_mid_base_axis",
                        true);
                    appendMidpointCandidate(
                        bridgeProposal->cellUniverse3MapOCenter,
                        mapAxis,
                        "map_o_mid_map_axis",
                        true);
                    appendMidpointCandidate(
                        0.5f * (bridgeProposal->cellUniverse3MapOCenter +
                                bridgeProposal->cellUniverse3MapDCenter),
                        baseAxis,
                        "map_od_mid_base_axis",
                        true);
                    appendMidpointCandidate(
                        0.5f * (bridgeProposal->cellUniverse3MapOCenter +
                                bridgeProposal->cellUniverse3MapDCenter),
                        mapAxis,
                        "map_od_mid_map_axis",
                        true);
                }
            }
        }
        if (bridgeProposal->hasAlternateSeedPair) {
            Candidate altCand;
            altCand.d1Pos = bridgeProposal->altD1Pos;
            altCand.d2Pos = bridgeProposal->altD2Pos;
            altCand.label = "bridge_tip_alt";
            altCand.sphereRadiusOverride = bridgeProposal->daughterSphereRadius;
            appendBridgeCandidate(altCand, false, false);
        }
        const float rodTipAxisPlaceMaxShape =
            std::max(0.0f,
                     probConfig
                         .pca_bridge_future_window_rod_tip_axis_place_max_parent_shape);
        const bool highShapeFutureAlignedRodTipAxisPlace =
            simulationConfig.celluniverse2_enabled &&
            bridgeProposal->daughterSphereRadius > 0.0f &&
            bridgeProposal->gapStartBin <= -8 &&
            bridgeProposal->immediateFutureCenterBacked &&
            bridgeProposal->centerSnapUsedAlignedPairFallback &&
            rodTipAxisPlaceMaxShape > 0.0f &&
            bridgeProposal->parentShapeElongation > rodTipAxisPlaceMaxShape;
        if (bridgeProposal->daughterSphereRadius > 0.0f &&
            !axisPlace.empty() && axisPlace.front().valid &&
            !highShapeFutureAlignedRodTipAxisPlace) {
            Candidate axisCand;
            axisCand.d1Pos = axisPlace.front().d1;
            axisCand.d2Pos = axisPlace.front().d2;
            axisCand.label = "bridge_axis_place";
            axisCand.sphereRadiusOverride = -1.0f;
            bridgeCandidates.push_back(axisCand);
        } else if (highShapeFutureAlignedRodTipAxisPlace &&
                   !axisPlace.empty() && axisPlace.front().valid) {
            std::cout << "  [Split Bridge AxisPlace Skip] " << parentName
                      << " reason=high_shape_future_aligned_rod_tip"
                      << " parentShape="
                      << bridgeProposal->parentShapeElongation
                      << " maxParentShape=" << rodTipAxisPlaceMaxShape
                      << " futureBoth="
                      << bridgeProposal->windowBothDaughtersSupported
                      << " futureMissing="
                      << bridgeProposal->windowMissingDaughterCount
                      << " futureBestMinBrightness="
                      << bridgeProposal->windowBestMatchedMinBrightness
                      << std::endl;
        }
        if (bridgeProposalOnly) {
            candidates.clear();
            candidates.insert(candidates.end(),
                              bridgeCandidates.begin(),
                              bridgeCandidates.end());
        } else {
            candidates.insert(candidates.begin(),
                              bridgeCandidates.begin(),
                              bridgeCandidates.end());
        }
        std::cout << "  [Split Bridge Inject] " << parentName
                  << " only=" << (bridgeProposalOnly ? 1 : 0)
                  << " d1=(" << bridgeCand.d1Pos.x << "," << bridgeCand.d1Pos.y
                  << "," << bridgeCand.d1Pos.z << ")"
                  << " d2=(" << bridgeCand.d2Pos.x << "," << bridgeCand.d2Pos.y
                  << "," << bridgeCand.d2Pos.z << ")"
                  << " alt=" << (bridgeProposal->hasAlternateSeedPair ? 1 : 0)
                  << " altD1=(" << bridgeProposal->altD1Pos.x << ","
                  << bridgeProposal->altD1Pos.y << ","
                  << bridgeProposal->altD1Pos.z << ")"
                  << " altD2=(" << bridgeProposal->altD2Pos.x << ","
                  << bridgeProposal->altD2Pos.y << ","
                  << bridgeProposal->altD2Pos.z << ")"
                  << " elong=" << bridgeProposal->elongation
                  << " gapBins=" << bridgeProposal->gapStartBin << "-"
                  << bridgeProposal->gapEndBin
                  << " sphereRadius=" << bridgeProposal->daughterSphereRadius
                  << " centerSlides=" << bridgeCenterSlideCount
                  << " evidenceMidpoints=" << bridgeEvidenceMidpointCount
                  << " slideRangeScale="
                  << probConfig.celluniverse3_injected_split_center_slide_range_scale
                  << " slideStepScale="
                  << probConfig.celluniverse3_injected_split_center_slide_step_scale
                  << " slideMinStep="
                  << probConfig.celluniverse3_injected_split_center_slide_min_step
                  << std::endl;
    }

    if (lumenProposal != nullptr) {
        Candidate lumenCand;
        lumenCand.d1Pos = lumenProposal->d1Pos;
        lumenCand.d2Pos = lumenProposal->d2Pos;
        lumenCand.label = "cell_lumen_primary";
        if (lumenProposalOnly) {
            candidates.clear();
            candidates.push_back(lumenCand);
        } else {
            candidates.insert(candidates.begin(), lumenCand);
        }
        std::cout << "  [Split CellLumen Inject] " << parentName
                  << " only=" << (lumenProposalOnly ? 1 : 0)
                  << " d1=(" << lumenCand.d1Pos.x << "," << lumenCand.d1Pos.y
                  << "," << lumenCand.d1Pos.z << ")"
                  << " d2=(" << lumenCand.d2Pos.x << "," << lumenCand.d2Pos.y
                  << "," << lumenCand.d2Pos.z << ")"
                  << " sep=" << cv::norm(lumenCand.d1Pos - lumenCand.d2Pos)
                  << " score=" << lumenProposal->elongation
                  << std::endl;
    }

    const int Kmax = std::max(1, probConfig.split_candidates_per_attempt);
    if (static_cast<int>(candidates.size()) > Kmax) {
        candidates.resize(Kmax);
    }

    // --- 4. Evaluate each candidate via a short burn-in ---
    // Save the pre-split state to revert cheaply.
    //
    // When _useBboxCost is true, build a single union bbox + exclusion mask
    // ONCE here covering parent + all candidate seed positions + drift
    // margin. All cost evaluations during burn-in / refine / final gate use
    // this same bbox + mask, so baseline and every candidate are compared on
    // the same voxel set (apples-to-apples). Neighbor positions don't
    // change during the split attempt, so the mask stays valid throughout.
    BoundingBox3D splitBbox;
    if (_useBboxCost) {
        std::vector<cv::Point3f> seedPoints;
        seedPoints.reserve(candidates.size() * 2);
        for (const auto &cand : candidates) {
            seedPoints.push_back(cand.d1Pos);
            seedPoints.push_back(cand.d2Pos);
        }
        const float pointR = _bboxMarginScale * std::max({srcMajor, srcB, srcMinor});
        splitBbox = computeUnionBboxWithPoints({cellIndex}, _bboxMarginScale,
                                               seedPoints, pointR);

        std::cout << "  [Split Bbox Init] " << parentName
                  << " bboxXYZ=(" << splitBbox.xMin << "-" << splitBbox.xMax
                  << ", " << splitBbox.yMin << "-" << splitBbox.yMax
                  << ", " << splitBbox.zMin << "-" << splitBbox.zMax << ")"
                  << " volume=" << splitBbox.volume()
                  << " maskSeedPoints=" << seedPoints.size()
                  << std::endl;

        // Install the shared split union bbox under each daughter
        // candidate name so perturbCell anchors daughters to the SAME
        // union bbox covering both lobes during burn-in + refine.
        // No Voronoi mask installed (cost uses empty mask; see below).
        if (splitBbox.isValid()) {
            _snapBboxes[parentName + "0"] = splitBbox;
            _snapBboxes[parentName + "1"] = splitBbox;
        }
    }

    // No Voronoi mask for split cost eval either — same reasoning as
    // perturbCell: neighbors' synth is constant across candidates and
    // baseline, cancels in all comparisons. Dropping the mask keeps
    // cost accounting honest about abandoned voxels.
    const std::vector<uint8_t> noSplitMask;
    auto evalImageCost = [&](const std::vector<cv::Mat> &synth) -> double {
        if (_useBboxCost) return calculateBboxCost(splitBbox, synth, noSplitMask);
        return _currentCost;  // legacy path: cached after refreshFullCostCache
    };

    const double baselineImageCost = _useBboxCost
        ? calculateBboxCost(splitBbox, _synthFrame, noSplitMask)
        : _currentCost;
    const double baselineOverlap = computeOverlapPenalty(probConfig.overlap_penalty_weight);
    const double baselineTotal = baselineImageCost + baselineOverlap;
    // Non-const: moved into callback copies at the end of the function
    // (std::move on const silently falls back to copy).
    std::vector<cv::Mat> savedSynth = _synthFrame;
    std::vector<double> savedPerSlice = _currentCostPerSlice;
    const double savedCost = _currentCost;
    std::vector<Ellipsoid> savedCells = cells;
    float parentSignalForCandidateGate = 0.0f;
    if (simulationConfig.celluniverse3_enabled && !_realFrame.empty()) {
        const auto parentCandidateBrightness =
            parent.measureBrightnessStats(_realFrame);
        const cv::Point3f parentCandidatePosition(
            parent.getX(), parent.getY(), parent.getZ());
        parentSignalForCandidateGate =
            std::max(0.0f,
                     parentCandidateBrightness.first -
                         backgroundAt(parentCandidatePosition));
    }
    int savedNonTrashCellCount = 0;
    for (const auto &cell : savedCells) {
        if (!cell.isTrash()) ++savedNonTrashCellCount;
    }

    int bestIdx = -1;
    double bestTotal = std::numeric_limits<double>::infinity();
    double bestSelectionScore = std::numeric_limits<double>::infinity();
    std::vector<Ellipsoid> bestCells;
    std::vector<cv::Mat> bestSynth;
    std::vector<double> bestPerSlice;
    double bestImageCost = 0.0;
    cv::Point3f bestSeedD1{0, 0, 0};
    cv::Point3f bestSeedD2{0, 0, 0};
    std::string bestLabel;
    bool bestFutureSupportedRodTipPrimary = false;

    int burnIters = std::max(0, probConfig.split_candidate_burn_in_iterations);
    if (lumenProposal != nullptr && lumenBurnInIterations >= 0) {
        burnIters = std::max(0, lumenBurnInIterations);
    }
    if (bridgeProposalOnly) {
        burnIters = 0;
    }
    std::cout << "  [Split Baseline] " << parentName
              << " imageCost=" << baselineImageCost
              << " overlap=" << baselineOverlap
              << " total=" << baselineTotal
              << " threshold=" << -probConfig.split_cost
              << " nCandidates=" << candidates.size()
              << " burnIters=" << burnIters
              << " bridgeOnly=" << (bridgeProposalOnly ? 1 : 0)
              << " lumenAttempt=" << (lumenProposal != nullptr ? 1 : 0)
              << std::endl;

    // Install tight position sigmas for candidate burn-in.
    //
    // Main-loop position sigmas (x=5, y=5, z=8) let a daughter drift
    // 15-25 voxels across 20 iters, far enough to escape the parent
    // footprint. Scaling by split_burn_in_pos_sigma_scale (0.4 default)
    // restricts burn-in to refinement distances (<10 voxels).
    //
    // Radii are not perturbed anywhere (config sigmas are zero; radii
    // are fit by iterative PCA each frame), so there is no radius-sigma
    // scaling here.
    //
    // Global static state mutation is safe here — single-threaded
    // optimizer, restored on every exit path below.
    const float posScale = std::max(0.0f, probConfig.split_burn_in_pos_sigma_scale);
    PerturbParams savedPerturbX = Ellipsoid::cellConfig.x;
    PerturbParams savedPerturbY = Ellipsoid::cellConfig.y;
    PerturbParams savedPerturbZ = Ellipsoid::cellConfig.z;
    Ellipsoid::cellConfig.x.sigma = savedPerturbX.sigma * posScale;
    Ellipsoid::cellConfig.y.sigma = savedPerturbY.sigma * posScale;
    Ellipsoid::cellConfig.z.sigma = savedPerturbZ.sigma * posScale;

    std::cout << "  [Split Sigmas] " << parentName
              << " posScale=" << posScale
              << " xSigma=" << savedPerturbX.sigma << "->" << Ellipsoid::cellConfig.x.sigma
              << " ySigma=" << savedPerturbY.sigma << "->" << Ellipsoid::cellConfig.y.sigma
              << " zSigma=" << savedPerturbZ.sigma << "->" << Ellipsoid::cellConfig.z.sigma
              << std::endl;

    for (size_t ci = 0; ci < candidates.size(); ++ci) {
        const auto &cand = candidates[ci];
        const bool candIsCellLumenPrior =
            cand.label == "cell_lumen_primary" && lumenUseDedicatedCostGate;
        const bool candIsPcaBridgeOnly =
            bridgeProposalOnly && cand.label == "bridge_primary";
        const bool futureBackedBridgeRescue =
            candIsPcaBridgeOnly &&
            bridgeProposal != nullptr &&
            bridgeProposal->futureWindowSplitRescue;
        const bool futureSupportedPcaBridgeDimBypass =
            candIsPcaBridgeOnly &&
            bridgeProposal != nullptr &&
            probConfig.pca_bridge_future_window_enabled &&
            bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
            bridgeProposal->windowBothDaughtersSupported >=
                std::max(1, probConfig.pca_bridge_future_window_min_both_daughter_support) &&
            bridgeProposal->windowMissingDaughterCount <=
                std::max(0, probConfig.pca_bridge_future_window_max_missing_daughters) &&
            bridgeProposal->windowParentPersists <=
                std::max(0, probConfig.pca_bridge_future_window_max_parent_persists);
        const bool futureSupportedRodTipPrimaryCandidate =
            candIsPcaBridgeOnly &&
            bridgeProposal != nullptr &&
            bridgeProposal->daughterSphereRadius > 0.0f &&
            bridgeProposal->centerSnapUsedAlignedPairFallback &&
            bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
            bridgeProposal->windowBothDaughtersSupported >=
                static_cast<int>(probConfig.split_future_rod_tip_primary_min_future_both) &&
            bridgeProposal->windowMissingDaughterCount <=
                static_cast<int>(probConfig.split_future_rod_tip_primary_max_missing_daughters) &&
            bridgeProposal->windowParentPersists == 0 &&
            bridgeProposal->windowBestMatchedMinBrightness >=
                probConfig.split_future_rod_tip_primary_min_brightness &&
            bridgeProposal->parentDistanceBalance >=
                probConfig.split_future_rod_tip_primary_min_parent_balance;
        const float activeDaughterVolumeScale =
            (candIsCellLumenPrior && lumenDaughterVolumeScale > 0.0f)
                ? lumenDaughterVolumeScale
                : volumeScale;
        Ellipsoid child1 = buildDaughter(parentName + "0", cand.d1Pos, parent,
                                         activeDaughterVolumeScale, srcMajor, srcB, srcMinor);
        Ellipsoid child2 = buildDaughter(parentName + "1", cand.d2Pos, parent,
                                         activeDaughterVolumeScale, srcMajor, srcB, srcMinor);
        auto applySphereRadiusOverride = [](Ellipsoid &daughter, float radius) {
            if (radius <= 0.0f) return;
            const auto &cfg = Ellipsoid::cellConfig;
            const float rA = std::clamp(
                radius,
                static_cast<float>(cfg.minARadius),
                static_cast<float>(cfg.maxARadius));
            const float rB = std::clamp(
                radius,
                static_cast<float>(cfg.maxBRadius > 0.0 ? cfg.minBRadius : cfg.minARadius),
                static_cast<float>(cfg.maxBRadius > 0.0 ? cfg.maxBRadius : cfg.maxARadius));
            const float rC = std::clamp(
                radius,
                static_cast<float>(cfg.minCRadius),
                static_cast<float>(cfg.maxCRadius));
            daughter.setRadii(rA, rB, rC);
        };
        applySphereRadiusOverride(child1, cand.sphereRadiusOverride);
        applySphereRadiusOverride(child2, cand.sphereRadiusOverride);

        // Replace parent with daughters.
        cells.erase(cells.begin() + cellIndex);
        cells.push_back(child1);
        cells.push_back(child2);
        const size_t d1Idx = cells.size() - 2;
        const size_t d2Idx = cells.size() - 1;

        // Render the synth with the new cell configuration (parent
        // removed, daughters added). Under bbox cost, only re-render
        // the z-slices affected by the parent + both daughters —
        // unaffected slices retain their existing render from savedSynth
        // (which is correct because those slices have no parent pixels
        // and the cost bbox doesn't reach them anyway).
        // For legacy full-image mode, do a full render (needed for
        // correct full-image cost).
        if (_useBboxCost) {
            // Compute the z-range touched by parent + both daughters.
            const float pMaxR = std::max({parent.getARadius(),
                                          parent.getBRadius(),
                                          parent.getCRadius()});
            const float d1MaxR = std::max({child1.getARadius(),
                                           child1.getBRadius(),
                                           child1.getCRadius()});
            const float d2MaxR = std::max({child2.getARadius(),
                                           child2.getBRadius(),
                                           child2.getCRadius()});
            const int nSlices = static_cast<int>(z_slices.size());
            const int zLo = std::max(0, static_cast<int>(std::floor(
                std::min({parent.getZ() - pMaxR,
                          child1.getZ() - d1MaxR,
                          child2.getZ() - d2MaxR}))));
            const int zHi = std::min(nSlices - 1, static_cast<int>(std::ceil(
                std::max({parent.getZ() + pMaxR,
                          child1.getZ() + d1MaxR,
                          child2.getZ() + d2MaxR}))));

            // Re-render only affected slices. Unaffected slices retain
            // the savedSynth reference (cv::Mat shallow copy = free).
            const cv::Size shape = getImageShape();
            for (int z = zLo; z <= zHi; ++z) {
                cv::Mat synthImage = makeSynthBackgroundSlice(shape, z);
                const float zf = static_cast<float>(z_slices[z]);
                for (const auto &cell : cells) {
                    const float cmr = std::max({cell.getARadius(),
                                                cell.getBRadius(),
                                                cell.getCRadius()});
                    if (std::abs(zf - cell.getZ()) > cmr) continue;
                    cell.draw(synthImage, simulationConfig, zf);
                }
                _synthFrame[z] = synthImage;
            }
        } else {
            _synthFrame = generateSynthFrame();
            refreshFullCostCache();
        }

        // Short alternating burn-in on each daughter.
        for (int it = 0; it < burnIters; ++it) {
            const size_t target = (it % 2 == 0) ? d1Idx : d2Idx;
            CostCallbackPair cp = perturbCell(target,
                                              probConfig.overlap_penalty_weight,
                                              /*useSignalGuidance=*/false,
                                              /*randomPerturbRadiusRatio=*/1.0f,
                                              /*pcaRefitWellFilledMove=*/false,
                                              /*useSignalMapGuidance=*/false);
            const bool accept = cp.first < 0.0;
            if (splitPerturbDebugPlacements != nullptr) {
                accumulateDebugCellPlacement(*splitPerturbDebugPlacements,
                                             cells[target],
                                             simulationConfig,
                                             splitPerturbDebugBrightness);
                if (splitPerturbDebugPlacementCount != nullptr) {
                    ++(*splitPerturbDebugPlacementCount);
                }
            }
            cp.second(accept);
        }

        // Under bbox flag, _currentCost is stale (perturbCell's bbox path
        // doesn't update it). Use the cached bbox+mask to score the
        // candidate's post-burn-in synth on the same voxel set as baseline.
        const double candImageCost = evalImageCost(_synthFrame);
        const double candOverlap = computeOverlapPenalty(probConfig.overlap_penalty_weight);
        const double candTotal = candImageCost + candOverlap;

        const Ellipsoid &candD1 = cells[d1Idx];
        const Ellipsoid &candD2 = cells[d2Idx];
        const float candDrift1 = static_cast<float>(cv::norm(
            cv::Point3f(candD1.getX(), candD1.getY(), candD1.getZ()) - cand.d1Pos));
        const float candDrift2 = static_cast<float>(cv::norm(
            cv::Point3f(candD2.getX(), candD2.getY(), candD2.getZ()) - cand.d2Pos));
        const cv::Point3f candD1Pos(candD1.getX(), candD1.getY(), candD1.getZ());
        const cv::Point3f candD2Pos(candD2.getX(), candD2.getY(), candD2.getZ());
        const bool bridgeTunnelSeedD1Inside =
            bridgeTunnelContainsPoint(cand.d1Pos);
        const bool bridgeTunnelSeedD2Inside =
            bridgeTunnelContainsPoint(cand.d2Pos);
        const bool bridgeTunnelFinalD1Inside =
            bridgeTunnelContainsPoint(candD1Pos);
        const bool bridgeTunnelFinalD2Inside =
            bridgeTunnelContainsPoint(candD2Pos);
        const float candDaughterSeparation =
            static_cast<float>(cv::norm(candD2Pos - candD1Pos));
        const float futureRodTipPrimaryMinCandidateSeparation =
            std::max(0.0f,
                     probConfig.bio_min_daughter_separation_parent_fraction) *
            std::max(1.0f, srcMaxR);
        const bool futureSupportedRodTipSelectionGuardCandidate =
            candIsPcaBridgeOnly &&
            bridgeProposal != nullptr &&
            bridgeProposal->daughterSphereRadius > 0.0f &&
            probConfig.pca_bridge_future_window_enabled &&
            bridgeProposal->centerSnapUsedAlignedPairFallback &&
            bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
            bridgeProposal->windowBothDaughtersSupported >= 1 &&
            bridgeProposal->windowMissingDaughterCount <= 2 &&
            bridgeProposal->windowParentPersists == 0 &&
            bridgeProposal->windowBestMatchedMinBrightness >=
                std::max(0.0f,
                         probConfig
                             .pca_bridge_future_window_rod_tip_balance_min_brightness);
        const bool futureRodTipPrimaryCollapsedCandidate =
            futureSupportedRodTipSelectionGuardCandidate &&
            futureRodTipPrimaryMinCandidateSeparation > 0.0f &&
            candDaughterSeparation < futureRodTipPrimaryMinCandidateSeparation;
        if (futureRodTipPrimaryCollapsedCandidate) {
            std::cout << "  [Split Cand Future RodTip Separation Gate] "
                      << parentName
                      << " idx=" << ci
                      << " label=" << cand.label
                      << " candSep=" << candDaughterSeparation
                      << " minSep="
                      << futureRodTipPrimaryMinCandidateSeparation
                      << " futureBoth="
                      << bridgeProposal->windowBothDaughtersSupported
                      << " futureMissing="
                      << bridgeProposal->windowMissingDaughterCount
                      << " futureBrightness="
                      << bridgeProposal->windowBestMatchedMinBrightness
                      << " action=reject_candidate"
                      << std::endl;
        }
        const bool bridgeTunnelAsymmetricCandidatePass =
            bridgeTunnelConstraintActive &&
            bridgeProposal != nullptr &&
            probConfig.celluniverse3_window_map_asymmetric_u_tunnel_rescue_enabled &&
            cand.label == "bridge_primary" &&
            ((bridgeTunnelSeedD1Inside && bridgeTunnelFinalD1Inside &&
              !bridgeTunnelSeedD2Inside && !bridgeTunnelFinalD2Inside) ||
             (bridgeTunnelSeedD2Inside && bridgeTunnelFinalD2Inside &&
              !bridgeTunnelSeedD1Inside && !bridgeTunnelFinalD1Inside)) &&
            std::max(bridgeProposal->cellUniverse3MapUSupportD1,
                     bridgeProposal->cellUniverse3MapUSupportD2) >=
                std::max(0.0f,
                         probConfig
                             .celluniverse3_window_map_primary_asymmetric_min_strong_u_support) &&
            std::min(bridgeProposal->cellUniverse3MapUSupportD1,
                     bridgeProposal->cellUniverse3MapUSupportD2) <=
                std::max(0.0f,
                         probConfig
                             .celluniverse3_window_map_asymmetric_u_tunnel_rescue_max_weak_u_support) &&
            bridgeProposal->windowBothDaughtersSupported >=
                std::max(1,
                         probConfig
                             .celluniverse3_window_map_asymmetric_u_tunnel_rescue_min_future_both) &&
            bridgeProposal->windowMissingDaughterCount <=
                std::max(0,
                         probConfig
                             .celluniverse3_window_map_asymmetric_u_tunnel_rescue_max_missing) &&
            bridgeProposal->windowParentPersists == 0 &&
            bridgeProposal->windowBestMatchedMinBrightness >=
                std::max(0.0f,
                         probConfig
                             .celluniverse3_window_map_asymmetric_u_tunnel_rescue_min_future_brightness) &&
            bridgeProposal->cellUniverse3MapRegionPenalty <=
                std::max(0.0f,
                         probConfig
                             .celluniverse3_window_map_asymmetric_u_tunnel_rescue_max_region_penalty) &&
            (!bridgeProposal->centerSnapApplied ||
             bridgeProposal->centerSnapScore <=
                 std::max(0.0f,
                          probConfig
                              .celluniverse3_window_map_asymmetric_u_tunnel_rescue_max_center_snap_score));
        const bool bridgeTunnelOutsideFutureCandidatePass =
            bridgeTunnelConstraintActive &&
            bridgeProposal != nullptr &&
            cand.label == "bridge_primary" &&
            !bridgeTunnelSeedD1Inside &&
            !bridgeTunnelSeedD2Inside &&
            !bridgeTunnelFinalD1Inside &&
            !bridgeTunnelFinalD2Inside &&
            bridgeProposal->windowBothDaughtersSupported >= 2 &&
            bridgeProposal->windowMissingDaughterCount == 0 &&
            bridgeProposal->windowParentPersists == 0 &&
            bridgeProposal->windowBestMatchedMinBrightness >=
                std::max(0.0f,
                         probConfig
                             .pca_bridge_future_window_rod_tip_balance_min_brightness) &&
            bridgeProposal->parentDistanceBalance >=
                std::max(0.0f,
                         probConfig
                             .pca_bridge_future_window_parent_balance_rescue_min) &&
            bridgeProposal->bioSeparationObserved >=
                bridgeProposal->bioSeparationRequired &&
            bridgeProposal->cellUniverse3MapRegionPenalty <= 2.0f;
        const bool bridgeTunnelCandidatePass =
            !bridgeTunnelConstraintActive ||
            (bridgeTunnelSeedD1Inside &&
             bridgeTunnelSeedD2Inside &&
             bridgeTunnelFinalD1Inside &&
             bridgeTunnelFinalD2Inside) ||
            bridgeTunnelAsymmetricCandidatePass ||
            bridgeTunnelOutsideFutureCandidatePass;
        std::cout << "  [Split Cand] " << parentName
                  << " idx=" << ci << "/" << candidates.size()
                  << " label=" << cand.label
                  << " seed1=(" << cand.d1Pos.x << "," << cand.d1Pos.y << "," << cand.d1Pos.z << ")"
                  << " final1=(" << candD1.getX() << "," << candD1.getY() << "," << candD1.getZ() << ")"
                  << " drift1=" << candDrift1
                  << " seed2=(" << cand.d2Pos.x << "," << cand.d2Pos.y << "," << cand.d2Pos.z << ")"
                  << " final2=(" << candD2.getX() << "," << candD2.getY() << "," << candD2.getZ() << ")"
                  << " drift2=" << candDrift2
                  << " total=" << candTotal
                  << " (image=" << candImageCost << " overlap=" << candOverlap << ")"
                  << std::endl;
        if (bridgeTunnelAsymmetricCandidatePass ||
            bridgeTunnelOutsideFutureCandidatePass) {
            std::cout << "  [Split Cand Tunnel Gate] " << parentName
                      << " idx=" << ci
                      << " label=" << cand.label
                      << " seedD1Inside=" << (bridgeTunnelSeedD1Inside ? 1 : 0)
                      << " seedD2Inside=" << (bridgeTunnelSeedD2Inside ? 1 : 0)
                      << " finalD1Inside=" << (bridgeTunnelFinalD1Inside ? 1 : 0)
                      << " finalD2Inside=" << (bridgeTunnelFinalD2Inside ? 1 : 0)
                      << " mapUSupportD1="
                      << bridgeProposal->cellUniverse3MapUSupportD1
                      << " mapUSupportD2="
                      << bridgeProposal->cellUniverse3MapUSupportD2
                      << " futureBoth="
                      << bridgeProposal->windowBothDaughtersSupported
                      << " futureMissing="
                      << bridgeProposal->windowMissingDaughterCount
                      << " futureBrightness="
                      << bridgeProposal->windowBestMatchedMinBrightness
                      << " action="
                      << (bridgeTunnelOutsideFutureCandidatePass
                              ? "allow_outside_future_pair"
                              : "allow_asymmetric_u_tunnel")
                      << std::endl;
        }
        if (!bridgeTunnelCandidatePass) {
            std::cout << "  [Split Cand Tunnel Gate] " << parentName
                      << " idx=" << ci
                      << " label=" << cand.label
                      << " seedD1Inside=" << (bridgeTunnelSeedD1Inside ? 1 : 0)
                      << " seedD2Inside=" << (bridgeTunnelSeedD2Inside ? 1 : 0)
                      << " finalD1Inside=" << (bridgeTunnelFinalD1Inside ? 1 : 0)
                      << " finalD2Inside=" << (bridgeTunnelFinalD2Inside ? 1 : 0)
                      << " tunnelBoxes="
                      << (bridgeProposal != nullptr
                              ? bridgeProposal
                                    ->cellUniverse3MapTunnelFlatIndices.size()
                              : 0)
                      << " action=reject_candidate"
                      << std::endl;
        }

        // Per-candidate edge brightness pre-filter. A daughter whose
        // position in the real image is below the edge_too_dim threshold
        // would fail the bridge gate anyway — filter it HERE so it can't
        // win the cost comparison and block a better candidate.
        //
        // This is the same biological signal as edge_too_dim (is there a
        // real cell at the daughter's position?) applied earlier in the
        // pipeline. Catches the f11 1f2ed pattern where a Z-direction
        // candidate placed d2 at z=57 (brightness 0.029) and won on cost
        // over the correct Y-direction candidate (daughters on bright
        // lobes, brightness >0.10). Without this filter, the Z-candidate
        // won, hit edge_too_dim at the bridge, and the whole split was
        // rejected — 1 frame late.
        //
        // Measure: average real-image brightness in a small 3D neighborhood
        // around each daughter center (3×3×3 voxels). Cheap and local.
        const float kMinDaughterBright =
            (candIsCellLumenPrior && lumenMinEdgeBrightness >= 0.0f)
                ? lumenMinEdgeBrightness
                : probConfig.bio_bridge_min_edge_brightness_absolute;
        const float activeMinDaughterBright = candIsPcaBridgeOnly
            ? probConfig.split_pca_bridge_edge_brightness_scale * kMinDaughterBright
            : kMinDaughterBright;
        auto measureLocalBrightness = [&](float cx, float cy, float cz) -> float {
            const int ix = static_cast<int>(std::round(cx));
            const int iy = static_cast<int>(std::round(cy));
            const int iz = static_cast<int>(std::round(cz));
            float sum = 0.0f;
            int cnt = 0;
            for (int dz = -1; dz <= 1; ++dz) {
                const int zz = iz + dz;
                if (zz < 0 || zz >= static_cast<int>(_realFrame.size())) continue;
                const cv::Mat &sl = _realFrame[zz];
                if (sl.type() != CV_32F) continue;
                for (int dy = -1; dy <= 1; ++dy) {
                    const int yy = iy + dy;
                    if (yy < 0 || yy >= sl.rows) continue;
                    const float *row = sl.ptr<float>(yy);
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int xx = ix + dx;
                        if (xx < 0 || xx >= sl.cols) continue;
                        sum += row[xx];
                        ++cnt;
                    }
                }
            }
            return (cnt > 0) ? (sum / cnt) : 0.0f;
        };

        const float d1LocalBright = measureLocalBrightness(
            candD1.getX(), candD1.getY(), candD1.getZ());
        const float d2LocalBright = measureLocalBrightness(
            candD2.getX(), candD2.getY(), candD2.getZ());
        const bool bothDaughtersBright =
            (d1LocalBright >= activeMinDaughterBright) &&
            (d2LocalBright >= activeMinDaughterBright);
        const bool cellUniverse3SignalCenterFutureCandidate =
            simulationConfig.celluniverse3_enabled &&
            candIsPcaBridgeOnly &&
            bridgeProposal != nullptr &&
            bridgeProposal->signalCenterScore >= 0.0f &&
            (futureBackedBridgeRescue || futureSupportedPcaBridgeDimBypass);
        const float signalCenterCurrentBackground = std::max(
            backgroundAt(cv::Point3f(
                candD1.getX(), candD1.getY(), candD1.getZ())),
            backgroundAt(cv::Point3f(
                candD2.getX(), candD2.getY(), candD2.getZ())));
        const float signalCenterCurrentDensityFloor =
            signalCenterCurrentBackground +
            std::max(
                std::max(0.0f,
                         probConfig
                             .celluniverse3_signal_center_future_density_floor_abs),
                std::max(
                    0.0f,
                    probConfig
                        .celluniverse3_signal_center_future_density_floor_parent_fraction) *
                    parentSignalForCandidateGate);
        const bool signalCenterCurrentDensityFloorPass =
            !cellUniverse3SignalCenterFutureCandidate ||
            std::min(d1LocalBright, d2LocalBright) >=
                signalCenterCurrentDensityFloor;
        const bool futureWindowDimBypass =
            (futureBackedBridgeRescue || futureSupportedPcaBridgeDimBypass) &&
            !bothDaughtersBright &&
            signalCenterCurrentDensityFloorPass;
        const float dimNearMissFloor =
            probConfig.split_future_near_dim_bypass_min_edge_fraction *
            activeMinDaughterBright;
        const bool futureWindowNearDimBypass =
            candIsPcaBridgeOnly &&
            bridgeProposal != nullptr &&
            bridgeProposal->bioSeparationSoftRescued &&
            bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
            bridgeProposal->windowBothDaughtersSupported >=
                static_cast<int>(probConfig.split_future_near_dim_bypass_min_future_both) &&
            bridgeProposal->windowMissingDaughterCount <=
                static_cast<int>(probConfig.split_future_near_dim_bypass_max_missing_daughters) &&
            bridgeProposal->windowParentPersists == 0 &&
            bridgeProposal->windowBestMatchedMinBrightness >=
                probConfig.split_future_near_dim_bypass_min_brightness &&
            bridgeProposal->parentDistanceBalance >=
                probConfig.split_future_near_dim_bypass_min_parent_balance &&
            std::min(d1LocalBright, d2LocalBright) >= dimNearMissFloor &&
            !bothDaughtersBright;
        const bool cellUniverse3WindowMapDimBypass =
            candIsPcaBridgeOnly &&
            bridgeProposal != nullptr &&
            bridgeProposal->cellUniverse3MapProposal &&
            bridgeProposal->cellUniverse3MapPriorConfident &&
            !bothDaughtersBright;

        // Quick valley pre-filter: project the parent's pixel cloud onto
        // the d1→d2 axis and measure gap vs max(edges). Same logic as the
        // final bridge gate but computed per-candidate BEFORE cost ranking.
        // Candidates with no valley (valleyFromBright > 0.85) or dim
        // daughters are filtered out so they can't win on cost and block
        // better candidates. The full bridge runs again on the winner
        // after refine+refit (Pass 2).
        float candValleyFromBright = 1.0f;
        {
            const cv::Point3f d1Pos(candD1.getX(), candD1.getY(), candD1.getZ());
            const cv::Point3f d2Pos(candD2.getX(), candD2.getY(), candD2.getZ());
            const cv::Point3f axVec = d2Pos - d1Pos;
            const float axLen = static_cast<float>(cv::norm(axVec));
            if (axLen > 1e-3f) {
                const cv::Point3f axDir(axVec.x/axLen, axVec.y/axLen, axVec.z/axLen);
                const cv::Point3f mid(
                    0.5f*(d1Pos.x+d2Pos.x), 0.5f*(d1Pos.y+d2Pos.y), 0.5f*(d1Pos.z+d2Pos.z));
                const float halfLen = 0.5f * axLen;
                const float gapHalf = std::max(0.15f * halfLen, 1.0f);

                // Match the final-bridge slab-min logic so the pre-filter
                // doesn't reject candidates whose mean-gap is artifact-
                // inflated but whose min cross-section is actually dark.
                static constexpr int kPreFilterSlabs = 5;
                std::array<double, kPreFilterSlabs> slabSum{};
                std::array<int,    kPreFilterSlabs> slabCnt{};
                const float slabW = (2.0f * gapHalf) / static_cast<float>(kPreFilterSlabs);
                double gapSum=0, e1Sum=0, e2Sum=0;
                int gapN=0, e1N=0, e2N=0;
                for (const auto &bp : pixels) {
                    const float dx = bp.pos.x - mid.x;
                    const float dy = bp.pos.y - mid.y;
                    const float dz = bp.pos.z - mid.z;
                    const float proj = dx*axDir.x + dy*axDir.y + dz*axDir.z;
                    if (std::abs(proj) > 1.5f*halfLen) continue;
                    if (std::abs(proj) < gapHalf) {
                        gapSum += bp.weight; ++gapN;
                        int bin = (slabW > 0.0f)
                            ? static_cast<int>((proj + gapHalf) / slabW)
                            : 0;
                        if (bin < 0) bin = 0;
                        if (bin >= kPreFilterSlabs) bin = kPreFilterSlabs - 1;
                        slabSum[bin] += bp.weight;
                        slabCnt[bin] += 1;
                    } else if (proj < -gapHalf && proj > -halfLen*1.1f) {
                        e1Sum += bp.weight; ++e1N;
                    } else if (proj > gapHalf && proj < halfLen*1.1f) {
                        e2Sum += bp.weight; ++e2N;
                    }
                }
                const float gBMean = (gapN>0) ? static_cast<float>(gapSum/gapN) : 0.0f;
                const int minPxPerSlab = std::max(3, gapN / (kPreFilterSlabs * 3));
                float gBMinSlab = std::numeric_limits<float>::infinity();
                int winSlab = -1;
                for (int i = 0; i < kPreFilterSlabs; ++i) {
                    if (slabCnt[i] >= minPxPerSlab) {
                        const float b = static_cast<float>(slabSum[i])
                                        / static_cast<float>(slabCnt[i]);
                        if (b < gBMinSlab) { gBMinSlab = b; winSlab = i; }
                    }
                }
                const float gB = (winSlab >= 0) ? gBMinSlab : gBMean;
                const float e1B = (e1N>0) ? static_cast<float>(e1Sum/e1N) : 0.0f;
                const float e2B = (e2N>0) ? static_cast<float>(e2Sum/e2N) : 0.0f;
                const float maxE = std::max(e1B, e2B);
                if (maxE > 1e-6f) candValleyFromBright = gB / maxE;
            }
        }
        const float valleyLimit =
            (candIsCellLumenPrior && lumenPrefilterMaxValleyRatio >= 0.0f)
                ? lumenPrefilterMaxValleyRatio
                : probConfig.bio_bridge_max_valley_ratio;
        const bool bypassPcaBridgeValley =
            candIsPcaBridgeOnly && !probConfig.pca_bridge_require_valley;
        const bool lumenPrefilterValleyIsSoft =
            candIsCellLumenPrior && lumenUseDedicatedCostGate && lumenSoftGateEnabled;
        const bool candPassesPreFilter = (bothDaughtersBright ||
                                          futureWindowDimBypass  ||
                                          futureWindowNearDimBypass ||
                                          cellUniverse3WindowMapDimBypass) &&
                                         (bypassPcaBridgeValley ||
                                          lumenPrefilterValleyIsSoft ||
                                          candValleyFromBright < valleyLimit);

        if (!candPassesPreFilter) {
            std::cout << "  [Split Cand PreFilter] " << parentName
                      << " idx=" << ci << " label=" << cand.label
                      << " d1Bright=" << d1LocalBright
                      << " d2Bright=" << d2LocalBright
                      << " minBright=" << activeMinDaughterBright
                      << " signalCenterFloor="
                      << signalCenterCurrentDensityFloor
                      << " valley=" << candValleyFromBright
                      << (bothDaughtersBright ? "" : " EDGE_DIM")
                      << (cellUniverse3SignalCenterFutureCandidate
                              ? " SIGNAL_CENTER_FUTURE" : "")
                      << (!signalCenterCurrentDensityFloorPass
                              ? " SIGNAL_CENTER_CURRENT_DENSITY_FLOOR" : "")
                      << (futureWindowDimBypass ? " FUTURE_DIM_BYPASS" : "")
                      << (futureWindowNearDimBypass ? " FUTURE_NEAR_DIM_BYPASS" : "")
                      << (cellUniverse3WindowMapDimBypass ? " WINDOW_MAP_DIM_BYPASS" : "")
                      << (bypassPcaBridgeValley ? " VALLEY_BYPASS" : "")
                      << (candValleyFromBright >= valleyLimit && !lumenPrefilterValleyIsSoft && !bypassPcaBridgeValley
                              ? " NO_VALLEY" : "")
                      << (candValleyFromBright >= valleyLimit && lumenPrefilterValleyIsSoft
                              ? " SOFT_VALLEY" : "")
                      << std::endl;
        }

        const double futureRodTipPrimarySelectionBonus =
            futureSupportedRodTipPrimaryCandidate
                ? std::max(static_cast<double>(
                               probConfig
                                   .split_future_rod_tip_primary_selection_bonus_abs),
                           static_cast<double>(
                               probConfig
                                   .split_future_rod_tip_primary_selection_bonus_fraction) *
                               baselineImageCost)
                : 0.0;
        bool cellUniverse3WindowMapCandidatePass = true;
        float cellUniverse3WindowMapCandidateMidShiftUnits = 0.0f;
        float cellUniverse3WindowMapCandidateSepFraction = 1.0f;
        float cellUniverse3WindowMapCandidateAxisAlignment = 1.0f;
        if (simulationConfig.celluniverse3_enabled &&
            probConfig.celluniverse3_window_map_candidate_gate_enabled &&
            candIsPcaBridgeOnly &&
            bridgeProposal != nullptr &&
            bridgeProposal->cellUniverse3MapProposal &&
            bridgeProposal->cellUniverse3MapPriorConfident) {
            const cv::Point3f baseD1 = bridgeProposal->d1Pos;
            const cv::Point3f baseD2 = bridgeProposal->d2Pos;
            const cv::Point3f baseMid = 0.5f * (baseD1 + baseD2);
            const cv::Point3f candMid = 0.5f * (cand.d1Pos + cand.d2Pos);
            const cv::Point3f baseVec = baseD2 - baseD1;
            const cv::Point3f candVec = cand.d2Pos - cand.d1Pos;
            const float baseSep = static_cast<float>(cv::norm(baseVec));
            const float candSep = static_cast<float>(cv::norm(candVec));
            const float normScale = std::max(1.0f, srcMaxR);
            cellUniverse3WindowMapCandidateMidShiftUnits =
                static_cast<float>(cv::norm(candMid - baseMid)) / normScale;
            if (baseSep > 1e-3f && candSep > 1e-3f) {
                cellUniverse3WindowMapCandidateSepFraction =
                    std::min(candSep / baseSep, baseSep / candSep);
                const cv::Point3f baseAxis = baseVec * (1.0f / baseSep);
                const cv::Point3f candAxis = candVec * (1.0f / candSep);
                cellUniverse3WindowMapCandidateAxisAlignment = std::abs(
                    baseAxis.x * candAxis.x +
                    baseAxis.y * candAxis.y +
                    baseAxis.z * candAxis.z);
            } else {
                cellUniverse3WindowMapCandidateSepFraction = 0.0f;
                cellUniverse3WindowMapCandidateAxisAlignment = 0.0f;
            }
            const float maxMidShift =
                std::max(
                    0.0f,
                    probConfig
                        .celluniverse3_window_map_candidate_max_midpoint_shift_scale);
            const float minSepFraction =
                std::clamp(
                    probConfig.celluniverse3_window_map_candidate_min_sep_fraction,
                    0.0f,
                    1.0f);
            const float minAxisAlignment =
                std::clamp(
                    probConfig
                        .celluniverse3_window_map_candidate_min_axis_alignment,
                    0.0f,
                    1.0f);
            cellUniverse3WindowMapCandidatePass =
                cellUniverse3WindowMapCandidateMidShiftUnits <= maxMidShift &&
                cellUniverse3WindowMapCandidateSepFraction >= minSepFraction &&
                cellUniverse3WindowMapCandidateAxisAlignment >= minAxisAlignment;
            if (!cellUniverse3WindowMapCandidatePass) {
                std::cout << "  [Split Cand WindowMap Gate] " << parentName
                          << " idx=" << ci
                          << " label=" << cand.label
                          << " midShiftUnits="
                          << cellUniverse3WindowMapCandidateMidShiftUnits
                          << " maxMidShift=" << maxMidShift
                          << " sepFraction="
                          << cellUniverse3WindowMapCandidateSepFraction
                          << " minSepFraction=" << minSepFraction
                          << " axisAlignment="
                          << cellUniverse3WindowMapCandidateAxisAlignment
                          << " minAxisAlignment=" << minAxisAlignment
                          << " mapUSupportD1="
                          << bridgeProposal->cellUniverse3MapUSupportD1
                          << " mapUSupportD2="
                          << bridgeProposal->cellUniverse3MapUSupportD2
                          << " action=reject_candidate"
                          << std::endl;
            }
        }
        bool cellUniverse3CleanSignalCandidatePass = true;
        float cellUniverse3CleanSignalCandidateMidShiftUnits = 0.0f;
        float cellUniverse3CleanSignalCandidateSepFraction = 1.0f;
        float cellUniverse3CleanSignalCandidateAxisAlignment = 1.0f;
        if (cleanSignalCandidateGateActive &&
            candIsPcaBridgeOnly &&
            bridgeProposal != nullptr) {
            const cv::Point3f baseD1 = cleanSignalCandidateBaseD1;
            const cv::Point3f baseD2 = cleanSignalCandidateBaseD2;
            const cv::Point3f baseMid = 0.5f * (baseD1 + baseD2);
            const cv::Point3f candMid = 0.5f * (candD1Pos + candD2Pos);
            const cv::Point3f baseVec = baseD2 - baseD1;
            const cv::Point3f candVec = candD2Pos - candD1Pos;
            const float baseSep = static_cast<float>(cv::norm(baseVec));
            const float candSep = static_cast<float>(cv::norm(candVec));
            const float normScale = std::max(1.0f, srcMaxR);
            cellUniverse3CleanSignalCandidateMidShiftUnits =
                static_cast<float>(cv::norm(candMid - baseMid)) / normScale;
            if (baseSep > 1e-3f && candSep > 1e-3f) {
                cellUniverse3CleanSignalCandidateSepFraction =
                    std::min(candSep / baseSep, baseSep / candSep);
                const cv::Point3f baseAxis = baseVec * (1.0f / baseSep);
                const cv::Point3f candAxis = candVec * (1.0f / candSep);
                cellUniverse3CleanSignalCandidateAxisAlignment = std::abs(
                    baseAxis.x * candAxis.x +
                    baseAxis.y * candAxis.y +
                    baseAxis.z * candAxis.z);
            } else {
                cellUniverse3CleanSignalCandidateSepFraction = 0.0f;
                cellUniverse3CleanSignalCandidateAxisAlignment = 0.0f;
            }
            const float maxMidShift =
                std::max(
                    0.0f,
                    probConfig
                        .celluniverse3_clean_signal_candidate_max_midpoint_shift_scale);
            const float minSepFraction =
                std::clamp(
                    probConfig
                        .celluniverse3_clean_signal_candidate_min_sep_fraction,
                    0.0f,
                    1.0f);
            const float minAxisAlignment =
                std::clamp(
                    probConfig
                        .celluniverse3_clean_signal_candidate_min_axis_alignment,
                    0.0f,
                    1.0f);
            cellUniverse3CleanSignalCandidatePass =
                cellUniverse3CleanSignalCandidateMidShiftUnits <=
                    maxMidShift &&
                cellUniverse3CleanSignalCandidateSepFraction >=
                    minSepFraction &&
                cellUniverse3CleanSignalCandidateAxisAlignment >=
                    minAxisAlignment;
            if (!cellUniverse3CleanSignalCandidatePass) {
                std::cout << "  [Split Cand CleanSignal Gate] "
                          << parentName
                          << " idx=" << ci
                          << " label=" << cand.label
                          << " midShiftUnits="
                          << cellUniverse3CleanSignalCandidateMidShiftUnits
                          << " maxMidShift=" << maxMidShift
                          << " sepFraction="
                          << cellUniverse3CleanSignalCandidateSepFraction
                          << " minSepFraction=" << minSepFraction
                          << " axisAlignment="
                          << cellUniverse3CleanSignalCandidateAxisAlignment
                          << " minAxisAlignment=" << minAxisAlignment
                          << " baseSep=" << baseSep
                          << " candSep=" << candSep
                          << " action=reject_candidate"
                          << std::endl;
            }
        }
        const double candSelectionScore =
            candTotal - futureRodTipPrimarySelectionBonus;

        if (candPassesPreFilter &&
            cellUniverse3WindowMapCandidatePass &&
            cellUniverse3CleanSignalCandidatePass &&
            bridgeTunnelCandidatePass &&
            !futureRodTipPrimaryCollapsedCandidate &&
            candSelectionScore < bestSelectionScore) {
            bestTotal = candTotal;
            bestSelectionScore = candSelectionScore;
            bestIdx = static_cast<int>(ci);
            // Move instead of copy — cells, _synthFrame, _currentCostPerSlice
            // are immediately overwritten from savedCells/savedSynth/savedPerSlice
            // below, so we can steal their contents here.
            bestCells = std::move(cells);
            bestSynth = std::move(_synthFrame);
            bestPerSlice = std::move(_currentCostPerSlice);
            bestImageCost = candImageCost;
            bestSeedD1 = cand.d1Pos;
            bestSeedD2 = cand.d2Pos;
            bestLabel = cand.label;
            bestFutureSupportedRodTipPrimary =
                futureSupportedRodTipPrimaryCandidate;
        }

        // Revert to pre-split state for the next candidate.
        cells = savedCells;
        _synthFrame = savedSynth;
        _currentCost = savedCost;
        _currentCostPerSlice = savedPerSlice;
    }

    if (bestIdx < 0) {
        Ellipsoid::cellConfig.x = savedPerturbX;
        Ellipsoid::cellConfig.y = savedPerturbY;
        Ellipsoid::cellConfig.z = savedPerturbZ;
        restoreLiveParent();
        return {0.0, noop};
    }

    // Log which candidate won the burn-in competition.
    const double preCostDiff = bestTotal - baselineTotal;
    std::cout << "  [Split Winner] " << parentName
              << " bestIdx=" << bestIdx << "/" << candidates.size()
              << " label=" << bestLabel
              << " preCostDiff=" << preCostDiff
              << " bestTotal=" << bestTotal
              << " baseline=" << baselineTotal
              << " seed1=(" << bestSeedD1.x << "," << bestSeedD1.y << "," << bestSeedD1.z << ")"
              << " seed2=(" << bestSeedD2.x << "," << bestSeedD2.y << "," << bestSeedD2.z << ")"
              << std::endl;

    // --- 4b. Final refine burn-in on the winning candidate ---
    // The candidate loop runs a short (~20 iter) burn-in per candidate so
    // the K=5 comparison is cheap. Now that we've picked a winner, give it
    // an extra refine pass with the same tight sigmas so the chosen
    // daughters can settle before bio/cost gates fire. This runs on the
    // best candidate's state (reinstalled), and the post-refine state is
    // re-captured as bestCells / bestSynth / etc.
    int refineIters = std::max(0, probConfig.split_final_refine_iterations);
    if (lumenProposal != nullptr && lumenRefineIterations >= 0) {
        refineIters = std::max(0, lumenRefineIterations);
    }
    if (bridgeProposalOnly) {
        refineIters = 0;
    }
    const int daughterRefitIters = std::max(0, probConfig.split_daughter_refit_iterations);
    if (refineIters > 0 || daughterRefitIters > 0) {
        // Reinstall the winning candidate's state.
        cells = bestCells;
        _synthFrame = bestSynth;
        _currentCostPerSlice = bestPerSlice;
        _currentCost = bestImageCost;

        const size_t d1IdxRefine = cells.size() - 2;
        const size_t d2IdxRefine = cells.size() - 1;
        const cv::Point3f preRefineD1(cells[d1IdxRefine].getX(),
                                        cells[d1IdxRefine].getY(),
                                        cells[d1IdxRefine].getZ());
        const cv::Point3f preRefineD2(cells[d2IdxRefine].getX(),
                                        cells[d2IdxRefine].getY(),
                                        cells[d2IdxRefine].getZ());
        // Pre-refine baseline: bestImageCost was set from candidate's
        // post-burn-in cost (bbox or full per flag), reinstalled into
        // _currentCost above for the legacy path.
        const double preRefineTotal = bestImageCost +
            computeOverlapPenalty(probConfig.overlap_penalty_weight);
        std::vector<Ellipsoid> preRefineBestCells = cells;
        std::vector<cv::Mat> preRefineBestSynth = _synthFrame;
        std::vector<double> preRefineBestPerSlice = _currentCostPerSlice;
        const double preRefineBestImageCost = bestImageCost;

        int refineAccepts = 0;
        for (int it = 0; it < refineIters; ++it) {
            const size_t target = (it % 2 == 0) ? d1IdxRefine : d2IdxRefine;
            CostCallbackPair cp = perturbCell(target,
                                              probConfig.overlap_penalty_weight,
                                              /*useSignalGuidance=*/false,
                                              /*randomPerturbRadiusRatio=*/1.0f,
                                              /*pcaRefitWellFilledMove=*/false,
                                              /*useSignalMapGuidance=*/false);
            const bool accept = cp.first < 0.0;
            if (splitPerturbDebugPlacements != nullptr) {
                accumulateDebugCellPlacement(*splitPerturbDebugPlacements,
                                             cells[target],
                                             simulationConfig,
                                             splitPerturbDebugBrightness);
                if (splitPerturbDebugPlacementCount != nullptr) {
                    ++(*splitPerturbDebugPlacementCount);
                }
            }
            if (accept) ++refineAccepts;
            cp.second(accept);
        }

        // --- A1: per-daughter PCA radius refit ---
        // After burn-in + positional refine pins daughter centers, each
        // daughter's radii (inherited from parent as 0.794 * src) are
        // still generic — a real daughter is usually smaller and may have
        // different aspect. Run a short PCA shape fit on each daughter,
        // using its built-in radii as the FIXED mask (same snap-mask
        // pattern as the per-frame shape fit), with Voronoi exclusion so
        // the sibling daughter and every other cell are claimants.
        //
        // Clamp fitted radii to [min * built, max * built] per axis:
        //   floor (min_fraction × built) — phantom daughter can't collapse
        //   ceiling (max_fraction × built) — newborn daughter can't bloat
        //     past ~1.1× built due to immature sibling-Voronoi boundary
        //     absorbing neighbor/halo pixels during refit.
        if (daughterRefitIters > 0) {
            const bool winningCleanFutureSupportedPcaBridge =
                bridgeProposalOnly &&
                bestLabel == "bridge_primary" &&
                bridgeProposal != nullptr &&
                bridgeProposal->gapStartBin >= 0 &&
                bridgeProposal->gapEndBin >= 0 &&
                probConfig.pca_bridge_future_window_enabled &&
                bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
                bridgeProposal->windowBothDaughtersSupported >=
                    std::max(1, probConfig.pca_bridge_future_window_min_both_daughter_support) &&
                bridgeProposal->windowMissingDaughterCount <=
                    std::max(0, probConfig.pca_bridge_future_window_max_missing_daughters) &&
                bridgeProposal->windowParentPersists <=
                    std::max(0, probConfig.pca_bridge_future_window_max_parent_persists);
            const float futureHalfspaceSnapLimit = std::max(
                probConfig.pca_bridge_future_window_match_distance *
                    probConfig.split_future_halfspace_snap_distance_scale,
                probConfig.pca_bridge_future_window_match_distance +
                    probConfig.pca_bridge_future_window_match_distance_per_frame);
            const bool winningStrongCleanFutureHalfspace =
                winningCleanFutureSupportedPcaBridge &&
                bridgeProposal->parentDistanceBalance >=
                    probConfig.split_clean_future_halfspace_min_parent_balance &&
                bridgeProposal->centerSnapMaxSeedDistance <= futureHalfspaceSnapLimit;
            const bool winningFutureBackedBridgeRescue =
                bridgeProposalOnly &&
                bestLabel == "bridge_primary" &&
                bridgeProposal != nullptr &&
                ((bridgeProposal->futureWindowSplitRescue &&
                  bridgeProposal->parentShapeElongation >=
                      probConfig.pca_bridge_future_window_min_parent_shape_for_cost_rescue) ||
                 winningStrongCleanFutureHalfspace);
            const float minFrac = std::max(0.0f, std::min(1.0f,
                probConfig.split_daughter_refit_min_radius_fraction));
            const float maxFrac = std::max(1.0f,
                probConfig.split_daughter_refit_max_radius_fraction);
            const bool bestUsesSphereOverride =
                bridgeProposalOnly &&
                (bestLabel == "bridge_primary" || bestLabel == "bridge_tip_alt") &&
                bridgeProposal != nullptr &&
                bridgeProposal->daughterSphereRadius > 0.0f;
            const bool winningFutureSupportedRodTipClampBypass =
                bestUsesSphereOverride &&
                bestLabel == "bridge_primary" &&
                probConfig.pca_bridge_future_window_enabled &&
                bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
                bridgeProposal->windowBothDaughtersSupported >= 1 &&
                bridgeProposal->windowMissingDaughterCount <= 2 &&
                bridgeProposal->windowParentPersists == 0 &&
                bridgeProposal->windowBestMatchedMinBrightness >=
                    std::max(0.0f,
                             probConfig
                                 .pca_bridge_future_window_rod_tip_balance_min_brightness);
            const bool lockFutureRodTipDaughterPosition =
                winningFutureSupportedRodTipClampBypass &&
                bridgeProposal->centerSnapUsedAlignedPairFallback;
            const float cleanFuturePcaBridgePositionLockSnapLimit =
                std::max(
                    futureHalfspaceSnapLimit,
                    std::max(
                        probConfig.split_clean_future_position_lock_min_snap_scale,
                        std::max(
                            0.0f,
                            probConfig
                                .pca_bridge_future_window_pca_snap_max_radius_scale)) *
                        std::max(1.0f, srcMaxR));
            const bool lockCleanFuturePcaBridgeDaughterPosition =
                simulationConfig.celluniverse2_enabled &&
                winningCleanFutureSupportedPcaBridge &&
                bestLabel == "bridge_primary" &&
                bridgeProposal != nullptr &&
                bridgeProposal->centerSnapUsedAlignedPairFallback &&
                bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
                bridgeProposal->windowMissingDaughterCount == 0 &&
                bridgeProposal->windowParentPersists == 0 &&
                bridgeProposal->windowBestMatchedMinBrightness >=
                    probConfig.split_immediate_pca_continuation_lock_min_brightness &&
                bridgeProposal->parentShapeElongation >=
                    probConfig.split_immediate_pca_continuation_min_parent_shape &&
                bridgeProposal->parentDistanceBalance >=
                    probConfig.split_immediate_pca_continuation_min_parent_balance &&
                bridgeProposal->parentDistanceBalance <=
                    probConfig.split_immediate_pca_continuation_max_parent_balance &&
                bridgeProposal->centerSnapMaxSeedDistance <=
                    cleanFuturePcaBridgePositionLockSnapLimit;
            const bool lockTwoFrameAlignedFuturePcaBridgeDaughterPosition =
                simulationConfig.celluniverse2_enabled &&
                winningCleanFutureSupportedPcaBridge &&
                bestLabel == "bridge_primary" &&
                bridgeProposal != nullptr &&
                bridgeProposal->centerSnapUsedAlignedPairFallback &&
                bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
                bridgeProposal->windowBothDaughtersSupported >=
                    std::max(2, probConfig.pca_bridge_future_window_min_both_daughter_support) &&
                bridgeProposal->windowMissingDaughterCount == 0 &&
                bridgeProposal->windowParentPersists == 0 &&
                bridgeProposal->windowBestMatchedMinBrightness >=
                    probConfig.split_one_frame_aligned_pca_continuation_lock_min_brightness &&
                bridgeProposal->parentShapeElongation >=
                    probConfig.split_one_frame_aligned_pca_continuation_min_parent_shape &&
                bridgeProposal->parentDistanceBalance >=
                    probConfig.split_one_frame_aligned_pca_continuation_min_parent_balance &&
                bridgeProposal->centerSnapMaxSeedDistance <=
                    std::max(cleanFuturePcaBridgePositionLockSnapLimit,
                             probConfig
                                 .split_one_frame_aligned_pca_continuation_snap_scale *
                                 std::max(1.0f, srcMaxR));
            const float sourceLongR = std::max({srcMajor, srcB, srcMinor});
            const float sourceShortR = std::min({srcMajor, srcB, srcMinor});
            const float sourceMidR = std::max(
                1e-3f, srcMajor + srcB + srcMinor - sourceLongR - sourceShortR);
            const float sourceLongMidRatio = sourceLongR / sourceMidR;
            const float sourceMidShortRatio =
                sourceMidR / std::max(1e-3f, sourceShortR);
            const bool lockSeverePostPcaRodFutureDaughterPosition =
                simulationConfig.celluniverse2_enabled &&
                probConfig.split_severe_post_pca_rod_future_rescue_enabled &&
                winningCleanFutureSupportedPcaBridge &&
                bestLabel == "bridge_primary" &&
                bridgeProposal != nullptr &&
                bridgeProposal->centerSnapUsedAlignedPairFallback &&
                bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
                bridgeProposal->windowBothDaughtersSupported >=
                    std::max(
                        1,
                        probConfig
                            .split_severe_post_pca_rod_future_rescue_min_future_both) &&
                bridgeProposal->windowMissingDaughterCount == 0 &&
                bridgeProposal->windowParentPersists == 0 &&
                bridgeProposal->parentShapeElongation >=
                    probConfig.split_severe_post_pca_rod_future_rescue_min_shape &&
                sourceLongMidRatio >=
                    probConfig
                        .split_severe_post_pca_rod_future_rescue_min_long_mid_ratio &&
                sourceMidShortRatio <=
                    probConfig
                        .split_severe_post_pca_rod_future_rescue_max_mid_short_ratio &&
                bridgeProposal->parentDistanceBalance >=
                    probConfig
                        .split_severe_post_pca_rod_future_rescue_min_parent_balance &&
                bridgeProposal->centerSnapMaxSeedDistance <=
                    std::max(
                        cleanFuturePcaBridgePositionLockSnapLimit,
                        probConfig
                                .split_severe_post_pca_rod_future_rescue_snap_scale *
                            std::max(1.0f, srcMaxR));
            const bool lockOneFrameAlignedFuturePcaBridgeDaughterPosition =
                simulationConfig.celluniverse2_enabled &&
                bridgeProposalOnly &&
                bestLabel == "bridge_primary" &&
                bridgeProposal != nullptr &&
                bridgeProposal->gapStartBin >= 0 &&
                bridgeProposal->gapEndBin >= 0 &&
                probConfig.pca_bridge_future_window_enabled &&
                bridgeProposal->centerSnapUsedAlignedPairFallback &&
                bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
                bridgeProposal->windowBothDaughtersSupported >= 1 &&
                bridgeProposal->windowMissingDaughterCount == 0 &&
                bridgeProposal->windowParentPersists == 0 &&
                bridgeProposal->windowBestMatchedMinBrightness >=
                    probConfig.split_one_frame_aligned_pca_continuation_lock_min_brightness &&
                bridgeProposal->parentShapeElongation >=
                    probConfig.split_one_frame_aligned_pca_continuation_min_parent_shape &&
                bridgeProposal->parentDistanceBalance >=
                    probConfig.split_one_frame_aligned_pca_continuation_min_parent_balance &&
                bridgeProposal->parentDistanceBalance <=
                    probConfig.split_one_frame_aligned_pca_continuation_max_parent_balance &&
                bridgeProposal->centerSnapMaxSeedDistance <=
                    std::max(cleanFuturePcaBridgePositionLockSnapLimit,
                             probConfig
                                 .split_one_frame_aligned_pca_continuation_snap_scale *
                                 std::max(1.0f, srcMaxR));
            const bool lockExactFutureCenterBridgeDaughterPosition =
                simulationConfig.celluniverse2_enabled &&
                bridgeProposalOnly &&
                bestLabel == "bridge_primary" &&
                bridgeProposal != nullptr &&
                probConfig.pca_bridge_future_window_enabled &&
                bridgeProposal->centerSnapApplied &&
                bridgeProposal->immediateFutureCenterBacked &&
                !bridgeProposal->centerSnapUsedAlignedPairFallback &&
                bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
                bridgeProposal->windowBothDaughtersSupported >=
                    std::max(2, probConfig.pca_bridge_future_window_min_both_daughter_support) &&
                bridgeProposal->windowMissingDaughterCount == 0 &&
                bridgeProposal->windowParentPersists == 0 &&
                bridgeProposal->windowBestMatchedMinBrightness >=
                    probConfig.split_exact_future_center_bridge_lock_min_brightness &&
                bridgeProposal->parentShapeElongation >=
                    probConfig.split_exact_future_center_bridge_min_parent_shape &&
                bridgeProposal->parentDistanceBalance >=
                    probConfig.split_exact_future_center_bridge_min_parent_balance &&
                bridgeProposal->centerSnapMaxSeedDistance <=
                    std::max(probConfig.split_exact_future_center_bridge_snap_abs,
                             probConfig.split_exact_future_center_bridge_snap_scale *
                                 std::max(1.0f, srcMaxR));
            const bool lockCurrentCleanPcaBridgeDaughterPosition =
                simulationConfig.celluniverse2_enabled &&
                bridgeProposalOnly &&
                bestLabel == "bridge_primary" &&
                bridgeProposal != nullptr &&
                bridgeProposal->gapStartBin >= 0 &&
                bridgeProposal->gapEndBin >= 0 &&
                bridgeProposal->centerSnapApplied &&
                !bridgeProposal->immediateFutureCenterBacked &&
                !bridgeProposal->centerSnapUsedAlignedPairFallback &&
                bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
                bridgeProposal->windowBothDaughtersSupported >= 1 &&
                bridgeProposal->windowMissingDaughterCount <=
                    std::max(0, probConfig.pca_bridge_future_window_max_missing_daughters) &&
                bridgeProposal->windowParentPersists == 0 &&
                bridgeProposal->windowBestMatchedMinBrightness >=
                    probConfig.split_current_locked_bridge_lock_min_brightness &&
                bridgeProposal->parentShapeElongation >=
                    probConfig.split_current_locked_bridge_min_parent_shape &&
                bridgeProposal->parentDistanceBalance >=
                    probConfig.split_current_locked_bridge_min_parent_balance &&
                bridgeProposal->centerSnapMaxSeedDistance <=
                    std::max(cleanFuturePcaBridgePositionLockSnapLimit,
                             probConfig.split_current_locked_bridge_snap_scale *
                                 std::max(1.0f, srcMaxR));
            const float generalCleanFuturePcaBridgeMinBrightness = std::min(
                probConfig.split_current_locked_bridge_lock_min_brightness,
                std::min(
                    probConfig.split_one_frame_aligned_pca_continuation_lock_min_brightness,
                    probConfig.split_exact_future_center_bridge_lock_min_brightness));
            const float generalCleanFuturePcaBridgeMinParentShape = std::min(
                probConfig.split_current_locked_bridge_min_parent_shape,
                std::min(
                    probConfig.split_one_frame_aligned_pca_continuation_min_parent_shape,
                    probConfig.split_exact_future_center_bridge_min_parent_shape));
            const float generalCleanFuturePcaBridgeMinParentBalance = std::min(
                probConfig.split_current_locked_bridge_min_parent_balance,
                std::min(
                    probConfig.split_one_frame_aligned_pca_continuation_min_parent_balance,
                    probConfig.split_exact_future_center_bridge_min_parent_balance));
            const float generalCleanFuturePcaBridgeMaxParentBalance = std::max(
                probConfig.split_current_locked_bridge_max_parent_balance,
                probConfig.split_one_frame_aligned_pca_continuation_max_parent_balance);
            const bool lockGeneralCleanFuturePcaBridgeDaughterPosition =
                simulationConfig.celluniverse2_enabled &&
                bridgeProposalOnly &&
                bestLabel == "bridge_primary" &&
                bridgeProposal != nullptr &&
                probConfig.pca_bridge_future_window_enabled &&
                bridgeProposal->centerSnapApplied &&
                (bridgeProposal->immediateFutureCenterBacked ||
                 bridgeProposal->centerSnapUsedAlignedPairFallback) &&
                bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
                bridgeProposal->windowBothDaughtersSupported >= 1 &&
                bridgeProposal->windowMissingDaughterCount == 0 &&
                bridgeProposal->windowParentPersists == 0 &&
                bridgeProposal->windowBestMatchedMinBrightness >=
                    generalCleanFuturePcaBridgeMinBrightness &&
                bridgeProposal->parentShapeElongation >=
                    generalCleanFuturePcaBridgeMinParentShape &&
                bridgeProposal->parentDistanceBalance >=
                    generalCleanFuturePcaBridgeMinParentBalance &&
                bridgeProposal->parentDistanceBalance <=
                    generalCleanFuturePcaBridgeMaxParentBalance &&
                bridgeProposal->centerSnapMaxSeedDistance <=
                    cleanFuturePcaBridgePositionLockSnapLimit;
            const bool dimButGeometricallyCleanSignalCenter =
                bridgeProposal != nullptr &&
                bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
                bridgeProposal->windowBothDaughtersSupported >= 2 &&
                bridgeProposal->windowMissingDaughterCount == 0 &&
                bridgeProposal->windowParentPersists == 0 &&
                bridgeProposal->windowBestMatchedMinBrightness >=
                    probConfig.split_dim_exact_future_signal_min_brightness &&
                bridgeProposal->parentDistanceBalance >=
                    probConfig.split_dim_exact_future_signal_min_parent_balance &&
                bridgeProposal->centerSnapMaxSeedDistance <=
                    probConfig.split_dim_exact_future_signal_snap_epsilon &&
                bridgeProposal->bioSeparationRequired >
                    probConfig.split_dim_exact_future_signal_snap_epsilon &&
                bridgeProposal->bioSeparationObserved >=
                    bridgeProposal->bioSeparationRequired;
            const bool lockCleanSignalCenterDaughterPosition =
                simulationConfig.celluniverse2_enabled &&
                bridgeProposalOnly &&
                bestLabel == "bridge_primary" &&
                bridgeProposal != nullptr &&
                bridgeProposal->gapStartBin <=
                    static_cast<int>(probConfig.split_clean_signal_center_gap_bin_max) &&
                bridgeProposal->gapEndBin <=
                    static_cast<int>(probConfig.split_clean_signal_center_gap_bin_max) &&
                probConfig.pca_bridge_future_window_enabled &&
                !bridgeProposal->centerSnapUsedAlignedPairFallback &&
                bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
                bridgeProposal->windowBothDaughtersSupported >=
                    std::max(1, probConfig.pca_bridge_future_window_min_both_daughter_support) &&
                bridgeProposal->windowMissingDaughterCount <=
                    std::max(0, probConfig.pca_bridge_future_window_max_missing_daughters) &&
                bridgeProposal->windowParentPersists == 0 &&
                (bridgeProposal->windowBestMatchedMinBrightness >=
                     probConfig.split_clean_signal_center_min_brightness ||
                 dimButGeometricallyCleanSignalCenter) &&
                bridgeProposal->parentShapeElongation >=
                    std::max(1.0f, probConfig.signal_center_split_min_parent_elongation);
            const bool lockExactSignalCenterDaughterPosition =
                simulationConfig.celluniverse2_enabled &&
                bridgeProposalOnly &&
                bestLabel == "bridge_primary" &&
                bridgeProposal != nullptr &&
                probConfig.signal_center_refit_position_lock_enabled &&
                bridgeProposal->signalCenterScore >= 0.0f &&
                bridgeProposal->centerSnapMaxSeedDistance <=
                    std::max(0.0f,
                             probConfig.signal_center_refit_position_lock_max_snap_distance) &&
                bridgeProposal->parentShapeElongation >=
                    std::max(1.0f, probConfig.signal_center_split_min_parent_elongation);
            const bool lockCellUniverse3SignalCenterFuturePosition =
                simulationConfig.celluniverse3_enabled &&
                probConfig
                    .celluniverse3_signal_center_future_position_lock_enabled &&
                bridgeProposalOnly &&
                bestLabel == "bridge_primary" &&
                bridgeProposal != nullptr &&
                bridgeProposal->signalCenterScore >= 0.0f &&
                !bridgeProposal->centerSnapUsedAlignedPairFallback &&
                bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
                bridgeProposal->windowBothDaughtersSupported >=
                    std::max(
                        1,
                        probConfig
                            .celluniverse3_signal_center_future_position_lock_min_future_both) &&
                bridgeProposal->windowMissingDaughterCount <=
                    std::max(
                        0,
                        probConfig
                            .celluniverse3_signal_center_future_position_lock_max_missing) &&
                bridgeProposal->windowParentPersists == 0 &&
                bridgeProposal->windowBestMatchedMinBrightness >=
                    probConfig
                        .celluniverse3_signal_center_future_position_lock_min_future_brightness &&
                bridgeProposal->parentShapeElongation >=
                    probConfig
                        .celluniverse3_signal_center_future_position_lock_min_parent_shape &&
                bridgeProposal->parentDistanceBalance >=
                    probConfig
                        .celluniverse3_signal_center_future_position_lock_min_parent_balance &&
                bridgeProposal->centerSnapMaxSeedDistance <=
                    std::max(
                        0.0f,
                        probConfig
                            .celluniverse3_signal_center_future_position_lock_max_snap_distance) &&
                bridgeProposal->bioSeparationRequired > 0.0f &&
                bridgeProposal->bioSeparationObserved >=
                    std::max(
                        0.0f,
                        probConfig
                            .celluniverse3_signal_center_future_position_lock_min_sep_fraction) *
                        bridgeProposal->bioSeparationRequired &&
                bridgeProposal->signalCenterSeparationRatio >=
                    probConfig
                        .celluniverse3_signal_center_future_position_lock_min_sep_ratio &&
                std::abs(bridgeProposal->signalCenterAxisAlignment) >=
                    probConfig
                        .celluniverse3_signal_center_future_position_lock_min_axis_alignment;
            const float cellUniverse3CleanFutureBridgeSnapLimit = std::max(
                cleanFuturePcaBridgePositionLockSnapLimit,
                std::max(
                    0.0f,
                    probConfig
                        .celluniverse3_clean_future_bridge_position_lock_max_snap_radius_scale) *
                    std::max(1.0f, srcMaxR));
            const bool lockCellUniverse3CleanFutureBridgeDaughterPosition =
                simulationConfig.celluniverse3_enabled &&
                probConfig
                    .celluniverse3_clean_future_bridge_position_lock_enabled &&
                bridgeProposalOnly &&
                bestLabel == "bridge_primary" &&
                bridgeProposal != nullptr &&
                probConfig.pca_bridge_future_window_enabled &&
                bridgeProposal->centerSnapApplied &&
                bridgeProposal->immediateFutureCenterBacked &&
                bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
                bridgeProposal->windowBothDaughtersSupported >=
                    std::max(
                        1,
                        probConfig
                            .celluniverse3_clean_future_bridge_position_lock_min_future_both) &&
                bridgeProposal->windowMissingDaughterCount <=
                    std::max(
                        0,
                        probConfig
                            .celluniverse3_clean_future_bridge_position_lock_max_missing) &&
                bridgeProposal->windowParentPersists == 0 &&
                bridgeProposal->windowBestMatchedMinBrightness >=
                    probConfig
                        .celluniverse3_clean_future_bridge_position_lock_min_future_brightness &&
                bridgeProposal->parentShapeElongation >=
                    probConfig
                        .celluniverse3_clean_future_bridge_position_lock_min_parent_shape &&
                bridgeProposal->parentDistanceBalance >=
                    probConfig
                        .celluniverse3_clean_future_bridge_position_lock_min_parent_balance &&
                bridgeProposal->centerSnapMaxSeedDistance <=
                    cellUniverse3CleanFutureBridgeSnapLimit;
            const bool lockBridgeAxisPlaceDaughterPosition =
                simulationConfig.celluniverse2_enabled &&
                bridgeProposalOnly &&
                bestLabel == "bridge_axis_place" &&
                bridgeProposal != nullptr &&
                bridgeProposal->daughterSphereRadius > 0.0f;
            const bool lockCellUniverse3DelayedMissingDaughterPosition =
                simulationConfig.celluniverse3_enabled &&
                probConfig
                    .celluniverse3_delayed_missing_daughter_refit_position_lock_enabled &&
                bridgeProposalOnly &&
                bestLabel == "bridge_primary" &&
                bridgeProposal != nullptr &&
                bridgeProposal->cellUniverse3DelayedMissingDaughter &&
                bridgeProposal->centerSnapApplied &&
                bridgeProposal->windowParentPersists == 0;
            const bool lockCellUniverse3WindowMapOverlapDaughterPosition =
                simulationConfig.celluniverse3_enabled &&
                probConfig.celluniverse3_window_map_overlap_position_lock_enabled &&
                bridgeProposalOnly &&
                bestLabel == "bridge_primary" &&
                bridgeProposal != nullptr &&
                bridgeProposal->cellUniverse3MapOverlapCenterProposal &&
                bridgeProposal->cellUniverse3MapPriorConfident &&
                bridgeProposal->cellUniverse3MapUSupportD1 >=
                    probConfig.celluniverse3_window_map_primary_support_min_u_support &&
                bridgeProposal->cellUniverse3MapUSupportD2 >=
                    probConfig.celluniverse3_window_map_primary_support_min_u_support &&
                bridgeProposal->cellUniverse3MapRegionPenalty <=
                    probConfig.celluniverse3_window_map_primary_support_max_region_penalty &&
                bridgeProposal->parentShapeElongation >=
                    probConfig.celluniverse3_window_map_neighbor_bridge_min_parent_shape &&
                bridgeProposal->parentShapeElongation <=
                    probConfig.celluniverse3_window_map_neighbor_bridge_max_parent_shape &&
                bridgeProposal->parentDistanceBalance >=
                    probConfig.celluniverse3_window_map_primary_support_min_parent_balance;
            const bool lockCellUniverse3TunnelDaughterPosition =
                bridgeTunnelConstraintActive &&
                bestLabel == "bridge_primary" &&
                bridgeProposal != nullptr &&
                bridgeProposal->cellUniverse3MapPriorEvaluated;
            const bool lockSplitDaughterRefitPosition =
                lockFutureRodTipDaughterPosition ||
                lockCleanFuturePcaBridgeDaughterPosition ||
                lockTwoFrameAlignedFuturePcaBridgeDaughterPosition ||
                lockSeverePostPcaRodFutureDaughterPosition ||
                lockOneFrameAlignedFuturePcaBridgeDaughterPosition ||
                lockExactFutureCenterBridgeDaughterPosition ||
                lockCurrentCleanPcaBridgeDaughterPosition ||
                lockGeneralCleanFuturePcaBridgeDaughterPosition ||
                lockCleanSignalCenterDaughterPosition ||
                lockExactSignalCenterDaughterPosition ||
                lockCellUniverse3SignalCenterFuturePosition ||
                lockCellUniverse3CleanFutureBridgeDaughterPosition ||
                lockBridgeAxisPlaceDaughterPosition ||
                lockCellUniverse3DelayedMissingDaughterPosition ||
                lockCellUniverse3WindowMapOverlapDaughterPosition ||
                lockCellUniverse3TunnelDaughterPosition;
            const char *splitDaughterRefitLockReason = "none";
            if (lockFutureRodTipDaughterPosition) {
                splitDaughterRefitLockReason = "future_supported_rod_tip";
            } else if (lockCleanFuturePcaBridgeDaughterPosition) {
                splitDaughterRefitLockReason = "clean_future_pca_bridge";
            } else if (lockTwoFrameAlignedFuturePcaBridgeDaughterPosition) {
                splitDaughterRefitLockReason =
                    "two_frame_aligned_future_pca_bridge";
            } else if (lockSeverePostPcaRodFutureDaughterPosition) {
                splitDaughterRefitLockReason =
                    "severe_post_pca_rod_future_bridge";
            } else if (lockOneFrameAlignedFuturePcaBridgeDaughterPosition) {
                splitDaughterRefitLockReason =
                    "one_frame_aligned_future_pca_bridge";
            } else if (lockExactFutureCenterBridgeDaughterPosition) {
                splitDaughterRefitLockReason = "clean_future_center_bridge";
            } else if (lockCurrentCleanPcaBridgeDaughterPosition) {
                splitDaughterRefitLockReason = "clean_current_pca_bridge";
            } else if (lockGeneralCleanFuturePcaBridgeDaughterPosition) {
                splitDaughterRefitLockReason =
                    "general_clean_future_pca_bridge";
            } else if (lockCleanSignalCenterDaughterPosition) {
                splitDaughterRefitLockReason = "clean_signal_center_split";
            } else if (lockExactSignalCenterDaughterPosition) {
                splitDaughterRefitLockReason = "exact_signal_center_split";
            } else if (lockCellUniverse3SignalCenterFuturePosition) {
                splitDaughterRefitLockReason =
                    "celluniverse3_signal_center_future";
            } else if (lockCellUniverse3CleanFutureBridgeDaughterPosition) {
                splitDaughterRefitLockReason =
                    "celluniverse3_clean_future_bridge";
            } else if (lockBridgeAxisPlaceDaughterPosition) {
                splitDaughterRefitLockReason = "bridge_axis_place_seed";
            } else if (lockCellUniverse3DelayedMissingDaughterPosition) {
                splitDaughterRefitLockReason =
                    "celluniverse3_delayed_missing_daughter";
            } else if (lockCellUniverse3WindowMapOverlapDaughterPosition) {
                splitDaughterRefitLockReason =
                    "celluniverse3_window_map_overlap";
            } else if (lockCellUniverse3TunnelDaughterPosition) {
                splitDaughterRefitLockReason =
                    "celluniverse3_window_map_tunnel";
            }
            const bool preserveCellUniverse3FutureBridgeSeedPosition =
                simulationConfig.celluniverse3_enabled &&
                (lockCleanFuturePcaBridgeDaughterPosition ||
                 lockTwoFrameAlignedFuturePcaBridgeDaughterPosition ||
                 lockOneFrameAlignedFuturePcaBridgeDaughterPosition ||
                 lockExactFutureCenterBridgeDaughterPosition ||
                 lockGeneralCleanFuturePcaBridgeDaughterPosition);
            const bool preserveSplitDaughterRefitSeedPosition =
                lockCellUniverse3DelayedMissingDaughterPosition ||
                lockCellUniverse3SignalCenterFuturePosition ||
                lockCellUniverse3CleanFutureBridgeDaughterPosition ||
                lockCellUniverse3WindowMapOverlapDaughterPosition ||
                lockCellUniverse3TunnelDaughterPosition ||
                preserveCellUniverse3FutureBridgeSeedPosition;
            const bool allowFutureBridgeDaughterPcaPositionUpdate =
                simulationConfig.celluniverse2_enabled &&
                probConfig
                    .split_daughter_refit_allow_future_bridge_position_update &&
                winningCleanFutureSupportedPcaBridge &&
                (lockCleanFuturePcaBridgeDaughterPosition ||
                 lockTwoFrameAlignedFuturePcaBridgeDaughterPosition ||
                 lockOneFrameAlignedFuturePcaBridgeDaughterPosition) &&
                bridgeProposal != nullptr &&
                bridgeProposal->windowBothDaughtersSupported >=
                    std::max(2,
                             probConfig
                                 .pca_bridge_future_window_min_both_daughter_support) &&
                bridgeProposal->windowMissingDaughterCount == 0 &&
                bridgeProposal->windowParentPersists == 0;
            bool allowFutureRodTipDaughterPcaPositionUpdate = false;
            bool allowFutureRodTipD1PcaPositionUpdate = false;
            bool allowFutureRodTipD2PcaPositionUpdate = false;
            float futureRodTipUnlockD1Mean = 0.0f;
            float futureRodTipUnlockD2Mean = 0.0f;
            float futureRodTipUnlockMinMean = 0.0f;
            float futureRodTipUnlockThreshold = 0.0f;
            if (simulationConfig.celluniverse2_enabled &&
                probConfig
                    .split_daughter_refit_allow_future_rod_tip_position_update &&
                lockFutureRodTipDaughterPosition &&
                bridgeProposal != nullptr &&
                bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
                bridgeProposal->windowBothDaughtersSupported >= 2 &&
                bridgeProposal->windowMissingDaughterCount == 0 &&
                bridgeProposal->windowParentPersists == 0 &&
                !_realFrame.empty()) {
                const auto parentBrightnessStats =
                    parent.measureBrightnessStats(_realFrame);
                const auto d1BrightnessStats =
                    cells[d1IdxRefine].measureBrightnessStats(_realFrame);
                const auto d2BrightnessStats =
                    cells[d2IdxRefine].measureBrightnessStats(_realFrame);
                const float parentBackground = backgroundAt(cv::Point3f(
                    parent.getX(), parent.getY(), parent.getZ()));
                const float localBackground = std::max(
                    backgroundAt(cv::Point3f(
                        cells[d1IdxRefine].getX(),
                        cells[d1IdxRefine].getY(),
                        cells[d1IdxRefine].getZ())),
                    backgroundAt(cv::Point3f(
                        cells[d2IdxRefine].getX(),
                        cells[d2IdxRefine].getY(),
                        cells[d2IdxRefine].getZ())));
                const float parentSignal =
                    std::max(0.0f,
                             parentBrightnessStats.first - parentBackground);
                const float adaptiveSignalThreshold = std::max(
                    std::max(0.0f,
                             probConfig
                                 .bio_min_daughter_mean_brightness_background_margin),
                    std::max(0.0f,
                             probConfig
                                 .bio_min_daughter_mean_brightness_parent_fraction) *
                        parentSignal);
                futureRodTipUnlockThreshold =
                    std::max(localBackground + adaptiveSignalThreshold,
                             std::max(0.0f,
                                      probConfig
                                          .bio_min_daughter_mean_brightness_absolute));
                futureRodTipUnlockD1Mean = d1BrightnessStats.first;
                futureRodTipUnlockD2Mean = d2BrightnessStats.first;
                futureRodTipUnlockMinMean =
                    std::min(futureRodTipUnlockD1Mean,
                             futureRodTipUnlockD2Mean);
                allowFutureRodTipD1PcaPositionUpdate =
                    futureRodTipUnlockD1Mean < futureRodTipUnlockThreshold;
                allowFutureRodTipD2PcaPositionUpdate =
                    futureRodTipUnlockD2Mean < futureRodTipUnlockThreshold;
                allowFutureRodTipDaughterPcaPositionUpdate =
                    allowFutureRodTipD1PcaPositionUpdate ||
                    allowFutureRodTipD2PcaPositionUpdate;
            }
            float dBuiltA = volumeScale * srcMajor;
            float dBuiltB = volumeScale * srcB;
            float dBuiltC = volumeScale * srcMinor;
            if (bestUsesSphereOverride) {
                const auto &cfg = Ellipsoid::cellConfig;
                const float r = bridgeProposal->daughterSphereRadius;
                dBuiltA = std::clamp(
                    r,
                    static_cast<float>(cfg.minARadius),
                    static_cast<float>(cfg.maxARadius));
                dBuiltB = std::clamp(
                    r,
                    static_cast<float>(cfg.maxBRadius > 0.0 ? cfg.minBRadius : cfg.minARadius),
                    static_cast<float>(cfg.maxBRadius > 0.0 ? cfg.maxBRadius : cfg.maxARadius));
                dBuiltC = std::clamp(
                    r,
                    static_cast<float>(cfg.minCRadius),
                    static_cast<float>(cfg.maxCRadius));
            }
            const float floorA = minFrac * dBuiltA;
            const float floorB = minFrac * dBuiltB;
            const float floorC = minFrac * dBuiltC;
            const float ceilA  = maxFrac * dBuiltA;
            const float ceilB  = maxFrac * dBuiltB;
            const float ceilC  = maxFrac * dBuiltC;

            auto buildRefitClaimSet = [&](size_t selfIdx) -> ClaimSet {
                ClaimSet others;
                for (size_t oi = 0; oi < cells.size(); ++oi) {
                    if (oi == selfIdx) continue;
                    const std::string oname = cells[oi].getName();
                    others[oname].push_back(cv::Point3f(
                        cells[oi].getX(), cells[oi].getY(), cells[oi].getZ()));
                }
                return others;
            };

            auto applySplitDaughterWeightedCenterPull =
                [&](size_t idx,
                    const char *label,
                    const char *stage,
                    const cv::Point3f &anchorPos) -> bool {
                if (!simulationConfig.celluniverse2_enabled ||
                    !simulationConfig.celluniverse2_weighted_center_pull_enabled ||
                    idx >= cells.size() ||
                    cells[idx].isTrash() ||
                    _realFrame.empty()) {
                    return false;
                }

                Ellipsoid &cell = cells[idx];
                const int maxZ = static_cast<int>(_realFrame.size()) - 1;
                const int maxY = _realFrame.empty() ? -1 : _realFrame[0].rows - 1;
                const int maxX = _realFrame.empty() ? -1 : _realFrame[0].cols - 1;
                if (maxX < 0 || maxY < 0 || maxZ < 0) {
                    return false;
                }

                const int iterations = std::max(
                    0, simulationConfig.celluniverse2_weighted_center_pull_iterations);
                if (iterations <= 0) {
                    return false;
                }
                const float requestedRadiusScale = std::max(
                    1.0f,
                    simulationConfig.celluniverse2_weighted_center_pull_radius_scale);
                const float stepFraction = std::clamp(
                    simulationConfig.celluniverse2_weighted_center_pull_step_fraction,
                    0.0f,
                    1.0f);
                const float minSignal = std::max(
                    0.0f,
                    simulationConfig.celluniverse2_weighted_center_pull_min_signal);
                const float nonuniformCvThreshold = std::max(
                    0.0f,
                    simulationConfig.celluniverse2_weighted_center_pull_nonuniform_cv);
                const bool crowdedShrink =
                    simulationConfig
                        .celluniverse2_weighted_center_pull_crowded_shrink_enabled;
                const float crowdedFraction = std::clamp(
                    simulationConfig
                        .celluniverse2_weighted_center_pull_crowded_halfspace_fraction,
                    0.05f,
                    0.95f);
                const float minCrowdedScale = std::clamp(
                    simulationConfig
                        .celluniverse2_weighted_center_pull_crowded_min_radius_scale,
                    1.0f,
                    requestedRadiusScale);
                const bool excludeNeighborOwned =
                    simulationConfig
                        .celluniverse2_weighted_center_pull_neighbor_exclusion_enabled;
                const float neighborExclusionScale = std::max(
                    0.0f,
                    simulationConfig
                        .celluniverse2_weighted_center_pull_neighbor_exclusion_scale);
                const bool voronoiOwnership =
                    simulationConfig
                        .celluniverse2_weighted_center_pull_voronoi_ownership_enabled;
                const float voronoiMarginUnits = std::max(
                    0.0f,
                    simulationConfig
                        .celluniverse2_weighted_center_pull_voronoi_margin_units);

                auto clampToFrame = [&](cv::Point3f p) {
                    p.x = std::clamp(p.x, 0.0f, static_cast<float>(maxX));
                    p.y = std::clamp(p.y, 0.0f, static_cast<float>(maxY));
                    p.z = std::clamp(p.z, 0.0f, static_cast<float>(maxZ));
                    return p;
                };

                auto scanBounds = [&](const cv::Point3f &center,
                                      float scanRadius,
                                      int &x0,
                                      int &x1,
                                      int &y0,
                                      int &y1,
                                      int &z0,
                                      int &z1) {
                    x0 = std::max(
                        0, static_cast<int>(std::floor(center.x - scanRadius)));
                    x1 = std::min(
                        maxX, static_cast<int>(std::ceil(center.x + scanRadius)));
                    y0 = std::max(
                        0, static_cast<int>(std::floor(center.y - scanRadius)));
                    y1 = std::min(
                        maxY, static_cast<int>(std::ceil(center.y + scanRadius)));
                    z0 = std::max(
                        0, static_cast<int>(std::floor(center.z - scanRadius)));
                    z1 = std::min(
                        maxZ, static_cast<int>(std::ceil(center.z + scanRadius)));
                };

                const cv::Point3f triggerPos(cell.getX(), cell.getY(), cell.getZ());
                const float triggerMaxR = std::max({
                    cell.getARadius(), cell.getBRadius(), cell.getCRadius()});
                if (triggerMaxR <= 1e-3f) {
                    return false;
                }
                int tx0, tx1, ty0, ty1, tz0, tz1;
                scanBounds(triggerPos, triggerMaxR, tx0, tx1, ty0, ty1, tz0, tz1);
                double triggerWeightSum = 0.0;
                double triggerWeightSqSum = 0.0;
                int triggerVoxels = 0;
                for (int z = tz0; z <= tz1; ++z) {
                    const cv::Mat &slice = _realFrame[static_cast<size_t>(z)];
                    for (int y = ty0; y <= ty1; ++y) {
                        const float *row = slice.ptr<float>(y);
                        for (int x = tx0; x <= tx1; ++x) {
                            const cv::Point3f p(static_cast<float>(x),
                                                static_cast<float>(y),
                                                static_cast<float>(z));
                            if (!cell.isPointInsideEllipsoid(p, 1.0f)) continue;
                            const float w = std::max(
                                0.0f, row[x] - backgroundAt(z, y, x));
                            if (w <= 0.0f) continue;
                            triggerWeightSum += static_cast<double>(w);
                            triggerWeightSqSum += static_cast<double>(w) * w;
                            ++triggerVoxels;
                        }
                    }
                }
                if (triggerWeightSum <= minSignal || triggerVoxels == 0) {
                    std::cout << "  [Split Daughter Weighted Center Pull] "
                              << parentName << " " << label
                              << " stage=" << stage
                              << " trigger=0 reason=no_interior_signal"
                              << " voxels=" << triggerVoxels
                              << " signal=" << triggerWeightSum
                              << std::endl;
                    return false;
                }
                const double triggerMean =
                    triggerWeightSum / static_cast<double>(triggerVoxels);
                const double triggerVar = std::max(
                    0.0,
                    triggerWeightSqSum / static_cast<double>(triggerVoxels) -
                        triggerMean * triggerMean);
                const double triggerCv =
                    (triggerMean > 1e-12) ? std::sqrt(triggerVar) / triggerMean : 0.0;
                if (triggerCv < nonuniformCvThreshold) {
                    std::cout << "  [Split Daughter Weighted Center Pull] "
                              << parentName << " " << label
                              << " stage=" << stage
                              << " trigger=0 reason=uniform_interior"
                              << " cv=" << triggerCv
                              << " threshold=" << nonuniformCvThreshold
                              << " voxels=" << triggerVoxels
                              << " signal=" << triggerWeightSum
                              << std::endl;
                    return false;
                }

                std::cout << "  [Split Daughter Weighted Center Pull] "
                          << parentName << " " << label
                          << " stage=" << stage
                          << " anchor=(" << anchorPos.x << ","
                          << anchorPos.y << "," << anchorPos.z << ")"
                          << " trigger=1"
                          << " cv=" << triggerCv
                          << " threshold=" << nonuniformCvThreshold
                          << " iterations=" << iterations
                          << " interiorVoxels=" << triggerVoxels
                          << " interiorSignal=" << triggerWeightSum
                          << std::endl;

                bool movedAny = false;
                for (int iter = 0; iter < iterations; ++iter) {
                    const cv::Point3f oldPos(cell.getX(), cell.getY(), cell.getZ());
                    const float cellMaxR = std::max({
                        cell.getARadius(), cell.getBRadius(), cell.getCRadius()});
                    if (cellMaxR <= 1e-3f) break;

                    float radiusScale = requestedRadiusScale;
                    const cv::Point3f searchCenter = anchorPos;
                    float nearestNeighborDistance =
                        std::numeric_limits<float>::infinity();
                    std::string nearestNeighborName;
                    if (crowdedShrink) {
                        for (size_t oi = 0; oi < cells.size(); ++oi) {
                            if (oi == idx || cells[oi].isTrash()) continue;
                            const cv::Point3f otherPos(cells[oi].getX(),
                                                       cells[oi].getY(),
                                                       cells[oi].getZ());
                            const float dist =
                                static_cast<float>(cv::norm(searchCenter - otherPos));
                            if (dist < nearestNeighborDistance) {
                                nearestNeighborDistance = dist;
                                nearestNeighborName = cells[oi].getName();
                            }
                        }
                        if (std::isfinite(nearestNeighborDistance)) {
                            const float neighborLimitedScale =
                                crowdedFraction * nearestNeighborDistance /
                                std::max(1.0f, cellMaxR);
                            radiusScale = std::min(
                                requestedRadiusScale,
                                std::max(minCrowdedScale, neighborLimitedScale));
                        }
                    }

                    const float scanRadius = std::max(1.0f, radiusScale * cellMaxR);
                    std::vector<Ellipsoid> nearbyNeighborMasks;
                    if ((excludeNeighborOwned && neighborExclusionScale > 0.0f) ||
                        voronoiOwnership) {
                        nearbyNeighborMasks.reserve(cells.size());
                        for (size_t oi = 0; oi < cells.size(); ++oi) {
                            if (oi == idx || cells[oi].isTrash()) continue;
                            const float otherMaxR = std::max({
                                cells[oi].getARadius(),
                                cells[oi].getBRadius(),
                                cells[oi].getCRadius()});
                            const cv::Point3f otherPos(cells[oi].getX(),
                                                       cells[oi].getY(),
                                                       cells[oi].getZ());
                            const float dist =
                                static_cast<float>(cv::norm(searchCenter - otherPos));
                            const float neighborReachScale = voronoiOwnership
                                ? std::max(neighborExclusionScale, requestedRadiusScale)
                                : neighborExclusionScale;
                            if (dist > scanRadius + neighborReachScale * otherMaxR) {
                                continue;
                            }
                            nearbyNeighborMasks.push_back(cells[oi]);
                        }
                    }

                    int x0, x1, y0, y1, z0, z1;
                    scanBounds(searchCenter, scanRadius, x0, x1, y0, y1, z0, z1);
                    Ellipsoid searchCell = cell;
                    searchCell.setPosition(searchCenter.x,
                                           searchCenter.y,
                                           searchCenter.z);
                    double weightSum = 0.0;
                    double sumX = 0.0;
                    double sumY = 0.0;
                    double sumZ = 0.0;
                    int rawVoxels = 0;
                    int excludedNeighborVoxels = 0;
                    int excludedOwnershipVoxels = 0;
                    int usedVoxels = 0;
                    const float ownershipMargin =
                        voronoiMarginUnits * std::max(1.0f, cellMaxR);
                    for (int z = z0; z <= z1; ++z) {
                        const cv::Mat &slice = _realFrame[static_cast<size_t>(z)];
                        for (int y = y0; y <= y1; ++y) {
                            const float *row = slice.ptr<float>(y);
                            for (int x = x0; x <= x1; ++x) {
                                const cv::Point3f p(static_cast<float>(x),
                                                    static_cast<float>(y),
                                                    static_cast<float>(z));
                                if (!searchCell.isPointInsideEllipsoid(
                                        p, radiusScale)) {
                                    continue;
                                }
                                ++rawVoxels;
                                bool neighborOwned = false;
                                for (const auto &neighbor : nearbyNeighborMasks) {
                                    if (excludeNeighborOwned &&
                                        neighborExclusionScale > 0.0f &&
                                        neighbor.isPointInsideEllipsoid(
                                            p, neighborExclusionScale)) {
                                        neighborOwned = true;
                                        break;
                                    }
                                    if (voronoiOwnership) {
                                        const cv::Point3f otherPos(
                                            neighbor.getX(),
                                            neighbor.getY(),
                                            neighbor.getZ());
                                        const float selfDist = static_cast<float>(
                                            cv::norm(p - searchCenter));
                                        const float otherDist = static_cast<float>(
                                            cv::norm(p - otherPos));
                                        if (otherDist + ownershipMargin < selfDist) {
                                            neighborOwned = true;
                                            ++excludedOwnershipVoxels;
                                            break;
                                        }
                                    }
                                }
                                if (neighborOwned) {
                                    ++excludedNeighborVoxels;
                                    continue;
                                }
                                const float w = std::max(
                                    0.0f, row[x] - backgroundAt(z, y, x));
                                if (w <= 0.0f) continue;
                                weightSum += static_cast<double>(w);
                                sumX += static_cast<double>(x) * w;
                                sumY += static_cast<double>(y) * w;
                                sumZ += static_cast<double>(z) * w;
                                ++usedVoxels;
                            }
                        }
                    }

                    if (weightSum <= minSignal || usedVoxels == 0) {
                        std::cout << "  [Split Daughter Weighted Center Pull] "
                                  << parentName << " " << label
                                  << " stage=" << stage
                                  << " iter=" << iter
                                  << " action=stop_no_owned_signal"
                                  << " radiusScale=" << radiusScale
                                  << " rawVoxels=" << rawVoxels
                                  << " excludedNeighborVoxels="
                                  << excludedNeighborVoxels
                                  << " excludedOwnershipVoxels="
                                  << excludedOwnershipVoxels
                                  << " usedVoxels=" << usedVoxels
                                  << " signal=" << weightSum
                                  << std::endl;
                        break;
                    }

                    const cv::Point3f target(
                        static_cast<float>(sumX / weightSum),
                        static_cast<float>(sumY / weightSum),
                        static_cast<float>(sumZ / weightSum));
                    cv::Point3f delta = target - oldPos;
                    const float targetMove =
                        static_cast<float>(cv::norm(delta));
                    cv::Point3f appliedDelta = delta * stepFraction;
                    float appliedMove =
                        static_cast<float>(cv::norm(appliedDelta));
                    const float maxMovePerIter = std::max(
                        0.0f,
                        simulationConfig
                                .celluniverse2_weighted_center_pull_max_move_units *
                            cellMaxR);
                    if (maxMovePerIter > 0.0f &&
                        appliedMove > maxMovePerIter &&
                        appliedMove > 1e-6f) {
                        appliedDelta *= (maxMovePerIter / appliedMove);
                        appliedMove = maxMovePerIter;
                    }

                    const cv::Point3f newPos = clampToFrame(oldPos + appliedDelta);
                    const float clampedMove =
                        static_cast<float>(cv::norm(newPos - oldPos));
                    std::cout << "  [Split Daughter Weighted Center Pull] "
                              << parentName << " " << label
                              << " stage=" << stage
                              << " iter=" << iter
                              << " radiusScale=" << radiusScale
                              << " requestedRadiusScale=" << requestedRadiusScale
                              << " searchCenter=(" << searchCenter.x << ","
                              << searchCenter.y << "," << searchCenter.z
                              << ")"
                              << " nearest=" << nearestNeighborName
                              << " nearestDist=" << nearestNeighborDistance
                              << " neighborMasks=" << nearbyNeighborMasks.size()
                              << " rawVoxels=" << rawVoxels
                              << " excludedNeighborVoxels="
                              << excludedNeighborVoxels
                              << " excludedOwnershipVoxels="
                              << excludedOwnershipVoxels
                              << " usedVoxels=" << usedVoxels
                              << " signal=" << weightSum
                              << " old=(" << oldPos.x << "," << oldPos.y
                              << "," << oldPos.z << ")"
                              << " target=(" << target.x << "," << target.y
                              << "," << target.z << ")"
                              << " targetMove=" << targetMove
                              << " appliedMove=" << clampedMove
                              << std::endl;

                    if (clampedMove <= 1e-3f) {
                        break;
                    }
                    cell.setPosition(newPos.x, newPos.y, newPos.z);
                    movedAny = true;
                }
                return movedAny;
            };

            auto refitOne = [&](size_t idx, const char *label) {
                const bool allowFutureRodTipThisDaughter =
                    (idx == d1IdxRefine && allowFutureRodTipD1PcaPositionUpdate) ||
                    (idx == d2IdxRefine && allowFutureRodTipD2PcaPositionUpdate);
                const float futureRodTipUnlockThisMean =
                    (idx == d1IdxRefine)
                        ? futureRodTipUnlockD1Mean
                        : ((idx == d2IdxRefine) ? futureRodTipUnlockD2Mean : 0.0f);
                const bool asymmetricTunnelOneInside =
                    lockCellUniverse3TunnelDaughterPosition &&
                    bridgeProposal != nullptr &&
                    ((bridgeProposal->cellUniverse3MapD1InsideTunnel ? 1 : 0) +
                     (bridgeProposal->cellUniverse3MapD2InsideTunnel ? 1 : 0)) == 1;
                const bool asymmetricTunnelD1Weak =
                    asymmetricTunnelOneInside &&
                    !bridgeProposal->cellUniverse3MapD1InsideTunnel &&
                    bridgeProposal->cellUniverse3MapD2InsideTunnel;
                const bool asymmetricTunnelD2Weak =
                    asymmetricTunnelOneInside &&
                    !bridgeProposal->cellUniverse3MapD2InsideTunnel &&
                    bridgeProposal->cellUniverse3MapD1InsideTunnel;
                const bool unlockAsymmetricTunnelWeakDaughter =
                    simulationConfig.celluniverse3_enabled &&
                    probConfig.celluniverse3_window_map_asymmetric_u_tunnel_rescue_enabled &&
                    bridgeProposalOnly &&
                    bestLabel == "bridge_primary" &&
                    bridgeProposal != nullptr &&
                    bridgeProposal->windowBothDaughtersSupported >=
                        std::max(1,
                                 probConfig.celluniverse3_window_map_asymmetric_u_tunnel_rescue_min_future_both) &&
                    bridgeProposal->windowMissingDaughterCount <=
                        std::max(0,
                                 probConfig.celluniverse3_window_map_asymmetric_u_tunnel_rescue_max_missing) &&
                    bridgeProposal->windowParentPersists == 0 &&
                    bridgeProposal->windowBestMatchedMinBrightness >=
                        probConfig.celluniverse3_window_map_asymmetric_u_tunnel_rescue_min_future_brightness &&
                    ((idx == d1IdxRefine && asymmetricTunnelD1Weak) ||
                     (idx == d2IdxRefine && asymmetricTunnelD2Weak));
                const bool effectiveSplitDaughterRefitPositionLock =
                    lockSplitDaughterRefitPosition &&
                    !allowFutureBridgeDaughterPcaPositionUpdate &&
                    !allowFutureRodTipThisDaughter &&
                    !unlockAsymmetricTunnelWeakDaughter;
                const bool effectivePreserveSplitDaughterRefitSeedPosition =
                    preserveSplitDaughterRefitSeedPosition &&
                    !unlockAsymmetricTunnelWeakDaughter;
                const float preA = cells[idx].getARadius();
                const float preB = cells[idx].getBRadius();
                const float preC = cells[idx].getCRadius();
                const cv::Point3f prePos(cells[idx].getX(),
                                         cells[idx].getY(),
                                         cells[idx].getZ());
                ClaimSet others = buildRefitClaimSet(idx);
                bool flattenedPlaneRotationApplied = false;
                if (effectiveSplitDaughterRefitPositionLock) {
                    std::cout << "  [Split Daughter Refit Position Lock] "
                              << parentName
                              << " " << label
                              << " reason=" << splitDaughterRefitLockReason
                              << " prePos=(" << prePos.x << ","
                              << prePos.y << "," << prePos.z << ")"
                              << std::endl;
                } else if (lockSplitDaughterRefitPosition) {
                    std::cout << "  [Split Daughter Refit Position Update Enabled] "
                              << parentName
                              << " " << label
                              << " originalLockReason="
                              << splitDaughterRefitLockReason
                              << " rodTipDimUnlock="
                              << (allowFutureRodTipThisDaughter ? 1 : 0)
                              << " asymmetricTunnelWeakUnlock="
                              << (unlockAsymmetricTunnelWeakDaughter ? 1 : 0)
                              << " rodTipMean="
                              << futureRodTipUnlockThisMean
                              << " rodTipMinMean="
                              << futureRodTipUnlockMinMean
                              << " rodTipThreshold="
                              << futureRodTipUnlockThreshold
                              << " futureBoth="
                              << (bridgeProposal
                                      ? bridgeProposal->windowBothDaughtersSupported
                                      : 0)
                              << " futureBrightness="
                              << (bridgeProposal
                                      ? bridgeProposal
                                            ->windowBestMatchedMinBrightness
                                      : 0.0f)
                              << " parentBalance="
                              << (bridgeProposal
                                      ? bridgeProposal->parentDistanceBalance
                                      : 0.0f)
                              << " prePos=(" << prePos.x << ","
                              << prePos.y << "," << prePos.z << ")"
                              << std::endl;
                }
                if (simulationConfig.celluniverse2_enabled &&
                    probConfig.split_daughter_flattened_plane_rotation_enabled) {
                    const float builtMax = std::max({dBuiltA, dBuiltB, dBuiltC});
                    const float builtMin = std::max(
                        1e-3f,
                        std::min({dBuiltA, dBuiltB, dBuiltC}));
                    const float builtShape = builtMax / builtMin;
                    if (builtShape >=
                        std::max(1.0f,
                                 probConfig.split_daughter_flattened_plane_min_shape)) {
                        const float planeRadius =
                            std::max(4.0f,
                                     probConfig
                                         .split_daughter_flattened_plane_radius_scale *
                                         builtMax);
                        GatherStats planeStats;
                        const std::vector<cv::Point3f> selfPlaneClaim{prePos};
                        const auto planePixels = gatherBrightPixelsVoronoi(
                            _realFrame,
                            *this,
                            prePos,
                            planeRadius,
                            selfPlaneClaim,
                            others,
                            &planeStats);
                        double targetTx = cells[idx].getThetaX();
                        double targetTy = cells[idx].getThetaY();
                        double targetTz = cells[idx].getThetaZ();
                        float planeRatio = 0.0f;
                        float longAxisRatio = 0.0f;
                        cv::Point3f planeNormal(0.0f, 0.0f, 1.0f);
                        cv::Point3f planeLongAxis(1.0f, 0.0f, 0.0f);
                        std::string mode;
                        const bool ok =
                            static_cast<int>(planePixels.size()) >=
                                probConfig
                                    .split_daughter_flattened_plane_min_pixels &&
                            estimateFlattenedPlaneRotation(
                                planePixels,
                                cells[idx],
                                std::max(
                                    1.0f,
                                    probConfig
                                        .split_daughter_flattened_plane_min_ratio),
                                std::max(
                                    1.0f,
                                    probConfig
                                        .split_daughter_flattened_plane_long_axis_ratio),
                                targetTx,
                                targetTy,
                                targetTz,
                                planeRatio,
                                longAxisRatio,
                                planeNormal,
                                planeLongAxis,
                                mode);
                        bool neighborConstrained = false;
                        std::string neighborGuardDetail;
                        if (ok &&
                            probConfig
                                .split_daughter_flattened_plane_neighbor_guard_enabled) {
                            cv::Point3f shortAxis(0.0f, 0.0f, 1.0f);
                            float shortLen = 0.0f;
                            cells[idx].worldSplitAxis(shortAxis, shortLen);
                            shortAxis = normalizedOr(
                                shortAxis,
                                cv::Point3f(0.0f, 0.0f, 1.0f));
                            const float maxNeighborDist =
                                std::max(
                                    0.0f,
                                    probConfig
                                        .split_daughter_flattened_plane_neighbor_distance_scale) *
                                builtMax;
                            const float maxLateral =
                                std::max(
                                    0.0f,
                                    probConfig
                                        .split_daughter_flattened_plane_neighbor_lateral_scale) *
                                builtMax;
                            const float minAxisAlign =
                                std::clamp(
                                    probConfig
                                        .split_daughter_flattened_plane_neighbor_axis_alignment,
                                    0.0f,
                                    1.0f);
                            const float minNormalAlign =
                                std::clamp(
                                    probConfig
                                        .split_daughter_flattened_plane_neighbor_normal_alignment,
                                    0.0f,
                                    1.0f);
                            float bestNeighborNormalAlign = 1.0f;
                            float bestNeighborAxisAlign = 0.0f;
                            float bestNeighborDist = 0.0f;
                            std::string bestNeighborName;
                            cv::Point3f bestNeighborDir(0.0f, 0.0f, 1.0f);

                            for (const auto &kv : others) {
                                for (const auto &op : kv.second) {
                                    cv::Point3f delta = op - prePos;
                                    const float dist =
                                        static_cast<float>(cv::norm(delta));
                                    if (dist < 1e-3f || dist > maxNeighborDist) {
                                        continue;
                                    }
                                    const cv::Point3f dir =
                                        delta * (1.0f / dist);
                                    const float axisAlign =
                                        std::abs(dir.dot(shortAxis));
                                    const float axial = dist * axisAlign;
                                    const float lateral = std::sqrt(std::max(
                                        0.0f,
                                        dist * dist - axial * axial));
                                    if (axisAlign < minAxisAlign ||
                                        lateral > maxLateral) {
                                        continue;
                                    }

                                    const float normalAlign =
                                        std::abs(dir.dot(planeNormal));
                                    if (normalAlign < bestNeighborNormalAlign) {
                                        bestNeighborNormalAlign = normalAlign;
                                        bestNeighborAxisAlign = axisAlign;
                                        bestNeighborDist = dist;
                                        bestNeighborName = kv.first;
                                        bestNeighborDir = dir;
                                    }
                                }
                            }

                            if (bestNeighborName.empty()) {
                                bestNeighborNormalAlign = 1.0f;
                            } else {
                                if (bestNeighborDir.dot(planeNormal) < 0.0f) {
                                    bestNeighborDir *= -1.0f;
                                }
                                if (bestNeighborNormalAlign < minNormalAlign) {
                                    rotationFromConstrainedPlaneNormal(
                                        cells[idx],
                                        bestNeighborDir,
                                        planeLongAxis,
                                        targetTx,
                                        targetTy,
                                        targetTz);
                                    planeNormal = bestNeighborDir;
                                    mode += "_neighbor_constrained";
                                    neighborConstrained = true;
                                }
                                std::ostringstream ss;
                                ss << " neighbor=" << bestNeighborName
                                   << " dist=" << bestNeighborDist
                                   << " axisAlign=" << bestNeighborAxisAlign
                                   << " normalAlign="
                                   << bestNeighborNormalAlign
                                   << " minNormalAlign="
                                   << minNormalAlign
                                   << " constrained="
                                   << (neighborConstrained ? 1 : 0);
                                neighborGuardDetail = ss.str();
                            }
                        }
                        if (ok) {
                            const float oldTx = cells[idx].getThetaX();
                            const float oldTy = cells[idx].getThetaY();
                            const float oldTz = cells[idx].getThetaZ();
                            cells[idx].setRotation(
                                static_cast<float>(targetTx),
                                static_cast<float>(targetTy),
                                static_cast<float>(targetTz));
                            flattenedPlaneRotationApplied = true;
                            std::cout << "  [Split Daughter Plane Rotation] "
                                      << parentName
                                      << " " << label
                                      << " mode=" << mode
                                      << " pixels=" << planePixels.size()
                                      << " planeRatio=" << planeRatio
                                      << " longAxisRatio=" << longAxisRatio
                                      << " planeNormal=("
                                      << planeNormal.x << ","
                                      << planeNormal.y << ","
                                      << planeNormal.z << ")"
                                      << neighborGuardDetail
                                      << " builtShape=" << builtShape
                                      << " radius=" << planeRadius
                                      << " oldTheta=(" << oldTx << ","
                                      << oldTy << "," << oldTz << ")"
                                      << " newTheta=(" << targetTx << ","
                                      << targetTy << "," << targetTz << ")"
                                      << std::endl;
                        } else {
                            std::cout << "  [Split Daughter Plane Rotation Skip] "
                                      << parentName
                                      << " " << label
                                      << " reason=pca_gate"
                                      << " pixels=" << planePixels.size()
                                      << " minPixels="
                                      << probConfig
                                             .split_daughter_flattened_plane_min_pixels
                                      << " planeRatio=" << planeRatio
                                      << " longAxisRatio=" << longAxisRatio
                                      << " planeNormal=("
                                      << planeNormal.x << ","
                                      << planeNormal.y << ","
                                      << planeNormal.z << ")"
                                      << neighborGuardDetail
                                      << " builtShape=" << builtShape
                                      << " radius=" << planeRadius
                                      << std::endl;
                        }
                    }
                }
                // Position update ENABLED for daughter refit (distinct
                // from mature cells' shape fit, which keeps it off to
                // let calibration own position). At birth the daughter's
                // centroid from burn-in is an estimate — letting the PCA
                // slide it toward the actual pixel centroid fixes the
                // "daughter drawn off-center" artifact.
                calibrateCellShapeViaPca(
                    idx, others,
                    daughterRefitIters,
                    Ellipsoid::cellConfig.pcaShapeRadiusScale,
                    Ellipsoid::cellConfig.pcaShapeMinPixels,
                    Ellipsoid::cellConfig.pcaShapeMaskScale,
                    Ellipsoid::cellConfig.pcaShapeConvergeRadius,
                    Ellipsoid::cellConfig.pcaShapeConvergeAngleDeg,
                    /*updatePosition=*/!effectiveSplitDaughterRefitPositionLock,
                    Ellipsoid::cellConfig.pcaShapeMaxPosShiftFraction,
                    dBuiltA, dBuiltB, dBuiltC);
                const float fitA = std::clamp(cells[idx].getARadius(), floorA, ceilA);
                const float fitB = std::clamp(cells[idx].getBRadius(), floorB, ceilB);
                const float fitC = std::clamp(cells[idx].getCRadius(), floorC, ceilC);
                cells[idx].setRadii(fitA, fitB, fitC);
                const cv::Point3f splitDaughterPullAnchor =
                    unlockAsymmetricTunnelWeakDaughter
                        ? ((idx == d1IdxRefine)
                               ? bridgeProposal->windowBestMatchedD1Pos
                               : bridgeProposal->windowBestMatchedD2Pos)
                        : prePos;
                if (effectivePreserveSplitDaughterRefitSeedPosition) {
                    cells[idx].setPosition(prePos.x, prePos.y, prePos.z);
                    std::cout << "  [Split Daughter Weighted Center Pull Skip] "
                              << parentName
                              << " " << label
                              << " stage=post_pca_shape_fit"
                              << " reason=" << splitDaughterRefitLockReason
                              << " lockedPos=(" << prePos.x << ","
                              << prePos.y << "," << prePos.z << ")"
                              << std::endl;
                } else {
                    applySplitDaughterWeightedCenterPull(
                        idx, label, "post_pca_shape_fit", splitDaughterPullAnchor);
                }
                if (flattenedPlaneRotationApplied) {
                    const cv::Point3f beforeExtraPos(cells[idx].getX(),
                                                     cells[idx].getY(),
                                                     cells[idx].getZ());
                    const float beforeExtraA = cells[idx].getARadius();
                    const float beforeExtraB = cells[idx].getBRadius();
                    const float beforeExtraC = cells[idx].getCRadius();
                    calibrateCellShapeViaPca(
                        idx, others,
                        daughterRefitIters,
                        Ellipsoid::cellConfig.pcaShapeRadiusScale,
                        Ellipsoid::cellConfig.pcaShapeMinPixels,
                        Ellipsoid::cellConfig.pcaShapeMaskScale,
                        Ellipsoid::cellConfig.pcaShapeConvergeRadius,
                        Ellipsoid::cellConfig.pcaShapeConvergeAngleDeg,
                        /*updatePosition=*/!effectiveSplitDaughterRefitPositionLock,
                        Ellipsoid::cellConfig.pcaShapeMaxPosShiftFraction,
                        dBuiltA, dBuiltB, dBuiltC);
                    const float extraFitA =
                        std::clamp(cells[idx].getARadius(), floorA, ceilA);
                    const float extraFitB =
                        std::clamp(cells[idx].getBRadius(), floorB, ceilB);
                    const float extraFitC =
                        std::clamp(cells[idx].getCRadius(), floorC, ceilC);
                    cells[idx].setRadii(extraFitA, extraFitB, extraFitC);
                    if (effectivePreserveSplitDaughterRefitSeedPosition) {
                        cells[idx].setPosition(prePos.x, prePos.y, prePos.z);
                        std::cout << "  [Split Daughter Weighted Center Pull Skip] "
                                  << parentName
                                  << " " << label
                                  << " stage=post_plane_rotation_extra_pca"
                                  << " reason=" << splitDaughterRefitLockReason
                                  << " lockedPos=(" << prePos.x << ","
                                  << prePos.y << "," << prePos.z << ")"
                                  << std::endl;
                    } else {
                        applySplitDaughterWeightedCenterPull(
                            idx, label, "post_plane_rotation_extra_pca", splitDaughterPullAnchor);
                    }
                    const cv::Point3f afterExtraPos(cells[idx].getX(),
                                                    cells[idx].getY(),
                                                    cells[idx].getZ());
                    std::cout << "  [Split Daughter Plane Rotation Extra PCA] "
                              << parentName
                              << " " << label
                              << " iters=" << daughterRefitIters
                              << " pre=(" << beforeExtraA << ","
                              << beforeExtraB << "," << beforeExtraC << ")"
                              << " post=(" << extraFitA << ","
                              << extraFitB << "," << extraFitC << ")"
                              << " prePos=(" << beforeExtraPos.x << ","
                              << beforeExtraPos.y << "," << beforeExtraPos.z
                              << ")"
                              << " postPos=(" << afterExtraPos.x << ","
                              << afterExtraPos.y << "," << afterExtraPos.z
                              << ")"
                              << " posShift="
                              << cv::norm(afterExtraPos - beforeExtraPos)
                              << std::endl;
                }
                const cv::Point3f postPos(cells[idx].getX(),
                                          cells[idx].getY(),
                                          cells[idx].getZ());
                std::cout << "  [Split Daughter Refit] " << parentName
                          << " " << label
                          << " iters=" << daughterRefitIters
                          << " built=(" << dBuiltA << "," << dBuiltB << "," << dBuiltC << ")"
                          << " floor=(" << floorA << "," << floorB << "," << floorC << ")"
                          << " ceil=(" << ceilA << "," << ceilB << "," << ceilC << ")"
                          << " pre=(" << preA << "," << preB << "," << preC << ")"
                          << " post=(" << fitA << "," << fitB << "," << fitC << ")"
                          << " prePos=(" << prePos.x << "," << prePos.y << "," << prePos.z << ")"
                          << " postPos=(" << postPos.x << "," << postPos.y << "," << postPos.z << ")"
                          << " posShift=" << cv::norm(postPos - prePos)
                          << std::endl;
            };

            refitOne(d1IdxRefine, "d1");
            refitOne(d2IdxRefine, "d2");

            if (simulationConfig.celluniverse2_enabled &&
                winningFutureSupportedRodTipClampBypass) {
                const cv::Point3f refitD1(cells[d1IdxRefine].getX(),
                                          cells[d1IdxRefine].getY(),
                                          cells[d1IdxRefine].getZ());
                const cv::Point3f refitD2(cells[d2IdxRefine].getX(),
                                          cells[d2IdxRefine].getY(),
                                          cells[d2IdxRefine].getZ());
                const float seedDistance =
                    static_cast<float>(cv::norm(preRefineD2 - preRefineD1));
                const float refitDistance =
                    static_cast<float>(cv::norm(refitD2 - refitD1));
                const float configuredMinDistance =
                    std::max(0.0f,
                             probConfig
                                 .bio_min_daughter_separation_parent_fraction) *
                    std::max(1.0f, srcMaxR);
                const float targetDistance =
                    std::min(seedDistance, configuredMinDistance);
                if (seedDistance > 1e-3f &&
                    targetDistance > 1e-3f &&
                    refitDistance < targetDistance) {
                    float loT = 0.0f;
                    float hiT = 1.0f;
                    for (int it = 0; it < 24; ++it) {
                        const float midT = 0.5f * (loT + hiT);
                        const cv::Point3f candD1 =
                            preRefineD1 + (refitD1 - preRefineD1) * midT;
                        const cv::Point3f candD2 =
                            preRefineD2 + (refitD2 - preRefineD2) * midT;
                        const float candDistance =
                            static_cast<float>(cv::norm(candD2 - candD1));
                        if (candDistance >= targetDistance) {
                            loT = midT;
                        } else {
                            hiT = midT;
                        }
                    }
                    const cv::Point3f clampedD1 =
                        preRefineD1 + (refitD1 - preRefineD1) * loT;
                    const cv::Point3f clampedD2 =
                        preRefineD2 + (refitD2 - preRefineD2) * loT;
                    cells[d1IdxRefine].setPosition(clampedD1.x,
                                                   clampedD1.y,
                                                   clampedD1.z);
                    cells[d2IdxRefine].setPosition(clampedD2.x,
                                                   clampedD2.y,
                                                   clampedD2.z);
                    std::cout << "  [Split Daughter RodTip Distance Clamp] "
                              << parentName
                              << " seedDistance=" << seedDistance
                              << " refitDistance=" << refitDistance
                              << " targetDistance=" << targetDistance
                              << " configuredMinDistance="
                              << configuredMinDistance
                              << " driftScale=" << loT
                              << " d1From=(" << refitD1.x << ","
                              << refitD1.y << "," << refitD1.z << ")"
                              << " d1To=(" << clampedD1.x << ","
                              << clampedD1.y << "," << clampedD1.z << ")"
                              << " d2From=(" << refitD2.x << ","
                              << refitD2.y << "," << refitD2.z << ")"
                              << " d2To=(" << clampedD2.x << ","
                              << clampedD2.y << "," << clampedD2.z << ")"
                              << std::endl;
                }
            }

            if (simulationConfig.celluniverse2_enabled &&
                probConfig.split_daughter_refit_keep_seed_halfspace_enabled) {
                cv::Point3f seedAxis = preRefineD2 - preRefineD1;
                const float seedDistance = static_cast<float>(cv::norm(seedAxis));
                if (seedDistance > 1e-3f) {
                    seedAxis *= (1.0f / seedDistance);
                    const cv::Point3f seedMidpoint =
                        0.5f * (preRefineD1 + preRefineD2);
                    const bool useFutureRodTipHalfspaceClamp =
                        winningFutureSupportedRodTipClampBypass &&
                        probConfig.split_future_rod_tip_refit_halfspace_min_fraction > 0.0f;
                    const bool useDelayedFutureHalfspaceClamp =
                        bridgeProposalOnly &&
                        bestLabel == "bridge_primary" &&
                        bridgeProposal != nullptr &&
                        probConfig.split_delayed_future_refit_halfspace_min_fraction > 0.0f &&
                        bridgeProposal->windowImmediateBothDaughtersSupported == 0 &&
                        bridgeProposal->windowBothDaughtersSupported >=
                            static_cast<int>(probConfig.split_delayed_future_pca_bridge_min_future_both) &&
                        bridgeProposal->windowMissingDaughterCount <=
                            static_cast<int>(probConfig.split_delayed_future_pca_bridge_max_missing_daughters) &&
                        bridgeProposal->windowParentPersists == 0 &&
                        bridgeProposal->windowBestMatchedMinBrightness >=
                            probConfig.split_delayed_future_pca_bridge_min_brightness &&
                        bridgeProposal->parentShapeElongation >=
                            probConfig.split_delayed_future_pca_bridge_min_parent_shape &&
                        bridgeProposal->parentDistanceBalance >=
                            probConfig.split_delayed_future_pca_bridge_min_parent_balance;
                    const float requestedHalfspaceMinFraction =
                        useFutureRodTipHalfspaceClamp
                            ? probConfig.split_future_rod_tip_refit_halfspace_min_fraction
                        : (useDelayedFutureHalfspaceClamp
                               ? std::max(
                                     probConfig.split_daughter_refit_halfspace_min_fraction,
                                     probConfig
                                         .split_delayed_future_refit_halfspace_min_fraction)
                        : (winningFutureBackedBridgeRescue
                               ? std::max(
                                     probConfig.split_daughter_refit_halfspace_min_fraction,
                                     probConfig.pca_bridge_future_window_refit_halfspace_min_fraction)
                               : probConfig.split_daughter_refit_halfspace_min_fraction));
                    const float minFraction = std::clamp(
                        requestedHalfspaceMinFraction,
                        0.0f,
                        0.95f);
                    const float minSigned = 0.5f * seedDistance * minFraction;

                    auto keepOnSeedSide = [&](size_t idx,
                                              float sideSign,
                                              const char *label) {
                        const cv::Point3f pos(cells[idx].getX(),
                                              cells[idx].getY(),
                                              cells[idx].getZ());
                        const float coord = (pos - seedMidpoint).dot(seedAxis);
                        float clampedCoord = coord;
                        if (sideSign < 0.0f && coord > -minSigned) {
                            clampedCoord = -minSigned;
                        } else if (sideSign > 0.0f && coord < minSigned) {
                            clampedCoord = minSigned;
                        }
                        if (std::abs(clampedCoord - coord) <= 1e-4f) {
                            return;
                        }

                        const cv::Point3f corrected =
                            pos + seedAxis * (clampedCoord - coord);
                        cells[idx].setPosition(corrected.x,
                                               corrected.y,
                                               corrected.z);
                        std::cout << "  [Split Daughter Halfspace Clamp] "
                                  << parentName
                                  << " " << label
                                  << " seedDistance=" << seedDistance
                                  << " minFraction=" << minFraction
                                  << " futureBackedBridge="
                                  << (winningFutureBackedBridgeRescue ? 1 : 0)
                                  << " delayedFutureBridge="
                                  << (useDelayedFutureHalfspaceClamp ? 1 : 0)
                                  << " coord=" << coord
                                  << " clampedCoord=" << clampedCoord
                                  << " from=(" << pos.x << "," << pos.y
                                  << "," << pos.z << ")"
                                  << " to=(" << corrected.x << ","
                                  << corrected.y << "," << corrected.z
                                  << ")"
                                  << std::endl;
                    };

                    if (winningFutureSupportedRodTipClampBypass &&
                        !useFutureRodTipHalfspaceClamp) {
                        std::cout << "  [Split Daughter Halfspace Clamp Bypass] "
                                  << parentName
                                  << " reason=future_supported_rod_tip"
                                  << " seedDistance=" << seedDistance
                                  << " windowBoth="
                                  << bridgeProposal->windowBothDaughtersSupported
                                  << " futureImmediate="
                                  << bridgeProposal->windowImmediateBothDaughtersSupported
                                  << " futureMissing="
                                  << bridgeProposal->windowMissingDaughterCount
                                  << " parentPersists="
                                  << bridgeProposal->windowParentPersists
                                  << " futureBestMinBrightness="
                                  << bridgeProposal->windowBestMatchedMinBrightness
                                  << std::endl;
                    } else {
                        if (useFutureRodTipHalfspaceClamp) {
                            std::cout << "  [Split Daughter Halfspace Clamp Mode] "
                                      << parentName
                                      << " reason=future_supported_rod_tip"
                                      << " minFraction=" << minFraction
                                      << " seedDistance=" << seedDistance
                                      << " futureBoth="
                                      << bridgeProposal->windowBothDaughtersSupported
                                      << " futureImmediate="
                                      << bridgeProposal->windowImmediateBothDaughtersSupported
                                      << " futureMissing="
                                      << bridgeProposal->windowMissingDaughterCount
                                      << " parentPersists="
                                      << bridgeProposal->windowParentPersists
                                      << " futureBestMinBrightness="
                                      << bridgeProposal->windowBestMatchedMinBrightness
                                      << std::endl;
                        }
                        keepOnSeedSide(d1IdxRefine, -1.0f, "d1");
                        keepOnSeedSide(d2IdxRefine, 1.0f, "d2");
                    }
                }
            }

            if (simulationConfig.celluniverse2_enabled &&
                probConfig.split_daughter_refit_align_short_axis_to_split_enabled) {
                const cv::Point3f alignedD1(cells[d1IdxRefine].getX(),
                                            cells[d1IdxRefine].getY(),
                                            cells[d1IdxRefine].getZ());
                const cv::Point3f alignedD2(cells[d2IdxRefine].getX(),
                                            cells[d2IdxRefine].getY(),
                                            cells[d2IdxRefine].getZ());
                cv::Point3f splitDir = alignedD2 - alignedD1;
                const float splitLen = static_cast<float>(cv::norm(splitDir));
                if (splitLen > 1e-3f) {
                    splitDir *= (1.0f / splitLen);
                    auto alignDaughterShortAxis = [&](size_t idx,
                                                       const char *label) {
                        const float oldTx = cells[idx].getThetaX();
                        const float oldTy = cells[idx].getThetaY();
                        const float oldTz = cells[idx].getThetaZ();
                        cv::Point3f oldShortAxis;
                        float oldShortLen = 0.0f;
                        cells[idx].worldSplitAxis(oldShortAxis, oldShortLen);
                        double targetTx = oldTx;
                        double targetTy = oldTy;
                        double targetTz = oldTz;
                        int shortSlot = 0;
                        float angleBefore = 0.0f;
                        if (!rotationAligningShortestAxisToDirection(
                                cells[idx],
                                splitDir,
                                targetTx,
                                targetTy,
                                targetTz,
                                &shortSlot,
                                &angleBefore)) {
                            return;
                        }
                        cells[idx].setRotation(static_cast<float>(targetTx),
                                               static_cast<float>(targetTy),
                                               static_cast<float>(targetTz));
                        cv::Point3f newShortAxis;
                        float newShortLen = 0.0f;
                        cells[idx].worldSplitAxis(newShortAxis, newShortLen);
                        const float alignment = std::abs(newShortAxis.dot(splitDir));
                        std::cout << "  [Split Daughter Axis Align] "
                                  << parentName
                                  << " " << label
                                  << " splitLen=" << splitLen
                                  << " shortSlot=" << shortSlot
                                  << " angleBeforeDeg=" << angleBefore
                                  << " alignment=" << alignment
                                  << " oldAxis=(" << oldShortAxis.x << ","
                                  << oldShortAxis.y << "," << oldShortAxis.z
                                  << ")"
                                  << " splitDir=(" << splitDir.x << ","
                                  << splitDir.y << "," << splitDir.z << ")"
                                  << " oldTheta=(" << oldTx << ","
                                  << oldTy << "," << oldTz << ")"
                                  << " newTheta=(" << targetTx << ","
                                  << targetTy << "," << targetTz << ")"
                                  << std::endl;
                    };
                    alignDaughterShortAxis(d1IdxRefine, "d1");
                    alignDaughterShortAxis(d2IdxRefine, "d2");
                }
            }

            // Radii changed → regenerate synth. Under bbox mode, only
            // re-render the z-slices affected by the two daughters (whose
            // radii just changed). Under legacy mode, full render.
            if (_useBboxCost) {
                const Ellipsoid &rd1 = cells[d1IdxRefine];
                const Ellipsoid &rd2 = cells[d2IdxRefine];
                const float rd1MaxR = std::max({rd1.getARadius(), rd1.getBRadius(), rd1.getCRadius()});
                const float rd2MaxR = std::max({rd2.getARadius(), rd2.getBRadius(), rd2.getCRadius()});
                // Include the PRE-refit extent too (built radii) in case
                // the refit shrunk the cell — old render needs clearing.
                const float builtMaxR = std::max({dBuiltA, dBuiltB, dBuiltC});
                const float extentR = std::max({rd1MaxR, rd2MaxR, builtMaxR});
                const int nSlices = static_cast<int>(z_slices.size());
                const int zLo = std::max(0, static_cast<int>(std::floor(
                    std::min(rd1.getZ(), rd2.getZ()) - extentR)));
                const int zHi = std::min(nSlices - 1, static_cast<int>(std::ceil(
                    std::max(rd1.getZ(), rd2.getZ()) + extentR)));
                const cv::Size shape = getImageShape();
                for (int z = zLo; z <= zHi; ++z) {
                    cv::Mat synthImage = makeSynthBackgroundSlice(shape, z);
                    const float zf = static_cast<float>(z_slices[z]);
                    for (const auto &cell : cells) {
                        const float cmr = std::max({cell.getARadius(),
                                                    cell.getBRadius(),
                                                    cell.getCRadius()});
                        if (std::abs(zf - cell.getZ()) > cmr) continue;
                        cell.draw(synthImage, simulationConfig, zf);
                    }
                    _synthFrame[z] = synthImage;
                }
            } else {
                _synthFrame = generateSynthFrame();
                refreshFullCostCache();
            }
        }

        // Re-capture refined state as new best. Capture diagnostic data
        // BEFORE moving cells (use-after-move is UB).
        bestImageCost = evalImageCost(_synthFrame);
        bestTotal = bestImageCost + computeOverlapPenalty(probConfig.overlap_penalty_weight);

        const cv::Point3f postRefineD1(cells[d1IdxRefine].getX(),
                                         cells[d1IdxRefine].getY(),
                                         cells[d1IdxRefine].getZ());
        const cv::Point3f postRefineD2(cells[d2IdxRefine].getX(),
                                         cells[d2IdxRefine].getY(),
                                         cells[d2IdxRefine].getZ());
        // Daughter radii diagnostic. Built = 0.794 * src; post-refit radii
        // reflect the PCA fit on each daughter's pixel cloud, clamped at
        // the configured floor fraction.
        // Read daughter state BEFORE moving cells — references into a
        // moved-from vector are dangling (UB). Copy radii by value.
        const Ellipsoid refinedD1 = cells[d1IdxRefine];
        const Ellipsoid refinedD2 = cells[d2IdxRefine];
        const float builtMajor = volumeScale * srcMajor;
        const float builtB     = volumeScale * srcB;
        const float builtMinor = volumeScale * srcMinor;
        std::cout << "  [Split Refine] " << parentName
                  << " iters=" << refineIters
                  << " accepts=" << refineAccepts
                  << " preTotal=" << preRefineTotal
                  << " postTotal=" << bestTotal
                  << " delta=" << (bestTotal - preRefineTotal)
                  << " refineDrift1=" << cv::norm(postRefineD1 - preRefineD1)
                  << " refineDrift2=" << cv::norm(postRefineD2 - preRefineD2)
                  << " d1=(" << postRefineD1.x << "," << postRefineD1.y << "," << postRefineD1.z << ")"
                  << " d2=(" << postRefineD2.x << "," << postRefineD2.y << "," << postRefineD2.z << ")"
                  << " builtR=(" << builtMajor << "," << builtB << "," << builtMinor << ")"
                  << " d1R=(" << refinedD1.getARadius() << "," << refinedD1.getBRadius() << "," << refinedD1.getCRadius() << ")"
                  << " d2R=(" << refinedD2.getARadius() << "," << refinedD2.getBRadius() << "," << refinedD2.getCRadius() << ")"
                  << std::endl;

        const bool refitFallbackLabel =
            bestLabel == "bridge_primary" || bestLabel == "bridge_tip_alt";
        const bool cleanFutureRefitFallbackCandidate =
            simulationConfig.celluniverse2_enabled &&
            probConfig.split_daughter_refit_cost_fallback_enabled &&
            bridgeProposalOnly &&
            refitFallbackLabel &&
            bridgeProposal != nullptr &&
            probConfig.pca_bridge_future_window_enabled &&
            bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
            bridgeProposal->windowBothDaughtersSupported >=
                std::max(1, probConfig.split_daughter_refit_cost_fallback_min_future_both) &&
            bridgeProposal->windowMissingDaughterCount <=
                std::max(0, probConfig.split_daughter_refit_cost_fallback_max_missing_daughters) &&
            bridgeProposal->windowParentPersists <=
                std::max(0, probConfig.split_daughter_refit_cost_fallback_max_parent_persists) &&
            bridgeProposal->parentShapeElongation >=
                probConfig.split_daughter_refit_cost_fallback_min_parent_shape;
        const bool refitWorsenedCandidate =
            bestTotal > preRefineTotal +
                std::max(0.0f, probConfig.split_daughter_refit_cost_fallback_min_worsen);
        const bool preRefineImprovesBaseline =
            !probConfig.split_daughter_refit_cost_fallback_require_pre_improvement ||
            preRefineTotal < baselineTotal;
        if (cleanFutureRefitFallbackCandidate &&
            refitWorsenedCandidate &&
            preRefineImprovesBaseline) {
            std::cout << "  [Split Refit Fallback] " << parentName
                      << " reason=refit_worsened_clean_future_split"
                      << " label=" << bestLabel
                      << " preTotal=" << preRefineTotal
                      << " postTotal=" << bestTotal
                      << " delta=" << (bestTotal - preRefineTotal)
                      << " baseline=" << baselineTotal
                      << " futureBoth="
                      << bridgeProposal->windowBothDaughtersSupported
                      << " futureImmediate="
                      << bridgeProposal->windowImmediateBothDaughtersSupported
                      << " futureMissing="
                      << bridgeProposal->windowMissingDaughterCount
                      << " parentPersists="
                      << bridgeProposal->windowParentPersists
                      << " parentShape="
                      << bridgeProposal->parentShapeElongation
                      << std::endl;
            bestImageCost = preRefineBestImageCost;
            bestTotal = preRefineTotal;
            bestCells = std::move(preRefineBestCells);
            bestSynth = std::move(preRefineBestSynth);
            bestPerSlice = std::move(preRefineBestPerSlice);
        } else {
            bestCells = std::move(cells);
            bestSynth = std::move(_synthFrame);
            bestPerSlice = std::move(_currentCostPerSlice);
        }

        // Revert to pre-split state — gates run on savedCells baseline
        // against bestCells (the refined winner).
        cells = savedCells;
        _synthFrame = savedSynth;
        _currentCost = savedCost;
        _currentCostPerSlice = savedPerSlice;
    }

    // Restore main-loop perturbation sigmas before the gate sequence.
    Ellipsoid::cellConfig.x = savedPerturbX;
    Ellipsoid::cellConfig.y = savedPerturbY;
    Ellipsoid::cellConfig.z = savedPerturbZ;

    // --- 5. Bio checks on the best candidate's final state ---
    // Rebuild daughter indices from bestCells (parent was at cellIndex,
    // daughters are the last two entries in bestCells since that's how we
    // replaced them during evaluation).
    const size_t d1IdxBest = bestCells.size() - 2;
    const size_t d2IdxBest = bestCells.size() - 1;
    const Ellipsoid &bestD1 = bestCells[d1IdxBest];
    const Ellipsoid &bestD2 = bestCells[d2IdxBest];

    const cv::Point3f bestD1Pos(bestD1.getX(), bestD1.getY(), bestD1.getZ());
    const cv::Point3f bestD2Pos(bestD2.getX(), bestD2.getY(), bestD2.getZ());
    const bool bestTunnelD1Inside = bridgeTunnelContainsPoint(bestD1Pos);
    const bool bestTunnelD2Inside = bridgeTunnelContainsPoint(bestD2Pos);
    const bool bestAsymmetricUTunnelPass =
        bridgeTunnelConstraintActive &&
        bridgeProposal != nullptr &&
        probConfig.celluniverse3_window_map_asymmetric_u_tunnel_rescue_enabled &&
        bestLabel == "bridge_primary" &&
        bestTunnelD1Inside != bestTunnelD2Inside &&
        std::max(bridgeProposal->cellUniverse3MapUSupportD1,
                 bridgeProposal->cellUniverse3MapUSupportD2) >=
            std::max(0.0f,
                     probConfig
                         .celluniverse3_window_map_primary_asymmetric_min_strong_u_support) &&
        std::min(bridgeProposal->cellUniverse3MapUSupportD1,
                 bridgeProposal->cellUniverse3MapUSupportD2) <=
            std::max(0.0f,
                     probConfig
                         .celluniverse3_window_map_asymmetric_u_tunnel_rescue_max_weak_u_support) &&
        bridgeProposal->windowBothDaughtersSupported >=
            std::max(1,
                     probConfig
                         .celluniverse3_window_map_asymmetric_u_tunnel_rescue_min_future_both) &&
        bridgeProposal->windowMissingDaughterCount <=
            std::max(0,
                     probConfig
                         .celluniverse3_window_map_asymmetric_u_tunnel_rescue_max_missing) &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->windowBestMatchedMinBrightness >=
            std::max(0.0f,
                     probConfig
                         .celluniverse3_window_map_asymmetric_u_tunnel_rescue_min_future_brightness) &&
        bridgeProposal->cellUniverse3MapRegionPenalty <=
            std::max(0.0f,
                     probConfig
                         .celluniverse3_window_map_asymmetric_u_tunnel_rescue_max_region_penalty) &&
        (!bridgeProposal->centerSnapApplied ||
         bridgeProposal->centerSnapScore <=
             std::max(0.0f,
                      probConfig
                          .celluniverse3_window_map_asymmetric_u_tunnel_rescue_max_center_snap_score));
    const bool bestOutsideFutureTunnelPass =
        bridgeTunnelConstraintActive &&
        bridgeProposal != nullptr &&
        bestLabel == "bridge_primary" &&
        !bestTunnelD1Inside &&
        !bestTunnelD2Inside &&
        bridgeProposal->windowBothDaughtersSupported >= 2 &&
        bridgeProposal->windowMissingDaughterCount == 0 &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->windowBestMatchedMinBrightness >=
            std::max(0.0f,
                     probConfig
                         .pca_bridge_future_window_rod_tip_balance_min_brightness) &&
        bridgeProposal->parentDistanceBalance >=
            std::max(0.0f,
                     probConfig
                         .pca_bridge_future_window_parent_balance_rescue_min) &&
        bridgeProposal->bioSeparationObserved >=
            bridgeProposal->bioSeparationRequired &&
        bridgeProposal->cellUniverse3MapRegionPenalty <= 2.0f;
    if (bestAsymmetricUTunnelPass || bestOutsideFutureTunnelPass) {
        std::cout << "  [Split Final Tunnel Gate] " << parentName
                  << " label=" << bestLabel
                  << " d1Inside=" << (bestTunnelD1Inside ? 1 : 0)
                  << " d2Inside=" << (bestTunnelD2Inside ? 1 : 0)
                  << " mapUSupportD1="
                  << bridgeProposal->cellUniverse3MapUSupportD1
                  << " mapUSupportD2="
                  << bridgeProposal->cellUniverse3MapUSupportD2
                  << " futureBoth="
                  << bridgeProposal->windowBothDaughtersSupported
                  << " futureMissing="
                  << bridgeProposal->windowMissingDaughterCount
                  << " futureBrightness="
                  << bridgeProposal->windowBestMatchedMinBrightness
                  << " action="
                  << (bestOutsideFutureTunnelPass
                          ? "allow_outside_future_pair"
                          : "allow_asymmetric_u_tunnel")
                  << std::endl;
    }
    if (bridgeTunnelConstraintActive &&
        !bestAsymmetricUTunnelPass &&
        !bestOutsideFutureTunnelPass &&
        (!bestTunnelD1Inside || !bestTunnelD2Inside)) {
        std::cout << "  [Split Final Tunnel Gate] " << parentName
                  << " label=" << bestLabel
                  << " d1Inside=" << (bestTunnelD1Inside ? 1 : 0)
                  << " d2Inside=" << (bestTunnelD2Inside ? 1 : 0)
                  << " d1=(" << bestD1Pos.x << "," << bestD1Pos.y
                  << "," << bestD1Pos.z << ")"
                  << " d2=(" << bestD2Pos.x << "," << bestD2Pos.y
                  << "," << bestD2Pos.z << ")"
                  << " tunnelBoxes="
                  << (bridgeProposal != nullptr
                          ? bridgeProposal
                                ->cellUniverse3MapTunnelFlatIndices.size()
                          : 0)
                  << " action=reject_split"
                  << std::endl;
        restoreLiveParent();
        return {0.0, noop};
    }
    const bool bestIsCellLumenPrior = (bestLabel == "cell_lumen_primary");
    const bool bestIsDeterministicSingleProposal =
        bridgeProposalOnly &&
        bestLabel == "bridge_primary" &&
        bridgeProposal != nullptr;
    const bool bestIsPcaBridgeOnly =
        bestIsDeterministicSingleProposal &&
        bridgeProposal->gapStartBin >= 0 &&
        bridgeProposal->gapEndBin >= 0;
    const bool bestHasCleanFutureBridgeSupport =
        bestIsPcaBridgeOnly &&
        probConfig.pca_bridge_future_window_enabled &&
        bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
        bridgeProposal->windowBothDaughtersSupported >=
            std::max(1, probConfig.pca_bridge_future_window_min_both_daughter_support) &&
        bridgeProposal->windowMissingDaughterCount <=
            std::max(0, probConfig.pca_bridge_future_window_max_missing_daughters) &&
        bridgeProposal->windowParentPersists <=
            std::max(0, probConfig.pca_bridge_future_window_max_parent_persists);
    const bool bestIsSignalCenterProposal =
        bestIsDeterministicSingleProposal &&
        bridgeProposal->gapStartBin <= -4 &&
        bridgeProposal->gapEndBin <= -4;
    const bool bestHasCleanFutureSignalSupport =
        bestIsSignalCenterProposal &&
        probConfig.pca_bridge_future_window_enabled &&
        bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
        bridgeProposal->windowBothDaughtersSupported >=
            std::max(2, probConfig.pca_bridge_future_window_min_both_daughter_support) &&
        bridgeProposal->windowMissingDaughterCount == 0 &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->windowBestMatchedMinBrightness >=
            std::max(
                0.0f,
                probConfig
                    .pca_bridge_future_window_geometry_rescue_min_brightness) &&
        bridgeProposal->centerSnapMaxSeedDistance <=
            std::max(
                std::max(0.0f, probConfig.pca_bridge_future_window_match_distance),
                probConfig.split_clean_pca_bridge_snap_base_scale * std::max(1.0f, srcMaxR));
    const bool bestHasCleanCurrentBridgeSupport =
        bestIsPcaBridgeOnly &&
        bridgeProposal->centerSnapApplied &&
        !bridgeProposal->immediateFutureCenterBacked &&
        !bridgeProposal->centerSnapUsedAlignedPairFallback &&
        (bridgeProposal->bioSeparationSoftRescued ||
         (bridgeProposal->bioSeparationRequired > 0.0f &&
          bridgeProposal->bioSeparationObserved >=
              bridgeProposal->bioSeparationRequired)) &&
        bridgeProposal->parentDistanceBalance >= probConfig.split_long_raw_pca_bridge_min_parent_balance &&
        bridgeProposal->centerSnapMaxSeedDistance <=
            std::max(
                std::max(0.0f, probConfig.pca_bridge_future_window_match_distance),
                0.75f * std::max(1.0f, srcMaxR));
    const bool bestHasCleanFutureSplitSupport =
        bestHasCleanFutureBridgeSupport || bestHasCleanFutureSignalSupport;
    const bool bestIsCellLumenPrepassFallback =
        bestIsCellLumenPrior && lumenProposal != nullptr &&
        lumenProposal->gapStartBin <= -2 && lumenProposal->gapEndBin <= -2;
    const bool bestIsCellLumenSnapshotSeedFallback =
        bestIsCellLumenPrior && lumenProposal != nullptr &&
        lumenProposal->gapStartBin == -3 && lumenProposal->gapEndBin == -3;
    const bool useCellLumenGateParams = bestIsCellLumenPrior && lumenUseDedicatedCostGate;
    const bool useCellLumenSoftGate = useCellLumenGateParams && lumenSoftGateEnabled;
    double lumenSoftGatePenaltyCost = 0.0;
    double splitSoftGeometryPenaltyCost = 0.0;
    int lumenLocalNeighborCount = 0;
    if (useCellLumenSoftGate && lumenDynamicOverlapEnabled) {
        const cv::Point3f parentPos(parent.getX(), parent.getY(), parent.getZ());
        const float densityRadius = std::max(0.0f, lumenLocalDensityRadiusScale) *
                                    std::max(1.0f, srcMaxR);
        for (size_t oi = 0; oi < cells.size(); ++oi) {
            if (oi == cellIndex) continue;
            const Ellipsoid &other = cells[oi];
            const cv::Point3f otherPos(other.getX(), other.getY(), other.getZ());
            if (cv::norm(otherPos - parentPos) <= densityRadius) {
                ++lumenLocalNeighborCount;
            }
        }
    }
    auto addLumenSoftGatePenalty = [&](const std::string &reason,
                                       double normalizedExcess,
                                       double penaltyFraction) {
        if (!useCellLumenSoftGate || normalizedExcess <= 0.0 || penaltyFraction <= 0.0) {
            return;
        }
        const double penalty = baselineImageCost * normalizedExcess * penaltyFraction;
        lumenSoftGatePenaltyCost += penalty;
        std::cout << "[Split CellLumen Soft Gate] " << parentName
                  << " reason=" << reason
                  << " normalizedExcess=" << normalizedExcess
                  << " penaltyFraction=" << penaltyFraction
                  << " penaltyCost=" << penalty
                  << " runningPenalty=" << lumenSoftGatePenaltyCost
                  << " localNeighbors=" << lumenLocalNeighborCount
                  << " baselineImageCost=" << baselineImageCost
                  << " bestLabel=" << bestLabel
                  << std::endl;
    };
    auto addSplitSoftGeometryPenalty = [&](const std::string &reason,
                                           double normalizedExcess,
                                           double penaltyFraction) {
        if (!simulationConfig.celluniverse2_enabled ||
            !probConfig.split_soft_geometry_gate_enabled ||
            normalizedExcess <= 0.0 ||
            penaltyFraction <= 0.0) {
            return;
        }
        const double penalty = baselineImageCost * normalizedExcess * penaltyFraction;
        splitSoftGeometryPenaltyCost += penalty;
        std::cout << "[Split Soft Geometry Gate] " << parentName
                  << " reason=" << reason
                  << " normalizedExcess=" << normalizedExcess
                  << " penaltyFraction=" << penaltyFraction
                  << " penaltyCost=" << penalty
                  << " runningPenalty=" << splitSoftGeometryPenaltyCost
                  << " baselineImageCost=" << baselineImageCost
                  << " bestLabel=" << bestLabel
                  << std::endl;
    };

    if (simulationConfig.celluniverse2_enabled &&
        bridgeProposal != nullptr &&
        bridgeProposal->bioSeparationSoftRescued) {
        addSplitSoftGeometryPenalty(
            "bio_separation_near_miss",
            static_cast<double>(
                std::max(0.0f,
                         bridgeProposal->bioSeparationSoftNormalizedExcess)),
            static_cast<double>(
                std::max(0.0f,
                         bridgeProposal->bioSeparationSoftPenaltyFraction)));
    }

    if (simulationConfig.celluniverse3_enabled &&
        bridgeProposal != nullptr &&
        bridgeProposal->cellUniverse3MapPriorEvaluated) {
        addSplitSoftGeometryPenalty(
            "celluniverse3_window_map_center",
            static_cast<double>(
                std::max(0.0f, bridgeProposal->cellUniverse3MapCenterPenalty)),
            static_cast<double>(
                std::max(0.0f,
                         simulationConfig
                             .celluniverse3_window_map_center_penalty_fraction)));
        addSplitSoftGeometryPenalty(
            "celluniverse3_window_map_axis",
            static_cast<double>(
                std::max(0.0f, bridgeProposal->cellUniverse3MapAxisPenalty)),
            static_cast<double>(
                std::max(0.0f,
                         simulationConfig
                             .celluniverse3_window_map_axis_penalty_fraction)));
        addSplitSoftGeometryPenalty(
            "celluniverse3_window_map_region",
            static_cast<double>(
                std::max(0.0f, bridgeProposal->cellUniverse3MapRegionPenalty)),
            static_cast<double>(
                std::max(0.0f,
                         simulationConfig
                             .celluniverse3_window_map_region_penalty_fraction)));
        std::cout << "[CellUniverse3 Window Map Split Score] "
                  << parentName
                  << " mapProposal="
                  << (bridgeProposal->cellUniverse3MapProposal ? 1 : 0)
                  << " confident="
                  << (bridgeProposal->cellUniverse3MapPriorConfident ? 1 : 0)
                  << " O=" << bridgeProposal->cellUniverse3MapOverlapBoxes
                  << " U=" << bridgeProposal->cellUniverse3MapUnionBoxes
                  << " D=" << bridgeProposal->cellUniverse3MapFutureOnlyBoxes
                  << " uSupportD1="
                  << bridgeProposal->cellUniverse3MapUSupportD1
                  << " uSupportD2="
                  << bridgeProposal->cellUniverse3MapUSupportD2
                  << " axisAlignment="
                  << bridgeProposal->cellUniverse3MapAxisAlignment
                  << " centerPenalty="
                  << bridgeProposal->cellUniverse3MapCenterPenalty
                  << " axisPenalty="
                  << bridgeProposal->cellUniverse3MapAxisPenalty
                  << " regionPenalty="
                  << bridgeProposal->cellUniverse3MapRegionPenalty
                  << " runningSplitSoftPenalty="
                  << splitSoftGeometryPenaltyCost
                  << " bestLabel=" << bestLabel
                  << std::endl;
    }

    // Drift from seed. A valid split is allowed to locally refine, but the
    // daughters should not explain the frame by walking far away from the
    // proposed division geometry.
    const float drift1 = static_cast<float>(cv::norm(
        bestD1Pos - bestSeedD1));
    const float drift2 = static_cast<float>(cv::norm(
        bestD2Pos - bestSeedD2));
    const float seedAxisLen = static_cast<float>(cv::norm(bestSeedD2 - bestSeedD1));
    const float finalAxisLen = static_cast<float>(cv::norm(bestD2Pos - bestD1Pos));
    const float parentMaxRadiusForSoftGeometry =
        std::max(1.0f, std::max({srcMajor, srcB, srcMinor}));
    const bool bestIsFutureSupportedBridgeAxisPlace =
        bridgeProposalOnly &&
        bestLabel == "bridge_axis_place" &&
        bridgeProposal != nullptr &&
        bridgeProposal->daughterSphereRadius > 0.0f &&
        probConfig.pca_bridge_future_window_enabled &&
        bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
        bridgeProposal->windowBothDaughtersSupported >=
            std::max(2, probConfig.pca_bridge_future_window_min_both_daughter_support) &&
        bridgeProposal->windowMissingDaughterCount == 0 &&
        bridgeProposal->windowParentPersists == 0;

    if (simulationConfig.celluniverse2_enabled &&
        probConfig.split_soft_geometry_gate_enabled) {
        const float softDriftFraction =
            std::max(0.0f,
                     probConfig.split_soft_max_daughter_seed_drift_fraction);
        const float softDriftLimit =
            softDriftFraction * parentMaxRadiusForSoftGeometry;
        if (softDriftLimit > 0.0f) {
            const float maxDrift = std::max(drift1, drift2);
            const double normalizedExcess =
                static_cast<double>(std::max(0.0f, maxDrift - softDriftLimit)) /
                std::max(1.0, static_cast<double>(softDriftLimit));
            addSplitSoftGeometryPenalty(
                "daughter_seed_drift",
                normalizedExcess,
                static_cast<double>(
                    std::max(0.0f, probConfig.split_soft_geometry_penalty_fraction)));
        }

        const float softFinalSepFraction =
            std::max(0.0f,
                     probConfig
                         .split_soft_max_final_separation_parent_fraction);
        const float softFinalSepLimit =
            softFinalSepFraction * parentMaxRadiusForSoftGeometry;
        if (softFinalSepLimit > 0.0f) {
            const double normalizedExcess =
                static_cast<double>(
                    std::max(0.0f, finalAxisLen - softFinalSepLimit)) /
                std::max(1.0, static_cast<double>(softFinalSepLimit));
            if (bestIsFutureSupportedBridgeAxisPlace && normalizedExcess > 0.0) {
                std::cout << "[Split Soft Gate Waived] " << parentName
                          << " reason=bridge_axis_place_final_separation"
                          << " normalizedExcess=" << normalizedExcess
                          << " finalAxisLen=" << finalAxisLen
                          << " softFinalSepLimit=" << softFinalSepLimit
                          << " windowBoth="
                          << bridgeProposal->windowBothDaughtersSupported
                          << " futureImmediate="
                          << bridgeProposal->windowImmediateBothDaughtersSupported
                          << " futureMissing="
                          << bridgeProposal->windowMissingDaughterCount
                          << " parentPersists="
                          << bridgeProposal->windowParentPersists
                          << std::endl;
            } else {
                addSplitSoftGeometryPenalty(
                    "daughter_final_separation",
                    normalizedExcess,
                    static_cast<double>(
                        std::max(0.0f, probConfig.split_soft_geometry_penalty_fraction)));
            }
        }
    }
    const float seedDx = bestSeedD2.x - bestSeedD1.x;
    const float seedDy = bestSeedD2.y - bestSeedD1.y;
    const float seedDz = bestSeedD2.z - bestSeedD1.z;
    const float seedLateralSeparation =
        std::sqrt(seedDx * seedDx + seedDy * seedDy);
    const float seedZDominance = seedAxisLen > 1e-3f
        ? std::abs(seedDz) / seedAxisLen
        : 0.0f;
    const float seedWindowLateralMin =
        useCellLumenGateParams
            ? std::max(0.0f, lumenMinPostRefitLateralSeparation)
            : 0.0f;
    const bool seedHasWindowLateralEvidence =
        seedLateralSeparation >= seedWindowLateralMin;
    const bool cleanWindowBackedLumenPrior =
        useCellLumenGateParams &&
        bestIsCellLumenPrior &&
        lumenProposal != nullptr &&
        lumenProposal->candidateIdA >= 0 &&
        lumenProposal->candidateIdB >= 0 &&
        lumenProposal->windowBothDaughtersSupported >= 2 &&
        lumenProposal->windowMissingDaughterCount == 0 &&
        lumenProposal->windowParentPersists == 0 &&
        lumenProposal->maxOverlapCostFractionOverride >= 0.0f;

    const float snapshotDriftMax = std::max(drift1, drift2);
    const float snapshotDriftMin = std::min(drift1, drift2);
    const float snapshotParentMinR =
        std::max(1e-5f, std::min({srcMajor, srcB, srcMinor}));
    const float snapshotParentShape = srcMaxR / snapshotParentMinR;
    const bool snapshotLargeShiftLooksSplitLike =
        useCellLumenGateParams &&
        bestIsCellLumenSnapshotSeedFallback &&
        lumenSnapshotSeedMaxRefitDrift >= 0.0f &&
        snapshotParentShape >= 1.65f &&
        seedAxisLen > 1e-3f &&
        finalAxisLen >= seedAxisLen * 2.20f &&
        snapshotDriftMin <= lumenSnapshotSeedMaxRefitDrift * 1.30f &&
        snapshotDriftMax <= lumenSnapshotSeedMaxRefitDrift * 2.10f;
    if (useCellLumenGateParams &&
        bestIsCellLumenSnapshotSeedFallback &&
        lumenSnapshotSeedMaxRefitDrift >= 0.0f &&
        snapshotDriftMax > lumenSnapshotSeedMaxRefitDrift &&
        !snapshotLargeShiftLooksSplitLike) {
        std::cout << "[Split Reject CellLumen snapshot refit drift] " << parentName
                  << " drift1=" << drift1
                  << " drift2=" << drift2
                  << " maxRefitDrift=" << lumenSnapshotSeedMaxRefitDrift
                  << " seedAxisLen=" << seedAxisLen
                  << " finalAxisLen=" << finalAxisLen
                  << " d1=(" << bestD1Pos.x << "," << bestD1Pos.y << "," << bestD1Pos.z << ")"
                  << " d2=(" << bestD2Pos.x << "," << bestD2Pos.y << "," << bestD2Pos.z << ")"
                  << " bestIdx=" << bestIdx
                  << " bestLabel=" << bestLabel
                  << std::endl;
        restoreLiveParent();
        return {0.0, noop};
    } else if (snapshotLargeShiftLooksSplitLike &&
               snapshotDriftMax > lumenSnapshotSeedMaxRefitDrift) {
        std::cout << "[Split CellLumen snapshot refit drift waived] " << parentName
                  << " drift1=" << drift1
                  << " drift2=" << drift2
                  << " maxRefitDrift=" << lumenSnapshotSeedMaxRefitDrift
                  << " seedAxisLen=" << seedAxisLen
                  << " finalAxisLen=" << finalAxisLen
                  << " parentShape=" << snapshotParentShape
                  << " reason=large_shift_axis_expansion"
                  << " bestIdx=" << bestIdx
                  << " bestLabel=" << bestLabel
                  << std::endl;
    }

    // CellLumen centers are excellent for XY-localizing real daughter cells,
    // but a single elongated nucleus can sometimes be over-segmented as two
    // centers stacked almost perfectly along interpolated Z. After the daughter
    // PCA refit, reject that pattern before the softer cost gates can rescue it.
    if (useCellLumenGateParams) {
        const float dx = bestD2Pos.x - bestD1Pos.x;
        const float dy = bestD2Pos.y - bestD1Pos.y;
        const float dz = bestD2Pos.z - bestD1Pos.z;
        const float lateralSeparation = std::sqrt(dx * dx + dy * dy);
        const float zDominance = finalAxisLen > 1e-3f
            ? std::abs(dz) / finalAxisLen
            : 0.0f;
        const float minConfigured = std::max(0.0f, lumenMinPostRefitLateralSeparation);
        const float minByRadius =
            std::max(0.0f, lumenMinPostRefitLateralSeparationRadiusScale) *
            std::max(1.0f, srcMaxR);
        const float minLateralSeparation = std::max(minConfigured, minByRadius);
        const float maxZDominance =
            std::clamp(lumenMaxZDominanceForLowLateralSeparation, 0.0f, 1.0f);
        if (minLateralSeparation > 0.0f &&
            lateralSeparation < minLateralSeparation &&
            zDominance > maxZDominance) {
            if (cleanWindowBackedLumenPrior && seedHasWindowLateralEvidence) {
                const double normalizedExcess =
                    static_cast<double>(minLateralSeparation - lateralSeparation) /
                    std::max(1.0, static_cast<double>(minLateralSeparation));
                addLumenSoftGatePenalty(
                    "post_refit_z_stack_duplicate",
                    normalizedExcess,
                    static_cast<double>(lumenSoftBridgeGapPenaltyFraction));
                std::cout << "[Split CellLumen Soft Gate Waived] " << parentName
                          << " reason=post_refit_z_stack_duplicate"
                          << " lateralSeparation=" << lateralSeparation
                          << " minLateralSeparation=" << minLateralSeparation
                          << " seedLateralSeparation=" << seedLateralSeparation
                          << " seedZDominance=" << seedZDominance
                          << " zDominance=" << zDominance
                          << " windowBoth="
                          << lumenProposal->windowBothDaughtersSupported
                          << " bestIdx=" << bestIdx
                          << " bestLabel=" << bestLabel
                          << std::endl;
            } else {
                std::cout << "[Split Reject CellLumen lateral] " << parentName
                          << " reason=post_refit_z_stack_duplicate"
                          << " lateralSeparation=" << lateralSeparation
                          << " minLateralSeparation=" << minLateralSeparation
                          << " configuredMin=" << minConfigured
                          << " radiusScaleMin=" << minByRadius
                          << " zDominance=" << zDominance
                          << " maxZDominance=" << maxZDominance
                          << " seedLateralSeparation=" << seedLateralSeparation
                          << " seedZDominance=" << seedZDominance
                          << " windowBacked=" << (cleanWindowBackedLumenPrior ? 1 : 0)
                          << " finalAxisLen=" << finalAxisLen
                          << " d1=(" << bestD1Pos.x << "," << bestD1Pos.y << "," << bestD1Pos.z << ")"
                          << " d2=(" << bestD2Pos.x << "," << bestD2Pos.y << "," << bestD2Pos.z << ")"
                          << " bestIdx=" << bestIdx
                          << " bestLabel=" << bestLabel
                          << std::endl;
                restoreLiveParent();
                return {0.0, noop};
            }
        }
    }

    if (probConfig.split_geometry_gate_enabled) {
        const float parentMaxRadius = std::max({srcMajor, srcB, srcMinor});
        const float driftFraction = probConfig.split_max_daughter_seed_drift_fraction;
        const float axisExpansionLimit = probConfig.split_max_daughter_axis_expansion;
        const float maxAllowedDrift = (driftFraction > 0.0f)
            ? parentMaxRadius * driftFraction
            : std::numeric_limits<float>::infinity();
        const bool driftTooLarge =
            driftFraction > 0.0f && std::max(drift1, drift2) > maxAllowedDrift;
        const bool axisExpandedTooMuch =
            axisExpansionLimit > 0.0f &&
            seedAxisLen > 1e-3f &&
            finalAxisLen > seedAxisLen * axisExpansionLimit;

        if (driftTooLarge || axisExpandedTooMuch) {
            std::cout << "[Split Reject bio] " << parentName
                      << " reason=daughter_geometry_drift"
                      << " drift1=" << drift1
                      << " drift2=" << drift2
                      << " maxAllowedDrift=" << maxAllowedDrift
                      << " parentMaxRadius=" << parentMaxRadius
                      << " driftFraction=" << driftFraction
                      << " seedAxisLen=" << seedAxisLen
                      << " finalAxisLen=" << finalAxisLen
                      << " axisExpansion=" << (seedAxisLen > 1e-3f ? finalAxisLen / seedAxisLen : 0.0f)
                      << " axisExpansionLimit=" << axisExpansionLimit
                      << " bestIdx=" << bestIdx
                      << " bestLabel=" << bestLabel
                      << std::endl;
            restoreLiveParent();
            return {0.0, noop};
        }
    }

    if (probConfig.split_axis_alignment_gate_enabled) {
        cv::Point3f parentShortAxis;
        float parentShortRadius = 0.0f;
        parent.worldSplitAxis(parentShortAxis, parentShortRadius);

        auto foldedAxisAngleDeg = [](const cv::Point3f &a,
                                     const cv::Point3f &b) -> float {
            const double an = cv::norm(a);
            const double bn = cv::norm(b);
            if (an <= 1e-9 || bn <= 1e-9) {
                return 0.0f;
            }
            const double dot = std::abs(
                (static_cast<double>(a.x) * b.x +
                 static_cast<double>(a.y) * b.y +
                 static_cast<double>(a.z) * b.z) / (an * bn));
            return static_cast<float>(
                std::acos(std::clamp(dot, 0.0, 1.0)) * 180.0 / M_PI);
        };

        cv::Point3f d1ShortAxis, d2ShortAxis;
        float d1ShortRadius = 0.0f;
        float d2ShortRadius = 0.0f;
        bestD1.worldSplitAxis(d1ShortAxis, d1ShortRadius);
        bestD2.worldSplitAxis(d2ShortAxis, d2ShortRadius);

        const float parentElongation = std::max(
            1.0f,
            (snapshotValid && snapshot.shapeElongation > 0.0f)
                ? snapshot.shapeElongation
                : parent.shapeElongation());
        const float sphereAngle =
            std::max(0.0f, probConfig.split_axis_alignment_sphere_angle_degrees);
        const float shrink =
            std::max(0.0f, probConfig.split_axis_alignment_elongation_shrink);
        const float minAngle =
            std::max(0.0f, probConfig.split_axis_alignment_min_angle_degrees);
        const float allowedAngle = std::max(
            minAngle,
            sphereAngle / (1.0f + shrink * std::max(0.0f, parentElongation - 1.0f)));
        float effectiveAllowedAngle = allowedAngle;
        const bool highConfidenceLumenAxisPrior =
            bestIsCellLumenPrior &&
            lumenProposal != nullptr &&
            lumenHighConfidenceMaxScore >= 0.0f &&
            lumenHighConfidenceAxisAlignmentDegrees >= 0.0f &&
            lumenProposal->elongation <= lumenHighConfidenceMaxScore;
        if (highConfidenceLumenAxisPrior) {
            effectiveAllowedAngle = std::max(
                effectiveAllowedAngle,
                std::max(0.0f, lumenHighConfidenceAxisAlignmentDegrees));
        }
        const float d1AxisAngle = foldedAxisAngleDeg(parentShortAxis, d1ShortAxis);
        const float d2AxisAngle = foldedAxisAngleDeg(parentShortAxis, d2ShortAxis);

        auto foldedAxisAlignment = [](const cv::Point3f &a,
                                      const cv::Point3f &b) -> float {
            const double an = cv::norm(a);
            const double bn = cv::norm(b);
            if (an <= 1e-9 || bn <= 1e-9) {
                return 0.0f;
            }
            return static_cast<float>(std::abs(
                (static_cast<double>(a.x) * b.x +
                 static_cast<double>(a.y) * b.y +
                 static_cast<double>(a.z) * b.z) / (an * bn)));
        };
        const cv::Point3f daughterSplitAxis = bestD2Pos - bestD1Pos;
        const float daughterSplitAxisLen =
            static_cast<float>(cv::norm(daughterSplitAxis));
        const float d1SplitLineAlignment =
            foldedAxisAlignment(daughterSplitAxis, d1ShortAxis);
        const float d2SplitLineAlignment =
            foldedAxisAlignment(daughterSplitAxis, d2ShortAxis);
        const bool generatedDaughtersAlignedToSplitAxis =
            simulationConfig.celluniverse2_enabled &&
            probConfig.split_daughter_refit_align_short_axis_to_split_enabled &&
            daughterSplitAxisLen > 1e-3f &&
            d1SplitLineAlignment >= 0.98f &&
            d2SplitLineAlignment >= 0.98f;

        if (d1AxisAngle > effectiveAllowedAngle || d2AxisAngle > effectiveAllowedAngle) {
            const float worstAxisAngle = std::max(d1AxisAngle, d2AxisAngle);
            const double normalizedAxisExcess =
                static_cast<double>(
                    std::max(0.0f, worstAxisAngle - effectiveAllowedAngle)) /
                std::max(1.0, 90.0 - static_cast<double>(effectiveAllowedAngle));
            const bool deterministicAxisWaived =
                simulationConfig.celluniverse2_enabled &&
                bestIsDeterministicSingleProposal &&
                (bestIsSignalCenterProposal ||
                 (bridgeProposal != nullptr &&
                  bridgeProposal->windowImmediateBothDaughtersSupported > 0));
            if (simulationConfig.celluniverse2_enabled &&
                generatedDaughtersAlignedToSplitAxis) {
                std::cout << "[Split Soft Gate Waived] " << parentName
                          << " reason=daughter_short_axis_alignment"
                          << " splitAxisAligned=1"
                          << " splitAxisLen=" << daughterSplitAxisLen
                          << " d1SplitLineAlignment=" << d1SplitLineAlignment
                          << " d2SplitLineAlignment=" << d2SplitLineAlignment
                          << " parentElongation=" << parentElongation
                          << " allowedAngleDeg=" << effectiveAllowedAngle
                          << " d1ParentAxisAngleDeg=" << d1AxisAngle
                          << " d2ParentAxisAngleDeg=" << d2AxisAngle
                          << " normalizedExcess=" << normalizedAxisExcess
                          << " signalCenterProposal="
                          << (bestIsSignalCenterProposal ? 1 : 0)
                          << " immediateFutureSupport="
                          << (bridgeProposal != nullptr
                                  ? bridgeProposal->windowImmediateBothDaughtersSupported
                                  : 0)
                          << " bestIdx=" << bestIdx
                          << " bestLabel=" << bestLabel
                          << std::endl;
            } else if (simulationConfig.celluniverse2_enabled) {
                addSplitSoftGeometryPenalty(
                    "daughter_short_axis_alignment",
                    normalizedAxisExcess,
                    static_cast<double>(
                        std::max(0.0f,
                                 probConfig.split_close_axis_penalty_fraction)));
                std::cout << "[Split Soft Gate] " << parentName
                          << " reason=daughter_short_axis_alignment"
                          << " parentElongation=" << parentElongation
                          << " allowedAngleDeg=" << effectiveAllowedAngle
                          << " d1AngleDeg=" << d1AxisAngle
                          << " d2AngleDeg=" << d2AxisAngle
                          << " d1SplitLineAlignment=" << d1SplitLineAlignment
                          << " d2SplitLineAlignment=" << d2SplitLineAlignment
                          << " normalizedExcess=" << normalizedAxisExcess
                          << " signalCenterProposal="
                          << (bestIsSignalCenterProposal ? 1 : 0)
                          << " immediateFutureSupport="
                          << (bridgeProposal != nullptr
                                  ? bridgeProposal->windowImmediateBothDaughtersSupported
                                  : 0)
                          << " bestIdx=" << bestIdx
                          << " bestLabel=" << bestLabel
                          << std::endl;
            } else if (deterministicAxisWaived) {
                std::cout << "[Split Soft Gate] " << parentName
                          << " reason=deterministic_daughter_axis_alignment"
                          << " parentElongation=" << parentElongation
                          << " allowedAngleDeg=" << effectiveAllowedAngle
                          << " d1AngleDeg=" << d1AxisAngle
                          << " d2AngleDeg=" << d2AxisAngle
                          << " signalCenterProposal="
                          << (bestIsSignalCenterProposal ? 1 : 0)
                          << " immediateFutureSupport="
                          << (bridgeProposal != nullptr
                                  ? bridgeProposal->windowImmediateBothDaughtersSupported
                                  : 0)
                          << " bestIdx=" << bestIdx
                          << " bestLabel=" << bestLabel
                          << std::endl;
            } else {
            if (useCellLumenSoftGate) {
                addLumenSoftGatePenalty(
                    "daughter_short_axis_misaligned",
                    normalizedAxisExcess,
                    static_cast<double>(lumenSoftAxisPenaltyFraction));
            } else {
            std::cout << "[Split Reject bio] " << parentName
                      << " reason=daughter_short_axis_misaligned"
                      << " parentElongation=" << parentElongation
                      << " allowedAngleDeg=" << effectiveAllowedAngle
                      << " baseAllowedAngleDeg=" << allowedAngle
                      << " highConfidenceLumenAxisPrior=" << (highConfidenceLumenAxisPrior ? 1 : 0)
                      << " sphereAngleDeg=" << sphereAngle
                      << " shrink=" << shrink
                      << " minAngleDeg=" << minAngle
                      << " parentShortAxis=(" << parentShortAxis.x << "," << parentShortAxis.y << "," << parentShortAxis.z << ")"
                      << " d1ShortAxis=(" << d1ShortAxis.x << "," << d1ShortAxis.y << "," << d1ShortAxis.z << ")"
                      << " d1AngleDeg=" << d1AxisAngle
                      << " d2ShortAxis=(" << d2ShortAxis.x << "," << d2ShortAxis.y << "," << d2ShortAxis.z << ")"
                      << " d2AngleDeg=" << d2AxisAngle
                      << " bestIdx=" << bestIdx
                      << " bestLabel=" << bestLabel
                      << std::endl;
            restoreLiveParent();
            return {0.0, noop};
            }
            }
        }

        if (simulationConfig.celluniverse2_enabled &&
            probConfig.split_daughter_axis_interaction_soft_gate_enabled) {
            const auto normalizeAxis = [](const cv::Point3f &axis) {
                const float norm = static_cast<float>(cv::norm(axis));
                if (norm <= 1e-6f) {
                    return cv::Point3f(0.0f, 0.0f, 0.0f);
                }
                return cv::Point3f(axis.x / norm, axis.y / norm, axis.z / norm);
            };
            const cv::Point3f d1Axis = normalizeAxis(d1ShortAxis);
            const cv::Point3f d2Axis = normalizeAxis(d2ShortAxis);
            if (cv::norm(d1Axis) > 0.0 && cv::norm(d2Axis) > 0.0) {
                const float d1ShortRadius = std::min({
                    bestD1.getARadius(), bestD1.getBRadius(), bestD1.getCRadius()});
                const float d2ShortRadius = std::min({
                    bestD2.getARadius(), bestD2.getBRadius(), bestD2.getCRadius()});
                const float d1MaxRadius = std::max({
                    bestD1.getARadius(), bestD1.getBRadius(), bestD1.getCRadius()});
                const float d2MaxRadius = std::max({
                    bestD2.getARadius(), bestD2.getBRadius(), bestD2.getCRadius()});
                const float avgShortRadius =
                    std::max(1.0f, 0.5f * (d1ShortRadius + d2ShortRadius));
                const float avgMaxRadius =
                    std::max(1.0f, 0.5f * (d1MaxRadius + d2MaxRadius));
                const float allowedLineDistance =
                    std::max(0.0f,
                             probConfig
                                 .split_daughter_axis_interaction_distance_fraction) *
                    avgShortRadius;
                const float allowedAlongDistance =
                    std::max(0.0f,
                             probConfig
                                 .split_daughter_axis_interaction_along_fraction) *
                    avgMaxRadius;

                const cv::Point3f w0 = bestD1Pos - bestD2Pos;
                const float axisDot = std::clamp(d1Axis.dot(d2Axis), -1.0f, 1.0f);
                const float foldedAxisDot = std::abs(axisDot);
                const float parallelAngle = std::clamp(
                    probConfig.split_daughter_axis_parallel_angle_degrees,
                    0.0f,
                    89.0f);
                const float parallelCos =
                    std::cos(parallelAngle * static_cast<float>(M_PI) / 180.0f);
                const bool nearlyParallel = foldedAxisDot >= parallelCos;
                const float denom = 1.0f - axisDot * axisDot;
                float lineDistance = 0.0f;
                float d1LineParam = 0.0f;
                float d2LineParam = 0.0f;
                float normalizedAlongExcess = 0.0f;

                if (denom <= 1e-4f || nearlyParallel) {
                    const cv::Point3f daughterDelta = bestD2Pos - bestD1Pos;
                    const float projection = d1Axis.dot(daughterDelta);
                    const cv::Point3f perpendicular(
                        daughterDelta.x - d1Axis.x * projection,
                        daughterDelta.y - d1Axis.y * projection,
                        daughterDelta.z - d1Axis.z * projection);
                    lineDistance = static_cast<float>(cv::norm(perpendicular));
                    d1LineParam = projection;
                    d2LineParam = 0.0f;
                } else {
                    const float d = d1Axis.dot(w0);
                    const float e = d2Axis.dot(w0);
                    d1LineParam = (axisDot * e - d) / denom;
                    d2LineParam = (e - axisDot * d) / denom;
                    const cv::Point3f closestOnD1(
                        bestD1Pos.x + d1Axis.x * d1LineParam,
                        bestD1Pos.y + d1Axis.y * d1LineParam,
                        bestD1Pos.z + d1Axis.z * d1LineParam);
                    const cv::Point3f closestOnD2(
                        bestD2Pos.x + d2Axis.x * d2LineParam,
                        bestD2Pos.y + d2Axis.y * d2LineParam,
                        bestD2Pos.z + d2Axis.z * d2LineParam);
                    lineDistance = static_cast<float>(
                        cv::norm(closestOnD1 - closestOnD2));
                    const float alongExcess =
                        std::max(0.0f,
                                 std::max(std::abs(d1LineParam),
                                          std::abs(d2LineParam)) -
                                     allowedAlongDistance);
                    normalizedAlongExcess =
                        alongExcess / std::max(1.0f, allowedAlongDistance);
                }

                const float distanceExcess =
                    std::max(0.0f, lineDistance - allowedLineDistance);
                const float normalizedDistanceExcess =
                    distanceExcess / std::max(1.0f, allowedLineDistance);
                const double normalizedExcess =
                    static_cast<double>(normalizedDistanceExcess +
                                        normalizedAlongExcess);
                addSplitSoftGeometryPenalty(
                    "daughter_short_axis_intersection",
                    normalizedExcess,
                    static_cast<double>(
                        std::max(
                            0.0f,
                            probConfig
                                .split_daughter_axis_interaction_penalty_fraction)));
                if (normalizedExcess > 0.0) {
                    std::cout << "[Split Soft Gate] " << parentName
                              << " reason=daughter_short_axis_intersection"
                              << " lineDistance=" << lineDistance
                              << " allowedLineDistance=" << allowedLineDistance
                              << " normalizedDistanceExcess="
                              << normalizedDistanceExcess
                              << " d1LineParam=" << d1LineParam
                              << " d2LineParam=" << d2LineParam
                              << " allowedAlongDistance=" << allowedAlongDistance
                              << " normalizedAlongExcess="
                              << normalizedAlongExcess
                              << " nearlyParallel=" << (nearlyParallel ? 1 : 0)
                              << " axisAngleDeg="
                              << foldedAxisAngleDeg(d1Axis, d2Axis)
                              << " bestIdx=" << bestIdx
                              << " bestLabel=" << bestLabel
                              << std::endl;
                }
            }
        }

        if (simulationConfig.celluniverse2_enabled &&
            probConfig.split_close_axis_soft_gate_enabled) {
            cv::Point3f daughterDelta = bestD2Pos - bestD1Pos;
            const float daughterDistance =
                static_cast<float>(cv::norm(daughterDelta));
            if (daughterDistance > 1e-3f) {
                daughterDelta *= (1.0f / daughterDistance);
                const float maxD1R = std::max({
                    bestD1.getARadius(), bestD1.getBRadius(), bestD1.getCRadius()});
                const float maxD2R = std::max({
                    bestD2.getARadius(), bestD2.getBRadius(), bestD2.getCRadius()});
                const float closeLimit =
                    std::max(0.0f,
                             probConfig.split_close_axis_distance_scale) *
                    std::max(1.0f, 0.5f * (maxD1R + maxD2R));
                const float minCenterAxisAlign =
                    std::clamp(
                        probConfig.split_close_axis_center_axis_alignment,
                        0.0f,
                        1.0f);
                const float d1CenterAlign =
                    std::abs(daughterDelta.dot(d1ShortAxis)) /
                    std::max(1e-6f, static_cast<float>(cv::norm(d1ShortAxis)));
                const float d2CenterAlign =
                    std::abs(daughterDelta.dot(d2ShortAxis)) /
                    std::max(1e-6f, static_cast<float>(cv::norm(d2ShortAxis)));
                if (closeLimit > 0.0f &&
                    daughterDistance <= closeLimit &&
                    d1CenterAlign >= minCenterAxisAlign &&
                    d2CenterAlign >= minCenterAxisAlign) {
                    const float daughterShortAxisAngle =
                        foldedAxisAngleDeg(d1ShortAxis, d2ShortAxis);
                    const float allowedDaughterAxisAngle =
                        std::max(
                            0.0f,
                            probConfig.split_close_axis_max_angle_degrees);
                    const double normalizedExcess =
                        static_cast<double>(
                            std::max(
                                0.0f,
                                daughterShortAxisAngle -
                                    allowedDaughterAxisAngle)) /
                        std::max(
                            1.0,
                            90.0 -
                                static_cast<double>(
                                    std::min(89.0f,
                                             allowedDaughterAxisAngle)));
                    addSplitSoftGeometryPenalty(
                        "close_daughter_short_axis_mismatch",
                        normalizedExcess,
                        static_cast<double>(
                            std::max(
                                0.0f,
                                probConfig
                                    .split_close_axis_penalty_fraction)));
                    if (normalizedExcess > 0.0) {
                        std::cout << "[Split Soft Gate] " << parentName
                                  << " reason=close_daughter_short_axis_mismatch"
                                  << " daughterDistance=" << daughterDistance
                                  << " closeLimit=" << closeLimit
                                  << " d1CenterAlign=" << d1CenterAlign
                                  << " d2CenterAlign=" << d2CenterAlign
                                  << " daughterShortAxisAngle="
                                  << daughterShortAxisAngle
                                  << " allowedAngle="
                                  << allowedDaughterAxisAngle
                                  << " bestIdx=" << bestIdx
                                  << " bestLabel=" << bestLabel
                                  << std::endl;
                    }
                }
            }
        }
    }

    if (probConfig.split_daughter_overlap_gate_enabled) {
        const float baseMaxDaughterOverlap = std::clamp(
            (useCellLumenGateParams && lumenMaxDaughterOverlapFraction >= 0.0f)
                ? lumenMaxDaughterOverlapFraction
                : probConfig.split_max_daughter_overlap_fraction,
            0.0f, 1.0f);
        float maxDaughterOverlap = baseMaxDaughterOverlap;
        if (useCellLumenSoftGate && lumenDynamicOverlapEnabled) {
            const float dynamicBonus =
                static_cast<float>(lumenLocalNeighborCount) *
                std::max(0.0f, lumenLocalDensityOverlapBonus);
            maxDaughterOverlap = std::min(
                std::clamp(lumenMaxDynamicDaughterOverlapFraction, baseMaxDaughterOverlap, 1.0f),
                baseMaxDaughterOverlap + dynamicBonus);
        }
        const float daughterOverlapScale = std::max(
            1e-3f, probConfig.split_daughter_overlap_scale);
        const float d1InD2Overlap = ellipsoidOverlapFractionOfFirst(
            _realFrame, bestD1, bestD2, daughterOverlapScale, daughterOverlapScale);
        const float d2InD1Overlap = ellipsoidOverlapFractionOfFirst(
            _realFrame, bestD2, bestD1, daughterOverlapScale, daughterOverlapScale);
        const float daughterOverlap = std::max(d1InD2Overlap, d2InD1Overlap);

        if (daughterOverlap > maxDaughterOverlap) {
            const float pcaBridgeSoftOverlapMax = std::clamp(
                probConfig.pca_bridge_soft_daughter_overlap_max, 0.0f, 1.0f);
            const float pcaBridgeFutureOverlapMax = std::clamp(
                probConfig.pca_bridge_future_window_rescue_overlap_max, 0.0f, 1.0f);
            const bool softPcaBridgeDaughterOverlap =
                simulationConfig.celluniverse2_enabled &&
                bestIsPcaBridgeOnly &&
                daughterOverlap <= pcaBridgeSoftOverlapMax;
            const bool immediateFutureBackedPcaBridgeDaughterOverlap =
                simulationConfig.celluniverse2_enabled &&
                bestIsPcaBridgeOnly &&
                probConfig.pca_bridge_future_window_enabled &&
                bridgeProposal != nullptr &&
                daughterOverlap <= pcaBridgeFutureOverlapMax &&
                bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
                bridgeProposal->windowParentPersists <=
                    std::max(0, probConfig.pca_bridge_future_window_max_parent_persists);
            const bool futureBackedPcaBridgeDaughterOverlap =
                simulationConfig.celluniverse2_enabled &&
                bestIsPcaBridgeOnly &&
                probConfig.pca_bridge_future_window_enabled &&
                bridgeProposal != nullptr &&
                daughterOverlap <= pcaBridgeFutureOverlapMax &&
                bridgeProposal->windowBothDaughtersSupported >=
                    std::max(1, probConfig.pca_bridge_future_window_min_both_daughter_support) &&
                bridgeProposal->windowMissingDaughterCount <=
                    std::max(0, probConfig.pca_bridge_future_window_max_missing_daughters) &&
                bridgeProposal->windowParentPersists <=
                    std::max(0, probConfig.pca_bridge_future_window_max_parent_persists);
            const bool futureBackedBridgeAxisPlaceDaughterOverlap =
                simulationConfig.celluniverse2_enabled &&
                bestIsFutureSupportedBridgeAxisPlace &&
                daughterOverlap <= pcaBridgeFutureOverlapMax;
            const bool futureBackedRodTipDaughterOverlap =
                simulationConfig.celluniverse2_enabled &&
                bestIsDeterministicSingleProposal &&
                bridgeProposal != nullptr &&
                bridgeProposal->daughterSphereRadius > 0.0f &&
                daughterOverlap <= pcaBridgeFutureOverlapMax &&
                bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
                bridgeProposal->windowBothDaughtersSupported >=
                    std::max(1, probConfig.pca_bridge_future_window_min_both_daughter_support) &&
                bridgeProposal->windowMissingDaughterCount <=
                    std::max(0, probConfig.pca_bridge_future_window_max_missing_daughters) &&
                bridgeProposal->windowParentPersists <=
                    std::max(0, probConfig.pca_bridge_future_window_max_parent_persists);
            const bool impossibleOverlap =
                (!useCellLumenSoftGate &&
                 !softPcaBridgeDaughterOverlap &&
                 !immediateFutureBackedPcaBridgeDaughterOverlap &&
                 !futureBackedPcaBridgeDaughterOverlap &&
                 !futureBackedBridgeAxisPlaceDaughterOverlap &&
                 !futureBackedRodTipDaughterOverlap) ||
                daughterOverlap > std::clamp(lumenHardMaxDaughterOverlapFraction, 0.0f, 1.0f);
            if (!impossibleOverlap) {
                if (softPcaBridgeDaughterOverlap ||
                    immediateFutureBackedPcaBridgeDaughterOverlap ||
                    futureBackedPcaBridgeDaughterOverlap ||
                    futureBackedBridgeAxisPlaceDaughterOverlap ||
                    futureBackedRodTipDaughterOverlap) {
                    std::cout << "[Split Soft Gate] " << parentName
                              << " reason="
                              << ((futureBackedRodTipDaughterOverlap)
                                      ? "rod_tip_future_window_overlap"
                                      : ((futureBackedBridgeAxisPlaceDaughterOverlap)
                                      ? "bridge_axis_place_future_window_overlap"
                                      : ((futureBackedPcaBridgeDaughterOverlap ||
                                          immediateFutureBackedPcaBridgeDaughterOverlap)
                                      ? "pca_bridge_future_window_overlap"
                                      : "pca_bridge_daughter_overlap")))
                              << " d1InD2Overlap=" << d1InD2Overlap
                              << " d2InD1Overlap=" << d2InD1Overlap
                              << " daughterOverlap=" << daughterOverlap
                              << " maxAllowed=" << maxDaughterOverlap
                              << " tolerance=" << pcaBridgeSoftOverlapMax
                              << " futureTolerance=" << pcaBridgeFutureOverlapMax
                              << " windowBoth="
                              << (bridgeProposal != nullptr
                                      ? bridgeProposal->windowBothDaughtersSupported
                                      : 0)
                              << " windowImmediateBoth="
                              << (bridgeProposal != nullptr
                                      ? bridgeProposal->windowImmediateBothDaughtersSupported
                                      : 0)
                              << " windowMissing="
                              << (bridgeProposal != nullptr
                                      ? bridgeProposal->windowMissingDaughterCount
                                      : 0)
                              << " windowParentPersists="
                              << (bridgeProposal != nullptr
                                      ? bridgeProposal->windowParentPersists
                                      : 0)
                              << " bestIdx=" << bestIdx
                              << " bestLabel=" << bestLabel
                              << std::endl;
                } else {
                    const double normalizedExcess =
                        static_cast<double>(daughterOverlap - maxDaughterOverlap) /
                        std::max(0.05, 1.0 - static_cast<double>(maxDaughterOverlap));
                    addLumenSoftGatePenalty(
                        "daughter_daughter_overlap",
                        normalizedExcess,
                        static_cast<double>(lumenSoftDaughterOverlapPenaltyFraction));
                }
            } else {
            std::cout << "[Split Reject bio] " << parentName
                      << " reason=daughter_daughter_overlap"
                      << " d1InD2Overlap=" << d1InD2Overlap
                      << " d2InD1Overlap=" << d2InD1Overlap
                      << " daughterOverlap=" << daughterOverlap
                      << " maxAllowed=" << maxDaughterOverlap
                      << " baseMaxAllowed=" << baseMaxDaughterOverlap
                      << " hardMaxAllowed=" << lumenHardMaxDaughterOverlapFraction
                      << " daughterScale=" << daughterOverlapScale
                      << " bestIdx=" << bestIdx
                      << " bestLabel=" << bestLabel
                      << std::endl;
            restoreLiveParent();
            return {0.0, noop};
            }
        }
    }

    // Daughter midpoint (shared by the bridge gate below for axis
    // projection and diagnostic logging). Previously also used by a
    // midpoint-near-snapshot-parent gate (2026-04-11 afternoon), but
    // that gate was removed because (a) the pre-pass grounds
    // snapshot.position onto the image bright region, which pulls the
    // reference toward the daughters and makes the check trivially
    // pass (see e3d03 f3 run 082245), and (b) the bridge brightness
    // gate catches the same false-split patterns directly via the
    // image content.
    const cv::Point3f daughterMidpoint(
        0.5f * (bestD1.getX() + bestD2.getX()),
        0.5f * (bestD1.getY() + bestD2.getY()),
        0.5f * (bestD1.getZ() + bestD2.getZ()));

    // 5a''. Bridge brightness gate — project the Voronoi-filtered bright
    // pixels (the same set PCA used) onto the daughter split axis. Real
    // divisions have a dim gap in the middle (dividing groove): low pixel
    // density and low mean brightness in the central bin. A fake split
    // over one continuous cell has uniform pixels across the middle.
    //
    // Normalized axis coordinate t in [-1, +1]:
    //   t = -1 ⇔ bestD1 center
    //   t =  0 ⇔ midpoint
    //   t = +1 ⇔ bestD2 center
    //
    // Only pixels with |t| < 1.5 are considered (ignore pixels way
    // outside the daughter span, which inflate statistics without
    // helping).
    //
    // Gap:  |t| < 0.3 (middle ~30% of the span)
    // Edge: 0.6 < |t| < 1.1 (near each daughter center)
    //
    // Single-metric rejection: gap / max(edge1, edge2) > bio_bridge_max_valley_ratio.
    // Adaptive bridge gate: measure brightness in the actual gap between
    // the two daughters' surfaces, not a fixed fraction of the axis.
    //
    // For each daughter, compute the ellipsoid radius along the split axis
    // (how far the surface extends toward the other daughter). The gap is
    // the region between those two surfaces. Edge zones are the regions
    // inside each daughter, away from the gap.
    bool bridgeCostRescueEligible = false;
    bool lumenStrongBridgeEvidence = false;
    float bridgeCostRescueValleyFromBright = 1.0f;
    float bridgeCostRescueWorstValleyRatio = 1.0f;
    float bridgeCostRescueGapDensity = 1.0f;
    float bridgeCostRescueEdgeBright = 0.0f;
    float lumenBridgeGapWidth = -std::numeric_limits<float>::infinity();
    {
        const cv::Point3f axisVec = bestD2Pos - bestD1Pos;
        const float axisLen = static_cast<float>(cv::norm(axisVec));
        const int totalBridgeCandidates = static_cast<int>(pixels.size());

        if (axisLen > 1e-3f && totalBridgeCandidates >= 1000) {
            const cv::Point3f axisDir(
                axisVec.x / axisLen,
                axisVec.y / axisLen,
                axisVec.z / axisLen);

            // Ellipsoid radius along an arbitrary direction: for an ellipsoid
            // with semi-axes (a,b,c) and rotation R, the support distance
            // along world direction d is ||diag(a,b,c) * R^T * d||.
            auto ellipsoidRadiusAlongDir = [](const Ellipsoid &e,
                                              const cv::Point3f &dir) -> float {
                std::array<double, 9> RT;
                e.generateInverseRotationMatrix(RT);
                // R^T * dir (world → local)
                const double lx = RT[0]*dir.x + RT[1]*dir.y + RT[2]*dir.z;
                const double ly = RT[3]*dir.x + RT[4]*dir.y + RT[5]*dir.z;
                const double lz = RT[6]*dir.x + RT[7]*dir.y + RT[8]*dir.z;
                // Scale by semi-axes
                const double sa = e.getARadius() * lx;
                const double sb = e.getBRadius() * ly;
                const double sc = e.getCRadius() * lz;
                return static_cast<float>(std::sqrt(sa*sa + sb*sb + sc*sc));
            };

            const float r1Along = ellipsoidRadiusAlongDir(bestD1, axisDir);
            const float r2Along = ellipsoidRadiusAlongDir(bestD2, axisDir);
            const float gapWidth = axisLen - r1Along - r2Along;

            // Coordinate system: project onto axisDir, origin at midpoint.
            // d1 is at -halfLen, d2 is at +halfLen.
            const float halfLen = 0.5f * axisLen;

            // ALWAYS measure the central region between daughters, even
            // when they geometrically overlap (gapWidth <= 0). A real split
            // produces a brightness valley at the midpoint; a false split
            // on a non-dividing cell has continuous brightness through the
            // center. The minimum gap half-width of 15% of halfLen
            // guarantees we sample enough pixels to detect this.
            const float minGapHalf = 0.30f * halfLen;
            const float surfaceGapHalf = 0.5f * gapWidth; // negative if overlapping
            const float effectiveGapHalf = std::max(minGapHalf, surfaceGapHalf);
            const float gapLo = -effectiveGapHalf;
            const float gapHi =  effectiveGapHalf;

            // Edge zones: near each daughter center, outside the gap.
            // d1 is at -halfLen, so its edge zone is [-halfLen-r1Along, gapLo]
            // d2 is at +halfLen, so its edge zone is [gapHi, halfLen+r2Along]
            const float edge1Lo = -halfLen - r1Along;
            const float edge1Hi = gapLo;
            const float edge2Lo = gapHi;
            const float edge2Hi = halfLen + r2Along;

            int totalInRange = 0;
            int gapCount = 0;
            int edge1Count = 0, edge2Count = 0;
            double gapBrightSum = 0.0;
            double edge1BrightSum = 0.0, edge2BrightSum = 0.0;

            // Slab-binned gap brightness (2026-04-21). The legacy mean
            // `gapBrightSum / gapCount` smears across the whole gap zone
            // (~15 vx on a moderate a510 case). Adaptive-cube pooling uses
            // cubes ~0.6×minR ≈ 9 vx wide, so a single pooled cube bridging
            // the gap averages bright neighbors with the narrow dark bridge,
            // inflating the mean and masking real valleys. Example: a510
            // f39 of run 212000 had mean gapBright=0.161 → ratio 0.76 →
            // bridge_flat reject; perfect_45's equivalent at f40 read 0.15
            // → ratio 0.66 only because its cell was more bloated and the
            // split axis long enough to place daughters 48 vx apart (many
            // consecutive cubes dominated by dark voxels).
            //
            // The slab-min approach samples "darkest cross-section along
            // the bridge": bin the gap zone into kGapSlabs slabs of equal
            // width along the axis, compute mean per slab, take the slab
            // with minimum brightness as gapBright. Insensitive to a
            // single contaminated cube because its effect is confined to
            // one slab; any neighboring slabs that are truly dark still
            // show through. Slab width ≈ gapWidth/kGapSlabs ≈ 3 vx at
            // kGapSlabs=5 — smaller than the pooling cube, so per-cube
            // artifacts cannot dominate all slabs at once.
            //
            // Guard: only consider slabs with ≥ minPixPerSlab pixels to
            // avoid single-pixel noise dominating. Fallback to legacy mean
            // if no slab qualifies.
            static constexpr int kGapSlabs = 5;
            std::array<double, kGapSlabs> slabSum{};
            std::array<int,    kGapSlabs> slabCount{};
            const float slabWidth = (gapHi - gapLo) / static_cast<float>(kGapSlabs);

            for (const auto &bp : pixels) {
                const cv::Point3f delta(
                    bp.pos.x - daughterMidpoint.x,
                    bp.pos.y - daughterMidpoint.y,
                    bp.pos.z - daughterMidpoint.z);
                const float proj =
                    delta.x * axisDir.x +
                    delta.y * axisDir.y +
                    delta.z * axisDir.z;

                if (proj < edge1Lo || proj > edge2Hi) continue;
                ++totalInRange;

                if (proj >= gapLo && proj <= gapHi) {
                    ++gapCount;
                    gapBrightSum += bp.weight;
                    int bin = (slabWidth > 0.0f)
                        ? static_cast<int>((proj - gapLo) / slabWidth)
                        : 0;
                    if (bin < 0) bin = 0;
                    if (bin >= kGapSlabs) bin = kGapSlabs - 1;
                    slabSum[bin] += bp.weight;
                    slabCount[bin] += 1;
                } else if (proj >= edge1Lo && proj <= edge1Hi) {
                    ++edge1Count;
                    edge1BrightSum += bp.weight;
                } else if (proj >= edge2Lo && proj <= edge2Hi) {
                    ++edge2Count;
                    edge2BrightSum += bp.weight;
                }
            }

            const int edgeCount = edge1Count + edge2Count;
            const double edgeBrightSum = edge1BrightSum + edge2BrightSum;

            const float gapDensity = (totalInRange > 0)
                ? static_cast<float>(gapCount) / static_cast<float>(totalInRange)
                : 0.0f;
            // Legacy mean — kept for diagnostic continuity in the log line.
            const float gapBrightMean = (gapCount > 0)
                ? static_cast<float>(gapBrightSum / gapCount)
                : 0.0f;

            // Slab-min: darkest cross-section across the gap zone. Requires
            // ≥ minPixPerSlab pixels in the slab to be considered (guards
            // against a single stray dark pixel winning). Falls back to the
            // legacy mean when no slab meets the threshold (sparse gaps,
            // very short effective gap half-width).
            const int minPixPerSlab = std::max(3, gapCount / (kGapSlabs * 3));
            float gapBrightMinSlab = std::numeric_limits<float>::infinity();
            int   minSlabIdx = -1;
            for (int i = 0; i < kGapSlabs; ++i) {
                if (slabCount[i] >= minPixPerSlab) {
                    const float b = static_cast<float>(slabSum[i])
                                    / static_cast<float>(slabCount[i]);
                    if (b < gapBrightMinSlab) {
                        gapBrightMinSlab = b;
                        minSlabIdx = i;
                    }
                }
            }
            const float gapBright = (minSlabIdx >= 0)
                ? gapBrightMinSlab
                : gapBrightMean;
            const float edgeBright = (edgeCount > 0)
                ? static_cast<float>(edgeBrightSum / edgeCount)
                : 0.0f;
            const float edge1Bright = (edge1Count > 0)
                ? static_cast<float>(edge1BrightSum / edge1Count)
                : 0.0f;
            const float edge2Bright = (edge2Count > 0)
                ? static_cast<float>(edge2BrightSum / edge2Count)
                : 0.0f;
            // Per-daughter valley ratios. Pooling edges (gap/edgeAvg)
            // hides asymmetry — a bright real daughter averages with a
            // dim phantom daughter and the gap still looks like a valley.
            // Checking gap against EACH edge independently catches the
            // phantom case: if gap >= edge on one side, that daughter is
            // in near-empty space, not a real cell body.
            const float valleyRatio1 = (edge1Bright > 1e-6f)
                ? (gapBright / edge1Bright)
                : 1.0f;
            const float valleyRatio2 = (edge2Bright > 1e-6f)
                ? (gapBright / edge2Bright)
                : 1.0f;
            const float worstValleyRatio = std::max(valleyRatio1, valleyRatio2);
            // Pooled ratio kept for logging only (diagnostic continuity).
            const float valleyRatio = (edgeBright > 1e-6f)
                ? (gapBright / edgeBright)
                : 0.0f;
            // Valley metric based on the BRIGHTER daughter edge only.
            //
            // Asymmetric division (one daughter inherits more cytoplasm
            // and renders brighter) is biologically normal. The dim-
            // daughter's edge ≈ gap is expected — it doesn't indicate
            // "no valley", just that this daughter is small. The signal
            // that actually matters is whether the brighter daughter
            // shows a drop from its cell body into the midpoint.
            //
            // Taking max(edge1, edge2) as the reference naturally handles
            // both symmetric and asymmetric cases: for symmetric daughters
            // the two edges are equal, `gap/max(edges) = gap/either_edge`;
            // for asymmetric daughters the brighter side drives the
            // decision and the dim side doesn't punish it.
            //
            // Replaces the previous worstValleyRatio-based tiered decision
            // (2026-04-15). The legacy two-tier path (gap density + worst
            // valley ratio) was retired — it punished legitimate asymmetric
            // division.
            const float brighterEdge = std::max(edge1Bright, edge2Bright);
            const float valleyFromBright = (brighterEdge > 1e-6f)
                ? (gapBright / brighterEdge)
                : 1.0f;

            std::cout << "  [Split Bridge] " << parentName
                      << " axisLen=" << axisLen
                      << " r1Along=" << r1Along
                      << " r2Along=" << r2Along
                      << " gapWidth=" << gapWidth
                      << " effGapHalf=" << effectiveGapHalf
                      << " totalInRange=" << totalInRange
                      << " gapCount=" << gapCount
                      << " edgeCount=" << edgeCount
                      << " gapDensity=" << gapDensity
                      << " gapBright=" << gapBright
                      << " gapBrightMean=" << gapBrightMean
                      << " gapBrightMinSlab=" << gapBrightMinSlab
                      << " minSlabIdx=" << minSlabIdx
                      << " edge1Bright=" << edge1Bright
                      << " edge2Bright=" << edge2Bright
                      << " edgeBright=" << edgeBright
                      << " valleyRatio1=" << valleyRatio1
                      << " valleyRatio2=" << valleyRatio2
                      << " worstValleyRatio=" << worstValleyRatio
                      << " valleyRatioPooled=" << valleyRatio
                      << " valleyFromBright=" << valleyFromBright
                      << std::endl;

            if (useCellLumenGateParams && lumenMinBridgeGapWidth >= 0.0f &&
                gapWidth < lumenMinBridgeGapWidth) {
                const float noValleyHardThreshold =
                    std::max(0.0f, probConfig.bio_bridge_no_valley_hard_threshold);
                const bool noValleyInsideOverlap =
                    edgeCount > 0 &&
                    valleyFromBright >= noValleyHardThreshold;
                if (noValleyInsideOverlap) {
                    const float priorScoreWithoutWindowBonus =
                        (lumenProposal != nullptr)
                            ? lumenProposal->elongation +
                                  std::max(0.0f, lumenProposal->balancedWindowBonus)
                            : std::numeric_limits<float>::infinity();
                    const bool futureCanSoftenNoValley =
                        cleanWindowBackedLumenPrior &&
                        seedHasWindowLateralEvidence &&
                        priorScoreWithoutWindowBonus <=
                            std::max(0.0f, lumenHighConfidenceMaxScore);
                    if (futureCanSoftenNoValley && useCellLumenSoftGate) {
                        const double normalizedExcess =
                            static_cast<double>(
                                valleyFromBright - noValleyHardThreshold);
                        addLumenSoftGatePenalty(
                            "lumen_overlap_no_valley",
                            normalizedExcess,
                            static_cast<double>(lumenSoftValleyPenaltyFraction));
                        std::cout << "[Split CellLumen Soft Gate Waived] " << parentName
                                  << " reason=lumen_overlap_no_valley"
                                  << " gapWidth=" << gapWidth
                                  << " valleyFromBright=" << valleyFromBright
                                  << " noValleyHardThreshold=" << noValleyHardThreshold
                                  << " seedLateralSeparation=" << seedLateralSeparation
                                  << " priorScoreWithoutWindowBonus="
                                  << priorScoreWithoutWindowBonus
                                  << " windowBoth="
                                  << lumenProposal->windowBothDaughtersSupported
                                  << std::endl;
                    } else {
                        std::cout << "[Split Reject bio] " << parentName
                                  << " reason=lumen_overlap_no_valley"
                                  << " gapWidth=" << gapWidth
                                  << " minBridgeGapWidth=" << lumenMinBridgeGapWidth
                                  << " valleyFromBright=" << valleyFromBright
                                  << " noValleyHardThreshold=" << noValleyHardThreshold
                                  << " gapBright=" << gapBright
                                  << " edge1Bright=" << edge1Bright
                                  << " edge2Bright=" << edge2Bright
                                  << " seedLateralSeparation=" << seedLateralSeparation
                                  << " priorScoreWithoutWindowBonus="
                                  << priorScoreWithoutWindowBonus
                                  << " axisLen=" << axisLen
                                  << " r1Along=" << r1Along
                                  << " r2Along=" << r2Along
                                  << std::endl;
                        restoreLiveParent();
                        return {0.0, noop};
                    }
                }
                if (useCellLumenSoftGate) {
                    const double normalizedExcess =
                        static_cast<double>(lumenMinBridgeGapWidth - gapWidth) /
                        std::max(1.0, static_cast<double>(std::max(1.0f, srcMaxR)));
                    addLumenSoftGatePenalty(
                        "lumen_bridge_gap_too_small",
                        normalizedExcess,
                        static_cast<double>(lumenSoftBridgeGapPenaltyFraction));
                } else {
                std::cout << "[Split Reject bio] " << parentName
                          << " reason=lumen_bridge_gap_too_small"
                          << " gapWidth=" << gapWidth
                          << " minBridgeGapWidth=" << lumenMinBridgeGapWidth
                          << " axisLen=" << axisLen
                          << " r1Along=" << r1Along
                          << " r2Along=" << r2Along
                          << std::endl;
                restoreLiveParent();
                return {0.0, noop};
                }
            }

            // Absolute edge-brightness gate: a daughter whose edge zone
            // is at or near background is sitting in empty space, even
            // if the other edge is bright enough to make a favorable
            // cost delta. Measured in the same real-image units as the
            // sigmoid-calibrated background (~0.0), so ~0.05 is ~5% above
            // background — well below any real cell body (~0.1-0.3).
            // Tunable via probConfig.bio_bridge_min_edge_brightness_absolute.
            const float kMinEdgeBrightAbsolute =
                (useCellLumenGateParams && lumenMinEdgeBrightness >= 0.0f)
                    ? lumenMinEdgeBrightness
                    : probConfig.bio_bridge_min_edge_brightness_absolute;
            const float activeMinEdgeBrightAbsolute =
                (bridgeProposalOnly && bestLabel == "bridge_primary")
                    ? 0.5f * kMinEdgeBrightAbsolute
                    : kMinEdgeBrightAbsolute;
            if (edge1Count > 0 && edge2Count > 0 &&
                std::min(edge1Bright, edge2Bright) < activeMinEdgeBrightAbsolute) {
                std::cout << "[Split Reject bio] " << parentName
                          << " reason=edge_too_dim"
                          << " edge1Bright=" << edge1Bright
                          << " edge2Bright=" << edge2Bright
                          << " minEdgeBright=" << std::min(edge1Bright, edge2Bright)
                          << " threshold=" << activeMinEdgeBrightAbsolute
                          << std::endl;
                restoreLiveParent();
                return {0.0, noop};
            }

            // Single-metric valley gate (2026-04-15 redesign):
            //   reject when gap brightness ≥ valleyLimit × max(edge1, edge2)
            //
            // This measures the brightness drop from the brighter daughter
            // edge into the gap. A real split (symmetric or asymmetric) has
            // valleyFromBright < 0.85 (gap is clearly darker than the
            // brighter edge). A phantom split has valleyFromBright ≈ 1 or
            // above (gap at least as bright as any edge).
            //
            // Replaces the previous two-tier logic that combined
            // worstValleyRatio + gapDensity. The old worst-of-two-sides
            // metric punished legitimate asymmetric division because the
            // dimmer daughter's edge ≈ gap inflated its ratio. The new
            // metric ignores that side correctly.
            const float valleyLimit =
                (useCellLumenGateParams && lumenBridgeMaxValleyRatio >= 0.0f)
                    ? lumenBridgeMaxValleyRatio
                    : probConfig.bio_bridge_max_valley_ratio;
            const bool bridgeFlat = valleyFromBright > valleyLimit;
            const bool denseFlatFutureRescue =
                bridgeProposalOnly &&
                bridgeProposal != nullptr &&
                bridgeProposal->windowBothDaughtersSupported >= 2 &&
                bridgeProposal->windowMissingDaughterCount == 0 &&
                bridgeProposal->windowParentPersists == 0 &&
                bridgeProposal->parentDistanceBalance >=
                    probConfig.split_dense_flat_future_min_parent_balance &&
                bridgeProposal->windowBestMatchedMinBrightness >=
                    probConfig.split_dense_flat_future_min_brightness &&
                gapDensity < probConfig.split_dense_flat_future_max_gap_density;
            const bool denseFlatRodTipFutureRescue =
                bridgeProposalOnly &&
                bestLabel == "bridge_primary" &&
                bridgeProposal != nullptr &&
                bridgeProposal->daughterSphereRadius > 0.0f &&
                bridgeProposal->immediateFutureCenterBacked &&
                bridgeProposal->centerSnapUsedAlignedPairFallback &&
                bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
                bridgeProposal->windowBothDaughtersSupported >= 1 &&
                bridgeProposal->windowMissingDaughterCount <= 2 &&
                bridgeProposal->windowParentPersists == 0 &&
                bridgeProposal->windowBestMatchedMinBrightness >=
                    probConfig.split_dense_flat_rod_tip_min_brightness &&
                bridgeProposal->parentDistanceBalance >=
                    probConfig.split_dense_flat_rod_tip_min_parent_balance &&
                valleyFromBright <=
                    probConfig.split_dense_flat_rod_tip_max_valley_from_bright;
            const bool denseFlatCellUniverse3DelayedMissingRescue =
                simulationConfig.celluniverse3_enabled &&
                bridgeProposalOnly &&
                bestLabel == "bridge_primary" &&
                bridgeProposal != nullptr &&
                bridgeProposal->cellUniverse3DelayedMissingDaughter &&
                probConfig
                    .celluniverse3_delayed_missing_daughter_dense_flat_rescue_enabled &&
                bridgeProposal->windowBothDaughtersSupported >=
                    std::max(
                        1,
                        probConfig
                            .celluniverse3_delayed_missing_daughter_dense_flat_min_future_both) &&
                bridgeProposal->windowMissingDaughterCount <=
                    std::max(
                        0,
                        probConfig
                            .celluniverse3_delayed_missing_daughter_dense_flat_max_missing) &&
                bridgeProposal->windowParentPersists == 0 &&
                bridgeProposal->windowBestMatchedMinBrightness >=
                    probConfig
                        .celluniverse3_delayed_missing_daughter_dense_flat_min_future_brightness &&
                valleyFromBright <=
                    probConfig
                        .celluniverse3_delayed_missing_daughter_dense_flat_max_valley_from_bright &&
                gapDensity <=
                    probConfig
                        .celluniverse3_delayed_missing_daughter_dense_flat_max_gap_density;
            if (edgeCount > 0 && bridgeFlat &&
                denseFlatCellUniverse3DelayedMissingRescue) {
                std::cout << "[CellUniverse3 Delayed Missing Dense Flat Rescue] "
                          << parentName
                          << " valleyFromBright=" << valleyFromBright
                          << " maxValley="
                          << probConfig
                                 .celluniverse3_delayed_missing_daughter_dense_flat_max_valley_from_bright
                          << " gapDensity=" << gapDensity
                          << " maxGapDensity="
                          << probConfig
                                 .celluniverse3_delayed_missing_daughter_dense_flat_max_gap_density
                          << " futureBoth="
                          << bridgeProposal->windowBothDaughtersSupported
                          << " futureMissing="
                          << bridgeProposal->windowMissingDaughterCount
                          << " futureBrightness="
                          << bridgeProposal->windowBestMatchedMinBrightness
                          << std::endl;
            }
            const bool denseFlatDeterministicBridge =
                bridgeProposalOnly &&
                bestLabel == "bridge_primary" &&
                bridgeFlat &&
                gapDensity >= probConfig.split_dense_flat_bridge_min_gap_density &&
                !denseFlatFutureRescue &&
                !denseFlatRodTipFutureRescue &&
                !denseFlatCellUniverse3DelayedMissingRescue;
            if (edgeCount > 0 && denseFlatDeterministicBridge) {
                std::cout << "[Split Reject bio] " << parentName
                          << " reason=dense_flat_bridge"
                          << " valleyFromBright=" << valleyFromBright
                          << " valleyLimit=" << valleyLimit
                          << " gapDensity=" << gapDensity
                          << " gapBright=" << gapBright
                          << " edge1Bright=" << edge1Bright
                          << " edge2Bright=" << edge2Bright
                          << " gapWidth=" << gapWidth
                          << " bestIdx=" << bestIdx
                          << " bestLabel=" << bestLabel
                          << std::endl;
                restoreLiveParent();
                return {0.0, noop};
            }
            const float maxDaughterSeedDrift = std::max(drift1, drift2);
            const float denseDriftingBridgeDriftLimit =
                probConfig.split_dense_drifting_bridge_seed_drift_fraction *
                std::max(1.0f, parentMaxRadiusForSoftGeometry);
            const bool denseDriftingBridgeGeometryTriggered =
                gapDensity >= probConfig.split_dense_drifting_bridge_min_gap_density &&
                valleyFromBright >=
                    probConfig.split_dense_drifting_bridge_min_valley_from_bright &&
                maxDaughterSeedDrift >= denseDriftingBridgeDriftLimit;
            const bool denseDriftingBridgeFutureWeak =
                (bridgeProposal == nullptr ||
                 bridgeProposal->windowBestMatchedMinBrightness <
                     probConfig.split_dense_drifting_bridge_future_brightness_floor ||
                 bridgeProposal->windowBothDaughtersSupported <
                     std::max(2, probConfig.pca_bridge_future_window_min_both_daughter_support) ||
                 bridgeProposal->windowMissingDaughterCount > 0 ||
                 bridgeProposal->windowParentPersists > 0);
            const float cellUniverse3MapPrimaryDenseBridgeDriftLimit =
                probConfig
                    .celluniverse3_window_map_primary_dense_bridge_max_drift_scale *
                std::max(1.0f, parentMaxRadiusForSoftGeometry);
            const float cellUniverse3MapMinUSupport =
                std::max(0.0f,
                         simulationConfig.celluniverse3_window_map_min_u_support);
            const bool cellUniverse3WindowMapPrimaryDenseBridgeRescue =
                simulationConfig.celluniverse3_enabled &&
                probConfig
                    .celluniverse3_window_map_primary_dense_bridge_rescue_enabled &&
                bridgeProposalOnly &&
                bestLabel == "bridge_primary" &&
                bridgeProposal != nullptr &&
                bridgeProposal->cellUniverse3MapProposal &&
                bridgeProposal->cellUniverse3MapPriorConfident &&
                bridgeProposal->windowParentPersists == 0 &&
                bridgeProposal->cellUniverse3MapUSupportD1 >= cellUniverse3MapMinUSupport &&
                bridgeProposal->cellUniverse3MapUSupportD2 >= cellUniverse3MapMinUSupport &&
                bridgeProposal->cellUniverse3MapRegionPenalty <= 0.25f &&
                valleyFromBright <=
                    probConfig
                        .celluniverse3_window_map_primary_dense_bridge_max_valley_from_bright &&
                gapDensity <=
                    probConfig
                        .celluniverse3_window_map_primary_dense_bridge_max_gap_density &&
                maxDaughterSeedDrift <= cellUniverse3MapPrimaryDenseBridgeDriftLimit;
            if (edgeCount > 0 &&
                denseDriftingBridgeGeometryTriggered &&
                denseDriftingBridgeFutureWeak &&
                cellUniverse3WindowMapPrimaryDenseBridgeRescue) {
                std::cout << "[CellUniverse3 Window Map Dense Bridge Rescue] "
                          << parentName
                          << " valleyFromBright=" << valleyFromBright
                          << " maxValley="
                          << probConfig
                                 .celluniverse3_window_map_primary_dense_bridge_max_valley_from_bright
                          << " gapDensity=" << gapDensity
                          << " maxGapDensity="
                          << probConfig
                                 .celluniverse3_window_map_primary_dense_bridge_max_gap_density
                          << " maxDaughterSeedDrift=" << maxDaughterSeedDrift
                          << " driftLimit="
                          << cellUniverse3MapPrimaryDenseBridgeDriftLimit
                          << " uSupportD1="
                          << bridgeProposal->cellUniverse3MapUSupportD1
                          << " uSupportD2="
                          << bridgeProposal->cellUniverse3MapUSupportD2
                          << " futureBoth="
                          << bridgeProposal->windowBothDaughtersSupported
                          << " futureMissing="
                          << bridgeProposal->windowMissingDaughterCount
                          << std::endl;
            }
            const bool asymmetricUTunnelDenseBridgeRescue =
                simulationConfig.celluniverse3_enabled &&
                probConfig.celluniverse3_window_map_asymmetric_u_tunnel_rescue_enabled &&
                bridgeProposalOnly &&
                bestLabel == "bridge_primary" &&
                bridgeProposal != nullptr &&
                bridgeProposal->cellUniverse3MapPriorEvaluated &&
                ((bridgeProposal->cellUniverse3MapD1InsideTunnel ? 1 : 0) +
                 (bridgeProposal->cellUniverse3MapD2InsideTunnel ? 1 : 0)) == 1 &&
                std::max(bridgeProposal->cellUniverse3MapUSupportD1,
                         bridgeProposal->cellUniverse3MapUSupportD2) >=
                    probConfig.celluniverse3_window_map_primary_support_min_u_support &&
                std::min(bridgeProposal->cellUniverse3MapUSupportD1,
                         bridgeProposal->cellUniverse3MapUSupportD2) <=
                    probConfig.celluniverse3_window_map_asymmetric_u_tunnel_rescue_max_weak_u_support &&
                bridgeProposal->windowBothDaughtersSupported >=
                    std::max(1,
                             probConfig.celluniverse3_window_map_asymmetric_u_tunnel_rescue_min_future_both) &&
                bridgeProposal->windowMissingDaughterCount <=
                    std::max(0,
                             probConfig.celluniverse3_window_map_asymmetric_u_tunnel_rescue_max_missing) &&
                bridgeProposal->windowParentPersists == 0 &&
                bridgeProposal->windowBestMatchedMinBrightness >=
                    probConfig.celluniverse3_window_map_asymmetric_u_tunnel_rescue_min_future_brightness &&
                valleyFromBright <=
                    probConfig.celluniverse3_window_map_primary_dense_bridge_max_valley_from_bright &&
                gapDensity <=
                    probConfig.celluniverse3_window_map_primary_dense_bridge_max_gap_density;
            if (edgeCount > 0 &&
                denseDriftingBridgeGeometryTriggered &&
                denseDriftingBridgeFutureWeak &&
                asymmetricUTunnelDenseBridgeRescue) {
                std::cout << "[CellUniverse3 Asymmetric U Tunnel Dense Bridge Rescue] "
                          << parentName
                          << " valleyFromBright=" << valleyFromBright
                          << " gapDensity=" << gapDensity
                          << " maxDaughterSeedDrift=" << maxDaughterSeedDrift
                          << " driftLimit=" << denseDriftingBridgeDriftLimit
                          << " uSupportD1="
                          << bridgeProposal->cellUniverse3MapUSupportD1
                          << " uSupportD2="
                          << bridgeProposal->cellUniverse3MapUSupportD2
                          << " futureBoth="
                          << bridgeProposal->windowBothDaughtersSupported
                          << " futureMissing="
                          << bridgeProposal->windowMissingDaughterCount
                          << " futureBrightness="
                          << bridgeProposal->windowBestMatchedMinBrightness
                          << std::endl;
            }
            const bool denseDriftingDeterministicBridge =
                simulationConfig.celluniverse2_enabled &&
                bridgeProposalOnly &&
                bestLabel == "bridge_primary" &&
                denseDriftingBridgeGeometryTriggered &&
                denseDriftingBridgeFutureWeak &&
                !cellUniverse3WindowMapPrimaryDenseBridgeRescue &&
                !asymmetricUTunnelDenseBridgeRescue;
            if (edgeCount > 0 && denseDriftingDeterministicBridge) {
                std::cout << "[Split Reject bio] " << parentName
                          << " reason=dense_drifting_bridge"
                          << " valleyFromBright=" << valleyFromBright
                          << " gapDensity=" << gapDensity
                          << " maxDaughterSeedDrift=" << maxDaughterSeedDrift
                          << " driftLimit="
                          << denseDriftingBridgeDriftLimit
                          << " futureBestMinBrightness="
                          << (bridgeProposal != nullptr
                                  ? bridgeProposal->windowBestMatchedMinBrightness
                                  : 0.0f)
                          << " futureBoth="
                          << (bridgeProposal != nullptr
                                  ? bridgeProposal->windowBothDaughtersSupported
                                  : 0)
                          << " futureMissing="
                          << (bridgeProposal != nullptr
                                  ? bridgeProposal->windowMissingDaughterCount
                                  : 0)
                          << " bestIdx=" << bestIdx
                          << " bestLabel=" << bestLabel
                          << std::endl;
                restoreLiveParent();
                return {0.0, noop};
            }
            const float weakBridgeParentShape =
                (bridgeProposal != nullptr)
                    ? bridgeProposal->parentShapeElongation
                    : snapshotParentShape;
            const bool weakNoValleyDriftingPcaBridge =
                simulationConfig.celluniverse2_enabled &&
                probConfig.split_weak_pca_bridge_strict_no_valley_enabled &&
                bridgeProposalOnly &&
                bestLabel == "bridge_primary" &&
                bridgeProposal != nullptr &&
                weakBridgeParentShape <=
                    probConfig
                        .split_weak_pca_bridge_strict_no_valley_max_parent_shape &&
                valleyFromBright >=
                    probConfig
                        .split_weak_pca_bridge_strict_no_valley_min_valley_from_bright &&
                gapDensity >=
                    probConfig
                        .split_weak_pca_bridge_strict_no_valley_min_gap_density &&
                maxDaughterSeedDrift >=
                    probConfig
                        .split_weak_pca_bridge_strict_no_valley_min_drift_fraction *
                        std::max(1.0f, parentMaxRadiusForSoftGeometry);
            if (edgeCount > 0 && weakNoValleyDriftingPcaBridge) {
                std::cout << "[Split Reject bio] " << parentName
                          << " reason=weak_no_valley_drifting_pca_bridge"
                          << " parentShape=" << weakBridgeParentShape
                          << " maxParentShape="
                          << probConfig
                                 .split_weak_pca_bridge_strict_no_valley_max_parent_shape
                          << " valleyFromBright=" << valleyFromBright
                          << " minValleyFromBright="
                          << probConfig
                                 .split_weak_pca_bridge_strict_no_valley_min_valley_from_bright
                          << " gapDensity=" << gapDensity
                          << " minGapDensity="
                          << probConfig
                                 .split_weak_pca_bridge_strict_no_valley_min_gap_density
                          << " maxDaughterSeedDrift=" << maxDaughterSeedDrift
                          << " driftLimit="
                          << probConfig
                                 .split_weak_pca_bridge_strict_no_valley_min_drift_fraction *
                                 std::max(1.0f, parentMaxRadiusForSoftGeometry)
                          << " futureBoth="
                          << bridgeProposal->windowBothDaughtersSupported
                          << " futureMissing="
                          << bridgeProposal->windowMissingDaughterCount
                          << " bestIdx=" << bestIdx
                          << " bestLabel=" << bestLabel
                          << std::endl;
                restoreLiveParent();
                return {0.0, noop};
            }
            const bool bypassPcaBridgeValley =
                bridgeProposalOnly &&
                bestLabel == "bridge_primary" &&
                (!probConfig.pca_bridge_require_valley ||
                 denseFlatRodTipFutureRescue);
            if (edgeCount > 0 && bridgeFlat && !bypassPcaBridgeValley) {
                const bool impossibleValley =
                    bestIsCellLumenPrepassFallback ||
                    !useCellLumenSoftGate ||
                    valleyFromBright > std::max(valleyLimit, lumenHardMaxValleyRatio);
                if (!impossibleValley) {
                    const double normalizedExcess =
                        static_cast<double>(valleyFromBright - valleyLimit) /
                        std::max(0.1, static_cast<double>(valleyLimit));
                    addLumenSoftGatePenalty(
                        "bridge_flat",
                        normalizedExcess,
                        static_cast<double>(lumenSoftValleyPenaltyFraction));
                } else {
                std::cout << "[Split Reject bio] " << parentName
                          << " reason=bridge_flat"
                          << " valleyFromBright=" << valleyFromBright
                          << " brighterEdge=" << brighterEdge
                          << " gapBright=" << gapBright
                          << " valleyRatio1=" << valleyRatio1
                          << " valleyRatio2=" << valleyRatio2
                          << " gapDensity=" << gapDensity
                          << " gapWidth=" << gapWidth
                          << " effGapHalf=" << effectiveGapHalf
                          << " valleyLimit=" << valleyLimit
                          << std::endl;
                restoreLiveParent();
                return {0.0, noop};
                }
            }

            const float rescueValleyLimit = std::max(
                0.0f, probConfig.split_bridge_cost_rescue_max_valley_ratio);
            const float rescueGapDensityLimit = std::max(
                0.0f, probConfig.split_bridge_cost_rescue_max_gap_density);
            const float rescueValleyRatio =
                probConfig.split_bridge_cost_rescue_require_two_sided_valley
                    ? worstValleyRatio
                    : valleyFromBright;
            bridgeCostRescueEligible =
                edgeCount > 0 &&
                rescueValleyRatio <= rescueValleyLimit &&
                gapDensity <= rescueGapDensityLimit;
            lumenStrongBridgeEvidence =
                useCellLumenGateParams &&
                edgeCount > 0 &&
                valleyFromBright <= valleyLimit &&
                std::min(edge1Bright, edge2Bright) >= activeMinEdgeBrightAbsolute;
            bridgeCostRescueValleyFromBright = valleyFromBright;
            bridgeCostRescueWorstValleyRatio = worstValleyRatio;
            bridgeCostRescueGapDensity = gapDensity;
            bridgeCostRescueEdgeBright = edgeBright;
            lumenBridgeGapWidth = gapWidth;
        }
    }

    // 5b. Size ratio, volume fraction, and buried checks.
    const bool futureSupportedMidpointRescue =
        simulationConfig.celluniverse2_enabled &&
        bridgeProposal != nullptr &&
        bridgeProposal->windowBothDaughtersSupported >=
            std::max(2, probConfig.pca_bridge_future_window_min_both_daughter_support) &&
        bridgeProposal->windowMissingDaughterCount == 0 &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->windowBestMatchedMinBrightness >=
            std::max(0.0f,
                     probConfig
                         .pca_bridge_future_window_rod_tip_balance_min_brightness);
    const bool futureSupportedRodTipMidpointRescue =
        futureSupportedMidpointRescue &&
        bridgeProposal->daughterSphereRadius > 0.0f;
    const bool immediateFutureRodTipBioRescue =
        simulationConfig.celluniverse2_enabled &&
        bridgeProposal != nullptr &&
        bridgeProposal->daughterSphereRadius > 0.0f &&
        bridgeProposal->immediateFutureCenterBacked &&
        bridgeProposal->centerSnapUsedAlignedPairFallback &&
        bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
        bridgeProposal->windowBothDaughtersSupported >= 1 &&
        bridgeProposal->windowMissingDaughterCount <= 2 &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->windowBestMatchedMinBrightness >=
            std::max(0.0f,
                     probConfig
                         .pca_bridge_future_window_rod_tip_balance_min_brightness) &&
        bridgeProposal->parentDistanceBalance >=
            probConfig.split_future_rod_tip_primary_min_parent_balance;
    const bool futureSupportedRodTipBioRescue =
        futureSupportedRodTipMidpointRescue ||
        immediateFutureRodTipBioRescue;

    ProbabilityConfig finalBioConfig = probConfig;
    const bool finalBioSeparationSoftRescue =
        simulationConfig.celluniverse2_enabled &&
        bridgeProposal != nullptr &&
        bridgeProposal->bioSeparationSoftRescued;
    const bool finalBioFutureSeparationRescue =
        simulationConfig.celluniverse2_enabled &&
        bridgeProposal != nullptr &&
        bridgeProposal->futureWindowSplitRescue &&
        probConfig.pca_bridge_future_window_min_separation_parent_fraction > 0.0f;
    if (finalBioSeparationSoftRescue || finalBioFutureSeparationRescue) {
        float activeSeparationFraction =
            std::max(0.0f, probConfig.bio_min_daughter_separation_parent_fraction);
        if (finalBioFutureSeparationRescue) {
            activeSeparationFraction = std::min(
                activeSeparationFraction,
                std::max(
                    0.0f,
                    probConfig.pca_bridge_future_window_min_separation_parent_fraction));
        }
        if (finalBioSeparationSoftRescue) {
            activeSeparationFraction = std::min(
                activeSeparationFraction,
                std::max(0.0f, probConfig.bio_min_daughter_separation_parent_fraction) *
                    std::clamp(probConfig.bio_separation_soft_min_fraction, 0.0f, 1.0f));
        }
        finalBioConfig.bio_min_daughter_separation_parent_fraction =
            activeSeparationFraction;
    }
    if (simulationConfig.celluniverse2_enabled && bridgeProposal != nullptr &&
        bridgeProposal->bioSeparationRequired > 0.0f && srcMaxR > 1e-3f) {
        const float effectiveSeparationFraction =
            bridgeProposal->bioSeparationRequired / std::max(1.0f, srcMaxR);
        finalBioConfig.bio_min_daughter_separation_parent_fraction = std::min(
            std::max(0.0f, finalBioConfig.bio_min_daughter_separation_parent_fraction),
            std::max(0.0f, effectiveSeparationFraction));
    }
    const bool futureSupportedPcaBridgeNearSeparationRescue =
        simulationConfig.celluniverse2_enabled &&
        bridgeProposalOnly &&
        bestLabel == "bridge_primary" &&
        bridgeProposal != nullptr &&
        !bridgeProposal->centerSnapUsedAlignedPairFallback &&
        bridgeProposal->windowBothDaughtersSupported >=
            std::max(2, probConfig.pca_bridge_future_window_min_both_daughter_support) &&
        bridgeProposal->windowMissingDaughterCount == 0 &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->windowBestMatchedMinBrightness >=
            probConfig.split_future_pca_bridge_near_sep_min_brightness &&
        bridgeProposal->parentDistanceBalance >=
            probConfig.split_future_pca_bridge_near_sep_min_parent_balance;
    if (futureSupportedPcaBridgeNearSeparationRescue) {
        const float nearSeparationFraction = std::clamp(
            probConfig.pca_bridge_future_window_near_separation_fraction,
            0.0f,
            1.0f);
        finalBioConfig.bio_min_daughter_separation_parent_fraction = std::min(
            finalBioConfig.bio_min_daughter_separation_parent_fraction,
            std::max(0.0f, probConfig.bio_min_daughter_separation_parent_fraction) *
                nearSeparationFraction);
    }
    const float cellUniverse3NearSepSnapLimit = std::max(
        std::max(0.0f, probConfig.pca_bridge_future_window_match_distance),
        std::max(
            0.0f,
            probConfig
                .celluniverse3_pca_bridge_near_separation_max_snap_radius_scale) *
            std::max(1.0f, srcMaxR));
    const bool cellUniverse3PcaBridgeNearSeparationRescue =
        simulationConfig.celluniverse3_enabled &&
        probConfig.celluniverse3_pca_bridge_near_separation_rescue_enabled &&
        bridgeProposalOnly &&
        bestLabel == "bridge_primary" &&
        bridgeProposal != nullptr &&
        bridgeProposal->windowBothDaughtersSupported >=
            std::max(
                1,
                probConfig
                    .celluniverse3_pca_bridge_near_separation_min_future_both) &&
        bridgeProposal->windowMissingDaughterCount <=
            std::max(
                0,
                probConfig
                    .celluniverse3_pca_bridge_near_separation_max_missing) &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->windowBestMatchedMinBrightness >=
            std::max(
                0.0f,
                probConfig
                    .celluniverse3_pca_bridge_near_separation_min_brightness) &&
        bridgeProposal->parentDistanceBalance >=
            std::max(
                0.0f,
                probConfig
                    .celluniverse3_pca_bridge_near_separation_min_parent_balance) &&
        bridgeProposal->centerSnapMaxSeedDistance <= cellUniverse3NearSepSnapLimit;
    if (cellUniverse3PcaBridgeNearSeparationRescue) {
        const float nearSeparationFraction = std::clamp(
            probConfig.celluniverse3_pca_bridge_near_separation_fraction,
            0.0f,
            1.0f);
        finalBioConfig.bio_min_daughter_separation_parent_fraction = std::min(
            finalBioConfig.bio_min_daughter_separation_parent_fraction,
            std::max(0.0f, probConfig.bio_min_daughter_separation_parent_fraction) *
                nearSeparationFraction);
    }
    const bool cellUniverse3AsymmetricUTunnelNearSeparationRescue =
        simulationConfig.celluniverse3_enabled &&
        probConfig.celluniverse3_window_map_asymmetric_u_tunnel_rescue_enabled &&
        bridgeProposalOnly &&
        bestLabel == "bridge_primary" &&
        bridgeProposal != nullptr &&
        bridgeProposal->cellUniverse3MapPriorEvaluated &&
        ((bridgeProposal->cellUniverse3MapD1InsideTunnel ? 1 : 0) +
         (bridgeProposal->cellUniverse3MapD2InsideTunnel ? 1 : 0)) == 1 &&
        std::max(bridgeProposal->cellUniverse3MapUSupportD1,
                 bridgeProposal->cellUniverse3MapUSupportD2) >=
            probConfig.celluniverse3_window_map_primary_support_min_u_support &&
        std::min(bridgeProposal->cellUniverse3MapUSupportD1,
                 bridgeProposal->cellUniverse3MapUSupportD2) <=
            probConfig.celluniverse3_window_map_asymmetric_u_tunnel_rescue_max_weak_u_support &&
        bridgeProposal->windowBothDaughtersSupported >=
            std::max(1,
                     probConfig.celluniverse3_window_map_asymmetric_u_tunnel_rescue_min_future_both) &&
        bridgeProposal->windowMissingDaughterCount <=
            std::max(0,
                     probConfig.celluniverse3_window_map_asymmetric_u_tunnel_rescue_max_missing) &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->windowBestMatchedMinBrightness >=
            probConfig.celluniverse3_window_map_asymmetric_u_tunnel_rescue_min_future_brightness;
    if (cellUniverse3AsymmetricUTunnelNearSeparationRescue) {
        const float nearSeparationFraction = std::clamp(
            probConfig.celluniverse3_window_map_asymmetric_u_tunnel_rescue_near_separation_fraction,
            0.0f,
            1.0f);
        finalBioConfig.bio_min_daughter_separation_parent_fraction = std::min(
            finalBioConfig.bio_min_daughter_separation_parent_fraction,
            std::max(0.0f, probConfig.bio_min_daughter_separation_parent_fraction) *
                nearSeparationFraction);
        std::cout << "[CellUniverse3 Asymmetric U Tunnel Near Separation Rescue] "
                  << parentName
                  << " fraction=" << nearSeparationFraction
                  << " futureBoth="
                  << bridgeProposal->windowBothDaughtersSupported
                  << " futureMissing="
                  << bridgeProposal->windowMissingDaughterCount
                  << " futureBrightness="
                  << bridgeProposal->windowBestMatchedMinBrightness
                  << " uSupportD1="
                  << bridgeProposal->cellUniverse3MapUSupportD1
                  << " uSupportD2="
                  << bridgeProposal->cellUniverse3MapUSupportD2
                  << std::endl;
    }
    const float midpointParentFraction =
        futureSupportedMidpointRescue
            ? std::max(
                  std::max(0.0f, probConfig.bio_max_midpoint_parent_fraction),
                  std::max(
                      0.0f,
                      probConfig
                          .pca_bridge_future_window_rod_tip_midpoint_parent_fraction))
            : std::max(0.0f, probConfig.bio_max_midpoint_parent_fraction);
    if (midpointParentFraction > 0.0f && snapshotValid) {
        const cv::Point3f daughterMidpoint(
            0.5f * (bestD1Pos.x + bestD2Pos.x),
            0.5f * (bestD1Pos.y + bestD2Pos.y),
            0.5f * (bestD1Pos.z + bestD2Pos.z));
        const float midpointDistance =
            static_cast<float>(cv::norm(daughterMidpoint - snapshot.position));
        const float midpointLimit = midpointParentFraction * std::max(1.0f, srcMaxR);
        const float midpointSeedDriftLimit =
            std::max(2.0f, 0.30f * std::max(1.0f, srcMaxR));
        const float midpointMaxDaughterSeedDrift = std::max(drift1, drift2);
	        const bool cleanFutureMidpointNearMiss =
	            futureSupportedMidpointRescue &&
	            bridgeProposal != nullptr &&
	            bridgeProposal->windowBothDaughtersSupported >= 2 &&
	            bridgeProposal->windowMissingDaughterCount == 0 &&
	            bridgeProposal->windowParentPersists == 0 &&
            bridgeProposal->windowBestMatchedMinBrightness >=
                probConfig.split_future_midpoint_near_miss_min_brightness &&
            midpointMaxDaughterSeedDrift <= midpointSeedDriftLimit &&
            midpointDistance <=
                probConfig.split_future_midpoint_near_miss_limit_scale *
                    std::max(1.0f, midpointLimit);
        const bool signalCenterMidpointNearMiss =
            simulationConfig.celluniverse2_enabled &&
            bridgeProposal != nullptr &&
            probConfig.signal_center_midpoint_near_miss_enabled &&
            bridgeProposal->signalCenterScore >=
                probConfig.signal_center_midpoint_near_miss_min_score &&
            bridgeProposal->signalCenterSeparationRatio >=
                probConfig.signal_center_midpoint_near_miss_min_separation_ratio &&
            bridgeProposal->signalCenterAxisAlignment >=
                probConfig.signal_center_midpoint_near_miss_min_axis_alignment &&
            bridgeProposal->windowBothDaughtersSupported >=
                std::max(1, probConfig.signal_center_midpoint_near_miss_min_future_both) &&
            bridgeProposal->windowMissingDaughterCount == 0 &&
            bridgeProposal->windowParentPersists == 0 &&
            bridgeProposal->windowBestMatchedMinBrightness >=
                probConfig.signal_center_midpoint_near_miss_min_future_brightness &&
            midpointMaxDaughterSeedDrift <= midpointSeedDriftLimit &&
            midpointDistance <=
                std::max(1.0f, probConfig.signal_center_midpoint_near_miss_limit_scale) *
                std::max(1.0f, midpointLimit);
	        if (midpointDistance > midpointLimit &&
	            !cleanFutureMidpointNearMiss &&
	            !signalCenterMidpointNearMiss) {
	            std::cout << "[Split Reject bio] " << parentName
	                      << " reason=daughter_midpoint_parent_drift"
	                      << " midpointDistance=" << midpointDistance
                      << " limit=" << midpointLimit
                      << " fraction=" << midpointParentFraction
                      << " futureSupportedMidpointRescue="
                      << (futureSupportedMidpointRescue ? 1 : 0)
	                      << " futureSupportedRodTipMidpointRescue="
	                      << (futureSupportedRodTipMidpointRescue ? 1 : 0)
	                      << " signalCenterMidpointNearMiss="
	                      << (signalCenterMidpointNearMiss ? 1 : 0)
	                      << " futureBestMinBrightness="
                      << (bridgeProposal != nullptr
                              ? bridgeProposal->windowBestMatchedMinBrightness
                              : 0.0f)
                      << " parentMaxR=" << srcMaxR
                      << " maxDaughterSeedDrift="
                      << midpointMaxDaughterSeedDrift
                      << " seedDriftLimit=" << midpointSeedDriftLimit
                      << " midpoint=(" << daughterMidpoint.x << ","
                      << daughterMidpoint.y << "," << daughterMidpoint.z << ")"
                      << " snapshot=(" << snapshot.position.x << ","
                      << snapshot.position.y << "," << snapshot.position.z << ")"
                      << " bestIdx=" << bestIdx
                      << " bestLabel=" << bestLabel
                      << std::endl;
            restoreLiveParent();
            return {0.0, noop};
        } else if (midpointDistance > midpointLimit) {
            std::cout << "[Split Soft Gate Waived] " << parentName
                      << " reason=clean_future_midpoint_parent_drift_near_miss"
                      << " midpointDistance=" << midpointDistance
                      << " limit=" << midpointLimit
                      << " fraction=" << midpointParentFraction
                      << " futureBestMinBrightness="
                      << (bridgeProposal != nullptr
                              ? bridgeProposal->windowBestMatchedMinBrightness
                              : 0.0f)
                      << " maxDaughterSeedDrift="
                      << midpointMaxDaughterSeedDrift
                      << " seedDriftLimit=" << midpointSeedDriftLimit
                      << " bestIdx=" << bestIdx
                      << " bestLabel=" << bestLabel
                      << std::endl;
        }
    }

    if (futureSupportedRodTipBioRescue) {
        finalBioConfig.bio_combined_volume_max_fraction = std::max(
            finalBioConfig.bio_combined_volume_max_fraction,
            std::max(
                0.0f,
                probConfig
                    .pca_bridge_future_window_rod_tip_combined_volume_max_fraction));
        finalBioConfig.bio_max_single_daughter_volume_fraction = std::max(
            finalBioConfig.bio_max_single_daughter_volume_fraction,
            std::max(
                0.0f,
                probConfig
                    .pca_bridge_future_window_rod_tip_single_daughter_volume_max_fraction));
    }
    const bool futureSupportedBridgeAxisBuriedRescue =
        futureSupportedRodTipMidpointRescue &&
        bestLabel == "bridge_axis_place" &&
        bridgeProposal != nullptr &&
        bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
        bridgeProposal->parentDistanceBalance >= probConfig.split_bridge_axis_buried_min_parent_balance &&
        bridgeProposal->windowBestMatchedMinBrightness >= probConfig.split_bridge_axis_buried_min_brightness;
    const bool cleanTwoFrameRodTipContinuationRescue =
        futureSupportedRodTipMidpointRescue &&
        bestLabel == "bridge_primary" &&
        bridgeProposal != nullptr &&
        bridgeProposal->windowBothDaughtersSupported >= 2 &&
        bridgeProposal->windowMissingDaughterCount == 0 &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->parentDistanceBalance >= probConfig.split_clean_rod_tip_continuation_min_parent_balance &&
        bridgeProposal->windowBestMatchedMinBrightness >= probConfig.split_clean_rod_tip_continuation_min_brightness;
    const float cleanPcaContinuationBaseSnapLimit =
        std::max(
            std::max(0.0f, probConfig.pca_bridge_future_window_match_distance),
            probConfig.split_clean_pca_continuation_base_snap_scale * std::max(1.0f, srcMaxR));
    const float cleanPcaContinuationSnapLimit =
        std::max(
            cleanPcaContinuationBaseSnapLimit,
            std::max(probConfig.split_clean_pca_continuation_min_snap_scale,
                     std::max(
                         0.0f,
                         probConfig
                             .pca_bridge_future_window_pca_snap_max_radius_scale)) *
                std::max(1.0f, srcMaxR));
    const bool cleanPcaContinuationImmediateSupport =
        bridgeProposal != nullptr &&
        (bridgeProposal->immediateFutureCenterBacked ||
         bridgeProposal->windowImmediateBothDaughtersSupported > 0);
    const bool immediateFuturePcaContinuationRescue =
        simulationConfig.celluniverse2_enabled &&
        bridgeProposalOnly &&
        bestLabel == "bridge_primary" &&
        bridgeProposal != nullptr &&
        cleanPcaContinuationImmediateSupport &&
        bridgeProposal->windowBothDaughtersSupported >= 2 &&
        bridgeProposal->windowMissingDaughterCount == 0 &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->windowBestMatchedMinBrightness >= probConfig.split_immediate_pca_continuation_min_brightness &&
        bridgeProposal->parentShapeElongation >= probConfig.split_immediate_pca_continuation_min_parent_shape &&
        bridgeProposal->parentDistanceBalance >= probConfig.split_immediate_pca_continuation_min_parent_balance &&
        bridgeProposal->centerSnapMaxSeedDistance <= cleanPcaContinuationSnapLimit;
    const bool oneFrameAlignedFuturePcaContinuationRescue =
        simulationConfig.celluniverse2_enabled &&
        bridgeProposalOnly &&
        bestLabel == "bridge_primary" &&
        bridgeProposal != nullptr &&
        cleanPcaContinuationImmediateSupport &&
        bridgeProposal->centerSnapUsedAlignedPairFallback &&
        bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
        bridgeProposal->windowBothDaughtersSupported >= 1 &&
        bridgeProposal->windowMissingDaughterCount == 0 &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->windowBestMatchedMinBrightness >= probConfig.split_one_frame_aligned_pca_continuation_min_brightness &&
        bridgeProposal->parentShapeElongation >= probConfig.split_one_frame_aligned_pca_continuation_min_parent_shape &&
        bridgeProposal->parentDistanceBalance >= probConfig.split_one_frame_aligned_pca_continuation_min_parent_balance &&
        bridgeProposal->parentDistanceBalance <= probConfig.split_one_frame_aligned_pca_continuation_max_parent_balance &&
        bridgeProposal->centerSnapMaxSeedDistance <=
            std::max(cleanPcaContinuationSnapLimit,
                     probConfig.split_one_frame_aligned_pca_continuation_snap_scale * std::max(1.0f, srcMaxR));
    const bool currentPcaContinuationRescue =
        simulationConfig.celluniverse2_enabled &&
        bridgeProposalOnly &&
        bestLabel == "bridge_primary" &&
        bridgeProposal != nullptr &&
        !bridgeProposal->immediateFutureCenterBacked &&
        bridgeProposal->windowBothDaughtersSupported >= 1 &&
        bridgeProposal->windowMissingDaughterCount <= 2 &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->windowBestMatchedMinBrightness >= probConfig.split_current_pca_continuation_min_brightness &&
        bridgeProposal->parentShapeElongation >= probConfig.split_current_pca_continuation_min_parent_shape &&
        bridgeProposal->parentDistanceBalance >= probConfig.split_current_pca_continuation_min_parent_balance &&
        bridgeProposal->centerSnapMaxSeedDistance <= cleanPcaContinuationSnapLimit;
    const bool cleanPcaBridgeContinuationRescue =
        immediateFuturePcaContinuationRescue ||
        oneFrameAlignedFuturePcaContinuationRescue ||
        currentPcaContinuationRescue;
    const float cellUniverse3CleanFutureBridgeBioSnapLimit = std::max(
        cleanPcaContinuationSnapLimit,
        std::max(
            0.0f,
            probConfig
                .celluniverse3_clean_future_bridge_position_lock_max_snap_radius_scale) *
            std::max(1.0f, srcMaxR));
    const bool cellUniverse3CleanFutureBridgeBioOwnershipRescue =
        simulationConfig.celluniverse3_enabled &&
        probConfig
            .celluniverse3_clean_future_bridge_bio_ownership_rescue_enabled &&
        bridgeProposalOnly &&
        bestLabel == "bridge_primary" &&
        bridgeProposal != nullptr &&
        probConfig.pca_bridge_future_window_enabled &&
        bridgeProposal->centerSnapApplied &&
        bridgeProposal->immediateFutureCenterBacked &&
        bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
        bridgeProposal->windowBothDaughtersSupported >=
            std::max(
                1,
                probConfig
                    .celluniverse3_clean_future_bridge_position_lock_min_future_both) &&
        bridgeProposal->windowMissingDaughterCount <=
            std::max(
                0,
                probConfig
                    .celluniverse3_clean_future_bridge_position_lock_max_missing) &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->windowBestMatchedMinBrightness >=
            probConfig
                .celluniverse3_clean_future_bridge_position_lock_min_future_brightness &&
        bridgeProposal->parentShapeElongation >=
            probConfig
                .celluniverse3_clean_future_bridge_position_lock_min_parent_shape &&
        bridgeProposal->parentDistanceBalance >=
            probConfig
                .celluniverse3_clean_future_bridge_position_lock_min_parent_balance &&
        bridgeProposal->centerSnapMaxSeedDistance <=
            cellUniverse3CleanFutureBridgeBioSnapLimit;
    const bool cellUniverse3SevereSignalCenterBridgeBioOwnershipRescue =
        simulationConfig.celluniverse3_enabled &&
        bestIsSignalCenterProposal &&
        bridgeProposal != nullptr &&
        probConfig.signal_center_severe_future_claim_enabled &&
        bridgeProposal->windowBothDaughtersSupported >=
            std::max(1,
                     probConfig.signal_center_severe_future_claim_min_future_both) &&
        bridgeProposal->windowMissingDaughterCount <=
            std::max(0, probConfig.signal_center_severe_future_claim_max_missing) &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->windowBestMatchedMinBrightness >=
            std::max(0.0f,
                     probConfig.signal_center_severe_future_claim_min_brightness) &&
        bridgeProposal->parentShapeElongation >=
            std::max(0.0f,
                     probConfig.signal_center_severe_future_claim_min_parent_shape) &&
        bridgeProposal->parentDistanceBalance >=
            std::max(0.0f,
                     probConfig.signal_center_severe_future_claim_min_parent_balance);
    const bool cellUniverse3WindowMapNeighborBridgeBypass =
        simulationConfig.celluniverse3_enabled &&
        probConfig.celluniverse3_window_map_neighbor_bridge_bypass_enabled &&
        bridgeProposalOnly &&
        bridgeProposal != nullptr &&
        bestLabel == "bridge_primary" &&
        bridgeProposal->cellUniverse3MapOverlapCenterProposal &&
        bridgeProposal->cellUniverse3MapPriorConfident &&
        bridgeProposal->cellUniverse3MapUSupportD1 >=
            probConfig.celluniverse3_window_map_primary_support_min_u_support &&
        bridgeProposal->cellUniverse3MapUSupportD2 >=
            probConfig.celluniverse3_window_map_primary_support_min_u_support &&
        bridgeProposal->cellUniverse3MapRegionPenalty <=
            probConfig.celluniverse3_window_map_primary_support_max_region_penalty &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->windowBestMatchedMinBrightness >=
            probConfig.celluniverse3_window_map_neighbor_bridge_min_future_brightness &&
        bridgeProposal->parentShapeElongation >=
            probConfig.celluniverse3_window_map_neighbor_bridge_min_parent_shape &&
        bridgeProposal->parentShapeElongation <=
            probConfig.celluniverse3_window_map_neighbor_bridge_max_parent_shape &&
        bridgeProposal->parentDistanceBalance >=
            probConfig.celluniverse3_window_map_primary_support_min_parent_balance;
    const bool skipExistingCellBuriedForBio =
        (useCellLumenGateParams && lumenSkipExistingCellBuriedCheck) ||
        futureSupportedBridgeAxisBuriedRescue ||
        cleanTwoFrameRodTipContinuationRescue ||
        cleanPcaBridgeContinuationRescue ||
        cellUniverse3CleanFutureBridgeBioOwnershipRescue;
    const bool skipNeighborBridgeForBio =
        (useCellLumenGateParams && lumenSkipNeighborBridgeCheck) ||
        cleanTwoFrameRodTipContinuationRescue ||
        cleanPcaBridgeContinuationRescue ||
        cellUniverse3CleanFutureBridgeBioOwnershipRescue ||
        cellUniverse3SevereSignalCenterBridgeBioOwnershipRescue ||
        cellUniverse3WindowMapNeighborBridgeBypass;
    const bool dimExactFutureSignalRodBlockerRescue =
        bestIsSignalCenterProposal &&
        bridgeProposal != nullptr &&
        bridgeProposal->windowBothDaughtersSupported >= 2 &&
        bridgeProposal->windowMissingDaughterCount == 0 &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->windowBestMatchedMinBrightness >= probConfig.split_dim_exact_future_signal_min_brightness &&
        bridgeProposal->parentDistanceBalance >= probConfig.split_dim_exact_future_signal_min_parent_balance &&
        bridgeProposal->centerSnapMaxSeedDistance <= probConfig.split_dim_exact_future_signal_snap_epsilon &&
        bridgeProposal->bioSeparationRequired >
            probConfig.split_dim_exact_future_signal_min_bio_separation &&
        bridgeProposal->bioSeparationObserved >=
            bridgeProposal->bioSeparationRequired;
    const bool ignoreSuspiciousRodBlockersForBio =
        simulationConfig.celluniverse2_enabled &&
        bridgeProposalOnly &&
        bridgeProposal != nullptr &&
        bestLabel == "bridge_primary" &&
        (bestHasCleanFutureSplitSupport ||
         dimExactFutureSignalRodBlockerRescue ||
         cellUniverse3WindowMapNeighborBridgeBypass) &&
        bridgeProposal->windowParentPersists == 0 &&
        (bridgeProposal->windowBestMatchedMinBrightness >= probConfig.split_ignore_suspicious_rod_min_brightness ||
         dimExactFutureSignalRodBlockerRescue ||
         cellUniverse3WindowMapNeighborBridgeBypass) &&
        (bridgeProposal->parentShapeElongation >=
             probConfig.split_ignore_suspicious_rod_min_parent_shape ||
         cellUniverse3WindowMapNeighborBridgeBypass);
    const bool delayedFuturePcaBridgeBioOwnershipRescue =
        simulationConfig.celluniverse2_enabled &&
        bridgeProposalOnly &&
        bestLabel == "bridge_primary" &&
        bridgeProposal != nullptr &&
        probConfig.split_delayed_future_existing_cell_buried_scale > 0.0f &&
        probConfig.split_delayed_future_existing_cell_buried_scale < 1.0f &&
        bridgeProposal->windowImmediateBothDaughtersSupported == 0 &&
        bridgeProposal->windowBothDaughtersSupported >=
            static_cast<int>(probConfig.split_delayed_future_pca_bridge_min_future_both) &&
        bridgeProposal->windowMissingDaughterCount <=
            static_cast<int>(probConfig.split_delayed_future_pca_bridge_max_missing_daughters) &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->windowBestMatchedMinBrightness >=
            probConfig.split_delayed_future_pca_bridge_min_brightness &&
        bridgeProposal->parentShapeElongation >=
            probConfig.split_delayed_future_pca_bridge_min_parent_shape &&
        bridgeProposal->parentDistanceBalance >=
            probConfig.split_delayed_future_pca_bridge_min_parent_balance;
    const float existingCellBuriedScaleForBio =
        delayedFuturePcaBridgeBioOwnershipRescue
            ? std::clamp(
                  probConfig.split_delayed_future_existing_cell_buried_scale,
                  0.0f,
                  1.0f)
            : 1.0f;

    std::string bioReason;
    if (!bioCheckDaughters(bestD1, bestD2, refParentVolume, srcMaxR,
                           bestCells, d1IdxBest, d2IdxBest,
                           finalBioConfig, bioReason,
                           existingCellBuriedScaleForBio,
                           skipExistingCellBuriedForBio,
                           skipNeighborBridgeForBio,
                           simulationConfig.celluniverse2_enabled &&
                               probConfig.split_bio_ignore_trash_neighbors_enabled,
                           ignoreSuspiciousRodBlockersForBio)) {
        std::cout << "[Split Reject bio] " << parentName
                  << " reason=" << bioReason
                  << " d1=(" << bestD1.getX() << "," << bestD1.getY() << "," << bestD1.getZ() << ")"
                  << " r1=(" << bestD1.getARadius() << "," << bestD1.getBRadius() << "," << bestD1.getCRadius() << ")"
                  << " d2=(" << bestD2.getX() << "," << bestD2.getY() << "," << bestD2.getZ() << ")"
                  << " r2=(" << bestD2.getARadius() << "," << bestD2.getBRadius() << "," << bestD2.getCRadius() << ")"
                  << " refParentVolume=" << refParentVolume
                  << " futureSupportedRodTipBioRescue="
                  << (futureSupportedRodTipBioRescue ? 1 : 0)
                  << " futureSupportedPcaBridgeNearSepRescue="
                  << (futureSupportedPcaBridgeNearSeparationRescue ? 1 : 0)
                  << " cellUniverse3PcaBridgeNearSepRescue="
                  << (cellUniverse3PcaBridgeNearSeparationRescue ? 1 : 0)
                  << " bridgeAxisBuriedRescue="
                  << (futureSupportedBridgeAxisBuriedRescue ? 1 : 0)
                  << " cleanTwoFrameRodTipContinuationRescue="
                  << (cleanTwoFrameRodTipContinuationRescue ? 1 : 0)
                  << " cleanPcaBridgeContinuationRescue="
                  << (cleanPcaBridgeContinuationRescue ? 1 : 0)
                  << " cellUniverse3CleanFutureBridgeBioRescue="
                  << (cellUniverse3CleanFutureBridgeBioOwnershipRescue ? 1 : 0)
                  << " cellUniverse3SevereSignalCenterBridgeBioRescue="
                  << (cellUniverse3SevereSignalCenterBridgeBioOwnershipRescue ? 1 : 0)
                  << " oneFrameAlignedPcaContinuationRescue="
                  << (oneFrameAlignedFuturePcaContinuationRescue ? 1 : 0)
                  << " cleanPcaImmediateSupport="
                  << (cleanPcaContinuationImmediateSupport ? 1 : 0)
                  << " delayedFutureBioOwnershipRescue="
                  << (delayedFuturePcaBridgeBioOwnershipRescue ? 1 : 0)
                  << " existingCellBuriedScale="
                  << existingCellBuriedScaleForBio
                  << " centerSnapMaxSeedDistance="
                  << (bridgeProposal != nullptr
                          ? bridgeProposal->centerSnapMaxSeedDistance
                          : 0.0f)
                  << " cleanPcaSnapLimit="
                  << cleanPcaContinuationSnapLimit
                  << " parentDistBalance="
                  << (bridgeProposal != nullptr
                          ? bridgeProposal->parentDistanceBalance
                          : 0.0f)
                  << " futureBestMinBrightness="
                  << (bridgeProposal != nullptr
                          ? bridgeProposal->windowBestMatchedMinBrightness
                          : 0.0f)
                  << " skipNeighborBridgeForBio="
                  << (skipNeighborBridgeForBio ? 1 : 0)
                  << " ignoreSuspiciousRodBlockersForBio="
                  << (ignoreSuspiciousRodBlockersForBio ? 1 : 0)
                  << " activeCombinedVolumeMax="
                  << finalBioConfig.bio_combined_volume_max_fraction
                  << " activeSingleDaughterVolumeMax="
                  << finalBioConfig.bio_max_single_daughter_volume_fraction
                  << std::endl;
        restoreLiveParent();
        return {0.0, noop};
    }

    const float minDaughterMeanAbs =
        std::max(0.0f, probConfig.bio_min_daughter_mean_brightness_absolute);
    const float minDaughterMeanParentFraction =
        std::max(0.0f, probConfig.bio_min_daughter_mean_brightness_parent_fraction);
    const float minDaughterBackgroundMargin =
        std::max(0.0f,
                 probConfig.bio_min_daughter_mean_brightness_background_margin);
    bool costBackedCleanFuturePcaBridgeDensityWaivedActive = false;
    if (!_realFrame.empty() &&
        (minDaughterMeanAbs > 0.0f ||
         minDaughterMeanParentFraction > 0.0f ||
         minDaughterBackgroundMargin > 0.0f)) {
        const auto parentBrightnessStats = parent.measureBrightnessStats(_realFrame);
        const auto d1BrightnessStats = bestD1.measureBrightnessStats(_realFrame);
        const auto d2BrightnessStats = bestD2.measureBrightnessStats(_realFrame);
        const float parentBackground = backgroundAt(cv::Point3f(
            parent.getX(), parent.getY(), parent.getZ()));
        const float localBackground = std::max(
            backgroundAt(cv::Point3f(
                bestD1.getX(), bestD1.getY(), bestD1.getZ())),
            backgroundAt(cv::Point3f(
                bestD2.getX(), bestD2.getY(), bestD2.getZ())));
        const float parentSignal =
            std::max(0.0f,
                     parentBrightnessStats.first - parentBackground);
        const float adaptiveSignalThreshold = std::max(
            minDaughterBackgroundMargin,
            minDaughterMeanParentFraction * parentSignal);
        const float densityThreshold = localBackground + adaptiveSignalThreshold;
        const float minDaughterMean =
            std::min(d1BrightnessStats.first, d2BrightnessStats.first);
        if (minDaughterMean < densityThreshold) {
            const float softMinFraction = std::clamp(
                probConfig.bio_daughter_density_soft_min_fraction, 0.0f, 1.0f);
            const bool weakFutureCurrentBridgeDensitySupport =
                simulationConfig.celluniverse2_enabled &&
                probConfig.pca_bridge_weak_future_current_fallback_enabled &&
                bestIsPcaBridgeOnly &&
                bridgeProposal != nullptr &&
                bridgeProposal->centerSnapUsedAlignedPairFallback &&
                !bridgeProposal->immediateFutureCenterBacked &&
                bridgeProposal->windowBothDaughtersSupported >= 1 &&
                bridgeProposal->windowMissingDaughterCount <=
                    probConfig
                        .pca_bridge_weak_future_current_fallback_max_missing_daughters &&
                bridgeProposal->windowParentPersists == 0 &&
                bridgeProposal->windowBestMatchedMinBrightness >=
                    probConfig
                        .pca_bridge_weak_future_current_fallback_min_future_brightness &&
                bridgeProposal->parentShapeElongation >=
                    probConfig
                        .pca_bridge_weak_future_current_fallback_min_parent_shape &&
                bridgeProposal->parentDistanceBalance >=
                    probConfig
                        .pca_bridge_weak_future_current_fallback_min_parent_balance &&
	                bridgeProposal->centerSnapMaxSeedDistance <=
	                    std::max(
	                        probConfig.pca_bridge_future_window_match_distance,
	                        probConfig
	                                .pca_bridge_weak_future_current_fallback_max_snap_radius_scale *
	                            std::max(1.0f, srcMaxR));
            const bool cellUniverse3SignalCenterFutureDensitySupport =
                simulationConfig.celluniverse3_enabled &&
                probConfig.celluniverse3_signal_center_future_density_rescue_enabled &&
                probConfig
                    .celluniverse3_signal_center_future_position_lock_enabled &&
                bridgeProposalOnly &&
                bestLabel == "bridge_primary" &&
                bridgeProposal != nullptr &&
                bridgeProposal->signalCenterScore >= 0.0f &&
                !bridgeProposal->centerSnapUsedAlignedPairFallback &&
                bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
                bridgeProposal->windowBothDaughtersSupported >=
                    std::max(
                        1,
                        probConfig
                            .celluniverse3_signal_center_future_position_lock_min_future_both) &&
                bridgeProposal->windowMissingDaughterCount <=
                    std::max(
                        0,
                        probConfig
                            .celluniverse3_signal_center_future_position_lock_max_missing) &&
                bridgeProposal->windowParentPersists == 0 &&
                bridgeProposal->windowBestMatchedMinBrightness >=
                    probConfig
                        .celluniverse3_signal_center_future_position_lock_min_future_brightness &&
                bridgeProposal->parentShapeElongation >=
                    probConfig
                        .celluniverse3_signal_center_future_position_lock_min_parent_shape &&
                bridgeProposal->parentDistanceBalance >=
                    probConfig
                        .celluniverse3_signal_center_future_position_lock_min_parent_balance &&
                bridgeProposal->centerSnapMaxSeedDistance <=
                    std::max(
                        0.0f,
                        probConfig
                            .celluniverse3_signal_center_future_position_lock_max_snap_distance) &&
                bridgeProposal->bioSeparationRequired > 0.0f &&
                bridgeProposal->bioSeparationObserved >=
                    std::max(
                        0.0f,
                        probConfig
                            .celluniverse3_signal_center_future_position_lock_min_sep_fraction) *
                        bridgeProposal->bioSeparationRequired &&
                bridgeProposal->signalCenterSeparationRatio >=
                    probConfig
                        .celluniverse3_signal_center_future_position_lock_min_sep_ratio &&
                std::abs(bridgeProposal->signalCenterAxisAlignment) >=
                    probConfig
                        .celluniverse3_signal_center_future_position_lock_min_axis_alignment;
            const bool densityFutureSupportOk =
                !probConfig.bio_daughter_density_soft_require_future_support ||
                (bridgeProposal != nullptr &&
                 (bridgeProposal->futureWindowSplitRescue ||
                  cellUniverse3SignalCenterFutureDensitySupport ||
                  cellUniverse3AsymmetricUTunnelNearSeparationRescue ||
                  weakFutureCurrentBridgeDensitySupport ||
                  bestFutureSupportedRodTipPrimary ||
                  bestHasCleanFutureSplitSupport));
            const bool oneFrameFutureBridgeDensityRescue =
                simulationConfig.celluniverse2_enabled &&
                bridgeProposal != nullptr &&
                bridgeProposal->futureWindowSplitRescue &&
                bridgeProposal->centerSnapUsedAlignedPairFallback &&
                bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
                bridgeProposal->windowBothDaughtersSupported == 1 &&
                bridgeProposal->windowMissingDaughterCount <= 2 &&
                bridgeProposal->parentShapeElongation >= probConfig.split_density_one_frame_min_parent_shape &&
                bridgeProposal->parentShapeElongation < probConfig.split_density_one_frame_max_parent_shape;
            const float activeSoftMinFraction =
                oneFrameFutureBridgeDensityRescue
                    ? std::min(softMinFraction, probConfig.split_density_one_frame_soft_min_fraction)
                    : ((simulationConfig.celluniverse2_enabled &&
                        (bestHasCleanFutureSplitSupport ||
                         (bridgeProposal != nullptr &&
                          bridgeProposal->futureWindowSplitRescue)) &&
                        densityFutureSupportOk)
                           ? std::min(softMinFraction, probConfig.split_density_future_soft_min_fraction)
                           : softMinFraction);
            const float hardDensityThreshold =
                localBackground + minDaughterMeanAbs * activeSoftMinFraction;
            const bool lockedCleanFuturePcaBridgeDensityContext =
                simulationConfig.celluniverse2_enabled &&
                bestIsPcaBridgeOnly &&
                bridgeProposal != nullptr &&
                bridgeProposal->futureWindowSplitRescue &&
                bridgeProposal->centerSnapUsedAlignedPairFallback &&
                bridgeProposal->windowImmediateBothDaughtersSupported > 0;
            const float lockedCleanFuturePcaBridgeDensityFloor =
                localBackground +
                std::max(probConfig.split_density_locked_floor_abs,
                         probConfig.split_density_locked_floor_parent_fraction * std::max(0.0f, parentSignal));
            const float cellUniverse3SignalCenterFutureDensityFloor =
                localBackground +
                std::max(
                    std::max(
                        0.0f,
                        probConfig.celluniverse3_signal_center_future_density_floor_abs),
                    std::max(
                        0.0f,
                        probConfig
                            .celluniverse3_signal_center_future_density_floor_parent_fraction) *
                        std::max(0.0f, parentSignal));
            const bool belowObviousBackgroundFloor =
                minDaughterMeanAbs <= 0.0f ||
                minDaughterMean < hardDensityThreshold ||
                (lockedCleanFuturePcaBridgeDensityContext &&
                 minDaughterMean < lockedCleanFuturePcaBridgeDensityFloor);
            const bool cellUniverse3SignalCenterFutureDensityWaived =
                cellUniverse3SignalCenterFutureDensitySupport &&
                minDaughterMean >= cellUniverse3SignalCenterFutureDensityFloor &&
                bridgeProposal != nullptr &&
                bridgeProposal->windowParentPersists == 0 &&
                bridgeProposal->windowBestMatchedMinBrightness >=
                    probConfig
                        .celluniverse3_signal_center_future_position_lock_min_future_brightness;
            const float densityCleanFuturePairSnapLimit =
                std::max(
                    std::max(
                        std::max(0.0f,
                                 probConfig.pca_bridge_future_window_match_distance),
                        0.75f * std::max(1.0f, srcMaxR)),
                    std::max(
                        0.0f,
                        probConfig
                            .pca_bridge_future_window_pca_snap_max_radius_scale) *
                        std::max(1.0f, srcMaxR));
            const bool lockedCleanFuturePcaBridgeDensityWaived =
                simulationConfig.celluniverse2_enabled &&
                bestIsPcaBridgeOnly &&
                bridgeProposal != nullptr &&
                bridgeProposal->futureWindowSplitRescue &&
                bridgeProposal->centerSnapUsedAlignedPairFallback &&
                bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
                !belowObviousBackgroundFloor &&
                bridgeProposal->windowBothDaughtersSupported >=
                    std::max(2,
                             probConfig
                                 .pca_bridge_future_window_min_both_daughter_support) &&
                bridgeProposal->windowMissingDaughterCount == 0 &&
                bridgeProposal->windowParentPersists == 0 &&
                bridgeProposal->windowBestMatchedMinBrightness >=
                    std::min(probConfig.split_immediate_pca_continuation_min_brightness,
                             probConfig.split_one_frame_aligned_pca_continuation_min_brightness) &&
                bridgeProposal->parentShapeElongation >=
                    std::min(probConfig.split_immediate_pca_continuation_min_parent_shape,
                             probConfig.split_one_frame_aligned_pca_continuation_min_parent_shape) &&
                bridgeProposal->parentDistanceBalance >=
                    std::min(probConfig.split_immediate_pca_continuation_min_parent_balance,
                             probConfig.split_one_frame_aligned_pca_continuation_min_parent_balance) &&
                bridgeProposal->centerSnapMaxSeedDistance <=
                    std::max(densityCleanFuturePairSnapLimit,
                             probConfig.split_one_frame_aligned_pca_continuation_snap_scale *
                                 std::max(1.0f, srcMaxR)) &&
                std::max(drift1, drift2) <=
                    std::max(probConfig.split_overlap_one_frame_signal_max_drift_abs,
                             probConfig.split_current_locked_bridge_max_drift_scale * std::max(1.0f, srcMaxR));
            const bool costBackedCleanFuturePcaBridgeDensityWaived =
                simulationConfig.celluniverse2_enabled &&
                bestIsPcaBridgeOnly &&
                bestLabel == "bridge_primary" &&
                bridgeProposal != nullptr &&
                bridgeProposal->futureWindowSplitRescue &&
                bridgeProposal->centerSnapUsedAlignedPairFallback &&
                bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
                bridgeProposal->windowBothDaughtersSupported >=
                    std::max(2,
                             probConfig
                                 .pca_bridge_future_window_min_both_daughter_support) &&
                bridgeProposal->windowMissingDaughterCount == 0 &&
                bridgeProposal->windowParentPersists == 0 &&
                bridgeProposal->windowBestMatchedMinBrightness >=
                    probConfig
                        .split_density_cost_backed_clean_future_min_future_brightness &&
                bridgeProposal->parentDistanceBalance >=
                    probConfig
                        .split_density_cost_backed_clean_future_min_parent_balance &&
                std::max(d1BrightnessStats.first, d2BrightnessStats.first) >=
                    densityThreshold &&
                (baselineImageCost - bestImageCost) >=
                    static_cast<double>(
                        probConfig
                            .split_density_cost_backed_clean_future_min_image_gain_abs) &&
                (baselineTotal - bestTotal) >=
                    static_cast<double>(
                        probConfig
                            .split_density_cost_backed_clean_future_min_total_gain_abs);
            costBackedCleanFuturePcaBridgeDensityWaivedActive =
                costBackedCleanFuturePcaBridgeDensityWaived;
            const bool softEligible =
                simulationConfig.celluniverse2_enabled &&
                probConfig.bio_daughter_density_soft_gate_enabled &&
                densityFutureSupportOk &&
                !belowObviousBackgroundFloor &&
                !lockedCleanFuturePcaBridgeDensityWaived &&
                !costBackedCleanFuturePcaBridgeDensityWaived &&
                !cellUniverse3SignalCenterFutureDensityWaived;
            const double normalizedExcess =
                static_cast<double>(densityThreshold - minDaughterMean) /
                std::max(1e-6, static_cast<double>(densityThreshold - localBackground));
            if (lockedCleanFuturePcaBridgeDensityWaived) {
                std::cout << "[Split Soft Gate Waived] " << parentName
                          << " reason=locked_clean_future_pca_bridge_density"
                          << " d1Mean=" << d1BrightnessStats.first
                          << " d1Std=" << d1BrightnessStats.second
                          << " d2Mean=" << d2BrightnessStats.first
                          << " d2Std=" << d2BrightnessStats.second
                          << " parentMean=" << parentBrightnessStats.first
                          << " background=" << localBackground
                          << " parentSignal=" << parentSignal
                          << " threshold=" << densityThreshold
                          << " hardThreshold=" << hardDensityThreshold
                          << " lockedDensityFloor="
                          << lockedCleanFuturePcaBridgeDensityFloor
                          << " normalizedExcess=" << normalizedExcess
                          << " futureSupportOk=" << (densityFutureSupportOk ? 1 : 0)
                          << " parentDistBalance="
                          << bridgeProposal->parentDistanceBalance
                          << " centerSnapMaxSeedDistance="
                          << bridgeProposal->centerSnapMaxSeedDistance
                          << " bestIdx=" << bestIdx
                          << " bestLabel=" << bestLabel
                          << std::endl;
            } else if (costBackedCleanFuturePcaBridgeDensityWaived) {
                std::cout << "[Split Soft Gate Waived] " << parentName
                          << " reason=cost_backed_clean_future_pca_bridge_density"
                          << " d1Mean=" << d1BrightnessStats.first
                          << " d1Std=" << d1BrightnessStats.second
                          << " d2Mean=" << d2BrightnessStats.first
                          << " d2Std=" << d2BrightnessStats.second
                          << " parentMean=" << parentBrightnessStats.first
                          << " background=" << localBackground
                          << " parentSignal=" << parentSignal
                          << " threshold=" << densityThreshold
                          << " hardThreshold=" << hardDensityThreshold
                          << " normalizedExcess=" << normalizedExcess
                          << " futureSupportOk=" << (densityFutureSupportOk ? 1 : 0)
                          << " parentDistBalance="
                          << bridgeProposal->parentDistanceBalance
                          << " futureMinBrightness="
                          << bridgeProposal->windowBestMatchedMinBrightness
                          << " futureMinBrightnessLimit="
                          << probConfig
                                 .split_density_cost_backed_clean_future_min_future_brightness
                          << " centerSnapMaxSeedDistance="
                          << bridgeProposal->centerSnapMaxSeedDistance
                          << " imageCostDiff="
                          << (bestImageCost - baselineImageCost)
                          << " totalCostDiff="
                          << (bestTotal - baselineTotal)
                          << " bestIdx=" << bestIdx
                          << " bestLabel=" << bestLabel
                          << std::endl;
            } else if (cellUniverse3SignalCenterFutureDensityWaived) {
                std::cout << "[Split Soft Gate Waived] " << parentName
                          << " reason=celluniverse3_signal_center_future_density"
                          << " d1Mean=" << d1BrightnessStats.first
                          << " d1Std=" << d1BrightnessStats.second
                          << " d2Mean=" << d2BrightnessStats.first
                          << " d2Std=" << d2BrightnessStats.second
                          << " parentMean=" << parentBrightnessStats.first
                          << " background=" << localBackground
                          << " parentSignal=" << parentSignal
                          << " threshold=" << densityThreshold
                          << " hardThreshold=" << hardDensityThreshold
                          << " signalCenterDensityFloor="
                          << cellUniverse3SignalCenterFutureDensityFloor
                          << " normalizedExcess=" << normalizedExcess
                          << " futureSupportOk=" << (densityFutureSupportOk ? 1 : 0)
                          << " futureMinBrightness="
                          << bridgeProposal->windowBestMatchedMinBrightness
                          << " parentDistBalance="
                          << bridgeProposal->parentDistanceBalance
                          << " bestIdx=" << bestIdx
                          << " bestLabel=" << bestLabel
                          << std::endl;
            } else if (softEligible) {
                addSplitSoftGeometryPenalty(
                    "daughter_density_brightness_near_miss",
                    normalizedExcess,
                    static_cast<double>(std::max(
                        0.0f,
                        probConfig.bio_daughter_density_soft_penalty_fraction)));
                std::cout << "[Split Soft Gate] " << parentName
                          << " reason=daughter_density_brightness_near_miss"
                          << " d1Mean=" << d1BrightnessStats.first
                          << " d1Std=" << d1BrightnessStats.second
                          << " d2Mean=" << d2BrightnessStats.first
                          << " d2Std=" << d2BrightnessStats.second
                          << " parentMean=" << parentBrightnessStats.first
                          << " background=" << localBackground
                          << " parentSignal=" << parentSignal
                          << " threshold=" << densityThreshold
                          << " adaptiveSignalThreshold=" << adaptiveSignalThreshold
                          << " hardThreshold=" << hardDensityThreshold
                          << " hardThresholdFraction="
                          << activeSoftMinFraction
                          << " normalizedExcess=" << normalizedExcess
                          << " futureSupportOk=" << (densityFutureSupportOk ? 1 : 0)
                          << " bestIdx=" << bestIdx
                          << " bestLabel=" << bestLabel
                          << std::endl;
            } else {
                std::cout << "[Split Reject bio] " << parentName
                          << " reason=daughter_density_brightness"
                          << " d1Mean=" << d1BrightnessStats.first
                          << " d1Std=" << d1BrightnessStats.second
                          << " d2Mean=" << d2BrightnessStats.first
                          << " d2Std=" << d2BrightnessStats.second
                          << " parentMean=" << parentBrightnessStats.first
                          << " background=" << localBackground
                          << " parentSignal=" << parentSignal
                          << " threshold=" << densityThreshold
                          << " adaptiveSignalThreshold=" << adaptiveSignalThreshold
                          << " hardThreshold=" << hardDensityThreshold
                          << " hardThresholdFraction="
                          << activeSoftMinFraction
                          << " absoluteMin=" << minDaughterMeanAbs
                          << " backgroundMargin=" << minDaughterBackgroundMargin
                          << " parentFraction=" << minDaughterMeanParentFraction
                          << " belowObviousBackgroundFloor=" << (belowObviousBackgroundFloor ? 1 : 0)
                          << " softEligible=" << (softEligible ? 1 : 0)
                          << " futureSupportOk=" << (densityFutureSupportOk ? 1 : 0)
                          << " weakFutureCurrentBridgeDensitySupport="
                          << (weakFutureCurrentBridgeDensitySupport ? 1 : 0)
                          << " cellUniverse3SignalCenterFutureDensitySupport="
                          << (cellUniverse3SignalCenterFutureDensitySupport ? 1 : 0)
                          << " signalCenterDensityFloor="
                          << cellUniverse3SignalCenterFutureDensityFloor
                          << " bestIdx=" << bestIdx
                          << " bestLabel=" << bestLabel
                          << std::endl;
                restoreLiveParent();
                return {0.0, noop};
            }
        }
    }

    if (finalBioSeparationSoftRescue) {
        const cv::Point3f finalD1Pos(bestD1.getX(), bestD1.getY(), bestD1.getZ());
        const cv::Point3f finalD2Pos(bestD2.getX(), bestD2.getY(), bestD2.getZ());
        const float finalDaughterSep =
            static_cast<float>(cv::norm(finalD2Pos - finalD1Pos));
        const float requiredSep =
            std::max(0.0f, probConfig.bio_min_daughter_separation_parent_fraction) *
            std::max(1.0f, srcMaxR);
        if (requiredSep > 0.0f && finalDaughterSep < requiredSep) {
            const double normalizedExcess =
                static_cast<double>(requiredSep - finalDaughterSep) /
                std::max(1.0, static_cast<double>(requiredSep));
            addSplitSoftGeometryPenalty(
                "final_bio_separation_near_miss",
                normalizedExcess,
                static_cast<double>(
                    std::max(0.0f,
                             probConfig.bio_separation_soft_penalty_fraction)));
            std::cout << "[Split Soft Gate] " << parentName
                      << " reason=final_bio_separation_near_miss"
                      << " finalSep=" << finalDaughterSep
                      << " requiredSep=" << requiredSep
                      << " hardFloorFraction="
                      << std::clamp(probConfig.bio_separation_soft_min_fraction,
                                    0.0f, 1.0f)
                      << " normalizedExcess=" << normalizedExcess
                      << " bestIdx=" << bestIdx
                      << " bestLabel=" << bestLabel
                      << std::endl;
        }
    }

    bool softOverlapAcceptedForCost = false;
    bool cleanContinuationDaughterOverlapAcceptedForCost = false;
    if (simulationConfig.celluniverse2_enabled) {
        std::string overlapA;
        std::string overlapB;
        float aInB = 0.0f;
        float bInA = 0.0f;
        const double overlapGateCostDiff = bestTotal - baselineTotal;
        const double overlapGateImageCostDiff = bestImageCost - baselineImageCost;
        const std::string acceptedSiblingA =
            bestIsDeterministicSingleProposal ? bestCells[d1IdxBest].getName() : "";
        const std::string acceptedSiblingB =
            bestIsDeterministicSingleProposal ? bestCells[d2IdxBest].getName() : "";
        const bool oneFrameRawPcaBridgeOverlapSupport =
            bestIsPcaBridgeOnly &&
            bridgeProposal != nullptr &&
            !bridgeProposal->centerSnapApplied &&
            !bridgeProposal->immediateFutureCenterBacked &&
            ((bridgeProposal->windowBothDaughtersSupported >= 1 &&
              bridgeProposal->windowMissingDaughterCount <= 2 &&
             bridgeProposal->windowBestMatchedMinBrightness >= probConfig.split_overlap_one_frame_raw_min_brightness) ||
             (bridgeProposal->parentShapeElongation >= probConfig.split_overlap_one_frame_raw_min_parent_shape &&
              finalAxisLen >= probConfig.split_long_raw_pca_bridge_min_axis_length_scale * std::max(1.0f, srcMaxR) &&
              bridgeProposal->parentDistanceBalance >= probConfig.split_overlap_one_frame_raw_min_parent_balance_secondary)) &&
            bridgeProposal->windowParentPersists == 0 &&
            bridgeProposal->parentDistanceBalance >= probConfig.split_overlap_one_frame_raw_min_parent_balance;
        const bool asymmetricRawPcaBridgeOverlapSupport =
            bestIsPcaBridgeOnly &&
            bridgeProposal != nullptr &&
            !bridgeProposal->centerSnapUsedAlignedPairFallback &&
            bridgeProposal->windowBothDaughtersSupported >= 1 &&
            bridgeProposal->windowMissingDaughterCount <= 2 &&
            bridgeProposal->windowParentPersists == 0 &&
            bridgeProposal->parentShapeElongation >= probConfig.split_asymmetric_raw_pca_bridge_min_parent_shape &&
            bridgeProposal->parentDistanceBalance >= probConfig.split_asymmetric_raw_pca_bridge_min_parent_balance &&
            finalAxisLen >= probConfig.split_asymmetric_raw_pca_bridge_min_axis_length_scale * std::max(1.0f, srcMaxR);
        const bool oneFrameCurrentFallbackPcaBridgeOverlapSupport =
            bestIsPcaBridgeOnly &&
            bridgeProposal != nullptr &&
            bridgeProposal->centerSnapApplied &&
            !bridgeProposal->immediateFutureCenterBacked &&
            bridgeProposal->centerSnapUsedAlignedPairFallback &&
            bridgeProposal->windowBothDaughtersSupported >= 1 &&
            bridgeProposal->windowMissingDaughterCount <= 2 &&
            bridgeProposal->windowParentPersists == 0 &&
            bridgeProposal->windowBestMatchedMinBrightness >= probConfig.pca_bridge_current_fallback_min_future_brightness &&
            bridgeProposal->parentDistanceBalance >= probConfig.pca_bridge_current_fallback_low_balance_min_parent_balance &&
            bridgeProposal->centerSnapMaxSeedDistance <=
                std::max(probConfig.pca_bridge_future_window_match_distance,
                         probConfig.pca_bridge_current_fallback_max_snap_radius_scale * std::max(1.0f, srcMaxR)) &&
            finalAxisLen >= probConfig.split_asymmetric_raw_pca_bridge_min_axis_length_scale * std::max(1.0f, srcMaxR);
        const bool oneFrameAlignedFuturePcaBridgeOverlapSupport =
            bestIsPcaBridgeOnly &&
            bestLabel == "bridge_primary" &&
            bridgeProposal != nullptr &&
            bridgeProposal->centerSnapApplied &&
            bridgeProposal->centerSnapUsedAlignedPairFallback &&
            bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
            bridgeProposal->windowBothDaughtersSupported >= 1 &&
            bridgeProposal->windowMissingDaughterCount == 0 &&
            bridgeProposal->windowParentPersists == 0 &&
            bridgeProposal->windowBestMatchedMinBrightness >= probConfig.split_one_frame_aligned_pca_continuation_min_brightness &&
            bridgeProposal->parentShapeElongation >= probConfig.split_one_frame_aligned_pca_continuation_min_parent_shape &&
            bridgeProposal->parentDistanceBalance >= probConfig.split_one_frame_aligned_pca_continuation_min_parent_balance &&
            bridgeProposal->parentDistanceBalance <= probConfig.split_one_frame_aligned_pca_continuation_max_parent_balance &&
            bridgeProposal->centerSnapMaxSeedDistance <=
                std::max(probConfig.pca_bridge_future_window_match_distance,
                         probConfig.split_one_frame_aligned_pca_continuation_snap_scale * std::max(1.0f, srcMaxR)) &&
            finalAxisLen >= probConfig.split_asymmetric_raw_pca_bridge_min_axis_length_scale * std::max(1.0f, srcMaxR);
        const bool twoFrameBridgeAxisPlaceOverlapSupport =
            bridgeProposalOnly &&
            bestLabel == "bridge_axis_place" &&
            bridgeProposal != nullptr &&
            bridgeProposal->daughterSphereRadius > 0.0f &&
            bridgeProposal->windowBothDaughtersSupported >= 2 &&
            bridgeProposal->windowMissingDaughterCount == 0 &&
            bridgeProposal->windowParentPersists == 0 &&
            bridgeProposal->windowBestMatchedMinBrightness >= probConfig.split_bridge_axis_place_future_min_brightness;
        const bool oneFrameSignalCenterOverlapSupport =
            bestIsSignalCenterProposal &&
            bridgeProposal != nullptr &&
            probConfig.pca_bridge_future_window_enabled &&
            bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
            bridgeProposal->windowBothDaughtersSupported >= 1 &&
            bridgeProposal->windowMissingDaughterCount == 0 &&
            bridgeProposal->windowParentPersists == 0 &&
            bridgeProposal->windowBestMatchedMinBrightness >= probConfig.split_overlap_one_frame_signal_min_brightness &&
            bridgeProposal->parentDistanceBalance >= probConfig.split_overlap_one_frame_signal_min_parent_balance &&
            std::max(drift1, drift2) <=
                std::max(probConfig.split_overlap_one_frame_signal_max_drift_abs,
                         probConfig.split_overlap_one_frame_signal_max_drift_scale * std::max(1.0f, srcMaxR));
        const bool twoFrameSignalCenterOverlapSupport =
            bestIsSignalCenterProposal &&
            bridgeProposal != nullptr &&
            probConfig.pca_bridge_future_window_enabled &&
            bridgeProposal->windowBothDaughtersSupported >= 2 &&
            bridgeProposal->windowMissingDaughterCount == 0 &&
            bridgeProposal->windowParentPersists == 0 &&
            (bridgeProposal->windowBestMatchedMinBrightness >= probConfig.split_overlap_two_frame_signal_min_brightness ||
             dimExactFutureSignalRodBlockerRescue) &&
            bridgeProposal->parentDistanceBalance >= probConfig.split_overlap_two_frame_signal_min_parent_balance;
        const bool severeSignalCenterOverlapSupport =
            simulationConfig.celluniverse3_enabled &&
            probConfig.celluniverse3_severe_signal_center_overlap_rescue_enabled &&
            bestIsSignalCenterProposal &&
            bridgeProposal != nullptr &&
            probConfig.signal_center_severe_future_claim_enabled &&
            bridgeProposal->windowBothDaughtersSupported >=
                std::max(1,
                         probConfig
                             .signal_center_severe_future_claim_min_future_both) &&
            bridgeProposal->windowMissingDaughterCount <=
                std::max(0,
                         probConfig.signal_center_severe_future_claim_max_missing) &&
            bridgeProposal->windowParentPersists == 0 &&
            bridgeProposal->windowBestMatchedMinBrightness >=
                std::max(0.0f,
                         probConfig.signal_center_severe_future_claim_min_brightness) &&
            bridgeProposal->parentShapeElongation >=
                std::max(0.0f,
                         probConfig.signal_center_severe_future_claim_min_parent_shape) &&
            bridgeProposal->parentDistanceBalance >=
                std::max(0.0f,
                         probConfig
                             .signal_center_severe_future_claim_min_parent_balance);
        const bool cleanFutureSupportedSplitOverlap =
            bestHasCleanFutureSignalSupport ||
            oneFrameSignalCenterOverlapSupport ||
            twoFrameSignalCenterOverlapSupport ||
            (bridgeProposal != nullptr &&
             (bridgeProposal->futureWindowSplitRescue ||
              bestHasCleanFutureSplitSupport ||
              bestHasCleanCurrentBridgeSupport ||
              oneFrameRawPcaBridgeOverlapSupport ||
              asymmetricRawPcaBridgeOverlapSupport ||
              oneFrameCurrentFallbackPcaBridgeOverlapSupport ||
              oneFrameAlignedFuturePcaBridgeOverlapSupport ||
              twoFrameBridgeAxisPlaceOverlapSupport));
        const bool strongFutureSupportedSplitOverlap =
            twoFrameBridgeAxisPlaceOverlapSupport ||
            (cleanFutureSupportedSplitOverlap &&
             bridgeProposal != nullptr &&
             bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
             bridgeProposal->windowBothDaughtersSupported >= 2 &&
             bridgeProposal->windowMissingDaughterCount == 0 &&
             bridgeProposal->windowParentPersists == 0 &&
             bridgeProposal->windowBestMatchedMinBrightness >= probConfig.split_bridge_axis_place_future_min_brightness);
        const float cleanFutureOverlapTolerance =
            twoFrameSignalCenterOverlapSupport
                ? std::max(
                      probConfig.split_overlap_signal_tolerance_min,
                      std::clamp(
                          probConfig.split_future_supported_hard_overlap_tolerance *
                              probConfig.split_overlap_signal_tolerance_multiplier,
                          probConfig.split_overlap_signal_tolerance_clamp_min,
                          probConfig.split_overlap_signal_tolerance_clamp_max))
            : (bestHasCleanFutureSignalSupport ||
               oneFrameSignalCenterOverlapSupport)
                ? std::max(
                      probConfig.split_overlap_signal_tolerance_min,
                      std::clamp(
                          probConfig.split_future_supported_hard_overlap_tolerance * probConfig.split_overlap_signal_tolerance_multiplier,
                          probConfig.split_overlap_signal_tolerance_clamp_min,
                          probConfig.split_overlap_signal_tolerance_clamp_max))
                : std::max(probConfig.split_overlap_default_tolerance_min,
                           std::clamp(
                               probConfig
                                   .split_future_supported_hard_overlap_tolerance,
                               probConfig.split_overlap_default_tolerance_min,
                               probConfig.split_overlap_default_tolerance_max));
        const float strongFutureOverlapTolerance =
            strongFutureSupportedSplitOverlap
                ? std::max(
                      cleanFutureOverlapTolerance,
                      twoFrameBridgeAxisPlaceOverlapSupport
                          ? probConfig.split_overlap_bridge_axis_tolerance
                          : (bestIsPcaBridgeOnly ? probConfig.split_overlap_pca_bridge_tolerance : probConfig.split_overlap_other_strong_tolerance))
                : cleanFutureOverlapTolerance;
        const float cleanCurrentBridgeOverlapTolerance =
            bestHasCleanCurrentBridgeSupport
                ? std::max(strongFutureOverlapTolerance, probConfig.split_overlap_clean_current_bridge_tolerance)
                : strongFutureOverlapTolerance;
        const float currentFallbackBridgeOverlapTolerance =
            oneFrameCurrentFallbackPcaBridgeOverlapSupport
                ? std::max(cleanCurrentBridgeOverlapTolerance, probConfig.split_overlap_current_fallback_bridge_tolerance)
                : cleanCurrentBridgeOverlapTolerance;
        const float alignedFutureBridgeOverlapTolerance =
            oneFrameAlignedFuturePcaBridgeOverlapSupport
                ? std::max(currentFallbackBridgeOverlapTolerance, probConfig.split_overlap_aligned_future_bridge_tolerance)
                : currentFallbackBridgeOverlapTolerance;
        const bool delayedFuturePcaBridgeOverlapSupport =
            simulationConfig.celluniverse2_enabled &&
            bestIsPcaBridgeOnly &&
            bestLabel == "bridge_primary" &&
            bridgeProposal != nullptr &&
            probConfig.split_delayed_future_hard_overlap_tolerance > 0.0f &&
            bridgeProposal->windowImmediateBothDaughtersSupported == 0 &&
            bridgeProposal->windowBothDaughtersSupported >=
                static_cast<int>(
                    probConfig.split_delayed_future_pca_bridge_min_future_both) &&
            bridgeProposal->windowMissingDaughterCount <=
                static_cast<int>(
                    probConfig
                        .split_delayed_future_pca_bridge_max_missing_daughters) &&
            bridgeProposal->windowParentPersists == 0 &&
            bridgeProposal->windowBestMatchedMinBrightness >=
                probConfig.split_delayed_future_pca_bridge_min_brightness &&
            bridgeProposal->parentShapeElongation >=
                probConfig.split_delayed_future_pca_bridge_min_parent_shape &&
            bridgeProposal->parentDistanceBalance >=
                probConfig.split_delayed_future_pca_bridge_min_parent_balance;
        const float delayedFutureOverlapTolerance =
            delayedFuturePcaBridgeOverlapSupport
                ? std::max(
                      alignedFutureBridgeOverlapTolerance,
                      probConfig.split_delayed_future_hard_overlap_tolerance)
                : alignedFutureBridgeOverlapTolerance;
        const float hardOverlapTolerance =
            (cleanFutureSupportedSplitOverlap || delayedFuturePcaBridgeOverlapSupport)
                ? delayedFutureOverlapTolerance
                : probConfig.split_overlap_hard_default_tolerance;
        const float generalCleanFutureOverlapMinBrightness = std::min(
            probConfig.split_current_locked_bridge_min_brightness,
            std::min(
                probConfig.split_one_frame_aligned_pca_continuation_min_brightness,
                probConfig.split_exact_future_center_bridge_min_brightness));
        const float generalCleanFutureOverlapMinParentShape = std::min(
            probConfig.split_current_locked_bridge_min_parent_shape,
            std::min(
                probConfig.split_one_frame_aligned_pca_continuation_min_parent_shape,
                probConfig.split_exact_future_center_bridge_min_parent_shape));
        const float generalCleanFutureOverlapMinParentBalance = std::min(
            probConfig.split_current_locked_bridge_min_parent_balance,
            std::min(
                probConfig.split_one_frame_aligned_pca_continuation_min_parent_balance,
                probConfig.split_exact_future_center_bridge_min_parent_balance));
        const float generalCleanFutureOverlapMaxParentBalance = std::max(
            probConfig.split_current_locked_bridge_max_parent_balance,
            probConfig.split_one_frame_aligned_pca_continuation_max_parent_balance);
        const float generalCleanFutureOverlapSnapLimit = std::max(
            probConfig.pca_bridge_future_window_match_distance,
            std::max(probConfig.split_clean_future_position_lock_min_snap_scale,
                     std::max(
                         0.0f,
                         probConfig.pca_bridge_future_window_pca_snap_max_radius_scale)) *
                std::max(1.0f, srcMaxR));
        const float generalCleanFutureOverlapMaxDrift = std::max(
            probConfig.split_locked_clean_future_pca_bridge_max_drift_abs,
            probConfig.split_locked_clean_future_pca_bridge_max_drift_scale *
                std::max(1.0f, srcMaxR));
        const float overlapGateMaxDaughterSeedDrift = std::max(drift1, drift2);
        const double generalCleanFutureImageGainRequired = std::max(
            static_cast<double>(
                probConfig.split_general_clean_future_pca_bridge_image_gain_abs),
            static_cast<double>(
                probConfig
                    .split_general_clean_future_pca_bridge_image_gain_fraction) *
                baselineImageCost);
        const bool crowdedGeneralCleanFuturePcaBridgeSupport =
            bestIsPcaBridgeOnly &&
            bestLabel == "bridge_primary" &&
            bridgeProposal != nullptr &&
            probConfig.pca_bridge_future_window_enabled &&
            bridgeProposal->centerSnapApplied &&
            (bridgeProposal->immediateFutureCenterBacked ||
             bridgeProposal->centerSnapUsedAlignedPairFallback) &&
            bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
            bridgeProposal->windowBothDaughtersSupported >= 1 &&
            bridgeProposal->windowMissingDaughterCount == 0 &&
            bridgeProposal->windowParentPersists == 0 &&
            bridgeProposal->windowBestMatchedMinBrightness >=
                generalCleanFutureOverlapMinBrightness &&
            bridgeProposal->parentShapeElongation >=
                generalCleanFutureOverlapMinParentShape &&
            bridgeProposal->parentDistanceBalance >=
                generalCleanFutureOverlapMinParentBalance &&
            bridgeProposal->parentDistanceBalance <=
                generalCleanFutureOverlapMaxParentBalance &&
            bridgeProposal->centerSnapMaxSeedDistance <=
                generalCleanFutureOverlapSnapLimit &&
            overlapGateMaxDaughterSeedDrift <= generalCleanFutureOverlapMaxDrift &&
            savedNonTrashCellCount >=
                probConfig
                    .split_general_clean_future_pca_bridge_crowded_cell_count_min;
        auto isSuspiciousRodOverlapBlocker = [&](const std::string &name) {
            if (name.empty() ||
                name == acceptedSiblingA ||
                name == acceptedSiblingB) {
                return false;
            }
            for (const auto &cell : savedCells) {
                if (cell.getName() != name || cell.isTrash()) continue;
                const float maxR = std::max({cell.getARadius(),
                                             cell.getBRadius(),
                                             cell.getCRadius()});
                const float minR = std::max(
                    1e-3f,
                    std::min({cell.getARadius(),
                              cell.getBRadius(),
                              cell.getCRadius()}));
                return maxR / minR >= probConfig.split_overlap_suspicious_rod_shape_ratio;
            }
            return false;
        };
        if (findNewOrWorseCellBodyOverlapIn(_realFrame,
                                            bestCells,
                                            savedCells,
                                            /*includeTrash=*/false,
                                            /*scale=*/1.0f,
                                            acceptedSiblingA,
                                            acceptedSiblingB,
                                            /*tolerance=*/hardOverlapTolerance,
                                            &overlapA,
                                            &overlapB,
                                            &aInB,
                                            &bInA)) {
            const bool cleanContinuationDaughterOverlap =
                cleanTwoFrameRodTipContinuationRescue &&
                !acceptedSiblingA.empty() &&
                ((overlapA == acceptedSiblingA ||
                  overlapA == acceptedSiblingB ||
                  overlapB == acceptedSiblingA ||
                  overlapB == acceptedSiblingB) &&
                 !((overlapA == acceptedSiblingA && overlapB == acceptedSiblingB) ||
                   (overlapA == acceptedSiblingB && overlapB == acceptedSiblingA))) &&
                aInB <= probConfig.split_overlap_clean_continuation_a_in_b_max &&
                bInA <= probConfig.split_overlap_clean_continuation_b_in_a_max;
            const bool cleanExactFutureSignalOverlap =
                twoFrameSignalCenterOverlapSupport &&
                bridgeProposal != nullptr &&
                bridgeProposal->centerSnapApplied &&
                bridgeProposal->immediateFutureCenterBacked &&
                !bridgeProposal->centerSnapUsedAlignedPairFallback &&
                bridgeProposal->centerSnapMaxSeedDistance <=
                    std::max(probConfig.split_overlap_exact_signal_snap_abs,
                             probConfig.split_overlap_exact_signal_snap_scale * std::max(1.0f, srcMaxR)) &&
                bridgeProposal->parentDistanceBalance >= probConfig.split_overlap_exact_signal_min_parent_balance &&
                bridgeProposal->windowBestMatchedMinBrightness >= probConfig.split_overlap_exact_signal_min_brightness &&
                aInB <= probConfig.split_overlap_exact_signal_a_in_b_max &&
                bInA <= probConfig.split_overlap_exact_signal_b_in_a_max;
            const bool cleanTwoFrameSignalOverlap =
                twoFrameSignalCenterOverlapSupport &&
                bridgeProposal != nullptr &&
                bridgeProposal->parentDistanceBalance >= probConfig.split_overlap_exact_signal_min_parent_balance &&
                bridgeProposal->windowBestMatchedMinBrightness >= probConfig.split_overlap_exact_signal_min_brightness &&
                aInB <= probConfig.split_overlap_exact_signal_a_in_b_max &&
                bInA <= probConfig.split_overlap_exact_signal_b_in_a_max;
            const bool severeSignalCenterSoftOverlap =
                severeSignalCenterOverlapSupport &&
                aInB <=
                    std::max(
                        0.0f,
                        probConfig
                            .celluniverse3_severe_signal_center_overlap_a_in_b_max) &&
                bInA <=
                    std::max(
                        0.0f,
                        probConfig
                            .celluniverse3_severe_signal_center_overlap_b_in_a_max);
            const bool highShapeRawRodSignalOverlap =
                bestIsSignalCenterProposal &&
                bestLabel == "bridge_primary" &&
                bridgeProposal != nullptr &&
                bridgeProposal->gapStartBin <= static_cast<int>(probConfig.split_high_shape_rod_signal_gap_start_max) &&
                bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
                bridgeProposal->windowBothDaughtersSupported >= static_cast<int>(probConfig.split_high_shape_rod_signal_min_future_both) &&
                bridgeProposal->windowMissingDaughterCount <= static_cast<int>(probConfig.split_high_shape_rod_signal_max_missing_daughters) &&
                bridgeProposal->windowParentPersists == 0 &&
                bridgeProposal->parentShapeElongation >= probConfig.split_high_shape_rod_signal_min_parent_shape &&
                bridgeProposal->parentDistanceBalance >= probConfig.split_high_shape_rod_signal_min_parent_balance &&
                bridgeCostRescueGapDensity <= probConfig.split_high_shape_rod_signal_max_gap_density &&
                bridgeCostRescueValleyFromBright <= probConfig.split_high_shape_rod_signal_max_valley_from_bright &&
                finalAxisLen >= probConfig.split_high_shape_rod_signal_min_axis_length_scale * std::max(1.0f, srcMaxR) &&
                aInB <= probConfig.split_overlap_high_shape_rod_a_in_b_max &&
                bInA <= probConfig.split_overlap_high_shape_rod_b_in_a_max;
            const bool asymmetricRawPcaBridgeSoftOverlap =
                probConfig.split_asymmetric_raw_pca_bridge_soft_overlap_enabled &&
                asymmetricRawPcaBridgeOverlapSupport &&
                bestLabel == "bridge_primary" &&
                bridgeProposal != nullptr &&
                bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
                aInB <= probConfig.split_asymmetric_raw_pca_bridge_soft_overlap_a_in_b_max &&
                bInA <= probConfig.split_asymmetric_raw_pca_bridge_soft_overlap_b_in_a_max &&
                overlapGateImageCostDiff <= -std::max(
                    static_cast<double>(probConfig.split_asymmetric_raw_pca_bridge_image_gain_abs),
                    static_cast<double>(probConfig.split_asymmetric_raw_pca_bridge_image_gain_fraction) *
                        baselineImageCost) &&
                overlapGateCostDiff <= -std::max(
                    static_cast<double>(probConfig.split_asymmetric_raw_pca_bridge_cost_limit_abs),
                    static_cast<double>(probConfig.split_asymmetric_raw_pca_bridge_cost_limit_fraction) *
                        baselineImageCost) &&
                bridgeCostRescueValleyFromBright <=
                    probConfig.split_asymmetric_raw_pca_bridge_max_valley_from_bright &&
                bridgeCostRescueGapDensity <=
                    probConfig.split_asymmetric_raw_pca_bridge_max_gap_density &&
                std::max(drift1, drift2) <= std::max(
                    probConfig.split_asymmetric_raw_pca_bridge_max_drift_abs,
                    probConfig.split_asymmetric_raw_pca_bridge_max_drift_scale *
                        std::max(1.0f, srcMaxR));
            const bool crowdedGeneralCleanFuturePcaBridgeSoftOverlap =
                crowdedGeneralCleanFuturePcaBridgeSupport &&
                overlapGateImageCostDiff <= -generalCleanFutureImageGainRequired &&
                aInB <=
                    probConfig
                        .split_general_clean_future_pca_bridge_overlap_a_in_b_max &&
                bInA <=
                    probConfig
                        .split_general_clean_future_pca_bridge_overlap_b_in_a_max &&
                bridgeCostRescueGapDensity <=
                    probConfig.split_locked_clean_future_pca_bridge_max_gap_density &&
                bridgeCostRescueValleyFromBright <=
                    probConfig
                        .split_locked_clean_future_pca_bridge_max_valley_from_bright;
            if (cleanContinuationDaughterOverlap ||
                cleanExactFutureSignalOverlap ||
                cleanTwoFrameSignalOverlap ||
                severeSignalCenterSoftOverlap ||
                highShapeRawRodSignalOverlap ||
                asymmetricRawPcaBridgeSoftOverlap ||
                crowdedGeneralCleanFuturePcaBridgeSoftOverlap) {
                softOverlapAcceptedForCost = true;
                if (cleanContinuationDaughterOverlap) {
                    cleanContinuationDaughterOverlapAcceptedForCost = true;
                }
                std::cout << "[Split Hard Overlap Bypass] "
                          << parentName
                          << " cellA=" << overlapA
                          << " cellB=" << overlapB
                          << " aInB=" << aInB
                          << " bInA=" << bInA
                          << " tolerance=" << hardOverlapTolerance
                          << " reason="
                          << (crowdedGeneralCleanFuturePcaBridgeSoftOverlap
                                  ? "general_clean_future_pca_bridge_overlap"
                              : (asymmetricRawPcaBridgeSoftOverlap
                                  ? "asymmetric_raw_pca_bridge_soft_overlap"
                              : (highShapeRawRodSignalOverlap
                                  ? "high_shape_raw_rod_signal_overlap"
                              : (severeSignalCenterSoftOverlap
                                  ? "celluniverse3_severe_signal_center_overlap"
                              : (cleanTwoFrameSignalOverlap
                                         ? "clean_two_frame_signal_overlap"
                              : (cleanExactFutureSignalOverlap
                                         ? "clean_exact_future_signal_overlap"
                                         : "clean_two_frame_rod_tip_continuation"))))))
                          << " bestIdx=" << bestIdx
                          << " bestLabel=" << bestLabel
                          << std::endl;
            } else {
            std::cout << "[Split Reject hard overlap] " << parentName
                      << " cellA=" << overlapA
                      << " cellB=" << overlapB
                      << " aInB=" << aInB
                      << " bInA=" << bInA
                      << " tolerance=" << hardOverlapTolerance
                      << " cleanFutureSupported="
                      << (cleanFutureSupportedSplitOverlap ? 1 : 0)
                      << " strongFutureSupported="
                      << (strongFutureSupportedSplitOverlap ? 1 : 0)
                      << " rawPcaOneFrameSupport="
                      << (oneFrameRawPcaBridgeOverlapSupport ? 1 : 0)
                      << " asymmetricRawPcaSupport="
                      << (asymmetricRawPcaBridgeOverlapSupport ? 1 : 0)
                      << " currentFallbackPcaSupport="
                      << (oneFrameCurrentFallbackPcaBridgeOverlapSupport ? 1 : 0)
                      << " oneFrameAlignedPcaSupport="
                      << (oneFrameAlignedFuturePcaBridgeOverlapSupport ? 1 : 0)
                      << " bridgeAxisPlaceSupport="
                      << (twoFrameBridgeAxisPlaceOverlapSupport ? 1 : 0)
                      << " oneFrameSignalOverlapSupport="
                      << (oneFrameSignalCenterOverlapSupport ? 1 : 0)
                      << " twoFrameSignalOverlapSupport="
                      << (twoFrameSignalCenterOverlapSupport ? 1 : 0)
                      << " severeSignalCenterOverlapSupport="
                      << (severeSignalCenterOverlapSupport ? 1 : 0)
                      << " delayedFutureOverlapSupport="
                      << (delayedFuturePcaBridgeOverlapSupport ? 1 : 0)
                      << " bestIdx=" << bestIdx
                      << " bestLabel=" << bestLabel
                      << std::endl;
            restoreLiveParent();
            return {0.0, noop};
            }
        } else if (!acceptedSiblingA.empty()) {
            softOverlapAcceptedForCost = true;
            std::cout << "[Split Soft Overlap Accepted] " << parentName
                      << " bestIdx=" << bestIdx
                      << " bestLabel=" << bestLabel
                      << " action=ignore_existing_or_sibling_overlap"
                      << std::endl;
        }
    }

    // --- 6. Cost check ---
    // costDiff is bbox-based when _useBboxCost is true (every cost
    // evaluation in burn-in / refine used the splitBbox + splitMask built
    // at the top of this function), or full-image L2 otherwise.
    // Both baseline and candidate were measured on the same voxel set,
    // so this is an apples-to-apples comparison.
    const double costDiff = bestTotal - baselineTotal;
    const double imageCostDiff = bestImageCost - baselineImageCost;
    const double overlapCostDiff = (bestTotal - bestImageCost) - (baselineTotal - baselineImageCost);

    // Adaptive split cost threshold: the larger of:
    // (1) the fixed split_cost from config
    // (2) split_cost_fraction × baselineImageCost (proportional to cell)
    // This prevents marginal splits on dim/small cells (low baseline cost)
    // from passing the fixed threshold while requiring the same fractional
    // improvement from bright/large cells.
    const bool useCellLumenCostGate = useCellLumenGateParams;
    const double activeSplitCost = useCellLumenCostGate
                                       ? static_cast<double>(std::max(0.0f, lumenSplitCost))
                                       : static_cast<double>(probConfig.split_cost);
    const double activeSplitCostFraction = useCellLumenCostGate
                                               ? static_cast<double>(std::max(0.0f, lumenSplitCostFraction))
                                               : static_cast<double>(probConfig.split_cost_fraction);
    const double adaptiveThreshold = std::max(
        activeSplitCost,
        activeSplitCostFraction * baselineImageCost);

    const bool lumenParentAnchoredProposal =
        bestIsCellLumenPrior && lumenProposal != nullptr &&
        lumenProposal->parentAnchored;
    auto isRealLumenCandidateId = [](int candidateId) {
        return candidateId >= 0 && candidateId < 1000000000;
    };
    const bool lumenParentAnchorOneRealCandidate =
        lumenParentAnchoredProposal &&
        lumenProposal != nullptr &&
        (isRealLumenCandidateId(lumenProposal->candidateIdA) !=
         isRealLumenCandidateId(lumenProposal->candidateIdB));
    const float parentAnchorRefitDriftLimit = std::max(
        lumenSnapshotSeedMaxRefitDrift >= 0.0f
            ? lumenSnapshotSeedMaxRefitDrift
            : 12.0f,
        seedAxisLen * 0.85f);
    const double parentAnchorWeakImageGain =
        std::max(80.0, 2.5 * static_cast<double>(
                           std::max(0.0f, lumenPositiveGateMinImageGain)));
    const bool parentAnchorCleanFutureDriftRescue =
        useCellLumenGateParams &&
        lumenParentAnchoredProposal &&
        lumenProposal != nullptr &&
        lumenProposal->windowBothDaughtersSupported >= 2 &&
        lumenProposal->windowMissingDaughterCount == 0 &&
        lumenProposal->windowParentPersists == 0 &&
        lumenProposal->neighborClaimPenalty <= 1e-5f &&
        lumenProposal->parentPersistencePenalty <= 1e-5f &&
        lumenProposal->elongation <= 0.0f &&
        imageCostDiff <= -30.0 &&
        overlapCostDiff <= baselineImageCost * 0.25 &&
        finalAxisLen >= seedAxisLen * 1.50f &&
        bridgeCostRescueValleyFromBright <= 0.65f &&
        lumenBridgeGapWidth >= 4.0f;
    if (useCellLumenGateParams &&
        lumenParentAnchoredProposal &&
        snapshotDriftMax > parentAnchorRefitDriftLimit &&
        costDiff > 0.0 &&
        imageCostDiff > -parentAnchorWeakImageGain &&
        !parentAnchorCleanFutureDriftRescue) {
        std::cout << "[Split Reject CellLumen parent anchor drift] " << parentName
                  << " drift1=" << drift1
                  << " drift2=" << drift2
                  << " maxRefitDrift=" << snapshotDriftMax
                  << " driftLimit=" << parentAnchorRefitDriftLimit
                  << " seedAxisLen=" << seedAxisLen
                  << " finalAxisLen=" << finalAxisLen
                  << " totalDiff=" << costDiff
                  << " imageDiff=" << imageCostDiff
                  << " overlapDiff=" << overlapCostDiff
                  << " weakImageGainLimit=" << parentAnchorWeakImageGain
                  << " d1=(" << bestD1Pos.x << "," << bestD1Pos.y << "," << bestD1Pos.z << ")"
                  << " d2=(" << bestD2Pos.x << "," << bestD2Pos.y << "," << bestD2Pos.z << ")"
                  << " bestIdx=" << bestIdx
                  << " bestLabel=" << bestLabel
                  << std::endl;
        restoreLiveParent();
        return {0.0, noop};
    }
    const bool parentAnchorWeakOverlapDuplicate =
        useCellLumenGateParams &&
        lumenParentAnchoredProposal &&
        costDiff > 0.0 &&
        imageCostDiff > -std::max(
                            30.0,
                            1.5 * static_cast<double>(
                                      std::max(
                                          0.0f,
                                          lumenPositiveGateMinImageGain))) &&
        overlapCostDiff > baselineImageCost * 0.70 &&
        lumenBridgeGapWidth < 0.0f &&
        bridgeCostRescueValleyFromBright > 0.55f &&
        snapshotDriftMax > std::max(8.0f, finalAxisLen * 0.50f);
    if (parentAnchorWeakOverlapDuplicate) {
        std::cout << "[Split Reject CellLumen parent anchor weak overlap duplicate] "
                  << parentName
                  << " drift1=" << drift1
                  << " drift2=" << drift2
                  << " maxRefitDrift=" << snapshotDriftMax
                  << " finalAxisLen=" << finalAxisLen
                  << " totalDiff=" << costDiff
                  << " imageDiff=" << imageCostDiff
                  << " overlapDiff=" << overlapCostDiff
                  << " overlapFraction="
                  << (baselineImageCost > 1e-9
                          ? overlapCostDiff / baselineImageCost
                          : 0.0)
                  << " bridgeGapWidth=" << lumenBridgeGapWidth
                  << " bridgeValleyFromBright="
                  << bridgeCostRescueValleyFromBright
                  << " priorScore="
                  << (lumenProposal != nullptr
                          ? lumenProposal->elongation
                          : 0.0f)
                  << " parentShapeElong="
                  << (lumenProposal != nullptr
                          ? lumenProposal->parentShapeElongation
                          : 1.0f)
                  << " bestIdx=" << bestIdx
                  << " bestLabel=" << bestLabel
                  << std::endl;
        restoreLiveParent();
        return {0.0, noop};
    }
    const bool parentAnchorWeakDriftBridgeDuplicate =
        useCellLumenGateParams &&
        lumenParentAnchorOneRealCandidate &&
        costDiff > 0.0 &&
        imageCostDiff > -60.0 &&
        overlapCostDiff > 0.0 &&
        lumenBridgeGapWidth < 0.0f &&
        bridgeCostRescueValleyFromBright > 0.75f &&
        snapshotDriftMax > std::max(8.0f, finalAxisLen * 0.45f) &&
        !parentAnchorCleanFutureDriftRescue;
    if (parentAnchorWeakDriftBridgeDuplicate) {
        std::cout << "[Split Reject CellLumen parent anchor weak drift duplicate] "
                  << parentName
                  << " drift1=" << drift1
                  << " drift2=" << drift2
                  << " maxRefitDrift=" << snapshotDriftMax
                  << " finalAxisLen=" << finalAxisLen
                  << " totalDiff=" << costDiff
                  << " imageDiff=" << imageCostDiff
                  << " overlapDiff=" << overlapCostDiff
                  << " bridgeGapWidth=" << lumenBridgeGapWidth
                  << " bridgeValleyFromBright="
                  << bridgeCostRescueValleyFromBright
                  << " priorScore="
                  << (lumenProposal != nullptr
                          ? lumenProposal->elongation
                          : 0.0f)
                  << " parentShapeElong="
                  << (lumenProposal != nullptr
                          ? lumenProposal->parentShapeElongation
                          : 1.0f)
                  << " bestIdx=" << bestIdx
                  << " bestLabel=" << bestLabel
                  << std::endl;
        restoreLiveParent();
        return {0.0, noop};
    }
    const bool parentAnchorLikelyUnclaimedBlob =
        useCellLumenGateParams &&
        lumenParentAnchorOneRealCandidate &&
        costDiff < 0.0 &&
        overlapCostDiff < -baselineImageCost * 0.10 &&
        imageCostDiff > -120.0 &&
        snapshotDriftMax > std::max(14.0f, seedAxisLen * 0.70f) &&
        finalAxisLen > seedAxisLen * 1.60f;
    const bool parentAnchorUnclaimedBlobWindowRescue =
        parentAnchorLikelyUnclaimedBlob &&
        lumenProposal != nullptr &&
        lumenProposal->windowBothDaughtersSupported >= 2 &&
        lumenProposal->windowMissingDaughterCount == 0 &&
        lumenProposal->windowParentPersists == 0 &&
        lumenProposal->neighborClaimPenalty <= 1e-5f &&
        lumenProposal->parentPersistencePenalty <= 1e-5f &&
        imageCostDiff <= -50.0 &&
        overlapCostDiff <= -baselineImageCost * 0.20 &&
        lumenBridgeGapWidth >= 8.0f &&
        bridgeCostRescueValleyFromBright <= probConfig.split_long_raw_pca_bridge_max_valley_from_bright;
    if (parentAnchorLikelyUnclaimedBlob &&
        !parentAnchorUnclaimedBlobWindowRescue) {
        std::cout << "[Split Reject CellLumen parent anchor unclaimed blob drift] "
                  << parentName
                  << " drift1=" << drift1
                  << " drift2=" << drift2
                  << " maxRefitDrift=" << snapshotDriftMax
                  << " seedAxisLen=" << seedAxisLen
                  << " finalAxisLen=" << finalAxisLen
                  << " totalDiff=" << costDiff
                  << " imageDiff=" << imageCostDiff
                  << " overlapDiff=" << overlapCostDiff
                  << " overlapFraction="
                  << (baselineImageCost > 1e-9
                          ? overlapCostDiff / baselineImageCost
                          : 0.0)
                  << " bridgeGapWidth=" << lumenBridgeGapWidth
                  << " bridgeValleyFromBright="
                  << bridgeCostRescueValleyFromBright
                  << " bestIdx=" << bestIdx
                  << " bestLabel=" << bestLabel
                  << std::endl;
        restoreLiveParent();
        return {0.0, noop};
    } else if (parentAnchorUnclaimedBlobWindowRescue) {
        std::cout << "[Split CellLumen Parent Anchor Unclaimed Blob Rescue] "
                  << parentName
                  << " drift1=" << drift1
                  << " drift2=" << drift2
                  << " maxRefitDrift=" << snapshotDriftMax
                  << " seedAxisLen=" << seedAxisLen
                  << " finalAxisLen=" << finalAxisLen
                  << " totalDiff=" << costDiff
                  << " imageDiff=" << imageCostDiff
                  << " overlapDiff=" << overlapCostDiff
                  << " bridgeGapWidth=" << lumenBridgeGapWidth
                  << " bridgeValleyFromBright="
                  << bridgeCostRescueValleyFromBright
                  << " windowBoth="
                  << lumenProposal->windowBothDaughtersSupported
                  << " bestIdx=" << bestIdx
                  << " bestLabel=" << bestLabel
                  << std::endl;
    }

    const float cleanPcaBridgeSnapDistanceLimit =
        std::max(
            std::max(0.0f, probConfig.pca_bridge_future_window_match_distance),
            0.75f * std::max(1.0f, srcMaxR));
    const float cleanPcaBridgeFuturePairSnapLimit =
        std::max(
            cleanPcaBridgeSnapDistanceLimit,
            std::max(
                0.0f,
                probConfig.pca_bridge_future_window_pca_snap_max_radius_scale) *
                std::max(1.0f, srcMaxR));
    const bool cleanPcaBridgeFutureGeometrySupport =
        bestIsPcaBridgeOnly &&
        bridgeProposal != nullptr &&
        bridgeProposal->futureWindowSplitRescue &&
        bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
        bridgeProposal->windowBothDaughtersSupported >=
            std::max(2, probConfig.pca_bridge_future_window_min_both_daughter_support) &&
        bridgeProposal->windowMissingDaughterCount == 0 &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->centerSnapMaxSeedDistance <=
            cleanPcaBridgeFuturePairSnapLimit &&
        bridgeProposal->parentDistanceBalance >= probConfig.split_clean_pca_bridge_future_geometry_min_parent_balance &&
        costDiff > 0.0 &&
        bridgeProposal->windowBestMatchedMinBrightness >=
            std::max(
                0.0f,
                probConfig
                    .pca_bridge_future_window_geometry_rescue_min_brightness);
    const bool cleanPcaBridgeFutureSoftSupport =
        bestIsPcaBridgeOnly &&
        bridgeProposal != nullptr &&
        bridgeProposal->futureWindowSplitRescue &&
        bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
        bridgeProposal->windowBothDaughtersSupported >=
            std::max(1, probConfig.pca_bridge_future_window_min_both_daughter_support) &&
        bridgeProposal->windowMissingDaughterCount <=
            std::max(0, probConfig.pca_bridge_future_window_max_missing_daughters) &&
        bridgeProposal->windowParentPersists <=
            std::max(0, probConfig.pca_bridge_future_window_max_parent_persists) &&
        !bridgeProposal->centerSnapUsedAlignedPairFallback &&
        bridgeProposal->centerSnapMaxSeedDistance <= cleanPcaBridgeSnapDistanceLimit &&
        bridgeProposal->parentShapeElongation >= probConfig.split_clean_pca_bridge_soft_min_parent_shape &&
        (bridgeProposal->parentShapeElongation >= probConfig.split_clean_pca_bridge_soft_strong_parent_shape ||
         bridgeProposal->windowBestMatchedMinBrightness >=
             std::max(probConfig.split_clean_pca_bridge_soft_min_brightness,
                      probConfig
                          .pca_bridge_future_window_rod_tip_balance_min_brightness)) &&
        bridgeCostRescueValleyFromBright <= probConfig.split_clean_pca_bridge_soft_max_valley_from_bright &&
        bridgeCostRescueGapDensity <= probConfig.split_long_raw_pca_bridge_max_gap_density &&
        imageCostDiff <= -std::max(static_cast<double>(probConfig.split_clean_pca_bridge_soft_image_gain_abs),
                                   static_cast<double>(probConfig.split_clean_pca_bridge_soft_image_gain_fraction) * baselineImageCost) &&
        costDiff <= std::max(static_cast<double>(probConfig.split_clean_pca_bridge_soft_cost_limit_abs),
                             static_cast<double>(probConfig.split_clean_pca_bridge_soft_cost_limit_fraction) * baselineImageCost);
    const float cellUniverse3WeakPcaBridgeSnapLimit = std::max(
        std::max(0.0f, probConfig.pca_bridge_future_window_match_distance),
        std::max(
            0.0f,
            probConfig
                .celluniverse3_weak_pca_bridge_cost_rescue_max_snap_radius_scale) *
            std::max(1.0f, srcMaxR));
    const double cellUniverse3WeakPcaBridgeCostLimit = std::max(
        static_cast<double>(
            std::max(
                0.0f,
                probConfig
                    .celluniverse3_weak_pca_bridge_cost_rescue_max_cost_abs)),
        static_cast<double>(
            std::max(
                0.0f,
                probConfig
                    .celluniverse3_weak_pca_bridge_cost_rescue_max_cost_fraction)) *
            baselineImageCost);
    const bool cellUniverse3WeakPcaBridgeImageCostOk =
        imageCostDiff <=
        static_cast<double>(
            std::max(
                0.0f,
                probConfig
                    .celluniverse3_weak_pca_bridge_cost_rescue_max_image_diff_abs));
    const bool cellUniverse3WeakPcaBridgeCostRescued =
        simulationConfig.celluniverse3_enabled &&
        probConfig.celluniverse3_weak_pca_bridge_cost_rescue_enabled &&
        bestIsPcaBridgeOnly &&
        bestLabel == "bridge_primary" &&
        bridgeProposal != nullptr &&
        bridgeProposal->centerSnapApplied &&
        bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
        bridgeProposal->windowBothDaughtersSupported >=
            std::max(
                1,
                probConfig
                    .celluniverse3_weak_pca_bridge_cost_rescue_min_future_both) &&
        bridgeProposal->windowMissingDaughterCount <=
            std::max(
                0,
                probConfig
                    .celluniverse3_weak_pca_bridge_cost_rescue_max_missing) &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->parentShapeElongation >=
            probConfig
                .celluniverse3_weak_pca_bridge_cost_rescue_min_parent_shape &&
        bridgeProposal->parentDistanceBalance >=
            probConfig
                .celluniverse3_weak_pca_bridge_cost_rescue_min_parent_balance &&
        bridgeProposal->windowBestMatchedMinBrightness >=
            probConfig
                .celluniverse3_weak_pca_bridge_cost_rescue_min_future_brightness &&
        bridgeProposal->centerSnapMaxSeedDistance <=
            cellUniverse3WeakPcaBridgeSnapLimit &&
        costDiff <= cellUniverse3WeakPcaBridgeCostLimit &&
        cellUniverse3WeakPcaBridgeImageCostOk &&
        bridgeCostRescueValleyFromBright <=
            probConfig
                .celluniverse3_weak_pca_bridge_cost_rescue_max_valley_from_bright &&
        bridgeCostRescueGapDensity <=
            probConfig
                .celluniverse3_weak_pca_bridge_cost_rescue_max_gap_density;
    const bool weakBridgeStartedWorse =
        bestIsPcaBridgeOnly &&
        bridgeProposal != nullptr &&
        preCostDiff > 0.0 &&
        bridgeCostRescueValleyFromBright > probConfig.split_weak_bridge_started_max_valley_from_bright &&
        bridgeCostRescueGapDensity > probConfig.split_weak_bridge_started_min_gap_density &&
        bridgeProposal->parentShapeElongation <
            probConfig.pca_bridge_future_window_min_parent_shape_for_cost_rescue &&
        !cleanPcaBridgeFutureSoftSupport &&
        !cleanPcaBridgeFutureGeometrySupport &&
        !cellUniverse3WeakPcaBridgeCostRescued;
    if (weakBridgeStartedWorse) {
        std::cout << "[Split Reject weak pca bridge] " << parentName
                  << " preCostDiff=" << preCostDiff
                  << " finalCostDiff=" << costDiff
                  << " valleyFromBright=" << bridgeCostRescueValleyFromBright
                  << " gapDensity=" << bridgeCostRescueGapDensity
                  << " parentShapeElong="
                  << bridgeProposal->parentShapeElongation
                  << " minStrongParentShape="
                  << probConfig.pca_bridge_future_window_min_parent_shape_for_cost_rescue
                  << " futureBoth="
                  << bridgeProposal->windowBothDaughtersSupported
                  << " futureImmediate="
                  << bridgeProposal->windowImmediateBothDaughtersSupported
                  << " futureMissing="
                  << bridgeProposal->windowMissingDaughterCount
                  << " centerSnapMaxSeedDistance="
                  << bridgeProposal->centerSnapMaxSeedDistance
                  << " cleanSnapDistanceLimit="
                  << cleanPcaBridgeSnapDistanceLimit
                  << " alignedPairFallback="
                  << (bridgeProposal->centerSnapUsedAlignedPairFallback ? 1 : 0)
                  << " cellUniverse3WeakPcaBridgeCostRescue="
                  << (cellUniverse3WeakPcaBridgeCostRescued ? 1 : 0)
                  << " cellUniverse3WeakPcaBridgeCostLimit="
                  << cellUniverse3WeakPcaBridgeCostLimit
                  << " cellUniverse3WeakPcaBridgeSnapLimit="
                  << cellUniverse3WeakPcaBridgeSnapLimit
                  << " bestIdx=" << bestIdx
                  << " bestLabel=" << bestLabel
                  << std::endl;
        restoreLiveParent();
        return {0.0, noop};
    }
    const bool lowShapeDelayedAlignedPcaBridge =
        bestIsPcaBridgeOnly &&
        bridgeProposal != nullptr &&
        bridgeProposal->centerSnapUsedAlignedPairFallback &&
        bridgeProposal->windowImmediateBothDaughtersSupported == 0 &&
        (bridgeProposal->windowBothDaughtersSupported <
             std::max(1, probConfig.split_low_shape_delayed_aligned_pca_bridge_min_future_both) ||
         bridgeProposal->windowMissingDaughterCount >
             std::max(0, probConfig.split_low_shape_delayed_aligned_pca_bridge_max_missing)) &&
        bridgeProposal->windowParentPersists <=
            std::max(0, probConfig.pca_bridge_future_window_max_parent_persists) &&
        bridgeProposal->parentShapeElongation <
            probConfig.split_low_shape_delayed_aligned_pca_bridge_min_parent_shape;
    if (lowShapeDelayedAlignedPcaBridge) {
        std::cout << "[Split Reject weak pca bridge] " << parentName
                  << " reason=low_shape_delayed_aligned_pca_bridge"
                  << " parentShapeElong="
                  << bridgeProposal->parentShapeElongation
                  << " minParentShape="
                  << probConfig.split_low_shape_delayed_aligned_pca_bridge_min_parent_shape
                  << " futureBoth="
                  << bridgeProposal->windowBothDaughtersSupported
                  << " minFutureBoth="
                  << probConfig.split_low_shape_delayed_aligned_pca_bridge_min_future_both
                  << " futureImmediate="
                  << bridgeProposal->windowImmediateBothDaughtersSupported
                  << " futureMissing="
                  << bridgeProposal->windowMissingDaughterCount
                  << " maxMissing="
                  << probConfig.split_low_shape_delayed_aligned_pca_bridge_max_missing
                  << " centerSnapMaxSeedDistance="
                  << bridgeProposal->centerSnapMaxSeedDistance
                  << " cleanSnapDistanceLimit="
                  << cleanPcaBridgeSnapDistanceLimit
                  << " parentDistBalance="
                  << bridgeProposal->parentDistanceBalance
                  << " bestIdx=" << bestIdx
                  << " bestLabel=" << bestLabel
                  << std::endl;
        restoreLiveParent();
        return {0.0, noop};
    }

    const double bridgeRescueLimit =
        static_cast<double>(std::max(
            0.0f, probConfig.split_bridge_cost_rescue_max_positive_fraction)) *
        baselineImageCost;
    const double geometryAdjustedCostDiff =
        costDiff + splitSoftGeometryPenaltyCost;
    const bool signalLikeBridgeCostRescueAllowed =
        !bestIsSignalCenterProposal ||
        (costDiff <= -adaptiveThreshold &&
         imageCostDiff <= -std::max(static_cast<double>(probConfig.split_signal_like_bridge_image_gain_abs),
                                   static_cast<double>(probConfig.split_signal_like_bridge_image_gain_fraction) * baselineImageCost));
    const bool bridgeCostRescued =
        probConfig.split_bridge_cost_rescue_enabled &&
        bridgeCostRescueEligible &&
        signalLikeBridgeCostRescueAllowed &&
        geometryAdjustedCostDiff >= -adaptiveThreshold &&
        geometryAdjustedCostDiff <= bridgeRescueLimit;
    const bool futureWindowStrictCostRescued =
        bestIsPcaBridgeOnly &&
        bridgeProposal != nullptr &&
        bridgeProposal->futureWindowSplitRescue &&
        bridgeCostRescueEligible &&
        bridgeProposal->parentShapeElongation >=
            probConfig.pca_bridge_future_window_min_parent_shape_for_cost_rescue &&
        geometryAdjustedCostDiff <= bridgeRescueLimit;
    const bool futureWindowSoftCostRescued =
        cleanPcaBridgeFutureSoftSupport &&
        costDiff <= std::max(static_cast<double>(probConfig.split_clean_pca_bridge_soft_cost_limit_abs),
                             static_cast<double>(probConfig.split_clean_pca_bridge_soft_cost_limit_fraction) * baselineImageCost);
    const bool futureWindowStrongCostRescued =
        bridgeProposal != nullptr &&
        bestHasCleanFutureBridgeSupport &&
        bridgeProposal->parentDistanceBalance >= probConfig.split_future_strong_bridge_min_parent_balance &&
        bridgeProposal->parentShapeElongation >=
            probConfig.pca_bridge_future_window_min_parent_shape_for_cost_rescue &&
        bridgeProposal->centerSnapMaxSeedDistance <=
            std::max(cleanPcaBridgeSnapDistanceLimit * probConfig.split_future_strong_bridge_snap_scale,
                     probConfig.pca_bridge_future_window_match_distance +
                         probConfig.pca_bridge_future_window_match_distance_per_frame) &&
        imageCostDiff <= -std::max(static_cast<double>(probConfig.split_future_strong_bridge_image_gain_abs),
                                   static_cast<double>(probConfig.split_future_strong_bridge_image_gain_fraction) * baselineImageCost) &&
        overlapCostDiff <= std::max(static_cast<double>(probConfig.split_future_strong_bridge_overlap_limit_abs),
                                    static_cast<double>(probConfig.split_future_strong_bridge_overlap_limit_fraction) * baselineImageCost) &&
        bridgeCostRescueValleyFromBright <= probConfig.split_future_strong_bridge_max_valley_from_bright &&
        bridgeCostRescueGapDensity <= probConfig.split_future_strong_bridge_max_gap_density &&
        costDiff <= bridgeRescueLimit &&
        geometryAdjustedCostDiff <= bridgeRescueLimit;
    const bool futureWindowGeometryCostRescued =
        cleanPcaBridgeFutureGeometrySupport &&
        bridgeProposal->parentShapeElongation >=
            std::max(
                0.0f,
                probConfig
                    .pca_bridge_future_window_geometry_rescue_min_parent_shape) &&
        imageCostDiff <= -std::max(static_cast<double>(probConfig.split_future_geometry_bridge_image_gain_abs),
                                   static_cast<double>(probConfig.split_future_geometry_bridge_image_gain_fraction) * baselineImageCost) &&
        overlapCostDiff <= static_cast<double>(probConfig.split_future_geometry_bridge_overlap_limit_fraction) * baselineImageCost &&
        bridgeCostRescueValleyFromBright <= probConfig.split_future_geometry_bridge_max_valley_from_bright &&
        bridgeCostRescueGapDensity <= probConfig.split_future_geometry_bridge_max_gap_density &&
        costDiff <= std::max(static_cast<double>(probConfig.split_future_geometry_bridge_cost_limit_abs),
                             static_cast<double>(probConfig.split_future_geometry_bridge_cost_limit_fraction) * baselineImageCost) &&
        geometryAdjustedCostDiff <= std::max(static_cast<double>(probConfig.split_future_geometry_bridge_cost_limit_abs),
                                             static_cast<double>(probConfig.split_future_geometry_bridge_cost_limit_fraction) * baselineImageCost);
    const bool twoFrameFutureBridgeCostRescued =
        bestIsPcaBridgeOnly &&
        bridgeProposal != nullptr &&
        bestHasCleanFutureBridgeSupport &&
        bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
        bridgeProposal->windowBothDaughtersSupported >=
            std::max(2, probConfig.pca_bridge_future_window_min_both_daughter_support) &&
        bridgeProposal->windowMissingDaughterCount == 0 &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->windowBestMatchedMinBrightness >= probConfig.split_two_frame_future_bridge_min_brightness &&
        bridgeProposal->parentDistanceBalance >= probConfig.split_two_frame_future_bridge_min_parent_balance &&
        bridgeProposal->centerSnapMaxSeedDistance <=
            std::max(cleanPcaBridgeFuturePairSnapLimit,
                     probConfig.split_two_frame_future_bridge_snap_scale * std::max(1.0f, srcMaxR)) &&
        imageCostDiff <= -static_cast<double>(probConfig.split_two_frame_future_bridge_image_gain_abs) &&
        overlapCostDiff <= std::max(static_cast<double>(probConfig.split_two_frame_future_bridge_overlap_limit_abs),
                                    static_cast<double>(probConfig.split_two_frame_future_bridge_overlap_limit_fraction) * baselineImageCost) &&
        costDiff <= std::max(static_cast<double>(probConfig.split_two_frame_future_bridge_cost_limit_abs),
                             static_cast<double>(probConfig.split_two_frame_future_bridge_cost_limit_fraction) * baselineImageCost) &&
        geometryAdjustedCostDiff <= std::max(static_cast<double>(probConfig.split_two_frame_future_bridge_geometry_limit_abs),
                                             static_cast<double>(probConfig.split_two_frame_future_bridge_geometry_limit_fraction) * baselineImageCost) &&
        bridgeCostRescueValleyFromBright <= probConfig.split_two_frame_future_bridge_max_valley_from_bright &&
        bridgeCostRescueGapDensity <= probConfig.split_two_frame_future_bridge_max_gap_density &&
        finalAxisLen >= probConfig.split_two_frame_future_bridge_min_axis_length_scale * std::max(1.0f, srcMaxR);
    const double futureSoftPenaltyRawGainLimit =
        std::max(bridgeRescueLimit, static_cast<double>(probConfig.split_future_soft_penalty_raw_gain_fraction) * baselineImageCost);
    const bool futureWindowSoftPenaltyRawGainRescued =
        bestIsPcaBridgeOnly &&
        bridgeProposal != nullptr &&
        bestHasCleanFutureBridgeSupport &&
        (!bridgeProposal->centerSnapUsedAlignedPairFallback ||
         probConfig
             .split_future_soft_penalty_raw_gain_allow_aligned_pair_fallback) &&
        bridgeProposal->parentDistanceBalance >=
            probConfig.split_future_soft_penalty_raw_gain_min_parent_balance &&
        splitSoftGeometryPenaltyCost > 0.0 &&
        costDiff <= -adaptiveThreshold &&
        geometryAdjustedCostDiff <= futureSoftPenaltyRawGainLimit &&
        bridgeCostRescueGapDensity <=
            probConfig.split_future_soft_penalty_raw_gain_max_gap_density &&
        bridgeCostRescueValleyFromBright <=
            probConfig.split_future_soft_penalty_raw_gain_max_valley_from_bright;
    const bool oneFrameBrightPcaBridgeSoftPenaltyRescued =
        bestIsPcaBridgeOnly &&
        bridgeProposal != nullptr &&
        bridgeProposal->bioSeparationSoftRescued &&
        !bridgeProposal->centerSnapUsedAlignedPairFallback &&
        bridgeProposal->windowBothDaughtersSupported >= 1 &&
        bridgeProposal->windowMissingDaughterCount <= 2 &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->windowBestMatchedMinBrightness >=
            probConfig.split_one_frame_bright_pca_bridge_soft_min_brightness &&
        bridgeProposal->parentShapeElongation >=
            probConfig.split_one_frame_bright_pca_bridge_soft_min_parent_shape &&
        bridgeProposal->parentDistanceBalance >=
            probConfig.split_one_frame_bright_pca_bridge_soft_min_parent_balance &&
        splitSoftGeometryPenaltyCost > 0.0 &&
        splitSoftGeometryPenaltyCost <=
            static_cast<double>(
                probConfig.split_one_frame_bright_pca_bridge_soft_penalty_fraction) *
                baselineImageCost &&
        imageCostDiff <= -adaptiveThreshold &&
        overlapCostDiff <=
            static_cast<double>(
                probConfig
                    .split_one_frame_bright_pca_bridge_soft_overlap_limit_fraction) *
                baselineImageCost &&
        geometryAdjustedCostDiff <=
            std::max(static_cast<double>(
                         probConfig
                             .split_one_frame_bright_pca_bridge_soft_geometry_limit_abs),
                     static_cast<double>(
                         probConfig
                             .split_one_frame_bright_pca_bridge_soft_geometry_limit_fraction) *
                         baselineImageCost) &&
        bridgeCostRescueGapDensity <=
            probConfig.split_one_frame_bright_pca_bridge_soft_max_gap_density &&
        bridgeCostRescueValleyFromBright <=
            probConfig
                .split_one_frame_bright_pca_bridge_soft_max_valley_from_bright &&
        finalAxisLen >=
            probConfig
                .split_one_frame_bright_pca_bridge_soft_min_axis_length_scale *
            std::max(1.0f, srcMaxR);
    const float maxDaughterSeedDrift = std::max(drift1, drift2);
    const bool lockedCleanFuturePcaBridgeCostRescued =
        bestIsPcaBridgeOnly &&
        bridgeProposal != nullptr &&
        bridgeProposal->futureWindowSplitRescue &&
        bridgeProposal->centerSnapUsedAlignedPairFallback &&
        bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
        bridgeProposal->windowBothDaughtersSupported >=
            std::max(2, probConfig.pca_bridge_future_window_min_both_daughter_support) &&
        bridgeProposal->windowMissingDaughterCount == 0 &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->windowBestMatchedMinBrightness >=
            probConfig.split_immediate_pca_continuation_min_brightness &&
        bridgeProposal->parentShapeElongation >=
            probConfig.split_immediate_pca_continuation_min_parent_shape &&
        bridgeProposal->parentDistanceBalance >=
            probConfig.split_immediate_pca_continuation_min_parent_balance &&
        bridgeProposal->parentDistanceBalance <=
            probConfig.split_immediate_pca_continuation_max_parent_balance &&
        bridgeProposal->centerSnapMaxSeedDistance <=
            std::max(cleanPcaBridgeFuturePairSnapLimit,
                     probConfig.split_locked_clean_future_pca_bridge_snap_scale *
                         std::max(1.0f, srcMaxR)) &&
        maxDaughterSeedDrift <=
            std::max(probConfig.split_locked_clean_future_pca_bridge_max_drift_abs,
                     probConfig.split_locked_clean_future_pca_bridge_max_drift_scale *
                         std::max(1.0f, srcMaxR)) &&
        costDiff <=
            std::max(static_cast<double>(
                         probConfig
                             .split_locked_clean_future_pca_bridge_cost_limit_abs),
                     static_cast<double>(
                         probConfig
                             .split_locked_clean_future_pca_bridge_cost_limit_fraction) *
                         baselineImageCost) &&
        geometryAdjustedCostDiff <=
            std::max(static_cast<double>(
                         probConfig
                             .split_locked_clean_future_pca_bridge_geometry_limit_abs),
                     static_cast<double>(
                         probConfig
                             .split_locked_clean_future_pca_bridge_geometry_limit_fraction) *
                         baselineImageCost) &&
        overlapCostDiff <=
            std::max(static_cast<double>(
                         probConfig
                             .split_locked_clean_future_pca_bridge_overlap_limit_abs),
                     static_cast<double>(
                         probConfig
                             .split_locked_clean_future_pca_bridge_overlap_limit_fraction) *
                         baselineImageCost) &&
        bridgeCostRescueGapDensity <=
            probConfig.split_locked_clean_future_pca_bridge_max_gap_density &&
        bridgeCostRescueValleyFromBright <=
            probConfig.split_locked_clean_future_pca_bridge_max_valley_from_bright &&
        finalAxisLen >=
            probConfig.split_locked_clean_future_pca_bridge_min_axis_length_scale *
            std::max(1.0f, srcMaxR);
    const float generalCleanFutureCostMinBrightness = std::min(
        probConfig.split_current_locked_bridge_min_brightness,
        std::min(
            probConfig.split_one_frame_aligned_pca_continuation_min_brightness,
            probConfig.split_exact_future_center_bridge_min_brightness));
    const float generalCleanFutureCostMinParentShape = std::min(
        probConfig.split_current_locked_bridge_min_parent_shape,
        std::min(
            probConfig.split_one_frame_aligned_pca_continuation_min_parent_shape,
            probConfig.split_exact_future_center_bridge_min_parent_shape));
    const float generalCleanFutureCostMinParentBalance = std::min(
        probConfig.split_current_locked_bridge_min_parent_balance,
        std::min(
            probConfig.split_one_frame_aligned_pca_continuation_min_parent_balance,
            probConfig.split_exact_future_center_bridge_min_parent_balance));
    const float generalCleanFutureCostMaxParentBalance = std::max(
        probConfig.split_current_locked_bridge_max_parent_balance,
        probConfig.split_one_frame_aligned_pca_continuation_max_parent_balance);
    const double generalCleanFutureImageGainRequired = std::max(
        static_cast<double>(
            probConfig.split_general_clean_future_pca_bridge_image_gain_abs),
        static_cast<double>(
            probConfig.split_general_clean_future_pca_bridge_image_gain_fraction) *
            baselineImageCost);
    const bool crowdedGeneralCleanFuturePcaBridgeSupport =
        bestIsPcaBridgeOnly &&
        bestLabel == "bridge_primary" &&
        bridgeProposal != nullptr &&
        probConfig.pca_bridge_future_window_enabled &&
        bridgeProposal->centerSnapApplied &&
        (bridgeProposal->immediateFutureCenterBacked ||
         bridgeProposal->centerSnapUsedAlignedPairFallback) &&
        bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
        bridgeProposal->windowBothDaughtersSupported >= 1 &&
        bridgeProposal->windowMissingDaughterCount == 0 &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->windowBestMatchedMinBrightness >=
            generalCleanFutureCostMinBrightness &&
        bridgeProposal->parentShapeElongation >=
            generalCleanFutureCostMinParentShape &&
        bridgeProposal->parentDistanceBalance >=
            generalCleanFutureCostMinParentBalance &&
        bridgeProposal->parentDistanceBalance <=
            generalCleanFutureCostMaxParentBalance &&
        bridgeProposal->centerSnapMaxSeedDistance <=
            std::max(cleanPcaBridgeFuturePairSnapLimit,
                     probConfig.split_locked_clean_future_pca_bridge_snap_scale *
                         std::max(1.0f, srcMaxR)) &&
        maxDaughterSeedDrift <=
            std::max(probConfig.split_locked_clean_future_pca_bridge_max_drift_abs,
                     probConfig.split_locked_clean_future_pca_bridge_max_drift_scale *
                         std::max(1.0f, srcMaxR)) &&
        savedNonTrashCellCount >=
            probConfig
                .split_general_clean_future_pca_bridge_crowded_cell_count_min;
    const float generalCleanFutureCrowdingAlpha =
        crowdedGeneralCleanFuturePcaBridgeSupport
            ? std::clamp(
                  static_cast<float>(
                      savedNonTrashCellCount -
                      probConfig
                          .split_general_clean_future_pca_bridge_crowded_cell_count_min) /
                      std::max(
                          1,
                          probConfig
                              .split_general_clean_future_pca_bridge_crowding_span),
                  0.0f, 1.0f)
            : 0.0f;
    const double generalCleanFutureCrowdingRelaxation =
        1.0 +
        static_cast<double>(generalCleanFutureCrowdingAlpha) *
            static_cast<double>(
                std::max(
                    1.0f,
                    probConfig
                        .split_general_clean_future_pca_bridge_max_relaxation) -
                1.0f);
    const double crowdedGeneralCleanFuturePcaBridgeCostLimit =
        std::max(
            static_cast<double>(
                probConfig.split_locked_clean_future_pca_bridge_cost_limit_abs),
            static_cast<double>(
                probConfig
                    .split_locked_clean_future_pca_bridge_cost_limit_fraction) *
                baselineImageCost) *
        generalCleanFutureCrowdingRelaxation;
    const double crowdedGeneralCleanFuturePcaBridgeGeometryLimit =
        std::max(
            static_cast<double>(
                probConfig
                    .split_locked_clean_future_pca_bridge_geometry_limit_abs),
            static_cast<double>(
                probConfig
                    .split_locked_clean_future_pca_bridge_geometry_limit_fraction) *
                baselineImageCost) *
        generalCleanFutureCrowdingRelaxation;
    const double crowdedGeneralCleanFuturePcaBridgeOverlapLimit =
        std::max(
            static_cast<double>(
                probConfig
                    .split_locked_clean_future_pca_bridge_overlap_limit_abs),
            static_cast<double>(
                probConfig
                    .split_locked_clean_future_pca_bridge_overlap_limit_fraction) *
                baselineImageCost) *
        generalCleanFutureCrowdingRelaxation;
    const bool crowdedGeneralCleanFuturePcaBridgeCostRescued =
        crowdedGeneralCleanFuturePcaBridgeSupport &&
        imageCostDiff <= -generalCleanFutureImageGainRequired &&
        costDiff <= crowdedGeneralCleanFuturePcaBridgeCostLimit &&
        geometryAdjustedCostDiff <=
            crowdedGeneralCleanFuturePcaBridgeGeometryLimit &&
        overlapCostDiff <= crowdedGeneralCleanFuturePcaBridgeOverlapLimit &&
        bridgeCostRescueGapDensity <=
            probConfig.split_locked_clean_future_pca_bridge_max_gap_density &&
        bridgeCostRescueValleyFromBright <=
            probConfig.split_locked_clean_future_pca_bridge_max_valley_from_bright &&
        finalAxisLen >=
            probConfig.split_locked_clean_future_pca_bridge_min_axis_length_scale *
            std::max(1.0f, srcMaxR);
    const double oneFrameAlignedLockedPcaBridgeCostLimit =
        std::max(static_cast<double>(
                     probConfig
                         .split_one_frame_aligned_locked_pca_bridge_cost_limit_abs),
                 static_cast<double>(
                     probConfig
                         .split_one_frame_aligned_locked_pca_bridge_cost_limit_fraction) *
                     baselineImageCost);
    const double oneFrameAlignedLockedPcaBridgeOverlapLimit =
        std::max(static_cast<double>(
                     probConfig
                         .split_one_frame_aligned_locked_pca_bridge_overlap_limit_abs),
                 static_cast<double>(
                     probConfig
                         .split_one_frame_aligned_locked_pca_bridge_overlap_limit_fraction) *
                     baselineImageCost);
    const double oneFrameAlignedLockedPcaBridgeImageGainRequired =
        std::max(static_cast<double>(
                     probConfig
                         .split_one_frame_aligned_locked_pca_bridge_image_gain_abs),
                 static_cast<double>(
                     probConfig
                         .split_one_frame_aligned_locked_pca_bridge_image_gain_fraction) *
                     baselineImageCost);
    const bool oneFrameAlignedLockedFuturePcaBridgeCostRescued =
        bestIsPcaBridgeOnly &&
        bestLabel == "bridge_primary" &&
        bridgeProposal != nullptr &&
        softOverlapAcceptedForCost &&
        bridgeProposal->centerSnapApplied &&
        bridgeProposal->centerSnapUsedAlignedPairFallback &&
        bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
        bridgeProposal->windowBothDaughtersSupported >= 1 &&
        bridgeProposal->windowMissingDaughterCount == 0 &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->windowBestMatchedMinBrightness >=
            probConfig.split_one_frame_aligned_pca_continuation_min_brightness &&
        bridgeProposal->parentShapeElongation >=
            probConfig.split_one_frame_aligned_pca_continuation_min_parent_shape &&
        bridgeProposal->parentDistanceBalance >=
            probConfig.split_one_frame_aligned_pca_continuation_min_parent_balance &&
        bridgeProposal->parentDistanceBalance <=
            probConfig.split_one_frame_aligned_pca_continuation_max_parent_balance &&
        bridgeProposal->centerSnapMaxSeedDistance <=
            std::max(cleanPcaBridgeFuturePairSnapLimit,
                     probConfig
                             .split_one_frame_aligned_pca_continuation_snap_scale *
                         std::max(1.0f, srcMaxR)) &&
        maxDaughterSeedDrift <=
            std::max(
                probConfig
                    .split_one_frame_aligned_locked_pca_bridge_max_drift_abs,
                probConfig
                    .split_one_frame_aligned_locked_pca_bridge_max_drift_scale *
                    std::max(1.0f, srcMaxR)) &&
        imageCostDiff <= -oneFrameAlignedLockedPcaBridgeImageGainRequired &&
        overlapCostDiff <= oneFrameAlignedLockedPcaBridgeOverlapLimit &&
        costDiff <= oneFrameAlignedLockedPcaBridgeCostLimit &&
        bridgeCostRescueGapDensity <=
            probConfig.split_one_frame_aligned_locked_pca_bridge_max_gap_density &&
        bridgeCostRescueValleyFromBright <=
            probConfig
                .split_one_frame_aligned_locked_pca_bridge_max_valley_from_bright &&
        finalAxisLen >=
            probConfig
                .split_one_frame_aligned_locked_pca_bridge_min_axis_length_scale *
            std::max(1.0f, srcMaxR);
    const double twoFrameAlignedLockedPcaBridgeCostLimit =
        std::max(static_cast<double>(
                     probConfig
                         .split_two_frame_aligned_locked_pca_bridge_cost_limit_abs),
                 static_cast<double>(
                     probConfig
                         .split_two_frame_aligned_locked_pca_bridge_cost_limit_fraction) *
                     baselineImageCost);
    const double twoFrameAlignedLockedPcaBridgeOverlapLimit =
        std::max(static_cast<double>(
                     probConfig
                         .split_two_frame_aligned_locked_pca_bridge_overlap_limit_abs),
                 static_cast<double>(
                     probConfig
                         .split_two_frame_aligned_locked_pca_bridge_overlap_limit_fraction) *
                     baselineImageCost);
    const double twoFrameAlignedLockedPcaBridgeImageGainRequired =
        std::max(static_cast<double>(
                     probConfig
                         .split_two_frame_aligned_locked_pca_bridge_image_gain_abs),
                 static_cast<double>(
                     probConfig
                         .split_two_frame_aligned_locked_pca_bridge_image_gain_fraction) *
                     baselineImageCost);
    const bool twoFrameAlignedLockedFuturePcaBridgeCostRescued =
        bestIsPcaBridgeOnly &&
        bestLabel == "bridge_primary" &&
        bridgeProposal != nullptr &&
        softOverlapAcceptedForCost &&
        probConfig.pca_bridge_future_window_enabled &&
        bridgeProposal->centerSnapApplied &&
        bridgeProposal->centerSnapUsedAlignedPairFallback &&
        bridgeProposal->windowBothDaughtersSupported >=
            std::max(2, probConfig.pca_bridge_future_window_min_both_daughter_support) &&
        bridgeProposal->windowMissingDaughterCount == 0 &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->windowBestMatchedMinBrightness >=
            probConfig.split_two_frame_aligned_locked_pca_bridge_min_brightness &&
        bridgeProposal->parentShapeElongation >=
            probConfig.split_two_frame_aligned_locked_pca_bridge_min_parent_shape &&
        bridgeProposal->parentDistanceBalance >=
            probConfig
                .split_two_frame_aligned_locked_pca_bridge_min_parent_balance &&
        bridgeProposal->parentDistanceBalance <=
            probConfig
                .split_two_frame_aligned_locked_pca_bridge_max_parent_balance &&
        bridgeProposal->centerSnapMaxSeedDistance <=
            std::max(cleanPcaBridgeFuturePairSnapLimit,
                     probConfig
                             .split_one_frame_aligned_pca_continuation_snap_scale *
                         std::max(1.0f, srcMaxR)) &&
        maxDaughterSeedDrift <=
            std::max(
                probConfig
                    .split_two_frame_aligned_locked_pca_bridge_max_drift_abs,
                probConfig
                    .split_two_frame_aligned_locked_pca_bridge_max_drift_scale *
                    std::max(1.0f, srcMaxR)) &&
        imageCostDiff <= -twoFrameAlignedLockedPcaBridgeImageGainRequired &&
        overlapCostDiff <= twoFrameAlignedLockedPcaBridgeOverlapLimit &&
        costDiff <= twoFrameAlignedLockedPcaBridgeCostLimit &&
        bridgeCostRescueGapDensity <=
            probConfig.split_two_frame_aligned_locked_pca_bridge_max_gap_density &&
        bridgeCostRescueValleyFromBright <=
            probConfig
                .split_two_frame_aligned_locked_pca_bridge_max_valley_from_bright &&
        finalAxisLen >=
            probConfig
                .split_two_frame_aligned_locked_pca_bridge_min_axis_length_scale *
            std::max(1.0f, srcMaxR);
    const bool lockedExactFutureCenterBridgeCostRescued =
        bridgeProposalOnly &&
        bestLabel == "bridge_primary" &&
        bridgeProposal != nullptr &&
        bridgeProposal->centerSnapApplied &&
        bridgeProposal->immediateFutureCenterBacked &&
        !bridgeProposal->centerSnapUsedAlignedPairFallback &&
        bridgeProposal->centerSnapMaxSeedDistance <=
            std::max(probConfig.split_exact_future_center_bridge_snap_abs,
                     probConfig.split_exact_future_center_bridge_snap_scale *
                         std::max(1.0f, srcMaxR)) &&
        bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
        bridgeProposal->windowBothDaughtersSupported >=
            std::max(2, probConfig.pca_bridge_future_window_min_both_daughter_support) &&
        bridgeProposal->windowMissingDaughterCount == 0 &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->windowBestMatchedMinBrightness >=
            probConfig.split_exact_future_center_bridge_min_brightness &&
        bridgeProposal->parentShapeElongation >=
            probConfig.split_exact_future_center_bridge_min_parent_shape &&
        bridgeProposal->parentDistanceBalance >=
            probConfig.split_locked_exact_future_bridge_min_parent_balance &&
        maxDaughterSeedDrift <=
            std::max(probConfig.split_locked_exact_future_bridge_max_drift_abs,
                     probConfig.split_locked_exact_future_bridge_max_drift_scale *
                         std::max(1.0f, srcMaxR)) &&
        costDiff <=
            std::max(static_cast<double>(
                         probConfig.split_locked_exact_future_bridge_cost_limit_abs),
                     static_cast<double>(
                         probConfig
                             .split_locked_exact_future_bridge_cost_limit_fraction) *
                         baselineImageCost) &&
        geometryAdjustedCostDiff <=
            std::max(static_cast<double>(
                         probConfig
                             .split_locked_exact_future_bridge_geometry_limit_abs),
                     static_cast<double>(
                         probConfig
                             .split_locked_exact_future_bridge_geometry_limit_fraction) *
                         baselineImageCost) &&
        overlapCostDiff <=
            std::max(static_cast<double>(
                         probConfig
                             .split_locked_exact_future_bridge_overlap_limit_abs),
                     static_cast<double>(
                         probConfig
                             .split_locked_exact_future_bridge_overlap_limit_fraction) *
                         baselineImageCost) &&
        bridgeCostRescueGapDensity <=
            probConfig.split_locked_exact_future_bridge_max_gap_density &&
        bridgeCostRescueValleyFromBright <=
            probConfig.split_locked_exact_future_bridge_max_valley_from_bright &&
        finalAxisLen >=
            probConfig.split_locked_exact_future_bridge_min_axis_length_scale *
            std::max(1.0f, srcMaxR);
    const bool oneFrameFuturePcaBridgeSoftPenaltyRawGainRescued =
        bestIsPcaBridgeOnly &&
        bridgeProposal != nullptr &&
        bridgeProposal->bioSeparationSoftRescued &&
        !bridgeProposal->centerSnapUsedAlignedPairFallback &&
        bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
        bridgeProposal->windowBothDaughtersSupported >=
            static_cast<int>(probConfig.split_future_near_dim_bypass_min_future_both) &&
        bridgeProposal->windowMissingDaughterCount <=
            static_cast<int>(probConfig.split_future_near_dim_bypass_max_missing_daughters) &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->windowBestMatchedMinBrightness >=
            probConfig.split_one_frame_future_pca_soft_min_brightness &&
        bridgeProposal->parentShapeElongation >=
            probConfig.split_one_frame_future_pca_soft_min_parent_shape &&
        bridgeProposal->parentDistanceBalance >=
            probConfig.split_one_frame_future_pca_soft_min_parent_balance &&
        splitSoftGeometryPenaltyCost > 0.0 &&
        costDiff <= -adaptiveThreshold &&
        imageCostDiff <= -adaptiveThreshold &&
        overlapCostDiff <=
            std::max(static_cast<double>(
                         probConfig.split_one_frame_future_pca_soft_overlap_limit_abs),
                     static_cast<double>(
                         probConfig
                             .split_one_frame_future_pca_soft_overlap_limit_fraction) *
                         baselineImageCost) &&
        geometryAdjustedCostDiff <=
            std::max(static_cast<double>(
                         probConfig
                             .split_one_frame_future_pca_soft_geometry_limit_abs),
                     static_cast<double>(
                         probConfig
                             .split_one_frame_future_pca_soft_geometry_limit_fraction) *
                         baselineImageCost) &&
        bridgeCostRescueGapDensity <=
            probConfig.split_one_frame_future_pca_soft_max_gap_density &&
        bridgeCostRescueValleyFromBright <=
            probConfig.split_one_frame_future_pca_soft_max_valley_from_bright &&
        finalAxisLen >=
            probConfig.split_one_frame_future_pca_soft_min_axis_length_scale *
            std::max(1.0f, srcMaxR) &&
        maxDaughterSeedDrift <=
            std::max(probConfig.split_one_frame_future_pca_soft_max_drift_abs,
                     probConfig.split_one_frame_future_pca_soft_max_drift_scale *
                         std::max(1.0f, srcMaxR));
    const double futureSignalSoftPenaltyRawGainLimit =
        std::max(bridgeRescueLimit,
                 static_cast<double>(
                     probConfig.split_future_signal_soft_penalty_fraction) *
                     baselineImageCost);
    const bool futureSignalSoftPenaltyRawGainRescued =
        bestHasCleanFutureSignalSupport &&
        bridgeProposal != nullptr &&
        bridgeProposal->parentDistanceBalance >=
            probConfig.split_future_signal_soft_min_parent_balance &&
        splitSoftGeometryPenaltyCost > 0.0 &&
        costDiff <= -adaptiveThreshold &&
        geometryAdjustedCostDiff <= futureSignalSoftPenaltyRawGainLimit &&
        imageCostDiff <=
            -std::max(static_cast<double>(
                          probConfig.split_future_signal_soft_image_gain_abs),
                      static_cast<double>(
                          probConfig.split_future_signal_soft_image_gain_fraction) *
                          baselineImageCost) &&
        overlapCostDiff <=
            static_cast<double>(
                probConfig.split_future_signal_soft_overlap_limit_fraction) *
                baselineImageCost &&
        bridgeCostRescueGapDensity <=
            probConfig.split_future_signal_soft_max_gap_density &&
        bridgeCostRescueValleyFromBright <=
            probConfig.split_future_signal_soft_max_valley_from_bright &&
        finalAxisLen >=
            probConfig.split_future_signal_soft_min_axis_length_scale *
            std::max(1.0f, srcMaxR) &&
        maxDaughterSeedDrift <=
            std::max(probConfig.split_future_signal_soft_max_drift_abs,
                     probConfig.split_future_signal_soft_max_drift_scale *
                         std::max(1.0f, srcMaxR));
    const bool futureSignalCleanCenterSoftPenaltyRescued =
        bestHasCleanFutureSignalSupport &&
        bridgeProposal != nullptr &&
        !bridgeProposal->centerSnapUsedAlignedPairFallback &&
        bridgeProposal->centerSnapMaxSeedDistance <=
            probConfig.split_dim_exact_future_signal_snap_epsilon &&
        bridgeProposal->parentDistanceBalance >=
            probConfig.split_future_signal_clean_center_min_parent_balance &&
        bridgeProposal->windowBestMatchedMinBrightness >=
            probConfig.split_future_signal_clean_center_min_brightness &&
        splitSoftGeometryPenaltyCost > 0.0 &&
        splitSoftGeometryPenaltyCost <=
            static_cast<double>(
                probConfig
                    .split_future_signal_clean_center_soft_penalty_fraction) *
                baselineImageCost &&
        costDiff <= -adaptiveThreshold &&
        imageCostDiff <= -adaptiveThreshold &&
        overlapCostDiff <=
            static_cast<double>(
                probConfig
                    .split_future_signal_clean_center_overlap_limit_fraction) *
                baselineImageCost &&
        bridgeCostRescueGapDensity <=
            probConfig.split_future_signal_clean_center_max_gap_density &&
        bridgeCostRescueValleyFromBright <=
            probConfig
                .split_future_signal_clean_center_max_valley_from_bright &&
        finalAxisLen >=
            probConfig
                .split_future_signal_clean_center_min_axis_length_scale *
            std::max(1.0f, srcMaxR) &&
        maxDaughterSeedDrift <=
            std::max(
                probConfig.split_future_signal_clean_center_max_drift_abs,
                probConfig.split_future_signal_clean_center_max_drift_scale *
                    std::max(1.0f, srcMaxR));
    const bool futureSignalNearThresholdCostRescued =
        bestHasCleanFutureSignalSupport &&
        bridgeProposal != nullptr &&
        bridgeProposal->parentDistanceBalance >=
            probConfig.split_future_signal_near_threshold_min_parent_balance &&
        costDiff <=
            static_cast<double>(
                probConfig.split_future_signal_near_threshold_cost_fraction) *
                adaptiveThreshold &&
        imageCostDiff <= -adaptiveThreshold &&
        overlapCostDiff <=
            static_cast<double>(
                probConfig
                    .split_future_signal_near_threshold_overlap_limit_fraction) *
                baselineImageCost &&
        splitSoftGeometryPenaltyCost <=
            static_cast<double>(
                probConfig
                    .split_future_signal_near_threshold_soft_penalty_fraction) *
                baselineImageCost &&
        bridgeCostRescueValleyFromBright <=
            probConfig
                .split_future_signal_near_threshold_max_valley_from_bright &&
        finalAxisLen >=
            probConfig
                .split_future_signal_near_threshold_min_axis_length_scale *
            std::max(1.0f, srcMaxR) &&
        maxDaughterSeedDrift <=
            std::max(probConfig.split_future_signal_near_threshold_max_drift_abs,
                     probConfig
                             .split_future_signal_near_threshold_max_drift_scale *
                         std::max(1.0f, srcMaxR));
    const bool futureSignalCleanValleyCostRescued =
        bestHasCleanFutureSignalSupport &&
        bridgeProposal != nullptr &&
        !bridgeProposal->centerSnapUsedAlignedPairFallback &&
        bridgeProposal->centerSnapMaxSeedDistance <=
            probConfig.split_dim_exact_future_signal_snap_epsilon &&
        bridgeProposal->parentDistanceBalance >=
            probConfig.split_future_signal_clean_valley_min_parent_balance &&
        bridgeProposal->windowBothDaughtersSupported >=
            std::max(2, probConfig.pca_bridge_future_window_min_both_daughter_support) &&
        bridgeProposal->windowMissingDaughterCount == 0 &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->windowBestMatchedMinBrightness >=
            probConfig.split_future_signal_clean_valley_min_brightness &&
        costDiff <=
            std::max(static_cast<double>(
                         probConfig
                             .split_future_signal_clean_valley_cost_limit_abs),
                     static_cast<double>(
                         probConfig
                             .split_future_signal_clean_valley_cost_limit_fraction) *
                         baselineImageCost) &&
        imageCostDiff <=
            -std::max(static_cast<double>(
                          probConfig
                              .split_future_signal_clean_valley_image_gain_abs),
                      static_cast<double>(
                          probConfig
                              .split_future_signal_clean_valley_image_gain_fraction) *
                          baselineImageCost) &&
        overlapCostDiff <=
            std::max(static_cast<double>(
                         probConfig
                             .split_future_signal_clean_valley_overlap_limit_abs),
                     static_cast<double>(
                         probConfig
                             .split_future_signal_clean_valley_overlap_limit_fraction) *
                         baselineImageCost) &&
        splitSoftGeometryPenaltyCost <=
            static_cast<double>(
                probConfig
                    .split_future_signal_clean_valley_soft_penalty_fraction) *
                baselineImageCost &&
        bridgeCostRescueGapDensity <=
            probConfig.split_future_signal_clean_valley_max_gap_density &&
        bridgeCostRescueValleyFromBright <=
            probConfig.split_future_signal_clean_valley_max_valley_from_bright &&
        finalAxisLen >=
            probConfig.split_future_signal_clean_valley_min_axis_length_scale *
            std::max(1.0f, srcMaxR) &&
        maxDaughterSeedDrift <=
            std::max(probConfig.split_future_signal_clean_valley_max_drift_abs,
                     probConfig
                             .split_future_signal_clean_valley_max_drift_scale *
                         std::max(1.0f, srcMaxR));
    const bool futureSignalLockedOverlapCostRescued =
        bestIsSignalCenterProposal &&
        bridgeProposal != nullptr &&
        softOverlapAcceptedForCost &&
        bridgeProposal->gapStartBin <= -4 &&
        bridgeProposal->gapEndBin <= -4 &&
        !bridgeProposal->centerSnapApplied &&
        !bridgeProposal->immediateFutureCenterBacked &&
        !bridgeProposal->centerSnapUsedAlignedPairFallback &&
        bridgeProposal->centerSnapMaxSeedDistance <=
            probConfig.split_dim_exact_future_signal_snap_epsilon &&
        bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
        bridgeProposal->windowBothDaughtersSupported >=
            std::max(2, probConfig.pca_bridge_future_window_min_both_daughter_support) &&
        bridgeProposal->windowMissingDaughterCount == 0 &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->windowBestMatchedMinBrightness >=
            probConfig.split_future_signal_locked_overlap_min_brightness &&
        bridgeProposal->parentShapeElongation >=
            probConfig.split_future_signal_locked_overlap_min_parent_shape &&
        bridgeProposal->parentDistanceBalance >=
            probConfig.split_future_signal_locked_overlap_min_parent_balance &&
        maxDaughterSeedDrift <=
            std::max(
                probConfig.split_future_signal_locked_overlap_max_drift_abs,
                probConfig.split_future_signal_locked_overlap_max_drift_scale *
                    std::max(1.0f, srcMaxR)) &&
        splitSoftGeometryPenaltyCost > 0.0 &&
        splitSoftGeometryPenaltyCost <=
            static_cast<double>(
                probConfig
                    .split_future_signal_locked_overlap_soft_penalty_fraction) *
                baselineImageCost &&
        imageCostDiff <=
            std::max(static_cast<double>(
                         probConfig
                             .split_future_signal_locked_overlap_image_limit_abs),
                     static_cast<double>(
                         probConfig
                             .split_future_signal_locked_overlap_image_limit_fraction) *
                         baselineImageCost) &&
        overlapCostDiff <=
            std::max(static_cast<double>(
                         probConfig
                             .split_future_signal_locked_overlap_overlap_limit_abs),
                     static_cast<double>(
                         probConfig
                             .split_future_signal_locked_overlap_overlap_limit_fraction) *
                         baselineImageCost) &&
        costDiff <=
            std::max(static_cast<double>(
                         probConfig
                             .split_future_signal_locked_overlap_cost_limit_abs),
                     static_cast<double>(
                         probConfig
                             .split_future_signal_locked_overlap_cost_limit_fraction) *
                         baselineImageCost) &&
        geometryAdjustedCostDiff <=
            std::max(static_cast<double>(
                         probConfig
                             .split_future_signal_locked_overlap_geometry_limit_abs),
                     static_cast<double>(
                         probConfig
                             .split_future_signal_locked_overlap_geometry_limit_fraction) *
                         baselineImageCost) &&
        bridgeCostRescueGapDensity <=
            probConfig.split_future_signal_locked_overlap_max_gap_density &&
        bridgeCostRescueValleyFromBright <=
            probConfig
                .split_future_signal_locked_overlap_max_valley_from_bright &&
        finalAxisLen >=
            probConfig.split_future_signal_locked_overlap_min_axis_length_scale *
            std::max(1.0f, srcMaxR);
    const double futureSignalBorderlineMovedCostLimit =
        std::max(static_cast<double>(
                     probConfig
                         .split_future_signal_borderline_moved_cost_limit_abs),
                 static_cast<double>(
                     probConfig
                         .split_future_signal_borderline_moved_cost_limit_fraction) *
                     baselineImageCost);
    const bool futureSignalBorderlineMovedCostRescued =
        bestHasCleanFutureSignalSupport &&
        bridgeProposal != nullptr &&
        bridgeProposal->immediateFutureCenterBacked &&
        bridgeProposal->centerSnapApplied &&
        !bridgeProposal->centerSnapUsedAlignedPairFallback &&
        bridgeProposal->centerSnapMaxSeedDistance <=
            probConfig.split_dim_exact_future_signal_snap_epsilon &&
        bridgeProposal->parentShapeElongation <=
            probConfig.signal_center_split_min_parent_elongation +
                probConfig
                    .split_future_signal_borderline_moved_parent_shape_slack &&
        bridgeProposal->parentDistanceBalance >=
            probConfig.split_future_signal_borderline_moved_min_parent_balance &&
        bridgeProposal->windowBestMatchedMinBrightness >=
            probConfig.split_future_signal_borderline_moved_min_brightness &&
        costDiff <= futureSignalBorderlineMovedCostLimit &&
        imageCostDiff <=
            std::max(static_cast<double>(
                         probConfig
                             .split_future_signal_borderline_moved_image_limit_abs),
                     static_cast<double>(
                         probConfig
                             .split_future_signal_borderline_moved_image_limit_fraction) *
                         baselineImageCost) &&
        overlapCostDiff <=
            static_cast<double>(
                probConfig
                    .split_future_signal_borderline_moved_overlap_limit_fraction) *
                baselineImageCost &&
        splitSoftGeometryPenaltyCost <=
            static_cast<double>(
                probConfig
                    .split_future_signal_borderline_moved_soft_penalty_fraction) *
                baselineImageCost &&
        bridgeCostRescueGapDensity <=
            probConfig.split_future_signal_borderline_moved_max_gap_density &&
        bridgeCostRescueValleyFromBright <=
            probConfig
                .split_future_signal_borderline_moved_max_valley_from_bright &&
        finalAxisLen >=
            probConfig
                .split_future_signal_borderline_moved_min_axis_length_scale *
            std::max(1.0f, srcMaxR) &&
        maxDaughterSeedDrift <= std::max(probConfig.split_long_raw_pca_bridge_max_drift_abs, probConfig.split_long_raw_pca_bridge_max_drift_scale * std::max(1.0f, srcMaxR));
    const bool futureRodTipCleanGapCostRescued =
        bestIsSignalCenterProposal &&
        bridgeProposal != nullptr &&
        bridgeProposal->daughterSphereRadius > 0.0f &&
        probConfig.pca_bridge_future_window_enabled &&
        bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
        bridgeProposal->windowBothDaughtersSupported >=
            std::max(2, probConfig.pca_bridge_future_window_min_both_daughter_support) &&
        bridgeProposal->windowMissingDaughterCount == 0 &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->windowBestMatchedMinBrightness >=
            std::max(0.0f,
                     probConfig
                         .pca_bridge_future_window_rod_tip_balance_min_brightness) &&
        overlapCostDiff <=
            static_cast<double>(
                probConfig
                    .split_future_rod_tip_clean_gap_overlap_limit_fraction) *
                baselineImageCost &&
        imageCostDiff <=
            std::max(static_cast<double>(
                         probConfig
                             .split_future_rod_tip_clean_gap_image_limit_abs),
                     static_cast<double>(
                         probConfig
                             .split_future_rod_tip_clean_gap_image_limit_fraction) *
                         baselineImageCost) &&
        geometryAdjustedCostDiff <=
            std::max(static_cast<double>(
                         probConfig
                             .split_future_rod_tip_clean_gap_geometry_limit_abs),
                     static_cast<double>(
                         probConfig
                             .split_future_rod_tip_clean_gap_geometry_limit_fraction) *
                         baselineImageCost) &&
        bridgeCostRescueGapDensity <=
            probConfig.split_future_rod_tip_clean_gap_max_gap_density &&
        bridgeCostRescueValleyFromBright <=
            probConfig.split_future_rod_tip_clean_gap_max_valley_from_bright &&
        finalAxisLen >=
            probConfig.split_future_rod_tip_clean_gap_min_axis_length_scale *
            std::max(1.0f, srcMaxR) &&
        maxDaughterSeedDrift <=
            std::max(probConfig.split_future_rod_tip_clean_gap_max_drift_abs,
                     probConfig.split_future_rod_tip_clean_gap_max_drift_scale *
                         std::max(1.0f, srcMaxR));
    const double futureRodTipPrimaryCostLimit =
        std::max(static_cast<double>(
                     probConfig.split_future_rod_tip_primary_cost_limit_abs),
                 static_cast<double>(
                     probConfig
                         .split_future_rod_tip_primary_cost_limit_fraction) *
                     baselineImageCost);
    const double futureRodTipPrimaryImageCostLimit =
        std::max(static_cast<double>(
                     probConfig.split_future_rod_tip_primary_image_limit_abs),
                 static_cast<double>(
                     probConfig
                         .split_future_rod_tip_primary_image_limit_fraction) *
                     baselineImageCost);
    const double futureRodTipPrimaryOverlapCostLimit =
        std::max(static_cast<double>(
                     probConfig.split_future_rod_tip_primary_overlap_limit_abs),
                 static_cast<double>(
                     probConfig
                         .split_future_rod_tip_primary_overlap_limit_fraction) *
                     baselineImageCost);
    const double futureRodTipPrimaryImageGainRequired =
        std::max(static_cast<double>(
                     probConfig.split_future_rod_tip_primary_image_gain_abs),
                 static_cast<double>(
                     probConfig
                         .split_future_rod_tip_primary_image_gain_fraction) *
                     baselineImageCost);
    const bool futureRodTipPrimaryRequiresCurrentImageGain =
        bridgeProposal != nullptr &&
        bridgeProposal->daughterSphereRadius > 0.0f &&
        bridgeProposal->centerSnapUsedAlignedPairFallback &&
        bridgeProposal->parentShapeElongation >=
            probConfig.pca_bridge_future_window_min_parent_shape_for_cost_rescue &&
        bridgeProposal->centerSnapMaxSeedDistance > cleanPcaBridgeFuturePairSnapLimit;
    const bool futureRodTipPrimaryHasCurrentImageGain =
        imageCostDiff <= -futureRodTipPrimaryImageGainRequired;
    const bool futureRodTipPrimaryStrongFutureEvidence =
        bridgeProposal != nullptr &&
        bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
        bridgeProposal->windowBothDaughtersSupported >= 2 &&
        bridgeProposal->windowMissingDaughterCount == 0 &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->windowBestMatchedMinBrightness >=
            probConfig.split_future_rod_tip_primary_min_brightness &&
        bridgeProposal->parentDistanceBalance >=
            probConfig.split_future_rod_tip_primary_min_parent_balance;
    const bool futureRodTipPrimaryCostRescued =
        (bestIsSignalCenterProposal || bestIsPcaBridgeOnly) &&
        bridgeProposal != nullptr &&
        bridgeProposal->daughterSphereRadius > 0.0f &&
        bridgeProposal->centerSnapUsedAlignedPairFallback &&
        bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
        bridgeProposal->windowBothDaughtersSupported >= 1 &&
        bridgeProposal->windowMissingDaughterCount <= 2 &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->windowBestMatchedMinBrightness >= probConfig.split_future_rod_tip_primary_min_brightness &&
        bridgeProposal->parentDistanceBalance >= probConfig.split_future_rod_tip_primary_min_parent_balance &&
        (!futureRodTipPrimaryRequiresCurrentImageGain ||
         futureRodTipPrimaryHasCurrentImageGain ||
         futureRodTipPrimaryStrongFutureEvidence) &&
        costDiff <= futureRodTipPrimaryCostLimit &&
        imageCostDiff <= futureRodTipPrimaryImageCostLimit &&
        overlapCostDiff <= futureRodTipPrimaryOverlapCostLimit &&
        bridgeCostRescueGapDensity <= probConfig.split_future_rod_tip_primary_max_gap_density &&
        bridgeCostRescueValleyFromBright <= probConfig.split_future_rod_tip_primary_max_valley_from_bright &&
        finalAxisLen >= probConfig.split_future_rod_tip_primary_min_axis_length_scale * std::max(1.0f, srcMaxR) &&
        maxDaughterSeedDrift <= std::max(probConfig.split_future_rod_tip_primary_max_drift_abs,
                                        probConfig.split_future_rod_tip_primary_max_drift_scale * std::max(1.0f, srcMaxR));
    const double oneFrameBridgeCostLimit =
        std::max(static_cast<double>(probConfig.split_one_frame_future_bridge_cost_limit_abs),
                 static_cast<double>(probConfig.split_one_frame_future_bridge_cost_limit_fraction) * baselineImageCost);
    const bool oneFrameFutureBridgeCostRescued =
        bestIsPcaBridgeOnly &&
        bridgeProposal != nullptr &&
        bridgeProposal->futureWindowSplitRescue &&
        bridgeProposal->centerSnapUsedAlignedPairFallback &&
        bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
        bridgeProposal->windowBothDaughtersSupported == 1 &&
        bridgeProposal->windowMissingDaughterCount <= 2 &&
        bridgeProposal->windowParentPersists <=
            std::max(0, probConfig.pca_bridge_future_window_max_parent_persists) &&
        bridgeProposal->parentShapeElongation >= probConfig.split_one_frame_future_bridge_min_parent_shape &&
        bridgeProposal->parentShapeElongation < probConfig.split_one_frame_future_bridge_max_parent_shape &&
        bridgeProposal->parentDistanceBalance >= probConfig.split_one_frame_future_bridge_min_parent_balance &&
        finalAxisLen >= std::max(1.0f, probConfig.split_one_frame_future_bridge_min_axis_length_scale * srcMaxR) &&
        bridgeProposal->centerSnapMaxSeedDistance <=
            std::max(cleanPcaBridgeFuturePairSnapLimit, probConfig.split_one_frame_future_bridge_snap_scale * srcMaxR) &&
        costDiff <= oneFrameBridgeCostLimit;
    const bool oneFramePcaBridgeOverlapCostRescued =
        bestIsPcaBridgeOnly &&
        bridgeProposal != nullptr &&
        !bridgeProposal->centerSnapUsedAlignedPairFallback &&
        bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
        bridgeProposal->windowBothDaughtersSupported == 1 &&
        bridgeProposal->windowMissingDaughterCount <=
            std::max(0, probConfig.split_one_frame_pca_bridge_max_future_missing) &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->parentShapeElongation >=
            probConfig.pca_bridge_future_window_min_parent_shape_for_cost_rescue &&
        bridgeProposal->parentDistanceBalance >= probConfig.split_one_frame_pca_bridge_min_parent_balance &&
        bridgeProposal->centerSnapMaxSeedDistance <= cleanPcaBridgeSnapDistanceLimit &&
        imageCostDiff <= -std::max(static_cast<double>(probConfig.split_one_frame_pca_bridge_image_gain_abs),
                                   static_cast<double>(probConfig.split_one_frame_pca_bridge_image_gain_fraction) * baselineImageCost) &&
        overlapCostDiff <= static_cast<double>(probConfig.split_one_frame_pca_bridge_overlap_limit_fraction) * baselineImageCost &&
        costDiff <= std::max(static_cast<double>(probConfig.split_one_frame_pca_bridge_cost_limit_abs),
                             static_cast<double>(probConfig.split_one_frame_pca_bridge_cost_limit_fraction) * baselineImageCost) &&
        bridgeCostRescueValleyFromBright <= probConfig.split_one_frame_pca_bridge_max_valley_from_bright &&
        bridgeCostRescueGapDensity <= probConfig.split_one_frame_pca_bridge_max_gap_density &&
        finalAxisLen >= probConfig.split_one_frame_pca_bridge_min_axis_length_scale * std::max(1.0f, srcMaxR);
    const double cleanFuturePcaBridgeOverlapRescueLimit = std::max(
        static_cast<double>(probConfig.split_clean_future_bridge_cost_limit_abs),
        static_cast<double>(probConfig.split_clean_future_bridge_cost_limit_fraction) *
            baselineImageCost);
    const double cleanFuturePcaBridgeImageGainRequired = std::max(
        static_cast<double>(probConfig.split_clean_future_bridge_image_gain_abs),
        static_cast<double>(probConfig.split_clean_future_bridge_image_gain_fraction) *
            baselineImageCost);
    const double cleanFuturePcaBridgeOverlapPenaltyLimit = std::max(
        static_cast<double>(probConfig.split_clean_future_bridge_overlap_limit_abs),
        static_cast<double>(probConfig.split_clean_future_bridge_overlap_limit_fraction) *
            baselineImageCost);
    const double alignedFuturePcaBridgeImageGainRequired = std::max(
        static_cast<double>(probConfig.split_aligned_future_bridge_image_gain_abs),
        static_cast<double>(probConfig.split_aligned_future_bridge_image_gain_fraction) *
            baselineImageCost);
    const double alignedFuturePcaBridgeOverlapPenaltyLimit = std::max(
        static_cast<double>(probConfig.split_aligned_future_bridge_overlap_limit_abs),
        static_cast<double>(probConfig.split_aligned_future_bridge_overlap_limit_fraction) *
            baselineImageCost);
    const bool cleanFuturePcaBridgeOverlapCostRescued =
        bestIsPcaBridgeOnly &&
        bridgeProposal != nullptr &&
        bestHasCleanFutureBridgeSupport &&
        bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
        bridgeProposal->windowBothDaughtersSupported >=
            std::max(2, probConfig.pca_bridge_future_window_min_both_daughter_support) &&
        bridgeProposal->windowMissingDaughterCount == 0 &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->windowBestMatchedMinBrightness >=
            probConfig.split_clean_future_bridge_min_brightness &&
        bridgeProposal->parentDistanceBalance >=
            probConfig.split_clean_future_bridge_min_parent_balance &&
        !bridgeProposal->centerSnapUsedAlignedPairFallback &&
        imageCostDiff <= -cleanFuturePcaBridgeImageGainRequired &&
        costDiff <= cleanFuturePcaBridgeOverlapRescueLimit &&
        overlapCostDiff <= cleanFuturePcaBridgeOverlapPenaltyLimit &&
        bridgeCostRescueValleyFromBright <=
            probConfig.split_clean_future_bridge_max_valley_from_bright &&
        bridgeCostRescueGapDensity <=
            probConfig.split_clean_future_bridge_max_gap_density &&
        finalAxisLen >= probConfig.split_clean_future_bridge_min_axis_length_scale *
                            std::max(1.0f, srcMaxR);
    const bool alignedFuturePcaBridgeImageGainCostRescued =
        bestIsPcaBridgeOnly &&
        bridgeProposal != nullptr &&
        bestHasCleanFutureBridgeSupport &&
        bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
        bridgeProposal->windowBothDaughtersSupported >=
            std::max(2, probConfig.pca_bridge_future_window_min_both_daughter_support) &&
        bridgeProposal->windowMissingDaughterCount == 0 &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->windowBestMatchedMinBrightness >=
            probConfig.split_aligned_future_bridge_min_brightness &&
        bridgeProposal->parentShapeElongation >=
            probConfig.split_aligned_future_bridge_min_parent_shape &&
        bridgeProposal->parentDistanceBalance >=
            probConfig.split_aligned_future_bridge_min_parent_balance &&
        bridgeProposal->centerSnapUsedAlignedPairFallback &&
        bridgeProposal->centerSnapMaxSeedDistance <=
            std::max(cleanPcaBridgeFuturePairSnapLimit,
                     probConfig.split_aligned_future_bridge_snap_scale *
                         std::max(1.0f, srcMaxR)) &&
        imageCostDiff <= -alignedFuturePcaBridgeImageGainRequired &&
        costDiff <= 0.0 &&
        overlapCostDiff <= alignedFuturePcaBridgeOverlapPenaltyLimit &&
        bridgeCostRescueValleyFromBright <=
            probConfig.split_aligned_future_bridge_max_valley_from_bright &&
        bridgeCostRescueGapDensity <=
            probConfig.split_aligned_future_bridge_max_gap_density &&
        finalAxisLen >= probConfig.split_aligned_future_bridge_min_axis_length_scale *
                            std::max(1.0f, srcMaxR);
    const double currentLockedPcaBridgeNearThresholdGainRequired = std::max(
        static_cast<double>(probConfig.split_current_locked_bridge_image_gain_abs),
        static_cast<double>(probConfig.split_current_locked_bridge_image_gain_fraction) *
            baselineImageCost);
    const bool currentLockedCleanPcaBridgeNearThresholdCostRescued =
        bestIsPcaBridgeOnly &&
        bestLabel == "bridge_primary" &&
        bridgeProposal != nullptr &&
        softOverlapAcceptedForCost &&
        bridgeProposal->centerSnapApplied &&
        !bridgeProposal->immediateFutureCenterBacked &&
        !bridgeProposal->centerSnapUsedAlignedPairFallback &&
        bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
        bridgeProposal->windowBothDaughtersSupported >=
            std::max(2, probConfig.pca_bridge_future_window_min_both_daughter_support) &&
        bridgeProposal->windowMissingDaughterCount == 0 &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->windowBestMatchedMinBrightness >=
            probConfig.split_current_locked_bridge_min_brightness &&
        bridgeProposal->parentShapeElongation >=
            probConfig.split_current_locked_bridge_min_parent_shape &&
        bridgeProposal->parentDistanceBalance >=
            probConfig.split_current_locked_bridge_min_parent_balance &&
        bridgeProposal->parentDistanceBalance <=
            probConfig.split_current_locked_bridge_max_parent_balance &&
        bridgeProposal->centerSnapMaxSeedDistance <=
            std::max(cleanPcaBridgeFuturePairSnapLimit,
                     probConfig.split_current_locked_bridge_snap_scale *
                         std::max(1.0f, srcMaxR)) &&
        maxDaughterSeedDrift <=
            std::max(probConfig.split_current_locked_bridge_max_drift_abs,
                     probConfig.split_current_locked_bridge_max_drift_scale *
                         std::max(1.0f, srcMaxR)) &&
        imageCostDiff <= -currentLockedPcaBridgeNearThresholdGainRequired &&
        overlapCostDiff <=
            std::max(static_cast<double>(probConfig.split_current_locked_bridge_overlap_limit_abs),
                     static_cast<double>(probConfig.split_current_locked_bridge_overlap_limit_fraction) *
                         baselineImageCost) &&
        costDiff <= static_cast<double>(probConfig.split_current_locked_bridge_cost_limit_fraction) * adaptiveThreshold &&
        geometryAdjustedCostDiff <=
            std::max(static_cast<double>(probConfig.split_current_locked_bridge_geometry_limit_abs),
                     static_cast<double>(probConfig.split_current_locked_bridge_geometry_limit_fraction) *
                         baselineImageCost) &&
        bridgeCostRescueGapDensity <=
            probConfig.split_current_locked_bridge_max_gap_density &&
        bridgeCostRescueValleyFromBright <=
            probConfig.split_current_locked_bridge_max_valley_from_bright &&
        finalAxisLen >= probConfig.split_current_locked_bridge_min_axis_length_scale *
                            std::max(1.0f, srcMaxR);
    const double softOverlapFuturePcaBridgeCostLimit = std::max(
        static_cast<double>(probConfig.split_soft_overlap_future_bridge_cost_limit_abs),
        static_cast<double>(probConfig.split_soft_overlap_future_bridge_cost_limit_fraction) *
            baselineImageCost);
    const bool softOverlapFuturePcaBridgeCostRescued =
        bestIsPcaBridgeOnly &&
        bridgeProposal != nullptr &&
        softOverlapAcceptedForCost &&
        bestHasCleanFutureBridgeSupport &&
        bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
        bridgeProposal->windowBothDaughtersSupported >=
            std::max(2, probConfig.pca_bridge_future_window_min_both_daughter_support) &&
        bridgeProposal->windowMissingDaughterCount == 0 &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->windowBestMatchedMinBrightness >=
            probConfig.split_soft_overlap_future_bridge_min_brightness &&
        bridgeProposal->parentDistanceBalance >=
            probConfig.split_soft_overlap_future_bridge_min_parent_balance &&
        bridgeProposal->centerSnapUsedAlignedPairFallback &&
        bridgeProposal->centerSnapMaxSeedDistance <=
            std::max(cleanPcaBridgeFuturePairSnapLimit,
                     probConfig.split_soft_overlap_future_bridge_snap_scale *
                         std::max(1.0f, srcMaxR)) &&
        costDiff <= softOverlapFuturePcaBridgeCostLimit &&
        imageCostDiff <=
            std::max(static_cast<double>(probConfig.split_soft_overlap_future_bridge_image_limit_abs),
                     static_cast<double>(probConfig.split_soft_overlap_future_bridge_image_limit_fraction) *
                         baselineImageCost) &&
        overlapCostDiff <=
            std::max(static_cast<double>(probConfig.split_soft_overlap_future_bridge_overlap_limit_abs),
                     static_cast<double>(probConfig.split_soft_overlap_future_bridge_overlap_limit_fraction) *
                         baselineImageCost) &&
        bridgeCostRescueGapDensity <=
            probConfig.split_soft_overlap_future_bridge_max_gap_density &&
        bridgeCostRescueValleyFromBright <=
            probConfig.split_soft_overlap_future_bridge_max_valley_from_bright &&
        finalAxisLen >= probConfig.split_soft_overlap_future_bridge_min_axis_length_scale *
                            std::max(1.0f, srcMaxR);
    const double stableFuturePcaBridgeNearThresholdGainRequired = std::max(
        static_cast<double>(probConfig.split_stable_future_bridge_image_gain_abs),
        static_cast<double>(probConfig.split_stable_future_bridge_image_gain_fraction) *
            baselineImageCost);
    const double stableFuturePcaBridgeNearThresholdOverlapLimit = std::max(
        static_cast<double>(probConfig.split_stable_future_bridge_overlap_limit_abs),
        static_cast<double>(probConfig.split_stable_future_bridge_overlap_limit_fraction) *
            baselineImageCost);
    const bool stableFuturePcaBridgeNearThresholdCostRescued =
        bestIsPcaBridgeOnly &&
        bestLabel == "bridge_primary" &&
        bridgeProposal != nullptr &&
        softOverlapAcceptedForCost &&
        bestHasCleanFutureBridgeSupport &&
        !bridgeProposal->centerSnapUsedAlignedPairFallback &&
        bridgeProposal->windowBothDaughtersSupported >=
            std::max(2, probConfig.pca_bridge_future_window_min_both_daughter_support) &&
        bridgeProposal->windowMissingDaughterCount == 0 &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->windowBestMatchedMinBrightness >=
            probConfig.split_stable_future_bridge_min_brightness &&
        bridgeProposal->parentShapeElongation >=
            probConfig.split_stable_future_bridge_min_parent_shape &&
        bridgeProposal->parentDistanceBalance >=
            probConfig.split_stable_future_bridge_min_parent_balance &&
        maxDaughterSeedDrift <=
            std::max(probConfig.split_stable_future_bridge_max_drift_abs,
                     probConfig.split_stable_future_bridge_max_drift_scale *
                         std::max(1.0f, srcMaxR)) &&
        imageCostDiff <= -stableFuturePcaBridgeNearThresholdGainRequired &&
        costDiff <= -stableFuturePcaBridgeNearThresholdGainRequired &&
        overlapCostDiff <= stableFuturePcaBridgeNearThresholdOverlapLimit &&
        bridgeCostRescueGapDensity <=
            probConfig.split_stable_future_bridge_max_gap_density &&
        bridgeCostRescueValleyFromBright <=
            probConfig.split_stable_future_bridge_max_valley_from_bright &&
        finalAxisLen >= probConfig.split_stable_future_bridge_min_axis_length_scale *
                            std::max(1.0f, srcMaxR);
    const bool cleanRodTipContinuationOverlapCostRescued =
        bridgeProposalOnly &&
        bestLabel == "bridge_primary" &&
        bridgeProposal != nullptr &&
        cleanContinuationDaughterOverlapAcceptedForCost &&
        bridgeProposal->daughterSphereRadius > 0.0f &&
        bridgeProposal->windowBothDaughtersSupported >= 2 &&
        bridgeProposal->windowMissingDaughterCount == 0 &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->windowBestMatchedMinBrightness >= probConfig.split_clean_rod_tip_continuation_min_brightness &&
        bridgeProposal->parentDistanceBalance >= probConfig.split_clean_rod_tip_continuation_min_parent_balance &&
        !bridgeProposal->centerSnapUsedAlignedPairFallback &&
        imageCostDiff <= -std::max(static_cast<double>(probConfig.split_clean_rod_tip_continuation_image_gain_abs),
                                   static_cast<double>(probConfig.split_clean_rod_tip_continuation_image_gain_fraction) * baselineImageCost) &&
        overlapCostDiff <= std::max(static_cast<double>(probConfig.split_clean_rod_tip_continuation_overlap_limit_abs),
                                    static_cast<double>(probConfig.split_clean_rod_tip_continuation_overlap_limit_fraction) * baselineImageCost) &&
        costDiff <= std::max(static_cast<double>(probConfig.split_clean_rod_tip_continuation_cost_limit_abs),
                             static_cast<double>(probConfig.split_clean_rod_tip_continuation_cost_limit_fraction) * baselineImageCost) &&
        bridgeCostRescueGapDensity <= probConfig.split_clean_rod_tip_continuation_max_gap_density &&
        bridgeCostRescueValleyFromBright <= probConfig.split_clean_rod_tip_continuation_max_valley_from_bright &&
        finalAxisLen >= probConfig.split_clean_rod_tip_continuation_min_axis_length_scale * std::max(1.0f, srcMaxR) &&
        maxDaughterSeedDrift <= std::max(probConfig.split_clean_rod_tip_continuation_max_drift_abs,
                                        probConfig.split_clean_rod_tip_continuation_max_drift_scale * std::max(1.0f, srcMaxR));
    const bool dimExactFutureSignalOverlapCostRescued =
        simulationConfig.celluniverse2_enabled &&
        bridgeProposalOnly &&
        bestLabel == "bridge_primary" &&
        bridgeProposal != nullptr &&
        dimExactFutureSignalRodBlockerRescue &&
        bridgeProposal->parentShapeElongation >= probConfig.split_dim_exact_future_signal_min_parent_shape &&
        bridgeProposal->parentDistanceBalance >= probConfig.split_dim_exact_future_signal_min_parent_balance &&
        bridgeProposal->centerSnapMaxSeedDistance <= probConfig.split_dim_exact_future_signal_snap_epsilon &&
        maxDaughterSeedDrift <= std::max(probConfig.split_dim_exact_future_signal_max_drift_abs,
                                        probConfig.split_dim_exact_future_signal_max_drift_scale * std::max(1.0f, srcMaxR)) &&
        imageCostDiff <= -std::max(static_cast<double>(probConfig.split_dim_exact_future_signal_image_gain_abs),
                                   static_cast<double>(probConfig.split_dim_exact_future_signal_image_gain_fraction) * baselineImageCost) &&
        overlapCostDiff <= std::max(static_cast<double>(probConfig.split_dim_exact_future_signal_overlap_limit_abs),
                                    static_cast<double>(probConfig.split_dim_exact_future_signal_overlap_limit_fraction) * baselineImageCost) &&
        costDiff <= std::max(static_cast<double>(probConfig.split_dim_exact_future_signal_cost_limit_abs),
                             static_cast<double>(probConfig.split_dim_exact_future_signal_cost_limit_fraction) * baselineImageCost) &&
        bridgeCostRescueGapDensity <= probConfig.split_dim_exact_future_signal_max_gap_density &&
        bridgeCostRescueValleyFromBright <= probConfig.split_dim_exact_future_signal_max_valley_from_bright &&
        finalAxisLen >= probConfig.split_dim_exact_future_signal_min_axis_length_scale * std::max(1.0f, srcMaxR);
    const double bridgeAxisPlaceFutureImageGainRequired =
        std::max(static_cast<double>(probConfig.split_bridge_axis_place_future_image_gain_abs),
                 static_cast<double>(probConfig.split_bridge_axis_place_future_image_gain_fraction) * baselineImageCost);
    const double bridgeAxisPlaceFutureOverlapPenaltyLimit =
        std::max(static_cast<double>(probConfig.split_bridge_axis_place_future_overlap_limit_abs),
                 static_cast<double>(probConfig.split_bridge_axis_place_future_overlap_limit_fraction) * baselineImageCost);
    const bool bridgeAxisPlaceFutureImageGainCostRescued =
        (bestIsPcaBridgeOnly || bridgeProposalOnly) &&
        bestLabel == "bridge_axis_place" &&
        bridgeProposal != nullptr &&
        bridgeProposal->daughterSphereRadius > 0.0f &&
        bridgeProposal->windowBothDaughtersSupported >= 2 &&
        bridgeProposal->windowMissingDaughterCount == 0 &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->windowBestMatchedMinBrightness >= probConfig.split_bridge_axis_place_future_min_brightness &&
        bridgeProposal->parentDistanceBalance >= probConfig.split_bridge_axis_place_future_min_parent_balance &&
        imageCostDiff <= -bridgeAxisPlaceFutureImageGainRequired &&
        overlapCostDiff <= bridgeAxisPlaceFutureOverlapPenaltyLimit &&
        costDiff <= std::max(static_cast<double>(probConfig.split_bridge_axis_place_future_cost_limit_abs),
                             static_cast<double>(probConfig.split_bridge_axis_place_future_cost_limit_fraction) * baselineImageCost) &&
        bridgeCostRescueValleyFromBright <= probConfig.split_bridge_axis_place_future_max_valley_from_bright &&
        bridgeCostRescueGapDensity <= probConfig.split_bridge_axis_place_future_max_gap_density &&
        finalAxisLen >= probConfig.split_bridge_axis_place_future_min_axis_length_scale * std::max(1.0f, srcMaxR) &&
        maxDaughterSeedDrift <= std::max(probConfig.split_bridge_axis_place_future_max_drift_abs,
                                        probConfig.split_bridge_axis_place_future_max_drift_scale * std::max(1.0f, srcMaxR));
    const double bridgeAxisPlaceNearThresholdRequired =
        std::max(static_cast<double>(probConfig.split_bridge_axis_place_near_image_gain_abs),
                 static_cast<double>(probConfig.split_bridge_axis_place_near_image_gain_adaptive_fraction) * adaptiveThreshold);
    const bool bridgeAxisPlaceCleanNearThresholdCostRescued =
        bridgeProposalOnly &&
        bestLabel == "bridge_axis_place" &&
        bridgeProposal != nullptr &&
        bridgeProposal->daughterSphereRadius > 0.0f &&
        bridgeProposal->windowParentPersists == 0 &&
        maxDaughterSeedDrift <= probConfig.split_bridge_axis_place_near_max_drift &&
        finalAxisLen >= probConfig.split_bridge_axis_place_near_min_axis_length_scale * std::max(1.0f, srcMaxR) &&
        imageCostDiff <= -bridgeAxisPlaceNearThresholdRequired &&
        costDiff <= -bridgeAxisPlaceNearThresholdRequired &&
        overlapCostDiff <= std::max(static_cast<double>(probConfig.split_bridge_axis_place_near_overlap_limit_abs),
                                    static_cast<double>(probConfig.split_bridge_axis_place_near_overlap_limit_fraction) * baselineImageCost) &&
        bridgeCostRescueValleyFromBright <= probConfig.split_bridge_axis_place_near_max_valley_from_bright &&
        bridgeCostRescueGapDensity <= probConfig.split_bridge_axis_place_near_max_gap_density;
    const double cellUniverse3AxisPlaceCostShortfall =
        costDiff + adaptiveThreshold;
    const double cellUniverse3AxisPlaceCostShortfallLimit =
        std::max(static_cast<double>(
                     probConfig
                         .celluniverse3_axis_place_near_threshold_cost_rescue_max_cost_shortfall_abs),
                 static_cast<double>(
                     probConfig
                         .celluniverse3_axis_place_near_threshold_cost_rescue_max_cost_shortfall_fraction) *
                     baselineImageCost);
    const double cellUniverse3AxisPlaceOverlapLimit =
        std::max(static_cast<double>(
                     probConfig
                         .celluniverse3_axis_place_near_threshold_cost_rescue_max_overlap_diff_abs),
                 static_cast<double>(
                     probConfig
                         .celluniverse3_axis_place_near_threshold_cost_rescue_max_overlap_diff_fraction) *
                     baselineImageCost);
    const bool cellUniverse3AxisPlaceNearThresholdCostRescued =
        simulationConfig.celluniverse3_enabled &&
        probConfig
            .celluniverse3_axis_place_near_threshold_cost_rescue_enabled &&
        (bestIsPcaBridgeOnly || bridgeProposalOnly) &&
        bestLabel == "bridge_axis_place" &&
        bridgeProposal != nullptr &&
        bridgeProposal->daughterSphereRadius > 0.0f &&
        bridgeProposal->windowBothDaughtersSupported >=
            std::max(
                1,
                probConfig
                    .celluniverse3_axis_place_near_threshold_cost_rescue_min_future_both) &&
        bridgeProposal->windowMissingDaughterCount <=
            std::max(
                0,
                probConfig
                    .celluniverse3_axis_place_near_threshold_cost_rescue_max_missing) &&
        bridgeProposal->windowParentPersists <=
            std::max(
                0,
                probConfig
                    .celluniverse3_axis_place_near_threshold_cost_rescue_max_parent_persists) &&
        bridgeProposal->parentShapeElongation >=
            probConfig
                .celluniverse3_axis_place_near_threshold_cost_rescue_min_parent_shape &&
        finalAxisLen >=
            probConfig
                .celluniverse3_axis_place_near_threshold_cost_rescue_min_axis_length_scale *
            std::max(1.0f, srcMaxR) &&
        maxDaughterSeedDrift <=
            probConfig
                .celluniverse3_axis_place_near_threshold_cost_rescue_max_seed_drift_scale *
            std::max(1.0f, srcMaxR) &&
        bridgeCostRescueGapDensity <=
            probConfig
                .celluniverse3_axis_place_near_threshold_cost_rescue_max_gap_density &&
        bridgeCostRescueValleyFromBright <=
            probConfig
                .celluniverse3_axis_place_near_threshold_cost_rescue_max_valley_from_bright &&
        imageCostDiff <= 0.0 &&
        splitSoftGeometryPenaltyCost <= 0.0 &&
        overlapCostDiff <= cellUniverse3AxisPlaceOverlapLimit &&
        cellUniverse3AxisPlaceCostShortfall >= 0.0 &&
        cellUniverse3AxisPlaceCostShortfall <=
            cellUniverse3AxisPlaceCostShortfallLimit;
    const bool longRawPcaBridgeNearThresholdCostRescued =
        bestIsPcaBridgeOnly &&
        bridgeProposal != nullptr &&
        softOverlapAcceptedForCost &&
        !bridgeProposal->centerSnapApplied &&
        !bridgeProposal->immediateFutureCenterBacked &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->parentShapeElongation >= probConfig.split_long_raw_pca_bridge_min_parent_shape &&
        bridgeProposal->parentDistanceBalance >= probConfig.split_long_raw_pca_bridge_min_parent_balance &&
        finalAxisLen >= probConfig.split_long_raw_pca_bridge_min_axis_length_scale * std::max(1.0f, srcMaxR) &&
        imageCostDiff <= -std::max(static_cast<double>(probConfig.split_long_raw_pca_bridge_image_gain_abs),
                                   static_cast<double>(probConfig.split_long_raw_pca_bridge_image_gain_fraction) * baselineImageCost) &&
        overlapCostDiff <= 0.0 &&
        costDiff <= -static_cast<double>(probConfig.split_long_raw_pca_bridge_cost_adaptive_fraction) * adaptiveThreshold &&
        bridgeCostRescueValleyFromBright <= probConfig.split_long_raw_pca_bridge_max_valley_from_bright &&
        bridgeCostRescueGapDensity <= probConfig.split_long_raw_pca_bridge_max_gap_density &&
        maxDaughterSeedDrift <= std::max(probConfig.split_long_raw_pca_bridge_max_drift_abs,
                                        probConfig.split_long_raw_pca_bridge_max_drift_scale * std::max(1.0f, srcMaxR));
    const bool asymmetricRawPcaBridgeNearThresholdCostRescued =
        bestIsPcaBridgeOnly &&
        bridgeProposal != nullptr &&
        softOverlapAcceptedForCost &&
        !bridgeProposal->centerSnapUsedAlignedPairFallback &&
        bridgeProposal->windowBothDaughtersSupported >= static_cast<int>(probConfig.split_asymmetric_raw_pca_bridge_min_future_both) &&
        bridgeProposal->windowMissingDaughterCount <= static_cast<int>(probConfig.split_asymmetric_raw_pca_bridge_max_missing_daughters) &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->parentShapeElongation >= probConfig.split_asymmetric_raw_pca_bridge_min_parent_shape &&
        bridgeProposal->parentDistanceBalance >= probConfig.split_asymmetric_raw_pca_bridge_min_parent_balance &&
        finalAxisLen >= probConfig.split_asymmetric_raw_pca_bridge_min_axis_length_scale * std::max(1.0f, srcMaxR) &&
        imageCostDiff <= -std::max(static_cast<double>(probConfig.split_asymmetric_raw_pca_bridge_image_gain_abs),
                                   static_cast<double>(probConfig.split_asymmetric_raw_pca_bridge_image_gain_fraction) * baselineImageCost) &&
        overlapCostDiff <= static_cast<double>(probConfig.split_asymmetric_raw_pca_bridge_overlap_limit_fraction) * baselineImageCost &&
        costDiff <= -std::max(static_cast<double>(probConfig.split_asymmetric_raw_pca_bridge_cost_limit_abs),
                              static_cast<double>(probConfig.split_asymmetric_raw_pca_bridge_cost_limit_fraction) * baselineImageCost) &&
        bridgeCostRescueValleyFromBright <= probConfig.split_asymmetric_raw_pca_bridge_max_valley_from_bright &&
        bridgeCostRescueGapDensity <= probConfig.split_asymmetric_raw_pca_bridge_max_gap_density &&
        maxDaughterSeedDrift <= std::max(probConfig.split_asymmetric_raw_pca_bridge_max_drift_abs,
                                        probConfig.split_asymmetric_raw_pca_bridge_max_drift_scale * std::max(1.0f, srcMaxR));
    const bool delayedFuturePcaBridgeNearMissCostRescued =
        bestIsPcaBridgeOnly &&
        bridgeProposal != nullptr &&
        bridgeProposal->bioSeparationSoftRescued &&
        !bridgeProposal->centerSnapUsedAlignedPairFallback &&
        bridgeProposal->windowImmediateBothDaughtersSupported == 0 &&
        bridgeProposal->windowBothDaughtersSupported >= static_cast<int>(probConfig.split_delayed_future_pca_bridge_min_future_both) &&
        bridgeProposal->windowMissingDaughterCount <= static_cast<int>(probConfig.split_delayed_future_pca_bridge_max_missing_daughters) &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->windowBestMatchedMinBrightness >= probConfig.split_delayed_future_pca_bridge_min_brightness &&
        bridgeProposal->parentShapeElongation >= probConfig.split_delayed_future_pca_bridge_min_parent_shape &&
        bridgeProposal->parentDistanceBalance >= probConfig.split_delayed_future_pca_bridge_min_parent_balance &&
        splitSoftGeometryPenaltyCost > 0.0 &&
        splitSoftGeometryPenaltyCost <= static_cast<double>(probConfig.split_delayed_future_pca_bridge_soft_penalty_fraction) * baselineImageCost &&
        imageCostDiff <= -std::max(static_cast<double>(probConfig.split_delayed_future_pca_bridge_image_gain_abs),
                                   static_cast<double>(probConfig.split_delayed_future_pca_bridge_image_gain_fraction) * baselineImageCost) &&
        overlapCostDiff <= std::max(static_cast<double>(probConfig.split_delayed_future_pca_bridge_overlap_limit_abs),
                                    static_cast<double>(probConfig.split_delayed_future_pca_bridge_overlap_limit_fraction) * baselineImageCost) &&
        costDiff <= std::max(static_cast<double>(probConfig.split_delayed_future_pca_bridge_cost_limit_abs),
                             static_cast<double>(probConfig.split_delayed_future_pca_bridge_cost_limit_fraction) * baselineImageCost) &&
        geometryAdjustedCostDiff <= std::max(static_cast<double>(probConfig.split_delayed_future_pca_bridge_geometry_limit_abs),
                                             static_cast<double>(probConfig.split_delayed_future_pca_bridge_geometry_limit_fraction) * baselineImageCost) &&
        bridgeCostRescueValleyFromBright <= probConfig.split_delayed_future_pca_bridge_max_valley_from_bright &&
        bridgeCostRescueGapDensity <= probConfig.split_delayed_future_pca_bridge_max_gap_density &&
        finalAxisLen >= probConfig.split_delayed_future_pca_bridge_min_axis_length_scale * std::max(1.0f, srcMaxR) &&
        maxDaughterSeedDrift <= std::max(probConfig.split_delayed_future_pca_bridge_max_drift_abs,
                                        probConfig.split_delayed_future_pca_bridge_max_drift_scale * std::max(1.0f, srcMaxR));
    const bool costBackedCleanFuturePcaBridgeNearThresholdCostRescued =
        bestIsPcaBridgeOnly &&
        bestLabel == "bridge_primary" &&
        bridgeProposal != nullptr &&
        softOverlapAcceptedForCost &&
        costBackedCleanFuturePcaBridgeDensityWaivedActive &&
        probConfig.pca_bridge_future_window_enabled &&
        bridgeProposal->centerSnapApplied &&
        bridgeProposal->centerSnapUsedAlignedPairFallback &&
        bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
        bridgeProposal->windowBothDaughtersSupported >=
            std::max(2, probConfig.pca_bridge_future_window_min_both_daughter_support) &&
        bridgeProposal->windowMissingDaughterCount == 0 &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->windowBestMatchedMinBrightness >=
            probConfig.split_density_cost_backed_clean_future_min_future_brightness &&
        imageCostDiff <=
            -static_cast<double>(
                probConfig
                    .split_density_cost_backed_clean_future_min_image_gain_abs) &&
        costDiff <=
            static_cast<double>(
                probConfig
                    .split_density_cost_backed_clean_future_cost_fraction) *
                adaptiveThreshold &&
        overlapCostDiff <=
            static_cast<double>(
                probConfig
                    .split_density_cost_backed_clean_future_overlap_limit_fraction) *
                baselineImageCost &&
        bridgeCostRescueGapDensity <=
            probConfig
                .split_density_cost_backed_clean_future_max_gap_density &&
        bridgeCostRescueValleyFromBright <=
            probConfig
                .split_density_cost_backed_clean_future_max_valley_from_bright &&
        finalAxisLen >=
            probConfig
                .split_density_cost_backed_clean_future_min_axis_length_scale *
            std::max(1.0f, srcMaxR) &&
        maxDaughterSeedDrift <=
            std::max(
                probConfig
                    .split_density_cost_backed_clean_future_max_drift_abs,
                probConfig
                    .split_density_cost_backed_clean_future_max_drift_scale *
                    std::max(1.0f, srcMaxR));
    const bool highShapeRawPcaBridgeBioNearMissCostRescued =
        bestIsPcaBridgeOnly &&
        bridgeProposal != nullptr &&
        bridgeProposal->bioSeparationSoftRescued &&
        !bridgeProposal->centerSnapApplied &&
        !bridgeProposal->immediateFutureCenterBacked &&
        !bridgeProposal->centerSnapUsedAlignedPairFallback &&
        bridgeProposal->windowBothDaughtersSupported == 0 &&
        bridgeProposal->windowMissingDaughterCount <= static_cast<int>(probConfig.split_high_shape_raw_pca_bridge_max_missing_daughters) &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->parentShapeElongation >= probConfig.split_high_shape_raw_pca_bridge_min_parent_shape &&
        bridgeProposal->parentDistanceBalance >= probConfig.split_high_shape_raw_pca_bridge_min_parent_balance &&
        splitSoftGeometryPenaltyCost > 0.0 &&
        splitSoftGeometryPenaltyCost <= static_cast<double>(probConfig.split_high_shape_raw_pca_bridge_soft_penalty_fraction) * baselineImageCost &&
        imageCostDiff <= std::max(static_cast<double>(probConfig.split_high_shape_raw_pca_bridge_image_limit_abs),
                                  static_cast<double>(probConfig.split_high_shape_raw_pca_bridge_image_limit_fraction) * baselineImageCost) &&
        overlapCostDiff <= std::max(static_cast<double>(probConfig.split_high_shape_raw_pca_bridge_overlap_limit_abs),
                                    static_cast<double>(probConfig.split_high_shape_raw_pca_bridge_overlap_limit_fraction) * baselineImageCost) &&
        geometryAdjustedCostDiff <= std::max(static_cast<double>(probConfig.split_high_shape_raw_pca_bridge_geometry_limit_abs),
                                             static_cast<double>(probConfig.split_high_shape_raw_pca_bridge_geometry_limit_fraction) * baselineImageCost) &&
        costDiff <= std::max(static_cast<double>(probConfig.split_high_shape_raw_pca_bridge_cost_limit_abs),
                             static_cast<double>(probConfig.split_high_shape_raw_pca_bridge_cost_limit_fraction) * baselineImageCost) &&
        bridgeCostRescueValleyFromBright <= probConfig.split_high_shape_raw_pca_bridge_max_valley_from_bright &&
        bridgeCostRescueGapDensity <= probConfig.split_high_shape_raw_pca_bridge_max_gap_density &&
        finalAxisLen >= probConfig.split_high_shape_raw_pca_bridge_min_axis_length_scale * std::max(1.0f, srcMaxR) &&
        maxDaughterSeedDrift <= std::max(probConfig.split_high_shape_raw_pca_bridge_max_drift_abs,
                                        probConfig.split_high_shape_raw_pca_bridge_max_drift_scale * std::max(1.0f, srcMaxR));
    const bool highShapeRawRodSignalNearTieCostRescued =
        bestIsSignalCenterProposal &&
        bestLabel == "bridge_primary" &&
        bridgeProposal != nullptr &&
        bridgeProposal->gapStartBin <= static_cast<int>(probConfig.split_high_shape_rod_signal_gap_start_max) &&
        bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
        bridgeProposal->windowBothDaughtersSupported >= static_cast<int>(probConfig.split_high_shape_rod_signal_min_future_both) &&
        bridgeProposal->windowMissingDaughterCount <= static_cast<int>(probConfig.split_high_shape_rod_signal_max_missing_daughters) &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->parentShapeElongation >= probConfig.split_high_shape_rod_signal_min_parent_shape &&
        bridgeProposal->parentDistanceBalance >= probConfig.split_high_shape_rod_signal_min_parent_balance &&
        imageCostDiff <= -std::max(static_cast<double>(probConfig.split_high_shape_rod_signal_image_gain_abs),
                                   static_cast<double>(probConfig.split_high_shape_rod_signal_image_gain_fraction) * baselineImageCost) &&
        overlapCostDiff <= std::max(static_cast<double>(probConfig.split_high_shape_rod_signal_overlap_limit_abs),
                                    static_cast<double>(probConfig.split_high_shape_rod_signal_overlap_limit_fraction) * baselineImageCost) &&
        costDiff <= std::max(static_cast<double>(probConfig.split_high_shape_rod_signal_cost_limit_abs),
                             static_cast<double>(probConfig.split_high_shape_rod_signal_cost_limit_fraction) * baselineImageCost) &&
        geometryAdjustedCostDiff <= std::max(static_cast<double>(probConfig.split_high_shape_rod_signal_geometry_limit_abs),
                                             static_cast<double>(probConfig.split_high_shape_rod_signal_geometry_limit_fraction) * baselineImageCost) &&
        bridgeCostRescueGapDensity <= probConfig.split_high_shape_rod_signal_max_gap_density &&
        bridgeCostRescueValleyFromBright <= probConfig.split_high_shape_rod_signal_max_valley_from_bright &&
        finalAxisLen >= probConfig.split_high_shape_rod_signal_min_axis_length_scale * std::max(1.0f, srcMaxR) &&
        maxDaughterSeedDrift <= std::max(probConfig.split_high_shape_rod_signal_max_drift_abs,
                                        probConfig.split_high_shape_rod_signal_max_drift_scale * std::max(1.0f, srcMaxR));
    const float severeCostLongR = std::max({srcMajor, srcB, srcMinor});
    const float severeCostShortR = std::min({srcMajor, srcB, srcMinor});
    const float severeCostMidR = std::max(
        1e-3f, srcMajor + srcB + srcMinor - severeCostLongR - severeCostShortR);
    const float severeCostLongMidRatio = severeCostLongR / severeCostMidR;
    const float severeCostMidShortRatio =
        severeCostMidR / std::max(1e-3f, severeCostShortR);
    const bool severePostPcaRodFutureCostRescued =
        simulationConfig.celluniverse2_enabled &&
        probConfig.split_severe_post_pca_rod_future_cost_rescue_enabled &&
        bridgeProposalOnly &&
        (bestLabel == "bridge_primary" || bestLabel == "bridge_tip_alt") &&
        bridgeProposal != nullptr &&
        bridgeProposal->immediateFutureCenterBacked &&
        bridgeProposal->centerSnapUsedAlignedPairFallback &&
        bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
        bridgeProposal->windowBothDaughtersSupported >=
            std::max(
                1,
                probConfig
                    .split_severe_post_pca_rod_future_rescue_min_future_both) &&
        bridgeProposal->windowMissingDaughterCount == 0 &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->parentShapeElongation >=
            probConfig.split_severe_post_pca_rod_future_rescue_min_shape &&
        severeCostLongMidRatio >=
            probConfig
                .split_severe_post_pca_rod_future_rescue_min_long_mid_ratio &&
        severeCostMidShortRatio <=
            probConfig
                .split_severe_post_pca_rod_future_rescue_max_mid_short_ratio &&
        bridgeProposal->parentDistanceBalance >=
            probConfig
                .split_severe_post_pca_rod_future_rescue_min_parent_balance &&
        bridgeCostRescueValleyFromBright <=
            probConfig
                .split_severe_post_pca_rod_future_cost_rescue_max_valley_from_bright &&
        bridgeCostRescueGapDensity <=
            probConfig
                .split_severe_post_pca_rod_future_cost_rescue_max_gap_density &&
        maxDaughterSeedDrift <=
            probConfig
                .split_severe_post_pca_rod_future_cost_rescue_max_drift_scale *
                std::max(1.0f, srcMaxR) &&
        costDiff <=
            static_cast<double>(
                probConfig.split_severe_post_pca_rod_future_cost_rescue_max_cost_abs) &&
        imageCostDiff <=
            static_cast<double>(
                probConfig
                    .split_severe_post_pca_rod_future_cost_rescue_max_image_diff_abs) &&
        overlapCostDiff <=
            static_cast<double>(
                probConfig
                    .split_severe_post_pca_rod_future_cost_rescue_max_overlap_diff_abs);
    const bool cellUniverse3WindowPenaltyPcaCostRescued =
        simulationConfig.celluniverse3_enabled &&
        probConfig.celluniverse3_window_soft_penalty_cost_rescue_enabled &&
        bestIsPcaBridgeOnly &&
        bestLabel == "bridge_primary" &&
        bridgeProposal != nullptr &&
        bridgeProposal->centerSnapApplied &&
        bridgeProposal->centerSnapUsedAlignedPairFallback &&
        !bridgeProposal->immediateFutureCenterBacked &&
        bridgeProposal->windowBothDaughtersSupported >=
            std::max(
                1,
                probConfig
                    .celluniverse3_window_soft_penalty_cost_rescue_min_future_both) &&
        bridgeProposal->windowMissingDaughterCount <=
            std::max(
                0,
                probConfig
                    .celluniverse3_window_soft_penalty_cost_rescue_max_missing_daughters) &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->windowBestMatchedMinBrightness >=
            probConfig
                .celluniverse3_window_soft_penalty_cost_rescue_min_future_brightness &&
        bridgeProposal->parentShapeElongation >=
            probConfig
                .celluniverse3_window_soft_penalty_cost_rescue_min_parent_shape &&
        bridgeProposal->parentDistanceBalance >=
            probConfig
                .celluniverse3_window_soft_penalty_cost_rescue_min_parent_balance &&
        costDiff <= 0.0 &&
        imageCostDiff <= 0.0 &&
        geometryAdjustedCostDiff <=
            static_cast<double>(
                probConfig
                    .celluniverse3_window_soft_penalty_cost_rescue_max_geometry_fraction) *
                baselineImageCost &&
        overlapCostDiff <=
            static_cast<double>(
                probConfig
                    .celluniverse3_window_soft_penalty_cost_rescue_max_overlap_fraction) *
                baselineImageCost &&
        finalAxisLen >=
            probConfig
                .celluniverse3_window_soft_penalty_cost_rescue_min_axis_length_scale *
            std::max(1.0f, srcMaxR) &&
        maxDaughterSeedDrift <=
            probConfig
                .celluniverse3_window_soft_penalty_cost_rescue_max_drift_scale *
            std::max(1.0f, srcMaxR) &&
        bridgeCostRescueValleyFromBright <=
            probConfig
                .celluniverse3_window_soft_penalty_cost_rescue_max_valley_from_bright &&
        bridgeCostRescueGapDensity <=
            probConfig
                .celluniverse3_window_soft_penalty_cost_rescue_max_gap_density;
    const bool cellUniverse3WindowUnionSoftPenaltyCostRescued =
        simulationConfig.celluniverse3_enabled &&
        probConfig
            .celluniverse3_window_union_soft_penalty_cost_rescue_enabled &&
        bestIsSignalCenterProposal &&
        bestLabel == "bridge_primary" &&
        bridgeProposal != nullptr &&
        bridgeProposal->signalCenterScore >= 0.0f &&
        !bridgeProposal->cellUniverse3MapProposal &&
        bridgeProposal->cellUniverse3MapPriorEvaluated &&
        bridgeProposal->cellUniverse3MapPriorConfident &&
        bridgeProposal->cellUniverse3MapUSupportD1 >=
            probConfig
                .celluniverse3_window_union_soft_penalty_cost_rescue_min_u_support &&
        bridgeProposal->cellUniverse3MapUSupportD2 >=
            probConfig
                .celluniverse3_window_union_soft_penalty_cost_rescue_min_u_support &&
        bridgeProposal->cellUniverse3MapRegionPenalty <=
            probConfig
                .celluniverse3_window_union_soft_penalty_cost_rescue_max_region_penalty &&
        bridgeProposal->windowBothDaughtersSupported >=
            std::max(
                1,
                probConfig
                    .celluniverse3_window_soft_penalty_cost_rescue_min_future_both) &&
        bridgeProposal->windowMissingDaughterCount <=
            std::max(
                0,
                probConfig
                    .celluniverse3_window_soft_penalty_cost_rescue_max_missing_daughters) &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->windowBestMatchedMinBrightness >=
            probConfig
                .celluniverse3_window_soft_penalty_cost_rescue_min_future_brightness &&
        bridgeProposal->parentShapeElongation >=
            probConfig
                .celluniverse3_window_soft_penalty_cost_rescue_min_parent_shape &&
        bridgeProposal->parentDistanceBalance >=
            probConfig
                .celluniverse3_window_soft_penalty_cost_rescue_min_parent_balance &&
        costDiff <= 0.0 &&
        imageCostDiff <= 0.0 &&
        geometryAdjustedCostDiff <=
            static_cast<double>(
                probConfig
                    .celluniverse3_window_soft_penalty_cost_rescue_max_geometry_fraction) *
                baselineImageCost &&
        overlapCostDiff <=
            static_cast<double>(
                probConfig
                    .celluniverse3_window_union_soft_penalty_cost_rescue_max_overlap_fraction) *
                baselineImageCost &&
        finalAxisLen >=
            probConfig
                .celluniverse3_window_soft_penalty_cost_rescue_min_axis_length_scale *
            std::max(1.0f, srcMaxR) &&
        maxDaughterSeedDrift <=
            probConfig
                .celluniverse3_window_soft_penalty_cost_rescue_max_drift_scale *
            std::max(1.0f, srcMaxR) &&
        bridgeCostRescueValleyFromBright <=
            probConfig
                .celluniverse3_window_soft_penalty_cost_rescue_max_valley_from_bright &&
        bridgeCostRescueGapDensity <=
            probConfig
                .celluniverse3_window_soft_penalty_cost_rescue_max_gap_density;
    const double cellUniverse3WindowMapPrimaryCostLimit =
        static_cast<double>(std::max(
            0.0f,
            probConfig
                .celluniverse3_window_map_primary_cost_rescue_max_cost_fraction)) *
        baselineImageCost;
    const double cellUniverse3WindowMapPrimaryOverlapLimit =
        static_cast<double>(std::max(
            0.0f,
            probConfig
                .celluniverse3_window_map_primary_cost_rescue_max_overlap_fraction)) *
        baselineImageCost;
    const bool cellUniverse3WindowMapPrimaryHasFutureSupport =
        bridgeProposal != nullptr &&
        (bridgeProposal->windowBothDaughtersSupported > 0 ||
         bridgeProposal->windowImmediateBothDaughtersSupported > 0);
    const float cellUniverse3WindowMapPrimaryCostMinAxisAlignment =
        (bridgeProposal != nullptr &&
         bridgeProposal->cellUniverse3MapOverlapCenterProposal &&
         cellUniverse3WindowMapPrimaryHasFutureSupport)
            ? std::max(
                  0.0f,
                  probConfig
                      .celluniverse3_window_map_overlap_cost_rescue_min_axis_alignment)
            : std::max(
                  0.0f,
                  probConfig
                      .celluniverse3_window_map_primary_cost_rescue_min_axis_alignment);
    const bool cellUniverse3WindowMapPrimaryCostRescued =
        simulationConfig.celluniverse3_enabled &&
        probConfig.celluniverse3_window_map_primary_cost_rescue_enabled &&
        bridgeProposalOnly &&
        bestLabel == "bridge_primary" &&
        bridgeProposal != nullptr &&
        bridgeProposal->centerSnapApplied &&
        bridgeProposal->cellUniverse3MapProposal &&
        bridgeProposal->cellUniverse3MapPriorConfident &&
        bridgeProposal->cellUniverse3MapUSupportD1 >=
            probConfig
                .celluniverse3_window_map_primary_cost_rescue_min_u_support &&
        bridgeProposal->cellUniverse3MapUSupportD2 >=
            probConfig
                .celluniverse3_window_map_primary_cost_rescue_min_u_support &&
        bridgeProposal->cellUniverse3MapRegionPenalty <=
            probConfig
                .celluniverse3_window_map_primary_cost_rescue_max_region_penalty &&
        std::abs(bridgeProposal->cellUniverse3MapAxisAlignment) >=
            cellUniverse3WindowMapPrimaryCostMinAxisAlignment &&
        bridgeProposal->parentDistanceBalance >=
            probConfig
                .celluniverse3_window_map_primary_cost_rescue_min_parent_balance &&
        costDiff <= cellUniverse3WindowMapPrimaryCostLimit &&
        imageCostDiff <=
            static_cast<double>(
                probConfig
                    .celluniverse3_window_map_primary_cost_rescue_max_image_diff_abs) &&
        overlapCostDiff <= cellUniverse3WindowMapPrimaryOverlapLimit &&
        finalAxisLen >=
            probConfig
                .celluniverse3_window_map_primary_cost_rescue_min_axis_length_scale *
            std::max(1.0f, srcMaxR) &&
        maxDaughterSeedDrift <=
            probConfig
                .celluniverse3_window_map_primary_cost_rescue_max_drift_scale *
            std::max(1.0f, srcMaxR) &&
        bridgeCostRescueValleyFromBright <=
            probConfig
                .celluniverse3_window_map_primary_cost_rescue_max_valley_from_bright &&
        bridgeCostRescueGapDensity <=
            probConfig
                .celluniverse3_window_map_primary_cost_rescue_max_gap_density;
    const bool cellUniverse3WindowSoftPenaltyCostRescued =
        cellUniverse3WindowPenaltyPcaCostRescued ||
        cellUniverse3WindowUnionSoftPenaltyCostRescued ||
        cellUniverse3WindowMapPrimaryCostRescued;
    const double cellUniverse3SignalCenterFutureCostLimit =
        std::max(
            static_cast<double>(
                probConfig
                    .celluniverse3_signal_center_future_cost_rescue_max_cost_abs),
            static_cast<double>(
                probConfig
                    .celluniverse3_signal_center_future_cost_rescue_max_cost_fraction) *
                baselineImageCost);
    const bool cellUniverse3SignalCenterFutureCostContext =
        simulationConfig.celluniverse3_enabled &&
        probConfig.celluniverse3_signal_center_future_cost_rescue_enabled &&
        probConfig.celluniverse3_signal_center_future_position_lock_enabled &&
        bridgeProposalOnly &&
        bestLabel == "bridge_primary" &&
        bridgeProposal != nullptr &&
        bridgeProposal->signalCenterScore >= 0.0f &&
        !bridgeProposal->centerSnapUsedAlignedPairFallback &&
        bridgeProposal->windowImmediateBothDaughtersSupported > 0 &&
        bridgeProposal->windowBothDaughtersSupported >=
            std::max(
                1,
                probConfig
                    .celluniverse3_signal_center_future_position_lock_min_future_both) &&
        bridgeProposal->windowMissingDaughterCount <=
            std::max(
                0,
                probConfig
                    .celluniverse3_signal_center_future_position_lock_max_missing) &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->windowBestMatchedMinBrightness >=
            probConfig
                .celluniverse3_signal_center_future_position_lock_min_future_brightness &&
        bridgeProposal->parentShapeElongation >=
            probConfig
                .celluniverse3_signal_center_future_position_lock_min_parent_shape &&
        bridgeProposal->parentDistanceBalance >=
            probConfig
                .celluniverse3_signal_center_future_position_lock_min_parent_balance &&
        bridgeProposal->centerSnapMaxSeedDistance <=
            std::max(
                0.0f,
                probConfig
                    .celluniverse3_signal_center_future_position_lock_max_snap_distance) &&
        bridgeProposal->bioSeparationRequired > 0.0f &&
        bridgeProposal->bioSeparationObserved >=
            std::max(
                0.0f,
                probConfig
                    .celluniverse3_signal_center_future_position_lock_min_sep_fraction) *
                bridgeProposal->bioSeparationRequired &&
        bridgeProposal->signalCenterSeparationRatio >=
            probConfig
                .celluniverse3_signal_center_future_position_lock_min_sep_ratio &&
        std::abs(bridgeProposal->signalCenterAxisAlignment) >=
            probConfig
                .celluniverse3_signal_center_future_position_lock_min_axis_alignment;
    const bool cellUniverse3SignalCenterFuturePositiveCostEvidenceOk =
        costDiff <= 0.0 ||
        (bridgeProposal != nullptr &&
         bridgeProposal->parentShapeElongation >=
             probConfig
                 .celluniverse3_signal_center_future_cost_rescue_positive_cost_min_parent_shape) ||
        (bridgeCostRescueValleyFromBright <=
             probConfig
                 .celluniverse3_signal_center_future_cost_rescue_positive_cost_max_valley_from_bright &&
         bridgeCostRescueGapDensity <=
             probConfig
                 .celluniverse3_signal_center_future_cost_rescue_positive_cost_max_gap_density);
    const bool cellUniverse3SignalCenterFutureCostRescued =
        cellUniverse3SignalCenterFutureCostContext &&
        costDiff <= cellUniverse3SignalCenterFutureCostLimit &&
        imageCostDiff <=
            static_cast<double>(
                probConfig
                    .celluniverse3_signal_center_future_cost_rescue_max_image_diff_abs) &&
        overlapCostDiff <=
            static_cast<double>(
                probConfig
                    .celluniverse3_signal_center_future_cost_rescue_max_overlap_diff_abs) &&
        finalAxisLen >=
            probConfig
                .celluniverse3_signal_center_future_cost_rescue_min_axis_length_scale *
            std::max(1.0f, srcMaxR) &&
        maxDaughterSeedDrift <=
            probConfig
                .celluniverse3_signal_center_future_cost_rescue_max_drift_scale *
            std::max(1.0f, srcMaxR) &&
        bridgeCostRescueValleyFromBright <=
            probConfig
                .celluniverse3_signal_center_future_cost_rescue_max_valley_from_bright &&
        bridgeCostRescueGapDensity <=
            probConfig
                .celluniverse3_signal_center_future_cost_rescue_max_gap_density &&
        cellUniverse3SignalCenterFuturePositiveCostEvidenceOk;
    const bool cellUniverse3DelayedMissingDaughterCostRescued =
        simulationConfig.celluniverse3_enabled &&
        probConfig.celluniverse3_delayed_missing_daughter_cost_rescue_enabled &&
        bridgeProposalOnly &&
        bestLabel == "bridge_primary" &&
        bridgeProposal != nullptr &&
        bridgeProposal->cellUniverse3DelayedMissingDaughter &&
        bridgeProposal->centerSnapApplied &&
        bridgeProposal->windowBothDaughtersSupported >=
            std::max(
                1,
                probConfig
                    .celluniverse3_delayed_missing_daughter_cost_rescue_min_future_both) &&
        bridgeProposal->windowMissingDaughterCount <=
            std::max(
                0,
                probConfig
                    .celluniverse3_delayed_missing_daughter_cost_rescue_max_missing) &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->windowBestMatchedMinBrightness >=
            probConfig
                .celluniverse3_delayed_missing_daughter_cost_rescue_min_future_brightness &&
        imageCostDiff <=
            static_cast<double>(
                probConfig
                    .celluniverse3_delayed_missing_daughter_cost_rescue_max_image_diff_abs) &&
        overlapCostDiff <=
            static_cast<double>(
                probConfig
                    .celluniverse3_delayed_missing_daughter_cost_rescue_max_overlap_diff_abs) &&
        costDiff <=
            static_cast<double>(
                probConfig
                    .celluniverse3_delayed_missing_daughter_cost_rescue_max_total_diff_abs) &&
        bridgeCostRescueValleyFromBright <=
            probConfig
                .celluniverse3_delayed_missing_daughter_cost_rescue_max_valley_from_bright &&
        bridgeCostRescueGapDensity <=
            probConfig
                .celluniverse3_delayed_missing_daughter_cost_rescue_max_gap_density &&
        maxDaughterSeedDrift <=
            std::max(
                probConfig
                    .celluniverse3_delayed_missing_daughter_cost_rescue_max_drift_abs,
                probConfig
                    .celluniverse3_delayed_missing_daughter_cost_rescue_max_drift_scale *
                    std::max(1.0f, srcMaxR));
    const bool cellUniverse3OutsideFuturePairCostRescued =
        simulationConfig.celluniverse3_enabled &&
        bestOutsideFutureTunnelPass &&
        bridgeProposalOnly &&
        bestLabel == "bridge_primary" &&
        bridgeProposal != nullptr &&
        costDiff <= -adaptiveThreshold &&
        imageCostDiff <= -adaptiveThreshold &&
        overlapCostDiff <=
            static_cast<double>(
                std::max(0.0f,
                         probConfig
                             .celluniverse3_window_map_primary_cost_rescue_max_overlap_fraction)) *
                baselineImageCost &&
        bridgeCostRescueValleyFromBright <=
            probConfig
                .celluniverse3_window_map_primary_cost_rescue_max_valley_from_bright &&
        bridgeCostRescueGapDensity <=
            probConfig
                .celluniverse3_window_map_primary_cost_rescue_max_gap_density;
    const double disconnectedFarPairRawGainRequired =
        std::max(0.0,
                 static_cast<double>(
                     probConfig
                         .signal_center_disconnected_far_pair_cost_rescue_min_raw_gain_fraction)) *
        adaptiveThreshold;
    const double disconnectedFarPairOverlapLimit =
        std::max(0.0,
                 static_cast<double>(
                     probConfig
                         .signal_center_disconnected_far_pair_cost_rescue_max_overlap_fraction)) *
        baselineImageCost;
    const bool cellUniverse3DisconnectedFarPairCostRescued =
        simulationConfig.celluniverse3_enabled &&
        probConfig.signal_center_disconnected_far_pair_cost_rescue_enabled &&
        bridgeProposalOnly &&
        bestLabel == "bridge_primary" &&
        bridgeProposal != nullptr &&
        bridgeProposal->signalCenterDisconnectedFarPairRescue &&
        bridgeProposal->centerSnapApplied &&
        bridgeProposal->windowBothDaughtersSupported >= 1 &&
        bridgeProposal->windowMissingDaughterCount <= 1 &&
        bridgeProposal->windowParentPersists == 0 &&
        bridgeProposal->windowBestMatchedMinBrightness >=
            probConfig.signal_center_disconnected_far_pair_min_brightness &&
        bridgeProposal->cellUniverse3MapRegionPenalty <= 2.0f &&
        bridgeProposal->signalCenterSeparationRatio >=
            probConfig.signal_center_disconnected_far_pair_min_sep_ratio &&
        costDiff <= 0.0 &&
        imageCostDiff <= -disconnectedFarPairRawGainRequired &&
        overlapCostDiff <= disconnectedFarPairOverlapLimit &&
        maxDaughterSeedDrift <=
            std::max(0.0f, 0.35f * std::max(1.0f, srcMaxR)) &&
        bridgeCostRescueValleyFromBright <=
            probConfig.signal_center_disconnected_far_pair_cost_rescue_max_valley_from_bright &&
        bridgeCostRescueGapDensity <=
            probConfig.signal_center_disconnected_far_pair_cost_rescue_max_gap_density;
    const bool futureWindowCostRescued =
        futureWindowStrictCostRescued || futureWindowSoftCostRescued ||
        futureWindowStrongCostRescued || futureWindowGeometryCostRescued ||
        twoFrameFutureBridgeCostRescued ||
        futureWindowSoftPenaltyRawGainRescued ||
        oneFrameBrightPcaBridgeSoftPenaltyRescued ||
        oneFrameFuturePcaBridgeSoftPenaltyRawGainRescued ||
        futureSignalSoftPenaltyRawGainRescued ||
        futureSignalCleanCenterSoftPenaltyRescued ||
        futureSignalNearThresholdCostRescued ||
        futureSignalCleanValleyCostRescued ||
        futureSignalLockedOverlapCostRescued ||
        futureSignalBorderlineMovedCostRescued ||
        futureRodTipCleanGapCostRescued ||
        futureRodTipPrimaryCostRescued ||
        oneFrameFutureBridgeCostRescued ||
        oneFramePcaBridgeOverlapCostRescued ||
        cleanFuturePcaBridgeOverlapCostRescued ||
        alignedFuturePcaBridgeImageGainCostRescued ||
        currentLockedCleanPcaBridgeNearThresholdCostRescued ||
        softOverlapFuturePcaBridgeCostRescued ||
        stableFuturePcaBridgeNearThresholdCostRescued ||
        lockedCleanFuturePcaBridgeCostRescued ||
        crowdedGeneralCleanFuturePcaBridgeCostRescued ||
        oneFrameAlignedLockedFuturePcaBridgeCostRescued ||
        twoFrameAlignedLockedFuturePcaBridgeCostRescued ||
        lockedExactFutureCenterBridgeCostRescued ||
        cleanRodTipContinuationOverlapCostRescued ||
        dimExactFutureSignalOverlapCostRescued ||
        bridgeAxisPlaceFutureImageGainCostRescued ||
        bridgeAxisPlaceCleanNearThresholdCostRescued ||
        cellUniverse3AxisPlaceNearThresholdCostRescued ||
        longRawPcaBridgeNearThresholdCostRescued ||
        asymmetricRawPcaBridgeNearThresholdCostRescued ||
        delayedFuturePcaBridgeNearMissCostRescued ||
        costBackedCleanFuturePcaBridgeNearThresholdCostRescued ||
        highShapeRawPcaBridgeBioNearMissCostRescued ||
        highShapeRawRodSignalNearTieCostRescued ||
        severePostPcaRodFutureCostRescued ||
        cellUniverse3WeakPcaBridgeCostRescued ||
        cellUniverse3WindowSoftPenaltyCostRescued ||
        cellUniverse3SignalCenterFutureCostRescued ||
        cellUniverse3DelayedMissingDaughterCostRescued ||
        cellUniverse3OutsideFuturePairCostRescued ||
        cellUniverse3DisconnectedFarPairCostRescued;
    double acceptedCostDiff = costDiff;
    if (bridgeCostRescued) {
        acceptedCostDiff = -std::max(1.0, adaptiveThreshold);
        std::cout << "[Split Cost Rescue] " << parentName
                  << " rawDiff=" << costDiff
                  << " geometryAdjustedDiff=" << geometryAdjustedCostDiff
                  << " splitSoftPenalty=" << splitSoftGeometryPenaltyCost
                  << " reportedDiff=" << acceptedCostDiff
                  << " maxPositive=" << bridgeRescueLimit
                  << " baselineImageCost=" << baselineImageCost
                  << " valleyFromBright=" << bridgeCostRescueValleyFromBright
                  << " worstValleyRatio=" << bridgeCostRescueWorstValleyRatio
                  << " maxValley=" << probConfig.split_bridge_cost_rescue_max_valley_ratio
                  << " gapDensity=" << bridgeCostRescueGapDensity
                  << " maxGapDensity=" << probConfig.split_bridge_cost_rescue_max_gap_density
                  << " edgeBright=" << bridgeCostRescueEdgeBright
                  << " bestIdx=" << bestIdx
                  << " bestLabel=" << bestLabel
                  << std::endl;
    }
    if (futureWindowCostRescued) {
        acceptedCostDiff = -std::max(1.0, adaptiveThreshold);
        std::cout << "[Split Future Window Cost Gate] " << parentName
                  << " totalDiff=" << costDiff
                  << " geometryAdjustedDiff=" << geometryAdjustedCostDiff
                  << " splitSoftPenalty=" << splitSoftGeometryPenaltyCost
                  << " imageDiff=" << imageCostDiff
                  << " overlapDiff=" << overlapCostDiff
                  << " reportedDiff=" << acceptedCostDiff
                  << " parentShapeElong="
                  << bridgeProposal->parentShapeElongation
                  << " minParentShape="
                  << probConfig.pca_bridge_future_window_min_parent_shape_for_cost_rescue
                  << " softRescue=" << (futureWindowSoftCostRescued ? 1 : 0)
                  << " strictRescue=" << (futureWindowStrictCostRescued ? 1 : 0)
                  << " strongRescue=" << (futureWindowStrongCostRescued ? 1 : 0)
                  << " geometryRescue="
                  << (futureWindowGeometryCostRescued ? 1 : 0)
                  << " twoFrameBridgeRescue="
                  << (twoFrameFutureBridgeCostRescued ? 1 : 0)
                  << " outsideFuturePairRescue="
                  << (cellUniverse3OutsideFuturePairCostRescued ? 1 : 0)
                  << " disconnectedFarPairRescue="
                  << (cellUniverse3DisconnectedFarPairCostRescued ? 1 : 0)
                  << " disconnectedFarPairRawGainRequired="
                  << disconnectedFarPairRawGainRequired
                  << " softPenaltyRawGainRescue="
                  << (futureWindowSoftPenaltyRawGainRescued ? 1 : 0)
                  << " softPenaltyRawGainLimit="
                  << futureSoftPenaltyRawGainLimit
                  << " oneFrameBrightPcaBridgeSoftPenaltyRescue="
                  << (oneFrameBrightPcaBridgeSoftPenaltyRescued ? 1 : 0)
                  << " oneFrameFuturePcaBridgeSoftPenaltyRawGainRescue="
                  << (oneFrameFuturePcaBridgeSoftPenaltyRawGainRescued ? 1 : 0)
                  << " signalSoftPenaltyRawGainRescue="
                  << (futureSignalSoftPenaltyRawGainRescued ? 1 : 0)
                  << " signalSoftPenaltyRawGainLimit="
                  << futureSignalSoftPenaltyRawGainLimit
                  << " signalCleanCenterSoftPenaltyRescue="
                  << (futureSignalCleanCenterSoftPenaltyRescued ? 1 : 0)
                  << " signalNearThresholdRescue="
                  << (futureSignalNearThresholdCostRescued ? 1 : 0)
                  << " signalBorderlineMovedRescue="
                  << (futureSignalBorderlineMovedCostRescued ? 1 : 0)
                  << " signalBorderlineMovedLimit="
                  << futureSignalBorderlineMovedCostLimit
                  << " futureRodTipCleanGapRescue="
                  << (futureRodTipCleanGapCostRescued ? 1 : 0)
                  << " futureRodTipPrimaryRescue="
                  << (futureRodTipPrimaryCostRescued ? 1 : 0)
                  << " futureRodTipPrimaryCostLimit="
                  << futureRodTipPrimaryCostLimit
                  << " futureRodTipPrimaryImageLimit="
                  << futureRodTipPrimaryImageCostLimit
                  << " futureRodTipPrimaryOverlapLimit="
                  << futureRodTipPrimaryOverlapCostLimit
                  << " maxDaughterSeedDrift=" << maxDaughterSeedDrift
                  << " oneFrameBridgeRescue="
                  << (oneFrameFutureBridgeCostRescued ? 1 : 0)
                  << " oneFramePcaBridgeOverlapRescue="
                  << (oneFramePcaBridgeOverlapCostRescued ? 1 : 0)
                  << " cleanFuturePcaBridgeOverlapRescue="
                  << (cleanFuturePcaBridgeOverlapCostRescued ? 1 : 0)
                  << " cleanFuturePcaBridgeOverlapRescueLimit="
                  << cleanFuturePcaBridgeOverlapRescueLimit
                  << " cleanFuturePcaBridgeImageGainRequired="
                  << cleanFuturePcaBridgeImageGainRequired
                  << " cleanFuturePcaBridgeOverlapPenaltyLimit="
                  << cleanFuturePcaBridgeOverlapPenaltyLimit
                  << " alignedFuturePcaBridgeImageGainRescue="
                  << (alignedFuturePcaBridgeImageGainCostRescued ? 1 : 0)
                  << " alignedFuturePcaBridgeImageGainRequired="
                  << alignedFuturePcaBridgeImageGainRequired
                  << " alignedFuturePcaBridgeOverlapPenaltyLimit="
                  << alignedFuturePcaBridgeOverlapPenaltyLimit
                  << " currentLockedCleanPcaBridgeNearThresholdRescue="
                  << (currentLockedCleanPcaBridgeNearThresholdCostRescued ? 1 : 0)
                  << " currentLockedPcaBridgeNearThresholdGainRequired="
                  << currentLockedPcaBridgeNearThresholdGainRequired
                  << " softOverlapFutureBridgeRescue="
                  << (softOverlapFuturePcaBridgeCostRescued ? 1 : 0)
                  << " softOverlapFutureCostLimit="
                  << softOverlapFuturePcaBridgeCostLimit
                  << " stableFutureNearThresholdBridgeRescue="
                  << (stableFuturePcaBridgeNearThresholdCostRescued ? 1 : 0)
                  << " stableFutureNearThresholdGainRequired="
                  << stableFuturePcaBridgeNearThresholdGainRequired
                  << " stableFutureNearThresholdOverlapLimit="
                  << stableFuturePcaBridgeNearThresholdOverlapLimit
                  << " lockedCleanFuturePcaBridgeRescue="
                  << (lockedCleanFuturePcaBridgeCostRescued ? 1 : 0)
                  << " crowdedGeneralCleanFuturePcaBridgeRescue="
                  << (crowdedGeneralCleanFuturePcaBridgeCostRescued ? 1 : 0)
                  << " crowdedGeneralCleanFutureCostLimit="
                  << crowdedGeneralCleanFuturePcaBridgeCostLimit
                  << " crowdedGeneralCleanFutureGeometryLimit="
                  << crowdedGeneralCleanFuturePcaBridgeGeometryLimit
                  << " crowdedGeneralCleanFutureOverlapLimit="
                  << crowdedGeneralCleanFuturePcaBridgeOverlapLimit
                  << " crowdedGeneralCleanFutureImageGainRequired="
                  << generalCleanFutureImageGainRequired
                  << " crowdedGeneralCleanFutureCrowdingRelaxation="
                  << generalCleanFutureCrowdingRelaxation
                  << " savedNonTrashCellCount=" << savedNonTrashCellCount
                  << " oneFrameAlignedLockedFuturePcaBridgeRescue="
                  << (oneFrameAlignedLockedFuturePcaBridgeCostRescued ? 1 : 0)
                  << " twoFrameAlignedLockedFuturePcaBridgeRescue="
                  << (twoFrameAlignedLockedFuturePcaBridgeCostRescued ? 1 : 0)
                  << " twoFrameAlignedLockedCostLimit="
                  << twoFrameAlignedLockedPcaBridgeCostLimit
                  << " twoFrameAlignedLockedOverlapLimit="
                  << twoFrameAlignedLockedPcaBridgeOverlapLimit
                  << " twoFrameAlignedLockedImageGainRequired="
                  << twoFrameAlignedLockedPcaBridgeImageGainRequired
                  << " oneFrameAlignedLockedCostLimit="
                  << oneFrameAlignedLockedPcaBridgeCostLimit
                  << " oneFrameAlignedLockedOverlapLimit="
                  << oneFrameAlignedLockedPcaBridgeOverlapLimit
                  << " oneFrameAlignedLockedImageGainRequired="
                  << oneFrameAlignedLockedPcaBridgeImageGainRequired
                  << " lockedExactFutureCenterBridgeRescue="
                  << (lockedExactFutureCenterBridgeCostRescued ? 1 : 0)
                  << " dimExactFutureSignalOverlapRescue="
                  << (dimExactFutureSignalOverlapCostRescued ? 1 : 0)
                  << " bridgeAxisPlaceFutureImageGainRescue="
                  << (bridgeAxisPlaceFutureImageGainCostRescued ? 1 : 0)
                  << " bridgeAxisPlaceFutureImageGainRequired="
                  << bridgeAxisPlaceFutureImageGainRequired
                  << " bridgeAxisPlaceFutureOverlapPenaltyLimit="
                  << bridgeAxisPlaceFutureOverlapPenaltyLimit
                  << " bridgeAxisPlaceCleanNearThresholdRescue="
                  << (bridgeAxisPlaceCleanNearThresholdCostRescued ? 1 : 0)
                  << " bridgeAxisPlaceNearThresholdRequired="
                  << bridgeAxisPlaceNearThresholdRequired
                  << " cellUniverse3AxisPlaceNearThresholdRescue="
                  << (cellUniverse3AxisPlaceNearThresholdCostRescued ? 1 : 0)
                  << " cellUniverse3AxisPlaceCostShortfall="
                  << cellUniverse3AxisPlaceCostShortfall
                  << " cellUniverse3AxisPlaceShortfallLimit="
                  << cellUniverse3AxisPlaceCostShortfallLimit
                  << " longRawPcaBridgeNearThresholdRescue="
                  << (longRawPcaBridgeNearThresholdCostRescued ? 1 : 0)
                  << " asymmetricRawPcaBridgeNearThresholdRescue="
                  << (asymmetricRawPcaBridgeNearThresholdCostRescued ? 1 : 0)
                  << " delayedFuturePcaBridgeNearMissRescue="
                  << (delayedFuturePcaBridgeNearMissCostRescued ? 1 : 0)
                  << " costBackedCleanFutureNearThresholdRescue="
                  << (costBackedCleanFuturePcaBridgeNearThresholdCostRescued
                          ? 1
                          : 0)
                  << " highShapeRawPcaBridgeBioNearMissRescue="
                  << (highShapeRawPcaBridgeBioNearMissCostRescued ? 1 : 0)
                  << " highShapeRawRodSignalNearTieRescue="
                  << (highShapeRawRodSignalNearTieCostRescued ? 1 : 0)
                  << " severePostPcaRodFutureCostRescue="
                  << (severePostPcaRodFutureCostRescued ? 1 : 0)
                  << " cellUniverse3WeakPcaBridgeCostRescue="
                  << (cellUniverse3WeakPcaBridgeCostRescued ? 1 : 0)
                  << " cellUniverse3WeakPcaBridgeImageCostOk="
                  << (cellUniverse3WeakPcaBridgeImageCostOk ? 1 : 0)
                  << " cellUniverse3WindowSoftPenaltyCostRescue="
                  << (cellUniverse3WindowSoftPenaltyCostRescued ? 1 : 0)
                  << " cellUniverse3WindowMapPrimaryCostRescue="
                  << (cellUniverse3WindowMapPrimaryCostRescued ? 1 : 0)
                  << " cellUniverse3WindowMapPrimaryCostLimit="
                  << cellUniverse3WindowMapPrimaryCostLimit
                  << " cellUniverse3WindowMapPrimaryOverlapLimit="
                  << cellUniverse3WindowMapPrimaryOverlapLimit
                  << " cellUniverse3SignalCenterFutureCostRescue="
                  << (cellUniverse3SignalCenterFutureCostRescued ? 1 : 0)
                  << " cellUniverse3SignalCenterPositiveCostEvidenceOk="
                  << (cellUniverse3SignalCenterFuturePositiveCostEvidenceOk ? 1 : 0)
                  << " cellUniverse3SignalCenterFutureCostLimit="
                  << cellUniverse3SignalCenterFutureCostLimit
                  << " cellUniverse3DelayedMissingDaughterCostRescue="
                  << (cellUniverse3DelayedMissingDaughterCostRescued ? 1 : 0)
                  << " cellUniverse3DisconnectedFarPairCostRescue="
                  << (cellUniverse3DisconnectedFarPairCostRescued ? 1 : 0)
                  << " severePostPcaRodFutureLongMidRatio="
                  << severeCostLongMidRatio
                  << " severePostPcaRodFutureMidShortRatio="
                  << severeCostMidShortRatio
                  << " oneFrameBridgeCostLimit=" << oneFrameBridgeCostLimit
                  << " centerSnapMaxSeedDistance="
                  << bridgeProposal->centerSnapMaxSeedDistance
                  << " cleanSnapDistanceLimit=" << cleanPcaBridgeSnapDistanceLimit
                  << " cleanFuturePairSnapLimit="
                  << cleanPcaBridgeFuturePairSnapLimit
                  << " parentDistBalance="
                  << bridgeProposal->parentDistanceBalance
                  << " alignedPairFallback="
                  << (bridgeProposal->centerSnapUsedAlignedPairFallback ? 1 : 0)
                  << " futureBoth="
                  << bridgeProposal->windowBothDaughtersSupported
                  << " futureImmediate="
                  << bridgeProposal->windowImmediateBothDaughtersSupported
                  << " futureMissing="
                  << bridgeProposal->windowMissingDaughterCount
                  << " parentPersists="
                  << bridgeProposal->windowParentPersists
                  << " bestIdx=" << bestIdx
                  << " bestLabel=" << bestLabel
                  << std::endl;
    }

    const double lumenMaxPositiveCost =
        useCellLumenCostGate
            ? static_cast<double>(std::max(0.0f, lumenMaxPositiveCostFraction)) * baselineImageCost
            : -adaptiveThreshold;
    const bool useCellLumenImageGate =
        useCellLumenCostGate && lumenUseImageCostGate;
    double effectiveLumenMaxOverlapCostFraction = lumenMaxOverlapCostFraction;
    const bool highConfidenceLumenPrior =
        useCellLumenCostGate &&
        lumenProposal != nullptr &&
        lumenHighConfidenceMaxScore >= 0.0f &&
        lumenHighConfidenceMaxOverlapCostFraction >= 0.0f &&
        lumenProposal->elongation <= lumenHighConfidenceMaxScore &&
        imageCostDiff <= 0.0;
    if (highConfidenceLumenPrior) {
        effectiveLumenMaxOverlapCostFraction = std::max(
            effectiveLumenMaxOverlapCostFraction,
            static_cast<double>(lumenHighConfidenceMaxOverlapCostFraction));
    }
    const bool windowBackedLumenPrior =
        useCellLumenCostGate && cleanWindowBackedLumenPrior;
    if (windowBackedLumenPrior) {
        effectiveLumenMaxOverlapCostFraction = std::max(
            effectiveLumenMaxOverlapCostFraction,
            static_cast<double>(
                lumenProposal->maxOverlapCostFractionOverride));
    }
    const bool partialParentAnchorWindowSupport =
        useCellLumenCostGate &&
        lumenParentAnchoredProposal &&
        lumenProposal != nullptr &&
        lumenParentAnchorOneRealCandidate &&
        lumenProposal->windowBothDaughtersSupported >=
            probConfig.lumen_partial_parent_window_min_both_supported &&
        lumenProposal->windowMissingDaughterCount <=
            probConfig.lumen_partial_parent_window_max_missing_daughters &&
        lumenProposal->windowParentPersists <=
            probConfig.lumen_partial_parent_window_max_parent_persists &&
        lumenProposal->parentShapeElongation >=
            probConfig.lumen_partial_parent_window_min_parent_shape &&
        lumenProposal->elongation <=
            probConfig.lumen_partial_parent_window_max_prior_score &&
        lumenProposal->neighborClaimPenalty <=
            probConfig.lumen_partial_parent_window_max_claim_penalty &&
        lumenProposal->continuationClaimSoftPenalty <=
            probConfig.lumen_partial_parent_window_max_continuation_penalty &&
        imageCostDiff <= 0.0 &&
        bridgeCostRescueValleyFromBright <=
            probConfig.lumen_partial_parent_window_max_valley_from_bright &&
        overlapCostDiff <=
            baselineImageCost *
                probConfig.lumen_partial_parent_window_max_overlap_cost_fraction &&
        snapshotDriftMax <=
            std::max(probConfig.lumen_partial_parent_window_max_snapshot_drift_abs,
                     finalAxisLen *
                         probConfig
                             .lumen_partial_parent_window_max_snapshot_drift_axis_fraction);
    if (partialParentAnchorWindowSupport) {
        effectiveLumenMaxOverlapCostFraction = std::max(
            effectiveLumenMaxOverlapCostFraction,
            static_cast<double>(
                probConfig
                    .lumen_partial_parent_window_max_overlap_cost_fraction));
    }
    const double lumenMaxOverlapCost =
        (useCellLumenCostGate && effectiveLumenMaxOverlapCostFraction >= 0.0f)
            ? static_cast<double>(effectiveLumenMaxOverlapCostFraction) * baselineImageCost
            : std::numeric_limits<double>::infinity();
    if (useCellLumenCostGate &&
        effectiveLumenMaxOverlapCostFraction >= 0.0f &&
        overlapCostDiff > lumenMaxOverlapCost) {
        const double hardOverlapCostLimit =
            (useCellLumenSoftGate && lumenHardMaxOverlapCostFraction >= 0.0f)
                ? static_cast<double>(lumenHardMaxOverlapCostFraction) * baselineImageCost
                : lumenMaxOverlapCost;
        const bool impossibleOverlapCost =
            !useCellLumenSoftGate || overlapCostDiff > hardOverlapCostLimit;
        if (!impossibleOverlapCost) {
            const double excess = std::max(0.0, overlapCostDiff - lumenMaxOverlapCost);
            const bool priorShapeRescued =
                lumenProposal != nullptr && lumenProposal->elongatedParentRescued;
            const double strongBridgeImageGain =
                std::max(static_cast<double>(
                             probConfig.lumen_strong_bridge_image_gain_abs),
                         static_cast<double>(
                             probConfig.lumen_strong_bridge_image_gain_multiplier) *
                             static_cast<double>(
                                 std::max(0.0f, lumenPositiveGateMinImageGain)));
            const bool strongBridgeOverlapEvidence =
                useCellLumenImageGate &&
                lumenStrongBridgeEvidence &&
                imageCostDiff <= -strongBridgeImageGain &&
                bridgeCostRescueValleyFromBright <=
                    probConfig.lumen_strong_bridge_max_valley_from_bright &&
                bridgeCostRescueGapDensity <=
                    probConfig.lumen_strong_bridge_max_gap_density &&
                (lumenBridgeGapWidth >=
                     probConfig.lumen_strong_bridge_min_gap_width ||
                 finalAxisLen >=
                     probConfig.lumen_strong_bridge_min_axis_length) &&
                priorShapeRescued;
            const bool configuredBridgeOverlapWaiver =
                lumenBridgeEvidenceWaivesOverlapSoftPenalty &&
                useCellLumenImageGate &&
                lumenStrongBridgeEvidence &&
                imageCostDiff <= lumenMaxPositiveCost &&
                priorShapeRescued;
            const bool bridgeEvidenceCanWaiveOverlap =
                configuredBridgeOverlapWaiver || strongBridgeOverlapEvidence;
            if (bridgeEvidenceCanWaiveOverlap) {
                std::cout << "[Split CellLumen Soft Gate Waived] " << parentName
                          << " reason=overlap_cost"
                          << " overlapDiff=" << overlapCostDiff
                          << " maxOverlap=" << lumenMaxOverlapCost
                          << " hardMaxOverlap=" << hardOverlapCostLimit
                          << " imageDiff=" << imageCostDiff
                          << " bridgeEvidence=1"
                          << " strongBridgeOverlap="
                          << (strongBridgeOverlapEvidence ? 1 : 0)
                          << " bridgeValleyFromBright="
                          << bridgeCostRescueValleyFromBright
                          << " bridgeGapDensity="
                          << bridgeCostRescueGapDensity
                          << " bridgeGapWidth="
                          << lumenBridgeGapWidth
                          << " snapshotSeedFallback=0"
                          << " elongatedParentRescued=" << (priorShapeRescued ? 1 : 0)
                          << " parentShapeElong="
                          << (lumenProposal != nullptr ? lumenProposal->parentShapeElongation : 1.0f)
                          << " baselineImageCost=" << baselineImageCost
                          << " bestIdx=" << bestIdx
                          << " bestLabel=" << bestLabel
                          << std::endl;
            } else {
                const double penalty = excess * std::max(0.0f, lumenSoftOverlapCostPenaltyWeight);
                lumenSoftGatePenaltyCost += penalty;
                std::cout << "[Split CellLumen Soft Gate] " << parentName
                          << " reason=overlap_cost"
                          << " overlapDiff=" << overlapCostDiff
                          << " maxOverlap=" << lumenMaxOverlapCost
                          << " hardMaxOverlap=" << hardOverlapCostLimit
                          << " penaltyWeight=" << lumenSoftOverlapCostPenaltyWeight
                          << " penaltyCost=" << penalty
                          << " runningPenalty=" << lumenSoftGatePenaltyCost
                          << " windowBacked=" << (windowBackedLumenPrior ? 1 : 0)
                          << " overlapFraction=" << effectiveLumenMaxOverlapCostFraction
                          << " baselineImageCost=" << baselineImageCost
                          << " bestIdx=" << bestIdx
                          << " bestLabel=" << bestLabel
                          << std::endl;
            }
        } else {
        std::cout << "[Split Reject CellLumen overlap] " << parentName
                  << " overlapDiff=" << overlapCostDiff
                  << " maxOverlap=" << lumenMaxOverlapCost
                  << " hardMaxOverlap=" << hardOverlapCostLimit
                  << " fraction=" << effectiveLumenMaxOverlapCostFraction
                  << " baseFraction=" << lumenMaxOverlapCostFraction
                  << " highConfidence=" << (highConfidenceLumenPrior ? 1 : 0)
                  << " windowBacked=" << (windowBackedLumenPrior ? 1 : 0)
                  << " priorScore=" << (lumenProposal != nullptr ? lumenProposal->elongation : 0.0f)
                  << " baselineImageCost=" << baselineImageCost
                  << " totalDiff=" << costDiff
                  << " imageDiff=" << imageCostDiff
                  << " bestIdx=" << bestIdx
                  << " bestLabel=" << bestLabel
                  << std::endl;
        restoreLiveParent();
        return {0.0, noop};
        }
    }
    const double rawLumenGateDiff =
        useCellLumenImageGate ? imageCostDiff : costDiff;
    const double totalSoftGatePenaltyCost =
        lumenSoftGatePenaltyCost + splitSoftGeometryPenaltyCost;
    const double lumenGateDiff =
        rawLumenGateDiff + totalSoftGatePenaltyCost;
    bool lumenPositiveGateHasImageSupport = true;
    const double lumenPositiveGateRequiredImageGain =
        std::max(static_cast<double>(std::max(0.0f, lumenPositiveGateMinImageGain)),
                 totalSoftGatePenaltyCost *
                     static_cast<double>(std::max(0.0f, lumenPositiveGateMinImageGainPenaltyRatio)));
    const double lumenPositiveGateSoftPenaltyFraction =
        (baselineImageCost > 1e-9)
            ? totalSoftGatePenaltyCost / baselineImageCost
            : std::numeric_limits<double>::infinity();
    const float lumenParentShapeElongation =
        (lumenProposal != nullptr) ? lumenProposal->parentShapeElongation : 1.0f;
    const float lumenPriorScore =
        (lumenProposal != nullptr) ? lumenProposal->elongation : std::numeric_limits<float>::infinity();
    const float lumenPriorScoreWithoutWindowBonus =
        (lumenProposal != nullptr)
            ? lumenPriorScore + std::max(0.0f, lumenProposal->balancedWindowBonus)
            : lumenPriorScore;
    const float lumenParentDistanceBalance =
        (lumenProposal != nullptr) ? lumenProposal->parentDistanceBalance : 1.0f;
    const float lumenParentPersistencePenalty =
        (lumenProposal != nullptr) ? lumenProposal->parentPersistencePenalty : 0.0f;
    const float lumenNeighborClaimPenalty =
        (lumenProposal != nullptr) ? lumenProposal->neighborClaimPenalty : 0.0f;
    const float lumenContinuationClaimSoftPenalty =
        (lumenProposal != nullptr) ? lumenProposal->continuationClaimSoftPenalty : 0.0f;
    const float lumenPositiveGateEffectiveElongatedMinShape =
        std::max(probConfig.lumen_positive_gate_min_shape_floor,
                 lumenPositiveGateElongatedParentMinShape);
    const bool lumenPositiveGateElongatedShapeOk =
        lumenProposal != nullptr &&
        lumenPositiveGateElongatedParentMinShape >= 0.0f &&
        lumenParentShapeElongation >= lumenPositiveGateEffectiveElongatedMinShape;
    const bool lumenPositiveGateElongatedRawOk =
        lumenPositiveGateElongatedMaxRawWorsening < 0.0f ||
        imageCostDiff <= static_cast<double>(lumenPositiveGateElongatedMaxRawWorsening);
    const bool lumenPositiveGateElongatedPenaltyOk =
        lumenPositiveGateElongatedMaxSoftPenaltyFraction < 0.0f ||
        lumenPositiveGateSoftPenaltyFraction <= static_cast<double>(lumenPositiveGateElongatedMaxSoftPenaltyFraction);
    const bool lumenPositiveGateElongatedScoreOk =
        lumenPositiveGateElongatedMaxScore < 0.0f ||
        lumenPriorScoreWithoutWindowBonus <= lumenPositiveGateElongatedMaxScore;
    const bool lumenCurrentParentDoesNotPersist =
        lumenParentPersistencePenalty <= probConfig.lumen_parent_persistence_epsilon;
    const bool lumenPositiveGateElongatedSupport =
        useCellLumenImageGate &&
        lumenCurrentParentDoesNotPersist &&
        lumenPositiveGateElongatedShapeOk &&
        lumenPositiveGateElongatedRawOk &&
        lumenPositiveGateElongatedPenaltyOk &&
        lumenPositiveGateElongatedScoreOk;
    const bool cleanFutureWindowSupport =
        useCellLumenImageGate &&
        lumenProposal != nullptr &&
        lumenProposal->candidateIdA >= 0 &&
        lumenProposal->candidateIdB >= 0 &&
        lumenProposal->windowBothDaughtersSupported >=
            probConfig.lumen_clean_window_min_both_supported &&
        lumenProposal->windowMissingDaughterCount <=
            probConfig.lumen_clean_window_max_missing_daughters &&
        lumenProposal->windowParentPersists <=
            probConfig.lumen_clean_window_max_parent_persists;
    const bool lumenWeakImageGeometryOk =
        lumenParentDistanceBalance >= probConfig.lumen_weak_image_min_parent_balance ||
        (lumenPositiveGateElongatedShapeOk && lumenPositiveGateElongatedScoreOk);
    const bool lumenRiskyContinuationClaim =
        lumenParentDistanceBalance < probConfig.lumen_risky_claim_parent_balance &&
        (lumenNeighborClaimPenalty >= probConfig.lumen_risky_claim_neighbor_penalty ||
         lumenContinuationClaimSoftPenalty >=
             probConfig.lumen_risky_claim_continuation_penalty);
    const bool lumenPositiveGateFutureTotalSupport =
        cleanFutureWindowSupport &&
        costDiff <= -adaptiveThreshold &&
        lumenPositiveGateSoftPenaltyFraction <=
            std::max(static_cast<double>(
                         probConfig.lumen_future_support_min_soft_penalty_fraction),
                     static_cast<double>(
                         std::max(0.0f,
                                  lumenPositiveGateElongatedMaxSoftPenaltyFraction)));
    const bool lumenPositiveGateFutureWindowSoftSupport =
        windowBackedLumenPrior &&
        seedHasWindowLateralEvidence &&
        lumenCurrentParentDoesNotPersist &&
        lumenWeakImageGeometryOk &&
        imageCostDiff <= 0.0 &&
        lumenPositiveGateSoftPenaltyFraction <=
            std::max(static_cast<double>(
                         probConfig.lumen_future_support_min_soft_penalty_fraction),
                     static_cast<double>(
                         std::max(0.0f,
                                  lumenPositiveGateElongatedMaxSoftPenaltyFraction)));
    const bool lumenCleanOneSidedWindowRescue =
        cleanFutureWindowSupport &&
        lumenPositiveGateElongatedShapeOk &&
        lumenNeighborClaimPenalty <=
            probConfig.lumen_clean_one_sided_max_neighbor_penalty &&
        lumenContinuationClaimSoftPenalty <=
            probConfig.lumen_clean_one_sided_max_continuation_penalty &&
        lumenParentPersistencePenalty <=
            probConfig.lumen_clean_one_sided_max_parent_persistence &&
        imageCostDiff <= 0.0 &&
        lumenPriorScoreWithoutWindowBonus <=
            probConfig.lumen_clean_one_sided_max_prior_score &&
        lumenPositiveGateSoftPenaltyFraction <=
            std::max(static_cast<double>(
                         probConfig.lumen_clean_one_sided_min_soft_penalty_fraction),
                     static_cast<double>(
                         std::max(0.0f,
                                  lumenPositiveGateElongatedMaxSoftPenaltyFraction)));
    const bool lumenPrepassFallbackSoftSupport =
        bestIsCellLumenPrepassFallback &&
        lumenPositiveGateElongatedShapeOk &&
        lumenCurrentParentDoesNotPersist &&
        lumenNeighborClaimPenalty <= probConfig.lumen_prepass_max_claim_penalty &&
        lumenContinuationClaimSoftPenalty <= probConfig.lumen_prepass_max_claim_penalty &&
        lumenParentDistanceBalance >= probConfig.lumen_prepass_min_parent_balance &&
        imageCostDiff <= -probConfig.lumen_prepass_min_image_gain &&
        bridgeCostRescueValleyFromBright <=
            probConfig.lumen_prepass_max_valley_from_bright &&
        bridgeCostRescueGapDensity <= probConfig.lumen_prepass_max_gap_density &&
        lumenPriorScoreWithoutWindowBonus <= probConfig.lumen_prepass_max_prior_score &&
        lumenPositiveGateSoftPenaltyFraction <=
            probConfig.lumen_prepass_max_soft_penalty_fraction;
    const bool lumenFutureContinuationConflictRescue =
        cleanFutureWindowSupport &&
        lumenProposal != nullptr &&
        lumenProposal->futureContinuationConflictRescued &&
        lumenCurrentParentDoesNotPersist &&
        lumenParentPersistencePenalty <= probConfig.lumen_prepass_max_claim_penalty &&
        lumenNeighborClaimPenalty <= probConfig.lumen_prepass_max_claim_penalty &&
        lumenContinuationClaimSoftPenalty >=
            probConfig.lumen_future_conflict_min_continuation_penalty &&
        lumenParentDistanceBalance >=
            probConfig.lumen_future_conflict_min_parent_balance &&
        imageCostDiff <= -probConfig.lumen_future_conflict_min_image_gain &&
        overlapCostDiff <=
            baselineImageCost *
                probConfig.lumen_future_conflict_max_overlap_cost_fraction &&
        bridgeCostRescueValleyFromBright <=
            probConfig.lumen_future_conflict_max_valley_from_bright &&
        lumenPriorScoreWithoutWindowBonus <=
            probConfig.lumen_future_conflict_max_prior_score &&
        lumenPositiveGateSoftPenaltyFraction <=
            probConfig.lumen_future_conflict_max_soft_penalty_fraction;
    const bool lumenCleanWindowTotalImprovement =
        windowBackedLumenPrior &&
        lumenCurrentParentDoesNotPersist &&
        !lumenRiskyContinuationClaim &&
        costDiff < 0.0 &&
        overlapCostDiff <= 0.0 &&
        imageCostDiff <= 0.0 &&
        lumenBridgeGapWidth >= probConfig.lumen_clean_window_min_gap_width &&
        bridgeCostRescueValleyFromBright <=
            probConfig.lumen_clean_window_max_valley_from_bright &&
        lumenPositiveGateSoftPenaltyFraction <=
            probConfig.lumen_clean_window_max_soft_penalty_fraction;
    const char *lumenPositiveGateSupportReason = "normal_image_gain";
    if (useCellLumenCostGate &&
        useCellLumenImageGate &&
        lumenPositiveGateRequiredImageGain > 0.0) {
        const double riskyClaimImageGain =
            lumenPositiveGateRequiredImageGain *
            (lumenRiskyContinuationClaim
                 ? static_cast<double>(
                       probConfig.lumen_risky_claim_image_gain_multiplier)
                 : 1.0);
        const bool rawImageGainSupport =
            imageCostDiff <= -riskyClaimImageGain;
        lumenPositiveGateHasImageSupport =
            rawImageGainSupport ||
            lumenPositiveGateElongatedSupport ||
            lumenPositiveGateFutureTotalSupport ||
            lumenPositiveGateFutureWindowSoftSupport ||
            lumenCleanOneSidedWindowRescue ||
            partialParentAnchorWindowSupport ||
            lumenFutureContinuationConflictRescue ||
            lumenCleanWindowTotalImprovement ||
            lumenPrepassFallbackSoftSupport;
        if (!rawImageGainSupport && lumenPositiveGateElongatedSupport) {
            lumenPositiveGateSupportReason = "elongated_parent_small_worsening";
        } else if (!rawImageGainSupport && lumenPositiveGateFutureTotalSupport) {
            lumenPositiveGateSupportReason = "future_window_total_improvement";
        } else if (!rawImageGainSupport && lumenPositiveGateFutureWindowSoftSupport) {
            lumenPositiveGateSupportReason = "future_window_soft_geometry";
        } else if (!rawImageGainSupport && lumenCleanOneSidedWindowRescue) {
            lumenPositiveGateSupportReason = "clean_one_sided_window_rescue";
        } else if (!rawImageGainSupport && partialParentAnchorWindowSupport) {
            lumenPositiveGateSupportReason = "partial_parent_anchor_window_support";
        } else if (!rawImageGainSupport && lumenFutureContinuationConflictRescue) {
            lumenPositiveGateSupportReason = "future_continuation_conflict_rescue";
        } else if (!rawImageGainSupport && lumenCleanWindowTotalImprovement) {
            lumenPositiveGateSupportReason = "clean_window_total_improvement";
        } else if (!rawImageGainSupport && lumenPrepassFallbackSoftSupport) {
            lumenPositiveGateSupportReason = "prepass_fallback_soft_support";
        }
    }
    const double lumenWeakSplitImageGain =
        std::max(static_cast<double>(probConfig.lumen_weak_split_image_gain_abs),
                 lumenPositiveGateRequiredImageGain *
                     static_cast<double>(
                         probConfig.lumen_weak_split_image_gain_multiplier));
    const bool lumenWeakImageSplit =
        useCellLumenCostGate &&
        useCellLumenImageGate &&
        bestIsCellLumenPrior &&
        windowBackedLumenPrior &&
        imageCostDiff > -lumenWeakSplitImageGain;
    const bool lumenLikelyContinuationHijack =
        lumenWeakImageSplit &&
        !lumenCleanOneSidedWindowRescue &&
        !lumenFutureContinuationConflictRescue &&
        !lumenCleanWindowTotalImprovement &&
        (lumenParentShapeElongation < probConfig.lumen_hijack_min_parent_shape ||
         lumenParentDistanceBalance < probConfig.lumen_hijack_min_parent_balance ||
         (lumenNeighborClaimPenalty >=
              probConfig.lumen_risky_claim_neighbor_penalty &&
          lumenParentDistanceBalance < probConfig.lumen_risky_claim_parent_balance));
    const bool lumenLikelyNeighborClaimDuplicate =
        lumenWeakImageSplit &&
        !lumenCleanOneSidedWindowRescue &&
        !lumenFutureContinuationConflictRescue &&
        !lumenCleanWindowTotalImprovement &&
        lumenNeighborClaimPenalty >=
            probConfig.lumen_neighbor_duplicate_min_claim_penalty &&
        imageCostDiff > -probConfig.lumen_neighbor_duplicate_min_image_gain &&
        (lumenParentDistanceBalance <
             probConfig.lumen_neighbor_duplicate_min_parent_balance ||
         (lumenBridgeGapWidth < probConfig.lumen_neighbor_duplicate_min_gap_width &&
          bridgeCostRescueValleyFromBright >
              probConfig.lumen_neighbor_duplicate_min_valley_from_bright));
    const bool lumenModerateClaimOverlapDuplicate =
        lumenWeakImageSplit &&
        !lumenCleanOneSidedWindowRescue &&
        !lumenFutureContinuationConflictRescue &&
        !lumenCleanWindowTotalImprovement &&
        lumenNeighborClaimPenalty >=
            probConfig.lumen_moderate_duplicate_min_claim_penalty &&
        lumenParentDistanceBalance <
            probConfig.lumen_moderate_duplicate_min_parent_balance &&
        overlapCostDiff >
            baselineImageCost *
                probConfig.lumen_moderate_duplicate_min_overlap_cost_fraction &&
        costDiff > 0.0 &&
        imageCostDiff > -probConfig.lumen_moderate_duplicate_min_image_gain;
    if (lumenLikelyContinuationHijack ||
        lumenLikelyNeighborClaimDuplicate ||
        lumenModerateClaimOverlapDuplicate) {
        std::cout << "[Split Reject CellLumen weak continuation hijack] " << parentName
                  << " imageDiff=" << imageCostDiff
                  << " weakImageGainThreshold=" << lumenWeakSplitImageGain
                  << " gateDiff=" << lumenGateDiff
                  << " rawGateDiff=" << rawLumenGateDiff
                  << " totalDiff=" << costDiff
                  << " overlapDiff=" << overlapCostDiff
                  << " parentShapeElong=" << lumenParentShapeElongation
                  << " parentDistBalance=" << lumenParentDistanceBalance
                  << " parentPersistencePenalty=" << lumenParentPersistencePenalty
                  << " neighborClaimPenalty=" << lumenNeighborClaimPenalty
                  << " continuationClaimSoftPenalty=" << lumenContinuationClaimSoftPenalty
                  << " neighborClaimDuplicate="
                  << (lumenLikelyNeighborClaimDuplicate ? 1 : 0)
                  << " moderateClaimOverlapDuplicate="
                  << (lumenModerateClaimOverlapDuplicate ? 1 : 0)
                  << " bridgeValleyFromBright="
                  << bridgeCostRescueValleyFromBright
                  << " bridgeGapWidth=" << lumenBridgeGapWidth
                  << " priorScore=" << lumenPriorScore
                  << " priorScoreNoWindowBonus=" << lumenPriorScoreWithoutWindowBonus
                  << " supportReason=" << lumenPositiveGateSupportReason
                  << " bestIdx=" << bestIdx
                  << " bestLabel=" << bestLabel
                  << std::endl;
        restoreLiveParent();
        return {0.0, noop};
    }
    // A negative gate diff is an actual reconstruction improvement. The
    // Lumen path only needs to reject candidates that get worse by more than
    // the configured positive allowance. For positive gate diffs, require
    // enough raw image improvement to justify any soft geometry penalty; this
    // avoids accepting duplicate splits that improve the image term only
    // weakly while exploding overlap.
    const bool lumenCostAccepted =
        useCellLumenCostGate &&
        lumenGateDiff <= lumenMaxPositiveCost &&
        lumenPositiveGateHasImageSupport;
    if (lumenCostAccepted) {
        acceptedCostDiff = -std::max(1.0, adaptiveThreshold);
        std::cout << "[Split CellLumen Cost Gate] " << parentName
                  << " gateDiff=" << lumenGateDiff
                  << " rawGateDiff=" << rawLumenGateDiff
                  << " softPenalty=" << lumenSoftGatePenaltyCost
                  << " splitSoftPenalty=" << splitSoftGeometryPenaltyCost
                  << " gateMode=" << (useCellLumenImageGate ? "image" : "total")
                  << " totalDiff=" << costDiff
                  << " imageDiff=" << imageCostDiff
                  << " overlapDiff=" << overlapCostDiff
                  << " reportedDiff=" << acceptedCostDiff
                  << " fixed=" << activeSplitCost
                  << " fraction=" << activeSplitCostFraction
                  << " maxPositive=" << lumenMaxPositiveCost
                  << " positiveImageSupport=1"
                  << " positiveSupportReason=" << lumenPositiveGateSupportReason
                  << " requiredImageGain=" << lumenPositiveGateRequiredImageGain
                  << " softPenaltyFraction=" << lumenPositiveGateSoftPenaltyFraction
                  << " parentShapeElong=" << lumenParentShapeElongation
                  << " priorScore=" << lumenPriorScore
                  << " priorScoreNoWindowBonus=" << lumenPriorScoreWithoutWindowBonus
                  << " windowBacked=" << (windowBackedLumenPrior ? 1 : 0)
                  << " parentDistBalance=" << lumenParentDistanceBalance
                  << " parentPersistencePenalty=" << lumenParentPersistencePenalty
                  << " neighborClaimPenalty=" << lumenNeighborClaimPenalty
                  << " continuationClaimSoftPenalty=" << lumenContinuationClaimSoftPenalty
                  << " riskyContinuationClaim=" << (lumenRiskyContinuationClaim ? 1 : 0)
                  << " baselineImageCost=" << baselineImageCost
                  << " bestIdx=" << bestIdx
                  << " bestLabel=" << bestLabel
                  << std::endl;
    }
    if (useCellLumenCostGate &&
        lumenGateDiff <= lumenMaxPositiveCost &&
        !lumenPositiveGateHasImageSupport) {
        std::cout << "[Split Reject CellLumen positive gate weak image] " << parentName
                  << " gateDiff=" << lumenGateDiff
                  << " rawGateDiff=" << rawLumenGateDiff
                  << " imageDiff=" << imageCostDiff
                  << " requiredImageGain=" << lumenPositiveGateRequiredImageGain
                  << " softPenaltyFraction=" << lumenPositiveGateSoftPenaltyFraction
                  << " parentShapeElong=" << lumenParentShapeElongation
                  << " priorScore=" << lumenPriorScore
                  << " priorScoreNoWindowBonus=" << lumenPriorScoreWithoutWindowBonus
                  << " elongatedShapeOk=" << (lumenPositiveGateElongatedShapeOk ? 1 : 0)
                  << " elongatedRawOk=" << (lumenPositiveGateElongatedRawOk ? 1 : 0)
                  << " elongatedPenaltyOk=" << (lumenPositiveGateElongatedPenaltyOk ? 1 : 0)
                  << " elongatedScoreOk=" << (lumenPositiveGateElongatedScoreOk ? 1 : 0)
                  << " currentParentDoesNotPersist=" << (lumenCurrentParentDoesNotPersist ? 1 : 0)
                  << " weakImageGeometryOk=" << (lumenWeakImageGeometryOk ? 1 : 0)
                  << " parentDistBalance=" << lumenParentDistanceBalance
                  << " parentPersistencePenalty=" << lumenParentPersistencePenalty
                  << " neighborClaimPenalty=" << lumenNeighborClaimPenalty
                  << " continuationClaimSoftPenalty=" << lumenContinuationClaimSoftPenalty
                  << " riskyContinuationClaim=" << (lumenRiskyContinuationClaim ? 1 : 0)
                  << " softPenalty=" << lumenSoftGatePenaltyCost
                  << " splitSoftPenalty=" << splitSoftGeometryPenaltyCost
                  << " totalDiff=" << costDiff
                  << " overlapDiff=" << overlapCostDiff
                  << " maxPositive=" << lumenMaxPositiveCost
                  << " bestIdx=" << bestIdx
                  << " bestLabel=" << bestLabel
                  << std::endl;
    }

    const double rejectGateDiff = useCellLumenImageGate
                                      ? lumenGateDiff
                                      : costDiff + totalSoftGatePenaltyCost;
    if (!bridgeCostRescued && !futureWindowCostRescued && !lumenCostAccepted &&
        rejectGateDiff >= -adaptiveThreshold) {
        std::cout << "[Split Reject cost] " << parentName
                  << " diff=" << rejectGateDiff
                  << " rawGateDiff=" << rawLumenGateDiff
                  << " softPenalty=" << lumenSoftGatePenaltyCost
                  << " splitSoftPenalty=" << splitSoftGeometryPenaltyCost
                  << " totalDiff=" << costDiff
                  << " imageDiff=" << imageCostDiff
                  << " overlapDiff=" << overlapCostDiff
                  << " geometryAdjustedDiff=" << geometryAdjustedCostDiff
                  << " highShapeRawPcaBridgeBioNearMissRescue="
                  << (highShapeRawPcaBridgeBioNearMissCostRescued ? 1 : 0)
                  << " highShapeRawRodSignalNearTieRescue="
                  << (highShapeRawRodSignalNearTieCostRescued ? 1 : 0)
                  << " oneFrameAlignedLockedFuturePcaBridgeRescue="
                  << (oneFrameAlignedLockedFuturePcaBridgeCostRescued ? 1 : 0)
                  << " twoFrameAlignedLockedFuturePcaBridgeRescue="
                  << (twoFrameAlignedLockedFuturePcaBridgeCostRescued ? 1 : 0)
                  << " stableFutureNearThresholdBridgeRescue="
                  << (stableFuturePcaBridgeNearThresholdCostRescued ? 1 : 0)
                  << " severePostPcaRodFutureCostRescue="
                  << (severePostPcaRodFutureCostRescued ? 1 : 0)
                  << " cellUniverse3WeakPcaBridgeCostRescue="
                  << (cellUniverse3WeakPcaBridgeCostRescued ? 1 : 0)
                  << " cellUniverse3WindowSoftPenaltyCostRescue="
                  << (cellUniverse3WindowSoftPenaltyCostRescued ? 1 : 0)
                  << " cellUniverse3WindowMapPrimaryCostRescue="
                  << (cellUniverse3WindowMapPrimaryCostRescued ? 1 : 0)
                  << " cellUniverse3WindowMapPrimaryCostMinAxis="
                  << cellUniverse3WindowMapPrimaryCostMinAxisAlignment
                  << " cellUniverse3WindowMapAxis="
                  << (bridgeProposal != nullptr
                          ? bridgeProposal->cellUniverse3MapAxisAlignment
                          : 0.0f)
                  << " cellUniverse3AxisPlaceNearThresholdRescue="
                  << (cellUniverse3AxisPlaceNearThresholdCostRescued ? 1 : 0)
                  << " cellUniverse3DelayedMissingDaughterCostRescue="
                  << (cellUniverse3DelayedMissingDaughterCostRescued ? 1 : 0)
                  << " costBackedCleanFutureNearThresholdRescue="
                  << (costBackedCleanFuturePcaBridgeNearThresholdCostRescued
                          ? 1
                          : 0)
                  << " parentShapeElong="
                  << (bridgeProposal != nullptr ? bridgeProposal->parentShapeElongation : 0.0f)
                  << " parentDistBalance="
                  << (bridgeProposal != nullptr ? bridgeProposal->parentDistanceBalance : 0.0f)
                  << " futureBoth="
                  << (bridgeProposal != nullptr ? bridgeProposal->windowBothDaughtersSupported : 0)
                  << " futureMissing="
                  << (bridgeProposal != nullptr ? bridgeProposal->windowMissingDaughterCount : 0)
                  << " parentPersists="
                  << (bridgeProposal != nullptr ? bridgeProposal->windowParentPersists : 0)
                  << " centerSnapApplied="
                  << (bridgeProposal != nullptr && bridgeProposal->centerSnapApplied ? 1 : 0)
                  << " immediateFutureBacked="
                  << (bridgeProposal != nullptr && bridgeProposal->immediateFutureCenterBacked ? 1 : 0)
                  << " alignedPairFallback="
                  << (bridgeProposal != nullptr && bridgeProposal->centerSnapUsedAlignedPairFallback ? 1 : 0)
                  << " valleyFromBright=" << bridgeCostRescueValleyFromBright
                  << " gapDensity=" << bridgeCostRescueGapDensity
                  << " finalAxisLen=" << finalAxisLen
                  << " srcMaxR=" << srcMaxR
                  << " maxDaughterSeedDrift=" << maxDaughterSeedDrift
                  << " gateMode=" << (useCellLumenImageGate ? "image" : "total")
                  << " mode=" << (_useBboxCost ? "bbox" : "full")
                  << " threshold=" << -adaptiveThreshold
                  << " (fixed=" << activeSplitCost
                  << " frac=" << activeSplitCostFraction << "×" << baselineImageCost
                  << " lumen_gate=" << (useCellLumenCostGate ? 1 : 0)
                  << " lumen_max_positive=" << lumenMaxPositiveCost << ")"
                  << " bestIdx=" << bestIdx
                  << " bestLabel=" << bestLabel
                  << " d1=(" << bestD1.getX() << "," << bestD1.getY() << "," << bestD1.getZ() << ")"
                  << " r1=(" << bestD1.getARadius() << "," << bestD1.getBRadius() << "," << bestD1.getCRadius() << ")"
                  << " drift1=" << drift1
                  << " d2=(" << bestD2.getX() << "," << bestD2.getY() << "," << bestD2.getZ() << ")"
                  << " r2=(" << bestD2.getARadius() << "," << bestD2.getBRadius() << "," << bestD2.getCRadius() << ")"
                  << " drift2=" << drift2
                  << std::endl;
        restoreLiveParent();
        return {0.0, noop};
    }

    // Accept: install the best candidate state. The callback applies on
    // accept; the caller uses perturbCell's contract where the callback is
    // invoked after the decision. To stay consistent with that contract we
    // return the (costDiff, callback) pair that installs bestCells state.
    // Move saved* into copies for the callback — savedCells/savedSynth/
    // savedPerSlice are not read again after this point.
    auto savedCellsCopy = std::move(savedCells);
    auto savedSynthCopy = std::move(savedSynth);
    auto savedPerSliceCopy = std::move(savedPerSlice);
    double savedCostCopy = savedCost;

    const cv::Point3f acceptedD1Pos(bestD1.getX(), bestD1.getY(), bestD1.getZ());
    const cv::Point3f acceptedD2Pos(bestD2.getX(), bestD2.getY(), bestD2.getZ());
    const cv::Point3f acceptedD1R(bestD1.getARadius(), bestD1.getBRadius(), bestD1.getCRadius());
    const cv::Point3f acceptedD2R(bestD2.getARadius(), bestD2.getBRadius(), bestD2.getCRadius());
    const float acceptedDrift1 = drift1;
    const float acceptedDrift2 = drift2;
    const cv::Point3f acceptedSeed1 = bestSeedD1;
    const cv::Point3f acceptedSeed2 = bestSeedD2;

    // Capture extras for the callback's reject branch — it needs to
    // undo the snapshot-state install so Phase B's live state is restored.
    const Ellipsoid liveParentCopy = liveParent;
    const Ellipsoid snapshotParentCopy = snapshotParent;
    const size_t cellIndexCopy = cellIndex;
    const bool snapshotValidCopy = snapshotValid;

    const std::string acceptedLabel = bestLabel;

    CallBackFunc callback = [this,
                             bestCells = std::move(bestCells),
                             bestSynth = std::move(bestSynth),
                             bestPerSlice = std::move(bestPerSlice),
                             bestImageCost,
                             savedCellsCopy = std::move(savedCellsCopy),
                             savedSynthCopy = std::move(savedSynthCopy),
                             savedPerSliceCopy = std::move(savedPerSliceCopy),
                             savedCostCopy,
                             parentName, costDiff = acceptedCostDiff,
                             acceptedD1Pos, acceptedD2Pos,
                             acceptedD1R, acceptedD2R, acceptedSeed1, acceptedSeed2,
                             acceptedDrift1, acceptedDrift2, acceptedLabel,
                             liveParentCopy, snapshotParentCopy, cellIndexCopy,
                             snapshotValidCopy](bool accept) mutable {
        // Snap bbox + snap position handling for daughter names.
        //
        // On ACCEPT: replace the burn-in-time shared splitBbox with per-
        // daughter snap bboxes centered on the final positions, and set
        // snap positions so the position-prior penalty (Change 37)
        // activates. Without this, newborn daughters had NO anchor post-
        // split — observed in run 205041 f3 where 12345..0 drifted from
        // (142, 176, 105) at split-accept to (-27, 265, 90) by f3 end
        // (175 px drift, exited image). Position prior shows priorWeight=30
        // in the log but was a no-op because _snapPositions had no entry
        // for the newborn daughter name.
        //
        // On REJECT: erase the stale daughter-name entries. The parent
        // keeps its own snap (never modified).
        const std::string d0Name = parentName + "0";
        const std::string d1Name = parentName + "1";
        if (accept) {
            this->cells = std::move(bestCells);
            this->_synthFrame = std::move(bestSynth);
            this->_currentCostPerSlice = std::move(bestPerSlice);
            this->_currentCost = bestImageCost;
            // Install snap anchors at the accepted daughter positions.
            // Use the daughter max-radius × bbox_margin_scale for the bbox.
            const float d0MaxR = std::max({acceptedD1R.x, acceptedD1R.y, acceptedD1R.z});
            const float d1MaxR = std::max({acceptedD2R.x, acceptedD2R.y, acceptedD2R.z});
            if (d0MaxR > 1e-3f) {
                BoundingBox3D b0 = this->computeBboxAtPoint(
                    acceptedD1Pos, d0MaxR, this->_bboxMarginScale);
                if (b0.isValid()) this->_snapBboxes[d0Name] = b0;
                this->_snapPositions[d0Name] = acceptedD1Pos;
            }
            if (d1MaxR > 1e-3f) {
                BoundingBox3D b1 = this->computeBboxAtPoint(
                    acceptedD2Pos, d1MaxR, this->_bboxMarginScale);
                if (b1.isValid()) this->_snapBboxes[d1Name] = b1;
                this->_snapPositions[d1Name] = acceptedD2Pos;
            }
            std::cout << "[Split Accepted] " << parentName
                      << " costDiff=" << costDiff
                      << " bestLabel=" << acceptedLabel
                      << " seed1=(" << acceptedSeed1.x << "," << acceptedSeed1.y << "," << acceptedSeed1.z << ")"
                      << " d1=(" << acceptedD1Pos.x << "," << acceptedD1Pos.y << "," << acceptedD1Pos.z << ")"
                      << " r1=(" << acceptedD1R.x << "," << acceptedD1R.y << "," << acceptedD1R.z << ")"
                      << " drift1=" << acceptedDrift1
                      << " seed2=(" << acceptedSeed2.x << "," << acceptedSeed2.y << "," << acceptedSeed2.z << ")"
                      << " d2=(" << acceptedD2Pos.x << "," << acceptedD2Pos.y << "," << acceptedD2Pos.z << ")"
                      << " r2=(" << acceptedD2R.x << "," << acceptedD2R.y << "," << acceptedD2R.z << ")"
                      << " drift2=" << acceptedDrift2
                      << std::endl;
        } else {
            // Reject path: restore the snapshot-parent-state first (which
            // is what savedCellsCopy holds), then swap cells[cellIndexCopy]
            // back to the live parent and re-render the affected z-range
            // so Phase B's live state isn't lost. Also erase any stale
            // daughter-name entries in snap maps (from burn-in installation).
            this->_snapBboxes.erase(d0Name);
            this->_snapBboxes.erase(d1Name);
            this->_snapPositions.erase(d0Name);
            this->_snapPositions.erase(d1Name);
            this->cells = std::move(savedCellsCopy);  // NOLINT: move in mutable lambda
            this->_synthFrame = std::move(savedSynthCopy);
            this->_currentCostPerSlice = std::move(savedPerSliceCopy);
            this->_currentCost = savedCostCopy;

            if (snapshotValidCopy && cellIndexCopy < this->cells.size()) {
                this->cells[cellIndexCopy] = liveParentCopy;
                int affMinR = -1, affMaxR = -1;
                Ellipsoid snapshotMutable = snapshotParentCopy;
                Ellipsoid liveMutable = liveParentCopy;
                auto revertedSynth = this->generateSynthFrameFast(snapshotMutable, liveMutable,
                                                                    &affMinR, &affMaxR);
                this->_synthFrame = revertedSynth;
                // Bbox mode keeps the full-image cache stale (Change 1).
                // Skip the incremental recompute on this reject path too.
                if (!this->_useBboxCost) {
                    std::vector<double> revertedPerSlice;
                    const double revertedCost = this->calculateIncrementalCost(revertedSynth,
                                                                                 affMinR, affMaxR,
                                                                                 revertedPerSlice);
                    this->_currentCost = revertedCost;
                    this->_currentCostPerSlice = revertedPerSlice;
                }
            }
        }
    };

    return {acceptedCostDiff, callback};
}

// Snapshot-driven daughter placement. Daughters built from previous frame's

std::vector<cv::Mat> Frame::getSynthFrame()
{
    return _synthFrame;
}
