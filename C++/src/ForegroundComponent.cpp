#include "../includes/BrightPixel.hpp"
#include "../includes/ForegroundModel.hpp"
#include "../includes/Frame.hpp"   // Frame::ClaimSet

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

#include <opencv2/core.hpp>

// Foreground-anchored gather (2026-07-31, Option 1 + Change 42 half-max). Instead
// of scanning a size-relative box (`gatherBrightPixelsVoronoi`, which re-clips the
// fit to a radius-derived cap), this flood-fills the cell's OWN blob using a
// PER-CELL, SELF-LIMITING intensity level anchored to fit-independent image facts:
//   peak  = robust local peak (99th pct of REAL intensity in a small CORE box,
//           sized off the FIXED frame-entry radius — never the live radius),
//   bg    = the STABLE per-frame background mean (_fgModel.muBg),
//   level = bg + 0.5*(peak - bg)  (standard FWHM: half of the cell's OWN contrast).
// The 6-connected BFS from the cell center keeps voxels with `I >= level`. Because
// `peak` and `bg` are image facts the fit cannot move, `level` is stable frame to
// frame → the region is stable → NO bloat feedback loop (the earlier global
// `pFg>0.5` criterion drifted: a bloating fit dragged muFg down, pushing the
// global boundary outward). No global model in the loop, no growth cap, no size
// constant — scale-invariant. Every voxel is Voronoi-split against the neighbor
// claim points (same rule as the box gather) so touching nuclei separate. Voxels
// return as BrightPixel{pos, weight = intensity − background}.
//
// LIVES IN ITS OWN TRANSLATION UNIT ON PURPOSE. Compiling this body inside
// Frame.cpp shifts the LTO FP scheduling of the SHARED PCA math in
// calibrateCellShapeViaPca and breaks the bit-exact legacy flag-OFF fit (verified
// empirically — noinline/optnone/histogram-instead-of-nth_element inside Frame.cpp
// all still perturbed it). Keeping it out of Frame.cpp's TU, plus noinline so LTO
// cannot fold it back into the caller, leaves the shared codegen untouched.
__attribute__((noinline))
std::vector<BrightPixel> gatherForegroundComponent(
    const std::vector<cv::Mat> &realFrame,
    const ForegroundModel &fgModel,
    float backgroundValue,
    const cv::Point3f &center,
    float safetyRadius,
    const std::vector<cv::Point3f> &selfClaimPoints,
    const Frame::ClaimSet &otherClaimSets)
{
    const float bgLevel = fgModel.muBg;      // STABLE per-frame background mean
    const float coreRadius = 0.125f * safetyRadius;  // 0.5×maskMaxR (safety=4×maskMaxR)
    std::vector<BrightPixel> kept;
    if (realFrame.empty() || safetyRadius <= 0.0f || selfClaimPoints.empty()) {
        std::cout << "  [HalfMax] cell@(" << center.x << "," << center.y << "," << center.z << ")"
                  << " peak=0 bg=" << bgLevel << " level=0 compVox=0" << std::endl;
        return kept;
    }

    const int rows = realFrame[0].rows;
    const int cols = realFrame[0].cols;
    const int slices = static_cast<int>(realFrame.size());

    // Per-cell half-max LEVEL from stable, fit-independent image facts.
    // peak = 99th percentile of REAL intensity within a CORE sphere (coreRadius,
    // off the FIXED frame-entry radius). Percentile (not raw max) ignores hot
    // voxels. level = bg + 0.5*(peak - bg). Computed ONCE per gather. A fixed
    // 256-bin histogram over [0,1] (heap-free, no sort) gives the percentile.
    float peak = bgLevel;
    {
        const float cr = std::max(1.0f, coreRadius);
        const int cx0 = std::max(0, static_cast<int>(std::floor(center.x - cr)));
        const int cx1 = std::min(cols - 1, static_cast<int>(std::ceil(center.x + cr)));
        const int cy0 = std::max(0, static_cast<int>(std::floor(center.y - cr)));
        const int cy1 = std::min(rows - 1, static_cast<int>(std::ceil(center.y + cr)));
        const int cz0 = std::max(0, static_cast<int>(std::floor(center.z - cr)));
        const int cz1 = std::min(slices - 1, static_cast<int>(std::ceil(center.z + cr)));
        const float cr2 = cr * cr;
        std::array<int, 256> hist{};
        long total = 0;
        for (int z = cz0; z <= cz1; ++z) {
            const cv::Mat &sl = realFrame[static_cast<size_t>(z)];
            if (sl.type() != CV_32F || sl.empty()) continue;
            const float dz = static_cast<float>(z) - center.z;
            for (int y = cy0; y <= cy1; ++y) {
                const float *rp = sl.ptr<float>(y);
                const float dy = static_cast<float>(y) - center.y;
                for (int x = cx0; x <= cx1; ++x) {
                    const float dx = static_cast<float>(x) - center.x;
                    if (dx * dx + dy * dy + dz * dz > cr2) continue;
                    int b = static_cast<int>(std::clamp(rp[x], 0.0f, 1.0f) * 255.0f);
                    if (b < 0) b = 0; else if (b > 255) b = 255;
                    ++hist[static_cast<size_t>(b)];
                    ++total;
                }
            }
        }
        if (total > 0) {
            const long targetCount = static_cast<long>(std::ceil(0.99 * static_cast<double>(total)));
            long cum = 0;
            for (int b = 0; b < 256; ++b) {
                cum += hist[static_cast<size_t>(b)];
                if (cum >= targetCount) { peak = (static_cast<float>(b) + 0.5f) / 255.0f; break; }
            }
        }
    }
    const float halfMaxLevel = bgLevel + 0.5f * (peak - bgLevel);

    const int minX = std::max(0, static_cast<int>(std::floor(center.x - safetyRadius)));
    const int maxX = std::min(cols - 1, static_cast<int>(std::ceil(center.x + safetyRadius)));
    const int minY = std::max(0, static_cast<int>(std::floor(center.y - safetyRadius)));
    const int maxY = std::min(rows - 1, static_cast<int>(std::ceil(center.y + safetyRadius)));
    const int minZ = std::max(0, static_cast<int>(std::floor(center.z - safetyRadius)));
    const int maxZ = std::min(slices - 1, static_cast<int>(std::ceil(center.z + safetyRadius)));
    if (minX > maxX || minY > maxY || minZ > maxZ) return kept;

    const int bw = maxX - minX + 1;
    const int bh = maxY - minY + 1;
    const int bd = maxZ - minZ + 1;
    const size_t boxVox = static_cast<size_t>(bw) *
                          static_cast<size_t>(bh) *
                          static_cast<size_t>(bd);

    const auto distSq = [](const cv::Point3f &a, const cv::Point3f &b) {
        const float dx = a.x - b.x;
        const float dy = a.y - b.y;
        const float dz = a.z - b.z;
        return dx * dx + dy * dy + dz * dz;
    };

    // Hard Voronoi split: keep the voxel only if its nearest self-claim is at
    // least as close as its nearest neighbor claim. Same rule as the box gather.
    const auto selfClaims = [&](int x, int y, int z) -> bool {
        const cv::Point3f p{static_cast<float>(x),
                            static_cast<float>(y),
                            static_cast<float>(z)};
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
        return otherBest >= selfBest;
    };

    const auto isForeground = [&](int x, int y, int z) -> bool {
        const cv::Mat &slice = realFrame[static_cast<size_t>(z)];
        if (slice.type() != CV_32F || slice.empty()) return false;
        return slice.ptr<float>(y)[x] >= halfMaxLevel;
    };

    const auto idxOf = [&](int x, int y, int z) -> size_t {
        return (static_cast<size_t>(z - minZ) * static_cast<size_t>(bh) +
                static_cast<size_t>(y - minY)) * static_cast<size_t>(bw) +
               static_cast<size_t>(x - minX);
    };

    // Seed at the voxel nearest the center. The center normally sits on the
    // bright nucleus; if it lands below the half-max level, fall back to the
    // nearest qualifying voxel in the box. If none qualifies, cell keeps shape.
    int seedX = std::clamp(static_cast<int>(std::lround(center.x)), minX, maxX);
    int seedY = std::clamp(static_cast<int>(std::lround(center.y)), minY, maxY);
    int seedZ = std::clamp(static_cast<int>(std::lround(center.z)), minZ, maxZ);
    if (!(isForeground(seedX, seedY, seedZ) && selfClaims(seedX, seedY, seedZ))) {
        float bestD = std::numeric_limits<float>::infinity();
        bool found = false;
        for (int z = minZ; z <= maxZ; ++z) {
            for (int y = minY; y <= maxY; ++y) {
                for (int x = minX; x <= maxX; ++x) {
                    if (!isForeground(x, y, z) || !selfClaims(x, y, z)) continue;
                    const cv::Point3f p{static_cast<float>(x),
                                        static_cast<float>(y),
                                        static_cast<float>(z)};
                    const float d = distSq(p, center);
                    if (d < bestD) {
                        bestD = d;
                        seedX = x; seedY = y; seedZ = z;
                        found = true;
                    }
                }
            }
        }
        if (!found) {
            std::cout << "  [HalfMax] cell@(" << center.x << "," << center.y << "," << center.z << ")"
                      << " peak=" << peak << " bg=" << bgLevel
                      << " level=" << halfMaxLevel << " compVox=0 (no_seed)" << std::endl;
            return kept;
        }
    }

    // 6-connected BFS over the Voronoi-clipped foreground component.
    std::vector<uint8_t> visited(boxVox, 0);
    std::vector<std::array<int, 3>> stack;
    stack.reserve(1024);
    stack.push_back({seedX, seedY, seedZ});
    visited[idxOf(seedX, seedY, seedZ)] = 1;
    static const int dx6[6] = {1, -1, 0, 0, 0, 0};
    static const int dy6[6] = {0, 0, 1, -1, 0, 0};
    static const int dz6[6] = {0, 0, 0, 0, 1, -1};
    while (!stack.empty()) {
        const std::array<int, 3> v = stack.back();
        stack.pop_back();
        const int x = v[0], y = v[1], z = v[2];
        const float I = realFrame[static_cast<size_t>(z)].ptr<float>(y)[x];
        kept.push_back({cv::Point3f(static_cast<float>(x),
                                    static_cast<float>(y),
                                    static_cast<float>(z)),
                        std::max(0.0f, I - backgroundValue)});
        for (int k = 0; k < 6; ++k) {
            const int nx = x + dx6[k];
            const int ny = y + dy6[k];
            const int nz = z + dz6[k];
            if (nx < minX || nx > maxX || ny < minY || ny > maxY ||
                nz < minZ || nz > maxZ) continue;
            const size_t ni = idxOf(nx, ny, nz);
            if (visited[ni]) continue;
            visited[ni] = 1;                       // mark before test → no revisit
            if (!isForeground(nx, ny, nz)) continue;
            if (!selfClaims(nx, ny, nz)) continue;
            stack.push_back({nx, ny, nz});
        }
    }
    std::cout << "  [HalfMax] cell@(" << center.x << "," << center.y << "," << center.z << ")"
              << " peak=" << peak
              << " bg=" << bgLevel
              << " level=" << halfMaxLevel
              << " compVox=" << kept.size()
              << std::endl;
    return kept;
}
