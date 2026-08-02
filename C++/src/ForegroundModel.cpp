#include "../includes/ForegroundModel.hpp"
#include <algorithm>
#include <cmath>

namespace {
// p-th percentile (p in [0,1]) of a mutable vector (partial sort). Empty -> 0.
float percentile(std::vector<float>& v, float p) {
    if (v.empty()) return 0.f;
    p = std::clamp(p, 0.f, 1.f);
    const size_t n = static_cast<size_t>(p * static_cast<float>(v.size() - 1));
    std::nth_element(v.begin(), v.begin() + n, v.end());
    return v[n];
}
// Population std around a given center.
float stdAround(const std::vector<float>& v, float center) {
    if (v.empty()) return 0.f;
    double s2 = 0.0;
    for (float x : v) { const double d = static_cast<double>(x) - center; s2 += d * d; }
    return static_cast<float>(std::sqrt(s2 / static_cast<double>(v.size())));
}
} // namespace

ForegroundModel estimateForegroundModel(const std::vector<cv::Mat>& real,
                                        const std::vector<uint8_t>& insideAnyCell,
                                        float minSigma) {
    ForegroundModel m;
    std::vector<float> fg, bg;
    fg.reserve(insideAnyCell.size() / 8 + 1);
    bg.reserve(insideAnyCell.size() + 1);
    double gsum = 0.0, gsq = 0.0;   // global intensity accumulators (for the sigma floor)
    size_t gn = 0;
    size_t idx = 0;
    for (const cv::Mat& slice : real) {
        if (slice.type() != CV_32F) { idx += static_cast<size_t>(slice.rows) * slice.cols; continue; }
        for (int y = 0; y < slice.rows; ++y) {
            const float* rr = slice.ptr<float>(y);
            for (int x = 0; x < slice.cols; ++x, ++idx) {
                const float I = rr[x];
                gsum += I; gsq += static_cast<double>(I) * I; ++gn;
                if (idx < insideAnyCell.size() && insideAnyCell[idx]) fg.push_back(I);
                else bg.push_back(I);
            }
        }
    }

    // Background: robust center (median) + spread. Background is genuinely dim.
    const float bgMedian = percentile(bg, 0.5f);
    m.muBg = bgMedian;
    m.sigBg = stdAround(bg, bgMedian);

    // Foreground: estimate from the BRIGHT HALF of inside-cell voxels (>= inside
    // median). The ellipsoid covers a bright core plus a large dim interior; if
    // we used the full-population median, muFg collapses to ~background and the
    // model goes flat (which froze the optimizer). The bright-half mean tracks
    // the actual signal the cell should be covering.
    if (!fg.empty()) {
        const float fgMedian = percentile(fg, 0.5f);
        double bsum = 0.0; size_t bn = 0;
        for (float x : fg) { if (x >= fgMedian) { bsum += x; ++bn; } }
        m.muFg = bn ? static_cast<float>(bsum / static_cast<double>(bn)) : fgMedian;
        double bs2 = 0.0;
        for (float x : fg) { if (x >= fgMedian) { const double d = static_cast<double>(x) - m.muFg; bs2 += d * d; } }
        m.sigFg = bn ? static_cast<float>(std::sqrt(bs2 / static_cast<double>(bn))) : m.sigBg;
    } else {
        // No foreground voxels: make logLR inert (everything reads background).
        m.muFg = m.muBg;
        m.sigFg = m.sigBg;
    }

    // Self-calibrating sigma floors that keep the cost landscape smooth enough
    // for greedy descent (and scale-free across [0,1] / [0,255] datasets):
    //  - sigFg floored at a small fraction of the global intensity std;
    //  - sigBg floored at a fraction of sigFg, so the background Gaussian is
    //    never razor-sharp relative to the foreground. A razor-sharp sigBg makes
    //    the cost near-discontinuous and dominated by a position-insensitive
    //    penalty, which froze greedy perturbation (0 moves accepted).
    const float globalStd = (gn > 0)
        ? static_cast<float>(std::sqrt(std::max(
              0.0, gsq / static_cast<double>(gn)
                       - (gsum / static_cast<double>(gn)) * (gsum / static_cast<double>(gn)))))
        : 0.f;
    m.sigFg = std::max(m.sigFg, std::max(minSigma, 0.05f * globalStd));
    m.sigBg = std::max(m.sigBg, std::max(minSigma, 0.25f * m.sigFg));
    m.finalize(minSigma);
    return m;
}
