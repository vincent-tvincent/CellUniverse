#include "../includes/SplitFeatureExtractor.hpp"
#include <algorithm>

static float clamp01(float x) { return std::min(1.0f, std::max(0.0f, x)); }

SplitFeatures buildSplitFeatures(const SplitFeatureInputs& in) {
    SplitFeatures f;
    const float minR = std::max(1e-3f, in.minR);
    f.elongation = clamp01((in.maxR / minR - 1.0f) / (2.0f - 1.0f)); // 1->0, 2->1
    // planarity: OBLATE/pancake index from sorted radii r1>=r2>=r3. (r2-r3)/r1 is
    // high for a disk (r2~r1>>r3, the metaphase plate) and ~0 for a sphere OR a
    // cigar (prolate: r2~r3) — the specific pre-division precursor that elongation
    // conflates. If midR is unset (<=0), fall back to the mean of max/min (neutral).
    {
        float r1 = in.maxR;
        float r2 = in.midR > 0.f ? in.midR : 0.5f * (in.maxR + in.minR);
        float r3 = in.minR;
        if (r1 < r2) std::swap(r1, r2);
        if (r2 < r3) std::swap(r2, r3);
        if (r1 < r2) std::swap(r1, r2);
        f.planarity = clamp01((r2 - r3) / std::max(1e-3f, r1));
    }
    f.valley     = clamp01(1.0f - in.valleyRatio);                   // dark bridge => high
    f.volumeRipe = clamp01(in.volume / std::max(1e-3f, in.birthVolume) - 1.0f); // doubled => 1
    f.centralDef = clamp01(1.0f - in.midBandBrightness / std::max(1e-3f, in.coreBrightness));
    f.centerSupport = clamp01(in.secondCenterSignal / std::max(1e-3f, in.secondCenterSignalRef));
    f.density    = clamp01(static_cast<float>(in.neighborsWithin) / std::max(1e-3f, in.densityRef));
    return f;
}
