#include <cassert>
#include <cmath>
#include <cstdio>
#include "../includes/SplitFeatureExtractor.hpp"

int main() {
    SplitFeatureInputs in{};
    // Round, small, single-lobe cell in isolation -> all features ~0.
    in.maxR=10; in.minR=10; in.volume=4188; in.birthVolume=4188; in.valleyRatio=1.2f;
    in.midBandBrightness=1.f; in.coreBrightness=1.f; in.secondCenterSignal=0;
    in.secondCenterSignalRef=100; in.neighborsWithin=0; in.densityRef=6;
    SplitFeatures f0 = buildSplitFeatures(in);
    assert(f0.elongation < 0.05f);
    assert(f0.valley < 0.05f);        // valleyRatio 1.2 => no dark bridge
    assert(f0.volumeRipe < 0.05f);
    assert(f0.centralDef < 0.05f);
    assert(f0.density < 0.01f);

    // Elongated, doubled-volume, dark-bridge, second-center, dumbbell -> strong.
    SplitFeatureInputs in2 = in;
    in2.maxR=20; in2.minR=10;               // elong 2.0 -> 1.0
    in2.volume=8376;                         // doubled -> volumeRipe ~1
    in2.valleyRatio=0.4f;                    // dark bridge -> valley 0.6
    in2.midBandBrightness=0.3f; in2.coreBrightness=1.f; // centralDef 0.7
    in2.secondCenterSignal=120;              // strong 2nd center -> centerSupport ~1
    SplitFeatures f1 = buildSplitFeatures(in2);
    assert(f1.elongation > 0.95f);
    assert(f1.valley > 0.55f && f1.valley < 0.65f);
    assert(f1.volumeRipe > 0.9f);
    assert(f1.centralDef > 0.65f);
    assert(f1.centerSupport > 0.9f);

    // Density normalizes and clamps.
    SplitFeatureInputs in3 = in; in3.neighborsWithin=9; in3.densityRef=6;
    assert(std::abs(buildSplitFeatures(in3).density - 1.0f) < 1e-4f); // 9/6 clamped to 1
    std::printf("test_split_feature_extractor: PASS\n");
    return 0;
}
