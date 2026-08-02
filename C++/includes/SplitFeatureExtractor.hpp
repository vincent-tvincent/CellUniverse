#pragma once
#include "SplitScore.hpp"

// Raw per-cell inputs, converted to normalized SplitFeatures by buildSplitFeatures.
struct SplitFeatureInputs {
    float maxR = 0.f, midR = 0.f, minR = 0.f;  // fitted semi-axes (max, mid, min) — sorted r1>=r2>=r3
    float volume = 0.f, birthVolume = 0.f;     // current vs birth ellipsoid volume
    float valleyRatio = 1.f;                   // mid-band / edge brightness; >=1 => no valley
    float midBandBrightness = 1.f, coreBrightness = 1.f; // for centralDef
    float secondCenterSignal = 0.f;            // signal of nearest UNCLAIMED CellLumen center inside, else 0
    float secondCenterSignalRef = 100.f;       // per-frame normalizer
    int   neighborsWithin = 0;                 // neighbor centers within k*maxR
    float densityRef = 6.f;                    // normalizer
};

// coverageGain is NOT set here (accept-time feature) — it stays 0.
SplitFeatures buildSplitFeatures(const SplitFeatureInputs& in);
