#pragma once
#include <cmath>

// Rung 2a: combine normalized per-cell features into a split logit / probability
// / accept decision. Image cost enters as ONE weighted feature (coverageGain),
// not as an authority — real divisions barely change pixel cost, so requiring a
// cost improvement would reject them.
struct SplitFeatures {           // normalized to [0,1] except coverageGain in [-1,1]
    float elongation   = 0.f;    // clamp((maxR/minR - 1)/(2-1), 0, 1)
    float planarity    = 0.f;    // clamp((r2-r3)/r1, 0, 1) — OBLATE/metaphase-pancake precursor
    float valley       = 0.f;    // clamp(1 - valleyRatio, 0, 1)  (dark bridge between lobes)
    float centerSupport= 0.f;    // strength of a 2nd CellLumen center inside the cell, [0,1]
    float volumeRipe   = 0.f;    // clamp(vol/birthVol - 1, 0, 1)  (~1 at doubled volume)
    float centralDef   = 0.f;    // clamp(1 - midBandBrightness/coreBrightness, 0, 1) (dumbbell)
    float density      = 0.f;    // clamp(neighborsWithin(k*maxR)/densityRef, 0, 1)  (SUPPRESSOR)
    float coverageGain = 0.f;    // clamp((parentCost-daughtersCost)/parentCost, -1, 1); accept-time only
};
struct SplitScoreWeights {
    float wElong=1.0f, wPlanarity=0.0f, wValley=2.0f, wCenter=2.0f, wVolume=1.0f, wCentral=1.0f, wDensity=2.0f, wCost=1.5f;
    float bias=-3.0f;            // logit bias (sets base rate ~ sigmoid(bias))
    float sAccept=0.0f;          // accept iff logit >= sAccept
};
inline float splitLogit(const SplitFeatures& f, const SplitScoreWeights& w) {
    return w.bias + w.wElong*f.elongation + w.wPlanarity*f.planarity + w.wValley*f.valley
         + w.wCenter*f.centerSupport + w.wVolume*f.volumeRipe + w.wCentral*f.centralDef
         - w.wDensity*f.density + w.wCost*f.coverageGain;
}
inline float splitProbability(const SplitFeatures& f, const SplitScoreWeights& w) {
    const float z = splitLogit(f, w);
    return z >= 0.f ? 1.0f/(1.0f+std::exp(-z)) : std::exp(z)/(1.0f+std::exp(z));
}
inline bool  splitScoreAccept(const SplitFeatures& f, const SplitScoreWeights& w) {
    return splitLogit(f, w) >= w.sAccept;
}

// Optional multi-factor split-decision context. When passed (non-null) to
// trySplitCellPhased, the split ACCEPT is decided by the score + geometric
// sanity instead of the CellLumen cost-gate/sentinel path.
struct SplitDecisionCtx {
    SplitFeatures parentFeatures;   // parent-only features (coverageGain filled in at accept time)
    SplitScoreWeights weights;
};
