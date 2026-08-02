#include <cassert>
#include <cstdio>
#include "../includes/SplitScore.hpp"

int main() {
    SplitScoreWeights w; // defaults
    SplitFeatures none;  // all zero -> strong no-split
    SplitFeatures ripe;  // strong division evidence
    ripe.elongation=1.f; ripe.valley=1.f; ripe.centerSupport=1.f; ripe.volumeRipe=1.f; ripe.centralDef=1.f;

    // Monotonic: more evidence -> higher probability.
    assert(splitProbability(ripe, w) > splitProbability(none, w));
    // Base rate is low with no evidence.
    assert(splitProbability(none, w) < 0.2f);
    // Ripe cell clears the accept threshold; empty cell does not.
    assert(splitScoreAccept(ripe, w));
    assert(!splitScoreAccept(none, w));
    // Density SUPPRESSES: same ripe cell in a crowded region should score lower.
    SplitFeatures ripeCrowded = ripe; ripeCrowded.density = 1.f;
    assert(splitLogit(ripeCrowded, w) < splitLogit(ripe, w));
    // Cost AS EVIDENCE: worse fit (coverageGain<0) lowers score, better raises it, neither vetoes.
    SplitFeatures ripeCostHurts = ripe; ripeCostHurts.coverageGain = -1.f;
    SplitFeatures ripeCostHelps = ripe; ripeCostHelps.coverageGain =  1.f;
    assert(splitLogit(ripeCostHurts, w) < splitLogit(ripe, w));
    assert(splitLogit(ripeCostHelps, w) > splitLogit(ripe, w));

    std::printf("test_split_score: PASS\n");
    return 0;
}
