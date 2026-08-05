#include "CandidateBatch.hpp"
#include "CellRefitGuards.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>

namespace {

void require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}

ChunkEvidence makeChunk()
{
    ChunkEvidence chunk;
    chunk.stableId = 41;
    chunk.weightedCenter = cv::Point3f(11.0f, 10.0f, 10.0f);
    chunk.geometricCenter = cv::Point3f(11.5f, 10.0f, 10.0f);
    chunk.robustCenter = cv::Point3f(10.8f, 10.0f, 10.0f);
    chunk.peakCenter = cv::Point3f(12.0f, 10.0f, 10.0f);
    chunk.brightness = 0.8f;
    chunk.confidence = 0.9f;
    chunk.voxelCount = 1000.0f;
    chunk.boxCount = 4;
    return chunk;
}

CandidateBatch makeBatch(const CandidateBatchConfig &config)
{
    CandidateBatchInput input;
    input.frame = 7;
    input.parentName = "cell_4";
    input.baselinePosition = cv::Point3f(10.0f, 10.0f, 10.0f);
    input.snapshotPosition = cv::Point3f(9.0f, 10.0f, 10.0f);
    input.parentMinRadius = 5.0f;
    input.evidence = {makeChunk()};
    return CandidateBatch(input, config);
}

} // namespace

int main()
{
    try {
        {
            float evidenceRatio = 0.0f;
            require(
                !cell_refit_guards::qualifiesPcaSeparationForSplitFirst(
                    9.0f, 10.0f, 100, 1.50f, 20, evidenceRatio) &&
                    std::abs(evidenceRatio - 0.90f) < 1.0e-7f,
                "frame-1-like PCA separation must remain post-continuation");
            require(
                !cell_refit_guards::qualifiesPcaSeparationForSplitFirst(
                    7.1f, 10.0f, 100, 1.50f, 20, evidenceRatio) &&
                    std::abs(evidenceRatio - 0.71f) < 1.0e-7f,
                "frame-5-like PCA separation must remain post-continuation");
            require(
                cell_refit_guards::qualifiesPcaSeparationForSplitFirst(
                    18.6f, 10.0f, 20, 1.50f, 20, evidenceRatio) &&
                    std::abs(evidenceRatio - 1.86f) < 1.0e-7f,
                "validated division-like PCA separation must run split-first");
            require(
                !cell_refit_guards::qualifiesPcaSeparationForSplitFirst(
                    18.6f, 0.0f, 20, 1.50f, 20, evidenceRatio),
                "invalid shortest radius must never qualify for split-first");

            require(
                cell_refit_guards::rejectSiblingRodOverlap(
                    1.960f, 2.270f, 0.1887f, 1.80f, 0.07f),
                "recorded 40/41 rod-overlap false split must be rejected");
            require(
                !cell_refit_guards::rejectSiblingRodOverlap(
                    1.616f, 1.597f, 0.00159f, 1.80f, 0.07f),
                "validated 20/21 split must survive the conjunctive gate");
            require(
                !cell_refit_guards::rejectSiblingRodOverlap(
                    1.95f, 1.55f, 0.15f, 1.80f, 0.07f),
                "one rod-like daughter alone must not reject a split");

            using LocalSplitDecision =
                cell_refit_guards::LocalComponentSplitDecision;
            require(
                cell_refit_guards::evaluateLocalComponentSplitPair(
                    1, 1, 0.0f, 10.0f, 45.0f, 0.0f, 0.8f,
                    0.0f, 14.0f, 0.0f, 0.0f) ==
                    LocalSplitDecision::VetoTooFewComponents,
                "one local threshold component must veto a premature split");
            require(
                cell_refit_guards::evaluateLocalComponentSplitPair(
                    2, 2, 39.5838f, 10.4552f, 41.8209f,
                    0.9952f, 0.80f, 3.886f, 14.2571f,
                    -19.7f, 19.9f) == LocalSplitDecision::Accept,
                "recorded cell-301 local component pair must pass");
            require(
                cell_refit_guards::evaluateLocalComponentSplitPair(
                    2, 2, 25.6181f, 10.0f, 45.0f,
                    0.17887f, 0.80f, 2.0f, 14.0f,
                    -12.0f, 12.0f) ==
                    LocalSplitDecision::VetoAxisAlignment,
                "cell-200-like off-axis pair must be rejected before split cost");
            require(
                cell_refit_guards::evaluateLocalComponentSplitPair(
                    0, 0, 0.0f, 10.0f, 45.0f, 0.0f, 0.8f,
                    0.0f, 14.0f, 0.0f, 0.0f) ==
                    LocalSplitDecision::FallbackNoEvidence,
                "zero local threshold evidence must preserve legacy fallback");
            require(
                cell_refit_guards::evaluateLocalComponentSplitPair(
                    2, 0, 0.0f, 10.0f, 45.0f, 0.0f, 0.8f,
                    0.0f, 14.0f, 0.0f, 0.0f) ==
                    LocalSplitDecision::FallbackNoEvidence,
                "filtered-out local components must preserve legacy fallback");
            require(
                cell_refit_guards::shouldFallbackForSaturatedSparseLocalEvidence(
                    true, 1.0f, 0.995f, 37, 1, 2, 64),
                "cell-51 saturated sparse residue must preserve legacy fallback");
            require(
                !cell_refit_guards::shouldFallbackForSaturatedSparseLocalEvidence(
                    true, 1.0f, 0.995f, 2714, 1, 1, 64),
                "substantial one-component body must remain an explicit veto");
            require(
                !cell_refit_guards::shouldFallbackForSaturatedSparseLocalEvidence(
                    false, 1.0f, 0.995f, 37, 1, 2, 64),
                "disabled saturated sparse fallback must not alter the veto");
            require(
                !cell_refit_guards::shouldFallbackForSaturatedSparseLocalEvidence(
                    true, 0.994f, 0.995f, 37, 1, 2, 64),
                "non-saturated sparse evidence must remain an explicit veto");
            require(
                !cell_refit_guards::shouldFallbackForSaturatedSparseLocalEvidence(
                    true, 1.0f, 0.995f, 65, 1, 2, 64),
                "support above the configured sparse cap must remain a veto");
            require(
                !cell_refit_guards::shouldFallbackForSaturatedSparseLocalEvidence(
                    true, 1.0f, 0.995f, 37, 1, 1, 64),
                "a sole raw component must remain a veto even when saturated");
            require(
                !cell_refit_guards::shouldFallbackForSaturatedSparseLocalEvidence(
                    true, 1.0f, 0.995f, 68, 1, 6, 64),
                "recorded Cell 200 support above the sparse cap must not activate hybrid");
            require(
                cell_refit_guards::shouldFallbackForSaturatedSparseLocalEvidence(
                    true, 0.995f, 0.995f, 37, 1, 2, 64),
                "configured normalized threshold floor must be inclusive");
            require(
                cell_refit_guards::isSaturatedSparseLocalEvidence(
                    1.0f, 0.995f, 37, 1, 2, 64) &&
                !cell_refit_guards::shouldFallbackForSaturatedSparseLocalEvidence(
                    false, 1.0f, 0.995f, 37, 1, 2, 64),
                "hybrid must recognize Cell 51 sparse saturation even when fallback is disabled");
            require(
                cell_refit_guards::shouldFallbackForSaturatedSparseLocalEvidence(
                    true, 1.0f, 0.995f, 37, 1, 2, 64),
                "hybrid disabled with fallback enabled must preserve legacy fallback");

            require(
                !cell_refit_guards::passesSaturatedSparseHybridValleyPrefilter(
                    0.970181f, false, true, 0.85f, 1.0f),
                "ordinary proposals must retain the strict valley threshold");
            require(
                !cell_refit_guards::passesSaturatedSparseHybridValleyPrefilter(
                    0.970181f, true, false, 0.85f, 1.0f),
                "disabled hybrid override must retain the ordinary threshold");
            require(
                cell_refit_guards::passesSaturatedSparseHybridValleyPrefilter(
                    0.970181f, true, true, 0.85f, 1.0f),
                "recorded Cell 51 hybrid valley ratio must pass its narrow override");
            require(
                cell_refit_guards::passesSaturatedSparseHybridValleyPrefilter(
                    1.0f, true, true, 0.85f, 1.0f),
                "hybrid maximum valley ratio must be inclusive");
            require(
                !cell_refit_guards::passesSaturatedSparseHybridValleyPrefilter(
                    1.01f, true, true, 0.85f, 1.0f),
                "hybrid valley override must not accept ratios above its ceiling");

            using SaturatedSparseHybridDecision =
                cell_refit_guards::SaturatedSparseHybridDecision;
            // Recorded frame-31 Cell 51: local CC=(103.455,140.242,106.909),
            // current global center idx2=(117.113,133.662,105.735). The
            // generic pair instead reached distant idx8 and failed its max
            // separation gate. Hybrid acceptance must stay compact and local.
            require(
                cell_refit_guards::evaluateSaturatedSparseHybridPair(
                    true, true, true, false, 17, 6,
                    0.306525f, 0.20f, 0.862229f, 0.70f,
                    15.204f, 12.4808f, 22.6924f,
                    10.821f, 17.0193f) ==
                    SaturatedSparseHybridDecision::Accept,
                "recorded Cell 51 local-plus-idx2 compact pair must pass");
            require(
                cell_refit_guards::evaluateSaturatedSparseHybridPair(
                    true, true, true, false, 2, 0,
                    0.238396f, 0.0f, 0.576503f, 0.0f,
                    61.69f, 12.4808f, 22.6924f,
                    10.821f, 17.0193f) ==
                    SaturatedSparseHybridDecision::VetoSeparation,
                "distant global idx8 must not replace Cell 51's compact pair");
            require(
                cell_refit_guards::evaluateSaturatedSparseHybridPair(
                    true, true, false, false, 17, 6,
                    0.306525f, 0.20f, 0.862229f, 0.70f,
                    15.204f, 12.4808f, 22.6924f,
                    10.821f, 17.0193f) ==
                    SaturatedSparseHybridDecision::VetoOutsideParentScope,
                "hybrid must reject a signal outside the local parent scope");
            require(
                cell_refit_guards::evaluateSaturatedSparseHybridPair(
                    true, true, true, false, 17, 6,
                    0.306525f, 0.20f, 0.862229f, 0.70f,
                    15.204f, 12.4808f, 22.6924f,
                    17.02f, 17.0193f) ==
                    SaturatedSparseHybridDecision::VetoMidpoint,
                "hybrid must reject a compact pair whose midpoint leaves the parent");
            require(
                cell_refit_guards::evaluateSaturatedSparseHybridPair(
                    true, true, true, false, 5, 6,
                    0.306525f, 0.20f, 0.862229f, 0.70f,
                    15.204f, 12.4808f, 22.6924f,
                    10.821f, 17.0193f) ==
                    SaturatedSparseHybridDecision::VetoSignalBoxes,
                "hybrid must reject a signal below the boxes floor");
            require(
                cell_refit_guards::evaluateSaturatedSparseHybridPair(
                    true, true, true, false, 17, 6,
                    0.19f, 0.20f, 0.862229f, 0.70f,
                    15.204f, 12.4808f, 22.6924f,
                    10.821f, 17.0193f) ==
                    SaturatedSparseHybridDecision::VetoSignalBrightness,
                "hybrid must reject a signal below the brightness floor");
            require(
                cell_refit_guards::evaluateSaturatedSparseHybridPair(
                    true, true, true, false, 17, 6,
                    0.306525f, 0.20f, 0.69f, 0.70f,
                    15.204f, 12.4808f, 22.6924f,
                    10.821f, 17.0193f) ==
                    SaturatedSparseHybridDecision::VetoSignalConfidence,
                "hybrid must reject a signal below the confidence floor");
            require(
                cell_refit_guards::evaluateSaturatedSparseHybridPair(
                    true, true, true, true, 17, 6,
                    0.306525f, 0.20f, 0.862229f, 0.70f,
                    15.204f, 12.4808f, 22.6924f,
                    10.821f, 17.0193f) ==
                    SaturatedSparseHybridDecision::VetoNeighborOwned,
                "hybrid must reject a neighbor-owned signal center");
            require(
                cell_refit_guards::evaluateSaturatedSparseHybridPair(
                    false, true, true, false, 17, 6,
                    0.306525f, 0.20f, 0.862229f, 0.70f,
                    15.204f, 12.4808f, 22.6924f,
                    10.821f, 17.0193f) ==
                    SaturatedSparseHybridDecision::FallbackNotEligible,
                "disabled hybrid must preserve the normal fallback");
            require(
                cell_refit_guards::evaluateSaturatedSparseHybridPair(
                    true, false, true, false, 17, 6,
                    0.306525f, 0.20f, 0.862229f, 0.70f,
                    15.204f, 12.4808f, 22.6924f,
                    10.821f, 17.0193f) ==
                    SaturatedSparseHybridDecision::FallbackNotEligible,
                "unsaturated, large-support, single-raw, and Cell-200 paths must not activate hybrid evidence");

            const std::array<float, 3> cell51Local{
                103.455f, 140.242f, 106.909f};
            const std::array<float, 3> cell51ObservedCenter{
                117.113f, 133.662f, 105.735f};
            const auto reflectedCell51 =
                cell_refit_guards::makeSaturatedSparseHybridReflectedSeed(
                    cell51Local, cell51ObservedCenter, 1.0f);
            require(
                reflectedCell51.valid &&
                std::abs(reflectedCell51.position[0] - 130.771f) < 1.0e-3f &&
                std::abs(reflectedCell51.position[1] - 127.082f) < 1.0e-3f &&
                std::abs(reflectedCell51.position[2] - 104.561f) < 1.0e-3f,
                "Cell 51 reflected seed must land opposite the local component around G");
            const auto unreflectedCell51 =
                cell_refit_guards::makeSaturatedSparseHybridReflectedSeed(
                    cell51Local, cell51ObservedCenter, 0.0f);
            require(
                unreflectedCell51.valid &&
                unreflectedCell51.position == cell51ObservedCenter,
                "zero reflection scale must keep the generated seed at G");
            require(
                !cell_refit_guards::makeSaturatedSparseHybridReflectedSeed(
                    cell51Local, cell51ObservedCenter,
                    std::numeric_limits<float>::quiet_NaN()).valid,
                "non-finite reflection scale must never produce a seed");
            using GeneratedSeedDecision =
                cell_refit_guards::SaturatedSparseHybridGeneratedSeedDecision;
            require(
                cell_refit_guards::evaluateSaturatedSparseHybridGeneratedSeed(
                    true, false, false) ==
                    GeneratedSeedDecision::VetoOutsideParentScope,
                "reflected seed outside the hybrid parent scope must be rejected");
            require(
                cell_refit_guards::evaluateSaturatedSparseHybridGeneratedSeed(
                    true, true, true) ==
                    GeneratedSeedDecision::VetoNeighborOwned,
                "neighbor-owned reflected seed must be rejected");

            const auto dropped = cell_refit_guards::detectBackgroundDrop(
                0.10, 0.05, 0.90, 0.05, 0.25);
            require(dropped.valid && dropped.dropped &&
                        dropped.observedModelContrastRatio < 0.07,
                    "collapsed observed/model contrast must trigger reattachment");
            const auto supported = cell_refit_guards::detectBackgroundDrop(
                0.55, 0.05, 0.65, 0.05, 0.25);
            require(supported.valid && !supported.dropped,
                    "a supported bright body must remain in place");
            require(
                cell_refit_guards::acceptBackgroundDropProbe(
                    true, 0.63924, 0.588953, 1.21832, 0.95),
                "CU4 must accept the recorded nearest viable bright center");
            require(
                !cell_refit_guards::acceptBackgroundDropProbe(
                    false, 0.63924, 0.588953, 1.21832, 0.95),
                "legacy rescue must retain its composite probe-score gate");
            require(
                !cell_refit_guards::acceptBackgroundDropProbe(
                    true, 0.50, 0.588953, 0.10, 0.95),
                "CU4 nearest-center policy must retain the brightness gate");
            require(
                cell_refit_guards::acceptBackgroundDropProbe(
                    true,
                    0.63924,
                    0.588953,
                    std::numeric_limits<double>::quiet_NaN(),
                    std::numeric_limits<double>::quiet_NaN()),
                "CU4 must not require the legacy score it intentionally ignores");
            require(
                !cell_refit_guards::acceptBackgroundDropProbe(
                    false,
                    0.63924,
                    0.588953,
                    std::numeric_limits<double>::quiet_NaN(),
                    0.95),
                "legacy rescue must reject a non-finite composite score");
            require(
                cell_refit_guards::preferNearestCenterClaim(
                    2.0, 4, 3.0, 1),
                "the nearer CU4 cell must receive a shared bright center");
            require(
                !cell_refit_guards::preferNearestCenterClaim(
                    3.0, 1, 2.0, 4),
                "a farther CU4 cell must not steal a shared bright center");
            require(
                cell_refit_guards::preferNearestCenterClaim(
                    std::numeric_limits<double>::quiet_NaN(),
                    1,
                    std::numeric_limits<double>::quiet_NaN(),
                    2),
                "non-finite distances must still have deterministic ordering");

            require(
                std::abs(cell_refit_guards::pcaLowSnrCutoff(
                             true, true, 0.20f, 0.60f) -
                         0.20f) < 1.0e-7f,
                "CU4 volume caps must not masquerade as a noise cutoff");
            require(
                std::abs(cell_refit_guards::pcaLowSnrCutoff(
                             true, false, 0.20f, 0.60f) -
                         0.60f) < 1.0e-7f,
                "disabled-mode SNR behavior must remain unchanged");
            require(
                cell_refit_guards::rescueLowSnrWithAnchoredComponent(
                    true, true, 24780, 1825, 18245, 1, 0.674755f,
                    0.75f, 1.50f),
                "recorded cell-20 support must qualify for anchored component rescue");
            require(
                !cell_refit_guards::rescueLowSnrWithAnchoredComponent(
                    true, true, 1993, 1039, 10385, 1, 4.30912f,
                    0.75f, 1.50f),
                "dropped cell-11 support must not qualify for shape rescue");
            require(
                !cell_refit_guards::rescueLowSnrWithAnchoredComponent(
                    true, true, 24780, 1825, 18245, 0, 0.674755f,
                    0.75f, 1.50f),
                "shape rescue must require an explicitly selected component");
            const std::array<std::array<float, 3>, 6> permutations{{
                {{8.0f, 2.0f, 1.0f}}, {{8.0f, 1.0f, 2.0f}},
                {{2.0f, 8.0f, 1.0f}}, {{1.0f, 8.0f, 2.0f}},
                {{2.0f, 1.0f, 8.0f}}, {{1.0f, 2.0f, 8.0f}}
            }};
            for (const auto &input : permutations) {
                const auto projected =
                    cell_refit_guards::projectAspectRatiosShrinkOnly(
                        input, 1.60f, 2.75f, 2.00f);
                require(projected.valid && projected.changed,
                        "severe rod must receive a valid hard projection");
                require(projected.afterLongMid <= 1.60001f &&
                            projected.afterMidShort <= 2.75001f &&
                            projected.afterLongShort <= 2.00001f,
                        "hard projection must satisfy every enabled ratio");
                for (size_t axis = 0; axis < 3; ++axis) {
                    require(projected.radii[axis] <= input[axis] + 1.0e-6f,
                            "hard aspect projection must be shrink-only");
                }
            }
            const auto sphere =
                cell_refit_guards::projectAspectRatiosShrinkOnly(
                    {3.0f, 3.0f, 3.0f}, 1.60f, 2.75f, 2.00f);
            require(sphere.valid && !sphere.changed,
                    "a valid sphere must remain unchanged");
            const auto oblate =
                cell_refit_guards::projectAspectRatiosShrinkOnly(
                    {3.0f, 3.0f, 2.0f}, 1.60f, 2.75f, 2.00f);
            require(oblate.valid && !oblate.changed,
                    "a valid oblate spheroid must remain unchanged");
            const auto recordedCell10 =
                cell_refit_guards::projectAspectRatiosShrinkOnly(
                    {21.856f, 15.0988f, 8.72155f},
                    1.60f, 2.75f, 2.00f);
            require(recordedCell10.valid && recordedCell10.changed &&
                        recordedCell10.afterLongShort <= 2.00001f,
                    "recorded cell-10 rod tip must be capped at 2:1");

            const auto recordedCell21Rounding =
                cell_refit_guards::rescueRounderShapeAtFixedVolume(
                    {22.6346f, 20.8780f, 14.6631f},
                    {22.6346f, 19.5500f, 13.1560f},
                    0.05f, 0.15f);
            require(
                recordedCell21Rounding.valid &&
                    recordedCell21Rounding.applied &&
                    recordedCell21Rounding.rescuedElongation <
                        recordedCell21Rounding.cappedElongation - 0.05f,
                "recorded cell-21 PCA fit must recover rounder proportions");
            require(
                std::abs(
                    recordedCell21Rounding.rescuedRadiusProduct /
                        recordedCell21Rounding.cappedRadiusProduct -
                    1.0) < 1.0e-5,
                "rounding rescue must preserve the temporal cap volume");
            const auto noRoundingForWorseFit =
                cell_refit_guards::rescueRounderShapeAtFixedVolume(
                    {8.0f, 2.0f, 1.0f}, {3.0f, 3.0f, 3.0f},
                    0.05f, 0.15f);
            require(
                noRoundingForWorseFit.valid &&
                    !noRoundingForWorseFit.applied,
                "a more elongated PCA fit must never trigger rounding rescue");

            const float percentile95SurfaceCorrection =
                cell_refit_guards::
                    ellipsoidAxisPercentileSurfaceCorrection(0.95f);
            require(
                std::abs(percentile95SurfaceCorrection - 1.232f) < 0.002f,
                "95th-percentile ellipsoid extents must be corrected from "
                "an interior chord to the surface radius");
            require(
                cell_refit_guards::
                    ellipsoidAxisPercentileSurfaceCorrection(1.0f) == 1.0f,
                "maximum component extent must not receive percentile "
                "surface correction");

            const auto growth =
                cell_refit_guards::reconcileIntegratedContrast(
                    1000, 0.20, 2000, 400.0,
                    1.0, 0.0, 1.0, 0.0);
            require(growth.valid && growth.applied &&
                        growth.contrastScale < 1.0 &&
                        std::abs(0.20 * growth.contrastScale - 0.10) < 1.0e-9,
                    "larger support must receive lower foreground contrast");
            const auto shrink =
                cell_refit_guards::reconcileIntegratedContrast(
                    1000, 0.20, 500, 100.0,
                    1.0, 0.0, 1.0, 0.0);
            require(shrink.valid && shrink.applied &&
                        shrink.contrastScale > 1.0 &&
                        std::abs(0.20 * shrink.contrastScale - 0.40) < 1.0e-9,
                    "smaller support must receive higher foreground contrast");
            const auto withinMargin =
                cell_refit_guards::reconcileIntegratedContrast(
                    1000, 0.20, 1000, 210.0,
                    1.0, 0.14, 1.0, 0.0);
            require(withinMargin.valid && !withinMargin.applied &&
                        std::abs(withinMargin.contrastScale - 1.0) < 1.0e-12,
                    "brightness-volume variation inside the deadband must remain");
            const auto siblingGroup =
                cell_refit_guards::reconcileIntegratedContrast(
                    1000, 0.20, 2000, 340.0,
                    1.0, 0.0, 1.0, 0.0);
            require(siblingGroup.valid && siblingGroup.contrastScale < 1.0,
                    "newborn siblings must share one parent-group scale");
            const auto zeroContrastShrink =
                cell_refit_guards::reconcileIntegratedContrast(
                    1000, 0.20, 500, 0.0,
                    1.0, 0.0, 0.50, 0.20);
            require(zeroContrastShrink.valid &&
                        zeroContrastShrink.applied &&
                        zeroContrastShrink.usedZeroContrastFallback &&
                        std::abs(
                            zeroContrastShrink.reconciledIntegratedContrast -
                            40.0) < 1.0e-9,
                    "zero-contrast shrinking cells need bounded recovery");
            const double background = 0.35;
            const double proposedBrightness = 0.55;
            require(background +
                            (proposedBrightness - background) *
                                growth.contrastScale <
                        proposedBrightness,
                    "nonzero background must be added after contrast scaling");
        }

        CandidateBatchConfig config;
        config.enabled = true;
        config.maxCandidatesPerParent = 8;
        config.expensiveTopK = 4;
        config.maxExactChunkCenters = 4;
        config.stochasticCandidatesPerParent = 2;
        config.deduplicationDistanceAbsolute = 0.05f;
        config.validate();

        CandidateBatch first = makeBatch(config);
        CandidateBatch second = makeBatch(config);
        require(!first.candidates().empty(), "batch must contain no-op");
        require(first.candidates().front().source == CandidateSource::NoOp,
                "candidate zero must be no-op");
        require(first.candidates().size() <= 8,
                "candidate cap must be enforced");
        require(first.candidates().size() == second.candidates().size(),
                "deterministic batches must have equal size");
        for (std::size_t i = 0; i < first.candidates().size(); ++i) {
            require(first.candidates()[i].stableId ==
                        second.candidates()[i].stableId,
                    "deterministic candidate order changed");
            require(cv::norm(first.candidates()[i].position -
                             second.candidates()[i].position) < 1.0e-7,
                    "deterministic candidate position changed");
        }
        std::set<std::uint64_t> stableIds;
        for (const CandidateProposal &candidate : first.candidates()) {
            require(stableIds.insert(candidate.stableId).second,
                    "candidate stable IDs must be unique within a batch");
        }

        CandidateBatchInput currentEvidenceInput;
        currentEvidenceInput.frame = 31;
        currentEvidenceInput.parentName = "cell_51";
        currentEvidenceInput.baselinePosition = cv::Point3f(10.0f, 10.0f, 10.0f);
        currentEvidenceInput.parentMinRadius = 5.0f;
        ChunkEvidence currentEvidence = makeChunk();
        currentEvidence.sourceFrameOffset = 0;
        currentEvidenceInput.evidence = {currentEvidence};
        CandidateBatchConfig lookaheadConfig = config;
        lookaheadConfig.lookaheadEvidenceEnabled = true;
        lookaheadConfig.lookaheadFrames = 2;
        lookaheadConfig.lookaheadPriorityPenaltyPerFrame = 0.40f;
        CandidateBatch currentEvidenceBatch(currentEvidenceInput,
                                            lookaheadConfig);

        CandidateBatchInput futureEvidenceInput = currentEvidenceInput;
        futureEvidenceInput.evidence[0].sourceFrameOffset = 2;
        CandidateBatch futureEvidenceBatch(futureEvidenceInput,
                                           lookaheadConfig);
        const auto findWeighted = [](const CandidateBatch &batch)
            -> const CandidateProposal * {
            for (const CandidateProposal &candidate : batch.candidates()) {
                if (candidate.source == CandidateSource::ChunkWeighted) {
                    return &candidate;
                }
            }
            return nullptr;
        };
        const CandidateProposal *currentWeighted =
            findWeighted(currentEvidenceBatch);
        const CandidateProposal *futureWeighted =
            findWeighted(futureEvidenceBatch);
        require(currentWeighted != nullptr && futureWeighted != nullptr,
                "current and future exact centers must both be representable");
        require(currentWeighted->evidenceFrameOffset == 0 &&
                    futureWeighted->evidenceFrameOffset == 2,
                "candidate diagnostics must retain evidence frame offset");
        require(std::abs(
                    (futureWeighted->cheapPriority -
                     currentWeighted->cheapPriority) - 0.80) < 1.0e-6,
                "future-center priority must pay the configured per-frame penalty");
        require(currentWeighted->stableId != futureWeighted->stableId,
                "future and current evidence must have distinct stable IDs");

        CandidateBatchInput otherParentInput;
        otherParentInput.frame = 7;
        otherParentInput.parentName = "cell_5";
        otherParentInput.baselinePosition =
            cv::Point3f(10.0f, 10.0f, 10.0f);
        otherParentInput.snapshotPosition =
            cv::Point3f(9.0f, 10.0f, 10.0f);
        otherParentInput.parentMinRadius = 5.0f;
        otherParentInput.evidence = {makeChunk()};
        CandidateBatch otherParent(otherParentInput, config);
        require(first.candidates().size() == otherParent.candidates().size(),
                "parent-name identity test requires matching candidate sets");
        for (std::size_t i = 1; i < first.candidates().size(); ++i) {
            require(first.candidates()[i].stableId !=
                        otherParent.candidates()[i].stableId,
                    "candidate stable IDs must include the parent identity");
        }

        const auto expensive = first.expensiveCandidateIndices();
        require(expensive.size() == 4,
                "expensive top-k must include exactly four candidates");
        require(expensive.front() == 0,
                "expensive set must include no-op");
        first.recordEvaluation(expensive[1], -5.0);
        first.recordEvaluation(expensive[2], -2.0);
        first.recordEvaluation(expensive[3], 1.0);
        const auto winner = first.selectWinner(100.0);
        require(winner.has_value() && *winner == expensive[1],
                "batch must select the best improving candidate");

        CandidateBatchConfig guarded = config;
        guarded.absoluteImprovementMargin = 6.0;
        CandidateBatch noWinner = makeBatch(guarded);
        const auto guardedExpensive = noWinner.expensiveCandidateIndices();
        noWinner.recordEvaluation(guardedExpensive[1], -5.0);
        require(!noWinner.selectWinner(100.0).has_value(),
                "improvement margin must preserve no-op");

        const double drawA = CandidateBatch::deterministicUnit(
            7, "cu4-test", 12, "cell_4", "split");
        const double drawB = CandidateBatch::deterministicUnit(
            7, "cu4-test", 12, "cell_4", "split");
        const double drawOther = CandidateBatch::deterministicUnit(
            7, "cu4-test", 12, "cell_5", "split");
        require(drawA == drawB, "deterministic draw changed");
        require(drawA >= 0.0 && drawA < 1.0,
                "deterministic draw must be in [0,1)");
        require(drawA != drawOther,
                "different parent keys should produce different draws");

        CandidateBatchInput multiChunkInput;
        multiChunkInput.frame = 15;
        multiChunkInput.parentName = "cell_5";
        multiChunkInput.baselinePosition = cv::Point3f(140.0f, 132.0f, 140.0f);
        multiChunkInput.parentMinRadius = 18.0f;
        ChunkEvidence firstChunk = makeChunk();
        firstChunk.stableId = 501;
        firstChunk.weightedCenter = cv::Point3f(133.0f, 121.0f, 168.0f);
        firstChunk.geometricCenter = cv::Point3f(133.5f, 121.5f, 167.5f);
        firstChunk.robustCenter = cv::Point3f(133.0f, 121.0f, 167.0f);
        firstChunk.peakCenter = cv::Point3f(132.0f, 120.0f, 168.0f);
        ChunkEvidence secondChunk = firstChunk;
        secondChunk.stableId = 502;
        secondChunk.weightedCenter = cv::Point3f(146.0f, 141.0f, 119.0f);
        secondChunk.geometricCenter = cv::Point3f(146.5f, 140.5f, 119.5f);
        secondChunk.robustCenter = cv::Point3f(146.0f, 141.0f, 119.0f);
        secondChunk.peakCenter = cv::Point3f(147.0f, 141.0f, 118.0f);
        multiChunkInput.evidence = {firstChunk, secondChunk};
        CandidateBatchConfig multiChunkConfig = config;
        multiChunkConfig.maxCandidatesPerParent = 12;
        multiChunkConfig.expensiveTopK = 6;
        multiChunkConfig.stochasticCandidatesPerParent = 0;
        CandidateBatch multiChunkBatch(multiChunkInput, multiChunkConfig);
        std::set<std::uint64_t> multiChunkIds;
        bool sawFirstChunk = false;
        bool sawSecondChunk = false;
        for (const CandidateProposal &candidate : multiChunkBatch.candidates()) {
            require(multiChunkIds.insert(candidate.stableId).second,
                    "multi-chunk candidate IDs must be unique");
            if (candidate.evidenceId == firstChunk.stableId) sawFirstChunk = true;
            if (candidate.evidenceId == secondChunk.stableId) sawSecondChunk = true;
            if (candidate.source == CandidateSource::ChunkWeighted ||
                candidate.source == CandidateSource::ChunkGeometric ||
                candidate.source == CandidateSource::ChunkRobust ||
                candidate.source == CandidateSource::ChunkPeak) {
                require(cv::norm(candidate.position) > 1.0f,
                        "populated chunk evidence must not create an origin proposal");
            }
        }
        require(sawFirstChunk && sawSecondChunk,
                "distinct chunks must retain distinct evidence identities");

        CandidateBatchConfig invalid = config;
        invalid.expensiveTopK = invalid.maxCandidatesPerParent + 1;
        bool threw = false;
        try {
            invalid.validate();
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        require(threw, "invalid top-k must be rejected");

        invalid = config;
        invalid.prioritySnapshotBias =
            std::numeric_limits<float>::quiet_NaN();
        threw = false;
        try {
            invalid.validate();
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        require(threw, "non-finite priority biases must be rejected");

        invalid = config;
        invalid.splitScheduleForceProbabilityThreshold = 1.01f;
        threw = false;
        try {
            invalid.validate();
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        require(threw, "split force probability threshold above one must be rejected");

        invalid = config;
        invalid.splitDaughterCooldownEvidenceBypassMinAgeFrames = -1;
        threw = false;
        try {
            invalid.validate();
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        require(threw, "negative cooldown bypass age must be rejected");

        invalid = config;
        invalid.splitDaughterCooldownEvidenceBypassMinSeparationShortRadiusFraction =
            std::numeric_limits<float>::quiet_NaN();
        threw = false;
        try {
            invalid.validate();
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        require(threw, "non-finite cooldown bypass ratio must be rejected");

        invalid = config;
        invalid.lookaheadFrames = -1;
        threw = false;
        try {
            invalid.validate();
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        require(threw, "negative lookahead frame count must be rejected");

        invalid = config;
        invalid.lookaheadPriorityPenaltyPerFrame =
            std::numeric_limits<float>::quiet_NaN();
        threw = false;
        try {
            invalid.validate();
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        require(threw, "non-finite lookahead penalty must be rejected");

        invalid = config;
        invalid.backgroundDropObservedTopFraction = 0.0f;
        threw = false;
        try {
            invalid.validate();
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        require(threw, "background-drop top fraction must be positive");

#ifdef CELLUNIVERSE_TEST_CONFIG_DIR
        const std::string configDir = CELLUNIVERSE_TEST_CONFIG_DIR;
        const YAML::Node traditionalNode =
            CellUniverseConfig::loadConfigYamlNode(
                configDir + "/config.yaml");
        BaseConfig traditional;
        traditional.explodeConfig(traditionalNode);
        require(!traditional.simulation.celluniverse4_enabled,
                "traditional config must not silently enable CU4");
        require(!traditional.candidateBatch.enabled,
                "traditional config must not enable candidate batches");
        require(!traditional.simulation.celluniverse4_component_filter_enabled,
                "traditional config must not enable CU4 component filtering");
        require(!traditional.simulation
                     .celluniverse4_pca_volume_cap_snr_decoupling_enabled,
                "traditional config must not alter PCA SNR decisions");
        require(!traditional.simulation
                     .celluniverse4_pca_component_low_snr_rescue_enabled,
                "traditional config must not rescue CU4 component fits");
        require(!traditional.simulation
                     .celluniverse4_pca_iterative_regather_enabled,
                "traditional config must not iteratively regather CU4 PCA support");
        require(
            traditional.simulation
                    .celluniverse4_pca_component_anchor_radius_scale == 0.0f,
            "traditional config must not enable CU4 PCA anchor trimming");
        require(
            !traditional.simulation
                 .celluniverse4_postfit_brightness_volume_reconcile_enabled,
            "traditional config must not enable CU4 brightness-volume coupling");
        require(traditional.cell != nullptr &&
                    !traditional.cell->pcaShapeRatioBoundHardLimitEnabled,
                "traditional config must not enable CU4 hard shape projection");

        const YAML::Node cu2Node =
            CellUniverseConfig::loadConfigYamlNode(
                configDir + "/config_celluniverse2.yaml");
        BaseConfig cu2;
        cu2.explodeConfig(cu2Node);
        require(cu2.simulation.celluniverse2_enabled,
                "CU2 profile must still enable CU2");
        require(!cu2.simulation.celluniverse4_enabled &&
                    !cu2.candidateBatch.enabled,
                "CU2 profile must remain isolated from CU4");

        const YAML::Node cu3Node =
            CellUniverseConfig::loadConfigYamlNode(
                configDir + "/config_celluniverse3.yaml");
        BaseConfig cu3;
        cu3.explodeConfig(cu3Node);
        require(cu3.simulation.celluniverse3_enabled,
                "CU3 profile must still enable CU3");
        require(!cu3.simulation.celluniverse4_enabled &&
                    !cu3.candidateBatch.enabled,
                "CU3 profile must remain isolated from CU4");

        const YAML::Node cu4Node =
            CellUniverseConfig::loadConfigYamlNode(
                configDir + "/config_celluniverse4.yaml");
        BaseConfig cu4;
        cu4.explodeConfig(cu4Node);
        require(cu4.simulation.celluniverse4_enabled,
                "CU4 profile must enable CU4");
        require(cu4.candidateBatch.enabled,
                "CU4 profile must enable candidate batches");
        require(
            cu4.candidateBatch
                .localComponentSplitSaturatedSparseHybridValleyOverrideEnabled &&
                std::abs(
                    cu4.candidateBatch
                        .localComponentSplitSaturatedSparseHybridMaxValleyRatio -
                    1.0f) < 1.0e-7f &&
                cu4.candidateBatch
                    .localComponentSplitSaturatedSparseHybridCostOverrideEnabled &&
                std::abs(
                    cu4.candidateBatch
                        .localComponentSplitSaturatedSparseHybridMinCostImprovement -
                    1500.0f) < 1.0e-7f,
            "CU4 must enable only the bounded saturated-sparse hybrid valley override");
        require(cu4.candidateBatch.expensiveTopK == 6,
                "CU4 profile must default to top-K 6");
        require(std::abs(
                    cu4.candidateBatch.splitScheduleForceProbabilityThreshold -
                    0.05f) < 1.0e-7f,
                "CU4 profile must enable the bounded split urgency threshold");
        require(
            cu4.candidateBatch.splitScheduleEvidenceFirstEnabled &&
                std::abs(
                    cu4.candidateBatch
                            .splitScheduleEvidenceFirstMinSeparationShortRadiusFraction -
                    1.50f) < 1.0e-7f &&
                cu4.candidateBatch
                        .splitScheduleEvidenceFirstMinKeptPixels == 20,
            "CU4 must prioritize only strong PCA-separation split evidence");
        require(!cu4.candidateBatch.splitRejectCompensationEnabled,
                "CU4 rejected splits must preserve the restored parent");
        require(cu4.simulation.celluniverse4_component_filter_enabled &&
                    cu4.simulation
                            .celluniverse4_component_connectivity_radius == 1 &&
                    cu4.simulation.celluniverse4_component_min_voxels == 20 &&
                    std::abs(
                        cu4.simulation
                            .celluniverse4_component_companion_min_fraction -
                        0.60f) < 1.0e-7f &&
                    cu4.simulation.celluniverse4_component_max_components == 1 &&
                    std::abs(
                        cu4.simulation
                            .celluniverse4_pca_component_anchor_radius_scale -
                        1.50f) < 1.0e-7f,
                "CU4 component support policy must default on");
        require(
            cu4.simulation
                .celluniverse4_pca_volume_cap_snr_decoupling_enabled,
            "CU4 must decouple volume caps from physical SNR decisions");
        require(
            cu4.simulation
                    .celluniverse4_pca_component_low_snr_rescue_enabled &&
                std::abs(
                    cu4.simulation
                        .celluniverse4_pca_component_low_snr_rescue_min_support_fraction -
                    0.75f) < 1.0e-7f &&
                std::abs(
                    cu4.simulation
                        .celluniverse4_pca_component_low_snr_rescue_max_anchor_distance -
                    1.50f) < 1.0e-7f,
            "CU4 must rescue only large components anchored near the tracked center");
        require(
            cu4.simulation.celluniverse4_pca_iterative_regather_enabled &&
                std::abs(
                    cu4.simulation
                            .celluniverse4_pca_iterative_regather_min_shift_radius_fraction -
                        0.48f) < 1.0e-7f &&
                cu4.simulation
                        .celluniverse4_pca_iterative_regather_max_steps == 4,
            "CU4 must enable bounded iterative owned-support PCA regathering");
        require(
            cu4.simulation
                    .celluniverse4_postfit_brightness_volume_reconcile_enabled &&
                std::abs(
                    cu4.simulation
                            .celluniverse4_postfit_brightness_volume_inverse_volume_exponent -
                    1.0f) < 1.0e-7f &&
                std::abs(
                    cu4.simulation
                            .celluniverse4_postfit_brightness_volume_log_deadband -
                    0.14f) < 1.0e-7f &&
                std::abs(
                    cu4.simulation
                            .celluniverse4_postfit_brightness_volume_strength -
                    0.50f) < 1.0e-7f &&
                std::abs(
                    cu4.simulation
                            .celluniverse4_postfit_brightness_volume_max_contrast_step_fraction -
                    0.20f) < 1.0e-7f &&
                cu4.simulation
                    .celluniverse4_postfit_brightness_volume_group_newborn_siblings &&
                cu4.simulation
                    .celluniverse4_postfit_brightness_volume_skip_resume_first_frame,
            "CU4 inverse size-brightness coupling must default on");
        require(
            cu4.cell != nullptr &&
                cu4.cell->pcaShapeRatioBoundEnabled &&
                cu4.cell->pcaShapeRatioBoundHardLimitEnabled &&
                cu4.cell->pcaShapeRatioBoundPreserveSplitEvidence &&
                cu4.cell->pcaShapeRatioBoundMinAgeFrames == 0 &&
                std::abs(cu4.cell->pcaShapeMaxLongMidRatio - 1.60f) <
                    1.0e-7f &&
                std::abs(cu4.cell->pcaShapeMaxMidShortRatio - 2.75f) <
                    1.0e-7f &&
                std::abs(cu4.cell->pcaShapeMaxLongShortRatio - 2.00f) <
                    1.0e-7f,
            "CU4 hard spheroid guard must default on");
        require(
            cu4.candidateBatch
                .splitDaughterCooldownEvidenceBypassEnabled &&
                cu4.candidateBatch
                        .splitDaughterCooldownEvidenceBypassMinAgeFrames == 6 &&
                std::abs(
                    cu4.candidateBatch
                        .splitDaughterCooldownEvidenceBypassMinSeparationShortRadiusFraction -
                    1.50f) < 1.0e-7f &&
                cu4.candidateBatch
                        .splitDaughterCooldownEvidenceBypassMinKeptPixels == 20,
            "CU4 evidence-qualified daughter cooldown bypass must default on");
        require(cu4.prob.celluniverse4_split_candidate_family_reserve_enabled &&
                    cu4.prob.celluniverse4_split_candidate_geometry_prefilter_enabled &&
                    cu4.prob.celluniverse4_split_use_winner_seed_axis_gate_enabled &&
                    cu4.prob.celluniverse4_split_finalist_retry_enabled &&
                    cu4.prob.celluniverse4_split_finalist_retry_limit == 4,
                "CU4 split candidate safeguards must default on");
        require(
            cu4.prob.celluniverse4_split_sibling_rod_overlap_gate_enabled &&
                std::abs(
                    cu4.prob
                            .celluniverse4_split_sibling_rod_min_raw_shape_ratio -
                    1.80f) < 1.0e-7f &&
                std::abs(
                    cu4.prob
                            .celluniverse4_split_sibling_rod_max_overlap_fraction -
                    0.07f) < 1.0e-7f,
            "CU4 must enable the conjunctive rod-overlap split rejection");
        require(
            cu4.candidateBatch.backgroundDropReattachEnabled &&
                std::abs(
                    cu4.candidateBatch
                            .backgroundDropObservedTopFraction -
                    0.30f) < 1.0e-7f &&
                std::abs(
                    cu4.candidateBatch
                            .backgroundDropMaxObservedModelContrastRatio -
                    0.25f) < 1.0e-7f &&
                cu4.candidateBatch.backgroundDropSiblingHalfspaceEnabled &&
                cu4.candidateBatch.backgroundDropHoldIfNoSafeCenter &&
                cu4.candidateBatch
                    .backgroundDropSkipCandidateBatchIfReattached,
            "CU4 must enable pre-perturb background-drop reattachment");

        const YAML::Node c6Node =
            CellUniverseConfig::loadConfigYamlNode(
                configDir +
                "/experiments/pavak-pos0-split-preprocess-20260801/cu4_normalized_n2v2_c6.yaml");
        BaseConfig c6;
        c6.explodeConfig(c6Node);
        require(
            c6.simulation
                    .celluniverse4_postfit_brightness_volume_reconcile_enabled &&
                c6.simulation
                    .celluniverse4_pca_volume_cap_snr_decoupling_enabled &&
                c6.simulation
                    .celluniverse4_pca_component_low_snr_rescue_enabled &&
                c6.cell != nullptr &&
                c6.cell->pcaShapeRatioBoundHardLimitEnabled &&
                c6.candidateBatch.splitScheduleEvidenceFirstEnabled &&
                c6.candidateBatch.backgroundDropReattachEnabled &&
                c6.candidateBatch
                    .backgroundDropExplicitSplitBypassHoldEnabled &&
                c6.candidateBatch.perParentFreshEvidencePoolEnabled &&
                c6.candidateBatch.lookaheadEvidenceEnabled &&
                c6.candidateBatch.lookaheadFrames == 2 &&
                std::abs(
                    c6.candidateBatch.lookaheadPriorityPenaltyPerFrame -
                    0.40f) < 1.0e-7f &&
                c6.candidateBatch.continuationSiblingHalfspaceEnabled &&
                std::abs(
                    c6.candidateBatch
                        .continuationSiblingHalfspaceToleranceMinRadiusFraction) <
                    1.0e-7f &&
                c6.prob.signal_center_split_enabled &&
                c6.prob.signal_center_neighbor_claim_gate_enabled &&
                c6.prob.signal_center_future_rescue_enabled &&
                !c6.prob
                    .signal_center_future_rescue_min_separation_gate_enabled &&
                c6.prob
                    .signal_center_future_rescue_max_separation_gate_enabled &&
                std::abs(
                    c6.prob
                            .signal_center_future_rescue_max_separation_parent_fraction -
                        5.0f) < 1.0e-7f &&
                c6.prob.signal_center_future_rescue_midpoint_gate_enabled &&
                std::abs(
                    c6.prob
                            .signal_center_future_rescue_max_midpoint_parent_fraction -
                        2.0f) < 1.0e-7f &&
                c6.prob
                    .signal_center_neighbor_owned_future_rescue_scope_gate_enabled &&
                c6.prob.signal_center_neighbor_owned_future_rescue_min_pairs == 3 &&
                c6.prob
                    .celluniverse4_rod_tip_split_recovery_enabled &&
                c6.prob.celluniverse4_rod_tip_hard_reject_enabled &&
                c6.prob
                    .celluniverse4_split_daughter_background_reattach_enabled &&
                std::abs(
                    c6.prob
                            .celluniverse4_split_daughter_background_probe_radius_parent_fraction -
                        0.25f) < 1.0e-7f &&
                std::abs(
                    c6.prob
                            .celluniverse4_split_daughter_background_reattach_max_distance_parent_radius_fraction -
                        4.0f) < 1.0e-7f &&
                c6.prob
                        .celluniverse4_split_daughter_background_reattach_min_center_boxes ==
                    2 &&
                std::abs(
                    c6.prob
                            .celluniverse4_rod_tip_min_long_short_ratio -
                        1.80f) < 1.0e-7f &&
                std::abs(
                    c6.prob
                            .celluniverse4_rod_tip_min_long_mid_ratio -
                        1.55f) < 1.0e-7f &&
                std::abs(c6.prob.split_cost - 2500.0f) < 1.0e-7f &&
                c6.prob
                    .celluniverse4_split_sibling_rod_overlap_gate_enabled,
            "Pos0 C6 experiment must enable the complete CU4 refit policy");

        const YAML::Node activePos0Node =
            CellUniverseConfig::loadConfigYamlNode(
                configDir +
                "/experiments/pavak-pos0-split-preprocess-20260801/cu4_bg_local_split_bright_review.yaml");
        BaseConfig activePos0;
        activePos0.explodeConfig(activePos0Node);
        require(
            activePos0.cell != nullptr &&
                std::abs(
                    activePos0.simulation
                            .celluniverse4_postfit_bright_cell_size_review_min_brightness -
                        0.30f) < 1.0e-7f &&
                activePos0.simulation
                    .celluniverse4_postfit_bright_cell_size_review_use_cell_brightness_threshold_enabled &&
                std::abs(
                    activePos0.simulation
                            .celluniverse4_postfit_bright_cell_size_review_brightness_relative_tolerance -
                        0.45f) < 1.0e-7f &&
                activePos0.simulation
                    .celluniverse4_postfit_bright_cell_size_review_stop_at_neighbor_surface_enabled &&
                activePos0.simulation
                    .celluniverse4_postfit_bright_cell_size_review_anisotropic_shape_enabled &&
                std::abs(
                    activePos0.simulation
                            .celluniverse4_postfit_bright_cell_size_review_shape_extent_percentile -
                        0.95f) < 1.0e-7f &&
                activePos0.cell->pcaShapeFitRoundingRescueEnabled &&
                std::abs(
                    activePos0.cell
                            ->pcaShapeFitRoundingRescueMinElongationImprovement -
                        0.05f) < 1.0e-7f &&
                std::abs(
                    activePos0.cell
                            ->pcaShapeFitRoundingRescueMaxAxisRedistributionFraction -
                        0.15f) < 1.0e-7f,
            "active Pos0 CU4 profile must enable bounded rounding rescue");

        YAML::Node invalidReviewTolerance = YAML::Clone(activePos0Node);
        invalidReviewTolerance["simulation"]
                              ["celluniverse4_postfit_bright_cell_size_review_brightness_relative_tolerance"] =
            1.0f;
        threw = false;
        try {
            BaseConfig rejected;
            rejected.explodeConfig(invalidReviewTolerance);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        require(
            threw,
            "component brightness tolerance must remain below one");

        const YAML::Node c6NormalizedOnlyNode =
            CellUniverseConfig::loadConfigYamlNode(
                configDir +
                "/experiments/pavak-pos0-split-preprocess-20260801/cu4_normalized_only_c6.yaml");
        BaseConfig c6NormalizedOnly;
        c6NormalizedOnly.explodeConfig(c6NormalizedOnlyNode);
        require(
            c6NormalizedOnly.simulation.celluniverse4_enabled &&
                c6NormalizedOnly.candidateBatch.enabled &&
                c6NormalizedOnly.simulation.preprocess_mode == "none" &&
                !c6NormalizedOnly.simulation.n2v2_preprocess_enabled &&
                c6NormalizedOnly.simulation
                    .frame_intensity_normalization_enabled &&
                !c6NormalizedOnly.simulation
                    .frame_intensity_normalization_before_n2v2_enabled,
            "CU4 normalized-only A/B profile must disable only N2V2");

        YAML::Node retryDisabled = YAML::Clone(cu4Node);
        retryDisabled["prob"]
                     ["celluniverse4_split_finalist_retry_enabled"] =
            false;
        retryDisabled["prob"]
                     ["celluniverse4_split_finalist_retry_limit"] = 2;
        BaseConfig disabledRetry;
        disabledRetry.explodeConfig(retryDisabled);
        require(
            !disabledRetry.prob.celluniverse4_split_finalist_retry_enabled &&
                disabledRetry.prob.celluniverse4_split_finalist_retry_limit == 2,
            "CU4 finalist retry controls must parse explicit overrides");

        YAML::Node invalidBrightness = YAML::Clone(cu4Node);
        invalidBrightness["simulation"]
                         ["celluniverse4_postfit_brightness_volume_strength"] =
            1.01f;
        threw = false;
        try {
            BaseConfig rejected;
            rejected.explodeConfig(invalidBrightness);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        require(threw, "CU4 brightness-volume strength above one must be rejected");

        YAML::Node invalidShapeAge = YAML::Clone(cu4Node);
        invalidShapeAge["cell"]["pcaShapeRatioBoundMinAgeFrames"] = -1;
        threw = false;
        try {
            BaseConfig rejected;
            rejected.explodeConfig(invalidShapeAge);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        require(threw, "CU4 hard shape minimum age must be nonnegative");

        YAML::Node invalidRoundingRescue = YAML::Clone(cu4Node);
        invalidRoundingRescue["cell"]
                              ["pcaShapeFitRoundingRescueEnabled"] = true;
        invalidRoundingRescue["cell"]
                              ["pcaShapeFitRoundingRescueMaxAxisRedistributionFraction"] =
            0.0f;
        threw = false;
        try {
            BaseConfig rejected;
            rejected.explodeConfig(invalidRoundingRescue);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        require(
            threw,
            "enabled rounding rescue must require positive redistribution");

        YAML::Node invalidHardShapeRatio = YAML::Clone(cu4Node);
        invalidHardShapeRatio["cell"]["pcaShapeMaxLongMidRatio"] = 1.0f;
        threw = false;
        try {
            BaseConfig rejected;
            rejected.explodeConfig(invalidHardShapeRatio);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        require(threw, "enabled hard shape ratios must be greater than one");

        YAML::Node invalidMode = YAML::Clone(cu4Node);
        invalidMode["simulation"]["celluniverse2_enabled"] = true;
        threw = false;
        try {
            BaseConfig rejected;
            rejected.explodeConfig(invalidMode);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        require(threw, "CU4 and CU2 combination must be rejected");

        invalidMode = YAML::Clone(cu4Node);
        invalidMode["cell_lumen"]["enabled"] = true;
        threw = false;
        try {
            BaseConfig rejected;
            rejected.explodeConfig(invalidMode);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        require(threw, "CU4 and CellLumen combination must be rejected");

        YAML::Node invalidComponentConfig = YAML::Clone(cu4Node);
        invalidComponentConfig["candidate_batch"]
                              ["local_component_split_saturated_sparse_hybrid_max_valley_ratio"] =
            1.01f;
        threw = false;
        try {
            BaseConfig rejected;
            rejected.explodeConfig(invalidComponentConfig);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        require(threw, "hybrid valley override above one must be rejected");

        invalidComponentConfig = YAML::Clone(cu4Node);
        invalidComponentConfig["candidate_batch"]
                              ["local_component_split_saturated_sparse_hybrid_min_cost_improvement"] =
            -1.0f;
        threw = false;
        try {
            BaseConfig rejected;
            rejected.explodeConfig(invalidComponentConfig);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        require(threw, "negative hybrid cost improvement must be rejected");

        invalidComponentConfig = YAML::Clone(cu4Node);
        invalidComponentConfig["simulation"]
                              ["celluniverse4_component_connectivity_radius"] =
            0;
        threw = false;
        try {
            BaseConfig rejected;
            rejected.explodeConfig(invalidComponentConfig);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        require(threw, "invalid CU4 component connectivity must be rejected");

        invalidComponentConfig = YAML::Clone(cu4Node);
        invalidComponentConfig["simulation"]
                              ["celluniverse4_pca_component_anchor_radius_scale"] =
            0.5;
        threw = false;
        try {
            BaseConfig rejected;
            rejected.explodeConfig(invalidComponentConfig);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        require(threw, "invalid CU4 PCA anchor radius scale must be rejected");

        invalidComponentConfig = YAML::Clone(cu4Node);
        invalidComponentConfig["simulation"]
                              ["celluniverse4_pca_component_low_snr_rescue_min_support_fraction"] =
            -0.1;
        threw = false;
        try {
            BaseConfig rejected;
            rejected.explodeConfig(invalidComponentConfig);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        require(threw, "invalid CU4 component rescue support fraction must be rejected");

        invalidComponentConfig = YAML::Clone(cu4Node);
        invalidComponentConfig["simulation"]
                              ["celluniverse4_pca_iterative_regather_min_shift_radius_fraction"] =
            -0.1;
        threw = false;
        try {
            BaseConfig rejected;
            rejected.explodeConfig(invalidComponentConfig);
        } catch (const std::invalid_argument &) {
            threw = true;
        }
        require(threw, "negative PCA iterative-regather shift must be rejected");
#endif

        std::cout << "CellUniverse4 candidate batch tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "CellUniverse4 candidate batch test failed: "
                  << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
