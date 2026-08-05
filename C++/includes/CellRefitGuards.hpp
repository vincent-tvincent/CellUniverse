#ifndef CELL_REFIT_GUARDS_HPP
#define CELL_REFIT_GUARDS_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

namespace cell_refit_guards {

inline bool qualifiesPcaSeparationForSplitFirst(
    float separation,
    float shortestRadius,
    int keptPixels,
    float minimumSeparationShortRadiusFraction,
    int minimumKeptPixels,
    float &separationRatio)
{
    separationRatio = 0.0f;
    if (!std::isfinite(separation) || separation < 0.0f ||
        !std::isfinite(shortestRadius) || shortestRadius <= 1.0e-3f ||
        !std::isfinite(minimumSeparationShortRadiusFraction) ||
        minimumSeparationShortRadiusFraction < 0.0f ||
        keptPixels < 0 || minimumKeptPixels < 0) {
        return false;
    }
    separationRatio = separation / shortestRadius;
    return std::isfinite(separationRatio) &&
           keptPixels >= minimumKeptPixels &&
           separationRatio >= minimumSeparationShortRadiusFraction;
}

inline bool rejectSiblingRodOverlap(
    float rawAspectD1,
    float rawAspectD2,
    float exactOverlapFraction,
    float minimumRawAspect,
    float maximumOverlapFraction)
{
    if (!std::isfinite(rawAspectD1) || !std::isfinite(rawAspectD2) ||
        !std::isfinite(exactOverlapFraction) ||
        !std::isfinite(minimumRawAspect) ||
        !std::isfinite(maximumOverlapFraction) ||
        rawAspectD1 < 1.0f || rawAspectD2 < 1.0f ||
        exactOverlapFraction < 0.0f || minimumRawAspect < 1.0f ||
        maximumOverlapFraction < 0.0f || maximumOverlapFraction > 1.0f) {
        return false;
    }
    return std::min(rawAspectD1, rawAspectD2) >= minimumRawAspect &&
           exactOverlapFraction > maximumOverlapFraction;
}

enum class LocalComponentSplitDecision {
    FallbackNoEvidence,
    VetoTooFewComponents,
    VetoSeparation,
    VetoAxisAlignment,
    VetoMidpoint,
    VetoParentStraddle,
    Accept,
};

// Pure, testable geometry gate for a split pair found by local thresholded
// connected components. A zero-component ROI is intentionally not a veto:
// its threshold evidence is absent, so the legacy proposal path may still
// evaluate the parent. Once an ROI has one or more components, however, it is
// a direct current-frame answer for this parent and must not be replaced by a
// global/future center pair.
inline LocalComponentSplitDecision evaluateLocalComponentSplitPair(
    int totalComponents,
    int usableComponents,
    float separation,
    float minSeparation,
    float maxSeparation,
    float axisAlignment,
    float minAxisAlignment,
    float midpointDistance,
    float maxMidpointDistance,
    float signedSideA,
    float signedSideB)
{
    if (totalComponents <= 0 || usableComponents <= 0) {
        return LocalComponentSplitDecision::FallbackNoEvidence;
    }
    if (usableComponents < 2) return LocalComponentSplitDecision::VetoTooFewComponents;
    if (!std::isfinite(separation) || !std::isfinite(minSeparation) ||
        !std::isfinite(maxSeparation) || separation < minSeparation ||
        separation > maxSeparation) {
        return LocalComponentSplitDecision::VetoSeparation;
    }
    if (!std::isfinite(axisAlignment) || !std::isfinite(minAxisAlignment) ||
        axisAlignment < minAxisAlignment) {
        return LocalComponentSplitDecision::VetoAxisAlignment;
    }
    if (!std::isfinite(midpointDistance) || !std::isfinite(maxMidpointDistance) ||
        midpointDistance > maxMidpointDistance) {
        return LocalComponentSplitDecision::VetoMidpoint;
    }
    if (!std::isfinite(signedSideA) || !std::isfinite(signedSideB) ||
        !(signedSideA <= 0.0f && signedSideB >= 0.0f)) {
        return LocalComponentSplitDecision::VetoParentStraddle;
    }
    return LocalComponentSplitDecision::Accept;
}

inline bool isSaturatedSparseLocalEvidence(
    float threshold, float minSaturatedThreshold, int totalSupportVoxels,
    int usableComponents, int totalComponents, int maxTotalSupportVoxels)
{
    return std::isfinite(threshold) && std::isfinite(minSaturatedThreshold) &&
           threshold >= minSaturatedThreshold && usableComponents == 1 &&
           totalComponents > 1 && maxTotalSupportVoxels >= 0 &&
           totalSupportVoxels <= maxTotalSupportVoxels;
}

inline bool shouldFallbackForSaturatedSparseLocalEvidence(
    bool enabled, float threshold, float minSaturatedThreshold, int totalSupportVoxels,
    int usableComponents, int totalComponents, int maxTotalSupportVoxels)
{
    return enabled && isSaturatedSparseLocalEvidence(
        threshold, minSaturatedThreshold, totalSupportVoxels, usableComponents,
        totalComponents, maxTotalSupportVoxels);
}

enum class SaturatedSparseHybridDecision {
    FallbackNotEligible,
    VetoOutsideParentScope,
    VetoNeighborOwned,
    VetoSignalBoxes,
    VetoSignalBrightness,
    VetoSignalConfidence,
    VetoSeparation,
    VetoMidpoint,
    Accept,
};

inline const char *saturatedSparseHybridDecisionName(
    SaturatedSparseHybridDecision decision)
{
    switch (decision) {
    case SaturatedSparseHybridDecision::FallbackNotEligible:
        return "saturated_sparse_hybrid_not_eligible";
    case SaturatedSparseHybridDecision::VetoOutsideParentScope:
        return "saturated_sparse_hybrid_parent_scope_gate";
    case SaturatedSparseHybridDecision::VetoNeighborOwned:
        return "saturated_sparse_hybrid_neighbor_owned_gate";
    case SaturatedSparseHybridDecision::VetoSignalBoxes:
        return "saturated_sparse_hybrid_signal_boxes_gate";
    case SaturatedSparseHybridDecision::VetoSignalBrightness:
        return "saturated_sparse_hybrid_signal_brightness_gate";
    case SaturatedSparseHybridDecision::VetoSignalConfidence:
        return "saturated_sparse_hybrid_signal_confidence_gate";
    case SaturatedSparseHybridDecision::VetoSeparation:
        return "saturated_sparse_hybrid_separation_gate";
    case SaturatedSparseHybridDecision::VetoMidpoint:
        return "saturated_sparse_hybrid_midpoint_gate";
    case SaturatedSparseHybridDecision::Accept:
        return "ok";
    }
    return "saturated_sparse_hybrid_unknown";
}

// Pure compact-pair gate for the exceptional saturated-sparse case. The
// caller establishes the saturated predicate from the local component search;
// this function never changes the ordinary global-center pairing limits.
inline SaturatedSparseHybridDecision evaluateSaturatedSparseHybridPair(
    bool hybridEnabled,
    bool saturatedSparsePredicate,
    bool signalInsideParentScope,
    bool signalNeighborOwned,
    int signalBoxes,
    int minimumSignalBoxes,
    float signalBrightness,
    float minimumSignalBrightness,
    float signalConfidence,
    float minimumSignalConfidence,
    float separation,
    float minimumSeparation,
    float maximumSeparation,
    float midpointDistance,
    float maximumMidpointDistance)
{
    if (!hybridEnabled || !saturatedSparsePredicate) {
        return SaturatedSparseHybridDecision::FallbackNotEligible;
    }
    if (!signalInsideParentScope) {
        return SaturatedSparseHybridDecision::VetoOutsideParentScope;
    }
    if (signalNeighborOwned) {
        return SaturatedSparseHybridDecision::VetoNeighborOwned;
    }
    if (signalBoxes < minimumSignalBoxes) {
        return SaturatedSparseHybridDecision::VetoSignalBoxes;
    }
    if (!std::isfinite(signalBrightness) ||
        !std::isfinite(minimumSignalBrightness) ||
        signalBrightness < minimumSignalBrightness) {
        return SaturatedSparseHybridDecision::VetoSignalBrightness;
    }
    if (!std::isfinite(signalConfidence) ||
        !std::isfinite(minimumSignalConfidence) ||
        signalConfidence < minimumSignalConfidence) {
        return SaturatedSparseHybridDecision::VetoSignalConfidence;
    }
    if (!std::isfinite(separation) || !std::isfinite(minimumSeparation) ||
        !std::isfinite(maximumSeparation) || separation < minimumSeparation ||
        separation > maximumSeparation) {
        return SaturatedSparseHybridDecision::VetoSeparation;
    }
    if (!std::isfinite(midpointDistance) ||
        !std::isfinite(maximumMidpointDistance) ||
        midpointDistance > maximumMidpointDistance) {
        return SaturatedSparseHybridDecision::VetoMidpoint;
    }
    return SaturatedSparseHybridDecision::Accept;
}

struct SaturatedSparseHybridReflectedSeed {
    bool valid = false;
    std::array<float, 3> position{};
};

// The observed signal center is often the centroid of the still-merged body,
// rather than the opposite daughter. Reflect it across the surviving local
// component so it remains an evidence anchor while the generated seed samples
// the other side of the body.
inline SaturatedSparseHybridReflectedSeed makeSaturatedSparseHybridReflectedSeed(
    const std::array<float, 3> &localComponent,
    const std::array<float, 3> &observedSignalCenter,
    float reflectionScale)
{
    SaturatedSparseHybridReflectedSeed result;
    if (!std::isfinite(reflectionScale) || reflectionScale < 0.0f) {
        return result;
    }
    for (size_t axis = 0; axis < result.position.size(); ++axis) {
        if (!std::isfinite(localComponent[axis]) ||
            !std::isfinite(observedSignalCenter[axis])) {
            return result;
        }
        result.position[axis] = observedSignalCenter[axis] +
            reflectionScale * (observedSignalCenter[axis] - localComponent[axis]);
        if (!std::isfinite(result.position[axis])) return result;
    }
    result.valid = true;
    return result;
}

enum class SaturatedSparseHybridGeneratedSeedDecision {
    VetoInvalid,
    VetoOutsideParentScope,
    VetoNeighborOwned,
    Accept,
};

inline const char *saturatedSparseHybridGeneratedSeedDecisionName(
    SaturatedSparseHybridGeneratedSeedDecision decision)
{
    switch (decision) {
    case SaturatedSparseHybridGeneratedSeedDecision::VetoInvalid:
        return "saturated_sparse_hybrid_generated_seed_invalid";
    case SaturatedSparseHybridGeneratedSeedDecision::VetoOutsideParentScope:
        return "saturated_sparse_hybrid_generated_seed_parent_scope_gate";
    case SaturatedSparseHybridGeneratedSeedDecision::VetoNeighborOwned:
        return "saturated_sparse_hybrid_generated_seed_neighbor_owned_gate";
    case SaturatedSparseHybridGeneratedSeedDecision::Accept:
        return "ok";
    }
    return "saturated_sparse_hybrid_generated_seed_unknown";
}

inline SaturatedSparseHybridGeneratedSeedDecision
evaluateSaturatedSparseHybridGeneratedSeed(
    bool generatedSeedValid,
    bool generatedSeedInsideParentScope,
    bool generatedSeedNeighborOwned)
{
    if (!generatedSeedValid) {
        return SaturatedSparseHybridGeneratedSeedDecision::VetoInvalid;
    }
    if (!generatedSeedInsideParentScope) {
        return SaturatedSparseHybridGeneratedSeedDecision::VetoOutsideParentScope;
    }
    if (generatedSeedNeighborOwned) {
        return SaturatedSparseHybridGeneratedSeedDecision::VetoNeighborOwned;
    }
    return SaturatedSparseHybridGeneratedSeedDecision::Accept;
}

inline float saturatedSparseHybridValleyLimit(
    bool saturatedSparseHybridPair,
    bool overrideEnabled,
    float ordinaryLimit,
    float hybridLimit)
{
    return saturatedSparseHybridPair && overrideEnabled
        ? hybridLimit
        : ordinaryLimit;
}

inline bool passesSaturatedSparseHybridValleyPrefilter(
    float observedRatio,
    bool saturatedSparseHybridPair,
    bool overrideEnabled,
    float ordinaryLimit,
    float hybridLimit)
{
    const float limit = saturatedSparseHybridValleyLimit(
        saturatedSparseHybridPair,
        overrideEnabled,
        ordinaryLimit,
        hybridLimit);
    return saturatedSparseHybridPair && overrideEnabled
        ? observedRatio <= limit
        : observedRatio < limit;
}

struct BackgroundDropDecision
{
    bool valid = false;
    bool dropped = false;
    double observedContrast = 0.0;
    double modelContrast = 0.0;
    double observedModelContrastRatio = 0.0;
};

inline BackgroundDropDecision detectBackgroundDrop(
    double observedTopMean,
    double meanBackground,
    double modelBrightness,
    double minimumModelContrast,
    double maximumObservedModelContrastRatio)
{
    BackgroundDropDecision result;
    if (!std::isfinite(observedTopMean) || !std::isfinite(meanBackground) ||
        !std::isfinite(modelBrightness) ||
        !std::isfinite(minimumModelContrast) ||
        !std::isfinite(maximumObservedModelContrastRatio) ||
        minimumModelContrast < 0.0 || maximumObservedModelContrastRatio < 0.0) {
        return result;
    }
    result.observedContrast = std::max(0.0, observedTopMean - meanBackground);
    result.modelContrast = std::max(0.0, modelBrightness - meanBackground);
    result.observedModelContrastRatio = result.modelContrast > 0.0
        ? result.observedContrast / result.modelContrast
        : std::numeric_limits<double>::infinity();
    result.dropped =
        result.modelContrast >= minimumModelContrast &&
        result.observedModelContrastRatio <=
            maximumObservedModelContrastRatio;
    result.valid = true;
    return result;
}

inline bool acceptBackgroundDropProbe(
    bool nearestViableCenterPolicy,
    double probeMean,
    double minimumProbeMean,
    double probeScore,
    double maximumProbeScore)
{
    if (!std::isfinite(probeMean) || !std::isfinite(minimumProbeMean)) {
        return false;
    }
    if (probeMean < minimumProbeMean) return false;
    if (nearestViableCenterPolicy) return true;
    return std::isfinite(probeScore) && std::isfinite(maximumProbeScore) &&
           probeScore <= maximumProbeScore;
}

inline double finiteRankingValue(double value)
{
    return std::isfinite(value)
        ? value
        : std::numeric_limits<double>::infinity();
}

inline bool preferNearestCenterClaim(
    double lhsDistance,
    std::size_t lhsCellIndex,
    double rhsDistance,
    std::size_t rhsCellIndex)
{
    const double lhs = finiteRankingValue(lhsDistance);
    const double rhs = finiteRankingValue(rhsDistance);
    if (lhs != rhs) return lhs < rhs;
    return lhsCellIndex < rhsCellIndex;
}

inline float pcaLowSnrCutoff(
    bool componentFilterEnabled,
    bool decoupleVolumeCap,
    float physicalCutoff,
    float finalSupportCutoff)
{
    if (!std::isfinite(physicalCutoff) ||
        !std::isfinite(finalSupportCutoff)) {
        return std::numeric_limits<float>::infinity();
    }
    return componentFilterEnabled && decoupleVolumeCap
        ? physicalCutoff
        : finalSupportCutoff;
}

inline bool rescueLowSnrWithAnchoredComponent(
    bool enabled,
    bool lowSnr,
    size_t retainedSupport,
    size_t minimumSupport,
    size_t innerOwned,
    size_t selectedComponentCount,
    float anchorDistance,
    float minimumSupportFraction,
    float maximumAnchorDistance)
{
    if (!enabled || !lowSnr || retainedSupport < minimumSupport ||
        innerOwned == 0 || selectedComponentCount == 0 ||
        !std::isfinite(anchorDistance) || anchorDistance < 0.0f ||
        !std::isfinite(minimumSupportFraction) ||
        minimumSupportFraction < 0.0f ||
        !std::isfinite(maximumAnchorDistance) ||
        maximumAnchorDistance < 0.0f) {
        return false;
    }
    const float supportFraction =
        static_cast<float>(retainedSupport) /
        static_cast<float>(innerOwned);
    const bool anchorCloseEnough = maximumAnchorDistance == 0.0f ||
        anchorDistance <= maximumAnchorDistance;
    return std::isfinite(supportFraction) &&
           supportFraction >= minimumSupportFraction &&
           anchorCloseEnough;
}

struct AspectRatioProjection
{
    std::array<float, 3> radii{};
    bool valid = false;
    bool changed = false;
    float beforeLongMid = 1.0f;
    float beforeMidShort = 1.0f;
    float beforeLongShort = 1.0f;
    float afterLongMid = 1.0f;
    float afterMidShort = 1.0f;
    float afterLongShort = 1.0f;
};

inline AspectRatioProjection projectAspectRatiosShrinkOnly(
    const std::array<float, 3> &input,
    float maxLongMid,
    float maxMidShort,
    float maxLongShort)
{
    AspectRatioProjection result;
    result.radii = input;
    for (const float radius : input) {
        if (!std::isfinite(radius) || radius <= 0.0f) return result;
    }

    const auto sortedIndices = [](const std::array<float, 3> &radii) {
        std::array<int, 3> indices{0, 1, 2};
        std::sort(indices.begin(), indices.end(), [&](int lhs, int rhs) {
            if (radii[lhs] != radii[rhs]) return radii[lhs] > radii[rhs];
            return lhs < rhs;
        });
        return indices;
    };
    const auto ratio = [](float high, float low) {
        return low > 1.0e-6f ? high / low : 1.0f;
    };

    const auto indices = sortedIndices(input);
    const float longest = input[indices[0]];
    const float middle = input[indices[1]];
    const float shortest = input[indices[2]];
    result.beforeLongMid = ratio(longest, middle);
    result.beforeMidShort = ratio(middle, shortest);
    result.beforeLongShort = ratio(longest, shortest);

    const bool limitLongMid = std::isfinite(maxLongMid) && maxLongMid > 1.0f;
    const bool limitMidShort =
        std::isfinite(maxMidShort) && maxMidShort > 1.0f;
    const bool limitLongShort =
        std::isfinite(maxLongShort) && maxLongShort > 1.0f;
    if (!limitLongMid && !limitMidShort && !limitLongShort) {
        result.valid = true;
        return result;
    }

    float projectedMiddle = middle;
    if (limitMidShort) {
        projectedMiddle = std::min(projectedMiddle, maxMidShort * shortest);
    }
    // If the longest axis is capped against the shortest, the middle axis
    // cannot remain larger than that same cap without changing axis order.
    if (limitLongShort) {
        projectedMiddle = std::min(projectedMiddle, maxLongShort * shortest);
    }

    float projectedLongest = longest;
    if (limitLongMid) {
        projectedLongest =
            std::min(projectedLongest, maxLongMid * projectedMiddle);
    }
    if (limitLongShort) {
        projectedLongest =
            std::min(projectedLongest, maxLongShort * shortest);
    }

    result.radii[indices[0]] = projectedLongest;
    result.radii[indices[1]] = projectedMiddle;
    result.radii[indices[2]] = shortest;
    result.afterLongMid = ratio(projectedLongest, projectedMiddle);
    result.afterMidShort = ratio(projectedMiddle, shortest);
    result.afterLongShort = ratio(projectedLongest, shortest);
    result.changed =
        std::abs(result.radii[0] - input[0]) > 1.0e-5f ||
        std::abs(result.radii[1] - input[1]) > 1.0e-5f ||
        std::abs(result.radii[2] - input[2]) > 1.0e-5f;
    result.valid = true;
    return result;
}

struct VolumePreservingRoundingRescue
{
    std::array<float, 3> radii{};
    bool valid = false;
    bool applied = false;
    float uncappedElongation = 1.0f;
    float cappedElongation = 1.0f;
    float rescuedElongation = 1.0f;
    float blend = 0.0f;
    double cappedRadiusProduct = 0.0;
    double rescuedRadiusProduct = 0.0;
};

// For uniformly occupied ellipsoid support, a coordinate percentile measures
// an interior chord, not the surface radius. For one normalized axis u in
// [-1, 1], P(|u| <= q) = 1.5 q - 0.5 q^3. Return the factor that converts the
// selected absolute-coordinate percentile back to the corresponding surface
// radius. This prevents a 95th-percentile size estimate from being biased low
// by roughly 19%, while retaining percentile robustness against isolated
// boundary noise.
inline float ellipsoidAxisPercentileSurfaceCorrection(float percentile)
{
    if (!std::isfinite(percentile) || percentile <= 0.0f ||
        percentile > 1.0f) {
        return 1.0f;
    }
    if (percentile >= 1.0f - 1.0e-6f) return 1.0f;

    double lower = 0.0;
    double upper = 1.0;
    for (int iteration = 0; iteration < 40; ++iteration) {
        const double q = 0.5 * (lower + upper);
        const double cdf = 1.5 * q - 0.5 * q * q * q;
        if (cdf < static_cast<double>(percentile)) {
            lower = q;
        } else {
            upper = q;
        }
    }
    const double coordinateQuantile = 0.5 * (lower + upper);
    if (!std::isfinite(coordinateQuantile) ||
        coordinateQuantile <= 1.0e-6) {
        return 1.0f;
    }
    return static_cast<float>(1.0 / coordinateQuantile);
}

// A temporal per-axis growth cap can accidentally reject a well-supported
// rounder PCA fit: the short axis is prevented from growing while the long
// axis remains large. Preserve the ordinary cap's radius product (and thus
// ellipsoid volume), but redistribute a bounded amount of that product toward
// the proportions of the uncapped PCA fit. Log-space interpolation preserves
// the product exactly while maxAxisRedistributionFraction limits the change
// applied to any one capped radius.
inline VolumePreservingRoundingRescue rescueRounderShapeAtFixedVolume(
    const std::array<float, 3> &uncapped,
    const std::array<float, 3> &capped,
    float minimumElongationImprovement,
    float maxAxisRedistributionFraction)
{
    VolumePreservingRoundingRescue result;
    result.radii = capped;
    if (!std::isfinite(minimumElongationImprovement) ||
        minimumElongationImprovement < 0.0f ||
        !std::isfinite(maxAxisRedistributionFraction) ||
        maxAxisRedistributionFraction <= 0.0f) {
        return result;
    }
    for (size_t axis = 0; axis < 3; ++axis) {
        if (!std::isfinite(uncapped[axis]) || uncapped[axis] <= 0.0f ||
            !std::isfinite(capped[axis]) || capped[axis] <= 0.0f) {
            return result;
        }
    }

    const auto elongation = [](const std::array<float, 3> &radii) {
        const auto limits = std::minmax_element(radii.begin(), radii.end());
        return *limits.second / *limits.first;
    };
    result.uncappedElongation = elongation(uncapped);
    result.cappedElongation = elongation(capped);
    result.rescuedElongation = result.cappedElongation;
    result.cappedRadiusProduct =
        static_cast<double>(capped[0]) * capped[1] * capped[2];
    const double uncappedProduct =
        static_cast<double>(uncapped[0]) * uncapped[1] * uncapped[2];
    result.rescuedRadiusProduct = result.cappedRadiusProduct;
    result.valid = std::isfinite(result.cappedRadiusProduct) &&
        result.cappedRadiusProduct > 0.0 &&
        std::isfinite(uncappedProduct) && uncappedProduct > 0.0;
    if (!result.valid ||
        result.cappedElongation - result.uncappedElongation <
            minimumElongationImprovement) {
        return result;
    }

    const double targetScale = std::cbrt(
        result.cappedRadiusProduct / uncappedProduct);
    std::array<double, 3> target{};
    double maximumLogChange = 0.0;
    for (size_t axis = 0; axis < 3; ++axis) {
        target[axis] = static_cast<double>(uncapped[axis]) * targetScale;
        if (!std::isfinite(target[axis]) || target[axis] <= 0.0) {
            result.valid = false;
            return result;
        }
        maximumLogChange = std::max(
            maximumLogChange,
            std::abs(std::log(target[axis] / capped[axis])));
    }
    if (maximumLogChange <= 1.0e-12) return result;

    const double maximumAllowedLogChange =
        std::log1p(static_cast<double>(maxAxisRedistributionFraction));
    result.blend = static_cast<float>(std::min(
        1.0, maximumAllowedLogChange / maximumLogChange));
    for (size_t axis = 0; axis < 3; ++axis) {
        const double rescued = std::exp(
            std::log(static_cast<double>(capped[axis])) +
            result.blend *
                (std::log(target[axis]) -
                 std::log(static_cast<double>(capped[axis]))));
        result.radii[axis] = static_cast<float>(rescued);
    }
    result.rescuedElongation = elongation(result.radii);
    result.rescuedRadiusProduct =
        static_cast<double>(result.radii[0]) *
        result.radii[1] * result.radii[2];
    result.applied =
        result.cappedElongation - result.rescuedElongation >=
            minimumElongationImprovement &&
        std::isfinite(result.rescuedRadiusProduct);
    if (!result.applied) {
        result.radii = capped;
        result.rescuedElongation = result.cappedElongation;
        result.rescuedRadiusProduct = result.cappedRadiusProduct;
    }
    return result;
}

struct BrightnessVolumeReconciliation
{
    bool valid = false;
    bool applied = false;
    bool usedZeroContrastFallback = false;
    double anchorIntegratedContrast = 0.0;
    double proposedIntegratedContrast = 0.0;
    double expectedIntegratedContrast = 0.0;
    double reconciledIntegratedContrast = 0.0;
    double logErrorBefore = 0.0;
    double logErrorAfter = 0.0;
    double contrastScale = 1.0;
};

inline BrightnessVolumeReconciliation reconcileIntegratedContrast(
    std::size_t anchorSupport,
    double anchorContrast,
    std::size_t currentSupport,
    double proposedIntegratedContrast,
    double inverseVolumeExponent,
    double logDeadband,
    double strength,
    double maximumContrastStepFraction)
{
    BrightnessVolumeReconciliation result;
    if (anchorSupport == 0 || currentSupport == 0 ||
        !std::isfinite(anchorContrast) || anchorContrast <= 0.0 ||
        !std::isfinite(proposedIntegratedContrast) ||
        proposedIntegratedContrast < 0.0 ||
        !std::isfinite(inverseVolumeExponent) || inverseVolumeExponent < 0.0 ||
        !std::isfinite(logDeadband) || logDeadband < 0.0 ||
        !std::isfinite(strength) || strength < 0.0 || strength > 1.0 ||
        !std::isfinite(maximumContrastStepFraction) ||
        maximumContrastStepFraction < 0.0) {
        return result;
    }

    result.anchorIntegratedContrast =
        static_cast<double>(anchorSupport) * anchorContrast;
    const double supportRatio =
        static_cast<double>(anchorSupport) /
        static_cast<double>(currentSupport);
    const double expectedContrast =
        anchorContrast * std::pow(supportRatio, inverseVolumeExponent);
    result.expectedIntegratedContrast =
        static_cast<double>(currentSupport) * expectedContrast;
    result.proposedIntegratedContrast = proposedIntegratedContrast;
    if (!std::isfinite(result.expectedIntegratedContrast) ||
        result.expectedIntegratedContrast <= 0.0) {
        return result;
    }

    // A cell that has already been driven exactly to the local background has
    // no positive contrast to multiply. Give it a bounded fraction of the
    // inverse-volume target instead of silently disabling the coupling. The
    // normal multiplicative path below remains unchanged for positive signal.
    if (proposedIntegratedContrast == 0.0) {
        double recoveryFraction = strength;
        if (maximumContrastStepFraction > 0.0) {
            recoveryFraction = std::min(
                recoveryFraction, maximumContrastStepFraction);
        }
        recoveryFraction = std::clamp(recoveryFraction, 0.0, 1.0);
        result.reconciledIntegratedContrast =
            result.expectedIntegratedContrast * recoveryFraction;
        result.logErrorBefore =
            -std::numeric_limits<double>::infinity();
        result.logErrorAfter = recoveryFraction > 0.0
            ? std::log(recoveryFraction)
            : -std::numeric_limits<double>::infinity();
        result.usedZeroContrastFallback = true;
        result.applied = result.reconciledIntegratedContrast > 0.0;
        result.valid = true;
        return result;
    }

    result.logErrorBefore = std::log(
        proposedIntegratedContrast / result.expectedIntegratedContrast);
    if (!std::isfinite(result.logErrorBefore)) return result;
    const double outsideDeadband =
        std::copysign(
            std::max(0.0, std::abs(result.logErrorBefore) - logDeadband),
            result.logErrorBefore);
    double scale = std::exp(-strength * outsideDeadband);
    if (maximumContrastStepFraction > 0.0) {
        scale = std::clamp(
            scale,
            std::max(0.0, 1.0 - maximumContrastStepFraction),
            1.0 + maximumContrastStepFraction);
    }
    if (!std::isfinite(scale) || scale <= 0.0) return result;

    result.contrastScale = scale;
    result.reconciledIntegratedContrast =
        proposedIntegratedContrast * scale;
    result.logErrorAfter = std::log(
        result.reconciledIntegratedContrast /
        result.expectedIntegratedContrast);
    result.applied = std::abs(scale - 1.0) > 1.0e-9;
    result.valid = true;
    return result;
}

} // namespace cell_refit_guards

#endif
