#include "../includes/CandidateBatch.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

std::uint64_t fnv1a(std::uint64_t hash, const std::string &value)
{
    constexpr std::uint64_t kPrime = 1099511628211ULL;
    for (const unsigned char c : value) {
        hash ^= static_cast<std::uint64_t>(c);
        hash *= kPrime;
    }
    return hash;
}

std::uint64_t fnv1a(std::uint64_t hash, std::uint64_t value)
{
    constexpr std::uint64_t kPrime = 1099511628211ULL;
    for (unsigned int byte = 0; byte < 8; ++byte) {
        hash ^= value & 0xffU;
        hash *= kPrime;
        value >>= 8U;
    }
    return hash;
}

std::uint64_t mix64(std::uint64_t value)
{
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

float pointDistance(const cv::Point3f &a, const cv::Point3f &b)
{
    return static_cast<float>(cv::norm(a - b));
}

double sourceBias(CandidateSource source, const CandidateBatchConfig &config)
{
    switch (source) {
    case CandidateSource::Snapshot:
        return config.prioritySnapshotBias;
    case CandidateSource::ChunkWeighted:
    case CandidateSource::ChunkGeometric:
    case CandidateSource::ChunkRobust:
    case CandidateSource::ChunkPeak:
    case CandidateSource::ChunkEdgeRefined:
        return config.priorityExactCenterBias;
    case CandidateSource::Stochastic:
        return config.priorityStochasticBias;
    case CandidateSource::NoOp:
        return -std::numeric_limits<double>::infinity();
    }
    return 0.0;
}

} // namespace

const char *candidateSourceName(CandidateSource source)
{
    switch (source) {
    case CandidateSource::NoOp: return "no_op";
    case CandidateSource::Snapshot: return "snapshot";
    case CandidateSource::ChunkWeighted: return "chunk_weighted";
    case CandidateSource::ChunkGeometric: return "chunk_geometric";
    case CandidateSource::ChunkRobust: return "chunk_robust";
    case CandidateSource::ChunkPeak: return "chunk_peak";
    case CandidateSource::ChunkEdgeRefined: return "chunk_edge_refined";
    case CandidateSource::Stochastic: return "stochastic";
    }
    return "unknown";
}

double CandidateBatch::deterministicUnit(std::uint64_t seed,
                                         const std::string &seedNamespace,
                                         int frame,
                                         const std::string &parentName,
                                         const std::string &operation,
                                         std::uint64_t ordinal)
{
    std::uint64_t hash = 1469598103934665603ULL ^ seed;
    hash = fnv1a(hash, seedNamespace);
    hash = fnv1a(hash, parentName);
    hash = fnv1a(hash, operation);
    hash = mix64(hash ^ static_cast<std::uint64_t>(static_cast<std::int64_t>(frame)));
    hash = mix64(hash ^ ordinal);
    return static_cast<double>(hash >> 11U) * (1.0 / 9007199254740992.0);
}

CandidateBatch::CandidateBatch(const CandidateBatchInput &input,
                               const CandidateBatchConfig &config)
    : config_(config)
{
    config_.validate();
    const float minRadius = std::max(1.0e-3f, input.parentMinRadius);
    const float dedupDistance = std::max(
        config_.deduplicationDistanceAbsolute,
        minRadius * config_.deduplicationDistanceMinRadiusFraction);
    const float associationDistance = std::max(
        config_.chunkAssociationDistanceAbsolute,
        minRadius * config_.chunkAssociationDistanceMinRadiusFraction);

    auto add = [&](CandidateSource source,
                   const cv::Point3f &position,
                   std::uint64_t evidenceId,
                   int evidenceFrameOffset,
                   float confidence,
                   bool pcaRefitSuggested,
                   std::uint64_t ordinal,
                   bool forceExpensiveTrial = false,
                   CandidateEvidenceFamily evidenceFamily =
                       CandidateEvidenceFamily::Component,
                   float evidenceThreshold =
                       std::numeric_limits<float>::quiet_NaN(),
                   bool trustworthyForAdaptiveHistory = false) {
        for (CandidateProposal &existing : proposals_) {
            if (pointDistance(existing.position, position) <= dedupDistance) {
                // Preserve the stronger provenance when a qualified recovery
                // center coincides with a generic evidence hypothesis.
                existing.forceExpensiveTrial =
                    existing.forceExpensiveTrial || forceExpensiveTrial;
                existing.trustworthyForAdaptiveHistory =
                    existing.trustworthyForAdaptiveHistory ||
                    trustworthyForAdaptiveHistory;
                return false;
            }
        }
        CandidateProposal proposal;
        std::uint64_t stableId = 1469598103934665603ULL;
        stableId = fnv1a(stableId, config_.deterministicSeedNamespace);
        stableId = fnv1a(stableId, input.parentName);
        stableId = fnv1a(
            stableId,
            static_cast<std::uint64_t>(static_cast<std::int64_t>(input.frame)));
        stableId = fnv1a(stableId, static_cast<std::uint64_t>(source));
        stableId = fnv1a(stableId, evidenceId);
        stableId = fnv1a(
            stableId,
            static_cast<std::uint64_t>(
                static_cast<std::int64_t>(evidenceFrameOffset)));
        stableId = fnv1a(stableId, ordinal);
        proposal.stableId = mix64(stableId);
        proposal.source = source;
        proposal.position = position;
        proposal.evidenceId = evidenceId;
        proposal.evidenceFrameOffset = evidenceFrameOffset;
        proposal.evidenceConfidence = confidence;
        proposal.evidenceFamily = evidenceFamily;
        proposal.evidenceThreshold = evidenceThreshold;
        proposal.trustworthyForAdaptiveHistory =
            trustworthyForAdaptiveHistory;
        proposal.pcaRefitSuggested = pcaRefitSuggested;
        proposal.forceExpensiveTrial = forceExpensiveTrial;
        proposal.cheapPriority =
            static_cast<double>(config_.priorityDistanceWeight) *
                pointDistance(input.baselinePosition, position) / minRadius -
            static_cast<double>(config_.priorityConfidenceWeight) * confidence +
            static_cast<double>(config_.lookaheadPriorityPenaltyPerFrame) *
                std::max(0, evidenceFrameOffset) +
            sourceBias(source, config_);
        proposals_.push_back(proposal);
        return true;
    };

    CandidateProposal noOp;
    noOp.stableId = 0;
    noOp.source = CandidateSource::NoOp;
    noOp.position = input.baselinePosition;
    noOp.cheapPriority = -std::numeric_limits<double>::infinity();
    noOp.evaluated = true;
    noOp.costDiff = 0.0;
    proposals_.push_back(noOp);

    if (config_.maxSnapshotAnchors > 0 && input.snapshotPosition.has_value()) {
        add(CandidateSource::Snapshot, *input.snapshotPosition,
            0, 0, 1.0f, false, 1);
    }

    int exactAdded = 0;
    if (config_.chunkEvidenceEnabled) {
        std::vector<ChunkEvidence> evidence = input.evidence;
        std::stable_sort(
            evidence.begin(), evidence.end(),
            [&](const ChunkEvidence &a, const ChunkEvidence &b) {
                const auto evidencePriority = [&](const ChunkEvidence &chunk) {
                    return
                        static_cast<double>(config_.priorityDistanceWeight) *
                            pointDistance(input.baselinePosition,
                                          chunk.weightedCenter) /
                            minRadius -
                        static_cast<double>(config_.priorityConfidenceWeight) *
                            chunk.confidence +
                        static_cast<double>(
                            config_.lookaheadPriorityPenaltyPerFrame) *
                            std::max(0, chunk.sourceFrameOffset);
                };
                const double aPriority = evidencePriority(a);
                const double bPriority = evidencePriority(b);
                if (aPriority != bPriority) {
                    return aPriority < bPriority;
                }
                if (a.sourceFrameOffset != b.sourceFrameOffset) {
                    return a.sourceFrameOffset < b.sourceFrameOffset;
                }
                return a.stableId < b.stableId;
            });
        for (const ChunkEvidence &chunk : evidence) {
            if (exactAdded >= config_.maxExactChunkCenters) break;
            if (chunk.boxCount < config_.chunkMinBoxes ||
                chunk.voxelCount < config_.chunkMinVoxels ||
                chunk.brightness < config_.chunkMinBrightness ||
                chunk.confidence < config_.chunkMinConfidence ||
                pointDistance(chunk.weightedCenter,
                              input.baselinePosition) > associationDistance) {
                continue;
            }
            const auto addExact = [&](CandidateSource source,
                                      const cv::Point3f &position,
                                      std::uint64_t ordinal,
                                      CandidateEvidenceFamily family =
                                          CandidateEvidenceFamily::Component) {
                if (exactAdded >= config_.maxExactChunkCenters) return;
                if (add(source, position, chunk.stableId,
                        chunk.sourceFrameOffset,
                        chunk.confidence, chunk.pcaRefitSuggested, ordinal,
                        chunk.forceExpensiveTrial &&
                            source == CandidateSource::ChunkWeighted,
                        family, chunk.threshold,
                        chunk.trustworthyForAdaptiveHistory)) {
                    ++exactAdded;
                }
            };
            if (config_.includeWeightedCenter) {
                addExact(CandidateSource::ChunkWeighted,
                         chunk.weightedCenter, 10);
            }
            if (config_.includeGeometricCenter) {
                addExact(CandidateSource::ChunkGeometric,
                         chunk.geometricCenter, 11);
            }
            if (config_.includeRobustCenter) {
                addExact(CandidateSource::ChunkRobust,
                         chunk.robustCenter, 12);
            }
            if (config_.includePeakCenter) {
                addExact(CandidateSource::ChunkPeak,
                         chunk.peakCenter, 13);
            }
            // This center is derived from a resolved local-component
            // boundary, not from a global map.  It is deliberately opt-in,
            // remains subject to the ordinary objective, and only receives
            // PCA when the source component already declared that reliable.
            if (config_.adaptiveEdgeRefinedHypothesesEnabled &&
                chunk.hasEdgeRefinedCenter) {
                addExact(CandidateSource::ChunkEdgeRefined,
                         chunk.edgeRefinedCenter, 14,
                         CandidateEvidenceFamily::EdgeRefined);
            }
        }
    }

    const float sigmaUpper = config_.stochasticSigmaAbsoluteMax;
    const float sigmaLower = std::min(
        sigmaUpper,
        std::max(config_.stochasticSigmaAbsoluteMin,
                 minRadius * config_.stochasticSigmaMinRadiusFraction));
    const float sigma = std::clamp(
        minRadius * config_.stochasticSigmaMaxRadiusFraction,
        sigmaLower, sigmaUpper);
    for (int i = 0; i < config_.stochasticCandidatesPerParent; ++i) {
        const double u1 = std::max(
            1.0e-12,
            deterministicUnit(config_.deterministicSeed,
                              config_.deterministicSeedNamespace,
                              input.frame, input.parentName,
                              "stochastic_u1", static_cast<std::uint64_t>(i)));
        const double u2 = deterministicUnit(
            config_.deterministicSeed, config_.deterministicSeedNamespace,
            input.frame, input.parentName,
            "stochastic_u2", static_cast<std::uint64_t>(i));
        const double u3 = deterministicUnit(
            config_.deterministicSeed, config_.deterministicSeedNamespace,
            input.frame, input.parentName,
            "stochastic_u3", static_cast<std::uint64_t>(i));
        const double radius = sigma * std::cbrt(u1);
        const double z = 2.0 * u2 - 1.0;
        const double phi = 6.28318530717958647692 * u3;
        const double xy = std::sqrt(std::max(0.0, 1.0 - z * z));
        const cv::Point3f offset(
            static_cast<float>(radius * xy * std::cos(phi)),
            static_cast<float>(radius * xy * std::sin(phi)),
            static_cast<float>(radius * z));
        add(CandidateSource::Stochastic,
            input.baselinePosition + offset, 0, 0, 0.0f, false,
            static_cast<std::uint64_t>(1000 + i));
    }

    if (proposals_.size() > 1) {
        std::stable_sort(proposals_.begin() + 1, proposals_.end(),
                         [](const CandidateProposal &a,
                            const CandidateProposal &b) {
                             if (a.cheapPriority != b.cheapPriority) {
                                 return a.cheapPriority < b.cheapPriority;
                             }
                             return a.stableId < b.stableId;
                         });
    }
    if (proposals_.size() >
        static_cast<std::size_t>(config_.maxCandidatesPerParent)) {
        proposals_.resize(static_cast<std::size_t>(
            config_.maxCandidatesPerParent));
    }
}

std::vector<std::size_t> CandidateBatch::expensiveCandidateIndices() const
{
    std::vector<std::size_t> result;
    if (proposals_.empty()) return result;
    result.push_back(0);
    if (config_.adaptiveTrialFitEnabled &&
        config_.adaptiveFamilyDiverseBeamEnabled) {
        const std::size_t beamWidth = static_cast<std::size_t>(std::max(
            1, config_.adaptiveFamilyDiverseBeamWidth));
        // Coarse deterministic pass: preserve one best candidate from each
        // image-evidence family before filling the remaining beam by the
        // existing priority ordering. No-op remains the unchanged baseline.
        for (const CandidateEvidenceFamily family : {
                 CandidateEvidenceFamily::Component,
                 CandidateEvidenceFamily::EdgeRefined}) {
            if (result.size() > beamWidth) break;
            for (std::size_t index = 1; index < proposals_.size(); ++index) {
                if (proposals_[index].evidenceId == 0 ||
                    proposals_[index].evidenceFamily != family) {
                    continue;
                }
                result.push_back(index);
                break;
            }
        }
        for (std::size_t index = 1;
             index < proposals_.size() && result.size() <= beamWidth;
             ++index) {
            if (std::find(result.begin(), result.end(), index) ==
                result.end()) {
                result.push_back(index);
            }
        }
    } else {
        const std::size_t limit = std::min(
            proposals_.size(),
            static_cast<std::size_t>(std::max(1, config_.expensiveTopK)));
        for (std::size_t index = 1; index < limit; ++index) {
            result.push_back(index);
        }
    }
    // The bounded rescue path may mark at most one evidence hypothesis. Give
    // it one ordinary trial even if broad generic candidates occupy top-K.
    for (std::size_t index = 1; index < proposals_.size(); ++index) {
        if (proposals_[index].forceExpensiveTrial &&
            std::find(result.begin(), result.end(), index) == result.end()) {
            result.push_back(index);
        }
    }
    return result;
}

void CandidateBatch::recordEvaluation(std::size_t index, double costDiff)
{
    if (index >= proposals_.size()) {
        throw std::out_of_range("candidate batch evaluation index is out of range");
    }
    if (!std::isfinite(costDiff)) {
        throw std::invalid_argument("candidate batch cost must be finite");
    }
    proposals_[index].evaluated = true;
    proposals_[index].costDiff = costDiff;
}

std::optional<std::size_t>
CandidateBatch::selectWinner(double baselineObjective) const
{
    if (proposals_.empty()) return std::nullopt;
    std::size_t bestIndex = 0;
    double bestCost = 0.0;
    for (std::size_t index = 1; index < proposals_.size(); ++index) {
        const CandidateProposal &candidate = proposals_[index];
        if (!candidate.evaluated) continue;
        if (candidate.costDiff < bestCost) {
            bestCost = candidate.costDiff;
            bestIndex = index;
        }
    }
    const double requiredImprovement = std::max(
        config_.absoluteImprovementMargin,
        config_.fractionalImprovementMargin *
            std::max(0.0, std::abs(baselineObjective)));
    if (bestIndex == 0 || -bestCost <= requiredImprovement) {
        return std::nullopt;
    }
    return bestIndex;
}
