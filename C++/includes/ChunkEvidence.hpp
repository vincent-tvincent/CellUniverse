#ifndef CHUNKEVIDENCE_HPP
#define CHUNKEVIDENCE_HPP

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include <opencv2/core.hpp>

// The local-component path can provide an ordinary component centroid and,
// when its thresholded boundary is well resolved, a separate
// boundary-derived ellipsoid-center hypothesis. Keep the family explicit so
// CandidateBatch can retain a small deterministic amount of diversity without
// relaxing ownership or geometry gates.
enum class CandidateEvidenceFamily
{
    Component,
    EdgeRefined
};

inline const char *candidateEvidenceFamilyName(CandidateEvidenceFamily family)
{
    switch (family) {
    case CandidateEvidenceFamily::Component: return "component";
    case CandidateEvidenceFamily::EdgeRefined: return "edge_refined";
    }
    return "unknown";
}

// Immutable, compact evidence published by the prepared-frame component
// extractor. It deliberately preserves alternative center definitions rather
// than collapsing them before the optimizer can compare hypotheses.
struct ChunkEvidence
{
    std::uint64_t stableId = 0;
    // Zero is evidence localized in the frame being optimized. Positive
    // values are explicit offline lookahead; they are penalized during
    // ranking so a future center cannot silently outrank equal current-frame
    // evidence.
    int sourceFrameOffset = 0;
    cv::Point3f weightedCenter{0.0f, 0.0f, 0.0f};
    cv::Point3f geometricCenter{0.0f, 0.0f, 0.0f};
    cv::Point3f robustCenter{0.0f, 0.0f, 0.0f};
    cv::Point3f peakCenter{0.0f, 0.0f, 0.0f};
    cv::Point3f bboxMin{0.0f, 0.0f, 0.0f};
    cv::Point3f bboxMax{0.0f, 0.0f, 0.0f};
    float brightness = 0.0f;
    float confidence = 0.0f;
    float voxelCount = 0.0f;
    float elongation = 1.0f;
    int boxCount = 0;
    CandidateEvidenceFamily family = CandidateEvidenceFamily::Component;
    // Normalized local-component threshold when available. Global signal
    // centers have no comparable local threshold and retain NaN.
    float threshold = std::numeric_limits<float>::quiet_NaN();
    cv::Point3f edgeRefinedCenter{0.0f, 0.0f, 0.0f};
    bool hasEdgeRefinedCenter = false;
    // Only evidence already known to be local, unclaimed, non-sparse,
    // non-saturated, and non-overlapping may update threshold history.
    bool trustworthyForAdaptiveHistory = false;
    bool neighborOwned = false;
    bool sparse = false;
    bool saturated = false;
    bool ambiguous = false;
    bool overlapsNeighbor = false;
    // A parent-local connected component may request that an exact-center
    // continuation candidate be evaluated with a PCA shape refit. Global
    // chunk evidence leaves this false.
    bool pcaRefitSuggested = false;
    // A uniquely verified recovery candidate may be guaranteed one ordinary
    // cost evaluation even when its distance priority would otherwise fall
    // outside the bounded top-K batch. It never bypasses any acceptance gate.
    bool forceExpensiveTrial = false;
};

std::string chunkEvidenceStableIdString(std::uint64_t stableId);

#endif
