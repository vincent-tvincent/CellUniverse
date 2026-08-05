#ifndef CHUNKEVIDENCE_HPP
#define CHUNKEVIDENCE_HPP

#include <cstddef>
#include <cstdint>
#include <string>

#include <opencv2/core.hpp>

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
};

std::string chunkEvidenceStableIdString(std::uint64_t stableId);

#endif
