#ifndef CANDIDATEBATCH_HPP
#define CANDIDATEBATCH_HPP

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "ChunkEvidence.hpp"
#include "ConfigTypes.hpp"

enum class CandidateSource
{
    NoOp,
    Snapshot,
    ChunkWeighted,
    ChunkGeometric,
    ChunkRobust,
    ChunkPeak,
    Stochastic
};

const char *candidateSourceName(CandidateSource source);

struct CandidateProposal
{
    std::uint64_t stableId = 0;
    CandidateSource source = CandidateSource::NoOp;
    cv::Point3f position{0.0f, 0.0f, 0.0f};
    std::uint64_t evidenceId = 0;
    int evidenceFrameOffset = 0;
    float evidenceConfidence = 0.0f;
    double cheapPriority = 0.0;
    bool evaluated = false;
    double costDiff = 0.0;
};

struct CandidateBatchInput
{
    int frame = 0;
    std::string parentName;
    cv::Point3f baselinePosition{0.0f, 0.0f, 0.0f};
    std::optional<cv::Point3f> snapshotPosition;
    float parentMinRadius = 1.0f;
    std::vector<ChunkEvidence> evidence;
};

class CandidateBatch
{
public:
    CandidateBatch(const CandidateBatchInput &input,
                   const CandidateBatchConfig &config);

    const std::vector<CandidateProposal> &candidates() const { return proposals_; }
    std::vector<std::size_t> expensiveCandidateIndices() const;
    void recordEvaluation(std::size_t index, double costDiff);
    std::optional<std::size_t> selectWinner(double baselineObjective) const;

    static double deterministicUnit(std::uint64_t seed,
                                    const std::string &seedNamespace,
                                    int frame,
                                    const std::string &parentName,
                                    const std::string &operation,
                                    std::uint64_t ordinal = 0);

private:
    CandidateBatchConfig config_;
    std::vector<CandidateProposal> proposals_;
};

#endif
