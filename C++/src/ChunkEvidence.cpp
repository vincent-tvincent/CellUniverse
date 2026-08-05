#include "../includes/ChunkEvidence.hpp"

#include <sstream>

std::string chunkEvidenceStableIdString(std::uint64_t stableId)
{
    std::ostringstream out;
    out << "chunk_" << stableId;
    return out.str();
}
