#ifndef RANGE_PREPROCESSOR_HPP
#define RANGE_PREPROCESSOR_HPP

#include "ConfigTypes.hpp"
#include "types.hpp"

#include <cstddef>
#include <ostream>
#include <string>
#include <vector>

struct RangePreprocessStats {
    float rangeUpperBound = 0.0f;
    int ranges = 0;
    int activeRanges = 0;
    int excludedRanges = 0;
    float finalThreshold = 0.0f;
    float brightBoostThreshold = 0.0f;
    float brightBoostEffectiveFactor = 1.0f;
    std::size_t brightBoostedVoxels = 0;
    std::size_t outputNonzero = 0;
};

class RangePreprocessor {
public:
    static ImageStack apply(const ImageStack &rawSlices,
                            const BaseConfig &config,
                            const std::string &imageFile,
                            RangePreprocessStats *stats,
                            std::ostream *logSink);

    static float percentileLinear(std::vector<float> values, float percentile);
};

#endif // RANGE_PREPROCESSOR_HPP
