#include "ZInterpolation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <opencv2/core.hpp>

namespace celluniverse::zinterpolation
{
namespace
{

std::size_t roundNonNegativeToEven(long double value)
{
    long double integral = 0.0L;
    const long double fraction = std::modf(value, &integral);
    long double rounded = integral;
    if (fraction > 0.5L ||
        (fraction == 0.5L && std::fmod(integral, 2.0L) != 0.0L))
    {
        rounded += 1.0L;
    }

    const long double maximum = static_cast<long double>(
        std::numeric_limits<std::size_t>::max());
    if (rounded < 0.0L || rounded > maximum)
    {
        throw std::overflow_error("Z interpolation interval count overflow");
    }
    return static_cast<std::size_t>(rounded);
}

void validateRatio(float ratio)
{
    if (!std::isfinite(ratio) || ratio < 1.0f)
    {
        throw std::invalid_argument(
            "Z interpolation ratio must be finite and at least 1");
    }
}

} // namespace

std::size_t outputSliceCount(std::size_t sourceSliceCount, float ratio)
{
    validateRatio(ratio);
    if (sourceSliceCount <= 1U)
    {
        return sourceSliceCount;
    }

    const std::size_t sourceIntervals = sourceSliceCount - 1U;
    const long double scaledIntervals =
        static_cast<long double>(sourceIntervals) *
        static_cast<long double>(ratio);
    const std::size_t outputIntervals =
        roundNonNegativeToEven(scaledIntervals);
    if (outputIntervals == std::numeric_limits<std::size_t>::max())
    {
        throw std::overflow_error("Z interpolation slice count overflow");
    }
    return outputIntervals + 1U;
}

std::vector<cv::Mat> resampleLinear(const std::vector<cv::Mat> &source,
                                    float ratio)
{
    const std::size_t outputCount = outputSliceCount(source.size(), ratio);
    if (outputCount > static_cast<std::size_t>(
            std::numeric_limits<int>::max()))
    {
        throw std::overflow_error(
            "Z interpolation output is too large for the parallel loop");
    }
    if (source.empty())
    {
        return {};
    }
    if (source.size() == 1U)
    {
        return {source.front()};
    }

    const std::size_t sourceIntervals = source.size() - 1U;
    const std::size_t outputIntervals = outputCount - 1U;
    std::vector<cv::Mat> output(outputCount);

    #pragma omp parallel for schedule(static)
    for (int outputIndex = 0;
         outputIndex < static_cast<int>(outputCount);
         ++outputIndex)
    {
        const std::size_t outputZ = static_cast<std::size_t>(outputIndex);
        const long double sourcePosition =
            static_cast<long double>(outputZ) *
            static_cast<long double>(sourceIntervals) /
            static_cast<long double>(outputIntervals);
        const std::size_t lower = static_cast<std::size_t>(
            std::floor(sourcePosition));
        const std::size_t upper = std::min(lower + 1U, sourceIntervals);
        const double weight = static_cast<double>(
            sourcePosition - static_cast<long double>(lower));

        if (upper == lower || weight == 0.0)
        {
            output[outputZ] = source[lower];
        }
        else
        {
            cv::addWeighted(source[lower], 1.0 - weight,
                            source[upper], weight, 0.0,
                            output[outputZ]);
        }
    }
    return output;
}

} // namespace celluniverse::zinterpolation
