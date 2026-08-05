#ifndef CELLUNIVERSE_Z_INTERPOLATION_HPP
#define CELLUNIVERSE_Z_INTERPOLATION_HPP

#include <cstddef>
#include <vector>

#include <opencv2/core/mat.hpp>

namespace celluniverse::zinterpolation
{

// Match the initializer's endpoint-preserving Z grid:
// round_to_even((sourceSliceCount - 1) * ratio) + 1.
std::size_t outputSliceCount(std::size_t sourceSliceCount, float ratio);

// Linearly resample the source stack onto the endpoint-preserving grid above.
std::vector<cv::Mat> resampleLinear(const std::vector<cv::Mat> &source,
                                    float ratio);

} // namespace celluniverse::zinterpolation

#endif
