#include "ZInterpolation.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace zi = celluniverse::zinterpolation;

namespace
{

void require(bool condition, const std::string &message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

void requireNear(float actual, float expected, float tolerance,
                 const std::string &message)
{
    require(std::abs(actual - expected) <= tolerance,
            message + ": actual=" + std::to_string(actual) +
            " expected=" + std::to_string(expected));
}

cv::Mat scalarSlice(float value)
{
    return cv::Mat(1, 1, CV_32F, cv::Scalar(value));
}

void requireInvalid(float ratio)
{
    bool threw = false;
    try
    {
        (void)zi::outputSliceCount(50U, ratio);
    }
    catch (const std::invalid_argument &)
    {
        threw = true;
    }
    require(threw, "invalid ratio was accepted");
}

} // namespace

int main()
{
    try
    {
        require(zi::outputSliceCount(33U, 7.0f) == 225U,
                "integer interpolation depth changed");
        require(zi::outputSliceCount(50U, 5.5f) == 271U,
                "fractional 5.5 interpolation depth mismatch");
        require(zi::outputSliceCount(2U, 2.5f) == 3U,
                "nearest-even lower tie mismatch");
        require(zi::outputSliceCount(2U, 3.5f) == 5U,
                "nearest-even upper tie mismatch");
        require(zi::outputSliceCount(1U, 5.5f) == 1U,
                "single-slice depth mismatch");

        const std::vector<cv::Mat> source = {
            scalarSlice(0.0f), scalarSlice(10.0f), scalarSlice(20.0f)};
        const std::vector<cv::Mat> integer = zi::resampleLinear(source, 2.0f);
        require(integer.size() == 5U, "integer resampling size mismatch");
        for (std::size_t index = 0; index < integer.size(); ++index)
        {
            requireNear(integer[index].at<float>(0, 0),
                        static_cast<float>(index) * 5.0f, 1.0e-5f,
                        "integer resampling value mismatch");
        }

        const std::vector<cv::Mat> fractional =
            zi::resampleLinear(source, 1.5f);
        require(fractional.size() == 4U,
                "fractional resampling size mismatch");
        requireNear(fractional.front().at<float>(0, 0), 0.0f, 1.0e-6f,
                    "first endpoint changed");
        requireNear(fractional[1].at<float>(0, 0), 20.0f / 3.0f, 1.0e-5f,
                    "first fractional interior mismatch");
        requireNear(fractional[2].at<float>(0, 0), 40.0f / 3.0f, 1.0e-5f,
                    "second fractional interior mismatch");
        requireNear(fractional.back().at<float>(0, 0), 20.0f, 1.0e-6f,
                    "last endpoint changed");

        requireInvalid(0.99f);
        requireInvalid(std::numeric_limits<float>::infinity());
        requireInvalid(std::numeric_limits<float>::quiet_NaN());

        std::cout << "Z interpolation tests passed\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "Z interpolation test failed: " << error.what() << '\n';
        return 1;
    }
}
