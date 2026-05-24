#include "N2V2Preprocessor.hpp"

#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace
{

void require(bool condition, const char *message)
{
    if (!condition)
    {
        throw std::runtime_error(message);
    }
}

void testPercentileLinear()
{
    const double p25 = n2v2::percentileLinear({1.0f, 2.0f, 3.0f, 4.0f}, 25.0);
    require(std::abs(p25 - 1.75) < 1e-6, "linear percentile p25 mismatch");

    const double p9999 = n2v2::percentileLinear({4.0f, 1.0f, 2.0f, 3.0f}, 99.99);
    require(p9999 > 3.99 && p9999 <= 4.0, "linear percentile p99.99 mismatch");
}

void testTileStarts()
{
    const std::vector<int> yStarts = n2v2::computeTileStartsForAxis(422, 256, 48);
    require(yStarts == std::vector<int>({0, 166}), "422 axis tile starts mismatch");

    const std::vector<int> xStarts = n2v2::computeTileStartsForAxis(517, 256, 48);
    require(xStarts == std::vector<int>({0, 208, 261}), "517 axis tile starts mismatch");
}

void testNoNetworkPipeline()
{
    n2v2::PreprocessConfig config;
    config.enableNetwork = false;
    config.backgroundSubtraction.enabled = true;
    config.backgroundSubtraction.percentile = 20.0;
    config.backgroundSubtraction.excludeZero = true;
    config.contrast.enabled = true;
    config.contrast.limitMode = n2v2::ContrastLimitMode::Absolute;
    config.contrast.lowLimit = 0.0;
    config.contrast.highLimit = 48.0;
    config.contrast.hasHighLimit = true;
    config.contrast.gamma = 1.0;
    config.output.dtype = n2v2::OutputDType::UInt16;

    cv::Mat slice(2, 4, CV_16U);
    unsigned short values[] = {0, 10, 20, 30, 40, 50, 60, 70};
    for (int y = 0; y < slice.rows; ++y)
    {
        for (int x = 0; x < slice.cols; ++x)
        {
            slice.at<unsigned short>(y, x) = values[y * slice.cols + x];
        }
    }

    n2v2::N2V2Preprocessor preprocessor(config);
    std::ostringstream log;
    n2v2::PreprocessResult result = preprocessor.processStack({slice}, "synthetic.tif", log);

    require(result.stack.size() == 1, "no-network output depth mismatch");
    require(result.stack[0].type() == CV_16U, "no-network output dtype mismatch");
    require(result.stack[0].rows == 2 && result.stack[0].cols == 4, "no-network shape mismatch");
    require(std::abs(result.summary.backgroundValue - 22.0) < 1e-6, "background percentile mismatch");
    require(result.summary.outputNonzero == 5, "output nonzero mismatch");
}

} // namespace

int main()
{
    try
    {
        testPercentileLinear();
        testTileStarts();
        testNoNetworkPipeline();
        std::cout << "n2v2_preprocess_test passed\n";
        return 0;
    }
    catch (const std::exception &exc)
    {
        std::cerr << "n2v2_preprocess_test failed: " << exc.what() << '\n';
        return 1;
    }
}
