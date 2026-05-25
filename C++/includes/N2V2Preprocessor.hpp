#ifndef CELLUNIVERSE_N2V2_PREPROCESSOR_HPP
#define CELLUNIVERSE_N2V2_PREPROCESSOR_HPP

#include "yaml-cpp/yaml.h"

#include <opencv2/opencv.hpp>

#include <filesystem>
#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

namespace n2v2
{

namespace fs = std::filesystem;

enum class OutputDType
{
    Preserve,
    UInt8,
    UInt16,
    Float32,
};

enum class ContrastLimitMode
{
    Absolute,
    Percentile,
};

enum class ContrastScope
{
    Stack,
    Slice,
};

struct BackgroundSubtractionConfig
{
    bool enabled = true;
    double percentile = 20.0;
    bool excludeZero = true;
    double clipMin = 0.0;
};

struct ContrastConfig
{
    bool enabled = true;
    ContrastLimitMode limitMode = ContrastLimitMode::Percentile;
    double lowLimit = 10.0;
    bool hasHighLimit = false;
    double highLimit = 0.0;
    double lowPercentile = 75.0;
    double highPercentile = 99.99;
    bool excludeZero = true;
    ContrastScope scope = ContrastScope::Stack;
    double gamma = 1.45;
    bool preserveZeroPixels = true;
};

struct OutputConfig
{
    OutputDType dtype = OutputDType::Preserve;
    bool writeIntermediate = false;
    bool quantizeBeforeContrast = true;
};

struct PreprocessConfig
{
    bool enableNetwork = true;
    fs::path modelPath;
    std::string device = "auto";
    int inferenceBatchSize = 16;
    std::vector<int> tileSize{256, 256};
    std::vector<int> tileOverlap{48, 48};

    double scalePercentile = 99.9;
    bool useNonzeroPixels = true;
    double fallbackScale = 65536.0;
    double careamicsMean = 0.33040523529052734;
    double careamicsStd = 0.23699833071884863;

    BackgroundSubtractionConfig backgroundSubtraction;
    ContrastConfig contrast;
    OutputConfig output;
};

struct FrameSummary
{
    fs::path inputPath;
    fs::path outputPath;
    fs::path intermediatePath;
    int z = 0;
    int y = 0;
    int x = 0;
    int inputCvType = -1;
    int outputCvType = -1;
    bool networkEnabled = true;
    double scale = 0.0;
    bool backgroundEnabled = false;
    double backgroundValue = 0.0;
    bool contrastEnabled = false;
    double contrastLow = 0.0;
    double contrastHigh = 0.0;
    long long inputNonzero = 0;
    long long outputNonzero = 0;
};

struct PreprocessResult
{
    std::vector<cv::Mat> stack;
    std::vector<cv::Mat> intermediateStack;
    FrameSummary summary;
};

class N2V2Preprocessor
{
public:
    explicit N2V2Preprocessor(PreprocessConfig config);
    ~N2V2Preprocessor();

    PreprocessResult processStack(const std::vector<cv::Mat> &rawStack,
                                  const fs::path &inputPath,
                                  std::ostream &log);

    const PreprocessConfig &config() const { return config_; }

private:
    PreprocessConfig config_;
    class Impl;
    std::unique_ptr<Impl> impl_;
};

PreprocessConfig loadPreprocessConfig(const fs::path &path);

std::vector<fs::path> resolveInputFrames(const std::string &input,
                                         int firstFrame,
                                         int lastFrame);

std::vector<cv::Mat> loadTiffStack(const fs::path &path);
void writeTiffStack(const fs::path &path, const std::vector<cv::Mat> &stack);
void writeSummaryCsv(const fs::path &path, const std::vector<FrameSummary> &summaries);

std::string cvTypeName(int type);
double percentileLinear(std::vector<float> values, double percentile);
std::vector<int> computeTileStartsForAxis(int axisSize, int patchSize, int overlap);

} // namespace n2v2

#endif
