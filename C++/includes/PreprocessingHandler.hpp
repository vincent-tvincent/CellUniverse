#ifndef PREPROCESSING_HANDLER_HPP
#define PREPROCESSING_HANDLER_HPP

#include "ConfigTypes.hpp"
#include "types.hpp"

#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

struct N2V2Runtime;

struct PreprocessedFrame
{
    std::vector<cv::Mat> stack;
    std::vector<cv::Mat> signalMap;
    float sampledBackground = 0.0f;
};

class PreprocessingHandler
{
public:
    PreprocessingHandler(const BaseConfig &config, std::string outputPath);
    ~PreprocessingHandler();

    std::vector<cv::Mat> probePreprocessedStack(const fs::path &imagePath,
                                                std::ostream &log) const;

    PreprocessedFrame preprocessFrame(const fs::path &imagePath,
                                      std::optional<float> edgeTarget,
                                      std::ostream &log) const;

    std::vector<PreprocessedFrame> preprocessBatch(const PathVec &imagePaths,
                                                   bool retainStacks,
                                                   std::ostream &log,
                                                   float *resolvedEdgeTarget) const;

private:
    N2V2Runtime &n2v2Runtime(std::ostream &log) const;

    const BaseConfig &config_;
    std::string outputPath_;
    mutable std::unique_ptr<N2V2Runtime> n2v2Runtime_;
};

#endif // PREPROCESSING_HANDLER_HPP
