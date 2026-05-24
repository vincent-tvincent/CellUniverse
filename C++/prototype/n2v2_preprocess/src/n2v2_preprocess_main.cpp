#include "N2V2Preprocessor.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace
{

void printUsage(const char *program)
{
    std::cerr << "Usage: " << program
              << " <firstFrame> <lastFrame> <inputPath|pattern> <outputDir> <configYaml>\n";
}

fs::path outputImagePath(const fs::path &outputDir, const fs::path &inputPath)
{
    return outputDir / "images" /
           (inputPath.stem().string() + "_preprocessed" + inputPath.extension().string());
}

fs::path outputIntermediatePath(const fs::path &outputDir, const fs::path &inputPath)
{
    return outputDir / "intermediate" /
           (inputPath.stem().string() + "_n2v2_denoised" + inputPath.extension().string());
}

fs::path outputLogPath(const fs::path &outputDir, const fs::path &inputPath)
{
    return outputDir / "logs" / (inputPath.stem().string() + ".log");
}

} // namespace

int main(int argc, char **argv)
{
    if (argc != 6)
    {
        printUsage(argv[0]);
        return 2;
    }

    try
    {
        const int firstFrame = std::stoi(argv[1]);
        const int lastFrame = std::stoi(argv[2]);
        const std::string inputPath = argv[3];
        const fs::path outputDir = argv[4];
        const fs::path configPath = argv[5];

        n2v2::PreprocessConfig config = n2v2::loadPreprocessConfig(configPath);
        n2v2::N2V2Preprocessor preprocessor(config);
        const std::vector<fs::path> inputs =
            n2v2::resolveInputFrames(inputPath, firstFrame, lastFrame);

        fs::create_directories(outputDir / "images");
        fs::create_directories(outputDir / "logs");
        if (config.output.writeIntermediate)
        {
            fs::create_directories(outputDir / "intermediate");
        }

        std::vector<n2v2::FrameSummary> summaries;
        summaries.reserve(inputs.size());

        for (std::size_t index = 0; index < inputs.size(); ++index)
        {
            const fs::path &input = inputs[index];
            const fs::path logPath = outputLogPath(outputDir, input);
            std::ofstream log(logPath);
            if (!log)
            {
                throw std::runtime_error("Failed to open log file: " + logPath.string());
            }

            std::cout << "Processing " << (index + 1) << "/" << inputs.size()
                      << ": " << input << std::endl;
            log << "Input: " << input << '\n';
            log << "Config: " << configPath << '\n';

            const std::vector<cv::Mat> rawStack = n2v2::loadTiffStack(input);
            n2v2::PreprocessResult result =
                preprocessor.processStack(rawStack, input, log);

            const fs::path imagePath = outputImagePath(outputDir, input);
            n2v2::writeTiffStack(imagePath, result.stack);
            result.summary.outputPath = imagePath;

            if (config.output.writeIntermediate)
            {
                const fs::path intermediatePath = outputIntermediatePath(outputDir, input);
                n2v2::writeTiffStack(intermediatePath, result.intermediateStack);
                result.summary.intermediatePath = intermediatePath;
            }

            log << "Output: " << result.summary.outputPath << '\n';
            if (!result.summary.intermediatePath.empty())
            {
                log << "Intermediate: " << result.summary.intermediatePath << '\n';
            }
            log << "Input nonzero: " << result.summary.inputNonzero << '\n';
            log << "Output nonzero: " << result.summary.outputNonzero << '\n';
            summaries.push_back(std::move(result.summary));
        }

        n2v2::writeSummaryCsv(outputDir / "preprocess_summary.csv", summaries);
        std::cout << "Wrote " << summaries.size()
                  << " preprocessed TIFFs to " << (outputDir / "images")
                  << std::endl;
        return 0;
    }
    catch (const std::exception &exc)
    {
        std::cerr << "celluniverse_preprocess: " << exc.what() << std::endl;
        return 1;
    }
}
