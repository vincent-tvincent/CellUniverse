#include "N2V2Preprocessor.hpp"

#include <torch/script.h>
#include <torch/torch.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace n2v2
{
namespace
{

struct AxisTile
{
    int coord = 0;
    int stitch = 0;
    int crop = 0;
    int cropSize = 0;
};

struct Tile2D
{
    AxisTile y;
    AxisTile x;
};

std::string lowerCopy(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

std::string deviceToString(const torch::Device &device)
{
    std::ostringstream stream;
    stream << device;
    return stream.str();
}

bool isTiffPath(const fs::path &path)
{
    const std::string ext = lowerCopy(path.extension().string());
    return ext == ".tif" || ext == ".tiff";
}

bool shouldSkipPath(const fs::path &path)
{
    const std::string name = path.filename().string();
    return name.empty() || name[0] == '.' || name.rfind("._", 0) == 0;
}

int requirePositiveInt(const YAML::Node &node, const char *name, int fallback)
{
    const int value = node ? node.as<int>() : fallback;
    if (value <= 0)
    {
        throw std::invalid_argument(std::string(name) + " must be positive");
    }
    return value;
}

std::vector<int> readIntPair(const YAML::Node &node,
                             const char *name,
                             std::vector<int> fallback)
{
    if (!node)
    {
        return fallback;
    }
    if (!node.IsSequence() || node.size() != 2)
    {
        throw std::invalid_argument(std::string(name) + " must contain exactly two integers");
    }
    fallback[0] = node[0].as<int>();
    fallback[1] = node[1].as<int>();
    if (fallback[0] <= 0 || fallback[1] <= 0)
    {
        throw std::invalid_argument(std::string(name) + " values must be positive");
    }
    return fallback;
}

OutputDType parseOutputDType(const std::string &value)
{
    const std::string dtype = lowerCopy(value);
    if (dtype == "preserve")
    {
        return OutputDType::Preserve;
    }
    if (dtype == "uint8")
    {
        return OutputDType::UInt8;
    }
    if (dtype == "uint16")
    {
        return OutputDType::UInt16;
    }
    if (dtype == "float32")
    {
        return OutputDType::Float32;
    }
    throw std::invalid_argument("output.dtype must be preserve, uint8, uint16, or float32");
}

ContrastLimitMode parseContrastLimitMode(const std::string &value)
{
    const std::string mode = lowerCopy(value);
    if (mode == "absolute")
    {
        return ContrastLimitMode::Absolute;
    }
    if (mode == "percentile")
    {
        return ContrastLimitMode::Percentile;
    }
    throw std::invalid_argument("contrast.limit_mode must be absolute or percentile");
}

ContrastScope parseContrastScope(const std::string &value)
{
    const std::string scope = lowerCopy(value);
    if (scope == "stack")
    {
        return ContrastScope::Stack;
    }
    if (scope == "slice")
    {
        return ContrastScope::Slice;
    }
    throw std::invalid_argument("contrast.scope must be stack or slice");
}

int outputDepth(OutputDType dtype, int inputDepth)
{
    if (dtype == OutputDType::UInt8)
    {
        return CV_8U;
    }
    if (dtype == OutputDType::UInt16)
    {
        return CV_16U;
    }
    if (dtype == OutputDType::Float32)
    {
        return CV_32F;
    }
    if (inputDepth == CV_8U || inputDepth == CV_16U || inputDepth == CV_32F)
    {
        return inputDepth;
    }
    return CV_16U;
}

double integerMaxForDepth(int depth)
{
    if (depth == CV_8U)
    {
        return 255.0;
    }
    if (depth == CV_16U)
    {
        return 65535.0;
    }
    return 1.0;
}

std::vector<AxisTile> computeAxisTiles(int axisSize, int patchSize, int overlap)
{
    if (axisSize <= 0 || patchSize <= 0)
    {
        throw std::invalid_argument("axis size and patch size must be positive");
    }
    if (overlap < 0 || overlap >= patchSize)
    {
        throw std::invalid_argument("tile overlap must be non-negative and smaller than tile size");
    }

    std::vector<AxisTile> tiles;
    const int step = patchSize - overlap;
    const int stop = std::max(1, axisSize - overlap);
    for (int i = 0; i < stop; i += step)
    {
        AxisTile tile;
        if (i == 0)
        {
            tile.coord = 0;
            tile.crop = 0;
            tile.stitch = 0;
            tile.cropSize = axisSize <= patchSize ? axisSize : patchSize - overlap / 2;
        }
        else if (i + patchSize < axisSize)
        {
            tile.coord = i;
            tile.crop = overlap / 2;
            tile.stitch = tile.coord + tile.crop;
            tile.cropSize = patchSize - overlap;
        }
        else
        {
            const int previousCropSize = tiles.empty() ? 1 : tiles.back().cropSize;
            const int previousStitch = tiles.empty() ? 0 : tiles.back().stitch;
            const int previousTileEnd = previousStitch + previousCropSize;
            tile.coord = std::max(0, axisSize - patchSize);
            tile.stitch = previousTileEnd;
            tile.crop = tile.stitch - tile.coord;
            tile.cropSize = axisSize - tile.stitch;
        }

        if (tile.cropSize > 0)
        {
            tiles.push_back(tile);
        }
    }
    return tiles;
}

std::vector<Tile2D> computeTiles2D(int rows, int cols, const PreprocessConfig &config)
{
    const auto yTiles = computeAxisTiles(rows, config.tileSize[0], config.tileOverlap[0]);
    const auto xTiles = computeAxisTiles(cols, config.tileSize[1], config.tileOverlap[1]);
    std::vector<Tile2D> tiles;
    tiles.reserve(yTiles.size() * xTiles.size());
    for (const auto &y : yTiles)
    {
        for (const auto &x : xTiles)
        {
            tiles.push_back(Tile2D{y, x});
        }
    }
    return tiles;
}

cv::Mat toGray(const cv::Mat &slice)
{
    if (slice.channels() == 1)
    {
        return slice.clone();
    }
    cv::Mat gray;
    if (slice.channels() == 3)
    {
        cv::cvtColor(slice, gray, cv::COLOR_BGR2GRAY);
    }
    else if (slice.channels() == 4)
    {
        cv::cvtColor(slice, gray, cv::COLOR_BGRA2GRAY);
    }
    else
    {
        throw std::runtime_error("Unsupported TIFF channel count: " + std::to_string(slice.channels()));
    }
    return gray;
}

std::vector<cv::Mat> toFloatStack(const std::vector<cv::Mat> &stack)
{
    std::vector<cv::Mat> output(stack.size());
    for (std::size_t i = 0; i < stack.size(); ++i)
    {
        stack[i].convertTo(output[i], CV_32F);
    }
    return output;
}

long long countNonzeroStack(const std::vector<cv::Mat> &stack)
{
    long long total = 0;
    for (const auto &slice : stack)
    {
        total += cv::countNonZero(slice != 0);
    }
    return total;
}

void collectValues(const std::vector<cv::Mat> &stack, bool excludeZero, std::vector<float> &values)
{
    values.clear();
    std::size_t reserveCount = 0;
    for (const auto &slice : stack)
    {
        reserveCount += slice.total();
    }
    values.reserve(reserveCount);
    for (const auto &slice : stack)
    {
        CV_Assert(slice.type() == CV_32F);
        for (int y = 0; y < slice.rows; ++y)
        {
            const float *row = slice.ptr<float>(y);
            for (int x = 0; x < slice.cols; ++x)
            {
                const float value = row[x];
                if (!std::isfinite(value))
                {
                    continue;
                }
                if (excludeZero && value == 0.0f)
                {
                    continue;
                }
                values.push_back(value);
            }
        }
    }
}

double stackPercentile(const std::vector<cv::Mat> &stack, double percentile, bool excludeZero)
{
    std::vector<float> values;
    collectValues(stack, excludeZero, values);
    return percentileLinear(std::move(values), percentile);
}

double slicePercentile(const cv::Mat &slice, double percentile, bool excludeZero)
{
    CV_Assert(slice.type() == CV_32F);
    std::vector<float> values;
    values.reserve(slice.total());
    for (int y = 0; y < slice.rows; ++y)
    {
        const float *row = slice.ptr<float>(y);
        for (int x = 0; x < slice.cols; ++x)
        {
            const float value = row[x];
            if (!std::isfinite(value))
            {
                continue;
            }
            if (excludeZero && value == 0.0f)
            {
                continue;
            }
            values.push_back(value);
        }
    }
    return percentileLinear(std::move(values), percentile);
}

std::vector<cv::Mat> normalizeForNetwork(const std::vector<cv::Mat> &rawFloat,
                                         double scale)
{
    std::vector<cv::Mat> normalized(rawFloat.size());
    const float invScale = scale > 0.0 ? static_cast<float>(1.0 / scale) : 0.0f;
    for (std::size_t i = 0; i < rawFloat.size(); ++i)
    {
        normalized[i] = rawFloat[i] * invScale;
        cv::min(normalized[i], 1.0f, normalized[i]);
        cv::max(normalized[i], 0.0f, normalized[i]);
    }
    return normalized;
}

std::vector<cv::Mat> multiplyStack(const std::vector<cv::Mat> &stack, double scale)
{
    std::vector<cv::Mat> output(stack.size());
    for (std::size_t i = 0; i < stack.size(); ++i)
    {
        output[i] = stack[i] * static_cast<float>(scale);
    }
    return output;
}

std::vector<cv::Mat> castStack(const std::vector<cv::Mat> &stack,
                               int depth,
                               bool inputIsNormalized)
{
    std::vector<cv::Mat> output(stack.size());
    const double scale = inputIsNormalized ? integerMaxForDepth(depth) : 1.0;
    for (std::size_t i = 0; i < stack.size(); ++i)
    {
        cv::Mat source;
        stack[i].convertTo(source, CV_32F);
        if (depth == CV_32F)
        {
            if (inputIsNormalized)
            {
                cv::min(source, 1.0f, source);
                cv::max(source, 0.0f, source);
            }
            output[i] = source;
            continue;
        }

        cv::Mat scaled = inputIsNormalized ? source * scale : source;
        scaled.convertTo(output[i], depth);
    }
    return output;
}

std::vector<cv::Mat> castAndClipStack(const std::vector<cv::Mat> &stack,
                                      int depth,
                                      bool inputIsNormalized)
{
    std::vector<cv::Mat> clipped(stack.size());
    const double maxValue = integerMaxForDepth(depth);
    for (std::size_t i = 0; i < stack.size(); ++i)
    {
        stack[i].convertTo(clipped[i], CV_32F);
        if (inputIsNormalized)
        {
            cv::min(clipped[i], 1.0f, clipped[i]);
            cv::max(clipped[i], 0.0f, clipped[i]);
        }
        else if (depth != CV_32F)
        {
            cv::min(clipped[i], maxValue, clipped[i]);
            cv::max(clipped[i], 0.0f, clipped[i]);
        }
    }
    return castStack(clipped, depth, inputIsNormalized);
}

std::vector<cv::Mat> applyBackgroundSubtraction(const std::vector<cv::Mat> &stack,
                                                const BackgroundSubtractionConfig &config,
                                                double &background)
{
    if (!config.enabled)
    {
        background = 0.0;
        return toFloatStack(stack);
    }
    background = stackPercentile(stack, config.percentile, config.excludeZero);
    if (!std::isfinite(background))
    {
        background = 0.0;
    }

    std::vector<cv::Mat> output(stack.size());
    for (std::size_t i = 0; i < stack.size(); ++i)
    {
        stack[i].convertTo(output[i], CV_32F);
        output[i] -= static_cast<float>(background);
        cv::max(output[i], static_cast<float>(config.clipMin), output[i]);
    }
    return output;
}

std::pair<double, double> absoluteLimits(const std::vector<cv::Mat> &stack,
                                         const ContrastConfig &config)
{
    double minValue = std::numeric_limits<double>::infinity();
    double maxValue = -std::numeric_limits<double>::infinity();
    for (const auto &slice : stack)
    {
        double sliceMin = 0.0;
        double sliceMax = 0.0;
        cv::minMaxLoc(slice, &sliceMin, &sliceMax);
        minValue = std::min(minValue, sliceMin);
        maxValue = std::max(maxValue, sliceMax);
    }
    if (!std::isfinite(minValue) || !std::isfinite(maxValue))
    {
        minValue = 0.0;
        maxValue = 0.0;
    }
    const double low = config.lowLimit;
    const double high = config.hasHighLimit ? config.highLimit : maxValue;
    return {low, high};
}

cv::Mat normalizeWithLimits(const cv::Mat &source, double low, double high)
{
    cv::Mat normalized(source.size(), CV_32F, cv::Scalar(0));
    if (high <= low)
    {
        return normalized;
    }
    source.convertTo(normalized, CV_32F, 1.0 / (high - low), -low / (high - low));
    cv::min(normalized, 1.0f, normalized);
    cv::max(normalized, 0.0f, normalized);
    return normalized;
}

std::vector<cv::Mat> applyContrast(const std::vector<cv::Mat> &stack,
                                   const ContrastConfig &config,
                                   double &lowOut,
                                   double &highOut)
{
    if (!config.enabled)
    {
        lowOut = 0.0;
        highOut = 0.0;
        return toFloatStack(stack);
    }

    std::vector<cv::Mat> source = toFloatStack(stack);
    std::vector<cv::Mat> output(source.size());
    std::vector<double> lows;
    std::vector<double> highs;

    if (config.scope == ContrastScope::Stack)
    {
        std::pair<double, double> limits =
            config.limitMode == ContrastLimitMode::Absolute
                ? absoluteLimits(source, config)
                : std::make_pair(
                      stackPercentile(source, config.lowPercentile, config.excludeZero),
                      stackPercentile(source, config.highPercentile, config.excludeZero));
        lowOut = limits.first;
        highOut = limits.second;
        for (std::size_t i = 0; i < source.size(); ++i)
        {
            output[i] = normalizeWithLimits(source[i], lowOut, highOut);
        }
    }
    else
    {
        lows.reserve(source.size());
        highs.reserve(source.size());
        for (std::size_t i = 0; i < source.size(); ++i)
        {
            std::pair<double, double> limits =
                config.limitMode == ContrastLimitMode::Absolute
                    ? absoluteLimits(std::vector<cv::Mat>{source[i]}, config)
                    : std::make_pair(
                          slicePercentile(source[i], config.lowPercentile, config.excludeZero),
                          slicePercentile(source[i], config.highPercentile, config.excludeZero));
            lows.push_back(limits.first);
            highs.push_back(limits.second);
            output[i] = normalizeWithLimits(source[i], limits.first, limits.second);
        }
        const auto meanFinite = [](const std::vector<double> &values) {
            double sum = 0.0;
            int count = 0;
            for (double value : values)
            {
                if (std::isfinite(value))
                {
                    sum += value;
                    ++count;
                }
            }
            return count > 0 ? sum / count : 0.0;
        };
        lowOut = meanFinite(lows);
        highOut = meanFinite(highs);
    }

    for (std::size_t i = 0; i < output.size(); ++i)
    {
        cv::Mat zeroMask = source[i] == 0.0f;
        if (config.gamma != 1.0)
        {
            cv::pow(output[i], config.gamma, output[i]);
        }
        if (config.preserveZeroPixels)
        {
            output[i].setTo(0.0f, zeroMask);
        }
        cv::min(output[i], 1.0f, output[i]);
        cv::max(output[i], 0.0f, output[i]);
    }
    return output;
}

torch::Device resolveTorchDevice(const std::string &requested)
{
    const std::string device = lowerCopy(requested);
    if (device.empty() || device == "auto")
    {
        return torch::cuda::is_available() ? torch::Device(torch::kCUDA)
                                           : torch::Device(torch::kCPU);
    }
    if (device == "cpu")
    {
        return torch::Device(torch::kCPU);
    }
    if (device == "cuda")
    {
        if (!torch::cuda::is_available())
        {
            throw std::runtime_error("n2v2_preprocess.device=cuda requested, but CUDA is not available");
        }
        return torch::Device(torch::kCUDA);
    }
    if (device.rfind("cuda:", 0) == 0)
    {
        if (!torch::cuda::is_available())
        {
            throw std::runtime_error("n2v2_preprocess.device=" + requested +
                                     " requested, but CUDA is not available");
        }
        const std::string indexText = device.substr(5);
        if (indexText.empty())
        {
            throw std::runtime_error("n2v2_preprocess.device has an empty CUDA index");
        }
        const int index = std::stoi(indexText);
        return torch::Device(torch::kCUDA, index);
    }
    throw std::runtime_error("n2v2_preprocess.device must be auto, cpu, cuda, or cuda:<index>");
}

std::string csvEscape(const fs::path &path)
{
    std::string value = path.string();
    if (value.find_first_of(",\"\n") == std::string::npos)
    {
        return value;
    }
    std::string escaped = "\"";
    for (char ch : value)
    {
        if (ch == '"')
        {
            escaped += "\"\"";
        }
        else
        {
            escaped += ch;
        }
    }
    escaped += '"';
    return escaped;
}

} // namespace

class N2V2Preprocessor::Impl
{
public:
    explicit Impl(const PreprocessConfig &config)
        : device(resolveTorchDevice(config.device))
    {
        networkLoaded = config.enableNetwork;
        if (!config.enableNetwork)
        {
            return;
        }
        if (config.modelPath.empty())
        {
            throw std::runtime_error("n2v2_preprocess.model_path is required when enable_network is true");
        }
        if (!fs::exists(config.modelPath))
        {
            throw std::runtime_error("n2v2_preprocess.model_path does not exist: " +
                                     config.modelPath.string());
        }
        module = torch::jit::load(config.modelPath.string(), device);
        module.eval();
    }

    std::vector<cv::Mat> predictNormalized(const std::vector<cv::Mat> &normalized,
                                           const PreprocessConfig &config,
                                           std::ostream &log)
    {
        torch::NoGradGuard noGrad;
        std::vector<cv::Mat> predicted(normalized.size());
        const int tileH = config.tileSize[0];
        const int tileW = config.tileSize[1];
        const int batchSize = std::max(1, config.inferenceBatchSize);

        for (std::size_t z = 0; z < normalized.size(); ++z)
        {
            CV_Assert(normalized[z].type() == CV_32F);
            predicted[z] = cv::Mat::zeros(normalized[z].size(), CV_32F);
            const std::vector<Tile2D> tiles = computeTiles2D(
                normalized[z].rows, normalized[z].cols, config);

            for (std::size_t start = 0; start < tiles.size(); start += static_cast<std::size_t>(batchSize))
            {
                const std::size_t end = std::min(tiles.size(), start + static_cast<std::size_t>(batchSize));
                const int currentBatch = static_cast<int>(end - start);
                torch::Tensor input = torch::empty(
                    {currentBatch, 1, tileH, tileW},
                    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
                auto inputAccessor = input.accessor<float, 4>();

                for (int b = 0; b < currentBatch; ++b)
                {
                    const Tile2D &tile = tiles[start + static_cast<std::size_t>(b)];
                    for (int yy = 0; yy < tileH; ++yy)
                    {
                        const int srcY = tile.y.coord + yy;
                        for (int xx = 0; xx < tileW; ++xx)
                        {
                            const int srcX = tile.x.coord + xx;
                            float value = 0.0f;
                            if (srcY >= 0 && srcY < normalized[z].rows &&
                                srcX >= 0 && srcX < normalized[z].cols)
                            {
                                value = normalized[z].ptr<float>(srcY)[srcX];
                            }
                            inputAccessor[b][0][yy][xx] =
                                static_cast<float>((value - config.careamicsMean) / config.careamicsStd);
                        }
                    }
                }

                torch::Tensor output = module.forward({input.to(device)}).toTensor();
                output = output.to(torch::kCPU).contiguous();
                if (output.dim() != 4 || output.size(0) != currentBatch ||
                    output.size(1) < 1 || output.size(2) < tileH || output.size(3) < tileW)
                {
                    throw std::runtime_error("N2V2 TorchScript model returned an unexpected output shape");
                }
                auto outputAccessor = output.accessor<float, 4>();

                for (int b = 0; b < currentBatch; ++b)
                {
                    const Tile2D &tile = tiles[start + static_cast<std::size_t>(b)];
                    for (int yy = 0; yy < tile.y.cropSize; ++yy)
                    {
                            const int tileY = tile.y.crop + yy;
                            const int dstY = tile.y.stitch + yy;
                            float *dstRow = predicted[z].ptr<float>(dstY);
                            for (int xx = 0; xx < tile.x.cropSize; ++xx)
                            {
                                const int tileX = tile.x.crop + xx;
                                const int dstX = tile.x.stitch + xx;
                                dstRow[dstX] = static_cast<float>(
                                    outputAccessor[b][0][tileY][tileX] * config.careamicsStd +
                                    config.careamicsMean);
                            }
                        }
                    }
            }

            if ((z + 1) % 8 == 0 || z + 1 == normalized.size())
            {
                log << "[N2V2] predicted_slice=" << (z + 1)
                    << "/" << normalized.size()
                    << " tiles=" << tiles.size()
                    << " backend=libtorch"
                    << " device=" << deviceToString(device)
                    << '\n';
            }
        }

        for (auto &slice : predicted)
        {
            cv::min(slice, 1.0f, slice);
            cv::max(slice, 0.0f, slice);
        }
        return predicted;
    }

    const torch::Device &selectedDevice() const
    {
        return device;
    }

    bool isNetworkLoaded() const
    {
        return networkLoaded;
    }

private:
    torch::Device device;
    torch::jit::script::Module module;
    bool networkLoaded = false;
};

N2V2Preprocessor::N2V2Preprocessor(PreprocessConfig config)
    : config_(std::move(config)),
      impl_(std::make_unique<Impl>(config_))
{
}

N2V2Preprocessor::~N2V2Preprocessor() = default;

PreprocessResult N2V2Preprocessor::processStack(const std::vector<cv::Mat> &rawStack,
                                                const fs::path &inputPath,
                                                std::ostream &log)
{
    if (rawStack.empty())
    {
        throw std::runtime_error("Cannot preprocess an empty stack");
    }

    const int inputDepth = rawStack.front().depth();
    const int finalDepth = outputDepth(config_.output.dtype, inputDepth);
    FrameSummary summary;
    summary.inputPath = inputPath;
    summary.z = static_cast<int>(rawStack.size());
    summary.y = rawStack.front().rows;
    summary.x = rawStack.front().cols;
    summary.inputCvType = rawStack.front().type();
    summary.networkEnabled = config_.enableNetwork;
    summary.inputNonzero = countNonzeroStack(rawStack);

    log << "[N2V2] runtime"
        << " backend=libtorch"
        << " requested_device=" << config_.device
        << " selected_device=" << deviceToString(impl_->selectedDevice())
        << " network_enabled=" << config_.enableNetwork
        << " network_loaded=" << impl_->isNetworkLoaded()
        << '\n';

    std::vector<cv::Mat> rawFloat = toFloatStack(rawStack);
    std::vector<cv::Mat> restoredFloat;

    if (config_.enableNetwork)
    {
        summary.scale = stackPercentile(rawFloat, config_.scalePercentile, config_.useNonzeroPixels);
        if (!std::isfinite(summary.scale) || summary.scale <= 0.0)
        {
            summary.scale = config_.fallbackScale;
        }
        log << "[N2V2] scale=" << summary.scale
            << " percentile=" << config_.scalePercentile
            << " use_nonzero=" << config_.useNonzeroPixels
            << '\n';
        const std::vector<cv::Mat> normalized = normalizeForNetwork(rawFloat, summary.scale);
        const std::vector<cv::Mat> predictedNormalized =
            impl_->predictNormalized(normalized, config_, log);
        restoredFloat = multiplyStack(predictedNormalized, summary.scale);
    }
    else
    {
        summary.scale = 0.0;
        restoredFloat = rawFloat;
        log << "[N2V2] network disabled; using raw stack intensities\n";
    }

    double background = 0.0;
    std::vector<cv::Mat> postBackground = applyBackgroundSubtraction(
        restoredFloat, config_.backgroundSubtraction, background);
    summary.backgroundEnabled = config_.backgroundSubtraction.enabled;
    summary.backgroundValue = background;
    log << "[Background] enabled=" << summary.backgroundEnabled
        << " value=" << summary.backgroundValue << '\n';

    const int intermediateDepth = outputDepth(OutputDType::Preserve, inputDepth);
    std::vector<cv::Mat> intermediate = castAndClipStack(postBackground, intermediateDepth, false);
    std::vector<cv::Mat> contrastInput =
        config_.output.quantizeBeforeContrast ? toFloatStack(intermediate) : postBackground;

    double contrastLow = 0.0;
    double contrastHigh = 0.0;
    std::vector<cv::Mat> postContrast = applyContrast(
        contrastInput, config_.contrast, contrastLow, contrastHigh);
    summary.contrastEnabled = config_.contrast.enabled;
    summary.contrastLow = contrastLow;
    summary.contrastHigh = contrastHigh;
    log << "[Contrast] enabled=" << summary.contrastEnabled
        << " low=" << summary.contrastLow
        << " high=" << summary.contrastHigh
        << " gamma=" << config_.contrast.gamma << '\n';

    std::vector<cv::Mat> finalStack = config_.contrast.enabled
        ? castAndClipStack(postContrast, finalDepth, true)
        : castAndClipStack(postContrast, finalDepth, false);
    summary.outputCvType = finalStack.front().type();
    summary.outputNonzero = countNonzeroStack(finalStack);

    return PreprocessResult{
        std::move(finalStack),
        std::move(intermediate),
        summary,
    };
}

PreprocessConfig loadPreprocessConfig(const fs::path &path)
{
    YAML::Node root = YAML::LoadFile(path.string());
    YAML::Node node = root["n2v2_preprocess"] ? root["n2v2_preprocess"] : root;

    PreprocessConfig config;
    if (node["enable_network"]) config.enableNetwork = node["enable_network"].as<bool>();
    if (node["model_path"]) config.modelPath = node["model_path"].as<std::string>();
    if (node["device"]) config.device = node["device"].as<std::string>();
    config.inferenceBatchSize = requirePositiveInt(node["inference_batch_size"],
                                                   "inference_batch_size",
                                                   config.inferenceBatchSize);
    config.tileSize = readIntPair(node["tile_size"], "tile_size", config.tileSize);
    config.tileOverlap = readIntPair(node["tile_overlap"], "tile_overlap", config.tileOverlap);
    if (config.tileOverlap[0] >= config.tileSize[0] || config.tileOverlap[1] >= config.tileSize[1])
    {
        throw std::invalid_argument("tile_overlap values must be smaller than tile_size values");
    }

    if (node["scale_percentile"]) config.scalePercentile = node["scale_percentile"].as<double>();
    if (node["use_nonzero_pixels"]) config.useNonzeroPixels = node["use_nonzero_pixels"].as<bool>();
    if (node["fallback_scale"]) config.fallbackScale = node["fallback_scale"].as<double>();
    if (node["careamics_mean"]) config.careamicsMean = node["careamics_mean"].as<double>();
    if (node["careamics_std"]) config.careamicsStd = node["careamics_std"].as<double>();
    if (config.scalePercentile <= 0.0 || config.scalePercentile > 100.0)
    {
        throw std::invalid_argument("scale_percentile must be in (0, 100]");
    }
    if (config.careamicsStd <= 0.0)
    {
        throw std::invalid_argument("careamics_std must be positive");
    }

    YAML::Node bg = node["background_subtraction"];
    if (bg)
    {
        if (bg["enabled"]) config.backgroundSubtraction.enabled = bg["enabled"].as<bool>();
        if (bg["percentile"]) config.backgroundSubtraction.percentile = bg["percentile"].as<double>();
        if (bg["exclude_zero"]) config.backgroundSubtraction.excludeZero = bg["exclude_zero"].as<bool>();
        if (bg["clip_min"]) config.backgroundSubtraction.clipMin = bg["clip_min"].as<double>();
    }
    if (config.backgroundSubtraction.percentile < 0.0 ||
        config.backgroundSubtraction.percentile > 100.0)
    {
        throw std::invalid_argument("background_subtraction.percentile must be in [0, 100]");
    }

    YAML::Node contrast = node["contrast"];
    if (contrast)
    {
        if (contrast["enabled"]) config.contrast.enabled = contrast["enabled"].as<bool>();
        if (contrast["limit_mode"]) config.contrast.limitMode =
            parseContrastLimitMode(contrast["limit_mode"].as<std::string>());
        if (contrast["low_limit"]) config.contrast.lowLimit = contrast["low_limit"].as<double>();
        if (contrast["high_limit"] && !contrast["high_limit"].IsNull())
        {
            config.contrast.highLimit = contrast["high_limit"].as<double>();
            config.contrast.hasHighLimit = true;
        }
        if (contrast["low_percentile"]) config.contrast.lowPercentile = contrast["low_percentile"].as<double>();
        if (contrast["high_percentile"]) config.contrast.highPercentile = contrast["high_percentile"].as<double>();
        if (contrast["exclude_zero"]) config.contrast.excludeZero = contrast["exclude_zero"].as<bool>();
        if (contrast["scope"]) config.contrast.scope = parseContrastScope(contrast["scope"].as<std::string>());
        if (contrast["gamma"]) config.contrast.gamma = contrast["gamma"].as<double>();
        if (contrast["preserve_zero_pixels"]) config.contrast.preserveZeroPixels =
            contrast["preserve_zero_pixels"].as<bool>();
    }
    if (config.contrast.lowPercentile < 0.0 || config.contrast.lowPercentile > 100.0 ||
        config.contrast.highPercentile < 0.0 || config.contrast.highPercentile > 100.0 ||
        config.contrast.lowPercentile >= config.contrast.highPercentile)
    {
        throw std::invalid_argument("contrast percentiles must satisfy 0 <= low < high <= 100");
    }
    if (config.contrast.gamma <= 0.0)
    {
        throw std::invalid_argument("contrast.gamma must be positive");
    }

    YAML::Node output = node["output"];
    if (output)
    {
        if (output["dtype"]) config.output.dtype = parseOutputDType(output["dtype"].as<std::string>());
        if (output["write_intermediate"]) config.output.writeIntermediate =
            output["write_intermediate"].as<bool>();
        if (output["quantize_before_contrast"]) config.output.quantizeBeforeContrast =
            output["quantize_before_contrast"].as<bool>();
    }

    return config;
}

std::vector<fs::path> resolveInputFrames(const std::string &input,
                                         int firstFrame,
                                         int lastFrame)
{
    if (firstFrame < 0 || lastFrame < firstFrame)
    {
        throw std::invalid_argument("Invalid frame range");
    }

    std::vector<fs::path> paths;
    if (input.find('%') != std::string::npos)
    {
        for (int frame = firstFrame; frame <= lastFrame; ++frame)
        {
            char buffer[4096];
            std::snprintf(buffer, sizeof(buffer), input.c_str(), frame);
            fs::path path(buffer);
            if (!fs::exists(path) || !fs::is_regular_file(path))
            {
                throw std::runtime_error("Input file not found: " + path.string());
            }
            if (!shouldSkipPath(path) && isTiffPath(path))
            {
                paths.push_back(path);
            }
        }
        return paths;
    }

    fs::path inputPath(input);
    if (fs::is_regular_file(inputPath))
    {
        if (!isTiffPath(inputPath) || shouldSkipPath(inputPath))
        {
            throw std::runtime_error("Input file is not a supported TIFF: " + inputPath.string());
        }
        paths.push_back(inputPath);
        return paths;
    }

    if (!fs::is_directory(inputPath))
    {
        throw std::runtime_error("Input path does not exist: " + inputPath.string());
    }

    for (const auto &entry : fs::directory_iterator(inputPath))
    {
        const fs::path path = entry.path();
        if (entry.is_regular_file() && isTiffPath(path) && !shouldSkipPath(path))
        {
            paths.push_back(path);
        }
    }
    std::sort(paths.begin(), paths.end());
    if (paths.empty())
    {
        throw std::runtime_error("No TIFF files found in input folder: " + inputPath.string());
    }
    if (firstFrame >= static_cast<int>(paths.size()))
    {
        throw std::out_of_range("firstFrame is outside the available TIFF list");
    }
    const int clampedLast = std::min(lastFrame, static_cast<int>(paths.size()) - 1);
    return std::vector<fs::path>(
        paths.begin() + firstFrame,
        paths.begin() + clampedLast + 1);
}

std::vector<cv::Mat> loadTiffStack(const fs::path &path)
{
    std::vector<cv::Mat> loaded;
    if (!cv::imreadmulti(path.string(), loaded, cv::IMREAD_ANYDEPTH | cv::IMREAD_GRAYSCALE))
    {
        throw std::runtime_error("Failed to read TIFF stack: " + path.string());
    }
    if (loaded.empty())
    {
        throw std::runtime_error("TIFF stack has zero slices: " + path.string());
    }

    std::vector<cv::Mat> stack;
    stack.reserve(loaded.size());
    cv::Size expectedSize;
    for (const auto &slice : loaded)
    {
        cv::Mat gray = toGray(slice);
        if (expectedSize.empty())
        {
            expectedSize = gray.size();
        }
        else if (gray.size() != expectedSize)
        {
            throw std::runtime_error("All TIFF slices must share the same dimensions: " + path.string());
        }
        stack.push_back(std::move(gray));
    }
    return stack;
}

void writeTiffStack(const fs::path &path, const std::vector<cv::Mat> &stack)
{
    if (stack.empty())
    {
        throw std::runtime_error("Cannot write an empty TIFF stack");
    }
    if (!path.parent_path().empty())
    {
        fs::create_directories(path.parent_path());
    }
    const std::vector<int> params = {
        cv::IMWRITE_TIFF_COMPRESSION, 1,
    };
    if (!cv::imwritemulti(path.string(), stack, params))
    {
        throw std::runtime_error("Failed to write TIFF stack: " + path.string());
    }
}

void writeSummaryCsv(const fs::path &path, const std::vector<FrameSummary> &summaries)
{
    fs::create_directories(path.parent_path());
    std::ofstream out(path);
    if (!out)
    {
        throw std::runtime_error("Failed to write summary CSV: " + path.string());
    }
    out << "input,output,intermediate,shape,input_type,output_type,network_enabled,"
           "scale,background_enabled,background_value,contrast_enabled,contrast_low,"
           "contrast_high,input_nonzero,output_nonzero\n";
    out << std::setprecision(12);
    for (const auto &item : summaries)
    {
        out << csvEscape(item.inputPath) << ','
            << csvEscape(item.outputPath) << ','
            << csvEscape(item.intermediatePath) << ','
            << item.z << 'x' << item.y << 'x' << item.x << ','
            << cvTypeName(item.inputCvType) << ','
            << cvTypeName(item.outputCvType) << ','
            << (item.networkEnabled ? "true" : "false") << ','
            << item.scale << ','
            << (item.backgroundEnabled ? "true" : "false") << ','
            << item.backgroundValue << ','
            << (item.contrastEnabled ? "true" : "false") << ','
            << item.contrastLow << ','
            << item.contrastHigh << ','
            << item.inputNonzero << ','
            << item.outputNonzero << '\n';
    }
}

std::string cvTypeName(int type)
{
    const int depth = CV_MAT_DEPTH(type);
    const int channels = CV_MAT_CN(type);
    std::string name;
    switch (depth)
    {
    case CV_8U: name = "uint8"; break;
    case CV_8S: name = "int8"; break;
    case CV_16U: name = "uint16"; break;
    case CV_16S: name = "int16"; break;
    case CV_32S: name = "int32"; break;
    case CV_32F: name = "float32"; break;
    case CV_64F: name = "float64"; break;
    default: name = "unknown"; break;
    }
    if (channels > 1)
    {
        name += "x" + std::to_string(channels);
    }
    return name;
}

double percentileLinear(std::vector<float> values, double percentile)
{
    if (values.empty())
    {
        return 0.0;
    }
    percentile = std::clamp(percentile, 0.0, 100.0);
    std::sort(values.begin(), values.end());
    const double position = (percentile / 100.0) * static_cast<double>(values.size() - 1);
    const auto lowerIndex = static_cast<std::size_t>(std::floor(position));
    const auto upperIndex = static_cast<std::size_t>(std::ceil(position));
    const double fraction = position - static_cast<double>(lowerIndex);
    const double lower = static_cast<double>(values[lowerIndex]);
    const double upper = static_cast<double>(values[upperIndex]);
    return lower + (upper - lower) * fraction;
}

std::vector<int> computeTileStartsForAxis(int axisSize, int patchSize, int overlap)
{
    const std::vector<AxisTile> tiles = computeAxisTiles(axisSize, patchSize, overlap);
    std::vector<int> starts;
    starts.reserve(tiles.size());
    for (const auto &tile : tiles)
    {
        starts.push_back(tile.coord);
    }
    return starts;
}

} // namespace n2v2
