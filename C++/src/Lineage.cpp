#include "../includes/Lineage.hpp"

namespace {
double maxInCurrentDynamicRange(int depth)
{
    switch (depth) {
        case CV_8U:  return 255.0;
        case CV_8S:  return 127.0;
        case CV_16U: return 65535.0;
        case CV_16S: return 32767.0;
        case CV_32S: return 2147483647.0;
        case CV_32F:
        case CV_64F:
            return 1.0;
        default:
            return 255.0;
    }
}

double computeAverageSliceMax(const std::vector<cv::Mat> &slices)
{
    if (slices.empty()) {
        return 255.0;
    }

    std::vector<double> sliceMaxima;
    sliceMaxima.reserve(slices.size());
    double globalMax = 0.0;
    for (const auto &slice : slices) {
        if (slice.empty()) {
            continue;
        }
        double maxValue = 0.0;
        cv::minMaxLoc(slice, nullptr, &maxValue);
        sliceMaxima.push_back(maxValue);
        globalMax = std::max(globalMax, maxValue);
    }

    if (sliceMaxima.empty()) {
        return 255.0;
    }

    const double informativeThreshold = std::max(1.0, globalMax * 0.10);
    std::vector<double> informativeMaxima;
    informativeMaxima.reserve(sliceMaxima.size());
    for (double m : sliceMaxima) {
        if (m >= informativeThreshold) {
            informativeMaxima.push_back(m);
        }
    }
    if (informativeMaxima.empty()) {
        informativeMaxima = sliceMaxima;
    }

    std::sort(informativeMaxima.begin(), informativeMaxima.end());
    const size_t mid = informativeMaxima.size() / 2;
    double medianMax = informativeMaxima[mid];
    if (informativeMaxima.size() % 2 == 0 && informativeMaxima.size() > 1) {
        medianMax = 0.5 * (informativeMaxima[mid - 1] + informativeMaxima[mid]);
    }

    const double scaleFloor = std::max(1.0, globalMax * 0.05);
    const double robustScale = std::max(medianMax, scaleFloor);
    return (robustScale > 0.0) ? robustScale : 255.0;
}

bool hasCalibrationZone(const SimulationConfig &sim)
{
    return sim.calibration_width > 0 && sim.calibration_height > 0 &&
           sim.calibration_x >= 0 && sim.calibration_y >= 0 && sim.calibration_z >= 0;
}

double computeCalibrationZoneMean(const std::vector<cv::Mat> &slices, const SimulationConfig &sim)
{
    if (!hasCalibrationZone(sim) || slices.empty()) {
        return -1.0;
    }

    const int nSlices = static_cast<int>(slices.size());
    const int z0 = std::clamp(sim.calibration_z, 0, nSlices - 1);
    const int zCount = std::max(1, sim.calibration_height);
    const int z1 = std::min(nSlices, z0 + zCount);

    double sum = 0.0;
    long long count = 0;
    for (int z = z0; z < z1; ++z) {
        const cv::Mat &slice = slices[z];
        if (slice.empty()) {
            continue;
        }

        const int x0 = std::max(0, sim.calibration_x);
        const int y0 = std::max(0, sim.calibration_y);
        const int x1 = std::min(slice.cols, sim.calibration_x + sim.calibration_width);
        const int y1 = std::min(slice.rows, sim.calibration_y + sim.calibration_width);
        if (x0 >= x1 || y0 >= y1) {
            continue;
        }

        const cv::Rect roi(x0, y0, x1 - x0, y1 - y0);
        const double roiMean = cv::mean(slice(roi))[0];
        const long long roiPixels = static_cast<long long>(roi.width) * roi.height;
        sum += roiMean * static_cast<double>(roiPixels);
        count += roiPixels;
    }

    if (count == 0) {
        return -1.0;
    }
    return sum / static_cast<double>(count);
}

double computeCalibrationZoneMeanAtZIndices(const std::vector<cv::Mat> &slices,
                                            const SimulationConfig &sim,
                                            const std::vector<int> &zIndices)
{
    if (!hasCalibrationZone(sim) || slices.empty() || zIndices.empty()) {
        return -1.0;
    }

    double sum = 0.0;
    long long count = 0;
    for (int z : zIndices) {
        if (z < 0 || z >= static_cast<int>(slices.size())) {
            continue;
        }
        const cv::Mat &slice = slices[z];
        if (slice.empty()) {
            continue;
        }

        const int x0 = std::max(0, sim.calibration_x);
        const int y0 = std::max(0, sim.calibration_y);
        const int x1 = std::min(slice.cols, sim.calibration_x + sim.calibration_width);
        const int y1 = std::min(slice.rows, sim.calibration_y + sim.calibration_width);
        if (x0 >= x1 || y0 >= y1) {
            continue;
        }

        const cv::Rect roi(x0, y0, x1 - x0, y1 - y0);
        const double roiMean = cv::mean(slice(roi))[0];
        const long long roiPixels = static_cast<long long>(roi.width) * roi.height;
        sum += roiMean * static_cast<double>(roiPixels);
        count += roiPixels;
    }

    if (count == 0) {
        return -1.0;
    }
    return sum / static_cast<double>(count);
}

double computeStackMean(const std::vector<cv::Mat> &slices)
{
    if (slices.empty()) {
        return 0.0;
    }
    double sum = 0.0;
    long long count = 0;
    for (const auto &slice : slices) {
        if (slice.empty()) {
            continue;
        }
        const double sliceMean = cv::mean(slice)[0];
        const long long pixels = static_cast<long long>(slice.rows) * slice.cols;
        sum += sliceMean * static_cast<double>(pixels);
        count += pixels;
    }
    if (count == 0) {
        return 0.0;
    }
    return sum / static_cast<double>(count);
}

void rescaleStack(std::vector<cv::Mat> &slices, double invScale)
{
    for (auto &slice : slices) {
        if (slice.empty()) {
            continue;
        }
        slice *= invScale;
        cv::min(slice, 1.0f, slice);
        cv::max(slice, 0.0f, slice);
    }
}
} // namespace

namespace utils
{
    template <typename T>
    void printMat(const cv::Mat &mat)
    {
        // Check if the matrix is empty
        if (mat.empty())
        {
            std::cout << "The matrix is empty." << std::endl;
            return;
        }

        // Iterate over matrix rows
        for (int i = 0; i < mat.rows; ++i)
        {
            for (int j = 0; j < mat.cols; ++j)
            {
                // Print each element. Use mat.at<T>(i,j) to access the element at [i,j] and cast it to the type T.
                // The type T should match the type of the elements in the matrix.
                std::cout << mat.at<T>(i, j) << " ";
            }
            std::cout << std::endl; // Newline for each row
        }
    }
    void applySigmoid(cv::Mat& img, float k = 10.f, float c = 0.5f)
    {
        // img.convertTo(img, CV_32F, 1.0 / 255.0);

        cv::Mat tmp = -k * (img - c);
        cv::exp(tmp, tmp);
        img = 1.0f / (1.0f + tmp);
    }
}

Image processImage(const Image &image, const BaseConfig &config, double scaleFactor, float sigmoidCenterOverride)
{
    Image processedImage;

    if (image.channels() == 3)
    {
        cv::cvtColor(image, processedImage, cv::COLOR_RGB2GRAY);
    }
    else
    {
        processedImage = image.clone();
    }

    // double maxValue = 0.0;
    // cv::minMaxLoc(processedImage, nullptr, &maxValue);
    // const double dynamicMax = maxInCurrentDynamicRange(processedImage.depth());
    // const double amplifyRatio = (dynamicMax > 0.0) ? (dynamicMax/maxValue) : 1.0;
    // const double scale = (amplifyRatio > 0.0) ? (1.0 / amplifyRatio) : 1.0;

    const double safeScale = (scaleFactor > 0.0) ? scaleFactor : 255.0;
    // std::cout << scaleFactor << std::endl;
    processedImage.convertTo(processedImage, CV_32F, 1.0 / safeScale);
    cv::min(processedImage, 1.0f, processedImage);
    SimulationConfig simConfig = config.simulation;
    const double blurSigma = (simConfig.blur_sigma > 0.0f) ? simConfig.blur_sigma : 1.5;
    cv::GaussianBlur(processedImage, processedImage, cv::Size(0, 0), blurSigma);
    const float sigmoidK = (simConfig.sigmoid_k > 0.0f) ? simConfig.sigmoid_k : 30.0f;
    float sigmoidCenter =
        (simConfig.sigmoid_center >= 0.0f) ? simConfig.sigmoid_center : config.simulation.background_color;
    if (sigmoidCenterOverride >= 0.0f) {
        sigmoidCenter = sigmoidCenterOverride;
    }
    sigmoidCenter += simConfig.sigmoid_center_offset;
    utils::applySigmoid(processedImage, sigmoidK, sigmoidCenter);

    return processedImage;
}

std::vector<cv::Mat> loadFrame(const std::string &imageFile, const BaseConfig &config, float *frameBackgroundOut)
{
    std::vector<cv::Mat> processedZSlices; // vector of matrices, each matrix is a 2D image
    std::vector<cv::Mat> interpolatedZSlices;
    float frameBackground = config.simulation.background_color;

    // Get the file extension
    std::string extension = imageFile.substr(imageFile.find_last_of('.') + 1);
    if (extension == "tiff" || extension == "tif")
    {
        std::vector<cv::Mat> tiffImage;
        cv::imreadmulti(imageFile, tiffImage, cv::IMREAD_ANYDEPTH | cv::IMREAD_COLOR);

        long unsigned numTiffSlices {tiffImage.size()};
        // assert(numTiffSlices == 33); // REMOVE: dataset-dependent, CE embryo has 249 slices
        if (numTiffSlices == 0) {
            throw std::runtime_error("TIFF has 0 slices: " + imageFile);
        }

        cv::Mat img = tiffImage[0];

        if (img.empty())
        {
            std::cout << "Error: Could not read the TIFF image" << std::endl;
            return processedZSlices;
        }

        // Convert to grayscale first, but defer preprocessing until after z-interpolation.
        std::vector<cv::Mat> grayZSlices;
        grayZSlices.reserve(numTiffSlices);
        for (unsigned i = 0; i < numTiffSlices; ++i)
        {
            cv::Mat graySlice;
            cv::cvtColor(tiffImage[i], graySlice, cv::COLOR_BGR2GRAY);
            grayZSlices.push_back(graySlice);
        }

        const int expandFactor = config.simulation.z_scaling; 
        // there will be (expandFactor-1) interpolated slices between each "real" one.
        // we need one extra at the very top to hold the top "real" z-Slice.

        unsigned numSynthSlices = expandFactor * (numTiffSlices-1) + 1; // 225 for 33 slices

        // for checking
        // std::cout << "Number of synthetic slices: " << numSynthSlices << std::endl;
        
        // Interpolate grayscale slices first.
        std::vector<cv::Mat> interpolatedGrayZSlices;
        interpolatedGrayZSlices.reserve(numSynthSlices);
        for (int synthSlice = 0; synthSlice < numSynthSlices; ++synthSlice) {
            int tiffSlice = int(synthSlice / expandFactor); // "real" slice index
            if (synthSlice % expandFactor == 0) {
                interpolatedGrayZSlices.push_back(grayZSlices[tiffSlice]);
            } else if (synthSlice % expandFactor == 1) {
                interpolateSlices(grayZSlices[tiffSlice],
                                  grayZSlices[tiffSlice + 1],
                                  interpolatedGrayZSlices,
                                  expandFactor - 1);
            }
        }

        const double avgSliceMax = computeAverageSliceMax(interpolatedGrayZSlices);
        // Calibration z-indexing is defined on the original (pre-interpolation) stack.
        const double rawCalibrationBg = computeCalibrationZoneMean(grayZSlices, config.simulation);
        if (rawCalibrationBg >= 0.0) {
            frameBackground = static_cast<float>(std::clamp(rawCalibrationBg / avgSliceMax, 0.0, 1.0));
            std::cout << "[Calibration] raw_mean=" << rawCalibrationBg
                      << " scale=" << avgSliceMax
                      << " sigmoid_center=" << frameBackground << std::endl;
        } else {
            std::cout << "[Calibration] no valid raw calibration voxels; fallback center="
                      << frameBackground << std::endl;
        }
        const float sigmoidCenter = frameBackground;
        std::cout << "[Sigmoid] center_used=" << (sigmoidCenter + config.simulation.sigmoid_center_offset)
                  << " (from_calibration_center=" << sigmoidCenter
                  << ", config_center=" << config.simulation.sigmoid_center
                  << ", center_offset=" << config.simulation.sigmoid_center_offset << ")" << std::endl;

        // Apply preprocessing on the full interpolated stack.
        processedZSlices.reserve(interpolatedGrayZSlices.size());
        for (const auto &slice : interpolatedGrayZSlices) {
            processedZSlices.push_back(processImage(slice, config, avgSliceMax, sigmoidCenter));
        }

        // Update frame background baseline from processed calibration area.
        std::vector<int> processedCalibZ;
        if (hasCalibrationZone(config.simulation)) {
            const int z0 = std::max(0, config.simulation.calibration_z);
            const int z1 = std::min(static_cast<int>(grayZSlices.size()),
                                    config.simulation.calibration_z + std::max(1, config.simulation.calibration_height));
            processedCalibZ.reserve(std::max(0, z1 - z0));
            for (int z = z0; z < z1; ++z) {
                processedCalibZ.push_back(z * expandFactor);
            }
        }
        const double processedCalibrationBg =
            computeCalibrationZoneMeanAtZIndices(processedZSlices, config.simulation, processedCalibZ);
        if (processedCalibrationBg >= 0.0) {
            frameBackground = static_cast<float>(std::clamp(processedCalibrationBg, 0.0, 1.0));
            std::cout << "[Calibration] processed_background=" << frameBackground << std::endl;
        }
        interpolatedZSlices = std::move(processedZSlices);

        // [PATCH] Validate synthetic slice count (only for TIFF branch)
        if (interpolatedZSlices.size() != numSynthSlices) {
            std::string errorMessage =
                "interpolatedZSlices must have exactly " + std::to_string(numSynthSlices) +
                " slices, but has " + std::to_string(interpolatedZSlices.size()) + " slices";
            throw std::runtime_error(errorMessage);
        }
    }
    else
    {
        // TODO: fix this
        cv::Mat img = cv::imread(imageFile);
        if (img.empty())
        {
            std::cout << "Error: Could not read the image" << std::endl;
            return processedZSlices;
        }

        if (img.channels() == 3)
        {
            cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
        }

        const std::vector<cv::Mat> singleSliceStack{img};
        const double avgSliceMax = computeAverageSliceMax(singleSliceStack);
        const double rawCalibrationBg = computeCalibrationZoneMean(singleSliceStack, config.simulation);
        if (rawCalibrationBg >= 0.0) {
            frameBackground = static_cast<float>(std::clamp(rawCalibrationBg / avgSliceMax, 0.0, 1.0));
            std::cout << "[Calibration] raw_mean=" << rawCalibrationBg
                      << " scale=" << avgSliceMax
                      << " sigmoid_center=" << frameBackground << std::endl;
        } else {
            std::cout << "[Calibration] no valid raw calibration voxels; fallback center="
                      << frameBackground << std::endl;
        }
        std::cout << "[Sigmoid] center_used=" << (frameBackground + config.simulation.sigmoid_center_offset)
                  << " (from_calibration_center=" << frameBackground
                  << ", config_center=" << config.simulation.sigmoid_center
                  << ", center_offset=" << config.simulation.sigmoid_center_offset << ")" << std::endl;
        processedZSlices.push_back(processImage(img, config, avgSliceMax, frameBackground));
        const double processedCalibrationBg = computeCalibrationZoneMean(processedZSlices, config.simulation);
        if (processedCalibrationBg >= 0.0) {
            frameBackground = static_cast<float>(std::clamp(processedCalibrationBg, 0.0, 1.0));
            std::cout << "[Calibration] processed_background=" << frameBackground << std::endl;
        }
        interpolatedZSlices = std::move(processedZSlices);
    }

    if (frameBackgroundOut != nullptr) {
        *frameBackgroundOut = frameBackground;
    }
    std::cout << std::to_string(interpolatedZSlices.size()) << "slices built successfully" << std::endl;
    return interpolatedZSlices;
}


Lineage::Lineage(std::map<std::string, std::vector<Spheroid>> initialCells,
                 PathVec imagePaths,
                 BaseConfig &config,
                 std::string outputPath,
                 int firstFrame,
                 int continueFrom)
: config(config), outputPath(outputPath), firstFrame(firstFrame)
{
    double prevFrameMean = -1.0;
    for (size_t i = 0; i < imagePaths.size(); ++i)
    {
        std::vector<Image> real_frame;
        float frameBackground = config.simulation.background_color;
        real_frame = loadFrame(imagePaths[i], config, &frameBackground);

        const double currentMeanBeforeAlign = computeStackMean(real_frame);
        if (prevFrameMean > 1e-9 && currentMeanBeforeAlign > 1e-9) {
            const double brightnessRatio = currentMeanBeforeAlign / prevFrameMean;
            if (brightnessRatio > 1e-6) {
                const double invScale = 1.0 / brightnessRatio;
                if (std::abs(brightnessRatio - 1.0) > 0.01) {
                    std::cout << "[Frame Align] frame_index=" << i
                              << " ratio=" << brightnessRatio
                              << " apply_scale=" << invScale << std::endl;
                    rescaleStack(real_frame, invScale);
                    frameBackground = static_cast<float>(std::clamp(frameBackground * invScale, 0.0, 1.0));
                }
            }
        }
        prevFrameMean = computeStackMean(real_frame);

        fs::path path(imagePaths[i]);
        //        std::cout << "Filename: " << path.filename() << std::endl;
        std::string file_name = path.filename();
        SimulationConfig frameSimConfig = config.simulation;
        frameSimConfig.z_slices = static_cast<int>(real_frame.size());
        frameSimConfig.background_color = frameBackground;

        if ((continueFrom == -1 || i < continueFrom) && initialCells.find(file_name) != initialCells.end())
        {
            const std::vector<Spheroid> &cells = initialCells.at(file_name);
            frames.emplace_back(real_frame, frameSimConfig, cells, outputPath, file_name);
        }
        else
        {
            frames.emplace_back(real_frame, frameSimConfig, std::vector<Spheroid>(), outputPath, file_name);
        }
    }
}
void Lineage::optimize(int frameIndex)
{
    if (frameIndex < 0 || static_cast<size_t>(frameIndex) >= frames.size())
    {
        throw std::invalid_argument("Invalid frame index");
    }

    Frame &frame = frames[frameIndex];
    size_t totalIterations = frame.length() * config.simulation.iterations_per_cell;
    std::cout << "Total iterations: " << totalIterations << std::endl;

    Cost costDiff = 0;
    double residSum = 0;
    double residCount = 0;
    double ovrResidual = 0;

    // Save pre-optimization cell shapes before Phase 1.
    // Phase 1 may collapse elongated/pancaked cells to spheres, shrinking the
    // search area for split detection. We preserve the original radii so the
    // PCA search boundary doesn't shrink.
    struct PreOptShape {
        float majorR;
        float minorR;
    };
    std::map<std::string, PreOptShape> preOptShapes;
    for (const auto &cell : frame.cells) {
        auto params = cell.getCellParams();
        preOptShapes[params.name] = {(float)params.majorRadius, (float)params.minorRadius};
    }

    // ============================================================
    // Phase 1: Perturbation-only optimization
    // Settle all existing cells into their best positions first.
    // ============================================================
    int displayFrame = firstFrame + frameIndex;
    std::cout << "[Phase 1] Perturbation optimization for frame " << displayFrame
              << " (" << frame.cells.size() << " cells, " << totalIterations << " iterations)" << std::endl;

    for (size_t i = 0; i < totalIterations; ++i) {
        if (costDiff < 0) {
            residSum += costDiff;
            residCount++;
        }
        if (i % 100 == 0) {
            ovrResidual = residSum / residCount;
            if (residCount > 0) {
                std::cout << "Frame " << displayFrame << ", iteration " << i
                          << " Difference of Residuals " << ovrResidual << std::endl;
            } else {
                std::cout << "Frame " << displayFrame << ", iteration " << i
                          << " -- No synthezised images selected" << std::endl;
            }
            residSum = 0;
            residCount = 0;
        }

        auto result = frame.perturb();
        costDiff = result.first;
        std::function<void(bool)> accept = result.second;
        accept(costDiff < 0);
    }

    // ============================================================
    // Phase 2: Post-optimization split detection
    // Greedy acceptance: evaluate candidates from current baseline,
    // apply only the best one, then recompute and repeat.
    // This avoids accepting multiple overlapping candidates that each
    // looked good only against an outdated baseline.
    // ============================================================
    const size_t initialCellCount = frame.cells.size();
    std::cout << "[Phase 2] Split detection for frame " << displayFrame
              << " (" << initialCellCount << " cells)" << std::endl;

    const double splitAttemptProb = std::clamp(static_cast<double>(config.prob.split), 0.0, 1.0);
    const double minRelativeSplitGain = 0.001; // require >=0.1% relative cost reduction
    const size_t maxSplitsThisFrame = std::max<size_t>(1, initialCellCount);
    std::mt19937 splitRng(std::random_device{}());
    std::uniform_real_distribution<double> u01(0.0, 1.0);

    struct SplitCandidate {
        std::string parentName;
        Spheroid child1;
        Spheroid child2;
        double costDiff;
        double relGain;
    };
    size_t splitsApplied = 0;
    for (size_t splitRound = 0; splitRound < maxSplitsThisFrame; ++splitRound) {
        if (splitAttemptProb <= 0.0) {
            break;
        }

        const double baselineCost = frame.calculateCost(frame.getSynthFrame());
        SplitCandidate bestCandidate;
        bool hasBest = false;

        std::vector<std::string> cellNames;
        cellNames.reserve(frame.cells.size());
        for (const auto &cell : frame.cells) {
            cellNames.push_back(cell.getCellParams().name);
        }

        for (const auto &name : cellNames) {
            if (splitAttemptProb < 1.0 && u01(splitRng) > splitAttemptProb) {
                continue;
            }

            size_t idx = SIZE_MAX;
            for (size_t j = 0; j < frame.cells.size(); ++j) {
                if (frame.cells[j].getCellParams().name == name) {
                    idx = j;
                    break;
                }
            }
            if (idx == SIZE_MAX) continue;

            float preOptMajorR = 0.0f, preOptMinorR = 0.0f;
            auto it = preOptShapes.find(name);
            if (it != preOptShapes.end()) {
                preOptMajorR = it->second.majorR;
                preOptMinorR = it->second.minorR;
            }

            auto result = frame.trySplitCell(idx, preOptMajorR, preOptMinorR);
            costDiff = result.first;
            std::function<void(bool)> callback = result.second;

            const double relGain = (baselineCost > 1e-9) ? (-costDiff / baselineCost) : 0.0;
            const bool passAbsGate = costDiff < -config.prob.split_cost;
            const bool passRelGate = relGain >= minRelativeSplitGain;

            if (passAbsGate && passRelGate) {
                Spheroid child1 = frame.cells[frame.cells.size() - 2];
                Spheroid child2 = frame.cells[frame.cells.size() - 1];

                if (!hasBest || costDiff < bestCandidate.costDiff) {
                    bestCandidate = {name, child1, child2, costDiff, relGain};
                    hasBest = true;
                }
            }

            // Always revert this trial candidate.
            callback(false);
        }

        if (!hasBest) {
            break;
        }

        size_t idx = SIZE_MAX;
        for (size_t j = 0; j < frame.cells.size(); ++j) {
            if (frame.cells[j].getCellParams().name == bestCandidate.parentName) {
                idx = j;
                break;
            }
        }
        if (idx == SIZE_MAX) {
            break;
        }

        frame.cells.erase(frame.cells.begin() + idx);
        frame.cells.push_back(bestCandidate.child1);
        frame.cells.push_back(bestCandidate.child2);
        frame.regenerateSynthFrame();
        splitsApplied++;

        std::cout << "[Split Accepted] " << bestCandidate.parentName << " split in frame "
                  << displayFrame << " (diff=" << bestCandidate.costDiff
                  << ", rel_gain=" << bestCandidate.relGain << ")" << std::endl;
    }

    // Regenerate synth frame and run post-split perturbation
    if (splitsApplied > 0) {
        size_t postSplitIters = 2 * config.simulation.iterations_per_cell * splitsApplied;
        for (size_t j = 0; j < postSplitIters; ++j) {
            auto presult = frame.perturb();
            presult.second(presult.first < 0);
        }
    }
}

void Lineage::saveImages(int frameIndex)
{
    if (frameIndex < 0 || static_cast<size_t>(frameIndex) >= frames.size())
    {
        throw std::invalid_argument("Invalid frame index");
    }

    std::vector<Image> realImages = frames[frameIndex].generateOutputFrame();
    std::vector<Image> synthImages = frames[frameIndex].generateOutputSynthFrame();
    int displayFrame = firstFrame + frameIndex;
    std::cout << "Saving images for frame " << displayFrame << "..." << std::endl;
    std::cout << "Real Image Type: " << realImages[0].type() << std::endl;
    std::cout << "Synth Image Type: " << synthImages[0].type() << std::endl;

    std::string realOutputPath = outputPath + "/real/" + std::to_string(displayFrame);
    if (!std::filesystem::exists(realOutputPath))
    {
        std::filesystem::create_directories(realOutputPath);
    }
    for (size_t i = 0; i < realImages.size(); ++i)
    {
        // Save real images
        cv::imwrite(realOutputPath + "/" + std::to_string(i) + ".png", realImages[i]);
    }

    std::string synthOutputPath = outputPath + "/synth/" + std::to_string(displayFrame);
    if (!std::filesystem::exists(synthOutputPath))
    {
        std::filesystem::create_directories(synthOutputPath);
    }
    for (size_t i = 0; i < synthImages.size(); ++i)
    {
        // Save synthetic images
        cv::imwrite(synthOutputPath + "/" + std::to_string(i) + ".png", synthImages[i]);
    }

    std::cout << "Done" << std::endl;
}

void Lineage::saveCells(int frameIndex)
{
    std::string cellsPath = outputPath + "/cells.csv";
    bool fileExists = std::filesystem::exists(cellsPath);

    // Append mode: each frame adds its rows as it finishes optimizing
    std::ofstream file(cellsPath, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << cellsPath << " for writing" << std::endl;
        return;
    }

    // Write header only for the first frame
    if (!fileExists || frameIndex == 0) {
        // Truncate if frame 0 (fresh run)
        if (frameIndex == 0) {
            file.close();
            file.open(cellsPath, std::ios::trunc);
        }
        file << "file,name,x,y,z,majorRadius,minorRadius,theta_x,theta_y,theta_z" << std::endl;
    }

    Frame &frame = frames[frameIndex];
    std::string imageName = frame.getImageName();

    for (const auto &cell : frame.cells) {
        SpheroidParams params = cell.getCellParams();
        // cell.printCellInfo();
        file << imageName << ","
             << params.name << ","
             << params.x << ","
             << params.y << ","
             << params.z << ","
             << params.majorRadius << ","
             << params.minorRadius << ","
             << params.theta_x << ","
             << params.theta_y << ","
             << params.theta_z
             << std::endl;
    }

    std::cout << "Saved " << frame.cells.size() << " cells for frame " << (firstFrame + frameIndex)
              << " to " << cellsPath << std::endl;
}

void Lineage::copyCellsForward(int to)
{
    if (to >= frames.size())
    {
        return;
    }
    // assumes cells have deepcopy copy constructors
    frames[to].cells = frames[to - 1].cells;
}

unsigned int Lineage::length()
{
    return frames.size();
}

const std::vector<Spheroid> &Lineage::getCells(int frameIndex) const
{
    if (frameIndex < 0 || static_cast<size_t>(frameIndex) >= frames.size())
    {
        throw std::invalid_argument("Lineage::getCells - invalid frameIndex");
    }
    return frames[frameIndex].cells;
}

std::vector<std::string> Lineage::getCellNames(int frameIndex) const
{
    const auto &cells = getCells(frameIndex);
    std::vector<std::string> names;
    names.reserve(cells.size());
    for (const auto &c : cells)
    {
        names.push_back(c.getCellParams().name);
    }
    return names;
}
