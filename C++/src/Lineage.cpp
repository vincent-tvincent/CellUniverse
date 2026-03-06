#include "../includes/Lineage.hpp"
#include <set>
#include <cmath>

namespace utils
{
    void applySigmoid(cv::Mat& img, float k, float c);
}

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

std::vector<float> buildBlurFactors(float maxBlurFactor, float step)
{
    const float startFactor = 1.0f; // 1.0 means no extra blur.
    const float endFactor = std::max(startFactor, maxBlurFactor);
    if (endFactor <= startFactor + 1e-6f) {
        return {startFactor};
    }

    const float safeStep = (step > 1e-6f) ? step : (endFactor - startFactor);
    std::vector<float> factors;
    for (float f = startFactor; f < endFactor - 1e-6f; f += safeStep) {
        factors.push_back(f);
    }
    if (factors.empty() || std::abs(factors.back() - endFactor) > 1e-6f) {
        factors.push_back(endFactor);
    }
    if (std::abs(factors.front() - startFactor) > 1e-6f) {
        factors.insert(factors.begin(), startFactor);
    }
    return factors;
}

void applyBlurFactor(Image &image, float blurFactor)
{
    // Blur factor 1.0 is intentionally treated as "no blur".
    const float sigma = std::max(0.0f, blurFactor - 1.0f);
    if (sigma <= 1e-6f) {
        return;
    }
    cv::GaussianBlur(image, image, cv::Size(0, 0), sigma);
}

void applySigmoidToStack(std::vector<cv::Mat> &slices, float sigmoidK, float sigmoidCenterUsed)
{
    for (auto &slice : slices) {
        utils::applySigmoid(slice, sigmoidK, sigmoidCenterUsed);
    }
}

std::vector<cv::Mat> fuseBlurSweepSigmoidStack(const std::vector<cv::Mat> &linearSlices,
                                               const BaseConfig &config,
                                               float frameSigmoidCenter)
{
    if (linearSlices.empty()) {
        return {};
    }

    const float sigmoidK = (config.simulation.sigmoid_k > 0.0f) ? config.simulation.sigmoid_k : 30.0f;
    const float sigmoidCenterUsed = frameSigmoidCenter + config.simulation.sigmoid_center_offset;
    const std::vector<float> blurFactors =
        buildBlurFactors(config.simulation.blur_sigma, config.simulation.blur_sweep_step);

    std::vector<cv::Mat> fusedSlices;
    fusedSlices.reserve(linearSlices.size());
    for (const auto &slice : linearSlices) {
        fusedSlices.push_back(cv::Mat::zeros(slice.size(), CV_32F));
    }

    std::vector<cv::Mat> workingSlices;
    workingSlices.reserve(linearSlices.size());
    for (float blurFactor : blurFactors) {
        workingSlices.clear();
        for (const auto &slice : linearSlices) {
            workingSlices.push_back(slice.clone());
        }

        for (auto &slice : workingSlices) {
            applyBlurFactor(slice, blurFactor);
        }
        applySigmoidToStack(workingSlices, sigmoidK, sigmoidCenterUsed);

        for (size_t i = 0; i < workingSlices.size(); ++i) {
            fusedSlices[i] += workingSlices[i];
        }
    }

    const float invCount = 1.0f / static_cast<float>(blurFactors.size());
    for (auto &slice : fusedSlices) {
        slice *= invCount;
        cv::min(slice, 1.0f, slice);
        cv::max(slice, 0.0f, slice);
    }

    std::cout << "[Blur Sweep] factors=" << blurFactors.size()
              << " range=[1," << std::max(1.0f, config.simulation.blur_sigma) << "]"
              << " step=" << config.simulation.blur_sweep_step << std::endl;
    return fusedSlices;
}

struct LoadedFrameData {
    std::vector<cv::Mat> linearZSlices;
    float sigmoidCenter = 0.0f;
    std::vector<int> calibrationZIndices;
};

constexpr double kPi = 3.14159265358979323846;

struct PreOptShape {
    float majorR;
    float minorR;
    float x;
    float y;
    float z;
};

struct Phase1Result {
    size_t totalIterations = 0;
    size_t acceptedCount = 0;
    std::map<std::string, PreOptShape> preOptShapes;
};

struct SplitCandidate {
    std::string parentName;
    Spheroid child1;
    Spheroid child2;
    double costDiff;
    double relGain;
    double expectedDaughterVolume;
};

struct AcceptedSplit {
    std::string parentName;
    std::string child1Name;
    std::string child2Name;
    double expectedDaughterVolume;
};

double spheroidVolume(const Spheroid &cell)
{
    const auto p = cell.getCellParams();
    return (4.0 / 3.0) * kPi * static_cast<double>(p.majorRadius) *
           static_cast<double>(p.majorRadius) * static_cast<double>(p.minorRadius);
}

double shapeVolume(float majorR, float minorR)
{
    return (4.0 / 3.0) * kPi * static_cast<double>(majorR) *
           static_cast<double>(majorR) * static_cast<double>(minorR);
}

size_t findCellIndexByName(const Frame &frame, const std::string &name)
{
    for (size_t i = 0; i < frame.cells.size(); ++i) {
        if (frame.cells[i].getCellParams().name == name) {
            return i;
        }
    }
    return SIZE_MAX;
}

std::map<std::string, double> collectPreviousVolumes(const std::vector<Frame> &frames, int frameIndex)
{
    std::map<std::string, double> previousVolumes;
    if (frameIndex <= 0) {
        return previousVolumes;
    }

    for (const auto &prevCell : frames[frameIndex - 1].cells) {
        previousVolumes[prevCell.getCellParams().name] = spheroidVolume(prevCell);
    }
    return previousVolumes;
}

std::map<std::string, PreOptShape> snapshotPreOptShapes(const Frame &frame)
{
    std::map<std::string, PreOptShape> preOptShapes;
    for (const auto &cell : frame.cells) {
        const auto params = cell.getCellParams();
        preOptShapes[params.name] = {
            static_cast<float>(params.majorRadius),
            static_cast<float>(params.minorRadius),
            params.x,
            params.y,
            params.z
        };
    }
    return preOptShapes;
}

Phase1Result runPhase1Perturbation(Frame &frame, const BaseConfig &config, int displayFrame)
{
    Phase1Result result;
    result.totalIterations = frame.length() * config.simulation.iterations_per_cell;
    result.preOptShapes = snapshotPreOptShapes(frame);

    std::cout << "Total iterations: " << result.totalIterations << std::endl;
    std::cout << "[Phase 1] Perturbation optimization for frame " << displayFrame
              << " (" << frame.cells.size() << " cells, " << result.totalIterations
              << " iterations)" << std::endl;

    Cost costDiff = 0;
    double residSum = 0;
    double residCount = 0;
    for (size_t i = 0; i < result.totalIterations; ++i) {
        if (costDiff < 0) {
            residSum += costDiff;
            residCount++;
        }
        if (i % 100 == 0) {
            if (residCount > 0) {
                const double avgResidual = residSum / residCount;
                std::cout << "Frame " << displayFrame << ", iteration " << i
                          << " Difference of Residuals " << avgResidual << std::endl;
            } else {
                std::cout << "Frame " << displayFrame << ", iteration " << i
                          << " -- No synthezised images selected" << std::endl;
            }
            residSum = 0;
            residCount = 0;
        }

        auto perturbResult = frame.perturb();
        costDiff = perturbResult.first;
        const bool accepted = (costDiff < 0);
        perturbResult.second(accepted);
        if (accepted) {
            result.acceptedCount++;
        }
    }

    return result;
}

std::vector<SplitCandidate> collectSplitCandidates(Frame &frame,
                                                   const BaseConfig &config,
                                                   const std::map<std::string, PreOptShape> &preOptShapes,
                                                   const Phase1Result &phase1,
                                                   int displayFrame)
{
    const size_t initialCellCount = frame.cells.size();
    std::cout << "[Phase 2] Split detection for frame " << displayFrame
              << " (" << initialCellCount << " cells)" << std::endl;

    const double acceptedFrac =
        (phase1.totalIterations > 0)
            ? (static_cast<double>(phase1.acceptedCount) / static_cast<double>(phase1.totalIterations))
            : 0.0;
    const double minRelativeSplitGain =
        (acceptedFrac < config.prob.phase1_accept_rate_threshold)
            ? config.prob.min_relative_split_gain_strict
            : config.prob.min_relative_split_gain_base;
    std::cout << "[Phase 2 Gate] frame " << displayFrame
              << " phase1_accept_rate=" << acceptedFrac
              << " min_rel_gain=" << minRelativeSplitGain
              << " gate_tol=" << std::max(0.0f, config.prob.split_gate_tolerance) << std::endl;

    std::vector<SplitCandidate> candidates;
    candidates.reserve(initialCellCount);
    const double baselineCost = frame.calculateCost(frame.getSynthFrame());

    std::vector<std::string> cellNames;
    cellNames.reserve(frame.cells.size());
    for (const auto &cell : frame.cells) {
        cellNames.push_back(cell.getCellParams().name);
    }

    for (const auto &name : cellNames) {
        const size_t idx = findCellIndexByName(frame, name);
        if (idx == SIZE_MAX) {
            continue;
        }

        float preOptMajorR = 0.0f, preOptMinorR = 0.0f;
        float preOptX = 0.0f, preOptY = 0.0f, preOptZ = 0.0f;
        auto preIt = preOptShapes.find(name);
        if (preIt != preOptShapes.end()) {
            preOptMajorR = preIt->second.majorR;
            preOptMinorR = preIt->second.minorR;
            preOptX = preIt->second.x;
            preOptY = preIt->second.y;
            preOptZ = preIt->second.z;
        }

        auto splitResult = frame.trySplitCell(
            idx,
            preOptMajorR, preOptMinorR,
            preOptX, preOptY, preOptZ,
            config.prob.split_elongation_threshold,
            config.prob.split_gate_tolerance);
        const double costDiff = splitResult.first;
        std::function<void(bool)> callback = splitResult.second;

        const double gateTol = std::max(0.0, static_cast<double>(config.prob.split_gate_tolerance));
        const double relGain = (baselineCost > 1e-9) ? (-costDiff / baselineCost) : 0.0;
        const bool passAbsGate = costDiff <= (-config.prob.split_cost + gateTol);
        const double strongAbsCutoff = -config.prob.split_cost * config.prob.strong_split_cost_multiplier;
        const auto parentParams = frame.cells[idx].getCellParams();
        const double parentMajor =
            (preOptMajorR > 0.0f) ? preOptMajorR : static_cast<double>(parentParams.majorRadius);
        const double parentMinor =
            (preOptMinorR > 0.0f) ? preOptMinorR : static_cast<double>(parentParams.minorRadius);
        const double parentAspectRatio = (parentMinor > 1e-6) ? (parentMajor / parentMinor) : 0.0;
        const bool passStrongAbsOverride =
            (costDiff <= (strongAbsCutoff + gateTol)) &&
            (parentAspectRatio + gateTol >= config.prob.strong_split_min_aspect_ratio);
        const bool passRelGate = (relGain + gateTol >= minRelativeSplitGain) || passStrongAbsOverride;

        if (passAbsGate && passRelGate) {
            Spheroid child1 = frame.cells[frame.cells.size() - 2];
            Spheroid child2 = frame.cells[frame.cells.size() - 1];
            const double parentVol = (preOptMajorR > 0.0f && preOptMinorR > 0.0f)
                                         ? shapeVolume(preOptMajorR, preOptMinorR)
                                         : spheroidVolume(frame.cells[idx]);
            const double child1Vol = spheroidVolume(child1);
            const double child2Vol = spheroidVolume(child2);
            const double daughterVol = child1Vol + child2Vol;
            const double maxDaughterVol = parentVol * config.prob.max_total_daughter_volume_ratio;
            if (parentVol > 1e-6 && daughterVol > maxDaughterVol) {
                std::cout << "[Split Skip] " << name
                          << " daughter_volume_gate=fail"
                          << " parent_vol=" << parentVol
                          << " daughter_total_vol=" << daughterVol
                          << " max_daughter_total_vol=" << maxDaughterVol
                          << " ratio=" << (daughterVol / parentVol)
                          << std::endl;
            } else {
                candidates.push_back({name, child1, child2, costDiff, relGain, parentVol * 0.5});
            }
        } else {
            std::cout << "[Split Skip] " << name
                      << " absGate=" << (passAbsGate ? "pass" : "fail")
                      << " relGate=" << (passRelGate ? "pass" : "fail")
                      << " strongAbsOverride=" << (passStrongAbsOverride ? "pass" : "fail")
                      << " diff=" << costDiff
                      << " split_cost=" << config.prob.split_cost
                      << " strong_abs_cutoff=" << strongAbsCutoff
                      << " aspect=" << parentAspectRatio
                      << " strong_min_aspect=" << config.prob.strong_split_min_aspect_ratio
                      << " rel_gain=" << relGain
                      << " min_rel_gain=" << minRelativeSplitGain
                      << " gate_tol=" << gateTol
                      << std::endl;
        }

        callback(false);
    }

    return candidates;
}

std::vector<AcceptedSplit> applySplitCandidates(Frame &frame,
                                                const std::vector<SplitCandidate> &candidates,
                                                std::map<std::string, double> &referenceVolumes,
                                                int displayFrame)
{
    std::vector<AcceptedSplit> acceptedSplits;
    acceptedSplits.reserve(candidates.size());
    for (const auto &candidate : candidates) {
        const size_t idx = findCellIndexByName(frame, candidate.parentName);
        if (idx == SIZE_MAX) {
            continue;
        }

        frame.cells.erase(frame.cells.begin() + idx);
        frame.cells.push_back(candidate.child1);
        frame.cells.push_back(candidate.child2);

        acceptedSplits.push_back({
            candidate.parentName,
            candidate.child1.getCellParams().name,
            candidate.child2.getCellParams().name,
            candidate.expectedDaughterVolume
        });

        const auto refIt = referenceVolumes.find(candidate.parentName);
        const double parentRefVolume =
            (refIt != referenceVolumes.end()) ? refIt->second : (candidate.expectedDaughterVolume * 2.0);
        referenceVolumes[candidate.child1.getCellParams().name] = std::max(1.0, parentRefVolume * 0.5);
        referenceVolumes[candidate.child2.getCellParams().name] = std::max(1.0, parentRefVolume * 0.5);

        std::cout << "[Split Accepted] " << candidate.parentName << " split in frame "
                  << displayFrame << " (diff=" << candidate.costDiff
                  << ", rel_gain=" << candidate.relGain << ")" << std::endl;
    }
    return acceptedSplits;
}

void runPostSplitPerturbation(Frame &frame, const BaseConfig &config, size_t splitsApplied)
{
    if (splitsApplied == 0) {
        return;
    }

    frame.regenerateSynthFrame();
    const size_t postSplitIters = 2 * config.simulation.iterations_per_cell * splitsApplied;
    for (size_t j = 0; j < postSplitIters; ++j) {
        auto presult = frame.perturb();
        presult.second(presult.first < 0);
    }
}

void runBrightnessRecovery(Frame &frame,
                           const BaseConfig &config,
                           int displayFrame,
                           const std::map<std::string, double> &previousVolumes,
                           const std::vector<AcceptedSplit> &acceptedSplits)
{
    if (frame.cells.empty()) {
        return;
    }

    const float brightnessFloor = std::clamp(
        frame.getBackgroundColor() + config.prob.brightness_retry_background_margin, 0.0f, 1.0f);
    const int maxAttempts = std::max(0, config.prob.brightness_retry_max_attempts);
    const int itersPerAttempt = std::max(0, config.prob.brightness_retry_iters_per_attempt);
    const float step = std::max(0.001f, config.prob.brightness_retry_step);
    const float minBrightnessRatio = std::clamp(
        config.prob.brightness_retry_min_brightness_ratio, 0.0f, 1.0f);
    if (maxAttempts <= 0) {
        return;
    }

    std::set<std::string> splitChildren;
    for (const auto &s : acceptedSplits) {
        splitChildren.insert(s.child1Name);
        splitChildren.insert(s.child2Name);
    }

    auto runCellRecovery = [&](const std::string &name, double targetVolume, const std::string &reason) {
        if (targetVolume <= 0.0) {
            return;
        }

        size_t idx = findCellIndexByName(frame, name);
        if (idx == SIZE_MAX) {
            return;
        }
        const float initialRecoveryBrightness = frame.cells[idx].getBrightness();

        for (int attempt = 1; attempt <= maxAttempts; ++attempt) {
            idx = findCellIndexByName(frame, name);
            if (idx == SIZE_MAX) {
                return;
            }

            const double currentVolume = spheroidVolume(frame.cells[idx]);
            if (currentVolume >= targetVolume) {
                std::cout << "[Brightness Recovery] frame " << displayFrame
                          << " cell=" << name
                          << " status=success"
                          << " reason=" << reason
                          << " attempts=" << (attempt - 1)
                          << " volume=" << currentVolume
                          << " target=" << targetVolume << std::endl;
                return;
            }

            const float oldBrightness = frame.cells[idx].getBrightness();
            if (oldBrightness <= brightnessFloor + 1e-6f) {
                std::cout << "[Brightness Recovery] frame " << displayFrame
                          << " cell=" << name
                          << " status=stopped_floor"
                          << " reason=" << reason
                          << " attempts=" << (attempt - 1)
                          << " brightness=" << oldBrightness
                          << " floor=" << brightnessFloor
                          << " volume=" << currentVolume
                          << " target=" << targetVolume << std::endl;
                return;
            }

            const float newBrightness = std::max(brightnessFloor, oldBrightness - step);
            if (initialRecoveryBrightness > 1e-6f) {
                const float brightnessRatio = newBrightness / initialRecoveryBrightness;
                if (brightnessRatio + 1e-6f < minBrightnessRatio) {
                    std::cout << "[Brightness Recovery] frame " << displayFrame
                              << " cell=" << name
                              << " status=stopped_ratio"
                              << " reason=" << reason
                              << " attempts=" << (attempt - 1)
                              << " brightness=" << oldBrightness << "->" << newBrightness
                              << " ratio=" << brightnessRatio
                              << " min_ratio=" << minBrightnessRatio
                              << " baseline_brightness=" << initialRecoveryBrightness
                              << " volume=" << currentVolume
                              << " target=" << targetVolume << std::endl;
                    return;
                }
            }
            frame.cells[idx].setBrightness(newBrightness, true);
            frame.regenerateSynthFrame();
            for (int k = 0; k < itersPerAttempt; ++k) {
                auto recoveryPerturb = frame.perturb();
                recoveryPerturb.second(recoveryPerturb.first < 0);
            }

            const double volumeAfter = spheroidVolume(frame.cells[idx]);
            std::cout << "[Brightness Recovery] frame " << displayFrame
                      << " cell=" << name
                      << " reason=" << reason
                      << " attempt=" << attempt
                      << " brightness=" << oldBrightness << "->" << newBrightness
                      << " volume=" << currentVolume << "->" << volumeAfter
                      << " target=" << targetVolume << std::endl;
        }

        idx = findCellIndexByName(frame, name);
        if (idx != SIZE_MAX) {
            const double finalVolume = spheroidVolume(frame.cells[idx]);
            std::cout << "[Brightness Recovery] frame " << displayFrame
                      << " cell=" << name
                      << " status=max_attempts"
                      << " reason=" << reason
                      << " attempts=" << maxAttempts
                      << " final_volume=" << finalVolume
                      << " target=" << targetVolume << std::endl;
        }
    };

    for (const auto &cell : frame.cells) {
        const auto name = cell.getCellParams().name;
        if (splitChildren.find(name) != splitChildren.end()) {
            continue;
        }

        auto prevIt = previousVolumes.find(name);
        if (prevIt == previousVolumes.end()) {
            continue;
        }
        const double prevVolume = prevIt->second;
        if (prevVolume <= 0.0) {
            continue;
        }

        const double curVolume = spheroidVolume(cell);
        if (curVolume < prevVolume * config.prob.nonsplit_shrink_trigger_ratio) {
            const double targetVolume = prevVolume * config.prob.nonsplit_recover_target_ratio;
            runCellRecovery(name, targetVolume, "non_split_shrink");
        }
    }

    for (const auto &splitInfo : acceptedSplits) {
        const double triggerVolume = splitInfo.expectedDaughterVolume * config.prob.split_shrink_trigger_ratio;
        const double targetVolume = splitInfo.expectedDaughterVolume * config.prob.split_recover_target_ratio;
        for (const auto &childName : {splitInfo.child1Name, splitInfo.child2Name}) {
            const size_t idx = findCellIndexByName(frame, childName);
            if (idx == SIZE_MAX) {
                continue;
            }
            const double childVolume = spheroidVolume(frame.cells[idx]);
            if (childVolume < triggerVolume) {
                runCellRecovery(childName, targetVolume, "split_shrink");
            }
        }
    }
}

void enforceVolumeCap(Frame &frame,
                      const BaseConfig &config,
                      std::map<std::string, double> &referenceVolumes,
                      int displayFrame)
{
    const double maxGrowthFactor = std::max(1.0, static_cast<double>(config.prob.max_volume_growth_from_initial));
    for (size_t i = 0; i < frame.cells.size(); ++i) {
        const auto params = frame.cells[i].getCellParams();
        auto refIt = referenceVolumes.find(params.name);
        if (refIt == referenceVolumes.end() || refIt->second <= 0.0) {
            continue;
        }

        const double allowedVolume = refIt->second * maxGrowthFactor;
        const double currentVolume = spheroidVolume(frame.cells[i]);
        if (currentVolume <= allowedVolume + 1e-6) {
            continue;
        }

        const double scale = std::cbrt(allowedVolume / currentVolume);
        const double targetMajor = static_cast<double>(params.majorRadius) * scale;
        const double targetMinor = static_cast<double>(params.minorRadius) * scale;

        std::unordered_map<std::string, float> capParams;
        capParams["majorRadius"] = static_cast<float>(targetMajor - static_cast<double>(params.majorRadius));
        capParams["minorRadius"] = static_cast<float>(targetMinor - static_cast<double>(params.minorRadius));
        frame.cells[i] = frame.cells[i].getParameterizedCell(capParams);

        std::cout << "[Volume Cap] frame " << displayFrame
                  << " cell=" << params.name
                  << " current_volume=" << currentVolume
                  << " allowed_volume=" << allowedVolume
                  << " scale=" << scale << std::endl;
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

Image preprocessLinear(const Image &image, const BaseConfig &config, double scaleFactor)
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
    return processedImage;
}

Image preprocessLinearNoBlur(const Image &image, double scaleFactor)
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

    const double safeScale = (scaleFactor > 0.0) ? scaleFactor : 255.0;
    processedImage.convertTo(processedImage, CV_32F, 1.0 / safeScale);
    cv::min(processedImage, 1.0f, processedImage);
    cv::max(processedImage, 0.0f, processedImage);
    return processedImage;
}

Image processImage(const Image &image, const BaseConfig &config, double scaleFactor, float sigmoidCenterOverride)
{
    Image processedImage = preprocessLinear(image, config, scaleFactor);

    SimulationConfig simConfig = config.simulation;
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

LoadedFrameData loadFrameLinearData(const std::string &imageFile, const BaseConfig &config)
{
    LoadedFrameData loadedFrame;
    std::vector<cv::Mat> linearZSlices; // vector of matrices, each matrix is a 2D image
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
            return loadedFrame;
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
        linearZSlices.reserve(interpolatedGrayZSlices.size());
        for (const auto &slice : interpolatedGrayZSlices) {
            linearZSlices.push_back(preprocessLinearNoBlur(slice, avgSliceMax));
        }

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
        interpolatedZSlices = std::move(linearZSlices);
        loadedFrame.calibrationZIndices = std::move(processedCalibZ);

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
            return loadedFrame;
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
        linearZSlices.push_back(preprocessLinearNoBlur(img, avgSliceMax));
        interpolatedZSlices = std::move(linearZSlices);
    }

    loadedFrame.sigmoidCenter = frameBackground;
    loadedFrame.linearZSlices = std::move(interpolatedZSlices);
    std::cout << std::to_string(loadedFrame.linearZSlices.size()) << "slices built successfully" << std::endl;
    return loadedFrame;
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
        LoadedFrameData loadedFrame = loadFrameLinearData(imagePaths[i], config);
        real_frame = std::move(loadedFrame.linearZSlices);
        float frameSigmoidCenter = loadedFrame.sigmoidCenter;

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
                    frameSigmoidCenter = static_cast<float>(std::clamp(frameSigmoidCenter * invScale, 0.0, 1.0));
                }
            }
        }
        prevFrameMean = computeStackMean(real_frame);

        const float sigmoidCenterUsed = frameSigmoidCenter + config.simulation.sigmoid_center_offset;
        std::cout << "[Sigmoid] center_used=" << sigmoidCenterUsed
                  << " (from_calibration_center=" << frameSigmoidCenter
                  << ", config_center=" << config.simulation.sigmoid_center
                  << ", center_offset=" << config.simulation.sigmoid_center_offset << ")" << std::endl;
        real_frame = fuseBlurSweepSigmoidStack(real_frame, config, frameSigmoidCenter);

        double processedCalibrationBg = -1.0;
        if (!loadedFrame.calibrationZIndices.empty()) {
            processedCalibrationBg =
                computeCalibrationZoneMeanAtZIndices(real_frame, config.simulation, loadedFrame.calibrationZIndices);
        } else {
            processedCalibrationBg = computeCalibrationZoneMean(real_frame, config.simulation);
        }
        if (processedCalibrationBg >= 0.0) {
            const float processedBgClamped = static_cast<float>(std::clamp(processedCalibrationBg, 0.0, 1.0));
            // Keep frame background anchored to pre-sigmoid calibration.
            // Processed-background can drift too low after blur-sweep fusion.
            std::cout << "[Calibration] processed_background=" << processedBgClamped
                      << " (not applied; using pre_sigmoid_background=" << frameBackground << ")" << std::endl;
        } else {
            std::cout << "[Calibration] no valid processed calibration voxels; fallback background="
                      << frameBackground << std::endl;
        }

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

    initializeBrightnessFromGroundTruth();
}


void Lineage::optimize(int frameIndex)
{
    if (frameIndex < 0 || static_cast<size_t>(frameIndex) >= frames.size())
    {
        throw std::invalid_argument("Invalid frame index");
    }

    Frame &frame = frames[frameIndex];
    const int displayFrame = firstFrame + frameIndex;
    const std::map<std::string, double> previousVolumes = collectPreviousVolumes(frames, frameIndex);

    // Regenerate _synthFrame so it reflects the current cells.
    // After copyCellsForward(), _synthFrame is stale (background-only from
    // the Frame constructor which received empty cells). Without this,
    // Phase 1 compares perturbations against a no-cell baseline, causing
    // any perturbation direction to be accepted during the bootstrap phase.
    frame.regenerateSynthFrame();

    const Phase1Result phase1 = runPhase1Perturbation(frame, config, displayFrame);
    const auto candidates = collectSplitCandidates(frame, config, phase1.preOptShapes, phase1, displayFrame);
    const auto acceptedSplits = applySplitCandidates(frame, candidates, referenceVolumes, displayFrame);
    runPostSplitPerturbation(frame, config, acceptedSplits.size());

    if (config.prob.enable_brightness_recovery && frameIndex > 0 && !frame.cells.empty()) {
        runBrightnessRecovery(frame, config, displayFrame, previousVolumes, acceptedSplits);
    }

    enforceVolumeCap(frame, config, referenceVolumes, displayFrame);

    frame.regenerateSynthFrame();
    updateFrameCellBrightness(frameIndex);
}

void Lineage::initializeBrightnessFromGroundTruth() {
    if (frames.empty()) {
        return;
    }
    if (frames[0].cells.empty()) {
        std::cout << "[Brightness Init] frame 1 has no initial cells; skipped" << std::endl;
        return;
    }
    referenceVolumes.clear();
    constexpr double kPi = 3.14159265358979323846;
    for (const auto &cell : frames[0].cells) {
        const auto p = cell.getCellParams();
        const double vol = (4.0 / 3.0) * kPi * static_cast<double>(p.majorRadius) *
                           static_cast<double>(p.majorRadius) * static_cast<double>(p.minorRadius);
        referenceVolumes[p.name] = std::max(1.0, vol);
    }
    updateFrameCellBrightness(0);
    std::cout << "[Brightness Init] initialized " << frames[0].cells.size()
              << " cells from frame 1 ground truth" << std::endl;
}

void Lineage::updateFrameCellBrightness(int frameIndex) {
    if (frameIndex < 0 || static_cast<size_t>(frameIndex) >= frames.size()) {
        return;
    }
    Frame &frame = frames[frameIndex];
    const auto &realFrame = frame.getRealFrame();
    const float emaAlpha = std::clamp(config.prob.brightness_update_ema_alpha, 0.0f, 1.0f);
    for (auto &cell : frame.cells) {
        const auto params = cell.getCellParams();
        const float oldBrightness = cell.getBrightness();
        const float measuredBrightness = cell.estimateBrightnessFromFrame(realFrame);
        const float blendedBrightness =
            cell.hasBrightnessHistory() ? ((1.0f - emaAlpha) * oldBrightness + emaAlpha * measuredBrightness)
                                        : measuredBrightness;
        cell.setBrightness(blendedBrightness, true);
        std::cout << "[Brightness Update] frame " << (firstFrame + frameIndex)
                  << " cell=" << params.name
                  << " old=" << oldBrightness
                  << " measured=" << measuredBrightness
                  << " blended=" << cell.getBrightness()
                  << " min=" << cell.getBrightnessMin()
                  << " max=" << cell.getBrightnessMax()
                  << std::endl;
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
