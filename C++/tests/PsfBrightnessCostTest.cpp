#include "Frame.hpp"
#include "SplitSiblingVolumeGuard.hpp"

#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void expectTrue(bool value, const std::string &message)
{
    if (!value) throw std::runtime_error(message);
}

void expectNear(double actual,
                double expected,
                double tolerance,
                const std::string &message)
{
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(
            message + ": expected=" + std::to_string(expected) +
            " actual=" + std::to_string(actual));
    }
}

SimulationConfig makeConfig(int zSlices, bool enabled, float sigma = 0.0f)
{
    SimulationConfig config;
    config.z_slices = zSlices;
    config.z_scaling = 1.0f;
    config.psf_brightness_cost_enabled = enabled;
    config.psf_brightness_cost_sigma = sigma;
    config.psf_brightness_prior_weight = 0.1f;
    config.psf_brightness_prior_tolerance = 0.0f;
    config.psf_brightness_contrast_floor = 0.05f;
    return config;
}

Ellipsoid makeCell(const std::string &name,
                   float brightness,
                   float radius = 1.0f,
                   bool isTrash = false)
{
    EllipsoidParams params(name,
                           1.0f,
                           1.0f,
                           0.0f,
                           radius,
                           radius,
                           0.0f,
                           0.0f,
                           0.0f,
                           brightness);
    params.bRadius = radius;
    params.isTrash = isTrash;
    return Ellipsoid(params);
}

BoundingBox3D makeBbox(int xMin,
                       int xMax,
                       int yMin,
                       int yMax,
                       int zMin,
                       int zMax)
{
    BoundingBox3D bbox;
    bbox.xMin = xMin;
    bbox.xMax = xMax;
    bbox.yMin = yMin;
    bbox.yMax = yMax;
    bbox.zMin = zMin;
    bbox.zMax = zMax;
    return bbox;
}

PsfBrightnessCostContext fixedContext(double candidateMean = 1.0,
                                      double priorMean = 1.0,
                                      double background = 0.0)
{
    PsfBrightnessCostContext context;
    context.candidate = {candidateMean, 0.0, true};
    context.prior = {priorMean, 0.0, true};
    context.background = background;
    context.valid = true;
    return context;
}

std::vector<double> gaussianKernel(double sigma)
{
    const int radius = static_cast<int>(std::floor(4.0 * sigma + 0.5));
    std::vector<double> kernel(static_cast<size_t>(2 * radius + 1));
    double sum = 0.0;
    for (int offset = -radius; offset <= radius; ++offset) {
        const double value = std::exp(
            -0.5 * static_cast<double>(offset * offset) / (sigma * sigma));
        kernel[static_cast<size_t>(offset + radius)] = value;
        sum += value;
    }
    for (double &value : kernel) value /= sum;
    return kernel;
}

void testConfigSwitchAndBounds()
{
    SimulationConfig defaults;
    expectTrue(defaults.psf_brightness_cost_enabled,
               "PSF brightness cost must default on");

    ProbabilityConfig probabilityDefaults;
    expectTrue(!probabilityDefaults
                    .split_acceptance_ignore_sibling_pair_overlap_cost,
               "sibling-overlap acceptance bypass must default off");
    const YAML::Node probabilityNode = YAML::Load(R"(
split_acceptance_ignore_sibling_pair_overlap_cost: true
 )");
    ProbabilityConfig probabilityEnabled;
    probabilityEnabled.explodeConfig(probabilityNode);
    expectTrue(probabilityEnabled
                   .split_acceptance_ignore_sibling_pair_overlap_cost,
               "sibling-overlap acceptance bypass was not parsed");

    const YAML::Node enabledNode = YAML::Load(R"(
iterations_per_cell: 1
z_scaling: 1.0
blur_sigma: 0.0
psf_brightness_cost_enabled: true
psf_brightness_cost_sigma: 1.25
psf_brightness_prior_weight: 0.2
psf_brightness_prior_tolerance: 0.007
psf_brightness_prior_spread: 0.03
psf_brightness_contrast_floor: 0.04
)");
    SimulationConfig enabled;
    enabled.explodeConfig(enabledNode);
    enabled.z_slices = 1;
    expectTrue(enabled.psf_brightness_cost_enabled,
               "explicit switch was not parsed");
    expectNear(enabled.psf_brightness_cost_sigma, 1.25, 1e-7,
               "sigma was not parsed");

    const YAML::Node disabledNode = YAML::Load(R"(
iterations_per_cell: 1
z_scaling: 1.0
blur_sigma: 0.0
psf_brightness_cost_enabled: false
)");
    SimulationConfig disabled;
    disabled.explodeConfig(disabledNode);
    expectTrue(!disabled.psf_brightness_cost_enabled,
               "explicit legacy rollback switch was not parsed");

    const YAML::Node oversizedSigma = YAML::Load(R"(
iterations_per_cell: 1
z_scaling: 1.0
blur_sigma: 0.0
psf_brightness_cost_sigma: 16.1
)");
    bool threw = false;
    try {
        SimulationConfig invalid;
        invalid.explodeConfig(oversizedSigma);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    expectTrue(threw, "oversized PSF sigma must be rejected");

    std::vector<cv::Mat> real = {cv::Mat::zeros(1, 1, CV_32F)};
    Frame psfFrame(real, enabled, {}, "", "psf_mode");
    psfFrame.setUseBboxCost(false, 3.0f);
    expectTrue(psfFrame.getUseBboxCost(),
               "PSF mode must force convolution-safe fixed-region cost");

    SimulationConfig legacyConfig = makeConfig(1, false);
    Frame legacyFrame(real, legacyConfig, {}, "", "legacy_mode");
    legacyFrame.setUseBboxCost(false, 3.0f);
    expectTrue(!legacyFrame.getUseBboxCost(),
               "disabled switch unexpectedly changed legacy cost mode");
}

void testVolumeWeightedMoments()
{
    const Ellipsoid small = makeCell("a", 0.2f, 1.0f);
    const Ellipsoid large = makeCell("b", 0.8f, 2.0f);
    const PsfBrightnessMoments moments =
        Frame::computePsfBrightnessMoments({&small, &large});
    const double expectedMean = (0.2 + 8.0 * 0.8) / 9.0;
    const double expectedVariance =
        (std::pow(0.2 - expectedMean, 2.0) +
         8.0 * std::pow(0.8 - expectedMean, 2.0)) / 9.0;
    expectTrue(moments.valid, "moments should be valid");
    expectNear(moments.mean, expectedMean, 1e-6,
               "volume-weighted mean mismatch");
    expectNear(moments.spread, std::sqrt(expectedVariance), 1e-6,
               "volume-weighted spread mismatch");
}

void testSymmetricImageAndBrightnessTerms()
{
    SimulationConfig config = makeConfig(1, true);
    config.asymmetric_cost_weight = 8.0f;
    std::vector<cv::Mat> real = {cv::Mat::zeros(1, 1, CV_32F)};
    std::vector<cv::Mat> synth = {
        cv::Mat(1, 1, CV_32F, cv::Scalar(1.0f))};
    Frame frame(real, config, {}, "", "symmetric");
    PsfBrightnessCostContext context = fixedContext(0.8, 1.0, 0.0);
    BboxCostBreakdown breakdown;
    const double cost = frame.calculateBboxCost(
        makeBbox(0, 0, 0, 0, 0, 0), synth, {}, -1, &context, &breakdown);
    expectNear(breakdown.imageCost, 1.0, 1e-7,
               "enabled image term must be symmetric squared L2");
    expectNear(breakdown.brightnessPriorCost, 0.004, 1e-6,
               "brightness prior term mismatch");
    expectNear(cost, 1.004, 1e-6, "composite cost mismatch");
}

void testLegacySwitchParity()
{
    SimulationConfig config = makeConfig(1, false);
    config.asymmetric_cost_weight = 8.0f;
    config.asymmetric_cost_threshold = 0.0f;
    std::vector<cv::Mat> real = {cv::Mat::zeros(1, 1, CV_32F)};
    std::vector<cv::Mat> synth = {
        cv::Mat(1, 1, CV_32F, cv::Scalar(1.0f))};
    Frame frame(real, config, {}, "", "legacy");
    const double cost = frame.calculateBboxCost(
        makeBbox(0, 0, 0, 0, 0, 0), synth, {});
    expectNear(cost, 8.0, 1e-12,
               "disabled switch must preserve legacy asymmetric L2");
}

void testBirthSiblingVolumeGuard()
{
    const std::array<float, 3> accepted20{
        25.0395f, 21.3703f, 15.8936f};
    const std::array<float, 3> accepted21{
        22.5855f, 20.2891f, 15.3679f};
    const SplitSiblingVolumeGuardResult pair20 =
        computeSplitSiblingVolumeGuard(accepted20, accepted21, 1.20f);
    expectTrue(pair20.valid && pair20.changed && pair20.firstIsLarger,
               "20/21 birth imbalance was not detected");
    expectNear(pair20.afterRatio, 1.20, 1e-6,
               "20/21 corrected ratio mismatch");
    expectTrue(pair20.largerRadiusScale > 0.99f &&
                   pair20.largerRadiusScale < 1.0f,
               "20 should receive only a small isotropic correction");

    const std::array<float, 3> accepted31{
        27.7316f, 24.9154f, 16.8456f};
    const std::array<float, 3> accepted30{
        23.7699f, 21.4161f, 14.9503f};
    const SplitSiblingVolumeGuardResult pair31 =
        computeSplitSiblingVolumeGuard(accepted31, accepted30, 1.20f);
    expectTrue(pair31.valid && pair31.changed && pair31.firstIsLarger,
               "31/30 birth imbalance was not detected");
    expectNear(pair31.afterRatio, 1.20, 1e-6,
               "31/30 corrected ratio mismatch");
    expectTrue(pair31.largerRadiusScale < 0.95f,
               "31 should receive a material isotropic correction");

    const SplitSiblingVolumeGuardResult balanced =
        computeSplitSiblingVolumeGuard(
            {10.0f, 9.0f, 8.0f}, {9.8f, 9.0f, 8.0f}, 1.20f);
    expectTrue(balanced.valid && !balanced.changed,
               "balanced siblings were modified");
}

void testRecentDaughterBirthVolumeEnvelope()
{
    const std::array<float, 3> birth{10.0f, 10.0f, 10.0f};
    const auto collapsed = computeSplitDaughterBirthVolumeEnvelope(
        {8.0f, 10.0f, 10.0f}, birth, 0.85f, 1.25f);
    expectTrue(collapsed.valid && collapsed.changed &&
                   collapsed.radiusScale > 1.0f,
               "recent daughter collapse was not restored to the envelope");
    expectNear(collapsed.targetVolume, 850.0, 1e-3,
               "recent daughter minimum volume mismatch");

    const auto expanded = computeSplitDaughterBirthVolumeEnvelope(
        {12.0f, 11.0f, 10.0f}, birth, 0.85f, 1.25f);
    expectTrue(expanded.valid && expanded.changed &&
                   expanded.radiusScale < 1.0f,
               "recent daughter expansion was not capped");
    expectNear(expanded.targetVolume, 1250.0, 1e-3,
               "recent daughter maximum volume mismatch");

    const auto stable = computeSplitDaughterBirthVolumeEnvelope(
        {10.0f, 10.0f, 10.0f}, birth, 0.85f, 1.25f);
    expectTrue(stable.valid && !stable.changed,
               "in-envelope daughter volume was modified");

    const auto invalid = computeSplitDaughterBirthVolumeEnvelope(
        {10.0f, 10.0f, 10.0f}, birth, 1.05f, 1.25f);
    expectTrue(!invalid.valid,
               "birth volume envelope accepted a minimum above birth");
}

void testCroppedBboxUsesOutsideHalo()
{
    constexpr double sigma = 1.0;
    const std::vector<double> kernel = gaussianKernel(sigma);
    const int radius = static_cast<int>((kernel.size() - 1) / 2);
    std::vector<cv::Mat> synth = {cv::Mat::zeros(1, 11, CV_32F)};
    synth[0].at<float>(0, 4) = 1.0f;
    std::vector<cv::Mat> real = {cv::Mat::zeros(1, 11, CV_32F)};
    real[0].at<float>(0, 5) =
        static_cast<float>(kernel[static_cast<size_t>(radius + 1)]);

    SimulationConfig config = makeConfig(1, true, sigma);
    config.psf_brightness_prior_weight = 0.0f;
    Frame frame(real, config, {}, "", "halo");
    PsfBrightnessCostContext context = fixedContext();
    const double cost = frame.calculateBboxCost(
        makeBbox(5, 5, 0, 0, 0, 0), synth, {}, -1, &context);
    expectNear(cost, 0.0, 1e-12,
               "cropped bbox ignored synth signal in its PSF halo");
}

void testNearestImageEdgeBehavior()
{
    constexpr double sigma = 1.0;
    const std::vector<double> kernel = gaussianKernel(sigma);
    const int radius = static_cast<int>((kernel.size() - 1) / 2);
    double expectedEdge = 0.0;
    for (int offset = -radius; offset <= 0; ++offset) {
        expectedEdge += kernel[static_cast<size_t>(offset + radius)];
    }

    std::vector<cv::Mat> synth = {cv::Mat::zeros(1, 11, CV_32F)};
    synth[0].at<float>(0, 0) = 1.0f;
    std::vector<cv::Mat> real = {cv::Mat::zeros(1, 11, CV_32F)};
    real[0].at<float>(0, 0) = static_cast<float>(expectedEdge);
    SimulationConfig config = makeConfig(1, true, sigma);
    config.psf_brightness_prior_weight = 0.0f;
    Frame frame(real, config, {}, "", "edge");
    PsfBrightnessCostContext context = fixedContext();
    const double cost = frame.calculateBboxCost(
        makeBbox(0, 0, 0, 0, 0, 0), synth, {}, -1, &context);
    expectNear(cost, 0.0, 1e-12,
               "image edge does not match nearest-boundary Gaussian");
}

void testMaskControlsBrightnessVoxelCount()
{
    SimulationConfig config = makeConfig(1, true);
    std::vector<cv::Mat> real = {cv::Mat::zeros(1, 2, CV_32F)};
    Frame frame(real, config, {}, "", "mask");
    PsfBrightnessCostContext context = fixedContext(0.8, 1.0, 0.0);
    BboxCostBreakdown breakdown;
    const double cost = frame.calculateBboxCost(
        makeBbox(0, 1, 0, 0, 0, 0), real, {1, 0}, -1,
        &context, &breakdown);
    expectTrue(breakdown.evaluatedVoxelCount == 1,
               "mask did not control evaluated voxel count");
    expectNear(cost, 0.004, 1e-6,
               "brightness term did not scale by selected voxel count");
}

void testLineageGroupAndPostFreezeRegistration()
{
    SimulationConfig config = makeConfig(1, true);
    const Ellipsoid parent = makeCell("2", 0.6f, 2.0f);
    std::vector<cv::Mat> real = {cv::Mat::zeros(3, 3, CV_32F)};
    Frame frame(real, config, {parent}, "", "prior");
    frame.setBackgroundColor(0.1f);
    frame.freezePsfBrightnessPriors();

    const Ellipsoid d0 = makeCell("20", 0.4f);
    const Ellipsoid d1 = makeCell("21", 0.8f);
    const PsfBrightnessCostContext group =
        frame.makePsfBrightnessCostContext({&d0, &d1}, d0.getName());
    expectTrue(group.valid, "daughter group did not inherit parent prior");
    expectNear(group.candidate.mean, 0.6, 1e-6,
               "daughter group mean mismatch");
    expectNear(group.candidate.spread, 0.2, 1e-6,
               "daughter group spread mismatch");
    expectNear(group.prior.mean, 0.6, 1e-6,
               "daughter group parent prior mismatch");

    const Ellipsoid orphan = makeCell("cellu3_orphan_3_1", 0.7f);
    expectTrue(frame.registerPsfBrightnessPriorIfMissing(orphan),
               "new independent cell prior was not registered");
    const PsfBrightnessCostContext orphanContext =
        frame.makePsfBrightnessCostContext({&orphan}, orphan.getName());
    expectTrue(orphanContext.valid,
               "registered independent-cell context is invalid");
    expectNear(orphanContext.prior.mean, 0.7, 1e-6,
               "independent-cell prior mismatch");

    const Ellipsoid trash = makeCell("trash", 0.5f, 1.0f, true);
    Frame trashFrame(real, config, {trash}, "", "trash");
    trashFrame.freezePsfBrightnessPriors();
    const PsfBrightnessCostContext trashContext =
        trashFrame.makePsfBrightnessCostContext({&trash}, trash.getName());
    expectTrue(trashContext.valid, "trash cell prior was skipped");
}

} // namespace

int main()
{
    // Production loads these global ellipsoid bounds from YAML before cells
    // are constructed. The focused test supplies a small valid envelope.
    Ellipsoid::cellConfig.minARadius = 0.1;
    Ellipsoid::cellConfig.maxARadius = 100.0;
    Ellipsoid::cellConfig.minBRadius = 0.1;
    Ellipsoid::cellConfig.maxBRadius = 100.0;
    Ellipsoid::cellConfig.minCRadius = 0.1;
    Ellipsoid::cellConfig.maxCRadius = 100.0;
    Ellipsoid::cellConfig.maxZ = 100.0f;

    const std::vector<std::pair<std::string, std::function<void()>>> tests = {
        {"config switch and bounds", testConfigSwitchAndBounds},
        {"volume-weighted moments", testVolumeWeightedMoments},
        {"symmetric image and brightness terms",
         testSymmetricImageAndBrightnessTerms},
        {"legacy switch parity", testLegacySwitchParity},
        {"birth sibling volume guard", testBirthSiblingVolumeGuard},
        {"recent daughter birth volume envelope",
         testRecentDaughterBirthVolumeEnvelope},
        {"cropped bbox halo", testCroppedBboxUsesOutsideHalo},
        {"nearest image edge", testNearestImageEdgeBehavior},
        {"mask voxel scaling", testMaskControlsBrightnessVoxelCount},
        {"lineage group and post-freeze registration",
         testLineageGroupAndPostFreezeRegistration},
    };

    int failures = 0;
    for (const auto &entry : tests) {
        try {
            entry.second();
            std::cout << "PASS: " << entry.first << '\n';
        } catch (const std::exception &error) {
            ++failures;
            std::cerr << "FAIL: " << entry.first << ": "
                      << error.what() << '\n';
        }
    }
    std::cout << "PSF brightness cost tests: "
              << (tests.size() - static_cast<size_t>(failures))
              << "/" << tests.size() << " passed\n";
    return failures == 0 ? 0 : 1;
}
