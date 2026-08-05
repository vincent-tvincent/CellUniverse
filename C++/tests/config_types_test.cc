#include <gtest/gtest.h>

#include "ConfigTypes.hpp"

TEST(ConfigTypesTest, FlatCellRotationRefineEnabledDefaultsToTrue) {
    const YAML::Node node = YAML::Load(R"(
cellType: ellipsoid
cell:
  x: {increase_prob: 0.0, decrease_prob: 0.0, mu: 0.0, sigma: 0.0}
  y: {increase_prob: 0.0, decrease_prob: 0.0, mu: 0.0, sigma: 0.0}
  z: {increase_prob: 0.0, decrease_prob: 0.0, mu: 0.0, sigma: 0.0}
  aRadius: {increase_prob: 0.0, decrease_prob: 0.0, mu: 0.0, sigma: 0.0}
  cRadius: {increase_prob: 0.0, decrease_prob: 0.0, mu: 0.0, sigma: 0.0}
  thetaX: {increase_prob: 0.0, decrease_prob: 0.0, mu: 0.0, sigma: 0.0}
  thetaY: {increase_prob: 0.0, decrease_prob: 0.0, mu: 0.0, sigma: 0.0}
  thetaZ: {increase_prob: 0.0, decrease_prob: 0.0, mu: 0.0, sigma: 0.0}
  minARadius: 1.0
  maxARadius: 10.0
  minCRadius: 1.0
  maxCRadius: 10.0
simulation:
  iterations_per_cell: 1
  background_color: 0.0
  z_scaling: 1.0
  blur_sigma: 0.0
prob: {}
)");

    BaseConfig config;
    config.explodeConfig(node);

    ASSERT_TRUE(config.cell);
    EXPECT_TRUE(config.cell->flatCellRotationRefineEnabled);
}

TEST(ConfigTypesTest, FlatCellRotationRefineEnabledCanBeDisabledFromYaml) {
    const YAML::Node node = YAML::Load(R"(
cellType: ellipsoid
cell:
  x: {increase_prob: 0.0, decrease_prob: 0.0, mu: 0.0, sigma: 0.0}
  y: {increase_prob: 0.0, decrease_prob: 0.0, mu: 0.0, sigma: 0.0}
  z: {increase_prob: 0.0, decrease_prob: 0.0, mu: 0.0, sigma: 0.0}
  aRadius: {increase_prob: 0.0, decrease_prob: 0.0, mu: 0.0, sigma: 0.0}
  cRadius: {increase_prob: 0.0, decrease_prob: 0.0, mu: 0.0, sigma: 0.0}
  thetaX: {increase_prob: 0.0, decrease_prob: 0.0, mu: 0.0, sigma: 0.0}
  thetaY: {increase_prob: 0.0, decrease_prob: 0.0, mu: 0.0, sigma: 0.0}
  thetaZ: {increase_prob: 0.0, decrease_prob: 0.0, mu: 0.0, sigma: 0.0}
  minARadius: 1.0
  maxARadius: 10.0
  minCRadius: 1.0
  maxCRadius: 10.0
  flatCellRotationRefineEnabled: false
simulation:
  iterations_per_cell: 1
  background_color: 0.0
  z_scaling: 1.0
  blur_sigma: 0.0
prob: {}
)");

    BaseConfig config;
    config.explodeConfig(node);

    ASSERT_TRUE(config.cell);
    EXPECT_FALSE(config.cell->flatCellRotationRefineEnabled);
}

TEST(ConfigTypesTest, PreprocessModeNoneIgnoresLegacyN2V2EnabledFlag) {
    const YAML::Node node = YAML::Load(R"(
iterations_per_cell: 1
z_scaling: 1.0
blur_sigma: 0.0
preprocess_mode: none
n2v2_preprocess:
  enabled: true
)");

    SimulationConfig config;
    config.explodeConfig(node);

    EXPECT_EQ(config.preprocess_mode, "none");
    EXPECT_FALSE(config.n2v2_preprocess_enabled);
}

TEST(ConfigTypesTest, PreprocessModeN2V2EnablesN2V2Internally) {
    const YAML::Node node = YAML::Load(R"(
iterations_per_cell: 1
z_scaling: 1.0
blur_sigma: 0.0
preprocess_mode: n2v2
)");

    SimulationConfig config;
    config.explodeConfig(node);

    EXPECT_EQ(config.preprocess_mode, "n2v2");
    EXPECT_TRUE(config.n2v2_preprocess_enabled);
}

TEST(ConfigTypesTest, PsfBrightnessCostDefaultsOnAndParsesExplicitSettings) {
    SimulationConfig defaults;
    EXPECT_TRUE(defaults.psf_brightness_cost_enabled);

    const YAML::Node node = YAML::Load(R"(
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
    SimulationConfig config;
    config.explodeConfig(node);

    EXPECT_TRUE(config.psf_brightness_cost_enabled);
    EXPECT_FLOAT_EQ(config.psf_brightness_cost_sigma, 1.25f);
    EXPECT_FLOAT_EQ(config.psf_brightness_prior_weight, 0.2f);
    EXPECT_FLOAT_EQ(config.psf_brightness_prior_tolerance, 0.007f);
    EXPECT_FLOAT_EQ(config.psf_brightness_prior_spread, 0.03f);
    EXPECT_FLOAT_EQ(config.psf_brightness_contrast_floor, 0.04f);

    const YAML::Node disabledNode = YAML::Load(R"(
iterations_per_cell: 1
z_scaling: 1.0
blur_sigma: 0.0
psf_brightness_cost_enabled: false
)");
    SimulationConfig disabled;
    disabled.explodeConfig(disabledNode);
    EXPECT_FALSE(disabled.psf_brightness_cost_enabled);
}

TEST(ConfigTypesTest, SplitBirthSiblingVolumeGuardDefaultsAndParses) {
    ProbabilityConfig defaults;
    EXPECT_TRUE(defaults.split_birth_sibling_volume_guard_enabled);
    EXPECT_FLOAT_EQ(defaults.split_birth_sibling_volume_ratio_max, 1.25f);
    EXPECT_EQ(defaults.split_sibling_volume_guard_recent_frames, 0);
    EXPECT_FLOAT_EQ(
        defaults.split_recent_daughter_volume_min_birth_factor, 0.0f);
    EXPECT_FLOAT_EQ(
        defaults.split_recent_daughter_volume_max_birth_factor, 0.0f);
    EXPECT_FALSE(defaults.split_acceptance_ignore_sibling_pair_overlap_cost);

    const YAML::Node node = YAML::Load(R"(
split_birth_sibling_volume_guard_enabled: false
split_birth_sibling_volume_ratio_max: 1.20
split_sibling_volume_guard_recent_frames: 8
split_recent_daughter_volume_min_birth_factor: 0.85
split_recent_daughter_volume_max_birth_factor: 1.25
split_acceptance_ignore_sibling_pair_overlap_cost: true
)");
    ProbabilityConfig config;
    config.explodeConfig(node);
    EXPECT_FALSE(config.split_birth_sibling_volume_guard_enabled);
    EXPECT_FLOAT_EQ(config.split_birth_sibling_volume_ratio_max, 1.20f);
    EXPECT_EQ(config.split_sibling_volume_guard_recent_frames, 8);
    EXPECT_FLOAT_EQ(
        config.split_recent_daughter_volume_min_birth_factor, 0.85f);
    EXPECT_FLOAT_EQ(
        config.split_recent_daughter_volume_max_birth_factor, 1.25f);
    EXPECT_TRUE(config.split_acceptance_ignore_sibling_pair_overlap_cost);
}

TEST(ConfigTypesTest, PsfBrightnessCostRejectsNonpositiveContrastFloor) {
    const YAML::Node node = YAML::Load(R"(
iterations_per_cell: 1
z_scaling: 1.0
blur_sigma: 0.0
psf_brightness_contrast_floor: 0.0
)");
    SimulationConfig config;
    EXPECT_THROW(config.explodeConfig(node), std::invalid_argument);
}

TEST(ConfigTypesTest, BackgroundStepAndCu4ReviewControlsParseFromYaml) {
    const YAML::Node simulationNode = YAML::Load(R"(
iterations_per_cell: 1
z_scaling: 1.0
blur_sigma: 0.0
initial_csv_background_intensity_step_limit_enabled: false
initial_csv_background_maximum_intensity_step: 0.08
celluniverse4_postfit_bright_cell_size_review_enabled: true
celluniverse4_postfit_bright_cell_size_review_min_brightness: 0.52
celluniverse4_postfit_bright_cell_size_review_attempts: 4
celluniverse4_postfit_bright_cell_size_review_radius_step_fraction: 0.03
celluniverse4_postfit_bright_cell_size_review_max_radius_scale: 1.14
celluniverse4_postfit_bright_cell_size_review_component_min_gain_voxels: 12
)");
    SimulationConfig simulation;
    simulation.explodeConfig(simulationNode);
    EXPECT_FALSE(simulation.initial_csv_background_intensity_step_limit_enabled);
    EXPECT_FLOAT_EQ(simulation.initial_csv_background_maximum_intensity_step,
                    0.08f);
    EXPECT_TRUE(
        simulation.celluniverse4_postfit_bright_cell_size_review_enabled);
    EXPECT_FLOAT_EQ(
        simulation.celluniverse4_postfit_bright_cell_size_review_min_brightness,
        0.52f);
    EXPECT_EQ(
        simulation.celluniverse4_postfit_bright_cell_size_review_attempts, 4);

    const YAML::Node candidateNode = YAML::Load(R"(
background_drop_current_local_split_only_enabled: true
background_drop_current_local_prepass_fallback_enabled: true
background_drop_current_local_prepass_fallback_min_kept_pixels: 800
background_drop_current_local_cluster_iterations: 9
background_drop_current_local_projected_seed_distance_radius_fraction: 1.6
background_drop_current_local_cluster_gather_radius_scale: 3.2
background_drop_current_local_cluster_min_weight_fraction: 0.15
background_drop_current_local_cluster_min_variance_reduction: 0.12
background_drop_current_local_prepass_fallback_max_anchor_distance_radius_fraction: 0.75
background_drop_current_local_prepass_fallback_max_midpoint_distance_radius_fraction: 0.60
background_drop_current_local_prepass_fallback_min_cost_improvement: 500.0
continuation_refine_anchor_distance_gate_enabled: false
continuation_refine_max_anchor_distance_min_radius_fraction: 1.25
)");
    CandidateBatchConfig candidateBatch;
    candidateBatch.explodeConfig(candidateNode);
    EXPECT_TRUE(candidateBatch.backgroundDropCurrentLocalSplitOnlyEnabled);
    EXPECT_TRUE(
        candidateBatch.backgroundDropCurrentLocalPrepassFallbackEnabled);
    EXPECT_EQ(
        candidateBatch.backgroundDropCurrentLocalPrepassFallbackMinKeptPixels,
        800);
    EXPECT_EQ(candidateBatch.backgroundDropCurrentLocalClusterIterations, 9);
    EXPECT_FLOAT_EQ(
        candidateBatch.backgroundDropCurrentLocalProjectedSeedDistanceRadiusFraction,
        1.6f);
    EXPECT_FLOAT_EQ(
        candidateBatch.backgroundDropCurrentLocalClusterGatherRadiusScale,
        3.2f);
    EXPECT_FLOAT_EQ(
        candidateBatch.backgroundDropCurrentLocalClusterMinWeightFraction,
        0.15f);
    EXPECT_FLOAT_EQ(
        candidateBatch.backgroundDropCurrentLocalClusterMinVarianceReduction,
        0.12f);
    EXPECT_FLOAT_EQ(
        candidateBatch
            .backgroundDropCurrentLocalPrepassFallbackMaxAnchorDistanceRadiusFraction,
        0.75f);
    EXPECT_FLOAT_EQ(
        candidateBatch
            .backgroundDropCurrentLocalPrepassFallbackMaxMidpointDistanceRadiusFraction,
        0.60f);
    EXPECT_FLOAT_EQ(
        candidateBatch
            .backgroundDropCurrentLocalPrepassFallbackMinCostImprovement,
        500.0f);
    EXPECT_FALSE(
        candidateBatch.continuationRefineAnchorDistanceGateEnabled);
    EXPECT_FLOAT_EQ(
        candidateBatch
            .continuationRefineMaxAnchorDistanceMinRadiusFraction,
        1.25f);
}
