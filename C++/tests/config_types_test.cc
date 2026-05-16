#include <gtest/gtest.h>

#include "ConfigTypes.hpp"

// Note: prior tests for `flatCellRotationRefineEnabled` were removed when the
// cell rotation refinement was deprecated and the field deleted from
// EllipsoidConfig. Tests for the deleted SimulationConfig fields
// `background_color` and `cell_color` were similarly removed (replaced by
// Frame::_backgroundValue and per-cell brightness — see other test files).
//
// This file intentionally exercises only currently-supported config behavior.

TEST(ConfigTypesTest, ExplodeConfigParsesEllipsoidCellSection) {
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
  z_scaling: 1.0
  blur_sigma: 0.0
prob: {}
)");

    BaseConfig config;
    config.explodeConfig(node);

    ASSERT_TRUE(config.cell);
    EXPECT_DOUBLE_EQ(config.cell->minARadius, 1.0);
    EXPECT_DOUBLE_EQ(config.cell->maxARadius, 10.0);
    EXPECT_DOUBLE_EQ(config.cell->minCRadius, 1.0);
    EXPECT_DOUBLE_EQ(config.cell->maxCRadius, 10.0);
}

TEST(ConfigTypesTest, AutoCalibrationFieldsHaveExpectedDefaults) {
    SimulationConfig cfg;
    EXPECT_TRUE(cfg.auto_calibrate_brightness_enabled);
    EXPECT_FLOAT_EQ(cfg.manual_background_intensity, -1.0f);
    EXPECT_FLOAT_EQ(cfg.manual_cell_intensity, -1.0f);
    EXPECT_FLOAT_EQ(cfg.calibration_cell_inner_fraction, 0.7f);
    EXPECT_FLOAT_EQ(cfg.calibration_pixel_trim_percent, 0.10f);
    EXPECT_FALSE(cfg.calibrated_preprocess_dynamic_range_enabled);
    EXPECT_FALSE(cfg.auto_calibrate_runtime_background_enabled);
    EXPECT_FLOAT_EQ(cfg.pca_shape_bg_floor, 0.01f);
    EXPECT_FLOAT_EQ(cfg.pca_shape_bg_margin, 0.02f);
    EXPECT_FLOAT_EQ(cfg.pca_shape_bg_sigma_k, 2.0f);
}

TEST(ConfigTypesTest, AutoDeriveGeometryFieldsHaveExpectedDefaults) {
    SimulationConfig cfg;
    EXPECT_TRUE(cfg.auto_derive_geometry_enabled);
    EXPECT_FLOAT_EQ(cfg.geometry_force_max_a_radius, -1.0f);
    EXPECT_FLOAT_EQ(cfg.geometry_force_max_b_radius, -1.0f);
    EXPECT_FLOAT_EQ(cfg.geometry_force_max_c_radius, -1.0f);
    EXPECT_FLOAT_EQ(cfg.geometry_force_min_a_radius, -1.0f);
    EXPECT_FLOAT_EQ(cfg.geometry_force_min_b_radius, -1.0f);
    EXPECT_FLOAT_EQ(cfg.geometry_force_min_c_radius, -1.0f);
    EXPECT_FLOAT_EQ(cfg.geometry_force_perturb_reference_radius, -1.0f);
    EXPECT_FLOAT_EQ(cfg.geometry_force_xy_sigma, -1.0f);
    EXPECT_FLOAT_EQ(cfg.geometry_force_z_sigma, -1.0f);
    EXPECT_FLOAT_EQ(cfg.geometry_force_position_prior_threshold, -1.0f);
    EXPECT_EQ(cfg.geometry_force_iterations_per_cell, -1);
    EXPECT_EQ(cfg.geometry_force_pca_bridge_min_side_voxels, -1);
}

TEST(ConfigTypesTest, ThreadCapFieldsHaveExpectedDefaults) {
    SimulationConfig cfg;
    EXPECT_EQ(cfg.global_thread_cap, 0);
    EXPECT_EQ(cfg.opencv_thread_cap, 1);
}

TEST(ConfigTypesTest, RangePreprocessFieldsHaveExpectedDefaults) {
    SimulationConfig cfg;
    EXPECT_FALSE(cfg.range_preprocess_enabled);
    EXPECT_EQ(cfg.range_preprocess_range_count, 100);
    EXPECT_FLOAT_EQ(cfg.range_preprocess_range_percentile, 99.0f);
    EXPECT_FLOAT_EQ(cfg.range_preprocess_occupancy_threshold_percent, 0.2f);
    EXPECT_FLOAT_EQ(cfg.range_preprocess_final_threshold_percentile, 90.0f);
    EXPECT_FLOAT_EQ(cfg.range_preprocess_bright_boost_fraction, 0.30f);
    EXPECT_FLOAT_EQ(cfg.range_preprocess_bright_boost_factor, 5.0f);
    EXPECT_TRUE(cfg.range_preprocess_blur_enabled);
    EXPECT_TRUE(cfg.range_preprocess_interpolate_z);
    EXPECT_TRUE(cfg.range_preprocess_auto_dynamic_range);
    EXPECT_FLOAT_EQ(cfg.range_preprocess_sigma, 6.0f);
    EXPECT_FLOAT_EQ(cfg.range_preprocess_real_ratio, 0.0f);
    EXPECT_FLOAT_EQ(cfg.range_preprocess_max_brightness, 65536.0f);
    EXPECT_FLOAT_EQ(cfg.range_preprocess_export_max_brightness, 65536.0f);
    EXPECT_TRUE(cfg.range_preprocess_skip_legacy_postprocess);
    EXPECT_TRUE(cfg.range_preprocess_bypass_load_blur);
    EXPECT_TRUE(cfg.range_preprocess_debug_stats);
}

TEST(ConfigTypesTest, RangePreprocessFieldsParseFromYaml) {
    SimulationConfig cfg;
    cfg.explodeConfig(YAML::Load(R"(
iterations_per_cell: 1
z_scaling: 1.0
blur_sigma: 0.0
range_preprocess_enabled: true
range_preprocess_range_count: 32
range_preprocess_range_percentile: 98.5
range_preprocess_occupancy_threshold_percent: 0.35
range_preprocess_final_threshold_percentile: 88.0
range_preprocess_bright_boost_fraction: 0.25
range_preprocess_bright_boost_factor: 4.0
range_preprocess_blur_enabled: false
range_preprocess_interpolate_z: false
range_preprocess_auto_dynamic_range: false
range_preprocess_sigma: 4.5
range_preprocess_real_ratio: 0.25
range_preprocess_max_brightness: 4095.0
range_preprocess_export_max_brightness: 255.0
range_preprocess_skip_legacy_postprocess: false
range_preprocess_bypass_load_blur: false
range_preprocess_debug_stats: false
pca_shape_bg_floor: 0.00002
pca_shape_bg_margin: 0.00003
)"));

    EXPECT_TRUE(cfg.range_preprocess_enabled);
    EXPECT_EQ(cfg.range_preprocess_range_count, 32);
    EXPECT_FLOAT_EQ(cfg.range_preprocess_range_percentile, 98.5f);
    EXPECT_FLOAT_EQ(cfg.range_preprocess_occupancy_threshold_percent, 0.35f);
    EXPECT_FLOAT_EQ(cfg.range_preprocess_final_threshold_percentile, 88.0f);
    EXPECT_FLOAT_EQ(cfg.range_preprocess_bright_boost_fraction, 0.25f);
    EXPECT_FLOAT_EQ(cfg.range_preprocess_bright_boost_factor, 4.0f);
    EXPECT_FALSE(cfg.range_preprocess_blur_enabled);
    EXPECT_FALSE(cfg.range_preprocess_interpolate_z);
    EXPECT_FALSE(cfg.range_preprocess_auto_dynamic_range);
    EXPECT_FLOAT_EQ(cfg.range_preprocess_sigma, 4.5f);
    EXPECT_FLOAT_EQ(cfg.range_preprocess_real_ratio, 0.25f);
    EXPECT_FLOAT_EQ(cfg.range_preprocess_max_brightness, 4095.0f);
    EXPECT_FLOAT_EQ(cfg.range_preprocess_export_max_brightness, 255.0f);
    EXPECT_FALSE(cfg.range_preprocess_skip_legacy_postprocess);
    EXPECT_FALSE(cfg.range_preprocess_bypass_load_blur);
    EXPECT_FALSE(cfg.range_preprocess_debug_stats);
    EXPECT_FLOAT_EQ(cfg.pca_shape_bg_floor, 0.00002f);
    EXPECT_FLOAT_EQ(cfg.pca_shape_bg_margin, 0.00003f);
}

TEST(ConfigTypesTest, SynthBackgroundFieldsHaveExpectedDefaults) {
    SimulationConfig cfg;
    EXPECT_FLOAT_EQ(cfg.synth_background_brightness_factor, 1.0f);

    ProbabilityConfig prob;
    EXPECT_FALSE(prob.pca_bridge_shape_fit_shortcut_enabled);
    EXPECT_FLOAT_EQ(prob.pca_bridge_black_bg_margin, 0.02f);
}

TEST(ConfigTypesTest, SplitCandidateTogglesHaveExpectedDefaults) {
    ProbabilityConfig prob;
    EXPECT_TRUE(prob.split_candidate_enable_shortest_axis);
    EXPECT_TRUE(prob.split_candidate_enable_image_pca_axis);
    EXPECT_TRUE(prob.split_candidate_enable_data_midpoint);
    EXPECT_TRUE(prob.split_candidate_enable_snapshot_midpoint);
    EXPECT_TRUE(prob.split_candidate_enable_primary_variant);
    EXPECT_TRUE(prob.split_candidate_enable_rotation_minus_variant);
    EXPECT_TRUE(prob.split_candidate_enable_rotation_plus_variant);
    EXPECT_TRUE(prob.split_candidate_enable_translation_minus_variant);
    EXPECT_TRUE(prob.split_candidate_enable_translation_plus_variant);
    EXPECT_TRUE(prob.split_candidate_enable_bridge_candidate);
}

TEST(ConfigTypesTest, SplitCandidateTogglesParseFromYaml) {
    ProbabilityConfig prob;
    prob.explodeConfig(YAML::Load(R"(
split_candidate_enable_shortest_axis: false
split_candidate_enable_image_pca_axis: false
split_candidate_enable_data_midpoint: false
split_candidate_enable_snapshot_midpoint: false
split_candidate_enable_primary_variant: false
split_candidate_enable_rotation_minus_variant: false
split_candidate_enable_rotation_plus_variant: false
split_candidate_enable_translation_minus_variant: false
split_candidate_enable_translation_plus_variant: false
split_candidate_enable_bridge_candidate: false
)"));

    EXPECT_FALSE(prob.split_candidate_enable_shortest_axis);
    EXPECT_FALSE(prob.split_candidate_enable_image_pca_axis);
    EXPECT_FALSE(prob.split_candidate_enable_data_midpoint);
    EXPECT_FALSE(prob.split_candidate_enable_snapshot_midpoint);
    EXPECT_FALSE(prob.split_candidate_enable_primary_variant);
    EXPECT_FALSE(prob.split_candidate_enable_rotation_minus_variant);
    EXPECT_FALSE(prob.split_candidate_enable_rotation_plus_variant);
    EXPECT_FALSE(prob.split_candidate_enable_translation_minus_variant);
    EXPECT_FALSE(prob.split_candidate_enable_translation_plus_variant);
    EXPECT_FALSE(prob.split_candidate_enable_bridge_candidate);
}
