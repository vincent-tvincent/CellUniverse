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
