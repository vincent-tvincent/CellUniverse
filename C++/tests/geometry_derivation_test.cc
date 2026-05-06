#include <gtest/gtest.h>
#include <string>
#include <vector>

#include "GeometryDerivation.hpp"
#include "CellUniverse.hpp"
#include "Ellipsoid.hpp"

TEST(DerivedGeometryTest, IsValidRequiresPositiveMaxRadiusAndIters) {
    DerivedGeometry g;
    EXPECT_FALSE(g.isValid());

    g.max_a_radius = 50.0f;
    g.max_b_radius = 50.0f;
    g.max_c_radius = 50.0f;
    g.min_a_radius = 8.0f;
    g.min_b_radius = 8.0f;
    g.min_c_radius = 8.0f;
    g.perturb_reference_radius = 25.0f;
    g.x_sigma = 5.0f;
    g.y_sigma = 5.0f;
    g.z_sigma = 8.0f;
    g.position_prior_threshold = 20.0f;
    g.iterations_per_cell = 250;
    g.pca_bridge_min_side_voxels = 40;
    EXPECT_TRUE(g.isValid());
}

TEST(DerivedGeometryTest, DescribeIncludesAllFields) {
    DerivedGeometry g;
    g.max_a_radius = 80.0f;
    g.min_a_radius = 7.8f;
    g.perturb_reference_radius = 39.0f;
    g.x_sigma = 5.85f;
    g.z_sigma = 14.4f;
    g.iterations_per_cell = 240;
    g.position_prior_threshold = 31.2f;

    const std::string desc = derivedGeometryDescribe(g);
    EXPECT_NE(desc.find("max_a=80"), std::string::npos);
    EXPECT_NE(desc.find("min_a=7.8"), std::string::npos);
    EXPECT_NE(desc.find("ref_r=39"), std::string::npos);
    EXPECT_NE(desc.find("xy_sigma=5.85"), std::string::npos);
    EXPECT_NE(desc.find("z_sigma=14.4"), std::string::npos);
    EXPECT_NE(desc.find("iters=240"), std::string::npos);
}

TEST(ImageSize3DTest, DefaultZeroAndAssignable) {
    ImageSize3D s;
    EXPECT_EQ(s.width, 0);
    EXPECT_EQ(s.height, 0);
    EXPECT_EQ(s.depth, 0);

    s.width = 600;
    s.height = 460;
    s.depth = 32;
    EXPECT_EQ(s.width, 600);
    EXPECT_EQ(s.height, 460);
    EXPECT_EQ(s.depth, 32);
}

namespace {

// Force Ellipsoid::cellConfig to test-friendly bounds before each Ellipsoid
// construction. Without this, the Ellipsoid ctor clamps radii against
// cellConfig.max*Radius and throws/wrongs the input. Earlier tests in the
// suite may have set cellConfig to small bounds (e.g. test maxARadius=20),
// so we MUST reset unconditionally (not just on first encounter).
void ensureEllipsoidConfigDefaults() {
    Ellipsoid::cellConfig.minARadius = 1.0;
    Ellipsoid::cellConfig.maxARadius = 200.0;
    Ellipsoid::cellConfig.minBRadius = 1.0;
    Ellipsoid::cellConfig.maxBRadius = 200.0;
    Ellipsoid::cellConfig.minCRadius = 1.0;
    Ellipsoid::cellConfig.maxCRadius = 200.0;
    Ellipsoid::cellConfig.maxZ = 1000.0f;
    Ellipsoid::cellConfig.minBrightness = 0.0;
    Ellipsoid::cellConfig.maxBrightness = 1.0;
}

Ellipsoid makeEll(const std::string &name, float ra, float rb, float rc, bool trash = false) {
    ensureEllipsoidConfigDefaults();
    EllipsoidParams p;
    p.name = name;
    p.x = 100.0f; p.y = 100.0f; p.z = 100.0f;
    p.aRadius = ra; p.bRadius = rb; p.cRadius = rc;
    p.theta_x = 0; p.theta_y = 0; p.theta_z = 0;
    p.brightness = 1.0f;
    p.isTrash = trash;
    return Ellipsoid(p);
}

}  // namespace

TEST(ComputeDerivedGeometryTest, FluoLikeFourCellsProducesExpectedFormulas) {
    BaseConfig cfg;
    cfg.simulation.auto_derive_geometry_enabled = true;

    // Fluo-like cells. Use cR == aR to avoid Ellipsoid ctor's oblate-invariant
    // clamp (`if (c > a) c = a`). In production, the fluo CSV has cR=48, aR=40
    // and PCA shape fit grows cR back after construction; for a pure unit
    // test of the formulas, equal radii avoid the clamp without losing
    // coverage of the formula behavior.
    std::vector<Ellipsoid> cells = {
        makeEll("cell_1", 40.0f, 40.0f, 40.0f),
        makeEll("cell_2", 40.0f, 40.0f, 40.0f),
        makeEll("cell_3", 38.0f, 38.0f, 38.0f),
        makeEll("cell_4", 38.0f, 38.0f, 38.0f),
        makeEll("trash_1", 10.0f, 10.0f, 10.0f, /*trash=*/true),
    };
    ImageSize3D image_size;
    image_size.width = 600; image_size.height = 460; image_size.depth = 32;

    DerivedGeometry g = CellUniverse::computeDerivedGeometry(cells, image_size, cfg);

    EXPECT_NEAR(g.mean_a_radius, 39.0f, 1e-3f);
    EXPECT_NEAR(g.mean_b_radius, 39.0f, 1e-3f);
    EXPECT_NEAR(g.mean_c_radius, 39.0f, 1e-3f);
    EXPECT_EQ(g.cell_count, 4u);

    EXPECT_NEAR(g.max_a_radius, 80.0f, 1e-3f);
    EXPECT_NEAR(g.max_b_radius, 80.0f, 1e-3f);
    EXPECT_NEAR(g.max_c_radius, 80.0f, 1e-3f);
    EXPECT_NEAR(g.min_a_radius, 7.8f, 1e-3f);
    EXPECT_NEAR(g.min_b_radius, 7.8f, 1e-3f);
    EXPECT_NEAR(g.min_c_radius, 7.8f, 1e-3f);
    EXPECT_NEAR(g.perturb_reference_radius, 39.0f, 1e-3f);
    EXPECT_NEAR(g.x_sigma, 5.85f, 1e-3f);
    EXPECT_NEAR(g.y_sigma, 5.85f, 1e-3f);
    EXPECT_NEAR(g.z_sigma, 11.7f, 1e-3f);   // mean_c × 0.30 = 39 × 0.30
    EXPECT_NEAR(g.position_prior_threshold, 31.2f, 1e-3f);
    EXPECT_EQ(g.iterations_per_cell, 150);
    EXPECT_TRUE(g.isValid());
}

TEST(ComputeDerivedGeometryTest, EmbryoLikeSixCellsProducesExpectedFormulas) {
    BaseConfig cfg;
    cfg.simulation.auto_derive_geometry_enabled = true;

    std::vector<Ellipsoid> cells = {
        makeEll("e3d03", 14.5f, 14.5f, 14.5f),
        makeEll("12345", 30.0f, 30.0f, 30.0f),
        makeEll("e9077", 28.0f, 28.0f, 28.0f),
        makeEll("1f2ed", 29.6f, 29.6f, 29.6f),
        makeEll("1f89ab", 31.1f, 31.1f, 31.1f),
        makeEll("8cbdf", 18.4f, 18.4f, 18.4f),
    };
    ImageSize3D image_size;
    image_size.width = 600; image_size.height = 460; image_size.depth = 32;

    DerivedGeometry g = CellUniverse::computeDerivedGeometry(cells, image_size, cfg);

    EXPECT_NEAR(g.mean_a_radius, 25.27f, 0.05f);
    EXPECT_NEAR(g.max_a_radius, 62.2f, 0.05f);
    EXPECT_NEAR(g.min_a_radius, 5.05f, 0.02f);
    EXPECT_NEAR(g.x_sigma, 3.79f, 0.02f);
    EXPECT_NEAR(g.z_sigma, 7.58f, 0.02f);
    EXPECT_EQ(g.iterations_per_cell, 180);
    EXPECT_NEAR(g.position_prior_threshold, 20.21f, 0.05f);
}

TEST(ComputeDerivedGeometryTest, ManualOverridesReplaceDerivedValues) {
    BaseConfig cfg;
    cfg.simulation.auto_derive_geometry_enabled = true;
    cfg.simulation.geometry_force_max_a_radius = 999.0f;
    cfg.simulation.geometry_force_xy_sigma = 7.7f;
    cfg.simulation.geometry_force_iterations_per_cell = 333;

    std::vector<Ellipsoid> cells = {
        makeEll("cell_1", 40.0f, 40.0f, 48.0f),
    };
    ImageSize3D image_size;
    image_size.width = 600; image_size.height = 460; image_size.depth = 32;

    DerivedGeometry g = CellUniverse::computeDerivedGeometry(cells, image_size, cfg);

    EXPECT_FLOAT_EQ(g.max_a_radius, 999.0f);
    EXPECT_FLOAT_EQ(g.x_sigma, 7.7f);
    EXPECT_FLOAT_EQ(g.y_sigma, 7.7f);
    EXPECT_EQ(g.iterations_per_cell, 333);
    EXPECT_NEAR(g.min_a_radius, 8.0f, 1e-3f);
    EXPECT_NEAR(g.position_prior_threshold, 32.0f, 1e-3f);
}

TEST(ComputeDerivedGeometryTest, EmptyCellsReturnsInvalidGeometry) {
    BaseConfig cfg;
    cfg.simulation.auto_derive_geometry_enabled = true;

    std::vector<Ellipsoid> cells;
    ImageSize3D image_size;
    image_size.width = 600; image_size.height = 460; image_size.depth = 32;

    DerivedGeometry g = CellUniverse::computeDerivedGeometry(cells, image_size, cfg);
    EXPECT_FALSE(g.isValid());
}

TEST(ComputeDerivedGeometryTest, IterationsClampedToMinAndMax) {
    BaseConfig cfg;
    cfg.simulation.auto_derive_geometry_enabled = true;

    std::vector<Ellipsoid> oneCell = { makeEll("c", 30.0f, 30.0f, 30.0f) };
    ImageSize3D imgSize;
    imgSize.width = 600; imgSize.height = 460; imgSize.depth = 32;
    DerivedGeometry g1 = CellUniverse::computeDerivedGeometry(oneCell, imgSize, cfg);
    EXPECT_EQ(g1.iterations_per_cell, 150);

    std::vector<Ellipsoid> manyCells;
    for (int i = 0; i < 30; ++i) {
        manyCells.push_back(makeEll("c" + std::to_string(i), 30.0f, 30.0f, 30.0f));
    }
    DerivedGeometry g30 = CellUniverse::computeDerivedGeometry(manyCells, imgSize, cfg);
    EXPECT_EQ(g30.iterations_per_cell, 500);
}

TEST(ApplyDerivedGeometryTest, MutatesEllipsoidCellConfigAndBaseConfig) {
    // Save originals to restore after test (Ellipsoid::cellConfig is a global
    // static — leaking modifications would pollute later tests in the suite).
    EllipsoidConfig savedCellCfg = Ellipsoid::cellConfig;

    BaseConfig cfg;
    cfg.cell = std::make_unique<EllipsoidConfig>();
    cfg.cell->maxARadius = 70.0;  // pre-derive value
    cfg.cell->minARadius = 5.0;
    cfg.cell->perturbSigmaReferenceRadius = 25.0f;
    cfg.cell->x.sigma = 5.0f;
    cfg.cell->y.sigma = 5.0f;
    cfg.cell->z.sigma = 8.0f;
    Ellipsoid::cellConfig = *cfg.cell;

    DerivedGeometry g;
    g.max_a_radius = 80.0f;
    g.max_b_radius = 80.0f;
    g.max_c_radius = 96.0f;
    g.min_a_radius = 7.8f;
    g.min_b_radius = 7.8f;
    g.min_c_radius = 9.6f;
    g.perturb_reference_radius = 39.0f;
    g.x_sigma = 5.85f;
    g.y_sigma = 5.85f;
    g.z_sigma = 14.4f;
    g.position_prior_threshold = 31.2f;
    g.iterations_per_cell = 240;
    g.pca_bridge_min_side_voxels = 60;
    g.cell_count = 4;
    g.mean_a_radius = 39.0f;
    g.mean_b_radius = 39.0f;
    g.mean_c_radius = 48.0f;

    CellUniverse::applyDerivedGeometry(g, cfg);

    // BaseConfig.cell.* mutated
    EXPECT_DOUBLE_EQ(cfg.cell->maxARadius, 80.0);
    // min_a_radius=7.8 is not exactly representable in either float or
    // double; the value flows through float (DerivedGeometry.min_a_radius)
    // before assignment to a double field, so compare via float promotion.
    EXPECT_DOUBLE_EQ(cfg.cell->minARadius, static_cast<double>(7.8f));
    EXPECT_DOUBLE_EQ(cfg.cell->maxCRadius, 96.0);
    EXPECT_FLOAT_EQ(cfg.cell->perturbSigmaReferenceRadius, 39.0f);
    EXPECT_FLOAT_EQ(cfg.cell->x.sigma, 5.85f);
    EXPECT_FLOAT_EQ(cfg.cell->y.sigma, 5.85f);
    EXPECT_FLOAT_EQ(cfg.cell->z.sigma, 14.4f);

    // Ellipsoid::cellConfig (the global static) ALSO mutated
    EXPECT_DOUBLE_EQ(Ellipsoid::cellConfig.maxARadius, 80.0);
    EXPECT_FLOAT_EQ(Ellipsoid::cellConfig.x.sigma, 5.85f);

    // SimulationConfig.iterations_per_cell mutated
    EXPECT_EQ(cfg.simulation.iterations_per_cell, 240);

    // ProbabilityConfig.position_prior_threshold mutated
    EXPECT_FLOAT_EQ(cfg.prob.position_prior_threshold, 31.2f);
    // pca_bridge_min_side_voxels mutated
    EXPECT_EQ(cfg.prob.pca_bridge_min_side_voxels, 60);

    // Restore for other tests
    Ellipsoid::cellConfig = savedCellCfg;
}
