#include <gtest/gtest.h>

#include "Ellipsoid.hpp"

namespace {
void ConfigureEllipsoidBounds() {
    Ellipsoid::cellConfig.minARadius = 1.0;
    Ellipsoid::cellConfig.maxARadius = 10.0;
    Ellipsoid::cellConfig.minCRadius = 0.5;
    Ellipsoid::cellConfig.maxCRadius = 8.0;
}
}

TEST(EllipsoidTest, ConstructorClampsRadiiToConfiguredBounds) {
    ConfigureEllipsoidBounds();

    Ellipsoid spheroid(EllipsoidParams("cellA", 0.0f, 0.0f, 0.0f, 20.0f, 0.1f));
    EllipsoidParams params = spheroid.getCellParams();

    EXPECT_DOUBLE_EQ(params.aRadius, 10.0);
    EXPECT_DOUBLE_EQ(params.cRadius, 0.5);
    EXPECT_TRUE(spheroid.checkConstraints());
}

TEST(EllipsoidTest, ConstructorEnforcesMinorRadiusNotGreaterThanMajor) {
    ConfigureEllipsoidBounds();

    Ellipsoid spheroid(EllipsoidParams("cellB", 0.0f, 0.0f, 0.0f, 2.0f, 5.0f));
    EllipsoidParams params = spheroid.getCellParams();

    EXPECT_DOUBLE_EQ(params.aRadius, 2.0);
    EXPECT_DOUBLE_EQ(params.cRadius, 2.0);
}

TEST(EllipsoidTest, DrawColorsCenterPixelAndLeavesFarPixelUnchanged) {
    ConfigureEllipsoidBounds();

    SimulationConfig simulationConfig;
    constexpr float kCellBrightness = 0.9f;
    constexpr float kBackground = 0.1f;

    cv::Mat image(21, 21, CV_32F, cv::Scalar(kBackground));
    // Pass per-cell brightness explicitly; draw() reads _brightness, not config.
    Ellipsoid spheroid(EllipsoidParams("cellC", 10.0f, 10.0f, 0.0f, 3.0f, 3.0f, 0.0f, 0.0f, 0.0f, kCellBrightness));

    spheroid.draw(image, simulationConfig, 0.0f);

    EXPECT_NEAR(image.at<float>(10, 10), kCellBrightness, 1e-6f);
    EXPECT_NEAR(image.at<float>(0, 0), kBackground, 1e-6f);
}

