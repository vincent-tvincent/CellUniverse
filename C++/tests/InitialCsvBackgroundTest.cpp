#include "BackgroundRegionTracker.hpp"
#include "CsvHandler.hpp"

#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace
{
void require(bool condition, const std::string &message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void requireNear(float actual,
                 float expected,
                 float tolerance,
                 const std::string &message)
{
    if (!std::isfinite(actual) ||
        std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(
            message + ": expected " + std::to_string(expected) +
            ", got " + std::to_string(actual));
    }
}

void testSchemaV2Fixture(const fs::path &fixtureDir)
{
    const fs::path csv = fixtureDir / "initial_schema_v2.csv";
    const InitialCsvMetadata metadata =
        CsvHandler::loadInitialCsvMetadata(csv.string());
    require(metadata.present, "schema-v2 metadata was not detected");
    require(metadata.initializerSchemaVersion == 2,
            "wrong initializer schema version");
    requireNear(metadata.zInterpolationRatio, 4.0f, 1.0e-6f,
                "wrong interpolation ratio");
    require(metadata.zCoordinateSpace == "scaled",
            "wrong coordinate space");

    const InitialCsvDocument document = CsvHandler::loadInitialCsv(
        csv.string(), "test.tif", 7.0f, 1.0f, "raw");
    require(document.hasBackground,
            "schema-v2 background record was not parsed");
    require(document.cells.size() == 1,
            "background record leaked into normal cells");
    require(document.background.name == "embryo_envelope",
            "wrong background record");
    requireNear(document.background.z, 10.0f, 1.0e-6f,
                "scaled schema-v2 z was rescaled");
    require(document.cells.front().hasRotation,
            "cell rotation was not preserved");
    require(document.cells.front().hasBrightness,
            "cell brightness was not preserved");
    requireNear(document.cells.front().thetaZ, 0.3f, 1.0e-6f,
                "cell theta_z mismatch");
    requireNear(document.cells.front().brightness, 0.9f, 1.0e-6f,
                "cell brightness mismatch");
}

void testLegacyFallback(const fs::path &fixtureDir)
{
    const fs::path csv = fixtureDir / "initial_legacy.csv";
    const InitialCsvMetadata metadata =
        CsvHandler::loadInitialCsvMetadata(csv.string());
    require(!metadata.present,
            "legacy CSV was incorrectly treated as schema-v2");
    const InitialCsvDocument document = CsvHandler::loadInitialCsv(
        csv.string(), "test.tif", 7.0f, 1.0f, "raw");
    require(!document.hasBackground,
            "legacy CSV unexpectedly created a background envelope");
    require(document.cells.size() == 1,
            "legacy cell count changed");
    requireNear(document.cells.front().z, 21.0f, 1.0e-6f,
                "legacy configured z scaling changed");
    require(!document.cells.front().hasRotation,
            "legacy cell unexpectedly acquired a rotation");
    require(!document.cells.front().hasBrightness,
            "legacy cell unexpectedly acquired a brightness override");
}

void testFutureSchemaRejected(const fs::path &fixtureDir)
{
    bool threw = false;
    try {
        (void)CsvHandler::loadInitialCsvMetadata(
            (fixtureDir / "initial_schema_future.csv").string());
    } catch (const std::runtime_error &) {
        threw = true;
    }
    require(threw, "future initializer schema was not rejected");
}

void testSoftBackgroundAndFreeze()
{
    BackgroundRegionTracker::SeedRecord seed;
    seed.center = cv::Point3f(10.0f, 10.0f, 10.0f);
    seed.radii = cv::Vec3f(10.0f, 8.0f, 6.0f);
    seed.rotation = cv::Vec3f(
        0.0f, 0.0f, static_cast<float>(CV_PI * 0.5));
    seed.coldBackground = 0.1f;
    seed.hotBackground = 0.5f;
    seed.softMargin = 0.1f;

    BackgroundRegionTracker tracker(seed);
    requireNear(tracker.membershipAt(seed.center), 1.0f, 1.0e-6f,
                "center membership mismatch");
    // With theta_z=pi/2, local +x maps to world +y.
    const cv::Point3f boundary(10.0f, 20.0f, 10.0f);
    requireNear(tracker.membershipAt(boundary), 0.5f, 1.0e-5f,
                "smoothstep boundary membership mismatch");
    requireNear(tracker.backgroundAt(boundary), 0.3f, 1.0e-5f,
                "smoothstep boundary background mismatch");
    requireNear(
        tracker.membershipAt(cv::Point3f(10.0f, 22.0f, 10.0f)),
        0.0f, 1.0e-6f, "outside membership mismatch");

    std::vector<cv::Mat> flat(
        21, cv::Mat(21, 21, CV_32F, cv::Scalar(0.2f)));
    require(!tracker.update(0, flat, {}),
            "first confirmed frame should retain seed geometry");
    require(!tracker.update(1, flat, {}),
            "flat frame should not move the background envelope");
    require(tracker.frozen(),
            "low-confidence flat frame did not freeze geometry");
    requireNear(tracker.currentState().center.x, seed.center.x, 1.0e-6f,
                "frozen tracker changed center");

    const auto rendered = tracker.render(21, 21, 21);
    require(rendered.background.size() == 21 &&
                rendered.membership.size() == 21,
            "rendered stack depth mismatch");
    requireNear(rendered.membership[10].at<float>(20, 10),
                0.5f, 1.0e-5f,
                "rendered membership differs from point sampling");
}

void testSmallShapeChangeAccepted()
{
    BackgroundRegionTracker::SeedRecord seed;
    seed.center = cv::Point3f(32.0f, 32.0f, 20.0f);
    seed.radii = cv::Vec3f(18.0f, 14.0f, 10.0f);
    seed.rotation = cv::Vec3f(0.0f, 0.0f, 0.0f);
    seed.coldBackground = 0.1f;
    seed.hotBackground = 0.5f;
    seed.softMargin = 0.08f;

    BackgroundRegionTracker tracker(seed);
    const auto initialFrame = tracker.render(41, 64, 64).background;
    require(!tracker.update(0, initialFrame, {}),
            "confirmed first-frame geometry should be held");

    BackgroundRegionTracker::SeedRecord shifted = seed;
    shifted.center.x += 1.5f;
    BackgroundRegionTracker shiftedTruth(shifted);
    const auto shiftedFrame =
        shiftedTruth.render(41, 64, 64).background;
    require(tracker.update(1, shiftedFrame, {}),
            "small high-confidence envelope shift was not accepted");
    require(!tracker.frozen(),
            "accepted small shift was marked frozen");
    require(tracker.confidence() >=
                tracker.options().minimumConfidence,
            "accepted shift did not meet confidence threshold");
    require(tracker.currentState().center.x > seed.center.x + 0.01f &&
                tracker.currentState().center.x <
                    shifted.center.x + 0.25f,
            "accepted center update did not move conservatively "
            "toward the shifted envelope: x=" +
                std::to_string(tracker.currentState().center.x));
    requireNear(tracker.currentState().rotation[0],
                seed.rotation[0], 1.0e-7f,
                "tracker changed fixed theta_x");
    requireNear(tracker.currentState().rotation[1],
                seed.rotation[1], 1.0e-7f,
                "tracker changed fixed theta_y");
    requireNear(tracker.currentState().rotation[2],
                seed.rotation[2], 1.0e-7f,
                "tracker changed fixed theta_z");
}

void validatePavakCsvs(int argc, char **argv)
{
    const std::array<int, 9> expectedCounts{
        6, 9, 6, 4, 7, 5, 7, 5, 4};
    for (int arg = 1; arg < argc; ++arg) {
        const fs::path csv(argv[arg]);
        const InitialCsvMetadata metadata =
            CsvHandler::loadInitialCsvMetadata(csv.string());
        require(metadata.present &&
                    metadata.initializerSchemaVersion == 2,
                csv.string() + ": missing schema-v2 metadata");
        requireNear(metadata.zInterpolationRatio, 4.0f, 1.0e-6f,
                    csv.string() + ": unexpected z ratio");
        require(metadata.zCoordinateSpace == "scaled",
                csv.string() + ": unexpected z coordinate space");
        const InitialCsvDocument document = CsvHandler::loadInitialCsv(
            csv.string(), "SPIMA_t001.tif", 4.0f, 1.0f, "scaled");
        require(document.hasBackground &&
                    document.background.name == "embryo_envelope",
                csv.string() + ": missing embryo envelope");

        const std::string position =
            csv.parent_path().filename().string();
        if (position.size() == 4 &&
            position.rfind("Pos", 0) == 0 &&
            position[3] >= '0' && position[3] <= '8') {
            const int index = position[3] - '0';
            require(
                static_cast<int>(document.cells.size()) ==
                    expectedCounts[static_cast<size_t>(index)],
                csv.string() + ": unexpected normal-cell count");
        }
        for (const InitialCellRecord &cell : document.cells) {
            require(!cell.cellId.empty() &&
                        cell.hasRotation &&
                        cell.hasBrightness,
                    csv.string() +
                        ": incomplete schema-v2 cell metadata");
        }
        std::cout << "[PASS] " << csv.string()
                  << " cells=" << document.cells.size()
                  << " ratio=" << metadata.zInterpolationRatio
                  << " zSpace=" << metadata.zCoordinateSpace
                  << '\n';
    }
}
}

int main(int argc, char **argv)
{
    try {
        if (argc > 1) {
            validatePavakCsvs(argc, argv);
            return 0;
        }

        const fs::path fixtureDir(CELLUNIVERSE_TEST_FIXTURE_DIR);
        testSchemaV2Fixture(fixtureDir);
        testLegacyFallback(fixtureDir);
        testFutureSchemaRejected(fixtureDir);
        testSoftBackgroundAndFreeze();
        testSmallShapeChangeAccepted();
        std::cout << "[PASS] initial CSV/background focused tests\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "[FAIL] " << error.what() << '\n';
        return 1;
    }
}
