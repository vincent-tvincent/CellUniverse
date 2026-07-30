#ifndef BACKGROUND_REGION_TRACKER_HPP
#define BACKGROUND_REGION_TRACKER_HPP

#include <cstddef>
#include <vector>

#include <opencv2/core.hpp>

#include "Ellipsoid.hpp"

// Conservative tracker for a rotated ellipsoidal two-region background.
//
// Coordinates use CellUniverse world order (x, y, z). Rotation follows the
// same convention as Ellipsoid: R = Rz * Ry * Rx and local = R^T * world.
// Rotation and soft-margin width remain fixed after configure(); only center,
// radii, and the two scalar background levels may change.
class BackgroundRegionTracker
{
public:
    struct SeedRecord
    {
        cv::Point3f center{0.0f, 0.0f, 0.0f};
        cv::Vec3f radii{1.0f, 1.0f, 1.0f};
        cv::Vec3f rotation{0.0f, 0.0f, 0.0f}; // theta_x, theta_y, theta_z
        float coldBackground = 0.0f;
        float hotBackground = 0.0f;
        float softMargin = 0.05f; // normalized radial distance
    };

    struct Options
    {
        // Geometry evidence is collected only in this normalized-radial band
        // around the predicted ellipsoid surface.
        float shellHalfWidth = 0.12f;
        int geometryStrideXY = 2;
        int geometryStrideZ = 1;
        float gaussianSigmaXY = 2.0f;
        bool smoothAcrossZ = true;
        float minimumOrientedGradient = 0.002f;
        std::size_t minimumEvidenceSamples = 256;
        std::size_t maximumEvidenceSamples = 100000;
        int minimumCoveredFaces = 4; // among +/- local x, y, z
        float minimumEvidenceFraction = 0.01f;
        float minimumConfidence = 0.25f;

        // Fixed-rotation, center/log-radius robust fit.
        int irlsIterations = 3;
        float huberScale = 1.5f;
        float geometryEmaAlpha = 0.25f;
        float maximumCenterShiftFraction = 0.03f;
        float maximumCenterShiftPixels = 3.0f;
        float maximumRadiusChangeFraction = 0.03f;
        float maximumRawUpdateMultiplier = 3.0f;
        float minimumRadius = 1.0f;
        float cellExclusionScale = 1.25f;

        // Robust hot/cold estimation outside cell supports.
        int intensityStride = 3;
        std::size_t minimumIntensitySamples = 256;
        std::size_t maximumIntensitySamplesPerRegion = 200000;
        float intensityTrimFraction = 0.10f;
        float intensityEmaAlpha = 0.25f;
        float maximumIntensityStep = 0.05f;
        float hotMembershipMinimum = 0.95f;
        float coldMembershipMaximum = 0.05f;
        float minimumRegionContrast = 0.002f;
        bool clampBackgroundToUnitRange = true;

        // The manually confirmed seed is authoritative for its source frame.
        bool holdSeedGeometryOnFirstUpdate = true;
        bool holdSeedIntensitiesOnFirstUpdate = true;
    };

    struct State
    {
        cv::Point3f center{0.0f, 0.0f, 0.0f};
        cv::Vec3f radii{1.0f, 1.0f, 1.0f};
        cv::Vec3f rotation{0.0f, 0.0f, 0.0f};
        float coldBackground = 0.0f;
        float hotBackground = 0.0f;
        float softMargin = 0.05f;

        int frameIndex = -1;
        float confidence = 0.0f;
        bool frozen = true; // geometry retained from the previous state

        std::size_t shellSamples = 0;
        std::size_t evidenceSamples = 0;
        std::size_t hotSamples = 0;
        std::size_t coldSamples = 0;
    };

    struct RenderedStacks
    {
        std::vector<cv::Mat> background; // CV_32F
        std::vector<cv::Mat> membership; // CV_32F, 0=cold and 1=hot
    };

    BackgroundRegionTracker() = default;
    explicit BackgroundRegionTracker(const SeedRecord &seed);
    BackgroundRegionTracker(const SeedRecord &seed, const Options &options);

    void configure(const SeedRecord &seed);
    void configure(const SeedRecord &seed, const Options &options);
    void reset();

    bool isConfigured() const noexcept { return configured_; }

    // Returns true only when this call accepted a geometry update. A false
    // return means the previous geometry was conservatively retained. Robust
    // hot/cold updates may still be accepted while geometry is frozen.
    bool update(const std::vector<cv::Mat> &currentFrame,
                const std::vector<Ellipsoid> &cells);
    bool update(int frameIndex,
                const std::vector<cv::Mat> &currentFrame,
                const std::vector<Ellipsoid> &cells);

    RenderedStacks render() const;
    RenderedStacks render(int zSlices, int rows, int cols) const;
    void render(int zSlices,
                int rows,
                int cols,
                std::vector<cv::Mat> &background,
                std::vector<cv::Mat> &membership) const;

    float membershipAt(const cv::Point3f &worldPoint) const;
    float backgroundAt(const cv::Point3f &worldPoint) const;
    float membershipAt(int z, int y, int x) const;
    float backgroundAt(int z, int y, int x) const;

    const State &currentState() const noexcept { return state_; }
    const SeedRecord &seedRecord() const noexcept { return seed_; }
    const Options &options() const noexcept { return options_; }
    float confidence() const noexcept { return state_.confidence; }
    bool frozen() const noexcept { return state_.frozen; }
    std::size_t updatesSeen() const noexcept { return updatesSeen_; }

private:
    SeedRecord seed_{};
    Options options_{};
    State state_{};
    bool configured_ = false;
    std::size_t updatesSeen_ = 0;
    int lastZSlices_ = 0;
    int lastRows_ = 0;
    int lastCols_ = 0;
};

#endif // BACKGROUND_REGION_TRACKER_HPP
