#ifndef COMPACTEXPORTER_HPP
#define COMPACTEXPORTER_HPP

#include "BackgroundRegionTracker.hpp"
#include "Frame.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace celluniverse::compact
{

enum class BackgroundKind
{
    Scalar,
    AnalyticEllipsoid,
    BinaryMask
};

struct AnalyticEllipsoidBackground
{
    cv::Point3f center{0.0f, 0.0f, 0.0f};
    cv::Vec3f radii{1.0f, 1.0f, 1.0f};
    cv::Vec3f rotation{0.0f, 0.0f, 0.0f};
    float cold = 0.0f;
    float hot = 0.0f;
    float softMargin = 0.0f;

    // Frame::addBackgroundOffset() changes the installed background field
    // without changing BackgroundRegionTracker::State. Reconstruction applies
    // this offset after cold/hot interpolation and clamps the result to [0,1].
    float additiveOffset = 0.0f;
    // Apply each delta in order with a clamp after every update. This retains
    // exact behavior when signed updates cross a saturation boundary.
    std::vector<float> offsetUpdates;
};

struct CompactBackground
{
    BackgroundKind kind = BackgroundKind::Scalar;
    float scalar = 0.0f;
    AnalyticEllipsoidBackground analytic;

    // Binary backgrounds use a z-major/y-major/x-minor, LSB-first bitset.
    // A zero bit selects cold and a one bit selects hot.
    float cold = 0.0f;
    float hot = 0.0f;
    int width = 0;
    int height = 0;
    int depth = 0;
    std::vector<std::uint8_t> maskBits;
};

struct CompactCell
{
    std::uint32_t drawOrder = 0;
    std::string name;
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    float aRadius = 0.0f;
    float bRadius = 0.0f;
    float cRadius = 0.0f;
    float thetaX = 0.0f;
    float thetaY = 0.0f;
    float thetaZ = 0.0f;
    float brightness = 0.0f;
    bool isTrash = false;
};

struct CompactFrameRecord
{
    int frame = 0;
    std::string sourceFrame;
    std::string pipelineMode;
    int width = 0;
    int height = 0;
    int depth = 0;
    float zInterpolationRatio = 1.0f;
    std::string zInterpolationSource = "config";
    std::string initialZSpace = "auto";
    CompactBackground background;
    std::vector<CompactCell> cells;
};

class CompactExporter
{
public:
    static constexpr int FormatVersion = 1;

    // Valid values are full, compact, and both. normalizeExportMode throws for
    // any other value so an invalid job cannot silently omit expected output.
    static std::string normalizeExportMode(const std::string &mode);
    static bool writesFull(const std::string &mode);
    static bool writesCompact(const std::string &mode);

    // Starts one compact-export session for outputRoot. Fresh sessions publish
    // only frames written by the current process, so stale frame JSON left in a
    // reused output directory is not advertised. For an in-place checkpoint
    // resume, pass resume_from as preserveFramesBefore; only records with
    // frame < resume_from are retained. Pass 0 for a fresh session.
    static void beginRun(const std::filesystem::path &outputRoot,
                         int preserveFramesBefore);

    // Captures the final, ordered state used by Frame::generateSynthFrame().
    // Pass analyticBackground only when the installed Frame background came
    // from that tracker state. Other spatial fields are inspected and encoded
    // as a scalar or exact two-value binary mask.
    static CompactFrameRecord captureFrame(
        const Frame &frame,
        int absoluteFrame,
        const std::string &pipelineMode,
        float zInterpolationRatio,
        const std::string &zInterpolationSource,
        const std::string &initialZSpace,
        const BackgroundRegionTracker::State *analyticBackground = nullptr);

    // Writes <outputRoot>/compact/frames/frame_XXXXXX.json, any deduplicated
    // CUBM mask, and an atomically refreshed compact/manifest.json. beginRun()
    // must be called once for outputRoot before the first frame is written.
    static void writeFrame(const std::filesystem::path &outputRoot,
                           const CompactFrameRecord &record);
};

} // namespace celluniverse::compact

#endif // COMPACTEXPORTER_HPP
