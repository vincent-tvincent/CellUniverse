#include "../includes/CompactExporter.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <map>
#include <mutex>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <type_traits>
#include <utility>

namespace fs = std::filesystem;

namespace celluniverse::compact
{
namespace
{

constexpr const char *kManifestSchema = "celluniverse.compact.manifest";
constexpr const char *kFrameSchema = "celluniverse.compact.frame";
constexpr std::uint16_t kCubmVersion = 1;
constexpr std::uint16_t kCubmFlagsLsbFirstZyx = 1;
constexpr std::uint64_t kFnvOffsetBasis = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
std::atomic<std::uint64_t> gTemporarySequence{0};
std::mutex gManifestSessionMutex;
std::map<std::string, std::map<int, std::string>> gManifestSessionFrames;

std::string lowerCopy(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) {
                       return static_cast<char>(std::tolower(ch));
                   });
    return value;
}

void requireFinite(float value, const char *field)
{
    if (!std::isfinite(value)) {
        throw std::runtime_error(
            std::string("compact export requires finite ") + field);
    }
}

std::string jsonEscape(const std::string &value)
{
    static constexpr char hex[] = "0123456789abcdef";
    std::string escaped;
    escaped.reserve(value.size() + 8);
    for (unsigned char ch : value) {
        switch (ch) {
        case '"': escaped += "\\\""; break;
        case '\\': escaped += "\\\\"; break;
        case '\b': escaped += "\\b"; break;
        case '\f': escaped += "\\f"; break;
        case '\n': escaped += "\\n"; break;
        case '\r': escaped += "\\r"; break;
        case '\t': escaped += "\\t"; break;
        default:
            if (ch < 0x20U) {
                escaped += "\\u00";
                escaped.push_back(hex[(ch >> 4U) & 0x0fU]);
                escaped.push_back(hex[ch & 0x0fU]);
            } else {
                escaped.push_back(static_cast<char>(ch));
            }
            break;
        }
    }
    return escaped;
}

template <typename Writer>
void writeAtomic(const fs::path &path,
                 std::ios::openmode mode,
                 Writer &&writer)
{
    fs::create_directories(path.parent_path());
    const auto ticks = std::chrono::steady_clock::now()
                           .time_since_epoch()
                           .count();
    const auto threadId =
        std::hash<std::thread::id>{}(std::this_thread::get_id());
    const auto sequence = gTemporarySequence.fetch_add(
        1, std::memory_order_relaxed);
    std::ostringstream temporaryName;
    temporaryName << path.string() << ".tmp." << ticks << '.'
                  << threadId << '.' << sequence;
    const fs::path temporary = temporaryName.str();

    try {
        {
            std::ofstream out(temporary, mode | std::ios::trunc);
            if (!out.is_open()) {
                throw std::runtime_error(
                    "unable to open compact export temporary file: " +
                    temporary.string());
            }
            writer(out);
            out.flush();
            if (!out.good()) {
                throw std::runtime_error(
                    "failed while writing compact export temporary file: " +
                    temporary.string());
            }
        }

        std::error_code ec;
        // On the Linux/NFS production target, rename within one directory is
        // atomic and replaces an existing destination. The fallback supports
        // standard-library implementations that reject replacement.
        fs::rename(temporary, path, ec);
        if (ec) {
            std::error_code removeError;
            fs::remove(path, removeError);
            ec.clear();
            fs::rename(temporary, path, ec);
        }
        if (ec) {
            throw std::runtime_error(
                "unable to publish compact export file " + path.string() +
                ": " + ec.message());
        }
    } catch (...) {
        std::error_code cleanupError;
        fs::remove(temporary, cleanupError);
        throw;
    }
}

void fnvByte(std::uint64_t &hash, std::uint8_t value)
{
    hash ^= static_cast<std::uint64_t>(value);
    hash *= kFnvPrime;
}

template <typename UInt>
void fnvLittleEndian(std::uint64_t &hash, UInt value)
{
    static_assert(std::is_unsigned<UInt>::value,
                  "FNV integer input must be unsigned");
    for (std::size_t i = 0; i < sizeof(UInt); ++i) {
        fnvByte(hash, static_cast<std::uint8_t>(
                          (value >> (8U * i)) & static_cast<UInt>(0xffU)));
    }
}

template <typename UInt>
void writeLittleEndian(std::ostream &out, UInt value)
{
    static_assert(std::is_unsigned<UInt>::value,
                  "CUBM integer fields must be unsigned");
    for (std::size_t i = 0; i < sizeof(UInt); ++i) {
        out.put(static_cast<char>(
            (value >> (8U * i)) & static_cast<UInt>(0xffU)));
    }
}

std::uint64_t maskHash(const CompactBackground &background)
{
    std::uint64_t hash = kFnvOffsetBasis;
    fnvLittleEndian(hash, static_cast<std::uint32_t>(background.width));
    fnvLittleEndian(hash, static_cast<std::uint32_t>(background.height));
    fnvLittleEndian(hash, static_cast<std::uint32_t>(background.depth));
    for (std::uint8_t byte : background.maskBits) {
        fnvByte(hash, byte);
    }
    return hash;
}

std::string maskFileName(const CompactBackground &background)
{
    std::ostringstream name;
    name << "mask_" << std::hex << std::setfill('0') << std::setw(16)
         << maskHash(background) << ".cubm";
    return name.str();
}

void writeCubmIfNeeded(const fs::path &maskPath,
                       const CompactBackground &background)
{
    const std::uint64_t voxelCount =
        static_cast<std::uint64_t>(background.width) *
        static_cast<std::uint64_t>(background.height) *
        static_cast<std::uint64_t>(background.depth);
    const std::uint64_t expectedPayloadBytes = (voxelCount + 7ULL) / 8ULL;
    if (background.maskBits.size() != expectedPayloadBytes) {
        throw std::runtime_error(
            "compact binary background bit count does not match dimensions");
    }

    constexpr std::uintmax_t headerBytes =
        4U + sizeof(std::uint16_t) + sizeof(std::uint16_t) +
        3U * sizeof(std::uint32_t) + sizeof(std::uint64_t);
    const std::uintmax_t expectedBytes =
        headerBytes + static_cast<std::uintmax_t>(background.maskBits.size());

    std::error_code ec;
    if (fs::exists(maskPath, ec) && !ec &&
        fs::file_size(maskPath, ec) == expectedBytes && !ec) {
        return;
    }

    writeAtomic(maskPath, std::ios::binary,
                [&](std::ostream &out) {
                    out.write("CUBM", 4);
                    writeLittleEndian(out, kCubmVersion);
                    writeLittleEndian(out, kCubmFlagsLsbFirstZyx);
                    writeLittleEndian(
                        out, static_cast<std::uint32_t>(background.width));
                    writeLittleEndian(
                        out, static_cast<std::uint32_t>(background.height));
                    writeLittleEndian(
                        out, static_cast<std::uint32_t>(background.depth));
                    writeLittleEndian(out, voxelCount);
                    out.write(
                        reinterpret_cast<const char *>(
                            background.maskBits.data()),
                        static_cast<std::streamsize>(
                            background.maskBits.size()));
                });
}

CompactBackground captureInstalledBackground(
    const Frame &frame,
    int width,
    int height,
    int depth)
{
    CompactBackground result;
    const auto &backgroundFrame = frame.getBackgroundFrame();
    if (backgroundFrame.empty()) {
        result.kind = BackgroundKind::Scalar;
        result.scalar = frame.getBackgroundValue();
        requireFinite(result.scalar, "scalar background");
        return result;
    }

    if (static_cast<int>(backgroundFrame.size()) != depth) {
        throw std::runtime_error(
            "compact export background depth does not match real frame");
    }

    bool haveFirst = false;
    bool haveSecond = false;
    float first = 0.0f;
    float second = 0.0f;
    for (int z = 0; z < depth; ++z) {
        const cv::Mat &slice = backgroundFrame[static_cast<std::size_t>(z)];
        if (slice.empty() || slice.type() != CV_32F ||
            slice.rows != height || slice.cols != width) {
            throw std::runtime_error(
                "compact export requires a dense CV_32F background stack "
                "matching the real frame");
        }
        for (int y = 0; y < height; ++y) {
            const float *row = slice.ptr<float>(y);
            for (int x = 0; x < width; ++x) {
                const float value = row[x];
                requireFinite(value, "background voxel");
                if (!haveFirst) {
                    first = value;
                    haveFirst = true;
                } else if (value != first &&
                           (!haveSecond || value != second)) {
                    if (haveSecond) {
                        throw std::runtime_error(
                            "non-analytic compact background contains more "
                            "than two values; provide its analytic descriptor");
                    }
                    second = value;
                    haveSecond = true;
                }
            }
        }
    }

    if (!haveFirst) {
        throw std::runtime_error(
            "compact export cannot encode an empty background field");
    }
    if (!haveSecond) {
        result.kind = BackgroundKind::Scalar;
        result.scalar = first;
        return result;
    }

    result.kind = BackgroundKind::BinaryMask;
    result.cold = std::min(first, second);
    result.hot = std::max(first, second);
    result.width = width;
    result.height = height;
    result.depth = depth;

    const std::uint64_t voxelCount =
        static_cast<std::uint64_t>(width) *
        static_cast<std::uint64_t>(height) *
        static_cast<std::uint64_t>(depth);
    result.maskBits.assign(
        static_cast<std::size_t>((voxelCount + 7ULL) / 8ULL), 0U);

    std::uint64_t linearIndex = 0;
    for (int z = 0; z < depth; ++z) {
        const cv::Mat &slice = backgroundFrame[static_cast<std::size_t>(z)];
        for (int y = 0; y < height; ++y) {
            const float *row = slice.ptr<float>(y);
            for (int x = 0; x < width; ++x, ++linearIndex) {
                const float value = row[x];
                if (value != result.cold && value != result.hot) {
                    throw std::runtime_error(
                        "binary compact background changed during capture");
                }
                if (value == result.hot) {
                    result.maskBits[static_cast<std::size_t>(
                        linearIndex >> 3U)] |= static_cast<std::uint8_t>(
                            1U << (linearIndex & 7U));
                }
            }
        }
    }
    return result;
}

void writeFloat(std::ostream &out, float value, const char *field)
{
    requireFinite(value, field);
    out << value;
}

std::string frameFileName(int frame)
{
    if (frame < 0) {
        throw std::invalid_argument(
            "compact export frame number must be non-negative");
    }
    std::ostringstream name;
    name << "frame_" << std::setfill('0') << std::setw(6) << frame
         << ".json";
    return name.str();
}

void writeBackgroundJson(std::ostream &out,
                         const CompactBackground &background,
                         const std::string &maskRelativePath)
{
    switch (background.kind) {
    case BackgroundKind::Scalar:
        out << "    \"background\": {\"kind\": \"scalar\", \"value\": ";
        writeFloat(out, background.scalar, "scalar background");
        out << "},\n";
        break;
    case BackgroundKind::AnalyticEllipsoid: {
        const auto &value = background.analytic;
        out << "    \"background\": {\n"
            << "      \"kind\": \"rotated_soft_ellipsoid\",\n"
            << "      \"center\": {\"x\": ";
        writeFloat(out, value.center.x, "background center x");
        out << ", \"y\": ";
        writeFloat(out, value.center.y, "background center y");
        out << ", \"z\": ";
        writeFloat(out, value.center.z, "background center z");
        out << "},\n      \"radii\": {\"a\": ";
        writeFloat(out, value.radii[0], "background radius a");
        out << ", \"b\": ";
        writeFloat(out, value.radii[1], "background radius b");
        out << ", \"c\": ";
        writeFloat(out, value.radii[2], "background radius c");
        out << "},\n      \"rotation\": {\"theta_x\": ";
        writeFloat(out, value.rotation[0], "background theta x");
        out << ", \"theta_y\": ";
        writeFloat(out, value.rotation[1], "background theta y");
        out << ", \"theta_z\": ";
        writeFloat(out, value.rotation[2], "background theta z");
        out << "},\n      \"cold\": ";
        writeFloat(out, value.cold, "background cold");
        out << ", \"hot\": ";
        writeFloat(out, value.hot, "background hot");
        out << ", \"soft_margin\": ";
        writeFloat(out, value.softMargin, "background soft margin");
        out << ", \"additive_offset\": ";
        writeFloat(out, value.additiveOffset,
                   "background additive offset");
        out << ",\n      \"offset_updates\": [";
        for (std::size_t index = 0;
             index < value.offsetUpdates.size(); ++index) {
            if (index > 0U) {
                out << ", ";
            }
            writeFloat(out, value.offsetUpdates[index],
                       "background offset update");
        }
        out << ']';
        out << "\n    },\n";
        break;
    }
    case BackgroundKind::BinaryMask:
        out << "    \"background\": {\n"
            << "      \"kind\": \"binary_mask\",\n"
            << "      \"cold\": ";
        writeFloat(out, background.cold, "binary background cold");
        out << ", \"hot\": ";
        writeFloat(out, background.hot, "binary background hot");
        out << ",\n      \"mask_format\": \"CUBM1\","
            << " \"mask_path\": \"" << jsonEscape(maskRelativePath)
            << "\"\n    },\n";
        break;
    }
}

void writeFrameJson(const fs::path &path,
                    const CompactFrameRecord &record,
                    const std::string &maskRelativePath)
{
    writeAtomic(path, std::ios::out,
                [&](std::ostream &out) {
                    out << std::setprecision(
                        std::numeric_limits<float>::max_digits10);
                    out << "{\n"
                        << "  \"schema\": \"" << kFrameSchema << "\",\n"
                        << "  \"version\": "
                        << CompactExporter::FormatVersion << ",\n"
                        << "  \"frame\": " << record.frame << ",\n"
                        << "  \"source_frame\": \""
                        << jsonEscape(record.sourceFrame) << "\",\n"
                        << "  \"pipeline_mode\": \""
                        << jsonEscape(record.pipelineMode) << "\",\n"
                        << "  \"dimensions\": {\"x\": " << record.width
                        << ", \"y\": " << record.height
                        << ", \"z\": " << record.depth << "},\n"
                        << "  \"coordinates\": {\n"
                        << "    \"cell_order\": \"xyz\","
                        << " \"volume_order\": \"zyx\",\n"
                        << "    \"space\": \"interpolated\","
                        << " \"z_interpolation_ratio\": ";
                    writeFloat(out, record.zInterpolationRatio,
                               "z interpolation ratio");
                    out << ",\n    \"z_interpolation_source\": \""
                        << jsonEscape(record.zInterpolationSource) << "\",\n"
                        << "    \"initial_z_space\": \""
                        << jsonEscape(record.initialZSpace) << "\"\n"
                        << "  },\n"
                        << "  \"render_contract\": {\n"
                        << "    \"id\": \"ellipsoid_rz_ry_rx_overwrite_cv8u_v1\",\n"
                        << "    \"rotation\": \"Rz*Ry*Rx\",\n"
                        << "    \"membership\": \"squared_local_radius<=1\",\n"
                        << "    \"overlap\": \"later_draw_order_overwrites\",\n"
                        << "    \"output\": \"opencv_float32_to_uint8_scale_255\"\n"
                        << "  },\n";
                    writeBackgroundJson(out, record.background,
                                        maskRelativePath);
                    out << "  \"cells\": [";
                    if (!record.cells.empty()) {
                        out << '\n';
                    }
                    for (std::size_t index = 0;
                         index < record.cells.size(); ++index) {
                        const CompactCell &cell = record.cells[index];
                        out << "    {\n"
                            << "      \"draw_order\": " << cell.drawOrder
                            << ", \"name\": \"" << jsonEscape(cell.name)
                            << "\",\n"
                            << "      \"center\": {\"x\": ";
                        writeFloat(out, cell.x, "cell x");
                        out << ", \"y\": ";
                        writeFloat(out, cell.y, "cell y");
                        out << ", \"z\": ";
                        writeFloat(out, cell.z, "cell z");
                        out << "},\n      \"radii\": {\"a\": ";
                        writeFloat(out, cell.aRadius, "cell radius a");
                        out << ", \"b\": ";
                        writeFloat(out, cell.bRadius, "cell radius b");
                        out << ", \"c\": ";
                        writeFloat(out, cell.cRadius, "cell radius c");
                        out << "},\n      \"rotation\": {\"theta_x\": ";
                        writeFloat(out, cell.thetaX, "cell theta x");
                        out << ", \"theta_y\": ";
                        writeFloat(out, cell.thetaY, "cell theta y");
                        out << ", \"theta_z\": ";
                        writeFloat(out, cell.thetaZ, "cell theta z");
                        out << "},\n      \"brightness\": ";
                        writeFloat(out, cell.brightness, "cell brightness");
                        out << ", \"is_trash\": "
                            << (cell.isTrash ? "true" : "false") << "\n"
                            << "    }";
                        if (index + 1U < record.cells.size()) {
                            out << ',';
                        }
                        out << '\n';
                    }
                    out << "  ]\n}\n";
                });
}

std::vector<std::pair<int, std::string>> readAdvertisedFrameFiles(
    const fs::path &compactRoot)
{
    const fs::path manifestPath = compactRoot / "manifest.json";
    std::error_code ec;
    if (!fs::exists(manifestPath, ec) || ec) {
        return {};
    }

    std::ifstream input(manifestPath);
    if (!input.is_open()) {
        throw std::runtime_error(
            "unable to read existing compact manifest: " +
            manifestPath.string());
    }

    const std::regex entryPattern(
        R"manifest(^\s*\{"frame": ([0-9]+), "path": "frames/(frame_([0-9]{6,})\.json)"\},?\s*$)manifest");
    bool sawSchema = false;
    bool sawVersion = false;
    bool sawFrameSchema = false;
    bool sawMaskSchema = false;
    bool sawFramesArray = false;
    std::map<int, std::string> advertised;
    std::string line;
    while (std::getline(input, line)) {
        sawSchema = sawSchema ||
            line.find(std::string("\"schema\": \"") +
                      kManifestSchema + "\"") != std::string::npos;
        sawVersion = sawVersion ||
            line.find("\"version\": 1") != std::string::npos;
        sawFrameSchema = sawFrameSchema ||
            line.find(std::string("\"frame_schema\": \"") +
                      kFrameSchema + "\"") != std::string::npos;
        sawMaskSchema = sawMaskSchema ||
            line.find("\"mask_schema\": \"CUBM1\"") != std::string::npos;
        sawFramesArray = sawFramesArray ||
            line.find("\"frames\": [") != std::string::npos;

        std::smatch match;
        if (!std::regex_match(line, match, entryPattern)) {
            continue;
        }
        int frame = 0;
        int fileFrame = 0;
        try {
            frame = std::stoi(match[1].str());
            fileFrame = std::stoi(match[3].str());
        } catch (const std::exception &) {
            throw std::runtime_error(
                "existing compact manifest contains an invalid frame number");
        }
        if (frame != fileFrame) {
            throw std::runtime_error(
                "existing compact manifest frame/path mismatch");
        }
        const std::string fileName = match[2].str();
        if (!fs::is_regular_file(compactRoot / "frames" / fileName, ec) ||
            ec) {
            throw std::runtime_error(
                "existing compact manifest advertises a missing frame: " +
                fileName);
        }
        if (!advertised.emplace(frame, fileName).second) {
            throw std::runtime_error(
                "existing compact manifest repeats frame " +
                std::to_string(frame));
        }
    }
    if (!input.eof() && input.fail()) {
        throw std::runtime_error(
            "failed while reading existing compact manifest: " +
            manifestPath.string());
    }
    if (!sawSchema || !sawVersion || !sawFrameSchema ||
        !sawMaskSchema || !sawFramesArray) {
        throw std::runtime_error(
            "existing compact manifest does not satisfy version-1 contract");
    }
    return {advertised.begin(), advertised.end()};
}

std::string manifestSessionKey(const fs::path &outputRoot)
{
    std::error_code ec;
    fs::path absolute = fs::absolute(outputRoot, ec);
    if (ec) {
        absolute = outputRoot;
    }
    return absolute.lexically_normal().string();
}

std::vector<std::pair<int, std::string>> orderedFrames(
    const std::map<int, std::string> &frames)
{
    return {frames.begin(), frames.end()};
}

void writeManifest(
    const fs::path &compactRoot,
    const std::vector<std::pair<int, std::string>> &frames)
{
    writeAtomic(compactRoot / "manifest.json", std::ios::out,
                [&](std::ostream &out) {
                    out << "{\n"
                        << "  \"schema\": \"" << kManifestSchema << "\",\n"
                        << "  \"version\": "
                        << CompactExporter::FormatVersion << ",\n"
                        << "  \"frame_schema\": \"" << kFrameSchema << "\",\n"
                        << "  \"mask_schema\": \"CUBM1\",\n"
                        << "  \"frames\": [";
                    if (!frames.empty()) {
                        out << '\n';
                    }
                    for (std::size_t index = 0; index < frames.size();
                         ++index) {
                        out << "    {\"frame\": " << frames[index].first
                            << ", \"path\": \"frames/"
                            << jsonEscape(frames[index].second) << "\"}";
                        if (index + 1U < frames.size()) {
                            out << ',';
                        }
                        out << '\n';
                    }
                    out << "  ]\n}\n";
                });
}

} // namespace

std::string CompactExporter::normalizeExportMode(const std::string &mode)
{
    const std::string normalized = lowerCopy(mode);
    if (normalized != "full" &&
        normalized != "compact" &&
        normalized != "both") {
        throw std::invalid_argument(
            "simulation.export_mode must be one of: full, compact, both");
    }
    return normalized;
}

bool CompactExporter::writesFull(const std::string &mode)
{
    const std::string normalized = normalizeExportMode(mode);
    return normalized == "full" || normalized == "both";
}

bool CompactExporter::writesCompact(const std::string &mode)
{
    const std::string normalized = normalizeExportMode(mode);
    return normalized == "compact" || normalized == "both";
}

void CompactExporter::beginRun(const fs::path &outputRoot,
                               int preserveFramesBefore)
{
    if (preserveFramesBefore < 0) {
        throw std::invalid_argument(
            "compact preserveFramesBefore must be non-negative");
    }
    const fs::path compactRoot = outputRoot / "compact";
    fs::create_directories(compactRoot / "frames");

    std::map<int, std::string> sessionFrames;
    if (preserveFramesBefore > 0) {
        for (const auto &[frame, fileName] :
             readAdvertisedFrameFiles(compactRoot)) {
            if (frame < preserveFramesBefore) {
                sessionFrames[frame] = fileName;
            }
        }
    }

    const std::string key = manifestSessionKey(outputRoot);
    std::lock_guard<std::mutex> lock(gManifestSessionMutex);
    gManifestSessionFrames[key] = std::move(sessionFrames);
    writeManifest(compactRoot, orderedFrames(gManifestSessionFrames[key]));
}

CompactFrameRecord CompactExporter::captureFrame(
    const Frame &frame,
    int absoluteFrame,
    const std::string &pipelineMode,
    float zInterpolationRatio,
    const std::string &zInterpolationSource,
    const std::string &initialZSpace,
    const BackgroundRegionTracker::State *analyticBackground)
{
    if (absoluteFrame < 0) {
        throw std::invalid_argument(
            "compact export frame number must be non-negative");
    }
    requireFinite(zInterpolationRatio, "z interpolation ratio");
    if (zInterpolationRatio <= 0.0f) {
        throw std::invalid_argument(
            "compact export z interpolation ratio must be positive");
    }

    const auto &realFrame = frame.getRealFrame();
    if (realFrame.empty() || realFrame.front().empty()) {
        throw std::runtime_error(
            "compact export requires the frame dimensions before images are "
            "released");
    }

    CompactFrameRecord record;
    record.frame = absoluteFrame;
    record.sourceFrame = frame.getImageName();
    record.pipelineMode = pipelineMode;
    record.width = realFrame.front().cols;
    record.height = realFrame.front().rows;
    record.depth = static_cast<int>(realFrame.size());
    record.zInterpolationRatio = zInterpolationRatio;
    record.zInterpolationSource = zInterpolationSource;
    record.initialZSpace = initialZSpace;

    for (const cv::Mat &slice : realFrame) {
        if (slice.empty() || slice.rows != record.height ||
            slice.cols != record.width) {
            throw std::runtime_error(
                "compact export real frame has inconsistent slice shapes");
        }
    }

    if (analyticBackground != nullptr && frame.hasBackgroundFrame()) {
        record.background.kind = BackgroundKind::AnalyticEllipsoid;
        record.background.analytic.center = analyticBackground->center;
        record.background.analytic.radii = analyticBackground->radii;
        record.background.analytic.rotation = analyticBackground->rotation;
        record.background.analytic.cold = analyticBackground->coldBackground;
        record.background.analytic.hot = analyticBackground->hotBackground;
        record.background.analytic.softMargin =
            analyticBackground->softMargin;
        record.background.analytic.additiveOffset =
            frame.getBackgroundFrameOffset();
        record.background.analytic.offsetUpdates =
            frame.getBackgroundFrameOffsetUpdates();
    } else {
        record.background = captureInstalledBackground(
            frame, record.width, record.height, record.depth);
    }

    record.cells.reserve(frame.cells.size());
    for (std::size_t index = 0; index < frame.cells.size(); ++index) {
        const EllipsoidParams params = frame.cells[index].getCellParams();
        CompactCell cell;
        cell.drawOrder = static_cast<std::uint32_t>(index);
        cell.name = params.name;
        cell.x = params.x;
        cell.y = params.y;
        cell.z = params.z;
        cell.aRadius = params.aRadius;
        cell.bRadius = params.bRadius;
        cell.cRadius = params.cRadius;
        cell.thetaX = params.theta_x;
        cell.thetaY = params.theta_y;
        cell.thetaZ = params.theta_z;
        cell.brightness = params.brightness;
        cell.isTrash = params.isTrash;
        record.cells.push_back(std::move(cell));
    }

    return record;
}

void CompactExporter::writeFrame(const fs::path &outputRoot,
                                 const CompactFrameRecord &record)
{
    const fs::path compactRoot = outputRoot / "compact";
    const fs::path framesDirectory = compactRoot / "frames";
    const fs::path masksDirectory = compactRoot / "masks";
    fs::create_directories(framesDirectory);

    const std::string key = manifestSessionKey(outputRoot);
    {
        std::lock_guard<std::mutex> lock(gManifestSessionMutex);
        if (gManifestSessionFrames.find(key) ==
            gManifestSessionFrames.end()) {
            throw std::logic_error(
                "CompactExporter::beginRun must be called before writeFrame");
        }
    }

    std::string maskRelativePath;
    if (record.background.kind == BackgroundKind::BinaryMask) {
        fs::create_directories(masksDirectory);
        const std::string fileName = maskFileName(record.background);
        writeCubmIfNeeded(masksDirectory / fileName, record.background);
        maskRelativePath = "masks/" + fileName;
    }

    writeFrameJson(framesDirectory / frameFileName(record.frame),
                   record, maskRelativePath);

    std::lock_guard<std::mutex> lock(gManifestSessionMutex);
    const auto session = gManifestSessionFrames.find(key);
    // beginRun cannot disappear within this process, so the session found
    // above remains valid while the frame payload is written.
    session->second[record.frame] = frameFileName(record.frame);
    writeManifest(compactRoot, orderedFrames(session->second));
}

} // namespace celluniverse::compact
