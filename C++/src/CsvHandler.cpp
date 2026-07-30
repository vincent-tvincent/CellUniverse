#include "CsvHandler.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace
{
std::string trim(std::string value)
{
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) {
        value.erase(value.begin());
    }
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) {
        value.pop_back();
    }
    return value;
}

std::vector<std::string> splitCsvLine(const std::string &line)
{
    std::vector<std::string> tokens;
    std::string token;
    bool inQuotes = false;

    for (char ch : line) {
        if (ch == '"') {
            inQuotes = !inQuotes;
            continue;
        }
        if (ch == ',' && !inQuotes) {
            tokens.push_back(trim(token));
            token.clear();
            continue;
        }
        token.push_back(ch);
    }
    tokens.push_back(trim(token));
    return tokens;
}

std::string normalizeHeaderName(const std::string &header)
{
    std::string normalized;
    for (char ch : header) {
        if (std::isalnum(static_cast<unsigned char>(ch))) {
            normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
        }
    }
    return normalized;
}

std::unordered_map<std::string, size_t> buildHeaderIndex(const std::vector<std::string> &headers)
{
    std::unordered_map<std::string, size_t> index;
    for (size_t i = 0; i < headers.size(); ++i) {
        index[normalizeHeaderName(headers[i])] = i;
    }
    return index;
}

bool findColumn(const std::unordered_map<std::string, size_t> &index,
                std::initializer_list<const char *> names,
                size_t &column)
{
    for (const char *name : names) {
        auto it = index.find(normalizeHeaderName(name));
        if (it != index.end()) {
            column = it->second;
            return true;
        }
    }
    return false;
}

bool getToken(const std::vector<std::string> &tokens, size_t column, std::string &value)
{
    if (column >= tokens.size()) {
        return false;
    }
    value = tokens[column];
    const std::string normalized = normalizeHeaderName(value);
    return !value.empty() && normalized != "none" && normalized != "null" && normalized != "nan";
}

size_t requireColumn(const std::unordered_map<std::string, size_t> &index,
                     const std::string &name)
{
    const std::string normalized = normalizeHeaderName(name);
    const auto it = index.find(normalized);
    if (it == index.end()) {
        throw std::runtime_error("missing required schema-v2 column '" + name + "'");
    }
    return it->second;
}

std::string requireToken(const std::vector<std::string> &tokens,
                         size_t column,
                         const std::string &field)
{
    std::string value;
    if (!getToken(tokens, column, value)) {
        throw std::runtime_error("missing required schema-v2 field '" + field + "'");
    }
    return value;
}

float parseFiniteFloat(const std::string &text, const std::string &field)
{
    size_t consumed = 0;
    float value = 0.0f;
    try {
        value = std::stof(text, &consumed);
    } catch (const std::exception &) {
        throw std::runtime_error("invalid floating-point value for '" + field + "': " + text);
    }
    if (consumed != text.size() || !std::isfinite(value)) {
        throw std::runtime_error("invalid finite floating-point value for '" + field + "': " + text);
    }
    return value;
}

int parseInteger(const std::string &text, const std::string &field)
{
    size_t consumed = 0;
    int value = 0;
    try {
        value = std::stoi(text, &consumed);
    } catch (const std::exception &) {
        throw std::runtime_error("invalid integer value for '" + field + "': " + text);
    }
    if (consumed != text.size()) {
        throw std::runtime_error("invalid integer value for '" + field + "': " + text);
    }
    return value;
}

bool parseStrictBool(const std::string &text, const std::string &field)
{
    const std::string normalized = normalizeHeaderName(text);
    if (normalized == "1" || normalized == "true" ||
        normalized == "yes" || normalized == "y") {
        return true;
    }
    if (normalized == "0" || normalized == "false" ||
        normalized == "no" || normalized == "n") {
        return false;
    }
    throw std::runtime_error("invalid boolean value for '" + field + "': " + text);
}

bool optionalFiniteFloat(const std::vector<std::string> &tokens,
                         size_t column,
                         const std::string &field,
                         float &value)
{
    std::string text;
    if (!getToken(tokens, column, text)) {
        return false;
    }
    value = parseFiniteFloat(text, field);
    return true;
}

bool nearlyEqual(float lhs, float rhs)
{
    const float scale = std::max({1.0f, std::abs(lhs), std::abs(rhs)});
    return std::abs(lhs - rhs) <= 1e-6f * scale;
}

bool parseBoolText(const std::string &value)
{
    const std::string normalized = normalizeHeaderName(value);
    return normalized == "1" ||
           normalized == "true" ||
           normalized == "yes" ||
           normalized == "y" ||
           normalized == "trash";
}

void parseOptionalTrashColumn(const std::vector<std::string> &tokens,
                              const std::unordered_map<std::string, size_t> &headerIndex,
                              InitialCellRecord &record)
{
    size_t trashCol = 0;
    if (!findColumn(headerIndex, {"isTrash", "trash", "is_trash"}, trashCol)) {
        record.isTrash = false;
        return;
    }

    std::string trashText;
    record.isTrash = getToken(tokens, trashCol, trashText) && parseBoolText(trashText);
}

float parseScaledFloat(const std::string &value, float scale)
{
    return std::stof(value) * scale;
}

bool parseNamedInitialCell(const std::vector<std::string> &tokens,
                           const std::unordered_map<std::string, size_t> &headerIndex,
                           float zScaling,
                           float initialRadiusScale,
                           InitialCellRecord &record)
{
    size_t filePathCol = 0;
    size_t cellNameCol = 0;
    size_t xCol = 0;
    size_t yCol = 0;
    size_t zCol = 0;
    size_t aRadiusCol = 0;
    size_t bRadiusCol = 0;
    size_t cRadiusCol = 0;

    const bool hasNamedCellColumns =
        findColumn(headerIndex, {"filePath", "file", "frame", "frameName", "image", "imageFile", "imagePath"}, filePathCol) &&
        findColumn(headerIndex, {"cellName", "name", "id", "cellId", "label"}, cellNameCol) &&
        findColumn(headerIndex, {"x"}, xCol) &&
        findColumn(headerIndex, {"y"}, yCol) &&
        findColumn(headerIndex, {"z"}, zCol) &&
        findColumn(headerIndex, {"aRadius", "majorRadius", "radiusA", "radius", "r"}, aRadiusCol) &&
        findColumn(headerIndex, {"cRadius", "minorRadius", "radiusC", "zRadius"}, cRadiusCol);

    if (!hasNamedCellColumns) {
        return false;
    }

    std::string xText;
    std::string yText;
    std::string zText;
    std::string aRadiusText;
    std::string cRadiusText;
    if (!getToken(tokens, filePathCol, record.filePath) ||
        !getToken(tokens, cellNameCol, record.cellName) ||
        !getToken(tokens, xCol, xText) ||
        !getToken(tokens, yCol, yText) ||
        !getToken(tokens, zCol, zText) ||
        !getToken(tokens, aRadiusCol, aRadiusText) ||
        !getToken(tokens, cRadiusCol, cRadiusText)) {
        throw std::runtime_error("missing named columns");
    }

    record.x = std::stof(xText);
    record.y = std::stof(yText);
    record.z = std::stof(zText) * zScaling;
    record.aRadius = parseScaledFloat(aRadiusText, initialRadiusScale);
    record.bRadius = record.aRadius;
    if (findColumn(headerIndex, {"bRadius", "radiusB", "middleRadius", "intermediateRadius"}, bRadiusCol)) {
        std::string bRadiusText;
        if (getToken(tokens, bRadiusCol, bRadiusText)) {
            record.bRadius = parseScaledFloat(bRadiusText, initialRadiusScale);
        }
    }
    record.cRadius = parseScaledFloat(cRadiusText, initialRadiusScale);
    parseOptionalTrashColumn(tokens, headerIndex, record);
    return true;
}

bool parseNamedNapariCell(const std::vector<std::string> &tokens,
                          const std::unordered_map<std::string, size_t> &headerIndex,
                          const std::string &firstFrameFile,
                          float zScaling,
                          float initialRadiusScale,
                          int lineCount,
                          InitialCellRecord &record)
{
    size_t cellTypeCol = 0;
    size_t xCol = 0;
    size_t yCol = 0;
    size_t zCol = 0;
    const bool hasNamedNapariColumns =
        findColumn(headerIndex, {"cell_type", "cellType", "type"}, cellTypeCol) &&
        findColumn(headerIndex, {"z"}, zCol) &&
        findColumn(headerIndex, {"y"}, yCol) &&
        findColumn(headerIndex, {"x"}, xCol);

    if (!hasNamedNapariColumns) {
        return false;
    }

    std::string cellType;
    std::string xText;
    std::string yText;
    std::string zText;
    if (!getToken(tokens, cellTypeCol, cellType) ||
        !getToken(tokens, xCol, xText) ||
        !getToken(tokens, yCol, yText) ||
        !getToken(tokens, zCol, zText)) {
        throw std::runtime_error("missing Napari columns");
    }

    record.filePath = !firstFrameFile.empty() ? firstFrameFile : std::string("t000.tif");
    record.cellName = cellType + "_" + std::to_string(lineCount + 1);
    record.x = std::stof(xText);
    record.y = std::stof(yText);
    record.z = std::stof(zText) * zScaling;
    record.aRadius = 10.0f * initialRadiusScale;
    record.bRadius = record.aRadius;
    record.cRadius = 10.0f * initialRadiusScale;
    parseOptionalTrashColumn(tokens, headerIndex, record);
    return true;
}

bool parsePositionalCell(const std::vector<std::string> &tokens,
                         float zScaling,
                         float initialRadiusScale,
                         InitialCellRecord &record)
{
    if (tokens.size() < 7) {
        return false;
    }

    record.filePath = tokens[0];
    record.cellName = tokens[1];
    record.x = std::stof(tokens[2]);
    record.y = std::stof(tokens[3]);
    record.z = std::stof(tokens[4]) * zScaling;
    record.aRadius = parseScaledFloat(tokens[5], initialRadiusScale);
    record.bRadius = record.aRadius;
    if (tokens.size() >= 8) {
        const std::string normalizedB = normalizeHeaderName(tokens[6]);
        if (!tokens[6].empty() && normalizedB != "none" && normalizedB != "null" && normalizedB != "nan") {
            record.bRadius = parseScaledFloat(tokens[6], initialRadiusScale);
        }
        record.cRadius = parseScaledFloat(tokens[7], initialRadiusScale);
    } else {
        record.cRadius = parseScaledFloat(tokens[6], initialRadiusScale);
    }
    return true;
}

bool parsePositionalNapariCell(const std::vector<std::string> &tokens,
                               const std::string &firstFrameFile,
                               float zScaling,
                               float initialRadiusScale,
                               int lineCount,
                               InitialCellRecord &record)
{
    if (tokens.size() != 4) {
        return false;
    }

    record.filePath = !firstFrameFile.empty() ? firstFrameFile : std::string("t000.tif");
    record.cellName = tokens[0] + "_" + std::to_string(lineCount + 1);
    record.z = std::stof(tokens[1]) * zScaling;
    record.y = std::stof(tokens[2]);
    record.x = std::stof(tokens[3]);
    record.aRadius = 10.0f * initialRadiusScale;
    record.bRadius = record.aRadius;
    record.cRadius = 10.0f * initialRadiusScale;
    return true;
}

constexpr int kSupportedInitializerSchemaVersion = 2;

const std::array<const char *, 21> kSchemaV2Columns = {
    "file",
    "name",
    "cellId",
    "x",
    "y",
    "z",
    "aRadius",
    "bRadius",
    "cRadius",
    "theta_x",
    "theta_y",
    "theta_z",
    "brightness",
    "coldBackgroundBrightness",
    "hotBackgroundBrightness",
    "isTrash",
    "isHotBackgroundRegion",
    "backgroundSoftMargin",
    "zInterpolationRatio",
    "zCoordinateSpace",
    "initializerSchemaVersion"
};

bool hasSchemaMetadataMarker(const std::unordered_map<std::string, size_t> &headerIndex)
{
    static const std::array<const char *, 7> markers = {
        "coldBackgroundBrightness",
        "hotBackgroundBrightness",
        "isHotBackgroundRegion",
        "backgroundSoftMargin",
        "zInterpolationRatio",
        "zCoordinateSpace",
        "initializerSchemaVersion"
    };
    for (const char *marker : markers) {
        if (headerIndex.find(normalizeHeaderName(marker)) != headerIndex.end()) {
            return true;
        }
    }
    return false;
}

std::unordered_map<std::string, size_t> buildSchemaV2HeaderIndex(
    const std::vector<std::string> &headers)
{
    if (headers.size() != kSchemaV2Columns.size()) {
        throw std::runtime_error(
            "schema-v2 initial CSV must contain exactly " +
            std::to_string(kSchemaV2Columns.size()) +
            " columns; found " + std::to_string(headers.size()));
    }

    std::unordered_map<std::string, size_t> index;
    for (size_t i = 0; i < headers.size(); ++i) {
        const std::string normalized = normalizeHeaderName(headers[i]);
        if (normalized.empty()) {
            throw std::runtime_error("schema-v2 initial CSV contains an empty column name");
        }
        if (!index.emplace(normalized, i).second) {
            throw std::runtime_error(
                "schema-v2 initial CSV contains duplicate column '" + headers[i] + "'");
        }
    }
    for (const char *column : kSchemaV2Columns) {
        requireColumn(index, column);
    }
    return index;
}

void accumulateSchemaV2Metadata(
    const std::vector<std::string> &tokens,
    size_t zRatioCol,
    size_t zSpaceCol,
    size_t schemaVersionCol,
    InitialCsvMetadata &metadata)
{
    const int schemaVersion = parseInteger(
        requireToken(tokens, schemaVersionCol, "initializerSchemaVersion"),
        "initializerSchemaVersion");
    if (schemaVersion > kSupportedInitializerSchemaVersion) {
        throw std::runtime_error(
            "initializer schema version " + std::to_string(schemaVersion) +
            " is newer than supported version " +
            std::to_string(kSupportedInitializerSchemaVersion));
    }
    if (schemaVersion != kSupportedInitializerSchemaVersion) {
        throw std::runtime_error(
            "unsupported initializer schema version " +
            std::to_string(schemaVersion) +
            "; expected version " +
            std::to_string(kSupportedInitializerSchemaVersion));
    }

    const float zRatio = parseFiniteFloat(
        requireToken(tokens, zRatioCol, "zInterpolationRatio"),
        "zInterpolationRatio");
    if (zRatio <= 0.0f) {
        throw std::runtime_error("zInterpolationRatio must be positive");
    }

    const std::string zSpace = normalizeHeaderName(
        requireToken(tokens, zSpaceCol, "zCoordinateSpace"));
    if (zSpace != "raw" && zSpace != "scaled") {
        throw std::runtime_error(
            "zCoordinateSpace must be either 'raw' or 'scaled'");
    }

    if (!metadata.present) {
        metadata.present = true;
        metadata.initializerSchemaVersion = schemaVersion;
        metadata.zInterpolationRatio = zRatio;
        metadata.zCoordinateSpace = zSpace;
        return;
    }
    if (metadata.initializerSchemaVersion != schemaVersion) {
        throw std::runtime_error(
            "conflicting initializerSchemaVersion values");
    }
    if (!nearlyEqual(metadata.zInterpolationRatio, zRatio)) {
        throw std::runtime_error(
            "conflicting zInterpolationRatio values");
    }
    if (metadata.zCoordinateSpace != zSpace) {
        throw std::runtime_error(
            "conflicting zCoordinateSpace values");
    }
}

InitialCsvMetadata parseSchemaV2MetadataOnly(
    std::ifstream &file,
    const std::string &firstLine)
{
    const std::vector<std::string> headers = splitCsvLine(firstLine);
    const std::unordered_map<std::string, size_t> headerIndex =
        buildSchemaV2HeaderIndex(headers);
    const size_t zRatioCol = requireColumn(headerIndex, "zInterpolationRatio");
    const size_t zSpaceCol = requireColumn(headerIndex, "zCoordinateSpace");
    const size_t schemaVersionCol =
        requireColumn(headerIndex, "initializerSchemaVersion");

    InitialCsvMetadata metadata;
    int physicalLine = 1;
    int dataRows = 0;
    std::string line;
    while (std::getline(file, line)) {
        ++physicalLine;
        if (trim(line).empty()) {
            continue;
        }
        ++dataRows;

        try {
            const std::vector<std::string> tokens = splitCsvLine(line);
            if (tokens.size() != headers.size()) {
                throw std::runtime_error(
                    "expected " + std::to_string(headers.size()) +
                    " fields but found " + std::to_string(tokens.size()));
            }
            accumulateSchemaV2Metadata(
                tokens, zRatioCol, zSpaceCol, schemaVersionCol, metadata);
        } catch (const std::exception &e) {
            throw std::runtime_error(
                "Invalid schema-v2 initial CSV metadata at file line " +
                std::to_string(physicalLine) + " (" + e.what() + "): " + line);
        }
    }

    if (dataRows == 0 || !metadata.present) {
        throw std::runtime_error("schema-v2 initial CSV contains no data rows");
    }
    return metadata;
}

InitialCsvDocument parseSchemaV2Document(
    std::ifstream &file,
    const std::string &firstLine,
    float zScaling,
    float initialRadiusScale)
{
    if (!std::isfinite(initialRadiusScale) || initialRadiusScale <= 0.0f) {
        throw std::runtime_error(
            "schema-v2 initial CSV requires a finite positive initialRadiusScale");
    }

    const std::vector<std::string> headers = splitCsvLine(firstLine);
    const std::unordered_map<std::string, size_t> headerIndex =
        buildSchemaV2HeaderIndex(headers);

    const size_t fileCol = requireColumn(headerIndex, "file");
    const size_t nameCol = requireColumn(headerIndex, "name");
    const size_t cellIdCol = requireColumn(headerIndex, "cellId");
    const size_t xCol = requireColumn(headerIndex, "x");
    const size_t yCol = requireColumn(headerIndex, "y");
    const size_t zCol = requireColumn(headerIndex, "z");
    const size_t aRadiusCol = requireColumn(headerIndex, "aRadius");
    const size_t bRadiusCol = requireColumn(headerIndex, "bRadius");
    const size_t cRadiusCol = requireColumn(headerIndex, "cRadius");
    const size_t thetaXCol = requireColumn(headerIndex, "theta_x");
    const size_t thetaYCol = requireColumn(headerIndex, "theta_y");
    const size_t thetaZCol = requireColumn(headerIndex, "theta_z");
    const size_t brightnessCol = requireColumn(headerIndex, "brightness");
    const size_t coldBackgroundCol =
        requireColumn(headerIndex, "coldBackgroundBrightness");
    const size_t hotBackgroundCol =
        requireColumn(headerIndex, "hotBackgroundBrightness");
    const size_t isTrashCol = requireColumn(headerIndex, "isTrash");
    const size_t isHotRegionCol = requireColumn(headerIndex, "isHotBackgroundRegion");
    const size_t softMarginCol = requireColumn(headerIndex, "backgroundSoftMargin");
    const size_t zRatioCol = requireColumn(headerIndex, "zInterpolationRatio");
    const size_t zSpaceCol = requireColumn(headerIndex, "zCoordinateSpace");
    const size_t schemaVersionCol =
        requireColumn(headerIndex, "initializerSchemaVersion");

    InitialCsvDocument document;
    int physicalLine = 1;
    int dataRows = 0;
    int backgroundRows = 0;
    std::string line;
    while (std::getline(file, line)) {
        ++physicalLine;
        if (trim(line).empty()) {
            continue;
        }
        ++dataRows;

        try {
            const std::vector<std::string> tokens = splitCsvLine(line);
            if (tokens.size() != headers.size()) {
                throw std::runtime_error(
                    "expected " + std::to_string(headers.size()) +
                    " fields but found " + std::to_string(tokens.size()));
            }

            accumulateSchemaV2Metadata(
                tokens, zRatioCol, zSpaceCol, schemaVersionCol,
                document.metadata);

            const float effectiveZScaling =
                document.metadata.zCoordinateSpace == "raw" ? zScaling : 1.0f;
            if (!std::isfinite(effectiveZScaling) || effectiveZScaling <= 0.0f) {
                throw std::runtime_error(
                    "raw zCoordinateSpace requires a finite positive caller z scaling");
            }

            const std::string filePath =
                requireToken(tokens, fileCol, "file");
            const std::string name =
                requireToken(tokens, nameCol, "name");
            std::string cellId;
            const bool hasCellId = getToken(tokens, cellIdCol, cellId);

            const float x = parseFiniteFloat(
                requireToken(tokens, xCol, "x"), "x");
            const float y = parseFiniteFloat(
                requireToken(tokens, yCol, "y"), "y");
            const float z = parseFiniteFloat(
                requireToken(tokens, zCol, "z"), "z") * effectiveZScaling;
            const float aRadius = parseFiniteFloat(
                requireToken(tokens, aRadiusCol, "aRadius"), "aRadius");
            const float bRadius = parseFiniteFloat(
                requireToken(tokens, bRadiusCol, "bRadius"), "bRadius");
            const float cRadius = parseFiniteFloat(
                requireToken(tokens, cRadiusCol, "cRadius"), "cRadius");
            if (aRadius <= 0.0f || bRadius <= 0.0f || cRadius <= 0.0f) {
                throw std::runtime_error(
                    "aRadius, bRadius, and cRadius must all be positive");
            }

            std::string thetaXText;
            std::string thetaYText;
            std::string thetaZText;
            const bool hasThetaX = getToken(tokens, thetaXCol, thetaXText);
            const bool hasThetaY = getToken(tokens, thetaYCol, thetaYText);
            const bool hasThetaZ = getToken(tokens, thetaZCol, thetaZText);
            const bool hasAnyRotation = hasThetaX || hasThetaY || hasThetaZ;
            const bool hasAllRotation = hasThetaX && hasThetaY && hasThetaZ;
            if (hasAnyRotation && !hasAllRotation) {
                throw std::runtime_error(
                    "theta_x, theta_y, and theta_z must be all present or all blank");
            }
            const float thetaX = hasAllRotation
                ? parseFiniteFloat(thetaXText, "theta_x") : 0.0f;
            const float thetaY = hasAllRotation
                ? parseFiniteFloat(thetaYText, "theta_y") : 0.0f;
            const float thetaZ = hasAllRotation
                ? parseFiniteFloat(thetaZText, "theta_z") : 0.0f;

            float brightness = 0.0f;
            const bool hasBrightness = optionalFiniteFloat(
                tokens, brightnessCol, "brightness", brightness);
            if (hasBrightness && (brightness < 0.0f || brightness > 1.0f)) {
                throw std::runtime_error("brightness must be in [0, 1]");
            }

            const bool isTrash = parseStrictBool(
                requireToken(tokens, isTrashCol, "isTrash"), "isTrash");
            const bool isHotRegion = parseStrictBool(
                requireToken(tokens, isHotRegionCol, "isHotBackgroundRegion"),
                "isHotBackgroundRegion");

            if (isHotRegion) {
                ++backgroundRows;
                if (backgroundRows > 1) {
                    throw std::runtime_error(
                        "schema-v2 initial CSV must contain exactly one hot background record");
                }
                if (isTrash) {
                    throw std::runtime_error(
                        "hot background record cannot also be marked as trash");
                }
                if (!hasAllRotation) {
                    throw std::runtime_error(
                        "hot background record requires theta_x, theta_y, and theta_z");
                }
                if (!hasBrightness) {
                    throw std::runtime_error(
                        "hot background record requires brightness");
                }

                const float coldBackground = parseFiniteFloat(
                    requireToken(tokens, coldBackgroundCol,
                                 "coldBackgroundBrightness"),
                    "coldBackgroundBrightness");
                const float hotBackground = parseFiniteFloat(
                    requireToken(tokens, hotBackgroundCol,
                                 "hotBackgroundBrightness"),
                    "hotBackgroundBrightness");
                const float softMargin = parseFiniteFloat(
                    requireToken(tokens, softMarginCol, "backgroundSoftMargin"),
                    "backgroundSoftMargin");
                if (coldBackground < 0.0f || coldBackground > 1.0f ||
                    hotBackground < 0.0f || hotBackground > 1.0f) {
                    throw std::runtime_error(
                        "coldBackgroundBrightness and hotBackgroundBrightness "
                        "must be in [0, 1]");
                }
                if (hotBackground < coldBackground) {
                    throw std::runtime_error(
                        "hotBackgroundBrightness must be greater than or equal "
                        "to coldBackgroundBrightness");
                }
                if (softMargin < 0.0f || softMargin >= 1.0f) {
                    throw std::runtime_error(
                        "backgroundSoftMargin must be in [0, 1); zero selects "
                        "a hard boundary");
                }
                if (!nearlyEqual(brightness, hotBackground)) {
                    throw std::runtime_error(
                        "hot background record brightness must equal "
                        "hotBackgroundBrightness");
                }

                document.hasBackground = true;
                document.background.filePath = filePath;
                document.background.name = name;
                document.background.x = x;
                document.background.y = y;
                document.background.z = z;
                document.background.aRadius = aRadius;
                document.background.bRadius = bRadius;
                document.background.cRadius = cRadius;
                document.background.thetaX = thetaX;
                document.background.thetaY = thetaY;
                document.background.thetaZ = thetaZ;
                document.background.brightness = brightness;
                document.background.coldBackgroundBrightness = coldBackground;
                document.background.hotBackgroundBrightness = hotBackground;
                document.background.backgroundSoftMargin = softMargin;
                continue;
            }

            if (!hasCellId) {
                throw std::runtime_error(
                    "normal schema-v2 cell record requires cellId");
            }
            std::string unused;
            if (getToken(tokens, coldBackgroundCol, unused) ||
                getToken(tokens, hotBackgroundCol, unused) ||
                getToken(tokens, softMarginCol, unused)) {
                throw std::runtime_error(
                    "background brightness and soft-margin fields must be blank "
                    "on normal cell records");
            }

            InitialCellRecord record;
            record.filePath = filePath;
            record.cellName = name;
            record.cellId = cellId;
            record.x = x;
            record.y = y;
            record.z = z;
            record.aRadius = aRadius * initialRadiusScale;
            record.bRadius = bRadius * initialRadiusScale;
            record.cRadius = cRadius * initialRadiusScale;
            record.thetaX = thetaX;
            record.thetaY = thetaY;
            record.thetaZ = thetaZ;
            record.brightness = brightness;
            record.hasRotation = hasAllRotation;
            record.hasBrightness = hasBrightness;
            record.isTrash = isTrash;
            document.cells.push_back(record);
        } catch (const std::exception &e) {
            throw std::runtime_error(
                "Invalid schema-v2 initial CSV row at file line " +
                std::to_string(physicalLine) + " (" + e.what() + "): " + line);
        }
    }

    if (dataRows == 0 || !document.metadata.present) {
        throw std::runtime_error("schema-v2 initial CSV contains no data rows");
    }
    if (backgroundRows != 1 || !document.hasBackground) {
        throw std::runtime_error(
            "schema-v2 initial CSV must contain exactly one hot background record");
    }
    if (document.cells.empty()) {
        throw std::runtime_error(
            "schema-v2 initial CSV must contain at least one normal cell record");
    }
    return document;
}

std::vector<InitialCellRecord> parseLegacyInitialCells(
    std::ifstream &file,
    const std::string &firstLine,
    const std::string &firstFrameFile,
    float zScaling,
    float initialRadiusScale,
    const std::string &initialZSpace)
{
    const bool resumeStateCsv =
        firstLine.find("theta_x") != std::string::npos &&
        firstLine.find("theta_y") != std::string::npos &&
        firstLine.find("theta_z") != std::string::npos;
    const bool zAlreadyScaled =
        (initialZSpace == "scaled") ||
        (initialZSpace == "auto" && resumeStateCsv);
    const float effectiveZScaling = zAlreadyScaled ? 1.0f : zScaling;
    const std::unordered_map<std::string, size_t> headerIndex =
        buildHeaderIndex(splitCsvLine(firstLine));

    std::vector<InitialCellRecord> records;
    std::string line;
    int lineCount = 0;
    while (std::getline(file, line)) {
        if (trim(line).empty()) {
            continue;
        }

        const std::vector<std::string> tokens = splitCsvLine(line);
        try {
            InitialCellRecord record;
            if (parseNamedInitialCell(tokens, headerIndex, effectiveZScaling,
                                      initialRadiusScale, record) ||
                parseNamedNapariCell(tokens, headerIndex, firstFrameFile,
                                     effectiveZScaling, initialRadiusScale,
                                     lineCount, record) ||
                parsePositionalCell(tokens, effectiveZScaling,
                                    initialRadiusScale, record) ||
                parsePositionalNapariCell(tokens, firstFrameFile,
                                          effectiveZScaling,
                                          initialRadiusScale, lineCount,
                                          record)) {
                records.push_back(record);
                ++lineCount;
                continue;
            }

            throw std::runtime_error(
                "Invalid initial CSV row at data line " +
                std::to_string(lineCount + 1) +
                " (expected named, 7/8-column, or Napari format): " + line);
        } catch (const std::exception &e) {
            throw std::runtime_error(
                "Invalid initial CSV row at data line " +
                std::to_string(lineCount + 1) +
                " (" + e.what() + "): " + line);
        }
    }
    return records;
}
}

InitialCsvMetadata CsvHandler::loadInitialCsvMetadata(
    const Path &initParamsPath)
{
    std::ifstream file(initParamsPath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open initial CSV: " + initParamsPath);
    }

    std::string firstLine;
    if (!std::getline(file, firstLine)) {
        return InitialCsvMetadata{};
    }

    const std::unordered_map<std::string, size_t> headerIndex =
        buildHeaderIndex(splitCsvLine(firstLine));
    if (!hasSchemaMetadataMarker(headerIndex)) {
        return InitialCsvMetadata{};
    }
    return parseSchemaV2MetadataOnly(file, firstLine);
}

InitialCsvDocument CsvHandler::loadInitialCsv(
    const Path &initParamsPath,
    const std::string &firstFrameFile,
    float zScaling,
    float initialRadiusScale,
    const std::string &initialZSpace)
{
    std::ifstream file(initParamsPath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open initial CSV: " + initParamsPath);
    }

    std::string firstLine;
    std::getline(file, firstLine);
    const std::unordered_map<std::string, size_t> headerIndex =
        buildHeaderIndex(splitCsvLine(firstLine));
    if (hasSchemaMetadataMarker(headerIndex)) {
        return parseSchemaV2Document(
            file, firstLine, zScaling, initialRadiusScale);
    }

    InitialCsvDocument document;
    document.cells = parseLegacyInitialCells(
        file, firstLine, firstFrameFile, zScaling,
        initialRadiusScale, initialZSpace);
    return document;
}

std::vector<InitialCellRecord> CsvHandler::loadInitialCells(
    const Path &initParamsPath,
    const std::string &firstFrameFile,
    float zScaling,
    float initialRadiusScale,
    const std::string &initialZSpace)
{
    return loadInitialCsv(
        initParamsPath, firstFrameFile, zScaling,
        initialRadiusScale, initialZSpace).cells;
}
