#ifndef CSVHANDLER_HPP
#define CSVHANDLER_HPP

#include "types.hpp"

#include <string>
#include <vector>

struct InitialCellRecord
{
    std::string filePath;
    std::string cellName;
    std::string cellId;
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
    bool hasRotation = false;
    bool hasBrightness = false;
    bool isTrash = false;
};

struct InitialCsvMetadata
{
    bool present = false;
    int initializerSchemaVersion = 0;
    float zInterpolationRatio = 1.0f;
    std::string zCoordinateSpace;
};

struct InitialBackgroundRecord
{
    std::string filePath;
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
    float coldBackgroundBrightness = 0.0f;
    float hotBackgroundBrightness = 0.0f;
    // Normalized transition width. Zero intentionally selects a hard boundary.
    float backgroundSoftMargin = 0.0f;
};

struct InitialCsvDocument
{
    std::vector<InitialCellRecord> cells;
    InitialCsvMetadata metadata;
    bool hasBackground = false;
    InitialBackgroundRecord background;
};

class CsvHandler
{
public:
    static InitialCsvMetadata loadInitialCsvMetadata(const Path &initParamsPath);

    static InitialCsvDocument loadInitialCsv(const Path &initParamsPath,
                                             const std::string &firstFrameFile,
                                             float zScaling,
                                             float initialRadiusScale,
                                             const std::string &initialZSpace = "auto");

    static std::vector<InitialCellRecord> loadInitialCells(const Path &initParamsPath,
                                                           const std::string &firstFrameFile,
                                                           float zScaling,
                                                           float initialRadiusScale,
                                                           const std::string &initialZSpace = "auto");
};

#endif
