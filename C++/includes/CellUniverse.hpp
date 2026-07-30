#ifndef CELLUNIVERSE_HPP
#define CELLUNIVERSE_HPP

#include <opencv2/opencv.hpp>
#include "ConfigTypes.hpp"
#include "Frame.hpp"
#include "types.hpp"
#include "Ellipsoid.hpp"
#include "BackgroundRegionTracker.hpp"

#include <array>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <set>
#include <unordered_map>
#include <optional>

namespace fs = std::filesystem;

class CellUniverse
{
public:
    CellUniverse(std::map<std::string, std::vector<Ellipsoid>> initialCells,
                 PathVec imagePaths,
                 BaseConfig &config,
                 std::string outputPath,
                 int firstFrame = 0,
                 int continueFrom = -1,
                 int selectedFrameCount = -1,
                 std::optional<BackgroundRegionTracker::SeedRecord>
                     initialCsvBackgroundSeed = std::nullopt);

    void optimize(int frameIndex);
    void saveImages(int frameIndex, const std::string &stage = "");
    void saveCompactFrame(int frameIndex);
    void saveCells(int frameIndex);
    void copyCellsForward(size_t to);
    // Memory optimization (M1): after this frame has been optimized, saved,
    // and its snapshot captured, release its image stacks. Cells and
    // snapshot metadata are retained. Enables long-horizon runs (60+ frames)
    // without 13+ GB memory peaks.
    void releaseFrameImages(int frameIndex);
    // Memory optimization (M2 — Option A): lazy per-frame load. Constructor
    // only samples percentiles; per-frame TIFF load + normalize + preprocess
    // happens on demand in this method. Main loop calls `prepareFrame(i)`
    // before `optimize(i)`. Keeps peak memory at ~1-2 frames (<1 GB for
    // 100+ frame runs vs 25+ GB before).
    void prepareFrame(int frameIndex);
    void prepareFrameWindow(int frameIndex);
    void preprocessAllFramesAlignedToMinimumBackground(bool loadIntoFrames);
    // Checkpoint save/load (Approach 2 — full state serialization).
    // saveCheckpoint(N) writes all state needed to resume AT frame N+1
    // to `{outputPath}/checkpoints/frame_{N:03d}.yaml`. Includes:
    // cells of frame N+1 (already copied forward), previousSnapshots,
    // cellShapeReference, cellShapeBirth, perFrameAdaptiveBackground[N],
    // perFrameMeanBrightness[N], frame N+1 backgroundValue, z_slices/maxZ.
    // loadCheckpoint(N, targetFrameIndex) reads checkpoint N and populates
    // the local frame that should be optimized next. Call BEFORE the main
    // loop starts. After loading, set the loop start to targetFrameIndex.
    void saveCheckpoint(int frameIndex);
    bool loadCheckpoint(int frameIndex, const std::string &checkpointPath);
    bool loadCheckpoint(int checkpointFrameIndex, int targetFrameIndex,
                        const std::string &checkpointPath);

    unsigned int length();

    // ---- Added for realtime viewer ----
    const std::vector<Ellipsoid> &getCells(int frameIndex) const;
    std::vector<std::string> getCellNames(int frameIndex) const;

private:
   BaseConfig config;
   std::vector<Frame> frames;
   std::string outputPath;
   int firstFrame;
   std::map<std::string, PreviousFrameSnapshot> previousSnapshots;
   // Frozen per-cell shape reference (a, b, c radii). Captured at cell
   // birth (frame 1 for initial cells; post-refit at split-accept for
   // daughters) and NEVER updated. Used as the pixel-collection mask
   // basis in subsequent frames' shape fits, decoupled from snap radii,
   // so a bloated fit in frame N can't compound into an even bigger
   // mask for frame N+1. See 2026-04-15 compounding-bloat analysis.
   std::map<std::string, std::array<float, 3>> cellShapeReference;
   // Birth-time radii. Captured once at first appearance, NEVER updated.
   // Used as the pixel-collection mask basis: mask = birth × maskScale.
   // Decoupled from the bounded ref so the mask can't participate in
   // feedback loops (neither upward bloat nor downward thinning).
   // The bounded ref is used ONLY for the fit-side growth cap.
   std::map<std::string, std::array<float, 3>> cellShapeBirth;
   std::unordered_map<std::string, int> cellFirstSeenFrame;
   std::unordered_map<std::string, int> trashDimFrameCounts;

   // M2 state: per-frame paths retained for lazy load and initial-cells map.
   PathVec imagePaths;
   size_t selectedFrameCount = 0;
   std::map<std::string, std::vector<Ellipsoid>> initialCells;
   float edgeBrightnessAlignmentTarget = 0.0f;
   bool edgeBrightnessAlignmentTargetInitialized = false;
   int continueFrom = -1;
   // Per-frame CellLumen split proposals. applyCellLumenRescue() builds these
   // from raw current-frame CellLumen centers; optimize() consumes them before
   // random split scheduling.
   std::unordered_map<int, std::unordered_map<std::string, BridgeSplitProposal>> cellLumenSplitPriors;
   // Parents whose best CellLumen split pair looked unsafe. The pre-pass
   // fallback may not resurrect these parents, otherwise a weak Lumen pair can
   // be rejected in ranking and then re-enter through the fallback path.
   std::unordered_map<int, std::set<std::string>> cellLumenSplitPriorRejectedBadParents;
   struct CellLumenCenterCandidate {
       cv::Point3f position{0.0f, 0.0f, 0.0f};
       float distance = 0.0f;
       int voxelCount = 0;
       float signal = 0.0f;
       int candidateId = -1;
   };
   std::unordered_map<int, std::unordered_map<std::string, CellLumenCenterCandidate>> cellLumenCenterCandidates;
   struct CellLumenLookaheadCandidate {
       cv::Point3f position{0.0f, 0.0f, 0.0f};
       int voxelCount = 0;
       float signal = 0.0f;
       int candidateId = -1;
   };
   std::unordered_map<int, std::vector<CellLumenLookaheadCandidate>> cellLumenLookaheadCandidates;
   struct PreparedFrameStack {
       std::vector<cv::Mat> realFrame;
       std::vector<cv::Mat> signalMap;
   };
   std::unordered_map<int, PreparedFrameStack> preparedFrameStacks;

   // Per-frame cached summaries for adaptive background (computed at end of
   // optimize(N); consumed by optimize(N+1) without needing frames[N]'s
   // image data.
   std::vector<float> perFrameAdaptiveBackground;
   std::vector<float> perFrameMeanBrightness;
   // Optional schema-v2 background envelope. When configured, this tracker
   // supplies a rotated, soft two-region background to every Frame. Its
   // geometry changes only after conservative image-evidence checks.
   BackgroundRegionTracker initialCsvBackgroundTracker;
   std::set<int> initialCsvBackgroundStateWrittenFrames;
   bool resumePreviousFrameSummaryValid = false;
   float resumePreviousAdaptiveBackground = 0.0f;
   float resumePreviousMeanBrightness = 0.0f;
   size_t resumePreviousCellCount = 0;
   int initialNonTrashCellCount = 0;
   int cumulativeAcceptedSplits = 0;
   std::unordered_map<int, std::vector<Frame::SignalCenter>>
       cellUniverse3WindowCentersByFrame;
   struct CellUniverse3WindowMapBox {
       cv::Point3f center{0.0f, 0.0f, 0.0f};
       int ix = 0;
       int iy = 0;
       int iz = 0;
       float priorMax = 0.0f;
       float priorSum = 0.0f;
       float futureMax = 0.0f;
       float futureSum = 0.0f;
       bool priorHot = false;
       bool futureHot = false;
       int voxels = 0;
   };
   struct CellUniverse3WindowMap {
       bool valid = false;
       int frameIndex = -1;
       int gridX = 0;
       int gridY = 0;
       int gridZ = 0;
       int boxSizeX = 1;
       int boxSizeY = 1;
       int boxSizeZ = 1;
       float priorThreshold = 0.0f;
       float futureThreshold = 0.0f;
       std::vector<CellUniverse3WindowMapBox> boxes;
   };
   std::unordered_map<int, CellUniverse3WindowMap> cellUniverse3WindowMapsByFrame;
   std::unordered_map<int, std::vector<cv::Mat>> cellUniverse3WindowProbabilityByFrame;
   std::unordered_map<int, std::vector<cv::Mat>> cellUniverse3WindowBackgroundByFrame;
   std::unordered_map<int, std::vector<cv::Mat>> cellUniverse3WindowBackgroundRegionByFrame;
   std::unordered_map<int, std::vector<cv::Mat>> cellUniverse3WindowBackgroundSampleByFrame;
   std::vector<cv::Mat> cellUniverse3GlobalMaxMap;
   float cellUniverse3GlobalHotThreshold = 0.0f;
   bool cellUniverse3GlobalMaxMapReady = false;
   struct CellUniverse3SplitHotspot {
       int frameIndex = -1;
       cv::Point3f center{0.0f, 0.0f, 0.0f};
       cv::Point3f axis{0.0f, 0.0f, 0.0f};
       float radius = 0.0f;
       float separation = 0.0f;
       float brightness = 0.0f;
       int boxes = 0;
       float score = std::numeric_limits<float>::max();
   };
   std::vector<CellUniverse3SplitHotspot> cellUniverse3SplitHotspots;
   struct CellUniverse3MissedSplitMemory {
       int frameIndex = -1;
       cv::Point3f parentPosition{0.0f, 0.0f, 0.0f};
       cv::Point3f splitAxis{0.0f, 0.0f, 0.0f};
       float parentMaxRadius = 0.0f;
       float parentShape = 1.0f;
       bool hasActivityRegionHint = false;
       cv::Point3f activityRegionCenter{0.0f, 0.0f, 0.0f};
       cv::Point3f activityRegionAxis{0.0f, 0.0f, 0.0f};
       float activityRegionRadius = 0.0f;
       float activityRegionBrightness = 0.0f;
       float activityRegionScore = std::numeric_limits<float>::max();
   };
   std::unordered_map<std::string, CellUniverse3MissedSplitMemory>
       cellUniverse3MissedSplitMemoryByCell;

   void prepareSignalCentersForFrame(int frameIndex,
                                     const std::vector<cv::Mat> &realFrame,
                                     bool keepLoaded);
   void installInitialCsvBackground(int frameIndex,
                                    const std::vector<cv::Mat> &realFrame);
   void writeInitialCsvBackgroundState(int frameIndex);
   std::vector<Frame::SignalCenter> buildCellUniverse3WindowCenters(
       int frameIndex,
       const std::vector<cv::Mat> &currentFrame);
   int rollingPreprocessWindowSize() const;
   size_t countNonTrashCellsInFrame(int frameIndex) const;
   double effectiveN2V2ContrastGamma() const;
   BaseConfig configForPreprocessing(int frameIndex) const;
   void invalidatePreparedFramesAfter(int frameIndex);
   double availableSystemMemoryGb() const;
   void pruneCellUniverse3PreparedFrameCache(int frameIndex,
                                             const std::string &reason);
   bool preparedFrameDiskCacheEnabled() const;
   fs::path preparedFrameDiskCacheDir() const;
   fs::path preparedFrameDiskCachePath(int frameIndex,
                                       const BaseConfig &preprocessConfig) const;
   bool loadPreparedFrameStackFromDiskCache(int frameIndex,
                                            const BaseConfig &preprocessConfig,
                                            PreparedFrameStack &prepared) const;
   void savePreparedFrameStackToDiskCache(int frameIndex,
                                          const BaseConfig &preprocessConfig,
                                          const PreparedFrameStack &prepared) const;
   void removePreparedFrameDiskCacheVariants(int frameIndex,
                                             const fs::path &keepPath) const;
   void prunePreparedFrameDiskCache(int frameIndex,
                                    int keepFuture,
                                    const std::string &reason) const;
   PreparedFrameStack loadPreparedFrameStack(int frameIndex);
   void cachePreparedFrameStack(int frameIndex);
   void applyCellLumenRescue(int frameIndex, const std::vector<cv::Mat> &preparedFrame);
   const std::vector<CellLumenLookaheadCandidate> &getCellLumenLookaheadCandidates(int frameIndex);
};

#endif
