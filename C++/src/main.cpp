#include <iostream>
#include <cstdio>
#include "types.hpp"
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include "ConfigTypes.hpp"
#include "CellFactory.hpp"
#include "CsvHandler.hpp"
#include "BackgroundRegionTracker.hpp"
#include "yaml-cpp/yaml.h"
#include "Ellipsoid.hpp"
#include "CellUniverse.hpp"
#include "CellLumen.hpp"
#include "CompactExporter.hpp"
#include "ImageHandler.hpp"
#include "LineageTreeCreator.hpp"
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <thread>
#ifdef _OPENMP
#include <omp.h>
#endif

class Args
{
public:
    std::string config{};
    std::string input{};
    int firstFrame = 0;
    int lastFrame = 0;
    std::string initial{};
    std::string output{};
    int continueFrom = -1;
    // Optional checkpoint-resume args (2026-04-22). When provided via the
    // run_celluniverse.sh launcher (sourced from the INI preset), these
    // override config.simulation.resume_from / resume_source_dir. Absent
    // (argc < 9) → resume defaults to disabled (0 / "").
    int resumeFromFrame = 0;
    std::string resumeSourceDir{};
};

class CellLumenArgs
{
public:
    std::string inputFile{};
    std::string output{};
    std::string config{};
    std::string csvOutput{};
};

// helper function to load the config
void loadConfig(const std::string &path, BaseConfig &config)
{
    YAML::Node node = CellUniverseConfig::loadConfigYamlNode(path);
    config.explodeConfig(node);
}

void applyRuntimeOverrides(BaseConfig &config)
{
    const char *threadEnv = std::getenv("CELLUNIVERSE_THREADS");
    std::string threadSource = "config";

    if (threadEnv != nullptr && std::string(threadEnv).size() > 0)
    {
        try
        {
            config.simulation.parallel_threads = std::stoi(threadEnv);
            threadSource = "CELLUNIVERSE_THREADS";
        }
        catch (const std::exception &)
        {
            std::cerr << "[WARN] Ignoring invalid CELLUNIVERSE_THREADS="
                      << threadEnv << "; using config value "
                      << config.simulation.parallel_threads << '\n';
        }
    }

    config.simulation.parallel_threads = std::max(1, config.simulation.parallel_threads);
    config.simulation.parallel_min_slices = std::max(1, config.simulation.parallel_min_slices);

    const unsigned int hardwareThreads = std::thread::hardware_concurrency();
    if (hardwareThreads > 0 &&
        config.simulation.parallel_threads > static_cast<int>(hardwareThreads))
    {
        std::cerr << "[WARN] Requested parallel_threads="
                  << config.simulation.parallel_threads
                  << " but hardware_concurrency=" << hardwareThreads
                  << "; clamping to hardware_concurrency." << '\n';
        config.simulation.parallel_threads = static_cast<int>(hardwareThreads);
    }

    cv::setNumThreads(config.simulation.parallel_threads);
#ifdef _OPENMP
    omp_set_num_threads(config.simulation.parallel_threads);
#endif

    const char *seedEnv = std::getenv("CELLUNIVERSE_SEED");
    std::cout << "[Runtime Parallelism] mode="
              << (config.simulation.parallel_threads > 1 ? "parallel_z_slices" : "single_thread")
              << " threads=" << config.simulation.parallel_threads
              << " source=" << threadSource
              << " hardware_concurrency=" << hardwareThreads
              << " parallel_min_slices=" << config.simulation.parallel_min_slices
              << " opencv_threads=" << cv::getNumThreads()
              << std::endl;
    if (seedEnv != nullptr && std::string(seedEnv).size() > 0)
    {
        std::cout << "[Runtime Random] CELLUNIVERSE_SEED=" << seedEnv << std::endl;
    }
    std::cout << "[Efficiency Metric] primary=seconds_per_frame"
              << " normalized=seconds_per_cell_iteration"
              << " realtime=iterations_per_second"
              << " note=split_attempts_are_reported_separately_because_they_are_much_more_expensive"
              << std::endl;
}

bool pathsReferToSameOutput(const fs::path &left, const fs::path &right)
{
    std::error_code leftError;
    std::error_code rightError;
    const fs::path leftAbsolute = fs::absolute(left, leftError);
    const fs::path rightAbsolute = fs::absolute(right, rightError);
    if (leftError || rightError) {
        return left.lexically_normal() == right.lexically_normal();
    }
    return leftAbsolute.lexically_normal() == rightAbsolute.lexically_normal();
}

int computeFutureContextLookahead(const BaseConfig &config)
{
    int lookahead = 0;
    const bool needsPreparedWindow =
        config.simulation.prepare_analyze_one_frame ||
        config.simulation.celluniverse3_enabled ||
        (config.cellLumen.enabled && config.cellLumen.fusionEnabled);

    if (!needsPreparedWindow || config.simulation.quit_after_preprocessing) {
        return 0;
    }

    if (config.simulation.celluniverse3_enabled) {
        lookahead = std::max(lookahead,
                             std::max(1, config.simulation.celluniverse3_window_radius));
    }

    if (config.simulation.celluniverse2_enabled &&
        config.prob.pca_bridge_future_window_enabled &&
        config.prob.pca_bridge_future_window_size > 1) {
        lookahead = std::max(
            lookahead,
            std::clamp(config.prob.pca_bridge_future_window_size, 2, 5) - 1);
    }

    if (config.cellLumen.enabled &&
        config.cellLumen.fusionEnabled &&
        config.cellLumen.fusionSplitPriorWindowEnabled &&
        config.cellLumen.fusionSplitPriorWindowSize > 1) {
        lookahead = std::max(
            lookahead,
            std::clamp(config.cellLumen.fusionSplitPriorWindowSize, 2, 5) - 1);
    }

    return lookahead;
}

Args initArgs(int argc, char *argv[]) {
    // parse args here
    Args args;

    args.firstFrame = std::stoi(argv[ff]);
    std::cout << "Loading args:\n";
    std::cout << "First frame: " << args.firstFrame << '\n'
              << std::flush;
    args.lastFrame = std::stoi(argv[lf]);
    std::cout << "Last frame: " << args.lastFrame << '\n'
              << std::flush;
    args.initial = argv[initial];
    std::cout << "Initial CSV path: " << args.initial << '\n'
              << std::flush;
    args.input = argv[input];
    std::cout << "Input: " << args.input << '\n'
              << std::flush;
    args.output = argv[output];
    std::cout << "Output folder: " << args.output << '\n'
              << std::flush;
    args.config = argv[config];
    std::cout << "Config file: " << args.config << '\n'
              << std::flush;
    args.continueFrom = -1;

    // Optional resume args (positions 7 and 8). Both must be present to
    // activate resume; either missing → leave defaults (resume disabled).
    // argc indexing: argv[0]=binary, argv[1]=ff, ..., argv[6]=initial,
    // argv[7]=resumeFrom, argv[8]=resumeSourceDir.
    if (argc > static_cast<int>(resumeFrom)) {
        try {
            args.resumeFromFrame = std::stoi(argv[resumeFrom]);
        } catch (const std::exception &) {
            args.resumeFromFrame = 0;
        }
        std::cout << "Resume from frame (arg): " << args.resumeFromFrame
                  << '\n' << std::flush;
    }
    if (argc > static_cast<int>(resumeSourceDir)) {
        args.resumeSourceDir = argv[resumeSourceDir];
        std::cout << "Resume source dir (arg): " << args.resumeSourceDir
                  << '\n' << std::flush;
    }

    return args;
}

CellLumenArgs initCellLumenArgs(char *argv[])
{
    CellLumenArgs args;
    args.inputFile = argv[2];
    args.output = argv[3];
    args.config = argv[4];
    args.csvOutput = argv[5];

    std::cout << "Loading CellLumen args:\n";
    std::cout << "Input frame: " << args.inputFile << '\n' << std::flush;
    std::cout << "Output folder: " << args.output << '\n' << std::flush;
    std::cout << "Config file: " << args.config << '\n' << std::flush;
    std::cout << "CSV output: " << args.csvOutput << '\n' << std::flush;
    return args;
}

int main(int argc, char *argv[])
{
    // Suppress OpenCV TIFF warnings (ColorMap tag noise from microscopy TIFFs)
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_ERROR);

    // Keep worker logs durable through backend pipes. This is useful when a
    // native process is killed before buffered '\n' diagnostics flush.
    std::cout.setf(std::ios::unitbuf);
    std::cerr.setf(std::ios::unitbuf);

    // Note: progress/status lines use std::endl for immediate visibility
    // through pipes (tee). Inner-loop diagnostics use '\n' for performance.

    if (argc >= 2 && std::string(argv[1]) == "--lineage-tree")
    {
        LineageTreeCreator::RenderOptions options;
        int argi = 2;
        while (argi < argc)
        {
            const std::string opt = argv[argi];
            if (opt == "--first-frame" && argi + 1 < argc)
            {
                options.firstFrame = std::stoi(argv[argi + 1]);
                argi += 2;
            }
            else if (opt == "--last-frame" && argi + 1 < argc)
            {
                options.lastFrame = std::stoi(argv[argi + 1]);
                argi += 2;
            }
            else if (opt == "--fps" && argi + 1 < argc)
            {
                options.fps = std::stod(argv[argi + 1]);
                argi += 2;
            }
            else
            {
                break;
            }
        }

        if (argc - argi < 2)
        {
            std::cerr << "Usage: celluniverse --lineage-tree [--first-frame N] [--last-frame N] [--fps F] "
                      << "<output.png|output.mp4> <cells.csv> [cells.csv ...]\n"
                      << "GIF output is handled by C++/scripts/make_lineage_tree_demo.py.\n";
            return 1;
        }

        std::vector<std::string> csvPaths;
        const std::string lineageOutput = argv[argi++];
        for (int i = argi; i < argc; ++i)
        {
            csvPaths.emplace_back(argv[i]);
        }

        const bool ok = LineageTreeCreator::renderCsvFiles(csvPaths, lineageOutput, options);
        return ok ? 0 : 1;
    }

    const std::string command = argc >= 2 ? std::string(argv[1]) : std::string();
    if (command == "--cell-lumen" || command == "--CellLumen")
    {
        if (argc < 6)
        {
            std::cerr << "Usage: celluniverse --cell-lumen <input_frame.tif> <output_dir> <config.yaml> <csv_output>\n";
            return 1;
        }

        CellLumenArgs args = initCellLumenArgs(argv);
        BaseConfig config;
        loadConfig(args.config, config);
        applyRuntimeOverrides(config);
        config.printConfig();

        fs::create_directories(args.output);
        if (celluniverse::compact::CompactExporter::writesCompact(
                config.simulation.export_mode))
        {
            celluniverse::compact::CompactExporter::beginRun(
                args.output, 0);
        }
        CellLumen cellLumen(config, fs::path(args.output));
        cellLumen.buildInitialCsvForFrame(fs::path(args.inputFile), fs::path(args.csvOutput));
        return 0;
    }

    // check user input
    if (argc < 7)
    {
        std::cerr << "Usage: celluniverse <firstFrame> <lastFrame> <input_pattern_or_dir_or_file> <output_dir> <config.yaml> <initial.csv>\n";
        return 1;
    }


    // parse args
    Args args = initArgs(argc, argv);

    // load config
    BaseConfig config;
    loadConfig(args.config, config);

    // CLI resume args (from run_celluniverse.sh INI preset) override whatever
    // was parsed from the YAML. Use argv-based detection so "absent CLI arg"
    // (argc < 8 / 9) does NOT clobber a YAML-set value.
    if (argc > static_cast<int>(resumeFrom)) {
        config.simulation.resume_from = args.resumeFromFrame;
    }
    if (argc > static_cast<int>(resumeSourceDir)) {
        config.simulation.resume_source_dir = args.resumeSourceDir;
    }

    applyRuntimeOverrides(config);

    // Schema-v2 initializer metadata is authoritative for the coordinate
    // contract. Read it before discovering/loading images so z interpolation
    // uses the same ratio that produced the CSV. Legacy CSVs have no metadata
    // marker and therefore retain the YAML configuration unchanged.
    const InitialCsvMetadata initialCsvMetadata =
        CsvHandler::loadInitialCsvMetadata(args.initial);
    if (initialCsvMetadata.present) {
        const float roundedRatio =
            std::round(initialCsvMetadata.zInterpolationRatio);
        if (std::abs(initialCsvMetadata.zInterpolationRatio - roundedRatio) >
                1.0e-5f ||
            roundedRatio < 1.0f) {
            throw std::runtime_error(
                "initializer schema-v2 zInterpolationRatio must be a positive "
                "integer for the current CellUniverse interpolation engine; "
                "refusing to truncate " +
                std::to_string(initialCsvMetadata.zInterpolationRatio));
        }
        const float configuredRatio = config.simulation.z_scaling;
        const std::string configuredSpace =
            config.simulation.initial_z_space;
        config.simulation.z_scaling = roundedRatio;
        config.simulation.z_scaling_source = "initial_csv";
        config.simulation.initial_z_space =
            initialCsvMetadata.zCoordinateSpace;
        std::cout << "[Initial CSV Metadata] schema="
                  << initialCsvMetadata.initializerSchemaVersion
                  << " zInterpolationRatio="
                  << config.simulation.z_scaling
                  << " zCoordinateSpace="
                  << config.simulation.initial_z_space
                  << " overrides=config"
                  << " previous_z_scaling=" << configuredRatio
                  << " previous_z_space=" << configuredSpace
                  << '\n';
    } else {
        std::cout << "[Initial CSV Metadata] schema=legacy"
                  << " action=use_regular_configuration"
                  << " z_scaling=" << config.simulation.z_scaling
                  << " initial_z_space="
                  << config.simulation.initial_z_space
                  << '\n';
    }
    config.printConfig();

    if (celluniverse::compact::CompactExporter::writesCompact(
            config.simulation.export_mode))
    {
        const bool preserveExistingFrames =
            config.simulation.resume_from > 0 &&
            !config.simulation.resume_source_dir.empty() &&
            pathsReferToSameOutput(
                args.output, config.simulation.resume_source_dir);
        const int preserveFramesBefore =
            preserveExistingFrames
                ? config.simulation.resume_from
                : 0;
        celluniverse::compact::CompactExporter::beginRun(
            args.output, preserveFramesBefore);
        std::cout << "[Compact Export] session="
                  << (preserveExistingFrames
                          ? "in_place_resume"
                          : "fresh_manifest")
                  << " output=" << (fs::path(args.output) / "compact")
                  << '\n';
    }

    // Load selected frames plus optional future context. The selected range
    // still controls optimize/export/checkpoint behavior; extra lookahead
    // frames only feed rolling window evidence if they exist.
    const int futureContextLookahead = computeFutureContextLookahead(config);
    const int loadLastFrame =
        (args.lastFrame >= 0 && futureContextLookahead > 0)
            ? args.lastFrame + futureContextLookahead
            : args.lastFrame;
    PathVec imageFilePaths = ImageHandler::getImageFilePaths(
        args.input,
        args.firstFrame,
        loadLastFrame,
        config,
        futureContextLookahead > 0,
        args.lastFrame);
    int selectedFrameCount = static_cast<int>(imageFilePaths.size());
    if (args.lastFrame >= args.firstFrame && args.lastFrame >= 0) {
        selectedFrameCount = args.lastFrame - args.firstFrame + 1;
    }
    selectedFrameCount =
        std::min(selectedFrameCount, static_cast<int>(imageFilePaths.size()));
    if (futureContextLookahead > 0) {
        std::cout << "[INFO] future context lookahead enabled: requested="
                  << args.firstFrame << ".." << args.lastFrame
                  << " load_last=" << loadLastFrame
                  << " selected_count=" << selectedFrameCount
                  << " loaded_count=" << imageFilePaths.size()
                  << " context_count="
                  << (static_cast<int>(imageFilePaths.size()) - selectedFrameCount)
                  << '\n';
    }

    // load cells
    std::string firstFrameFile;
    if (!imageFilePaths.empty()) {
        firstFrameFile = imageFilePaths.front().filename().string();
        std::cout << "[INFO] firstFrameFile=" << firstFrameFile << '\n';
    } else {
        std::cerr << "[WARN] imageFilePaths is empty; cannot determine initial frame filename." << '\n';
    }

    std::optional<BackgroundRegionTracker::SeedRecord>
        initialCsvBackgroundSeed;
    if (initialCsvMetadata.present) {
        const float initialRadiusScale =
            config.cell ? config.cell->initialRadiusScale : 1.0f;
        const InitialCsvDocument initialCsv = CsvHandler::loadInitialCsv(
            args.initial,
            firstFrameFile,
            config.simulation.z_scaling,
            initialRadiusScale,
            config.simulation.initial_z_space);
        if (initialCsv.hasBackground) {
            BackgroundRegionTracker::SeedRecord seed;
            seed.center = cv::Point3f(
                initialCsv.background.x,
                initialCsv.background.y,
                initialCsv.background.z);
            seed.radii = cv::Vec3f(
                initialCsv.background.aRadius,
                initialCsv.background.bRadius,
                initialCsv.background.cRadius);
            seed.rotation = cv::Vec3f(
                initialCsv.background.thetaX,
                initialCsv.background.thetaY,
                initialCsv.background.thetaZ);
            seed.coldBackground =
                initialCsv.background.coldBackgroundBrightness;
            seed.hotBackground =
                initialCsv.background.hotBackgroundBrightness;
            seed.softMargin =
                initialCsv.background.backgroundSoftMargin;
            initialCsvBackgroundSeed = seed;

            // Keep scalar-only code paths safe while the spatial background
            // field is installed during frame preparation.
            if (config.cell) {
                config.cell->backgroundColor = seed.coldBackground;
            }
            std::cout << "[Initial CSV Background] source="
                      << initialCsv.background.name
                      << " center=(" << seed.center.x << ","
                      << seed.center.y << "," << seed.center.z << ")"
                      << " radii=(" << seed.radii[0] << ","
                      << seed.radii[1] << "," << seed.radii[2] << ")"
                      << " rotation=(" << seed.rotation[0] << ","
                      << seed.rotation[1] << "," << seed.rotation[2] << ")"
                      << " cold=" << seed.coldBackground
                      << " hot=" << seed.hotBackground
                      << " softMargin=" << seed.softMargin
                      << " dynamic_tracking=enabled"
                      << '\n';
        }
    }

    if (config.cellType == "ellipsoid" && config.cell) {
        Ellipsoid::cellConfig = *config.cell;
    }

    if (config.simulation.quit_after_preprocessing) {
        CellUniverse preprocessOnlyLineage({}, imageFilePaths, config, args.output,
                                           args.firstFrame, args.continueFrom,
                                           selectedFrameCount,
                                           initialCsvBackgroundSeed);
        preprocessOnlyLineage.preprocessAllFramesAlignedToMinimumBackground(false);
        std::cout << "[DEBUG] quit_after_preprocessing=true; exiting after preprocessing/load phase." << std::endl;
        return 0;
    }

    // load cells here
    CellFactory cellFactory(config);
    std::map<Path, std::vector<Ellipsoid>> cells = cellFactory.createCells(args.initial, config.simulation.z_slices / 2,
                                                                        config.simulation.z_scaling, firstFrameFile,
                                                                        config.simulation.initial_z_space);
    // create lineage
    CellUniverse lineage = CellUniverse(cells, imageFilePaths, config,
                                        args.output, args.firstFrame,
                                        args.continueFrom, selectedFrameCount,
                                        initialCsvBackgroundSeed);
    const bool prepareAnalyzeOneFrame =
        (config.simulation.prepare_analyze_one_frame ||
         config.simulation.celluniverse3_enabled ||
         (config.cellLumen.enabled && config.cellLumen.fusionEnabled)) &&
        !config.simulation.quit_after_preprocessing;
    if (prepareAnalyzeOneFrame) {
        std::cout << "[INFO] prepare_analyze_one_frame=true; each frame will be prepared immediately before optimize()."
                  << std::endl;
        if (config.cellLumen.enabled && config.cellLumen.fusionEnabled &&
            !config.simulation.prepare_analyze_one_frame) {
            std::cout << "[INFO] cell_lumen.fusionEnabled=true; forcing per-frame prepare so CellLumen rescue uses the shared prepared stack before optimization."
                      << std::endl;
        }
        if (config.simulation.celluniverse3_enabled &&
            !config.simulation.prepare_analyze_one_frame) {
            std::cout << "[INFO] celluniverse3_enabled=true; forcing per-frame prepare so windowed max/sum guidance can use a rolling local frame window."
                      << std::endl;
        }
    } else {
        lineage.preprocessAllFramesAlignedToMinimumBackground(true);
    }

    // Checkpoint resume (Approach 2): resume_from is the absolute dataset
    // frame to analyze next. Load `{resume_source_dir}/checkpoints/
    // frame_{resume_from - 1:03d}.txt`, install its copied-forward cells into
    // the local frame corresponding to resume_from, then start the loop there.
    int loopStart = 0;
    if (config.simulation.resume_from > 0 && !config.simulation.resume_source_dir.empty()) {
        const int resumeFrame = config.simulation.resume_from;
        const int checkpointFrame = resumeFrame - 1;
        const int targetLocalFrame = resumeFrame - args.firstFrame;
        if (targetLocalFrame < 0 ||
            targetLocalFrame >= static_cast<int>(lineage.length())) {
            std::cerr << "[Resume] invalid resume_from=" << resumeFrame
                      << " for requested frame range [" << args.firstFrame
                      << "," << args.lastFrame << "]; running from local frame 0\n";
        } else {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "frame_%03d.txt", checkpointFrame);
            const std::string ckptPath =
                config.simulation.resume_source_dir + "/checkpoints/" + buf;
            if (lineage.loadCheckpoint(checkpointFrame, targetLocalFrame, ckptPath)) {
                loopStart = targetLocalFrame;
                std::cout << "[Resume] skipping absolute frames "
                          << args.firstFrame << ".." << (resumeFrame - 1)
                          << " (local 0.." << (loopStart - 1) << ")"
                          << " — loaded checkpoint from " << ckptPath << std::endl;
            } else {
                std::cerr << "[Resume] checkpoint load failed, running from frame 0\n";
            }
        }
    }

    // Run
    auto start = std::chrono::steady_clock::now();
    for (int frame = loopStart; frame < lineage.length(); ++frame)
    {
        // M2 Option A: lazy-load this frame's images. In on-demand mode,
        // keep a rolling prepared-stack window sized to the CellLumen future
        // evidence window, then consume the current frame from that cache.
        if (prepareAnalyzeOneFrame) {
            lineage.prepareFrameWindow(frame);
        }
        lineage.prepareFrame(frame);

        lineage.optimize(frame);

        lineage.copyCellsForward(frame + 1);

        lineage.saveImages(frame);
        lineage.saveCompactFrame(frame);

        lineage.saveCells(frame);

        if (config.simulation.release_analyzed_exported_frames) {
            // Release image-heavy stacks after exported outputs and saved cells.
            // Later frames use copied cells, checkpoints, and cached summaries.
            lineage.releaseFrameImages(frame);
        }

        // Checkpoint for potential future resume.
        lineage.saveCheckpoint(frame);
    }
    auto end = std::chrono::steady_clock::now(); // timer end


    // end this program
    std::chrono::duration<double> elapsed_seconds = end - start;

    std::cout << "Time elapsed: " << elapsed_seconds.count() << " seconds" << std::endl;

    return 0;
}
