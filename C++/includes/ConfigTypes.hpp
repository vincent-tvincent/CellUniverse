// ConfigTypes.hpp
#ifndef CONFIGTYPES_HPP
#define CONFIGTYPES_HPP
#include <vector>
#include <random>
#include <string>
#include "yaml-cpp/yaml.h"
#include <iostream>

class SimulationConfig {
public:
    int iterations_per_cell;
    float background_color;
    float cell_color;
    int padding;
    float z_scaling;
    float blur_sigma;
    float sigmoid_k;
    float sigmoid_center;
    float sigmoid_center_offset;
    int calibration_x;
    int calibration_y;
    int calibration_z;
    int calibration_width;
    int calibration_height;
    int z_slices;
    std::vector<int> z_values;

    // Constructor with default values
    SimulationConfig() : iterations_per_cell(0), background_color(0.0f), cell_color(0.0f),
                         padding(0), z_scaling(1.0f), blur_sigma(0.0f),
                         sigmoid_k(30.0f), sigmoid_center(-1.0f), sigmoid_center_offset(0.045f),
                         calibration_x(-1), calibration_y(-1), calibration_z(-1),
                         calibration_width(0), calibration_height(0), z_slices(-1) {
    }
    void explodeConfig(const YAML::Node node) {
        iterations_per_cell = node["iterations_per_cell"].as<int>();
        background_color = node["background_color"].as<float>();
        cell_color = node["cell_color"].as<float>();
        padding = node["padding"].as<int>();
        z_scaling = node["z_scaling"].as<float>();
        if (node["blur_sigma"]) {
            blur_sigma = node["blur_sigma"].as<float>();
        }
        if (node["sigmoid_k"]) {
            sigmoid_k = node["sigmoid_k"].as<float>();
        }
        if (node["sigmoid_center"]) {
            sigmoid_center = node["sigmoid_center"].as<float>();
        }
        if (node["sigmoid_center_offset"]) {
            sigmoid_center_offset = node["sigmoid_center_offset"].as<float>();
        }
        if (node["calibration_x"]) {
            calibration_x = node["calibration_x"].as<int>();
        }
        if (node["calibration_y"]) {
            calibration_y = node["calibration_y"].as<int>();
        }
        if (node["calibration_z"]) {
            calibration_z = node["calibration_z"].as<int>();
        }
        if (node["calibration_width"]) {
            calibration_width = node["calibration_width"].as<int>();
        }
        if (node["calibration_height"]) {
            calibration_height = node["calibration_height"].as<int>();
        }
    }
    void printConfig() {
        std::cout << "Simulation Config\n";
        std::cout << "iterations_per_cell: " << iterations_per_cell << std::endl;
        std::cout << "background_color: " << background_color << std::endl;
        std::cout << "cell_color: " << cell_color << std::endl;
        std::cout << "padding: " << padding << std::endl;
        std::cout << "z_scaling: " << z_scaling << std::endl;
        std::cout << "blur_sigma: " << blur_sigma << std::endl;
        std::cout << "sigmoid_k: " << sigmoid_k << std::endl;
        std::cout << "sigmoid_center: " << sigmoid_center << std::endl;
        std::cout << "sigmoid_center_offset: " << sigmoid_center_offset << std::endl;
        std::cout << "calibration_x: " << calibration_x << std::endl;
        std::cout << "calibration_y: " << calibration_y << std::endl;
        std::cout << "calibration_z: " << calibration_z << std::endl;
        std::cout << "calibration_width: " << calibration_width << std::endl;
        std::cout << "calibration_height: " << calibration_height << std::endl;
        std::cout << "z_slices: " << z_slices << std::endl;
    }
};

class ProbabilityConfig {
public:
    float perturbation;
    float split;
    float split_cost;
    float split_elongation_threshold;
    float min_relative_split_gain_base;
    float min_relative_split_gain_strict;
    float phase1_accept_rate_threshold;
    float nonsplit_shrink_trigger_ratio;
    float nonsplit_recover_target_ratio;
    float split_shrink_trigger_ratio;
    float split_recover_target_ratio;
    float brightness_retry_step;
    int brightness_retry_max_attempts;
    int brightness_retry_iters_per_attempt;
    float brightness_retry_background_margin;
    bool enable_brightness_recovery;
    ProbabilityConfig()
        : perturbation(0.0f),
          split(0.0f),
          split_cost(0.0f),
          split_elongation_threshold(1.3f),
          min_relative_split_gain_base(0.01f),
          min_relative_split_gain_strict(0.02f),
          phase1_accept_rate_threshold(0.02f),
          nonsplit_shrink_trigger_ratio(0.78f),
          nonsplit_recover_target_ratio(0.90f),
          split_shrink_trigger_ratio(0.55f),
          split_recover_target_ratio(0.80f),
          brightness_retry_step(0.02f),
          brightness_retry_max_attempts(10),
          brightness_retry_iters_per_attempt(120),
          brightness_retry_background_margin(0.01f),
          enable_brightness_recovery(true) {
    }

    void explodeConfig(const YAML::Node& node) {
        if (node["perturbation"]) {
            perturbation = node["perturbation"].as<float>();
        }
        if (node["split"]) {
            split = node["split"].as<float>();
        }
        if (node["split_cost"]) {
            split_cost = node["split_cost"].as<float>();
        }
        if (node["split_elongation_threshold"]) {
            split_elongation_threshold = node["split_elongation_threshold"].as<float>();
        }
        if (node["min_relative_split_gain_base"]) {
            min_relative_split_gain_base = node["min_relative_split_gain_base"].as<float>();
        }
        if (node["min_relative_split_gain_strict"]) {
            min_relative_split_gain_strict = node["min_relative_split_gain_strict"].as<float>();
        }
        if (node["phase1_accept_rate_threshold"]) {
            phase1_accept_rate_threshold = node["phase1_accept_rate_threshold"].as<float>();
        }
        if (node["nonsplit_shrink_trigger_ratio"]) {
            nonsplit_shrink_trigger_ratio = node["nonsplit_shrink_trigger_ratio"].as<float>();
        }
        if (node["nonsplit_recover_target_ratio"]) {
            nonsplit_recover_target_ratio = node["nonsplit_recover_target_ratio"].as<float>();
        }
        if (node["split_shrink_trigger_ratio"]) {
            split_shrink_trigger_ratio = node["split_shrink_trigger_ratio"].as<float>();
        }
        if (node["split_recover_target_ratio"]) {
            split_recover_target_ratio = node["split_recover_target_ratio"].as<float>();
        }
        if (node["brightness_retry_step"]) {
            brightness_retry_step = node["brightness_retry_step"].as<float>();
        }
        if (node["brightness_retry_max_attempts"]) {
            brightness_retry_max_attempts = node["brightness_retry_max_attempts"].as<int>();
        }
        if (node["brightness_retry_iters_per_attempt"]) {
            brightness_retry_iters_per_attempt = node["brightness_retry_iters_per_attempt"].as<int>();
        }
        if (node["brightness_retry_background_margin"]) {
            brightness_retry_background_margin = node["brightness_retry_background_margin"].as<float>();
        }
        if (node["enable_brightness_recovery"]) {
            enable_brightness_recovery = node["enable_brightness_recovery"].as<bool>();
        }
    }
    void printConfig() {
        std::cout << "Probability Config\n";
        std::cout << "perturbation: " << perturbation << std::endl;
        std::cout << "split: " << split << std::endl;
        std::cout << "split_cost: " << split_cost << std::endl;
        std::cout << "split_elongation_threshold: " << split_elongation_threshold << std::endl;
        std::cout << "min_relative_split_gain_base: " << min_relative_split_gain_base << std::endl;
        std::cout << "min_relative_split_gain_strict: " << min_relative_split_gain_strict << std::endl;
        std::cout << "phase1_accept_rate_threshold: " << phase1_accept_rate_threshold << std::endl;
        std::cout << "nonsplit_shrink_trigger_ratio: " << nonsplit_shrink_trigger_ratio << std::endl;
        std::cout << "nonsplit_recover_target_ratio: " << nonsplit_recover_target_ratio << std::endl;
        std::cout << "split_shrink_trigger_ratio: " << split_shrink_trigger_ratio << std::endl;
        std::cout << "split_recover_target_ratio: " << split_recover_target_ratio << std::endl;
        std::cout << "brightness_retry_step: " << brightness_retry_step << std::endl;
        std::cout << "brightness_retry_max_attempts: " << brightness_retry_max_attempts << std::endl;
        std::cout << "brightness_retry_iters_per_attempt: " << brightness_retry_iters_per_attempt << std::endl;
        std::cout << "brightness_retry_background_margin: " << brightness_retry_background_margin << std::endl;
        std::cout << "enable_brightness_recovery: " << (enable_brightness_recovery ? "true" : "false") << std::endl;
    }
};

class CellParams {
    //The CellParams class stores the parameters of a particular cell.
public:
    std::string name;
    CellParams(const std::string& name_)
        : name(name_){
    }
};

class PerturbParams {
    //Used with a CellConfig to add perturb parameters.
public:
    float prob;
    float mu;
    float sigma;
    void explodeParams(const YAML::Node& node) {
        prob = node["prob"].as<float>();
        mu = node["mu"].as<float>();
        sigma = node["sigma"].as<float>();
    }
    [[nodiscard]] float getPerturbOffset() const {
        std::random_device rd; // Obtain a random number from hardware
        std::mt19937 gen(rd()); // Seed the generator
        std::uniform_real_distribution<> dis(0.0, 1.0); // Distribution for probability

        if (dis(gen) < prob) {
            std::normal_distribution<> d(mu, sigma); // Distribution for the Gaussian
            return d(gen);
        } else {
            return mu;
        }
    }
};

class CellConfig {
    // A pure abstract base class for SphereConfig and BacilliConfig
public:
    // pure virtual function for exploding the configuration
    virtual void explodeConfig(const YAML::Node& node) = 0;
    virtual ~CellConfig() = default;
//    virtual CellConfig& operator=(const CellConfig& other) = 0;
};

class SphereConfig: public CellConfig {
public:
    PerturbParams x{};
    PerturbParams y{};
    PerturbParams z{};
    PerturbParams radius{};
    double minRadius{};
    double maxRadius{};
    double boundingBoxScale{1.0};
    ~SphereConfig() = default;

    void explodeConfig(const YAML::Node& node) override
    {
        x.explodeParams(node["x"]);
        y.explodeParams(node["y"]);
        z.explodeParams(node["z"]);
        radius.explodeParams(node["radius"]);
        minRadius = node["minRadius"].as<double>();
        maxRadius = node["maxRadius"].as<double>();
        if (node["boundingBoxScale"]) {
            boundingBoxScale = node["boundingBoxScale"].as<double>();
        }
    }
};

class SpheroidConfig: public CellConfig {
public:
    PerturbParams x{};
    PerturbParams y{};
    PerturbParams z{};
    PerturbParams majorRadius{};
    PerturbParams minorRadius{};
    PerturbParams thetaX{};
    PerturbParams thetaY{};
    PerturbParams thetaZ{};
    double minMajorRadius{};
    double maxMajorRadius{};
    double minMinorRadius{};
    double maxMinorRadius{};
    ~SpheroidConfig() = default;

    void explodeConfig(const YAML::Node& node) override
    {
        x.explodeParams(node["x"]);
        y.explodeParams(node["y"]);
        z.explodeParams(node["z"]);
        majorRadius.explodeParams(node["majorRadius"]);
        minorRadius.explodeParams(node["minorRadius"]);
        thetaX.explodeParams(node["thetaX"]);
        thetaY.explodeParams(node["thetaY"]);
        thetaZ.explodeParams(node["thetaZ"]);

        minMajorRadius = node["minMajorRadius"].as<double>();
        maxMajorRadius = node["maxMajorRadius"].as<double>();
        minMinorRadius = node["minMinorRadius"].as<double>();
        maxMinorRadius = node["maxMinorRadius"].as<double>();
    }
};

class BaseConfig {
public:
    std::string cellType;
    SpheroidConfig* cell;
    SimulationConfig simulation;
    ProbabilityConfig prob;
    BaseConfig() :cell(nullptr) {};
    BaseConfig& operator=(const BaseConfig& other) {
        if (this != &other) {
            if (other.cell != nullptr) {
                cell = new SpheroidConfig(*other.cell);
            }
            else {
                cell = nullptr;
            }
            simulation = other.simulation;
            prob = other.prob;
        }
        return *this;
    }
    BaseConfig(const BaseConfig& other) {
        if (other.cell != nullptr) {
            cell = new SpheroidConfig(*other.cell);
        }
        else {
            cell = nullptr;
        }
        simulation = other.simulation;
        prob = other.prob;
    }
    ~BaseConfig() {
        delete cell;
    }
    // load the BaseConfig with a YAML node
    void explodeConfig(const YAML::Node& node) {
        cellType = node["cellType"].as<std::string>();
        // TODO: Now cell is always pointing to a spheroid config, make it more dynamic later
        cell = new SpheroidConfig;
        cell->explodeConfig(node["cell"]);
        simulation.explodeConfig(node["simulation"]);
        prob.explodeConfig(node["prob"]);
    }
    // for debug
    void printConfig() {
        simulation.printConfig();
        prob.printConfig();
    }
};

#endif
