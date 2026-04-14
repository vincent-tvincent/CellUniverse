#ifndef SPHEROID_HPP
#define SPHEROID_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <array>
#include <algorithm>
#include <string>

#include "ConfigTypes.hpp"
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class SpheroidParams : public CellParams
{
public:
    float x;
    float y;
    float z;
    // Triaxial ellipsoid semi-axes. When `bRadius` is zero it is treated as
    // oblate (`bRadius == aRadius`) by the Spheroid constructor for backward
    // compatibility with pre-triaxial CSVs.
    float aRadius; // a axis
    float bRadius;     // b axis (second equatorial axis, 0 = oblate fallback)
    float cRadius; // c axis
    float theta_x; // rotation about x-axis (radians)
    float theta_y; // rotation about y-axis (radians)
    float theta_z; // rotation about z-axis (radians)
    float brightness; // per-cell rendering brightness [0,1]

    SpheroidParams() : CellParams(""), x(0), y(0), z(0), aRadius(0), bRadius(0), cRadius(0), theta_x(0), theta_y(0), theta_z(0), brightness(0.5f) {}
    SpheroidParams(const std::string &name, float x, float y, float z, float aRadius, float cRadius)
        : CellParams(name), x(x), y(y), z(z), aRadius(aRadius), bRadius(0), cRadius(cRadius), theta_x(0), theta_y(0), theta_z(0), brightness(0.5f) {}
    SpheroidParams(const std::string &name, float x, float y, float z, float aRadius, float cRadius, float theta_x, float theta_y, float theta_z, float brightness = 0.5f)
        : CellParams(name), x(x), y(y), z(z), aRadius(aRadius), bRadius(0), cRadius(cRadius), theta_x(theta_x), theta_y(theta_y), theta_z(theta_z), brightness(brightness) {}
};

// Per-cell snapshot captured at end of each frame. Authoritative source for
// next-frame split decisions. Driven by the triaxial fitted shape —
// shapeElongation = max(a,b,c) / min(a,b,c). Historical field names
// `longAxisDir` / `longAxisLength` are retained, but the active split
// pipeline now stores the chosen split axis there, which is the shortest
// fitted axis.
struct PreviousFrameSnapshot
{
    bool valid = false;
    float shapeElongation = 1.0f;
    cv::Point3f longAxisDir{1.0f, 0.0f, 0.0f};
    float longAxisLength = 0.0f;
    cv::Point3f position{0.0f, 0.0f, 0.0f};
    float aRadius = 0.0f;
    float bRadius = 0.0f;
    float cRadius = 0.0f;
    float thetaX = 0.0f;
    float thetaY = 0.0f;
    float thetaZ = 0.0f;
    float brightness = 0.5f;
};

struct PerturbDirections
{
    int brightness = 0;
    int aRadius = 0;
    int bRadius = 0;
    int cRadius = 0;
};

class Spheroid
{
    private:
        std::string _name;
        cv::Point3f _position;
        double _a_radius; // a axis
        double _b_radius;     // b axis (triaxial; oblate fallback = _a_radius when 0)
        double _c_radius; // c axis
        double a, b, c;
        double _theta_x;  // rotation angle about x-axis (radians)
        double _theta_y;  // rotation angle about y-axis (radians)
        double _theta_z;  // rotation angle about z-axis (radians)
        float _brightness; // per-cell rendering brightness [0,1]
        PerturbParams _aRadiusPerturbParams;
        PerturbParams _cRadiusPerturbParams;
        PerturbParams _bRadiusPerturbParams;
        PerturbParams _brightnessPerturbParams;

        // Inverse rotation: transforms world-space displacement back to local (upright) frame
        void inverseRotatePoint(double dx, double dy, double dz,
                                double &lx, double &ly, double &lz) const;

        void generateInverseRotationMatrix(std::array<double, 9> &R_T) const;

        bool computeSliceBounds(const cv::Mat &image, float z,
                                int &minX, int &maxX, int &minY, int &maxY) const;

    public:
        static SpheroidConfig cellConfig;
        
        Spheroid(const SpheroidParams &init_props);
        
        Spheroid() : _a_radius(0), _b_radius(0), _c_radius(0), _theta_x(0), _theta_y(0), _theta_z(0), _brightness(0.5f) {}

        // Lightweight accessors — avoid getCellParams() copy in tight loops
        const std::string& getName() const { return _name; }
        float getX() const { return _position.x; }
        float getY() const { return _position.y; }
        float getZ() const { return _position.z; }
        float getARadius() const { return static_cast<float>(_a_radius); }
        float getBRadius() const { return static_cast<float>(_b_radius); }
        float getCRadius() const { return static_cast<float>(_c_radius); }
        // Max/min of the three fitted semi-axes. Used by the triaxial plan as
        // the elongation signal for snapshot-based split classification.
        float shapeElongation() const {
            const double mn = std::min({_a_radius, _b_radius, _c_radius});
            const double mx = std::max({_a_radius, _b_radius, _c_radius});
            return (mn > 1e-6) ? static_cast<float>(mx / mn) : 1.0f;
        }

        // Historical name retained. Returns the world-space split axis and
        // its length. The active split pipeline defines this as the shortest
        // of the three fitted semi-axes and uses it to compute D1_seed/D2_seed
        // in snapshot-based split placement:
        //   D1_seed = snapshot.center - 0.5 * longAxisLength * longAxisDir
        //   D2_seed = snapshot.center + 0.5 * longAxisLength * longAxisDir
        void worldLongAxis(cv::Point3f &dir, float &length) const;
        float getBrightness() const { return _brightness; }
        float getARadiusIncreaseProbability() const { return _aRadiusPerturbParams.increase_prob; }
        float getARadiusDecreaseProbability() const { return _aRadiusPerturbParams.decrease_prob; }
        float getCRadiusIncreaseProbability() const { return _cRadiusPerturbParams.increase_prob; }
        float getCRadiusDecreaseProbability() const { return _cRadiusPerturbParams.decrease_prob; }
        float getBRadiusIncreaseProbability() const { return _bRadiusPerturbParams.increase_prob; }
        float getBRadiusDecreaseProbability() const { return _bRadiusPerturbParams.decrease_prob; }
        float getBrightnessIncreaseProbability() const { return _brightnessPerturbParams.increase_prob; }
        float getBrightnessDecreaseProbability() const { return _brightnessPerturbParams.decrease_prob; }
        void setBrightness(float brightness);
        void setARadiusPerturbProbabilities(float increaseProbability, float decreaseProbability);
        void setCRadiusPerturbProbabilities(float increaseProbability, float decreaseProbability);
        void setBRadiusPerturbProbabilities(float increaseProbability, float decreaseProbability);
        void setBrightnessPerturbProbabilities(float increaseProbability, float decreaseProbability);
        void blendAdaptivePerturbProbabilitiesWithConfig(float brightnessTrust,
                                                         float aRadiusTrust,
                                                         float cRadiusTrust,
                                                         float bRadiusTrust);
        void blendBrightnessPerturbProbabilitiesWithConfig(float trust);
        void adjustARadiusPerturbProbability(int direction, float delta);
        void adjustCRadiusPerturbProbability(int direction, float delta);
        void adjustBRadiusPerturbProbability(int direction, float delta);
        void adjustBrightnessPerturbProbability(int direction, float delta);
        float measureMeanBrightness(const std::vector<cv::Mat> &image) const;
        std::pair<float, float> measureBrightnessStats(const std::vector<cv::Mat> &image) const;

        void printCellInfo() const {
            std::cout << "Spheroid name: " << _name << " x: " << _position.x << " y: " << _position.y << " z: " << _position.z << " aRadius: " << _a_radius << " bRadius: " << _b_radius << " cRadius: " << _c_radius << " theta_x: " << _theta_x << " theta_y: " << _theta_y << " theta_z: " << _theta_z << " brightness: " << _brightness << '\n';
        }

        void draw(cv::Mat &image, const SimulationConfig &simulationConfig, float z = 0) const;

        void drawOutline(cv::Mat &image, float color, float z = 0) const;

        [[nodiscard]] Spheroid getPerturbedCell(PerturbDirections *directions = nullptr) const;

        // Test if point is inside this cell's (optionally scaled) ellipsoid
        bool isPointInsideEllipsoid(const cv::Point3f &worldPoint,
                                    float scaleFactor = 1.0f) const;

        bool checkConstraints() const;

        SpheroidParams getCellParams() const;

        [[nodiscard]] std::pair<std::vector<float>, std::vector<float>> calculateCorners() const;

        std::pair<std::vector<float>, std::vector<float>> calculateMinimumBox(Spheroid &perturbed_cell) const;

        cv::Point3f get_center() const;

        void print() const;
};

#endif
