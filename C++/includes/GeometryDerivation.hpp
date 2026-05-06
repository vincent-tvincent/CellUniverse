#ifndef GEOMETRY_DERIVATION_HPP
#define GEOMETRY_DERIVATION_HPP

#include <cstddef>
#include <sstream>
#include <string>

// Image dimensions in raw (pre z-interpolation) coordinates. Used by
// computeDerivedGeometry as a future-proofing knob — currently the formulas
// don't depend on image_size, but it's plumbed so future enhancements can
// clamp derived bounds against image dimensions without changing the API.
struct ImageSize3D {
    int width = 0;
    int height = 0;
    int depth = 0;  // raw (pre z-interpolation) z-slice count
};

// Geometry parameters derived from initial-CSV statistics + z_scaling.
// Produced by CellUniverse::computeDerivedGeometry. Applied to
// Ellipsoid::cellConfig and BaseConfig before CellUniverse is constructed.
//
// All radius/sigma values are in PIXEL coordinates (interpolated z-space
// for *_c_radius and z_sigma; cell positions in the initial CSV are in
// raw z and CellFactory multiplies by z_scaling, so c-radius math here
// is in interpolated z just like the rest of the optimizer).
struct DerivedGeometry {
    float max_a_radius = 0.0f;
    float max_b_radius = 0.0f;
    float max_c_radius = 0.0f;
    float min_a_radius = 0.0f;
    float min_b_radius = 0.0f;
    float min_c_radius = 0.0f;
    float perturb_reference_radius = 0.0f;
    float x_sigma = 0.0f;
    float y_sigma = 0.0f;
    float z_sigma = 0.0f;
    float position_prior_threshold = 0.0f;
    int iterations_per_cell = 0;
    int pca_bridge_min_side_voxels = 0;
    std::size_t cell_count = 0;
    float mean_a_radius = 0.0f;
    float mean_b_radius = 0.0f;
    float mean_c_radius = 0.0f;

    // Span-only validity. A default-constructed DerivedGeometry has zero
    // max radius and zero iterations — clearly invalid. This matches the
    // BrightnessCalibration::isValid() span-only design from Change 14.
    bool isValid() const {
        return max_a_radius > 1e-3f
            && perturb_reference_radius > 1e-3f
            && iterations_per_cell > 0;
    }
};

inline std::string derivedGeometryDescribe(const DerivedGeometry &g) {
    std::ostringstream oss;
    oss << "max_a=" << g.max_a_radius
        << " max_b=" << g.max_b_radius
        << " max_c=" << g.max_c_radius
        << " min_a=" << g.min_a_radius
        << " min_b=" << g.min_b_radius
        << " min_c=" << g.min_c_radius
        << " ref_r=" << g.perturb_reference_radius
        << " xy_sigma=" << g.x_sigma
        << " z_sigma=" << g.z_sigma
        << " prior_thr=" << g.position_prior_threshold
        << " iters=" << g.iterations_per_cell
        << " bridge_min_voxels=" << g.pca_bridge_min_side_voxels
        << " n_cells=" << g.cell_count
        << " mean_a=" << g.mean_a_radius
        << " mean_b=" << g.mean_b_radius
        << " mean_c=" << g.mean_c_radius;
    return oss.str();
}

#endif // GEOMETRY_DERIVATION_HPP
