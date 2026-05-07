#ifndef CALIBRATION_TYPES_HPP
#define CALIBRATION_TYPES_HPP

#include <algorithm>
#include <cstddef>
#include <sstream>
#include <string>

enum class BrightnessCalibrationSource {
    Uninitialized,
    AutoFrameZero,
    ManualOverride,
};

struct BrightnessCalibration {
    float background_intensity = 0.0f;
    float cell_intensity = 0.0f;
    std::size_t cell_pixel_count = 0;
    std::size_t background_pixel_count = 0;
    BrightnessCalibrationSource source = BrightnessCalibrationSource::Uninitialized;

    // Mean of frame 0's raw pixel values. Used by applyCalibratedPreprocess
    // for per-frame brightness-ratio adjustment of (background, cell) anchors
    // to handle photobleaching. See Change 16 plan
    // docs/plans/2026-05-05-auto-derive-geometry-from-initial-csv.md.
    float frame_0_mean = 0.0f;

    bool isValid() const {
        return (cell_intensity - background_intensity) > 1e-3f;
    }

    float normalize(float raw) const {
        if (!isValid()) return raw;
        const float span = cell_intensity - background_intensity;
        const float t = (raw - background_intensity) / span;
        return std::clamp(t, 0.0f, 1.0f);
    }
};

inline std::string calibrationDescribe(const BrightnessCalibration &cal) {
    std::ostringstream oss;
    const char *src = "uninitialized";
    switch (cal.source) {
        case BrightnessCalibrationSource::AutoFrameZero: src = "auto"; break;
        case BrightnessCalibrationSource::ManualOverride: src = "manual"; break;
        case BrightnessCalibrationSource::Uninitialized: src = "uninitialized"; break;
    }
    oss << "background=" << cal.background_intensity
        << " cell=" << cal.cell_intensity
        << " span=" << (cal.cell_intensity - cal.background_intensity)
        << " bg_pixels=" << cal.background_pixel_count
        << " cell_pixels=" << cal.cell_pixel_count
        << " source=" << src
        << " frame_0_mean=" << cal.frame_0_mean;
    return oss.str();
}

#endif // CALIBRATION_TYPES_HPP
