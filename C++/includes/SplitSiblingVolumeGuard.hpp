#pragma once

#include <algorithm>
#include <array>
#include <cmath>

struct SplitSiblingVolumeGuardResult {
    bool valid = false;
    bool changed = false;
    bool firstIsLarger = false;
    double firstVolume = 0.0;
    double secondVolume = 0.0;
    double beforeRatio = 1.0;
    double afterRatio = 1.0;
    float largerRadiusScale = 1.0f;
};

inline SplitSiblingVolumeGuardResult computeSplitSiblingVolumeGuard(
    const std::array<float, 3> &firstRadii,
    const std::array<float, 3> &secondRadii,
    float maxVolumeRatio)
{
    SplitSiblingVolumeGuardResult result;
    result.firstVolume =
        static_cast<double>(firstRadii[0]) * firstRadii[1] * firstRadii[2];
    result.secondVolume =
        static_cast<double>(secondRadii[0]) * secondRadii[1] * secondRadii[2];

    if (!std::isfinite(result.firstVolume) ||
        !std::isfinite(result.secondVolume) ||
        result.firstVolume <= 0.0 || result.secondVolume <= 0.0 ||
        !std::isfinite(maxVolumeRatio) || maxVolumeRatio < 1.0f) {
        return result;
    }

    result.valid = true;
    result.firstIsLarger = result.firstVolume >= result.secondVolume;
    const double larger = std::max(result.firstVolume, result.secondVolume);
    const double smaller = std::min(result.firstVolume, result.secondVolume);
    result.beforeRatio = larger / smaller;
    result.afterRatio = result.beforeRatio;
    if (result.beforeRatio <= static_cast<double>(maxVolumeRatio)) {
        return result;
    }

    const double targetLarger = static_cast<double>(maxVolumeRatio) * smaller;
    result.largerRadiusScale = static_cast<float>(
        std::cbrt(targetLarger / larger));
    result.afterRatio = static_cast<double>(maxVolumeRatio);
    result.changed =
        std::isfinite(result.largerRadiusScale) &&
        result.largerRadiusScale > 0.0f &&
        result.largerRadiusScale < 1.0f;
    return result;
}

struct SplitDaughterBirthVolumeEnvelopeResult {
    bool valid = false;
    bool changed = false;
    double currentVolume = 0.0;
    double birthVolume = 0.0;
    double targetVolume = 0.0;
    float radiusScale = 1.0f;
};

inline SplitDaughterBirthVolumeEnvelopeResult
computeSplitDaughterBirthVolumeEnvelope(
    const std::array<float, 3> &currentRadii,
    const std::array<float, 3> &birthRadii,
    float minBirthFactor,
    float maxBirthFactor)
{
    SplitDaughterBirthVolumeEnvelopeResult result;
    result.currentVolume = static_cast<double>(currentRadii[0]) *
        currentRadii[1] * currentRadii[2];
    result.birthVolume = static_cast<double>(birthRadii[0]) *
        birthRadii[1] * birthRadii[2];
    if (!std::isfinite(result.currentVolume) ||
        !std::isfinite(result.birthVolume) ||
        result.currentVolume <= 0.0 || result.birthVolume <= 0.0 ||
        !std::isfinite(minBirthFactor) ||
        !std::isfinite(maxBirthFactor) ||
        minBirthFactor <= 0.0f ||
        minBirthFactor > 1.0f ||
        maxBirthFactor < 1.0f ||
        maxBirthFactor < minBirthFactor) {
        return result;
    }

    result.valid = true;
    const double minVolume =
        result.birthVolume * static_cast<double>(minBirthFactor);
    const double maxVolume =
        result.birthVolume * static_cast<double>(maxBirthFactor);
    result.targetVolume =
        std::clamp(result.currentVolume, minVolume, maxVolume);
    if (std::abs(result.targetVolume - result.currentVolume) <=
        1e-9 * std::max(1.0, result.currentVolume)) {
        return result;
    }
    result.radiusScale = static_cast<float>(
        std::cbrt(result.targetVolume / result.currentVolume));
    result.changed = std::isfinite(result.radiusScale) &&
        result.radiusScale > 0.0f;
    return result;
}
