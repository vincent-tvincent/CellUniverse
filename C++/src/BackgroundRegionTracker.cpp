#include "BackgroundRegionTracker.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>

namespace
{
constexpr double kEpsilon = 1e-12;

using Matrix3 = std::array<double, 9>;
using Vector6 = cv::Vec<double, 6>;
using Matrix6 = cv::Matx<double, 6, 6>;

struct RadialGeometry
{
    double radial = 0.0;
    cv::Vec3d local{0.0, 0.0, 0.0};
    cv::Vec3d normalizedLocal{0.0, 0.0, 0.0};
    cv::Vec3d radialGradientWorld{0.0, 0.0, 0.0};
};

struct EvidenceSample
{
    Vector6 jacobian{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double residual = 0.0;
    double weight = 0.0;
};

template <typename T>
class DeterministicReservoir
{
public:
    DeterministicReservoir(std::size_t capacity, std::uint64_t seed)
        : capacity_(capacity), randomState_(seed)
    {
        values_.reserve(capacity_);
    }

    void consider(const T &value)
    {
        ++seen_;
        if (capacity_ == 0) {
            return;
        }
        if (values_.size() < capacity_) {
            values_.push_back(value);
            return;
        }

        // Deterministic Algorithm-R reservoir sampling. This keeps memory
        // bounded without biasing samples toward the first z-slices.
        randomState_ = randomState_ * 6364136223846793005ULL +
                       1442695040888963407ULL;
        const std::size_t replacement =
            static_cast<std::size_t>(randomState_ % seen_);
        if (replacement < capacity_) {
            values_[replacement] = value;
        }
    }

    const std::vector<T> &values() const noexcept { return values_; }
    std::size_t seen() const noexcept { return seen_; }

private:
    std::size_t capacity_ = 0;
    std::size_t seen_ = 0;
    std::uint64_t randomState_ = 0;
    std::vector<T> values_;
};

bool finiteFloat(float value)
{
    return std::isfinite(static_cast<double>(value));
}

bool finitePoint(const cv::Point3f &point)
{
    return finiteFloat(point.x) && finiteFloat(point.y) && finiteFloat(point.z);
}

bool finiteVector(const cv::Vec3f &value)
{
    return finiteFloat(value[0]) &&
           finiteFloat(value[1]) &&
           finiteFloat(value[2]);
}

float clampUnit(float value)
{
    return std::clamp(value, 0.0f, 1.0f);
}

int countSetBits(unsigned int bits)
{
    int count = 0;
    while (bits != 0U) {
        count += static_cast<int>(bits & 1U);
        bits >>= 1U;
    }
    return count;
}

Matrix3 makeRotationMatrix(const cv::Vec3f &rotation)
{
    const double sx = std::sin(static_cast<double>(rotation[0]));
    const double cx = std::cos(static_cast<double>(rotation[0]));
    const double sy = std::sin(static_cast<double>(rotation[1]));
    const double cy = std::cos(static_cast<double>(rotation[1]));
    const double sz = std::sin(static_cast<double>(rotation[2]));
    const double cz = std::cos(static_cast<double>(rotation[2]));

    // R = Rz * Ry * Rx, stored row-major.
    return Matrix3{
        cz * cy,
        cz * sy * sx - sz * cx,
        cz * sy * cx + sz * sx,
        sz * cy,
        sz * sy * sx + cz * cx,
        sz * sy * cx - cz * sx,
        -sy,
        cy * sx,
        cy * cx
    };
}

cv::Vec3d worldToLocal(const Matrix3 &rotation,
                       const cv::Point3f &center,
                       const cv::Point3f &worldPoint)
{
    const double dx = static_cast<double>(worldPoint.x - center.x);
    const double dy = static_cast<double>(worldPoint.y - center.y);
    const double dz = static_cast<double>(worldPoint.z - center.z);

    // local = R^T * (world - center).
    return cv::Vec3d{
        rotation[0] * dx + rotation[3] * dy + rotation[6] * dz,
        rotation[1] * dx + rotation[4] * dy + rotation[7] * dz,
        rotation[2] * dx + rotation[5] * dy + rotation[8] * dz
    };
}

cv::Vec3d localToWorldDirection(const Matrix3 &rotation,
                                const cv::Vec3d &localDirection)
{
    return cv::Vec3d{
        rotation[0] * localDirection[0] +
            rotation[1] * localDirection[1] +
            rotation[2] * localDirection[2],
        rotation[3] * localDirection[0] +
            rotation[4] * localDirection[1] +
            rotation[5] * localDirection[2],
        rotation[6] * localDirection[0] +
            rotation[7] * localDirection[1] +
            rotation[8] * localDirection[2]
    };
}

bool calculateRadialGeometry(const BackgroundRegionTracker::State &state,
                             const Matrix3 &rotation,
                             const cv::Point3f &worldPoint,
                             RadialGeometry &out)
{
    if (state.radii[0] <= 0.0f ||
        state.radii[1] <= 0.0f ||
        state.radii[2] <= 0.0f) {
        return false;
    }

    out.local = worldToLocal(rotation, state.center, worldPoint);
    for (int axis = 0; axis < 3; ++axis) {
        out.normalizedLocal[axis] =
            out.local[axis] / static_cast<double>(state.radii[axis]);
    }
    const double radialSquared = out.normalizedLocal.dot(out.normalizedLocal);
    out.radial = std::sqrt(std::max(0.0, radialSquared));

    if (out.radial > kEpsilon) {
        const cv::Vec3d localGradient{
            out.local[0] /
                (out.radial * static_cast<double>(state.radii[0]) *
                 static_cast<double>(state.radii[0])),
            out.local[1] /
                (out.radial * static_cast<double>(state.radii[1]) *
                 static_cast<double>(state.radii[1])),
            out.local[2] /
                (out.radial * static_cast<double>(state.radii[2]) *
                 static_cast<double>(state.radii[2]))
        };
        out.radialGradientWorld =
            localToWorldDirection(rotation, localGradient);
    } else {
        out.radialGradientWorld = cv::Vec3d{0.0, 0.0, 0.0};
    }
    return std::isfinite(out.radial);
}

float membershipFromRadial(double radial, float softMargin)
{
    if (!std::isfinite(radial)) {
        return 0.0f;
    }
    if (softMargin <= 1e-8f) {
        return radial <= 1.0 ? 1.0f : 0.0f;
    }

    // This exactly mirrors the initializer contract:
    // u = clamp((1 + margin - radial) / (2 * margin), 0, 1)
    // weight = u^2 * (3 - 2u)
    const double u = std::clamp(
        (1.0 + static_cast<double>(softMargin) - radial) /
            (2.0 * static_cast<double>(softMargin)),
        0.0,
        1.0);
    return static_cast<float>(u * u * (3.0 - 2.0 * u));
}

float membershipAtState(const BackgroundRegionTracker::State &state,
                        const Matrix3 &rotation,
                        const cv::Point3f &worldPoint)
{
    RadialGeometry radial;
    if (!calculateRadialGeometry(state, rotation, worldPoint, radial)) {
        return 0.0f;
    }
    return membershipFromRadial(radial.radial, state.softMargin);
}

void validateSeed(const BackgroundRegionTracker::SeedRecord &seed)
{
    if (!finitePoint(seed.center) ||
        !finiteVector(seed.radii) ||
        !finiteVector(seed.rotation) ||
        !finiteFloat(seed.coldBackground) ||
        !finiteFloat(seed.hotBackground) ||
        !finiteFloat(seed.softMargin)) {
        throw std::invalid_argument(
            "BackgroundRegionTracker seed contains a non-finite value");
    }
    if (seed.radii[0] <= 0.0f ||
        seed.radii[1] <= 0.0f ||
        seed.radii[2] <= 0.0f) {
        throw std::invalid_argument(
            "BackgroundRegionTracker seed radii must be positive");
    }
    if (seed.softMargin < 0.0f) {
        throw std::invalid_argument(
            "BackgroundRegionTracker soft margin must be nonnegative");
    }
}

void validateOptions(const BackgroundRegionTracker::Options &options)
{
    if (!finiteFloat(options.shellHalfWidth) ||
        options.shellHalfWidth <= 0.0f ||
        options.shellHalfWidth >= 1.0f) {
        throw std::invalid_argument(
            "BackgroundRegionTracker shellHalfWidth must be in (0, 1)");
    }
    if (options.geometryStrideXY <= 0 ||
        options.geometryStrideZ <= 0 ||
        options.intensityStride <= 0) {
        throw std::invalid_argument(
            "BackgroundRegionTracker sample strides must be positive");
    }
    if (!finiteFloat(options.gaussianSigmaXY) ||
        options.gaussianSigmaXY < 0.0f ||
        !finiteFloat(options.minimumOrientedGradient) ||
        options.minimumOrientedGradient < 0.0f ||
        options.minimumEvidenceSamples == 0 ||
        options.maximumEvidenceSamples == 0 ||
        options.maximumEvidenceSamples < options.minimumEvidenceSamples ||
        options.minimumCoveredFaces < 1 ||
        options.minimumCoveredFaces > 6 ||
        !finiteFloat(options.minimumEvidenceFraction) ||
        options.minimumEvidenceFraction <= 0.0f ||
        !finiteFloat(options.minimumConfidence) ||
        options.minimumConfidence < 0.0f ||
        options.minimumConfidence > 1.0f) {
        throw std::invalid_argument(
            "BackgroundRegionTracker geometry evidence options are invalid");
    }
    if (options.irlsIterations <= 0 ||
        !finiteFloat(options.huberScale) ||
        options.huberScale <= 0.0f ||
        !finiteFloat(options.geometryEmaAlpha) ||
        options.geometryEmaAlpha < 0.0f ||
        options.geometryEmaAlpha > 1.0f ||
        !finiteFloat(options.maximumCenterShiftFraction) ||
        options.maximumCenterShiftFraction <= 0.0f ||
        !finiteFloat(options.maximumCenterShiftPixels) ||
        options.maximumCenterShiftPixels <= 0.0f ||
        !finiteFloat(options.maximumRadiusChangeFraction) ||
        options.maximumRadiusChangeFraction <= 0.0f ||
        !finiteFloat(options.maximumRawUpdateMultiplier) ||
        options.maximumRawUpdateMultiplier < 1.0f ||
        !finiteFloat(options.minimumRadius) ||
        options.minimumRadius <= 0.0f ||
        !finiteFloat(options.cellExclusionScale) ||
        options.cellExclusionScale <= 0.0f) {
        throw std::invalid_argument(
            "BackgroundRegionTracker geometry update options are invalid");
    }
    if (options.minimumIntensitySamples == 0 ||
        options.maximumIntensitySamplesPerRegion == 0 ||
        options.maximumIntensitySamplesPerRegion <
            options.minimumIntensitySamples ||
        !finiteFloat(options.intensityTrimFraction) ||
        options.intensityTrimFraction < 0.0f ||
        options.intensityTrimFraction >= 0.5f ||
        !finiteFloat(options.intensityEmaAlpha) ||
        options.intensityEmaAlpha < 0.0f ||
        options.intensityEmaAlpha > 1.0f ||
        !finiteFloat(options.maximumIntensityStep) ||
        options.maximumIntensityStep < 0.0f ||
        !finiteFloat(options.hotMembershipMinimum) ||
        !finiteFloat(options.coldMembershipMaximum) ||
        options.hotMembershipMinimum < 0.0f ||
        options.hotMembershipMinimum > 1.0f ||
        options.coldMembershipMaximum < 0.0f ||
        options.coldMembershipMaximum > 1.0f ||
        options.coldMembershipMaximum >= options.hotMembershipMinimum ||
        !finiteFloat(options.minimumRegionContrast) ||
        options.minimumRegionContrast < 0.0f) {
        throw std::invalid_argument(
            "BackgroundRegionTracker intensity options are invalid");
    }
}

std::vector<cv::Mat> makeFloatStack(const std::vector<cv::Mat> &frame,
                                    int &rows,
                                    int &cols)
{
    if (frame.empty()) {
        throw std::invalid_argument(
            "BackgroundRegionTracker update requires a nonempty frame");
    }

    rows = frame.front().rows;
    cols = frame.front().cols;
    if (rows <= 0 || cols <= 0 || frame.front().channels() != 1) {
        throw std::invalid_argument(
            "BackgroundRegionTracker requires nonempty single-channel slices");
    }

    std::vector<cv::Mat> result;
    result.reserve(frame.size());
    for (const cv::Mat &slice : frame) {
        if (slice.empty() ||
            slice.rows != rows ||
            slice.cols != cols ||
            slice.channels() != 1) {
            throw std::invalid_argument(
                "BackgroundRegionTracker frame slices must have one shape and one channel");
        }
        cv::Mat converted;
        slice.convertTo(converted, CV_32F);
        result.push_back(std::move(converted));
    }
    return result;
}

std::vector<cv::Mat> makeSmoothedStack(
    const std::vector<cv::Mat> &frame,
    const BackgroundRegionTracker::Options &options)
{
    std::vector<cv::Mat> xySmoothed;
    xySmoothed.reserve(frame.size());
    for (const cv::Mat &slice : frame) {
        cv::Mat smoothed;
        if (options.gaussianSigmaXY > 1e-6f) {
            cv::GaussianBlur(slice,
                             smoothed,
                             cv::Size(),
                             options.gaussianSigmaXY,
                             options.gaussianSigmaXY,
                             cv::BORDER_REPLICATE);
        } else {
            smoothed = slice.clone();
        }
        xySmoothed.push_back(std::move(smoothed));
    }

    if (!options.smoothAcrossZ || xySmoothed.size() <= 1) {
        return xySmoothed;
    }

    std::vector<cv::Mat> result(xySmoothed.size());
    for (std::size_t z = 0; z < xySmoothed.size(); ++z) {
        const std::size_t previous = z > 0 ? z - 1 : z;
        const std::size_t next =
            z + 1 < xySmoothed.size() ? z + 1 : z;
        if (previous == z && next == z) {
            result[z] = xySmoothed[z].clone();
        } else if (previous == z) {
            cv::addWeighted(xySmoothed[z], 0.75,
                            xySmoothed[next], 0.25,
                            0.0, result[z]);
        } else if (next == z) {
            cv::addWeighted(xySmoothed[previous], 0.25,
                            xySmoothed[z], 0.75,
                            0.0, result[z]);
        } else {
            cv::Mat neighbors;
            cv::addWeighted(xySmoothed[previous], 0.5,
                            xySmoothed[next], 0.5,
                            0.0, neighbors);
            cv::addWeighted(xySmoothed[z], 0.5,
                            neighbors, 0.5,
                            0.0, result[z]);
        }
    }
    return result;
}

bool pointExcludedByCells(const cv::Point3f &point,
                          const std::vector<Ellipsoid> &cells,
                          float exclusionScale)
{
    for (const Ellipsoid &cell : cells) {
        const float maximumRadius =
            std::max({cell.getARadius(),
                      cell.getBRadius(),
                      cell.getCRadius()}) * exclusionScale;
        if (!finiteFloat(maximumRadius) || maximumRadius <= 0.0f) {
            continue;
        }

        const float dx = point.x - cell.getX();
        const float dy = point.y - cell.getY();
        const float dz = point.z - cell.getZ();
        if (dx * dx + dy * dy + dz * dz >
            maximumRadius * maximumRadius) {
            continue;
        }
        if (cell.isPointInsideEllipsoid(point, exclusionScale)) {
            return true;
        }
    }
    return false;
}

double medianOf(std::vector<double> values)
{
    if (values.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const std::size_t middle = values.size() / 2;
    std::nth_element(values.begin(),
                     values.begin() + static_cast<std::ptrdiff_t>(middle),
                     values.end());
    const double upper = values[middle];
    if ((values.size() & 1U) != 0U) {
        return upper;
    }
    const auto lowerIt =
        std::max_element(values.begin(),
                         values.begin() + static_cast<std::ptrdiff_t>(middle));
    return 0.5 * (upper + *lowerIt);
}

double dotJacobian(const Vector6 &jacobian, const Vector6 &value)
{
    double result = 0.0;
    for (int index = 0; index < 6; ++index) {
        result += jacobian[index] * value[index];
    }
    return result;
}

bool solveRobustGeometry(const std::vector<EvidenceSample> &samples,
                         const BackgroundRegionTracker::Options &options,
                         Vector6 &solution,
                         double &medianAbsoluteResidual)
{
    if (samples.size() < 6) {
        return false;
    }

    std::vector<double> gradientWeights;
    gradientWeights.reserve(samples.size());
    for (const EvidenceSample &sample : samples) {
        if (std::isfinite(sample.weight) && sample.weight > 0.0) {
            gradientWeights.push_back(sample.weight);
        }
    }
    const double medianGradient = medianOf(gradientWeights);
    if (!std::isfinite(medianGradient) || medianGradient <= 0.0) {
        return false;
    }
    const double maximumBaseWeight = std::max(
        medianGradient * 4.0,
        static_cast<double>(options.minimumOrientedGradient));

    solution = Vector6::all(0.0);
    for (int iteration = 0; iteration < options.irlsIterations; ++iteration) {
        std::vector<double> currentAbsoluteResiduals;
        currentAbsoluteResiduals.reserve(samples.size());
        for (const EvidenceSample &sample : samples) {
            currentAbsoluteResiduals.push_back(std::abs(
                sample.residual +
                dotJacobian(sample.jacobian, solution)));
        }
        const double medianResidual = medianOf(currentAbsoluteResiduals);
        if (!std::isfinite(medianResidual)) {
            return false;
        }
        const double robustSigma =
            std::max(1e-4, 1.4826 * medianResidual);
        const double huberCutoff =
            std::max(1e-4,
                     static_cast<double>(options.huberScale) * robustSigma);

        Matrix6 normal = Matrix6::zeros();
        Vector6 rightHandSide = Vector6::all(0.0);
        for (const EvidenceSample &sample : samples) {
            const double currentResidual =
                sample.residual +
                dotJacobian(sample.jacobian, solution);
            const double absoluteResidual = std::abs(currentResidual);
            const double huberWeight =
                absoluteResidual <= huberCutoff
                    ? 1.0
                    : huberCutoff / std::max(absoluteResidual, kEpsilon);
            const double weight =
                std::min(sample.weight, maximumBaseWeight) * huberWeight;
            if (!std::isfinite(weight) || weight <= 0.0) {
                continue;
            }

            for (int row = 0; row < 6; ++row) {
                rightHandSide[row] -=
                    weight * sample.jacobian[row] * sample.residual;
                for (int col = 0; col < 6; ++col) {
                    normal(row, col) +=
                        weight *
                        sample.jacobian[row] *
                        sample.jacobian[col];
                }
            }
        }

        double diagonalMean = 0.0;
        for (int axis = 0; axis < 6; ++axis) {
            diagonalMean += normal(axis, axis);
        }
        diagonalMean /= 6.0;
        const double ridge = std::max(1e-12, diagonalMean * 1e-8);
        for (int axis = 0; axis < 6; ++axis) {
            normal(axis, axis) += ridge;
        }

        Vector6 next = Vector6::all(0.0);
        if (!cv::solve(normal, rightHandSide, next, cv::DECOMP_SVD)) {
            return false;
        }
        for (int axis = 0; axis < 6; ++axis) {
            if (!std::isfinite(next[axis])) {
                return false;
            }
        }

        double changeSquared = 0.0;
        for (int axis = 0; axis < 6; ++axis) {
            const double difference = next[axis] - solution[axis];
            changeSquared += difference * difference;
        }
        solution = next;
        if (changeSquared <= 1e-12) {
            break;
        }
    }

    std::vector<double> finalAbsoluteResiduals;
    finalAbsoluteResiduals.reserve(samples.size());
    for (const EvidenceSample &sample : samples) {
        finalAbsoluteResiduals.push_back(std::abs(
            sample.residual +
            dotJacobian(sample.jacobian, solution)));
    }
    medianAbsoluteResidual = medianOf(finalAbsoluteResiduals);
    return std::isfinite(medianAbsoluteResidual);
}

float robustTrimmedMean(const std::vector<float> &input,
                        float trimFraction)
{
    if (input.empty()) {
        return std::numeric_limits<float>::quiet_NaN();
    }

    std::vector<float> values;
    values.reserve(input.size());
    for (float value : input) {
        if (finiteFloat(value)) {
            values.push_back(value);
        }
    }
    if (values.empty()) {
        return std::numeric_limits<float>::quiet_NaN();
    }

    std::sort(values.begin(), values.end());
    std::size_t trim = static_cast<std::size_t>(
        std::floor(static_cast<double>(values.size()) *
                   static_cast<double>(trimFraction)));
    if (2 * trim >= values.size()) {
        trim = 0;
    }

    double sum = 0.0;
    for (std::size_t index = trim;
         index < values.size() - trim;
         ++index) {
        sum += static_cast<double>(values[index]);
    }
    const std::size_t count = values.size() - 2 * trim;
    return count > 0
        ? static_cast<float>(sum / static_cast<double>(count))
        : std::numeric_limits<float>::quiet_NaN();
}

float updateScalarConservatively(
    float current,
    float seed,
    float candidate,
    const BackgroundRegionTracker::Options &options)
{
    float base = finiteFloat(current) ? current : seed;
    if (!finiteFloat(candidate)) {
        return base;
    }
    if (options.clampBackgroundToUnitRange) {
        base = clampUnit(base);
        candidate = clampUnit(candidate);
    }
    const float proposedDelta =
        options.intensityEmaAlpha * (candidate - base);
    const float boundedDelta = options.intensityStepLimitEnabled
        ? std::clamp(
              proposedDelta,
              -options.maximumIntensityStep,
              options.maximumIntensityStep)
        : proposedDelta;
    float updated = base + boundedDelta;
    if (options.clampBackgroundToUnitRange) {
        updated = clampUnit(updated);
    }
    return finiteFloat(updated) ? updated : seed;
}

void estimateRegionIntensities(
    const std::vector<cv::Mat> &frame,
    const BackgroundRegionTracker::State &state,
    const BackgroundRegionTracker::Options &options,
    const std::vector<Ellipsoid> &cells,
    float &coldEstimate,
    float &hotEstimate,
    std::size_t &coldCount,
    std::size_t &hotCount)
{
    DeterministicReservoir<float> coldSamples(
        options.maximumIntensitySamplesPerRegion,
        0x853c49e6748fea9bULL);
    DeterministicReservoir<float> hotSamples(
        options.maximumIntensitySamplesPerRegion,
        0xda3e39cb94b95bdbULL);

    const Matrix3 rotation = makeRotationMatrix(state.rotation);
    const int stride = options.intensityStride;
    const int offset = stride / 2;
    for (int z = std::min(offset, static_cast<int>(frame.size()) - 1);
         z < static_cast<int>(frame.size());
         z += stride) {
        const cv::Mat &slice = frame[static_cast<std::size_t>(z)];
        for (int y = std::min(offset, slice.rows - 1);
             y < slice.rows;
             y += stride) {
            const float *row = slice.ptr<float>(y);
            for (int x = std::min(offset, slice.cols - 1);
                 x < slice.cols;
                 x += stride) {
                const float value = row[x];
                if (!finiteFloat(value)) {
                    continue;
                }
                const cv::Point3f point(
                    static_cast<float>(x),
                    static_cast<float>(y),
                    static_cast<float>(z));
                if (pointExcludedByCells(
                        point, cells, options.cellExclusionScale)) {
                    continue;
                }

                const float membership =
                    membershipAtState(state, rotation, point);
                if (membership >= options.hotMembershipMinimum) {
                    hotSamples.consider(value);
                } else if (membership <= options.coldMembershipMaximum) {
                    coldSamples.consider(value);
                }
            }
        }
    }

    coldCount = coldSamples.seen();
    hotCount = hotSamples.seen();
    coldEstimate =
        coldCount >= options.minimumIntensitySamples
            ? robustTrimmedMean(coldSamples.values(),
                                options.intensityTrimFraction)
            : std::numeric_limits<float>::quiet_NaN();
    hotEstimate =
        hotCount >= options.minimumIntensitySamples
            ? robustTrimmedMean(hotSamples.values(),
                                options.intensityTrimFraction)
            : std::numeric_limits<float>::quiet_NaN();
}

} // namespace

BackgroundRegionTracker::BackgroundRegionTracker(const SeedRecord &seed)
{
    configure(seed, Options{});
}

BackgroundRegionTracker::BackgroundRegionTracker(
    const SeedRecord &seed,
    const Options &options)
{
    configure(seed, options);
}

void BackgroundRegionTracker::configure(const SeedRecord &seed)
{
    configure(seed, Options{});
}

void BackgroundRegionTracker::configure(
    const SeedRecord &seed,
    const Options &options)
{
    validateSeed(seed);
    validateOptions(options);

    seed_ = seed;
    options_ = options;
    if (options_.clampBackgroundToUnitRange) {
        seed_.coldBackground = clampUnit(seed_.coldBackground);
        seed_.hotBackground = clampUnit(seed_.hotBackground);
    }
    configured_ = true;
    reset();
}

void BackgroundRegionTracker::reset()
{
    updatesSeen_ = 0;
    lastZSlices_ = 0;
    lastRows_ = 0;
    lastCols_ = 0;

    state_ = State{};
    if (!configured_) {
        return;
    }
    state_.center = seed_.center;
    state_.radii = seed_.radii;
    state_.rotation = seed_.rotation;
    state_.coldBackground = seed_.coldBackground;
    state_.hotBackground = seed_.hotBackground;
    state_.softMargin = seed_.softMargin;
    state_.frameIndex = -1;
    state_.confidence = 1.0f;
    state_.frozen = true;
}

bool BackgroundRegionTracker::update(
    const std::vector<cv::Mat> &currentFrame,
    const std::vector<Ellipsoid> &cells)
{
    const int inferredFrame =
        state_.frameIndex >= 0 ? state_.frameIndex + 1 : 0;
    return update(inferredFrame, currentFrame, cells);
}

bool BackgroundRegionTracker::update(
    int frameIndex,
    const std::vector<cv::Mat> &currentFrame,
    const std::vector<Ellipsoid> &cells)
{
    if (!configured_) {
        throw std::logic_error(
            "BackgroundRegionTracker must be configured before update");
    }

    int rows = 0;
    int cols = 0;
    const std::vector<cv::Mat> floatFrame =
        makeFloatStack(currentFrame, rows, cols);
    lastZSlices_ = static_cast<int>(floatFrame.size());
    lastRows_ = rows;
    lastCols_ = cols;

    const bool firstUpdate = updatesSeen_ == 0;
    const Matrix3 rotation = makeRotationMatrix(state_.rotation);
    const float minimumRadius =
        std::min({state_.radii[0], state_.radii[1], state_.radii[2]});
    const double fitCenterScale =
        std::max(static_cast<double>(options_.minimumRadius),
                 static_cast<double>(minimumRadius));

    std::size_t shellSamples = 0;
    std::size_t evidenceSamples = 0;
    unsigned int coveredFaces = 0U;
    float geometryConfidence = 0.0f;
    bool geometryAccepted = false;

    if (!(firstUpdate && options_.holdSeedGeometryOnFirstUpdate)) {
        const std::vector<cv::Mat> smoothed =
            makeSmoothedStack(floatFrame, options_);
        DeterministicReservoir<EvidenceSample> evidence(
            options_.maximumEvidenceSamples,
            0x9e3779b97f4a7c15ULL +
                static_cast<std::uint64_t>(
                    std::max(frameIndex, 0)));

        const double supportScale =
            1.0 + static_cast<double>(options_.shellHalfWidth);
        const double extentX = supportScale * std::sqrt(
            std::pow(rotation[0] * state_.radii[0], 2.0) +
            std::pow(rotation[1] * state_.radii[1], 2.0) +
            std::pow(rotation[2] * state_.radii[2], 2.0));
        const double extentY = supportScale * std::sqrt(
            std::pow(rotation[3] * state_.radii[0], 2.0) +
            std::pow(rotation[4] * state_.radii[1], 2.0) +
            std::pow(rotation[5] * state_.radii[2], 2.0));
        const double extentZ = supportScale * std::sqrt(
            std::pow(rotation[6] * state_.radii[0], 2.0) +
            std::pow(rotation[7] * state_.radii[1], 2.0) +
            std::pow(rotation[8] * state_.radii[2], 2.0));

        const int xMinimum = std::max(
            1,
            static_cast<int>(std::floor(state_.center.x - extentX)));
        const int xMaximum = std::min(
            cols - 2,
            static_cast<int>(std::ceil(state_.center.x + extentX)));
        const int yMinimum = std::max(
            1,
            static_cast<int>(std::floor(state_.center.y - extentY)));
        const int yMaximum = std::min(
            rows - 2,
            static_cast<int>(std::ceil(state_.center.y + extentY)));
        const int zMinimum = std::max(
            0,
            static_cast<int>(std::floor(state_.center.z - extentZ)));
        const int zMaximum = std::min(
            static_cast<int>(smoothed.size()) - 1,
            static_cast<int>(std::ceil(state_.center.z + extentZ)));

        const double expectedGradientSign =
            state_.hotBackground >= state_.coldBackground ? -1.0 : 1.0;
        if (xMinimum <= xMaximum &&
            yMinimum <= yMaximum &&
            zMinimum <= zMaximum &&
            std::abs(state_.hotBackground -
                     state_.coldBackground) >=
                options_.minimumRegionContrast) {
            for (int z = zMinimum;
                 z <= zMaximum;
                 z += options_.geometryStrideZ) {
                const cv::Mat &slice =
                    smoothed[static_cast<std::size_t>(z)];
                for (int y = yMinimum;
                     y <= yMaximum;
                     y += options_.geometryStrideXY) {
                    for (int x = xMinimum;
                         x <= xMaximum;
                         x += options_.geometryStrideXY) {
                        const cv::Point3f point(
                            static_cast<float>(x),
                            static_cast<float>(y),
                            static_cast<float>(z));
                        RadialGeometry radial;
                        if (!calculateRadialGeometry(
                                state_, rotation, point, radial) ||
                            std::abs(radial.radial - 1.0) >
                                options_.shellHalfWidth) {
                            continue;
                        }
                        ++shellSamples;

                        if (pointExcludedByCells(
                                point,
                                cells,
                                options_.cellExclusionScale)) {
                            continue;
                        }

                        const double radialGradientNorm =
                            cv::norm(radial.radialGradientWorld);
                        if (radialGradientNorm <= kEpsilon) {
                            continue;
                        }
                        const cv::Vec3d outwardNormal =
                            radial.radialGradientWorld /
                            radialGradientNorm;

                        const float *row = slice.ptr<float>(y);
                        const double imageGradientX =
                            0.5 * static_cast<double>(
                                row[x + 1] - row[x - 1]);
                        const double imageGradientY =
                            0.5 * static_cast<double>(
                                slice.ptr<float>(y + 1)[x] -
                                slice.ptr<float>(y - 1)[x]);
                        double imageGradientZ = 0.0;
                        if (z > 0 &&
                            z + 1 < static_cast<int>(smoothed.size())) {
                            imageGradientZ =
                                0.5 * static_cast<double>(
                                    smoothed[static_cast<std::size_t>(z + 1)]
                                            .ptr<float>(y)[x] -
                                    smoothed[static_cast<std::size_t>(z - 1)]
                                            .ptr<float>(y)[x]);
                        } else if (z + 1 <
                                   static_cast<int>(smoothed.size())) {
                            imageGradientZ = static_cast<double>(
                                smoothed[static_cast<std::size_t>(z + 1)]
                                    .ptr<float>(y)[x] -
                                row[x]);
                        } else if (z > 0) {
                            imageGradientZ = static_cast<double>(
                                row[x] -
                                smoothed[static_cast<std::size_t>(z - 1)]
                                    .ptr<float>(y)[x]);
                        }

                        const cv::Vec3d imageGradient{
                            imageGradientX,
                            imageGradientY,
                            imageGradientZ
                        };
                        const double orientedGradient =
                            expectedGradientSign *
                            imageGradient.dot(outwardNormal);
                        if (!std::isfinite(orientedGradient) ||
                            orientedGradient <
                                options_.minimumOrientedGradient) {
                            continue;
                        }

                        EvidenceSample sample;
                        // Unknown center variables are normalized by
                        // fitCenterScale so center and log-radius columns have
                        // comparable conditioning.
                        sample.jacobian[0] =
                            -radial.radialGradientWorld[0] *
                            fitCenterScale;
                        sample.jacobian[1] =
                            -radial.radialGradientWorld[1] *
                            fitCenterScale;
                        sample.jacobian[2] =
                            -radial.radialGradientWorld[2] *
                            fitCenterScale;
                        sample.jacobian[3] =
                            -radial.normalizedLocal[0] *
                            radial.normalizedLocal[0] /
                            std::max(radial.radial, kEpsilon);
                        sample.jacobian[4] =
                            -radial.normalizedLocal[1] *
                            radial.normalizedLocal[1] /
                            std::max(radial.radial, kEpsilon);
                        sample.jacobian[5] =
                            -radial.normalizedLocal[2] *
                            radial.normalizedLocal[2] /
                            std::max(radial.radial, kEpsilon);
                        sample.residual = radial.radial - 1.0;
                        sample.weight = orientedGradient;
                        evidence.consider(sample);

                        int dominantAxis = 0;
                        if (std::abs(radial.normalizedLocal[1]) >
                            std::abs(radial.normalizedLocal[dominantAxis])) {
                            dominantAxis = 1;
                        }
                        if (std::abs(radial.normalizedLocal[2]) >
                            std::abs(radial.normalizedLocal[dominantAxis])) {
                            dominantAxis = 2;
                        }
                        const int face =
                            2 * dominantAxis +
                            (radial.normalizedLocal[dominantAxis] >= 0.0
                                 ? 1
                                 : 0);
                        coveredFaces |= (1U << static_cast<unsigned int>(face));
                    }
                }
            }
        }

        evidenceSamples = evidence.seen();
        Vector6 fit = Vector6::all(0.0);
        double medianResidual = 0.0;
        const bool fitSolved =
            evidenceSamples >= options_.minimumEvidenceSamples &&
            countSetBits(coveredFaces) >= options_.minimumCoveredFaces &&
            solveRobustGeometry(
                evidence.values(), options_, fit, medianResidual);

        if (fitSolved) {
            std::vector<double> gradientWeights;
            gradientWeights.reserve(evidence.values().size());
            for (const EvidenceSample &sample : evidence.values()) {
                gradientWeights.push_back(sample.weight);
            }
            const double medianGradient = medianOf(gradientWeights);

            const double evidenceFactor = std::min(
                1.0,
                static_cast<double>(evidenceSamples) /
                    static_cast<double>(
                        options_.minimumEvidenceSamples));
            const double evidenceFraction =
                shellSamples > 0
                    ? static_cast<double>(evidenceSamples) /
                        static_cast<double>(shellSamples)
                    : 0.0;
            const double fractionFactor = std::min(
                1.0,
                evidenceFraction /
                    static_cast<double>(
                        options_.minimumEvidenceFraction));
            const double faceFactor = std::min(
                1.0,
                static_cast<double>(countSetBits(coveredFaces)) /
                    static_cast<double>(
                        options_.minimumCoveredFaces));
            const double gradientFactor =
                std::isfinite(medianGradient) && medianGradient > 0.0
                    ? medianGradient /
                        (medianGradient +
                         static_cast<double>(
                             options_.minimumOrientedGradient))
                    : 0.0;
            const double residualFactor = std::clamp(
                1.0 -
                    medianResidual /
                        static_cast<double>(
                            options_.shellHalfWidth),
                0.0,
                1.0);

            cv::Vec3d rawCenterShift{
                fit[0] * fitCenterScale,
                fit[1] * fitCenterScale,
                fit[2] * fitCenterScale
            };
            const double rawCenterMagnitude =
                cv::norm(rawCenterShift);
            const double fractionalCenterCap =
                fitCenterScale *
                static_cast<double>(
                    options_.maximumCenterShiftFraction);
            const double centerCap = std::max(
                0.25,
                std::min(
                    static_cast<double>(
                        options_.maximumCenterShiftPixels),
                    fractionalCenterCap));
            const double rawRadiusMagnitude =
                std::max({std::abs(fit[3]),
                          std::abs(fit[4]),
                          std::abs(fit[5])});

            const bool implausible =
                rawCenterMagnitude >
                    centerCap *
                    static_cast<double>(
                        options_.maximumRawUpdateMultiplier) ||
                rawRadiusMagnitude >
                    static_cast<double>(
                        options_.maximumRadiusChangeFraction) *
                    static_cast<double>(
                        options_.maximumRawUpdateMultiplier);
            const double plausibilityFactor =
                implausible
                    ? 0.0
                    : std::min(
                        1.0,
                        std::min(
                            centerCap /
                                std::max(rawCenterMagnitude,
                                         kEpsilon),
                            static_cast<double>(
                                options_.maximumRadiusChangeFraction) /
                                std::max(rawRadiusMagnitude,
                                         kEpsilon)));

            geometryConfidence = static_cast<float>(
                evidenceFactor *
                fractionFactor *
                faceFactor *
                gradientFactor *
                residualFactor *
                plausibilityFactor);

            if (!implausible &&
                geometryConfidence >= options_.minimumConfidence) {
                if (rawCenterMagnitude > centerCap) {
                    rawCenterShift *=
                        centerCap / rawCenterMagnitude;
                }
                const double alpha =
                    static_cast<double>(
                        options_.geometryEmaAlpha);
                state_.center.x += static_cast<float>(
                    alpha * rawCenterShift[0]);
                state_.center.y += static_cast<float>(
                    alpha * rawCenterShift[1]);
                state_.center.z += static_cast<float>(
                    alpha * rawCenterShift[2]);

                for (int axis = 0; axis < 3; ++axis) {
                    const double boundedLogChange =
                        std::clamp(
                            fit[axis + 3],
                            -static_cast<double>(
                                options_.maximumRadiusChangeFraction),
                            static_cast<double>(
                                options_.maximumRadiusChangeFraction));
                    const double updatedRadius =
                        static_cast<double>(state_.radii[axis]) *
                        std::exp(alpha * boundedLogChange);
                    state_.radii[axis] = static_cast<float>(
                        std::max(
                            static_cast<double>(
                                options_.minimumRadius),
                            updatedRadius));
                }
                geometryAccepted = true;
            }
        }
    } else {
        // The seed describes the manually confirmed source frame.
        geometryConfidence = 1.0f;
    }

    float coldEstimate =
        std::numeric_limits<float>::quiet_NaN();
    float hotEstimate =
        std::numeric_limits<float>::quiet_NaN();
    std::size_t coldSamples = 0;
    std::size_t hotSamples = 0;
    estimateRegionIntensities(
        floatFrame,
        state_,
        options_,
        cells,
        coldEstimate,
        hotEstimate,
        coldSamples,
        hotSamples);

    const bool holdSeedIntensities =
        firstUpdate &&
        options_.holdSeedIntensitiesOnFirstUpdate;
    bool intensityAccepted = false;
    bool intensitySnapped = false;
    if (!holdSeedIntensities) {
        bool acceptIntensityEstimates =
            finiteFloat(coldEstimate) && finiteFloat(hotEstimate);
        const float seedContrast =
            seed_.hotBackground - seed_.coldBackground;
        if (acceptIntensityEstimates &&
            std::abs(seedContrast) >=
                options_.minimumRegionContrast) {
            if (seedContrast > 0.0f) {
                acceptIntensityEstimates =
                    hotEstimate - coldEstimate >=
                    options_.minimumRegionContrast;
            } else {
                acceptIntensityEstimates =
                    coldEstimate - hotEstimate >=
                    options_.minimumRegionContrast;
            }
        }
        if (acceptIntensityEstimates) {
            intensityAccepted = true;
            intensitySnapped =
                firstUpdate &&
                options_.snapIntensityEstimatesOnFirstUpdate;
            if (intensitySnapped) {
                state_.coldBackground =
                    options_.clampBackgroundToUnitRange
                        ? clampUnit(coldEstimate)
                        : coldEstimate;
                state_.hotBackground =
                    options_.clampBackgroundToUnitRange
                        ? clampUnit(hotEstimate)
                        : hotEstimate;
            } else {
                state_.coldBackground = updateScalarConservatively(
                    state_.coldBackground,
                    seed_.coldBackground,
                    coldEstimate,
                    options_);
                state_.hotBackground = updateScalarConservatively(
                    state_.hotBackground,
                    seed_.hotBackground,
                    hotEstimate,
                    options_);
            }
        }
    }

    state_.frameIndex = frameIndex;
    state_.confidence = geometryConfidence;
    state_.frozen = !geometryAccepted;
    state_.shellSamples = shellSamples;
    state_.evidenceSamples = evidenceSamples;
    state_.hotSamples = hotSamples;
    state_.coldSamples = coldSamples;
    state_.coveredFaces = countSetBits(coveredFaces);
    state_.coldCandidate =
        finiteFloat(coldEstimate) ? coldEstimate : state_.coldBackground;
    state_.hotCandidate =
        finiteFloat(hotEstimate) ? hotEstimate : state_.hotBackground;
    state_.intensityAccepted = intensityAccepted;
    state_.intensitySnapped = intensitySnapped;
    ++updatesSeen_;
    return geometryAccepted;
}

BackgroundRegionTracker::RenderedStacks
BackgroundRegionTracker::render() const
{
    if (lastZSlices_ <= 0 || lastRows_ <= 0 || lastCols_ <= 0) {
        throw std::logic_error(
            "BackgroundRegionTracker render requires a prior update or explicit dimensions");
    }
    return render(lastZSlices_, lastRows_, lastCols_);
}

BackgroundRegionTracker::RenderedStacks
BackgroundRegionTracker::render(
    int zSlices,
    int rows,
    int cols) const
{
    RenderedStacks result;
    render(zSlices,
           rows,
           cols,
           result.background,
           result.membership);
    return result;
}

void BackgroundRegionTracker::render(
    int zSlices,
    int rows,
    int cols,
    std::vector<cv::Mat> &background,
    std::vector<cv::Mat> &membership) const
{
    if (!configured_) {
        throw std::logic_error(
            "BackgroundRegionTracker must be configured before render");
    }
    if (zSlices <= 0 || rows <= 0 || cols <= 0) {
        throw std::invalid_argument(
            "BackgroundRegionTracker render dimensions must be positive");
    }

    background.assign(
        static_cast<std::size_t>(zSlices),
        cv::Mat());
    membership.assign(
        static_cast<std::size_t>(zSlices),
        cv::Mat());
    const Matrix3 rotation = makeRotationMatrix(state_.rotation);
    for (int z = 0; z < zSlices; ++z) {
        cv::Mat backgroundSlice(rows, cols, CV_32F);
        cv::Mat membershipSlice(rows, cols, CV_32F);
        for (int y = 0; y < rows; ++y) {
            float *backgroundRow =
                backgroundSlice.ptr<float>(y);
            float *membershipRow =
                membershipSlice.ptr<float>(y);
            for (int x = 0; x < cols; ++x) {
                const cv::Point3f point(
                    static_cast<float>(x),
                    static_cast<float>(y),
                    static_cast<float>(z));
                const float weight =
                    membershipAtState(state_, rotation, point);
                membershipRow[x] = weight;
                backgroundRow[x] =
                    state_.coldBackground +
                    (state_.hotBackground -
                     state_.coldBackground) * weight;
            }
        }
        background[static_cast<std::size_t>(z)] =
            std::move(backgroundSlice);
        membership[static_cast<std::size_t>(z)] =
            std::move(membershipSlice);
    }
}

float BackgroundRegionTracker::membershipAt(
    const cv::Point3f &worldPoint) const
{
    if (!configured_) {
        throw std::logic_error(
            "BackgroundRegionTracker must be configured before sampling");
    }
    if (!finitePoint(worldPoint)) {
        throw std::invalid_argument(
            "BackgroundRegionTracker sample point must be finite");
    }
    const Matrix3 rotation =
        makeRotationMatrix(state_.rotation);
    return membershipAtState(state_, rotation, worldPoint);
}

float BackgroundRegionTracker::backgroundAt(
    const cv::Point3f &worldPoint) const
{
    const float weight = membershipAt(worldPoint);
    return state_.coldBackground +
           (state_.hotBackground -
            state_.coldBackground) * weight;
}

float BackgroundRegionTracker::membershipAt(
    int z,
    int y,
    int x) const
{
    if (lastZSlices_ <= 0 || lastRows_ <= 0 || lastCols_ <= 0) {
        throw std::logic_error(
            "BackgroundRegionTracker indexed sampling requires a prior update");
    }
    if (z < 0 || z >= lastZSlices_ ||
        y < 0 || y >= lastRows_ ||
        x < 0 || x >= lastCols_) {
        throw std::out_of_range(
            "BackgroundRegionTracker sample index is outside the current frame");
    }
    return membershipAt(cv::Point3f(
        static_cast<float>(x),
        static_cast<float>(y),
        static_cast<float>(z)));
}

float BackgroundRegionTracker::backgroundAt(
    int z,
    int y,
    int x) const
{
    const float weight = membershipAt(z, y, x);
    return state_.coldBackground +
           (state_.hotBackground -
            state_.coldBackground) * weight;
}
