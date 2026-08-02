#pragma once
#include <opencv2/core.hpp>

// Weighted 3D sample consumed by the PCA shape-fit math (weighted centroid +
// covariance + radii). Defined here so it can be shared between Frame.cpp (which
// also keeps a token-identical local definition — see the ODR note there) and the
// out-of-line gatherForegroundComponent in ForegroundComponent.cpp. Keeping the
// half-max gather in its OWN translation unit is deliberate: adding its body to
// Frame.cpp shifts the LTO FP scheduling of the SHARED PCA math and breaks the
// bit-exact legacy flag-OFF fit.
struct BrightPixel
{
    cv::Point3f pos;   // world coordinates (x, y, z in interpolated-z space)
    float weight;      // pixel intensity above background
};
