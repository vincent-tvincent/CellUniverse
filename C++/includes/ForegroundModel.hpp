#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <opencv2/core.hpp>

// Rung 1 (self-calibrating data term). A per-frame two-population intensity
// model: voxels inside cells are "foreground" (bright), voxels outside are
// "background" (dim + noise). Both Gaussians are estimated from THIS frame's
// own pixels each frame, so the term self-calibrates to each dataset's
// brightness/pedestal/noise instead of relying on absolute thresholds.
//
// The per-voxel negative log-likelihood replaces the raw-L2 residual: a
// voxel a cell claims should look foreground, one it does not claim should
// look background. Because sigBg is measured from the real background, noise
// is *modelled* (expected), not penalised.
struct ForegroundModel {
    float muFg = 0.f, sigFg = 1.f, muBg = 0.f, sigBg = 1.f;
    float logSigFg = 0.f, logSigBg = 0.f; // precomputed std::log(sigma*)
    bool  valid = false;

    // Negative log-likelihood of intensity under each Gaussian, dropping the
    // shared 0.5*log(2*pi) constant (cancels in every cost delta).
    inline float nllFg(float I) const { float z = (I - muFg) / sigFg; return 0.5f * z * z + logSigFg; }
    inline float nllBg(float I) const { float z = (I - muBg) / sigBg; return 0.5f * z * z + logSigBg; }
    // Foreground log-likelihood ratio: > 0 ⇒ this intensity looks foreground.
    inline float logLR(float I) const { return nllBg(I) - nllFg(I); }

    // Soft foreground probability P(fg|I) = sigmoid(logLR), in [0,1]. This is the
    // BOUNDED per-voxel signal used by the L2-magnitude-compatible data term:
    // a soft-classification cost `claimed ? (1-pFg) : pFg` stays in [0,1]/voxel
    // (like L2), unlike the raw NLL whose (I/sigBg)^2 blow-up froze the optimizer.
    inline float pFg(float I) const {
        const float z = logLR(I);
        // Numerically stable sigmoid.
        if (z >= 0.f) { const float e = std::exp(-z); return 1.0f / (1.0f + e); }
        const float e = std::exp(z);
        return e / (1.0f + e);
    }

    void finalize(float minSigma) {
        sigFg = std::max(sigFg, minSigma);
        sigBg = std::max(sigBg, minSigma);
        logSigFg = std::log(sigFg);
        logSigBg = std::log(sigBg);
        valid = true;
    }
};

// Robust per-frame estimate. `insideAnyCell` is 1 where a voxel is inside any
// current cell (flattened z-major, same total voxel count as `real`), else 0.
ForegroundModel estimateForegroundModel(const std::vector<cv::Mat>& real,
                                        const std::vector<uint8_t>& insideAnyCell,
                                        float minSigma);
