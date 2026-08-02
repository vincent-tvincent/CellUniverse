#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
#include <random>
#include <opencv2/core.hpp>
#include "../includes/ForegroundModel.hpp"

// Build a 1-slice volume: background voxels ~ N(30,5), a 16x16 foreground
// block ~ N(200,20). `inside` marks the foreground block.
static void makeSynthFrame(std::vector<cv::Mat>& real, std::vector<uint8_t>& inside) {
    const int H = 64, W = 64;
    cv::Mat slice(H, W, CV_32F);
    std::mt19937 rng(1234);
    std::normal_distribution<float> bg(30.f, 5.f), fg(200.f, 20.f);
    std::vector<uint8_t> in(static_cast<size_t>(H) * W, 0);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            bool isFg = (x >= 24 && x < 40 && y >= 24 && y < 40);
            slice.at<float>(y, x) = isFg ? fg(rng) : bg(rng);
            in[static_cast<size_t>(y) * W + x] = isFg ? 1 : 0;
        }
    }
    real = { slice };
    inside = in;
}

int main() {
    std::vector<cv::Mat> real;
    std::vector<uint8_t> inside;
    makeSynthFrame(real, inside);
    ForegroundModel m = estimateForegroundModel(real, inside, /*minSigma=*/1.0f);

    assert(m.valid);
    // Background center recovered (median of the dim population).
    assert(std::abs(m.muBg - 30.f) < 4.f);
    // Foreground center is the mean of the BRIGHT HALF of inside voxels, so for
    // N(200,20) it sits a bit above 200 (~mu + 0.8*sigma ≈ 216), not at 200.
    assert(m.muFg > 205.f && m.muFg < 235.f);
    // Populations are clearly separated and neither sigma collapsed.
    assert(m.muFg > m.muBg + 3.f * m.sigBg);
    assert(m.sigFg > 0.f && m.sigBg > 0.f);
    assert(m.sigBg >= 0.25f * m.sigFg - 1e-3f); // sigBg floor keeps the cost smooth

    // logLR sign: bright looks foreground (>0), dim looks background (<0).
    assert(m.logLR(216.f) > 0.f);
    assert(m.logLR(30.f) < 0.f);

    // nll is minimized at the mean and grows away from it.
    assert(m.nllFg(m.muFg) < m.nllFg(m.muFg + 3.f * m.sigFg));
    assert(m.nllBg(m.muBg) < m.nllBg(m.muBg + 3.f * m.sigBg));

    std::printf("test_foreground_model: PASS\n");
    return 0;
}
