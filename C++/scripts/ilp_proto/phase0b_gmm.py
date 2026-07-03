#!/usr/bin/env python3
"""
Phase 0b (GMM) — validate the Gaussian-mixture / splatting representation + EM fit.

This is the fit-story check for the chosen representation:
  cell = mixture of a few anisotropic Gaussians (each = a rotated Ellipsoid),
  fit by EM, K-selection (1/2/3) gives the nested shape/split/ILP hierarchy.

Two questions, on the 5 GT SEG slices (102 nuclei):
  1) CEILING: does an EM-fit K-Gaussian mixture level-set reach the shape ceiling
     (and does EM beat the hand-rolled Nelder-Mead metaball ~0.90)?
  2) HIERARCHY: does BIC K-selection recover structure -- K=1 for round nuclei,
     K>=2 for concave / dumbbell (bilobed) nuclei? That is the nested-hypothesis claim.

Fit uses the pure SHAPE (uniform samples over the GT mask) so this isolates the
representation's capability from tracker fit quality / z-registration. (Intensity-
weighted EM on the real/ volume is the natural follow-up that also tests the fit gap.)

Usage:
  .venv-eval/bin/python scripts/ilp_proto/phase0b_gmm.py
"""
from __future__ import annotations
import glob, os
import numpy as np
import tifffile
from skimage.measure import regionprops
from sklearn.mixture import GaussianMixture

SEG_DIR = "/Users/jihangli/MCS/3D_Cell_Tracking/Fluo-N3DH-CE/01_GT/SEG"


def crop(gt_full, label, pad=6):
    mask = (gt_full == label)
    ys, xs = np.nonzero(mask)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, gt_full.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, gt_full.shape[1])
    return mask[y0:y1, x0:x1]


def best_threshold_jaccard(logp, gt_flat):
    """Exact best-threshold Jaccard: level sets are nested by threshold, so sweep
    pixels in descending density and take the max Jaccard over all prefixes."""
    order = np.argsort(logp)[::-1]
    gt_sorted = gt_flat[order]
    ngt = gt_flat.sum()
    inter = np.cumsum(gt_sorted)                 # gt-positive pixels included so far
    added = np.arange(1, len(order) + 1)
    union = ngt + added - inter
    jac = inter / np.maximum(union, 1)
    return float(jac.max())


def fit_eval(gt):
    ys, xs = np.nonzero(gt)
    X = np.stack([xs, ys], 1).astype(float)
    H, W = gt.shape
    GY, GX = np.mgrid[0:H, 0:W]
    grid = np.stack([GX.ravel(), GY.ravel()], 1).astype(float)
    gt_flat = gt.ravel()
    out = {}
    bic = {}
    for K in (1, 2, 3):
        if len(X) <= K * 3:
            out[K] = 0.0; bic[K] = np.inf; continue
        gm = GaussianMixture(K, covariance_type="full", reg_covar=1e-3,
                             n_init=2, max_iter=100, random_state=0)
        gm.fit(X)
        logp = gm.score_samples(grid)
        out[K] = best_threshold_jaccard(logp, gt_flat)
        bic[K] = gm.bic(X)
    kstar = min(bic, key=bic.get)
    return out, kstar


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(SEG_DIR, "*.tif"))):
        gt_full = tifffile.imread(f)
        name = os.path.basename(f).replace("man_seg_", "").replace(".tif", "")
        for r in regionprops(gt_full):
            if r.area < 200:
                continue
            gt = crop(gt_full, r.label)
            jac, kstar = fit_eval(gt)
            rows.append(dict(slice=name, label=r.label, area=int(r.area),
                             sol=r.solidity, ecc=r.eccentricity,
                             j1=jac[1], j2=jac[2], j3=jac[3], kstar=kstar))

    n = len(rows)
    def m(k): return float(np.mean([r[k] for r in rows]))
    print(f"\n===== EM-GMM CEILING & K-SELECTION — {n} nuclei =====")
    print(f"  K=1 (1 Gaussian)  mean Jaccard = {m('j1'):.3f}   [ellipse baseline ~0.877]")
    print(f"  K=2               mean Jaccard = {m('j2'):.3f}   ({m('j2')-m('j1'):+.3f} vs K=1)")
    print(f"  K=3               mean Jaccard = {m('j3'):.3f}   ({m('j3')-m('j1'):+.3f} vs K=1)")
    best_of_k = np.mean([max(r['j1'], r['j2'], r['j3']) for r in rows])
    print(f"  best-of-K (oracle) mean Jaccard = {best_of_k:.3f}")

    # K-selection vs shape type
    from collections import Counter
    ksel = Counter(r['kstar'] for r in rows)
    print(f"\n  BIC K-selection distribution: {dict(sorted(ksel.items()))}")
    # round (high solidity, low ecc) should be K=1; concave/dumbbell K>=2
    round_ = [r for r in rows if r['sol'] > 0.94 and r['ecc'] < 0.6]
    hard_  = [r for r in rows if r['sol'] < 0.90]
    def pct_k1(g): return 100 * np.mean([r['kstar'] == 1 for r in g]) if g else 0
    def pct_kge2(g): return 100 * np.mean([r['kstar'] >= 2 for r in g]) if g else 0
    print(f"  round nuclei (sol>0.94,ecc<0.6, n={len(round_)}):  {pct_k1(round_):.0f}% picked K=1")
    print(f"  concave nuclei (sol<0.90, n={len(hard_)}):          {pct_kge2(hard_):.0f}% picked K>=2")

    # the known dumbbell / big-gain nuclei from union-of-2 test
    print("\n  known dumbbell/bilobed nuclei -- did GMM see 2 lobes?")
    targets = [("028_018", 2012), ("162_010", 1604), ("141_014", 1207),
               ("162_010", 1370), ("028_018", 2303)]
    for sl, ar in targets:
        cand = [r for r in rows if r['slice'] == sl and abs(r['area'] - ar) <= 60]
        if cand:
            r = cand[0]
            print(f"    {sl} area={r['area']:>4} sol={r['sol']:.2f}: "
                  f"J(K1)={r['j1']:.3f} J(K2)={r['j2']:.3f} -> BIC picked K={r['kstar']}")


if __name__ == "__main__":
    main()
