#!/usr/bin/env python3
"""
Phase 0b (metaball) — ceiling for union-of-N ellipsoids and a smooth metaball.

0b only tested hard-union N=2 (0.909). The chosen representation is union-of-N /
metaball, so we need the ceiling as a function of N (how many blobs to hit the
"both equally" bar ~0.93+), and whether a SMOOTH metaball beats a hard union.

Models (2D, per GT nucleus, optimized for max Jaccard):
  union-N   : OR of N ellipses            (5N params)          -- hard seams
  metaball  : sum_i Gaussian(c_i, sigma_i) >= T                (4N+1 params) -- smooth

Usage:
  .venv-eval/bin/python scripts/ilp_proto/phase0b_metaball.py [--sample 40]
"""
from __future__ import annotations
import argparse, glob, os
import numpy as np
import tifffile
from scipy.optimize import minimize
from scipy.cluster.vq import kmeans2
from skimage.measure import regionprops

from phase0b_shape_ceiling import raster_ellipse, jaccard, moment_ellipse

SEG_DIR = "/Users/jihangli/MCS/3D_Cell_Tracking/Fluo-N3DH-CE/01_GT/SEG"


def crop(gt_full, label, pad=8):
    mask = (gt_full == label)
    ys, xs = np.nonzero(mask)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, gt_full.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, gt_full.shape[1])
    return mask[y0:y1, x0:x1]


# ---- union of N ellipses ----
def raster_unionN(shape, p, N):
    m = np.zeros(shape, bool)
    for i in range(N):
        m |= raster_ellipse(shape, p[5 * i:5 * i + 5])
    return m


def init_unionN(gt, N):
    ys, xs = np.nonzero(gt)
    pts = np.stack([xs, ys], 1).astype(float)
    if len(pts) < N:
        base = moment_ellipse(gt); return np.tile(base, N)
    cen, lab = kmeans2(pts, N, minit="++", seed=0)
    params = []
    for i in range(N):
        sel = lab == i
        if sel.sum() < 5:
            params.append(moment_ellipse(gt)); continue
        sub = np.zeros_like(gt); sub[ys[sel], xs[sel]] = True
        params.append(moment_ellipse(sub))
    return np.concatenate(params)


# ---- smooth metaball: sum_i A_i exp(-((x-cx)^2+(y-cy)^2)/(2 s_i^2)) >= T ----
def raster_metaball(shape, p, N):
    YY, XX = np.mgrid[0:shape[0], 0:shape[1]]
    f = np.zeros(shape, float)
    for i in range(N):
        cx, cy, s = p[3 * i], p[3 * i + 1], max(p[3 * i + 2], 0.5)
        f += np.exp(-(((XX - cx) ** 2 + (YY - cy) ** 2) / (2 * s * s)))
    T = p[3 * N]
    return f >= max(T, 1e-3)


def init_metaball(gt, N):
    ys, xs = np.nonzero(gt)
    pts = np.stack([xs, ys], 1).astype(float)
    cen, lab = kmeans2(pts, N, minit="++", seed=0)
    p = []
    for i in range(N):
        sel = lab == i
        s = max(np.sqrt(sel.sum() / np.pi), 2.0) if sel.sum() > 3 else 4.0
        p += [cen[i, 0], cen[i, 1], s]
    p += [0.5]          # threshold
    return np.array(p)


def opt(gt, raster, x0, N, restarts=3, maxiter=None):
    if maxiter is None:
        maxiter = 250 * N
    best_v, best_p = -1.0, x0
    for k in range(restarts):
        start = x0 if k == 0 else x0 * (1.0 + 0.15 * ((-1) ** np.arange(len(x0))))
        res = minimize(lambda p: -jaccard(raster(gt.shape, p, N), gt), start,
                       method="Nelder-Mead",
                       options={"maxiter": maxiter, "xatol": 1e-2, "fatol": 1e-4})
        if -res.fun > best_v:
            best_v, best_p = -res.fun, res.x
    return best_v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0, help="limit to N nuclei (0=all)")
    args = ap.parse_args()

    nuclei = []
    for f in sorted(glob.glob(os.path.join(SEG_DIR, "*.tif"))):
        gt_full = tifffile.imread(f)
        name = os.path.basename(f).replace("man_seg_", "").replace(".tif", "")
        for r in regionprops(gt_full):
            if r.area >= 200:
                nuclei.append((name, r.label, r.solidity, r.eccentricity,
                               crop(gt_full, r.label)))
    if args.sample:
        # stratified-ish: hardest (low solidity) first + spread
        nuclei.sort(key=lambda t: t[2])
        step = max(1, len(nuclei) // args.sample)
        nuclei = nuclei[:args.sample // 2] + nuclei[::step]
        seen = set(); uniq = []
        for n in nuclei:
            k = (n[0], n[1])
            if k not in seen:
                seen.add(k); uniq.append(n)
        nuclei = uniq[:args.sample]

    res = {k: [] for k in ["ell", "u2", "u3", "u4", "mb3", "mb4"]}
    for name, lab, sol, ecc, gt in nuclei:
        me = moment_ellipse(gt)
        res["ell"].append(jaccard(raster_ellipse(gt.shape, me), gt))
        res["u2"].append(opt(gt, raster_unionN, init_unionN(gt, 2), 2))
        res["u3"].append(opt(gt, raster_unionN, init_unionN(gt, 3), 3))
        res["u4"].append(opt(gt, raster_unionN, init_unionN(gt, 4), 4))
        res["mb3"].append(opt(gt, raster_metaball, init_metaball(gt, 3), 3))
        res["mb4"].append(opt(gt, raster_metaball, init_metaball(gt, 4), 4))

    # unions are supersets of ellipse; guard numeric dropouts monotonic
    for i in range(len(res["ell"])):
        res["u2"][i] = max(res["u2"][i], res["ell"][i])
        res["u3"][i] = max(res["u3"][i], res["u2"][i])
        res["u4"][i] = max(res["u4"][i], res["u3"][i])

    n = len(nuclei)
    def m(k): return float(np.mean(res[k]))
    print(f"\n===== UNION-of-N / METABALL CEILING — {n} nuclei =====")
    print(f"  ellipse            mean={m('ell'):.3f}")
    print(f"  union-2  (10 DOF)  mean={m('u2'):.3f}  ({m('u2')-m('ell'):+.3f})")
    print(f"  union-3  (15 DOF)  mean={m('u3'):.3f}  ({m('u3')-m('ell'):+.3f})")
    print(f"  union-4  (20 DOF)  mean={m('u4'):.3f}  ({m('u4')-m('ell'):+.3f})")
    print(f"  metaball-3 (10 DOF) mean={m('mb3'):.3f}  ({m('mb3')-m('ell'):+.3f})  [smooth]")
    print(f"  metaball-4 (13 DOF) mean={m('mb4'):.3f}  ({m('mb4')-m('ell'):+.3f})  [smooth]")
    print(f"\n  reference: radial K=5=0.930, K=8=0.961 (SPHARM proxy)")


if __name__ == "__main__":
    main()
