#!/usr/bin/env python3
"""
Phase 0b — Shape-ceiling measurement (Workstream S go/no-go).

Question (from docs/plans/2026-07-02-shape-and-ilp-buildplan.md):
  Does a richer shape primitive raise the best-achievable per-slice SEG (Jaccard)
  ceiling MEANINGFULLY above the plain ellipsoid? If not, don't build the shape change.

Method:
  The CTC Fluo-N3DH-CE 01 GT SEG is 5 sparse 2D slices (man_seg_FFF_ZZZ.tif). For each
  GT nucleus outline we compute the BEST achievable 2D Jaccard for three primitives,
  by optimizing each primitive's parameters to maximize overlap with the GT mask:

    ellipse       (x'/a)^2 + (y'/b)^2 <= 1                         -- current model
    superellipse  |x'/a|^r + |y'/b|^r <= 1                         -- S1 (adds exponent)
    union-of-2    ellipse_1 OR ellipse_2                           -- S2 (dumbbell/dividing)

  (x',y') = rotate((x,y)-center, -theta). Jaccard = |shape & gt| / |shape | gt|.
  This is a CEILING: the fitter is free to place the ideal shape, so it upper-bounds
  what the tracker could ever score with that primitive on that nucleus.

Gate:
  * superellipse lifts the ceiling over ellipse by a worthwhile margin  -> build S1
  * union-of-2 captures the dumbbell/dividing nuclei ellipse can't      -> build S2

Usage:
  .venv-eval/bin/python scripts/ilp_proto/phase0b_shape_ceiling.py
  .venv-eval/bin/python scripts/ilp_proto/phase0b_shape_ceiling.py --slice 21   # one slice
"""
from __future__ import annotations
import argparse, glob, os
import numpy as np
import tifffile
from scipy.optimize import minimize

SEG_DIR = "/Users/jihangli/MCS/3D_Cell_Tracking/Fluo-N3DH-CE/01_GT/SEG"


# ----------------------------------------------------------------------------
# Rasterizers (operate on a local Y,X grid relative to a crop origin)
# ----------------------------------------------------------------------------
def _rot_coords(YY, XX, cx, cy, theta):
    ct, st = np.cos(-theta), np.sin(-theta)
    dx = XX - cx; dy = YY - cy
    xp = ct * dx - st * dy
    yp = st * dx + ct * dy
    return xp, yp


def raster_ellipse(shape, p):
    cx, cy, a, b, th = p
    a = max(a, 0.5); b = max(b, 0.5)
    YY, XX = np.mgrid[0:shape[0], 0:shape[1]]
    xp, yp = _rot_coords(YY, XX, cx, cy, th)
    return (xp / a) ** 2 + (yp / b) ** 2 <= 1.0


def raster_superellipse(shape, p):
    cx, cy, a, b, th, r = p
    a = max(a, 0.5); b = max(b, 0.5); r = min(max(r, 0.3), 6.0)
    YY, XX = np.mgrid[0:shape[0], 0:shape[1]]
    xp, yp = _rot_coords(YY, XX, cx, cy, th)
    return np.abs(xp / a) ** r + np.abs(yp / b) ** r <= 1.0


def raster_union2(shape, p):
    e1 = raster_ellipse(shape, p[0:5])
    e2 = raster_ellipse(shape, p[5:10])
    return e1 | e2


def jaccard(mask, gt):
    inter = np.logical_and(mask, gt).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(mask, gt).sum()
    return inter / union


# ----------------------------------------------------------------------------
# Moment-based initialization
# ----------------------------------------------------------------------------
def moment_ellipse(gt):
    ys, xs = np.nonzero(gt)
    cy, cx = ys.mean(), xs.mean()
    cov = np.cov(np.stack([xs - cx, ys - cy]))
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]; evecs = evecs[:, order]
    a = 2.0 * np.sqrt(max(evals[0], 1e-6))
    b = 2.0 * np.sqrt(max(evals[1], 1e-6))
    th = np.arctan2(evecs[1, 0], evecs[0, 0])
    return np.array([cx, cy, a, b, th])


def optimize_shape(gt, raster, x0, restarts=4, maxiter=400):
    best_val, best_p = -1.0, x0
    rng_scales = [0.0, 0.1, 0.2, 0.35]
    for k in range(restarts):
        s = rng_scales[k % len(rng_scales)]
        # deterministic perturbation (no RNG needed): scale each param by (1+/-s)
        pert = x0 * (1.0 + s * (np.array([(-1) ** i for i in range(len(x0))])))
        start = pert if k > 0 else x0

        def obj(p):
            return -jaccard(raster(gt.shape, p), gt)

        res = minimize(obj, start, method="Nelder-Mead",
                       options={"maxiter": maxiter, "xatol": 1e-2, "fatol": 1e-4})
        v = -res.fun
        if v > best_val:
            best_val, best_p = v, res.x
    return best_val, best_p


def split_init(gt, me):
    """Init two ellipses by cutting the nucleus along its major axis at the centroid."""
    cx, cy, a, b, th = me
    ys, xs = np.nonzero(gt)
    # project onto major axis direction
    ux, uy = np.cos(th), np.sin(th)
    proj = (xs - cx) * ux + (ys - cy) * uy
    side = proj >= 0
    parts = []
    for sel in (side, ~side):
        if sel.sum() < 5:
            parts.append(me.copy()); continue
        sub = np.zeros_like(gt); sub[ys[sel], xs[sel]] = True
        parts.append(moment_ellipse(sub))
    return np.concatenate(parts)


# ----------------------------------------------------------------------------
# Per-nucleus evaluation
# ----------------------------------------------------------------------------
def eval_nucleus(gt_full, label):
    mask = (gt_full == label)
    ys, xs = np.nonzero(mask)
    pad = 8
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, gt_full.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, gt_full.shape[1])
    gt = mask[y0:y1, x0:x1]
    me = moment_ellipse(gt)

    j_ell, p_ell = optimize_shape(gt, raster_ellipse, me)
    se0 = np.append(p_ell, 2.0)
    j_sup, p_sup = optimize_shape(gt, raster_superellipse, se0)
    u0 = split_init(gt, p_ell)
    j_uni, p_uni = optimize_shape(gt, raster_union2, u0, restarts=3, maxiter=600)
    # union should never lose to ellipse (superset); guard numeric dropouts
    j_uni = max(j_uni, j_ell)
    exponent = p_sup[5]
    return {"area": int(mask.sum()), "ell": j_ell, "sup": j_sup, "uni": j_uni,
            "exp": exponent}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", type=int, default=None, help="frame number to restrict to")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(SEG_DIR, "*.tif")))
    if args.slice is not None:
        files = [f for f in files if f"_{args.slice:03d}_" in os.path.basename(f)]

    all_rows = []
    print(f"{'slice':>18} {'nuc':>4} {'area':>5} {'ellipse':>8} {'superell':>9} "
          f"{'union2':>7} {'exp':>5}")
    for f in files:
        gt_full = tifffile.imread(f)
        labs = [l for l in np.unique(gt_full) if l != 0]
        name = os.path.basename(f).replace("man_seg_", "").replace(".tif", "")
        for l in labs:
            r = eval_nucleus(gt_full, l)
            r["slice"] = name
            all_rows.append(r)
            print(f"{name:>18} {l:>4} {r['area']:>5} {r['ell']:>8.3f} "
                  f"{r['sup']:>9.3f} {r['uni']:>7.3f} {r['exp']:>5.2f}")

    A = all_rows
    def mean(k): return float(np.mean([r[k] for r in A]))
    def med(k):  return float(np.median([r[k] for r in A]))
    print("\n================ CEILING SUMMARY (per-nucleus best Jaccard) ================")
    print(f"  nuclei: {len(A)}")
    print(f"  ellipse      mean={mean('ell'):.3f}  median={med('ell'):.3f}   (current model)")
    print(f"  superellipse mean={mean('sup'):.3f}  median={med('sup'):.3f}   "
          f"(+{mean('sup')-mean('ell'):+.3f} vs ellipse)")
    print(f"  union-of-2   mean={mean('uni'):.3f}  median={med('uni'):.3f}   "
          f"(+{mean('uni')-mean('ell'):+.3f} vs ellipse)")
    # where does union win big? (dumbbell / dividing nuclei)
    gains = sorted(A, key=lambda r: r["uni"] - r["ell"], reverse=True)
    print("\n  biggest union-of-2 gains (dividing/dumbbell candidates):")
    for r in gains[:8]:
        print(f"    {r['slice']} area={r['area']:>4}  ell={r['ell']:.3f} -> "
              f"uni={r['uni']:.3f}  (+{r['uni']-r['ell']:.3f})")
    fitted_exp = np.median([r["exp"] for r in A])
    print(f"\n  median fitted superellipse exponent: {fitted_exp:.2f} "
          f"(2.0 = plain ellipse; >2 boxier, <2 pointier)")


if __name__ == "__main__":
    main()
