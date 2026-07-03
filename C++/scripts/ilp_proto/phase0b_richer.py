#!/usr/bin/env python3
"""
Phase 0b (extended) — add a star-convex radial-Fourier boundary to the ceiling test.

Motivation: 0b tested ellipse / superellipse / union-of-2. It did NOT test the prime
candidate for "smooth irregular shape like the SEG outline": a star-convex boundary
  r(theta) = a0 + sum_{k=1..K} [a_k cos(k*theta) + b_k sin(k*theta)]
whose 3D analog is a spherical-harmonic (SPHARM) surface. This measures its Jaccard
ceiling at K = 3/5/8 harmonics and renders overlays for representative nuclei so we
can pick a representation against evidence.

Star-convex limit: if a nucleus boundary is multi-valued in theta from the centroid
(deep concavity / true dumbbell), the radial model caps out -> that gap is exactly
where union-of-N / metaball is needed. The figure makes that visible.

Usage:
  .venv-eval/bin/python scripts/ilp_proto/phase0b_richer.py            # summary + figure
"""
from __future__ import annotations
import glob, os
import numpy as np
import tifffile
from skimage.measure import find_contours, regionprops
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phase0b_shape_ceiling import (raster_ellipse, raster_superellipse, raster_union2,
                                   jaccard, moment_ellipse, optimize_shape, split_init)

SEG_DIR = "/Users/jihangli/MCS/3D_Cell_Tracking/Fluo-N3DH-CE/01_GT/SEG"
OUT_PNG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase0b_shapes.png")


def fit_radial(gt, K):
    ys, xs = np.nonzero(gt); cy, cx = ys.mean(), xs.mean()
    conts = find_contours(gt.astype(float), 0.5)
    if not conts:
        return None
    pts = np.vstack(conts)                       # (row=y, col=x)
    th = np.arctan2(pts[:, 0] - cy, pts[:, 1] - cx)
    r = np.hypot(pts[:, 0] - cy, pts[:, 1] - cx)
    cols = [np.ones_like(th)]
    for k in range(1, K + 1):
        cols += [np.cos(k * th), np.sin(k * th)]
    A = np.stack(cols, 1)
    coef, *_ = np.linalg.lstsq(A, r, rcond=None)
    return (cx, cy, coef, K)


def raster_radial(shape, params):
    cx, cy, coef, K = params
    YY, XX = np.mgrid[0:shape[0], 0:shape[1]]
    th = np.arctan2(YY - cy, XX - cx)
    rb = np.full(th.shape, coef[0]); idx = 1
    for k in range(1, K + 1):
        rb += coef[idx] * np.cos(k * th) + coef[idx + 1] * np.sin(k * th); idx += 2
    rr = np.hypot(YY - cy, XX - cx)
    return rr <= np.maximum(rb, 0.1)


def crop(gt_full, label, pad=8):
    mask = (gt_full == label)
    ys, xs = np.nonzero(mask)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, gt_full.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, gt_full.shape[1])
    return mask[y0:y1, x0:x1]


def eval_all():
    files = sorted(glob.glob(os.path.join(SEG_DIR, "*.tif")))
    rows = []
    for f in files:
        gt_full = tifffile.imread(f)
        name = os.path.basename(f).replace("man_seg_", "").replace(".tif", "")
        for r in regionprops(gt_full):
            if r.area < 200:
                continue
            gt = crop(gt_full, r.label)
            me = moment_ellipse(gt)
            j_ell = jaccard(raster_ellipse(gt.shape, me), gt)   # moment ellipse (fast baseline)
            rads = {}
            for K in (3, 5, 8):
                p = fit_radial(gt, K)
                rads[K] = jaccard(raster_radial(gt.shape, p), gt) if p else 0.0
            rows.append(dict(slice=name, label=r.label, area=int(r.area),
                             solidity=r.solidity, ecc=r.eccentricity,
                             ell=j_ell, r3=rads[3], r5=rads[5], r8=rads[8]))
    return rows


def summary(rows):
    def m(k): return float(np.mean([r[k] for r in rows]))
    print(f"\n===== RADIAL-FOURIER (SPHARM proxy) CEILING — {len(rows)} nuclei =====")
    print(f"  ellipse (moment)     mean={m('ell'):.3f}   [0b optimized ellipse = 0.887]")
    print(f"  radial K=3  (7 DOF)  mean={m('r3'):.3f}   ({m('r3')-m('ell'):+.3f} vs ellipse)")
    print(f"  radial K=5  (11 DOF) mean={m('r5'):.3f}   ({m('r5')-m('ell'):+.3f} vs ellipse)")
    print(f"  radial K=8  (17 DOF) mean={m('r8'):.3f}   ({m('r8')-m('ell'):+.3f} vs ellipse)")
    # where does radial K=5 still fail (star-convexity limit -> needs union/metaball)?
    worst = sorted(rows, key=lambda r: r["r5"])[:8]
    print("\n  nuclei where radial K=5 still struggles (deep concavity / dumbbell):")
    for r in worst:
        print(f"    {r['slice']} area={r['area']:>4} sol={r['solidity']:.2f} "
              f"ecc={r['ecc']:.2f}  ell={r['ell']:.3f} r5={r['r5']:.3f} r8={r['r8']:.3f}")


def pick_representatives(rows):
    by = {}
    by["round"] = max(rows, key=lambda r: r["solidity"] - r["ecc"])
    by["elongated"] = max(rows, key=lambda r: r["ecc"])
    by["concave"] = min(rows, key=lambda r: r["solidity"])
    by["radial-hard"] = min(rows, key=lambda r: r["r5"])
    return by


def render_figure(reps):
    files = {os.path.basename(f).replace("man_seg_", "").replace(".tif", ""): f
             for f in glob.glob(os.path.join(SEG_DIR, "*.tif"))}
    models = ["ellipse", "superellipse", "radial K=5", "union-of-2"]
    fig, axes = plt.subplots(len(reps), len(models), figsize=(3 * len(models), 3 * len(reps)))
    for ri, (tag, r) in enumerate(reps.items()):
        gt_full = tifffile.imread(files[r["slice"]])
        gt = crop(gt_full, r["label"])
        me = moment_ellipse(gt)
        j_ell, p_ell = optimize_shape(gt, raster_ellipse, me)
        j_sup, p_sup = optimize_shape(gt, raster_superellipse, np.append(p_ell, 2.0))
        p_r5 = fit_radial(gt, 5); j_r5 = jaccard(raster_radial(gt.shape, p_r5), gt)
        j_uni, p_uni = optimize_shape(gt, raster_union2, split_init(gt, p_ell), restarts=3, maxiter=600)
        fits = [(raster_ellipse(gt.shape, p_ell), j_ell),
                (raster_superellipse(gt.shape, p_sup), j_sup),
                (raster_radial(gt.shape, p_r5), j_r5),
                (raster_union2(gt.shape, p_uni), max(j_uni, j_ell))]
        for ci, (mdl, (fit, jv)) in enumerate(zip(models, fits)):
            ax = axes[ri, ci]
            ax.imshow(gt, cmap="Greys", alpha=0.55)
            for c in find_contours(fit.astype(float), 0.5):
                ax.plot(c[:, 1], c[:, 0], "r-", lw=1.8)
            ax.set_title(f"{mdl}\nJ={jv:.3f}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            if ci == 0:
                ax.set_ylabel(f"{tag}\n{r['slice']} a={r['area']}\nsol={r['solidity']:.2f}",
                              fontsize=8, rotation=0, ha="right", va="center")
    fig.suptitle("Shape primitives vs GT SEG nuclei (gray=GT, red=best fit)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_PNG, dpi=110)
    print(f"\n  figure -> {OUT_PNG}")


if __name__ == "__main__":
    rows = eval_all()
    summary(rows)
    render_figure(pick_representatives(rows))
