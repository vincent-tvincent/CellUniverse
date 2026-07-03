# Phase 0b — Shape-ceiling measurement: RESULTS

**Date:** 2026-07-02
**Script:** `phase0b_shape_ceiling.py`
**Data:** the 5 CTC Fluo-N3DH-CE 01 GT SEG slices (`man_seg_FFF_ZZZ.tif`), 102 nuclei total.
**Gate (build plan):** does a richer shape primitive raise the best-achievable per-slice SEG (Jaccard) ceiling *meaningfully* over the ellipse? If not, don't build the shape change.

## Method
For each GT nucleus outline, optimize each primitive's parameters (Nelder-Mead, moment init, multi-restart) to **maximize** 2D Jaccard against the GT mask. This is a *ceiling*: the fitter places the ideal shape, so it upper-bounds what the tracker could ever score with that primitive. (Nelder-Mead is local → reported numbers are conservative **lower bounds** on the true ceiling.)

Primitives: ellipse `(x'/a)²+(y'/b)²≤1` · superellipse `|x'/a|ʳ+|y'/b|ʳ≤1` · union-of-2-ellipses.

## Results (102 nuclei, per-nucleus best Jaccard)

| primitive | mean | median | Δ vs ellipse | note |
|---|---|---|---|---|
| **ellipse** (current) | 0.887 | 0.893 | — | already high |
| **superellipse** (S1) | 0.892 | 0.897 | **+0.005** | median fitted exponent **2.00** → best superellipse ≈ an ellipse |
| **union-of-2** (S2) | 0.909 | 0.912 | **+0.022** | gain concentrated on dividing nuclei |

**Union-of-2's gain is localized to dumbbell / dividing nuclei** (the rest are already near-elliptical):

| slice | area | ellipse → union2 | gain |
|---|---|---|---|
| 028_018 | 2012 | 0.799 → 0.944 | **+0.145** |
| 162_010 | 1604 | 0.816 → 0.916 | +0.100 |
| 141_014 | 1207 | 0.787 → 0.873 | +0.086 |
| 162_010 | 1370 | 0.826 → 0.909 | +0.084 |
| 028_018 | 2303 | 0.836 → 0.902 | +0.066 |

~8 of 102 nuclei (dividing) account for essentially all of the union-of-2 gain.

## Verdict per gate
- **S1 superquadric — GATE FAILED.** Ceiling gain +0.005; median fitted exponent is 2.00 (nuclei are genuinely elliptical, not boxy/pointy). **Do not build S1** — it buys nothing on this dataset.
- **S2 union-of-2 / metaball — targeted PASS only.** +0.022 overall, but +0.08–0.145 on the ~8% dividing nuclei ellipse structurally cannot represent (concave pinch). Worth building **for the dividing-nucleus case**, not as a broad SEG win.

## The bigger finding (reframes the whole shape plan)
**The ellipse ceiling is already ~0.89, but the reported SEG is 0.358.** The ~0.53 gap between what an ellipse *could* score (0.89) and what the tracker *does* score (0.358) is **not a shape-primitive problem** — a richer primitive can close at most +0.005 (S1) to +0.022 (S2) of it. The dominant SEG loss must be elsewhere:
- fit/placement not reaching the ceiling (radii/rotation/center off),
- 3D↔slice z-extent mismatch,
- missed / spurious cells and the SEG >0.5-overlap matching cutoff (unmatched GT scores 0).

**Recommended next diagnostic (before ANY shape build):** measure the tracker's *actual* per-nucleus Jaccard on these 5 slices (render committed cells, extract the GT SEG z-plane via the ~6.5× z-scale, match to GT labels). That localizes the 0.53 gap to fit vs placement vs missing — which is almost certainly where the real SEG points are, not the primitive.

## Extended ceilings — richer primitives (2026-07-02, `phase0b_richer.py` / `phase0b_metaball.py`)

Added after the maintainer noted the nuclei are visibly non-elliptical (confirmed: **28% have solidity < 0.92 = genuine concavity**; only 38% are ellipse-friendly) and that the model must also generalize to non-elliptical cells (HeLa). Goal reset to **"both equally"** (lift N3DH-CE SEG *and* generalize).

**Radial-Fourier / SPHARM proxy** (`r(θ)=a0+Σ aₖcos+bₖsin`, full 102 nuclei):

| primitive | DOF | mean Jaccard |
|---|---|---|
| ellipse | 5 | 0.877 |
| superquadric | 6 | 0.892 |
| union-of-2 (hard) | 10 | 0.909 |
| radial K=5 | 11 | **0.930** |
| radial K=8 | 17 | **0.961** |

**Union-of-N / metaball** (40-nucleus subset weighted toward hard/concave; ellipse baseline 0.852 here):

| primitive | DOF | mean Jaccard (hard subset) |
|---|---|---|
| ellipse | 5 | 0.852 |
| metaball-3 (smooth Gaussian) | 10 | 0.887 |
| union-2 | 10 | 0.893 |
| metaball-4 | 13 | 0.901 |
| union-3 | 15 | 0.909 |
| union-4 | 20 | 0.919 |

Reads:
- **SPHARM/radial is the most DOF-efficient for smooth-irregular shape** (K=8 → 0.961 at 17 DOF). Its limit is true dumbbell/branching (multi-valued from center).
- **Union-of-N needs high N (≈4–5, 20–25 DOF) to approach the SPHARM ceiling** on smooth shapes — heavier for SA fitting — but uniquely handles branching and **unifies shape + split + ILP 2-cell hypothesis** in one representation. Hard union beat the (soft Gaussian) metaball at equal DOF in this test.
- Visual overlays: `phase0b_shapes.png` (ellipse badly misses the bent/bilobed cell; radial and union follow it).

**Open representation decision (maintainer to pick):** SPHARM-primary (+union-2 for divisions) vs union-of-N/metaball-only. Trade: SPHARM = higher smooth-shape fidelity per DOF; union-of-N = branching + native division/ILP integration, at more DOF.

## Gaussian-mixture / splatting fit check (2026-07-02, `phase0b_gmm.py`)

Chosen representation = cell as a mixture of a few anisotropic Gaussians (each a rotated `Ellipsoid`), fit by EM. Tested on the 102 SEG nuclei (uniform mask samples → isolates representation from tracker fit quality).

**1. Ceiling — VALIDATED.** EM-fit K-Gaussian mixture level-set, best-threshold Jaccard:

| K | mean Jaccard |
|---|---|
| 1 (single Gaussian) | 0.882 (≈ ellipse 0.877 — sanity check passes) |
| 2 | 0.903 |
| 3 | **0.927** |

EM matches/beats the hand-rolled Nelder-Mead metaball and is a robust CPU fitter. The representation clears the "both equally" bar at K=3.

**2. Automatic K-selection (BIC) — UNRELIABLE.** BIC on dense pixel samples over-segments: with all pixels it picked K=3 for 100/102 nuclei (0% of round nuclei got K=1). It's resolution/sample-count dependent, and no fixed sample cap cleanly separates structure:

| sample cap | round→K=1 | concave→K≥2 |
|---|---|---|
| 80 | 83% (good) | 27% (misses dumbbells) |
| 150 | 62% | 55% (coin-flip) |
| 300 | 14% | 91% (over-segments all) |

**Design implication (coherent with 0a):** do **not** rely on per-frame BIC to pick the cell/lobe count. The GMM **emits** the nested K=1/2/3 hypotheses; **selection is the windowed ILP's job** (temporal evidence), not a single-frame criterion. This is exactly 0a's lesson — single-frame 1-vs-2 is ambiguous; neighbors decide. So Workstream **S (GMM hypothesis emission)** and **L (ILP selection)** genuinely require each other; the committed per-frame render can use a conservative default K with K=2/3 carried as split hypotheses.

## Combined 0a + 0b steer
- Drop **S1 (superquadric)** entirely.
- **S2 (union-of-2 / metaball)** keeps a dual, *modest* justification: the dividing-nucleus SEG cases here **and** the ILP's 2-cell hypothesis generator (Phase 0a). Still not a big-win item on its own.
- The largest available SEG lever appears to be **closing the ellipse-fit gap**, not enriching the shape. Confirm with the actual-Jaccard diagnostic next.
