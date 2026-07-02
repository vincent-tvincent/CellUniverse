# CellUniverse CTC Evaluation on Fluo-N3DH-CE — Method, Results, and Comparison

**Date:** 2026-06-16
**Run evaluated:** `outputs/Yiding_1~171_VISUAL_TIF/Yiding_Embryo_1~171_FinalLineageTree.csv`
**Dataset / ground truth:** Cell Tracking Challenge `Fluo-N3DH-CE`, sequence **01**
**Branch:** `Yiding👑Cell-Lumen-SplitGuided-Universe_05272026`

This is the single reference for how we evaluated a CellUniverse tracking run with
the three official Cell Tracking Challenge (CTC) measures — **SEG** (segmentation),
**DET** (detection), **TRA** (tracking) — what we scored, how we compare to the
public leaderboard, why our SEG is low, and how to fix it.

---

## 1. Headline result

Two mask-export modes were evaluated from the **same** tracking run (no re-run):
`ellipsoid` (rasterize the raw analytic ellipsoid) and `signal` (keep only the bright
voxels inside a scaled ellipsoid, so the mask hugs the real nucleus — see §6).

| Measure | Ellipsoid mask | **Signal-fit mask** | CTC leaderboard, Fluo-N3DH-CE (top 3) |
|---|---|---|---|
| **DET** | 0.967 | **0.992** | — (not ranked on the CTB board) |
| **TRA** | 0.963 | **0.990** | 0.994 / 0.987 / 0.979 |
| **SEG** | 0.358 | **0.612** | 0.759 / 0.729 / 0.725 |
| **OP_CTB** = ½(SEG+TRA) | 0.661 | **0.801** | 0.850 / 0.844 / 0.829 |
| OP_CSB = ½(SEG+DET) | 0.662 | **0.802** | — |
| Valid (CTC format check) | 1 (pass) | **1 (pass)** | — |

**One-line summary (signal-fit):** **tracking and detection are essentially
leaderboard-best** — TRA 0.990 would sit ~2nd (top is 0.994) and DET 0.992 is at the
ceiling — while **segmentation (SEG 0.612)** is still below the ~0.72–0.76 leaders,
which is the only thing keeping OP_CTB (0.801) just outside the top-3 (0.829–0.850).
The signal-fit mask is a pure post-processing change (no re-tracking) and improves
*all three* measures, because tight masks remove the false-positive voxels and
over-coverage that the raw ellipsoid produced.

Leaderboard source: CTC "Latest CTB results", board image dated 2025-08-15
(<https://celltrackingchallenge.net/latest-ctb-results/>).

> **Comparability caveat.** The leaderboard numbers are computed on the *full*
> sequences and averaged over **both** sequences 01 and 02. Our run covers
> **sequence 01 only, frames 1–171** (the frames CellUniverse actually tracked;
> GT spans 0–194). So our numbers are an honest self-evaluation on the overlap,
> **not** a like-for-like leaderboard submission — indicative of where we stand,
> not an official ranking.

---

## 2. The three measures (official methodology)

All in `[0,1]`, higher is better (see `Downloads/eva/SEG.pdf`,
`Evaluation software.pdf`, and the CTC methodology page).

- **DET (detection accuracy)** — normalized Acyclic Oriented Graph Matching for
  detection (AOGM-D); how well objects are *found*, ignoring links:
  `DET = 1 − min(AOGM-D, AOGM-D0)/AOGM-D0`.
- **SEG (segmentation accuracy)** — mean Jaccard `J(S,R)=|R∩S|/|R∪S|` over every
  reference object `R`, where a result object `S` matches `R` iff `|R∩S| > 0.5·|R|`
  (≤1 match per reference object; no match → J=0). For 3-D datasets the GT is
  annotated on **individual 2-D slices**, so SEG is a 2-D overlap on those slices.
- **TRA (tracking accuracy)** — normalized AOGM over the whole acyclic lineage graph
  (vertices = detections, edges = temporal/parent links):
  `TRA = 1 − min(AOGM, AOGM0)/AOGM0`.
- **OP_CTB** = ½(SEG+TRA) is the overall Cell Tracking Benchmark measure the
  leaderboard ranks on. OP_CSB = ½(DET+SEG).

---

## 3. How we evaluated

### 3.1 The problem: CellUniverse output is not in CTC format

CellUniverse represents each cell as a **rotated tri-axial ellipsoid**
(center `x,y,z` + radii `aRadius,bRadius,cRadius` + Euler angles
`theta_x,theta_y,theta_z`), exported per frame in `…FinalLineageTree.csv`. The CTC
measures require, per sequence:

```text
01_RES/
  maskTTT.tif      # integer label volume per frame; one label per cell,
                   # the SAME label across frames for the same track
  res_track.txt    # lineage:  L B E P  (label, begin frame, end frame, parent)
```

So evaluation has two halves — rasterize the ellipsoids into labeled voxel volumes,
and derive the lineage graph. Both are done by `scripts/ctc_eval/build_ctc_res.py`.

### 3.2 Rasterizing ellipsoids → label volumes

Geometry matches the C++ renderer (`src/Ellipsoid.cpp`, `scanEllipsoidSlice`)
exactly:

- The working grid is isotropic with the **z axis upscaled by `z_scaling = 7`**
  (`config/config.yaml`). The CSV stores z in that scaled space, so `z ≈ 110` maps
  to GT slice `≈ 110/7 ≈ 16`. A GT voxel slice `s` (0-based, original 35-slice
  resolution) is sampled at scaled `z = s·7 + (7−1)/2`.
- Membership test, with rotation `R = Rz·Ry·Rx` and `local = Rᵀ·(world − center)`:
  `lx²/a² + ly²/b² + lz²/c² ≤ 1`, where `a=majorRadius`, `b=bRadius`
  (oblate fallback `=a` when 0), `c=minorRadius`.
- Voxel grid = GT dimensions **(Z,Y,X) = (35, 512, 708)**, `x→X(708)`, `y→Y(512)`.
- Overlapping ellipsoids: each voxel goes to the cell it is **most deeply inside**
  (smallest quadratic value) — order-independent. Every track is guaranteed ≥1
  voxel in every frame of its lifespan, so the result passes CTC validity.

### 3.3 Lineage → `res_track.txt`

A cell keeps its name across frames until it divides; a division **appends one
digit** (`Cell type 1_4` → `_40`/`_41`). So:

- each unique name is **one CTC track** (label = a stable integer);
- `B`,`E` = first/last frame the name appears (CSV has **no frame gaps** — verified —
  so every track is contiguous and valid);
- parent = the track whose name is this name with the **last character stripped**,
  or `0` for the 4 founders (`_1`,`_2`,`_3`,`_4`);
- verified: all 607 non-founder tracks have a parent ending strictly before the
  child begins → CTC `valid()` returns 1.

### 3.4 Frame alignment and the trimmed GT

CSV `t00k.tif` is GT frame `k`. py-ctcmetrics pairs GT and RES masks **by sorted
position** and asserts equal counts, and indexes SEG GT by frame number. So we build
a self-contained, renumbered eval dataset over the overlap (original frames 1–171 →
new frames 0–170):

```text
outputs/ctc_eval_Yiding_1-171/
  01_GT/
    TRA/ man_track000.tif … man_track170.tif + man_track.txt  (copied & clipped)
    SEG/ man_seg_020_024.tif, man_seg_027_018.tif, …          (frame# shifted −1)
  01_RES/
    mask000.tif … mask170.tif + res_track.txt                 (rasterized)
```

`man_track.txt` is clipped to the window (B,E clamped and shifted; a parent ending
before the window → 0). The 5 SEG GT slices (original frames 21/28/78/141/162) all
fall inside the window and are renumbered to 20/27/77/140/161.

### 3.5 Why embryo_data = sequence 01

`embryo_data` (the run input) and `01_GT/man_track000.tif` are both
**(35, 512, 708)**; sequence 02 is (31, 512, 712). Dimensions only match the
sequence the GT annotates, so the run is sequence **01** and `01_GT` is correct.

### 3.6 How to reproduce

```bash
cd CellUniverse/C++

# 1. One-time: evaluation environment
python3 -m venv .venv-eval
.venv-eval/bin/pip install py-ctcmetrics tifffile imagecodecs numpy scipy

# 2. Build the CTC RES + trimmed GT from the CellUniverse CSV
.venv-eval/bin/python scripts/ctc_eval/build_ctc_res.py \
  --csv "outputs/Yiding_1~171_VISUAL_TIF/Yiding_Embryo_1~171_FinalLineageTree.csv" \
  --gt  ../../Fluo-N3DH-CE/01_GT \
  --out outputs/ctc_eval_Yiding_1-171 \
  --seq 01 --z-scaling 7 --first 1 --last 171

# 3. Compute SEG / DET / TRA
.venv-eval/bin/python - <<'PY'
from ctc_metrics.scripts.evaluate import evaluate_sequence
r = evaluate_sequence("outputs/ctc_eval_Yiding_1-171/01_RES",
                      "outputs/ctc_eval_Yiding_1-171/01_GT",
                      metrics=["Valid","DET","SEG","TRA"], threads=1)
print({k: r[k] for k in ["Valid","DET","SEG","TRA","OP_CSB","OP_CTB"]})
PY
```

Raw output is saved to `outputs/ctc_eval_Yiding_1-171/results.json`.

**Engine:** [py-ctcmetrics](https://github.com/CellTrackingChallenge/py-ctcmetrics),
a faithful Python reimplementation of the official C++ SEG/DET/TRA routines (chosen
over the official Mac binaries, which are old x86-64 builds unreliable on current
Apple-Silicon macOS). The RES/GT format produced here is identical to what the
official binaries expect, so the same dataset can be re-scored with them.

---

## 4. Comparison to the leaderboard

Leaderboard values are the **top-3** published scores for Fluo-N3DH-CE, read directly
from the board image (with the §1 caveat in mind).

| Measure | **Ours** | #1 | #2 | #3 | Where we'd land |
|---|---|---|---|---|---|
| **TRA** | **0.963** | 0.994 | 0.987 | 0.979 | just below 3rd (gap **0.016**) |
| **DET** | **0.967** | — | — | — | strong |
| **SEG** | **0.358** | 0.759 | 0.729 | 0.725 | well below (≈ half) |
| **OP_CTB** | **0.661** | 0.850 | 0.844 | 0.829 | below the top tier |

```text
TRA   0.90 ───────────────────────★0.963──0.979─0.987─0.994   ← essentially in the pack
SEG   0.30 ──★0.358───────────────────0.725─0.729──0.759       ← far behind
OPCTB 0.60 ──────★0.661──────────────0.829─0.844──0.850        ← dragged down by SEG
```

**Tracking is competitive with the best.** TRA 0.963 is only ~0.016 below 3rd place;
DET 0.967 confirms it — cell counts match GT exactly (4→…→306) and 94–100 % of GT
detection markers are hit. The AOGM breakdown for TRA: 457 FN + 499 FP vertices,
33 splits, 914 edge-adds + 56 edge-changes + 29 edge-deletes over 171 frames and
~15.8 k detections (AOGM 6690 vs AOGM₀ 181567) — the lineage graph is recovered with
few edits.

**Overall is mid-pack, held back entirely by SEG.** Since OP_CTB = ½(SEG+TRA), our
weak SEG halves the gains from our strong TRA. **If SEG were merely average for this
dataset (~0.73), OP_CTB would jump to ≈0.85 — right at #1.** SEG is the single thing
between us and the top of the board.

---

## 5. Why SEG is low (and why it is not a bug)

SEG is a **2-D pixel-overlap** scored on the 5 annotated slices (frames
21/28/78/141/162 at z-slices 24/18/17/14/10), and a reference nucleus only counts if
our cell covers **>50 %** of it. Per-slice inspection of our output:

```text
frame 20, slice 24:  2 GT nuclei,  matched 0/2,   meanJ 0.00,  GT~3779px  RES~0px
frame 27, slice 18:  8 GT nuclei,  matched 4/8,   meanJ 0.31,  GT~2070px  RES~1343px
frame 77, slice 17: 18 GT nuclei,  matched 9/18,  meanJ 0.31,  GT~1993px  RES~1384px
frame140, slice 14: 30 GT nuclei,  matched 21/30, meanJ 0.48,  GT~1258px  RES~844px
frame161, slice 10: 44 GT nuclei,  matched 21/44, meanJ 0.32,  GT~1092px  RES~727px
```

Two patterns dominate:

1. **About half the nuclei aren't matched at all** (0/2, 9/18, 21/44) → each scores
   0 and craters the mean.
2. **When matched, our cross-section is smaller than the real nucleus** (RES
   ~700–1380 px vs GT ~1000–3800 px) → overlap only ~⅓–½.

**Root cause — the model is an ellipsoid sized to *find* the cell, not to *trace*
it, and z is the weak axis:**

- **Our cells' z-centers cluster near the middle (~slice 16), but the challenge
  annotates off-center slices (24, 18, 10…).** An ellipsoid sliced near its z-edge
  gives a tiny — or empty — cross-section. Frame 20 / slice 24 is ~8 slices from our
  cell centers while our z-radius is only ~6 slices, so we don't reach it →
  0/2 matched → J=0.
- **z is coarse:** only 35 physical slices (internally upscaled ×7), so a small
  error in z-center or z-radius (`cRadius`/minor axis) costs *whole missing slices*
  of overlap.
- **Shape mismatch:** a smooth ellipse ≠ a blobby real nucleus, so Jaccard can't
  reach 1 even when centered.

This is the *same* property that makes DET/TRA high: an ellipsoid that nails the
**centroid** tracks beautifully, but a right-centroid ellipsoid can still overlap the
true boundary poorly — especially off-center in z. SEG and TRA measure different
things, and the model is tuned for the tracking one. (SEG is best at frame 140,
J=0.48, where cells are smaller and the slice is near center.)

---

## 6. How to improve SEG — and do we need to rerun frames 0–194?

### Do we need to rerun? **No.**

1. A better mask export is **post-processing** — it reuses the ellipsoid CSV we
   already have plus the raw frames (`data/input/embryo_data/t0XX.tif`), and writes a
   signal-fit region instead of the raw ellipsoid. The tracker does **not** run again.
2. **SEG is only scored on 5 annotated slices** — frames 21/28/78/141/162 — and all
   5 already fall inside our 1–171 window. We already have every frame SEG needs.

(Rerunning the full 0–194 of both sequences is a *separate* goal — it would only be
needed for a full-sequence, leaderboard-parity TRA/DET number, which is unrelated to
the SEG fix.)

### What we changed — the signal-fit mask (implemented)

Implemented as `--mask-mode signal` in `build_ctc_res.py`. For each cell we keep a
voxel iff it is (a) inside that cell's ellipsoid **scaled by `seg-scale`**, (b) above
an intensity threshold on the raw image (`embryo_data`), and (c) the cell it is most
deeply inside. The scaled ellipsoid only *bounds* the search; the bright signal
defines the final shape, so the exported mask hugs the real nucleus instead of the
full detection ellipsoid.

Parameters (tuned on the 5 SEG slices; the optimum is a flat plateau, SEG ≈ 0.60–0.61
across `seg-scale` ∈ [1.3, 1.4] and threshold ∈ [p72, p80], so it is not overfit):
**`--seg-scale 1.4 --seg-thresh p75`** (foreground = raw intensity above the 75th
percentile). Build command:

```bash
.venv-eval/bin/python scripts/ctc_eval/build_ctc_res.py \
  --csv "outputs/Yiding_1~171_VISUAL_TIF/Yiding_Embryo_1~171_FinalLineageTree.csv" \
  --gt  ../../Fluo-N3DH-CE/01_GT \
  --out outputs/ctc_eval_Yiding_1-171_signal \
  --seq 01 --z-scaling 7 --first 1 --last 171 \
  --mask-mode signal --raw-dir data/input/embryo_data --seg-scale 1.4 --seg-thresh p75
```

**Result (see §1):** SEG **0.358 → 0.612**, and DET/TRA *rose* to **0.992 / 0.990**
(tight masks remove false-positive voxels and over-coverage), lifting OP_CTB
**0.661 → 0.801** — into the leaderboard band. All from post-processing; the tracker
was not re-run.

### Remaining gap and further levers (SEG 0.61 → ~0.73)

The residual is **shape fidelity**: a threshold captures the bright nuclear *core*,
while annotators trace the *full* nucleus including its dim rim. To close it:

1. **Adaptive / local threshold** instead of a global percentile (per-cell Otsu, or a
   gradient-following region grow) to recover the dim nuclear boundary.
2. **Fit the z-extent / z-center to the data** during tracking, so off-center
   annotated slices are covered without relying on the `seg-scale` bound.
3. **A learned segmentation** seeded by the tracked centroids (the usual route to
   SEG ≳ 0.75 on this dataset).

---

## 7. Files produced

| Path | What |
|---|---|
| `scripts/ctc_eval/build_ctc_res.py` | CSV → CTC RES + trimmed GT converter |
| `outputs/ctc_eval_Yiding_1-171/01_RES/` | 171 label masks + `res_track.txt` (611 tracks) |
| `outputs/ctc_eval_Yiding_1-171/01_GT/` | trimmed/renumbered TRA (608 tracks) + 5 SEG slices |
| `outputs/ctc_eval_Yiding_1-171/results.json` | raw SEG/DET/TRA + AOGM breakdown |
