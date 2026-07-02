# Phase 0a — Windowed-ILP offline validation: RESULTS

**Date:** 2026-07-02
**Script:** `windowed_ilp_proto.py`  (PuLP + bundled CBC, CPU, label-free)
**Data:** committed run `outputs/Yiding_1~171_VISUAL_TIF/` (CSV + `real/N_real.tif` image), GT `01_GT/TRA/man_track.txt`
**Gate (from build plan):** *does a W=1 window flip known division-timing / merge errors the per-frame greedy got wrong?*

## Setup (honesty rules)
- **Image evidence** = the tracker's own preprocessed `real/N_real.tif` volume (the exact image it fit). We never score against GT masks (that would encode the answer).
- **Detection score** = `Σ(intensity − background)` over a hypothesis's voxels (MDL-style): a too-big 1-cell blob that spans the dark gap loses voxels; a spurious 2nd daughter over background adds negative voxels, so over-splitting is penalised. 1-cell vs 2-cell are directly comparable.
- **Geometry** matches C++ exactly (`R = Rz·Ry·Rx`, `local = Rᵀ(world−center)`, `a=major, b=bRadius, c=minor`).
- **GT truth**: founders matched at f0 → `cell_0↔GT256, cell_1↔GT1, cell_2↔GT511, cell_3↔GT640` (xy within 1–2 px; z scale ≈ 6.5×, GT is 35 slices vs 239).

## Results

### 1. ILP core is correct — `--selftest` PASS
Synthetic lineage that truly divides at frame 2, with a single-frame noise spike making per-frame greedy pick "1 cell" at frame 3 (a flicker / un-division). Greedy flickers; the ILP holds the division and stays 2-cell. **The solver + flow/event/disjointness constraints work.**

### 2. Real over-split cases — the window did NOT override the image
Cell-count says tracker > GT at f9–10 and f34 (candidate over-splits). Per-lineage truth via the f0 correspondence:

| Lineage | GT id | GT div frame | tracker div | image at commit | ILP result | verdict |
|---|---|---|---|---|---|---|
| `cell_3`/`_1_3` | 511 | **f7** | f9 | decisive merge f6–8, split f9+ | keeps merge till f9 | **LATE split** — 2-cell hyp at f7–8 doesn't exist offline → *not fixable here* (needs S2/L1) |
| `cell_3`/`_1_4` | 640 | **f11** | f9 | **decisive split at f9 (1.8×)** | split at f9 (=greedy=tracker) | image already bimodal 2 frames before GT marks it → **GT-annotation gap, not a greedy error**; window correctly does not override |
| `_1_200` | (deep) | ~f33–34 | f34 | **ambiguous (≈1% margin at f33)** | split at f33 | image genuinely ambiguous; window agreed with the razor-thin per-frame call (landed f33, tracker f34) |

**On real data the W=1..3 window never flipped a known greedy timing error toward GT.** In every real case the ILP agreed with the per-frame image argmax: where the image was decisive it followed it; where it was ambiguous (f34) it didn't override.

## Why the offline test is underpowered (the key caveat)
The committed CSV only stores the hypotheses greedy **kept**. The ILP's actual value is choosing among **competing** hypotheses — especially picking a 2-cell fit at a frame where greedy committed only 1 cell (the LATE-split case, the *majority* of real timing errors here: f4, f7, f18…). Those competing 2-cell hypotheses require an **image 2-blob fit** (Workstream **S2 metaball / L1 emission**) — they cannot be reconstructed from committed output. So this offline harness can validate the *machinery* (it does) but cannot fairly validate the *payoff* on real data.

## Verdict & recommendation
- **Literal gate:** offline W=1 changed nothing vs greedy on real data → **not a green light for L2 on its own.**
- **But not a red light either:** the experiment is structurally blind to the ILP's main use (real competing hypotheses), which don't exist until S2/L1.
- **Recommendation:** do **not** build the full windowed ILP (L2) next. Build **Workstream S first** — S1 superquadric (the SEG win, low risk) then **S2** (image 2-blob / metaball = the real 2-cell hypothesis generator). Then re-run *this exact harness* fed by S2's emitted hypotheses (that is Phase L1→L2). The prototype is written to be reused for that.

This matches the plan's dependency graph: **S2 must precede L1**; the ILP is only as good as the hypotheses it selects from.

## Naming / schema note (per maintainer, 2026-07-02)
- Prototype reporting uses readable lineage labels (founder + GT id), not raw `Cell type 1_N`.
- The CSV **column** rename (`majorRadius/bRadius/minorRadius` → clearer semi-axis names, plus a shape-exponent column) should ride with **S1**, because the superquadric changes what the radii mean anyway. Not applied to existing committed CSVs (run evidence).
