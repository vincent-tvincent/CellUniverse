# Conference Poster — Design Doc (efficiency-forward)

**Project:** CellUniverse — lightweight 3D cell tracking & lineage reconstruction
**Validation:** Cell Tracking Challenge `Fluo-N3DH-CE` (*C. elegans* embryo, fluorescent nuclei)
**Audience:** primarily biology, professional / quantitative
**Date:** 2026-06-17

Decision (2026-06-17): the full-sequence benchmark is **mid-pack** (OP_CTB 0.75), so we
**do not** make the leaderboard the headline. Instead we lead with **efficiency** —
CellUniverse reaches competitive tracking **without GPUs, without training, and without
labelled data** — and use the benchmark as honest supporting validation.

> **Defaults to confirm:** A0 **landscape** (1189 × 841 mm), 3-column grid. Set the real
> title, authors, affiliations, logos before printing.

---

## 1. The one-sentence story

> *CellUniverse reconstructs the full cell lineage of a developing C. elegans embryo
> from 3D microscopy using a classical, training-free optimiser that runs on a laptop
> CPU — reaching tracking accuracy (TRA ≈ 0.95, DET ≈ 0.97) competitive with deep
> networks that need GPUs and large annotated training sets.*

The takeaway a passer-by should get in 5 seconds: **"accurate cell tracking + full
lineage, with none of the deep-learning infrastructure — no GPU, no training data."**

---

## 2. Why efficiency is the right headline

The Cell Tracking Challenge leaders on this dataset are **deep neural networks**: they
require (a) a **GPU**, (b) **manually annotated training data**, and (c) **hours–days of
training** before they can track a single frame. CellUniverse is a **model-based
stochastic optimiser** — it fits parametric ellipsoids directly to the raw image, so it
has **zero training cost and no data requirement**, yet lands close on accuracy. For a
biologist who wants to run a tool on their own embryo movies today, that trade is the
whole point.

| | **CellUniverse (ours)** | Typical CTC leaderboard method |
|---|---|---|
| Approach | Classical model fitting (ellipsoids + stochastic search) | Deep neural network |
| Labelled training data | **None** | Required (ELEPHANT: ~2% nuclei hand-labelled) |
| GPU required | **No** (multicore CPU) | Yes (GTX 1080 Ti / 2× Titan RTX) |
| Training time | **None** (runs immediately) | ≥30 GPU-hours (ELEPHANT) up to multi-day |
| Runtime | ≈ 53 s/frame (~70 cells) on an 8-core laptop CPU | sub-second/frame GPU inference — **after** training |
| Peak memory | ≈ 2.9 GB RAM (no VRAM) | GPU VRAM + host RAM |
| Tracking accuracy (TRA) | **0.95** | 0.98–0.99 |
| Detection accuracy (DET) | **0.97** | — |

*(This table is Table 1 — the centerpiece. Keep it honest: we are slightly behind on
accuracy and we are NOT faster per frame, but our **total cost to a result on a new
dataset is far lower** — zero annotation, zero training, no GPU.)*

> **Honesty guardrail (important).** Do **not** claim CellUniverse is "faster" than deep
> nets per frame — it is not (≈53 s/frame vs sub-second GPU inference). The true, defensible
> efficiency claim is **total cost of ownership**: no GPU, no labelled training data, and
> no training step, so a biologist can point it at a new embryo movie and get a lineage
> immediately, whereas a deep method first needs annotation + GPU training (hours–days).
> Frame the headline as *"no training, no GPU, no labels"* — not *"faster."*

---

## 3. Layout map (A0 landscape, 3 columns)

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│  TITLE  ·  Authors  ·  Affiliations                         [Lab logo] [QR]   │  banner
├──────────────────────────┬──────────────────────────┬────────────────────────┤
│ COLUMN 1                  │ COLUMN 2                  │ COLUMN 3                │
│ 1. Background & Question  │ 3. Method (pipeline)      │ 5. Efficiency (HEADLINE)│
│    why lineage matters    │    [Fig 1 pipeline]       │    [Table 1 cost vs DL] │
│    [Fig 0 hero render]    │ 4. Key idea: ellipsoid +  │    [Fig 4 cost-accuracy]│
│ 2. The Challenge          │    split guidance, no NN  │ 6. Validation (CTC)     │
│    dense 3D, divisions    │    [Fig 2 ellipsoid fit]  │    [Table 2 benchmark]  │
│    [Fig 0b raw vs model]  │                           │    [Fig 5 GT overlay]   │
│                           │                           │ 7. Results: lineage     │
│                           │                           │    [Fig 3 lineage tree] │
│                           │                           │    [Fig 6 count-vs-time]│
│                           │                           │ 8. Limitations & Future │
├──────────────────────────┴──────────────────────────┴────────────────────────┤
│  References · Acknowledgements · Code/Contact · QR to demo video               │  footer
└─────────────────────────────────────────────────────────────────────────────┘
```

The two figures people will photograph: **Fig 4 (cost-vs-accuracy)** and **Fig 3
(lineage tree)**. Make them the largest.

---

## 4. Section-by-section content (short bullets)

**1. Background & Question** *(biology hook)*
- The *C. elegans* embryo builds itself by a precise series of cell divisions — the
  **cell lineage** — central to developmental biology.
- 3D time-lapse microscopy captures it; reconstructing the lineage by hand is infeasible.
- **Q:** can we recover the complete lineage automatically — *without* the cost of deep
  learning?

**2. The Challenge**
- Nuclei are densely packed and touch (>300 cells late); they divide, drift, vary in
  brightness; z-resolution is coarse.
- Need correct identity through time and correct divisions.

**3. Method — pipeline** *(Fig 1)*
- Per frame, fit a **tri-axial ellipsoid** per nucleus directly to the 3D image
  (position, radii, orientation) by stochastic optimisation.
- **Split guidance (CellLumen):** a raw-signal detector proposes divisions so dense,
  dividing nuclei separate correctly.
- Identity carried across frames → **lineage tree**.

**4. Key idea — model fitting, not learning** *(Fig 2)*
- A compact parametric cell the optimiser refines frame to frame.
- No network, no weights, no training set — just the image and the model.

**5. Efficiency — the headline** *(Table 1, Fig 4)*
- **No GPU, no training, no labelled data.** Dependencies: OpenCV + OpenMP (CPU).
- Runs on a laptop-class **8-core CPU**, ≈ **53 s/frame** (~70 cells), ≈ **2.9 GB** peak RAM.
- Competitive accuracy with **zero setup cost** — point it at a new movie and go.
- (Honest framing: the win is no-training/no-GPU total cost, not per-frame speed.)

**6. Validation — Cell Tracking Challenge** *(Table 2, Fig 5)*
- Scored with the **official CTC measures** on the full sequence (frames 0–194).
- **DET 0.97, TRA 0.95** — strong detection and tracking; SEG 0.55 (segmentation is the
  current limit). OP_CTB 0.75.

**7. Results — the reconstructed lineage** *(Fig 3, Fig 6)*
- Full lineage across 195 frames (6 → 322 cells).
- Cell-count-vs-time tracks ground truth → divisions captured.

**8. Limitations & Future Work**
- Segmentation boundary fidelity (bright core vs full nucleus); dense-frame merge errors.
- Faster still / streaming; generalise across embryos & modalities.

---

## 5. Figure & table inventory

Legend: ✅ have · 🟡 quick to make · ✏️ draw.

| # | Figure / table | Shows | Source | Priority |
|---|---|---|---|---|
| **Table 1** | **Cost vs deep learning** | training data / GPU / training time / runtime / accuracy | §2 (numbers below) | 🟡 headline |
| **Fig 4** | **Cost-vs-accuracy plot** | x = setup cost (none→GPU+training), y = TRA; ours in the cheap-and-accurate corner | 🟡 make | headline |
| **Fig 0** | Hero 3D render | tracked nuclei colored by lineage | still from `viz/demo_3d_f1-f10.mp4` / `demo_ellipsoid_3d` | ✅/🟡 |
| **Fig 1** | Pipeline schematic | 3D frames → ellipsoid fit + split guidance → lineage | draw | ✏️ |
| **Fig 2** | Ellipsoid + division | nucleus with overlaid ellipsoid; a division | snapshot | 🟡 |
| **Fig 3** | **Lineage tree** | dendrogram, founders → divisions vs time (652 tracks) | from `merged_0-194.csv` | 🟡 must |
| **Table 2** | Benchmark | DET/SEG/TRA/OP_CTB, ours vs leaderboard top-3 | below | 🟡 |
| **Fig 5** | GT-vs-mask overlay | our mask vs expert GT on annotated slices | `outputs/ctc_eval_Yiding_1-171_signal/overlay_seg.png` | ✅ |
| **Fig 6** | Cell count vs time | tracked count/frame vs GT (divisions on time) | `merged_0-194.csv` + GT | 🟡 |

### Table 2 — benchmark numbers (full sequence 01, frames 0–194)

| Measure | **CellUniverse** | CTC leaderboard, Fluo-N3DH-CE (top 3) |
|---|---|---|
| DET | **0.967** | — |
| TRA | **0.954** | 0.994 / 0.987 / 0.979 |
| SEG | **0.547** | 0.759 / 0.729 / 0.725 |
| OP_CTB = ½(SEG+TRA) | **0.751** | 0.850 / 0.844 / 0.829 |

*Footnote:* official CTC measures (py-ctcmetrics), full sequence 01 (195 frames), trash
objects excluded; leaderboard values are averaged over both sequences — indicative
comparison, not a ranked submission.

### Fig 4 — barrier-to-entry / time-to-result (the headline figure) — BUILT

File: `outputs/poster_figs/fig4_cost_comparison.png`. Two panels:
- **Left, requirements matrix:** rows = Manual / Deep learning / CellUniverse; columns =
  GPU required, labelled training data, training step, runs on commodity CPU. CellUniverse
  is all-green (none needed); deep learning is all-red except CPU.
- **Right, "time to first lineage on a new dataset"** (log hours): Manual ~2–3 months,
  Deep-learning training-only ≥30 GPU-h (+ GPU + ~2% labels), **CellUniverse ~12 h, CPU,
  no setup.** Footnote is explicit that the deep bar is training-only and per-frame
  inference is fast once trained — so we never claim a per-frame speed win.

### Justifying the time claim (sourced)

The honest argument is **total cost-to-result / barrier-to-entry**, not per-frame speed:

- The **Cell Tracking Challenge publishes no runtime metric** today (only accuracy);
  the original benchmark even noted classical methods can be among the fastest, and the
  classical CPU method **KTH-SE/Baxter tops Fluo-N3DH-CE** — so "classical + CPU" is no
  accuracy handicap here (Ulman 2017, Nature Methods; Magnusson & Jaldén 2015, IEEE TMI).
- Every competitive **deep** tracker on this dataset carries costs absent from the
  accuracy tables: a **GPU**, a **training step**, and **labelled data**. The most
  annotation-efficient one on Fluo-N3DH-CE, **ELEPHANT**, still needs **~30 GPU-hours of
  training, a GPU (GTX 1080 Ti), and ~2 % of nuclei hand-annotated** (Sugawara 2022,
  eLife); **KIT-GE** trains on **2× Titan RTX** (Scherr 2020, PLOS ONE).
- Manual annotation is the field's real bottleneck: CTC gold-truth coverage is only
  **17.8 %** "due to the labor-intensive nature of manual annotations" (Maška 2023,
  Nat. Methods), and manual lineaging runs **2–3 months** (Wolff 2018, eLife).
- **CellUniverse:** a full 195-frame, ~300-cell lineage in **~12 h on a commodity 8-core
  CPU, zero labels, zero training, no GPU.** The 12 h *is* the entire cost — on hardware
  any lab already owns.

One-line poster claim: *"From raw movie to full lineage in ~12 h on a laptop CPU — no
GPU, no training, no labelled data; deep methods need all three before they can start."*

---

## 6. Style guidance (bio conference)

- Less text, more figures (~30–40% area). Bullets, not paragraphs.
- One accent color for "ours" everywhere; neutral grey for baselines/GT.
- Big fonts: title ≥ 80 pt, headers ≥ 36 pt, body ≥ 24 pt, captions ≥ 20 pt.
- Self-contained captions stating the takeaway.
- Color-blind-safe palette; label the overlay colors.
- QR codes: demo video + code/contact. Biologists love the moving 3D render.
- Keep AOGM/Jaccard math off the poster (one-line footnote at most).

---

## 7. What to emphasize vs cut

**Emphasize:** Table 1 + Fig 4 (cost vs deep learning), the lineage tree (Fig 3), the
hero render (Fig 0). **Cut:** AOGM internals, CSV→CTC plumbing, per-slice SEG analysis,
the two-run merge mechanics — Q&A material.

---

## 8. Title options (efficiency-forward)

- "Cell lineage reconstruction without deep learning: a training-free, CPU-only tracker
  for the *C. elegans* embryo"
- "No GPU, no training, full lineage: lightweight 3D cell tracking validated on the Cell
  Tracking Challenge"
- "CellUniverse: competitive embryo cell tracking at a fraction of the compute"

---

## 9. A0 portrait variant

Header, then 2 columns. Left = sections 1→4 (Background, Challenge, Method, Key idea).
Right = sections 5→8 (Efficiency, Validation, Lineage, Limitations). Keep Table 1/Fig 4
and Fig 3 large in the right column.

---

## 10. Figures — status

Generated in `outputs/poster_figs/` (PNG, 150 dpi):
- ✅ **Fig 0** `fig0_hero.png` — embryo nuclei in 3D at frames 30/90/160, colored by lineage.
- ✅ **Fig 3** `fig3_lineage_tree.png` — full lineage dendrogram (652 cells, 4 founders).
- ✅ **Fig 4** `fig4_cost_comparison.png` — requirements matrix + time-to-result bar.
- ✅ **Fig 5** `fig5_gt_overlay.png` — mask vs GT on annotated slices.
- ✅ **Fig 6** `fig6_count_vs_time.png` — tracked cell count vs GT over time.

Still to do (by hand): **Fig 1 pipeline schematic** — I can give a precise box/arrow spec.
Measured numbers now in Table 1: ~53 s/frame (~70 cells), ~2.9 GB RAM, full run ~12 h.
