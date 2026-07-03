# Design — Generative metaball shape model + joint segmentation/tracking ILP

**Date:** 2026-07-02
**Status:** Design (brainstorm output). Next: implementation plan (`writing-plans`).
**Depends on:** `docs/plans/2026-07-02-post-ultrack-directions.md`, `docs/plans/2026-07-02-shape-and-ilp-buildplan.md`, `scripts/ilp_proto/RESULTS.md`, `scripts/ilp_proto/RESULTS_phase0b.md`.

## 0. What was validated first (why this design)

Four offline experiments (no C++ touched) gated every choice below:

- **0a — windowed ILP** (`scripts/ilp_proto/`): the CBC/PuLP ILP core works (removes flicker), but on committed output it could not override the per-frame image call — the real timing errors are **late splits whose 2-cell hypothesis doesn't exist in committed data**. ⇒ the ILP needs a real hypothesis *generator*.
- **0b — shape ceiling**: ellipse per-slice Jaccard ceiling ≈ **0.89** vs reported SEG **0.358** ⇒ the SEG loss is dominantly a **fit/placement gap, not a primitive gap**. Superquadric adds nothing (+0.005). But **28% of nuclei are genuinely concave** and HeLa is worse ⇒ an ellipse/superquadric (both convex) cannot represent them.
- **richer primitives**: radial/SPHARM 0.93–0.96, but star-convex (StarDist confirms: "not good for non-convex cells"); union-of-N / metaball 0.90–0.92 and handles concavity/dumbbell.
- **Gaussian-mixture (EM)**: an EM-fit mixture reaches **0.927 at K=3**, reuses `Ellipsoid`, is CPU/label-free. **But automatic K-selection (BIC) is unreliable** (over-segments; sample-count dependent) ⇒ cell-count must be decided by temporal evidence (the ILP), not per-frame.

**Net:** one representation — a Gaussian-mixture cell that *emits* nested K=1/2/3 hypotheses with the ILP *selecting* — unifies shape fidelity, the Ultrack nested-contour requirement, the split mechanism, and the ILP hypothesis generator. Label-free, CPU, incremental. This is a CellUniverse-native realization of the **Ultrack** pattern (multi-hypothesis pool → temporal ILP — itself already label-free/training-free); our distinctive ingredient is a *generative, temporally-primed* model-fit source rather than per-frame discriminative segmenters, and we may reuse Ultrack's solver rather than rebuild it (see §2.2). Learned methods (CELLECT/Cellpose) own the raw leaderboard; the no-label/no-GPU/physics-interpretable lane is ours.

## 1. Goal & success criteria

A 3D cell shape model that (a) represents non-elliptical / concave / HeLa-like cells and (b) reliably fits — **"both equally":** lift N3DH-CE SEG *and* generalize. Success:

- SEG on the 5 GT slices improves over the ellipse baseline with **no** regression in splits/positions/sizes vs `best22`.
- The windowed ILP (fed real hypotheses) fixes ≥ a few known division-timing errors greedy missed, with no new spurious splits.
- Feature flags off ⇒ pipeline is byte-for-byte today's.
- Runs label-free, CPU-only, no learned model.

## 2. Architecture overview

Two hypothesis generators feed one arbiter:

```text
 top-down: tracked cells --PCA--> EM Gaussian-mixture fit --> {1-cell, 2-cell, 3-cell} hypotheses ┐
                                                                                                   ├─> windowed ILP ─> joint segmentation + lineage ─> (Chan-Vese eval-mask) ─> checkpoint/render
 bottom-up: raw signal --CellLumen blob/seed detection --------> {appearance, division, rescue} ───┘
```

The ILP does **joint** segmentation-selection *and* tracking over a sliding window `[t-W..t+W]` (start W=1 ⇒ t-1,t,t+1): it selects the set of segments **and** their links (parent→child / appear / disappear / divide) maximizing image-fit + temporal consistency, commits the center frame, slides forward. Segmentation and tracking are co-determined — the "most promising segmentation" is the one that also tracks consistently.

### 2.1 Full pipeline — step by step

Setup (once): **frame-0 seeding** — CellLumen raw-signal detection auto-seeds the initial cell set (or the initial CSV in dev mode). This is just the ILP's frame-0 "appearance" set.

Per frame `t`, as it enters the sliding window:

1. **Load (+ minimal preprocess)** frame `t` — TIFF → working volume. **Reconsidered (2026-07-02):** the current heavy preprocessing (percentile sigmoid contrast, dim subtraction) was tuned for the *SA cost function*; intensity-weighted EM treats brightness as a **density**, so strong contrast remapping distorts it and is counterproductive. Reduce to a light background estimate / denoise (CellLumen still needs a background level) or none — the exact minimal set is settled in offline validation.
2. **Position calibration** — predict each carried-over cell's position into frame `t` (this is the temporal prior that inits the EM fit). *(unchanged)*
3. **Top-down generation (EM fit + SA configuration search)** — (a) per tracked cell, intensity-weighted EM fits the Gaussian-mixture **jointly: center (mean) + rotation (covariance) + shape** (`K_shape` for fidelity), init from the previous-frame state or CellLumen peaks. **EM subsumes SA's *old* role** (position via mean, rotation via covariance). (b) **SA is repurposed to its genuine role — spatiotemporal configuration search** (§3.7): *propagate* previous-frame cells into `t` (temporal prior) and run **birth / death / split / merge / move** proposals driven by residual image energy, generating *temporally-primed, off-menu* hypotheses no per-frame segmenter would produce (recovers missed cells, proposes divisions).
4. **Hypothesis emission** — emit the **1/2/3-cell** (EM `K`-decomposition) **and SA-proposed** (birth / split / merge / propagated) hypotheses, each with an **image-fit cost** (rendered-occupancy vs intensity residual).
5. **Bottom-up hypothesis generation (CellLumen)** — raw-signal blob/seed detection on frame `t` → emit **appearance / division / rescue** candidates (bright blobs with no tracked cell) with costs. Fills what top-down structurally can't (late/missed splits, new cells).
6. **Hypothesis pool assembly** — merge top-down + bottom-up into `candidates_t.csv`; score **source agreement** (both propose the same segment → confidence bonus); pre-filter obviously-bad candidates (bio-gates, pruning) to bound ILP size. *(log any pruning)*
7. **Temporal link construction** — build links between frame `t` hypotheses and frames `t-1 .. t+1` hypotheses, weighted by overlap/IoU.
8. **Windowed ILP solve** — once `[t-1, t, t+1]` is populated, CBC selects the globally consistent set: which hypotheses (segmentation) **and** which links + events appear/disappear/divide (tracking), maximizing image-fit + temporal consistency, subject to hypothesis disjointness + flow conservation.
9. **Commit center frame** — selected hypotheses become that frame's committed cells; selected links become lineage edges; update the tracked-cell set carried forward.
10. **Chan-Vese eval-mask refinement** *(optional, gated)* — snap each committed cell's **render mask** to the image edge in a narrow band. Output/eval only; the parametric metaball state is unchanged.
11. **Checkpoint + render** — write committed frame state to `checkpoints/frame_NNN.txt`; emit synth/real renders + logs. *(unchanged interface)*
12. **Slide window** forward by 1 → repeat from step 1 for the next incoming frame.

Cross-cutting: **online calibration (D2)** — estimate median cell size / brightness / count from the early frames and feed them as the data-derived scales steps 3–6 need; update through the run.

After the last frame:

13. **Lineage tree** construction + render, assembled from the ILP-selected links.

```text
 setup:  frame-0 CellLumen auto-seed ─┐
                                       v
 ┌──────────────────────── per incoming frame t ─────────────────────────┐
 │ 1 load+preprocess → 2 position calib                                   │
 │           │                                                            │
 │           ├─ 3 EM fit + SA moves(birth/death/split/merge) ─ 4 emit    │  top-down
 │           └─ 5 CellLumen raw detect ─────────────── emit app/div/rescue│  bottom-up
 │                          │                                             │
 │                 6 pool + agreement + prune  →  7 temporal links        │
 └──────────────────────────────────┬────────────────────────────────────┘
                                     v   (window [t-1, t, t+1])
                         8 WINDOWED ILP  (joint seg + track, CBC)
                                     v
                    9 commit center frame  →  10 Chan-Vese eval mask (opt)
                                     v
                    11 checkpoint + render  →  12 slide window +1
                                     v  (after last frame)
                         13 lineage tree build + render
```

### 2.2 Relationship to Ultrack (and where SA takes a different route)

**What Ultrack actually does (verified against the paper, PMC11398427):** hypothesis generation is **per-frame independent** — candidates come from multiple *non-trained* segmentations / parameter sweeps (γ=0.1…1.0) / channels merged into one ultrametric contour map; it is **not** temporally-primed. Temporal information enters in only two places, **neither of which generates segments**: (1) the ILP linking, and (2) an *optional optical-flow* step that warps candidate **centroids** toward neighboring frames to improve *association scoring* ("warps associations, not generates new segmentations"). Crucially, **the ILP selects from a FIXED pre-generated candidate set and can never create a new segment.**

**The opening.** Ultrack's hard limit is that frozen, per-frame-independent menu: a cell no segmenter proposed can never appear, and no segment is ever born from temporal continuity. That is exactly what **SA** can do that a fixed-menu ILP structurally cannot.

**SA's different route — temporally-primed, off-menu generation.** A spatiotemporal Marked-Point-Process / RJMCMC search (§3.7) that *propagates* previous-frame models into frame `t` and proposes **birth / death / split / merge / move** driven by residual image energy — creating segments that (a) are born from temporal continuity and (b) were in no per-frame menu (recovering fully-missed cells). This does **not** contradict the earlier "don't replace ILP *selection* with SA" finding (ILP is exact for selection over fixed hypotheses); it is a *different* use of SA — **generating** the hypotheses the ILP could never contain.

Two compositions:

- **Hybrid (recommended):** SA *generates* the temporally-primed + off-menu candidates; the **ILP selects** exactly among the enriched pool. SA supplies what Ultrack's menu lacks; the ILP keeps exact puzzle-solving.
- **Pure-SA (Plan B, §9):** SA does the whole joint problem — generation + selection + linking — as one annealed energy minimization, no ILP. Maximally different from Ultrack; approximate; needs careful move/schedule design.

| | Ultrack | This design (hybrid) |
|---|---|---|
| Hypothesis menu | fixed, per-frame-independent | **SA-generated: temporally-primed + unbounded** (birth recovers missed cells) |
| Temporal info in generation | none (flow only warps centroids for *matching*) | **yes** — models propagated frame→frame before refinement |
| Selection | ILP over fixed menu | ILP over the SA-enriched menu (or pure-SA) |
| Sources | multiple non-trained segmenters | EM metaball + SA moves + CellLumen (+ optional classical segmenters) |

**One line:** Ultrack = *exact selection over a frozen, per-frame-independent menu*; ours = *SA-driven, temporally-primed, off-menu generation (recovering what no menu contains), handed to the ILP for exact selection.*

Two things we still adopt from Ultrack: **(a) multi-method diversity** — keep several diverse label-free generators in the pool (metaball at multiple `K`/thresholds, CellLumen, optionally classical watershed); **(b) reuse-vs-rebuild the solver** — we may emit hypotheses in Ultrack's format and use its ILP rather than rebuild it (fork for the implementation plan).

## 3. Components

### 3.1 `MetaballCell` (the model)

A cell = `N` (1–3) anisotropic Gaussian components, each mean+covariance = a rotated `Ellipsoid` (center, 3 angles, 3 scales) → reuses existing rasterization/rotation. Occupancy = level set of the summed field `Σ wᵢ·Gᵢ(x) ≥ T`. **N=1 with T calibrated to the current 95th-percentile radius ≡ today's single ellipsoid** (zero behavior change off). Owns representation + rasterization + threshold only; not fitting or selection. Two distinct K's, never conflated:
- `K_shape` = Gaussians to render *one cell's* irregular outline (fidelity), bounded by a residual-improvement threshold.
- `cell-count` = 1 cell vs 2 — the division question, decided by the ILP, never per-frame BIC.

### 3.2 EM fitting pass (top-down generator)

Slots in where the PCA+SA shape/position fit is today, on the same Voronoi-filtered bright-voxel set. **Intensity-weighted EM** (brightness ∝ density → targets the 0b fit-gap) fits center+covariance+shape jointly, deterministic init from the temporal prior (or CellLumen seeds in crowded frames). **This subsumes PCA (shape) and SA's *old* position/rotation search.** SA is not dropped but **repurposed** to spatiotemporal configuration search (§3.7) — its genuine role in this design. Augment-not-replace: committed single-cell shape stays calibrated to the current ellipse; EM adds components for fidelity and emits the K=2 region decomposition as the data-driven split candidate (replaces the shortest-axis two-ellipsoid seeding in `trySplitCellPhased`; bio-gates remain as a cheap pre-filter).

### 3.3 CellLumen (bottom-up generator)

CellLumen's raw-signal blob/seed detection (already emits a candidate graph) feeds the **same** ILP hypothesis pool. Roles:
1. **Fills the 0a gap** — generates the missing division/appearance 2-cell hypotheses the top-down model structurally can't invent.
2. **Seeds EM** in crowded late frames (its design target) where PCA-axis init fails.
3. **Unifies appearances / rescue / frame-0 auto-seed** as ILP "appear" hypotheses validated by temporal support.
4. **Agreement = confidence** — when top-down and bottom-up independently propose the same segment, the ILP objective rewards it.

### 3.4 Hypothesis emission → sidecar

Per region, emit competing hypotheses (H1=1 cell, H2=2 cells, opt H3) each with an **image-fit cost** (rendered-occupancy vs intensity residual — the 0a score). Written to `candidates_NNN.csv`: `frame, region_id, hypothesis_id, source(model|lumen), components, fit_cost`. This is the CellUniverse analog of Ultrack's UCM hierarchy.

### 3.5 Chan-Vese refinement (output-side, folded in)

Region-based level set (no edges needed — ideal for fuzzy fluorescence), initialized at the committed metaball surface, run a few iterations in a **narrow band** so it snaps to the true image edge without ballooning/leaking into neighbors. **Refines only the rendered/eval mask that SEG scores; the parametric metaball stays the tracked state** (perturbation, checkpoint, temporal prior) — reusing the existing tracked-shape-vs-eval-mask decoupling (`gotchas.md` birth-mask/bounded-reference). Low blast radius; closes the 0b fit-gap on output.

### 3.6 Windowed ILP (arbiter)

The 0a core. Consumes the sidecar (both sources), selects hypotheses + links over `[t-W..t+W]`, commits center, slides by 1. Objective = Σ fit-weight·select + Σ link-weight·link − event penalties; constraints = hypothesis disjointness + flow conservation. Output = joint segmentation + lineage. **Solver option (§2.2):** build in C++ (Coin-OR CBC) *or* emit hypotheses in Ultrack's format and reuse its hierarchy+ILP — decided in the implementation plan.

### 3.7 SA move set (spatiotemporal hypothesis generator)

SA's role here is **configuration search**, not shape/position refinement (EM does that). A Marked-Point-Process / RJMCMC move set proposes changes to the cell configuration over the window, each scored by the global energy (image likelihood + temporal continuity + division/appearance priors). This is what makes our hypothesis menu **temporally-primed and unbounded** — the genuine departure from Ultrack's frozen per-frame menu (§2.2).

| Move | What it proposes | Fills the gap |
|---|---|---|
| **propagate** | carry a previous-frame metaball into `t`, refine to the image | the temporally-primed hypothesis no per-frame menu contains |
| **birth** | instantiate a cell where residual (unexplained) brightness is high | recovers a fully-missed cell; frame-0 seed; mid-run appearance |
| **death** | remove a cell explaining no signal | disappearance / spurious suppression |
| **split** | divide a cell into two (seeded by the EM `K=2` decomposition) | division hypothesis (esp. the late splits 0a couldn't fix) |
| **merge** | fuse two cells into one | over-segmentation fix |
| **move** | local position/rotation nudge | residual refinement if EM alone is insufficient |

Two operating modes: **(a) generator** — SA proposes, the ILP selects (recommended hybrid); **(b) pure-SA** — SA also accepts/commits via annealing on the global energy, no ILP (Plan B, §9). Config: `simulation.sa_moves_enabled` (default off), per-move enable + proposal rates, `simulation.sa_mode` (`generator` | `pure`).

## 4. Configuration (all gated; default off ⇒ current pipeline)

| Key | Default | Effect |
|---|---|---|
| `cell.metaball_enabled` | false | Gaussian-mixture model; false = single-ellipsoid PCA exactly |
| `cell.metaball_max_components` | 1 | K_shape cap |
| `cell.metaball_level_set_threshold` | (data-derived) | calibrated so K=1 ≡ percentile ellipse |
| `probability.emit_split_hypotheses` | false | write candidate sidecar |
| `simulation.chan_vese_refine` | false | eval-mask boundary polish (+band/iters) |
| `simulation.ilp_enabled` | false | windowed ILP selection |
| `simulation.ilp_window_W` | 1 | window half-width (t-1,t,t+1) |
| `simulation.sa_moves_enabled` | false | SA configuration moves — birth/death/split/merge/propagate (§3.7) |
| `simulation.sa_mode` | generator | `generator` (SA proposes, ILP selects) or `pure` (SA commits, no ILP) |
| `cell_lumen.emit_hypotheses_to_ilp` | false | CellLumen candidates into the pool |

Prefer **data-derived** params (percentiles, fraction of estimated median cell size) over absolute constants (C2/C3 from the plan) — the goal is a new dataset running with ~no retuning.

## 5. Validation

**Offline first (continue the 0-phase discipline):**
- EM-GMM on the *real image* (not clean masks) → does it reach the 0.93 ceiling? Measures the fit-gap and the Chan-Vese benefit.
- Re-run the 0a ILP on the late-split cases (f4/f7/f18) **fed by CellLumen hypotheses** → the decisive "does bottom-up + ILP fix what greedy/top-down couldn't" test.
- **SA move-set value (the SA route, §3.7):** does an SA **birth/propagate** pass recover the late-split/missed cells (f4/f7/f18) that top-down EM alone misses — the off-menu, temporally-primed hypotheses Ultrack's fixed per-frame menu structurally can't contain? This is the decisive test that SA generation adds what the ILP-over-fixed-menu cannot.
- **Preprocessing level:** compare EM fit quality on raw vs light-denoise vs full-sigmoid volumes → fixes the minimal step-1 preprocessing (or confirms none).

**In-C++ incremental (per `rules/workflow.md`):** build → run ~22 frames → diff vs `best22` on splits/positions/sizes/**shapes** + SEG on the 5 slices (frames 21/28/78/141/162) + CTC DET/TRA/SEG vs `01_GT`. **Feature-off regression check must match today byte-for-byte.** Stop-and-analyze on any regression.

## 6. Risks & mitigations

| Risk | Mitigation |
|---|---|
| EM convergence in crowded frames | small K cap; CellLumen seeding; deterministic init |
| Over-segmentation (BIC unreliable) | cell-count is the ILP's job, not per-frame; conservative default K |
| ILP instance size on 100+-cell frames | small W; candidate pruning / SA warm-start; log any pruning |
| Chan-Vese leaking into neighbors | narrow band + neighbor-aware constraint |
| Regressing working single-cell fits | K=1 calibration + flag-off identity |
| Scope (this is large) | strict incremental rollout, each step gated + tested vs best22 |

## 7. Rollout order (each step: build → run 22f → compare)

1. **Offline validation** — EM-GMM on real image; CellLumen-fed ILP on late-split cases. Gate: bottom-up + ILP fixes the 0a gap.
2. **`MetaballCell` + K=1 calibration** — zero-regression scaffold (flag off = identical).
3. **EM `K_shape` single-cell shape fidelity** — SEG on 5 slices, no tracking regression.
4. **Chan-Vese eval-mask polish** — SEG closure.
5. **Hypothesis emission (metaball K=2 + CellLumen) → sidecar** — additive, committed path unchanged.
6. **Windowed ILP W=1 selection** — the big one; TRA/division-timing vs greedy, no new spurious splits.
7. **Data-derived params / online calibration (D2) + HeLa/Tribolium generalization.**

## 8. Decision gates (proceed only if)

| After | Proceed only if |
|---|---|
| 1 offline | EM-GMM reaches ~0.92 on real image AND CellLumen-fed ILP fixes ≥ a few late-split cases |
| 2 scaffold | flag-off output byte-identical to today |
| 3 K_shape | SEG improves on the 5 slices, no split/position regression vs best22 |
| 4 Chan-Vese | further SEG gain, no neighbor leakage |
| 5 emission | sidecar correct, committed output unchanged when ILP off |
| 6 ILP | TRA/division-timing improves vs greedy, no new spurious splits |
| 7 generalize | a second dataset runs with no per-dataset retuning |

## 9. Out of scope / fallbacks (Plan B)

- If EM-GMM on real images doesn't reach ceiling → limiter is fit/placement; pivot to the fit-quality pass before more shape work.
- If smooth Gaussian tails underfit crisp boundaries → swap per-cell primitive to SPHARM/radial (same emit→select architecture).
- If parametric fitting can't reach the shape on hard frames → **slice-stack segmentation** (2D per-slice → stack → smooth; = the Cellpose-3D pattern, label-free version). Paradigm shift to segment-first; last resort, documented not chosen.
- If per-cell over-segmentation is untameable → restrict the mixture to split candidates only (default K=1 committed; K≥2 emitted only as division hypotheses).
- **Pure-SA variant (§2.2):** if the hybrid's ILP proves too heavy (instance size on 100+-cell frames) or the fixed-menu selection too limiting, SA does the whole joint generation + selection + linking as one annealed energy minimization — no ILP. Maximally different from Ultrack; approximate; the move-set + annealing-schedule design is the risk.

Each fallback is a localized swap; the emit-hierarchy → ILP-select architecture is fixed regardless of which primitive fills the "blob" slot.
