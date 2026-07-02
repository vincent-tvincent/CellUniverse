# Post-Ultrack Future Directions — Research Findings + Roadmap

**Date:** 2026-07-02
**Context:** Maintainer's ideas after reading the Ultrack paper (Bragantini et al., *Nature Methods* 2025 / arXiv 2308.04526). Deep-research pass (23 sources, 24/25 claims adversarially verified, 1 refuted).
**Scope:** Where CellUniverse (C++, no-GPU, no-label; triaxial-ellipsoid + PCA fit + simulated-annealing perturbation + CellLumen rescue; benchmark Fluo-N3DH-CE) should go to close the SEG gap and improve tracking.

---

## Headline conclusion (reframes all three ideas)

> **Two jobs need two optimizers. Simulated annealing is right for *shape fitting* (continuous, generative). It is *wrong* for the *selection* puzzle. Keep them separate — don't make SA do Ultrack's ILP job.**

## Thread 1 — What Ultrack's ILP actually is (well-sourced)

Ultrack's ILP is **jointly a segmentation-selector AND a linker in one integer program** — not a post-hoc linker.

- **Decision variables:** `y_p` (segment p selected), `x_{p,q}` (temporal link p→q), `x_α`/`x_β`/`x_δ` (appear/disappear/divide).
- **Objective:** `max Σ w_{p,q}·x_{p,q} + Σ (w_α·x_α + w_β·x_β + w_δ·x_δ)` — overlap/IoU link weights + event penalties.
- **Key constraint (resolves competing hypotheses):** disjointness `y_p + y_q ≤ 1` for nested p⊂q; each pixel used once; flow-conservation couples selection to linking.
- **Candidate hypotheses:** hierarchical watershed on **ultrametric contour maps** (foreground + boundary-confidence map), *classical / non-learned*, nested multi-scale.
- **Solver:** Gurobi (recommended) or open-source **CBC/COIN-OR**; NP-hard, LP-relax + branch-and-bound.
- Structurally = **weighted set-packing / Maximum-Weight-Independent-Set (MWIS)**.
- **Not DL-dependent** — "competitive results with or without deep learning" → compatible with our no-label/no-GPU stance.

Sources: arxiv.org/abs/2308.04526, royerlab.github.io/ultrack, biorxiv 2024.09.02.610652, nature.com/articles/s41592-025-02778-0.

## Thread 2 — Can SA replace the ILP solver? (mostly no)

- Selection (MWIS/set-packing) **is expressible as QUBO** → SA-attackable in principle.
- **But the one direct empirical comparison found SA loses:** Metropolis/Boltzmann sampling on the 0-1 selection program converged "not very impressive[ly]" vs MILP branch-and-bound (MHT literature, patent US9291708B2 + JAIF survey).
- **Refuted (0/3 votes):** the claim that "exact solvers fail at scale, SA wins." Do **not** treat SA as a free substitute.
- **Verdict:** if we add a selection stage, keep it **exact (CBC — open-source, CPU)**. Reserve SA for shape fitting.

## Thread 3 — Modeling irregular shapes (partially sourced)

Strong label-free, CPU, optimization precedents for our exact paradigm:
- **GOCELL** (Med Image Anal 2018): globally-optimal model-based nucleus segmentation via implicitly-parameterized shapes → convex/init-independent **but only for a single isolated object and only because the shape is an ellipse/ellipsoid** (our exact primitive). Richer shapes + multi-object "collaborating shapes" **lose** the guarantee.
- **Marked Point Processes** (Descombes/Zerubia): detect an *unknown number* of objects via variable-dimension energy minimization solved by **SA with birth/death (RJMCMC)** — already applied to nuclei. Principled version of our CellLumen-rescue + split logic.
- **StarDist star-convex polyhedron** (radial distances along fixed rays) = the "radial polygon per slice" representation; the *representation* is fittable without the neural net.

**Honesty flag:** the per-representation DOF-vs-SA-cost table below is **engineering judgment, not benchmark-verified** (the research could not ground it). Principle (sound but unverified): SA cost grows fast with DOF.

| Representation | Extra DOF vs ellipsoid | SA-friendly | Dumbbell/dividing | Call |
|---|---|---|---|---|
| **Superquadric** (+exponents) | +1–2 | yes | partial | **best first step** |
| **Union-of-2-ellipsoids / metaball** | ~+7 | workable | **direct** | **best for division problem** |
| Star-convex per-slice (StarDist rep) | many | heavy | yes | high SA cost |
| SPHARM surface | many | heavy | yes | overkill for SA |
| Level-set / snakes | ~pixel-count | no | yes | **incompatible with SA** |

## Segment-first vs generative-fit — does it matter?

Structurally yes (Ultrack *selects among pre-made* hypotheses; we *generate-and-fit*), but **no head-to-head benchmark on Fluo-N3DH-CE exists**. Open question: does a **global temporal ILP** beat our **per-frame greedy** split decisions? Unknown → test before rewriting.

---

## Prioritized roadmap (flagged by stance-compatibility)

**Tier 1 — do first (compatible, high value/effort, low risk)**
1. **Superquadric shape upgrade** (+1–2 DOF, SA-friendly, one-line inside-test change). Cheapest SEG win. Backed by GOCELL (implicit-shape model-fitting is a published lever).
2. **Union-of-two-ellipsoids / metaball for splitting cells.** Dividing nucleus = dumbbell; one ellipsoid can't fit it. Targets SEG gap *and* division geometry at modest DOF. Highest-leverage change for our organism.

**Tier 2 — the Ultrack idea, in our stance (compatible)**
3. **Global selection stage, kept EXACT.** SA emits a small hierarchy of candidate per-cell shapes → **CBC-solved ILP/MWIS** selects the temporally-consistent, non-overlapping subset. Grafts Ultrack's global consistency onto our label-free fit. Do **not** use SA for this selection.
4. **MPP-style birth/death for variable cell count.** Reframe CellLumen-rescue + split as principled variable-dimension SA (RJMCMC).

**Tier 3 — longer horizon**
5. **Modify/drop PCA shape-fit** — becomes init-only if we move to superquadric/metaball. Don't drop until the richer model is validated (it's a cheap initializer).
6. Star-convex-per-slice only if superquadric+metaball plateau on SEG.

**Off-stance — avoid**
- Full level-set/deformable (too many DOF for SA).
- Replacing the ILP solver with SA (Thread 2 refutation).
- Learned segmentation front-end (breaks no-label).

## The single most valuable next experiment
Before any rewrite: **run a global-ILP linking step (CBC) over the existing per-frame ellipsoid hypotheses and measure whether TRA / division-timing beats the current per-frame greedy split logic.** Cheap test; no published source answers it for our data.

## Caveats
- Solidly primary-sourced: Ultrack ILP, QUBO/MWIS reduction, GOCELL, Marked Point Processes, StarDist star-convex.
- **Not** verified: the per-shape DOF/SA-cost table (engineering judgment); and "global ILP vs per-frame splits on Fluo-N3DH-CE" (hence experiment #1).
- Solver comparisons (SA vs MILP) shift with Gurobi/CBC releases — treat as directional.
