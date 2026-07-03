# Build Plan — Richer Shapes + Windowed ILP Temporal Consistency

**Date:** 2026-07-02
**Depends on:** `docs/plans/2026-07-02-post-ultrack-directions.md` (research + rationale).
**Goal:** Close the SEG gap (irregular / dividing nuclei) AND resolve per-frame "1 cell vs 2 cells" ambiguity using multi-frame temporal consistency — while staying **CPU, no-GPU, no-label**.

## The two workstreams and why they converge

- **Workstream S (Shape):** replace/augment the triaxial ellipsoid so a cell (esp. a dividing one) can be modeled accurately per slice.
- **Workstream L (Linking):** add a **windowed ILP** that selects, across a sliding window of frames, the globally consistent set of cell hypotheses — so an ambiguous frame is decided by its neighbors.

They converge at **hypothesis emission**: the windowed ILP can only pick "2 cells" at frame *t* if Workstream S actually *produced* a 2-cell shape hypothesis there. So the metaball/split shape model (S) is what feeds the ILP (L). Build them to meet in the middle.

## Guiding principles

1. **Validate before building.** Two cheap offline experiments (Phase 0) decide go/no-go before any large C++ change.
2. **Everything behind a config flag**, default = current behavior. Exponent=2 → ellipsoid; ILP disabled → current per-frame greedy.
3. **Every C++ phase:** build → run ~22 frames → compare to `best22` baseline on splits / positions / sizes / shapes / SEG (per `rules/workflow.md`). Stop-and-analyze on regression.
4. **SA does continuous (shape); ILP (CBC, exact) does discrete (selection).** Never make SA solve the ILP.

## Hard constraints: label-free, generalizable, self-calibrating (added 2026-07-02)

The Cell Tracking Challenge forbids **any** form of manual label at inference — **including the seed `initial.csv`** (which is a hand-/GT-derived label). This makes three things first-class requirements, not nice-to-haves:

- **C1 — No `initial.csv`.** The first-frame cell set must be **detected automatically from raw signal** (Workstream D). The same detector handles new-cell *appearance* mid-run — it is the ILP `appearance` event, not a special case.
- **C2 — Generalize across datasets with few knobs.** Prefer **data-derived** parameters (percentiles, relative scales, estimated cell size/count) over absolute constants. The ILP's overlap/IoU weights are scale-free — fewer magic numbers than the current split-gate stack. Target: a new dataset runs with ~no per-dataset tuning.
- **C3 — Auto-calibrate through the run.** Estimate characteristic cell size / brightness / count from early frames and adapt online; use ILP selection confidence as a recalibration signal. Push knobs out of config into estimation.

These are the throughline: **move parameters from config into data-derived estimation.** Every phase below should be evaluated against "does this add a dataset-specific knob?" — if yes, prefer a data-derived alternative.

---

## Phase 0 — Cheap validation experiments (do FIRST, decides everything)

These are offline, no C++ change, and tell us whether the whole direction pays off.

### 0a. ILP validation on EXISTING outputs (Workstream L go/no-go)

- **Input:** an existing run CSV (e.g. `outputs/Yiding_1~171_VISUAL_TIF/resume_194/merged_0-194.csv`, columns `file,name,x,y,z,aRadius,bRadius,cRadius,...`).
- **Build a Python `W=1` windowed ILP prototype** (`scripts/ilp_proto/`): for each known-hard ambiguous cell, form the 1-cell and 2-cell hypotheses (from the existing splits/merges + a synthetic merged blob), solve the windowed selection ILP (variables `y_p` select, `x_pq` link, event vars; objective = overlap weight + event penalties; constraints = disjointness + flow conservation) with **CBC via PuLP**.
- **Metric:** does the 3-frame ILP flip the *known* division-timing / merge errors (from the GT `man_track.txt`) that the per-frame greedy logic got wrong?
- **Tooling:** `.venv-eval/bin/pip install pulp` (bundles a CBC binary — CPU, license-free).
- **Gate:** if a `W=1` window fixes ≥ a few known hard cases → proceed to Workstream L. If it changes nothing → the temporal-ILP direction is not worth the build; stop here.

### 0b. Shape ceiling measurement (Workstream S go/no-go)

- Offline, on the 5 CTC SEG-annotated slices (frames 21/28/78/141/162): fit a **superquadric** and a **union-of-2-ellipsoids** to each GT nucleus (or to raw signal) and compute the best-achievable Jaccard.
- **Gate:** confirm superquadric lifts the per-slice SEG ceiling meaningfully above the ellipsoid's, and that union-of-2 captures the dumbbell/dividing nuclei. If the ceiling barely moves, don't build the shape change.

---

## Workstream D — Label-free detection & auto-calibration (CTC-compliance prerequisite)

### Phase D1 — Auto first-frame segmentation (replace `initial.csv`) — **BACK BURNER**

> **Deferred (2026-07-02).** Required for a real CTC submission, but NOT on the near-term critical path. Keep the `initial.csv` path for dev iteration; pick D1 up before an actual submission.

- **Goal:** produce the initial cell set at frame 0 from raw signal alone — no seed CSV.
- **Chosen approach when resumed:** reuse **`CellLumen` in its ground-truth-builder mode, applied to frame 0 ONLY**, to auto-seed the initial cells. CellLumen already does raw-signal blob/seed detection for the rescue path and for GT building — running it once on frame 0 emits the initial cell set with no label. (Only frame 0 needs it; subsequent new cells are the ILP `appearance` event.)
- **Principled upgrade (optional, later):** a **Marked Point Process** birth/death SA (RJMCMC) detects an a-priori-unknown number of nuclei from intensity — the published label-free version. Consider only if CellLumen's frame-0 recall proves dataset-fragile.
- **Insertion:** `src/main.cpp` initial-cell load path + `CellFactory` — add an "auto-detect frame 0" mode that bypasses the CSV and calls the CellLumen frame-0 seeder (`src/CellLumen.cpp`).
- **Config:** `simulation.auto_seed_first_frame` (default false → current CSV path unchanged).
- **Effort:** medium. **Risk:** medium-high — frame-0 detection quality sets the whole run; mitigated because the windowed ILP + appearance/disappearance events (L2) can recover missed/spurious initial cells over the first few frames.

### Phase D2 — Data-derived parameters & online calibration (C2 + C3)

- Audit the config for **absolute constants** that should be **data-derived**: intensity thresholds → percentiles; size gates → fraction of estimated median cell size; split constants → relative to the fitted shape distribution.
- **Online estimation:** from the first N frames estimate median cell radius, brightness distribution, and typical count; feed these as the priors/scales the later stages need, and update them as the run proceeds.
- **Effort:** medium, incremental (can be done knob-by-knob). **Risk:** low. **Payoff:** a new dataset runs with little/no tuning (C2), and the run self-adjusts (C3).

## Workstream S — Shape

### Phase S1 — Superquadric (Tier 1, cheapest SEG win)

- **Model change:** ellipsoid inside-test `(x/a)²+(y/b)²+(z/c)² ≤ 1` → superquadric `|x/a|^r + |y/b|^r + |z/c|^r ≤ 1`.
- **Insertion points (confirmed):**
  - `src/Ellipsoid.cpp:631` `isPointInsideEllipsoid` — add exponent to the implicit test.
  - `src/Ellipsoid.cpp:12` `scanEllipsoidSlice` + `:429` `drawWithRotation` — same exponent in the rasterizer.
  - `includes/Ellipsoid.hpp` `EllipsoidParams` — add `float shapeExponent = 2.0f;` (2.0 = exact ellipsoid → zero behavior change when off).
  - `src/Frame.cpp:2522` `calibrateCellShapeViaPca` — PCA still gives axes/center; **SA fits the exponent** (add a perturb param, or a small grid search 1.6–3.0).
- **Config:** `cell.shape_exponent_enabled` (default false), perturb block for the exponent.
- **Effort:** small. **Risk:** low (superset of ellipsoid). **Verify:** SEG on the 5 slices + shape/size vs baseline.

### Phase S2 — Union-of-2-ellipsoids / metaball (Tier 1 item 2; the "2-cell hypothesis" generator)

- **Model:** a dividing/ambiguous cell = smooth implicit union of two sub-ellipsoids (blend field ≥ threshold). ~+7 params (second blob: center offset + radii + blend).
- **Insertion:** new candidate shape used inside the split pipeline `src/Frame.cpp:3148` `trySplitCellPhased` / `bioCheckDaughters (:2119)` — the split candidate becomes a real 2-blob fit, not just two seeded ellipsoids. SA fits the 2-blob params to the blob.
- **Why now:** this is the shape that *produces the two-cell hypothesis* the windowed ILP will select. It directly targets both SEG (dumbbell shape) and division geometry.
- **Effort:** medium. **Risk:** medium (more DOF → watch SA convergence; keep the metaball path split-only, not global).

---

## Workstream L — Windowed ILP

### Phase L1 — Hypothesis emission (the S→L bridge)

- **Change:** for ambiguous cells, `Frame` **emits both** the 1-cell fit (S1) and the 2-cell fit (S2) with their fit costs, instead of greedily committing in `trySplitCellPhased`.
- **Output:** a per-frame candidate sidecar (extend the CSV writer at `src/CellUniverse.cpp:10648`, or a parallel `candidates_NNN.csv`): `frame, candidate_id, parent_cell, shape_params, fit_cost`. This is the CellUniverse analog of Ultrack's UCM hierarchy.
- **Config:** `probability.emit_split_hypotheses` (default false → current committing path unchanged).
- **Effort:** medium. **Risk:** low (additive; doesn't change committed output when off).

### Phase L2 — Windowed CBC ILP selection (Tier 2)

- **First as a Python post-process** (reuse the Phase-0a prototype, now fed by real emitted candidates), THEN port to C++ once validated.
- **ILP:** window `[t-W .. t+W]`; variables `y_p` (select), `x_{p,q}` (link), `x_α/x_β/x_δ` (appear/disappear/divide); objective `max Σ w_{p,q}x_{p,q} + Σ event penalties`; constraints = disjointness `y_p+y_q≤1` for conflicting hypotheses + flow conservation. Solve with CBC, commit center frame(s), **slide by 1**.
- **Knobs:** window `W` (start 1), candidate pruning (SA warm-start / drop obviously-bad segments to bound instance size on dense frames).
- **C++ port (later):** Coin-OR CBC C++ API, or a lightweight MWIS local-search if instances stay small. Keep it exact where feasible.
- **Effort:** large. **Risk:** medium-high (instance size on 100+-cell frames — mitigate with W and pruning).

### Phase L3 — (Optional) Alternating ILP↔SA loop (Tier 2 deeper)

- Couple: ILP selects → SA refines selected shapes given the selection → re-solve, until stable. Only if L2 shows clear value over the one-shot pipeline.

---

## Ordering / dependency graph

```text
Phase 0a (ILP proto) ─┐                          ┌─ L1 emit ─ L2 windowed ILP ─ (L3 alt-loop)
                      ├─ decide ─ S1 superquadric ┤
Phase 0b (shape ceil)─┘                          └─ S2 metaball ───────────────┘ (feeds L1)

D1 auto-seed (CTC prerequisite) ── validated alongside ── D2 data-derived params (cross-cutting, every phase)
```

- **0a and 0b run in parallel, first.** They are the go/no-go gates for L and S.
- **D1 (auto-seed) is required for any real CTC submission** — build it in parallel; it reuses CellLumen and de-risks by leaning on L2's appearance/disappearance recovery. Until D1 lands, keep the CSV path for dev iteration only (not submission).
- **D2 is cross-cutting** — apply the "data-derived not absolute" test in every phase; don't add dataset-specific knobs.
- **S1 is independent** and low-risk — build once 0b passes, in parallel with the ILP track.
- **S2 must precede/accompany L1** (S2 makes the two-cell hypothesis real).
- **L2 depends on L1.** L3 is optional and last.

## Decision gates

| After | Proceed only if | Result (2026-07-02) |
|-------|-----------------|---------------------|
| Phase 0a | `W=1` ILP fixes known hard 1-vs-2 cases the greedy logic missed | **INCONCLUSIVE** — ILP core validated (self-test), but on real committed data W=1–3 never overrode the per-frame image call; the real errors are *late splits* needing competing hypotheses that don't exist until S2/L1. Not a green light for L2 alone. `scripts/ilp_proto/RESULTS.md` |
| Phase 0b | superquadric/metaball raises the per-slice SEG ceiling meaningfully | **S1 FAILED / S2 targeted** — ellipse ceiling already 0.887; superellipse +0.005 (median exponent 2.00 → don't build S1); union-of-2 +0.022 overall but +0.08–0.145 on ~8% dividing nuclei. Big finding: 0.89 ceiling vs 0.358 actual ⇒ SEG loss is a **fit/placement gap, not a primitive gap**. `scripts/ilp_proto/RESULTS_phase0b.md` |
| Phase D1 | auto-detected frame-0 cells match GT count/positions within tolerance on ≥2 datasets (no CSV) | back burner |
| Phase D2 | the same config runs a second dataset (e.g. Tribolium) with no per-dataset retuning | not started |
| Phase S1 | SEG improves on the 5 slices with no split/position regression vs best22 | **DROPPED** — 0b ceiling gain negligible |
| Phase L2 | windowed ILP improves TRA / division-timing vs per-frame greedy, no new spurious splits | gated on S2/L1 producing real hypotheses |

### Phase 0 outcome & revised direction (2026-07-02)

Both cheap gates are done. Combined steer:

1. **Drop S1 (superquadric)** — nuclei are genuinely elliptical; ceiling gain +0.005.
2. **S2 (union-of-2 / metaball)** keeps only a *modest, dual* justification: the ~8% dividing nuclei **and** the ILP's 2-cell hypothesis generator. Not a standalone win.
3. **The dominant SEG lever is closing the ellipse-fit gap (0.89 ceiling → 0.358 actual), not shape richness.** Next diagnostic (before any shape/ILP build): measure the tracker's *actual* per-nucleus Jaccard on the 5 SEG slices to localize the gap (fit vs placement vs missed/matching).

## Stance compatibility

- ✅ Compatible: superquadric, metaball, hypothesis emission, CBC ILP (all CPU, label-free).
- ❌ Avoid: level-set/deformable shapes (too many DOF for SA); learned segmentation front-end (breaks no-label); SA replacing the ILP solver (evidence: underperforms MILP).

## Tooling

- ILP prototype: `.venv-eval/bin/pip install pulp` (bundled CBC).
- Later in-C++ ILP: Coin-OR CBC (open-source) or lightweight MWIS heuristic.
- No new GPU / no learned-model dependency at any phase.

## Open questions carried from research

- Real DOF-vs-SA-convergence cost per shape (superquadric/metaball) — measure empirically in S1/S2.
- ILP instance size on dense late-embryo frames (100+ cells) — measure in L2; may force pruning or smaller W.
- Whether global windowed ILP actually beats per-frame greedy on Fluo-N3DH-CE — answered by Phase 0a, then L2.
