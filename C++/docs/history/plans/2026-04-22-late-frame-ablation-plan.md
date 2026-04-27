# Late-Frame Ablation Plan — 2026-04-22

**Branch:** `jl_voronoi_bleed_penalty_04212026`
**Goal:** Measure, in isolation, the effect of each candidate fix for the late-frame
drift + false-split + cell-bloat patterns observed past f60.

## Problem summary

Post-f60 runs exhibit three interacting failure modes:

1. **Drift into empty space** — cells abandon their bright region, float into
   dark areas. No cost pulls them back (no neighbor → no bleed; synth=0 real=0
   residual=0). Example: e3d03 jumps ~38 vx between f64 and f65; another cell
   ends up above the cluster at f71 with no bright signal beneath it.
2. **Cell bloat below the 1.8× birth cap** — e3d03 reaches aR=44.9 (birth ~30),
   a5100 reaches aR=51 (birth ~28). PCA axes drift, split candidates miscalibrate,
   daughters either stick inside parent (cost-rejected) or overshoot to neighbor
   bright regions (false-split misattribution).
3. **Cost-gate lockout for genuinely-dividing cells** — `23001` tried to split
   at f66 with clean drifts (3.3/4.2) and passed the bio gate easily, but the
   proportional cost threshold (`split_cost_fraction × baselineImageCost`)
   scales up as the parent's bad-fit inflates baseline, creating a perverse
   loop where the more visibly-dividing the cell becomes, the harder the gate
   rejects its split.

## Candidate fixes

### Fix C — Continuous birth-growth cap (supersedes the 1.8× gated cap)

Log-barrier penalty on `aR / birthR`:
- < 1.1× birth: no penalty
- 1.1× – 1.3× birth: quadratic ramp up to modest penalty
- 1.3× – 1.6× birth: steep penalty, most perturbs get rejected
- > 1.6× birth: hard reject

Replaces the current gated `(aR ≥ 1.8 × birth) AND (elong ≥ 1.8)` clamp. No
threshold cliff, no elongation gate coupling.

Primary target: e3d03 bloat to aR=44.9 would be resisted from aR=33 onward;
a5100 bloat to aR=51 never happens — split happens earlier with cleaner axes.

### Fix A — Brightness-centroid anchor prior

Per frame, per cell: compute
`centroid_brightness(cell) = Σ(real[p] × p) / Σ(real[p])`
over voxels inside cell's snap-Voronoi territory, weighted by real-image brightness.

Add `weight × ‖cell.pos − centroid_brightness‖²` to the perturbation cost.

Closes the "empty-space drift has no penalty" gap. If a cell drifts into an
unoccupied dark region its centroid pulls it toward the nearest bright signal.

### Fix B — Multi-frame snap median

Current: snap = last frame's optimized position.
Proposed: snap = component-wise median of last K optimized positions (K=5).

Makes single-frame drift transient: a 38-vx jump at f65 doesn't become the new
anchor — the median still reflects f60-f64 values.

### Fix D — Split persistence requirement

A split candidate must clear both bio and cost gates at nearly-matching daughter
positions across N consecutive frames (e.g., N=2, daughters within 10 vx
inter-frame) before committing.

Targets opportunistic one-frame accepts (e3d03 at f67 with drifts 18/33 after
being close-to-threshold at f66 — persistence would catch the drift mismatch
between frames and reject).

## Methodology

### Pre-work

1. **RNG seed config (done 2026-04-22)**: `mc_rng_seed: int` in
   `ConfigTypes.hpp`, seeded in `CellUniverse::optimize`. Value `0` keeps
   production random_device behavior; values `>0` give deterministic main-RNG
   trajectories. Thread-local RNGs (perturb samplers, signal-guided) still use
   random_device — expected <2% residual variance.

2. **Comparison script (C++/scripts/compare_ablation.py)**: takes two output
   directories, emits a delta report with:
   - Per-frame max/mean `aRadius` and count of cells with `aR > 1.3 × birth`
   - Per-cell drift magnitude between consecutive frames
   - Split-accept list diff (added, removed, frame-shifted)
   - Splits with refine drifts > 15 vx (geometric-ugliness indicator)
   - Bio-reject count by reason (buried_in, bridging_to, bridge_flat, edge_too_dim)

3. **Controlled starting point**: resume from frame 60 of
   `output_jihang_20260422_004745/checkpoints/frame_059.txt`. This is the last
   checkpoint BEFORE the preprocessing anomaly window (f62/f64). Each ablation
   test resumes from this exact state.

### Test schedule

All tests: `resume_from=60`, `last_frame=72`, `mc_rng_seed=42`.

| Test | Config on top of current | Primary effect measured | Decision threshold |
|---|---|---|---|
| **T0** | current code unchanged | baseline metrics | reference |
| **T1** | + Fix C (continuous cap) | max aR bounded; downstream drift reduced | max aR < 1.4× birth on all frames; bad-spot count ≤ T0/2 |
| **T2** | + Fix A (centroid anchor) | drifted cells pulled back | drift > 20 vx count ≤ T0/3 |
| **T3** | + Fix D (split persistence) | fewer ugly-drift splits | splits with drifts > 15 count ≤ T0/3 |
| **T4** | + Fix B (median snap) | frame-to-frame drift variance down | single-frame drift > 15 vx count ≤ T0/2 |
| **T5** | best of T1-T4 combined | all metrics converge | all above thresholds simultaneously |

### Stopping rule

If T1 (Fix C alone) achieves all thresholds, stop. Additional fixes cost
complexity and have risk of interacting with each other; don't stack unless
T1 leaves residual issues.

### Output per test

Each test produces one markdown file in `C++/docs/plans/ablation/`:
- `T0-baseline.md`
- `T1-fix-c-continuous-cap.md`
- `T2-fix-a-centroid-anchor.md` (if run)
- `T3-fix-d-split-persistence.md` (if run)
- `T4-fix-b-median-snap.md` (if run)
- `T5-combo.md` (if run)

Each file records: config diff, output dir, comparison-script report verbatim,
screenshot evidence, verdict (accept / reject / tune further).

## Artifacts

- Starting checkpoint: `outputs/output_jihang_20260422_004745/checkpoints/frame_059.txt`
- Ablation output dirs: `outputs/output_jihang_ablation_T{N}_<timestamp>/`
  (via `output_name_rule=output_ablation_T{N}_{timestamp}` in INI preset)
- Reports: `C++/docs/plans/ablation/T{N}-*.md`
- Comparison script: `C++/scripts/compare_ablation.py`

## Post-ablation

Once winner(s) identified, the full production run returns to `mc_rng_seed=0`
(random_device) for normal multi-seed runs. The seed mechanism stays in place
for future ablation or debugging needs.
