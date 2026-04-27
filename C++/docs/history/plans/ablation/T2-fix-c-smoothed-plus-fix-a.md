# T2 — Fix C (smoothed) + Fix A (brightness-centroid anchor)

**Output dir:** `outputs/output_ablation_T2_20260422_210422`
**Delta vs T1:**
- Fix C hard-reject smoothed: `1e9 plateau` replaced with `boundary + slope × (ratio − end)`
- Fix A added: `brightness_centroid_anchor_weight: 50.0`
- Centroid computed once per frame from snap-Voronoi map + real image, `[Brightness Centroid]` log
- Same seed=42, same resume from f60

**Run stopped at f67 mid-frame** (user called off for analysis + session-save).

## Per-frame trajectory (through f66)

| Frame | T0 n | T1 n | T2 n | Notes |
|---|---|---|---|---|
| 61 | 30 | 30 | 30 | parity |
| 62 | 30 | 30 | 30 | parity |
| 63 | 30 | 30 | 30 | parity |
| 64 | 30 | 30 | 30 | parity |
| 65 | 35 | 34 | **34** | T1 and T2 match — e3d03 FP suppressed in both |
| 66 | 38 | 38 | **37** | T2 adds f111 (rescued from z=0 drift) + deep sub-splits |
| 67 (partial) | 43 | 42 | 38+ | T2 accepted a5101 (drift 26/13) and e3d03 (**drift1 = 63.7**) before stop |

## Fix C (smoothed) behavior

- `[Bloat Cap]` log confirms every frame with `birth_installed=N/N` coverage.
- Hard-reject zone replaced with `boundary + slope × (ratio - end)` extrapolation. Weight=1000 → boundary = 12000, slope = 100000 per unit-ratio past end. Cells stuck above 1.6× birth now have a pull-back gradient.
- Did NOT observe the bloat-cap-trap pattern from T1 on a5100. Further measurement needed in aggregate on a completed run.

## Fix A (brightness-centroid anchor) observations

- Per-frame centroid build time: 19-43 ms. Negligible overhead.
- **Massively increased perturb accept rate.** T1 f65: 153 accepted; T2 f65: **569 accepted** (3.7×). Cells actively moving toward their image-centroid targets.
- **Rescued f111 from z=0 drift**: in T1 f70 it was at z=0.00 (stuck at boundary). In T2 f66 it split with daughters at z=70 and z=47 (mid-z). Centroid anchor pulled it back before it reached the boundary.
- **Unintended interaction at f67**: e3d03 split accepted with **drift1 = 63.7** and cost −119k. Likely caused by e3d03's snap-Voronoi territory bleeding into neighbor brightness — centroid pointed toward a neighbor's bright region, pulling e3d03's split candidates across the midline. Fix A amplified e3d03's late-frame misattribution risk.

## Interpretation

**Wins:**
- Boundary-drift rescue (f111 at z=0) ✓
- FP suppression at f65 (no e3d03) maintained from T1 ✓
- Bloat cap smoothed (no hard-reject plateau) ✓

**Concerns:**
- Deep-level sub-splits (`a51101` f66 drifts 17.8/18.4; `a51110` f66 drifts 37/28) — likely over-splitting. Centroid anchor may be creating position instability that triggers split attempts at random.
- **e3d03 f67 FP with drift1=63.7 is a regression vs T1** (T1 had no e3d03 FP in f67 at all). Centroid anchor may be causing e3d03's split candidates to reach into neighbor territory.

## Root-cause analysis — why Fix A hurts e3d03

Fix A's design assumes a cell's Voronoi territory is ROUGHLY the bright region the cell biologically owns. For e3d03 that assumption fails: its territory overlaps with dividing neighbors (23101, cb cells). The centroid of this mixed territory is PULLED TOWARD the neighbor's bright signal, because:

- e3d03's own bright region has moderate brightness (~0.4-0.5 mean)
- A dividing neighbor's peaks can be brighter (~0.6-0.8 peak)
- Weighted mean shifts TOWARD the brighter neighbor's peak inside e3d03's territory

Result: centroid-anchor penalty ENCOURAGES e3d03 to move toward its neighbor, which is exactly the misattribution that drives e3d03's FP splits.

## Options for T3 / T2b

**T2b** — Fix A with narrower territory:
- Only count pixels within `radius × some factor` of cell center (not the full Voronoi territory)
- Prevents cells from having centroids dragged by distant bright regions

**T2c** — Fix A only for ISOLATED cells:
- If cell's Voronoi territory is surrounded mostly by background (not other cells), apply centroid pull
- Cells surrounded by neighbors (e3d03's case) skip the pull — rely on bleed penalty + position prior alone

**T3** — Move to Fix D (split persistence requirement):
- e3d03 f65 and f67 split attempts would not match (daughter positions differ hugely)
- Persistence across frames would reject e3d03 anyway
- Doesn't depend on centroid fixes

## Recommendation

**Proceed to T3 (Fix D persistence)** — Fix A has a real bug (centroid drag toward neighbors) that undoes its gains in crowded regions. Fix D directly prevents opportunistic one-frame FPs regardless of centroid issues. If T3 passes with Fix C alone (no A), that's the simpler solution.

## Artifacts

- Logs: `outputs/output_ablation_T2_20260422_210422/debug_log.txt` (partial, up to f67 pre-Optimize-Done)
- Splits accepted in T2: a5111, a5110, a5010, a5011 (f65), a51101, f111, a51110 (f66), a5101, e3d03 (f67 mid)
- Centroid build times: 19-43 ms/frame (acceptable overhead)

## Session state at stop

- T2 stopped mid-f67 before Optimize Done emitted.
- Checkpoint files 60-65 saved.
- Next test: T3 (Fix D split persistence) OR T2b (narrowed centroid territory) per user direction.

## Visual confirmation (2026-04-22 22:02)

User-provided screenshot from T2 f65 shows **multiple cells clumped together in the central cluster** — ellipsoids visibly stacked/coincident, overlapping the same bright region with wrong shape. This extends the e3d03 FP pattern from a single-cell anomaly to a **cluster-wide failure mode** of Fix A:

When N cells occupy a crowded region, their snap-Voronoi territories necessarily touch and often overlap with neighbors' bright peaks. Each cell's brightness-weighted centroid is therefore a weighted average that includes its neighbors' brightness. With all N cells' centroids dragged toward the same cluster-wide brightness mass, Fix A pulls them TOWARD EACH OTHER. Net effect: crowded-cluster cells collapse into a clump, shape fits fail, downstream PCA becomes noisy, split attempts misfire.

This is the same underlying mechanism as the e3d03 regression, just scaled to the full cluster. **Confirms T2b priority: narrow centroid computation to pixels within the cell's own radius** (e.g., `< 1.5 × cell.aR`), preventing Fix A from seeing neighbor brightness at all.

The f65 clumping at T2 is the upstream cause of the deep-lineage over-splitting observed at f66 (`a51101`, `a51110` with big drifts) — once cells are clumped, the optimizer's split candidates scan outward from clumped centers and can land on bright neighbors' signals, triggering accepts at geometrically weird daughter positions.
