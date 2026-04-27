# T2b — Narrowed brightness-centroid territory (radius_factor=1.5)

**Output dir:** `outputs/output_ablation_T2b_20260422_223025` (partial, stopped mid-f67)
**Delta vs T1:** `brightness_centroid_anchor_weight: 0 → 50`, `brightness_centroid_radius_factor: 1.5` (new). Fix C unchanged. seed=42.
**Status:** FAILED — pivoting to T3.

## Hypothesis

Restricting brightness-centroid integration to voxels within
`1.5 × cell.majorRadius` of the cell's own center would eliminate the
neighbor-brightness drag that made T2 worse than T1, while preserving Fix A's
rescue of the z=0 drifter (`1f2ed10d:111`).

## Result — partial run (f61-f66 optimize-done + f67 first split before stop)

### Aggregate deltas vs T1 (from `compare_ablation.py`)

| Metric | T1 | T2b (partial) | Direction |
|---|---|---|---|
| Total splits | 17 (full f61-f72) | 6 (partial f61-f67) | — |
| Splits with drift > 15 | 1 | 2 | **worse** |
| Max aR across frames | 57.4 | **65.0** | **worse** |
| Mean per-frame drift | 8.97 | **21.97** | **2.45× worse** |
| Cell-frames drift > 25 | 12 | 33 | **~3× worse** |
| f66 cell count | 38 | 36 | behind (missed 23101 TP) |

### Split-diff (partial vs T1 f61-f66)

**Only in T1 at f66 (T2b missed):**
- `e9077677:000` d1=1.8 d2=0.8 cost=-127k — clean TP
- `e9077677:101` d1=13.1 d2=12.3 cost=-61k — borderline
- `12345679:101` (23101) d1=4.1 d2=5.2 cost=-82k — the clean TP T1 gained over T0
- `12345679:000` d1=8.7 d2=7.0 cost=-45k — clean TP

**Only in T2b:**
- `e9077677:1101` (a51101) f66 d1=12.8 d2=10.8 cost=-151k — suspicious deep-lineage, same cell T2 accepted (T2 drifts were 17.8/18.4; T2b cleaner but same cell)
- `1f2ed10d:111` f66 d1=41.8 d2=7.6 cost=-64k — **the intended rescue**, daughters at mid-z (70, 47) away from z=0 boundary. This is the one Fix A win preserved.

### Decisive signal — f67 e3d03 FP reproduced

Before the user stopped the run, f67 processed its first split:
```
[Split Accepted] e3d03428 drift1=53.0 drift2=18.1 cost=-53947
```

T2's e3d03 FP was drift1=63.7. T2b reduced magnitude (53 vs 63.7) but the FP
**class** is still present. Radius gate didn't structurally prevent it — only
damped severity. The Voronoi territory at f67 is already asymmetric enough
that even voxels within `1.5 × aR` of e3d03's center contain misattributed
neighbor brightness.

## Visual — f65 clumping

User-provided f65 screenshot shows cluster clumping still present (though
milder than T2's f65): synth ellipsoids still stack in the central cluster.
Root cause same as above: in dense regions, nearest-neighbor distances are
often < `1.5 × cell.aR`, so the radius gate doesn't exclude the neighbor's
voxels from the cell's Voronoi territory.

## Verdict

**FAILED.** T2b is worse than T1 on nearly every aggregate metric:
- Drift up 2.45×, cell-frames with drift > 25 up ~3×
- Max aR up to 65 (from T1's 57.4) — Fix A drives excess movement that
  some cells parlay into bloat past the barrier cap
- Missed 4 clean TPs at f66 alone
- e3d03 FP still reproduced at f67 (damped but present)

The one intended win (`1f2ed10d:111` rescue at f66) is real, but the cost
across the cell population is too high. Fix A's **directional pull is wrong**
even when spatially restricted — because the pull target (brightness
centroid) is a function of the real-image signal in a region that is not
always the cell's own signal in crowded late-frame conditions.

## Decision

**Disable Fix A entirely and pivot to T3 (Fix D split persistence).** Fix D
rejects splits that don't pass bio+cost gates at nearly-matching daughter
positions across ≥2 consecutive frames. It sidesteps the centroid-geometry
question entirely. Expected to:

1. Kill `e3d03` at f67 regardless of drift/centroid (f65 attempt positions
   don't match f67 attempt positions).
2. Kill `cb11` ugly split at f72 (single-frame opportunistic, no persistence).
3. Not rescue `1f2ed10d:111` z=0 drift (that's Fix A territory) — accept as
   known residual until a territory-independent drift fix is found (e.g.
   z-boundary reflecting penalty).

## Next-session config reset (post-T2b)

```yaml
mc_rng_seed: 42                              # ablation mode
bloat_cap_barrier_weight: 1000.0             # Fix C stays on
brightness_centroid_anchor_weight: 0.0       # Fix A OFF — shelved
brightness_centroid_radius_factor: 1.5       # inert at weight=0
```

`user_input_configurations.ini [jihang]`:
`output_name_rule=output_ablation_T3_{timestamp}`, `resume_from=60`,
`resume_source_dir=...004745`, `last_frame=72`.

T3 implementation scope (new session):
- Cross-frame state carry for per-cell last-attempt daughter positions.
- `trySplitCellPhased`: before accept, check stored last-frame attempt →
  compute position distance + cost similarity → reject if mismatch.
- New config: `split_persistence_required_frames: int = 2`,
  `split_persistence_position_tolerance_vx: float = 10.0`,
  `split_persistence_cost_tolerance_ratio: float = 0.25`.
