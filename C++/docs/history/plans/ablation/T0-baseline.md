# T0 — Baseline (current 2026-04-22 code, seed=42)

**Output dir:** `outputs/output_ablation_T0_20260422_161659`
**Resume from:** f60 of `outputs/output_jihang_20260422_004745/checkpoints/frame_059.txt`
**Frame range:** f61-f72 (13 frames; checkpoint state loaded at end of f59)
**RNG:** `mc_rng_seed: 42` (main MC RNG deterministic; thread-local RNGs still `random_device`)
**Config:** current 2026-04-22 code unchanged — bleed penalty on (w=0.5), slab-min bridge on, velocity cap off, score==0 rollback reset on, birth growth cap gated at 1.8×/1.8elong, voronoi_cost_enabled off.

## Reference metrics (from `compare_ablation.py T0 T0`, restricted to f61-f72)

### Per-frame trajectory

| Frame | Cells | max aR | Bloated (aR > 1.3 × 30 = 39) |
|---|---|---|---|
| 61 | 30 | 45.6 | 8 |
| 62 | 30 | 46.7 | 6 |
| 63 | 30 | 47.4 | 13 |
| 64 | 30 | 49.0 | 14 |
| 65 | 35 | 49.6 | 12 |
| 66 | 38 | 52.0 | 13 |
| 67 | 43 | 54.6 | 10 |
| 68 | 46 | 46.7 | 10 |
| 69 | 47 | 49.0 | 9 |
| 70 | 48 | 51.5 | 10 |
| 71 | 49 | 54.0 | 6 |
| 72 | 50 | 47.1 | 6 |

### Aggregate metrics

| Metric | Value |
|---|---|
| Total splits accepted (f61-f72) | **20** |
| Splits with drift1 or drift2 > 15 vx | **3** |
| Max aR across all frames | **54.6** |
| Mean of per-frame mean aR | **35.3** |
| Frames with any bloated cell (aR > 1.3 × 30) | **12 / 12** |
| Cell-frames with drift > 15 vx | **51** |
| Cell-frames with drift > 25 vx | **13** |
| Mean per-frame drift (vx) | **9.24** |

### Bio-reject counts

| Reason | Count |
|---|---|
| bridge_flat | 10 |
| d1_buried | 11 |
| d2_bridging | 1 |
| d2_buried | 15 |

## Split events (seeded trajectory, f61-f72)

| Frame | Cell | drift1 / drift2 | costDiff | Note |
|---|---|---|---|---|
| 65 | e3d03 | 1.7 / 2.2 | -100k | **FP** (e3d03 doesn't biologically split) — clean drifts |
| 65 | a5111 | 0.5 / 0.2 | -107k | clean |
| 65 | a5110 | 4.4 / 4.7 | -58k | clean |
| 65 | a5010 | 0.7 / 9.2 | -166k | one-sided drift |
| 65 | a5011 | 11.1 / 11.1 | -124k | moderate drifts |
| 66 | a5000 | 3.4 / 0.9 | -85k | clean |
| 66 | 23000 | 5.9 / 4.6 | -43k | clean |
| 66 | a5101 | 3.1 / 3.7 | -59k | clean |
| 67 | a5001 | 1.2 / 0.6 | -183k | ultra-clean |
| 67 | 23011 | 3.3 / 3.5 | -153k | clean |
| 67 | 23001 | 6.0 / 6.1 | -28k | **TP** — caught cleanly at f67 (vs never-caught in run 124756 with random seed) |
| 67 | e3d03...c8d0 | 14.6 / **36.0** | -96k | **FP cascade** (child of e3d03 FP) — ugly asymmetric |
| 67 | 23100 | 9.7 / 10.2 | -120k | moderate |
| 68 | 23111 | 5.4 / 5.7 | -121k | clean |
| 68 | 23110 | 1.1 / 0.4 | -50k | ultra-clean |
| 68 | a5100 | 26.4 / 17.5 | -76k | **asymmetric** — bloat-driven geometry |
| 69 | e3d03...c8d00 | 6.4 / 8.7 | -28k | **FP cascade 3rd level** |
| 70 | 23010 | 1.2 / 0.6 | -129k | ultra-clean |
| 71 | cb10 | 0.4 / 0.3 | -83k | ultra-clean |
| 72 | cb11 | 12.0 / **50.2** | -140k | **massive asymmetric** — bloat-driven |

## Known issues visible in T0

1. **e3d03 FP at f65** (clean drifts 1.7/2.2 — the bloat has made the FP "look legitimate" geometrically). Then cascades to c8d0 at f67 and c8d00 at f69.
2. **Bloat progression**: max aR reaches 54.6 at f67, a5100 is aR=41-51 across f67-f68 (1.4-1.7× birth of ~28). Birth cap at 1.8× + elong ≥ 1.8 doesn't trigger.
3. **Ugly-drift splits** (drift > 15): a5100 f68 (26/17), c8d0 f67 (drift2=36), cb11 f72 (drift2=50). All three attributable to parent bloat making split candidates search far from seed.
4. **Persistent bloat across frames**: 12 / 12 frames have at least one cell above 1.3× birth.

## Decision thresholds for T1 (vs T0)

Targets set in `docs/plans/2026-04-22-late-frame-ablation-plan.md`:

- **Max aR across all frames** ≤ T0/2 (threshold: ≤ 27 or so would be ideal; realistic: ≤ 40 = no cell above 1.3× birth). **T0 value: 54.6.**
- **Frames with any bloated cell** ≤ 6 (half of 12). **T0 value: 12.**
- **Cell-frames with drift > 15 vx** ≤ 25 (half of 51). **T0 value: 51.**
- **Splits with drift > 15 vx** ≤ 1 (was 3 in T0). **T0 value: 3.**
- **e3d03 FP cascade eliminated** (no e3d03 / c8d0 / c8d00 in T1 accepts). **T0 value: 3 e3d03-lineage splits.**

If T1 hits all of these, stop and declare Fix C sufficient.

## Notes

- **Deterministic seed worked**: same-seed compare (T0 vs T0) yields zero deltas — confirmed.
- **23001 split** is RNG-dependent. In run 124756 (random seed) it was cost-rejected across f66-f71; in T0 (seed=42) it accepts cleanly at f67. Option 1 (cost_fraction 0.01) from the earlier analysis may be less urgent than originally thought; the issue was a specific RNG-trajectory dead end.
- Run time: extrapolated ~170 min wall-clock. Actual elapsed not parsed from log tail; will add once extracted.
