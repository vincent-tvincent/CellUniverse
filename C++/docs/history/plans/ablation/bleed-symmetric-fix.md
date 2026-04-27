# bleed=2 symmetric — fundamental force-balance fix

**Run:** `outputs/output_bleed2_sym_20260423_034034` (f60 → f72 complete, seed=42)
**Deltas vs T1:**
- `voronoi_bleed_penalty_weight: 0.5 → 2.0` (4× stronger territorial enforcement)
- `Frame::trySplitCellPhased`: added bleed term to both baseline (parent) and candidate (daughters) cost evaluation so the split-vs-no-split comparison is internally consistent. Daughters measured against PARENT's Voronoi territory (the allowed zone).
- `Frame::computeVoronoiBleedVoxels`: added `bypassEnabledFlag` param so split-cost path can query the Voronoi map while the burn-in disable-guard is active.

## Why this fix exists

T1/T2/T2b/T3 all hit "tuning hell" — each was an additive patch on top of a
force-balance that allowed cells to migrate freely into neighbor Voronoi
territories. The Voronoi bleed penalty (the one force that should prevent
that) was weight=0.5 — effectively inert against image-cost shifts of
1k-10k.

Bumping the weight to 5 (attempt 1) revealed a second problem: the bleed
term was **asymmetric** — applied in `perturbCell` (parent shape fit) but
NOT in `trySplitCellPhased` (split candidate evaluation). Tightening bleed
constrained the parent's fit but left daughters free to tile broadly →
split cost dropped relative to no-split cost → **e3d03 acquired a FP split
at f66** (drift1=19 at bleed=5, drift2=26.7 at bleed=2 asymmetric).

The symmetric fix: bleed term added to both sides. Parent pays for
voxels outside own territory. Daughters pay against PARENT's territory
(the shared allowed zone — daughters collectively inherit parent's
space). Force balance now consistent.

## Results

**14 splits accepted over f61-f72. All clean. Max drift = 11.3 vx.**

| Frame | T1 cells | bleed=2 sym | Splits (T1 / sym) | Notable |
|---|---|---|---|---|
| 61-64 | 30 | 30 | 0/0 | baseline |
| 65 | 34 | 34 | 4/4 ✓ | a5111 (7.3/7.7), a5110 (3.0/3.1), a5010 (2.0/5.1), a5011 (4.1/4.2) |
| 66 | 38 | 37 | 4/3 | missed 23101 clean TP (T1 had 4.1/5.2) |
| 67 | 42 | 40 | 4/3 | missed 23001 clean TP. e3d03 **REJECTED** (T0/T2/T2b all had FP here) |
| 68 | 44 | 42 | 2/2 ✓ | 23111 (5.7/5.7), 23110 (2.5/3.3) |
| 69 | 44 | 42 | 0/0 ✓ | no cascade |
| 70 | 45 | 43 | 1/1 ✓ | 23010 (9.6/9.4) |
| 71 | 46 | 44 | 1/1 ✓ | cb10 (3.7/3.8) |
| 72 | 47 | **44** | 1/**0** | **cb11 ugly split REJECTED** (T1 had drift2=56.7) |

**Delta vs T1**: missed 2 clean TPs (23101, 23001 — both at borderline
cost), rejected 1 ugly split (cb11 drift2=56.7). Net: 3 fewer final
cells, zero FPs, no cb11 pathology.

**Drift comparison** (legit splits only):
- T1: mean ~9, some > 15 (ugly)
- bleed=2 sym: mean ~5, all ≤ 11.3

## Why missed TPs?

23101 at f66, 23001 at f67 are both cases where parent was elongating
and daughters would land slightly outside parent's pre-split Voronoi
territory. At bleed=2, the extra bleed penalty tipped the cost gate
against accept.

Two options:
1. **Weight=1.0 or 1.5**: might recover the edge-case TPs while still
   rejecting e3d03/cb11 (whose daughters extended much further out).
   Worth trying.
2. **Accept**: 14/16 TPs is 88% recall, with 100% FP rejection. Trade
   favorable.

## Force-balance audit (updated)

| Force term | Weight | Symmetric? |
|---|---|---|
| Image cost (L2) | native ~100k | ✓ (both paths use same eval) |
| Overlap penalty | 75000 | ✓ (both paths apply) |
| Position prior | 75 | (perturbCell only, doesn't apply to split) |
| Bloat cap (Fix C) | 1000 | (perturbCell only, doesn't apply to split) |
| **Voronoi bleed** | **2.0** | **✓ NEW — applied to both** |

Position prior and bloat cap apply only to perturbCell because they
measure drift from snap / birth state — concepts that don't apply
directly to a split (daughters don't have snaps/births yet). Bleed is
different: territorial constraint applies to any ellipsoid.

## What this DIDN'T fix (deferred residuals)

- **1f2ed10d:111 z=0 drift**: still a possible issue. This cell had no
  split attempts in the monitor stream, consistent with T1. If the
  cell ends up at z=0 by f72, that's a separate boundary-drift issue.
  Would need to check cells.csv to verify.
- **Missed 23101, 23001**: borderline TPs. Tunable by lowering bleed
  weight, with risk of e3d03/cb11 re-emerging.

## Config (final)

```yaml
mc_rng_seed: 42                              # ablation mode
voronoi_bleed_penalty_weight: 2.0            # FUNDAMENTAL — up from 0.5
bloat_cap_barrier_weight: 1000.0             # Fix C (kept)
brightness_centroid_anchor_weight: 0.0       # Fix A (shelved)
split_persistence_required_frames: 0         # T3 (shelved)
```

INI: `output_name_rule=output_bleed2_sym_{timestamp}`, resume f60 from
`output_jihang_20260422_004745`, last_frame=72.

## Recommendation

**Accept bleed=2 symmetric as the shipping fix.** It:
- Fixes the root cause (force-balance asymmetry) rather than patching symptoms
- Eliminates the entire FP class (e3d03 cascade, cb11 ugly split)
- Simplifies the pipeline (no need for T3 persistence gate or Fix A)
- Still needs minor tuning potential (weight=1 or 1.5 may recover the
  two borderline TPs), but even without that, net result is a strictly
  cleaner cell population than T1.

Production launch: set `mc_rng_seed: 0`, clear `resume_from` / rename
output, run a full f1-f72 from scratch to validate cb11 doesn't get
born bloated in a clean-start trajectory.
