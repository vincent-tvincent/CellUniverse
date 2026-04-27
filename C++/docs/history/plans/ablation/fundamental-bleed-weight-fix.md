# Fundamental fix: strengthen Voronoi bleed penalty

**Status:** IN FLIGHT (2026-04-23, post-T3 failure)
**Config delta vs T1:** `voronoi_bleed_penalty_weight: 0.5 → 5.0`. All other
force-term weights unchanged. Fix C on. Fix A off. T3 gate off.

## Why T0/T1/T2/T2b/T3 all hit "tuning hell"

Each ablation attempted a different *additive* fix on top of the same
broken force balance:

- **T0 baseline**: image-cost dominates, overlap fires only at IoU > 0,
  bleed penalty at 0.5 is inert. Cells freely migrate toward cluster
  bright mass. Produces FPs (e3d03 f65/f67) and bloat (cb11 f72).
- **T1 (Fix C)**: caps bloat via log-barrier on radii. Doesn't touch cell
  motion — just prevents cells from inflating beyond 1.6× birth. Killed
  e3d03 f67 FP *indirectly* (e3d03 was inflating on neighbor brightness;
  cap prevented that). Didn't kill cb11 (born large — cap blind). Didn't
  touch drift-into-boundary (f111) or cluster clumping.
- **T2 (Fix C + Fix A full-territory)**: adds quadratic centroid pull.
  Centroid computed over cell's snap-Voronoi territory. In crowded
  clusters, territory includes neighbor brightness → centroid sits on
  cluster mass, not cell's own blob → pull direction inverted → all cells
  drag toward center → visible clumping at f65. Also amplified e3d03 FP
  (drift1=63.7).
- **T2b (Fix C + Fix A, narrowed territory)**: centroid integration
  restricted to voxels within 1.5 × majorRadius of cell center. Tolerance
  chosen to exclude far-away neighbors while keeping own-footprint
  brightness. Failed because **nearest-neighbor distance < 1.5 × aR in
  crowded clusters** — narrowing didn't exclude neighbor brightness,
  just reduced its weight. Mean drift 2.45× worse than T1. e3d03 FP
  reappeared at f67 with drift1=53.
- **T3 (Fix C + split persistence)**: reject splits unless same cell
  produced matching candidate at previous frame. Attempted both absolute
  position (v1 — rejected by parent-drift) and relative-to-parent (v2 —
  rejected by burn-in random-walk). Legit splits had 15-25 vx relative-
  position variability between consecutive frames, indistinguishable
  from random-drift FPs. 8-cell gap vs T1 at f66. Can't separate signal
  from noise via position matching.

## Root cause — force balance audit (from `Frame::perturbCell` delta cost)

| Force term | Weight | What it does | Effective at f65+ |
|---|---|---|---|
| Image cost (L2) | — (native ~100k scale) | Pull toward bright pixels not yet covered | **Dominant** |
| Overlap penalty | 75000 | Push apart cells with IoU > 0 | Only at physical overlap |
| Position prior | 75 | Pull toward snap if drift > 20 vx | Very weak |
| Bloat cap (Fix C) | 1000 | Penalize r > 1.1 × birth | Caps size, not motion |
| **Voronoi bleed** | **0.5** | Penalize voxels outside own territory | **Effectively inert** |

At bleed weight 0.5: a cell drifting 1000 voxels into a neighbor's
Voronoi territory pays 500. Image cost shifts during a single perturb
sweep are 1k–10k. The penalty is ignored.

**The one force that should enforce "each cell stays in its bright blob"
is weighted so low it does nothing.** All other ablation attempts added
new forces on top of this broken balance instead of fixing it. Each new
force had to be strong enough to overcome the image-cost pull, which made
it over-eager (Fix A pulled cells toward neighbors; T3 rejected legit
splits) — because the underlying issue is that **cells are allowed to
migrate freely through each other's territories** and patches had to
compensate for that by being loud.

## The structural fix

Raise `voronoi_bleed_penalty_weight` from 0.5 to 5.0 (10×).

At weight 5: a cell drifting 1000 voxels into a neighbor territory pays
5000. Now comparable to image-cost shifts. Cells still can't physically
overlap (overlap penalty fires), can't bloat (bloat cap fires), AND can't
territorially invade (bleed penalty fires). The Voronoi partition becomes
a soft-but-respected constraint.

This attacks the f65 clumping mechanism directly: cells that would drift
toward neighbor bright regions pay a cost for rendering outside their
own Voronoi partition. They have to either stay put (if their own
partition still has signal) or shrink (not grow into neighbor territory).

## Predictions

- **f65 clumping**: should diminish or disappear — no force pulling cells
  into neighbor territory. Cells stay centered on their own bright blob.
- **e3d03 f65/f67 FP**: e3d03 was a cell with marginal signal drifting
  into neighbor brightness and "finding" a split candidate there. With
  strong bleed penalty, e3d03 stays in its own (sparse) territory; no
  spurious bright signal for a split to attach to.
- **cb11 f72**: cb11 was born large; drift at f72 gave the split
  direction asymmetry. If bleed prevents that drift, cb11 may not
  produce a cost-profitable split candidate.
- **1f2ed10d:111 z=0 drift**: partially addressed. z=0 is the image
  boundary; cell's Voronoi territory extends to boundary if no
  neighbor is on that side. Bleed doesn't help when the territory
  itself extends to z=0. May still need a z-boundary specific fix.
- **Overall splits**: legit splits should still fire (daughter regions
  are inside the parent's old territory). Count should approach T1's
  17.

## Failure modes to watch for

- **Cells stuck in wrong Voronoi partition**: if a cell is initialized
  at a position where its snap-Voronoi territory doesn't cover its own
  real brightness, strong bleed penalty traps it there. Mitigated by
  Voronoi rebuild per frame + after each split accept.
- **Legit split movement blocked**: if the parent's territory is tight,
  daughter burn-in needs to move into fresh pixels. Bleed may oppose
  that. Monitor split accept rate.
- **Over-correction → cells shrink**: cells may shrink to fit inside
  their territory rather than render outside it. Bloat cap has a lower
  bound (ratio ≥ 1.1) but no floor on shrinkage. Watch cell radii.

## Test config

- `mc_rng_seed: 42` (ablation determinism)
- `voronoi_bleed_penalty_weight: 5.0` (up from default 0.5)
- `bloat_cap_barrier_weight: 1000.0` (Fix C on)
- `brightness_centroid_anchor_weight: 0.0` (Fix A off)
- `split_persistence_required_frames: 0` (T3 off)
- INI: `output_name_rule=output_bleed5_{timestamp}`, resume f60 from
  `output_jihang_20260422_004745`, last_frame=72.

## Iteration plan

1. **Run bleed=5.0**. Compare vs T1 (Fix C only). Three outcomes:
   - Passes: f65 clumping gone, splits count near T1, no e3d03 FP, no
     cb11 ugly split. → This is the fix. Ship.
   - Too weak: still clumps / FPs at f65/f67. Bump to 25 or 50, rerun.
   - Too strong: splits blocked, cells shrink, overall cost worsens.
     Back down to 2 or 3.
2. If multiple strengths tested and none works: the fundamental force
   balance isn't fixable by weight alone. Next step: **territory-
   exclusive image cost** — compute image L2 only within each cell's
   own Voronoi voxels. Bigger change but decouples cells' costs.
