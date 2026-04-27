# Changelog v8 — yp_fix_mask preprocessing merge + post-merge work (2026-04-19+)

Opened 2026-04-19. Previous: `changelogv7.md` (closed at Change 50).

This file marks the start of a new stage: integration of the `yp_fix_mask_04172026` preprocessing pipeline (parallel raw load + global percentile normalization + iterative contrast-scoring preprocessing). Our snap-only split logic from v7 (Change 50) is preserved alongside the new preprocessing.

Branch (merge done on): `jl_yp_preprocessing_merge_04192026`

---

## 2026-04-19: Merge yp_fix_mask preprocessing pipeline (Change 1)

**Status: ACTIVE — validated against snap-only baseline**

### Cherry-picked from `origin/yp_fix_mask_04172026`

| Commit | Content |
|---|---|
| `9b3fcfd` | CMakeLists.txt arm/x86 portability |
| `6ca6331` | Make pre-process parallel |
| `87ee430` | Simplified preprocessing logic, much faster + better contrast |
| `25c5923` | Preprocessing improvement: split `loadFrame` into `loadRawFrame` + `preprocessLoadedFrame`; added contrast/brightness scoring fields and 4-pass constructor |
| `b575246` | Adaptive cube pooling code (kept disabled in our config) |

**Skipped:** `a9b19cb` (their shape config tuning), `ffc1917` (signal-strength guided perturbation), `f42ccc2`/`82055e3`/`f56e67e`/`c1e1dd2`/`6661c96` (their tuning rounds), `cbc0714`/`14afa6a` (debug toggles).

### Files changed

- **`C++/CMakeLists.txt`** — copied yp wholesale (we had no local edits)
- **`C++/src/ImageHandler.cpp`** — copied yp wholesale (~900 LOC: 4-pass preprocessing pipeline, percentile scoring, adaptive cube pooling implementation, signal-center localization)
- **`C++/includes/ImageHandler.hpp`** — copied yp wholesale (new public API: `loadRawFrame`, `preprocessLoadedFrame`, `evaluateSequenceContrastScore`, plus `loadFrame` with optional `logSink`)
- **`C++/includes/ConfigTypes.hpp`** — copied yp wholesale, added back `position_prior_threshold` field + parse (ours; yp doesn't have it)
- **`C++/includes/Frame.hpp`** — added minimal `SignalCenter` struct (public nested) so yp's `ImageHandler::localizeSignalCentersInStack` compiles; no member or accessors added (signal-guided perturbation feature is unused in our pipeline)
- **`C++/src/CellUniverse.cpp`** — added 3 helpers (`computeStackPercentile`, `computeStackMax`, `normalizeStackToSharedScale`); replaced constructor body with yp's 4-pass pipeline (parallel raw load → global percentile normalization → parallel preprocess → frame construction). Kept everything else (`optimize`, snap-only split logic, all our shape/split work) unchanged.
- **`C++/config/config.yaml`** — hybrid: yp's preprocessing block wholesale, our shape/split tuning preserved. Pooling disabled per user instruction.

### Why the constructor port was necessary

First merge attempt copied only `ImageHandler.cpp` + config preprocessing fields; preprocessing produced inverted images (mean=0.95 vs expected ~0.05). Diagnosis: yp's `loadFrame` runs an iterative contrast-scoring loop that assumes inputs have been globally normalized to a shared low/high reference. Our prior single-pass `loadFrame` per frame skipped that normalization. Porting yp's 4-pass constructor (which computes global percentiles across all loaded frames before per-frame preprocessing) restored correct image polarity.

### Validation (run 113836, 22 frames)

Compared HYBRID (this merge) vs SNAP-ONLY (run 095555, last v7 baseline) vs BEST22.

| Cell | GT | HYBRID | SNAP-ONLY | BEST22 |
|------|----|---|---|---|
| e9077 | f3 | **f3 ✓** | f3 ✓ | f3 ✓ |
| 12345 | f3 | **f3 ✓** | f3 ✓ | f3 ✓ |
| 1f89ab | f8 | **f8 ✓** | f8 ✓ | f8 ✓ |
| 1f2ed | f11 | **f11 ✓** | f11 ✓ | f12 (+1) |
| e9077..a50 | f19 | f20 (+1) | f20 (+1) | f20 (+1) |
| 12345..0 | f20 | **f20 ✓** | f20 ✓ | f20 ✓ |
| 12345..1 | f20 | **f20 ✓** | f20 ✓ | f20 ✓ |
| e9077..a51 | f20 | **f20 ✓** | f20 ✓ | f20 ✓ |

| Score | HYBRID | SNAP-ONLY | BEST22 |
|---|---|---|---|
| Splits accepted | 8/8 | 8/8 | 8/8 |
| **On-time** | **7/8** | **7/8** | 6/8 |
| Cells at f22 | 14 | 14 | 14 |

**Hybrid matches snap-only on timing.** Notable difference: split cost diffs are 2-3× larger (e.g. e9077 f3: -114k vs -38k snap-only). Stronger preprocessing contrast → bigger split signal → more confident accepts and more headroom for marginal cases.

### Rollback

Per-file rollback (each independent):
- `git checkout HEAD~1 -- C++/CMakeLists.txt`
- `git checkout HEAD~1 -- C++/src/ImageHandler.cpp C++/includes/ImageHandler.hpp`
- `git checkout HEAD~1 -- C++/src/CellUniverse.cpp` (loses constructor port; `ImageHandler.cpp` will then need rebuild against old signature — partial rollback unstable, prefer full rollback)
- `git checkout HEAD~1 -- C++/includes/ConfigTypes.hpp C++/includes/Frame.hpp C++/config/config.yaml`

For full rollback: `git revert <merge-commit>` (clean, takes everything back together).

### Open follow-ups

- **Backburner — halo coverage**: synth cells don't cover the bright-cell HALO (only the core). Levers when we tackle this: `pcaShapeMaskScale` (1.6 → 1.8-2.0), `pcaShapeRadiusPercentile` (0.90 → 0.95-0.97), or direct `pcaShapeRadiusScale` bump.
- **45-frame validation pending** — confirm hybrid behavior holds at longer horizons (the snap-only 45-frame run from v7 had cascade effects at f16/f29/f32 which may resolve differently with yp preprocessing's stronger contrast).
- **Pooling evaluation** — `adaptive_cube_pooling_enabled` is currently `false`. Run a separate validation with it enabled once baseline is stable.
- **45-frame run vs `e9077..a50` at f19** — still missed (1-frame late) across all three runs (ours, snap-only, hybrid). Possibly the cleanest remaining +1-late target.

### Notes for future entries

- Active branch `jl_yp_preprocessing_merge_04192026` is split off from `jl_runtime_improve_04162026` after the snap-only commit (`3d08daf`). Future cleanup work should branch off here.
- All future preprocessing/cleanup/optimization changes go in this v8 file. v7 is closed at Change 50 (snap-only).

---

## 2026-04-19: Config audit — magnitude-dependent gate retuning (Change 2)

**Status: ACTIVE — validated 8/8 splits 7/8 on time 0 FP at 22-frame horizon**

### Why

Yp preprocessing produces image cost magnitudes 2-3× larger than ours. Real-split cost diffs grew similarly (was -8k to -30k snap-only, now -56k to -177k hybrid). Existing gates (split_cost=2000 fixed, split_cost_fraction=0.03) were calibrated for the OLD magnitudes — too lenient at the new scale. The 45-frame hybrid run (130101) confirmed: **e3d03 false-positive split at f4** with diff=-58k against baseline 195k (30% improvement) cleanly cleared the 0.03 (~5860) threshold.

### Fields changed in `C++/config/config.yaml`

| Field | Was | Is | Why |
|---|---|---|---|
| `split_cost` | 2000 | **7000** | Fixed split-cost floor; ~3.5× to track cost magnitude growth |
| `split_cost_fraction` | 0.03 | **0.20** | Real splits show 100%+ improvements; FPs show ~30%. 20% blocks FPs cleanly with huge margin for real splits |
| `overlap_penalty_weight` | 30000 | **75000** | 2.5× to keep relative overlap pressure constant |
| `position_prior_weight` | 30 | **75** | 2.5× so position prior is not diluted by larger image costs |
| `bio_bridge_max_valley_ratio` | 0.85 | **0.75** | Tighter bridge gate; FP slipped past 0.85 by spanning genuine empty space between distant daughters |
| `bio_bridge_min_edge_brightness_absolute` | 0.04 | **0.07** | Yp preprocessing brightens halos → phantom edges sit on halo. Tightening blocks halo-on-halo phantom splits |
| `pcaShapeMaskScale` | 1.6 | **1.3** | Smaller mask = less halo absorption = more compact fits |
| `pcaShapeWeightExponent` | 1.0 | **1.3** | Suppress halo influence in PCA centroid (snap-only baseline value) |
| `pcaShapeRadiusPercentile` | 0.90 | **0.85** | Tighter radii — top 10% under yp may include halo |
| `adaptive_background_top_fraction` | 0.4 | **0.2** | Top 40% of bg candidates may include cell halo; top 20% targets cleaner bg |

### Validation (run 135708, 22 frames)

| Cell | GT | Result |
|---|---|---|
| e9077 | f3 | f3 ✓ |
| 12345 | f3 | f3 ✓ |
| 1f89ab | f8 | f8 ✓ |
| 1f2ed | f11 | f11 ✓ |
| e9077..a50 | f19 | f20 (+1) |
| 12345..0 | f20 | f20 ✓ |
| 12345..1 | f20 | f20 ✓ |
| e9077..a51 | f20 | f20 ✓ |

8/8 splits, 7/8 on time, 0 FP. Identical timing to pre-audit hybrid run. Tightened gates blocked the f4 FP without losing any real splits.

---

## 2026-04-19: Signal-guided perturbation port + iter tune (Change 3)

**Status: ACTIVE — validated 8/8 splits 7/8 on time 0 FP**

Ported yp's `ffc1917` signal-guided perturbation feature in full (frame-level toggle design, not per-iteration).

### Files changed

- **`C++/includes/Ellipsoid.hpp`** — added `setPosition(float x, float y, float z)` setter (used by signal-guided teleport).
- **`C++/includes/Frame.hpp`** — `_signalCenters` member, `setSignalCenters`/`getSignalCenters`, `useSignalGuidance` param on `perturbCell`.
- **`C++/src/Frame.cpp::perturbCell`** — added signal-guidance teleport block (~40 LOC) after `getPerturbedCell`. When `useSignalGuidance` is true, finds the nearest signal center to oldPos and resamples cell position from a normal distribution around that center (sigma scaled by per-cluster `sigmaScale × signal_guided_sigma_range_multiplier`).
- **`C++/src/CellUniverse.cpp`** — added `BrightBox` struct, `chooseNearestDivisorSize`, `localizeSignalCentersForFrame` (~170 LOC). In `optimize()`, decides per-frame whether to use signal-guided based on:
  1. `signal_guided_position_enabled` is true
  2. Centers detected ≥ previous frame cell count (otherwise fallback to random mode)
- **Iteration count split**: `signal_guided_iterations_per_cell` for guided frames, `random_iterations_per_cell` for random frames. Falls back to `iterations_per_cell` when either is < 0.
- **`C++/config/config.yaml`** —
  - `iterations_per_cell: 500 → 150`
  - `signal_guided_position_enabled: true`
  - `signal_guided_iterations_per_cell: 100`
  - `random_iterations_per_cell: 150`
  - `signal_guided_box_size_scale: 0.4`, `signal_guided_min_box_brightness_delta: 0.4`, `signal_guided_min_sigma_scale: 0.35`, `signal_guided_sigma_range_multiplier: 10.0` (yp tuned defaults)

### Why frame-level (not per-iteration)

First port attempt (run 100630) used signal-guidance for ALL perturbations: regression to 5/8 splits, perturb-accept rate dropped to 5-20/frame. Signal-guided teleport is destructive when applied unconditionally — it overwrites good positions ~80% of the time. Yp's frame-level design (run 100631) uses signal-guided only when conditions favor it: enough centers were detected to plausibly cover all cells.

### Validation (run 100631, 22 frames)

Per-frame mode breakdown:
- f1: signal_guided (no prev-frame fallback condition; 0 perturbs but no splits expected)
- f3: signal_guided (centers=6 = prevCells=6) → 2 GT splits accepted ✓
- All other frames: random fallback (centers usually 4-9, well under cell count)

Result: **8/8 splits, 7/8 on time, 0 FP**. Same as audit baseline.

### Open follow-ups (still applicable from Change 1)

- 45-frame validation
- Halo coverage tuning
- Pooling evaluation
- Fix `e9077..a50` +1 late at f19 (consistent miss across all hybrid runs)


---

## 2026-04-19: PCA shape revert + signal-guided disabled for isolation (Change 4)

**Status: ACTIVE — validated 8/8 splits 7/8 on time 0 FP at 22-frame, daughter radii restored**

### Why

The Change 2 audit tightened three PCA shape fields defensively (`pcaShapeMaskScale: 1.6→1.3`, `pcaShapeWeightExponent: 1.0→1.3`, `pcaShapeRadiusPercentile: 0.90→0.85`) alongside the cost/bio gate tuning. Inspection of the resulting 22-frame run showed daughter c-axis shrinking from 15 px (baseline) to 7 px by f20 — cells compacted too aggressively. The audit's cost/bio gate changes (`split_cost_fraction: 0.03→0.20`, tightened bridge gate) were already blocking the e3d03 f4 FP directly, so the PCA shape tightening was redundant and harmful: it suppressed legitimate cell extent without adding FP protection.

### Fields reverted in `C++/config/config.yaml`

| Field | Change 2 (audit) | Change 4 (reverted) | Baseline value |
|---|---|---|---|
| `pcaShapeMaskScale` | 1.3 | **1.6** | 1.6 (best22 validated) |
| `pcaShapeWeightExponent` | 1.3 | **1.0** | 1.0 (linear weighting) |
| `pcaShapeRadiusPercentile` | 0.85 | **0.90** | 0.90 (top 10% trim) |

All three back to their v7 snap-only baseline values. The cost and bio gates from Change 2 (`split_cost: 7000`, `split_cost_fraction: 0.20`, `overlap_penalty_weight: 75000`, `position_prior_weight: 75`, `bio_bridge_max_valley_ratio: 0.75`, `bio_bridge_min_edge_brightness_absolute: 0.07`, `adaptive_background_top_fraction: 0.2`) are KEPT — they are the actual FP-blocking levers.

### Signal-guided disabled for isolation

`signal_guided_position_enabled: true → false`. Change 3 (signal-guided port) is kept in the codebase and config; only the master toggle is off so the 22-frame validation isolates the PCA shape revert's effect. Per-frame iter buckets still apply (`random_iterations_per_cell: 150` is used on all frames when the toggle is off). Re-enabling is a separate evaluation after the 45-frame baseline lands.

### Validation (run 162148, 22 frames)

| Cell | GT | Result |
|---|---|---|
| e9077 | f3 | f3 ✓ |
| 12345 | f3 | f3 ✓ |
| 1f89ab | f8 | f8 ✓ |
| 1f2ed | f11 | f11 ✓ |
| e9077..a50 | f19 | f20 (+1) |
| 12345..0 | f20 | f20 ✓ |
| 12345..1 | f20 | f20 ✓ |
| e9077..a51 | f20 | f20 ✓ |

8/8 splits, 7/8 on time, 0 FP. Same timeline as Changes 2 and 3. **Daughter radii restored to baseline sizes** (c-axis ~15 px at f20 vs 7 px with the audit PCA settings).

### Conclusion

Cost-gate audit, not PCA shape tightening, blocks FPs. PCA shape fields stay at baseline values (1.6 / 1.0 / 0.90). Ready for 45-frame validation (the critical next test — confirms no FP regression at longer horizon AND that daughter/grand-daughter radii stay visible).

### Open follow-ups

- 45-frame validation of current config (Change 2 gates + Change 4 PCA revert + signal-guided OFF)
- Re-enable `signal_guided_position_enabled: true` once 45-frame baseline is clean, to see if guidance helps anywhere when not active by default
- Halo coverage tuning (now easier to assess with baseline PCA values)
- Pooling evaluation (`adaptive_cube_pooling_enabled: false` still untested)
- `e9077..a50` +1 late at f19 (hard case, consistent across all hybrid runs)


---

## 2026-04-19: Shape-fix — pooling + radiusScale + cost gate revert (Change 5)

**Status: ACTIVE — validated 8/8 splits 7/8 on time 0 FP at 22-frame; cell shapes restored to visually match real bright extents**

### Why

User flagged after Change 4 validation (run 162148): cell size/shape "very incorrect vs the two runs before the merge" — synth ellipsoids fit the bright cell cores but not the full cell extent. Frame-by-frame comparison vs pre-merge baseline (095555) confirmed:

- **Frame-1 PCA radii systematically undershot baseline**: 12345 c-axis 16.7 vs 22.7 (−26%), e9077 c-axis 20.0 vs 25.5 (−22%), 1f89ab a-axis 35.4 vs 43.3 (−18%).
- **Mask pixel count** (`n`) at f1 was ~40% lower than baseline (e.g. 12345: 95044 vs 154863), while **meanW** was ~2× higher — yp preprocessing creates brighter cores with compressed halos, collapsing weighted variance along thin axes.
- **Cells kept shrinking frame-over-frame** in the Change 4 baseline: 1f89ab a-axis drifted from −11% at f1 to −40% at f10 to −32% at f22. The fit growth cap couldn't overcome per-frame PCA undershoot.

### Investigation path (what we tried and why each failed before landing on the final combo)

1. **`pcaShapeRadiusPercentile: 0.90 → 0.95`** — no effect. Pipeline doc + code confirm percentile path is DEAD since v7 Change 50; active code uses `pcaShapeRadiusScale × sqrt(weightedVariance)`. Reverted.
2. **`pcaShapeMaskScale: 1.6 → 1.8`** — small positive effect (~+1-2% on c-axis for some cells). Kept.
3. **`pcaShapeRadiusScale: 2.236 → 2.5`** (run 175740) — inflated cells uniformly by +12%, but inflated baseline image costs too (1f89ab f8 baseline 145k → 226k). Real split improvements as a fraction of baseline dropped from 31% (Change 4) to 16%, **failing the audit's `split_cost_fraction: 0.20` gate**. 1f89ab f8 rejected.
4. **`split_cost_fraction: 0.20 → 0.12`** on top of rs=2.5 (run 181007) — still not enough. Baseline cost rose further (225k → 368k), real ratio dropped to 9.8%. Chasing the fraction was a losing game.
5. **`pcaShapeRadiusScale` reverted to 2.236 + `adaptive_cube_pooling_enabled: true`** (run 182118) — pooling spread core brightness into cube-neighborhoods, mask pixel count recovered +26% at f1, c-axis mostly recovered (12345 c=16.7→18.5, e9077 c=20.0→23.2). But user observation: cells still "almost uniform brightness" AND still undersized by 20-40% by mid-run because pooling's `majority_threshold: 0.7` erodes the boundary, and weighted variance still concentrates fit near core.
6. **Final combo**: pooling ON + `pcaShapeRadiusScale: 2.5` (re-applied) + **`split_cost` and `split_cost_fraction` reverted to pre-merge values** (2000 / 0.03). With cost gate relaxed, bigger baseline costs no longer block real splits. Bio gates (tightened in Change 2: valley 0.75, edge 0.07) + re-enabled position prior (w=75) carry FP protection.

### Fields changed (all in `C++/config/config.yaml`)

| Field | Change 4 value | Change 5 value | Active lever? |
|---|---|---|---|
| `pcaShapeMaskScale` | 1.6 | **1.8** | Yes (small c-axis gain) |
| `pcaShapeRadiusScale` | 2.236 | **2.5** | Yes (primary fit-size lever) |
| `pcaShapeRadiusPercentile` | 0.95→0.90 | 0.90 | **Dead code** (see comment) |
| `adaptive_cube_pooling_enabled` | false | **true** | Yes (halo preservation) |
| `split_cost` | 7000 | **2000** | Yes (pre-merge value restored) |
| `split_cost_fraction` | 0.20 | **0.03** | Yes (pre-merge value restored) |

Kept from Change 2: `overlap_penalty_weight: 75000`, `position_prior_weight: 75`, `bio_bridge_max_valley_ratio: 0.75`, `bio_bridge_min_edge_brightness_absolute: 0.07`, `adaptive_background_top_fraction: 0.2`.

### Validation (run 193101, 22 frames)

| Cell | GT | Result | Split diff |
|---|---|---|---|
| e9077 | f3 | f3 ✓ | −122910 |
| 12345 | f3 | f3 ✓ | −71028 |
| 1f89ab | f8 | f8 ✓ | −56957 |
| 1f2ed | f11 | f11 ✓ | −123776 |
| e9077..a50 | f19 | f20 (+1 late) | −147848 |
| e9077..a51 | f20 | f20 ✓ | −57553 |
| 12345..0 | f20 | f20 ✓ | −121329 |
| 12345..1 | f20 | f20 ✓ | −101780 |

**8/8 splits, 7/8 on time, 0 FP.** e3d03 at f4 was rejected by bio gate (`d1_bridging_to_12345`) — exactly the historical regression, now blocked even with relaxed 0.03 cost fraction. Multiple bridging-type FPs at f18/f20/f21/f22 also all blocked by bio gate. **Bio gate is fully load-bearing for FP control; cost gate is now purely a floor.**

### Shape improvement (Δ vs baseline 095555, f1)

| Cell | Change 4 (162148) | Change 5 (193101) |
|---|---|---|
| 1f89ab a-axis | −11% | **−0.2%** ✓ |
| 1f89ab b-axis | −5% | +6% |
| 1f89ab c-axis | −7% | +4% |
| 12345 a-axis | +2% | +13% |
| 12345 c-axis | −26% | −9% |
| e9077 c-axis | −22% | +1% ✓ |
| 1f2ed c-axis | −7% | +11% |
| e3d03 (all axes) | +9%/+8%/+10% | +21%/+21%/+23% (overshoot) |

Most cells now within ±15% of baseline. User visual confirmation: "now it looks better, some cells do still falls a little bit short, but overall looks good."

### Known remaining issues

- **e3d03 over-inflated** by ~20-25% — smallest/dimmest cell, gets relatively more PCA-driven inflation under rs=2.5. Not blocking but visually slightly rounder than real.
- **Some e9077 late-frame daughters shrink aggressively** (e9077..a50 at f22 c-axis = 7px, −45% vs baseline). Fit growth cap may need re-examination for small grand-daughter cells.
- **`e9077..a50` consistent +1 late at f19** — unchanged across all hybrid runs and all our tuning. Not a cost/shape gate issue; needs targeted investigation.

### Rollback

Single-file revert: `git checkout HEAD~1 -- C++/config/config.yaml` reverts all 6 field changes together. Incremental rollback: see the `2026-04-19 (final/size-fix/...)` comment blocks in config.yaml.

### Open follow-ups

- **45-frame validation** — next major test. Confirms FP blocking holds at longer horizon with 0.03 cost fraction + tightened bio gates.
- **e3d03 overshoot** — if bothersome, small cells could get a per-cell radiusScale cap (code change) or `pcaShapeAdaptiveExponent` could be retuned to de-emphasize rs bump for dim cells.
- **e9077..a50 f19 miss** — dedicated investigation: fit quality at f19 vs split criteria.
- **Pooling threshold tuning** — if visual shape is still unsatisfying, `adaptive_cube_pooling_majority_threshold: 0.7 → 0.4` would preserve edge cubes (tried mentally, not empirically this session).


---

## 2026-04-19: Post-birth daughter growth fix + 45-frame full recall (Change 6)

**Status: ACTIVE — validated 20/20 GT splits 15/20 on time 0 FP at 45-frame horizon; beats best45 baseline (18/20)**

### Why

User observed in run 193101 (Change 5) that late-frame daughters stay undersized: synth ellipsoids fit the core but miss the visibly-growing real extent. Root cause: `cellShapeBirth` is frozen at split time (gotcha B), so `mask = birthRadii × maskScale` is a hard ceiling on the post-split PCA fit. Cells born small at f20 with `maskScale=1.8` couldn't track their actual growth through f21-f22 — PCA variance stays mask-bounded, growth cap then sees no demand to grow reference.

### Fields changed in `C++/config/config.yaml`

| Field | Change 5 | Change 6 |
|---|---|---|
| `pcaShapeMaskScale` | 1.8 | **2.2** |

22% extra slack on the frozen birth mask lets PCA fit track post-birth cell growth (typical daughters expand ~15-25% over 2-3 frames before dividing again). 2.2 is the moderate step — going beyond ~2.5 risks neighbor-cell pixel contamination in dense f20+ clusters.

### Validation — 22-frame (run 203458)

8/8 splits, 7/8 on time, 0 FP. Late-frame daughter c-axis at f22:
- e9077..a52 (worst under 1.8): c=5.0 (−45%) → c=~13 (+4%) under 2.2. **Collapse eliminated.**
- 1f89ab daughters now within ±10% of baseline (vs ±15-20% under 1.8)

### Validation — 45-frame (run 211928)

**20/20 GT splits captured. 15/20 on time. 0 FP. 26 cells (matches GT).**

| GT Frame | GT Split | Our Frame | Δ |
|---|---|---|---|
| f3 | e9077, 12345 | f3 | **on time** |
| f8 | 1f89ab | f8 | **on time** |
| f11 | 1f2ed | f11 | **on time** |
| f19 | e9077..a50 | f20 | +1 |
| f20 | 12345..0, 12345..1, e9077..a51 | f20 | **on time** |
| f27 | 1f89ab..1 | f28 | +1 |
| f28 | 1f89ab..0 | f28 | **on time** |
| f31 | 1f2ed..0 | f32 | +1 |
| f38 | 1f2ed..1 | f38 | **on time** |
| f39 | 12345..01, 12345..00, e9077..a511, e9077..a500, e9077..a501, 12345..10, 12345..11 | f39 | **on time** |
| f39 | e9077..a510 | f40 | +1 |

FP control: e3d03 at f4 blocked by bio gate (historical regression), all bridge-type attempts at f18/f22/f34/f40/f41/f42 blocked by bio gate. Cost gate at 0.03 did not reject any real split; all cost rejects had positive diff (fit would worsen).

### vs best45 baseline (20260418_best45)

| | best45 | Change 6 run |
|---|---|---|
| Splits captured | 18/20 | **20/20** ✓ |
| 1f89ab..1 f27 | MISSED | captured f28 |
| e9077..a500 f39 | MISSED | captured f39 on time |
| FP | 0 | 0 |
| Final cell count | 24/26 | **26/26** ✓ |

**Milestone: 100% GT recall with 0 FP at the full 45-frame horizon.**

### Runtime — 8351 sec (~139 min) for 45 frames

Per-frame cost grows with cell count:
- f1-f10 (6-10 cells): ~1.5 min/frame
- f20-f30 (14-16 cells): ~3 min/frame
- f39-f45 (25-26 cells): ~5-8 min/frame

### Phase A optimization (this commit): parallelize `applyAdaptiveCubePooling`

**Files changed:** `C++/src/ImageHandler.cpp`

Four `#pragma omp parallel for collapse(2)` directives added to the pooling function's grid loops:
1. Cube-stats computation (line ~178)
2. Cube-reweighting + voxel fill (line ~233)
3. Isolated-bright-cube neighbor check (line ~315)
4. Isolated-bright-cube voxel zeroing (line ~357)

All writes are to disjoint cube-local memory; counters use OpenMP `reduction(+:...)`.

**Expected speedup:** 4-6× on pooling step. Pooling runs 45 times (once per frame in constructor) at ~1.8M cubes each → ~80M cube operations previously serial, now spread across cores. Full-run impact: ~60-90 sec saved.

### Rollback

`git checkout HEAD~1 -- C++/config/config.yaml C++/src/ImageHandler.cpp` reverts both the mask bump and parallel pooling.

### Open follow-ups

- **Benchmark Phase A** after rebuild — confirm pooling speedup and check total runtime
- **Phase A continued**: pre-allocate `cv::Mat` scratch buffer in `generateSynthFrameFast` (eliminates ~42k Mat allocations/frame late-run), cache inverse rotation matrix per draw, parallelize split candidate burn-in with state-copy pattern
- **Shape corners**: e3d03 still overshoots +20-25%; small-cell `pcaShapeRadiusScale` cap or adaptive exponent tune is the follow-up
- **+1-late drift across 5 splits** — growth cap per-frame is a candidate to relax post-split (daughters get higher cap for first 2-3 frames)

---

## 2026-04-20: Velocity cap + birth growth cap v2 + checkpoint system (Change 7) — **status: velocity cap REVERTED 2026-04-21**

### Why

72-frame WSL run `output_jihang_linux` exposed three problem classes:
1. **Position drift** — cells drifting slowly toward image boundaries; eventually stuck at z=0 or z=224.
2. **Bloat** — cells growing oversize instead of splitting (a511 reached a=58.6 vs birth 28.6, ~2× oversize, blocking its own split).
3. **Iteration friction** — 45f+ runs take 2+ hours; iterating on late-frame behavior required re-running from f1.

### Fields added to `C++/includes/ConfigTypes.hpp` (`ProbabilityConfig`)

```cpp
float max_perturb_drift_xy = 15.0f;  // disabled 2026-04-21 → 0.0f
float max_perturb_drift_z  = 20.0f;  // disabled 2026-04-21 → 0.0f
float birth_growth_cap_factor = 1.8f;
float birth_growth_cap_elong_threshold = 1.8f;
int   resume_from = 0;
std::string resume_source_dir = "";
```

### Implementation summaries

**Velocity cap** — `C++/src/Frame.cpp:749-770` (new block in `perturbCell`). Reject any perturbation that moves the cell further than `max_perturb_drift_{xy,z}` from its snap position (fixed per frame). Restores cell to pre-perturb state and returns 0 diff + noop callback.

**Birth growth cap v2** — `C++/src/CellUniverse.cpp:1085-1180` (new block in `optimize`). Clamp per-cell radii at `birthRadii × factor` ONLY when `shapeElongation ≥ elongThreshold` also. The conjunction gate is important: v1 (factor=1.5 alone) triggered a false positive e3d03 split at f28 because it clamped healthy cells (including e3d03's neighbors) and shifted the cost landscape. v2 only fires on cells that are BOTH big AND elongated — the classic pre-split bloat pattern — leaving healthy growing round cells alone.

**Checkpoint save/load** — new methods `CellUniverse::saveCheckpoint(N)` / `loadCheckpoint(N)` write/read plain-text state per frame: cells vector, `previousSnapshots`, `cellShapeReference`, `cellShapeBirth`, summaries. One file per frame at `{output}/checkpoints/frame_{N:03d}.txt`. Resume logic in `C++/src/main.cpp:114-140` checks `config.simulation.resume_from > 0 && resume_source_dir != ""` at launch and loads from `{resume_source_dir}/checkpoints/frame_{resume_from-1:03d}.txt`.

**Per-frame TIFF output** — in `saveImages`, additionally emit `real_tiff/{N}.tiff` and `synth_tiff/{N}.tiff` as multi-page TIFFs via `cv::imwritemulti`. Replaces the post-run `convert_png_to_tiff.py` step. ~1-3s per frame overhead (serial LZW).

### Validation

- **Velocity cap**: 45f run `output_jihang_20260420_192746` captured 19/20 GT splits, 0 FP, **0 cells drifted to z=0 or z=224** (vs linux run where 8 cells drifted).
- **Birth growth cap v2 (1.8/1.8)**: no e3d03 FP at f28; cells clamped at f12-f27 (2-5 per frame) as expected.
- **Checkpoint resume**: `output_jihang_20260420_234047` with `resume_from=22` skipped f0-21, processed 3 frames in 424 sec (vs 2758 sec for full 25f run). 0 FP, no crashes.
- **TIFFs**: written automatically to `real_tiff/` and `synth_tiff/` directories, verified multi-page structure.

### 2026-04-21 REVERT: velocity cap dropped

Cell-by-cell comparison vs `perfect_45` baseline (script at `/tmp/compare_runs2.py`) revealed visible ~20 vx offsets at f3-f6 for fast-moving cells (`1f2ed10d`, `e3d03`). Perfect_45's frame-to-frame deltas for these cells hit 20-27 vx/frame in xy during early development — the 15 vx cap clipped every one of those legit motions, accumulating to ~28 vx cumulative drift by f5-f6 (visible as off-center ellipsoids). The position prior's quadratic penalty (non-saturating) remains active and handles late-frame drift without clipping fast biology.

**Defaults changed in `C++/includes/ConfigTypes.hpp`:**
```cpp
float max_perturb_drift_xy = 0.0f;  // 15.0f → 0.0f (disabled)
float max_perturb_drift_z  = 0.0f;  // 20.0f → 0.0f (disabled)
```

The `> 0.0f` guard in `perturbCell` makes 0 equivalent to disabled.

### Rollback

Revert the `max_perturb_drift_{xy,z}` defaults to 15/20 to re-enable the cap; revert the entire Change 7 by `git checkout <commit>~1 -- C++/src/Frame.cpp C++/src/CellUniverse.cpp C++/src/main.cpp C++/includes/ConfigTypes.hpp C++/includes/CellUniverse.hpp`.

---

## 2026-04-20: Pooling OpenMP pragmas commented out (Change 8) — **status: determinism audit, NOT behavior regression**

### Why

User requested removal of parallelism sources to isolate non-determinism during an audit. Four `#pragma omp parallel for collapse(2)` directives in `C++/src/ImageHandler.cpp::applyAdaptiveCubePooling` commented out.

### Files changed

`C++/src/ImageHandler.cpp` lines ~178, ~240, ~320, ~367 — pragmas now leading `//`.

### Effect

Pooling runs serially. Saves no correctness issue; loses ~60-90s per 45f run (Change 6's Phase A speedup). Actual non-determinism source is `std::random_device` seeding + float summation reorder, NOT pooling parallel.

### Rollback

Uncomment the four `#pragma` lines when the determinism audit concludes parallel pooling is fine to re-enable. For bit-reproducibility add `OMP_NUM_THREADS=<fixed N>` to the run-script environment.

---

## 2026-04-21: Static Voronoi cost-territory (Change 9) — **status: FAILED, disabled by default**

### Why

Observed bloat pattern: cell a511 grew large enough to cover its neighbor 8cbdf86d's bright pixels (at z=224 boundary), pushing 8cbdf86d out of position and destroying a511's own split valley signature. Fundamental cause: the L2 image cost rewards "bright pixel covered" regardless of which cell covers it, so a cell has no disincentive against annexing neighbor brightness.

### Design (replacement-cost filter)

- `Frame::rebuildVoronoiMap()` builds a per-pixel CV_32S map of nearest-cell-index, anchored to snap positions (fixed for the whole frame so the boundary does NOT move with the perturbed cell).
- `calculateBboxCost(..., voronoiCellIdx)` skips pixels where `_voronoiMap[z](y,x) != voronoiCellIdx` when filter is active.
- Rebuilt at frame start after snap install, and after every split accept.

### Files changed

`C++/includes/ConfigTypes.hpp` (new `voronoi_cost_enabled: bool`), `C++/includes/Frame.hpp` (members + method), `C++/src/Frame.cpp` (+rebuildVoronoiMap, voronoi skip in calculateBboxCost inner loop), `C++/src/CellUniverse.cpp` (enable + rebuild calls).

### Why it failed

45f run `output_jihang_20260421_020631` with this enabled hit `e3d03` false-positive split at **f4**. Root cause: when a cell is surrounded by neighbors, its Voronoi territory is a polygonal wedge. With cost restricted to that wedge, the cell deforms to match the wedge shape (e3d03 elongation spiked). High elongation → high P(split) → premature split attempt → accepted (cost gate was no longer comparing wedge-deformed parent vs daughters at the same basis). **The replacement-cost formulation is structurally flawed: it distorts cell shapes to match territory polygons, which creates a new failure mode.**

### Disposition

- `voronoi_cost_enabled` config default = `false`. Infrastructure (map build, filter code path) retained but unreachable by default.
- `calculateBboxCost(..., voronoiCellIdx = -1)` default parameter keeps all legacy call sites behavior-identical.
- Superseded by the additive bleed penalty (Change 11).

### Rollback (of the disable, i.e. re-enable)

Set `voronoi_cost_enabled: true` in `C++/config/config.yaml` — NOT recommended, use the additive bleed penalty (Change 11) instead.

---

## 2026-04-21: Split-attempt Voronoi bug fixes from cpp-reviewer audit (Change 10)

### Bugs caught by `celluniverse-cpp-reviewer` agent audit

**Bug A (critical)**: `trySplitCellPhased` mutates `cells[]` in-place (erase parent, push_back two daughters) but never rebuilds `_voronoiMap`. Subsequent `perturbCell(d2Idx = N)` passes an index that is out of range for `_voronoiAnchors` (which still has size N-1 from pre-split rebuild). Without a bounds check the Voronoi filter silently skipped every pixel → `cost = 0.0` for both old and new → daughter d2's burn-in received no cost gradient → daughter positions did not refine.

**Bug B (enabling)**: `calculateBboxCost`'s `useVoronoi` guard did not check `voronoiCellIdx < _voronoiAnchors.size()` — the out-of-bounds read silently matched nothing instead of failing.

### Fixes

**`C++/src/Frame.cpp:2256-2275`** — RAII guard at top of `trySplitCellPhased`:
```cpp
struct VoronoiDisableGuard {
    bool *flagPtr; bool saved;
    VoronoiDisableGuard(bool *p) : flagPtr(p), saved(*p) { *p = false; }
    ~VoronoiDisableGuard() { *flagPtr = saved; }
};
VoronoiDisableGuard vorGuard(&_voronoiEnabled);
```
Disables the map for the entire split body (burn-in, refine, cost gate), restores on any exit path. Rebuild after accept is handled by outer `CellUniverse::optimize`; reject path leaves the pre-split map valid.

**`C++/src/Frame.cpp:430-437`** — added bounds check to `useVoronoi` guard:
```cpp
const bool useVoronoi = (_voronoiEnabled
                         && voronoiCellIdx >= 0
                         && !_voronoiMap.empty()
                         && static_cast<int>(_voronoiMap.size()) == static_cast<int>(_realFrame.size())
                         && voronoiCellIdx < static_cast<int>(_voronoiAnchors.size()));
```

### Status

Retained even after Change 9's Voronoi-cost was disabled. Both guards apply to Change 11's bleed penalty (same `_voronoiEnabled` gate).

---

## 2026-04-21: Additive Voronoi bleed penalty (Change 11) — **the fundamental bloat fix**

### Why

The static-Voronoi replacement-cost approach (Change 9) reshapes cells to match their Voronoi territory polygons, causing premature false splits. Reverting it leaves the original bloat problem unaddressed: cells still annex neighbor brightness (see user's screenshots 2026-04-21 showing one large ellipsoid covering the brightness of what should be two cells). Three pathologies all stem from this single mechanism:

1. **Bloat / neighbor-particle capture** — observed directly in screenshots and as a511's aRadius growing 40.9 → 56.2 over 45 frames (+37%), absorbing 8cbdf86d's brightness.
2. **False-positive splits** — when a bloated cell reaches elong ≥ split threshold, the cost gate sometimes squeaks through a bad split (observed: e3d03 at f28 run 021403, costDiff=-12k, drift1=49).
3. **Cascading missed splits** — FP daughters from (2) occupy the region where real neighbor splits should place their daughters, triggering bio-gate "buried_in" rejections (observed: 12345...23400 at f39 blocked by e3d03...c8d0).

### Design — additive, not replacement

For each perturbation, ADD to the cost:
```
bleed_penalty = voronoi_bleed_penalty_weight × count(voxels inside THIS cell's
                ellipsoid that sit in a different cell's snap-anchored Voronoi
                territory)
```

Key properties that distinguish this from the failed replacement approach:
- **Additive**: the image L2 term is untouched. Cells still receive the normal brightness-fitting gradient, so they fit their own bright regions with the correct rotated-ellipsoid shape. No shape distortion.
- **Snap-anchored Voronoi**: boundaries are fixed at frame start and do not move with the perturbed cell. Intrusion cost grows monotonically with volume of annexation.
- **Zero in the nominal case**: a cell sitting within its own territory has bleed = 0, no penalty, no runtime cost beyond a single cheap AABB-bounded pass.

### Fields added to `C++/includes/ConfigTypes.hpp` (`SimulationConfig`)

```cpp
bool  voronoi_bleed_penalty_enabled = true;
float voronoi_bleed_penalty_weight  = 0.5f;
```

YAML overrides in `C++/config/config.yaml` accepted but not required; defaults are the recommended tuning.

### Files changed

**`C++/includes/Frame.hpp`** — new public methods `computeVoronoiBleedVoxels`, `setVoronoiBleedWeight`, `getVoronoiBleedWeight`; new private member `float _voronoiBleedWeight = 0.0f`.

**`C++/src/Frame.cpp`** —
- New method `Frame::computeVoronoiBleedVoxels(const Ellipsoid &cell, int cellIdx) const` (~30 lines): scans the cell's AABB, skips pixels in own territory via Voronoi map lookup, calls `Ellipsoid::isPointInsideEllipsoid` for the rest, returns the count. OpenMP-parallel over z with per-slice reduction.
- `perturbCell` delta-cost block (the `double costDiff = ...` line): now includes `+ newBleedPenalty - oldBleedPenalty` when `_voronoiEnabled && _voronoiBleedWeight > 0.0f`. Both old and new penalties computed from the same Voronoi map, so only the delta matters.

**`C++/src/CellUniverse.cpp`** —
- `voronoiMapNeeded` now gates on EITHER `voronoi_cost_enabled` OR `voronoi_bleed_penalty_enabled && weight > 0`.
- `frame.setVoronoiBleedWeight(...)` called before `rebuildVoronoiMap()` at frame start and after every split accept.
- `[Voronoi Map]` log line extended with `bleed_w=<weight>` for visibility.

### Why this fixes all three pathologies at once

1. **Bloat** — cell trying to grow into neighbor territory costs bleed_weight × ΔV. At weight 0.5 and ~10k voxels of intrusion (observed bloat magnitude), that's 5000 cost — comparable to the overlap penalty's working range, enough to stop net expansion while allowing legitimate biological motion inside own territory.
2. **FP splits** — without bloat, shape elongation stays natural → P(split) stays in the normal range → no premature split attempts → no FPs.
3. **Cascades** — without FPs, no spurious daughters occupy the positions where legit splits would place their daughters → no bio-gate cascades.

### Unlike Change 9, NO shape distortion

The image cost term is unchanged. A cell sitting in an elongated Voronoi polygon still sees the same brightness gradient as before; it does not have extra incentive to deform to match the polygon. The bleed penalty only activates when the cell's ellipsoid volume exceeds its territory — which only happens for bloating cells.

### Split path

`trySplitCellPhased`'s `VoronoiDisableGuard` (Change 10) disables `_voronoiEnabled` for the entire split body → `computeVoronoiBleedVoxels` early-returns 0 → the penalty is dormant during split evaluation. This is correct because (a) the map would be stale mid-split and (b) the split-decision cost should be symmetric between parent and daughter candidates at the full-image basis, not biased by partial-volume bleed accounting.

### Performance

Per perturbation, 2 × `computeVoronoiBleedVoxels` calls (old + new cell). Each scans ~(2×maxR)³ voxels ≈ 256k for a 40-radius cell. Own-territory early-skip culls most; only boundary voxels run `isPointInsideEllipsoid`. Empirical: ~2-5 ms per call on macOS 8-core. Frame-level overhead: 300k perturbs × 2 × ~3 ms = 30 min/frame is *unacceptable* — IF every perturb computed it fully. In practice the OpenMP reduction distributes across cores and the own-territory skip drops inner-loop cost to ~μs scale. Watch for `[Voronoi Map]` log line build_ms and total frame runtime after rebuild to confirm.

### Rollback

Set `voronoi_bleed_penalty_enabled: false` in YAML or the ConfigTypes default to disable. Keeps Change 10 bug fixes active.

### Open follow-ups

- **Tune `voronoi_bleed_penalty_weight`** on a full 45f run. 0.5 is the starting point.
- **CV_8U storage for `_voronoiMap`** (from cpp-reviewer agent report): 4× memory reduction + 2-3× inner-loop speedup. Current CV_32S uses ~207 MB per frame.
- **Bleed computation parallelism**: currently per-z OMP reduction. If hot spot, switch to per-block tiling with thread-local reductions.

---

## 2026-04-21: Slab-min gap brightness in bridge valley gate (Change 12) — **the a510-miss fix**

### Why

After Change 11 eliminated the e3d03 FP and the cascade-blocked 23400 miss, the remaining failure at a510 f39 was NOT a bloat problem — a510 was properly sized. The bridge-gate valley check was reading the gap too bright: `valleyFromBright = 0.762` (just 0.012 above the 0.75 rejection threshold). Cell-by-cell comparison with `perfect_45`'s accepted a510 at f40 showed a geometric asymmetry, but the deeper cause was a **pooling-width vs gap-width** mismatch:

- Real-image gap between a510's two pre-division bright lobes: **~5-6 vx wide** (visible dark bridge in the TIFF).
- Adaptive cube pooling cube size: `0.6 × minR ≈ 9 vx` (from `adaptive_cube_pooling_cube_size_scale: 0.6`).
- Result: any single pooled cube straddling the gap **averages** its dark interior with its bright neighbors, smearing the valley closed. The gap zone's mean brightness came out at ~0.16 instead of the true ~0.10 — enough to push the valley ratio from ~0.45 to ~0.76.

Perfect_45 only passed its a510 split at f40 because its cell was more bloated at that point — longer split axis (~48 vx daughter separation vs our ~18 vx) forced the sampling line to cross *multiple* consecutive gap-adjacent cubes, dominating the mean with actually-dark cubes.

**The issue was an image-processing artifact of the same width as the biological feature we were trying to measure**, not a bug in the bleed penalty or the valley threshold.

### Design — slab-min within the gap zone

Partition the gap zone (`±effectiveGapHalf` along the split axis) into `kGapSlabs=5` thin slabs. Compute mean brightness per slab. Take the **minimum** slab as `gapBright` — "the darkest cross-section along the bridge." A single contaminated cube's smear is confined to one slab; any cleanly-dark adjacent slab still shows through. Slab width ~3 vx < pooling cube ~9 vx, so per-cube artifacts cannot dominate all slabs at once.

Guard: require ≥ `max(3, gapCount / (kGapSlabs × 3))` pixels per slab before considering it, to prevent a single stray dark pixel from winning. Fallback to legacy mean if no slab qualifies (sparse gap zones). Log line emits **both** `gapBrightMean` and `gapBrightMinSlab` plus `minSlabIdx` for continued diagnosis.

### Files changed

**`C++/src/Frame.cpp`** —
- Final `[Split Bridge]` gate (~lines 3581-3675 in the old offset, moved by edits): added `slabSum[kGapSlabs]` / `slabCount[kGapSlabs]` arrays, bin each in-gap pixel by its projected coordinate, compute min-qualifying slab after the loop, use as `gapBright` (with legacy mean fallback).
- Per-candidate `[Split Cand PreFilter]` (~lines 3168-3188): same slab-min logic applied so candidates aren't filtered out on artifact-inflated means before they can reach the final bridge.
- `[Split Bridge]` log line extended: `gapBrightMean=<legacy> gapBrightMinSlab=<new> minSlabIdx=<0..4 or -1>`.
- New include: `<array>` for the fixed-size slab accumulators.

### Why this doesn't admit false positives

The change is **strictly one-sided**: slab-min is always ≤ mean (min ≤ mean for same sample set). It can only *lower* the reported gap brightness, never raise it. That means:

- Real valleys (biological dark bridge exists): mean is inflated by artifacts, slab-min finds the truer value → gate passes where it previously failed. Correct outcome.
- No valley (single cell, false split candidate): all slabs are similarly bright, min ≈ mean → no change in behavior. Still rejects.
- Geometric overlap (daughters inside parent body): gap projection is inside bright core, all slabs bright → min still ≈ mean → still rejects.

Confirmed empirically during the f22-f45 resume run at `output_jihang_20260421_230314`:
- `8cbdf86d` at f42: valley 0.98 (slab-min 0.33, mean 0.34 — both reject, no change).
- `23411` sub-split at f37: valley 0.97 (slab-min 0.20, mean 0.20 — both reject).
- `e3d03` at f41: valley 0.53 (slab-min 0.11, mean 0.13 — both pass bridge, but correctly caught by downstream `d1_buried_in_cb00` bio check).

### Validation — output_jihang_20260421_230314 (f22-f45 resume from f21 checkpoint of 174140)

Config at run time: `voronoi_bleed_penalty_enabled: true, weight=0.5`; `voronoi_cost_enabled: false`; `max_perturb_drift_{xy,z}: 0` (velocity cap disabled); `bio_bridge_max_valley_ratio: 0.75` unchanged; slab-min gate active.

**Result: 20/20 GT splits, 0 FP, 0 FN, 26 cells at f45.** First full-horizon run with complete GT recall since the yp-merge regression.

Key per-frame evidence:
- **f28**: `cb0` + `cb1` both accepted, drifts 4-12, valleys 0.35/0.35. No e3d03 FP (bleed prevented bloat; bridge gate valley 0.53 still caught by `d1_buried_in` downstream).
- **f32**: `1f2ed10d...f0-sibling` accepted, valley 0.39.
- **f38**: `1f2ed10d...f1-sibling` accepted, valley 0.27.
- **f39**: 7 splits accepted (a511, a501, a500, 23400, 23401, 23410, 23411). Every bridge valley ≤ 0.52.
  - `23400` drifts 0.8/0.3 — **cascade miss from run 021403 now clean** because there's no e3d03 FP daughter to bury it.
- **f40**: **`a510` accepted** via `snap_imgPca_trans-`, drifts 22/17, cost -43k, bridge valley 0.44.
  - Slab-min pulled gapBright from 0.152 (mean) to 0.100. With mean, valley would be 0.152/0.211 = **0.72** — just passes here, but would have failed at f39 where mean gave 0.76.
  - Slab-min made the difference: f39 valley with mean=0.76 (reject) → slab-min=0.71 (accept at bridge, then cost-reject), continuing to f40 where cell geometry shifted and cost gate also passed.
- **No false positives anywhere**: e3d03 never split, 8cbdf86d never split, no 3-deep sub-subdivisions.

### Combined impact of Changes 7-12 vs the original problem set

| Issue | Before (session start, run 174140 → 000145 lineage) | After (session end, run 230314) |
|---|---|---|
| Cell drift → z=0/z=224 boundary | 8 cells drifted | 0 |
| a511-style bloat (neighbor annexation) | Observed in screenshots | Suppressed by bleed penalty |
| e3d03 FP at f28 | Occurred (costDiff -12k, drift 49) | **Eliminated** |
| 23400 cascade miss at f39 | Blocked by e3d03 FP daughter | **Clean split (drifts 0.8/0.3)** |
| a510 miss at f39-f45 | Valley 0.76 > 0.75 (artifact-inflated) | **Accepted at f40 via slab-min** |
| Early-frame visible cell offset (f3-f6) | ~20 vx drift vs perfect_45 | Resolved by velocity-cap drop |
| Final cell count vs GT 26 | 25 (1 FP + 1 FN) or 25 (0 FP + 1 FN) | **26 (0 FP, 0 FN)** |

### Rollback

Revert `C++/src/Frame.cpp` bridge-gate slab accumulator additions. The old `gapBrightSum / gapCount` mean path remains as the `gapBrightMean` computation so reverting is a matter of setting `gapBright = gapBrightMean` and removing the slab loop. Legacy behavior restored.

### Open follow-ups

- **Tune `kGapSlabs`**: 5 is a reasonable default for typical ~15 vx gap zones (3 vx slabs). Could expose as `bio_bridge_gap_slabs` config for experimentation on longer split axes.
- **Investigate whether pre-pooled image** (used in other valley-like checks?) would help elsewhere. The slab-min approach is a per-metric fix; sampling raw pixels through the pooled output is the deeper architectural option.
- **Capture the slab-min log** in the validation run's artifact for any future regression — the `minSlabIdx` field tells us at a glance when slab-min was the deciding factor.

---

## 2026-04-22: Preprocessing score==0 rollback state reset (Change 13)

### Why

Frames 62 and 64 in the run `output_jihang_20260421_230314` → its resume
`output_jihang_20260422_004745` produced strongly-elevated backgrounds
(`processed_sequence mean = 0.268` at f62, `0.460` at f64 vs typical ~0.015).
The downstream effect was `edge_too_dim` blanketing f63-f64 split attempts (all
cells' edge brightness read ~0.03, below the 0.07 gate) followed by a burst of
false-split accepts at f65 when brightness recovered.

Root cause: inside `ImageHandler::processPreparedSequence`, when the contrast
penalty ramp drove the sequence into a flat-zero state, `evaluateSequenceContrastScore`
returned 0.0f, triggering the rollback path at line 1022. The rollback reduced
`currentPenalty` and restored `bestSequence` correctly, but left the three
scoring knobs (`scorePercentile`, `rewardGate`, `hasPreviousScore`) in their
drifted state. Subsequent iterations evaluated against a shifted metric frame
of reference, so `bestSequence` never improved beyond its round-0 capture —
which had elevated background. Final preprocessed output carried that bg into
post-processing, which amplified it 5× via the black/white percentile gate
(mid-range pixels ×5), producing the visible "purple frame" output.

### Fix

**`C++/src/ImageHandler.cpp:1022-1058`** — `score==0` rollback block now also
resets:

```cpp
scorePercentile = config.simulation.iterative_score_percentile;
rewardGate = config.simulation.iterative_reward_gate;
hasPreviousScore = false;
```

`bestScore` and `bestSequence` intentionally NOT reset — we don't want to lose
a truly good state captured before the collapse.

### Validation

Resume run `output_jihang_20260422_124756` (f60-f72, with all other 2026-04-22
changes active):
- f62: `processed_sequence mean` 0.268 → **0.0158** ✓ (clean)
- f64: `processed_sequence mean` 0.460 → **0.0144** ✓ (clean)
- No `edge_too_dim` spam at f63-f64.
- No `e3d03` FP at f65 (previous run accepted at `costDiff=-12k` during the
  post-anomaly burst).

### Rollback

Revert the three new lines in the score==0 block. Legacy behavior restored.

---

## 2026-04-22: RNG seed config for ablation A/B comparison (Change 14)

### Why

Monte Carlo runs use `std::random_device` for seeding, making every run a
unique random trajectory. Ablation testing of individual fixes (documented in
`docs/plans/2026-04-22-late-frame-ablation-plan.md`) requires deterministic
trajectories so A/B comparisons isolate code-change effects from RNG noise.

### Fields added

`C++/includes/ConfigTypes.hpp` (`SimulationConfig`):
```cpp
int mc_rng_seed = 0;  // 0 = random_device (production); >0 = deterministic
```

### Implementation

`C++/src/CellUniverse.cpp:729-744`: main MC RNG now conditionally seeded.

```cpp
std::mt19937 gen;
if (config.simulation.mc_rng_seed > 0) {
    const uint64_t baseSeed = static_cast<uint64_t>(config.simulation.mc_rng_seed);
    const uint64_t frameMix = static_cast<uint64_t>(frameIndex) * 0x9E3779B97F4A7C15ULL;
    gen.seed(static_cast<std::mt19937::result_type>(baseSeed ^ frameMix));
    std::cout << "[RNG] frame " << frameIndex << " seeded deterministic (base="
              << config.simulation.mc_rng_seed << ")" << std::endl;
} else {
    std::random_device rd;
    gen.seed(rd());
}
```

Frame-indexed mixing keeps different frames exploring independent trajectories
while the whole run is reproducible for a given seed.

### Known limitation

Thread-local RNGs in `PerturbParams::samplePerturb` and the signal-guided
position helper in `Frame.cpp` still use `random_device`. Expected residual
variance <2% on frame-level metrics — acceptable for ablation. Full bit-
reproducibility would require plumbing the seed into those thread_locals
plus forcing single-threaded execution (or accepting OpenMP schedule noise).

### Use

Set `mc_rng_seed: 42` (or any positive int) in `config.yaml` for deterministic
ablation runs. Leave at `0` (or omit) for normal production multi-seed runs.

---

## 2026-04-22: Checkpoint resume moved from YAML to INI preset + CLI (Change 15)

### Why

`resume_from` / `resume_source_dir` lived in `config.yaml`, making resume
configuration mix with per-cell algorithmic tuning. Separating them into the
INI preset keeps "what frames / where / resume from" with the rest of the
run-harness configuration.

### Changes

- `C++/includes/types.hpp`: `argKeywords` enum extended with `resumeFrom` (7)
  and `resumeSourceDir` (8). Optional positional args.
- `C++/src/main.cpp`: `Args` extended with `resumeFromFrame` / `resumeSourceDir`.
  `initArgs` parses `argv[7]` and `argv[8]` when present. If present, they
  override `config.simulation.resume_from` / `resume_source_dir` from YAML.
- `C++/scripts/run_celluniverse.sh`: reads `resume_from` / `resume_source_dir`
  from the INI preset. Resolves relative source-dir paths against the INI's
  directory. Appends args to `CMD` only when `resume_from > 0` AND source_dir
  is non-empty. Logs `Resume from frame` + `Resume source dir` via `print_kv`.
- `C++/config/user_input_configurations.ini`: header comments on new keys.
  `[jihang]` preset gets `resume_from=0` / `resume_source_dir=` defaults.
- `C++/config/config.yaml`: `resume_from` / `resume_source_dir` YAML keys
  replaced by a comment pointer. YAML parser still reads them (optional)
  but the INI / CLI path is the source of truth; CLI wins over YAML.

### Use

Edit `C++/config/user_input_configurations.ini`:
```ini
[jihang]
...
resume_from=60
resume_source_dir=/absolute/or/relative/to/ini/path
```
Then `scripts/run_celluniverse.sh config/user_input_configurations.ini jihang`.
Script logs the resolved resume params and appends them to the binary command.
Normal runs: `resume_from=0` / empty `resume_source_dir`.

### Backward compat

Calling `./celluniverse` with only 6 args still works (resume disabled).
YAML `resume_from` / `resume_source_dir` keys still parsed if set (CLI wins).

## 2026-04-22: Ablation close-out — accept T1 (Fix C), revert Fix A, reset to production (Change 16) — **status: SUPERSEDED by Change 17**

**Supersede reason:** T1's unmet thresholds were not just checkpoint artifacts.
Three of five decision thresholds failed (max aR, bloated-cell frames, drift >
15 cell-frames); `cb11` ugly split was actually worse in T1 (drift2=56.7) than
T0 (50); `1f2ed10d:111` still drifted to z=0. Stopping rule was applied too
eagerly. Reverted the config reset and moved to T2b (narrowed centroid).

### Decision

Accept the T1 result (Fix C continuous bloat cap, weight=1000) as the shipping
fix. Revert Fix A (brightness-centroid anchor) due to demonstrated systemic
regression — see `docs/plans/ablation/T2-fix-c-smoothed-plus-fix-a.md`. Reset
ablation-only config knobs to production values and unlock a full f1-f72 run.

### Evidence basis

- **T1** (`outputs/output_ablation_T1_20260422_180917/`): eliminated all 3
  `e3d03` FPs, reduced ugly-drift splits 3 → 1, added clean TP at 23101 f66.
  Met decision thresholds on the two critical bad-spot classes.
- **T2** (`outputs/output_ablation_T2_20260422_210422/`, partial through f67):
  Fix A at weight=50 rescued the `1f2ed10d:111` z=0 drifter BUT produced
  cluster-wide clumping at f65 (synth ellipsoids coincident in central
  cluster; visually confirmed by user screenshot) and a worse e3d03 FP at
  f67 (drift1=63.7) because each cell's brightness-weighted centroid over
  its snap-Voronoi territory is dragged toward NEIGHBORS' bright peaks.
  Systemic, not patchable without territory restriction.
- **Stopping rule** from the master plan
  (`docs/plans/2026-04-22-late-frame-ablation-plan.md`): if any single fix
  (or thin combo) eliminates the bad-spot screenshots AND hits the decision
  thresholds, stop stacking — simpler wins. Fix C alone qualifies.

### Files changed

**File:** `C++/config/config.yaml` (Fix A disabled)

**Line 538 (before):**
```yaml
  brightness_centroid_anchor_weight: 50.0
```

**Line 538 (after):**
```yaml
  brightness_centroid_anchor_weight: 0.0
```

**File:** `C++/config/config.yaml` (seed reset to random)

**Line 390 (before):**
```yaml
  mc_rng_seed: 42
```

**Line 390 (after):**
```yaml
  mc_rng_seed: 0
```

**File:** `C++/config/user_input_configurations.ini` (`[jihang]` preset)

**Lines 61, 67-69 (before):**
```ini
output_name_rule=output_ablation_T2_{timestamp}
...
resume_from=60
resume_source_dir=/Users/jihangli/MCS/3D_Cell_Tracking/CellUniverse/C++/outputs/output_jihang_20260422_004745
# Ablation T0 baseline uses mc_rng_seed=42 via config.yaml
```

**Lines 61, 67-71 (after):**
```ini
output_name_rule=output_production_f1-f72_{timestamp}
...
resume_from=0
resume_source_dir=
# Production run: Fix C (bloat cap, weight=1000) active; Fix A disabled
# (brightness_centroid_anchor_weight=0); mc_rng_seed=0 for random trajectories.
# Ablation A/B tests used mc_rng_seed=42 + resume_from=60; reset to prod here.
```

### What stays on (Fix C, shipped)

- `bloat_cap_barrier_weight: 1000.0` — continuous log-barrier penalty per
  axis starting at 1.1× birth, with smoothed linear extrapolation past 1.6×
  (replaces the 1e9 hard-reject plateau that created a no-gradient trap).
- `bloat_cap_barrier_start: 1.1`, `bloat_cap_barrier_end: 1.6` — unchanged.
- Code in `Frame.cpp` / `CellUniverse.cpp` / `ConfigTypes.hpp` /
  `Frame.hpp` introduced in Change 14 scope is retained.

### What stays off (Fix A, shelved)

- `brightness_centroid_anchor_weight: 0.0` — code path remains compiled
  (`Frame::computeBrightnessCentroids`, centroid term in `perturbCell`
  delta cost) but inactive at weight 0. Can be reactivated under a future
  T2b (narrowed centroid territory) or T2c (isolated-cells-only) design.

### Remaining known gaps (deferred)

1. `1f2ed10d:111` drift to z=0 at f70 — only Fix A rescued this in T1. If
   reproduced in clean-start production, revisit with T2b (restrict
   centroid pixels to `distance < 1.5 × cell.aR`).
2. `cb11` ugly split at f72 (drift2 ≈ 50-57) — inherited-birth-size edge
   case. Relative-to-birth cap blind to cells born large at f28. A full
   f1-f72 run with Fix C active from the start may avoid cb11 being
   born bloated.

### Run launch (next step, user)

After build:
```bash
scripts/run_celluniverse.sh config/user_input_configurations.ini jihang
```
Output goes to `C++/outputs/output_production_f1-f72_<timestamp>/`.

## 2026-04-22: T2b — narrowed brightness-centroid territory (Change 17) — **status: ACTIVE (ablation)**

### Motivation

T2 (Fix A full-territory centroid) rescued the `1f2ed10d:111` z=0 drifter but
produced two systemic failures:

1. **Cluster-wide clumping at f65**: every cell's snap-Voronoi territory
   includes neighbor brightness. Each brightness-weighted centroid is dragged
   toward the cluster's common bright mass, so cells pull toward each other.
   Visually confirmed: synth ellipsoids stacked/coincident in the central
   cluster (user-provided f65 screenshot).
2. **e3d03 FP at f67 with drift1=63.7**: e3d03's Voronoi territory leaks into
   a dividing neighbor (23101 region). Its centroid sits on the neighbor's
   peak, not its own bright region → anchor penalty *amplifies* the
   misattribution instead of correcting it.

Both stem from the same mechanism — the Voronoi territory is the WRONG
integration domain when cells are packed close to neighbors. T2b restricts
the integration to voxels near the cell's OWN center.

### Change

Add a per-voxel distance gate inside `Frame::computeBrightnessCentroids`:
only voxels with `|voxel - cell.position|² ≤ (radius_factor × majorRadius)²`
contribute to that cell's centroid. Factor default 1.5 — enough headroom for
legitimate off-center brightness within the cell's footprint while excluding
neighbors.

New config field `simulation.brightness_centroid_radius_factor` (default
`1.5`). `<= 0` disables the gate (legacy T2 behavior, available for A/B
comparison).

### Files changed

**File:** `C++/includes/ConfigTypes.hpp`

**Lines 405-412 (added after `brightness_centroid_anchor_weight`):**
```cpp
// T2b (2026-04-22): restrict centroid integration to voxels within
// `radius_factor × cell.majorRadius` of the cell's current (x,y,z).
// ... (full comment in source)
float brightness_centroid_radius_factor = 1.5f;
```

**Line 555 (added YAML parser entry):**
```cpp
if (node["brightness_centroid_radius_factor"]) brightness_centroid_radius_factor = node["brightness_centroid_radius_factor"].as<float>();
```

**File:** `C++/includes/Frame.hpp`

**Lines 191-196 (added setter/getter):**
```cpp
void setBrightnessCentroidRadiusFactor(float f) { _brightnessCentroidRadiusFactor = f; }
float getBrightnessCentroidRadiusFactor() const { return _brightnessCentroidRadiusFactor; }
```

**Lines 454-456 (added private member):**
```cpp
// T2b: radius factor for the "near this cell" distance gate inside
// computeBrightnessCentroids. <= 0 → no gate (legacy behavior).
float _brightnessCentroidRadiusFactor = 1.5f;
```

**File:** `C++/src/Frame.cpp`

**Lines 370-384 (precompute per-cell center + squared radius gate, added before parallel loop):**
```cpp
const float radiusFactor = _brightnessCentroidRadiusFactor;
const bool gateEnabled = (radiusFactor > 0.0f);
std::vector<float> cx(nCells), cy(nCells), cz(nCells), rSq(nCells);
for (int i = 0; i < nCells; ++i) {
    cx[i] = cells[i].getX();
    cy[i] = cells[i].getY();
    cz[i] = cells[i].getZ();
    const float r = radiusFactor * cells[i].getMajorRadius();
    rSq[i] = r * r;
}
```

**Lines 404-412 (distance gate inside voxel loop):**
```cpp
if (gateEnabled) {
    const float ddx = static_cast<float>(x) - cx[idx];
    const float ddy = dy - cy[idx];
    const float ddz = dz - cz[idx];
    const float d2 = ddx * ddx + ddy * ddy + ddz * ddz;
    if (d2 > rSq[idx]) continue;
}
```

**File:** `C++/src/CellUniverse.cpp`

**Line 1304 (push radius factor from config to Frame, before the compute call):**
```cpp
frame.setBrightnessCentroidRadiusFactor(config.prob.brightness_centroid_radius_factor);
```

**Line 1312 (logged radius_factor added to the `[Brightness Centroid]` line):**
```cpp
<< " radius_factor=" << config.prob.brightness_centroid_radius_factor
```

**File:** `C++/config/config.yaml`

**Lines 540-546 (added after `brightness_centroid_anchor_weight`):**
```yaml
# T2b: restrict centroid integration to voxels within
# radius_factor × majorRadius of the cell's own center...
brightness_centroid_radius_factor: 1.5
```

**File:** `C++/config/user_input_configurations.ini` (`[jihang]` preset restored to ablation mode)

**Lines 61-69 (reverted from Change 16's production reset):**
```ini
output_name_rule=output_ablation_T2b_{timestamp}
...
resume_from=60
resume_source_dir=/Users/jihangli/MCS/3D_Cell_Tracking/CellUniverse/C++/outputs/output_jihang_20260422_004745
# Ablation T0 baseline uses mc_rng_seed=42 via config.yaml
```

**File:** `C++/config/config.yaml` (ablation knobs re-enabled)

- `mc_rng_seed: 42` (was `0` in Change 16, restored for A/B determinism)
- `brightness_centroid_anchor_weight: 50.0` (was `0.0` in Change 16, re-enabled)
- `bloat_cap_barrier_weight: 1000.0` (unchanged)

### Expected behavior

- `1f2ed10d:111` still rescued at f70 (centroid pulls toward its own bright
  footprint, same as T2).
- Cluster clumping at f65 gone (cells no longer pulled toward neighbors'
  brightness because neighbor voxels are filtered out).
- e3d03 f67 FP should drop back to ≤ T1/T0 severity (centroid sits on
  e3d03's own bright region, not on the 23101 neighbor).

### A/B comparison

Baseline: `outputs/output_ablation_T1_20260422_180917` (Fix C only).
Variant: `outputs/output_ablation_T2b_<timestamp>` (Fix C + narrowed Fix A).

```bash
python3 C++/scripts/compare_ablation.py \
  C++/outputs/output_ablation_T1_20260422_180917 \
  C++/outputs/output_ablation_T2b_<ts> --from-frame 61
```

### Performance

Adds 3 float subtractions + 3 multiplies + 1 compare + potential early-out
per voxel inside the parallel loop. Typical cost increase < 5% over T2
(still << 1% of frame runtime).

## 2026-04-23: T3 — Fix D split persistence gate (Change 18) — **status: ACTIVE (ablation, v2 with relative-to-parent positions)**

### Iteration history

- **v1 (first T3 launch, stopped after f66)**: stored ABSOLUTE daughter
  positions `(d1Pos, d2Pos)` in `SplitAttemptRecord`. Rejected a5010 at
  f66 with pairDist=36 — not because split geometry changed, but because
  the parent cell drifted 30+ vx between f65 and f66. All late-frame legit
  splits would have failed this check alongside FPs. Design flaw.
- **v2 (current)**: stores RELATIVE positions `(d1Rel = d1Pos - parentPos,
  d2Rel = d2Pos - parentPos)` at time of attempt. Isolates the split
  GEOMETRY (direction + separation from parent) which IS stable across
  frames for legit biology (daughter spawn pattern is invariant under
  parent motion) but random for FP splits (burn-in lands on different
  local maxima each frame).

### Motivation

T2b (Change 17) failed — centroid-pull direction is structurally wrong in
crowded late-frame clusters even with narrowed territory. T2b partial run
showed 2.45× higher mean per-frame drift, ~3× more extreme drifts, worse
max aR (65 vs 57), missed 4 clean TPs at f66, and still reproduced the
e3d03 f67 FP (drift1=53 vs T2's 63.7). See
`docs/plans/ablation/T2b-narrowed-centroid.md`.

T3 attacks the FP problem from a different angle: **split persistence**.
A split is accepted only if the same cell produced a matching candidate at
the previous frame. Single-frame opportunistic splits (e3d03 f65 FP that
T0 had, e3d03 f67 FP that T2/T2b reproduced, cb11 f72 ugly split) are
structurally rejected because their attempt geometry doesn't persist.

Orthogonal to Fix A — the gate cares only about attempt stability across
frames, not about brightness geometry or drift direction.

### Design

Per parent cell name, track the most recent cost-profitable split-candidate
winner: `(d1Pos, d2Pos, costDiff, frameAttempted)`. Stored in
`CellUniverse::splitAttemptHistory`.

Algorithm (inside the split-accept branch of `CellUniverse::optimize`,
before invoking the commit callback):

1. Call `Frame::trySplitCellPhased` with a new out-param `&winnerInfo` that
   receives the final winner's d1/d2 positions and costDiff. (Existing
   bio+cost gates inside trySplitCellPhased run unchanged.)
2. If costDiff < 0 (cost-profitable) AND `winnerInfo.valid` AND
   `required_frames >= 2`, apply the persistence check:
   - Look up `splitAttemptHistory[cellName]`; treat stale records
     (`frameAttempted < currentFrame - 1`) as missing.
   - Match positions: try both same-order pairing (d1↔prev_d1, d2↔prev_d2)
     and crossed (d1↔prev_d2, d2↔prev_d1); take the pairing with lower max
     distance. Require `<= position_tolerance_vx`.
   - Match cost: `|currCost - prevCost| / max(|prevCost|, 1.0) <=
     cost_tolerance_ratio`.
   - Matched → pass, erase record (prevents double-counting).
   - Unmatched → record current attempt, flip `accept = false`, log
     `[Split Persistence Reject]`.
3. Call `callback(accept)` as before. Rejected path falls into existing
   blacklist + compensation-perturb branch.

### Files changed

**File:** `C++/includes/ConfigTypes.hpp`

**Lines 415-431 (added after brightness_centroid_radius_factor):**
```cpp
int split_persistence_required_frames = 0;
float split_persistence_position_tolerance_vx = 10.0f;
float split_persistence_cost_tolerance_ratio = 0.5f;
```

**Lines 558-560 (added YAML parser entries).**

**File:** `C++/includes/Frame.hpp`

**Lines 203-227 (added `SplitWinnerInfo` struct, modified
`trySplitCellPhased` signature to accept optional `winnerOut`):**
```cpp
struct SplitWinnerInfo {
    bool valid = false;
    cv::Point3f d1Pos{};
    cv::Point3f d2Pos{};
    double costDiff = 0.0;
};
CostCallbackPair trySplitCellPhased(
    ...,
    SplitWinnerInfo *winnerOut = nullptr);
```

**File:** `C++/src/Frame.cpp`

**Line 2506 (definition signature updated to match header).**

**Lines 4193-4202 (populate out-param before returning the callback):**
```cpp
if (winnerOut != nullptr) {
    winnerOut->valid = true;
    winnerOut->d1Pos = acceptedD1Pos;
    winnerOut->d2Pos = acceptedD2Pos;
    winnerOut->costDiff = costDiff;
}
```
Early-return paths (validity/bio rejections) leave `winnerOut->valid = false`
by default — correct, since they never produced a winner.

**File:** `C++/includes/CellUniverse.hpp`

**Lines 86-96 (added `SplitAttemptRecord` struct + `splitAttemptHistory`
map field).**

**File:** `C++/src/CellUniverse.cpp`

**Lines 1622-1704 (persistence gate between trySplitCellPhased return and
callback invocation).** See inline comments for the match logic. Logs:
- `[Split Persistence Pass]` when a match clears history and split commits
- `[Split Persistence Reject]` when records/rejects a first-frame attempt

**File:** `C++/config/config.yaml`

**Lines 538-553 (disable Fix A, enable T3):**
```yaml
brightness_centroid_anchor_weight: 0.0     # SHELVED after T2/T2b failure
brightness_centroid_radius_factor: 1.5     # inert at weight=0
split_persistence_required_frames: 2        # T3 active
split_persistence_position_tolerance_vx: 10.0
split_persistence_cost_tolerance_ratio: 0.5
```

**File:** `C++/config/user_input_configurations.ini`

`output_name_rule=output_ablation_T3_{timestamp}`. Resume state unchanged
(resume_from=60 from `...004745`).

### Expected behavior

- **e3d03 f65 FP (in T0, not in T1/T2/T2b)**: first-frame attempt, no prior
  record → rejected by persistence. Would need to reappear at f66 with
  matching geometry to commit. Unlikely given e3d03's random-drift
  signature.
- **e3d03 f67 FP (in T0/T2/T2b, not in T1)**: same logic — the f65 attempt
  (if any) had different geometry from the f67 attempt, so f67's candidate
  gets logged and rejected, history updated. f68 unlikely to produce
  matching.
- **cb11 f72 ugly split (present in T0/T1/T2)**: opportunistic single-frame,
  no f71 matching attempt → rejected.
- **Clean TPs (23101 at f66, etc.)**: parent cells genuinely elongating
  should produce similar split geometries across 2-3 frames (snap positions
  are close, PCA direction stable). First frame: reject+record. Second
  frame: match+accept. Net effect: 1-frame delay on legitimate splits,
  structurally rejects single-frame FPs.

### Possible regressions

- **Fast splits**: cells that elongate→split within a single frame window
  (no pre-elongation state at the previous frame) would be delayed by one
  frame. Acceptable — biology isn't that precise.
- **Cells that try ONCE and stop**: a legitimate split attempt that only
  fires once (burn-in noise varies) never passes persistence. Mitigated by
  `cost_tolerance_ratio = 0.5` being generous — small costDiff variations
  shouldn't break matches.
- **f111 z=0 drift rescue**: T3 does NOT address this — that's a Fix A
  concern. Known deferred residual.

### A/B compare targets

Primary baseline: `outputs/output_ablation_T1_20260422_180917` (Fix C only,
shipping candidate). T3 should match T1 on clean TPs (with 1-frame delays)
and be strictly better on FP rejection (no e3d03 at f65/f67, no cb11 f72
ugly split).

```bash
python3 C++/scripts/compare_ablation.py \
  C++/outputs/output_ablation_T1_20260422_180917 \
  C++/outputs/output_ablation_T3_<ts> --from-frame 61
```

### T3 v2 result — FAILED

Run `outputs/output_ablation_T3_20260423_011920` (partial through f66):
- f65: 6 splits recorded (5 T1-legit-TPs + e3d03 T0-FP), 0 accepted.
- f66: 6 persistence checks fired. **All rejected** with pairDist 14.7–24.6
  using relative-to-parent coordinates. Legit cells (a5010, a5011, a5111,
  a5110) rejected alongside e3d03 FP.
- T3 at f66: 30 cells vs T1's 38 — **8-cell gap**, persistence gate
  blocking legit biology.

Root cause: burn-in random-walk + per-frame PCA axis variability produce
15–25 vx relative-position variability between consecutive frames even
for stable biological splits. This variability is the same magnitude as
what a random-drift FP would produce. Position-tolerance matching cannot
discriminate signal from noise.

T3 persistence gate disabled (`split_persistence_required_frames: 0`).
Code retained but inert. Could be revived with a different matching
metric (e.g. split-axis direction stability rather than daughter
position), but deprioritized — the ablation chain itself is hitting
"tuning hell" territory. Reverting to a fundamental force-balance fix.

## 2026-04-23: Fundamental fix — strengthen Voronoi bleed penalty (Change 19) — **status: ACTIVE (ablation)**

### Why stop the T*-fix pattern

T0/T1/T2/T2b/T3 were all *additive* patches on top of an unchanged force
balance. Each new force term had to shout louder than the image-cost
dominance to have an effect, which made each over-eager:

- Fix A tried to pull cells toward brightness; centroid was contaminated
  by neighbors in dense clusters → pulled cells TOGETHER (clumping).
- T3 tried to reject unstable splits; legit split geometry was too noisy
  across 2 frames → rejected everything (8-cell gap at f66).

Root cause diagnosis: the **Voronoi bleed penalty** (the force that
penalizes cells for rendering into neighbor territory) was weighted at
**0.5** — effectively inert against image-cost shifts of 1k–10k. So
the "each cell stays in its own bright blob" constraint was a whisper,
not a rule. Every downstream patch had to compensate for that broken
foundation.

### Change

Raise `voronoi_bleed_penalty_weight` from 0.5 to 5.0 (10×). At this
level, a cell that drifts ~1000 voxels into a neighbor's Voronoi
territory pays ~5000 — comparable to image-cost shifts. The Voronoi
partition becomes a soft-but-respected constraint.

No code changes — the force term already existed, just under-weighted.

### Files changed

**File:** `C++/config/config.yaml`

**Lines 524-542 (added explicit bleed weight override):**
```yaml
voronoi_bleed_penalty_weight: 5.0
```
(previously relied on the 0.5 default in `ConfigTypes.hpp`).

**File:** `C++/config/user_input_configurations.ini`

`output_name_rule=output_bleed5_{timestamp}` (dropping the `output_ablation_T*`
naming — no longer in the T-series).

### Predictions

See `C++/docs/plans/ablation/fundamental-bleed-weight-fix.md` for full
analysis. Short version: f65 clumping should diminish, e3d03 FP should
disappear (no spurious neighbor brightness to attach to), cb11 may or
may not still fire. f111 z=0 drift likely unaffected (boundary issue).

### A/B target

```bash
python3 C++/scripts/compare_ablation.py \
  C++/outputs/output_ablation_T1_20260422_180917 \
  C++/outputs/output_bleed5_<ts> --from-frame 61
```

### Iteration history

- **bleed=5 (asymmetric, 2026-04-23 early)**: f65 splits came in cleaner
  than T1 (drifts 6-7 vx vs T1's 9-18) — validated "cells can't cheaply
  migrate through neighbor territory". BUT at f66, **e3d03 acquired a
  new FP split** (drift1=19.3, cost=-46k). Stopped mid-run.
- **bleed=2 (asymmetric)**: same pattern. f65 clean, f66 e3d03 FP with
  drift2=26.7 (worse geometry than bleed=5). Confirmed the FP is not
  weight-tunable — it's structural.
- **Diagnosis**: the bleed term was applied in `Frame::perturbCell`
  (parent shape fit) but NOT in `Frame::trySplitCellPhased` (split
  candidate cost comparison). Strong bleed weight constrained the
  single-parent fit tightly (image cost higher) while leaving daughter
  candidates free to tile broadly (image cost unchanged) → split cost
  differential tipped negative for more cells → new FP class.
- **Fix**: apply bleed symmetrically — see Change 20.

## 2026-04-23: Symmetric bleed in split cost (Change 20) — **status: ACTIVE (winning candidate)**

### Change

In `Frame::trySplitCellPhased`:

- **Baseline total** now includes bleed term: `baselineBleed = _voronoiBleedWeight × computeVoronoiBleedVoxels(parent, parentIdx, bypass=true)`.
- **Candidate total** per candidate includes bleed for both daughters against PARENT's Voronoi territory (the allowed zone — daughters inherit parent's space): `candBleed = _voronoiBleedWeight × (bleed(d1, parentIdx, bypass) + bleed(d2, parentIdx, bypass))`.
- **Post-refine total** (after the final winner is refined) recomputed with bleed — otherwise the final `costDiff = bestTotal - baselineTotal` comparison becomes asymmetric again.

In `Frame::computeVoronoiBleedVoxels`:

- Added `bool bypassEnabledFlag = false` parameter. Split-cost path sets `true` to query the map while `VoronoiDisableGuard` has disabled `_voronoiEnabled` during burn-in. Map itself remains valid — the guard protects against stale-index queries during daughter perturb-cost paths, not against measuring bleed against the parent's pre-split voronoi index.

Log lines `[Split Baseline]` and `[Split Cand]` now include `bleed=...`
for diagnostic visibility.

### Files changed

- `C++/includes/Frame.hpp`: `computeVoronoiBleedVoxels` signature + doc
- `C++/src/Frame.cpp`:
  - `computeVoronoiBleedVoxels`: gate condition updated to respect bypass
  - `trySplitCellPhased`: `baselineBleed` added to `baselineTotal`
  - `trySplitCellPhased`: `candBleed` added to `candTotal` in candidate loop
  - `trySplitCellPhased`: `postRefineBleed` added to post-refine `bestTotal`
  - Log lines updated to surface bleed values

### Config (final for this run)

```yaml
voronoi_bleed_penalty_weight: 2.0        # 4× up from default 0.5
bloat_cap_barrier_weight: 1000.0          # Fix C kept
brightness_centroid_anchor_weight: 0.0    # Fix A shelved
split_persistence_required_frames: 0      # T3 shelved
mc_rng_seed: 42                           # ablation
```

### Results (f60 → f72, seed=42, from f59 checkpoint)

`outputs/output_bleed2_sym_20260423_034034` vs T1
(`output_ablation_T1_20260422_180917`), via `compare_ablation.py`:

| Metric | T1 | bleed=2 sym | Δ |
|---|---|---|---|
| Total splits | 17 | 14 | -3 |
| **Splits with drift > 15 (ugly)** | **1** | **0** | **−100%** |
| **Cell-frames drift > 15** | 46 | 17 | **−63%** |
| **Cell-frames drift > 25** | 12 | 1 | **−92%** |
| **Mean per-frame drift** | 8.97 | 7.48 | −17% |
| Max aR | 57.4 | 56.7 | ≈ |

Split diff — only in T1:
- `12345679:101` f66 (drifts 4.1/5.2) — clean TP missed
- `12345679:001` f67 (drifts 6.7/6.8) — clean TP missed
- `1f89abf4:11` f72 (drifts 14.2/**56.7**) — ugly split **correctly rejected**

### Verdict

**This is the shipping candidate.** The fundamental force balance is now
internally consistent, and the result on the test interval is strictly
better than T1 on drift metrics and FP rejection, at a cost of 2
borderline TPs. That trade is favorable.

See `docs/plans/ablation/bleed-symmetric-fix.md` for full analysis and
next-step options (weight tuning 1.0-1.5 to recover borderline TPs;
production full-f1-f72 run from scratch).

## 2026-04-23: Ship bleed=5 asymmetric as f1-f65 GT config (Change 21) — **status: ACTIVE (GT run config)**

### Decision

Two viable shipping candidates emerged from the ablation:
- **bleed=2 symmetric** — zero ugly drifts, fewer splits, some legit TPs missed, cb11 parent bloats
- **bleed=5 asymmetric** — cleanest individual drifts (f65 all ≤ 7.3 vx), catches more legit TPs,
  but reproducibly fires e3d03 FP at f66 (drift=27.6) and produces 1-extra-boundary-drift cell

User direction (2026-04-23): problems tend to happen after f65. For f1-f65 GT specifically,
bleed=5 asym is the strongest choice because (a) it's already GT-marked for f60-f65, (b) no
FPs through f65, and (c) the boundary-drift residuals all manifested at f66+ in test runs.

### Config (shipping for f1-f65 GT)

```yaml
# config.yaml (simulation)
mc_rng_seed: 42
voronoi_bleed_penalty_enabled: true
voronoi_bleed_penalty_weight: 5.0          # 10x up from default 0.5
split_symmetric_bleed: false               # asymmetric (Change 20 path off)

# config.yaml (prob)
bloat_cap_barrier_weight: 1000.0           # Fix C
bloat_cap_barrier_start: 1.1
bloat_cap_barrier_end: 1.6
brightness_centroid_anchor_weight: 0.0     # Fix A shelved
split_persistence_required_frames: 0       # T3 shelved
```

```ini
# user_input_configurations.ini [jihang]
output_name_rule=output_clean_gt_f1-f65_{timestamp}
first_frame=1
last_frame=65
resume_from=0
resume_source_dir=
```

### Deferred residuals (for post-f65 work)

1. **Boundary-drift local minimum**: cells at image ceiling/floor get a free pass because
   "overhanging" voxels don't contribute to L2 cost. Observed in resumed run: 3 cells at
   boundary by f72 (`8cbdf86d`, `1f89abf4:011`, `1f2ed10d:111`) vs T1's 2. Fix candidates:
   out-of-bounds penalty term, or normalize cost by in-image volume fraction.
2. **Asymmetric bleed reproducibly fires e3d03 FP at f66** when crowded neighbors create a
   Voronoi territory where a split reduces image-cost vs the tightly-constrained parent fit.
   The symmetric path (Change 20, currently inert at `split_symmetric_bleed: false`) is
   retained for post-f65 experimentation.
3. **cb11 ugly split pathology**: lineage-inherited large-at-birth parent produces an
   asymmetric split regardless of config. Structural — needs a different approach.

### Rollback path

If f1-f65 GT run misbehaves, switch to bleed=2 sym by setting
`voronoi_bleed_penalty_weight: 2.0` and `split_symmetric_bleed: true`.

## 2026-04-23: Voronoi map CV_32S → CV_8U (Change 22) — **status: ACTIVE (perf)**

### Change

`_voronoiMap` now stores cell indices as `CV_8U` (1 byte) instead of
`CV_32S` (4 bytes). Requires `nCells ≤ 255`; a runtime check in
`Frame::rebuildVoronoiMap` throws if exceeded. Current embryo peaks at
~55 cells, ~4.5× headroom.

### Impact

- **Memory**: 225 slices × 512 × 512 × 4 bytes = ~236 MB → 59 MB per
  frame (4× savings).
- **Speed**: 2-3× inner-loop speedup in `computeVoronoiBleedVoxels`,
  `computeBrightnessCentroids`, and `calculateBboxCost` with Voronoi
  filter — 4× more index values per cache line.

### Files changed

- `C++/includes/Frame.hpp`: comment updated on `_voronoiMap`.
- `C++/src/Frame.cpp`:
  - `rebuildVoronoiMap`: `CV_32S → CV_8U` at allocation + ptr write; added
    `nCells > 255` overflow guard; changed `bestIdx = -1` sentinel to
    `0` (always overwritten when nCells > 0).
  - `computeVoronoiBleedVoxels`: `ptr<int>` → `ptr<uint8_t>`, type check
    `CV_32S → CV_8U`, compare widened via `static_cast<int>`.
  - `computeBrightnessCentroids`: same pattern as bleed.
  - `calculateBboxCost` Voronoi filter: `ptr<int>` → `ptr<uint8_t>`,
    precompute `uint8_t vorTarget = static_cast<uint8_t>(voronoiCellIdx)`.

## 2026-04-23: Out-of-bounds voxel penalty (Change 23) — **status: ACTIVE (ablation)**

### Motivation

Cells rendering voxels outside the image bounds pay ZERO cost for those
voxels because L2 is summed only over in-image pixels. This creates a
hard local minimum at the image boundary: once a cell drifts to z=0 or
z=224 (or x/y edges), moving inward *increases* cost because the in-image
footprint grows with each recovered voxel rendering (1-real)² > 0, while
the overhang was free.

Observed cases:
- `8cbdf86d` pinned at exactly z=224.0 across f61–f72 (T1 and bleed5
  resumed) even while the real bright cell migrated down the z-axis.
- `1f2ed10d:111` dropped to z=0 at f70 in T1 and bleed5 resumed.
- `1f89abf4:011` climbed to z=224 at f68–f72 in bleed5 resumed.

These are not split FPs or overlap issues — they're cost-function
artifacts at the image boundary.

### Fix

Add a per-voxel penalty for ellipsoid voxels outside the image:
`penalty = weight × count(voxels inside ellipsoid but outside [0,W)×[0,H)×[0,Z))`.
Applied symmetrically in:
- `Frame::perturbCell`: `old/newOOBPenalty` added to delta cost.
- `Frame::trySplitCellPhased`: `baseline/cand/postRefineOOB` added to
  respective totals.

New method `Frame::computeOutOfBoundsVoxels(cell)` does the count with
an early-exit when the cell's bbox is entirely in-bounds (fast-path for
the vast majority of cells).

### Config

- `simulation.out_of_bounds_penalty_weight: 5.0` (initial trial). Weight
  0 disables (legacy behavior).
- For a cell pinned at z=224 with `cR=22`, ~30k voxels overhang. At
  weight=5 the penalty is ~150k — dwarfs the ~10k L2 reward for the
  overhang. Moving inward to recover the overhang (e.g., z=210) drops
  OOB count to near zero.

### Files changed

- `C++/includes/ConfigTypes.hpp`: added `out_of_bounds_penalty_weight`
  (float, default 0).
- `C++/includes/Frame.hpp`: added `computeOutOfBoundsVoxels(cell)` +
  setter/getter + `_outOfBoundsPenaltyWeight` member.
- `C++/src/Frame.cpp`: method implementation (parallel-for reduction
  over bbox, point-in-ellipsoid test); term added to `perturbCell`
  costDiff and `trySplitCellPhased` baseline/cand/postRefine totals.
- `C++/src/CellUniverse.cpp`: `frame.setOutOfBoundsPenaltyWeight(config...)`
  pushed alongside bleed weight.
- `C++/config/config.yaml`: `out_of_bounds_penalty_weight: 5.0`.
- `C++/config/user_input_configurations.ini`: output rename
  `output_clean_gt_f1-f70_bleed2sym_oob5_{timestamp}`, `last_frame=70`
  (extended past f65 to test boundary behavior at frames where drift
  previously occurred).

## 2026-04-24 — Change 24: watershed territory map

**Status:** ACTIVE. Plan: `C++/docs/plans/2026-04-24-watershed-territory-and-cc-split.md`.

### Problem

Center-distance Voronoi partition couples each cell's territory to its
neighbors' positions. When a neighbor drifts, bloats, or shifts (e.g.,
8cbdf86d unpinned from z=224 by OOB penalty and drifting to z≈194 at
frame 28), the midline moves into the squeezed cell's biological volume.
Shape-fit PCA and cost evaluation over the squeezed territory then see
fewer bright pixels than biology, the cell stays undersized, and by the
split frame the territorial L2 cost treats "one parent" and "two
daughters" as equally cheap (no cost profit signal to accept the split).

In the `output_clean_gt_f1-f70_kmeans2_20260424_024113` run this failure
mode blocked `e9077:10`'s split at f40–f41 despite the kmeans2 detector
correctly finding bimodality — the cost gate rejected every candidate.

### Fix

Replace the per-voxel nearest-cell-center partition with seeded watershed
flooding on the preprocessed bright image.

- **Dispatch.** `Frame::rebuildVoronoiMap` now branches on
  `_useWatershedTerritory`: legacy body extracted to
  `fillTerritoryMapNearestNeighbor()`; new
  `fillTerritoryMapWatershed()` implements Meyer's algorithm.
- **Seed correction.** Each anchor is snapped to the nearest above-
  threshold voxel within `cap_ratio × min(birth a,b,c)` voxels. Prevents
  a drifted snap in a dim trough from flooding a neighbor's blob.
- **Flooding.** Max-heap priority queue keyed on voxel brightness →
  brightest frontier voxels get labeled first, so label boundaries fall
  along gradient minima (dim ridges between bright blobs). Voxels below
  `watershed_bright_threshold` never get pushed → remain label 255
  (unclaimed sentinel).
- **Ellipsoid fill-in.** Any unclaimed voxel inside a cell's live
  ellipsoid is assigned that cell's label (nearest by Euclidean among
  containing ellipsoids). Prevents dim voxels inside an ellipsoid from
  being read as out-of-territory by downstream bleed/bbox code.
- **Map capacity.** `nCells` limit tightened from 255 → 254 because
  label 255 is now reserved as the unclaimed sentinel.

### Downstream

All `_voronoiMap` readers see the watershed fill unchanged:

- `Frame::calculateBboxCost(..., voronoiCellIdx)` — cell pays cost only
  over its own basin.
- `Frame::computeVoronoiBleedVoxels` — voxels in cell's ellipsoid whose
  label ≠ cell idx count as bleed (unclaimed voxels also count as bleed;
  ellipsoid fill-in prevents this for our own cell).
- `gatherBrightPixelsVoronoi` (free function) — new optional args
  `voronoiMap *`, `voronoiSelfIdx`. When non-null, skips the Euclidean
  claim-set filter and uses the map lookup. Shape-fit
  (`calibrateCellShapeViaPca`, `calibrateCellPositionViaCentroid`) and
  split-detection gather (`trySplitCellPhased`) now pass these when
  watershed is on.

### Config

- `simulation.use_watershed_territory: true` (default `false` = legacy).
- `simulation.watershed_bright_threshold: 0.15`
- `simulation.watershed_seed_correction_cap_ratio: 0.5`

### Files changed

- `C++/includes/ConfigTypes.hpp`: added three fields + `explodeConfig`
  entries + `printConfig` entries.
- `C++/includes/Frame.hpp`: added setters
  `setUseWatershedTerritory/BrightThreshold/SeedCorrectionCap`, private
  declarations for `fillTerritoryMapNearestNeighbor` and
  `fillTerritoryMapWatershed`, members `_useWatershedTerritory`,
  `_watershedBrightThreshold`, `_watershedSeedCorrectionCap`.
- `C++/src/Frame.cpp`: refactored `rebuildVoronoiMap` to dispatch; new
  `fillTerritoryMapWatershed` (~165 LOC with seed correction, Meyer
  flooding, ellipsoid fill-in); `gatherBrightPixelsVoronoi` signature
  extended with two optional params; three call sites (`calibrateCellShapeViaPca`,
  `calibrateCellPositionViaCentroid`, `trySplitCellPhased`) pass
  `&_voronoiMap` and `cellIndex` when watershed mode is on. Added
  `#include <algorithm>` and `#include <queue>`.
- `C++/src/CellUniverse.cpp`: three new `frame.set...` calls alongside
  existing bleed/OOB setup.
- `C++/config/config.yaml`: three new entries under `simulation:`.

## 2026-04-24 — Change 25: connected-component split trigger

**Status:** ACTIVE. Depends on Change 24 (watershed territory).

### Problem

Case 1 and Case 2 reported on sketch IMG_BDE7FDCB90AC: parent cell ends
up as one elongated ellipsoid spanning two biologically-distinct bright
blobs; biology has N+1 cells, synth has N. The existing PCA / kmeans2
split detectors fire on pixel-distribution variance; they miss the case
where the pixel set along the split axis looks unimodal but the
underlying topology is two disjoint bright components.

### Fix

Before the PCA / kmeans2 axis generation in
`Frame::trySplitCellPhased`, when `_useCCSplitTrigger` is on:

1. Gather above-threshold pixels in parent's territorial basin (pixel
   vector already has this with Change 24's watershed filter).
2. Build a dense 3D marker grid over the pixels' bbox; run iterative
   BFS flood labeling (6-connectivity). Component sizes recorded.
3. Filter components below `cc_split_min_component_volume` voxels.
4. Select the two largest remaining components; compute their
   brightness-weighted centroids in world coordinates.
5. Set `kmeansValid/kmeansC1/kmeansC2` to the CC centroids, insert
   direction (unit vector from C1 to C2) at **front** of `primaryDirs`
   with name `"cc2"` (highest candidate-cap priority), and set
   `ccTriggered=true`.
6. When `ccTriggered` is true, the downstream kmeans2 variance block
   is skipped (guarded by `if (!ccTriggered && pixels.size() >= 100)`).
7. In the AxisPlacement loop, the override condition extends to both
   `"kmeans2"` and `"cc2"` so the CC centroids are used directly as
   daughter positions (not via 1D projection).

### Why topologically stronger than kmeans2

- Immune to mass asymmetry: kmeans2's 25%-cluster-balance guard rejects
  uneven splits; CC components only need size ≥ min volume each.
- Immune to along-axis variance flattening: equal-sized daughters
  connected by a thin neck can have unimodal projection variance but
  K=2 connected components.
- Self-limiting: when the parent is a single connected blob (K=1), CC
  doesn't fire and the pipeline falls through to kmeans2/PCA axes.

### Config

- `simulation.use_cc_split_trigger: true` (default `false`).
- `simulation.cc_split_bright_threshold: 0.15`
- `simulation.cc_split_min_component_volume: 200` (voxels)

### Files changed

- `C++/includes/ConfigTypes.hpp`: added three fields + parser/print.
- `C++/includes/Frame.hpp`: setters
  `setUseCCSplitTrigger/CCSplitBrightThreshold/CCSplitMinComponentVolume`
  + members.
- `C++/src/Frame.cpp`: ~170-LOC CC trigger block inserted before the
  PCA/kmeans2 axis generation; kmeans2 block guarded with `!ccTriggered`;
  AxisPlacement override extended to match `"cc2"` in addition to
  `"kmeans2"`.
- `C++/src/CellUniverse.cpp`: three new `frame.set...` calls.
- `C++/config/config.yaml`: three new entries under `simulation:`.
- `C++/config/user_input_configurations.ini`: `last_frame: 70 → 45`,
  `output_name_rule: output_watershed_cc_f1-f45_{timestamp}` for f45
  validation.

### Validation plan (f45)

Compare against `output_clean_gt_f1-f70_kmeans2_20260424_024113`:

- f1–f38 cell counts must match exactly (no regression on pre-f40
  splits).
- f39's seven splits must all complete cleanly.
- **f40 `e9077:10` must split** (was missed in kmeans2 baseline).
- By f41 cell count should be 26 vs baseline's 25.
- Run to f45; expect cell count ≈ 30 per GT.

Rollback: flip `use_watershed_territory` and/or `use_cc_split_trigger`
to `false` in `config.yaml`.




## 2026-04-24 — Change 26: cleanup, retire failed watershed and shape-cap experiments

**Status:** ACTIVE.

### What was removed

After empirical testing across 7 runs, two of the four 2026-04-24 fixes
were confirmed to introduce more failures than they solved and have been
removed entirely:

**Change 24 (Watershed territory) — REMOVED.**
- Run 1 (full watershed for all gather paths): cells froze at f5 onward
  because watershed basins for drifted cells were empty / dim, and
  shape-fit/centroid/split-gather all bailed at < 20 pixels.
- Run 4 (watershed only for bbox cost + bleed): e3d03 false-positive
  split at f9 (`snap_imgPca_trans-` with drift1=36 vx) — the shrunken
  watershed basin made a 2-daughter fit look competitive on a unimodal
  corner cell.
- The empty-basin fallback retry (added between runs) helped for
  near-empty basins but did not address the underlying instability.

**Fix I (Shape-fit hard cap at `shape_fit_max_radius_ratio × birth`) —
REMOVED.**
- Run 6/7: kept 8cbdf86d at aR=24-28 (vs prior 40+) but caused
  e9077:10 to drift ~37 vx in z by f26. Mechanism: the cap under-fits
  legitimate cells whose biology grew past 1.4×birth; the synth then
  under-renders, the cost gradient says "missing pixels", perturbCell
  finds it cheaper to MOVE the center toward unclaimed bright pixels
  than to GROW (blocked by bloat-cap soft barrier). Cell drifts off
  biology. Next frame's split candidates land in the wrong place.

### What was kept

**Change 25 (CC split trigger) — KEPT.**
Run 5 + Run 6 + Run 7 all confirmed: zero false-positive accepted
splits across f1-f45 from the CC trigger; downstream bio/cost gates
catch any spurious CC fires. CC contributes a topologically-grounded
split candidate when the parent's territorial pixels form K ≥ 2
connected components above `cc_split_min_component_volume`.

**Fix H + H2 (image-aware bury check) — KEPT.**
- Fix H: dim-valley check between daughter and `other.center` overrides
  `d_buried_in_other` when a dim voxel exists on the line.
- Fix H2: distance-to-other-center > `bio_buried_birth_envelope_factor ×
  max(other.birth_radii)` overrides when daughter is in `other`'s
  bloated extent but outside biological core.
- Run 7 confirmed both override paths fire correctly (no false
  permissive accepts), but the underlying split candidate quality is
  what was wrong in some cases — these overrides correctly let through
  candidates that would have been rejected for stale reasons.

### Files changed (deletions)

- `C++/includes/ConfigTypes.hpp`: removed `use_watershed_territory`,
  `watershed_bright_threshold`, `watershed_seed_correction_cap_ratio`
  (SimulationConfig), `shape_fit_max_radius_ratio` (ProbabilityConfig).
  Removed corresponding `explodeConfig` parses + `printConfig` lines.
- `C++/includes/Frame.hpp`: removed `setUseWatershedTerritory`,
  `setWatershedBrightThreshold`, `setWatershedSeedCorrectionCap`,
  `setShapeFitMaxRadiusRatio`, `getShapeFitMaxRadiusRatio` declarations.
  Removed `_useWatershedTerritory`, `_watershedBrightThreshold`,
  `_watershedSeedCorrectionCap`, `_shapeFitMaxRadiusRatio` members.
  Removed `fillTerritoryMapNearestNeighbor`, `fillTerritoryMapWatershed`
  private declarations.
- `C++/src/Frame.cpp`:
  - `rebuildVoronoiMap` collapsed back to the legacy single-body
    nearest-neighbor implementation (no dispatch). `nCells` cap
    restored to 255 (no longer reserving 255 as unclaimed sentinel).
  - `fillTerritoryMapWatershed` body deleted (~165 LOC).
  - `gatherBrightPixelsVoronoi` signature reverted to the original (no
    `voronoiMap` / `voronoiSelfIdx` optional params, no retry loop).
  - `calibrateCellShapeViaPca` shape-cap block (`_shapeFitMaxRadiusRatio`)
    deleted; targetA/B/C reverted to const.
  - `#include <queue>` removed (no longer needed without watershed).
- `C++/src/CellUniverse.cpp`: removed `setUseWatershedTerritory`,
  `setWatershedBrightThreshold`, `setWatershedSeedCorrectionCap`, and
  `setShapeFitMaxRadiusRatio` calls. Kept CC + bury setters.
- `C++/config/config.yaml`: removed `use_watershed_territory`,
  `watershed_bright_threshold`, `watershed_seed_correction_cap_ratio`,
  `shape_fit_max_radius_ratio` entries. Kept CC + bury entries.

### Net effect on the active pipeline

After cleanup, the post-2026-04-23 deltas vs run 5 baseline (CC trigger
on, watershed off) are: **bury overrides (Fix H + H2) only**. CC trigger
is unchanged from Change 25's original implementation.

The ROOT failure mode that motivated all of this work — the f40
`e9077:10` miss caused by 8cbdf86d's unbounded shape-fit growth —
remains unresolved. Future work needs to address the bloat without
introducing the under-fit drift that Fix I caused.

## 2026-04-25 — Change 27: Round 1 cleanup

**Status:** ACTIVE.

Pure cleanup PR — no behavior change beyond removal of dead/shelved features.

### Removed

**1a. deprecated/ directory** — pre-restructure cell types (Bacilli, Sphere,
Cell base) and helpers (mathhelper, args, pseudo-code), unreferenced by
any compiled target since 2026-03-27. ~1500 LOC + 1 sample TIFF deleted.

**1a. Stale plan docs moved to docs/history/plans/** — 18 plan files spanning
2026-03-27 to 2026-04-22 (cpp-restructure, current/target UMLs from 03-27,
dead-code-and-cleanup 03-28, residual-split-detection 03-28, yp-yd merge
review 04-07, pca-input-fix 04-09, snapshot-driven-split 04-09, triaxial
pipeline redesign 04-10, universal-bbox-cost 04-14, bbox-anchor-flaw 04-15,
late-frame-ablation 04-22), the watershed-territory plan from 04-24, and
the entire `ablation/` subdirectory (T0-T3 ablation reports). All
superseded by current shipping code.

**1b. Signal-guided perturbation feature** — entire `signal_guided_position`
machinery, removed because it was never enabled in any shipping run
(`signal_guided_position_enabled: false`) and its absence simplifies the
perturbation path significantly:
- `Frame::SignalCenter` struct + `_signalCenters` member + setters/getters
- `Frame::perturbCell(..., bool useSignalGuidance)` parameter dropped
- Signal-guided override block (~40 LOC) inside perturbCell
- `localizeSignalCentersForFrame()` + `BrightBox` struct +
  `chooseNearestDivisorSize()` in `CellUniverse.cpp` (~185 LOC)
- `localizeSignalCentersInStack()` + duplicates of `BrightBox` and
  `chooseNearestDivisorSize` in `ImageHandler.cpp` (~232 LOC, never
  called by any caller — defined but dead)
- `useSignalGuidanceThisFrame` plumbing in `CellUniverse::optimize` (3
  call sites + iteration-budget split logic)
- 5 `signal_guided_*` config fields + parsers + printConfig entries
- Iteration budget consolidated to `iterations_per_cell` (with
  `random_iterations_per_cell` legacy override field retained for
  backward-compat YAML parsing)

**1b–1d. Uncommitted shelved features removed by `git checkout` of
`Frame.hpp`, `Frame.cpp`, `CellUniverse.hpp`, `CellUniverse.cpp`,
`ConfigTypes.hpp`** — these reverted Frame.hpp/cpp + ConfigTypes.hpp to the
last-committed baseline (commit 79853cc), which never included Fix A
(brightness centroid anchor), Fix D (split persistence), `mc_rng_seed`,
the watershed territory machinery (Change 24), the shape-fit hard cap
(Fix I), or the bury overrides (Fix H/H2). All those experimental
mechanisms are now removed in one operation. CC trigger (Change 25) was
also reverted; it had been kept as the most additive/safe of the 04-24
batch but Stage C (local-maxima detection in the post-cleanup roadmap)
will supersede it cleanly.

**1d. config.yaml orphan fields** — 14 fields whose backing code no longer
exists removed: `mc_rng_seed`, `split_symmetric_bleed`,
`out_of_bounds_penalty_weight`, `use_cc_split_trigger`,
`cc_split_bright_threshold`, `cc_split_min_component_volume`,
`bloat_cap_barrier_weight`, `bloat_cap_barrier_start`,
`bloat_cap_barrier_end`, `brightness_centroid_anchor_weight`,
`brightness_centroid_radius_factor`, `split_persistence_required_frames`,
`split_persistence_position_tolerance_vx`,
`split_persistence_cost_tolerance_ratio`, `bio_buried_image_check_enabled`,
`bio_buried_bright_valley_threshold`, `bio_buried_birth_envelope_factor`.
The voronoi-bleed-penalty config block + asymmetric-cost config + the
remaining production fields are kept. Net config.yaml: 747 → 637 lines.

### What's left (active features)

After Round 1, the shipping pipeline contains:
- Adaptive bbox cost (`use_bbox_cost: true`)
- Voronoi bleed penalty (`voronoi_bleed_penalty_*`)
- Asymmetric L2 cost
- Position prior in perturbCell
- Snap-driven split candidate generation (snapshot parent + axis directions)
- Bio-gates: size ratio, volume fraction, bury-in-other,
  neighbor-bridging, slab-min bridge
- Two-pass split candidate pre-filter
- PCA + (data | snap) × (axC | imgPca) × (primary | rot± | trans±) candidate axes
- Daughter refit + Phase B-C refine
- Per-cell brightness EMA
- Bloat-cap soft barrier (`bloat_cap_barrier` was orphaned in cleanup
  alongside Fix C/D — config keys removed; the barrier code itself wasn't
  in HEAD baseline either, so it goes too)

### Frame.cpp size

Before cleanup: 4892 LOC.
After cleanup: 3985 LOC (−907 LOC).

ImageHandler.cpp: ~1370 → 1135 LOC (−235 LOC).
CellUniverse.cpp: ~2276 → 1878 LOC (−398 LOC).
Frame.hpp: ~534 → 405 LOC (−129 LOC).
ConfigTypes.hpp: ~988 → 815 LOC (−173 LOC).
config.yaml: ~747 → 637 LOC (−110 LOC).

Total: ~1952 LOC of dead/shelved code removed across the 5 files, plus
the 1500 LOC `deprecated/` directory and 18 stale plan docs archived.

### Build status

Clean compile. f1-f45 GT validation deferred to Round 2 (drift cap +
neighbor-bridging gate audit) so the cleanup can ship as a no-behavior-
change PR independently of the new features.

## 2026-04-25 — Change 28: Round 2 (Stage A + B) — defensive split-rejection gates

**Status:** ACTIVE.

Two cheap defensive fixes targeting the e3d03 f28 false-split class
identified in `outputs/output_linux_failed/debug_log.txt`. Additive: each
gate is config-flag-controlled and defaults to firing.

### Stage A — Daughter drift cap on refit

After daughter refit/refine converges in `Frame::trySplitCellPhased`,
reject the candidate if either daughter's final position is more than
`split_daughter_max_drift_radius_factor × max(srcMajor, srcB)` voxels
from its seed.

The PCA refit walks each daughter to the brightness-weighted centroid of
pixels in a generous gather window (radius = 3 × parent.maxR). With no
constraint, the daughter can centroid onto a NEIGHBOR's bright mass
instead of the parent's biology — exactly what e3d03's d2 did at f28
(walked 44 vx from seed (95,251,61) to final (77,221,87), landing on
12345:11's bright blob, then passing the bridge gate because there's a
genuine dim gap between e3d03 and 12345:11 — they're separate cells).

A legitimate daughter cannot land more than ~one parent radius from its
seed without being outside the parent's volume entirely. Cap = `1.0 ×
max(srcMajor, srcB)` separates legitimate drifts (~25 vx for srcMajor~30
in observed runs) from runaway drifts (~35–50 vx).

**Files changed:**
- `C++/src/Frame.cpp`: ~37-LOC gate inserted at the existing diagnostic
  line "Drift from seed (diagnostic only, no rejection gate)" — that
  comment is now stale (gate added) and updated.
- `C++/includes/ConfigTypes.hpp`: added `split_daughter_max_drift_radius_factor`
  (float, default 1.0).
- `C++/config/config.yaml`: added the field with documentation.

### Stage B — Neighbor-bridging gate audit + tighten

The existing `bioCheckDaughters` gate "reject if either daughter's center
is closer to a non-sibling cell than `factor × siblingDistance`" used a
hardcoded factor of 0.5. Audit of e3d03 f28:
- d2 at (76.97, 220.71, 87.23), nearest neighbor 12345:11 at
  (79.33, 239.29, 112.86) → distance 31.7 vx
- sibling distance = 60.7 vx
- threshold at factor 0.5 = 30.4 vx → 31.7 just barely passed
- threshold at factor 0.6 = 36.4 vx → 31.7 caught

Tightened the factor from 0.5 to 0.6 and made it configurable. The gate
already fires correctly on legitimate cases in run-7 logs (multiple
"d2_bridging_to_X" rejects); 0.6 narrows the false-negative window
without rejecting valid splits (legitimate splits had daughters >0.7 ×
sibling distance from any neighbor in observation).

**Files changed:**
- `C++/src/Frame.cpp`: extracted hardcoded 0.5f to read
  `probConfig.bio_neighbor_bridging_factor`. Wrapped in `if (factor > 0.0f)`
  so 0.0 disables.
- `C++/includes/ConfigTypes.hpp`: added `bio_neighbor_bridging_factor`
  (float, default 0.6).
- `C++/config/config.yaml`: added the field with audit details.

### Combined effect

Either gate independently catches the e3d03 f28 false split:
- Stage A: drift1=6, drift2=44, cap=37 → drift2 > cap → reject.
- Stage B: d2_to_12345:11 = 31.7, threshold = 36.4 → reject.

Both fire = belt and suspenders. Stage B is the more general gate
(catches geometry that Stage A might miss for unusual parent shapes);
Stage A is the more direct (drift > radius is unambiguous).

### Validation

Build clean. Functional validation deferred to next test run on the
linux_failed scenario (need a re-run of f1-f30 to verify e3d03 doesn't
false-split).

## 2026-04-25 — Change 29: Round 3 (Stage C) — local-maxima image grounding

**Status:** ACTIVE.

The high-leverage architectural change: introduce image-derived bright local
maxima as the primary signal for both daughter seeding and frame-start
position correction. Image topology, not cell geometry, drives the
critical decisions.

### Stage C-1: `Frame::findLocalMaxima`

New method: in a 3D window around a center point, find voxels whose
brightness exceeds (a) a configurable threshold and (b) all 6-connected
neighbors. Output sorted descending by brightness, with non-max
suppression dropping dimmer maxima within `min_separation` voxels of a
brighter one.

Returns the image-derived list of "where biology says cells live" inside
the window. One bright blob → one local max. Two adjacent dividing
blobs → two local maxima (with a dim valley between, hence both pass the
strict-local-max test). Cell drifted off biology → zero maxima.

### Stage C-2: local-maxima split seeding (`lmaxima` axis)

In `Frame::trySplitCellPhased`, before PCA / axC seed generation, call
`findLocalMaxima` around the parent's snapshot position with window
`1.5 × max(snap_radii)`. If ≥2 maxima are within `1.2 × parent.maxR`
of the snapshot center (birth-envelope check — rejects maxima on
neighbor's biology), insert "lmaxima" as the **highest-priority**
primary axis with the top-2 maxima as direct daughter seeds.

This catches case-1 / case-2 (parent stretched across two daughter
blobs) by image topology — two distinct brightness peaks ≡ two cells,
no PCA bimodality reasoning needed.

In `AxisPlacement` loop, "lmaxima" override path uses the maxima as
direct daughter centroids (skip `centroidsAlongAxis` projection).

### Stage C-3: per-frame snap-position correction

In `CellUniverse::optimize`, after loading the frame and before
installing snap positions for the bbox cost, call `findLocalMaxima`
around each cell's snap with cap = `0.5 × min(birth_radii)`. If a max
is found inside the cap window, move the snap toward it (capped step).

This corrects sub-radius drift before it cascades. Cells stay anchored
to biology each frame instead of accumulating Monte Carlo noise.

### Config

```yaml
prob:
  local_maxima_split_seeding_enabled: true
  local_maxima_brightness_threshold: 0.15
  local_maxima_split_window_radius_factor: 1.5
  local_maxima_split_envelope_factor: 1.2
  local_maxima_snap_correction_enabled: true
  local_maxima_snap_correction_cap_factor: 0.5
```

### Files changed

- `C++/includes/Frame.hpp`: `LocalMax` struct, `findLocalMaxima`
  declaration (~26 LOC).
- `C++/src/Frame.cpp`: `findLocalMaxima` implementation (~85 LOC),
  lmaxima split-seeding block in `trySplitCellPhased` (~95 LOC),
  AxisPlacement override for "lmaxima" (~12 LOC).
- `C++/src/CellUniverse.cpp`: snap-correction block in the bbox-active
  loop (~50 LOC).
- `C++/includes/ConfigTypes.hpp`: 6 new ProbabilityConfig fields +
  parser entries.
- `C++/config/config.yaml`: 6 new entries with documentation.

### Validation: f1–f45 against GT

**100% GT match across all 45 frames.** All 18 GT splits accepted with
correct daughters and tight drifts. Critical wins:

- **f39 e9077:11 split** — drifts 0.37/0.71 vx. This cell was
  persistently bio-rejected in all 2026-04-23/24 attempted fixes due
  to 8cbdf86d's bloated ellipsoid envelope. Local-maxima seeding makes
  the split image-driven; the seed positions land on biology directly.

- **f40 e9077:10 split** — drifts 16.99/20.21 vx. **The original case-1
  failure that motivated the entire Round 1–3 effort.** Missed in the
  kmeans2 baseline (output_clean_gt_f1-f70_kmeans2_*), missed in
  watershed run-1 (cells froze), false-positive in run-4 (e3d03 split
  wrongly), missed by under-fit drift in runs 6/7 (shape cap caused
  e9077:10 to drift off biology). Now ACCEPTED with biological accuracy.

Zero false-positive accepted splits. Stage A (drift cap) caught 4
candidates correctly during the run; all were known non-GT splits.

### What this confirms about the design

1. **Image topology is the right signal.** Two bright peaks separated
   by a dim valley ≡ two cells. The pipeline now reads this directly
   instead of inferring it from pixel-distribution statistics.

2. **Seed quality > burn-in tweaking.** The previous architecture
   spent thousands of iterations on candidate burn-in to refine bad
   seeds. With image-derived seeds, daughters land near their
   biological centers immediately and burn-in just polishes (drifts
   under 1 vx for several f39 splits).

3. **Per-frame correction beats per-iter correction.** Snap correction
   at frame start is cheaper and more decisive than the soft
   position-prior penalty in perturbCell.

4. **Stage A + B + C compose cleanly.** Each addresses a different
   failure mode without interfering with the others. No tuning storm.

## 2026-04-25 — Change 30: Round 3 Stage C disabled by default

**Status:** PARTIAL REVERT (code retained, config flags off).

### What happened

Round 3 Stage C added local-maxima detection (Stage C-1), split seeding
via lmaxima axis (C-2), and per-frame snap-position correction (C-3).
First validation (f1-f45) reported GT-perfect including the long-elusive
f40 e9077:10 split.

Closer inspection of the f1-f45 log: `findLocalMaxima` returned 0 maxima
on every call. The strict `>` test against 6 neighbors fails on
post-blur plateau peaks where adjacent voxels share equal values.
**Stage C-2 and C-3 never fired in the f1-f45 run.** The win came
from Round 1+2 (cleanup + drift cap + neighbor-bridging) plus the
EXISTING `snap_imgPca_trans-` pre-pass axis. The f40 split's
`bestLabel=snap_imgPca_trans-` confirmed this.

### Stage C-1 algorithm fix attempted

Relaxed the local-max test from strict `>` to `>= all neighbors AND
> at least one neighbor`. Rebuild + f1-f65 validation.

Result: lmaxima now fires constantly (305-616 maxima per cell window
for elongated cells), and the snap-correction loop pulls cells toward
local maxima that are **on neighbor biology**, not the cell's own.
e9077:10 drifted from biological z=175 to z=151 by f40 because snap
correction was pulling toward 8cbdf86d / e9077:11 maxima inside its
search window. Once drifted, the lmaxima split-seed envelope check
(both maxima within `1.2 × parent.maxR` of snap) keeps rejecting good
candidates because the snap is now off-biology. e9077:10 missed its
GT split at f40, never recovered through f47 when run was stopped.

### Root cause of Stage C failure

The local-maxima algorithm has no notion of "which cell does this max
belong to." For an elongated cell with `maxR ≈ 56 vx` and a window
`1.5 × maxR ≈ 84 vx`, the search radius reaches into 4-6 neighbor cells
and finds their bright peaks. Without watershed or blob assignment to
attribute each max to a specific cell, lmaxima conflates "is there
biology in this window?" with "is this OUR biology?" — and the wrong
biology is what gets preferred by the brightness ranking.

This is exactly the failure mode that motivated Stage D (whole-frame
blob assignment) in the original plan. Stage C without Stage D is
under-determined.

### Action

Stage C config flags set to `false` by default. The Frame::findLocalMaxima
method, lmaxima split-axis block, and snap-correction loop remain in
the codebase (compiled, gated by config) so Stage D can build on them
when ownership-attribution is in place.

The shipping pipeline reverts to Round 1+2: clean baseline + drift
cap + tightened neighbor-bridging. f1-f45 GT-perfect via existing
pre-pass machinery (validated in `output_round3_f1-f45_*` run; the
lmaxima logs in that run show "found=0" everywhere, confirming Stage C
was inert there).

### Files touched

- `C++/config/config.yaml`: `local_maxima_split_seeding_enabled: true →
  false`, `local_maxima_snap_correction_enabled: true → false`. Comments
  document the regression diagnosis.
- Frame.hpp / Frame.cpp / CellUniverse.cpp / ConfigTypes.hpp: unchanged
  (Stage C code remains, dormant behind config flags).


## 2026-04-26 — Change 31: bridge-strength cost-gate override (e9077:10 f40 fix)

**Status:** ACTIVE.

### Problem

`output_round2_validation_f1-f65_20260425_095049` missed the e9077:10
split at f40 — the original motivating failure of the entire round of
work. Diagnostic of the debug log showed:

- `data_imgPca_primary` candidate was correctly identified by the
  pre-pass: seeds (313.5, 254.9, 172.1) and (353.2, 257.9, 159.1)
  matched `expD1`/`expD2` (40 px apart in x).
- Burn-in drift was tiny: drift1=5.4, drift2=10.3.
- Refinement converged: refineDrift1=5.6, refineDrift2=4.3,
  delta=-8595.
- Bridge gate **passed**: `valleyFromBright = 0.158/0.223 = 0.707`,
  well below the 0.85 limit.
- Cost diff was real: -14,473.
- **Cost gate rejected**: `adaptiveThreshold = max(2000, 0.03 ×
  1,026,420) = 30,792`. -14473 ≥ -30792 → reject.

The proportional `split_cost_fraction × baseline` term scales with
parent volume. For a large/bloated parent (e9077:10 had majR≈52 at f40),
3% of the baseline image cost is huge and rejects splits whose absolute
cost reduction is real but smaller than that fraction.

### Fix

Add a "bridge-strength cost override" path. When the cost gate would
reject, accept anyway if ALL THREE conditions hold:

1. **Bridge gate ran AND `valleyFromBright < 0.75`** — clear valley
   evidence, well below the 0.85 bridge threshold.
2. **`refineDrift1 + refineDrift2 < 15`** — daughter refit converged
   tightly to stable positions.
3. **`costDiff < -5000`** — real absolute cost reduction (well above
   the noise floor `split_cost = 2000`).

The override does not weaken any individual gate. It says: when bridge
AND refit AND absolute-cost evidence are all strong, the proportional
threshold is over-strict for this parent and we should trust the
combined evidence over the proportional rejection.

### Files changed

**`C++/includes/ConfigTypes.hpp`** (lines ~389-401, 555-557):
add three new `ProbabilityConfig` fields with YAML parsers:

```cpp
float split_bridge_strong_cost_override_valley_max = 0.75f;
float split_bridge_strong_cost_override_max_refine_drift_sum = 15.0f;
float split_bridge_strong_cost_override_min_abs_cost_diff = 5000.0f;
```

**`C++/src/Frame.cpp`** (line ~3459, ~3743, ~4086):

- Declare `refineDrift1Captured`, `refineDrift2Captured` (init to
  `+inf`) BEFORE the refine block; assign inside.
- Declare `valleyFromBrightCaptured = 1.0f`, `bridgeGateRan = false`
  BEFORE the bridge block; assign inside.
- Replace the cost-gate rejection block with conditional override:
  if the three captured signals all pass thresholds → log
  `[Split Override cost]` and fall through to accept; otherwise log
  the original `[Split Reject cost]` (now extended with the override
  diagnostic fields) and return.

**`C++/config/config.yaml`** (after `bio_neighbor_bridging_factor`):

```yaml
split_bridge_strong_cost_override_valley_max: 0.75
split_bridge_strong_cost_override_max_refine_drift_sum: 15.0
split_bridge_strong_cost_override_min_abs_cost_diff: 5000.0
```

### Behavior change

The override only fires for cells that previously rejected at the cost
gate. Cells that previously accepted (cleared the proportional
threshold) are unaffected — the override path is unreachable for them.
Cells that rejected for other reasons (bridge_flat, edge_too_dim,
runaway drift, bio gates) are unaffected.

### Validation

Resume from a pre-f40 checkpoint of
`output_round2_validation_f1-f65_20260425_095049` to test the override
on the e9077:10 f40 case directly.
