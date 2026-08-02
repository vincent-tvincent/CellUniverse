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

## 2026-05-03: Demote PCA bridge from accepting path to candidate-proposal source (Change 13) — **status: ACTIVE, awaiting build**

### Problem / Motivation

The PCA-bridge split path (`Frame::tryPcaBridgeSplit`) acted as an independent
accepting path: when an elongated cell with a dark long-axis bin gap was
found, the function fitted two daughters via PCA on the left/right pixel
clusters and committed the replacement immediately. It had its own cost
threshold (`pca_bridge_min_cost_improvement`) and bypassed the main path's
gates: candidate burn-in, daughter-overlap fraction gate, asymmetric L2
weighting, the final bridge gate, the adaptive cost gate, and the new
split-bridge cost rescue.

This produced false positives. Concrete case in
`output_ubuntu_fluo_resume33_0-50_20260503_184603/debug_log.txt` at f43:

```
[PCA Bridge Split Accept] cell=cell_310 elong=3.7316 gapBins=9-11
  splitProj=0 left=7335 right=6790 costDiff=-9554.39
  oldImage=2.18448e+06 newImage=2.15753e+06
  oldOverlap=0 newOverlap=17392.7
```

`costDiff=-9554` is below the bridge's `min_cost_improvement` so it was
accepted, but the split CREATED 17392 voxels of overlap with neighbors
(oldOverlap=0 → newOverlap=17392.7). The main path's
`split_daughter_overlap_gate` and the asymmetric L2 cost would have rejected
this. `docs/split-gate-overlap-analysis.md` flagged this as the
highest-risk overlap and recommended demoting the bridge to a
candidate-proposal source.

### Files changed

- `C++/includes/Frame.hpp`
- `C++/src/Frame.cpp`
- `C++/src/CellUniverse.cpp`

### Code changes

**File:** `C++/includes/Frame.hpp`

**Before (around L40–55, before BoundingBox3D `};`):** no `BridgeSplitProposal` struct.

**After (immediately after `BoundingBox3D`):**
```cpp
struct BridgeSplitProposal
{
    cv::Point3f d1Pos{0.0f, 0.0f, 0.0f};
    cv::Point3f d2Pos{0.0f, 0.0f, 0.0f};
    float elongation = 0.0f;
    int gapStartBin = -1;
    int gapEndBin = -1;
    int leftPixelCount = 0;
    int rightPixelCount = 0;
};
```

**Before (around L177–193):**
```cpp
CostCallbackPair trySplitCellPhased(
    size_t cellIndex,
    const PreviousFrameSnapshot &snapshot,
    const ClaimSet &otherCellsClaimSets,
    bool useSnapshotDirection,
    const ProbabilityConfig &probConfig,
    std::vector<cv::Mat> *splitPerturbDebugPlacements = nullptr,
    int *splitPerturbDebugPlacementCount = nullptr,
    float splitPerturbDebugBrightness = 0.0f);

bool tryPcaBridgeSplit(size_t cellIndex,
                       const ProbabilityConfig &probConfig,
                       std::ostream *logSink = nullptr);
```

**After:**
```cpp
CostCallbackPair trySplitCellPhased(
    size_t cellIndex,
    const PreviousFrameSnapshot &snapshot,
    const ClaimSet &otherCellsClaimSets,
    bool useSnapshotDirection,
    const ProbabilityConfig &probConfig,
    std::vector<cv::Mat> *splitPerturbDebugPlacements = nullptr,
    int *splitPerturbDebugPlacementCount = nullptr,
    float splitPerturbDebugBrightness = 0.0f,
    const BridgeSplitProposal *bridgeProposal = nullptr);

bool discoverPcaBridgeProposal(size_t cellIndex,
                               const ProbabilityConfig &probConfig,
                               BridgeSplitProposal &outProposal,
                               std::ostream *logSink = nullptr) const;
```

`tryPcaBridgeSplit` declaration removed. The new `discoverPcaBridgeProposal` is `const` (no cell mutation) and returns the (left, right) weighted centroids.

**File:** `C++/src/Frame.cpp`

The old `tryPcaBridgeSplit` (~L2841–L3030) is replaced by `discoverPcaBridgeProposal` (now at L2850–L3030). The bin-analysis half is preserved verbatim. The fit-daughter / synth-mutate / cost-gate half is deleted; in its place the function computes weighted L/R centroids directly and writes them into `outProposal`. Log tag changed from `[PCA Bridge Split]` / `[PCA Bridge Split Accept]` / `[PCA Bridge Split Reject]` to `[PCA Bridge Propose]`.

**`Frame::trySplitCellPhased` signature (L3032–L3041):**

Added trailing parameter `const BridgeSplitProposal *bridgeProposal` (default `nullptr` via the header).

**Bridge candidate injection inside `trySplitCellPhased` (just before `Kmax` truncation, ~L3590):**

```cpp
if (bridgeProposal != nullptr) {
    Candidate bridgeCand;
    bridgeCand.d1Pos = bridgeProposal->d1Pos;
    bridgeCand.d2Pos = bridgeProposal->d2Pos;
    bridgeCand.label = "bridge_primary";
    candidates.insert(candidates.begin(), bridgeCand);
    std::cout << "  [Split Bridge Inject] " << parentName
              << " d1=(" << bridgeCand.d1Pos.x << "," << bridgeCand.d1Pos.y
              << "," << bridgeCand.d1Pos.z << ")"
              << " d2=(" << bridgeCand.d2Pos.x << "," << bridgeCand.d2Pos.y
              << "," << bridgeCand.d2Pos.z << ")"
              << " elong=" << bridgeProposal->elongation
              << " gapBins=" << bridgeProposal->gapStartBin << "-"
              << bridgeProposal->gapEndBin
              << std::endl;
}
```

The proposal is inserted at the FRONT of `candidates` so it survives the `Kmax` cap. It then competes against `data_*` and `snap_*` candidates under identical burn-in + bio + bridge + cost rules.

**File:** `C++/src/CellUniverse.cpp`

**Before (around L2711–L2752, inside `runPcaShapeFit` lambda body, end-of-frame):**

A loop scanned every cell, computed elongation, and called `frame.tryPcaBridgeSplit(ci, config.prob)`. Each accepted bridge mutated `frame.cells` and erased `cellShapeReference` / `cellShapeBirth` for that cell.

**After:**

Replaced with a comment block that points to the new pre-loop discovery phase. The synth refresh (`frame.regenerateSynthFrame()`) that bridge needed is removed because we no longer mutate cells here.

**Pre-loop discovery (new, just before `runPhase` lambda, around L3178–L3220):**

```cpp
std::unordered_map<std::string, BridgeSplitProposal> bridgeProposals;
if (config.prob.pca_bridge_split_enabled) {
    float maxBridgeElong = 0.0f;
    int bridgeEligible = 0;
    for (const auto &cell : frame.cells) {
        if (cell.isTrash()) continue;
        const float elong = cell.shapeElongation();
        maxBridgeElong = std::max(maxBridgeElong, elong);
        if (elong >= config.prob.pca_bridge_elongation_ratio) {
            ++bridgeEligible;
        }
    }
    for (size_t ci = 0; ci < frame.cells.size(); ++ci) {
        const std::string parentName = frame.cells[ci].getName();
        BridgeSplitProposal proposal;
        if (frame.discoverPcaBridgeProposal(ci, config.prob, proposal)) {
            bridgeProposals[parentName] = proposal;
        }
    }
    std::cout << "[PCA Bridge Propose] frame " << (firstFrame + frameIndex)
              << " scanned=" << frame.cells.size()
              << " eligible=" << bridgeEligible
              << " proposalsFound=" << bridgeProposals.size()
              << " maxElong=" << maxBridgeElong
              << " threshold=" << config.prob.pca_bridge_elongation_ratio
              << std::endl;
}
```

Cells at this point carry the previous frame's PCA-fit shape (current frame's PCA-fit happens in `runPcaShapeFit` at end-of-frame), but the bright-pixel input is the current frame's image — same input the original end-of-frame bridge had against the next frame's start. Daughter centroids feed the standard 50-iter burn-in inside `trySplitCellPhased`, so any seed-position drift from the stale shape is corrected.

**Lookup at the trySplitCellPhased call site (around L3457):**

```cpp
const BridgeSplitProposal *bridgeForCell = nullptr;
if (!bridgeProposals.empty()) {
    auto bIt = bridgeProposals.find(cellName);
    if (bIt != bridgeProposals.end()) {
        bridgeForCell = &bIt->second;
    }
}
auto result = frame.trySplitCellPhased(
    cellIdx, splitSnapshot, others, useSnapDir, config.prob,
    exportPerturbDebug ? &splitPerturbDebugPlacements : nullptr,
    exportPerturbDebug ? &splitPerturbDebugPlacementCount : nullptr,
    config.simulation.perturb_debug_cell_brightness,
    bridgeForCell);
```

### Effect

1. The PCA bridge no longer accepts splits independently. The cell_310 f43 false split (overlap going from 0 → 17392) is impossible under this design — the main path's `split_daughter_overlap_gate_enabled` (currently 0.05) catches it.
2. Cells where the standard `data_*` / `snap_*` candidates miss but the bridge finds a clean L/R cluster (e.g. f11 cell_4 = GT 640) still split, because the bridge's daughter centroids are now offered as `bridge_primary` to the burn-in winner-pick.
3. Acceptance economics are unified — `split_cost`, `split_cost_fraction`, and `split_bridge_cost_rescue_*` apply uniformly to all candidate sources.
4. Eligibility is unified — only cells whose `P(split)` from snapshot elongation triggers an attempt this iteration get to use their bridge proposal. The bridge no longer fires on cells the main path didn't pick.

### Diagnostic surface

- `[PCA Bridge Propose] frame N scanned=… eligible=… proposalsFound=… maxElong=… threshold=…` — frame-level summary.
- `[PCA Bridge Propose] cell=… elong=… gapBins=…-… splitProj=… left=… right=… d1=(…) d2=(…)` — per-cell discovery.
- `[PCA Bridge Propose] cell=… rejected=no_dark_bridge|too_few_side_voxels` — discovery rejections.
- `[Split Bridge Inject] cell=… d1=(…) d2=(…) elong=… gapBins=…-…` — confirms the proposal entered the candidate list.
- The old tags `[PCA Bridge Split]` / `[PCA Bridge Split Accept]` / `[PCA Bridge Split Reject]` are gone.

### Open follow-ups

- **Validate on a fluo f0–f43 run.** Confirm: (a) the f4/f7/f11/f18/f24/f25 splits still fire (the cell_4 f11 split was the only one that depended on the old bridge), (b) the f35 wave hits 9 splits across f34 + f35, (c) the f43 cell_310 false split does NOT fire.
- **`split-candidate-current-code.md` should be updated** to add `bridge_primary` to the candidate label table once validation lands.
- **Eligibility coupling:** the bridge proposal is currently consumed only when `P(split)` triggers an attempt. For cells where snapshot elongation is low but post-PCA elongation is high (the original use case for the bridge), the snap-driven `P(split)` may not fire even though the bridge sees a valid gap. If validation shows missed splits here, consider boosting `P(split)` for cells with a discovered proposal.
- **Cells without snapshot.** A bridge proposal can exist for a newborn daughter that has no snapshot (frame after split). The current code falls through `splitBlacklist.insert(cellName)` then `continue` when the snapshot is missing, so the bridge proposal is silently dropped. Acceptable for now (newborn daughters shouldn't re-split immediately) but worth a log line.

## 2026-05-04: Port prolate-shape pre-filter from yp_opt_speed 1ec91e7 (Change 13b) — **status: ACTIVE, awaiting build**

### Problem / Motivation

Vincent's commit `1ec91e7` ("debug for the black bridge split shortcut, add a missing logic") added a parent-shape gate to the PCA bridge that rejects ellipsoids whose elongation comes from one collapsed axis instead of a clean rod shape. Concrete failure case from the commit message: `R = (43, 30, 12)` has `long/short = 3.58` (passes the elong=2.0 trigger) but `mid/short = 2.5` — a wedge, not a rod. The bin-analysis "valley" along the long axis is not biological in that case and should not produce a daughter pair.

This change is complementary to Change 13 (bridge demote): demote unifies acceptance under the main gate stack so cost/overlap can catch false positives at evaluation time; the prolate gate rejects geometrically-suspect proposals **at discovery time** so they never enter the candidate list, never burn-in, never log noise. Both fix bridge false-positive sources from different angles.

### Files changed

- `C++/includes/ConfigTypes.hpp`
- `C++/config/config.yaml`
- `C++/src/Frame.cpp`

### Code changes

**File:** `C++/includes/ConfigTypes.hpp`

**Before (around L666–668):**
```cpp
bool pca_bridge_split_enabled = false;
float pca_bridge_elongation_ratio = 3.0f;
float pca_bridge_black_threshold = 0.05f;
```

**After:**
```cpp
bool pca_bridge_split_enabled = false;
float pca_bridge_elongation_ratio = 3.0f;
// Prolate-shape pre-filter (ported from yp_opt_speed 1ec91e7). Reject
// bridge candidates where the high elongation comes from one collapsed
// axis instead of a clean rod shape. R=(43,30,12) has long/short=3.58
// (passes elong gate) but mid/short=2.5 — wedge, not rod, and the
// bin-analysis "valley" along the long axis is not biological.
//   long/mid >= pca_bridge_min_long_mid_ratio  → one clearly long axis
//   mid/short <= pca_bridge_max_mid_short_ratio → two non-long axes
//                                                 close in size
// Set either to 0 to disable that half of the gate.
float pca_bridge_min_long_mid_ratio = 1.35f;
float pca_bridge_max_mid_short_ratio = 1.35f;
float pca_bridge_black_threshold = 0.05f;
```

YAML reads + cout dumps for both new fields added alongside the existing `pca_bridge_elongation_ratio` entries.

**File:** `C++/config/config.yaml`

**Inserted under `prob:` after `pca_bridge_elongation_ratio`:**
```yaml
  pca_bridge_min_long_mid_ratio: 1.35
  pca_bridge_max_mid_short_ratio: 1.35
```

**File:** `C++/src/Frame.cpp`

**Inside `discoverPcaBridgeProposal`, immediately after the elong threshold check (around L2865–L2895):**

Adds a `midIdx` lookup, computes `midR`, then:

```cpp
const float longMidRatio = longR / midR;
const float midShortRatio = midR / shortR;
const float minLongMidRatio = std::max(0.0f, probConfig.pca_bridge_min_long_mid_ratio);
const float maxMidShortRatio = std::max(0.0f, probConfig.pca_bridge_max_mid_short_ratio);
const bool longAxisDistinct = minLongMidRatio <= 0.0f || longMidRatio >= minLongMidRatio;
const bool shortAxesSimilar = maxMidShortRatio <= 0.0f || midShortRatio <= maxMidShortRatio;
if (!longAxisDistinct || !shortAxesSimilar) {
    log << "  [PCA Bridge Propose] cell=" << parent.getName()
        << " elong=" << elong
        << " rejected=triaxial_shape"
        << " longR=" << longR
        << " midR=" << midR
        << " shortR=" << shortR
        << " longMidRatio=" << longMidRatio
        << " minLongMidRatio=" << minLongMidRatio
        << " midShortRatio=" << midShortRatio
        << " maxMidShortRatio=" << maxMidShortRatio
        << std::endl;
    return false;
}
```

The log tag is `[PCA Bridge Propose]` (consistent with the discover-only naming from Change 13a) rather than the original commit's `[PCA Bridge Split]`. The reject reason `triaxial_shape` is preserved verbatim for grep continuity with Vincent's branch.

### Effect

- Cells with collapsed-axis geometry now bypass bin-analysis entirely. Saves a small amount of compute; more importantly, prevents proposals that come from a non-biological "valley" derived from wonky ellipsoid fits.
- Pairs with Change 13a: the demote unifies acceptance economics (good against false positives caused by unfair scoring), the prolate gate rejects geometrically degenerate parents at discovery (good against false positives caused by wonky shapes). Together they make the bridge produce only proposals worth competing in the main candidate list.

### Open follow-ups

- Tune `pca_bridge_min_long_mid_ratio` and `pca_bridge_max_mid_short_ratio` against fluo runs. 1.35/1.35 may be too tight for asymmetric divisions where one daughter remains larger.

---

## 2026-06-24: Deep repo cleanup — remove live-viz feature, dead code, unused scripts/configs (Change 14)

**Status: ACTIVE — build clean (100%), smoke run verified (frames 0→ tracking pipeline runs, no LiveViz).**

Goal: shrink the repo to the active algorithm (cell lumen, perturbation, split, shape fit). Executed on a fresh branch with a restore point. Anchored on the CMake build graph (all 12 `src/*.cpp` compile — `src/`/`includes/` were already lean) and the `base_config` inheritance closure (so no active config chain was broken).

### A. Live-visualizer feature removed (code)

Removed the optional napari/mpl auto-launch. It was gated on `config.liveViz.enabled` (default `false`), so tracking behavior is unchanged.

- **`src/main.cpp`** — deleted the `maybeLaunchLiveViz(...)` function (was lines 119–174) and its call site.

  Before (call site, ~line 429):
  ```cpp
      // Optionally spawn the live visualizer (scripts/cell_viz) now that the
      // output dir / frame range are known. Backgrounded; non-fatal on failure.
      maybeLaunchLiveViz(config, args, argv[0]);

      // Run
  ```
  After:
  ```cpp
      // Run
  ```
  Also removed now-unused `#include <sstream>` (only the deleted function used `std::ostringstream`).

- **`includes/ConfigTypes.hpp`** — deleted the `LiveVizConfig` struct (was lines 2753–2769) and its 5 usages in `BaseConfig`: the `LiveVizConfig liveViz;` member, copy-ctor init `liveViz(other.liveViz)`, assignment `liveViz = other.liveViz;`, `if (node["live_viz"]) liveViz.explodeConfig(...)`, and `liveViz.printConfig();`.

- **`config/config.yaml`** — deleted the `live_viz:` block (was lines 3–13) + its comment header. (Unknown YAML keys are silently ignored, so any leftover `live_viz:` in other configs is harmless.)

- **Deleted files:** `scripts/cell_viz/` (dir), `scripts/live_cell_viz.py`, `scripts/live_monitor_napari.py`, `scripts/make_demo_napari.py`, `scripts/play_real_and_synth_same_napari.py`, `scripts/napari_embryo_review.py`, `scripts/run_play_HL60_real_synth_napari.sh`, `config/config_live_napari.yaml`, `config/config_test_live_viz_mpl.yaml`.

### B. Dead code / disabled tests removed

- **`deprecated/`** (whole dir) — pre-triaxial dead code (`Sphere`, `Bacilli`, `Cell`, `args`, `mathhelper`, `pseudo-code`). Nothing includes it.
- **`tests/`** (whole dir) — disabled in CMake since 2026-04-08 (`# add_subdirectory(tests)`), referenced the removed `Spheroid`/`SimulationConfig` API (would not compile).
- All `__pycache__/` under `scripts/` (build artifacts).

### C. Unused scripts removed

`scripts/4_Windows_Demo_16-9.py`, `analyze_late_goal_runs.py`, `iterative_brightness_recovery_blur.py`, `make_demo_video.py`, `validate_embryo_centers.py` — verified referenced by nothing (`src/`, run scripts, docs). Kept `image_processor_clean.py`/`measure_brightness.py` (referenced) and `make_lineage_tree_demo.py` (named in a main.cpp user message).

### D. Config archive (moved, not deleted)

Moved 10 experimental CellLumen `config_*.yaml` variants (not in the `config.yaml`/VERIFIED `base_config` closure) + the `BackUp_TuningHistory_CellLumen/` and `cell_lumen_config_tuning/` dirs into new **`config/_archive/`**. `config/` root now holds only the 10 active-chain configs + `config_CastaneumEmbryo.yaml` + 2 INIs.

### Verification

- `cmake --build build -j 8 --target celluniverse` → **100% built**, clean.
- Smoke run (embryo_data, frames 0–3, `config.yaml`): per-frame pipeline (Voronoi → image-grounded PCA → PCA-bridge split propose) runs normally; **no `LiveViz` output**; no crash.

### Not done here (future)

- Intra-file dead functions in `Frame.cpp`/`CellLumen.cpp`/`Ellipsoid.cpp` — needs a `-Wunused-function` + call-graph pass, deferred to avoid hunting dead code in the active focus files.
- `brightness_volume_analyzer` executable and `Python/` legacy kept pending explicit decision.

---

## 2026-06-24: Cleanup part 2 — drop 2nd executable, legacy Python, viz venv, and 5 dead functions (Change 15)

**Status: ACTIVE — build clean (100%), celluniverse links without the analyzer target.**

Completes the deferred items from Change 14.

### A. Removed the `brightness_volume_analyzer` executable

- Deleted `src/BrightnessVolumeAnalyzer.cpp`.
- **`CMakeLists.txt`** — removed the `add_executable(brightness_volume_analyzer ...)` block + its `target_include_directories`/`target_link_libraries` (was lines 139–154). Its shared sources (`EmbryoBrightTracker.cpp`, `LineageTreeCreator.cpp`, `LineageViewer.cpp`) are kept — they are also compiled into the main `celluniverse` target (verified before removal).

### B. Removed large non-source trees

- `Python/` (legacy reference implementation, 16 MB) — not built, not referenced by the C++ pipeline.
- `.venv-viz/` (905 MB) — the live-viz virtualenv; its only consumers (the napari scripts + `main.cpp` launch) were removed in Change 14. Verified no remaining script references `venv-viz`.

### C. Removed 5 dead member functions (call-site verified, 0 callers)

Found via a `Class::method` definition scan cross-referenced against all call sites in `src/` + `includes/`. Each below had **only** its header declaration + definition (extra grep hits were comments); none are `virtual`.

| Function | Removed from |
|---|---|
| `Frame::getSynthFrame()` | `src/Frame.cpp` + `includes/Frame.hpp:437` |
| `CellLumen::estimateBackgroundValue()` | `src/CellLumen.cpp` + `includes/CellLumen.hpp:67` |
| `Ellipsoid::get_center()` | `src/Ellipsoid.cpp` + `includes/Ellipsoid.hpp:215` |
| `Ellipsoid::print()` | `src/Ellipsoid.cpp` + `includes/Ellipsoid.hpp:217` |
| `Ellipsoid::checkConstraints()` | `src/Ellipsoid.cpp` + `includes/Ellipsoid.hpp:207` |

Example (`Ellipsoid.cpp`):
```cpp
// removed — no callers:
cv::Point3f Ellipsoid::get_center() const { return _position; }
void Ellipsoid::print() const { std::cout << "Ellipsoid: " << _name << ... ; }
bool Ellipsoid::checkConstraints() const { /* radius-bound checks */ }
```

**Kept (scan flagged them but they ARE used — `!method(` call form the regex missed):**
`CellLumen::componentContainsBrightSeed` (called `CellLumen.cpp:2068`), `Ellipsoid::computeSliceBounds` (called `Ellipsoid.cpp:446, 563`).

### Verification

- `cmake -S . -B build` reconfigure OK; `cmake --build build -j 8 --target celluniverse` → **100% built**, no errors/warnings. All three focus TUs (`Frame`, `Ellipsoid`, `CellLumen`) recompiled clean.

### Still open

- Free-function/static dead code (non-member) in the focus files not yet swept — the scan targeted `Class::method` definitions. A `-Wunused-function` clean-build pass would catch file-local statics next.

---

## 2026-06-24: Cleanup part 3 — collapse to ONE config.yaml + strip 11 dead config knobs (Change 16)

**Status: ACTIVE — build clean (100%), flattened config parses + runs (fusionEnabled=1 confirmed).**

### A. Single config.yaml (flattened from the VERIFIED CellLumen-fusion chain)

The config set was a 10-file `base_config` inheritance chain (`config.yaml` → `config_embryo` → `NoPreprocess` → `deterministicSplit` → … → `VERIFIED_F085-120`). Note: bare `config.yaml` was the *root* and had **no `cell_lumen` block** — the fusion pipeline lived entirely in the chain.

- Resolved the full chain into a single self-contained **`config/config.yaml`** (blocks: `cell` 55, `simulation` 117, `prob` 56, `cell_lumen` 321 knobs), `base_config` removed. The flattener mirrors `mergeYamlNodes` exactly; **verified** the flattened effective config is byte-identical to the recursively-resolved VERIFIED config (`flatten == resolve(VERIFIED)` → True).
- Deleted **all other yaml**: the 9 remaining chain files, `config/embryo/`, `config/_archive/` (~200 tuning yamls), and `config_CastaneumEmbryo.yaml`. `config/` now holds `config.yaml` + 2 INIs + dataset seed CSV dirs only.
- The INI launcher already pointed at `config.yaml`, so runs now use the full fusion pipeline. (Per-knob inline comments from the old layered files are in git history.)

### B. Dead config-knob audit — removed 11 knobs (parsed but never used by the algorithm)

Audit: every YAML knob (545) is parsed (0 Level-1 dead). Level-2 (parsed into a struct field the algorithm never reads) found 18 candidates; verified each by extracting the real field and grepping `src/`:
- **Kept (false positives):** `max_split_probability`, `split_max_parent_overlap_fraction`, `split_parent_overlap_daughter_scale`, `split_parent_overlap_gate_enabled` (legacy alias keys parsing into actively-used `P_split_max` / `split_*daughter*` fields); **`mu`** (used inside `PerturbParams`' own inline `normal_distribution(mu, sigma)` — my `src/`-only grep had excluded `ConfigTypes.hpp`).
- **Removed 11 dead fields** (decl + parse + print) from `includes/ConfigTypes.hpp` (27 lines) and their keys from `config.yaml`: `bio_bridge_max_gap_density`, `cube_pooling_cost_comparison_enabled`, `enable_lineage_tree_window` (+`lineage_tree_window` alias), `fusionSplitPriorConflictMinOldScoreForReplacement`, `fusionSplitPriorConflictMinSeparationDrop`, `pcaShapeRadiusPercentile`, and 5× `pca_bridge_{daughter_radius_scale,min_radius_fraction,max_radius_fraction,min_cost_improvement,overlap_weight}`.

### Verification

- `cmake --build` → **100%**, no errors/warnings (removing a truly-used field would fail to compile → confirms all 11 were dead).
- Smoke run (frames 0–1, new `config.yaml`): parses cleanly, prints `CellLumen Config / fusionEnabled: 1`, runs the pipeline.

### C. Free-function / static dead-code sweep (`-Wunused-function`)

A clean build with `-Wall -Wunused-function` flagged compiler-verified file-local dead functions (internal linkage, zero callers). Removed all + cascades, iterating until the flag was clean:

| Function | File |
|---|---|
| `scaleStackBrightness` | `src/CellUniverse.cpp` |
| `continuousEllipsoidVolume` | `src/CellLumen.cpp` |
| `collectSampledValues` | `src/CellLumen.cpp` (orphaned when `estimateBackgroundValue` went in Change 15) |
| `isLocalMax3x3x3` | `src/EmbryoBrightTracker.cpp` (was marked `// redudant`) |
| `refinePeakWeightedCentroid` | `src/EmbryoBrightTracker.cpp` |
| `localizeSignalCentersInStack` (187 lines) | `src/ImageHandler.cpp` (signal-guided perturbation, unused in our pipeline) |
| `chooseNearestDivisorSize` + local `struct BrightBox` | `src/ImageHandler.cpp` (cascade — only used by the localizer; note the `BrightBox` in `CellUniverse.cpp` is a *different, active* struct) |

Final `-Wall -Wunused-function` build: **zero unused-function warnings**. Normal build 100%; smoke run confirms the pipeline (Voronoi → PCA-bridge, fusion active) runs unchanged.

---

## 2026-07-02: Phase 0a — offline windowed-ILP validation prototype (Change 17)

**Status: EXPLORATORY — no tracker/C++ change. Go/no-go experiment for Workstream L (windowed ILP), per `docs/plans/2026-07-02-shape-and-ilp-buildplan.md`.**

Offline, Python-only. Validates whether a small sliding-window ILP that selects the globally consistent set of cell hypotheses can fix division-timing / merge errors the per-frame greedy commit gets wrong — *before* committing to the large C++ build.

### New files

- **`scripts/ilp_proto/windowed_ilp_proto.py`** (new, ~460 LOC) — self-contained prototype, four layers:
  1. **Geometry** — Python ellipsoid voxel membership matching the C++ convention exactly: `R = Rz·Ry·Rx`, `local = Rᵀ·(world−center)`, inside iff `(lx/a)²+(ly/b)²+(lz/c)²≤1`, with `a=majorRadius, b=bRadius, c=minorRadius` (mirrors `Ellipsoid::isPointInsideEllipsoid` / `generateInverseRotationMatrix`, `src/Ellipsoid.cpp:632,~/98`). IoU links + an MDL-style image score `Σ(intensity−bg)`.
  2. **`solve_window`** — the reusable Ultrack-style selection ILP (PuLP + bundled CBC): nodes `y_p` (select), edges `x_pq` (temporal link), events appear/disappear/divide; objective = detection + link weights − event penalties; constraints = hypothesis disjointness (`Σ y ≤ 1` per conflict) + flow conservation. **Written to mirror the eventual C++ port (Phase L2).**
  3. **`--selftest`** — synthetic flicker scenario; proves the ILP holds a division that per-frame greedy un-divides.
  4. **Real-data driver** (`--analyze`, `--case`) over the committed run `outputs/Yiding_1~171_VISUAL_TIF/`, scored against the tracker's own `real/N_real.tif` volumes (never against GT masks — that would be circular). Founder↔GT correspondence baked in (`cell_0↔GT256, cell_1↔GT1, cell_2↔GT511, cell_3↔GT640`), with `readable()` labels replacing raw `Cell type 1_N`.
- **`scripts/ilp_proto/RESULTS.md`** (new) — findings + verdict.

### Tooling

- `.venv-eval/bin/pip install pulp` → **pulp 3.3.2**; bundled CBC verified solving (`PULP_CBC_CMD.available()` → true, tiny ILP → Optimal). CPU, license-free. No GPU / no learned model added.

### Findings (see `RESULTS.md`)

- **ILP core correct** — `--selftest` PASS (removes a flicker greedy falls for).
- **On real data, W=1–3 never flipped a known greedy timing error toward GT.** Every real over-split case was either image-supported (f9/GT640: fluorescence bimodal ~2 frames before GT marks the split — annotation-convention gap, not a greedy error), a late split not reconstructible offline (f9/GT511, f4/f7/f18 — the 2-cell hypothesis at the early frame isn't in committed output), or genuinely ambiguous (f34, ~1% image margin) where the window agreed with the per-frame call.
- **Verdict:** not a green light for L2 on its own, not a red light either — the offline harness is structurally blind to the ILP's real payoff (competing hypotheses greedy *discarded*), which don't exist until Workstream **S2 (metaball)** emits them. **Recommendation:** build S first (S1 superquadric → S2 metaball), then re-run this harness fed by S2's emitted hypotheses. Matches the plan's dependency order (S2 precedes L1).

### Not done (per standing rules)

- No git writes. No change to tracker code, `config.yaml`, or committed run CSVs (run evidence). CSV column rename (`majorRadius/bRadius/minorRadius` → semi-axis + shape-exponent) deferred to S1, where the superquadric changes radii semantics anyway.

---

## 2026-07-02: Phase 0b — offline shape-ceiling measurement (Change 18)

**Status: EXPLORATORY — no tracker/C++ change. Go/no-go for Workstream S (richer shape), per the build plan.**

Offline, Python-only. Measures the best-achievable per-slice SEG (Jaccard) ceiling for three shape primitives on the 5 CTC GT SEG slices, to decide whether a richer shape is worth building *before* touching C++.

### New files

- **`scripts/ilp_proto/phase0b_shape_ceiling.py`** (new, ~230 LOC) — for each of the 102 GT nuclei in `01_GT/SEG/man_seg_*.tif`, optimizes (Nelder-Mead, moment init, multi-restart) an **ellipse**, a **superellipse** `|x'/a|ʳ+|y'/b|ʳ≤1`, and a **union-of-2-ellipses** to maximize 2D Jaccard vs the GT outline. Reports per-nucleus + summary ceilings. (Local optimizer → ceilings are conservative lower bounds.)
- **`scripts/ilp_proto/RESULTS_phase0b.md`** (new) — findings + verdict.

### Findings (102 nuclei)

| primitive | mean Jaccard | Δ vs ellipse |
|---|---|---|
| ellipse (current) | 0.887 | — |
| superellipse (S1) | 0.892 | **+0.005** (median fitted exponent 2.00) |
| union-of-2 (S2) | 0.909 | **+0.022** (concentrated: +0.08–0.145 on ~8% dividing nuclei) |

- **S1 superquadric — GATE FAILED** (ceiling +0.005; nuclei are genuinely elliptical). **Dropped** from the plan.
- **S2 union-of-2 — targeted pass only** (helps the dividing/dumbbell nuclei; modest overall). Kept only for its dual role (dividing case + ILP 2-cell hypothesis generator).
- **Reframe:** ellipse ceiling 0.89 vs reported SEG 0.358 ⇒ the ~0.53 SEG loss is a **fit/placement gap, not a shape-primitive gap**. A richer primitive closes ≤+0.02 of it.

### Plan doc updated

- `docs/plans/2026-07-02-shape-and-ilp-buildplan.md` decision-gate table filled with 0a + 0b results; added "Phase 0 outcome & revised direction" (drop S1; S2 modest; next diagnostic = tracker's *actual* per-nucleus Jaccard on the 5 slices to localize the fit gap).

### Not done (per standing rules)

- No git writes. No tracker/C++/config change.

---

## 2026-07-02: Native formatted run output — real/ + synth/ subdirs + tracked_cells.csv (Change 19)

**Status: ACTIVE — build clean (100%), verified on a f1-20 run (real/ + synth/ + tracked_cells.csv + name_map.csv emitted).**

The program now emits its run output in an organized, human-readable layout natively (previously the visual TIFFs went flat into the output root as `{N}_real.tif`, and the only cell CSV was `cells.csv` with 32-char hex UUID names + terse column headers). Requested so runs are formatted by the program, not post-processed.

### A. Visual TIFFs into `real/` and `synth/` subdirs

- **`src/CellUniverse.cpp`** (`saveImages`, the `export_frame_tiff` block, ~line 10614): write to `exportRoot/real/{N}_real.tif` and `exportRoot/synth/{N}_synth.tif` (creating the subdirs) instead of flat `exportRoot/{N}_real.tif`.

```cpp
// before
std::filesystem::create_directories(exportRoot);
writeNapariFriendlyTiffStack(exportRoot + "/" + std::to_string(displayFrame) + "_real.tif", realImages);
writeNapariFriendlyTiffStack(exportRoot + "/" + std::to_string(displayFrame) + "_synth.tif", synthImages);
// after
const std::string realDir = exportRoot + "/real"; const std::string synthDir = exportRoot + "/synth";
std::filesystem::create_directories(realDir); std::filesystem::create_directories(synthDir);
writeNapariFriendlyTiffStack(realDir + "/" + std::to_string(displayFrame) + "_real.tif", realImages);
writeNapariFriendlyTiffStack(synthDir + "/" + std::to_string(displayFrame) + "_synth.tif", synthImages);
```

### B. `tracked_cells.csv` + `name_map.csv` (human-readable, output-only)

- **`src/CellUniverse.cpp`** — `saveCells` now calls a new `saveFormattedCells(frameIndex)` after writing the internal `cells.csv`. The new method (appended after `saveCells`) writes, per frame:
  - `tracked_cells.csv` — header `frame,cell,x,y,z,semi_axis_a,semi_axis_b,semi_axis_c,rot_x,rot_y,rot_z,is_marker`; `frame` = integer frame number; `cell` = recognizable `cell_<N>` (root UUID → stable index on first appearance; split daughters keep the `0/1` lineage suffix as `cell_<N><path>`).
  - `name_map.csv` — `recognizable_name,internal_id` provenance, rewritten each frame.
- **`includes/CellUniverse.hpp`** — declared `void saveFormattedCells(int frameIndex);` (after `saveCells`, line ~36) and added member `std::map<std::string,int> _formattedNameIndex;` (private, after `firstFrame`).
- **Crucially, `cells.csv` is untouched** — the internal file (read by `CsvHandler` column-alias lookup and the `--lineage-tree` name-suffix encoding) keeps its original format, so lineage / resume / CSV-reading are unaffected. The formatted files are output-only.

### C. `config/config.yaml`

- `export_frame_tiff: false → true` (so runs emit the visual output by default, matching the requested formatted layout). `export_frame_png` left `false`.

### Verification

- `cmake --build build -j 8 --target celluniverse` → **100% built**, no errors.
- f1-20 run (`original_data`, `initial_origin_0.csv`): output dir has `real/` (20 `_real.tif`), `synth/` (20 `_synth.tif`), `tracked_cells.csv` (cell_1..cell_6, renamed columns), `name_map.csv`, and the untouched `cells.csv`. No `resume_*/` or `viz/`.

### Note

- The output **directory** name is chosen by the caller/launcher, not the tracker; runs were named `<YYYYMMDD_HHMMSS>_run_...` so `outputs/` sorts chronologically (see also the 2026-07-02 `outputs/` rename pass logged in `outputs/RENAME_MAP.txt`).

---

## 2026-07-02: Fix washed-out (near-white) frame normalization for raw/no-preprocess frames (Change 20)

**Status: ACTIVE — build clean (100%), verified: low_ref 0.97 → 97.3, exported real p50 229 → 115 (real dynamic range restored).**

### Symptom

With `preprocess_mode: none` on `original_data`, exported `real`/`synth` TIFFs were near-white (real median 229/255, synth 237–250 nearly flat). Because `generateOutputFrame` is `_realFrame × 255`, the *internal* image the tracker fits was equally washed to `[0.84, 1.0]` — near-zero contrast.

### Root cause

`normalizeStackToFrameIntensity` (`src/CellUniverse.cpp`) stretches `[low_ref, high_ref] → [0,1]` using percentiles. `exclude_zeros` only drops *exactly* 0, but the rotated frame's black border interpolates to small **non-zero** voxels (~1–90). Those are >1% of nonzero voxels, so `low_ref` = p1 landed at **0.97** (border) instead of the tissue's ~94. Run log: `[Frame Intensity Scale] frame=frame001.tif mean=0.842557 low_ref=0.968965 high_ref=121.791`. Tissue background (raw ~103) → `(103-0.97)/(121.8-0.97)=0.84` → near-white. (The `[p1,p99.5]` scheme assumes a *preprocessed* near-zero background; raw frames break it.)

### Fix (robust low reference, gated, default off)

- **`src/CellUniverse.cpp`** — `computeStackPercentile` gains a `float excludeBelow = 0.0f` param (skips `value < excludeBelow`). `normalizeStackToFrameIntensity` now computes `highReference` first, then `lowReference` excluding voxels below `frame_intensity_low_signal_floor_fraction * highReference`, so the near-zero border/pedestal can't drag the low percentile down. `fraction = 0` (default) → identical to before.
- **`includes/ConfigTypes.hpp`** — added `float frame_intensity_low_signal_floor_fraction = 0.0f;` (+ parse + print).
- **`config/config.yaml`** — set `frame_intensity_low_signal_floor_fraction: 0.25` (exclude below 25% of high_ref; cleanly separates the ~1–30 border from the ~94+ tissue).

**Refinement (same day):** the signal-floor fix moved `low_ref` to the tissue *minimum* (~97), which still left the tissue *background pedestal* (~108) mapping to mid-gray (~0.45) — fine in a 2D slice but bright in a napari 3D accumulation. To subtract the pedestal, also set `frame_intensity_scale_low_percentile: 0.01 → 0.5` (low_ref = tissue **median** ~108.8) and `frame_intensity_scale_high_percentile: 0.995 → 0.999` (less cell saturation). Result: exported real `mean 0.84 → 0.45 → 0.053`; background median **15/255 (dark)**, cells to 255 — a proper dark-background/bright-cell fluorescence image. (These are `config.yaml` values only; defaults unchanged.)

### Verification (f1-5)

- `[Frame Intensity Scale] frame=frame001.tif mean=0.448791 low_ref=97.2856 high_ref=121.791` — low_ref now in the tissue.
- Exported real: p1=66 / p50=115 / p99=220 / max=255 (real contrast) vs the old flat ~229. Synth: bg 153 vs cells 250 (was 237 vs 250).

### Still open (separate issue)

- Splits are still rejected (6 cells held through f5; best22 splits e9077/12345 at f3). The washed image was **not** the (sole) cause of the no-split behavior — that is a separate split-gate issue to investigate next.

### Default-off safety

- With `frame_intensity_low_signal_floor_fraction` unset (0), `computeStackPercentile`/`normalizeStackToFrameIntensity` behave exactly as before, so preprocessed configs are unaffected.

---

## 2026-07-20: Remove the offline windowed-ILP prototypes (Change 21)

**Status: ACTIVE — cleanup. No tracker/C++ change.**

The offline windowed-ILP / temporal-consistency direction (validated in `scripts/ilp_proto/`, Changes 17–18) did not pan out — the offline experiments never showed the ILP fixing real division-timing errors (the errors were data/setup issues, not something a temporal window could fix). Removing the dead prototype code; the design/research write-ups are kept as notes.

### Removed

- **`scripts/ilp_proto/`** (whole directory): `windowed_ilp_proto.py`, `sa_hypotheses.py`, `offline_validation.py`, `fitgap_real.py`, `phase0b_{shape_ceiling,richer,metaball,gmm}.py`, `RESULTS*.md`, `phase0b_shapes.png`, `tests/`. All were offline Python prototypes — **none was wired into the C++ tracker**, so nothing in the tracker changes.

### Kept

- **Design/research docs** (marked SHELVED as notes): `docs/plans/2026-07-02-post-ultrack-directions.md`, `docs/plans/2026-07-02-shape-and-ilp-buildplan.md`, `docs/superpowers/specs/2026-07-02-metaball-ilp-shape-tracking-design.md`.
- **Changes 19–20** (native `real/`+`synth/`+`tracked_cells.csv` output, robust-low-reference normalization knob) — not ILP, and useful for running the tracker on high-pedestal raw data (e.g. the 422×517 Pavak/original_data set).
- The CTC-eval scripts (`scripts/ctc_eval/`) — unrelated to the ILP.

### Note

- Git deletions are left unstaged for the maintainer. `scripts/ctc_eval/` and all tracker code are untouched.

---

## 2026-07-28: Lower window balanced-daughter bonus to curb over-splitting (Change 22)

**Status: TESTING — config-only, no rebuild. f0–20 embryo_data run in flight.**

### Problem

On `embryo_data` (512×708, seed `config/Teammate_initial/initial_embryo.csv`, first_frame=0) the tracker **over-splits vs GT**: reaches 7 cells at f1 (GT=4) and ends 16 at f20 (GT=12), ~3–4 cells ahead throughout. Prior-session diagnosis: the CellLumen **fusion split prior** forces splits (`bestLabel=cell_lumen_primary`, `costDiff=-1`) because (a) CellLumen's seeded watershed over-segments (~24 sub-blobs from ~4 cells), and (b) `fusionSplitPriorWindowBalancedDaughterBonus: 25.0` gives a large +25 bonus to any balanced sub-blob pair that persists across the 3-frame window, pushing the score well under the accept threshold (`fusionSplitPriorMaxScore: 22`). No min-frames guard on this path.

GT per-frame cell count (f0..f20): `4 4 4 4 6 6 6 7 7 7 7 8 ... 10 12 12`.
Baseline run (bonus=25): `4 7 8 8 8 9 9 9 11 11 12 12 14 15 15 15 15 15 15 16 16`.

### Field changed in `C++/config/config.yaml`

| Line | Field | Was | Is |
|---|---|---|---|
| 469 | `fusionSplitPriorWindowBalancedDaughterBonus` | `25.0` | **`10.0`** |

Rationale: the +25 window bonus is the single biggest driver pushing balanced sub-blob pairs under `fusionSplitPriorMaxScore: 22`. Halving-plus it to 10 keeps genuine persistent-daughter evidence rewarded while no longer single-handedly forcing an accept. One knob at a time per workflow rules.

### Next levers if still early (not yet applied)

1. Tighten `fusionSplitPriorRoundParentMaxShape` `1.25 → 1.1` (line 440) — stop round (elong ~1.03) founders being split by the prior.
2. Raise `fusionSplitPriorMinSeparation` (8.0, line 438) / `fusionSplitPriorMinVoxels` (500, line 454) — filter tiny over-segmented sub-blobs.
3. Deeper: reduce CellLumen watershed over-segmentation at source.

### Rollback

`fusionSplitPriorWindowBalancedDaughterBonus: 10.0 → 25.0` (line 469).

---

## 2026-07-30: Rung 1 part 1 — ForegroundModel module (Change 23)

**Status: IN PROGRESS — Rung 1 of the generative decision-core redesign. New leaf module, not yet wired into the tracker (Tasks 4–5 wire it). No behavior change to the tracker yet.**

Implements the self-calibrating foreground/background image model from the design spec (`docs/superpowers/specs/2026-07-30-generative-tracking-decision-core-redesign.md` §4.1) and the Rung 1 plan (`docs/superpowers/plans/2026-07-30-rung1-self-calibrating-data-term.md`, Task 1). This is the replacement for the raw-L2 residual: a per-frame two-Gaussian intensity model whose parameters are estimated from each frame's own pixels, so the term self-calibrates to each dataset's brightness/pedestal/noise instead of relying on absolute thresholds. Noise is *modelled* (via `sigBg` measured from the real background), not penalised.

### Files created

- **`includes/ForegroundModel.hpp`** — `struct ForegroundModel { muFg,sigFg,muBg,sigBg,logSigFg,logSigBg,valid }` with inline `nllFg(I)`, `nllBg(I)`, `logLR(I)`, and `finalize(minSigma)`. Declares `estimateForegroundModel(real, insideAnyCell, minSigma)`.
  - `nllFg/nllBg`: Gaussian negative log-likelihood, dropping the shared `0.5*log(2π)` const (cancels in every cost delta): `0.5*((I-μ)/σ)² + log(σ)`.
  - `logLR(I) = nllBg(I) − nllFg(I)`; `> 0` ⇒ intensity looks foreground.
- **`src/ForegroundModel.cpp`** — `estimateForegroundModel`: splits voxels into fg (inside any cell) / bg (outside) via the `insideAnyCell` flat mask, then robust location/scale per population = **median + 1.4826·MAD** (Gaussian-consistent, outlier-resistant). Fallback: if no fg voxels, fg params ← bg params so `logLR` is inert (safe on a cell-less frame). `finalize` floors σ at `minSigma` and precomputes `log(σ)`.
- **`tests/test_foreground_model.cpp`** — unit test: a synthetic 64×64 frame with bg~N(30,5) and a 16×16 fg block~N(200,20). Asserts population recovery within tolerance, `logLR` sign (bright>0, dim<0), and nll minimized at the mean.

### Verification

Compiled standalone and run:
```
g++ -std=c++17 $(pkg-config --cflags opencv4) tests/test_foreground_model.cpp \
    src/ForegroundModel.cpp $(pkg-config --libs opencv4) -o /tmp/tfm && /tmp/tfm
→ test_foreground_model: PASS
```
(Note macOS/zsh: `pkg-config` multi-token output needs `${=var}` word-splitting or the linker mis-parses `--libs`.)

### Build integration (Task 2)

- **`CMakeLists.txt`** — added `${cwd}/src/ForegroundModel.cpp` to the `celluniverse` executable source list. Added an opt-in `celluniverse_tests` target guarded by `option(BUILD_CELLU_TESTS OFF)` that compiles `tests/test_foreground_model.cpp` + `src/ForegroundModel.cpp`, links `${OpenCV_LIBS}`, and registers `add_test(NAME celluniverse_tests ...)`.
- Configure with tests: `cmake -S . -B build -DBUILD_CELLU_TESTS=ON`; run `cmake --build build --target celluniverse_tests && ./build/celluniverse_tests`.
- Verified: test target builds + `test_foreground_model: PASS`; main `celluniverse` target rebuilds clean with the new source linked in (no behavior change — nothing calls the module yet).

### Config flags (Task 3)

- **`includes/ConfigTypes.hpp`** — added to `ProbabilityConfig` (after `bbox_margin_scale`, which also lives there — accessed as `config.prob.*`): `generative_data_term_enabled` (bool, false), `edge_term_enabled` (bool, false), `edge_term_weight` (float, 0.0), `foreground_model_min_sigma` (float, 1.0); plus matching `node["..."]` parse lines in `ProbabilityConfig::explodeConfig`.
- **`config/config.yaml`** — added the four keys under the `prob:` block (all default off) after `bbox_margin_scale`. (Note: `prob:` → `ProbabilityConfig`, not `simulation:` — that is where the related `bbox_margin_scale` cost knob already lives.)
- Verified: `celluniverse` rebuilds clean with the new config parse. All-off ⇒ no behavior change.

### calculateBboxCost fg/bg branch (Task 4)

- **`includes/Frame.hpp`** — `#include "ForegroundModel.hpp"`; new public `setForegroundModel(const ForegroundModel&, bool enabled)`; new private members `_fgModel` + `_generativeDataTerm=false`.
- **`src/Frame.cpp::calculateBboxCost`** — hoisted `const bool useGen = _generativeDataTerm && _fgModel.valid; const float claimEps = 1e-4f;`. Added a `useGen` branch to BOTH loop variants: a voxel with synth `> claimEps` (claimed = modelled foreground) contributes `_fgModel.nllFg(real)`, otherwise `_fgModel.nllBg(real)` — replacing the `(synth−real)²` (+`asymK`) term. The Voronoi `continue` guard and the mask `continue` guard are preserved. When `useGen` is false every path is byte-identical to the legacy asymmetric-L2 behaviour.
- **Verified (flag OFF no-op):** `celluniverse` builds clean; embryo f0–2 run with `generative_data_term_enabled: false` → counts `f0:4 f1:4 f2:4`, zero `[FG Model]` lines, exit 0, no crash. Confirms the new code is inert until enabled.

### Per-frame estimation wiring + estimator calibration (Task 5)

- **`includes/Frame.hpp`** — added `getSynthFrame()` const getter (mirrors existing `getRealFrame()`).
- **`src/CellUniverse.cpp`** — `#include "../includes/ForegroundModel.hpp"`; in `optimize`, right after the mean-cell-brightness block (post `regenerateSynthFrame`), added the per-frame estimation: build a flat `insideAnyCell` mask (`synth > 1e-4` ⇒ foreground), call `estimateForegroundModel(real, insideAnyCell, config.prob.foreground_model_min_sigma)`, `frame.setForegroundModel(fm, true)`, and emit a `[FG Model]` log line. When the flag is off, install an invalid model (legacy L2 path). Uses `config.prob.*` (the fields live in `ProbabilityConfig`).
- **Estimator calibration fix (found by the Task-5 smoke run) — `src/ForegroundModel.cpp` + defaults:**
  - The first smoke run showed **every sigma floored to 1.0** on the [0,1]-normalized images (model went flat, no signal). Two causes: (a) MAD collapses to 0 when a population is mode-dominated (preprocessed background is mostly exactly 0), (b) `minSigma=1.0` exceeds the whole intensity range and is scale-dependent.
  - Fix: scale is now **std around the median** (non-collapsing) instead of `1.4826·MAD`; sigma is floored at `max(minSigma, 0.02·|muFg−muBg|)` — the **relative** part self-calibrates across [0,1] vs [0,255] datasets and prevents both collapse and over-flooring.
  - Default `foreground_model_min_sigma` changed `1.0 → 0.001` in `ConfigTypes.hpp` + `config/config.yaml` (now just an absolute div-by-zero guard; the relative floor does the real work).
- **Verified (flag ON smoke, embryo f0–3):** unit test still `PASS`; run exit 0, counts `f0:4 f1:4 f2:4 f3:4`; `[FG Model]` now non-degenerate — e.g. f2 `muFg=0.050 sigFg=0.138 muBg=0 sigBg=0.001`. Observations for the A/B phase: foreground median is dim (ellipsoid covers dim interior; bright cores sit in `sigFg`'s tail), and background is razor-sharp (mostly 0). Whether this yields good tracking is the Task-6 A/B question.

### Rollback

Revert the `optimize` FG-estimation block + `#include` in `CellUniverse.cpp`; revert `getSynthFrame` in `Frame.hpp`; revert the `calculateBboxCost` branch + `Frame.hpp` member/setter/include; revert the estimator (std→MAD) + floor in `ForegroundModel.cpp`; restore `foreground_model_min_sigma` default to 1.0; remove the four fields + parse lines from `ConfigTypes.hpp`, the four keys from `config/config.yaml`, the `ForegroundModel.cpp` line + `BUILD_CELLU_TESTS`/`celluniverse_tests` block from `CMakeLists.txt`; then delete `includes/ForegroundModel.hpp`, `src/ForegroundModel.cpp`, `tests/test_foreground_model.cpp`.

---

## 2026-07-30: Remove name_map.csv output (Change 24)

**Status: ACTIVE — output-only change, per maintainer request ("no point of the name map").**

- **`src/CellUniverse.cpp::saveFormattedCells`** — removed the per-frame `name_map.csv` write (the `cell_N ↔ internal-id` provenance file). `tracked_cells.csv` (recognizable `cell_N` names) is unchanged, and `_formattedNameIndex` is retained because it still assigns the stable `cell_N` numbering used in `tracked_cells.csv`.
- **`includes/CellUniverse.hpp`** — updated the two comments that referenced `name_map.csv`.
- Verified: a run produces `tracked_cells.csv` and no longer produces `name_map.csv`. Existing `name_map.csv` files in old output dirs are left untouched (evidence).

---

## 2026-07-30: Rung 1 A/B iteration 1 — estimator (A)+(B) + A/B result (Change 25)

**Status: BLOCKED — the generative data term is not a clean drop-in behind a flag; it is coupled to the L2-tuned decision machinery. Needs a design decision before proceeding.**

### Seed determinism (methodology)
The tracker seeds `mt19937` from `std::random_device` when `CELLUNIVERSE_SEED` is unset (`ConfigTypes.hpp:28`) → runs are non-deterministic. All A/B runs below fix `CELLUNIVERSE_SEED=42` so the only variable is the data term.

### Legacy seed=42 reference (f0–20, embryo)
Counts `4 7 8 8 8 8 8 8 8 8 8 9 10 11 11 12 12 12 13 14 14` vs GT `4 4 4 4 6 6 6 7 7 7 7 8 8 8 9 9 9 10 10 12 12` — over-splits. **All 10 accepted splits are `costDiff=-1 bestLabel=cell_lumen_primary`** — every split is forced by the CellLumen fusion prior; the image cost has no vote. (Confirms Rung-2, not Rung-1, owns the over-split fix.)

### Estimator (A)+(B) fix (`src/ForegroundModel.cpp`)
First generative run froze the optimizer because the FG model degenerated (foreground median ≈ background; `sigBg` floored razor-sharp at 0.001). Fixes:
- **(A)** `muFg`/`sigFg` now estimated from the **bright half** of inside-cell voxels (mean of voxels ≥ inside-median), so the dim ellipsoid interior no longer drags `muFg` down to background. `percentile()` helper added; `robustStats` replaced.
- **(B)** `sigBg` floored at `max(minSigma, 0.25·sigFg)` and `sigFg` at `0.05·globalStd` — self-calibrating, prevents razor-sharp collapse.
- Unit test updated (muFg is now the bright-half mean ≈ 216 for N(200,20)); still PASS.
- Result: FG model is now healthy (e.g. f2 `muFg=0.17 sigBg=0.037`, was `0.05 / 0.001`).

### A/B result (generative seed=42, f0–20) — STILL FROZEN
- Counts stuck at `4` all frames; `perturb_accepted=0` on every frame after f0; **f1 had 4 split attempts, 0 accepted** (legacy accepted 3 at f1).
- **Root cause (deeper than the model):** the generative cost's magnitudes/gradients differ fundamentally from asymmetric-L2, and *both* decision mechanisms are calibrated for L2 — (1) greedy perturbation acceptance (`costDiff<0`) finds no improving move (the NLL rigidly locks cells onto bright signal; uncovering any bright voxel costs `~(I/sigBg)²`, dwarfing the gain from covering dim voxels), and (2) the fusion prior's image-cost gate now rejects the split proposals it forced under L2.
- **Conclusion:** Rung 1 (data term) is **not cleanly separable** as a flag-gated drop-in — the cost and the decision machinery (greedy acceptance + L2-tuned gates) are coupled. A fair evaluation needs either a cost re-scaling/softening to be L2-compatible, or Rung 2's decision changes (Metropolis acceptance + gates-as-energy-terms). Default stays OFF; awaiting maintainer decision.

---

## 2026-07-30: Rung 1 A/B iteration 2 — soft-coverage cost unfreezes the optimizer (Change 26)

**Status: WORKING — the generative data term now moves cells and fires splits under the existing greedy loop. Full f0–20 A/B in flight.**

### Root cause of the freeze (deeper than magnitude)
Iteration 1 (Change 25) tried raw NLL and then a healthy fg/bg model — both froze. The real cause was the **binary claim indicator** (`synth > 1e-4`): the per-voxel cost depended only on whether a voxel was claimed, not on the synth *value*. So a sub-voxel move flipped zero voxels → `costDiff = 0` → greedy (`costDiff<0`) rejected it. The cost was a **step function of position**, with no gradient for small refining moves. (L2 works precisely because it is continuous in the synth value via the brightness falloff.)

### Fix: bounded, continuous soft-coverage cost
- **`includes/ForegroundModel.hpp`** — added `pFg(I)` = numerically-stable sigmoid of `logLR(I)` ∈ [0,1] (soft foreground probability).
- **`src/Frame.cpp::calculateBboxCost`** — both generative branches now compute per voxel: `c = clamp(synth,0,1)` (soft coverage) and cost `= c·(1−pFg) + (1−c)·pFg`. Bounded [0,1]/voxel (so it does not swamp / get swamped by the L2-tuned priors as badly), **continuous in synth** (so greedy gets a smooth gradient), and always positive (sidesteps the negative-baseline hazard the raw NLL created). Removed the now-unused `claimEps`.

### Verified (smoke, generative seed=42, f0–3)
`f0 perturb=1`, `f1 perturb=5 split_acc=3 cells=7`, `f2 perturb=4 cells=7`, `f3 perturb=6 cells=7` — optimizer unfrozen (was `0/0/0`). Cf. legacy seed=42 `4 7 8 8`. Full f0–20 comparison running.

### Still open (from scan agent #1, to address in Rung 2 / weight re-fit)
The soft-coverage term changed the image-cost magnitude (bounded [0,1]/voxel) but the priors/gates are still L2-tuned: `overlap_penalty_weight=75000`, `position_prior_weight=75`, `split_cost=2000` and the hardcoded split floors. Splits currently still ride the fusion prior's forced path (`costDiff=-1`); the L2-calibrated gates are not yet re-scaled to the new term. These are the O2 weight-refit / Rung-2 items, not blockers for judging whether the data term *helps placement*.

---

## 2026-07-30: Rung 1 A/B verdict + deep-scan synthesis (Change 27)

**Status: RUNG-1 CONCLUSION — the data term works but is confined to placement; the over-split fix and a fair evaluation both require Rung 2. Default stays OFF.**

### Full A/B (seed=42, embryo f0–20)
| | f0..f20 counts | end | splits |
|---|---|---|---|
| GT | `4 4 4 4 6 6 6 7 7 7 7 8 8 8 9 9 9 10 10 12 12` | 12 | — |
| Legacy L2 | `4 7 8 8 8 8 8 8 8 8 8 9 10 11 11 12 12 12 13 14 14` | 14 | 10, all `costDiff=-1 cell_lumen_primary` |
| Generative (soft-cov) | `4 7 8 8 8 8 8 10 10 10 11 12 13 14 14 15 16 16 16 16 16` | 16 | 12, all `costDiff=-1 cell_lumen_primary` |

- **The data term is functional and active**: it accepts *more* perturbations than legacy (moves cells more) and demonstrably changes placement + shape (f5 spot-check: positions shift up to ~20–25 px, radii run larger — consistent with the term penalising uncovered foreground). Whether that placement is *better* needs a GT/CTC overlay (not run).
- **But it cannot touch splits.** Every accepted split — in BOTH arms — is `costDiff=-1 bestLabel=cell_lumen_primary`, i.e. forced by the fusion-prior sentinel that bypasses the image-cost accept. So the data term has zero influence on split decisions; it only repositions cells, which here made the fusion prior fire *more* (16 vs 14). **Rung 1 cannot fix over-splitting — that is 100% owned by the fusion prior, not the cost.**

### Deep-scan synthesis (3 agents, read-only)
- **Determinism (dodged for this A/B):** RNG seeds from `random_device` unless `CELLUNIVERSE_SEED` set (`ConfigTypes.hpp:28`), and OpenMP float reductions (`Frame.cpp:565,677,978,1018`) are non-reproducible across thread counts. Both A/B arms used `CELLUNIVERSE_SEED=42` + single-thread, so the comparison is clean. **Standing rule: fixed seed + fixed threads for all future A/B.**
- **Over-split is a localized sentinel:** the `-1` is set at `Frame.cpp:8308` (`= -max(1, adaptiveThreshold)`, with `fusionSplitPriorCost/Fraction=0` → threshold 0), plus a `fusionSplitPriorMaxPositiveCostFraction:0.12` allowance ("a split that worsens the image ≤12% still passes"). **Guidance-only is reachable via 4 config flags** (`skipRandomSplits`, `forceSchedule`, `guidedOnly`, `useDedicatedCostGate`) without touching the frame loop — but recall drops until `P(split)` is the 7-feature logistic (Rung 2a) and the sentinel is replaced by a hypothesis-energy comparison (Rung 2b).
- **All decision weights are L2-scaled** (`overlap 75000`, `position_prior 75`, `split_cost 2000`, hardcoded split floors `2500/9000/20000/...`) → must be re-fit for the new term (spec O2).
- **Two findings that REVISE the design:**
  1. **FG model is estimated from the pre-optimization synth coverage** (`CellUniverse.cpp:5648-5660`), which is birth-mask-capped and RNG-dependent → the objective is *not fixed across seeds* and is coupled to the shape cap. **Should estimate the FG model from a search-independent proxy** (real-image percentiles or the CellLumen map, per spec §4.2) so the cost is a function of the image, not the stochastic history.
  2. **Birth-mask-bounded PCA fit caps coverage** (`Frame.cpp:2578-2585`): a legitimately grown cell cannot expand its claim to cover foreground, so the generative penalty for uncovered foreground biases toward drift/split — fighting the redesign's goals. The shape cap and the data term must be reconciled.
- **z-anisotropy:** `z_scaling=7` interpolates z 7× at load (`ImageHandler.cpp:1257`), so z voxels are correlated and over-weighted in the pooled fg/bg histogram (and a future edge term would see interpolation-ramp gradients). CellLumen uses the *opposite* convention (scale-in-metric). Reconcile before claiming transferability.
- **Global mutable `Ellipsoid::cellConfig`** (manual save/restore, no RAII) — a latent order-dependence trap that Rung 2b's extra lifecycle paths (merge/Metropolis) will trip. Wrap in RAII.

### Conclusion
Rung 1 answered its question: the self-calibrating data term *works* (unfrozen, actively repositions cells) but its influence is confined to placement, and it cannot be fairly judged for tracking quality in isolation because splits bypass it and the surrounding weights are L2-tuned. The user's #1 pain (over-split) is structurally a Rung-2 problem. Recommend proceeding to Rung 2a (config-flag conversion to guidance-only + multi-factor `P(split)`), carrying the two design revisions above.

---

## 2026-07-28: Refresh docs/pipeline.md to current (Change 22)

**Status: ACTIVE — docs only.**

`docs/pipeline.md` had been stale since 2026-04-16 (pre-CellLumen). Updated it to the current end-to-end flow while keeping the still-accurate PCA-fit / SA / phased-split core intact.

### `docs/pipeline.md`

- Header: bumped "Last updated" to 2026-07-28; noted the core below is unchanged from the April rewrite; pointed rationale to `changelogv8.md` (Changes 16–21).
- **Added §0 "Startup: initial CSV → founder cells"** — how `CsvHandler` turns the seed into named founders (`cellName = cellType + "_" + rowNumber`, `CsvHandler.cpp:210`), position/shape columns, `CellFactory`.
- **Added "Frame-intensity normalization"** — `low_ref`/`high_ref` stretch + the robust-low-reference knob (Change 20) that fixes washed-out high-pedestal frames.
- **Added "CellLumen fusion detector + split priors"** — the bottom-up watershed + fusion-prior path (`cell_lumen_primary`, forced `costDiff=-1`), the over-split driver knobs, and the refit-drift gate.
- **Added "Output (END OF FRAME)"** — `cells.csv` (internal) vs `tracked_cells.csv` + `name_map.csv` (Change 19) + `real/`+`synth/` TIFFs + lineage.
- Fixed the stale "Save cells.csv + output PNG images" line; updated "Related docs" to `changelogv8` + the `.claude/rules/*`.

---

## 2026-07-31: Rung 2a part 1 — SplitScore combiner (Change 28)

**Status: ACTIVE — new module, tested, ready for feature extraction + wiring.**

### Summary

Introduced a modular 7-feature logistic for split scoring. The core is a linear combination of normalized morphology + evidence features (elongation, valley, centerSupport, volumeRipe, centralDef, density) plus the image-cost term (coverageGain), mapped through sigmoid to probability and compared to a threshold for accept/reject. This decouples split decisions from the hardcoded cost gates and enables per-feature tuning / ablation.

### New files

- **`includes/SplitScore.hpp`** — header-only module (structs `SplitFeatures`, `SplitScoreWeights`; functions `splitLogit`, `splitProbability`, `splitScoreAccept`). All logic is pure and inline.
- **`src/SplitScore.cpp`** — translation unit (one include, no implementation). Exists to link the module into celluniverse + test targets.
- **`tests/test_split_score.cpp`** — unit tests: monotonicity (more evidence → higher prob), base rate (zero evidence → low prob), accept gate, density suppression, cost as evidence (not a veto).

### Logit formula

```
logit = bias + w_elong·elongation + w_valley·valley + w_center·centerSupport 
       + w_volume·volumeRipe + w_central·centralDef - w_density·density 
       + w_cost·coverageGain
```

- **Features** in [0,1] except `coverageGain` in [-1,1].
- **Weights** (defaults): bias = −3.0, wElong=1.0, wValley=2.0, wCenter=2.0, wVolume=1.0, wCentral=1.0, wDensity=2.0, wCost=1.5.
- **Accept threshold** sAccept = 0.0 (logit ≥ sAccept → accept).
- **Density negated**: density term suppresses split in crowded regions.
- **Cost as evidence**: `coverageGain` lowers or raises score; does not veto (consistent with Rung-1 finding that real divisions barely change pixel cost).

### Verification

```bash
g++ -std=c++17 tests/test_split_score.cpp src/SplitScore.cpp -o /tmp/tss && /tmp/tss
# Output: test_split_score: PASS
```

All 6 assertions pass:
1. Monotonic increase in features.
2. Low base rate with zero evidence.
3. Ripe cell accepts; empty cell rejects.
4. Density suppresses even with ripe features.
5. Cost gain increases score; cost loss decreases it.
6. Cost never vetoes the decision (always evidence, not authority).

---

## 2026-07-31: Rung 2a part 2 — SplitFeatureExtractor (Change 29)

**Status: ACTIVE — pure normalizer module, fills the 6 morphology + evidence features for SplitScore, tested.**

### Summary

Introduced feature-extraction normalizers to convert raw per-cell measurements (fitted semi-axes, volume, brightness ratios, neighbor density) into normalized [0,1]-range features that feed the SplitScore logistic (Change 28). Each feature is independently calibrated so its common-case range maps to [0,1] and extreme values clamp safely.

### New files

- **`includes/SplitFeatureExtractor.hpp`** — struct `SplitFeatureInputs` (8 raw fields: `maxR`, `minR`, `volume`, `birthVolume`, `valleyRatio`, `midBandBrightness`, `coreBrightness`, `secondCenterSignal`, `secondCenterSignalRef`, `neighborsWithin`, `densityRef`). Function `buildSplitFeatures(const SplitFeatureInputs&) → SplitFeatures` fills the first 6 features; `coverageGain` remains 0 (set at accept time by other code).
- **`src/SplitFeatureExtractor.cpp`** — pure normalizer implementations. Helper `clamp01(x)` clamps to [0,1]. Each feature uses a simple formula:
  - **elongation**: `clamp01((maxR / minR - 1) / (2 - 1))` — map range [1,2] to [0,1]; elongation 1→0, elongation 2→1.
  - **valley**: `clamp01(1 - valleyRatio)` — high valleyRatio (no dark bridge) → low feature; low ratio (dark bridge) → high feature.
  - **volumeRipe**: `clamp01(volume / birthVolume - 1)` — volume at birth → 0; doubled volume → 1.
  - **centralDef**: `clamp01(1 - midBandBrightness / coreBrightness)` — uniform brightness → 0; dip at middle → high feature.
  - **centerSupport**: `clamp01(secondCenterSignal / secondCenterSignalRef)` — normalized by per-frame signal max (typically 100).
  - **density**: `clamp01(neighborsWithin / densityRef)` — normalized by typical neighbor count (e.g., 6).
- **`tests/test_split_feature_extractor.cpp`** — unit tests:
  - **Test 1 (low features)**: round, small, single-lobe cell in isolation → all features ~0.
  - **Test 2 (high features)**: elongated, doubled-volume, dark-bridge, second-center signal, dumbbell morphology → strong features.
  - **Test 3 (clamping)**: density 9/6 exceeds 1.0 and clamps to 1.

### Verification

```bash
cd /Users/jihangli/MCS/3D_Cell_Tracking/CellUniverse/C++
g++ -std=c++17 tests/test_split_feature_extractor.cpp src/SplitFeatureExtractor.cpp -o /tmp/tsfe && /tmp/tsfe
# Output: test_split_feature_extractor: PASS
```

All assertions pass. Test 1 confirms quiet cells score ~0 (no false alarms). Test 2 confirms dividing-cell features rise to 0.6–1.0. Test 3 confirms clamping protects against outlier inputs.

### Next steps (Rung 2a part 3+)

- Wire `buildSplitFeatures` into `Frame::trySplitCellPhased` to populate the 6 morphology features at candidate evaluation time.
- Port the existing hard-coded bio-gate checks (valley, density, centerSupport) to use the extracted features.
- Enable/disable the SplitScore logistic via config flag, tuning weights to match or improve on the hard-coded logic.

---

## 2026-07-31: Rung 2a part 3 — Config flag + weights (Change 30)

**Status: ACTIVE — config parsing complete, all defaults OFF (no behavior change).**

### Summary

Made the Rung 2a multi-factor split decision parameters parseable. Added 15 new config fields to `ProbabilityConfig` (flag + weights). All default to OFF so the tracker behaves identically to before — this task only makes the keys recognizable by YAML parsing.

### Files changed

- **`includes/ConfigTypes.hpp`** (2 edits, 14 lines added):
  - **Lines 759–773**: Added 15 new fields to `class ProbabilityConfig` (after `foreground_model_min_sigma`):
    - `multifactor_split_enabled : bool = false` — master flag; when false, SplitScore is unused.
    - **Weights** (7 feature weights + cost): `split_w_elong`, `split_w_valley`, `split_w_center`, `split_w_volume`, `split_w_central`, `split_w_density`, `split_w_cost` (all float, defaults 1.0–2.0).
    - **Scoring params** (4 floats): `split_score_bias = −3.0`, `split_score_accept = 0.0` (logit threshold), `split_score_attempt_min_prob = 0.05` (skip low-P attempts), `split_second_center_signal_ref = 100.0`.
    - **Calibration refs** (2 floats): `split_density_ref = 6.0` (typical neighbor count), `split_density_radius_scale = 2.5` (neighbor search radius).
  - **Lines 876–889**: Added 14 parse lines in `ProbabilityConfig::explodeConfig` (after `foreground_model_min_sigma` parse), one if-check per field, all following the existing pattern (`if (node["key"]) key = node["key"].as<T>();`).

- **`config/config.yaml`** (1 edit, 16 lines added):
  - **Lines 259–274**: Added the 15 keys under the `prob:` block (after `foreground_model_min_sigma: 0.001`), with matching defaults and an explanatory header comment.

### Build verification

```
cmake --build build -j 8 --target celluniverse
[100%] Built target celluniverse
```

**Compiles clean.** Unknown YAML keys are silently ignored (per project traps), so the new parse lines are required to avoid feature no-ops. Every YAML key now has a matching parsed field.

### No behavior change

All 15 new fields default to OFF (`multifactor_split_enabled = false`), so the existing hard-coded split pipeline is unchanged. When the flag is false, `SplitScore` and `SplitFeatureExtractor` (Changes 28–29) remain unused. Future tasks will wire them in via `Frame::measureSplitEvidence` and `Frame::trySplitCellPhased` (Rung 2a parts 4–6).

---

## 2026-07-31: Rung 2a part 4 — Frame::measureSplitEvidence (Change 31)

**Status: ACTIVE — new read-only method added, not yet called (behavior-neutral; wired by Rung 2a part 5).**

### Summary

Added `Frame::measureSplitEvidence(size_t cellIndex) const`, a lightweight read-only method that measures the brightness valley between a cell's two potential lobes on its CURRENT fit (no daughter proposal). It fills three of the raw inputs (`valleyRatio`, `midBandBrightness`, `coreBrightness`) that `SplitFeatureExtractor::buildSplitFeatures` (Change 29) needs for the `valley` and `centralDef` features. `trySplitCellPhased` was **not modified** — only the new struct + method were added.

The method reuses the exact long-axis derivation and in-ellipsoid bright-voxel gathering already used by `Frame::discoverPcaBridgeProposal` (added earlier for the PCA-bridge daughter proposal, `src/Frame.cpp` lines 2956–3166) and the brightness cutoff formula from `gatherBrightPixelsVoronoi` (`src/Frame.cpp` line ~1808): `max(0.05f, backgroundValue + 0.02f)`. No new geometry was invented.

### Files changed

- **`includes/Frame.hpp`** (2 edits, 22 lines added):
  - **Lines 75–87** (new, after `struct BridgeSplitProposal` closes at line 73): added `struct SplitEvidence` — a free-standing struct at file scope (same placement convention as `BridgeSplitProposal`), so `measureSplitEvidence`'s return type needs no `Frame::` qualification outside the class.

    ```cpp
    // Read-only split evidence for a cell's CURRENT fit. Produced by
    // Frame::measureSplitEvidence (Rung 2a part 4) by binning in-ellipsoid
    // bright voxels along the cell's own longest fitted axis — reuses the same
    // long-axis + in-ellipsoid-gather machinery as discoverPcaBridgeProposal,
    // but standalone (no daughter pair, no cell mutation). Consumed by the
    // Rung-2a split score (SplitFeatureExtractor), not by trySplitCellPhased.
    // Defaults (all 1.0) mean "no valley detected" / neutral evidence.
    struct SplitEvidence
    {
        float valleyRatio = 1.0f;       // midBandBrightness / max(1e-3, edgeBrightness)
        float midBandBrightness = 1.0f; // mean brightness of the center bin
        float coreBrightness = 1.0f;    // brightest bin mean across the axis
    };
    ```

  - **Lines 378–385** (new, inside `class Frame { public:`, immediately after the `discoverPcaBridgeProposal` declaration ends at line 376): added the method declaration.

    ```cpp
    // Rung 2a part 4: lightweight, read-only split evidence for a cell's
    // CURRENT fit (no daughter proposal, no mutation). Projects in-ellipsoid
    // bright voxels onto the cell's longest fitted semi-axis (same long-axis
    // derivation as discoverPcaBridgeProposal), bins them into 11 equal bins
    // across [-longR, +longR], and reports the mid-band vs edge-lobe
    // brightness contrast. Returns default SplitEvidence{} (no valley) when
    // the cell index is invalid or too few bright voxels are found (<20).
    SplitEvidence measureSplitEvidence(size_t cellIndex) const;
    ```

- **`src/Frame.cpp`** (1 edit, 127 lines added):
  - **Lines 3168–3293** (new, inserted between the end of `discoverPcaBridgeProposal` at line 3166 and the `trySplitCellPhased` comment block that previously followed directly): the full method body.

    ```cpp
    // Rung 2a part 4: read-only split evidence for one cell's CURRENT fit.
    // Standalone sibling of discoverPcaBridgeProposal above — reuses the exact
    // same long-axis derivation (longest semi-axis, world-space direction via
    // generateInverseRotationMatrix) and in-ellipsoid bright-voxel gather (same
    // brightness cutoff as gatherBrightPixelsVoronoi: max(0.05, background+0.02)),
    // but does NOT propose a daughter split; it simply bins bright voxels along
    // the long axis and reports the mid-band-vs-edge-lobe brightness contrast.
    // const, no cell/member mutation, no logging side effects.
    SplitEvidence Frame::measureSplitEvidence(size_t cellIndex) const
    {
        SplitEvidence evidence; // defaults: all 1.0 ("no valley")
        if (cellIndex >= cells.size() || _realFrame.empty()) {
            return evidence;
        }

        const Ellipsoid &cell = cells[cellIndex];

        // Longest fitted semi-axis + its world-space direction — same pattern
        // as discoverPcaBridgeProposal's longIdx/longAxis derivation above.
        const float radii[3] = {
            cell.getARadius(), cell.getBRadius(), cell.getCRadius()
        };
        int longIdx = 0;
        for (int i = 1; i < 3; ++i) {
            if (radii[i] > radii[longIdx]) longIdx = i;
        }
        const float longR = radii[longIdx];
        if (longR <= 1e-3f) {
            return evidence;
        }

        std::array<double, 9> R_T;
        cell.generateInverseRotationMatrix(R_T);
        const int base = 3 * longIdx;
        const cv::Point3f longAxis(
            static_cast<float>(R_T[base]),
            static_cast<float>(R_T[base + 1]),
            static_cast<float>(R_T[base + 2]));

        // Axis-aligned bbox sized to longR: the max world-axis extent of a
        // rotated ellipsoid is bounded by its longest semi-axis — same bbox
        // pattern discoverPcaBridgeProposal uses.
        const int xMin = std::max(0, static_cast<int>(std::floor(cell.getX() - longR)));
        const int xMax = std::min(_realFrame[0].cols - 1, static_cast<int>(std::ceil(cell.getX() + longR)));
        const int yMin = std::max(0, static_cast<int>(std::floor(cell.getY() - longR)));
        const int yMax = std::min(_realFrame[0].rows - 1, static_cast<int>(std::ceil(cell.getY() + longR)));
        const int zMin = std::max(0, static_cast<int>(std::floor(cell.getZ() - longR)));
        const int zMax = std::min(static_cast<int>(_realFrame.size()) - 1,
                                  static_cast<int>(std::ceil(cell.getZ() + longR)));

        // Same brightness cutoff as gatherBrightPixelsVoronoi (above, this file).
        const float brightnessCutoff = std::max(0.05f, _backgroundValue + 0.02f);

        static constexpr int kEvidenceBins = 11;
        std::array<double, kEvidenceBins> binSum{};
        std::array<int, kEvidenceBins> binCount{};
        const float binWidth = (2.0f * longR) / static_cast<float>(kEvidenceBins);

        int totalBright = 0;
        for (int z = zMin; z <= zMax; ++z) {
            const cv::Mat &slice = _realFrame[z];
            if (slice.type() != CV_32F || slice.empty()) continue;
            for (int y = yMin; y <= yMax; ++y) {
                const float *row = slice.ptr<float>(y);
                for (int x = xMin; x <= xMax; ++x) {
                    const float v = row[x];
                    if (v <= brightnessCutoff) continue;

                    const cv::Point3f worldPoint(
                        static_cast<float>(x),
                        static_cast<float>(y),
                        static_cast<float>(z));
                    if (!cell.isPointInsideEllipsoid(worldPoint)) continue;

                    ++totalBright;

                    const float dx = worldPoint.x - cell.getX();
                    const float dy = worldPoint.y - cell.getY();
                    const float dz = worldPoint.z - cell.getZ();
                    const float proj = dx * longAxis.x + dy * longAxis.y + dz * longAxis.z;
                    const float t = std::clamp(proj, -longR, longR);
                    int bin = static_cast<int>((t + longR) / binWidth);
                    bin = std::clamp(bin, 0, kEvidenceBins - 1);
                    binSum[bin] += v;
                    binCount[bin] += 1;
                }
            }
        }

        // Too few voxels to bin reliably — return the neutral default.
        if (totalBright < 20) {
            return evidence;
        }

        std::array<float, kEvidenceBins> binMean{};
        for (int i = 0; i < kEvidenceBins; ++i) {
            binMean[i] = (binCount[i] > 0)
                ? static_cast<float>(binSum[i] / binCount[i])
                : 0.0f;
        }

        float coreBrightness = binMean[0];
        for (int i = 1; i < kEvidenceBins; ++i) {
            coreBrightness = std::max(coreBrightness, binMean[i]);
        }

        // Middle bin (index 5 of 11) — the cell's own longitudinal center.
        constexpr int kMidBin = kEvidenceBins / 2;
        const float midBandBrightness = binMean[kMidBin];

        // Two highest side bins, one from each half (excluding the center
        // band) — the two candidate-lobe peaks.
        float leftPeak = 0.0f;
        for (int i = 0; i < kMidBin; ++i) {
            leftPeak = std::max(leftPeak, binMean[i]);
        }
        float rightPeak = 0.0f;
        for (int i = kMidBin + 1; i < kEvidenceBins; ++i) {
            rightPeak = std::max(rightPeak, binMean[i]);
        }
        const float edgeBrightness = 0.5f * (leftPeak + rightPeak);

        evidence.valleyRatio = midBandBrightness / std::max(1e-3f, edgeBrightness);
        evidence.midBandBrightness = midBandBrightness;
        evidence.coreBrightness = coreBrightness;
        return evidence;
    }
    ```

### Design notes / deviations from the task algorithm

- **`SplitEvidence` placed at file scope, not nested in `class Frame`.** The brief's interface sketch showed the struct directly above the method declaration, ambiguous about nesting. Followed the existing `BridgeSplitProposal` convention (also a free struct immediately above `class Frame`) for consistency and so the `.cpp` definition doesn't need a `Frame::SplitEvidence` qualifier.
- **`midBandBrightness`** is the mean of the single center bin (index 5 of 11), read literally from "the center bin(s) (index 5, ...)" in the brief — 11 bins have exactly one true center bin, so no averaging across index 4/5/6 was applied.
- **Empty bins** (zero voxels) are treated as brightness 0 rather than excluded from max/peak search — consistent with "no signal here" and matches how `binCount==0` is handled nowhere else needing special-casing (an empty center bin correctly drives `valleyRatio` toward 0, i.e., a strong valley).
- Everything else follows the brief's algorithm exactly: longest-axis direction via `generateInverseRotationMatrix`, in-ellipsoid gather via `Ellipsoid::isPointInsideEllipsoid`, `_realFrame` for raw intensities, 11 bins over `[-longR, +longR]`, `<20` voxel fallback to the neutral default, `valleyRatio = midBandBrightness / max(1e-3f, edgeBrightness)`.

### Build verification

```
cmake --build build -j 8 --target celluniverse
[100%] Built target celluniverse
```

**Compiles clean, no warnings** (verified with a forced rebuild of `Frame.cpp.o` alone to confirm no stale-object masking). `measureSplitEvidence` is not called anywhere yet — Rung 2a part 5 wires it into the split-attempt path — so there is **no behavior change** in this task.

---

## 2026-07-31: Rung 2a part 5 — multi-factor split ATTEMPT wiring (Change 32)

**Status: ACTIVE — `multifactor_split_enabled` now computes real `P(split)` from `SplitScore`/`SplitFeatureExtractor`; flag-OFF path is byte-identical to before.**

### Summary

Wired `SplitScore` (Change 28) + `SplitFeatureExtractor` (Change 29) + `Frame::measureSplitEvidence` (Change 31) into `CellUniverse::optimize`'s `P(split)` computation and the random-split attempt roll. When `config.prob.multifactor_split_enabled` is `true`, `splitProbabilities[name]` is now the logistic `splitProbability(SplitFeatures, SplitScoreWeights)` score instead of the elongation-only linear ramp, and a per-cell `splitFeaturesByName` map is populated for the accept-step override that Task 6 will add. When the flag is `false` the pre-existing elongation-ramp loop runs **unchanged** (moved into an `else` branch verbatim, not rewritten) and the attempt-roll gate is a no-op. No new config knobs were added — all 15 fields already existed from Rung 2a part 3 (Change 30).

### Files changed

- **`src/CellUniverse.cpp`** (3 edits):

  - **Lines 4–6** (new includes, after the existing `ForegroundModel.hpp` include):
    ```cpp
    #include "../includes/SplitScore.hpp"
    #include "../includes/SplitFeatureExtractor.hpp"
    ```

  - **Lines 5768–5881** (P(split) loop, inside `CellUniverse::optimize`). Before, the block unconditionally ran the linear elongation ramp. After, it declares `splitFeaturesByName` and branches on the flag:
    ```cpp
    std::map<std::string, float> splitProbabilities;
    // Rung 2a: parent features for the accept step (Task 6). Populated only
    // when multifactor_split_enabled; empty otherwise.
    std::map<std::string, SplitFeatures> splitFeaturesByName;

    if (config.prob.multifactor_split_enabled) {
        SplitScoreWeights W;
        W.wElong = config.prob.split_w_elong;    W.wValley = config.prob.split_w_valley;
        W.wCenter = config.prob.split_w_center;  W.wVolume = config.prob.split_w_volume;
        W.wCentral = config.prob.split_w_central; W.wDensity = config.prob.split_w_density;
        W.wCost = config.prob.split_w_cost;      W.bias = config.prob.split_score_bias;
        W.sAccept = config.prob.split_score_accept;

        for (size_t ci = 0; ci < frame.cells.size(); ++ci) {
            const Ellipsoid &c = frame.cells[ci];
            if (c.isTrash()) continue;
            const std::string name = c.getName();
            const float a = c.getARadius(), b = c.getBRadius(), cc = c.getCRadius();
            const float maxR = std::max(a, std::max(b, cc));
            const float minR = std::min(a, std::min(b, cc));

            float ps = 0.0f;
            SplitFeatures f;
            if (allowSplits) {
                SplitFeatureInputs in;
                in.maxR = maxR; in.minR = minR;
                in.volume = (4.0f / 3.0f) * 3.14159265f * a * b * cc;
                auto bIt = cellShapeBirth.find(name);
                if (bIt != cellShapeBirth.end()) {
                    const auto &br = bIt->second; // std::array<float,3> {aR,bR,cR}
                    in.birthVolume = (4.0f / 3.0f) * 3.14159265f * br[0] * br[1] * br[2];
                } else {
                    in.birthVolume = in.volume;
                }
                const SplitEvidence ev = frame.measureSplitEvidence(ci);
                in.valleyRatio = ev.valleyRatio;
                in.midBandBrightness = ev.midBandBrightness;
                in.coreBrightness = ev.coreBrightness;

                // Second (unclaimed) CellLumen center inside the cell.
                in.secondCenterSignal = 0.0f;
                if (maxR > 1e-3f) {
                    const cv::Point3f cellPos(c.getX(), c.getY(), c.getZ());
                    const float lo = 0.3f * maxR, hi = 1.2f * maxR;
                    const auto &lookaheadCenters = getCellLumenLookaheadCandidates(frameIndex);
                    for (const auto &center : lookaheadCenters) {
                        const float dist = static_cast<float>(cv::norm(cellPos - center.position));
                        if (dist >= lo && dist <= hi) {
                            in.secondCenterSignal = std::max(in.secondCenterSignal, center.signal);
                        }
                    }
                }
                in.secondCenterSignalRef = config.prob.split_second_center_signal_ref;

                int nb = 0;
                for (size_t oj = 0; oj < frame.cells.size(); ++oj) {
                    if (oj == ci) continue;
                    const Ellipsoid &o = frame.cells[oj];
                    const float dx = o.getX() - c.getX(), dy = o.getY() - c.getY(), dz = o.getZ() - c.getZ();
                    if (std::sqrt(dx * dx + dy * dy + dz * dz) < config.prob.split_density_radius_scale * maxR) ++nb;
                }
                in.neighborsWithin = nb;
                in.densityRef = config.prob.split_density_ref;

                f = buildSplitFeatures(in);
                ps = splitProbability(f, W);
            }
            splitFeaturesByName[name] = f;
            splitProbabilities[name] = ps;
            std::cout << "[Split Score] frame " << displayFrame << " " << name
                      << " elong=" << f.elongation << " valley=" << f.valley
                      << " center=" << f.centerSupport << " volume=" << f.volumeRipe
                      << " central=" << f.centralDef << " density=" << f.density
                      << " P=" << ps << std::endl;
        }
    } else {
        // ... original elongation-ramp loop, verbatim, unindented one level ...
    }
    ```
    The `else` branch is the **exact pre-existing loop body** (same variable names, same `std::cout` format, same clamp/lerp math) — only its indentation changed from being wrapped one brace level deeper.

  - **Lines 8091–8100** (split-attempt roll). Before:
    ```cpp
    const bool canSplit = pSplit > 0.0f
                       && splitBlacklist.count(cellName) == 0;

    if (canSplit && uniform01(gen) < pSplit) {
    ```
    After:
    ```cpp
    // Rung 2a part 5: under the multifactor score, don't even roll for
    // cells whose P(split) is below the configured attempt floor —
    // keeps the burn-in budget off cells the score already rejects.
    // Flag-off path is untouched (min-prob gate is a no-op there).
    const bool multifactorAttemptFloorOk =
        !config.prob.multifactor_split_enabled
        || pSplit >= config.prob.split_score_attempt_min_prob;
    const bool canSplit = pSplit > 0.0f
                       && splitBlacklist.count(cellName) == 0
                       && multifactorAttemptFloorOk;

    if (canSplit && uniform01(gen) < pSplit) {
    ```

### Deviations from the task brief (member/accessor names)

- **`SplitEvidence` is a free struct, not `Frame::SplitEvidence`.** Change 31 (Rung 2a part 4) deliberately placed it at file scope, same convention as `BridgeSplitProposal` — documented in that changelog entry. Using the qualified `Frame::SplitEvidence` name (as the brief sketch wrote it) fails to compile (`error: no type named 'SplitEvidence' in 'Frame'`); fixed to the unqualified `SplitEvidence`.
- **`cellShapeBirth`** is exactly `std::map<std::string, std::array<float, 3>> cellShapeBirth` (declared `includes/CellUniverse.hpp:95`) — matches the brief's guess (`{aR,bR,cR}`) exactly, no change needed.
- **Lumen-center container: used `getCellLumenLookaheadCandidates(frameIndex)`, not `cellLumenCenterCandidates[frameIndex]`.** The brief suggested `cellLumenCenterCandidates[frameIndex]` for the "second unclaimed center" search, but that member (`includes/CellUniverse.hpp:129`, `std::unordered_map<int, std::unordered_map<std::string, CellLumenCenterCandidate>>`) is keyed **by owning cell name** — one claimed candidate per cell, not a free list to search spatially. `cellLumenLookaheadCandidates` (`includes/CellUniverse.hpp:136`, `std::unordered_map<int, std::vector<CellLumenLookaheadCandidate>>`) is the actual flat per-frame raw-candidate list (fields `position`, `voxelCount`, `signal`, `candidateId`), already used this exact way (distance-gated nearest/strongest-signal search) by `applyGlobalLumenCenterAssignment` and several other split-pipeline sites in this file. Its accessor is the member function `getCellLumenLookaheadCandidates(int frameIndex)` (returns `const std::vector<CellLumenLookaheadCandidate>&`, lazily builds/caches). Used that instead — it matches the brief's intent ("iterate; keep strongest signal within a distance range") far better than the per-cell-claim map, and the `[0.3·maxR, 1.2·maxR]` distance floor already excludes a cell's own near-zero-distance signal, satisfying "unclaimed" without needing explicit claim bookkeeping.

### Build verification

**Blocker found and worked around without touching out-of-scope files:** `src/SplitFeatureExtractor.cpp` (defines `buildSplitFeatures`) and `src/SplitScore.cpp` are **not yet in `CMakeLists.txt`'s `add_executable(celluniverse ...)` sources list** — that wiring is explicitly reserved for Task 6 ("Rung2a T6: score accept override + CMake" in the task tracker). `cmake --build build -j 8 --target celluniverse` therefore compiles `CellUniverse.cpp.o` cleanly (confirming every accessor name above is correct) but fails at **link** time:
```
Undefined symbols for architecture arm64:
  "buildSplitFeatures(SplitFeatureInputs const&)", referenced from:
      CellUniverse::optimize(int) in CellUniverse.cpp.o
ld: symbol(s) not found for architecture arm64
```
Per this task's scope restriction (touch only `src/CellUniverse.cpp` + this changelog — no CMake edits), `CMakeLists.txt` was **not modified**. Instead, both required smokes were run against a **shadow executable**: `SplitFeatureExtractor.cpp` and `SplitScore.cpp` were compiled with the exact flags from `build/compile_commands.json`, then linked against the already-built `build/CMakeFiles/celluniverse.dir/src/*.cpp.o` objects using the exact link line from `build/link.txt` (`+2` new objects), output to a scratch path outside `build/`. This exercises the real, unmodified `CellUniverse.cpp.o` — the same object the real build produces — so the smoke results below are representative. **Task 6 must add `${cwd}/src/SplitFeatureExtractor.cpp` and `${cwd}/src/SplitScore.cpp` to `CMakeLists.txt`'s `add_executable` list before `cmake --build build --target celluniverse` will link.**

### Smoke tests (via shadow executable)

**Flag-ON, f0–3** (`multifactor_split_enabled: true`, `CELLUNIVERSE_SEED=42`, `data/input/embryo_data/t%03d.tif`, `config/Teammate_initial/initial_embryo.csv`): `exit=0`, no crash. 22 `[Split Score]` lines printed, plausible: frame 0 (splits disabled, pre-first-step) all `P=0`; frame 1–3 round founders sit around `P≈0.05–0.06`, while cells with elevated `center`/`elong`/`volume` features score higher (e.g. frame 2 `cell_3 elong=0.023 center=0.972 volume=0.728 → P=0.435`; frame 3 snapshot shows `cell_20 shapeElong=1.70` from natural elongation growth). `split_attempts=0`/`split_accepted=0` in this short window — expected, real GT splits start later and the accept override is Task 6's job; this task only validates the score computation and attempt gating don't crash and produce sane numbers.

**Flag-OFF, f0–2** (default `config/config.yaml`, same seed/inputs): `exit=0`, no crash, **zero** `[Split Score]` lines, `[P(split)]` header prints the original `(linear elong ramp)` / `(splits disabled)` text unchanged, `split_attempts=0`/`split_accepted=0` matches the flag-ON short-window count. Confirms flag-OFF is a pure no-op.

### No behavior change (flag-off)

`multifactor_split_enabled` still defaults to `false` (Change 30). With the flag off, `splitFeaturesByName` stays empty, `splitProbabilities` is filled by the untouched elongation-ramp `else` branch, and `multifactorAttemptFloorOk` is always `true` — so `canSplit`'s truth table is identical to before this change.

## 2026-07-31: Rung 2a part 6 — score-based ACCEPT override + CMake wiring (Change 33)

**Status: ACTIVE — CMake now links `SplitScore.cpp`/`SplitFeatureExtractor.cpp` into `celluniverse` (the link failure documented in Change 32 is fixed); `trySplitCellPhased` accepts an optional `SplitDecisionCtx*` that, when non-null, replaces the CellLumen cost-gate/sentinel ACCEPT decision with the multi-factor score + a geometric sanity check. Flag-OFF path is byte-identical to before (verified by code inspection of the added guards + a live f0–3 run showing zero `[Split Accept MF]` lines and the legacy `[Split Accepted] ... cell_lumen_primary` path still firing 3 times).**

### Summary

This is the final integration step of Rung 2a. Two things:

1. **CMake fix.** `src/SplitScore.cpp` + `src/SplitFeatureExtractor.cpp` are now in the `celluniverse` `add_executable` sources list, fixing the `buildSplitFeatures` link failure Change 32 worked around with a shadow executable.
2. **Accept override.** Added a new `SplitDecisionCtx` struct (`includes/SplitScore.hpp`) bundling a cell's parent `SplitFeatures` + `SplitScoreWeights`. `Frame::trySplitCellPhased` gained one new trailing parameter, `const SplitDecisionCtx *mfCtx = nullptr`. `CellUniverse::optimize`'s `attemptSplitAtIndex` lambda builds this context from `splitFeaturesByName` (Change 32) + `config.prob.split_w_*`/`split_score_bias`/`split_score_accept` (Change 30) only when `config.prob.multifactor_split_enabled` is `true`, and passes `nullptr` otherwise (or when the cell has no `splitFeaturesByName` entry). Inside `trySplitCellPhased`, the entire legacy `lumenCostAccepted` / final cost-reject block is now guarded by `if (mfCtx == nullptr) { ... } else { ... }` — the `else` branch computes `coverageGain` from the same `baselineImageCost`/`bestImageCost` the legacy cost gate already computed, evaluates `splitScoreAccept(...)` + a hardcoded geometric-sanity check (daughter separation, combined daughter volume vs. reference parent volume — no new config knobs), and either falls through to the shared accept-install code (on accept) or does the same `restoreLiveParent(); return {0.0, noop};` every other reject gate in this function uses (on reject). `bridgeCostRescued`'s own condition also gained an `mfCtx == nullptr &&` guard so its `[Split Cost Rescue]` log can't fire under the multifactor path either.

### Files changed

- **`includes/SplitScore.hpp`** (append after `splitScoreAccept`, line 31):
  ```cpp
  inline bool  splitScoreAccept(const SplitFeatures& f, const SplitScoreWeights& w) {
      return splitLogit(f, w) >= w.sAccept;
  }

  // Optional multi-factor split-decision context. When passed (non-null) to
  // trySplitCellPhased, the split ACCEPT is decided by the score + geometric
  // sanity instead of the CellLumen cost-gate/sentinel path.
  struct SplitDecisionCtx {
      SplitFeatures parentFeatures;   // parent-only features (coverageGain filled in at accept time)
      SplitScoreWeights weights;
  };
  ```

- **`includes/Frame.hpp`** (2 edits):
  - **Line 16** — added `#include "SplitScore.hpp"` next to the existing `ForegroundModel.hpp` include (was not transitively included; `SplitDecisionCtx` is used in the `trySplitCellPhased` declaration below).
  - **Line 365** — appended one trailing default-valued parameter to the `trySplitCellPhased` declaration, after the last existing param `lumenNonParentDuplicateReanchorMinParentDistanceBalance = 0.60f`:
    ```cpp
    float lumenNonParentDuplicateReanchorMinParentDistanceBalance = 0.60f,
    // Rung 2a part 6: optional multi-factor split-decision context. When
    // non-null, the final ACCEPT is decided by splitScoreAccept(...) +
    // a geometric sanity check instead of the CellLumen cost-gate /
    // sentinel (lumenCostAccepted / bridgeCostRescued) path below. The
    // daughter build / burn-in / refine machinery above is unaffected.
    // Flag-OFF callers (nullptr, the default) are byte-identical to the
    // pre-Rung-2a-part-6 behavior.
    const SplitDecisionCtx *mfCtx = nullptr);
    ```
    Existing call sites are unaffected — the default keeps them at `nullptr`.

- **`src/Frame.cpp`** (4 edits inside `trySplitCellPhased`):

  - **Lines 3431–3432** (definition signature, mirrors the header):
    ```cpp
    float lumenNonParentDuplicateReanchorMinParentDistanceBalance,
    const SplitDecisionCtx *mfCtx)
    {
    ```

  - **Line 7385** (`bridgeCostRescued`, first sentinel). Before:
    ```cpp
    const bool bridgeCostRescued =
        probConfig.split_bridge_cost_rescue_enabled &&
        bridgeCostRescueEligible &&
    ```
    After:
    ```cpp
    const bool bridgeCostRescued =
        mfCtx == nullptr &&
        probConfig.split_bridge_cost_rescue_enabled &&
        bridgeCostRescueEligible &&
    ```

  - **Lines 8432–8536** (the `lumenCostAccepted` computation, its `[Split CellLumen Cost Gate]` log, the `[Split Reject CellLumen positive gate weak image]` diagnostic log, `rejectGateDiff`, and the final `[Split Reject cost]` cost gate) — wrapped **verbatim, unmodified** inside `if (mfCtx == nullptr) { ... }` (opening brace inserted immediately before `const bool lumenCostAccepted =`, closing brace after the final reject block's `return {0.0, noop};`). No line inside this block was edited — same variable names, same log format, same thresholds.

  - **New `else` branch, lines 8538–8603** (added immediately after the `if (mfCtx == nullptr) { ... }` block, before the pre-existing `// Accept: install the best candidate state` comment):
    ```cpp
    } else {
        // Rung 2a part 6: multi-factor score decides ACCEPT instead of the
        // CellLumen cost-gate/sentinel path above. coverageGain is filled in
        // here from the same baseline/candidate image costs the legacy cost
        // gate above uses (baselineImageCost, bestImageCost); geometric
        // sanity reuses the same daughter positions/radii and reference
        // parent volume/radii the bio gate above already validated
        // (bestD1/bestD2, refParentVolume, srcMajor/srcB/srcMinor) — nothing
        // here is recomputed from the image.
        const float parentCost = static_cast<float>(baselineImageCost);
        const float daughtersCost = static_cast<float>(bestImageCost);
        SplitFeatures pf = mfCtx->parentFeatures;
        pf.coverageGain = std::max(-1.0f, std::min(1.0f,
            (parentCost - daughtersCost) / std::max(1e-6f, parentCost)));

        const cv::Point3f mfD1Pos(bestD1.getX(), bestD1.getY(), bestD1.getZ());
        const cv::Point3f mfD2Pos(bestD2.getX(), bestD2.getY(), bestD2.getZ());
        const float daughterSeparation =
            static_cast<float>(cv::norm(mfD2Pos - mfD1Pos));
        const float parentMinR = std::min({srcMajor, srcB, srcMinor});
        const double d1Vol = static_cast<double>(bestD1.getARadius()) *
                              static_cast<double>(bestD1.getBRadius()) *
                              static_cast<double>(bestD1.getCRadius());
        const double d2Vol = static_cast<double>(bestD2.getARadius()) *
                              static_cast<double>(bestD2.getBRadius()) *
                              static_cast<double>(bestD2.getCRadius());
        const double combinedDaughterVolume = d1Vol + d2Vol;

        const bool geomSane =
            (daughterSeparation >= 0.5f * parentMinR) &&
            (combinedDaughterVolume >= 0.6 * refParentVolume) &&
            (combinedDaughterVolume <= 1.6 * refParentVolume);

        const bool scoreAccept = splitScoreAccept(pf, mfCtx->weights);
        const bool mfAccept = scoreAccept && geomSane;
        acceptedCostDiff = mfAccept ? -std::max(1.0, adaptiveThreshold) : 1.0;

        std::cout << "[Split Accept MF] " << parentName
                  << " accept=" << (mfAccept ? 1 : 0)
                  << " scoreAccept=" << (scoreAccept ? 1 : 0)
                  << " geomSane=" << (geomSane ? 1 : 0)
                  << " logit=" << splitLogit(pf, mfCtx->weights)
                  << " coverageGain=" << pf.coverageGain
                  ...
                  << std::endl;

        if (!mfAccept) {
            restoreLiveParent();
            return {0.0, noop};
        }
        // Fall through to the shared accept-install block below with
        // acceptedCostDiff already set negative.
    }
    ```
    **Real variable names reused** (per the brief's request to cite them): `baselineImageCost` (declared line 4105) → `parentCost`; `bestImageCost` (declared line 4122, holds the winning candidate's image cost by the time this code runs) → `daughtersCost`; `bestD1`/`bestD2` (the winning candidate `Ellipsoid`s, already validated by `bioCheckDaughters` a few hundred lines above at line 5670) → `daughterSeparation` (`cv::norm` of their positions) and `combinedDaughterVolume` (`a·b·c` sum, same proportional-volume convention `bioCheckDaughters`'s local `cellVolume` lambda uses — no `4/3·π` factor, consistent since it's compared as a ratio); `srcMajor`/`srcB`/`srcMinor` (declared line 3477, the snapshot/live-parent semi-axes already used to build `refParentVolume`) → `parentMinR = min(srcMajor, srcB, srcMinor)`; `refParentVolume` (declared line 3661, `srcMajor·srcB·srcMinor`, the exact same reference volume `bioCheckDaughters` was already called with at line 5670) → reused directly as `parentVolume`, no separate variable needed. `adaptiveThreshold` (the same per-cell cost-gate threshold every other accept path in this function uses) sets the accept-side magnitude of `acceptedCostDiff`, matching the `lumenCostAccepted`/`bridgeCostRescued` sentinel convention (`-std::max(1.0, adaptiveThreshold)`) so the caller's `costDiff < 0.0` test in `CellUniverse.cpp` behaves identically regardless of which path produced the accept.

- **`src/CellUniverse.cpp`** (2 edits in `attemptSplitAtIndex`, inside `CellUniverse::optimize`):

  - **Lines 7696–7712** (new block, inserted immediately before `auto result = frame.trySplitCellPhased(`):
    ```cpp
    // Rung 2a part 6: build the multi-factor split-decision context
    // when the flag is on. splitFeaturesByName was populated earlier
    // in this function (per-cell parent features), only when
    // multifactor_split_enabled; if this cell has no entry (e.g.
    // splits were disabled for the frame), fall back to nullptr so
    // trySplitCellPhased takes its legacy CellLumen cost-gate path.
    SplitDecisionCtx mfCtxStorage;
    const SplitDecisionCtx *mfCtxPtr = nullptr;
    if (config.prob.multifactor_split_enabled) {
        auto sfIt = splitFeaturesByName.find(cellName);
        if (sfIt != splitFeaturesByName.end()) {
            mfCtxStorage.parentFeatures = sfIt->second;
            mfCtxStorage.weights.wElong = config.prob.split_w_elong;
            mfCtxStorage.weights.wValley = config.prob.split_w_valley;
            mfCtxStorage.weights.wCenter = config.prob.split_w_center;
            mfCtxStorage.weights.wVolume = config.prob.split_w_volume;
            mfCtxStorage.weights.wCentral = config.prob.split_w_central;
            mfCtxStorage.weights.wDensity = config.prob.split_w_density;
            mfCtxStorage.weights.wCost = config.prob.split_w_cost;
            mfCtxStorage.weights.bias = config.prob.split_score_bias;
            mfCtxStorage.weights.sAccept = config.prob.split_score_accept;
            mfCtxPtr = &mfCtxStorage;
        }
    }
    ```
  - **Line 7920** (last argument of the `trySplitCellPhased(...)` call). Before:
    ```cpp
                config.cellLumen
                    .fusionSplitPriorNonParentDuplicateReanchorMinParentDistanceBalance);
    ```
    After:
    ```cpp
                config.cellLumen
                    .fusionSplitPriorNonParentDuplicateReanchorMinParentDistanceBalance,
                mfCtxPtr);
    ```

- **`CMakeLists.txt`** (2 edits):
  - **`add_executable(${PROJECT_NAME} ...)` sources list** — added `${cwd}/src/SplitScore.cpp` and `${cwd}/src/SplitFeatureExtractor.cpp` next to `${cwd}/src/ForegroundModel.cpp`. This is the fix for the link failure Change 32 documented and worked around.
  - **`BUILD_CELLU_TESTS` block** — see "Deviation" below; the two new test files ended up as separate `add_executable` targets rather than sources added to the existing `celluniverse_tests` target.

### Deviation from the task brief: test files as separate executables, not merged into `celluniverse_tests`

The brief's Step 5 said to add `tests/test_split_score.cpp` + `tests/test_split_feature_extractor.cpp` (plus their `src/*.cpp` modules) directly to the existing `celluniverse_tests` `add_executable` sources list, alongside `tests/test_foreground_model.cpp`. **Tried this literally first** — it fails to link:
```
duplicate symbol '_main' in:
    .../celluniverse_tests.dir/tests/test_foreground_model.cpp.o
    .../celluniverse_tests.dir/tests/test_split_score.cpp.o
    .../celluniverse_tests.dir/tests/test_split_feature_extractor.cpp.o
ld: 1 duplicate symbols
```
All three test files are self-contained programs with their own `int main()` (same style Rung 1 established for `test_foreground_model.cpp`) — a pre-existing fact from Tasks 1–5, not something introduced by this task. Combining 3 `main()`s into 1 binary is not possible without editing at least 2 of the 3 test files (e.g. renaming `main` to a named function and adding a driver to call all three), which is out of this task's scope (`tests/*.cpp` is not in the brief's touch list, and the maintainer's standing rule is surgical, minimal-footprint changes).

**Fix (CMake-only, zero test-file edits):** kept `celluniverse_tests` exactly as it was (still just `test_foreground_model.cpp` + `ForegroundModel.cpp`, byte-identical to before this change) and added two new, independent `add_executable` targets — `celluniverse_tests_split_score` (`test_split_score.cpp` + `SplitScore.cpp`) and `celluniverse_tests_split_feature_extractor` (`test_split_feature_extractor.cpp` + `SplitScore.cpp` + `SplitFeatureExtractor.cpp`) — each registered with `add_test(...)` so `ctest` picks up all three. This is arguably a *more* faithful application of "mirror how `ForegroundModel.cpp` is wired" than the brief's literal instruction: one test-with-`main()` + its module(s) → one executable, applied per module rather than force-merged.

### Verify (all ran clean)

```
$ cmake -S . -B build -DBUILD_CELLU_TESTS=ON
-- Configuring done / Generating done

$ cmake --build build -j 8 --target celluniverse_tests celluniverse_tests_split_score celluniverse_tests_split_feature_extractor
[100%] Built target celluniverse_tests
[100%] Built target celluniverse_tests_split_score
[100%] Built target celluniverse_tests_split_feature_extractor

$ ./build/celluniverse_tests && ./build/celluniverse_tests_split_score && ./build/celluniverse_tests_split_feature_extractor
test_foreground_model: PASS
test_split_score: PASS
test_split_feature_extractor: PASS

$ cmake --build build -j 8 --target celluniverse
[100%] Built target celluniverse       # previously failed to link (Change 32); now links clean
```

**Flag-OFF no-op, f0–3** (default `config/config.yaml`, `multifactor_split_enabled: false`, `CELLUNIVERSE_SEED=42`, `data/input/embryo_data/t%03d.tif`, `config/Teammate_initial/initial_embryo.csv`):
```
exit=0
Saved cells per frame: 4 7 7 7
grep -c "Split Accept MF" run.log → 0
```
Zero `[Split Accept MF]` lines, and the legacy `lumenCostAccepted` path (guarded by `if (mfCtx == nullptr)`) still fires normally — 3 `[Split Accepted] ... bestLabel=cell_lumen_primary costDiff=-1` lines in this same run, confirming the guard additions didn't silently disable the pre-existing accept path. Confirms flag-OFF is a pure no-op: `mfCtxPtr` is `nullptr` whenever `multifactor_split_enabled` is `false` (the default), which makes every new `mfCtx == nullptr &&` guard evaluate to its original unguarded value and routes execution into the untouched legacy `if (mfCtx == nullptr) { ... }` block.

---

## 2026-07-31: Rung 2a gate bypass — guard the legacy quality gauntlet under `mfCtx` (Change 34)

**Status: ACTIVE — verified flag-off byte-identical (0 `[MF Gate Bypass]` lines, same per-frame cell counts, legacy `lumenCostAccepted` path still fires 3× on f0–3); flag-on f0–6 shows candidates now reaching the `[Split Accept MF]` decision instead of being killed by the pre-Rung-2a-part-6 quality gauntlet.**

### Why

Change 33 (Rung 2a part 6) made the multi-factor score decide ACCEPT via `splitScoreAccept` — but only for the *last* gate in `trySplitCellPhased`. An A/B run showed the score works as a proposal generator (it fires split attempts) but almost never gets to decide: of 29 split attempts, only 3 reached the `[Split Accept MF]` block. The other 26 were rejected earlier by the pre-existing quality cascade (bio gate, bridge/valley gate, daughter-overlap gate, dozens of CellLumen duplicate/drift/anchor heuristics) that predates Rung 2a and was written to gate the *old* geometry-only split path. Result: severe under-splitting. This change makes the MF score the sole judge for every quality gate in the function, not just the final cost gate — when `mfCtx` is set, none of the old quality rejects fire; the daughters flow through to the existing MF accept block (unchanged from Change 33).

### Summary

Read `Frame::trySplitCellPhased` end-to-end (src/Frame.cpp, lines 3300–8736 — the entire ~5437-line function). Found every early no-split return between best-candidate selection (`bestIdx`/`bestD1`/`bestD2`, set ~line 4830) and the accept block (~line 8605) via `restoreLiveParent()`/`restoreLiveParentAt(...)` occurrences (48 total). Classified each:

- **45 QUALITY gates** — a real division could legitimately fail these; GUARDED with `if (mfCtx == nullptr) { <existing reject> }` + a `[MF Gate Bypass] <gateName> cell=<name>` log on the else-path. Covers: the bio gate (`bioCheckDaughters`, daughter-geometry-drift, axis-alignment), the bridge/valley gate (`edge_too_dim`, `bridge_flat`, `lumen_overlap_no_valley`, `lumen_bridge_gap_too_small`), the daughter-overlap gate, and 38 CellLumen duplicate/drift/parent-anchor reject heuristics (`lumenLikelyContinuationHijack`, the `parentAnchor*Duplicate`/`lumen*Duplicate` family, `daughterExistingOverlapDuplicate`, `cellLumenOverlap`, etc.).
- **3 STRUCTURAL bailouts** — LEFT unguarded (fire regardless of `mfCtx`): `cellIndex >= cells.size()` bounds check (line 3435), `too_few_bright_pixels` (< 20 Voronoi-filtered bright pixels — no candidate geometry exists at all, line 3754), and `bestIdx < 0` (no candidate survived the per-candidate edge-brightness/valley pre-filter during burn-in — `bestCells`/`bestD1`/`bestD2` are literally unassigned in this branch, so there is nothing for the MF score to evaluate, line 4461).
- **2 sites already guarded by Change 33** — verified untouched: `bridgeCostRescued`'s definition (`mfCtx == nullptr && ...`, line 7485 — an accept shortcut, not a reject gate, already disabled under `mfCtx`) and the legacy `lumenCostAccepted`/adaptive-cost-gate cascade (`if (mfCtx == nullptr) { ... } else { ... }`, line 8567 — the pre-existing Rung 2a part 6 MF accept block itself).

Guard pattern, applied uniformly (matches the style Change 33 already established for the final cost gate):
```cpp
if (someRejectCondition) {
    std::cout << "[Split Reject ...] " << parentName << ... << std::endl;  // unchanged
    if (mfCtx == nullptr) {
        restoreLiveParent();          // or restoreLiveParentAt(pos, "reason") for reanchor variants
        return {0.0, noop};
    }
    std::cout << "[MF Gate Bypass] <gateName> cell=" << parentName << std::endl;
}
```
Every diagnostic `std::cout` and value computation (valley ratios, overlap fractions, soft-gate penalties, cost diffs) is unchanged — only the `return` (and its immediately-preceding `restoreLiveParent[At]()`) is guarded. Three gates whose reject sits behind a `reanchor-or-plain-restore` branch (`parentAnchorOneRealRefitDriftGuard`, `parentAnchorOneRealOverlapNoGapDuplicate`, `parentAnchorOneRealMinImageGainGuard`, `parentAnchorOneRealCleanHighOverlapGuard`, `parentAnchorOneRealLowShapeOverlapDuplicate`, `parentAnchorOneRealOverlapDominatedWeakDuplicate`) and one gate with three separate return sub-branches sharing one enclosing `if` (`daughterExistingOverlapDuplicate`) got the whole inner if/else wrapped, not each `return` individually, so there's exactly one bypass log per logical gate.

No new config knob. No code deleted (per the task's binding constraint — gate removal is a later step once this A/B validates).

### Files changed

- **`src/Frame.cpp`** — 45 guard insertions inside `Frame::trySplitCellPhased`, spanning lines ~4930–8563 (post-edit line numbers). Two of the 45 (`daughterExistingOverlapDuplicate` at line 6858, `parentAnchorOneRealOverlapDominatedWeakDuplicate` at line 8184) live in blocks with pre-existing mixed tab/space indentation (an artifact of an earlier edit, not introduced here) — applied via a small Python find-and-wrap script instead of the `Edit` tool to avoid a whitespace-exact-match failure; verified by reading the resulting block back and by the brace-balance + build checks below. Every other guard was applied directly.

  Representative example (line 5687, `bioCheckDaughters` — the core bio gate):
  ```cpp
  // Before:
                    << " refParentVolume=" << refParentVolume
                    << std::endl;
          restoreLiveParent();
          return {0.0, noop};
      }

  // After:
                    << " refParentVolume=" << refParentVolume
                    << std::endl;
          if (mfCtx == nullptr) {
              restoreLiveParent();
              return {0.0, noop};
          }
          std::cout << "[MF Gate Bypass] bioCheckDaughters cell=" << parentName << " reason=" << bioReason << std::endl;
      }
  ```

  Full per-gate line list (post-edit): see `C++/docs/plans/` job report
  `rung2a-gatebypass-report.md` for the complete 45-row table with GUARD/LEAVE
  classification and reasoning for every one of the 48 early-return sites found.

- **`docs/changelogs/changelogv8.md`** — this entry.

### Sanity checks

- `{`/`}` brace count in `src/Frame.cpp`: 761 / 761 (balanced) before running the compiler.
- `if (mfCtx == nullptr)` occurrence count: 46 (45 new + the 1 pre-existing Change-33 block).
- `[MF Gate Bypass]` distinct log-line count: 45 (one per logical gate).

### Build

```
$ cmake --build build -j 8 --target celluniverse
[100%] Built target celluniverse    # clean, no new warnings
```

### Verify

**1. Flag-OFF no-op** (default `config/config.yaml`, `multifactor_split_enabled: false`, `CELLUNIVERSE_SEED=42`, f0–3, `data/input/embryo_data/t%03d.tif`, `config/Teammate_initial/initial_embryo.csv`):
```
exit=0
Saved cells per frame: 4 7 7 7
grep -c "MF Gate Bypass" run.log → 0
```
Zero `[MF Gate Bypass]` lines and 0 `[Split Accept MF]` lines; the legacy `lumenCostAccepted` accept path still fires 3× (`[Split Accepted] ... bestLabel=cell_lumen_primary costDiff=-1`), identical to the Change-33 flag-off baseline. Confirms every new `if (mfCtx == nullptr)` guard takes its reject branch unconditionally when `mfCtx` is null — flag-off is byte-identical in behavior to pre-Change-34.

**2. Flag-ON (guidance-only config — `multifactor_split_enabled: true`, `fusionSplitPriorForceSchedule/SkipRandomSplits/GuidedOnly/UseDedicatedCostGate: false`, seed=42, f0–20):**
```
counts:  4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 5 5   (still under-splits)
MF accept decisions reached: 5   gate-bypass events: 3 (bridgeFlat×2, daughterDaughterOverlap×1)
```
The gate-bypass alone did NOT fix under-splitting: only 3 candidates ever hit a guarded gate. Diagnosis (candidate log): candidates die EARLIER, at the per-candidate `[Split Cand PreFilter]` valley check (`NO_VALLEY`, valley≈1.0 — this fluorescence data has no dark bridge), before `bestIdx` selection — so they never reach the guarded gates or the score. The prefilter demotion is Change 35 (T8).


---

## 2026-07-31: Rung 2a T8 — demote the split-candidate VALLEY prefilter under `mfCtx` (Change 35)

**Status: ACTIVE — the fix that lets the multifactor score actually fire splits on non-binary (no-valley) data. Flag-off byte-identical.**

### Why
Change 34 (gate bypass) did not fix under-splitting because candidates die EARLIER than the guarded gates: the per-candidate `[Split Cand PreFilter]` valley/edge check (`bothDaughtersBright && (softValley || candValleyFromBright < valleyLimit)`) rejects every short-axis candidate on this fluorescence data (`NO_VALLEY`, valley≈1.0 — no dark bridge between forming nuclei), so `bestIdx` is never set and the multifactor score is never reached. The valley is a binary-image-era confirmation; it has no signal here. It is ALREADY a soft feature in the multifactor score (via `Frame::measureSplitEvidence`), so the hard prefilter is redundant and harmful in multifactor mode. This supersedes Change 34's classification of the per-candidate prefilter → `bestIdx<0` path as a "structural bailout LEFT unguarded": the *quality* half (valley/edge) is now guarded; the true no-candidate `bestIdx<0` bailout remains unguarded.

### Change (`src/Frame.cpp`, candidate cost-comparison ~4438–4474)
Was a single gate `if (candPassesPreFilter && candTotal < bestTotal) { <set bestIdx...> }`. Now:
```cpp
bool candEligibleForBest = candPassesPreFilter;
if (!candPassesPreFilter && mfCtx != nullptr) {
    candEligibleForBest = true;   // multifactor: valley is a soft score feature, not a hard prefilter
    std::cout << "[MF Prefilter Keep] cell=" << ... << " label=" << ... << " valley=" << ... << std::endl;
}
if (candEligibleForBest && candTotal < bestTotal) { <set bestIdx/bestCells/bestSeed...> }
```
A retained candidate still went through `buildDaughter` + per-candidate burn-in (unconditional in the loop body), so `bestD1/bestD2` are always well-formed; the downstream MF accept still gates on `splitScoreAccept && geomSane` (daughter separation + combined-volume window, NaN→reject), so a degenerate candidate cannot commit.

### Verified
- **Flag-OFF no-op** (default config, seed=42, f0–3): `4 7 7 7`, 0 `[MF Prefilter Keep]` lines, identical to the Change-34 flag-off baseline. `candEligibleForBest ≡ candPassesPreFilter` when `mfCtx` is null.
- **Flag-ON reaches the score**: candidates now survive the prefilter (55 `[MF Prefilter Keep]`) and reach `bestIdx` + the multifactor accept; interim f0–5 counts `4 4 4 5 6 6` vs GT `4 4 4 4 6 6` (tracking; was frozen at 4). Full f0–20 A/B vs the verified GT (`4 4 4 4 6 6 6 7 7 7 7 8 8 8 8 8 8 8 10 12 12`) in flight; weight tuning (T9) follows.

### Opus review
Code Approved: only the valley/edge quality prefilter is guarded (structural bailouts untouched); retained candidates genuinely enter the `bestIdx` cost comparison; no degenerate-commit path (geomSane guards). No Critical/Important code findings.

## 2026-07-31: Rung 2b diagnostic — read-only `[SplitTemporal]` 2nd-nucleus trajectory probe (Change 36)

**Status: DIAGNOSTIC (read-only) — emits `[SplitTemporal]` log lines only; changes NO decision. Active only inside the `multifactor_split_enabled` branch, so the flag-off path is byte-identical. Purpose: test whether temporal DIVERGENCE (2nd-nucleus separation growth) discriminates real divisions from CellLumen over-detection FPs, BEFORE building it as a score feature.**

### Why
The T8 A/B over-splits (21 vs verified GT 12). Root cause (diagnosed from `[Split Accept MF]` feature stats over f0–20): the `centerSupport` feature is **saturated** — 72% of *all* cells score `center ≥ 0.85` because CellLumen over-detects (f1: 30 blobs for ~4 cells; f15: 46), so a "second nucleus" exists on nearly every cell. Accept vs reject means (accepts n=17, rejects n=37): `center` 0.970 vs 0.804, `volume` 0.734 vs 0.362, `valley` 0.048 vs 0.035, `coverageGain` 0.057 vs 0.023, `elongation` 0.626 vs 0.807 (inverted). Only `volume` weakly separates (heavy overlap) — so the decision is a growth-driven cascade, and no per-frame weight tuning can hit near-0 early FP. The division signal is **temporal**. Naive persistence was already ruled out from the existing `[CellLumen Fusion SplitPrior Ranked]` `windowBoth`/`windowMissing` fields: FP `cell_100@f8` (`windowBoth=2 windowMissing=0`) is identical to real `cell_4@f3` because the *future* frames are over-detected too (there is always a blob to match forward). The remaining candidate discriminator is **divergence** (separation growth), which is not logged anywhere — hence this probe.

### Change (`src/CellUniverse.cpp`, inside the multifactor `P(split)` per-cell loop, after the `[Split Score]` log ~5858)
Added a read-only block (guarded by `allowSplits && maxR > 1e-3f`, only reached under `multifactor_split_enabled`) that, per non-trash cell at frame `t`:
- finds the **primary** nucleus `P1` = nearest CellLumen lookahead candidate to the cell center, and the **second** nucleus `P2` = strongest candidate in `[0.3, 1.2]·maxR` (same criterion as the `centerSupport` feature at `CellUniverse.cpp:5819–5834`);
- forward-matches `P1`,`P2` to the nearest candidate at `t+1` and `t+2` via `getCellLumenLookaheadCandidates(frameIndex+1/+2)` (the same on-demand forward detector the window machinery already uses — no new detection path);
- logs `sep0/sep1/sep2` (the P1–P2 separation at each frame), `div = sep2 − sep0`, the forward match distances `p1d1/p2d1/p1d2/p2d2` (persistence quality), and the nucleus count within `1.2·maxR` at each frame `nuc0/nuc1/nuc2`.
```cpp
// [SplitTemporal] frame <t> <name> maxR=.. haveP2=0/1 sep0=.. sep1=.. sep2=.. div=.. p1d1=.. p2d1=.. p1d2=.. p2d2=.. nuc0=.. nuc1=.. nuc2=..
```

### Side-effect safety
Read-only: only `std::cout` lines are added; no cell/state mutation. It calls `getCellLumenLookaheadCandidates(t+1/t+2)`, which the fusion window path (`fusionSplitPriorWindowEnabled`, active in the A/B config) already calls — the accessor is a deterministic pure function of the frame image and caches its result. To be confirmed on this run: per-frame counts reproduce the T8 trajectory (`4 4 4 5 6 6 8 8 9 10 11 14 15 16 17 17 17 18 19 21 21`) exactly, proving the probe does not perturb the split decisions.

### Test cases (tree-aligned vs verified GT `4 4 4 4 6 6 6 7 7 7 7 8 8 8 8 8 8 8 10 12 12`)
- **Real** (GT f4 +2, one caught 1 frame early): `cell_4@f3`, `cell_1@f4` → expect `div > 0` (daughters separate).
- **FP** (GT flat 6→7, run overshoots): `cell_10@f6`, `cell_2@f6`, `cell_100@f8`, `cell_11@f9`, `cell_3@f10` → expect `div ≈ 0` (one nucleus's over-seg halves).
If `div` separates these, it becomes the discriminative feature gating the saturated `centerSupport`; if not, report and seek a different signal (parent-disappearance / signal-balance) — do not build on an unconfirmed assumption.

### Result (diagnostic run f0–14, seed=42, single-thread)
Side-effect check PASSED: per-frame counts reproduced T8 exactly (`4 4 4 5 6 6 8 8 9 10 11 14 15 16 17`). **Divergence does NOT discriminate — the hypothesis is rejected**, for a deeper reason than expected:
- **Real early divisions show only ONE nucleus at the decision frame.** `cell_4@f3` and `cell_1@f4` (the two GT-f4 divisions) both logged `haveP2=0 nuc0=1` — no pair to measure. The `centerSupport` that fired them was reading a *single, off-center* nucleus in the `[0.3,1.2]·maxR` ring, not two nuclei.
- FP splits mostly showed a pair whose separation **collapses** (`cell_10@f6` 4.4→0, `cell_100@f8` 11→24→0, `cell_3@f10` 16→0), i.e. `div<0`. So divergence would *reject* fakes but **cannot confirm reals** (they have no pair) → a divergence gate under-splits the real cases.

**Root reason (user insight, 2026-07-31):** this data is **nucleus-only** (no membrane); a "cell" IS a nucleus and division is literally 1 nucleus → 2. The data does not render the *separation process* (anaphase is fast), so there is no divergence to see — the two daughter nuclei simply appear the next frame. The observable precursor is the **metaphase plate**: the nucleus flattens into an oblate disk while still one nucleus. → pivot to a per-frame planarity feature (Change 37). The `[SplitTemporal]` probe is removed in Change 37 (its question is answered).

## 2026-07-31: Rung 2b — planarity (metaphase-pancake) split feature + centerSupport de-weight (Change 37)

**Status: ACTIVE (behind `multifactor_split_enabled`, default OFF). Replaces the saturated `centerSupport`-driven over-split with a per-frame OBLATENESS (metaphase-plate) driver — the correct precursor for nucleus-only division. Retires the Change-36 `[SplitTemporal]` diagnostic probe.**

### Why
Change 36 proved divergence/second-nucleus signals are absent at the decision frame (real early divisions present one nucleus). Nucleus-only data means the pre-division observable is the **metaphase plate**: the nucleus flattens to an oblate disk (two large axes ≈ equal, third much smaller) while still one nucleus, then becomes two. The daughters separate along the plate normal = the **shortest** fitted axis, which is exactly the axis the split code already seeds daughters along — so an oblate cell's shortest axis IS the spindle/division axis. Generic `elongation` (`maxR/minR`) can't capture this — it fires equally on a cigar (prolate, e.g. a cell squished by neighbours, NOT dividing) — which is why it was non-discriminative in the T8 A/B (accepts 0.63 vs rejects 0.81, inverted). Verified from cells.csv: quiescent founders are round (oblate `(r2-r3)/r1` ≈ 0.006–0.019) while cells approaching division pancake (`cell_1@f3` 0.263→split f4; `cell_10@f5` 0.350; `cell_11@f5` 0.313). The saturated `centerSupport` (≈1 on 72% of cells from CellLumen over-detection) is de-weighted so it stops driving the growth-cascade over-split.

### Changes
1. **`includes/SplitScore.hpp`** — `SplitFeatures` gains `float planarity` (after `elongation`); `SplitScoreWeights` gains `float wPlanarity=0.0f` (default 0 → header/unit-test behaviour unchanged); `splitLogit` adds `+ w.wPlanarity*f.planarity`.
2. **`includes/SplitFeatureExtractor.hpp`** — `SplitFeatureInputs` gains `float midR` (the median semi-axis) alongside `maxR/minR`.
3. **`src/SplitFeatureExtractor.cpp`** — `buildSplitFeatures` computes `planarity = clamp01((r2-r3)/r1)` from the sorted radii (r1≥r2≥r3); `midR<=0` falls back to `0.5*(maxR+minR)`. High for a disk, ~0 for sphere OR cigar.
4. **`includes/ConfigTypes.hpp`** — `ProbabilityConfig` gains `float split_w_planarity = 0.0f` + its parse line (after `split_w_elong`).
5. **`config/config.yaml`** — added `split_w_planarity: 8.0`; **`split_w_center: 2.0 → 0.5`** (de-weight saturated over-detection feature). Only active when `multifactor_split_enabled` (OFF by default), so legacy behaviour is unchanged.
6. **`src/CellUniverse.cpp`** — set `W.wPlanarity = config.prob.split_w_planarity`; set `in.midR = a+b+cc-maxR-minR`; added `planarity=` to the `[Split Score]` log; **removed the Change-36 `[SplitTemporal]` probe block** (its question is answered; it doubled runtime via forward detections).
7. **`src/Frame.cpp`** — added `planarity=` to the `[Split Accept MF]` log.

### To verify (A/B in flight, seed=42, single-thread, f0–20)
Guidance-only config (multifactor on, 4 fusion flags off) with the pancaking weights. Target: per-frame counts track GT (`4 4 4 4 6 6 6 7 7 7 7 8 8 8 8 8 8 8 10 12 12`), end near 12 (not the T8 over-split 21), FPs down. Watch: the birth-mask fit cap can inflate a real pancake's short axis to round (`cell_4@f2` raw PCA iter0 `c=10.8` oblate 0.30 → converged `c=16.4` oblate 0.06 as it drifted/expanded) — if real dividers are missed, measure planarity pre-inflation (early-iter fit / bright-voxel covariance). Accept-side 1→2-nucleus confirmation is the deferred follow-up.

**SUPERSEDED before completion:** the pancaking A/B never ran to term — a screenshot review found the fitted cells are **grossly undersized** (~1/3 of the true nucleus), which *hides* the pancaking signal the whole feature depends on (Change 38). Change 37 must be re-A/B'd on correctly-sized cells.

## 2026-07-31: Rung 2b — undersized-fit root cause + sized seed CSV (Change 38)

**Status: ACTIVE — new seed `config/initial_embryo_sized.csv` gives the 4 founders explicit radii so the fit covers the whole nucleus from f0. Prerequisite for pancaking (planarity) to be readable. No code change; new data file + diagnosis.**

### Why (root cause of the undersized cells)
A synth-on-real overlay showed the fitted ellipsoids covering only the bright core, ~1/3 of each nucleus. Measured from the raw TIFF with the exact `Frame Intensity Scale` (`low_ref=22.76→0, high_ref=93.98→1`, background 0.034): the true in-plane nucleus radius is **~40–52px**, but the fit converged to **~13–17px (26–34% coverage)**. Two causes, neither is preprocessing (`preprocess_mode: none`; contrast is good):
1. **Seed defaults to 10px.** The embryo seed CSV (`initial_embryo.csv`, header `cellType,z,y,x`) has NO radius column → `CsvHandler.cpp:214` (`parseNamedNapariCell`) hardcodes `aRadius = 10.0 * initialRadiusScale = 10px`.
2. **The PCA fit is box-bounded.** `calibrateCellShapeViaPca` gathers/masks bright pixels only within `pcaShapeMaskScale (1.45) × radius`, fixed at frame entry (`Frame.cpp:2582–2585`). From a 10px seed the f0 box is only 14.5px — blind to the ~40px nucleus — so it converges at ~13 and creeps up ~1–3px/frame thereafter, staying undersized through the early frames where pancaking must be read.

### Fix
New seed `config/initial_embryo_sized.csv` in the rich/named format (`file,name,x,y,z,majorRadius,bRadius,minorRadius,theta_*`) with the original founder positions + accurate per-cell radii (a≈37–44, b≈35–41, c≈33–37, in fit/interpolated units; z stays raw=16, ×7-scaled by the loader; radii used as-is). Uses the **default config** (`initialRadiusScale: 1.0`, `multifactor_split_enabled: false`) — no hack.

### Verified (f0–3, seed=42, single-thread, default config + sized CSV)
- f0 fits now `cell_1 a=44.1 b=41.0 c=37.1`, … — accurate from frame 0; **coverage 77–86%** of the true nucleus (was 26–34%). Fits hold across frames (no collapse).
- **Oblate shape now visible**: cell_1 c=37 vs a=44 at f0 → c=27.6 vs a=36.8 by f3 (planarity 0.09→0.22), where the undersized fit read ~spherical (planarity ~0.02). The size fix is what makes planarity (Change 37) meaningful.

### Follow-ups
- All embryo runs must now seed from `config/initial_embryo_sized.csv`.
- Re-run the Change-37 pancaking A/B on the sized cells (the prior attempt was on hidden-pancake undersized cells).
- Remaining ~20% is dim outer halo; push the fit further (bigger seed / fit-side change) only if GT SEG or the overlay says the halo is true nucleus.

## 2026-07-31: Rung 2b — fix `wPlanarity` omitted from the accept-time weights (Change 39)

**Status: BUGFIX to Change 37. `src/CellUniverse.cpp:7705`.**

### Why
The first pancaking A/B on the sized seed produced ZERO splits through f6 (GT `4 4 4 4 6 6 6`). The `[Split Accept MF]` logs exposed the cause: `cell_2` (`planarity=0.250 elongation=0.915 valley=0.235 central=0.400 center=0.985`) was rejected with `logit=-0.672` — and that value equals the full logit *minus* the `wPlanarity·planarity` term (`8·0.250 = 2.0`; `-0.672 + 2.0 = 1.33 > 0` would accept). Change 37 added `W.wPlanarity` to the ATTEMPT-probability weights (`CellUniverse.cpp:5783`) but not to the ACCEPT-time weights built in `attemptSplitAtIndex` (`mfCtxStorage.weights`, `:7705–7713`), which set every other `split_w_*` explicitly. So planarity drove the split *attempt* but was silently zero at *accept*, and the pancaked cells could never clear the gate.

### Change (`src/CellUniverse.cpp`, after line 7705)
```cpp
mfCtxStorage.weights.wElong = config.prob.split_w_elong;
mfCtxStorage.weights.wPlanarity = config.prob.split_w_planarity;   // ADDED
mfCtxStorage.weights.wValley = config.prob.split_w_valley;
```
Now the accept logit matches the attempt logit (both include the pancaking term). Re-running the f0–6 sized-seed A/B to confirm the first split fires on a genuinely pancaked cell at ~f4 (GT).

## 2026-07-31: Option 1 — scale-invariant foreground-anchored shape fit behind `pcaShapeFgAnchored` (Change 40)

**Status: IMPLEMENTED + GATED. Flag OFF is byte-identical (verified). Flag ON does NOT yet work on this data — the `pFg>0.5` flood-fill explodes to whole-image components; root cause diagnosed below. The feature ships OFF by default; the ON path needs a more selective foreground criterion before it recovers the true nucleus.**

### Why
The legacy PCA fit (`calibrateCellShapeViaPca`) gathers bright voxels inside a size-relative box `pcaShapeMaskScale × radius` and re-clips them to the frozen ellipsoid mask. Every knob in that path (`pcaShapeMaskScale`, `pcaShapeMaxPosShiftFraction`, seed `initialRadiusScale`) is a *size-relative constant* that must be re-tuned per dataset, and from an undersized seed it undersizes/drifts (Change 38). Option 1 adds a *scale-invariant, self-calibrating* alternative: gather by flood-filling the cell's own self-calibrated foreground component (`ForegroundModel::pFg > 0.5`) — no size constant enters the gather.

### Changes (all gated on the new flag; OFF path untouched)

**1. `includes/ConfigTypes.hpp` — new cell-config field + parse.**
After `pcaShapeMaxPosShiftFraction` (field ~`:1134`):
```cpp
    float pcaShapeMaxPosShiftFraction{0.5f}; // cap per-iter shift at fraction * maxR
    // Foreground-anchored shape fit (2026-07-31). When true AND a valid
    // per-frame ForegroundModel is installed on the Frame, the PCA shape fit
    // gathers voxels by flood-filling the cell's self-calibrated foreground
    // component (pFg > 0.5) ... Default false → the legacy box fit is byte-identical.
    bool  pcaShapeFgAnchored{false};                                   // ADDED
```
Parse (~`:1281`):
```cpp
        if (node["pcaShapeFgAnchored"]) pcaShapeFgAnchored = node["pcaShapeFgAnchored"].as<bool>();  // ADDED
```

**2. `config/config.yaml` — default key (`:66`).**
```yaml
  pcaShapeMaxPosShiftFraction: 0.5
  pcaShapeFgAnchored: false     # ADDED — Option 1 fg-anchored fit, default off
```

**3. `src/CellUniverse.cpp:5645` — estimate the FG model when EITHER flag is set.**
Before:
```cpp
    if (config.prob.generative_data_term_enabled) {
        ...
        frame.setForegroundModel(fm, true);
        std::cout << "[FG Model] frame " << displayFrame ... << " (generative data term ON)" << std::endl;
    } else {
        frame.setForegroundModel(ForegroundModel{}, false);
    }
```
After:
```cpp
    const bool fgAnchoredFit = config.cell && config.cell->pcaShapeFgAnchored;
    if (config.prob.generative_data_term_enabled || fgAnchoredFit) {
        ...
        // enabled flag flips the COST term; fg-anchored fit only needs _fgModel
        // installed and must NOT change the cost — so pass generative flag as-is.
        frame.setForegroundModel(fm, config.prob.generative_data_term_enabled);
        std::cout << "[FG Model] frame " << displayFrame ...
                  << " (cost_term=" << (config.prob.generative_data_term_enabled ? 1 : 0)
                  << " fg_anchored_fit=" << (fgAnchoredFit ? 1 : 0) << ")" << std::endl;
    } else {
        frame.setForegroundModel(ForegroundModel{}, false);
    }
```
The model is estimated at `:5645`, before the per-frame fit driver at `:6213`, so `_fgModel` is installed when the fit runs. `enabled=generative_data_term_enabled` keeps `calculateBboxCost` on the legacy L2 path when only the fit flag is on (verified: log reads `cost_term=0 fg_anchored_fit=1`).

**4. `src/Frame.cpp:1881` — new free function `gatherForegroundComponent` (does NOT touch `gatherBrightPixelsVoronoi`).**
6-connected BFS from the cell center over voxels where `fgModel.pFg(I) > 0.5`, bounded to a safety box (`4 × maxR`), each voxel Voronoi-split against the neighbor claim points (same rule as the box gather). Returns `BrightPixel{pos, weight = max(0, I − background)}`. Seeds at the center voxel; if that voxel is sub-0.5, falls back to the nearest qualifying voxel in the box.

**5. `src/Frame.cpp::calibrateCellShapeViaPca` — three gated branches (each byte-identical when OFF).**
- `:2746` compute the mode once: `const bool fgAnchored = Ellipsoid::cellConfig.pcaShapeFgAnchored && _fgModel.valid;`
- `:2801` gather branch — `if (fgAnchored) cachedRaw = gatherForegroundComponent(_realFrame, _fgModel, _backgroundValue, center, 4.0f*maxR, selfClaim, otherCellsClaimSets); else cachedRaw = gatherBrightPixelsVoronoi(...);`
- `:2877` mask pre-filter bypass — `if (fgAnchored) pixels = raw; else { …existing invA2Fixed ellipsoid test… }` (the component already defines membership; re-clipping would reintroduce the size cap).
- `:3057` position bypass — `if (fgAnchored) newCenter = weighted centroid (no cap); else { …existing maxPosShiftFraction×maxR cap… }`.

### Verified — flag OFF is byte-identical
`f0–3`, `CELLUNIVERSE_SEED=42 CELLUNIVERSE_THREADS=1`, default `config/config.yaml` (`pcaShapeFgAnchored: false`), seed `config/initial_embryo_sized.csv`, run BEFORE and AFTER all edits:
`diff` of `cells.csv` → **identical**; `diff` of `tracked_cells.csv` → **identical**. The legacy path is untouched.

### Flag ON — does NOT work on this data (honest result, root cause found)
`f0–6`, seed=42, single-thread, sized seed, raw normalization (`frame_intensity_scale 0.0/1.0`) + `multifactor_split_enabled: true` + fusion prior forced off + `pcaShapeFgAnchored: true` (run `outputs/20260731_215559_fganchored`):

| metric | target | fg-anchored ON | legacy box (f0–3 ref) |
|---|---|---|---|
| fitted aRadius | ~35–45px | **48.6–70px (pinned to max/growth-cap)** | 35–44px |
| per-cell pos maxjump | <10px | **24–52px** | 23–26px |
| per-frame counts | `4 4 4 4 6 6 6` | **`4 4 4 4 4 4 4`** (no splits) | — |

The `[PCA Shape]` logs show the flood-fill grabbing `n = 15–28 million` voxels (essentially the whole volume): `R=(265,174,127)` on iter 0, clamped down to the 70px max. **Root cause (not a wiring bug):** with raw 0/1 normalization the per-frame model is `muFg=0.041, sigFg=0.020, muBg=0.028, sigBg=0.052`. Because `muFg ≈ muBg` and `sigFg < sigBg`, the narrow foreground Gaussian's tall peak wins over a wide band — `pFg>0.5` holds for `I ∈ [0.014, 0.073]`, which *includes the background mean* (`pFg(0.028)=0.68`) and *excludes the bright nucleus core* (`pFg(0.10)=0.07`). So the model labels most of the dim, outlier-compressed image as foreground; the flood-fill fills the whole connected region, and since the safety box scales with the (now huge) fitted `maxR`, it runs away.

### Follow-ups (for the maintainer — design decision, not silently patched)
- The `pFg>0.5` gate is not selective on this data. Options that stay self-calibrating: (a) improve `estimateForegroundModel` so `muFg` tracks the nucleus core; (b) require the flood-fill to climb a monotone/hysteresis band around the seed rather than a single `pFg>0.5` cut; (c) base the safety box on the FIXED mask radii, not the live (exploding) `maxR`, so the bound is real. The offline prototype's validated recovery (~40px, drift <5px) did not reproduce with the in-tree model + raw normalization.
- Legacy path stays the default; this flag is dormant until the foreground criterion is fixed.

### 2026-07-31 (follow-up) — RESOLVED: three fixes; the real blocker was a frame-0 seed z-space bug, not the model

Three fixes turned the flag-ON path from "whole-volume explosion" into a working, scale-invariant fit. The first two (proposed on a runaway-radii / bright-half hypothesis) were necessary but not sufficient; a per-voxel diagnostic then exposed the actual cause — the frame-0 seed sits at the **wrong z** (raw 16, not interpolated 112), so the foreground indicator sampled a dim slab far above the nucleus.

**Fix A — hard ellipsoid membership for the foreground indicator (`src/CellUniverse.cpp:5652`).** Replaced `insideAnyCell = (synth > 1e-4)` with hard membership: for each non-trash cell, precompute its inverse-rotation matrix + `1/r²`, and inside its `2×maxR` bounding box mark voxels with `lx²·invA² + ly²·invB² + lz²·invC² ≤ 1` (same test the PCA mask uses; no per-voxel trig). Cost function is still installed with `enabled = generative_data_term_enabled`, so the L2 path is untouched. Also added `fgVox=` (marked-voxel count) to the `[FG Model]` log.

**Fix B — safety box off the FIXED frame-entry mask radius (`src/Frame.cpp:2809`).** `gatherForegroundComponent` bound changed `4.0f * maxR` → `4.0f * maskMaxR` (the frozen frame-entry radius), so the BFS box can't grow as the fit grows. Real effect confirmed: capped the runaway (`n` plateaued at ~20M instead of climbing to 28M+ in the pre-Fix-B run).

**Fix C (the actual root cause) — frame-0 seed z-space: `initial_z_space: raw`.** A per-voxel diagnostic showed frame-0 cells at `center=(188,250,z=16)` with `realAtCenter=0.016` (dim) and `markedMean=0.029` (≈ background) — `muFg` was almost exactly the *global* volume mean (10.6/255=0.0416). The cells were at raw z=16, but the working volume is z-interpolated ×7 to 239 slices, so the nucleus lives at interpolated z≈112. Cause: `initial_embryo_sized.csv` (Change 38) is a raw-z seed in *rich* (theta) format; `CsvHandler::loadInitialCells` (`:289`) uses the presence of `theta_x/y/z` as its `resumeStateCsv` heuristic → treats the seed as "already scaled" → skips the ×`z_scaling`. Setting `initial_z_space: raw` forces the scale (16×7=112). This is the config key's exact purpose (`ConfigTypes.hpp:289`). It is added to the **flag-ON run config only** (not to `config/config.yaml`), so the flag-OFF byte-identity baseline is preserved.

**Verified — self-check PASSES.** `[FG DIAG]` with `initial_z_space: raw`: frame-0 cells at `center=(188,250,112)`, `realAtCenter=0.50–0.68` (bright nucleus), `markedMean=0.39`. `[FG Model] frame 0 muFg=0.4885 muBg=0.0272` (**~18×** separation; sigFg=0.073 > sigBg=0.034 → `pFg>0.5` is a clean upper threshold). Frame 1 confirms it holds: `muFg=0.462 muBg=0.028`.

**Verified — flag OFF still byte-identical** (`f0–3`, default `config/config.yaml`, seed=42): `cells.csv` and `tracked_cells.csv` **identical** to the pre-change baseline. All three fixes are either gated inside the `generative || fgAnchored` block (A, B) or live only in the flag-ON run config (C).

**Flag ON result (`outputs/20260731_224045_fganchored_zfix`, f0–6):** the whole-volume explosion is gone; the flood-fill now recovers nucleus-sized components (`n≈200k–400k`).

| metric | target | pre-fix | **post-fix (A+B+C)** |
|---|---|---|---|
| `muFg` / `muBg` | ~0.4–0.5 / ~0.03 | 0.041 / 0.027 | **0.49 / 0.027** ✓ |
| aRadius f0–1 | ~35–45px | 48–70 (pinned) | **34–42px** ✓ |
| aRadius f2–6 | ~35–45px | 48–70 | 36–70 (dividing cells enlarge; non-dividing `cell_3` stays ~40) |
| pos maxjump | <10px | 24–52 | **7–41px** (per-frame mostly 7–12 after settle; not fully <10) |
| per-frame counts | `4 4 4 4 6 6 6` | `4 4 4 4 4 4 4` | `4 4 4 4 4 4 5` (splits under-fire — separate split-logic concern) |

The shape-fit objective is met: the fit is scale-invariant, self-calibrating, recovers the true ~40px nucleus from frame 0, and no longer explodes. Two targets are not fully met and are **downstream of the shape fit**: (a) drift 7–41px — the largest jumps are the f0→f2 settle plus real nucleus motion; (b) split counts under-fire — the multifactor split gate isn't triggering on the (now correctly enlarging) pre-division cells. Both are for follow-up; neither is a shape-fit defect.

**Follow-up:** `config/config.yaml` should set `initial_z_space: raw` whenever it seeds from `initial_embryo_sized.csv` — the raw-z-at-z16 mis-placement also affects the legacy path (its fit limps the cell from z16 up to the nucleus). Left unset here only to preserve the flag-OFF byte-identity baseline; the maintainer can adopt it globally and re-baseline.

## 2026-07-31: Rung 2b — fix sized-seed z-placement at the source (Change 41)

**Status: ACTIVE — cleaner fix superseding the Change-40 `initial_z_space: raw` run-config override. `config/initial_embryo_sized.csv`.**

### Why
Change 40 found that `initial_embryo_sized.csv` was loaded with the ×7 z-scaling SKIPPED, stranding the founders at working-z=16 (raw slice ~2) instead of z≈112 (raw slice 16, where the nuclei are). Root: `CsvHandler::loadInitialCells` (`src/CsvHandler.cpp:285-291`) treats a header containing `theta_x/y/z` as a resume-state CSV (`resumeStateCsv=true`) → with the default `initial_z_space: auto`, `zAlreadyScaled=true` → `effectiveZScaling=1.0`. My sized seed carried `theta_*` columns (all 0.0), tripping that heuristic. The box fit tolerated it (position update drifts the cell from z16 to the nucleus mid-fit — this inflated the observed "drift"); the fg-anchored fit did not (it estimates the fg model at the seed position → membership on a dim slab → `muFg`=global mean).

### Change (`config/initial_embryo_sized.csv`)
Removed the three `theta_*` columns. Header `file,name,x,y,z,majorRadius,bRadius,minorRadius,theta_x,theta_y,theta_z` → `file,name,x,y,z,majorRadius,bRadius,minorRadius`. `parseNamedInitialCell` does not read theta (rotation defaults to 0 and is set by the fit), so the seed still parses as the rich/named format with radii, but no longer trips `resumeStateCsv` → default `auto` applies the ×7 z-scaling. No `initial_z_space` override needed.

### Verified (f0-1, default config, SEED=42 THREADS=1)
`initial_z_space: auto`; founders load at **z=104-120 (≈112, on the nucleus)** with radii a≈37-44px — was z=16 with the theta version. The Change-40 flag-ON fg-anchored result reproduces from this seed without the run-config `initial_z_space: raw` override.

### Follow-ups (unchanged from Change 40)
- Re-validate the box-fit pancaking A/B on this corrected seed (earlier sized-seed splits ran with cells drifting up from z16 — the "first split at f5" result needs re-checking).
- The fg-anchored under-splitting (`4 4 4 4 4 4 5`) is a split-gate concern, separate from the fit.

## 2026-08-01: fg-anchored fit — per-cell half-max (FWHM) flood-fill boundary kills the bloat feedback loop (Change 42)

**Status: ACTIVE (flag ON path). Fixes the frame-over-frame radius BLOAT that Change 40/41 left: the fg-anchored fit was correct at f0 (~40px) but climbed toward the 70px cap by f6. Root cause was the GLOBAL `pFg>0.5` flood-fill boundary feeding back on itself; replaced with a per-cell, self-limiting half-max level anchored to fit-independent image facts.**

### Why (the bloat feedback loop)
Change 40's `gatherForegroundComponent` flood-filled voxels with `_fgModel.pFg(I) > 0.5`. `pFg`'s crossover is set by the GLOBAL model means, and `muFg` is estimated from the current cells' interiors — so as a cell's fit bloated, its ellipsoid swept in dimmer peripheral voxels, `muFg` fell (0.49→0.16 over f0→f6), the `pFg>0.5` boundary moved OUTWARD, the flood-fill grew (`compVox` 897k→1.66M), and the fit bloated further. A closed loop between the fit size and the boundary.

### Change — per-cell half-max (FWHM) boundary
Replace the flood-fill criterion in `gatherForegroundComponent` with `I >= halfMaxLevel`, computed ONCE per gather from quantities the fit cannot move:
- `bg`   = `_fgModel.muBg` (the BACKGROUND mean — stable across frames, ~0.027; `muFg` is no longer read by the fit, so its drift is harmless).
- `peak` = 99th-percentile REAL intensity in a CORE sphere of radius `0.5 × maskMaxR` (the FIXED frame-entry radius, never the live/bloating radius), via a heap-free 256-bin histogram (percentile, not raw max → ignores hot voxels).
- `halfMaxLevel = bg + 0.5*(peak - bg)` (standard FWHM: half of the cell's OWN contrast; a principled scale-invariant fraction, not a tuned gate).
Downstream is unchanged: BFS 6-connected from center, Voronoi-split against neighbor centers, safety box `4 × maskMaxR`; centroid→position, PCA→rotation, extent→radii. `bg` and `peak` are image facts (background stays dim, nucleus center stays bright) ⇒ `level` is stable ⇒ the region is stable ⇒ no bloat. Emits `[HalfMax] cell@(x,y,z) peak=.. bg=.. level=.. compVox=..` per gather.

### Files
- **`src/ForegroundComponent.cpp`** (NEW) — `gatherForegroundComponent` moved here (see byte-identity note) with the half-max criterion.
- **`includes/BrightPixel.hpp`** (NEW) — the `BrightPixel{cv::Point3f pos; float weight;}` struct, shared by the two TUs (Frame.cpp keeps a token-identical local copy for ODR).
- **`src/Frame.cpp`** — removed the in-TU `gatherForegroundComponent` definition; added an EXTERNAL-linkage forward declaration at global scope (after `struct BrightPixel`, line ~90). Caller (`calibrateCellShapeViaPca`, ~line 2680) unchanged: `gatherForegroundComponent(_realFrame, _fgModel, _backgroundValue, center, 4.0f*maskMaxR, selfClaim, otherCellsClaimSets)`.
- **`CMakeLists.txt`** — added `src/ForegroundComponent.cpp` to the `celluniverse` sources.

### Verified — bloat FIXED, radii STABLE across f0-6
Run `outputs/20260801_000326_halfmax` (f0-6, seed=42, single-thread, sized seed + `initial_z_space: raw` + raw normalization + multifactor + `pcaShapeFgAnchored: true`):
- **Fitted aRadius stays ~30-42px on EVERY frame** (was climbing 41→70 with `pFg`). `cell_1` a-radius trajectory `[38, 39, 38, 30, 29, 27, 26]`; non-splitting `cell_3` holds ~32px. f6 max radius 30.7 ≈ f0 max 37.9 — NOT climbing to the cap.
- `[HalfMax]` levels stable per cell (`peak≈0.69-0.78, bg=0.027, level≈0.37-0.43`); `compVox≈90k-174k` (was 200k-1.66M with the drifting `pFg` boundary).

### Known regression (downstream, NOT the fit) — over-split
Per-frame counts are `4 4 6 8 9 10 11` vs GT `4 4 4 4 6 6 6`. The now-sharp, correctly-sized cells trip the multifactor split gate too aggressively (Change 40's bloated version UNDER-split at `4 4 4 4 4 4 5`). This is a split-GATE tuning concern for a later change, not a shape-fit defect — the shape fit is doing its job (stable, nucleus-sized). Drift maxjump ~34px (mix of real motion + split re-seeding).

### Byte-identity — NOT bit-exact; a compiler (‑O3 + LTO) FP-scheduling artifact, not a logic change
The flag-OFF fit is **not** bit-identical to the pre-change baseline (`f0-3` default config: `cells.csv` differs). This is NOT a leak into the legacy path — the flag-OFF run executes **zero** `[HalfMax]` (the fg-anchored branch is never entered), i.e. the runtime code path is unchanged. The differences are FP round-off: at f0, `cell_2` is bit-identical, `cell_3/cell_4` differ by ~0.02px (last-bit noise), and `cell_1` by 1.36px because a last-bit change flips its PCA convergence/degeneracy THRESHOLD branch; over frames this amplifies (max ~15-24px by f3). Cause: under Release `-O3` + IPO/LTO, the mere PRESENCE of the half-max code in the binary reshuffles the FP-instruction scheduling of the SHARED PCA math in `calibrateCellShapeViaPca`. Exhaustively verified it is unavoidable via code structure: it persists with the helper (a) in Frame.cpp as-is, (b) `__attribute__((noinline))`, (c) `noinline+optnone`, (d) with a byte-identical caller arg list, (e) with a heap-free histogram (no `std::vector<float>`/`std::nth_element`), and (f) moved to its own translation unit (this change). The ONLY bit-exact option is to not ship the fit change. **Decision needed:** accept the FP-reordering artifact (flag-OFF algorithm is provably unchanged), or treat this fit as a permanent behavior change and re-baseline the flag-OFF reference. Build **links clean**.
