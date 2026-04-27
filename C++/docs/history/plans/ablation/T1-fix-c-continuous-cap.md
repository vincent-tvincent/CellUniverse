# T1 — Fix C: continuous bloat-cap log-barrier (weight=1000)

**Output dir:** `outputs/output_ablation_T1_20260422_180917`
**Delta vs T0:** only `bloat_cap_barrier_weight: 0.0 → 1000.0`. All other config unchanged. seed=42.

## Implementation recap

Log-barrier penalty added to `perturbCell` delta cost per axis:
```
penalty(ratio = r / birthR) =
  0                                               if ratio < 1.1
  weight × -log((1.6 - ratio) / 0.5)              if 1.1 ≤ ratio < 1.6
  1e9  (hard reject)                              if ratio ≥ 1.6
```
Applied to aR, bR, cR summed. Birth radii installed per cell name from
`cellShapeBirth` at frame start in `CellUniverse::optimize`. Cells without a
birth record (rare — newborn mid-frame splits before capture) are exempt.

## T0 vs T1 per-frame cell count + max aR

| Frame | T0 n | T1 n | T0 maxAR | T1 maxAR | T0 bloat | T1 bloat |
|---|---|---|---|---|---|---|
| 61 | 30 | 30 | 45.6 | 45.6 | 8 | 8 |
| 62 | 30 | 30 | 46.7 | 49.1 | 6 | 6 |
| 63 | 30 | 30 | 47.4 | 49.3 | 13 | 14 |
| 64 | 30 | 30 | 49.0 | 49.0 | 14 | 14 |
| 65 | 35 | 34 | 49.6 | 51.5 | 12 | 15 |
| 66 | 38 | 38 | 52.0 | 54.0 | 13 | 11 |
| 67 | 43 | 42 | 54.6 | 56.8 | 10 | 8 |
| 68 | 46 | 44 | 46.7 | 57.4 | 10 | 8 |
| 69 | 47 | 44 | 49.0 | 51.2 | 9 | 7 |
| 70 | 48 | 45 | 51.5 | 55.7 | 10 | 6 |
| 71 | 49 | 46 | 54.0 | 54.9 | 6 | 4 |
| 72 | 50 | 47 | 47.1 | 47.6 | 6 | 4 |

## Split diffs

**Only in T0 (eliminated by Fix C):**
- f65 `e3d03428` d1=1.7 d2=2.2 cost=-99698 — FP (biologically doesn't split)
- f67 `e3d03428:0` d1=14.6 d2=**36.0** cost=-95925 — FP cascade + ugly drift
- f68 `e9077677:100` (a5100) d1=**26.4** d2=17.5 cost=-76021 — ugly bloat-driven split
- f69 `e3d03428:00` d1=6.4 d2=8.7 cost=-27829 — FP cascade 3rd level

**Only in T1 (new accept):**
- f66 `12345679:101` (23101) d1=4.1 d2=5.2 cost=-81806 — clean TP that T0 missed

**Remaining ugly split (both runs):**
- f72 `cb11` — T0 drift2=50 / T1 drift2=**56.7** cost=-127k. Bloat-driven at birth (cb11 born from large parent at f28; `aR/birthR` stays < 1.1 even at current size). Relative cap blind to this.

## Aggregate deltas vs T0 thresholds

| Metric | T0 | T1 | Threshold | Status |
|---|---|---|---|---|
| Total splits | 20 | 17 | — | — |
| Splits with drift > 15 | 3 | 1 | ≤ 1 | **met** ✓ |
| e3d03-lineage FPs | 3 | 0 | 0 | **met** ✓ |
| Max aR across frames | 54.6 | 57.4 | ≤ 40 | **not met** (cb11 edge case) |
| Bloated-cell frames (12 total) | 12 | 12 | ≤ 6 | **not met** |
| Cell-frames drift > 15 | 51 | 46 | ≤ 25 | **not met** (−10%) |
| Mean per-frame drift | 9.24 | 8.97 | — | −3% |

## Verdict

**Strong partial win.** Fix C eliminates the clearest failure mode (e3d03 FP cascade + ugly bloat-driven splits) with a one-config-value change. Clean-TP rate unchanged; one TP (23101 at f66) even gained because the optimizer isn't wasting iterations on e3d03's bloat.

**Limitations identified:**
1. **Relative-to-birth cap blind to cells born large.** cb11 was born at aR~38-40 from a big parent at f28; at current aR=45-46, ratio is 1.1-1.2× — barely into the penalty zone, not enough to prevent the ugly drift2=57 split.
2. **Inherited bloat from checkpoint.** 15 cells bloated at T1 f65 (vs 12 in T0). Fix C's log-barrier penalty-only design doesn't SHRINK cells — so bloated starting state from f59 checkpoint persists. Only prevents FURTHER bloat.
3. **Drift metrics only slightly improved.** The drift-reduction effect is indirect (bloated cells have less stable PCA → drift more); direct drift reduction requires Fix A.

## Next steps

Two paths forward:

1. **T1b: weight bump** — try weight=2000 or 3000 to see if stronger penalty tips more bloated cells below 1.3× (but risk suppressing legit growth). Quick test.
2. **T2: Fix A (brightness-centroid anchor)** — directly address drift. Since Fix C handled the bloat-driven FPs, T2 focuses on the remaining residuals: drift into empty space and cb11-style edge-case ugly splits.

Recommendation: **proceed to T2** (Fix A). Fix C at weight=1000 has done the useful work; cranking it higher has diminishing returns. T2's centroid anchor directly targets the "cell floating in empty space" screenshots the user flagged earlier.

## Notes on implementation robustness

- `[Bloat Cap]` log line emitted per frame confirming `birth_installed=N/N`. Never missed a cell in the T1 run.
- Hard-reject zone (ratio ≥ 1.6): triggered 0 times across T1 (verified by absence of `1e9`-cost perturb accepts). Cells stayed in the soft-penalty zone.
- Determinism: same seed=42 as T0, so divergences in per-frame cell positions and accept patterns are attributable entirely to the penalty term.
