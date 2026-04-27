# Ground-Truth trajectory — frames 1 → 65 (combined runs)

**Last updated:** 2026-04-23

This is the stitched ground-truth reference trajectory used to score ablation
runs through f65. Each segment is sourced from the highest-quality run available
for that frame range.

## Source mapping

| Frames | Source run | Status | Notes |
|---|---|---|---|
| f1–f45 | `output_jihang_20260419_211928_perfect_45` | present | "perfect_45" reference. End state: 26 cells at f45. Matches ground-truth splits memory (e9077, 12345, 1f89abf, 1f2ed10d lineages through f39 + e9077:10 split at f40). |
| **f46–f59** | — (was `output_jihang_20260422_004745`) | **DELETED 2026-04-23** | Source directory was removed between earlier audits and now. The f59 checkpoint (still referenced by ablation-run INIs) was created by that run; the cells.csv for f46-f59 is gone. **Need to re-run or recover from backup.** |
| f60 | f59 checkpoint only (no csv entry) | partial | Captured in `checkpoints/frame_059.txt` state — 30 cells entering f61, but not written as a cells.csv row. |
| f61–f65 | `output_bleed5_20260423_021617` | present | **GT-marked 2026-04-23**: cleanest f65 drifts of all ablation runs (all 4 splits ≤ 7.3 vx), 4 clean TPs, no e3d03 FP. |

## Per-frame GT trajectory

| Frame | Cells | Splits this frame (children names) | Parents that split | Max aR | Source |
|---|---|---|---|---|---|
| 1 | 6 | — | — | 45.5 | perfect_45 |
| 2 | 6 | — | — | 45.8 | perfect_45 |
| 3 | 8 | 12345679...0, 12345679...1, e9077677...0, e9077677...1 | 12345679, e9077677 | 47.7 | perfect_45 |
| 4 | 8 | — | — | 49.3 | perfect_45 |
| 5 | 8 | — | — | 49.7 | perfect_45 |
| 6 | 8 | — | — | 52.8 | perfect_45 |
| 7 | 8 | — | — | 52.8 | perfect_45 |
| 8 | 9 | 1f89abf4...0, 1f89abf4...1 | 1f89abf4 | 52.8 | perfect_45 |
| 9 | 9 | — | — | 51.7 | perfect_45 |
| 10 | 9 | — | — | 46.5 | perfect_45 |
| 11 | 10 | 1f2ed10d...0, 1f2ed10d...1 | 1f2ed10d | 37.8 | perfect_45 |
| 12 | 10 | — | — | 39.7 | perfect_45 |
| 13 | 10 | — | — | 41.7 | perfect_45 |
| 14 | 10 | — | — | 43.8 | perfect_45 |
| 15 | 10 | — | — | 45.8 | perfect_45 |
| 16 | 10 | — | — | 46.5 | perfect_45 |
| 17 | 10 | — | — | 45.6 | perfect_45 |
| 18 | 10 | — | — | 44.4 | perfect_45 |
| 19 | 10 | — | — | 40.9 | perfect_45 |
| 20 | 14 | 12345679...00, 12345679...01, 12345679...10, 12345679...11, e9077677...00, e9077677...01, e9077677...10, e9077677...11 | 12345679...0, 12345679...1, e9077677...0, e9077677...1 | 42.9 | perfect_45 |
| 21 | 14 | — | — | 45.1 | perfect_45 |
| 22 | 14 | — | — | 45.4 | perfect_45 |
| 23 | 14 | — | — | 48.5 | perfect_45 |
| 24 | 14 | — | — | 48.9 | perfect_45 |
| 25 | 14 | — | — | 48.6 | perfect_45 |
| 26 | 14 | — | — | 47.7 | perfect_45 |
| 27 | 14 | — | — | 47.4 | perfect_45 |
| 28 | 16 | 1f89abf4...00, 1f89abf4...01, 1f89abf4...10, 1f89abf4...11 | 1f89abf4...0, 1f89abf4...1 | 48.3 | perfect_45 |
| 29 | 16 | — | — | 45.2 | perfect_45 |
| 30 | 16 | — | — | 45.1 | perfect_45 |
| 31 | 16 | — | — | 45.3 | perfect_45 |
| 32 | 17 | 1f2ed10d...10, 1f2ed10d...11 | 1f2ed10d...1 | 44.9 | perfect_45 |
| 33 | 17 | — | — | 43.9 | perfect_45 |
| 34 | 17 | — | — | 45.5 | perfect_45 |
| 35 | 17 | — | — | 45.6 | perfect_45 |
| 36 | 17 | — | — | 45.4 | perfect_45 |
| 37 | 17 | — | — | 49.9 | perfect_45 |
| 38 | 18 | 1f2ed10d...00, 1f2ed10d...01 | 1f2ed10d...0 | 52.4 | perfect_45 |
| 39 | 25 | 12345679...000, 12345679...001, 12345679...010, 12345679...011, 12345679...100, 12345679...101, 12345679...110, 12345679...111, e9077677...000, e9077677...001, e9077677...010, e9077677...011, e9077677...110, e9077677...111 | 12345679...00, 12345679...01, 12345679...10, 12345679...11, e9077677...00, e9077677...01, e9077677...11 | 48.4 | perfect_45 |
| 40 | 26 | e9077677...100, e9077677...101 | e9077677...10 | 40.3 | perfect_45 |
| 41 | 26 | — | — | 48.2 | perfect_45 |
| 42 | 26 | — | — | 51.6 | perfect_45 |
| 43 | 26 | — | — | 55.4 | perfect_45 |
| 44 | 26 | — | — | 53.9 | perfect_45 |
| 45 | 26 | — | — | 53.1 | perfect_45 |
| **46** | — | GAP — data deleted | — | — | — |
| **47** | — | GAP — data deleted | — | — | — |
| **48** | — | GAP — data deleted | — | — | — |
| **49** | — | GAP — data deleted | — | — | — |
| **50** | — | GAP — data deleted | — | — | — |
| **51** | — | GAP — data deleted | — | — | — |
| **52** | — | GAP — data deleted | — | — | — |
| **53** | — | GAP — data deleted | — | — | — |
| **54** | — | GAP — data deleted | — | — | — |
| **55** | — | GAP — data deleted | — | — | — |
| **56** | — | GAP — data deleted | — | — | — |
| **57** | — | GAP — data deleted | — | — | — |
| **58** | — | GAP — data deleted | — | — | — |
| **59** | — | GAP — data deleted | — | — | — |
| **60** | — | GAP — data deleted | — | — | — |
| 61 | 30 | 1f2ed10d...100, 1f2ed10d...101, 1f2ed10d...110, 1f2ed10d...111, 1f89abf4...000, 1f89abf4...001, 1f89abf4...010, 1f89abf4...011 | 1f2ed10d...10, 1f2ed10d...11, 1f89abf4...00, 1f89abf4...01 | 45.6 | bleed5_asym GT-marked 2026-04-23 |
| 62 | 30 | — | — | 45.3 | bleed5_asym GT-marked 2026-04-23 |
| 63 | 30 | — | — | 46.3 | bleed5_asym GT-marked 2026-04-23 |
| 64 | 30 | — | — | 46.0 | bleed5_asym GT-marked 2026-04-23 |
| 65 | 34 | e9077677...0100, e9077677...0101, e9077677...0110, e9077677...0111, e9077677...1100, e9077677...1101, e9077677...1110, e9077677...1111 | e9077677...010, e9077677...011, e9077677...110, e9077677...111 | 46.0 | bleed5_asym GT-marked 2026-04-23 |

## Split event summary (f1–f65)

**Total split events in covered ranges:** 11

- **f3** (perfect_45): 4 split(s) — parent(s) 12345679, e9077677 → daughters 12345679...0, 12345679...1, e9077677...0, e9077677...1
- **f8** (perfect_45): 2 split(s) — parent(s) 1f89abf4 → daughters 1f89abf4...0, 1f89abf4...1
- **f11** (perfect_45): 2 split(s) — parent(s) 1f2ed10d → daughters 1f2ed10d...0, 1f2ed10d...1
- **f20** (perfect_45): 8 split(s) — parent(s) 12345679...0, 12345679...1, e9077677...0, e9077677...1 → daughters 12345679...00, 12345679...01, 12345679...10, 12345679...11, e9077677...00, e9077677...01, e9077677...10, e9077677...11
- **f28** (perfect_45): 4 split(s) — parent(s) 1f89abf4...0, 1f89abf4...1 → daughters 1f89abf4...00, 1f89abf4...01, 1f89abf4...10, 1f89abf4...11
- **f32** (perfect_45): 2 split(s) — parent(s) 1f2ed10d...1 → daughters 1f2ed10d...10, 1f2ed10d...11
- **f38** (perfect_45): 2 split(s) — parent(s) 1f2ed10d...0 → daughters 1f2ed10d...00, 1f2ed10d...01
- **f39** (perfect_45): 14 split(s) — parent(s) 12345679...00, 12345679...01, 12345679...10, 12345679...11, e9077677...00, e9077677...01, e9077677...11 → daughters 12345679...000, 12345679...001, 12345679...010, 12345679...011, 12345679...100, 12345679...101, 12345679...110, 12345679...111, e9077677...000, e9077677...001, e9077677...010, e9077677...011, e9077677...110, e9077677...111
- **f40** (perfect_45): 2 split(s) — parent(s) e9077677...10 → daughters e9077677...100, e9077677...101
- **f61** (bleed5_asym GT-marked 2026-04-23): 8 split(s) — parent(s) 1f2ed10d...10, 1f2ed10d...11, 1f89abf4...00, 1f89abf4...01 → daughters 1f2ed10d...100, 1f2ed10d...101, 1f2ed10d...110, 1f2ed10d...111, 1f89abf4...000, 1f89abf4...001, 1f89abf4...010, 1f89abf4...011
- **f65** (bleed5_asym GT-marked 2026-04-23): 8 split(s) — parent(s) e9077677...010, e9077677...011, e9077677...110, e9077677...111 → daughters e9077677...0100, e9077677...0101, e9077677...0110, e9077677...0111, e9077677...1100, e9077677...1101, e9077677...1110, e9077677...1111

## Comparing other ablation runs at f60-f65

For the 60-65 interval, other runs deviate from the GT (bleed5_asym) as follows:

| Run | f65 cells | f65 split drifts (d1/d2) | Notes |
|---|---|---|---|
| **bleed5_asym (GT)** | 34 | 6.2/6.3, 7.1/7.3, 4.3/6.4, 4.2/4.3 | Cleanest drifts |
| bleed2_asym | 34 | 7.9/8.3, 7.8/8.0, 0.97/6.4, 12.9/12.8 | One borderline drift (12.9) |
| bleed2_sym | 34 | 7.3/7.7, 3.0/3.1, 2.0/5.1, 4.1/4.2 | Clean, marginally below GT |
| T1 (Fix C only) | 34 | 3.0/3.4, 6.9/7.7, 6.6/9.3, 12.4/12.3 | 1 borderline (12.4) |
| T0 (baseline) | 35 | includes e3d03 FP (drift 1.7/2.2) | Extra FP split |
| T2 (Fix A full) | — | f65 screenshots showed clumping | Stopped mid-f67 |
| T2b (Fix A 1.5r) | 34 | — | Same splits as T1/bleed but clumping visible, stopped mid-f67 |
| T3 (persistence) | 30 | all rejected (first-frame) | Deferred all f65 splits to f66 |

## Open action

1. **Recover or re-generate f46-f59 data.** No surviving run has cells.csv for this range. Options:
   a. Re-run `output_jihang_20260422_004745`-equivalent from f1 with the same config → regenerates the trajectory and the f59 checkpoint together.
   b. Check backups (iCloud, Time Machine, git LFS, etc.) for the deleted `output_jihang_20260422_004745` directory.
2. Until restored, f46-f59 state in every ablation run is inherited from the (still-on-disk) `checkpoints/frame_059.txt` held inside this GT table's source directories or ablation run dirs. That checkpoint is the only artifact of that missing trajectory.
