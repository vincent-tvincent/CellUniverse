# Hysteresis Binarization with Cell-Position Seeding — Implementation Plan

**Goal:** Replace the linear-clip step at the end of `applyCalibratedPreprocess` (`ImageHandler.cpp:1557-1567`) with a hysteresis binarization that uses known cell positions as seeds. This preserves dim cells and dim chromatin gaps inside cells — both of which the current clip-to-[0,1] step destroys when raw intensity sits at or below the bg anchor.

**Branch:** current (auto_calibrate path). No branch switch needed.

**Why now:** Today's contrast-stretch experiments confirmed that pure intensity-based normalization can't both (a) suppress bg noise and (b) preserve dim cell structure. The right fix is to commit to binarization (per the original CellUniverse 2D paper) and use the cell positions we already maintain to anchor preservation of dim cells.

**Active changelog:** `C++/docs/changelogs/changelogv10.md` (binarization work = Change 4).

---

## Diagnosis (the failure modes we're fixing)

The current `applyCalibratedPreprocess` does:

```
output = clip( (raw - bg_anchor) / (cell_anchor - bg_anchor),  0,  1 )
```

Three failure modes for dim cells:

1. **Cell content removed.** `bg_anchor = P10(bg_pixels)` is well below `bg_mean`. Internal chromatin gaps in cells (raw values near `bg_mean`, > P10) end up below `bg_anchor` after remap → clip to 0 → cells look hollow.

2. **Dim cells lose structure.** A cell whose typical core is below `cell_anchor` (which is the median of cell pixels — half the cells are darker than this) normalizes to < 1 with internal variation collapsed.

3. **Whole cell disappears.** If a cell's brightest pixel is ≤ `bg_anchor`, every pixel clips to 0. The cell vanishes from the binary mask. Linear remap with clipping has no mechanism to preserve a region that doesn't beat the global threshold.

Hysteresis thresholding plus position-aware seeding solves all three.

---

## Algorithm

**Inputs:** raw frame after photobleach-ratio adjustment (already computed); `bg_anchor`, `cell_anchor` from auto_calibrate; `cells` from CSV/snapshot with `(position, rotation, radii)`.

**Compute thresholds (relative to anchor span):**
```
span    = cell_anchor - bg_anchor
T_high  = bg_anchor + binarize_high_fraction × span     # default 0.50
T_low   = bg_anchor + binarize_low_fraction  × span     # default 0.10
```

**Step A — high mask:** `M_high(p) = (raw(p) >= T_high)`. Definite cell pixels.

**Step B — low mask:** `M_low(p) = (raw(p) >= T_low)`. Could-be-cell pixels (bg-noise floor cleared).

**Step C — cell-position seeding (the dim-cell preservation mechanism):**
```
For each non-trash cell in cells:
  seed_radii = cell.radii × binarize_seed_scale   # default 0.5
  For each voxel inside scaled-ellipsoid(cell.position, cell.rotation, seed_radii):
    if M_low[voxel] == 1:                         # gating: only seed where signal exists
      M_high[voxel] = 1                           # promote to definite cell
```

Reuses `pointInsideScaledEllipsoid` already in CellUniverse.cpp.

**Step D — morphological reconstruction (geodesic dilation):**
```
binary = M_high.clone()
for iter in 0..max_iters:
    next = dilate3D(binary, 3x3x3 26-connectivity kernel)
    next = next AND M_low                         # constrain to eligibility
    if next == binary: break                      # converged
    binary = next
return binary
```

Output: `binary` is 0/1 per voxel. Replaces the clipped grayscale output.

---

## Why this fixes each failure mode

| Failure | How the fix handles it |
|---|---|
| Cell content removed (chromatin gaps) | Gap voxels are above `T_low`, connected to cell core (above `T_high`) → reconstruction preserves them as 1 |
| Dim cell with at least some bright pixels | Bright pixels seed `M_high` directly; the rest (above `T_low`, connected) get pulled in by reconstruction |
| Uniformly-dim cell, no `T_high` pixels | Cell-position seeding plants 12.5% of cell volume into `M_high` (gated by `T_low`) → reconstruction expands from there |
| Cell drifted ≤ ~10 voxels between frames | Seed is at OLD position, but cell's `M_low` region is connected to seed via halo → reconstruction follows halo to new position |
| Cell drifted hugely (broke connection) | No expansion. Cell drops from binary mask. Snapshot still exists; trash removal catches it after a few frames |
| Brand-new cell (not in any snapshot) | Standard hysteresis preserves it via `T_high` if it has bright pixels (same as old behavior) |
| Touching cells | Both seeded individually, but their `M_low` regions connect → unified blob. Standard split detection finds the geometric bridge. |

---

## Files Touched

| File | Role | Change kind |
|---|---|---|
| `C++/includes/ConfigTypes.hpp` | Add `binarize_*` config fields, YAML parse, log dump. | Modify |
| `C++/includes/ImageHandler.hpp` | Update `applyCalibratedPreprocess` signature to take `cells` (or expose binarize as a separate function). | Modify |
| `C++/src/ImageHandler.cpp` | Implement `hysteresisBinarizeWithSeeds`, replace clip step in `applyCalibratedPreprocess`. | Modify |
| `C++/src/CellUniverse.cpp` | Pass cell positions through to `applyCalibratedPreprocess` (frame 0: from initial CSV; frame N≥1: from `previousSnapshots`). | Modify |
| `C++/config/config.yaml` | Add new fields with documented defaults. | Modify |
| `C++/docs/changelogs/changelogv10.md` | Append Change 4 entries (one per phase). | Modify |

---

## Phase Breakdown

- **Phase 1 — Foundation:** Config fields, helper declaration, defaults. No behavior change unless `binarize_enabled = true`.
- **Phase 2 — Implementation:** `hysteresisBinarizeWithSeeds` function; wire into `applyCalibratedPreprocess`; plumb cell positions.
- **Phase 3 — Validation:** Build, run preprocess-only with image export on both fluo f0-5 and original_data f1-5, inspect, iterate parameters.
- **Phase 4 — Full-pipeline validation:** With binarize on, run full pipeline f0-5 on both datasets; verify splits behave; iterate cost/threshold tuning if needed.
- **Phase 5 — Cleanup:** Default-off bandaids that become irrelevant under binary input (brightness EMA, anchor-based gather thresholds, etc.).

---

## Phase 1 — Foundation

### Task 1: Add `binarize_*` config fields

**File:** `C++/includes/ConfigTypes.hpp`

After the auto_calibrate block:

```cpp
// Hysteresis binarization with cell-position seeding (v10 Change 4,
// 2026-05-08). Replaces the linear clip-to-[0,1] step in
// applyCalibratedPreprocess. Output is a binary mask (0 = bg, 1 = cell)
// preserving dim cells and dim chromatin gaps that the linear clip
// destroys. See docs/plans/2026-05-08-binarize-with-cell-seeding.md.
bool  binarize_enabled = false;            // off by default until Phase 4 validation
float binarize_high_fraction = 0.50f;      // T_high = bg_anchor + 0.50 × (cell - bg)
float binarize_low_fraction  = 0.10f;      // T_low  = bg_anchor + 0.10 × (cell - bg)
float binarize_seed_scale    = 0.5f;       // inner-ellipsoid radius factor for seed
bool  binarize_seed_skip_trash = true;     // exclude trash cells from seeding
int   binarize_max_iters = 50;             // dilation iterations (typically 10-20 needed)
```

YAML parse + printConfig dump entries follow existing patterns (see `auto_calibrate_*` parses around ConfigTypes.hpp:481-484).

### Task 2: Declare `hysteresisBinarizeWithSeeds`

**File:** `C++/includes/ImageHandler.hpp`

Add static method declaration:

```cpp
// Hysteresis binarize with cell-position seeding. Returns a binary stack
// (0/1 float values) of the same shape as `input`. Modes:
//   1. Pixels >= t_high → 1 (definite cell).
//   2. Inner-ellipsoid voxels at known cell positions, where input >= t_low,
//      → 1 (preserves dim cells).
//   3. Iterative geodesic dilation: pixels >= t_low connected to a 1 → 1.
//   4. Everything else → 0.
static std::vector<cv::Mat> hysteresisBinarizeWithSeeds(
    const std::vector<cv::Mat> &input,
    float t_low,
    float t_high,
    const std::vector<Ellipsoid> &cells,
    float seed_scale,
    bool skip_trash,
    int max_iters,
    float z_scaling,
    std::ostream *logSink = nullptr);
```

### Task 3: Update applyCalibratedPreprocess signature

**File:** `C++/includes/ImageHandler.hpp`, `C++/src/ImageHandler.cpp`

Add `const std::vector<Ellipsoid> &cells` parameter, defaulting to empty for backward compatibility (legacy callers without cell list still work, just no seeding).

Update all call sites in `CellUniverse.cpp` (3 places: `loadFrame`, `prepareFrame`, the eager-load path) to pass `cells` (the live cell list, which is the initial CSV at frame 0 and the post-optimization snapshot at frame N≥1).

### Acceptance for Phase 1

- Build green; existing tests pass.
- `cost_metric=l2_intensity` and `binarize_enabled=false` (default) → byte-identical output to current behavior.

---

## Phase 2 — Implementation

### Task 4: Implement `hysteresisBinarizeWithSeeds`

**File:** `C++/src/ImageHandler.cpp`

Location: alongside other static helpers in the anonymous namespace, OR as a public `ImageHandler::` static method.

Pseudocode:
```cpp
std::vector<cv::Mat> hysteresisBinarizeWithSeeds(
    const std::vector<cv::Mat> &input,
    float t_low, float t_high,
    const std::vector<Ellipsoid> &cells,
    float seed_scale, bool skip_trash, int max_iters,
    float z_scaling,
    std::ostream *logSink)
{
    // Allocate M_high, M_low (uint8 binary, same shape as input).
    // Step A & B: per-voxel threshold.
    // Step C: for each cell, mark inner-ellipsoid voxels in M_high if M_low is 1.
    // Step D: iterative dilate3D + AND with M_low until convergence.
    // Convert binary uint8 → float (0.0 / 1.0) for downstream cost.
    // Log iterations + final pixel count + reconstruction stats.
}
```

`dilate3D` helper: 3D 26-connectivity dilation, OpenMP-parallelized over z.

### Task 5: Wire into `applyCalibratedPreprocess`

**File:** `C++/src/ImageHandler.cpp` `applyCalibratedPreprocess`

Replace the clip step (lines 1557-1567) when `binarize_enabled = true`:

```cpp
std::vector<cv::Mat> linearScaled;  // grayscale linear remap (not clipped)
// ... compute linearScaled by anchor remap, NO clipping ...

if (config.simulation.binarize_enabled) {
    const float t_low_raw  = bg_n + config.simulation.binarize_low_fraction  * (cell_n - bg_n);
    const float t_high_raw = bg_n + config.simulation.binarize_high_fraction * (cell_n - bg_n);
    // Note: thresholds are in RAW intensity units (post-photobleach ratio),
    // applied to the input before the anchor remap. So we binarize the
    // anchor-mapped scale: t_low_mapped = (t_low_raw - bg_n) / span_n
    //                                   = binarize_low_fraction.
    // i.e., on the [0, 1] anchor-mapped scale, t_low/t_high are just the
    // configured fractions.
    linearScaled = hysteresisBinarizeWithSeeds(
        linearScaled,
        config.simulation.binarize_low_fraction,
        config.simulation.binarize_high_fraction,
        cells,
        config.simulation.binarize_seed_scale,
        config.simulation.binarize_seed_skip_trash,
        config.simulation.binarize_max_iters,
        config.simulation.z_scaling,
        &log);
} else {
    // Default: clip to [0, 1] (existing behavior).
    for (auto &slice : linearScaled) {
        cv::min(slice, 1.0f, slice);
        cv::max(slice, 0.0f, slice);
    }
}
```

### Task 6: Plumb cell positions in CellUniverse.cpp

**File:** `C++/src/CellUniverse.cpp`

Three call sites to update (around lines 1907, 1973, 2402). Each gets the current cell list passed through to `ImageHandler::preprocessLoadedFrame` → `applyCalibratedPreprocess`. The cells are already in scope at those sites.

### Acceptance for Phase 2

- Build green, all tests pass.
- With `binarize_enabled = false` (default): output unchanged.
- With `binarize_enabled = true`: output is 0/1 binary stack.
- Smoke run completes f0-5 on fluo and on original_data without crashing.

---

## Phase 3 — Visual validation (preprocess-only)

### Task 7: Run both datasets, inspect

```yaml
# config.yaml
binarize_enabled: true
quit_after_preprocessing: true
export_preprocessed_images: true
```

```bash
scripts/run_celluniverse.sh config/user_input_config_embryo.ini embryo  # f0-5
scripts/run_celluniverse.sh config/user_input_configurations.ini jihang  # f1-5
```

Open `outputs/output_*/preprocessed/*.tif` in Fiji/Napari.

**Acceptance criteria:**

- Bg pixels: 0 (true black).
- Cell cores: 1 (white).
- Cell halo / dim chromatin gaps inside cells: 1 (preserved by reconstruction).
- All initial-CSV cells visible as binary blobs (no disappearance).
- Trash cells: visible if they have above-T_high pixels; absent if uniformly dim AND not seeded.

If any cell disappears that shouldn't:
- Try `binarize_seed_scale = 0.7` (bigger seed)
- Try `binarize_low_fraction = 0.05` (looser low threshold)

If too much bg noise leaks in as binary 1:
- Try `binarize_high_fraction = 0.6` (tighter high threshold)
- Try `binarize_low_fraction = 0.2` (tighter low threshold)

---

## Phase 4 — Full-pipeline validation

### Task 8: Full pipeline f0-5 both datasets

```yaml
binarize_enabled: true
quit_after_preprocessing: false
export_preprocessed_images: false
```

**Acceptance:**

- Fluo f0-5: 2 splits at f4 (cells 1, 256) within ±1 frame.
- Original_data f1-5: 2 splits at f3 (cells e9077, 12345) within ±1 frame.
- No crashes, no `[NCC Fallback]` storm.
- Frame timing within 2× legacy (binarize adds reconstruction work but skips grayscale-cost machinery).

If splits don't fire:
- Check if synth render needs adjustment (binary input vs Gaussian halo synth = mismatched scale)
- Likely need to flip `synth_render_gaussian_enabled = false` for the binary path (use hard-mask synth)
- Or stay on grayscale synth + use NCC cost (which is scale-invariant by construction)

### Task 9: Validate at scale

If Phase 4 passes, run fluo 0-20 with binarize on. Acceptance: 8/8 GT splits, cell counts match.

---

## Phase 5 — Cleanup (after Phase 4 passes)

The following knobs become irrelevant under binary input. Default-off (don't delete; keep parsing for backward compat):

- `brightnessUpdateBlend`, `brightnessMeanAmplification` — synth brightness no longer matters
- `auto_derive_background_from_frame_zero`, `auto_derive_background_stddev_multiplier`, `auto_derive_halo_cutoff_fraction` — local stats become trivial on binary
- `pca_shape_bg_floor` — gather cutoff = 0.5 on binary
- `bio_bridge_min_edge_brightness_absolute`, `pca_bridge_black_threshold` — geometric on binary
- `synth_render_gaussian_*` — hard-mask synth is the natural pair for binary real

Each toggle gets its own validation pass before flipping.

---

## Risks / things to verify

- **Reconstruction speed.** 3D dilation over 49M voxels × 10-20 iters = potentially expensive. Need to benchmark; if > 5s/frame, switch to a scanline-based connected-components implementation.
- **Cell drift between frames.** With `seed_scale = 0.5`, the seed covers ~50% of the cell width. A cell that drifted by more than 25% of its width could break the connection. Stage 0a position refinement should catch this in the next frame, but worth monitoring.
- **Seed bleed across touching cells.** If two cells' inner ellipsoids overlap (rare but possible after a split before refit), the seed unifies them. Same as today's connected-blob behavior; not a new problem.
- **Binary vs grayscale synth mismatch.** Today's render path produces grayscale synth (via per-cell `_brightness`). Comparing binary real to grayscale synth via L2 may behave oddly. Two options:
  1. Switch synth to hard-mask render when `binarize_enabled = true`.
  2. Keep grayscale synth and use NCC cost (already implemented and validated).
  Recommend option 1 for the cleanest binary-vs-binary cost path. Option 2 as fallback.
- **Bridge gate on binary.** Today's bridge looks at intensity along d1→d2 axis with `brightness ≤ blackThreshold` voxel binning. On binary input, this becomes "count 0-voxels along axis" which works trivially — but check the threshold semantics still make sense.

---

## Rollback

Single config flag flip: `binarize_enabled: false` reverts to existing clip behavior. No code is removed in Phases 1-4. Phase 5 cleanup only changes defaults.

---

## Success criteria

This plan is complete when:

1. Both fluo f0-5 and original_data f1-5 run cleanly with `binarize_enabled = true`.
2. Visual inspection of preprocessed output shows: bg = black, cells = white, dim cells preserved, internal chromatin gaps preserved.
3. Full pipeline detects expected splits within ±1 frame on both datasets.
4. Frame timing within 2× legacy.
5. ≥ 5 absolute-brightness knobs default-off in Phase 5 without regression.
