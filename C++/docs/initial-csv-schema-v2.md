# Initial CSV schema v2

CellUniverse accepts the initializer's 21-column schema-v2 CSV in addition to
all legacy initial-cell formats. Schema-v2 metadata is authoritative only when
the metadata columns are present. A CSV without those columns follows the
existing YAML/configuration behavior unchanged.

## Runtime precedence

For schema v2:

1. `zInterpolationRatio` overrides `simulation.z_scaling` before image
   discovery and interpolation.
2. `zCoordinateSpace` overrides `simulation.initial_z_space`.
3. Per-cell `theta_x`, `theta_y`, `theta_z`, and `brightness` initialize each
   ellipsoid. Blank optional values retain the old zero-rotation/configured
   brightness fallback.
4. The one row with `isHotBackgroundRegion=1` is background metadata and is
   never created as a cell.
5. That row's cold/hot levels, rotated ellipsoid, and soft margin initialize a
   per-frame spatial background field. The cold value is also installed as the
   scalar fallback for code that has no voxel location.

The current interpolation implementation requires a positive integer ratio.
Schema-v2 input with a fractional ratio fails with a clear error rather than
being silently truncated.

The reader rejects missing or duplicate required columns, conflicting
row-level metadata, unsupported/future schema versions, invalid radii or
brightness, more than one background row, `hot < cold`, and soft margins
outside `[0, 1)`.

## Two-region background

The initializer and C++ runtime use the same field:

```text
local = transpose(Rz * Ry * Rx) * (world - center)
radial = length(local / radii)
u = clamp((1 + softMargin - radial) / (2 * softMargin), 0, 1)
membership = u^2 * (3 - 2u)
background = cold + membership * (hot - cold)
```

`softMargin=0` is a hard boundary. Otherwise, `radial=1` has membership `0.5`.

The manually confirmed first-frame envelope is retained exactly. On later
frames, `BackgroundRegionTracker` keeps rotation fixed and proposes only small
center/radius changes from oriented boundary evidence in a narrow shell. It:

- excludes current cell ellipsoids from shape and intensity samples;
- uses robust fitting, EMA smoothing, per-frame movement caps, face coverage,
  and a confidence gate;
- freezes the previous geometry when evidence is sparse, flat, inconsistent,
  or implausibly large;
- robustly refreshes cold/hot levels from high-confidence interior/exterior
  samples, retaining current/seed values when those estimates are unsafe.

The resulting per-voxel field is installed before signal-center localization,
PCA fitting, weighted-center pulls, synthesis, and CellUniverse3 window
guidance. It supersedes the legacy dataset-wide maximum-intensity hot-region
map for runs that contain schema-v2 envelope metadata. Runs without that
metadata retain the existing scalar or configured dual-background logic.

Each processed frame appends an inspectable state row to
`initial_csv_background_states.csv` in the output directory. The row records
the accepted/frozen shape, levels, confidence, and evidence counts.

## Verification

Build and run the focused test:

```bash
cmake --build build_webui_local --target initial_csv_background_test -j 8
ctest --test-dir build_webui_local -R '^initial_csv_background_test$' \
  --output-on-failure
```

The test covers schema-v2 parsing, envelope exclusion, rotations/brightness,
legacy fallback, future-version rejection, exact rotated smoothstep sampling,
low-confidence shape freezing, and acceptance of a small high-confidence
per-frame shape shift while rotation remains fixed. The same executable can
validate generated CSV files by passing them as positional arguments.

`brightness_volume_analyzer` has its own historical CSV reader and is not part
of this initial-cell ingestion path.
