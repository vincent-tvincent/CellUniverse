# CellUniverse compact export v1

Compact export stores the state needed to reconstruct the synthetic frame
without writing a real or synthetic PNG/TIFF stack for every analyzed frame.
The source microscopy TIFF remains the source of the real-frame display.

## Configuration

Set `simulation.export_mode` in `config.yaml`:

- `full` (default) preserves the existing PNG/TIFF export behavior.
- `compact` writes compact metadata and suppresses full real/synthetic frame
  image export.
- `both` writes both representations.

`simulation.export_frame_png` and `simulation.export_frame_tiff` continue to
select formats when `export_mode` is `full` or `both`. They do not cause frame
images to be written in `compact` mode. `cells.csv`, checkpoints, and
non-frame diagnostic artifacts retain their existing behavior.

The standalone `--cell-lumen` path follows the same mode. In compact mode it
writes the compact frame and does not write its `_real.tif` or `_synth.tif`
preview stacks.

## Directory layout

```text
<job-output>/
  compact/
    manifest.json
    frames/
      frame_000000.json
      frame_000001.json
    masks/
      mask_<content-hash>.cubm
```

Frame numbers are zero-padded to at least six digits. Binary masks are
content-addressed and reused across frames with the same dimensions and mask
bits.

Each completed file is first written beside its destination with a `.tmp`
suffix and then renamed into place. The frame record is published before the
manifest is refreshed, so readers should use the manifest as the set of
complete, discoverable frames.

A fresh process session starts a fresh manifest even when its output directory
already contains old frame JSON. This prevents a shorter rerun from advertising
stale frames. An in-place checkpoint resume preserves only completed records
advertised by the prior manifest whose absolute frame number is less than
`resume_from`; stale records at or after the resume point, and JSON omitted by
the prior manifest, are not advertised. Stale, unadvertised JSON may remain on
disk and can be removed separately after confirming it is no longer needed.

## Manifest

`compact/manifest.json` has schema `celluniverse.compact.manifest`, version
`1`, and this shape:

```json
{
  "schema": "celluniverse.compact.manifest",
  "version": 1,
  "frame_schema": "celluniverse.compact.frame",
  "mask_schema": "CUBM1",
  "frames": [
    {"frame": 0, "path": "frames/frame_000000.json"}
  ]
}
```

## Frame record

A `celluniverse.compact.frame` version-1 record contains:

- `frame`: absolute CellUniverse frame number.
- `source_frame`: source microscopy filename. The job input location resolves
  this filename; compact export does not duplicate the source TIFF.
- `pipeline_mode`: `traditional`, `celluniverse2`, `celluniverse3`,
  `celllumen`, or `cell_lumen_fusion`.
- `dimensions`: interpolated optimizer dimensions as `{x, y, z}`.
- `coordinates`: declares cell coordinate order `xyz`, dense volume order
  `zyx`, interpolated coordinate space, the effective
  `z_interpolation_ratio`, its `z_interpolation_source`, and the configured
  `initial_z_space`.
- `render_contract`: the synthetic-frame rasterization contract.
- `background`: one of the background encodings below.
- `cells`: cells in exact draw order.

Cell centers and radii are expressed in interpolated optimizer voxels, not
physical micrometers. All `theta_x`, `theta_y`, and `theta_z` rotations are in
radians.

Every cell contains:

```json
{
  "draw_order": 0,
  "name": "AB",
  "center": {"x": 1.0, "y": 2.0, "z": 3.0},
  "radii": {"a": 4.0, "b": 5.0, "c": 6.0},
  "rotation": {"theta_x": 0.0, "theta_y": 0.0, "theta_z": 0.0},
  "brightness": 0.5,
  "is_trash": false
}
```

Floating-point values are serialized with `max_digits10`, sufficient to
round-trip the C++ `float` state.

## Real-frame reconstruction

The compact export intentionally does not store a real-frame image. A reader
loads `source_frame` from the job input and displays the raw microscopy stack
after z interpolation with the record's effective
`z_interpolation_ratio`.

For a raw stack with `N > 1` z slices and ratio `r >= 1`, CellUniverse creates
`M = round_to_even((N - 1) * r) + 1` slices. For output index `k`, let
`position = k * (N - 1) / (M - 1)`, `source = floor(position)`, and
`t = position - source`. The output is:

```text
(1 - t) * raw[source] + t * raw[source + 1]
```

The first and final output slices therefore preserve the source endpoints.
The ratio in this record is the requested runtime value; the exact discrete
grid scale is `(M - 1) / (N - 1)`.
`z_interpolation_source` is `config` for a YAML-derived ratio, `initial_csv`
when supported schema-v2 initializer metadata overrides it, and
`cell_lumen_profile` for standalone CellLumen profile selection. The recorded
ratio therefore takes precedence over a default read directly from
`config.yaml`. The main pipeline records the finite ratio used by
`ImageHandler`, including fractional values. Standalone CellLumen records its
separately selected positive integer preview factor; its already-scaled cell
coordinates remain explicit in the cell records.

## Synthetic-frame reconstruction

Create the background volume first, then process `cells` in increasing
`draw_order`. For each integer voxel center `(x, y, z)`:

1. Build `R = Rz(theta_z) * Ry(theta_y) * Rx(theta_x)`.
2. Compute `local = transpose(R) * (voxel - center)`.
3. The voxel belongs to the cell when:

   ```text
   (local.x / a)^2 + (local.y / b)^2 + (local.z / c)^2 <= 1
   ```

4. Set every member voxel to the cell's `brightness`.

Later cells overwrite earlier cells in overlap regions. Trash cells are
rendered by the same rule; `is_trash` is retained for viewer semantics.
To match the legacy synthetic TIFF intensity conversion, convert the final
float32 volume to unsigned 8-bit with OpenCV's saturating conversion and scale
`255`. The geometric rule reconstructs the same ellipsoids, draw order, and
intensities. A reader that evaluates every voxel independently may differ at a
small number of boundary voxels from the legacy C++ rasterizer, which advances
rotated coordinates with incremental double-precision additions.

This gives two independently recoverable logical layers: the background
volume and the cell-only draw result. Compositing the ordered cells over the
background reproduces the synthetic frame.

## Background encodings

### Scalar

```json
{"kind": "scalar", "value": 0.02}
```

Fill the volume with `value`.

### Rotated soft ellipsoid

`kind` is `rotated_soft_ellipsoid`. The record stores `center`, `radii`,
`rotation`, `cold`, `hot`, `soft_margin`, and `additive_offset`.
It also stores `offset_updates`, the ordered deltas whose sum is
`additive_offset`.

For each voxel, use the same `Rz * Ry * Rx` rotation and calculate:

```text
local = transpose(R) * (voxel - center)
radial = sqrt((local.x/a)^2 + (local.y/b)^2 + (local.z/c)^2)
```

When `soft_margin <= 1e-8`, membership is one at `radial <= 1` and zero
otherwise. Otherwise:

```text
u = clamp((1 + soft_margin - radial) / (2 * soft_margin), 0, 1)
membership = u^2 * (3 - 2*u)
background = cold + (hot - cold) * membership
for delta in offset_updates:
  background = clamp(background + delta, 0, 1)
```

`additive_offset` records the cumulative `Frame::addBackgroundOffset()` change
to the installed spatial field. Version-1 readers should apply
`offset_updates` in order because a clamp follows every runtime update; using
the cumulative value once is equivalent only when intermediate saturation
does not occur. `Frame::setBackgroundColor()` is deliberately not included
because it changes the scalar fallback but not the installed spatial field.

### Binary mask

```json
{
  "kind": "binary_mask",
  "cold": 0.01,
  "hot": 0.03,
  "mask_format": "CUBM1",
  "mask_path": "masks/mask_<content-hash>.cubm"
}
```

A zero bit selects `cold`; a one bit selects `hot`.

`CUBM1` is a little-endian binary format:

| Offset | Size | Field |
| ---: | ---: | --- |
| 0 | 4 | ASCII `CUBM` |
| 4 | 2 | unsigned version, `1` |
| 6 | 2 | flags, `1` means LSB-first z/y/x bit order |
| 8 | 4 | unsigned width (`x`) |
| 12 | 4 | unsigned height (`y`) |
| 16 | 4 | unsigned depth (`z`) |
| 20 | 8 | unsigned voxel count |
| 28 | ceil(voxel count / 8) | packed mask bits |

The linear voxel index is `(z * height + y) * width + x`. Bit `i` is stored
in byte `i / 8`, at bit position `i % 8` (least-significant bit first).

## Recovery utility

`scripts/recover_compact_export.py` materializes the background-only and
cells-only uint8 ZYX TIFF layers for one manifest frame:

```bash
python scripts/recover_compact_export.py \
  --compact /path/to/output/compact \
  --frame 0 \
  --output-dir /path/to/recovered
```

Recovered content is separated by type:

```text
recovered/
  background/
    frame_000000.background.tif
  cells/
    frame_000000.cells.tif
  centers/
    cell_centers_ids.csv
  center_id_stacks/  # optional
    frame_000000.center_ids.tif
  recovery/
    frame_000000.recovery.json
    frame_000000.center_ids.recovery.json  # optional
    recovery_batch.json
  tools/
    view_recovered_synth.py
```

The recovery utility creates the applicable data folders. Package-local
viewers or other helpers belong under `tools/`; they are not generated by the
recovery command.

Use `--all-frames` to recover the complete manifest. The optional
`--center-id-sidecar` writes one sparse
`centers/cell_centers_ids.csv` table with
`frame,source_frame,cell_id,z,y,x,draw_order,is_trash,brightness`. It is
intended for a time-aware Points layer with text IDs and deliberately avoids a
third dense image volume.

```bash
python scripts/recover_compact_export.py \
  --compact /path/to/output/compact \
  --all-frames \
  --center-id-sidecar \
  --output-dir /path/to/recovered
```

### Optional raster center-ID layer

`--center-id-stacks` independently adds a visualization layer; it is not a
third input to synthetic-frame reconstruction and does not require the sparse
CSV. The output is a forced-Deflate uint8 ZYX TIFF per frame, using `0` for
empty space, `255` for the center marker, and `192` for the adjacent raster ID:

```bash
python scripts/recover_compact_export.py \
  --compact /path/to/output/compact \
  --all-frames \
  --resume \
  --center-id-stacks \
  --center-marker-shape sphere \
  --center-marker-radius 2 \
  --output-dir /path/to/recovered
```

The default marker is a radius-2 sphere; `cube` and radii from 1 through 16
are also supported. Centers and radii are expressed in interpolated optimizer
voxels. Cell IDs must be decimal strings and remain strings so leading zeros
are preserved. Deterministic 5x7 glyphs are placed beside their markers without
marker/text bounding-box overlap and occupy a three-slice Z slab.

Each `frame_NNNNNN.center_ids.recovery.json` record uses schema
`celluniverse.compact.center-id-recovery` and binds the marker/render contract
to the compact-frame SHA-256 and output TIFF size/SHA-256. Resume verifies and
skips this optional layer independently, without regenerating valid background
or cells TIFFs. `--overwrite-center-id-stacks` requires `--resume` and replaces
only the center-ID TIFFs and their completion records.

Every frame has a recovery JSON completion marker containing input/output
SHA-256 hashes. `--resume` verifies those hashes and binds each completed
triplet to its current manifest membership and frame path, frame record, mask,
dimensions, pipeline, coordinates, and compression before skipping it. An
append-only manifest can therefore add new frames without invalidating older
unchanged triplets. Partial, corrupt, input-mismatched, or no-longer-advertised
outputs fail rather than being silently reused.

Recovery records use output-root-relative grouped paths. Resume also accepts
the basename-only paths written by the earlier v1 recovery utility after those
files have been moved into the grouped tree. The writer rejects legacy-flat,
mixed, symlinked, or non-directory category paths so it cannot silently create
duplicate frames or escape the budget root.

The default decimal 10 GB output-root budget recursively includes every
content folder, including optional center-ID TIFFs, existing regular files,
temporary TIFF coexistence, and sidecars. Atomic temporary files stay beside
the applicable final file.
`--all-frames --dry-run` validates every frame and reports the
committed-output estimate without creating output. Streaming runtime checks
enforce the hard limit while each TIFF is written.

## Writer constraints

- Compact mode writes no reconstructed real or synthetic frame images.
- A non-analytic installed background must be scalar or contain exactly two
  distinct float values. The writer fails instead of silently approximating a
  more complex field.
- The format is versioned. Readers must reject unsupported schema versions or
  render-contract identifiers rather than guessing.
