# CellUniverse interactive initializer

This local-first tool opens one 3D microscopy frame in Napari and guides the
user through:

1. embryo membrane/background-region detection,
2. solid cell-region and per-slice watershed-seed review,
3. Z interpolation of the image and ID-preserving cell boundaries, followed by
   robust center recalculation,
4. per-cell brightness-offset fitting and simulated-environment preview, and
5. final refined-ellipsoid review and `initial.csv` export.

The source pixels are immutable. The upper clamp and all filter responses are
preview data only.

By default, each stage shows the source image, one final result overlay, and
numbered centers where applicable. The colored cell overlay represents solid
enclosed cell regions—not edge pixels. DoG, LoG, fused response, thresholded
edges, and other intermediate layers remain hidden unless **Diagnostics** is
enabled.

Step 2 exposes a **Fused edge threshold** slider. Lower values admit weaker
DoG/LoG boundary responses; higher values retain only stronger responses.
Those thresholded edges act as soft watershed guidance and do not remove
pixels from the final filled cell regions.

The **Discard enclosed 3D chunks smaller than** control measures the complete
26-connected candidate volume in raw voxels after per-slice closing. It does
not apply the threshold independently to each Z cross-section. Likewise,
discarding a small bright center core removes only that core as a watershed
seed; the surviving foreground chunk receives a distance-based fallback center
instead of being deleted.

In Step 2, each yellow point is an accepted bright-core watershed-seed
observation on that Z slice. Seed points are intentionally unnumbered. A
separate high-visibility overlay places exactly one zero-based ID at the robust
3D center of every solid cell, matching the IDs reported by interpolation and
ellipsoid fitting. If a candidate chunk has no usable bright core,
the detector supplies a distance-based fallback seed. To reject a false seed,
select one or more yellow points with Napari's select tool and click
**Mark selected yellow center(s) wrong**:

- If the seed lies on a 26-connected 3D component detached from that label's
  selected cell body, and no other accepted seed supports the detached
  component, the complete detached component is removed.
- The selected solid cell body remains byte-for-byte unchanged even if every
  one of its seeds is rejected. This includes rejecting the only seed on one
  ordinary Z cross-section while the cell continues through neighboring
  layers. To delete an entire false cell, use **Mark picked segment wrong**
  instead of rejecting its seed observations.

The yellow observations are review and component-selection evidence; they are
not averaged into the exported XYZ position. For each positive cell label, the
initializer first selects the 26-connected 3D component with the most accepted
seed support, breaking ties by voxel count. Short, laterally aligned components
whose Z ranges do not overlap are joined when they are separated by no more
than two missing source slices; a lateral island occupying the same Z range is
not joined. The initializer then computes an intensity-weighted XY centroid on
each occupied Z slice. A robust tilted centerline keeps the main run of slice
centroids and automatically excludes lateral outliers, after which the
retained voxels jointly determine X, Y, and Z. Local percentile normalization
and a geometric weight floor prevent one saturated voxel from dominating the
chunk. If the weighted center falls outside a concave label, it is projected
to the nearest retained cell voxel. A center lying in an inferred short Z gap
is preserved instead of being snapped into one half of the cell. For a cell
visible on exactly three layers, a separate conservative rule removes only a
grossly displaced slice when the other two centers form a compact low-motion
pair; ambiguous three-layer geometry is kept for user review.

A rejected seed remains excluded from future seed/component-selection evidence,
but it does not remove a valid segmentation slice and is not a prerequisite for
calculating the chunk-derived center. Only the isolated-chunk rule above changes
segmentation pixels; a selected solid cell remains valid with zero accepted
seeds, and rejecting the last seed on a normal cell slice never creates a Z
gap. Automatically excluded slice centroids are shown as
unnumbered magenta X markers only when **Diagnostics** is enabled. Select a red
X and click **Restore selected red center(s)** to restore both the seed
observation and the exact detached 3D chunk snapshot that was removed. If a
later label edit occupies any saved voxel, restoration is blocked rather than
silently returning a partial chunk. Corrected slices must be reviewed again
before continuing. Changing detector parameters creates a new candidate set
and intentionally resets manual seed decisions.

To reject an entire false-positive segmentation, click **Pick a colored
cell in the viewer**, click its filled label with Napari's picker, then
click **Mark picked segment wrong**. This removes that label through all source
Z slices and rejects its associated centers. **Restore most recently marked
segment** restores the exact saved mask and the centers that were accepted
when it was removed.

To merge any number of cells that should be one solid 3D chunk:

1. Click **Pick a colored cell in the viewer** and click one intended cell.
2. Click **Mark/unmark picked cell for merge**. The marked cell is highlighted
   in cyan in the **Cells marked for merge** layer.
3. Repeat the pick and mark action for every cell that belongs in the same
   chunk. Picking and toggling a marked cell again removes it; **Clear marked
   cells** resets the complete selection.
4. Click **Merge all marked cells**.

There is no target cell to choose. The merge deterministically retains the
smallest current label and absorbs every other marked label; normal Step 2
review may later compact the displayed IDs. It only relabels voxels and never
adds, removes, or bridges foreground. The complete marked union must be one
connected 3D cell or a set of short, laterally aligned Z continuations. A
laterally detached union is refused because downstream center calculation
would otherwise split it again. All seed observations and rejection records
are reassigned to the retained ID, the marked selection is cleared after a
successful merge, affected slices are marked for review, and
interpolation/fitting is invalidated.

For a merged pair that cannot be separated by the automatic controls:

1. Click **Draw separator line on current Z**.
2. Draw one magenta line completely across the shared neck of the colored
   segment.
3. Optionally move to other representative Z slices and draw more lines to
   control how the separator tilts through the stack. More than one guide on
   the same layer is also retained.
4. If a guide repeatedly selects the wrong colored segment, raise **Optional
   line-target guard**. Its default `0` is off, so it does not reject an
   intentional guide merely because the line also crosses a nearby label.
5. Click **Apply all separator lines**.

All lines targeting the same segment are fitted into one 3D separation plane.
With a single line, that line is extended vertically through Z. Every voxel of
the target is classified directly by plane side on every layer: the
smaller-X/left side keeps the source ID, and the larger-X/right side receives a
new ID. A horizontal separator uses Y as the deterministic fallback. The
algorithm does not require the drawn pixels to disconnect the old 3D label and
does not use watershed propagation. Plane pixels are assigned to one side, so
the result has two solid adjacent labels with no hollow gap.
Existing accepted and rejected seed observations are reassigned to the child
that actually contains them. If a new child had no prior observation, exactly
one geometric fallback seed is added; the initializer no longer creates one
new yellow point on every occupied Z slice. Affected slices are marked for
review, and downstream interpolation/fitting is invalidated. **Undo last split
or merge** restores the exact state before the most recent successful manual
split or merge, including labels, seed decisions, rejection records, and
reviewed slices. It remains available while reviewing affected slices, and is
cleared when a different segmentation edit replaces it. The magenta guides
remain editable until they are cleared.

While a preview is running, the wizard displays a progress bar and elapsed
time. Step 2 reuses a bounded preprocessing cache between slider changes, and
diagnostic image layers are created only when **Diagnostics** is enabled.

When Step 2 is confirmed, a cell ID that still covers separate, non-aligned 3D
pieces is repaired automatically. The robustly selected piece keeps its ID;
each other piece, or short aligned group of pieces, is measured against
**Discard enclosed 3D chunks smaller than**. A sub-threshold detached child is
discarded automatically; every retained child receives a new cell ID. Existing
seed decisions follow the correct retained cell, seed observations belonging
to discarded support are removed, and only a retained child with no prior seed
observation receives one fallback seed. No connecting bridge is invented. The
automatic cleanup/split can be restored exactly with **Undo last split or
merge** after returning to Step 2.

The Z step interpolates the source image and every cell boundary independently
while preserving its cross-slice ID. It then recalculates every robust
intensity-weighted center from the interpolated image and label volume instead
of merely multiplying the source Z coordinate. Short aligned same-ID Z gaps
are filled by signed-distance interpolation before the output grid is built,
so one missing source layer cannot silently delete half of a cell. Step-3
centroid outliers are independently recomputed and shown as unnumbered magenta
X markers when **Diagnostics** is enabled. This step performs no membrane or
cell-ellipsoid fitting. The user first confirms that the interpolated geometry
and center locations are correct. It normally opens in the 3D-volume view. The
checkbox can be turned off to recover a safe 2D slice view if the local OpenGL
renderer fails.

Interpolation can expose tiny disconnected satellites when a noisy
source-space connection is stretched through Z. The initializer does not
silently delete those source-backed voxels. If the robust center must ignore
more than 1% of any interpolated label, Step 3 names that cell and blocks
ellipsoid fitting until the user returns to Step 2 to correct, split, or remove
the ambiguous region. It also blocks before fitting if any interpolated cell
contains fewer than eight voxels. Sub-percent satellites remain diagnostic
rather than forcing a risky automatic topology change.

The Cell fitting step then creates an immutable baseline ellipsoid for each
interpolated cell. It provides one nonnegative brightness-offset slider per
cell. An offset of zero preserves that cell's baseline ellipsoid. A positive
offset searches only inside the immutable baseline ellipsoid and refits the
cell's radii and orientation from the brighter accepted volume while keeping
its robust chunk-derived center; moving the slider is always evaluated from
the same baseline rather than compounding earlier adjustments. Clicking a cell
ellipsoid in the 3D fitting view selects, scrolls to, and focuses its matching
brightness-offset slider. The 3D preview renders cell ellipsoids transparently
so the image remains visible through them, while the background-envelope
ellipsoid is rendered more visibly in its separate lower layer. The final
review and export use the refined cell ellipsoids, not the immutable baselines.

Cell fitting and final review normally show the interpolated 3D volume together
with the original smooth, translucent cell and background-envelope Surface
models. Exactly one numbered ID is anchored at each cell's 3D center. The 2D
fallback remains a real model preview rather than only the original
segmentation: it rasterizes the current rotated ellipsoids, overlays every
fitted cell cross-section, outlines the fitted background envelope, reuses the
same single center ID for each cell, and supports clicking a cross-section to
focus its slider. Each page remembers its own 3D/fallback choice. If the macOS
Qt/VisPy canvas becomes blank or reports a `GLError`, uncheck that page's 3D box
to restore the 2D renderer.

The launcher explicitly selects VisPy's full desktop-OpenGL `gl+` backend
before importing Napari. This avoids the observed restricted `pyopengl2`
`GL_INVALID_OPERATION` failure while drawing a 3D Volume and supports Napari's
instanced 3D point overlays.

After all current cell fits finish, the initializer re-estimates both
background brightness levels in the interpolated image. It excludes the hard
union of every finalized cell ellipsoid. High-confidence membrane-interior
voxels supply the bright/hot value, high-confidence exterior voxels supply the
dim/cold value, and the soft transition band is excluded from both. The
initializer records the robust-estimation method, voxel counts, and
distribution statistics. This final `interpolated_fitted_cells_excluded`
estimate replaces the earlier source-space diagnostic estimate; export is
blocked if it is missing or stale.

## Run

From the remote-development control workspace:

```bash
./.venv/bin/python initialize_frame.py \
  "/Volumes/vincent/2026_bright_channel/Pos4/SPIMA/SPIMA_t001.tif" \
  --output-dir .
```

The prepared `.venv` uses Python 3.12, Napari, and PyQt6. The output directory
defaults to the current directory.

## Current integration boundary

CSV schema version 2 contains explicit rotation, hot-background region,
soft-margin, Z-ratio, coordinate-space, `coldBackgroundBrightness`, and
`hotBackgroundBrightness` columns. Both background levels are populated only
on the hot background-envelope row; cell rows leave those two columns blank.
The envelope row's ordinary `brightness` value is the same final hot
background value. The companion `initial.initializer.json` records that value,
the cold value, the exact estimation stage, and structured calculation
statistics.

The current remote CellUniverse C++ parser does not yet consume all of these
fields, including the explicit cold-background brightness. Do not use the new
CSV with the production tracker until the planned remote parser and
fractional-Z integration has been implemented and verified.
