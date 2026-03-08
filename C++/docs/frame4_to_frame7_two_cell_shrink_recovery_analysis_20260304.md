# Frame 4-7 Two-Cell Shrink + Brightness Recovery Analysis (2026-03-04)

Output analyzed:

- `/Volumes/vincent/celluniverse/outputs/output_yuancen_20260304_152817`

Cells analyzed:

- `123456793425243543545020349852341`
- `e9077677575842b1a2925729fbcfb3a50`

Data sources:

- `cells.csv`
- `debug_log.txt`

## 1) Volume and shape trajectory (frame 4 to frame 7)

### Cell `123456793425243543545020349852341`

- Frame 4: `major=18.1614`, `minor=17.3620`, `vol=23987.612`
  - Previous frame volume (frame 3): `32416.828`
  - Ratio: `0.740` -> trigger check (`< 0.78`) = `YES`
- Frame 5: `major=17.6119`, `minor=10.3086`, `vol=13393.705`
  - Ratio vs frame 4: `0.558` -> trigger `YES`
- Frame 6: `major=15.0000`, `minor=5.0000`, `vol=4712.389`
  - Ratio vs frame 5: `0.352` -> trigger `YES`
- Frame 7: `major=15.0000`, `minor=5.0000`, `vol=4712.389`
  - Ratio vs frame 6: `1.000` -> trigger `NO`

### Cell `e9077677575842b1a2925729fbcfb3a50`

- Frame 4: `major=16.7215`, `minor=16.7215`, `vol=19584.582`
  - Previous frame volume (frame 3): `24347.685`
  - Ratio: `0.804` -> trigger `NO`
- Frame 5: `major=15.0000`, `minor=11.2482`, `vol=10601.179`
  - Ratio vs frame 4: `0.541` -> trigger `YES`
- Frame 6: `major=15.0000`, `minor=5.0000`, `vol=4712.389`
  - Ratio vs frame 5: `0.445` -> trigger `YES`
- Frame 7: `major=15.0000`, `minor=5.0000`, `vol=4712.389`
  - Ratio vs frame 6: `1.000` -> trigger `NO`

## 2) Brightness recovery logs for these two cells

### Frame 4

- `123...341`: recovery attempts 1-2, then `status=stopped_ratio`
  - baseline `0.546081`, candidate step `0.506081->0.486081`
  - ratio `0.890126 < min_ratio 0.9`
- `e907...b3a50`: no recovery event in frame 4

### Frame 5

- `123...341`: attempts 1-2, then `status=stopped_ratio`
  - ratio `0.893178 < 0.9`
- `e907...b3a50`: attempts 1-3, then `status=stopped_ratio`
  - ratio `0.871416 < 0.9`

### Frame 6

- `123...341`: attempts 1-2, then `status=stopped_ratio`
  - ratio `0.881897 < 0.9`
- `e907...b3a50`: attempts 1-2, then `status=stopped_ratio`
  - ratio `0.886727 < 0.9`

### Frame 7

- No brightness recovery events for these cells (and frame-level recovery count was 0).

## 3) Brightness update trend (gets dimmer each frame)

### Cell `123...341`

- Frame 4 blended: `0.561681`
- Frame 5 blended: `0.508032`
- Frame 6 blended: `0.404833`
- Frame 7 blended: `0.342333`

### Cell `e907...b3a50`

- Frame 4 blended: `0.622161`
- Frame 5 blended: `0.529694`
- Frame 6 blended: `0.454144`
- Frame 7 blended: `0.411791`

## 4) Why behavior looks wrong by frame 6-7

1. Both cells collapse to minimum-size boundary by frame 6 (`major=15`, `minor=5`).
2. Recovery trigger is based on frame-to-frame shrink (`cur < prev * 0.78`).
3. Once stuck at the min-size floor, frame 7 has no additional shrink (`ratio=1.0`), so recovery does not trigger.
4. Recovery attempts in frames 4-6 are repeatedly cut short by the ratio stop gate (`status=stopped_ratio`).
5. Measured brightness keeps decreasing; EMA brightness update keeps pulling both cells dimmer each frame.

## 5) Net diagnosis

The observed failure is a combined effect of:

- hard size floor reached early,
- shrink-only recovery trigger (no absolute-smallness trigger),
- early termination by `stopped_ratio`,
- and continued downward brightness updates.

This combination can lock cells into a dim + undersized state where frame-to-frame recovery no longer activates.
