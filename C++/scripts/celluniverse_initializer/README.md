# CellUniverse initializer 0.2.4

This directory is a self-contained source delivery of the guided Napari
initializer. It does not include a virtual environment or dataset.

## Install

From this directory:

```bash
./setup_venv.sh
```

The setup script creates `.venv` beside the launcher and installs the pinned
runtime ranges from `requirements.txt`.

## Run

The initializer consumes one multipage 3D TIFF (or a directory of ordered 2D
TIFF slices):

```bash
./run_initializer.sh /path/to/SPIMA_t001.tif --output-dir /path/to/output
```

The default initial Z interpolation ratio is `4.0`. Override it with:

```bash
./run_initializer.sh /path/to/SPIMA_t001.tif \
  --output-dir /path/to/output \
  --initial-z-ratio 4
```

Napari requires a graphical desktop. When launching on Vulcan, use a graphical
remote session or SSH with trusted X11 forwarding; otherwise install and run
this same source delivery on a workstation with access to the server data.

For a noninteractive startup check in an environment with a working Qt
offscreen backend:

```bash
QT_QPA_PLATFORM=offscreen ./run_initializer.sh \
  /path/to/SPIMA_t001.tif \
  --output-dir /tmp/celluniverse-codex/initializer-smoke \
  --headless-smoke
```

See `INITIALIZER_GUIDE.md` for the complete five-step workflow and manual
review controls.

## Integration boundary

The exported schema-2 `initial.csv` and companion JSON include the two-region
background, soft margin, rotations, and fractional/interpolated Z metadata.
The current CellUniverse C++ ingestion path does not yet consume all of those
fields. Treat the output as initializer data prepared for the pending parser
integration; do not claim a production tracking run until that integration is
implemented and verified.
