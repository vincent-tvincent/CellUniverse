from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

from .algorithms import load_volume
from .exporter import fingerprint_file
from .model import InitializerSession


def _configure_stable_vispy_backend() -> str:
    """Select VisPy's full desktop-GL backend before Napari creates a canvas.

    PyOpenGL's ``pyopengl2`` wrapper has produced ``GL_INVALID_OPERATION`` while
    drawing a Napari 3D Volume on this Mac. VisPy's ``gl+`` backend exposes the
    complete non-deprecated desktop API required by Napari's 3D Volume and
    instanced Points visuals instead of the restricted test wrapper.
    """

    from vispy import use
    from vispy.gloo import gl

    use(gl="gl+")
    return str(gl.current_backend.__name__).rsplit(".", 1)[-1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="initialize_frame.py",
        description=(
            "Open a guided Napari workflow that detects an embryo background "
            "envelope, initializes cell ellipsoids, and exports initial.csv."
        ),
    )
    parser.add_argument(
        "frame",
        type=Path,
        help="Multipage 3D TIFF frame or directory of ordered 2D TIFF slices",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Export directory (default: current working directory)",
    )
    parser.add_argument(
        "--initial-z-ratio",
        type=float,
        default=4.0,
        help="Initial Z interpolation ratio shown in the wizard (default: 4.0)",
    )
    parser.add_argument(
        "--headless-smoke",
        action="store_true",
        help="Create a headless ViewerModel/widget, validate startup, and exit",
    )
    return parser


def _fingerprint(path: Path) -> str:
    if path.is_file():
        return fingerprint_file(path)
    digest = hashlib.sha256(str(path.resolve()).encode())
    for item in sorted(path.iterdir()):
        if item.is_file():
            stat = item.stat()
            digest.update(item.name.encode())
            digest.update(str(stat.st_size).encode())
            digest.update(str(stat.st_mtime_ns).encode())
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = args.frame.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if args.initial_z_ratio <= 0:
        print("error: --initial-z-ratio must be positive", file=sys.stderr)
        return 2
    try:
        raw = load_volume(source)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Keep Napari/Numba caches in the project-isolated temporary root. This
    # avoids scattering task files through the macOS user cache and works
    # around Numba cache-locator failures when the environment lives on an
    # external volume.
    runtime_root = Path("/tmp/celluniverse-codex/initializer-runtime")
    cache_root = runtime_root / "cache"
    numba_cache = runtime_root / "numba-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    numba_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(numba_cache))

    try:
        vispy_gl_backend = _configure_stable_vispy_backend()
        import napari
        from qtpy import API_NAME, QT_VERSION
    except Exception as exc:
        print(
            "error: Napari and a Qt6 backend are required. Run this launcher with "
            "the prepared local environment:\n"
            "  ./.venv/bin/python initialize_frame.py <frame> [--output-dir DIR]\n"
            f"Import failure: {exc}",
            file=sys.stderr,
        )
        return 2

    from .widget import InitializerWizard

    session = InitializerSession(
        source_path=source,
        output_dir=output_dir,
        raw=raw,
        source_fingerprint=_fingerprint(source),
        z_ratio=float(args.initial_z_ratio),
    )
    print(
        f"Loaded {source}: shape(Z,Y,X)={raw.shape}, dtype={raw.dtype}; "
        f"Napari {napari.__version__}, {API_NAME} {QT_VERSION}, "
        f"VisPy GL backend {vispy_gl_backend}"
    )
    if args.headless_smoke:
        # macOS offscreen QOpenGL contexts are not reliable. ViewerModel tests
        # all layer and widget construction without creating a Vispy canvas.
        from napari.components import ViewerModel
        from qtpy import QtWidgets

        application = QtWidgets.QApplication.instance()
        if application is None:
            application = QtWidgets.QApplication([])
        viewer_model = ViewerModel()
        wizard = InitializerWizard(viewer_model, session)
        wizard.close()
        print("Headless Napari ViewerModel/initializer widget smoke test: PASS")
        return 0
    viewer = napari.Viewer(
        title=f"CellUniverse Initializer — {source.name}",
    )
    wizard = InitializerWizard(viewer, session)
    viewer.window.add_dock_widget(
        wizard,
        name="CellUniverse Initializer",
        area="right",
    )
    napari.run()
    return 0
