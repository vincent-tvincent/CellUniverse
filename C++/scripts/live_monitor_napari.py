#!/usr/bin/env python3
"""Live Napari monitor for CellUniverse output folders."""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Napari writes generated theme/icon assets through the standard cache path.
# Defaulting to /tmp keeps this script usable on read-only home directories.
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/celluniverse_napari_cache")
os.environ.setdefault("XDG_CONFIG_HOME", "/tmp/celluniverse_napari_config")

import napari
import numpy as np
import tifffile
try:
    import dask.array as da
    from dask import delayed
except Exception:  # pragma: no cover - optional runtime dependency
    da = None
    delayed = None
from qtpy.QtCore import QTimer


FRAME_RE = re.compile(r"^(?:t)?(\d+)$")


@dataclass(frozen=True)
class LayerSpec:
    name: str
    relative_dir: Path
    opacity: float
    colormap: str
    z_offset_steps: int


# Bottom-to-top render order. The last layer is intentionally the top overlay.
LAYER_SPECS = (
    LayerSpec("real", Path("tiff/real"), 1.0, "gray", 0),
    LayerSpec("split_placements", Path("perturb_debug/split_placements"), 0.3, "bop purple", 1),
    LayerSpec("movement_placements", Path("perturb_debug/movement_placements"), 0.3, "cyan", 2),
    LayerSpec("synth", Path("tiff/synth"), 0.7, "gray", 3),
    LayerSpec("cell_centers", Path("perturb_debug/cell_centers"), 1.0, "bop orange", 4),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Open a Napari window and live-reload CellUniverse real/debug/synth "
            "TIFF stacks as complete frames arrive."
        )
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="CellUniverse output directory containing tiff/ and perturb_debug/.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Polling interval in minutes. Default: 1.0.",
    )
    parser.add_argument(
        "--z-offset",
        type=float,
        default=0.05,
        help="Per-layer Z offset in voxels. Default: 0.05.",
    )
    parser.add_argument(
        "--eager",
        action="store_true",
        help="Load full TIFF stacks into RAM. Default uses lazy dask arrays when dask is installed.",
    )
    parser.add_argument(
        "--stable-age",
        type=float,
        default=2.0,
        help="Minimum seconds since last file modification before reading. Default: 2.0.",
    )
    return parser.parse_args()


def frame_id(path: Path) -> int | None:
    match = FRAME_RE.match(path.stem)
    if not match:
        return None
    return int(match.group(1))


def tif_paths_by_frame(folder: Path) -> dict[int, Path]:
    frames: dict[int, Path] = {}
    if not folder.is_dir():
        return frames

    for path in folder.iterdir():
        if not path.is_file() or path.suffix.lower() not in {".tif", ".tiff"}:
            continue
        fid = frame_id(path)
        if fid is not None:
            frames[fid] = path
    return frames


def complete_frame_ids(root: Path) -> tuple[int, ...]:
    per_layer_ids: list[set[int]] = []
    for spec in LAYER_SPECS:
        layer_frames = tif_paths_by_frame(root / spec.relative_dir)
        per_layer_ids.append(set(layer_frames))

    if not per_layer_ids:
        return ()

    return tuple(sorted(set.intersection(*per_layer_ids)))


def stable_file(path: Path, stable_age_seconds: float) -> bool:
    try:
        stat = path.stat()
    except OSError:
        return False

    age_seconds = max(0.0, time.time() - (stat.st_mtime_ns / 1_000_000_000.0))
    return stat.st_size > 0 and age_seconds >= stable_age_seconds


def read_tiff_metadata(path: Path) -> tuple[tuple[int, ...], np.dtype]:
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        return tuple(series.shape), np.dtype(series.dtype)


def read_layer_stack(paths: Iterable[Path], lazy: bool, stable_age_seconds: float):
    paths = tuple(paths)
    for path in paths:
        if not stable_file(path, stable_age_seconds):
            raise RuntimeError(f"{path} is still changing or too new")

    if lazy:
        if da is None or delayed is None:
            raise RuntimeError("lazy loading requires dask; rerun with --eager or install dask")
        chunks = []
        for path in paths:
            shape, dtype = read_tiff_metadata(path)
            lazy_frame = delayed(tifffile.imread)(str(path))
            chunks.append(da.from_delayed(lazy_frame, shape=shape, dtype=dtype))
        if not chunks:
            return da.empty((0,), dtype=np.float32)
        return da.stack(chunks, axis=0)

    frames = []
    for path in paths:
        frames.append(tifffile.imread(path))

    if not frames:
        return np.empty((0,), dtype=np.float32)

    return np.stack(frames, axis=0)


def z_translate(data: np.ndarray, offset: float) -> tuple[float, ...]:
    translate = [0.0] * data.ndim
    if data.ndim >= 4:
        translate[1] = offset
    elif data.ndim >= 3:
        translate[0] = offset
    return tuple(translate)


class LiveMonitor:
    def __init__(
        self,
        output_dir: Path,
        interval_minutes: float,
        z_offset: float,
        lazy: bool,
        stable_age_seconds: float,
    ) -> None:
        self.output_dir = output_dir
        self.z_offset = z_offset
        self.lazy = lazy
        self.stable_age_seconds = stable_age_seconds
        self.viewer = napari.Viewer(title=f"CellUniverse live monitor: {output_dir.name}")
        self.loaded_frame_ids: tuple[int, ...] = ()
        self.timer = QTimer()
        self.timer.setInterval(int(max(0.01, interval_minutes) * 60_000))
        self.timer.timeout.connect(self.poll)

    def start(self) -> None:
        self.poll()
        self.timer.start()
        napari.run()

    def poll(self) -> None:
        frame_ids = complete_frame_ids(self.output_dir)
        if not frame_ids or frame_ids == self.loaded_frame_ids:
            return

        try:
            layer_data = self.load_all_layers(frame_ids)
        except Exception as exc:
            print(f"[live-monitor] Waiting for complete/stable export: {exc}", file=sys.stderr)
            return

        for spec, data in layer_data:
            translate = z_translate(data, spec.z_offset_steps * self.z_offset)
            if spec.name in self.viewer.layers:
                layer = self.viewer.layers[spec.name]
                layer.data = data
                layer.opacity = spec.opacity
                layer.colormap = spec.colormap
                layer.translate = translate
            else:
                self.viewer.add_image(
                    data,
                    name=spec.name,
                    opacity=spec.opacity,
                    colormap=spec.colormap,
                    translate=translate,
                )

        self.loaded_frame_ids = frame_ids
        self.viewer.dims.ndisplay = 3
        self.viewer.dims.set_point(0, len(frame_ids) - 1)
        mode = "lazy" if self.lazy else "eager"
        print(
            f"[live-monitor] Loaded {len(frame_ids)} complete frames ({mode}) "
            f"({frame_ids[0]}..{frame_ids[-1]})"
        )

    def load_all_layers(self, frame_ids: tuple[int, ...]):
        loaded = []
        for spec in LAYER_SPECS:
            layer_frames = tif_paths_by_frame(self.output_dir / spec.relative_dir)
            paths = [layer_frames[fid] for fid in frame_ids]
            loaded.append((spec, read_layer_stack(paths, self.lazy, self.stable_age_seconds)))
        return loaded


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if not output_dir.is_dir():
        print(f"Output directory does not exist: {output_dir}", file=sys.stderr)
        return 2

    missing = [str(spec.relative_dir) for spec in LAYER_SPECS if not (output_dir / spec.relative_dir).is_dir()]
    if missing:
        print(f"Output directory is missing required layer folders: {', '.join(missing)}", file=sys.stderr)
        return 2

    lazy = not args.eager
    if lazy and da is None:
        print("[live-monitor] dask is not installed; falling back to eager RAM loading.", file=sys.stderr)
        lazy = False

    LiveMonitor(output_dir, args.interval, args.z_offset, lazy, args.stable_age).start()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
