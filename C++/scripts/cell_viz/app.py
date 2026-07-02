"""Orchestration: wire watcher -> readers -> scene -> renderer, live or batch."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import tifffile

from . import readers
from .model import FrameData
from .scene import build_scene
from .watcher import OutputDirWatcher


def _raw_path(pattern: str, frame: int) -> Path:
    # pattern uses printf style, e.g. data/input/embryo_data/t%03d.tif
    return Path(pattern % frame)


def load_raw(pattern: str, frame: int):
    p = _raw_path(pattern, frame)
    if not p.exists():
        return None
    img = tifffile.imread(str(p))
    return img


def resolve_z_scale(run_dir: Path, cli_z_scale: float | None, first_frame: int) -> float:
    if cli_z_scale:
        return cli_z_scale
    ckpt = run_dir / "checkpoints" / f"frame_{first_frame:03d}.txt"
    z_slices, max_z = readers.read_checkpoint_zinfo(ckpt)
    if z_slices and max_z:
        return max(1.0, max_z / z_slices)
    return 7.0  # dataset default (config z_scaling)


class VizApp:
    def __init__(self, run_dir, input_pattern, first_frame, last_frame,
                 z_scale=None, mode="3d", show_labels=True):
        self.run_dir = Path(run_dir)
        self.input_pattern = input_pattern
        self.first = first_frame
        self.last = last_frame
        # z-scale for the RAW fallback only; the interpolated output frame is 1:1.
        self.z_scale_raw = resolve_z_scale(self.run_dir, z_scale, first_frame)
        self.mode = mode
        self.show_labels = show_labels
        self.watcher = OutputDirWatcher(self.run_dir, first_frame, last_frame)
        self._frames: dict[int, FrameData] = {}
        self._warned_raw = False

    def frame_data(self, n: int) -> FrameData:
        if n not in self._frames:
            self._frames[n] = readers.load_frame(self.run_dir, n)
        return self._frames[n]

    def image_for(self, n: int):
        """Return ``(image, z_scale)`` for frame ``n``.

        Prefer CellUniverse's z-interpolated ``<run>/<n>_real.tif`` (full depth,
        cells map 1:1 so z_scale=1). Fall back to the raw input frame (needs the
        config z-scale) only if export_frame_tiff was off — warn once.
        """
        real_tif = self.run_dir / f"{n}_real.tif"
        if real_tif.exists():
            return tifffile.imread(str(real_tif)), 1.0
        if not self._warned_raw:
            print(f"[viz] WARNING: {real_tif.name} not found — falling back to raw "
                  f"(set simulation.export_frame_tiff: true for interpolated frames).",
                  flush=True)
            self._warned_raw = True
        return load_raw(self.input_pattern, n), self.z_scale_raw

    def synth_for(self, n: int):
        """The actual C++ synth volume ``<run>/<n>_synth.tif`` (z-interpolated), or None."""
        synth_tif = self.run_dir / f"{n}_synth.tif"
        if synth_tif.exists():
            return tifffile.imread(str(synth_tif))
        return None

    def scene_for(self, n: int, z_scale: float, reveal=None):
        curr = self.frame_data(n)
        prev = self.frame_data(n - 1) if (n - 1) in self.watcher.completed_frames() else None
        return build_scene(curr, prev, z_scale, reveal=reveal,
                           show_labels=self.show_labels)

    def _render_mpl(self, n, recorder, imageio):
        from .mpl_render import render_frame_synth
        img, z_scale = self.image_for(n)
        prev = self.frame_data(n - 1) if (n - 1) in self.watcher.completed_frames() else None
        png = recorder.frames_dir / f"frame_{n:03d}.png"
        recorder.frames_dir.mkdir(parents=True, exist_ok=True)
        render_frame_synth(self.frame_data(n), prev, img, png, n, z_scale=z_scale,
                           show_labels=self.show_labels, synth=self.synth_for(n))
        recorder.capture(imageio.imread(png), n)
        print(f"[viz] rendered frame {n} ({len(self.frame_data(n).cells)} cells)",
              flush=True)

    # ---- batch / headless via matplotlib (no OpenGL needed) ----
    def run_batch_mpl(self, fps=4, save_pngs=True):
        import imageio.v2 as imageio

        from .recorder import VideoRecorder

        frames = self.watcher.completed_frames()
        if not frames:
            raise SystemExit(f"No completed frames under {self.run_dir}/checkpoints")
        recorder = VideoRecorder(self.run_dir, frames[0], frames[-1], fps=fps,
                                 save_pngs=False)
        for n in frames:
            self._render_mpl(n, recorder, imageio)
        recorder.close()
        print(f"[viz] wrote {recorder.frames_written} frames -> {recorder.video_path}",
              flush=True)
        return recorder.video_path

    # ---- live headless tailer via matplotlib (no display, no GL) ----
    def run_live_mpl(self, interval=2.0, fps=4, idle_timeout=120.0):
        """Poll the output dir and render each new completed frame as it lands.

        Used by the baked-in auto-launch on display-less machines. Stops once the
        last requested frame is done, or after ``idle_timeout`` with no new frame.
        """
        import time

        import imageio.v2 as imageio

        from .recorder import VideoRecorder

        first = self.first if self.first is not None else 0
        last = self.last if self.last is not None else first
        recorder = VideoRecorder(self.run_dir, first, last, fps=fps, save_pngs=False)
        print(f"[viz] live mpl tailer watching {self.run_dir} "
              f"(frames {first}..{last})", flush=True)
        idle = 0.0
        while True:
            fresh = self.watcher.new_frames()
            if fresh:
                idle = 0.0
                for n in fresh:
                    self._render_mpl(n, recorder, imageio)
            else:
                idle += interval
            if self.watcher.is_finished() and not fresh:
                break
            if idle >= idle_timeout:
                print(f"[viz] idle {idle:.0f}s, stopping tailer", flush=True)
                break
            time.sleep(interval)
        recorder.close()
        print(f"[viz] wrote {recorder.frames_written} frames -> {recorder.video_path}",
              flush=True)
        return recorder.video_path

    # ---- batch / headless: render every completed frame to a video ----
    def run_batch(self, record="settled", fps=4, save_pngs=True):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        import napari  # noqa: WPS433  (deferred: needs Qt)
        from .recorder import VideoRecorder
        from .render import NapariRenderer

        frames = self.watcher.completed_frames()
        if not frames:
            raise SystemExit(f"No completed frames under {self.run_dir}/checkpoints")
        sample, _ = self.image_for(frames[0])
        raw_shape = sample.shape if sample is not None else (239, 512, 708)

        viewer = napari.Viewer(show=False, ndisplay=3 if self.mode == "3d" else 2)
        renderer = NapariRenderer(viewer, raw_shape)
        recorder = VideoRecorder(self.run_dir, frames[0], frames[-1], fps=fps,
                                 save_pngs=save_pngs)
        try:
            for n in frames:
                img, z_scale = self.image_for(n)
                if img is not None:
                    renderer.set_raw(img)
                renderer.set_synth(self.synth_for(n))
                renderer.render(self.scene_for(n, z_scale))
                rgb = viewer.screenshot(canvas_only=True, flash=False)
                recorder.capture(rgb, n)
                print(f"[viz] rendered frame {n} "
                      f"({len(self.frame_data(n).cells)} cells)")
        finally:
            recorder.close()
            viewer.close()
        print(f"[viz] wrote {recorder.frames_written} frames -> {recorder.video_path}")
        return recorder.video_path

    # ---- live: interactive window that tails the run ----
    def run_live(self, interval=2.0, fit_step_ms=120, record="settled", fps=4):
        import napari
        from qtpy.QtCore import QTimer

        from .recorder import VideoRecorder
        from .render import NapariRenderer

        # show the first available frame (or a blank canvas)
        frames = self.watcher.completed_frames()
        f0 = frames[0] if frames else self.first
        sample, _ = self.image_for(f0) if frames else (None, 1.0)
        raw_shape = sample.shape if sample is not None else (239, 512, 708)

        viewer = napari.Viewer(ndisplay=3 if self.mode == "3d" else 2)
        renderer = NapariRenderer(viewer, raw_shape)
        recorder = (VideoRecorder(self.run_dir, self.first, self.last, fps=fps)
                    if record != "off" else None)

        state = {"current": None, "recorded": set()}

        def show_frame(n: int):
            img, z_scale = self.image_for(n)
            if img is not None:
                renderer.set_raw(img)
            renderer.set_synth(self.synth_for(n))
            renderer.render(self.scene_for(n, z_scale))
            viewer.title = f"CellUniverse viz — frame {n}"
            state["current"] = n
            if recorder is not None and n not in state["recorded"]:
                rgb = viewer.screenshot(canvas_only=True, flash=False)
                recorder.capture(rgb, n)
                state["recorded"].add(n)

        if frames:
            show_frame(frames[-1])

        def poll():
            fresh = self.watcher.new_frames()
            if fresh:
                show_frame(max(fresh))
            if recorder is not None and self.watcher.is_finished():
                recorder.close()

        timer = QTimer()
        timer.timeout.connect(poll)
        timer.start(int(interval * 1000))
        self._timer = timer  # prevent GC; viewer owns the event loop
        napari.run()
        if recorder is not None:
            recorder.close()
