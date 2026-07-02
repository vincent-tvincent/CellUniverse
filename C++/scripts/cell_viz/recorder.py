"""Capture rendered frames to a video for playback."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def _fit_to(rgb: np.ndarray, shape) -> np.ndarray:
    """Pad/crop an RGB frame to a fixed (H, W) so every video frame matches."""
    th, tw = shape
    h, w = rgb.shape[:2]
    out = np.zeros((th, tw, 3), dtype=rgb.dtype)
    ch, cw = min(h, th), min(w, tw)
    out[:ch, :cw] = rgb[:ch, :cw, :3]
    return out


class VideoRecorder:
    def __init__(self, out_dir: Path, first: int, last: int, fps: int = 4,
                 save_pngs: bool = True):
        self.viz_dir = Path(out_dir) / "viz"
        self.frames_dir = self.viz_dir / "frames"
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        if save_pngs:
            self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.save_pngs = save_pngs
        self.video_path = self.viz_dir / f"playback_f{first}-f{last}.mp4"
        self._writer = None
        self._fps = fps
        self._count = 0
        self._shape = None  # fixed (H, W), even dims, set from first frame

    def _ensure_writer(self):
        if self._writer is None:
            # macro_block_size=16 pads to a libx264-friendly size; quality knob.
            self._writer = imageio.get_writer(
                self.video_path, fps=self._fps, macro_block_size=16,
                codec="libx264", quality=8,
            )

    def capture(self, rgb, frame_number: int):
        """Append one rendered RGB(A) array; optionally save a PNG too."""
        if rgb is None:
            return
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        if self._shape is None:
            h, w = rgb.shape[:2]
            self._shape = (h + (h % 2), w + (w % 2))  # even dims
        rgb = _fit_to(rgb, self._shape)
        self._ensure_writer()
        self._writer.append_data(rgb)
        self._count += 1
        if self.save_pngs:
            imageio.imwrite(self.frames_dir / f"frame_{frame_number:03d}.png", rgb)

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    @property
    def frames_written(self) -> int:
        return self._count
