"""Detect completed frames in a running (or finished) output dir.

A frame N is "done and safe to read" once ``checkpoints/frame_0NN.txt`` exists —
the checkpoint is written after cells.csv / candidate_graph for that frame.
"""

from __future__ import annotations

import re
from pathlib import Path

_CKPT_RE = re.compile(r"frame_(\d+)\.txt$")


class OutputDirWatcher:
    def __init__(self, run_dir: Path, first_frame: int | None = None,
                 last_frame: int | None = None):
        self.run_dir = Path(run_dir)
        self.first_frame = first_frame
        self.last_frame = last_frame
        self._seen: set[int] = set()

    def completed_frames(self) -> list[int]:
        """All frame numbers whose checkpoint exists, in range, sorted."""
        ckpt_dir = self.run_dir / "checkpoints"
        if not ckpt_dir.is_dir():
            return []
        frames = []
        for p in ckpt_dir.glob("frame_*.txt"):
            m = _CKPT_RE.search(p.name)
            if not m:
                continue
            n = int(m.group(1))
            if self.first_frame is not None and n < self.first_frame:
                continue
            if self.last_frame is not None and n > self.last_frame:
                continue
            frames.append(n)
        return sorted(frames)

    def new_frames(self) -> list[int]:
        """Completed frames not yet returned by a previous call."""
        fresh = [n for n in self.completed_frames() if n not in self._seen]
        self._seen.update(fresh)
        return fresh

    def is_finished(self) -> bool:
        """True once the last requested frame has a checkpoint."""
        if self.last_frame is None:
            return False
        return self.last_frame in set(self.completed_frames())
