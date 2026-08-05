#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path


try:
    import numpy  # noqa: F401
    import tifffile  # noqa: F401
except ModuleNotFoundError:
    sys.modules.setdefault("numpy", types.ModuleType("numpy"))
    sys.modules.setdefault("tifffile", types.ModuleType("tifffile"))


SCRIPT = Path(
    os.environ.get(
        "CELLUNIVERSE_RECOVERY_SCRIPT",
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "recover_compact_export.py",
    )
)
SPEC = importlib.util.spec_from_file_location("recover_compact_export", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_contract(root: Path, ratio: object) -> None:
    frame_dir = root / "frames"
    frame_dir.mkdir()
    manifest = {
        "schema": MODULE.MANIFEST_SCHEMA,
        "version": MODULE.FORMAT_VERSION,
        "frame_schema": MODULE.FRAME_SCHEMA,
        "mask_schema": "CUBM1",
        "frames": [{"frame": 0, "path": "frames/frame_000000.json"}],
    }
    frame = {
        "schema": MODULE.FRAME_SCHEMA,
        "version": MODULE.FORMAT_VERSION,
        "frame": 0,
        "source_frame": "SPIMA_t000.tif",
        "pipeline_mode": "traditional",
        "dimensions": {"x": 1, "y": 1, "z": 1},
        "coordinates": {
            "cell_order": "xyz",
            "volume_order": "zyx",
            "space": "interpolated",
            "z_interpolation_ratio": ratio,
            "initial_z_space": "scaled",
            "z_interpolation_source": "initial_csv",
        },
        "render_contract": dict(MODULE.RENDER_CONTRACT),
        "background": {"kind": "scalar", "value": 0.0},
        "cells": [],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (frame_dir / "frame_000000.json").write_text(
        json.dumps(frame), encoding="utf-8"
    )


def load_ratio(ratio: object):
    with tempfile.TemporaryDirectory(prefix="celluniverse-ratio-test-") as tmp:
        root = Path(tmp)
        write_contract(root, ratio)
        return MODULE._load_contract(root, 0)


class FractionalCompactRatioTest(unittest.TestCase):
    def test_fractional_ratio_is_accepted(self) -> None:
        accepted = load_ratio(5.5)
        self.assertTrue(
            math.isclose(
                accepted.coordinates["z_interpolation_ratio"], 5.5
            )
        )

    def test_invalid_ratios_are_rejected(self) -> None:
        invalid_values = (
            0.999,
            -1.0,
            float("nan"),
            float("inf"),
            float("-inf"),
            True,
            False,
            "5.5",
        )
        for invalid in invalid_values:
            with self.subTest(ratio=invalid):
                with self.assertRaises(MODULE.RecoveryError):
                    load_ratio(invalid)


if __name__ == "__main__":
    unittest.main()
