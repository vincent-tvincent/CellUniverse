"""Unit tests for the pure cell_viz layers (no GUI).

Run from C++/:  .venv-viz/bin/python -m pytest tests/cell_viz -q
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pytest

CPP_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CPP_ROOT / "scripts"))

from cell_viz import geometry, lineage, readers  # noqa: E402


def _smoke_run() -> Path:
    matches = sorted(
        p for p in glob.glob(str(CPP_ROOT / "outputs" / "output_smoketest_f85-88_*"))
        if Path(p).is_dir()
    )
    if not matches:
        pytest.skip("no smoke-test run dir found under outputs/")
    return Path(matches[-1])


# ---- lineage ----

def test_parent_of_split_daughter():
    assert lineage.parent_of("Cell type 1_20100") == "Cell type 1_2010"
    assert lineage.parent_of("Cell type 1_200100") == "Cell type 1_20010"


def test_parent_of_root_and_trash():
    assert lineage.parent_of("Cell type 1_3") is None
    assert lineage.parent_of("trash_5") is None
    assert lineage.parent_of("no_digits") is None


def test_is_daughter_of():
    assert lineage.is_daughter_of("Cell type 1_20010", "Cell type 1_2001")
    assert not lineage.is_daughter_of("Cell type 1_20010", "Cell type 1_2000")


# ---- readers (against real smoke fixtures) ----

def test_read_cells_csv_frame_filter():
    run = _smoke_run()
    cells = readers.read_cells_csv(run / "cells.csv", 88)
    assert len(cells) == 76, f"expected 76 cells at f88, got {len(cells)}"
    assert all(c.a_radius > 0 for c in cells if not c.is_trash)
    names = {c.name for c in cells}
    assert any(n.startswith("Cell type 1_") for n in names)


def test_candidate_graph_parsing():
    run = _smoke_run()
    cg = run / "candidate_graph" / "frame_86_candidates.csv"
    lumen, splits, fit_order = readers.read_candidate_graph(cg)
    assert len(lumen) == 92, f"expected 92 lumen centers, got {len(lumen)}"
    assert splits, "expected at least one accepted split at f86"
    # accepted split parents must be real cell names
    assert all(s.parent.startswith("Cell type") for s in splits)
    # lumen centers carry signal/voxels
    assert all(lc.signal > 0 for lc in lumen)
    assert len(fit_order) >= 61


def test_load_frame_consistency():
    run = _smoke_run()
    fd = readers.load_frame(run, 86)
    assert fd.frame == 86
    assert fd.cells and fd.lumen_centers
    # candidate_graph "selected" splits are PROPOSALS (seeds), not guaranteed
    # commits — so we only require they carry valid seed coordinates.
    for s in fd.splits:
        assert s.parent.startswith("Cell type")
        assert any(v != 0 for v in s.d1) and any(v != 0 for v in s.d2)


def test_committed_splits_f85_to_f86():
    run = _smoke_run()
    prev = readers.load_frame(run, 85)
    curr = readers.load_frame(run, 86)
    events = lineage.committed_splits(prev, curr)
    # f85->f86 grew 61->71 cells: real splits must be detected here.
    assert events, "expected committed splits between f85 and f86"
    for parent_cell, d1, d2 in events:
        assert lineage.parent_of(d1.name) == parent_cell.name
        assert lineage.parent_of(d2.name) == parent_cell.name
        assert d1.name != d2.name
        # parent ghost has a real position to draw
        assert parent_cell.a_radius > 0


def test_checkpoint_zinfo():
    run = _smoke_run()
    z_slices, max_z = readers.read_checkpoint_zinfo(run / "checkpoints" / "frame_085.txt")
    assert z_slices == 35
    assert max_z == 238


# ---- geometry ----

def test_rotation_identity():
    R = geometry.rotation_matrix(0, 0, 0)
    assert np.allclose(R, np.eye(3))


def test_rotation_orthonormal():
    R = geometry.rotation_matrix(0.3, -0.7, 1.1)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_wireframe_axis_aligned_extent():
    # axis-aligned ellipsoid at origin; z_scale=7 compresses the z axis index.
    wf = geometry.ellipsoid_wireframe(
        center=(100.0, 200.0, 70.0), radii=(30.0, 20.0, 14.0),
        angles=(0.0, 0.0, 0.0), z_scale=7.0,
    )
    rings = np.vstack(wf["rings"])  # (N,3) in (z,y,x)
    # Discrete ring sampling may fall just short of the exact extreme, never past
    # it, so the bound is [r - small, r]. tol covers 48-point spacing.
    tol = 0.2
    # x extent = 100 ± 30
    assert 130.0 - tol <= rings[:, 2].max() <= 130.0 + 1e-6
    assert 70.0 - 1e-6 <= rings[:, 2].min() <= 70.0 + tol
    # y extent = 200 ± 20
    assert 220.0 - tol <= rings[:, 1].max() <= 220.0 + 1e-6
    # z index extent = (70 ± 14)/7 = [8, 12]
    assert 12.0 - tol <= rings[:, 0].max() <= 12.0 + 1e-6
    assert 8.0 - 1e-6 <= rings[:, 0].min() <= 8.0 + tol


def test_center_napari_zscale():
    c = geometry.center_napari((100.0, 200.0, 70.0), z_scale=7.0)
    assert np.allclose(c, [10.0, 200.0, 100.0])


# ---- scene (pure, no napari) ----

def test_build_scene_counts_and_reveal():
    from cell_viz import scene as scene_mod

    run = _smoke_run()
    curr = readers.load_frame(run, 86)
    prev = readers.load_frame(run, 85)
    s = scene_mod.build_scene(curr, prev, z_scale=7.0)
    # one silhouette outline per cell (no rainbow rings / axes)
    assert len(s.ring_paths) == len(curr.cells)
    # all lumen centers drawn
    assert s.lumen_points.shape[0] == len(curr.lumen_centers)
    # committed splits -> 1 ghost silhouette + 2 links each
    events = lineage.committed_splits(prev, curr)
    assert len(s.ghost_paths) == len(events)
    assert len(s.link_paths) == 2 * len(events)
    # reveal limits cells
    s5 = scene_mod.build_scene(curr, prev, z_scale=7.0, reveal=5)
    assert len(s5.ring_paths) == 5


# ---- brightness (checkpoint) + synth ----

def test_checkpoint_brightness_attached():
    run = _smoke_run()
    cells = readers.load_frame(run, 86).cells
    # brightness must be populated from the checkpoint; real cells near ~0.98,
    # and at least one dimmer cell exists (e.g. 1_310 ~0.74 in the smoke run).
    brights = [c.brightness for c in cells if not c.is_trash]
    assert brights and all(0.0 < b <= 1.0 for b in brights)
    assert min(brights) < 0.95, "expected at least one dim cell from checkpoint"


def test_projected_silhouette_axis_aligned():
    # axis-aligned: xy projection ellipse extent = a in x, b in y.
    xy = geometry.projected_silhouette_xy(
        center=(100.0, 200.0, 70.0), radii=(30.0, 20.0, 14.0), angles=(0, 0, 0), n=64
    )
    assert np.isclose(xy[:, 0].max() - xy[:, 0].min(), 60.0, atol=0.5)
    assert np.isclose(xy[:, 1].max() - xy[:, 1].min(), 40.0, atol=0.5)


def test_synth_projection_fills_brightness():
    from cell_viz.model import Cell
    from cell_viz.synth import synth_projection_2d

    cell = Cell("c", x=50.0, y=40.0, z=70.0, a_radius=10.0, b_radius=10.0,
                c_radius=10.0, theta_x=0, theta_y=0, theta_z=0, is_trash=False,
                brightness=0.8)
    img = synth_projection_2d([cell], (80, 100))
    assert np.isclose(img.max(), 0.8, atol=1e-6)
    assert np.isclose(img[40, 50], 0.8, atol=1e-6)  # center filled
    assert img[0, 0] == 0.0                          # far corner empty
