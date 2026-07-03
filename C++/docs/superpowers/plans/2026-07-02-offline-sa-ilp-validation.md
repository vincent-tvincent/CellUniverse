# Offline SA-Route Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove offline (Python only, no C++) that SA-style *generation* of temporally-primed, off-menu hypotheses (propagate / birth / image-driven split) lets the windowed ILP recover the known late-split divisions (f4/f7) that the committed tracker menu structurally lacks — the decisive gate for the whole shape+ILP design.

**Architecture:** Reuse the existing `scripts/ilp_proto/windowed_ilp_proto.py` (the CBC/PuLP windowed ILP core + `Cell` geometry + loaders). Add two modules: `sa_hypotheses.py` (the SA move generators — residual map, birth, propagate, image-driven split) and `offline_validation.py` (the experiment: build an SA-enriched hypothesis pool over a 3-frame window, solve, and compare the ILP-implied division frame to GT with vs without SA hypotheses). Success = with SA hypotheses the ILP lands the division within ±1 of GT on cases where committed-only cannot.

**Tech Stack:** Python 3.11 in `.venv-eval` (numpy, scipy, scikit-learn, scikit-image, tifffile, pulp/CBC — all present). Tests via pytest (installed in Task 1).

## Global Constraints

- **Python only this plan** — no C++, no build. Label-free, CPU, no learned model (project rule).
- **Interpreter:** always `.venv-eval/bin/python` (run from `C++/`). Never system python.
- **Git is the maintainer's** (project rule): the agent NEVER runs `git add/commit/push`. Each task's final "Checkpoint" step means *verify the deliverable and report it ready*; the maintainer commits.
- **Reuse, don't duplicate:** import from `scripts/ilp_proto/windowed_ilp_proto.py` — `Cell`, `solve_window`, `Node`, `Conflict`, `iou`, `image_score`, `load_tracker`, `load_real`, `gt_divisions`, `tracker_lineage`, `build_lineage_window`, `split_parent`, `FOUNDER_GT`.
- **Data (verified present):** run CSV `outputs/Yiding_1~171_VISUAL_TIF/resume_194/merged_0-194.csv`; real volumes `outputs/Yiding_1~171_VISUAL_TIF/real/{N}_real.tif` (frames 1–171, shape (239,512,708), CSV coords are in this space); GT `…/Fluo-N3DH-CE/01_GT/TRA/man_track.txt`.
- **Founder→GT map (verified):** `Cell type 1_1`=GT256, `1_2`=GT1, `1_3`=GT511, `1_4`=GT640.
- **Late-split target cases (verified):** GT1 & GT256 divide at **f4** (tracker at f5), GT511 divides at **f7** (tracker at f9). These are the cells whose committed CSV lacks a 2-cell hypothesis at the true division frame.
- **`Cell` fields:** `name,x,y,z,a,b,c,tx,ty,tz` (a=major,b=bRadius,c=minor). `Cell.voxels(scale=1.0)` returns a set of `(z,y,x)`. `image_score(vox, vol, bg, norm)` = summed signed `(intensity-bg)/norm`.

---

### Task 1: Test scaffold + smoke import

**Files:**
- Create: `scripts/ilp_proto/tests/__init__.py` (empty)
- Create: `scripts/ilp_proto/tests/test_smoke.py`

**Interfaces:**
- Consumes: existing `windowed_ilp_proto` module symbols.
- Produces: a working `pytest` setup all later tasks use; confirms imports + data paths.

- [ ] **Step 1: Install pytest**

Run: `.venv-eval/bin/pip install pytest`
Expected: `Successfully installed pytest-...`

- [ ] **Step 2: Write the smoke test**

Create `scripts/ilp_proto/tests/__init__.py` (empty file) and `scripts/ilp_proto/tests/test_smoke.py`:

```python
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import windowed_ilp_proto as W


def test_imports_and_data_present():
    assert hasattr(W, "solve_window") and hasattr(W, "Cell")
    assert os.path.exists(W.RUN_CSV), W.RUN_CSV
    assert os.path.exists(W.GT_TRACK), W.GT_TRACK
    vol, used = W.load_real(5)
    assert vol is not None and vol.ndim == 3


def test_founder_map_and_late_cases():
    # the cells whose committed CSV lacks a 2-cell hyp at the true division frame
    assert W.FOUNDER_GT["Cell type 1_2"] == ("B", 1)     # GT1 divides f4
    assert W.FOUNDER_GT["Cell type 1_3"] == ("C", 511)   # GT511 divides f7
```

- [ ] **Step 3: Run the smoke test**

Run: `cd /Users/jihangli/MCS/3D_Cell_Tracking/CellUniverse/C++ && .venv-eval/bin/python -m pytest scripts/ilp_proto/tests/test_smoke.py -v`
Expected: 2 passed.

- [ ] **Step 4: Checkpoint**

Verify both tests pass; report Task 1 ready for the maintainer to commit (`scripts/ilp_proto/tests/`).

---

### Task 2: Residual map + birth candidates

**Files:**
- Create: `scripts/ilp_proto/sa_hypotheses.py`
- Test: `scripts/ilp_proto/tests/test_sa_hypotheses.py`

> **Scope note:** `residual_map` + `birth_candidates` are the *birth* move-set primitives (spec §3.7). This plan builds and unit-tests them (they are part of "the SA move set"), but the decisive late-split experiment (Task 5) is recovered by the *split* + *propagate* moves — the birth primitives are consumed later by the full hypothesis-pool assembly and the C++ port. This keeps the move set complete while the offline gate stays focused on the split/propagate path. (The other §3.7 moves — death, merge — are deferred to the C++ move-set plan.)

**Interfaces:**
- Consumes: `windowed_ilp_proto.Cell`.
- Produces:
  - `residual_map(vol, cells, bg, scale=1.2) -> np.ndarray` — `(vol-bg)` clipped ≥0 with voxels inside any cell zeroed (unexplained signal).
  - `birth_candidates(residual, cell_radius, min_total=2000.0, max_n=4) -> list[Cell]` — one `Cell` per bright residual blob (isotropic radius = `cell_radius`, name `birth_i`).

- [ ] **Step 1: Write the failing test**

Create `scripts/ilp_proto/tests/test_sa_hypotheses.py`:

```python
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import windowed_ilp_proto as W
import sa_hypotheses as SA


def _cell(name, x, y, z, r=6.0):
    return W.Cell(name, x, y, z, r, r, r, 0.0, 0.0, 0.0)


def test_residual_zeros_explained_and_keeps_unexplained():
    vol = np.full((20, 40, 40), 10.0)
    vol[10, 10, 10] = 100.0      # explained (inside cell A)
    vol[10, 10, 30] = 100.0      # unexplained (no cell)
    res = SA.residual_map(vol, [_cell("A", 10, 10, 10, r=4)], bg=10.0, scale=1.2)
    assert res[10, 10, 10] == 0.0          # inside cell -> explained
    assert res[10, 10, 30] > 0.0           # far bright voxel -> residual


def test_birth_finds_unexplained_blob():
    res = np.zeros((20, 40, 40))
    res[8:13, 8:13, 28:33] = 50.0          # a bright residual blob near (x=30,y=10,z=10)
    births = SA.birth_candidates(res, cell_radius=5.0, min_total=1000.0, max_n=4)
    assert len(births) == 1
    b = births[0]
    assert abs(b.x - 30) < 3 and abs(b.y - 10) < 3 and abs(b.z - 10) < 3
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv-eval/bin/python -m pytest scripts/ilp_proto/tests/test_sa_hypotheses.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sa_hypotheses'`.

- [ ] **Step 3: Write the implementation**

Create `scripts/ilp_proto/sa_hypotheses.py`:

```python
"""SA move generators for the offline validation (propagate / birth / split)."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy import ndimage
import windowed_ilp_proto as W


def residual_map(vol, cells, bg, scale=1.2):
    """(vol-bg) clipped >=0, with voxels inside any cell zeroed = unexplained signal."""
    res = np.clip(vol.astype(float) - bg, 0.0, None)
    Z, Y, X = vol.shape
    for c in cells:
        for (z, y, x) in c.voxels(scale):
            if 0 <= z < Z and 0 <= y < Y and 0 <= x < X:
                res[z, y, x] = 0.0
    return res


def birth_candidates(residual, cell_radius, min_total=2000.0, max_n=4):
    """One Cell per connected bright residual blob with enough total signal."""
    thr = max(residual.max() * 0.3, 1.0)
    mask = residual >= thr
    lab, n = ndimage.label(mask)
    out = []
    for i in range(1, n + 1):
        sel = lab == i
        total = float(residual[sel].sum())
        if total < min_total:
            continue
        zz, yy, xx = np.nonzero(sel)
        out.append((total, W.Cell(f"birth_{i}", float(xx.mean()), float(yy.mean()),
                                  float(zz.mean()), cell_radius, cell_radius, cell_radius,
                                  0.0, 0.0, 0.0)))
    out.sort(key=lambda t: t[0], reverse=True)
    return [c for _, c in out[:max_n]]
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv-eval/bin/python -m pytest scripts/ilp_proto/tests/test_sa_hypotheses.py -v`
Expected: 2 passed.

- [ ] **Step 5: Checkpoint**

Report Task 2 ready (`sa_hypotheses.py`, `tests/test_sa_hypotheses.py`).

---

### Task 3: Propagate + image-driven split generators

**Files:**
- Modify: `scripts/ilp_proto/sa_hypotheses.py` (append)
- Test: `scripts/ilp_proto/tests/test_sa_hypotheses.py` (append)

**Interfaces:**
- Consumes: `windowed_ilp_proto.Cell`; `sklearn.mixture.GaussianMixture`.
- Produces:
  - `propagate_cell(prev, vol, bg, scale=1.3) -> Cell` — prev cell re-centered on the intensity-weighted centroid of bright voxels in its footprint (radii/rotation kept), name `prop_<prev.name>`.
  - `split_region_em(cell, vol, bg, scale=1.5, seed=0) -> list[Cell]` — EM(K=2) on intensity-sampled voxel coords → two daughter `Cell`s at the component means (radii = parent×0.7, theta = parent theta), names `<cell.name>0` / `<cell.name>1`.

- [ ] **Step 1: Write the failing test (append)**

Append to `scripts/ilp_proto/tests/test_sa_hypotheses.py`:

```python
def test_propagate_recenters_on_signal():
    vol = np.full((20, 40, 40), 10.0)
    vol[10, 12, 22] = 200.0                # signal offset from the prev center
    prev = _cell("P", 20, 12, 10, r=8)     # footprint covers the bright voxel
    prop = SA.propagate_cell(prev, vol, bg=10.0, scale=1.5)
    assert prop.name == "prop_P"
    assert abs(prop.x - 22) < 4 and abs(prop.z - 10) < 4   # pulled toward signal


def test_split_em_separates_two_lobes():
    vol = np.full((20, 60, 60), 10.0)
    vol[10, 20, 18:24] = 150.0             # lobe 1 near x=21
    vol[10, 20, 38:44] = 150.0             # lobe 2 near x=41
    parent = _cell("Q", 31, 20, 10, r=16)  # covers both lobes
    d = SA.split_region_em(parent, vol, bg=10.0, seed=0)
    assert len(d) == 2
    xs = sorted(c.x for c in d)
    assert xs[0] < 28 and xs[1] > 34       # one daughter per lobe
    assert {c.name for c in d} == {"Q0", "Q1"}
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv-eval/bin/python -m pytest scripts/ilp_proto/tests/test_sa_hypotheses.py -k "propagate or split_em" -v`
Expected: FAIL with `AttributeError: module 'sa_hypotheses' has no attribute 'propagate_cell'`.

- [ ] **Step 3: Write the implementation (append)**

Append to `scripts/ilp_proto/sa_hypotheses.py`:

```python
from sklearn.mixture import GaussianMixture


def _bright_voxels(cell, vol, bg, scale):
    """(coords Nx3 as (x,y,z), weights) for bright voxels inside cell footprint."""
    Z, Y, X = vol.shape
    pts, wts = [], []
    for (z, y, x) in cell.voxels(scale):
        if 0 <= z < Z and 0 <= y < Y and 0 <= x < X:
            w = float(vol[z, y, x]) - bg
            if w > 0:
                pts.append((x, y, z)); wts.append(w)
    return np.array(pts, float), np.array(wts, float)


def propagate_cell(prev, vol, bg, scale=1.3):
    pts, wts = _bright_voxels(prev, vol, bg, scale)
    if len(pts) == 0:
        return W.Cell("prop_" + prev.name, prev.x, prev.y, prev.z, prev.a, prev.b,
                      prev.c, prev.tx, prev.ty, prev.tz)
    c = np.average(pts, axis=0, weights=wts)
    return W.Cell("prop_" + prev.name, float(c[0]), float(c[1]), float(c[2]),
                  prev.a, prev.b, prev.c, prev.tx, prev.ty, prev.tz)


def split_region_em(cell, vol, bg, scale=1.5, seed=0, n_sample=600):
    pts, wts = _bright_voxels(cell, vol, bg, scale)
    if len(pts) < 6:
        return [cell]
    rng = np.random.default_rng(seed)
    p = wts / wts.sum()
    idx = rng.choice(len(pts), size=min(n_sample, len(pts)), replace=True, p=p)
    gm = GaussianMixture(2, covariance_type="full", reg_covar=1e-2,
                         n_init=2, random_state=seed).fit(pts[idx])
    out = []
    for i, m in enumerate(gm.means_):
        out.append(W.Cell(cell.name + str(i), float(m[0]), float(m[1]), float(m[2]),
                          cell.a * 0.7, cell.b * 0.7, cell.c * 0.7,
                          cell.tx, cell.ty, cell.tz))
    return out
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv-eval/bin/python -m pytest scripts/ilp_proto/tests/test_sa_hypotheses.py -v`
Expected: 4 passed.

- [ ] **Step 5: Checkpoint**

Report Task 3 ready.

---

### Task 4: SA-enriched window solve

**Files:**
- Create: `scripts/ilp_proto/offline_validation.py`
- Test: `scripts/ilp_proto/tests/test_offline_validation.py`

**Interfaces:**
- Consumes: `windowed_ilp_proto` (`load_tracker`, `load_real`, `Node`, `Conflict`, `iou`, `image_score`, `solve_window`, `merged_cell`); `sa_hypotheses` (`split_region_em`, `propagate_cell`).
- Produces:
  - `sa_window(byframe, parent_name, div_frame, w_half=1, use_sa=True) -> (fs, per_frame, res)` — like `build_lineage_window` but when `use_sa=True` the **1-cell hypothesis is the temporally-primed `propagate_cell(parent)`** and the **2-cell hypothesis is the image-driven `split_region_em(parent)`**; when `use_sa=False` the 2-cell hypothesis is absent (committed-only menu). Returns window frames, per-frame `{merge, split}` nodes, and the `solve_window` result. (Param is `w_half`, NOT `W` — `W` is the imported module alias.)
  - `implied_division_frame(fs, per_frame, res) -> int|None` — first frame whose 2-cell nodes are all selected.

- [ ] **Step 1: Write the failing test**

Create `scripts/ilp_proto/tests/test_offline_validation.py`:

```python
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import windowed_ilp_proto as W
import offline_validation as OV


def test_sa_window_runs_and_returns_frames():
    byframe = W.load_tracker()
    fs, per_frame, res = OV.sa_window(byframe, "Cell type 1_2", div_frame=5, w_half=1, use_sa=True)
    assert len(fs) >= 2
    assert res.status == "Optimal"
    d = OV.implied_division_frame(fs, per_frame, res)
    assert d is None or d in fs
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv-eval/bin/python -m pytest scripts/ilp_proto/tests/test_offline_validation.py -v 2>&1 | grep -v TIFFReadDirectory`
Expected: FAIL with `ModuleNotFoundError: No module named 'offline_validation'`.

- [ ] **Step 3: Write the implementation**

Create `scripts/ilp_proto/offline_validation.py`:

```python
"""Decisive offline test: does SA-generated (temporally-primed propagate + image-
driven split) hypothesis generation let the windowed ILP recover late-split
divisions the committed per-frame menu lacks?"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import windowed_ilp_proto as W
import sa_hypotheses as SA

NORM = 1000.0


def sa_window(byframe, parent_name, div_frame, w_half=1, use_sa=True):
    lo, hi = div_frame - 1 - w_half, div_frame + w_half
    frames = [f for f in range(lo, hi + 1) if f in byframe]
    nodes, links, conflicts, per_frame = [], [], [], {}

    for f in frames:
        cells = byframe[f]
        vol, _ = W.load_real(f)
        bg = float(np.percentile(vol, 50)) if vol is not None else 0.0
        d0, d1 = cells.get(parent_name + "0"), cells.get(parent_name + "1")
        par = cells.get(parent_name)

        if d0 and d1:                       # committed daughters already exist
            two = [d0, d1]
            one = W.merged_cell(two)
        elif par and use_sa:                # SA: propagate 1-cell, image-split 2-cell
            two = SA.split_region_em(par, vol, bg, seed=f)
            one = SA.propagate_cell(par, vol, bg)
        elif par:                           # committed-only: 1-cell, NO 2-cell hyp
            two, one = [], par
        else:
            per_frame[f] = None
            continue

        vm = one.voxels()
        nM = W.Node(f"M_{f}", f, W.image_score(vm, vol, bg, NORM), vm)
        nodes.append(nM)
        s_nodes = []
        for i, cc in enumerate(two):
            vc = cc.voxels()
            nC = W.Node(f"S{i}_{f}", f, W.image_score(vc, vol, bg, NORM), vc)
            nodes.append(nC); s_nodes.append(nC)
            conflicts.append(W.Conflict([nM.nid, nC.nid]))
        per_frame[f] = {"merge": nM, "split": s_nodes}

    fs = [f for f in frames if per_frame.get(f)]
    for i in range(len(fs) - 1):
        A = [per_frame[fs[i]]["merge"]] + per_frame[fs[i]]["split"]
        B = [per_frame[fs[i + 1]]["merge"]] + per_frame[fs[i + 1]]["split"]
        for p in A:
            for q in B:
                w = W.iou(p.vox, q.vox)
                if w > 0.02:
                    links.append((p.nid, q.nid, w))
    res = W.solve_window(nodes, links, conflicts, fs)
    return fs, per_frame, res


def implied_division_frame(fs, per_frame, res):
    for f in fs:
        pf = per_frame[f]
        if pf and pf["split"] and all(n.nid in res.selected for n in pf["split"]):
            return f
    return None
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv-eval/bin/python -m pytest scripts/ilp_proto/tests/test_offline_validation.py -v 2>&1 | grep -v TIFFReadDirectory`
Expected: 1 passed.

- [ ] **Step 5: Checkpoint**

Report Task 4 ready.

---

### Task 5: Decisive experiment — GT recovery with vs without SA

**Files:**
- Modify: `scripts/ilp_proto/offline_validation.py` (append `main`)
- Test: `scripts/ilp_proto/tests/test_offline_validation.py` (append)

**Interfaces:**
- Consumes: `sa_window`, `implied_division_frame`; `windowed_ilp_proto.load_tracker`.
- Produces: `gt_recovery(parent_name, gt_div_frame) -> dict` with keys `sa_div`, `nosa_div`, `gt`, `recovered` (bool: `sa_div` within ±1 of GT while `nosa_div` is not).

- [ ] **Step 1: Write the failing test (append)**

Append to `scripts/ilp_proto/tests/test_offline_validation.py`:

```python
def test_gt_recovery_shape():
    r = OV.gt_recovery("Cell type 1_2", gt_div_frame=4)
    assert set(r) == {"sa_div", "nosa_div", "gt", "recovered"}
    assert r["gt"] == 4
    assert isinstance(r["recovered"], bool)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv-eval/bin/python -m pytest scripts/ilp_proto/tests/test_offline_validation.py -k gt_recovery -v 2>&1 | grep -v TIFFReadDirectory`
Expected: FAIL with `AttributeError: module 'offline_validation' has no attribute 'gt_recovery'`.

- [ ] **Step 3: Write the implementation (append)**

Append to `scripts/ilp_proto/offline_validation.py`:

```python
def gt_recovery(parent_name, gt_div_frame):
    byframe = W.load_tracker()
    fs_a, pf_a, res_a = sa_window(byframe, parent_name, gt_div_frame, use_sa=True)
    fs_n, pf_n, res_n = sa_window(byframe, parent_name, gt_div_frame, use_sa=False)
    sa_div = implied_division_frame(fs_a, pf_a, res_a)
    nosa_div = implied_division_frame(fs_n, pf_n, res_n)
    ok = (sa_div is not None and abs(sa_div - gt_div_frame) <= 1
          and (nosa_div is None or abs(nosa_div - gt_div_frame) > 1))
    return {"sa_div": sa_div, "nosa_div": nosa_div, "gt": gt_div_frame, "recovered": ok}


def main():
    # (parent, gt_div_frame). GT511 is late-by-2 (tracker f9) -> committed menu has
    # NO in-window 2-cell hyp = the decisive case. GT1/GT256 are late-by-1 (tracker
    # f5) -> within the +/-1 GT tolerance = controls (nothing to recover).
    cases = [("Cell type 1_3", 7), ("Cell type 1_2", 4), ("Cell type 1_1", 4)]
    print(f"{'parent':>16} {'GT':>3} {'noSA':>5} {'SA':>4} {'recov':>6}  note")
    n_wrong, n_recovered = 0, 0
    for parent, g in cases:
        r = gt_recovery(parent, g)
        committed_wrong = (r["nosa_div"] is None or abs(r["nosa_div"] - g) > 1)
        if committed_wrong:
            n_wrong += 1
            n_recovered += int(r["recovered"])
        note = "committed WRONG" if committed_wrong else "control (within +/-1)"
        print(f"{parent:>16} {r['gt']:>3} {str(r['nosa_div']):>5} "
              f"{str(r['sa_div']):>4} {str(r['recovered']):>6}  {note}")
    ok = n_wrong > 0 and n_recovered == n_wrong
    print(f"\nGATE: of {n_wrong} case(s) the committed menu got wrong, SA recovered "
          f"{n_recovered}. {'PASS -> SA generates the off-menu hypothesis the ILP needs; '
          'green-light Workstream L/S' if ok else 'FAIL -> report whether the image at the '
          'GT frame even supports 2 cells (GT-vs-image timing gap, per 0a)'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify the test passes**

Run: `.venv-eval/bin/python -m pytest scripts/ilp_proto/tests/test_offline_validation.py -v 2>&1 | grep -v TIFFReadDirectory`
Expected: all passed.

- [ ] **Step 5: Run the decisive experiment**

Run: `.venv-eval/bin/python scripts/ilp_proto/offline_validation.py 2>&1 | grep -v TIFFReadDirectory`
Expected: a table of 3 cases + a GATE line. Record the actual `noSA`/`SA`/`recovered` values (this is the experiment result, not a pass/fail of code).

- [ ] **Step 6: Checkpoint**

Report Task 5 ready and paste the GATE table output.

---

### Task 6: Fit-gap check (EM-GMM on the real image) + results writeup

**Files:**
- Create: `scripts/ilp_proto/fitgap_real.py`
- Create: `scripts/ilp_proto/tests/test_fitgap.py`
- Create: `scripts/ilp_proto/RESULTS_offline_sa.md`

**Interfaces:**
- Consumes: `windowed_ilp_proto.load_real`; the 5 SEG slices; `sa_hypotheses.split_region_em`.
- Produces: `RESULTS_offline_sa.md` capturing (a) the GT-recovery gate table from Task 5 and (b) the EM-on-real fit-gap number.

- [ ] **Step 1: Write the fit-gap probe test**

Create `scripts/ilp_proto/tests/test_fitgap.py`:

```python
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import fitgap_real as FG


def test_real_frame_has_signal_under_a_tracked_cell():
    # a tracked cell's footprint should capture above-background signal in real vol
    frac = FG.mean_signal_fraction(frame=5)
    assert 0.0 < frac <= 1.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv-eval/bin/python -m pytest scripts/ilp_proto/tests/test_fitgap.py -v 2>&1 | grep -v TIFFReadDirectory`
Expected: FAIL with `ModuleNotFoundError: No module named 'fitgap_real'`.

- [ ] **Step 3: Write the implementation**

Create `scripts/ilp_proto/fitgap_real.py`:

```python
"""Fit-gap probe: how much of a tracked cell's footprint signal does the committed
ellipse actually capture on the REAL preprocessed volume (0b's fit-gap on real data)."""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import windowed_ilp_proto as W


def mean_signal_fraction(frame):
    byframe = W.load_tracker()
    vol, _ = W.load_real(frame)
    bg = float(np.percentile(vol, 50))
    total_in, total_all = 0.0, 0.0
    Z, Y, X = vol.shape
    cells = list(byframe[frame].values())
    inside = set()
    for c in cells:
        inside |= c.voxels(1.0)
    for (z, y, x) in inside:
        if 0 <= z < Z and 0 <= y < Y and 0 <= x < X:
            total_in += max(float(vol[z, y, x]) - bg, 0.0)
    total_all = float(np.clip(vol.astype(float) - bg, 0.0, None).sum())
    return total_in / max(total_all, 1.0)


def main():
    for f in (5, 21, 78):
        print(f"frame {f}: fraction of above-bg signal captured by committed cells "
              f"= {mean_signal_fraction(f):.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify the test passes**

Run: `.venv-eval/bin/python -m pytest scripts/ilp_proto/tests/test_fitgap.py -v 2>&1 | grep -v TIFFReadDirectory`
Expected: 1 passed.

- [ ] **Step 5: Run both probes and write results**

Run: `.venv-eval/bin/python scripts/ilp_proto/fitgap_real.py 2>&1 | grep -v TIFFReadDirectory`
Then create `scripts/ilp_proto/RESULTS_offline_sa.md` with: the Task 5 GATE table (SA vs noSA division recovery on f4/f7), the fit-gap fractions, and a one-paragraph verdict — **does SA generation recover ≥2/3 late-split cases the committed menu could not? (gate for Workstream L/S)** and what the fit-gap fraction implies for the Chan-Vese/refinement need.

- [ ] **Step 6: Checkpoint**

Report Task 6 ready; paste `RESULTS_offline_sa.md` verdict.

---

## Decision gate (end of plan)

The denominator is only the cases where the **committed-only** menu is actually wrong (no in-window 2-cell hypothesis, or division off by ≥2 from GT). The decisive such case is **GT511** (`Cell type 1_3`, GT f7, tracker f9); GT1/GT256 are late-by-1 controls inside the ±1 tolerance (nothing to recover).

- **PASS** (SA recovers every committed-wrong case — at minimum GT511) → SA *generation* produces the temporally-primed, off-menu hypothesis the ILP needs and lands the division within ±1 of GT where the fixed menu cannot. The SA-route thesis holds; proceed to rollout step 2 (the zero-regression C++ `MetaballCell` scaffold) in a follow-up plan.
- **FAIL** (SA does not recover GT511) → report *why* from the Task 5 numbers: if the real image at f7 is not bimodal (SA split can't find 2 lobes), the tracker's f9 matches the image and the discrepancy is a GT-vs-image timing gap (the 0a finding), not a fixable generation gap — revisit the thesis before any C++.
