#!/usr/bin/env python3
"""
Phase 0a — Offline windowed-ILP prototype (Workstream L go/no-go).

Goal (from docs/plans/2026-07-02-shape-and-ilp-buildplan.md):
  Validate, WITHOUT any C++ change, whether a small sliding-window ILP that selects
  the globally consistent set of cell hypotheses can fix division-timing / merge
  errors that the current per-frame *greedy* commit gets wrong -- decided by GT.

What this file is:
  A self-contained reference implementation of the windowed selection ILP (CBC via
  PuLP) plus an honest offline harness. The ILP core (`solve_window`) is written to
  mirror the future C++ port (Phase L2): nodes = cell hypotheses, edges = temporal
  links, events = appear/disappear/divide, objective = detection + link weights minus
  event penalties, constraints = hypothesis disjointness + flow conservation.

Honesty constraints (important -- read before trusting any number):
  * IMAGE EVIDENCE is the tracker's own preprocessed `real/{N}_real.tif` volume
    (the exact image it fit against). We do NOT score hypotheses against GT masks;
    that would be circular (GT masks encode the division frame we are trying to test).
  * GT is used ONLY as truth: `man_track.txt` gives the true division frames.
  * Detection/link geometry uses the C++ ellipsoid convention exactly
    (R = Rz*Ry*Rx, local = R^T*(world-center), inside if sum((l/axis)^2)<=1;
     a=majorRadius, b=bRadius, c=minorRadius).

Offline scope / known limit:
  The committed CSV only stores the hypotheses the greedy logic KEPT. A *late* split
  (tracker stayed 1 cell where GT had 2) has no stored 2-cell hypothesis at the early
  frame; inventing one honestly needs an image 2-blob fit == Workstream S2/L1, not
  available from a committed CSV. So offline we can validate:
    (A) the ILP core is correct                     -> synthetic self-test
    (B) premature/over-splits (tracker > GT count)  -> 2-cell hyp exists in CSV,
        synthesize the 1-cell merge, let the ILP suppress the transient split.
  Late splits are reported but flagged as "needs L1/S2 image hypotheses".

Usage:
  .venv-eval/bin/python scripts/ilp_proto/windowed_ilp_proto.py --selftest
  .venv-eval/bin/python scripts/ilp_proto/windowed_ilp_proto.py --analyze
  .venv-eval/bin/python scripts/ilp_proto/windowed_ilp_proto.py --case 9   # window solve
"""
from __future__ import annotations
import argparse, csv, math, os, sys
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pulp

# ----------------------------------------------------------------------------
# Paths (relative to C++/)
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
CPP  = os.path.abspath(os.path.join(HERE, "..", ".."))
RUN_CSV  = os.path.join(CPP, "outputs/Yiding_1~171_VISUAL_TIF/resume_194/merged_0-194.csv")
REAL_DIR = os.path.join(CPP, "outputs/Yiding_1~171_VISUAL_TIF/real")
GT_TRACK = "/Users/jihangli/MCS/3D_Cell_Tracking/Fluo-N3DH-CE/01_GT/TRA/man_track.txt"

# Founder correspondence, established by matching the 4 f0 cells to GT centroids
# (xy within 1-2 px; z scale ~6.5x). Lets us print recognizable lineage labels
# instead of the raw "Cell type 1_N" names, and know each lineage's true GT.
FOUNDER_GT = {"Cell type 1_1": ("A", 256), "Cell type 1_2": ("B", 1),
              "Cell type 1_3": ("C", 511), "Cell type 1_4": ("D", 640)}

def readable(name: str) -> str:
    """Map 'Cell type 1_3100' -> 'C.100 (GT511-lineage)'. The digits after the
    founder token are the division path (0/1 at each split)."""
    for base, (letter, gt) in FOUNDER_GT.items():
        if name == base or name.startswith(base):
            path = name[len(base):]
            return f"{letter}.{path} (GT{gt}-lineage)" if path else f"{letter} (GT{gt})"
    return name

# ----------------------------------------------------------------------------
# Cell model (matches C++ Ellipsoid convention)
# ----------------------------------------------------------------------------
@dataclass
class Cell:
    name: str
    x: float; y: float; z: float
    a: float; b: float; c: float          # a=major, b=bRadius, c=minor
    tx: float; ty: float; tz: float       # theta_x/y/z

    @staticmethod
    def from_row(r: dict) -> "Cell":
        return Cell(r["name"],
                    float(r["x"]), float(r["y"]), float(r["z"]),
                    float(r["majorRadius"]), float(r["bRadius"]), float(r["minorRadius"]),
                    float(r["theta_x"]), float(r["theta_y"]), float(r["theta_z"]))

    def rot_T(self) -> np.ndarray:
        cx, sx = math.cos(self.tx), math.sin(self.tx)
        cy, sy = math.cos(self.ty), math.sin(self.ty)
        cz, sz = math.cos(self.tz), math.sin(self.tz)
        # R = Rz*Ry*Rx  (C++ generateInverseRotationMatrix)
        R = np.array([
            [cz*cy, cz*sy*sx - sz*cx, cz*sy*cx + sz*sx],
            [sz*cy, sz*sy*sx + cz*cx, sz*sy*cx - cz*sx],
            [-sy,   cy*sx,            cy*cx           ],
        ])
        return R.T  # inverse-rotate world->local

    def voxels(self, scale: float = 1.0):
        """Return integer (z,y,x) voxel coords inside this ellipsoid, as a set."""
        a, b, c = max(self.a*scale, 1e-3), max(self.b*scale, 1e-3), max(self.c*scale, 1e-3)
        RT = self.rot_T()
        maxr = max(a, b, c)
        out = set()
        x0, y0, z0 = self.x, self.y, self.z
        for zz in range(int(math.floor(z0-maxr)), int(math.ceil(z0+maxr))+1):
            for yy in range(int(math.floor(y0-maxr)), int(math.ceil(y0+maxr))+1):
                for xx in range(int(math.floor(x0-maxr)), int(math.ceil(x0+maxr))+1):
                    d = np.array([xx-x0, yy-y0, zz-z0])
                    lx, ly, lz = RT @ d
                    if (lx/a)**2 + (ly/b)**2 + (lz/c)**2 <= 1.0:
                        out.add((zz, yy, xx))
        return out


def merged_cell(cells, name="MERGED") -> Cell:
    """Synthetic 1-cell hypothesis: 'these cells are really ONE nucleus'.

    Sized like a single real cell (mean daughter radii scaled by 2^(1/3) for the
    combined volume), NOT an enclosing ellipsoid -- an enclosing blob would cover the
    dark gap and lose by construction, which is not an honest 1-vs-2 comparison."""
    cx = np.mean([c.x for c in cells]); cy = np.mean([c.y for c in cells]); cz = np.mean([c.z for c in cells])
    k = 2.0 ** (1.0 / 3.0) if len(cells) > 1 else 1.0
    ra = k * np.mean([c.a for c in cells])
    rb = k * np.mean([c.b for c in cells])
    rc = k * np.mean([c.c for c in cells])
    tx = float(np.mean([c.tx for c in cells])); ty = float(np.mean([c.ty for c in cells]))
    tz = float(np.mean([c.tz for c in cells]))
    return Cell(name, cx, cy, cz, ra, rb, rc, tx, ty, tz)


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------
def iou(va: set, vb: set) -> float:
    if not va or not vb:
        return 0.0
    inter = len(va & vb)
    if inter == 0:
        return 0.0
    return inter / len(va | vb)


def image_score(vox: set, vol: np.ndarray, bg: float, norm: float = 1.0) -> float:
    """Honest per-hypothesis detection score against the real (preprocessed) volume.

    score = sum(intensity - bg) over occupied voxels  (MDL-style explanatory score),
    optionally divided by `norm` to keep magnitudes O(10).
      * bright voxels (over a nucleus)  -> positive
      * dark voxels (background gap)    -> negative
    Consequences that make 1-vs-2 comparable and honest:
      * a too-big 1-cell blob that spans the dark gap between two nuclei eats negative
        voxels -> lower score.
      * a spurious 2nd daughter placed over background (over-split) adds mostly
        negative voxels -> does NOT raise the total, so over-splitting is penalised.
    Summing node weights in the ILP then equals the signed signal over the selected
    union (minus double-count on overlap, which the geometry keeps small)."""
    if not vox:
        return 0.0
    Z, Y, X = vol.shape
    acc = 0.0
    for (z, y, x) in vox:
        if 0 <= z < Z and 0 <= y < Y and 0 <= x < X:
            acc += float(vol[z, y, x]) - bg
    return acc / norm


# ----------------------------------------------------------------------------
# Windowed selection ILP  (the reusable core -- mirrors the future C++ port)
# ----------------------------------------------------------------------------
@dataclass
class Node:
    nid: str
    frame: int
    weight: float                       # detection score (image evidence)
    vox: set = field(default=None, repr=False)

@dataclass
class Conflict:
    """Mutually-exclusive hypotheses at one frame (<=1 may be selected)."""
    members: list

@dataclass
class WindowResult:
    status: str
    objective: float
    selected: set                       # node ids
    links: list                         # (p,q)
    divisions: list                     # node ids that divide
    appears: list
    disappears: list


def solve_window(nodes, links, conflicts, frames,
                 c_appear=1.0, c_disappear=1.0, c_div=0.5,
                 penalize_boundary=False, msg=False) -> WindowResult:
    """
    nodes:     list[Node]
    links:     list[(pid, qid, weight)]  temporal edges, frame(p)+1 == frame(q)
    conflicts: list[Conflict]            disjoint hypothesis groups
    frames:    sorted list of frame indices in the window

    Ultrack-style flow ILP:
      max  sum w_p y_p + sum w_e x_e  -  c_a*appear - c_d*disappear - c_div*division
      s.t. appear_p + sum_in x = y_p                    (one parent or appearance)
           disappear_p + sum_out x = y_p + division_p   (one child, or two if division)
           division_p <= y_p
           sum_{p in C} y_p <= 1   for each conflict C
    Boundary appear/disappear (first/last frame in window) are free unless
    penalize_boundary -- window edges are not real births/deaths.
    """
    node = {n.nid: n for n in nodes}
    fmin, fmax = min(frames), max(frames)
    P = pulp.LpProblem("window", pulp.LpMaximize)

    y = {n.nid: pulp.LpVariable(f"y_{n.nid}", cat="Binary") for n in nodes}
    x = {(p, q): pulp.LpVariable(f"x_{p}__{q}", cat="Binary") for (p, q, w) in links}
    ap = {n.nid: pulp.LpVariable(f"ap_{n.nid}", cat="Binary") for n in nodes}
    dp = {n.nid: pulp.LpVariable(f"dp_{n.nid}", cat="Binary") for n in nodes}
    dv = {n.nid: pulp.LpVariable(f"dv_{n.nid}", cat="Binary") for n in nodes}

    lw = {(p, q): w for (p, q, w) in links}
    ins = defaultdict(list); outs = defaultdict(list)
    for (p, q, w) in links:
        outs[p].append(q); ins[q].append(p)

    # objective
    obj = []
    for n in nodes:
        obj.append(n.weight * y[n.nid])
    for (p, q, w) in links:
        obj.append(w * x[(p, q)])
    for n in nodes:
        at_start = (n.frame == fmin); at_end = (n.frame == fmax)
        ca = 0.0 if (at_start and not penalize_boundary) else c_appear
        cd = 0.0 if (at_end and not penalize_boundary) else c_disappear
        obj.append(-ca * ap[n.nid])
        obj.append(-cd * dp[n.nid])
        obj.append(-c_div * dv[n.nid])
    P += pulp.lpSum(obj)

    # flow constraints
    for n in nodes:
        i = n.nid
        P += ap[i] + pulp.lpSum(x[(p, i)] for p in ins[i]) == y[i], f"in_{i}"
        P += dp[i] + pulp.lpSum(x[(i, q)] for q in outs[i]) == y[i] + dv[i], f"out_{i}"
        P += dv[i] <= y[i], f"divsel_{i}"

    # disjointness
    for k, C in enumerate(conflicts):
        P += pulp.lpSum(y[m] for m in C.members) <= 1, f"conf_{k}"

    P.solve(pulp.PULP_CBC_CMD(msg=1 if msg else 0))
    st = pulp.LpStatus[P.status]
    sel = {i for i in y if y[i].value() and y[i].value() > 0.5}
    lk = [(p, q) for (p, q) in x if x[(p, q)].value() and x[(p, q)].value() > 0.5]
    dvs = [i for i in dv if dv[i].value() and dv[i].value() > 0.5]
    aps = [i for i in ap if ap[i].value() and ap[i].value() > 0.5 and i in sel]
    dps = [i for i in dp if dp[i].value() and dp[i].value() > 0.5 and i in sel]
    return WindowResult(st, pulp.value(P.objective), sel, lk, dvs, aps, dps)


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_tracker():
    rows = list(csv.DictReader(open(RUN_CSV)))
    byframe = defaultdict(dict)          # frame -> name -> Cell
    for r in rows:
        f = int(r["file"][1:4])
        byframe[f][r["name"]] = Cell.from_row(r)
    return byframe


def tracker_lineage(byframe):
    """Reconstruct tracker divisions from name suffixes: parent P -> P+'0', P+'1'."""
    appear = defaultdict(list)
    for f in sorted(byframe):
        for n in byframe[f]:
            appear[n].append(f)
    allnames = set(appear)
    divs = []
    for n in allnames:
        ch = [c for c in allnames if c == n + "0" or c == n + "1"]
        if len(ch) >= 2:
            cf = min(min(appear[c]) for c in ch)
            divs.append((cf, n, sorted(ch)))
    divs.sort()
    return divs, appear


def gt_divisions():
    gt = [list(map(int, l.split())) for l in open(GT_TRACK) if l.strip()]
    children = defaultdict(list); begin = {}; end = {}
    for L, B, E, P in gt:
        begin[L] = B; end[L] = E
        if P != 0:
            children[P].append(L)
    divs = [(end[p] + 1, p, ch) for p, ch in children.items() if len(ch) >= 2]
    divs.sort()
    def count(f):
        return sum(1 for L, B, E, P in gt if B <= f <= E)
    return divs, count


def load_real(frame_idx):
    """Load the preprocessed 'real' volume aligned to CSV frame `frame_idx`.
    Auto-detect the real/<M> index (M==frame_idx or frame_idx+1) by which choice
    puts more signal under an arbitrary reference -- resolved by caller if needed."""
    import tifffile
    for cand in (frame_idx, frame_idx + 1):
        p = os.path.join(REAL_DIR, f"{cand}_real.tif")
        if os.path.exists(p):
            return tifffile.imread(p), cand
    return None, None


# ----------------------------------------------------------------------------
# Experiments
# ----------------------------------------------------------------------------
def selftest():
    """Prove the ILP core: a lineage that truly divides at window-frame 2, with a
    single-frame NOISE spike that makes per-frame greedy pick '1 cell' at frame 3
    (a flicker / un-division). The ILP must keep the division at f2 and stay 2-cell."""
    print("=== ILP core self-test ===")
    frames = [0, 1, 2, 3, 4]
    nodes = []; links = []; conflicts = []
    # frame 0,1: one real parent (high score). frames 2..4: two daughters (high),
    # plus a competing 1-cell merged hypothesis (low, EXCEPT spiked at f3).
    def add(nid, f, w, vox):
        nodes.append(Node(nid, f, w, vox)); return nid
    # parent
    add("P0", 0, 10.0, {(0,)}); add("P1", 1, 10.0, {(1,)})
    # daughters at 2,3,4
    for f in (2, 3, 4):
        add(f"A{f}", f, 8.0, {(f, "a")})
        add(f"B{f}", f, 8.0, {(f, "b")})
    # merged 1-cell hypotheses at 2,3,4 (normally worse; spike at f3)
    add("M2", 2, 6.0, {(2,)})
    add("M3", 3, 20.0, {(3,)})   # <-- noise spike: greedy prefers merge here
    add("M4", 4, 6.0, {(4,)})
    # conflicts: merged Mf excludes each daughter at frame f
    for f in (2, 3, 4):
        conflicts.append(Conflict([f"M{f}", f"A{f}"]))
        conflicts.append(Conflict([f"M{f}", f"B{f}"]))
    # links (temporal): parent chain, division P1->A2,B2, daughter chains, merge chain
    def L(p, q, w): links.append((p, q, w))
    L("P0", "P1", 9.0)
    L("P1", "A2", 7.0); L("P1", "B2", 7.0)   # division
    L("P1", "M2", 7.0)
    L("A2", "A3", 7.0); L("B2", "B3", 7.0)
    L("A2", "M3", 4.0); L("B2", "M3", 4.0)
    L("M2", "M3", 6.0); L("M2", "A3", 4.0); L("M2", "B3", 4.0)
    L("A3", "A4", 7.0); L("B3", "B4", 7.0)
    L("M3", "M4", 6.0); L("M3", "A4", 4.0); L("M3", "B4", 4.0)
    L("A3", "M4", 4.0); L("B3", "M4", 4.0)

    # greedy per-frame argmax over the conflict groups
    greedy = {}
    for f in (2, 3, 4):
        opts = {"merge": next(n.weight for n in nodes if n.nid == f"M{f}"),
                "split": next(n.weight for n in nodes if n.nid == f"A{f}")
                         + next(n.weight for n in nodes if n.nid == f"B{f}")}
        greedy[f] = max(opts, key=opts.get)
    print("  greedy per-frame choice:", greedy,
          "  <- flicker at f3" if greedy[3] == "merge" else "")

    res = solve_window(nodes, links, conflicts, frames)
    ilp = {}
    for f in (2, 3, 4):
        ilp[f] = "split" if (f"A{f}" in res.selected and f"B{f}" in res.selected) else "merge"
    print(f"  ILP status={res.status} obj={res.objective:.1f}")
    print("  ILP per-frame choice:  ", ilp)
    ok = all(ilp[f] == "split" for f in (2, 3, 4)) and greedy[3] == "merge"
    print("  RESULT:", "PASS - ILP removed the flicker greedy fell for" if ok
          else "FAIL")
    return ok


def analyze():
    """Compare tracker (name-encoded) divisions to GT, list honest offline test cases."""
    byframe = load_tracker()
    tdivs, appear = tracker_lineage(byframe)
    gdivs, gcount = gt_divisions()
    print("=== tracker vs GT cell count (frames 0..40) ===")
    over = []; under = []
    for f in range(0, 41):
        tc = len(byframe.get(f, {})); gc = gcount(f)
        tag = ""
        if tc > gc: tag = "  OVER-split (offline-testable)"; over.append(f)
        elif tc < gc: tag = "  under/late (needs L1/S2 image hyp)"; under.append(f)
        print(f"  f{f:>2}: tracker={tc:>2} gt={gc:>2}{tag}")
    print(f"\n  over-split frames (2-cell hyp exists in CSV): {over}")
    print(f"  late frames (need image 2-blob hyp):          {under}")
    print(f"\n  tracker divisions: {len(tdivs)}   GT divisions(<=194): {len(gdivs)}")
    print("  first 8 tracker divisions:", [(f, p) for f, p, ch in tdivs[:8]])
    print("  first 8 GT divisions:     ", [(f, p) for f, p, ch in gdivs[:8]])
    return over, under


def build_lineage_window(byframe, parent_name, div_frame, W=1, vol_cache=None):
    """For a tracker division (parent -> parent0/parent1 at div_frame), build the
    per-frame 1-cell vs 2-cell hypotheses over [div_frame-1-W .. div_frame+W] and
    score them against the real volumes. Returns (nodes, links, conflicts, frames)."""
    import tifffile
    lo = div_frame - 1 - W
    hi = div_frame + W
    frames = [f for f in range(lo, hi + 1) if f in byframe]
    nodes = []; links = []; conflicts = []
    per_frame = {}   # f -> {'merge': Node, 'split':(Node,Node)}

    for f in frames:
        cells = byframe[f]
        # the two daughters if present this frame, else the parent
        d0 = cells.get(parent_name + "0"); d1 = cells.get(parent_name + "1")
        par = cells.get(parent_name)
        vol, used = load_real(f)
        bg = float(np.percentile(vol, 50)) if vol is not None else 0.0

        two = []
        if d0 and d1:
            two = [d0, d1]
        elif par:
            # synthesize a 2-cell split of the parent along its shortest axis
            two = split_parent(par)
        one_src = two if two else ([par] if par else [])
        if not one_src:
            per_frame[f] = None; continue
        one = merged_cell(one_src, name=f"M_{f}")

        NORM = 1000.0
        vm = one.voxels()
        nM = Node(f"M_{f}", f, image_score(vm, vol, bg, NORM), vm)
        nodes.append(nM)
        s_nodes = []
        for i, cc in enumerate(two):
            vc = cc.voxels()
            nC = Node(f"S{i}_{f}", f, image_score(vc, vol, bg, NORM), vc)
            nodes.append(nC); s_nodes.append(nC)
            conflicts.append(Conflict([nM.nid, nC.nid]))
        per_frame[f] = {"merge": nM, "split": s_nodes}

    # temporal links by voxel IoU across consecutive frames
    fs = [f for f in frames if per_frame.get(f)]
    for i in range(len(fs) - 1):
        f, g = fs[i], fs[i + 1]
        A = [per_frame[f]["merge"]] + per_frame[f]["split"]
        B = [per_frame[g]["merge"]] + per_frame[g]["split"]
        for p in A:
            for q in B:
                w = iou(p.vox, q.vox)
                if w > 0.02:
                    links.append((p.nid, q.nid, w))
    return nodes, links, conflicts, fs, per_frame


def split_parent(par: Cell):
    """Seed two daughters along the parent's shortest local axis (C++ split geometry)."""
    RT = par.rot_T()          # world->local ; columns of R = rows of RT
    R = RT.T
    axes = [(par.a, R[:, 0]), (par.b, R[:, 1]), (par.c, R[:, 2])]
    length, d = min(axes, key=lambda t: t[0])
    d = d / (np.linalg.norm(d) + 1e-9)
    off = 0.5 * length * d
    ctr = np.array([par.x, par.y, par.z])
    out = []
    for s, nm in ((+1, "0"), (-1, "1")):
        p = ctr + s * off
        out.append(Cell(par.name + nm, p[0], p[1], p[2],
                        par.a * 0.7, par.b * 0.7, par.c * 0.7,
                        par.tx, par.ty, par.tz))
    return out


def run_case(div_frame, parent_name=None, W=1):
    byframe = load_tracker()
    tdivs, appear = tracker_lineage(byframe)
    gdivs, gcount = gt_divisions()
    if parent_name is None:
        cands = [(f, p, ch) for (f, p, ch) in tdivs if f == div_frame]
        if not cands:
            print(f"no tracker division committed at f{div_frame}; "
                  f"divisions near it: {[(f,p) for f,p,ch in tdivs if abs(f-div_frame)<=2]}")
            return
        parent_name = cands[0][1]
    print(f"=== window solve: parent {readable(parent_name)} "
          f"[{parent_name!r}] committed division at f{div_frame} ===")
    nodes, links, conflicts, fs, per_frame = build_lineage_window(byframe, parent_name, div_frame, W)
    print(f"  window frames: {fs}")
    print("  per-frame detection scores (merge vs split-sum):")
    for f in fs:
        pf = per_frame[f]
        ms = pf["merge"].weight
        ss = sum(n.weight for n in pf["split"])
        greedy = "split" if ss > ms else "merge"
        print(f"    f{f}: merge={ms:6.2f}  split={ss:6.2f}  greedy->{greedy}")
    res = solve_window(nodes, links, conflicts, fs)
    print(f"  ILP status={res.status} obj={res.objective:.2f}")
    print("  ILP per-frame choice:")
    ilp_div = None
    for f in fs:
        pf = per_frame[f]
        sp = all(n.nid in res.selected for n in pf["split"])
        mg = pf["merge"].nid in res.selected
        ch = "split" if sp else ("merge" if mg else "none")
        if sp and ilp_div is None:
            ilp_div = f
        print(f"    f{f}: {ch}")
    print(f"  ILP-implied division frame: {ilp_div}   tracker committed: {div_frame}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--case", type=int, help="tracker division frame to window-solve")
    ap.add_argument("--parent", type=str, default=None)
    ap.add_argument("-W", type=int, default=1)
    args = ap.parse_args()
    if args.selftest:
        ok = selftest(); sys.exit(0 if ok else 1)
    if args.analyze:
        analyze(); return
    if args.case is not None:
        run_case(args.case, args.parent, args.W); return
    ap.print_help()


if __name__ == "__main__":
    main()
