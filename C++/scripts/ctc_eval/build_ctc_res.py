#!/usr/bin/env python3
"""Convert a CellUniverse FinalLineageTree CSV into a Cell Tracking Challenge
(CTC) RES folder, and build a trimmed/renumbered GT folder covering exactly the
frames we tracked, so that py-ctcmetrics can compute SEG / DET / TRA.

Why this script exists
----------------------
CellUniverse represents each cell as a rotated tri-axial ellipsoid (center +
3 radii + 3 Euler angles), exported per frame in a CSV. The CTC measures need
*integer label volumes* (one connected component per cell, the label stable
across frames) plus a `res_track.txt` lineage file. This script rasterizes the
ellipsoids into label volumes that match the C++ renderer's geometry exactly,
and derives the lineage from the cell-name encoding.

Geometry (must match C++ src/Ellipsoid.cpp)
-------------------------------------------
* Working grid is isotropic with the z axis upscaled by `z_scaling` (=7 for the
  embryo config). A model z of ~110 therefore maps to GT slice ~110/7 ≈ 16.
* Membership test (identical to scanEllipsoidSlice): with R = Rz·Ry·Rx and
  local = R^T · (world - center),  lx²/a² + ly²/b² + lz²/c² ≤ 1.
  a = majorRadius, b = bRadius (oblate fallback a when 0), c = minorRadius.
* A GT voxel slice s (0-based, original resolution) is sampled at expanded
  z = s*z_scaling + (z_scaling-1)/2  (block midpoint).

Lineage (cell-name encoding)
----------------------------
A cell keeps its name across frames until it divides; a division appends one
digit (e.g. `Cell type 1_4` -> `Cell type 1_40` / `_41`). So each unique name is
one CTC track and its parent is the name with the last character stripped.

Frame mapping
-------------
CSV file `t00k.tif` is GT frame k. We evaluate the overlap k = 1..171 and
renumber both RES and GT to 0..170 (CTC is 0-based and py-ctcmetrics pairs GT
and RES purely by sorted position, asserting equal counts).
"""
import argparse
import csv
import os
import re
import shutil
from collections import defaultdict

import numpy as np
import tifffile


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True, help="FinalLineageTree CSV")
    p.add_argument("--gt", required=True, help="Source CTC GT dir (has TRA/ and SEG/)")
    p.add_argument("--out", required=True, help="Output eval dataset dir")
    p.add_argument("--seq", default="01")
    p.add_argument("--z-scaling", type=float, default=7.0)
    p.add_argument("--first", type=int, default=1, help="first original frame to keep")
    p.add_argument("--last", type=int, default=171, help="last original frame to keep")
    p.add_argument("--shape", default=None,
                   help="Z,Y,X of label volume; default = taken from GT man_track000.tif")
    # ----- mask mode -----
    p.add_argument("--mask-mode", choices=["ellipsoid", "signal"], default="ellipsoid",
                   help="ellipsoid: rasterize the raw analytic ellipsoid (detection-shaped). "
                        "signal: keep only bright voxels inside a scaled ellipsoid so the mask "
                        "hugs the real nucleus (improves SEG; needs --raw-dir).")
    p.add_argument("--raw-dir", default=None,
                   help="Raw image dir (e.g. data/input/embryo_data) for --mask-mode signal")
    p.add_argument("--seg-scale", type=float, default=1.6,
                   help="signal mode: scale the ellipsoid bound (lets the signal define the "
                        "shape and reach off-center z-slices)")
    p.add_argument("--seg-thresh", default="otsu",
                   help="signal mode foreground threshold: 'otsu', 'pXX' (percentile, e.g. p92), "
                        "or an absolute intensity value")
    p.add_argument("--skip-gt", action="store_true",
                   help="do not (re)build the trimmed GT (reuse an existing one)")
    return p.parse_args()


def foreground_threshold(raw, spec):
    """Resolve the --seg-thresh spec to an absolute intensity for one raw volume."""
    if spec == "otsu":
        from skimage.filters import threshold_otsu
        return float(threshold_otsu(raw))
    if spec.startswith("p"):
        return float(np.percentile(raw, float(spec[1:])))
    return float(spec)


def frame_index(fname):
    return int(re.search(r"t(\d+)\.tif", fname).group(1))


def rotation_T(tx, ty, tz):
    """R^T for R = Rz·Ry·Rx (rows of R^T = columns of R). Matches C++."""
    cx, sx = np.cos(tx), np.sin(tx)
    cy, sy = np.cos(ty), np.sin(ty)
    cz, sz = np.cos(tz), np.sin(tz)
    R00 = cz * cy;            R01 = cz * sy * sx - sz * cx; R02 = cz * sy * cx + sz * sx
    R10 = sz * cy;            R11 = sz * sy * sx + cz * cx; R12 = sz * sy * cx - cz * sx
    R20 = -sy;                R21 = cy * sx;                R22 = cy * cx
    # local = R^T d  ->  lx = R00 dx + R10 dy + R20 dz, etc.
    return (R00, R10, R20, R01, R11, R21, R02, R12, R22)


def rasterize_frame(cells, shape, z_scaling):
    """cells: list of dicts with x,y,z,a,b,c,tx,ty,tz,label.
    Returns uint16 (Z,Y,X). Overlaps resolved by smallest ellipsoid value
    (each voxel assigned to the cell it is most deeply inside)."""
    Z, Y, X = shape
    labels = np.zeros(shape, dtype=np.uint16)
    bestval = np.full(shape, np.inf, dtype=np.float32)
    zoff = (z_scaling - 1.0) / 2.0
    for cd in cells:
        cx, cy, cz = cd["x"], cd["y"], cd["z"]
        a, b, c = cd["a"], cd["b"], cd["c"]
        tx, ty, tz = cd["tx"], cd["ty"], cd["tz"]
        rmax = max(a, b, c)
        x0 = max(0, int(np.floor(cx - rmax)));  x1 = min(X - 1, int(np.ceil(cx + rmax)))
        y0 = max(0, int(np.floor(cy - rmax)));  y1 = min(Y - 1, int(np.ceil(cy + rmax)))
        z0 = max(0, int(np.floor((cz - rmax - zoff) / z_scaling)))
        z1 = min(Z - 1, int(np.ceil((cz + rmax - zoff) / z_scaling)))
        if x0 > x1 or y0 > y1 or z0 > z1:
            continue
        xs = np.arange(x0, x1 + 1)
        ys = np.arange(y0, y1 + 1)
        zs = np.arange(z0, z1 + 1)
        zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
        dx = xx - cx
        dy = yy - cy
        dz = zz * z_scaling + zoff - cz
        R = rotation_T(tx, ty, tz)
        lx = R[0] * dx + R[1] * dy + R[2] * dz
        ly = R[3] * dx + R[4] * dy + R[5] * dz
        lz = R[6] * dx + R[7] * dy + R[8] * dz
        val = (lx * lx) / (a * a) + (ly * ly) / (b * b) + (lz * lz) / (c * c)
        inside = val <= 1.0
        if not inside.any():
            # guarantee presence: stamp the center voxel
            zc = int(round((cz - zoff) / z_scaling)); zc = min(max(zc, 0), Z - 1)
            yc = min(max(int(round(cy)), 0), Y - 1)
            xc = min(max(int(round(cx)), 0), X - 1)
            labels[zc, yc, xc] = cd["label"]; bestval[zc, yc, xc] = 0.0
            continue
        sub_best = bestval[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
        sub_lab = labels[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
        take = inside & (val < sub_best)
        sub_best[take] = val[take].astype(np.float32)
        sub_lab[take] = cd["label"]
    return labels


def rasterize_frame_signal(cells, shape, z_scaling, raw, thresh, scale):
    """Signal-fit variant. A voxel is assigned to a cell iff it is (a) inside that
    cell's ellipsoid scaled by `scale`, (b) above the intensity threshold, and
    (c) the cell it is most deeply inside. The scaled ellipsoid only bounds the
    search; the bright signal defines the final shape, so the mask hugs the real
    nucleus instead of the full detection ellipsoid."""
    Z, Y, X = shape
    labels = np.zeros(shape, dtype=np.uint16)
    bestval = np.full(shape, np.inf, dtype=np.float32)
    fg = raw > thresh
    zoff = (z_scaling - 1.0) / 2.0
    for cd in cells:
        cx, cy, cz = cd["x"], cd["y"], cd["z"]
        a, b, c = cd["a"] * scale, cd["b"] * scale, cd["c"] * scale
        tx, ty, tz = cd["tx"], cd["ty"], cd["tz"]
        rmax = max(a, b, c)
        x0 = max(0, int(np.floor(cx - rmax)));  x1 = min(X - 1, int(np.ceil(cx + rmax)))
        y0 = max(0, int(np.floor(cy - rmax)));  y1 = min(Y - 1, int(np.ceil(cy + rmax)))
        z0 = max(0, int(np.floor((cz - rmax - zoff) / z_scaling)))
        z1 = min(Z - 1, int(np.ceil((cz + rmax - zoff) / z_scaling)))
        if x0 > x1 or y0 > y1 or z0 > z1:
            continue
        xs = np.arange(x0, x1 + 1); ys = np.arange(y0, y1 + 1); zs = np.arange(z0, z1 + 1)
        zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
        dx = xx - cx; dy = yy - cy; dz = zz * z_scaling + zoff - cz
        R = rotation_T(tx, ty, tz)
        lx = R[0] * dx + R[1] * dy + R[2] * dz
        ly = R[3] * dx + R[4] * dy + R[5] * dz
        lz = R[6] * dx + R[7] * dy + R[8] * dz
        val = (lx * lx) / (a * a) + (ly * ly) / (b * b) + (lz * lz) / (c * c)
        sub_fg = fg[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
        take = (val <= 1.0) & sub_fg & (val < bestval[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1])
        bestval[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1][take] = val[take].astype(np.float32)
        labels[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1][take] = cd["label"]
    return labels


def ensure_presence(labels_per_frame, tracks):
    """After rasterization, make sure every track label appears in every frame
    of its [B,E] range (CTC validity). Stamp the recorded center if missing."""
    for name, t in tracks.items():
        lbl = t["label"]
        for fr in range(t["B"], t["E"] + 1):
            vol = labels_per_frame[fr]
            if not (vol == lbl).any():
                zc, yc, xc = t["centers"][fr]
                vol[zc, yc, xc] = lbl


def main():
    args = parse_args()
    rows = list(csv.DictReader(open(args.csv)))

    gt_tra = os.path.join(args.gt, "TRA")
    if args.shape:
        Z, Y, X = (int(v) for v in args.shape.split(","))
    else:
        ref = tifffile.imread(os.path.join(gt_tra, "man_track000.tif"))
        Z, Y, X = ref.shape
    shape = (Z, Y, X)
    print(f"label volume shape (Z,Y,X) = {shape}, z_scaling = {args.z_scaling}")

    # ----- group rows by (new frame, name) -----
    keep = lambda k: args.first <= k <= args.last
    byname = defaultdict(list)
    rows_by_newframe = defaultdict(list)
    for r in rows:
        k = frame_index(r["file"])
        if not keep(k):
            continue
        nf = k - args.first
        byname[r["name"]].append(nf)
        rows_by_newframe[nf].append(r)

    names = sorted(byname)
    label_of = {n: i + 1 for i, n in enumerate(names)}  # 1..N, deterministic
    n_frames = args.last - args.first + 1

    # ----- res_track.txt + per-track metadata -----
    tracks = {}
    for n in names:
        frs = sorted(byname[n])
        parent_name = n[:-1]
        parent = label_of.get(parent_name, 0)
        if parent and max(byname[parent_name]) >= frs[0]:
            parent = 0  # overlap -> cannot be parent (should not happen)
        tracks[n] = {"label": label_of[n], "B": frs[0], "E": frs[-1],
                     "parent": parent, "centers": {}}

    if args.mask_mode == "signal" and not args.raw_dir:
        raise SystemExit("--mask-mode signal requires --raw-dir")

    # ----- rasterize each frame -----
    zoff = (args.z_scaling - 1.0) / 2.0
    labels_per_frame = {}
    for nf in range(n_frames):
        cells = []
        for r in rows_by_newframe[nf]:
            a = float(r["majorRadius"]); b = float(r["bRadius"]); c = float(r["minorRadius"])
            if b <= 0.0:
                b = a  # oblate fallback (matches Ellipsoid ctor)
            cx = float(r["x"]); cy = float(r["y"]); cz = float(r["z"])
            cells.append({"x": cx, "y": cy, "z": cz, "a": a, "b": b, "c": c,
                          "tx": float(r["theta_x"]), "ty": float(r["theta_y"]),
                          "tz": float(r["theta_z"]), "label": label_of[r["name"]]})
            zc = min(max(int(round((cz - zoff) / args.z_scaling)), 0), Z - 1)
            yc = min(max(int(round(cy)), 0), Y - 1)
            xc = min(max(int(round(cx)), 0), X - 1)
            tracks[r["name"]]["centers"][nf] = (zc, yc, xc)
        if args.mask_mode == "signal":
            orig = nf + args.first
            raw = tifffile.imread(os.path.join(args.raw_dir, f"t{orig:03d}.tif")).astype(np.float32)
            T = foreground_threshold(raw, args.seg_thresh)
            labels_per_frame[nf] = rasterize_frame_signal(
                cells, shape, args.z_scaling, raw, T, args.seg_scale)
        else:
            labels_per_frame[nf] = rasterize_frame(cells, shape, args.z_scaling)
        if (nf + 1) % 20 == 0 or nf == n_frames - 1:
            print(f"  rasterized frame {nf+1}/{n_frames} ({len(cells)} cells)")

    ensure_presence(labels_per_frame, tracks)

    # ----- write RES -----
    res_dir = os.path.join(args.out, f"{args.seq}_RES")
    os.makedirs(res_dir, exist_ok=True)
    for nf in range(n_frames):
        tifffile.imwrite(os.path.join(res_dir, f"mask{nf:03d}.tif"),
                         labels_per_frame[nf], compression="zlib")
    with open(os.path.join(res_dir, "res_track.txt"), "w") as f:
        for n in names:
            t = tracks[n]
            f.write(f"{t['label']} {t['B']} {t['E']} {t['parent']}\n")
    print(f"wrote {n_frames} masks + res_track.txt ({len(names)} tracks) to {res_dir}")

    # ----- write trimmed/renumbered GT -----
    if args.skip_gt:
        print("skipping GT build (--skip-gt)")
    else:
        build_gt(args, n_frames)


def build_gt(args, n_frames):
    src_tra = os.path.join(args.gt, "TRA")
    src_seg = os.path.join(args.gt, "SEG")
    dst_gt = os.path.join(args.out, f"{args.seq}_GT")
    dst_tra = os.path.join(dst_gt, "TRA")
    dst_seg = os.path.join(dst_gt, "SEG")
    os.makedirs(dst_tra, exist_ok=True)
    os.makedirs(dst_seg, exist_ok=True)

    # TRA label images: copy original frame k -> new frame k-first
    for k in range(args.first, args.last + 1):
        src = os.path.join(src_tra, f"man_track{k:03d}.tif")
        dst = os.path.join(dst_tra, f"man_track{k - args.first:03d}.tif")
        shutil.copyfile(src, dst)

    # man_track.txt: clip [B,E] to window, shift, fix parents of dropped tracks
    kept = {}
    lines = []
    for ln in open(os.path.join(src_tra, "man_track.txt")):
        ln = ln.strip()
        if not ln:
            continue
        L, B, E, P = (int(v) for v in ln.split())
        lines.append((L, B, E, P))
    keep_labels = {L for (L, B, E, P) in lines
                   if not (E < args.first or B > args.last)}
    with open(os.path.join(dst_tra, "man_track.txt"), "w") as f:
        for (L, B, E, P) in lines:
            if E < args.first or B > args.last:
                continue
            nb = max(B, args.first) - args.first
            ne = min(E, args.last) - args.first
            parent = P if (P in keep_labels) else 0
            f.write(f"{L} {nb} {ne} {parent}\n")
            kept[L] = (nb, ne, parent)
    print(f"wrote trimmed GT TRA ({len(kept)} tracks) to {dst_tra}")

    # SEG: rename man_seg_FFF_SSS.tif -> frame FFF-first
    n_seg = 0
    for fn in sorted(os.listdir(src_seg)):
        m = re.match(r"man_seg_(\d+)_(\d+)\.tif", fn)
        if not m:
            continue
        F = int(m.group(1)); S = m.group(2)
        if not (args.first <= F <= args.last):
            continue
        dst = os.path.join(dst_seg, f"man_seg_{F - args.first:03d}_{S}.tif")
        shutil.copyfile(os.path.join(src_seg, fn), dst)
        n_seg += 1
    print(f"copied {n_seg} SEG GT slices to {dst_seg}")


if __name__ == "__main__":
    main()
