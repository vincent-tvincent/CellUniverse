#!/usr/bin/env python3
"""Build a CTC RES + trimmed GT from the TWO CellUniverse runs concatenated:
run1 = FinalLineageTree (frames 1-171), run2 = resume (frames 174-194).

NOTE: the two runs use independent label schemes (verified: <26% suffix overlap,
matched cells jump ~143px in 3 frames), and frames 172-173 are missing. So the
combined lineage is NOT continuous across the 171->174 boundary: run1 tracks end at
171 and run2 tracks begin fresh at 174 (parent 0). We drop 172-173 and renumber the
kept frames {1..171, 174..194} to contiguous 0-based indices 0..191, so CTC pairing
works. Masks use the signal-fit mode (scale 1.4, p75 threshold).

This is an honest "what do we get if we glue them together" evaluation, not a clean
single-run lineage.
"""
import argparse, csv, os, re, shutil
from collections import defaultdict
import numpy as np
import tifffile

import build_ctc_res as B  # rotation_T, rasterize_frame_signal, foreground_threshold, ensure_presence


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run1", required=True, help="FinalLineageTree CSV (frames 1-171)")
    p.add_argument("--run2", required=True, help="resume cells.csv (frames 174-194)")
    p.add_argument("--gt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--seq", default="01")
    p.add_argument("--z-scaling", type=float, default=7.0)
    p.add_argument("--seg-scale", type=float, default=1.4)
    p.add_argument("--seg-thresh", default="p75")
    return p.parse_args()


def fidx(f):
    return int(re.search(r"(\d+)", f).group(1))


def load_run(path, prefix, rad_cols):
    """Return list of dicts with normalized fields; name prefixed to keep runs disjoint."""
    a_col, b_col, c_col = rad_cols
    out = []
    for r in csv.DictReader(open(path)):
        out.append({
            "orig": fidx(r["file"]),
            "name": prefix + r["name"],
            "x": float(r["x"]), "y": float(r["y"]), "z": float(r["z"]),
            "a": float(r[a_col]), "b": float(r[b_col]), "c": float(r[c_col]),
            "tx": float(r["theta_x"]), "ty": float(r["theta_y"]), "tz": float(r["theta_z"]),
        })
    return out


def main():
    args = parse_args()
    # kept orig frames -> 0-based new index (drop 172,173)
    kept = list(range(1, 172)) + list(range(174, 195))   # 1..171, 174..194
    new_of = {orig: i for i, orig in enumerate(kept)}     # orig -> 0..191
    n_frames = len(kept)
    raw_of = {i: orig for orig, i in new_of.items()}      # new -> orig (for raw lookup)

    rows = load_run(args.run1, "A:", ("majorRadius", "bRadius", "minorRadius"))
    rows += load_run(args.run2, "B:", ("aRadius", "bRadius", "cRadius"))

    ref = tifffile.imread(os.path.join(args.gt, "TRA", "man_track000.tif"))
    Z, Y, X = ref.shape
    shape = (Z, Y, X)
    print(f"shape {shape}, {n_frames} kept frames (172-173 dropped), z_scaling {args.z_scaling}")

    byname = defaultdict(list)
    rows_by_nf = defaultdict(list)
    for r in rows:
        if r["orig"] not in new_of:
            continue
        nf = new_of[r["orig"]]
        byname[r["name"]].append(nf)
        rows_by_nf[nf].append(r)

    names = sorted(byname)
    label_of = {n: i + 1 for i, n in enumerate(names)}

    tracks = {}
    for n in names:
        frs = sorted(byname[n])
        parent_name = n[:-1]                       # strip last digit (within same run prefix)
        parent = label_of.get(parent_name, 0)
        if parent and max(byname[parent_name]) >= frs[0]:
            parent = 0
        tracks[n] = {"label": label_of[n], "B": frs[0], "E": frs[-1],
                     "parent": parent, "centers": {}}

    zoff = (args.z_scaling - 1.0) / 2.0
    labels_per_frame = {}
    for nf in range(n_frames):
        cells = []
        for r in rows_by_nf[nf]:
            a, b, c = r["a"], (r["b"] if r["b"] > 0 else r["a"]), r["c"]
            cells.append({"x": r["x"], "y": r["y"], "z": r["z"], "a": a, "b": b, "c": c,
                          "tx": r["tx"], "ty": r["ty"], "tz": r["tz"], "label": label_of[r["name"]]})
            zc = min(max(int(round((r["z"] - zoff) / args.z_scaling)), 0), Z - 1)
            yc = min(max(int(round(r["y"])), 0), Y - 1)
            xc = min(max(int(round(r["x"])), 0), X - 1)
            tracks[r["name"]]["centers"][nf] = (zc, yc, xc)
        raw = tifffile.imread(os.path.join(args.raw_dir, f"t{raw_of[nf]:03d}.tif")).astype(np.float32)
        T = B.foreground_threshold(raw, args.seg_thresh)
        labels_per_frame[nf] = B.rasterize_frame_signal(cells, shape, args.z_scaling, raw, T, args.seg_scale)
        if (nf + 1) % 20 == 0 or nf == n_frames - 1:
            print(f"  frame {nf+1}/{n_frames} (orig t{raw_of[nf]:03d}, {len(cells)} cells)")

    B.ensure_presence(labels_per_frame, tracks)

    # write RES
    res_dir = os.path.join(args.out, f"{args.seq}_RES")
    os.makedirs(res_dir, exist_ok=True)
    for nf in range(n_frames):
        tifffile.imwrite(os.path.join(res_dir, f"mask{nf:03d}.tif"),
                         labels_per_frame[nf], compression="zlib")
    with open(os.path.join(res_dir, "res_track.txt"), "w") as f:
        for n in names:
            t = tracks[n]
            f.write(f"{t['label']} {t['B']} {t['E']} {t['parent']}\n")
    print(f"wrote {n_frames} masks + res_track.txt ({len(names)} tracks)")

    # ----- GT (drop 172,173; renumber to 0..191) -----
    dst_tra = os.path.join(args.out, f"{args.seq}_GT", "TRA")
    dst_seg = os.path.join(args.out, f"{args.seq}_GT", "SEG")
    os.makedirs(dst_tra, exist_ok=True)
    os.makedirs(dst_seg, exist_ok=True)
    src_tra = os.path.join(args.gt, "TRA")
    for orig, nf in new_of.items():
        shutil.copyfile(os.path.join(src_tra, f"man_track{orig:03d}.tif"),
                        os.path.join(dst_tra, f"man_track{nf:03d}.tif"))
    # man_track.txt: remap each track's kept frames -> new indices
    keep_labels = set()
    lines = []
    for ln in open(os.path.join(src_tra, "man_track.txt")):
        ln = ln.strip()
        if not ln:
            continue
        L, Bf, Ef, P = (int(v) for v in ln.split())
        new_frames = [new_of[f] for f in range(Bf, Ef + 1) if f in new_of]
        if not new_frames:
            continue
        keep_labels.add(L)
        lines.append((L, min(new_frames), max(new_frames), P))
    with open(os.path.join(dst_tra, "man_track.txt"), "w") as f:
        for (L, nb, ne, P) in lines:
            f.write(f"{L} {nb} {ne} {P if P in keep_labels else 0}\n")
    print(f"wrote trimmed GT TRA ({len(lines)} tracks)")

    n_seg = 0
    for fn in sorted(os.listdir(os.path.join(args.gt, "SEG"))):
        m = re.match(r"man_seg_(\d+)_(\d+)\.tif", fn)
        if not m:
            continue
        F, S = int(m.group(1)), m.group(2)
        if F not in new_of:
            continue
        shutil.copyfile(os.path.join(args.gt, "SEG", fn),
                        os.path.join(dst_seg, f"man_seg_{new_of[F]:03d}_{S}.tif"))
        n_seg += 1
    print(f"copied {n_seg} SEG GT slices")


if __name__ == "__main__":
    main()
