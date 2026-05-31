#!/usr/bin/env python3
"""Gamma tuning for original CellUniverse tracking with N2V2 enabled.

Runs 25-frame Fluo windows with generated YAML configs only. CellLumen is
explicitly disabled; N2V2 network preprocessing remains enabled. Results are
scored against GT and cross-checked against recent successful Fluo outputs.
"""

import csv
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path('/home/puv/celluniverse/CellUniverse/C++')
ROOT = Path('/home/puv/output_fluo/tuning_fluo_gt_20260529/gamma_25frame_original_n2v2_recompiled_r6_20260531_025423')
GTFILLED_INITIAL_125_149 = ROOT / 'initials/initial_125_149_gtfilled_from_gt_centers.csv'
DRIVER_PATH = Path('/home/puv/output_fluo/tuning_fluo_gt_20260529/continuous_gt_watch_r26/continuous_gt_watch_r26.py')
PARENT_CONFIG = REPO / 'config/config.yaml'
BINARY = REPO / 'build/celluniverse'
INPUT_PATTERN = '/extra/wayne2/src/CellUniverse/celltrackingchallenge.net/Fluo-N3DH-CE-0train/Fluo-N3DH-CE/01/t%03d.tif'
CPUSET = '0-99'
THREADS = 20
MAX_WORKERS = 3
DISK_LIMIT_BYTES = 20 * 1024**3
REFERENCE_RUNS = {
    'ref_0': Path('/home/puv/output_fluo/output_fluo_0-239_20260525_022357'),
    'ref_42': Path('/home/puv/output_fluo/output_fluo_42-239_resume_from_022357_20260525_064217'),
    'ref_102': Path('/home/puv/output_fluo/output_fluo_102-239_resume_from_t101_latest_20260526_074318'),
}
GAMMAS = [1.45, 1.46, 1.47, 1.48, 1.49, 1.50, 1.52, 1.43, 1.42, 1.40, 1.385, 1.38, 1.36, 1.35, 1.32, 1.30, 1.28, 1.25, 1.55, 1.60, 1.62, 1.65, 1.68, 1.75, 1.85, 2.10, 1.00]
BATCHES = [
    {'name': '000_024', 'run_start': 0, 'run_end': 24, 'score_start': 0, 'score_end': 24, 'initial_kind': 'file', 'initial_source': str(REPO / 'config/embryo/initial_embryo_0.csv'), 'ref': 'ref_0'},
    {'name': '025_049', 'run_start': 25, 'run_end': 49, 'score_start': 25, 'score_end': 49, 'initial_kind': 'extract', 'initial_source': str(REFERENCE_RUNS['ref_0'] / 'cells.csv'), 'initial_frame': 25, 'ref': 'ref_0'},
    {'name': '050_074', 'run_start': 50, 'run_end': 74, 'score_start': 50, 'score_end': 74, 'initial_kind': 'extract', 'initial_source': str(REFERENCE_RUNS['ref_42'] / 'cells.csv'), 'initial_frame': 50, 'ref': 'ref_42'},
    {'name': '075_099', 'run_start': 75, 'run_end': 99, 'score_start': 75, 'score_end': 99, 'initial_kind': 'extract', 'initial_source': str(REFERENCE_RUNS['ref_42'] / 'cells.csv'), 'initial_frame': 75, 'ref': 'ref_42'},
    {'name': '100_124', 'run_start': 100, 'run_end': 124, 'score_start': 100, 'score_end': 124, 'initial_kind': 'extract', 'initial_source': str(REFERENCE_RUNS['ref_42'] / 'cells.csv'), 'initial_frame': 100, 'ref': 'ref_42'},
    {'name': '125_149', 'run_start': 125, 'run_end': 149, 'score_start': 125, 'score_end': 149, 'initial_kind': 'file', 'initial_source': str(GTFILLED_INITIAL_125_149), 'ref': 'ref_102'},
    {'name': '150_174', 'run_start': 150, 'run_end': 174, 'score_start': 150, 'score_end': 174, 'initial_kind': 'file', 'initial_source': str(REPO / 'config/embryo/initial_embryo_150_from_f149_fusion.csv'), 'ref': 'ref_102'},
    # Deferred: created after the best 150..174 run finishes, by copying its frame 174 state to t175.
    {'name': '175_199_gt_to194', 'run_start': 175, 'run_end': 194, 'score_start': 175, 'score_end': 194, 'initial_kind': 'deferred_best_previous', 'depends_on': '150_174', 'source_frame': 174, 'target_frame': 175, 'ref': None},
]
DIRS = {k: ROOT / k for k in ['configs', 'runs', 'initials', 'metrics', 'logs', 'docs']}
METRICS = DIRS['metrics'] / 'gamma_metrics.csv'
SUMMARY = DIRS['metrics'] / 'gamma_batch_summary.csv'
RELATION = DIRS['metrics'] / 'gamma_vs_cell_count.csv'
STATUS = ROOT / 'status.json'
DECISIONS = ROOT / 'decisions.jsonl'
LOGBOOK = DIRS['docs'] / 'gamma_tuning_logbook.md'


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def ensure_dirs():
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)


def append_jsonl(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(obj, sort_keys=True) + '\n')


def write_json(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    tmp.replace(path)


def append_log(title, lines):
    ensure_dirs()
    with LOGBOOK.open('a') as f:
        f.write(f'\n## {now()} - {title}\n')
        for line in lines:
            f.write(f'- {line}\n')


def load_driver():
    spec = importlib.util.spec_from_file_location('r26_driver', DRIVER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def set_path(data, dotted, value):
    cur = data
    parts = dotted.split('.')
    for key in parts[:-1]:
        cur = cur.setdefault(key, {})
    cur[parts[-1]] = value


def parse_frame(value):
    m = re.search(r'(\d+)', str(value))
    return int(m.group(1)) if m else None


def extract_initial(src, frame, out):
    src = Path(src)
    rows = []
    with src.open(newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        for row in reader:
            if parse_frame(row.get('file', row.get('frame', ''))) == frame:
                row = dict(row)
                row['file'] = f't{frame:03d}.tif'
                rows.append(row)
    if not rows:
        raise RuntimeError(f'no rows for frame {frame} in {src}')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def rewrite_frame_in_csv(src, source_frame, target_frame, out):
    rows = []
    with Path(src).open(newline='') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        for row in reader:
            if parse_frame(row.get('file', row.get('frame', ''))) == source_frame:
                row = dict(row)
                row['file'] = f't{target_frame:03d}.tif'
                rows.append(row)
    if not rows:
        raise RuntimeError(f'no rows for frame {source_frame} in {src}')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def prepare_initial(batch):
    if batch.get('name') == '125_149':
        if not GTFILLED_INITIAL_125_149.exists():
            raise RuntimeError(f'missing corrected 125_149 initial: {GTFILLED_INITIAL_125_149}')
        return GTFILLED_INITIAL_125_149
    out = DIRS['initials'] / f"initial_{batch['name']}.csv"
    if out.exists():
        return out
    if batch['initial_kind'] == 'file':
        shutil.copy2(batch['initial_source'], out)
    elif batch['initial_kind'] == 'extract':
        extract_initial(batch['initial_source'], batch['initial_frame'], out)
    elif batch['initial_kind'] == 'deferred_best_previous':
        raise RuntimeError(f"deferred initial missing for {batch['name']}: {out}")
    else:
        raise RuntimeError(f"unknown initial kind {batch['initial_kind']}")
    return out


def make_config(batch, gamma):
    with PARENT_CONFIG.open() as f:
        data = yaml.safe_load(f)
    overrides = {
        'cell_lumen.enabled': False,
        'cell_lumen.fusionEnabled': False,
        'cell_lumen.fusionSplitPriorEnabled': False,
        'cell.initialBrightness': 1.0,
        'cell.maxBrightness': 1.0,
        'cell.trashPcaShapeMaxOriginalRadiusFactor': 1.25,
        'simulation.preprocess_mode': 'n2v2',
        'simulation.n2v2_preprocess.enable_network': True,
        'simulation.n2v2_preprocess.contrast.gamma': float(gamma),
        'simulation.parallel_threads': THREADS,
        'simulation.export_frame_png': False,
        'simulation.export_frame_tiff': False,
        'simulation.export_preprocessed_images': False,
        'simulation.export_signal_debug_images': False,
        'simulation.export_perturb_debug_images': False,
        'simulation.export_perturb_cell_center_debug_images': False,
        'simulation.release_analyzed_exported_frames': True,
    }
    for k, v in overrides.items():
        set_path(data, k, v)
    name = f"gamma_{gamma:.2f}_{batch['name']}".replace('.', 'p')
    path = DIRS['configs'] / f'{name}.yaml'
    with path.open('w') as f:
        yaml.safe_dump(data, f, sort_keys=False)
    append_jsonl(DECISIONS, {'time': now(), 'event': 'config_generated', 'batch': batch['name'], 'gamma': gamma, 'config': str(path), 'overrides': overrides})
    return path


def env_for_threads():
    env = os.environ.copy()
    for key in ['CELLUNIVERSE_THREADS', 'OMP_NUM_THREADS', 'OPENCV_FOR_THREADS_NUM', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        env[key] = str(THREADS)
    return env


def task_id(batch, gamma):
    return f"{batch['name']}__gamma_{gamma:.2f}".replace('.', 'p')


def launch_task(batch, gamma):
    initial = prepare_initial(batch)
    config = make_config(batch, gamma)
    tid = task_id(batch, gamma)
    run_dir = DIRS['runs'] / tid
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config, run_dir / 'config_used.yaml')
    shutil.copy2(initial, run_dir / 'initial_used.csv')
    cmd = ['taskset', '-c', CPUSET, str(BINARY), str(batch['run_start']), str(batch['run_end']), INPUT_PATTERN, str(run_dir), str(config), str(initial)]
    log = (run_dir / 'debug_log.txt').open('ab', buffering=0)
    log.write((
        f"[CMD] {' '.join(cmd)}\nStarted: {now()}\n"
        f"method: original_celluniverse_no_celllumen\n"
        f"n2v2_network: enabled\ncontrast_gamma: {gamma}\n"
        f"batch: {batch['name']} score={batch['score_start']}..{batch['score_end']}\n"
        f"threads: {THREADS} cpuset: {CPUSET}\n==========================================\n"
    ).encode())
    proc = subprocess.Popen(cmd, cwd=str(BINARY.parent), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env_for_threads(), start_new_session=True, close_fds=True)
    meta = {'task_id': tid, 'batch': batch, 'gamma': gamma, 'pid': proc.pid, 'status': 'running', 'run_dir': str(run_dir), 'config': str(config), 'initial': str(initial), 'threads': THREADS, 'cpuset': CPUSET, 'launched_at': now()}
    write_json(run_dir / 'run_meta.json', meta)
    append_jsonl(DECISIONS, {'time': now(), 'event': 'run_launched', **meta})
    return {'proc': proc, 'log': log, **meta}


def read_predictions(cells_csv, start, end):
    rows = {}
    cells_csv = Path(cells_csv)
    if not cells_csv.exists():
        return rows
    raw_z = []
    with cells_csv.open(newline='') as f:
        for row in csv.DictReader(f):
            fr = parse_frame(row.get('file', row.get('frame', '')))
            if fr is None or fr < start or fr > end or str(row.get('isTrash', row.get('trash', '0'))).lower() in ['1', 'true', 'yes']:
                continue
            try:
                raw_z.append(float(row.get('z', row.get('center_z'))))
            except Exception:
                pass
    scale = 1.0 if raw_z and max(raw_z) > 60.0 else 7.0
    with cells_csv.open(newline='') as f:
        for row in csv.DictReader(f):
            fr = parse_frame(row.get('file', row.get('frame', '')))
            if fr is None or fr < start or fr > end or str(row.get('isTrash', row.get('trash', '0'))).lower() in ['1', 'true', 'yes']:
                continue
            try:
                rec = {'frame': fr, 'name': str(row.get('name', row.get('cell', ''))), 'x': float(row.get('x', row.get('center_x'))), 'y': float(row.get('y', row.get('center_y'))), 'z': float(row.get('z', row.get('center_z'))) * scale}
                rows.setdefault(fr, []).append(rec)
            except Exception:
                continue
    return rows


def dist(a, b):
    return math.sqrt((a['x']-b['x'])**2 + (a['y']-b['y'])**2 + (a['z']-b['z'])**2)


def threshold(frame):
    # Early Fluo cells are large; do not over-prune coordinate-only drift.
    # Counts/splits still fail immediately. Later, as cells narrow, tighten.
    if frame < 75: return 60.0
    if frame < 100: return 45.0
    if frame < 150: return 32.0
    return 28.0


def greedy_match(gt_rows, pred_rows, th):
    if not gt_rows:
        return [], [], list(pred_rows)
    if not pred_rows:
        return [], list(gt_rows), []
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        large = 1.0e9
        cost = np.full((len(gt_rows), len(pred_rows)), large, dtype=float)
        dist_matrix = np.zeros((len(gt_rows), len(pred_rows)), dtype=float)
        for gi, g in enumerate(gt_rows):
            for pi, p in enumerate(pred_rows):
                d = dist(g, p)
                dist_matrix[gi, pi] = d
                if d <= th:
                    cost[gi, pi] = d
        row_ind, col_ind = linear_sum_assignment(cost)
        matches = []
        ug, up = set(), set()
        for gi, pi in zip(row_ind, col_ind):
            d = float(dist_matrix[gi, pi])
            if d <= th:
                ug.add(int(gi)); up.add(int(pi))
                matches.append((gt_rows[int(gi)], pred_rows[int(pi)], d))
        missing = [g for i, g in enumerate(gt_rows) if i not in ug]
        extra = [p for i, p in enumerate(pred_rows) if i not in up]
        return matches, missing, extra
    except Exception:
        opts = []
        for gi, g in enumerate(gt_rows):
            for pi, p in enumerate(pred_rows):
                d = dist(g, p)
                if d <= th:
                    opts.append((d, gi, pi))
        opts.sort()
        ug, up, matches = set(), set(), []
        for d, gi, pi in opts:
            if gi in ug or pi in up: continue
            ug.add(gi); up.add(pi); matches.append((gt_rows[gi], pred_rows[pi], d))
        missing = [g for i,g in enumerate(gt_rows) if i not in ug]
        extra = [p for i,p in enumerate(pred_rows) if i not in up]
        return matches, missing, extra

def births(pred, frame):
    if frame <= 0: return []
    prev = {p['name'] for p in pred.get(frame-1, []) if p.get('name')}
    return [p for p in pred.get(frame, []) if p.get('name') and p['name'] not in prev]


def score_task(run_dir, batch, gamma):
    d = load_driver()
    gt = d.load_gt()
    pred = read_predictions(Path(run_dir) / 'cells.csv', batch['score_start'], batch['score_end'])
    ref_pred = {}
    if batch.get('ref') and (REFERENCE_RUNS[batch['ref']] / 'cells.csv').exists():
        ref_pred = read_predictions(REFERENCE_RUNS[batch['ref']] / 'cells.csv', batch['score_start'], batch['score_end'])
    total_dist = max_dist = 0.0
    matched = missing = extra = missed_splits = extra_splits = 0
    ref_total = ref_count = ref_max = 0.0
    rows = []
    gt_counts = []
    pred_counts = []
    for fr in range(batch['score_start'], batch['score_end'] + 1):
        gt_rows = gt.get(fr, [])
        pred_rows = pred.get(fr, [])
        gt_counts.append(len(gt_rows)); pred_counts.append(len(pred_rows))
        m, miss, ex = greedy_match(gt_rows, pred_rows, threshold(fr))
        ds = [x[2] for x in m]
        gt_birth = [g for g in gt_rows if g.get('parent', 0) > 0]
        pred_birth = births(pred, fr)
        sm, smiss, sex = greedy_match(gt_birth, pred_birth, threshold(fr))
        if fr == batch['score_start']:
            smiss, sex = [], []
        rm, _, _ = greedy_match(ref_pred.get(fr, []), pred_rows, max(60.0, threshold(fr) * 1.5)) if ref_pred else ([], [], [])
        rds = [x[2] for x in rm]
        ref_total += sum(rds); ref_count += len(rds); ref_max = max(ref_max, max(rds) if rds else 0.0)
        matched += len(m); missing += len(miss); extra += len(ex); missed_splits += len(smiss); extra_splits += len(sex)
        total_dist += sum(ds); max_dist = max(max_dist, max(ds) if ds else 0.0)
        rows.append({'frame': fr, 'gt_count': len(gt_rows), 'pred_count': len(pred_rows), 'matched': len(m), 'missing': len(miss), 'extra': len(ex), 'missed_splits': len(smiss), 'extra_splits': len(sex), 'mean_dist': (sum(ds)/len(ds) if ds else 0.0), 'max_dist': (max(ds) if ds else 0.0), 'threshold': threshold(fr), 'ref_mean_dist': (sum(rds)/len(rds) if rds else 0.0), 'ref_matches': len(rm)})
    mean_dist = total_dist / matched if matched else 999.0
    ref_mean = ref_total / ref_count if ref_count else 0.0
    score = 2000*(missed_splits + extra_splits) + 500*(missing + extra) + mean_dist + 5*max_dist + 0.25*ref_mean
    passed = missing == 0 and extra == 0 and missed_splits == 0 and extra_splits == 0 and max_dist <= max(threshold(f) for f in range(batch['score_start'], batch['score_end']+1))
    result = {'time': now(), 'batch': batch['name'], 'run_start': batch['run_start'], 'run_end': batch['run_end'], 'score_start': batch['score_start'], 'score_end': batch['score_end'], 'gamma': gamma, 'run_dir': str(run_dir), 'status': 'scored', 'passed': int(passed), 'score': score, 'missing_cells': missing, 'extra_cells': extra, 'missed_splits': missed_splits, 'extra_splits': extra_splits, 'matched_cells': matched, 'mean_dist': mean_dist, 'max_dist': max_dist, 'ref_mean_dist': ref_mean, 'ref_max_dist': ref_max, 'gt_mean_count': sum(gt_counts)/len(gt_counts), 'pred_mean_count': sum(pred_counts)/len(pred_counts)}
    detail_path = DIRS['metrics'] / f"frames_{task_id(batch, gamma)}.csv"
    write_csv(detail_path, rows, ['frame','gt_count','pred_count','matched','missing','extra','missed_splits','extra_splits','mean_dist','max_dist','threshold','ref_mean_dist','ref_matches'])
    return result


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        for row in rows:
            w.writerow(row)


def append_metric(row):
    fields = ['time','batch','run_start','run_end','score_start','score_end','gamma','status','passed','score','missing_cells','extra_cells','missed_splits','extra_splits','matched_cells','mean_dist','max_dist','ref_mean_dist','ref_max_dist','gt_mean_count','pred_mean_count','run_dir']
    write_csv(METRICS, [row], fields)



def run_contract_current(row):
    run_dir = row.get('run_dir')
    if not run_dir:
        return True
    run_path = Path(run_dir)
    if row.get('batch') == '125_149':
        meta_path = run_path / 'run_meta.json'
        initial = None
        if meta_path.exists():
            try:
                initial = json.loads(meta_path.read_text()).get('initial')
            except Exception:
                initial = None
        if initial and Path(initial) != GTFILLED_INITIAL_125_149:
            return False
        if not initial:
            try:
                if not ((run_path / 'initial_used.csv').exists() and (run_path / 'initial_used.csv').read_bytes() == GTFILLED_INITIAL_125_149.read_bytes()):
                    return False
            except Exception:
                return False
    log_path = run_path / 'debug_log.txt'
    if log_path.exists():
        try:
            for line in log_path.read_text(errors='ignore').splitlines():
                if '[N2V2] scale=' in line:
                    return float(line.split('scale=', 1)[1].split()[0]) >= 10.0
        except Exception:
            return False
    return True


def load_completed_ids():
    done = set()
    for r in read_metric_rows():
        done.add((r['batch'], float(r['gamma'])))
    return done


def update_summaries():
    if not METRICS.exists():
        return
    rows = list(csv.DictReader(METRICS.open(newline='')))
    by_batch = {}
    for r in rows:
        try:
            by_batch.setdefault(r['batch'], []).append(r)
        except Exception:
            pass
    summary_rows = []
    relation_rows = []
    for batch, vals in sorted(by_batch.items()):
        vals = sorted(vals, key=lambda r: float(r['score']))
        best = vals[0]
        summary_rows.append({'batch': batch, 'best_gamma': best['gamma'], 'best_score': best['score'], 'passed': best['passed'], 'gt_mean_count': best['gt_mean_count'], 'pred_mean_count': best['pred_mean_count'], 'tested_gammas': ';'.join(v['gamma'] for v in vals)})
        for v in vals:
            relation_rows.append({'batch': batch, 'gamma': v['gamma'], 'score': v['score'], 'gt_mean_count': v['gt_mean_count'], 'pred_mean_count': v['pred_mean_count'], 'missing_cells': v['missing_cells'], 'extra_cells': v['extra_cells'], 'mean_dist': v['mean_dist'], 'max_dist': v['max_dist']})
    for path in [SUMMARY, RELATION]:
        if path.exists(): path.unlink()
    write_csv(SUMMARY, summary_rows, ['batch','best_gamma','best_score','passed','gt_mean_count','pred_mean_count','tested_gammas'])
    write_csv(RELATION, relation_rows, ['batch','gamma','score','gt_mean_count','pred_mean_count','missing_cells','extra_cells','mean_dist','max_dist'])


def disk_bytes(path):
    total = 0
    for root, dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root)/name).stat().st_size
            except OSError:
                pass
    return total


def all_tasks(include_deferred=False):
    tasks = []
    for b in BATCHES:
        if b.get('initial_kind') == 'deferred_best_previous' and not include_deferred:
            continue
        for g in GAMMAS:
            tasks.append((b, g))
    return tasks


def read_metric_rows():
    if not METRICS.exists():
        return []
    return [row for row in csv.DictReader(METRICS.open(newline='')) if run_contract_current(row)]


def batch_completed(batch_name):
    rows = [r for r in read_metric_rows() if r.get('batch') == batch_name]
    return len({float(r['gamma']) for r in rows}) >= len(GAMMAS)


def best_metric_for_batch(batch_name):
    rows = [r for r in read_metric_rows() if r.get('batch') == batch_name]
    if not rows:
        return None
    return sorted(rows, key=lambda r: float(r['score']))[0]


def ensure_deferred_initials_and_tasks(queue):
    queued_keys = {(b['name'], float(g)) for b, g in queue}
    completed = load_completed_ids()
    added = 0
    for b in BATCHES:
        if b.get('initial_kind') != 'deferred_best_previous':
            continue
        initial = DIRS['initials'] / f"initial_{b['name']}.csv"
        if not initial.exists():
            dep = b['depends_on']
            if not batch_completed(dep):
                continue
            best = best_metric_for_batch(dep)
            if not best:
                continue
            rewrite_frame_in_csv(Path(best['run_dir']) / 'cells.csv', b['source_frame'], b['target_frame'], initial)
            append_log(f"Generated deferred initial for {b['name']}", [
                f"dependency batch: `{dep}`",
                f"source run: `{best['run_dir']}`",
                f"source frame: `{b['source_frame']}` -> target frame: `{b['target_frame']}`",
                f"output: `{initial}`",
            ])
        for g in GAMMAS:
            key = (b['name'], float(g))
            if key not in completed and key not in queued_keys:
                queue.append((b, g)); queued_keys.add(key); added += 1
    return added


def auto():
    ensure_dirs()
    append_log('Started gamma 25-frame tuning oldparams interleaved r3', [
        'method: original CellUniverse tracking; CellLumen disabled in every generated YAML',
        'NN preprocessing: `simulation.preprocess_mode=n2v2`, `n2v2_preprocess.enable_network=true`',
        f'gamma candidates, launch priority order: `{GAMMAS}`',
        'batches: first 200 frames as 25-frame groups; launch order interleaves independent windows before testing the next gamma',
        f'CPU budget: max `{MAX_WORKERS}` workers x `{THREADS}` threads = `{MAX_WORKERS*THREADS}` logical cores',
        f'disk budget: `<20GB` under `{ROOT}`',
    ])
    completed_ids = load_completed_ids()
    available_batches = [b for b in BATCHES if b.get('initial_kind') != 'deferred_best_previous']
    # Interleave by window first: start the same gamma across independent
    # 25-frame windows, then cycle the other gammas through those windows.
    queue = [(b, g) for g in GAMMAS for b in available_batches if (b['name'], float(g)) not in completed_ids]
    ensure_deferred_initials_and_tasks(queue)
    active = []
    while queue or active:
        active = [a for a in active if a['proc'].poll() is None or not a.get('scored')]
        for a in list(active):
            if a['proc'].poll() is not None and not a.get('scored'):
                try:
                    row = score_task(Path(a['run_dir']), a['batch'], a['gamma'])
                    append_metric(row)
                    append_jsonl(DECISIONS, {'time': now(), 'event': 'run_scored', **row})
                except Exception as e:
                    row = {'time': now(), 'batch': a['batch']['name'], 'run_start': a['batch']['run_start'], 'run_end': a['batch']['run_end'], 'score_start': a['batch']['score_start'], 'score_end': a['batch']['score_end'], 'gamma': a['gamma'], 'status': 'score_failed', 'passed': 0, 'score': 999999, 'missing_cells': 999, 'extra_cells': 999, 'missed_splits': 999, 'extra_splits': 999, 'matched_cells': 0, 'mean_dist': 999, 'max_dist': 999, 'ref_mean_dist': 0, 'ref_max_dist': 0, 'gt_mean_count': 0, 'pred_mean_count': 0, 'run_dir': a['run_dir']}
                    append_metric(row)
                    append_jsonl(DECISIONS, {'time': now(), 'event': 'score_failed', 'task': a['task_id'], 'error': str(e)})
                a['scored'] = True
                try:
                    a['log'].close()
                except Exception:
                    pass
                update_summaries()
                ensure_deferred_initials_and_tasks(queue)
        active = [a for a in active if not a.get('scored')]
        while queue and len(active) < MAX_WORKERS:
            b, g = queue.pop(0)
            if disk_bytes(ROOT) > DISK_LIMIT_BYTES:
                append_jsonl(DECISIONS, {'time': now(), 'event': 'disk_limit_pause', 'bytes': disk_bytes(ROOT)})
                queue.insert(0, (b, g)); break
            active.append(launch_task(b, g))
        ensure_deferred_initials_and_tasks(queue)
        write_json(STATUS, {'updated_at': now(), 'root': str(ROOT), 'queued': len(queue), 'active': [{'task_id': a['task_id'], 'pid': a['pid'], 'batch': a['batch']['name'], 'gamma': a['gamma'], 'run_dir': a['run_dir']} for a in active], 'completed': len(load_completed_ids()), 'disk_bytes': disk_bytes(ROOT), 'max_workers': MAX_WORKERS, 'threads': THREADS})
        time.sleep(20)
    update_summaries()
    write_json(STATUS, {'updated_at': now(), 'root': str(ROOT), 'queued': 0, 'active': [], 'completed': len(load_completed_ids()), 'disk_bytes': disk_bytes(ROOT), 'status': 'complete'})
    append_log('Completed gamma 25-frame tuning oldparams interleaved r3', [f'metrics: `{METRICS}`', f'summary: `{SUMMARY}`', f'relation: `{RELATION}`'])


def status():
    ensure_dirs()
    if STATUS.exists():
        print(STATUS.read_text())
    else:
        print('No status yet')
    if METRICS.exists():
        rows = list(csv.DictReader(METRICS.open(newline='')))
        print(f'metrics rows: {len(rows)}')
        for r in sorted(rows, key=lambda x: float(x['score']))[:10]:
            print(r['batch'], r['gamma'], r['score'], 'pass=', r['passed'], 'miss/extra=', r['missing_cells'], r['extra_cells'], 'dist=', r['mean_dist'], r['max_dist'])
    if SUMMARY.exists():
        print('\nsummary:')
        print(SUMMARY.read_text())


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'status'
    if cmd == 'auto':
        auto()
    elif cmd == 'status':
        status()
    else:
        raise SystemExit(f'unknown command {cmd}')
