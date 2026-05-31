#!/usr/bin/env python3
"""Refresh live gamma-tuning CSV summaries for the r3 original/N2V2 sweep."""

import csv
import importlib.util
import json
import math
import os
import re
import time
from pathlib import Path

ROOT = Path('/home/puv/output_fluo/tuning_fluo_gt_20260529/gamma_25frame_original_n2v2_recompiled_r6_20260531_025423')
REPO = Path('/home/puv/celluniverse/CellUniverse/C++')
DRIVER_PATH = REPO / 'scripts/gamma_25frame_tuning_oldparams_interleaved_r3.py'
GT_DRIVER_PATH = Path('/home/puv/output_fluo/tuning_fluo_gt_20260529/continuous_gt_watch_r26/continuous_gt_watch_r26.py')

KNOWN = ROOT / 'metrics/gamma_known_results.csv'
BEST = REPO / 'docs/gamma_window_best_so_far.csv'
LOGBOOK = ROOT / 'docs/gamma_tuning_logbook.md'
GTFILLED_INITIAL_125_149 = ROOT / 'initials/initial_125_149_gtfilled_from_gt_centers.csv'
INTERRUPT_MARKERS = [
    'INTERRUPTED_FOR_PRIORITY',
    'INTERRUPTED_FOR_16T_REALLOC',
    'INTERRUPTED_FOR_FASTSPLIT_BRANCH',
    'INTERRUPTED_FOR_TARGETED_SPLITGATE_BRANCH',
    'INTERRUPTED_FOR_INTERMEDIATE_GAMMA_BRANCH',
    'INTERRUPTED_FOR_LATE_RELAUNCH',
    'INTERRUPTED_FOR_WINDOW_REBALANCE',
    'INTERRUPTED_FOR_EARLY_REGRESSION_BRANCH',
    'INTERRUPTED_FOR_RESUME126_PROBE',
    'PRUNED_GT_FAILURE',
]


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_frame(value):
    match = re.search(r'(\d+)', str(value))
    return int(match.group(1)) if match else None


def threshold(frame):
    if frame < 75:
        return 60.0
    if frame < 100:
        return 45.0
    if frame < 150:
        return 32.0
    return 28.0


def read_predictions(cells_csv):
    cells_csv = Path(cells_csv)
    if not cells_csv.exists():
        return {}
    with cells_csv.open(newline='') as handle:
        raw_rows = list(csv.DictReader(handle))
    z_values = []
    for row in raw_rows:
        try:
            z_values.append(float(row.get('z', row.get('center_z'))))
        except Exception:
            pass
    z_scale = 1.0 if z_values and max(z_values) > 60.0 else 7.0
    frames = {}
    for row in raw_rows:
        frame = parse_frame(row.get('file', row.get('frame', '')))
        if frame is None:
            continue
        if str(row.get('isTrash', row.get('trash', '0'))).lower() in ('1', 'true', 'yes'):
            continue
        try:
            frames.setdefault(frame, []).append({
                'x': float(row.get('x', row.get('center_x'))),
                'y': float(row.get('y', row.get('center_y'))),
                'z': float(row.get('z', row.get('center_z'))) * z_scale,
                'name': str(row.get('name', row.get('cell', ''))),
            })
        except Exception:
            continue
    return frames


def distance(a, b):
    return math.sqrt((a['x'] - b['x']) ** 2 + (a['y'] - b['y']) ** 2 + (a['z'] - b['z']) ** 2)


def greedy_match(gt_rows, pred_rows, limit):
    if not gt_rows:
        return 0, 0, len(pred_rows), []
    if not pred_rows:
        return 0, len(gt_rows), 0, []
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        large = 1.0e9
        cost = np.full((len(gt_rows), len(pred_rows)), large, dtype=float)
        dist_matrix = np.zeros((len(gt_rows), len(pred_rows)), dtype=float)
        for gt_i, gt_row in enumerate(gt_rows):
            for pred_i, pred_row in enumerate(pred_rows):
                dist_value = distance(gt_row, pred_row)
                dist_matrix[gt_i, pred_i] = dist_value
                if dist_value <= limit:
                    cost[gt_i, pred_i] = dist_value
        row_ind, col_ind = linear_sum_assignment(cost)
        used_gt = set()
        used_pred = set()
        distances = []
        for gt_i, pred_i in zip(row_ind, col_ind):
            dist_value = float(dist_matrix[gt_i, pred_i])
            if dist_value <= limit:
                used_gt.add(int(gt_i))
                used_pred.add(int(pred_i))
                distances.append(dist_value)
        return len(used_gt), len(gt_rows) - len(used_gt), len(pred_rows) - len(used_pred), distances
    except Exception:
        options = []
        for gt_i, gt_row in enumerate(gt_rows):
            for pred_i, pred_row in enumerate(pred_rows):
                dist_value = distance(gt_row, pred_row)
                if dist_value <= limit:
                    options.append((dist_value, gt_i, pred_i))
        options.sort()
        used_gt = set()
        used_pred = set()
        distances = []
        for dist_value, gt_i, pred_i in options:
            if gt_i in used_gt or pred_i in used_pred:
                continue
            used_gt.add(gt_i)
            used_pred.add(pred_i)
            distances.append(dist_value)
        return len(used_gt), len(gt_rows) - len(used_gt), len(pred_rows) - len(used_pred), distances

def pid_alive(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def initial_contract_ok(batch_name, run_dir):
    if batch_name != '125_149':
        return True
    meta_path = Path(run_dir) / 'run_meta.json'
    initial_path = None
    if meta_path.exists():
        try:
            initial_path = json.loads(meta_path.read_text()).get('initial')
        except Exception:
            initial_path = None
    if initial_path and Path(initial_path) == GTFILLED_INITIAL_125_149:
        return True
    copied = Path(run_dir) / 'initial_used.csv'
    try:
        return copied.exists() and copied.read_bytes() == GTFILLED_INITIAL_125_149.read_bytes()
    except Exception:
        return False


def interrupted_or_pruned(run_dir):
    run_dir = Path(run_dir)
    return any((run_dir / marker).exists() for marker in INTERRUPT_MARKERS)


def preprocessing_contract_ok(run_dir):
    log_path = Path(run_dir) / 'debug_log.txt'
    if not log_path.exists():
        return True
    try:
        text = log_path.read_text(errors='ignore')
    except Exception:
        return False
    for line in text.splitlines():
        if '[N2V2] scale=' not in line:
            continue
        try:
            scale = float(line.split('scale=', 1)[1].split()[0])
        except Exception:
            return False
        return scale >= 10.0
    return True


def classify(frames, state):
    if not frames:
        return 'running_no_scored_output' if state == 'running' else 'no_scored_output'
    first_bad = next((f for f in frames if f['is_bad']), None)
    if first_bad:
        return 'true_fail_or_needs_tuning'
    grace_frames = [f['frame'] for f in frames if f['early_split_grace']]
    if grace_frames:
        if frames[-1]['frame'] > max(grace_frames):
            return 'recovered_after_possible_early_split'
        return 'possible_one_frame_early_split'
    if state == 'running':
        return 'promising_so_far'
    return 'stopped_clean_so_far'


def scan_run(gt, batch, gamma, run_dir, state, pid):
    pred = read_predictions(run_dir / 'cells.csv')
    frames = []
    if pred:
        start = int(batch['score_start'])
        end = min(max(pred), int(batch['score_end']))
        for frame in range(start, end + 1):
            if frame not in pred:
                continue
            gt_rows = gt.get(frame, [])
            pred_rows = pred.get(frame, [])
            matched, missing, extra, distances = greedy_match(gt_rows, pred_rows, threshold(frame))
            next_count = len(gt.get(frame + 1, [])) if frame + 1 <= int(batch['score_end']) else None
            early_split_grace = bool(
                extra and not missing and next_count is not None
                and next_count > len(gt_rows)
                and extra <= next_count - len(gt_rows)
                and matched == len(gt_rows)
            )
            max_dist = max(distances) if distances else 0.0
            is_bad = bool((missing or extra or max_dist > threshold(frame)) and not early_split_grace)
            frames.append({
                'frame': frame,
                'gt_count': len(gt_rows),
                'pred_count': len(pred_rows),
                'matched': matched,
                'missing': missing,
                'extra': extra,
                'mean_dist': sum(distances) / len(distances) if distances else 0.0,
                'max_dist': max_dist,
                'early_split_grace': early_split_grace,
                'is_bad': is_bad,
            })
    first_bad = next((f for f in frames if f['is_bad']), None)
    last = frames[-1] if frames else {}
    clean_frames = 0
    for frame in frames:
        if frame['is_bad']:
            break
        clean_frames += 1
    classification = classify(frames, state)
    if state != 'running' and frames and int(frames[-1]['frame']) < int(batch['score_end']):
        classification = 'stopped_incomplete'
    return {
        'time': now(),
        'batch': batch['name'],
        'gamma': f'{float(gamma):.2f}',
        'run': run_dir.name,
        'state': state,
        'classification': classification,
        'frames_scored': len(frames),
        'clean_prefix_frames': clean_frames,
        'last_frame': last.get('frame', ''),
        'last_gt_count': last.get('gt_count', ''),
        'last_pred_count': last.get('pred_count', ''),
        'first_bad_frame': first_bad.get('frame', '') if first_bad else '',
        'first_bad_gt_count': first_bad.get('gt_count', '') if first_bad else '',
        'first_bad_pred_count': first_bad.get('pred_count', '') if first_bad else '',
        'first_bad_missing': first_bad.get('missing', '') if first_bad else '',
        'first_bad_extra': first_bad.get('extra', '') if first_bad else '',
        'mean_dist_last': last.get('mean_dist', ''),
        'max_dist_last': last.get('max_dist', ''),
        'pid': pid or '',
        'run_dir': str(run_dir),
        'note': note_for(batch['name'], float(gamma), run_dir.name, first_bad, last),
    }


def note_for(batch_name, gamma, run_name, first_bad, last):
    if run_name == '025_049__gamma_1p45_resume28_relaxed' and last and not first_bad:
        return 'resume from frame28 remains clean after frame35 split burst; current local best signal for 25-49'
    if batch_name == '000_024' and gamma == 1.40 and last and not first_bad:
        last_frame = int(last.get('frame', -1)) if last.get('frame', '') != '' else -1
        if last_frame >= 12:
            return 'boundary probe: survived through frame12; must still complete 0-24'
        if last_frame >= 11:
            return 'boundary probe: survived through frame11; must still clear frame12 and complete 0-24'
        return 'boundary probe: fixes frame4 missed split so far; must still survive frame11 over-split risk'
    if first_bad and first_bad.get('missing'):
        return 'missed GT cell/split before completing window'
    if first_bad and first_bad.get('extra'):
        return 'extra predicted cell/split before completing window'
    return ''


def best_rows(rows):
    output = []
    for batch in sorted({row['batch'] for row in rows}):
        vals = [row for row in rows if row['batch'] == batch]
        def rank(row):
            complete = row['last_frame'] != '' and int(row['last_frame']) >= batch_end(batch)
            clean = int(row['clean_prefix_frames'] or 0)
            has_bad = row['first_bad_frame'] != ''
            class_rank = {
                'stopped_clean_so_far': 0,
                'promising_so_far': 0,
                'recovered_after_possible_early_split': 0,
                'possible_one_frame_early_split': 2,
                'true_fail_or_needs_tuning': 3,
                'running_no_scored_output': 4,
                'no_scored_output': 5,
                'stopped_incomplete': 6,
            }.get(row['classification'], 9)
            bad_frame = int(row['first_bad_frame']) if row['first_bad_frame'] != '' else 999
            max_dist = float(row['max_dist_last'] or 999)
            return (not complete, -clean, has_bad, class_rank, -bad_frame, max_dist)
        best = sorted(vals, key=rank)[0]
        complete = best['last_frame'] != '' and int(best['last_frame']) >= batch_end(batch)
        output.append({
            'time': now(),
            'batch': batch,
            'best_gamma_so_far': best['gamma'],
            'best_run_so_far': best['run'],
            'classification': best['classification'],
            'frames_scored': best['frames_scored'],
            'clean_prefix_frames': best['clean_prefix_frames'],
            'last_frame': best['last_frame'],
            'complete_window': int(complete),
            'note': best['note'],
        })
    return output


def batch_end(batch_name):
    for batch in DRIVER.BATCHES:
        if batch['name'] == batch_name:
            return int(batch['score_end'])
    return 999


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def append_log(rows, best):
    changed = []
    for row in rows:
        if row['classification'] in ('promising_so_far', 'stopped_clean_so_far') and row['note']:
            changed.append(f"{row['run']}: {row['note']} (last frame {row['last_frame']})")
    with LOGBOOK.open('a') as handle:
        handle.write(f"\n## {now()} - Refreshed gamma known-results snapshot\n")
        handle.write(f"- wrote: `{KNOWN}`\n")
        handle.write(f"- wrote: `{BEST}`\n")
        for item in best:
            handle.write(
                f"- best-so-far {item['batch']}: gamma {item['best_gamma_so_far']} "
                f"via `{item['best_run_so_far']}`, {item['classification']}, "
                f"last frame {item['last_frame']}, complete={item['complete_window']}\n"
            )
        for item in changed[:8]:
            handle.write(f"- observation: {item}\n")


def main():
    global DRIVER
    DRIVER = load_module('gamma_driver', DRIVER_PATH)
    gt_driver = load_module('gt_driver', GT_DRIVER_PATH)
    gt = gt_driver.load_gt()
    status = json.loads((ROOT / 'status.json').read_text()) if (ROOT / 'status.json').exists() else {}
    active_by_run = {}
    for item in status.get('active', []):
        active_by_run[Path(item['run_dir']).name] = item
    rows = []
    for batch in DRIVER.BATCHES:
        run_base = ROOT / 'runs'
        if not run_base.exists():
            continue
        prefix = f"{batch['name']}__gamma_"
        for run_dir in sorted(p for p in run_base.iterdir() if p.is_dir() and p.name.startswith(prefix)):
            match = re.search(r'gamma_(\d+)p(\d+)', run_dir.name)
            if not match:
                continue
            gamma = float(f"{match.group(1)}.{match.group(2)}")
            if interrupted_or_pruned(run_dir):
                continue
            if not preprocessing_contract_ok(run_dir):
                continue
            if not initial_contract_ok(batch['name'], run_dir):
                continue
            active = active_by_run.get(run_dir.name, {})
            if not active:
                meta_path = run_dir / 'run_meta.json'
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text())
                        active = {
                            'pid': meta.get('pid'),
                            'run_dir': meta.get('run_dir', str(run_dir)),
                            'task_id': meta.get('task_id', run_dir.name),
                        }
                    except Exception:
                        active = {}
            pid = active.get('pid')
            state = 'running' if pid_alive(pid) else ('interrupted_or_done' if (run_dir / 'cells.csv').exists() else 'no_cells_output')
            rows.append(scan_run(gt, batch, gamma, run_dir, state, pid))
    fields = [
        'time', 'batch', 'gamma', 'run', 'state', 'classification', 'frames_scored',
        'clean_prefix_frames', 'last_frame', 'last_gt_count', 'last_pred_count',
        'first_bad_frame', 'first_bad_gt_count', 'first_bad_pred_count',
        'first_bad_missing', 'first_bad_extra', 'mean_dist_last', 'max_dist_last',
        'pid', 'run_dir', 'note',
    ]
    write_csv(KNOWN, rows, fields)
    best = best_rows(rows)
    write_csv(BEST, best, [
        'time', 'batch', 'best_gamma_so_far', 'best_run_so_far', 'classification',
        'frames_scored', 'clean_prefix_frames', 'last_frame', 'complete_window', 'note',
    ])
    append_log(rows, best)
    print(f'wrote {KNOWN}')
    print(f'wrote {BEST}')
    for row in best:
        print(f"{row['batch']}: gamma {row['best_gamma_so_far']} {row['classification']} last={row['last_frame']} complete={row['complete_window']}")


if __name__ == '__main__':
    main()
