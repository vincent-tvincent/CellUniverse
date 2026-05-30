#!/usr/bin/env python3
"""Early-prune watchdog for gamma_25frame_tuning_oldparams_interleaved_r3.py.

This is experiment tooling only. It watches the active gamma tuning runs,
compares completed frames against GT counts and tolerant centroid matches, and
terminates runs that have already made an unrecoverable count/match error.
"""

import csv
import importlib.util
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/home/puv/output_fluo/tuning_fluo_gt_20260529/gamma_25frame_original_n2v2_oldparams_interleaved_r3')
STATUS = ROOT / 'status.json'
DECISIONS = ROOT / 'decisions.jsonl'
LOGBOOK = ROOT / 'docs/gamma_tuning_logbook.md'
DRIVER_PATH = Path('/home/puv/output_fluo/tuning_fluo_gt_20260529/continuous_gt_watch_r26/continuous_gt_watch_r26.py')
REPO_DRIVER = Path('/home/puv/celluniverse/CellUniverse/C++/scripts/gamma_25frame_tuning_oldparams_interleaved_r3.py')
SLEEP_SECONDS = 30


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def append_jsonl(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(obj, sort_keys=True) + '\n')


def append_log(title, lines):
    LOGBOOK.parent.mkdir(parents=True, exist_ok=True)
    with LOGBOOK.open('a') as f:
        f.write(f'\n## {now()} - {title}\n')
        for line in lines:
            f.write(f'- {line}\n')


def load_gt():
    spec = importlib.util.spec_from_file_location('r26_driver', DRIVER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_gt()


def load_batches():
    spec = importlib.util.spec_from_file_location('gamma_driver', REPO_DRIVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {batch['name']: batch for batch in module.BATCHES}


def parse_frame(value):
    m = re.search(r'(\d+)', str(value))
    return int(m.group(1)) if m else None


def threshold(frame):
    # Early Fluo cells are large; do not over-prune coordinate-only drift.
    # Counts/splits still fail immediately. Later, as cells narrow, tighten.
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
    rows = list(csv.DictReader(cells_csv.open()))
    z_values = []
    for row in rows:
        try:
            z_values.append(float(row.get('z', row.get('center_z'))))
        except Exception:
            pass
    z_scale = 1.0 if z_values and max(z_values) > 60.0 else 7.0
    pred = {}
    for row in rows:
        fr = parse_frame(row.get('file', row.get('frame', '')))
        if fr is None:
            continue
        if str(row.get('isTrash', row.get('trash', '0'))).lower() in ('1', 'true', 'yes'):
            continue
        try:
            pred.setdefault(fr, []).append({
                'x': float(row.get('x', row.get('center_x'))),
                'y': float(row.get('y', row.get('center_y'))),
                'z': float(row.get('z', row.get('center_z'))) * z_scale,
            })
        except Exception:
            continue
    return pred


def dist(a, b):
    return math.sqrt((a['x'] - b['x']) ** 2 + (a['y'] - b['y']) ** 2 + (a['z'] - b['z']) ** 2)


def match_counts(gt_rows, pred_rows, limit):
    options = []
    for gi, g in enumerate(gt_rows):
        for pi, p in enumerate(pred_rows):
            d = dist(g, p)
            if d <= limit:
                options.append((d, gi, pi))
    options.sort()
    used_gt = set()
    used_pred = set()
    distances = []
    for d, gi, pi in options:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        distances.append(d)
    return len(used_gt), len(gt_rows) - len(used_gt), len(pred_rows) - len(used_pred), distances


def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def active_controller_alive():
    proc = subprocess.run(
        ['pgrep', '-f', str(REPO_DRIVER) + ' auto'],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return bool(proc.stdout.strip())


def should_prune(gt, batches, active):
    batch_name = active['batch']
    batch = batches.get(batch_name)
    if not batch:
        return None
    run_dir = Path(active['run_dir'])
    pred = read_predictions(run_dir / 'cells.csv')
    if not pred:
        return None
    start = int(batch['score_start'])
    end = min(max(pred), int(batch['score_end']))
    failures = []
    for frame in range(start, end + 1):
        if frame not in pred:
            continue
        gt_rows = gt.get(frame, [])
        pred_rows = pred.get(frame, [])
        matched, missing, extra, distances = match_counts(gt_rows, pred_rows, threshold(frame))
        max_dist = max(distances) if distances else 0.0
        early_split_grace = False
        if extra and not missing and frame + 1 <= int(batch['score_end']):
            next_gt_count = len(gt.get(frame + 1, []))
            current_gt_count = len(gt_rows)
            # GT can lag a visually obvious split by one frame. Do not prune if
            # every current GT cell is matched and the extra prediction exactly
            # matches the next frame's GT count; the next frame will verify it.
            early_split_grace = (
                matched == current_gt_count
                and next_gt_count > current_gt_count
                and 0 < extra <= next_gt_count - current_gt_count
            )
        if (missing or extra or max_dist > threshold(frame)) and not early_split_grace:
            failures.append({
                'frame': frame,
                'gt_count': len(gt_rows),
                'pred_count': len(pred_rows),
                'matched': matched,
                'missing': missing,
                'extra': extra,
                'max_dist': max_dist,
                'threshold': threshold(frame),
            })
    return failures[0] if failures else None


def prune(active, reason):
    pid = int(active['pid'])
    event = {
        'time': now(),
        'event': 'watchdog_prune',
        'task_id': active['task_id'],
        'pid': pid,
        'batch': active['batch'],
        'gamma': active['gamma'],
        'reason': reason,
    }
    append_jsonl(DECISIONS, event)
    append_log(
        f"Watchdog prune {active['task_id']}",
        [
            f"batch: {active['batch']}",
            f"gamma: {active['gamma']}",
            f"pid: {pid}",
            f"reason: {reason}",
        ],
    )
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


def run_once(gt, batches):
    if not STATUS.exists():
        return False
    status = json.loads(STATUS.read_text())
    for active in status.get('active', []):
        pid = active.get('pid')
        if not pid or not pid_alive(pid):
            continue
        reason = should_prune(gt, batches, active)
        if reason:
            prune(active, reason)
    return status.get('status') == 'complete'


def main():
    gt = load_gt()
    batches = load_batches()
    append_log('Started gamma watchdog oldparams interleaved r3', [
        'Prunes active gamma runs after completed frames show GT count/match failure.',
        'Uses tolerant centroid thresholds and interpolated GT z from the continuous GT loader.',
    ])
    while True:
        complete = run_once(gt, batches)
        if complete and not active_controller_alive():
            append_log('Stopped gamma watchdog oldparams interleaved r3', ['controller complete and no active auto driver found'])
            return
        time.sleep(SLEEP_SECONDS)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
