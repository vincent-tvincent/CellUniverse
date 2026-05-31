#!/usr/bin/env python3
"""Direct refill monitor for corrected r5 gamma tuning.

Keeps at most 8 CellUniverse workers alive under the r5 tuning root, refreshes
known-result docs, prunes true GT failures, and launches the next queued
window/gamma candidate. This is intentionally separate from the older scheduler
because r5 uses direct worker launches and run_meta.json PID tracking.
"""

import csv
import importlib.util
import json
import os
import signal
import subprocess
import time
from pathlib import Path

REPO = Path('/home/puv/celluniverse/CellUniverse/C++')
DRIVER_PATH = REPO / 'scripts/gamma_25frame_tuning_oldparams_interleaved_r3.py'
UPDATER = REPO / 'scripts/gamma_update_known_results_r3.py'
ROOT = Path('/home/puv/output_fluo/tuning_fluo_gt_20260529/gamma_25frame_original_n2v2_oldparams_interleaved_r5_brightness1_trash125')
KNOWN = ROOT / 'metrics/gamma_known_results.csv'
LOG = ROOT / 'logs/gamma_r5_refill_monitor.log'
DECISIONS = ROOT / 'decisions.jsonl'
MAX_WORKERS = 6
SLEEP_SECONDS = 30

# Keep the queue dense around the historically useful 1.45 region, with
# intermediate values included so we are not only testing coarse steps.
GAMMA_QUEUE = [
    1.45, 1.44, 1.46, 1.435, 1.43, 1.47, 1.425, 1.42, 1.48,
    1.415, 1.40, 1.49, 1.50, 1.385, 1.38, 1.52, 1.36, 1.35,
    1.32, 1.30, 1.55, 1.60, 1.25, 1.65, 1.75, 1.85, 2.10, 1.00,
]
WINDOW_PRIORITY = [
    '125_149', '150_174', '100_124', '075_099', '050_074', '025_049', '000_024',
]


def load_driver():
    spec = importlib.util.spec_from_file_location('gamma_driver', DRIVER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def append_jsonl(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        handle.write(json.dumps(obj, sort_keys=True) + '\n')


def log(message):
    line = f'{now()} {message}'
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a') as handle:
        handle.write(line + '\n')
    print(line, flush=True)


def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def ps_lines():
    out = subprocess.run(['ps', '-eo', 'pid,stat,nlwp,pcpu,etime,args'], text=True, stdout=subprocess.PIPE).stdout
    return out.splitlines()[1:]


def live_workers(driver):
    workers = []
    root_s = str(ROOT / 'runs')
    for line in ps_lines():
        if root_s not in line or '/build/celluniverse' not in line:
            continue
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue
        pid = int(parts[0])
        args = parts[5]
        run_dir = None
        for tok in args.split():
            if tok.startswith(root_s):
                run_dir = Path(tok)
                break
        meta = {}
        if run_dir and (run_dir / 'run_meta.json').exists():
            try:
                meta = json.loads((run_dir / 'run_meta.json').read_text())
            except Exception:
                meta = {}
        workers.append({
            'pid': pid,
            'line': line.strip(),
            'run_dir': str(run_dir) if run_dir else '',
            'task_id': meta.get('task_id', run_dir.name if run_dir else ''),
            'batch': (meta.get('batch') or {}).get('name', ''),
            'gamma': meta.get('gamma', ''),
        })
    return workers


def write_status(driver, workers):
    status = {
        'updated_at': now(),
        'root': str(ROOT),
        'mode': 'r5_direct_refill_monitor',
        'active': [
            {
                'pid': w['pid'],
                'run_dir': w['run_dir'],
                'task_id': w['task_id'],
                'batch': w['batch'],
                'gamma': w['gamma'],
            }
            for w in workers
        ],
        'max_workers': MAX_WORKERS,
        'threads': driver.THREADS,
        'cpuset': driver.CPUSET,
    }
    driver.write_json(ROOT / 'status.json', status)


def run_updater():
    try:
        subprocess.run(['python3', str(UPDATER)], cwd=str(REPO), check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=90)
    except Exception as exc:
        log(f'updater_error {exc}')


def prune_true_failures():
    if not KNOWN.exists():
        return []
    pruned = []
    with KNOWN.open(newline='') as handle:
        for row in csv.DictReader(handle):
            if row.get('state') != 'running':
                continue
            if row.get('classification') != 'true_fail_or_needs_tuning':
                continue
            pid = row.get('pid')
            if not pid or not pid_alive(pid):
                continue
            run_dir = Path(row['run_dir'])
            marker = run_dir / 'PRUNED_GT_FAILURE'
            reason = {
                'time': now(),
                'event': 'pruned_true_gt_failure',
                'pid': int(pid),
                'run': row.get('run'),
                'batch': row.get('batch'),
                'gamma': row.get('gamma'),
                'first_bad_frame': row.get('first_bad_frame'),
                'missing': row.get('first_bad_missing'),
                'extra': row.get('first_bad_extra'),
                'run_dir': row.get('run_dir'),
            }
            marker.write_text(json.dumps(reason, indent=2, sort_keys=True) + '\n')
            append_jsonl(DECISIONS, reason)
            log(f"prune pid={pid} run={row.get('run')} bad_frame={row.get('first_bad_frame')} missing={row.get('first_bad_missing')} extra={row.get('first_bad_extra')}")
            try:
                os.kill(int(pid), signal.SIGTERM)
                time.sleep(3)
                if pid_alive(pid):
                    os.kill(int(pid), signal.SIGKILL)
            except Exception as exc:
                log(f'prune_kill_error pid={pid} err={exc}')
            pruned.append(row)
    return pruned


def existing_run_names():
    runs = ROOT / 'runs'
    if not runs.exists():
        return set()
    return {p.name for p in runs.iterdir() if p.is_dir()}


def build_queue(driver):
    batches = {b['name']: b for b in driver.BATCHES if b.get('initial_kind') != 'deferred_best_previous'}
    queue = []
    for gamma in GAMMA_QUEUE:
        for name in WINDOW_PRIORITY:
            if name in batches:
                queue.append((batches[name], float(gamma)))
    return queue


def launch_next(driver, workers):
    existing = existing_run_names()
    live_names = {w['task_id'] for w in workers}
    launched = []
    for batch, gamma in build_queue(driver):
        if len(workers) + len(launched) >= MAX_WORKERS:
            break
        tid = driver.task_id(batch, gamma)
        if tid in existing or tid in live_names:
            continue
        try:
            task = driver.launch_task(batch, gamma)
        except Exception as exc:
            append_jsonl(DECISIONS, {
                'time': now(), 'event': 'launch_failed', 'task_id': tid,
                'batch': batch['name'], 'gamma': gamma, 'error': str(exc),
            })
            log(f'launch_failed task={tid} err={exc}')
            continue
        launched.append({
            'pid': task['pid'],
            'run_dir': task['run_dir'],
            'task_id': task['task_id'],
            'batch': batch['name'],
            'gamma': gamma,
        })
        driver.append_log('Refill monitor launched gamma candidate', [
            f'task: `{tid}`',
            f'batch: `{batch["name"]}` gamma: `{gamma}`',
            'reason: free worker slot in corrected r5 direct sweep',
            'contract: original CellUniverse, CellLumen disabled, N2V2 enabled, brightness=1.0, trash elongation cap=1.25',
        ])
        log(f'launched task={tid} pid={task["pid"]}')
    return launched


def main():
    driver = load_driver()
    driver.ensure_dirs()
    log('monitor_start')
    append_jsonl(DECISIONS, {
        'time': now(),
        'event': 'r5_refill_monitor_started',
        'max_workers': MAX_WORKERS,
        'threads': driver.THREADS,
        'cpuset': driver.CPUSET,
        'gamma_queue': GAMMA_QUEUE,
        'window_priority': WINDOW_PRIORITY,
    })
    while True:
        workers = live_workers(driver)
        write_status(driver, workers)
        run_updater()
        pruned = prune_true_failures()
        if pruned:
            time.sleep(2)
            workers = live_workers(driver)
            write_status(driver, workers)
            run_updater()
        workers = live_workers(driver)
        launched = launch_next(driver, workers)
        if launched:
            time.sleep(2)
            workers = live_workers(driver)
            write_status(driver, workers)
            run_updater()
        else:
            write_status(driver, workers)
        log(f'heartbeat active={len(workers)} launched={len(launched)}')
        time.sleep(SLEEP_SECONDS)


if __name__ == '__main__':
    main()
