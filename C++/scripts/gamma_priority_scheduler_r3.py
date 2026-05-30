#!/usr/bin/env python3
import csv
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path

DRIVER = Path('/home/puv/celluniverse/CellUniverse/C++/scripts/gamma_25frame_tuning_oldparams_interleaved_r3.py')
spec = importlib.util.spec_from_file_location('r3', DRIVER)
r3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r3)
ROOT = r3.ROOT
SLEEP_SECONDS = 20

GAMMA_PRIORITY = [1.43, 1.42, 1.75, 1.25, 2.1, 1.0]
BASELINE_GAMMA = 1.45


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def append_log(title, lines):
    r3.LOGBOOK.parent.mkdir(parents=True, exist_ok=True)
    with r3.LOGBOOK.open('a') as f:
        f.write(f'\n## {now()} - {title}\n')
        for line in lines:
            f.write(f'- {line}\n')


def metrics_rows():
    if not r3.METRICS.exists():
        return []
    with r3.METRICS.open(newline='') as f:
        return list(csv.DictReader(f))


def metric_keys():
    keys = set()
    for row in metrics_rows():
        try:
            keys.add((row['batch'], float(row['gamma'])))
        except Exception:
            pass
    return keys


def failed_batches():
    failed = []
    passed = set()
    tested = {}
    for row in metrics_rows():
        batch = row.get('batch')
        try:
            no_output = float(row.get('pred_mean_count', 0)) == 0.0 and int(float(row.get('matched_cells', 0))) == 0
        except Exception:
            no_output = False
        try:
            tested.setdefault(batch, set()).add(float(row.get('gamma')))
        except Exception:
            pass
        if row.get('passed') == '1':
            passed.add(batch)
        elif batch and not no_output and batch not in failed:
            failed.append(batch)
    names = [b for b in failed if b not in passed]
    batch_order = {b['name']: i for i, b in enumerate(r3.BATCHES)}
    return sorted(names, key=lambda b: (-len(tested.get(b, set())), batch_order.get(b, 999)))


def live_workers():
    proc = subprocess.run(['ps', '-eo', 'pid,args'], text=True, stdout=subprocess.PIPE, check=False)
    workers = []
    for line in proc.stdout.splitlines()[1:]:
        if str(ROOT) not in line or '/build/celluniverse' not in line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        pid = int(parts[0])
        args = parts[1]
        run_dir = None
        for token in args.split():
            if str(ROOT / 'runs') in token:
                run_dir = token
                break
        if not run_dir:
            continue
        meta_path = Path(run_dir) / 'run_meta.json'
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
            workers.append({'pid': pid, 'run_dir': run_dir, 'task_id': meta.get('task_id'), 'batch': meta['batch']['name'], 'gamma': float(meta['gamma'])})
        except Exception:
            continue
    return workers


def score_finished(live):
    live_pids = {w['pid'] for w in live}
    done = metric_keys()
    for meta_path in sorted((ROOT / 'runs').glob('*/run_meta.json')):
        try:
            meta = json.loads(meta_path.read_text())
            batch = meta['batch']
            gamma = float(meta['gamma'])
            key = (batch['name'], gamma)
            if key in done or int(meta['pid']) in live_pids:
                continue
            if (Path(meta['run_dir']) / 'INTERRUPTED_FOR_PRIORITY').exists():
                r3.append_jsonl(r3.DECISIONS, {'time': now(), 'event': 'priority_score_skipped_interrupted', 'task_id': meta.get('task_id'), 'run_dir': meta.get('run_dir')})
                continue
            row = r3.score_task(Path(meta['run_dir']), batch, gamma)
            r3.append_metric(row)
            r3.append_jsonl(r3.DECISIONS, {'time': now(), 'event': 'priority_run_scored', **row})
            done.add(key)
            append_log(f"Priority scored {meta['task_id']}", [
                f"passed: {row['passed']}",
                f"score: {row['score']}",
                f"missing/extra cells: {row['missing_cells']}/{row['extra_cells']}",
                f"missed/extra splits: {row['missed_splits']}/{row['extra_splits']}",
                f"mean/max distance: {row['mean_dist']}/{row['max_dist']}",
            ])
            r3.update_summaries()
        except Exception as exc:
            r3.append_jsonl(r3.DECISIONS, {'time': now(), 'event': 'priority_score_failed', 'meta': str(meta_path), 'error': str(exc)})


def task_exists(batch_name, gamma, live, done):
    if (batch_name, float(gamma)) in done:
        return True
    for w in live:
        if w['batch'] == batch_name and abs(float(w['gamma']) - float(gamma)) < 1e-9:
            return True
    run_dir = ROOT / 'runs' / f"{batch_name}__gamma_{gamma:.2f}".replace('.', 'p')
    return run_dir.exists() and (run_dir / 'run_meta.json').exists()


def build_queue(live):
    done = metric_keys()
    by_name = {b['name']: b for b in r3.BATCHES if b.get('initial_kind') != 'deferred_best_previous'}
    queue = []
    failed_names = failed_batches()
    for name in failed_names:
        b = by_name.get(name)
        if not b:
            continue
        for gamma in GAMMA_PRIORITY:
            if not task_exists(name, gamma, live, done):
                queue.append((b, gamma, 'failed-window-priority'))
    for b in by_name.values():
        if not task_exists(b['name'], BASELINE_GAMMA, live, done):
            queue.append((b, BASELINE_GAMMA, 'baseline-window-sweep'))
    for gamma in GAMMA_PRIORITY:
        for b in by_name.values():
            if not task_exists(b['name'], gamma, live, done):
                queue.append((b, gamma, 'gamma-grid'))
    return queue


def write_status(live, queue_len):
    r3.write_json(r3.STATUS, {
        'updated_at': now(),
        'root': str(ROOT),
        'queued': queue_len,
        'active': [{'task_id': w['task_id'], 'pid': w['pid'], 'batch': w['batch'], 'gamma': w['gamma'], 'run_dir': w['run_dir']} for w in live],
        'completed': len(metric_keys()),
        'disk_bytes': r3.disk_bytes(ROOT),
        'max_workers': r3.MAX_WORKERS,
        'threads': r3.THREADS,
        'scheduler': 'priority_r3',
    })


def main():
    r3.ensure_dirs()
    append_log('Started priority scheduler r3', [
        'Keeps existing workers alive, but prioritizes failed-window alternate gamma trials before continuing broad sweep.',
        f'Max active workers remains {r3.MAX_WORKERS} x {r3.THREADS} = {r3.MAX_WORKERS * r3.THREADS} thread tokens.',
    ])
    while True:
        live = live_workers()
        score_finished(live)
        live = live_workers()
        queue = build_queue(live)
        while len(live) < r3.MAX_WORKERS and queue:
            if r3.disk_bytes(ROOT) > r3.DISK_LIMIT_BYTES:
                r3.append_jsonl(r3.DECISIONS, {'time': now(), 'event': 'priority_disk_limit_pause', 'bytes': r3.disk_bytes(ROOT)})
                break
            batch, gamma, reason = queue.pop(0)
            try:
                launched = r3.launch_task(batch, gamma)
                r3.append_jsonl(r3.DECISIONS, {'time': now(), 'event': 'priority_launch', 'task_id': launched['task_id'], 'reason': reason, 'batch': batch['name'], 'gamma': gamma})
                append_log(f"Priority launched {launched['task_id']}", [f'reason: {reason}', f'pid: {launched["pid"]}'])
            except Exception as exc:
                r3.append_jsonl(r3.DECISIONS, {'time': now(), 'event': 'priority_launch_failed', 'batch': batch['name'], 'gamma': gamma, 'error': str(exc)})
            live = live_workers()
            queue = build_queue(live)
        write_status(live, len(queue))
        if not live and not queue:
            r3.update_summaries()
            append_log('Completed priority scheduler r3', [f'metrics: {r3.METRICS}', f'summary: {r3.SUMMARY}', f'relation: {r3.RELATION}'])
            break
        time.sleep(SLEEP_SECONDS)

if __name__ == '__main__':
    main()
