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

GAMMA_PRIORITY = [1.45, 1.46, 1.47, 1.48, 1.49, 1.50, 1.52, 1.43, 1.42, 1.40, 1.385, 1.38, 1.36, 1.35, 1.32, 1.30, 1.28, 1.25, 1.55, 1.60, 1.62, 1.65, 1.68, 1.75, 1.85, 2.10, 1.00]
BASELINE_GAMMA = 1.45
MAX_THREAD_TOKENS = 96
ACCEPTED_BATCHES = set()


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
        rows = list(csv.DictReader(f))
    return [row for row in rows if metric_row_is_current(row)]


def metric_row_is_current(row):
    run_dir = row.get('run_dir')
    if not run_dir:
        return True
    if row.get('batch') == '125_149':
        meta_path = Path(run_dir) / 'run_meta.json'
        initial = None
        if meta_path.exists():
            try:
                initial = json.loads(meta_path.read_text()).get('initial')
            except Exception:
                initial = None
        if initial and Path(initial) != r3.GTFILLED_INITIAL_125_149:
            return False
        if not initial:
            try:
                if not ((Path(run_dir) / 'initial_used.csv').exists() and (Path(run_dir) / 'initial_used.csv').read_bytes() == r3.GTFILLED_INITIAL_125_149.read_bytes()):
                    return False
            except Exception:
                return False
    log_path = Path(run_dir) / 'debug_log.txt'
    if not log_path.exists():
        return False
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
        # Correct raw-TIFF N2V2 runs have Fluo scale values around 100-250.
        # Old pre-fix runs used normalized input and show scale around 0.x.
        return scale >= 10.0
    return True


def metric_keys():
    keys = set()
    for row in metrics_rows():
        try:
            keys.add((row['batch'], float(row['gamma'])))
        except Exception:
            pass
    return keys


def passed_batches():
    accepted = set(ACCEPTED_BATCHES)
    accepted.update({row.get('batch') for row in metrics_rows() if row.get('passed') == '1'})
    known = ROOT / 'metrics' / 'gamma_known_results.csv'
    if known.exists():
        try:
            with known.open(newline='') as f:
                for row in csv.DictReader(f):
                    if row.get('complete_window') == '1' and row.get('classification') not in {'true_fail_or_needs_tuning', 'running_no_scored_output'}:
                        accepted.add(row.get('batch'))
        except Exception:
            pass
    return accepted


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
            workers.append({'pid': pid, 'run_dir': run_dir, 'task_id': meta.get('task_id'), 'batch': meta['batch']['name'], 'gamma': float(meta['gamma']), 'threads': int(meta.get('threads', r3.THREADS))})
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
            marker_names = ['INTERRUPTED_FOR_PRIORITY', 'INTERRUPTED_FOR_16T_REALLOC', 'INTERRUPTED_FOR_FASTSPLIT_BRANCH', 'INTERRUPTED_FOR_TARGETED_SPLITGATE_BRANCH', 'INTERRUPTED_FOR_INTERMEDIATE_GAMMA_BRANCH', 'INTERRUPTED_FOR_LATE_RELAUNCH', 'INTERRUPTED_FOR_WINDOW_REBALANCE', 'INTERRUPTED_FOR_RESUME126_PROBE', 'PRUNED_GT_FAILURE']
            marker = next((name for name in marker_names if (Path(meta['run_dir']) / name).exists()), None)
            if marker:
                r3.append_jsonl(r3.DECISIONS, {'time': now(), 'event': 'priority_score_skipped_interrupted', 'task_id': meta.get('task_id'), 'run_dir': meta.get('run_dir'), 'marker': marker})
                continue
            if not r3.run_contract_current({'batch': batch['name'], 'run_dir': meta.get('run_dir')}):
                r3.append_jsonl(r3.DECISIONS, {'time': now(), 'event': 'priority_score_skipped_invalid_contract', 'task_id': meta.get('task_id'), 'run_dir': meta.get('run_dir'), 'batch': batch['name']})
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
    passed_names = passed_batches()
    by_name = {b['name']: b for b in r3.BATCHES if b.get('initial_kind') != 'deferred_best_previous' and b['name'] not in passed_names}
    failed_names = set(failed_batches())
    late_priority = {'125_149': 0, '150_174': 1, '100_124': 2, '050_074': 3, '075_099': 4, '000_024': 5}
    batch_order = sorted(by_name.values(), key=lambda b: late_priority.get(b['name'], 99))
    queue = []

    # Cover all windows, but put the under-sampled dense late windows first.
    # Each gamma pass walks the 25-frame batches in order, so the live pool
    # naturally spans early/mid/late windows while still prioritizing failures.
    for gamma in GAMMA_PRIORITY:
        for b in batch_order:
            if not task_exists(b['name'], gamma, live, done):
                reason = 'failed-window-round-robin' if b['name'] in failed_names else 'gamma-grid-round-robin'
                queue.append((b, gamma, reason))

    for b in batch_order:
        if not task_exists(b['name'], BASELINE_GAMMA, live, done):
            queue.append((b, BASELINE_GAMMA, 'baseline-window-sweep'))
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
        'active_thread_tokens': sum(int(w.get('threads', r3.THREADS)) for w in live),
        'max_thread_tokens': MAX_THREAD_TOKENS,
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
        while len(live) < r3.MAX_WORKERS and sum(int(w.get('threads', r3.THREADS)) for w in live) + r3.THREADS <= MAX_THREAD_TOKENS and queue:
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
