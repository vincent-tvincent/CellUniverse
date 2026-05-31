#!/usr/bin/env python3
"""Kill active r3 gamma runs once GT scoring shows a real failure.

This monitor is intentionally conservative: it only stops live CellUniverse
workers whose current scored prefix is classified as true_fail_or_needs_tuning by
our GT matcher. Possible one-frame early split grace and no-output states are not
killed.
"""

import importlib.util
import json
import os
import signal
import subprocess
import time
from pathlib import Path

REPO = Path('/home/puv/celluniverse/CellUniverse/C++')
ROOT = Path('/home/puv/output_fluo/tuning_fluo_gt_20260529/gamma_25frame_original_n2v2_recompiled_r6_20260531_025423')
UPDATER_PATH = REPO / 'scripts/gamma_update_known_results_r3.py'
DRIVER_PATH = REPO / 'scripts/gamma_25frame_tuning_oldparams_interleaved_r3.py'
GT_DRIVER_PATH = Path('/home/puv/output_fluo/tuning_fluo_gt_20260529/continuous_gt_watch_r26/continuous_gt_watch_r26.py')
DECISIONS = ROOT / 'decisions.jsonl'
LOGBOOK = ROOT / 'docs/gamma_tuning_logbook.md'
SLEEP_SECONDS = 60


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def append_jsonl(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        handle.write(json.dumps(obj, sort_keys=True) + '\n')


def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def load_status():
    active = []
    status_path = ROOT / 'status.json'
    if status_path.exists():
        try:
            active.extend(json.loads(status_path.read_text()).get('active', []))
        except Exception:
            pass

    seen = {(str(item.get('run_dir')), int(item.get('pid', -1))) for item in active}
    proc = subprocess.run(
        ['ps', '-eo', 'pid,args'],
        text=True,
        stdout=subprocess.PIPE,
        check=False,
    )
    for line in proc.stdout.splitlines()[1:]:
        if str(ROOT) not in line or '/build/celluniverse' not in line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        run_dir = None
        for token in parts[1].split():
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
            batch = meta.get('batch', {})
            item = {
                'pid': pid,
                'run_dir': str(run_dir),
                'task_id': meta.get('task_id'),
                'batch': batch.get('name') if isinstance(batch, dict) else batch,
                'gamma': meta.get('gamma'),
            }
        except Exception:
            continue
        key = (item['run_dir'], item['pid'])
        if key not in seen:
            active.append(item)
            seen.add(key)
    return active


def main_once():
    updater = load_module('gamma_update_known_results_r3', UPDATER_PATH)
    driver = load_module('gamma_driver_r3', DRIVER_PATH)
    gt_driver = load_module('gt_driver_r26', GT_DRIVER_PATH)
    gt = gt_driver.load_gt()
    batch_by_name = {batch['name']: batch for batch in driver.BATCHES}
    killed = []
    for active in load_status():
        pid = active.get('pid')
        if not pid_alive(pid):
            continue
        run_dir = Path(active.get('run_dir', ''))
        batch = batch_by_name.get(active.get('batch'))
        if not batch or not run_dir.exists():
            continue
        interrupt_markers = [
            'INTERRUPTED_FOR_PRIORITY',
            'INTERRUPTED_FOR_16T_REALLOC',
            'INTERRUPTED_FOR_FASTSPLIT_BRANCH',
            'INTERRUPTED_FOR_TARGETED_SPLITGATE_BRANCH',
            'INTERRUPTED_FOR_INTERMEDIATE_GAMMA_BRANCH',
            'INTERRUPTED_FOR_LATE_RELAUNCH',
            'INTERRUPTED_FOR_WINDOW_REBALANCE',
            'INTERRUPTED_FOR_RESUME126_PROBE',
        ]
        if any((run_dir / marker).exists() for marker in interrupt_markers) or (run_dir / 'PRUNED_GT_FAILURE').exists():
            continue
        if not updater.preprocessing_contract_ok(run_dir):
            continue
        if not updater.initial_contract_ok(batch['name'], run_dir):
            continue
        row = updater.scan_run(gt, batch, active.get('gamma'), run_dir, 'running', pid)
        if row.get('classification') != 'true_fail_or_needs_tuning':
            continue
        reason = (
            f"GT live prune: first_bad_frame={row.get('first_bad_frame')} "
            f"missing={row.get('first_bad_missing')} extra={row.get('first_bad_extra')} "
            f"last_frame={row.get('last_frame')} gamma={active.get('gamma')}"
        )
        (run_dir / 'PRUNED_GT_FAILURE').write_text(reason + '\n')
        try:
            os.kill(int(pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        time.sleep(1)
        if pid_alive(pid):
            try:
                os.kill(int(pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        event = {'time': now(), 'event': 'live_gt_prune', 'pid': pid, 'task_id': active.get('task_id'), 'batch': batch['name'], 'gamma': active.get('gamma'), 'run_dir': str(run_dir), 'reason': reason, 'scan': row}
        append_jsonl(DECISIONS, event)
        killed.append(event)
    if killed:
        LOGBOOK.parent.mkdir(parents=True, exist_ok=True)
        with LOGBOOK.open('a') as handle:
            handle.write(f"\n## {now()} - Live GT pruned failing runs\n")
            for item in killed:
                handle.write(f"- `{item['task_id']}` pid={item['pid']}: {item['reason']}\n")
    return killed


def main():
    append_jsonl(DECISIONS, {'time': now(), 'event': 'live_pruner_started', 'sleep_seconds': SLEEP_SECONDS})
    while True:
        try:
            main_once()
        except Exception as exc:
            append_jsonl(DECISIONS, {'time': now(), 'event': 'live_pruner_error', 'error': str(exc)})
        time.sleep(SLEEP_SECONDS)


if __name__ == '__main__':
    main()
