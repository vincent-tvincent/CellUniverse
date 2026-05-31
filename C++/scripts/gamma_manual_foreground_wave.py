#!/usr/bin/env python3
"""Foreground gamma tuning wave runner.

Keeps CellUniverse workers as children of this foreground process and emits
heartbeats so the tool session does not reap detached workers.
"""
import importlib.util
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

REPO = Path('/home/puv/celluniverse/CellUniverse/C++')
SPEC = importlib.util.spec_from_file_location('drv', REPO / 'scripts/gamma_25frame_tuning_oldparams_interleaved_r3.py')
drv = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(drv)

TASKS = [
    ('000_024', 1.45, 'r6_recompiled_canary_original_gamma'),
    ('075_099', 1.45, 'r6_recompiled_mid_canary'),
    ('125_149', 1.40, 'r6_recompiled_late_lower_no_valley_canary'),
]


def launch(batch, gamma, label):
    initial = drv.prepare_initial(batch)
    config = drv.make_config(batch, gamma)
    base = drv.task_id(batch, gamma)
    tid = f'{base}__{label}_{int(time.time())}'
    run_dir = drv.DIRS['runs'] / tid
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config, run_dir / 'config_used.yaml')
    shutil.copy2(initial, run_dir / 'initial_used.csv')
    cmd = ['taskset', '-c', drv.CPUSET, str(drv.BINARY), str(batch['run_start']), str(batch['run_end']), drv.INPUT_PATTERN, str(run_dir), str(config), str(initial)]
    log = (run_dir / 'debug_log.txt').open('ab', buffering=0)
    log.write((
        f"[CMD] {' '.join(cmd)}\nStarted: {drv.now()}\n"
        f"method: original_celluniverse_no_celllumen\n"
        f"n2v2_network: enabled\ncontrast_gamma: {gamma}\n"
        f"batch: {batch['name']} score={batch['score_start']}..{batch['score_end']}\n"
        f"manual_wave: r6_recompiled_foreground\n"
        f"threads: {drv.THREADS} cpuset: {drv.CPUSET}\n"
        "==========================================\n"
    ).encode())
    proc = subprocess.Popen(cmd, cwd=str(drv.BINARY.parent), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=drv.env_for_threads(), close_fds=True)
    meta = {'task_id': tid, 'batch': batch, 'gamma': gamma, 'pid': proc.pid, 'status': 'running', 'run_dir': str(run_dir), 'config': str(config), 'initial': str(initial), 'threads': drv.THREADS, 'cpuset': drv.CPUSET, 'launched_at': drv.now(), 'manual_wave': 'r6_recompiled_foreground', 'reason': label}
    drv.write_json(run_dir / 'run_meta.json', meta)
    drv.append_jsonl(drv.DECISIONS, {'time': drv.now(), 'event': 'r6_recompiled_foreground_launch', **meta})
    return {'proc': proc, 'log': log, **meta}


def main():
    drv.ensure_dirs()
    batches = {b['name']: b for b in drv.BATCHES}
    active = [launch(batches[name], gamma, label) for name, gamma, label in TASKS]
    drv.write_json(drv.STATUS, {'updated_at': drv.now(), 'root': str(drv.ROOT), 'mode': 'r6_recompiled_foreground', 'active': [{'pid': a['pid'], 'run_dir': a['run_dir'], 'task_id': a['task_id'], 'batch': a['batch']['name'], 'gamma': a['gamma']} for a in active], 'max_workers': len(active), 'threads': drv.THREADS, 'cpuset': drv.CPUSET})
    print(json.dumps({'event': 'r6_recompiled_started', 'active': [{'pid': a['pid'], 'task_id': a['task_id']} for a in active]}, indent=2), flush=True)
    while active:
        still = []
        for a in active:
            code = a['proc'].poll()
            if code is None:
                still.append(a)
            else:
                a['log'].close()
                print(json.dumps({'event': 'worker_exit', 'time': drv.now(), 'task_id': a['task_id'], 'pid': a['pid'], 'returncode': code}, sort_keys=True), flush=True)
        active = still
        drv.write_json(drv.STATUS, {'updated_at': drv.now(), 'root': str(drv.ROOT), 'mode': 'r6_recompiled_foreground', 'active': [{'pid': a['pid'], 'run_dir': a['run_dir'], 'task_id': a['task_id'], 'batch': a['batch']['name'], 'gamma': a['gamma']} for a in active], 'max_workers': len(TASKS), 'threads': drv.THREADS, 'cpuset': drv.CPUSET})
        print(json.dumps({'event': 'heartbeat', 'time': drv.now(), 'active': [{'pid': a['pid'], 'task_id': a['task_id']} for a in active]}, sort_keys=True), flush=True)
        time.sleep(5)
    print(json.dumps({'event': 'r6_recompiled_complete', 'time': drv.now()}, sort_keys=True), flush=True)

if __name__ == '__main__':
    main()
