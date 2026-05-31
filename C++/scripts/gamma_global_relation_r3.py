#!/usr/bin/env python3
"""Summarize window-local gamma results and fit simple global gamma hints.

This is analysis-only. It reads the current r3 known-results/metrics outputs and
writes docs artifacts so the running sweep can be watched without overwriting any
tracking config.
"""

import csv
import math
from pathlib import Path

ROOT = Path('/home/puv/output_fluo/tuning_fluo_gt_20260529/gamma_25frame_original_n2v2_oldparams_interleaved_r3')
REPO = Path('/home/puv/celluniverse/CellUniverse/C++')
KNOWN = ROOT / 'metrics/gamma_known_results.csv'
METRICS = ROOT / 'metrics/gamma_metrics.csv'
OUT_CSV = REPO / 'docs/gamma_global_relation_live.csv'
OUT_MD = REPO / 'docs/gamma_global_relation_live.md'


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def to_float(value, default=math.nan):
    try:
        if value == '':
            return default
        return float(value)
    except Exception:
        return default


def batch_end(batch):
    ends = {
        '000_024': 24,
        '025_049': 49,
        '050_074': 74,
        '075_099': 99,
        '100_124': 124,
        '125_149': 149,
        '150_174': 174,
        '175_199_gt_to194': 194,
    }
    return ends.get(batch, 999)


def rank_known(row):
    complete = row.get('last_frame') not in ('', None) and int(float(row.get('last_frame') or 0)) >= batch_end(row.get('batch'))
    clean = int(float(row.get('clean_prefix_frames') or 0))
    bad = row.get('first_bad_frame') not in ('', None)
    class_rank = {
        'stopped_clean_so_far': 0,
        'promising_so_far': 0,
        'recovered_after_possible_early_split': 0,
        'possible_one_frame_early_split': 2,
        'true_fail_or_needs_tuning': 3,
        'running_no_scored_output': 4,
        'no_scored_output': 5,
    }.get(row.get('classification'), 9)
    max_dist = to_float(row.get('max_dist_last'), 999.0)
    return (not complete, -clean, bad, class_rank, max_dist)


def main():
    known = read_csv(KNOWN)
    metrics = read_csv(METRICS)
    best = []
    for batch in sorted({r.get('batch') for r in known if r.get('batch')}):
        vals = [r for r in known if r.get('batch') == batch]
        if vals:
            best.append(sorted(vals, key=rank_known)[0])
    metrics_by_run = {Path(m.get('run_dir', '')).name: m for m in metrics if m.get('run_dir')}
    rows = []
    for row in best:
        gamma = f"{to_float(row.get('gamma')):.2f}"
        metric = metrics_by_run.get(row.get('run', ''), {})
        rows.append({
            'batch': row.get('batch', ''),
            'best_gamma_live': gamma,
            'classification': row.get('classification', ''),
            'complete_window': int(row.get('last_frame') not in ('', None) and int(float(row.get('last_frame') or 0)) >= batch_end(row.get('batch'))),
            'clean_prefix_frames': row.get('clean_prefix_frames', ''),
            'last_frame': row.get('last_frame', ''),
            'first_bad_frame': row.get('first_bad_frame', ''),
            'gt_mean_count': metric.get('gt_mean_count', ''),
            'pred_mean_count': metric.get('pred_mean_count', ''),
            'mean_dist': metric.get('mean_dist', row.get('mean_dist_last', '')),
            'max_dist': metric.get('max_dist', row.get('max_dist_last', '')),
            'run': row.get('run', ''),
        })
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open('w', newline='') as handle:
        fields = ['batch', 'best_gamma_live', 'classification', 'complete_window', 'clean_prefix_frames', 'last_frame', 'first_bad_frame', 'gt_mean_count', 'pred_mean_count', 'mean_dist', 'max_dist', 'run']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    passed = [r for r in rows if r['complete_window'] == 1 and r['classification'] not in {'true_fail_or_needs_tuning', 'no_scored_output', 'running_no_scored_output'}]
    gammas = [to_float(r['best_gamma_live']) for r in passed if not math.isnan(to_float(r['best_gamma_live']))]
    global_line = 'No global gamma conclusion yet: not all windows have a passing completed local result.'
    if len(passed) >= 2:
        global_line = f"Current completed-window gamma range: {min(gammas):.2f}..{max(gammas):.2f}; keep fitting after late windows finish."
    with OUT_MD.open('w') as handle:
        handle.write('# Live Gamma Relation\n\n')
        handle.write(global_line + '\n\n')
        handle.write('This file is live analysis only. It does not overwrite default YAML. GT remains the primary source; recent successful Fluo runs are cross-validation.\n\n')
        handle.write('| batch | gamma | class | complete | last | gt_mean_count | max_dist | run |\n')
        handle.write('|---|---:|---|---:|---:|---:|---:|---|\n')
        for r in rows:
            handle.write(f"| {r['batch']} | {r['best_gamma_live']} | {r['classification']} | {r['complete_window']} | {r['last_frame']} | {r['gt_mean_count']} | {r['max_dist']} | `{r['run']}` |\n")
    print(f'wrote {OUT_CSV}')
    print(f'wrote {OUT_MD}')


if __name__ == '__main__':
    main()
