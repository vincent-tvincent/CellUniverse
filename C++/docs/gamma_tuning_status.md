# Gamma Tuning Status

Last updated: 2026-05-30T11:06:30-0700

Human-visible best-so-far CSV:

- `docs/gamma_window_best_so_far.csv`

Detailed per-run CSV remains outside the repo under the tuning root:

- `/home/puv/output_fluo/tuning_fluo_gt_20260529/gamma_25frame_original_n2v2_oldparams_interleaved_r3/metrics/gamma_known_results.csv`

Current experiment root:

- `/home/puv/output_fluo/tuning_fluo_gt_20260529/gamma_25frame_original_n2v2_oldparams_interleaved_r3`

Current runtime:

- Method: original CellUniverse tracking, CellLumen disabled.
- N2V2: enabled, using raw TIFF input before normalization.
- Scheduler: round-robin across 25-frame windows so the pool does not over-focus on one range; the final `175-194` GT-backed window is queued automatically once a usable `150-174` predecessor is available.
- Worker cap: `12` workers x `8` threads = `96` thread tokens. Existing old workers may remain `12` threads until they finish, and the scheduler token gate prevents total live thread tokens from exceeding `96`.
- CPU set: `0-99`.
- Disk usage recorded by scheduler: `65846775` bytes, below the 20 GB cap.
- Queued remaining: scheduler recomputes after the expanded gamma queue is loaded.

Active workers:

- `000_024__gamma_1p48` pid=486385 gamma=1.48 batch=000_024
- `025_049__gamma_1p40` pid=486446 gamma=1.4 batch=025_049
- `100_124__gamma_1p42` pid=488690 gamma=1.42 batch=100_124
- `150_174__gamma_1p45` pid=489123 gamma=1.45 batch=150_174
- `050_074__gamma_1p43` pid=489136 gamma=1.43 batch=050_074
- `075_099__gamma_1p43` pid=489149 gamma=1.43 batch=075_099
- `125_149__gamma_1p43` pid=489189 gamma=1.43 batch=125_149
- `150_174__gamma_1p43` pid=489202 gamma=1.43 batch=150_174

## Best So Far Snapshot

```csv
time,batch,best_gamma_so_far,best_run_so_far,classification,frames_scored,clean_prefix_frames,last_frame,complete_window,note
2026-05-30T10:52:30-0700,000_024,1.75,000_024__gamma_1p75,true_fail_or_needs_tuning,13,12,12,0,extra predicted cell/split before completing window
2026-05-30T10:52:30-0700,025_049,1.45,025_049__gamma_1p45_resume28_relaxed,true_fail_or_needs_tuning,12,11,40,0,missed GT cell/split before completing window
2026-05-30T10:52:30-0700,050_074,1.45,050_074__gamma_1p45,stopped_clean_so_far,7,7,56,0,
2026-05-30T10:52:30-0700,075_099,1.45,075_099__gamma_1p45,possible_one_frame_early_split,11,11,85,0,
2026-05-30T10:52:30-0700,100_124,1.45,100_124__gamma_1p45,true_fail_or_needs_tuning,3,2,102,0,missed GT cell/split before completing window
2026-05-30T10:52:30-0700,125_149,1.43,125_149__gamma_1p43,running_no_scored_output,0,0,,0,
2026-05-30T10:52:30-0700,150_174,1.43,150_174__gamma_1p43,running_no_scored_output,0,0,,0,
```

## Note on gamma 1.45

The old export `/home/puv/output_fluo/output_fluo_0-239_20260525_022357` is evidence that gamma `1.45` used to work well for the earliest frames. In particular, frame `11` has `8` non-trash predicted cells there, matching the GT count.

The controlled reruns before the N2V2-order fix were not identical experiments even with the same gamma: they used a loaded-stack N2V2 route. The current code now routes N2V2 through raw TIFF input first, then normalizes the N2V2 output, so newly launched gamma runs after `2026-05-30T10:36` are the runs to trust for this comparison. As of `2026-05-30T11:08`, the queued gamma set was expanded around the old working band: `1.45`, `1.46`, `1.47`, `1.48`, `1.49`, `1.50`, `1.52`, plus lower and wider probes for later condensed windows.

## 2026-05-30T23:09:21-0700 - Corrected Late-Window Evaluation
- Invalidated old conclusions from pre-fix normalized-input N2V2 runs; those run directories were archived under the tuning root.
- Invalidated greedy-matcher late-frame conclusions; dense frames now use scipy optimal assignment for GT/pred centroid matching.
- Backed up old greedy metrics and rebuilt `gamma_metrics.csv` from finished corrected raw-scale runs.
- Relaunched full-window `150_174` retries for gamma `1.35`, `1.40`, `1.45`, and `1.48` because optimal assignment showed the previous partial first-frame outputs were clean, not failures.
- Current CPU plan remains 12 workers x 8 threads = 96 thread tokens under the 100-core cap.
