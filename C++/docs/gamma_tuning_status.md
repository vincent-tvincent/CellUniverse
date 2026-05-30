# Gamma Tuning Status

Last updated: 2026-05-30T10:38:22-0700

Human-visible best-so-far CSV:

- `docs/gamma_window_best_so_far.csv`

Detailed per-run CSV remains outside the repo under the tuning root:

- `/home/puv/output_fluo/tuning_fluo_gt_20260529/gamma_25frame_original_n2v2_oldparams_interleaved_r3/metrics/gamma_known_results.csv`

Current experiment root:

- `/home/puv/output_fluo/tuning_fluo_gt_20260529/gamma_25frame_original_n2v2_oldparams_interleaved_r3`

Current runtime:

- Method: original CellUniverse tracking, CellLumen disabled.
- N2V2: enabled, using raw TIFF input before normalization.
- Worker cap: `4` workers x `24` threads = `96` thread tokens.
- CPU set: `0-99`.
- Disk usage recorded by scheduler: `63577952` bytes, below the 20 GB cap.
- Queued remaining: `14`.

Active workers:

- `075_099__gamma_1p43` pid=472560 gamma=1.43 batch=075_099
- `100_124__gamma_1p42` pid=473810 gamma=1.42 batch=100_124
- `050_074__gamma_1p43` pid=473835 gamma=1.43 batch=050_074
- `050_074__gamma_1p42` pid=473860 gamma=1.42 batch=050_074

## Best So Far Snapshot

```csv
time,batch,best_gamma_so_far,best_run_so_far,classification,frames_scored,clean_prefix_frames,last_frame,complete_window,note
2026-05-30T10:37:30-0700,000_024,1.75,000_024__gamma_1p75,true_fail_or_needs_tuning,13,12,12,0,extra predicted cell/split before completing window
2026-05-30T10:37:30-0700,025_049,1.45,025_049__gamma_1p45_resume28_relaxed,true_fail_or_needs_tuning,12,11,40,0,missed GT cell/split before completing window
2026-05-30T10:37:30-0700,050_074,1.45,050_074__gamma_1p45,stopped_clean_so_far,7,7,56,0,
2026-05-30T10:37:30-0700,075_099,1.45,075_099__gamma_1p45,possible_one_frame_early_split,11,11,85,0,
2026-05-30T10:37:30-0700,100_124,1.45,100_124__gamma_1p45,true_fail_or_needs_tuning,3,2,102,0,missed GT cell/split before completing window
2026-05-30T10:37:30-0700,125_149,1.45,125_149__gamma_1p45,no_scored_output,0,0,,0,
2026-05-30T10:37:30-0700,150_174,1.45,150_174__gamma_1p45,no_scored_output,0,0,,0,
```

## Note on gamma 1.45

The old export `/home/puv/output_fluo/output_fluo_0-239_20260525_022357` is evidence that gamma `1.45` used to work well for the earliest frames. In particular, frame `11` has `8` non-trash predicted cells there, matching the GT count.

The controlled reruns before the N2V2-order fix were not identical experiments even with the same gamma: they used a loaded-stack N2V2 route. The current code now routes N2V2 through raw TIFF input first, then normalizes the N2V2 output, so newly launched gamma runs after `2026-05-30T10:36` are the runs to trust for this comparison.
