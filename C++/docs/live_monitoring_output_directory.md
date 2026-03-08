# Live Monitoring Output Directory

This document records the exact method used to monitor a running output directory and report frame-by-frame split results.

## Goal

For each completed frame, report:

- whether any cell split
- which parent cell IDs split
- how many cells split

## Directory and Log

Given an output directory:

`/Volumes/vincent/celluniverse/outputs/output_<preset>_<timestamp>`

the monitored file is:

`debug_log.txt`

Quick checks:

```bash
ls -la /Volumes/vincent/celluniverse/outputs/output_<preset>_<timestamp>
tail -n 50 /Volumes/vincent/celluniverse/outputs/output_<preset>_<timestamp>/debug_log.txt
```

## Completion and Split Markers

The parser uses these log markers:

- Split accepted line:
  - `[Split Accepted] <cell_id> split in frame <N> (...)`
- Frame completion boundary:
  - `Saving images for frame <N>...`
  - followed by `Done`

Interpretation:

- Count accepted split parents for frame `N`.
- Emit frame result only when `Done` is reached for the same `Saving images for frame <N>...`.

## Live Monitor Command

```bash
tail -n0 -F /Volumes/vincent/celluniverse/outputs/output_<preset>_<timestamp>/debug_log.txt | perl -ne '
if (/\[Split Accepted\] (\S+) split in frame (\d+)/) {
  my ($cell,$f)=($1,$2);
  my $k="$f|$cell";
  if (!exists $seen{$k}) {
    $seen{$k}=1;
    $split_count{$f}++;
    $split_list{$f} = defined($split_list{$f}) && length($split_list{$f}) ? "$split_list{$f},$cell" : $cell;
  }
}
if (/Saving images for frame (\d+)/) {
  $current_frame=$1;
}
if (/^Done\s*$/ && defined($current_frame)) {
  my $count = $split_count{$current_frame} // 0;
  my $list = $split_list{$current_frame} // "none";
  print "FRAME_COMPLETE frame=$current_frame split_count=$count split_cells=$list\n";
  $|=1;
  undef $current_frame;
}
if (/Processing finished/) {
  print "MONITOR_END processing_finished\n";
  $|=1;
  exit 0;
}
'
```

## Expected Output Format

The monitor emits one line per completed frame:

```text
FRAME_COMPLETE frame=<N> split_count=<K> split_cells=<id1,id2,...|none>
```

Example:

```text
FRAME_COMPLETE frame=3 split_count=2 split_cells=123...,e907...
```

## Stop Monitoring

Press `Ctrl+C` in the monitor session.

## Notes

- This logic reports accepted splits only (not attempted or skipped splits).
- If a frame has no accepted split lines before its `Done`, it reports `split_count=0 split_cells=none`.
- To investigate false negatives, inspect `[Split Skip]` lines for the same frame.
