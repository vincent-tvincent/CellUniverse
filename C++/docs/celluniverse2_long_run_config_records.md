# CellUniverse2 Long-Run Configuration Records

This document records every CellUniverse2 configuration/code version that has
been observed to process more than 50 continuous frames, and marks which of
those runs were also validated as correct against GT centers. It is meant to
preserve the exact evidence needed for future tuning: run path, frame range,
resume source, validation method, caveats, and the next known failure.

## Validation Rules

- A "continuous span" means consecutive dataset frames processed by one run.
- Fresh `0-to-n` runs are stronger evidence than resume runs.
- Resume runs are recorded with their source checkpoint and should not be used
  as proof that earlier frames still pass unless the source run is also listed.
- GT labels are a guide, not an absolute oracle. If a split appears one frame
  early but both daughters map cleanly to GT in the next frame, record it as a
  GT timing caveat rather than forcing the pipeline to match the delayed label.
- For each tuning change, check whether the new threshold conflicts with prior
  fixes before launching a long smoke run.

## Completeness Audit

- Latest sweep date: `2026-06-13`.
- Sweep root:
  `/run/media/blue-lobster/disk3/celluniverse_output/outputs_fluo`.
- Inclusion rule for the sweep: any run directory with a `checkpoints/` folder
  containing a consecutive frame span longer than `50` frames.
- Current disk sweep result: `4` retained run directories currently contain a
  consecutive checkpoint span longer than `50` frames:
  `run_003_0_to_200_dense_flat_gate`,
  `run_058_from0_to200_stale_daughter_cleanup`, and
  `run_059_from0_to200_axis_place_shape_gate`, plus active fresh run
  `run_072_from0_to200_center_claim_guard`.
- Additional records retained from prior validation: older qualifying runs are
  still listed below when they were validated before cleanup or when their
  checkpoint/debug-log evidence had already been summarized. Their original run
  directories may no longer be retained on disk, so they are lower-confidence
  than the live sweep entries.
- Latest active-run status: `run_072_from0_to200_center_claim_guard` is the
  current fresh run for the center-claim guard checkout. It processed through
  checkpoint `85`, validated through checkpoint `84`, and stopped at the next
  true count failure, frame `85`. The old hidden-assignment symptom at frame
  `85` was replaced by a plain missing GT center (`GT148`).
  Focused follow-ups `run_060`, `run_061`, `run_062`, and `run_063` are recorded under
  `run_059` as frontier evidence, but none is promoted as a fresh long-run
  configuration because they resumed from frame `85` and did not establish a
  new continuous `0-to-n` span.

## Configuration Version Ledger

This ledger is the quick lookup for every known configuration/code version that
cleared at least one continuous 50+ frame span. Detailed records below are the
source of truth for promoted runs; the historical index keeps older or
checkpoint-only evidence visible without treating it as current validation.

| Version family | Strongest qualifying run | Continuous span | Validation strength | Main configuration/code identity | Status |
| --- | --- | ---: | --- | --- | --- |
| June 11 dense-flat gate | `run_003_0_to_200_dense_flat_gate` | `0-59` | checkpoint-only | CellUniverse2 deterministic split probes, local-bbox adaptive background, gamma split decay, dense-flat gate family | Superseded by later validator-backed runs. |
| Current midpoint rescue | `run_053_from0_to65_current_midpoint_rescue` | `0-65` | completed + validator-backed | current-midpoint rescue with future-window support, bio separation soft gate, split soft geometry gate, density gate | Fresh proof that current tuning crossed 50 frames. |
| Current midpoint rescue plus frame-18 corrections | `run_054_from0_to200_current_midpoint_rescue` | `0-85` processed, `0-84` validated | validator-backed to next true failure | `run_053` family plus rod-tip flatness threshold `1.70` and dense drifting no-dark `bridge_primary` protection | Superseded by later fixes; frame `85` hidden assignment failure. |
| No-dark future-axis gate | `run_055_from0_to200_no_dark_future_axis_gate` | `0-68` | count-validated stopped run | rejects no-dark raw PCA bridge projected from far reverted immediate-future snap | Provisional comparison version; preserved early behavior. |
| No-dark future-axis follow-up | `run_057_from0_to200_after_run010_followup` | `0-78` processed, `0-74` validated | validator-backed to next true failure | same family as `run_055`, retested from fresh frame `0` | First persistent mismatch at frame `75` from stale extra `cell_3011`. |
| Stale daughter cleanup | `run_058_from0_to200_stale_daughter_cleanup` | `0-85` processed, `0-84` validated | validator-backed to next true failure | stale split-daughter cleanup tokens: enabled, min age `4`, overlap `0.50` | Fixed old frame-75 count symptom; failed hidden assignment at frame `85`. |
| Axis-place high-shape cutoff | `run_059_from0_to200_axis_place_shape_gate` | `0-85` processed, `0-84` validated | validator-backed to next true failure | `run_058` family plus `pca_bridge_future_window_rod_tip_axis_place_max_parent_shape=2.20` | Current strongest fresh run; frame `85` is the frontier. |
| Signal-center claim guard | `run_072_from0_to200_center_claim_guard` | `0-85` processed, `0-84` validated | validator-backed to next true failure | current checkout plus local signal-center claim arbitration guard for future-supported blockers | Fixed the frame-85 hidden assignment form but still missed GT148 at frame `85`. |
| June 16 tuning halfspace `0.18` | `output_ubuntu_fluo_fresh_0-90_celluniverse2_tuning_20260616_093027` | `0-84` validated, frame `85` fails | validator-backed to known frontier | tuning YAML with `split_daughter_refit_halfspace_min_fraction=0.18` and current crowding/future-supported split logic | Stronger halfspace clamp stays clean through the early/mid frontier and still lands on the same old frame-85 GT148 miss rather than creating a new earlier regression. |
| Older June 9/10 smoke families | see Historical Fresh-Run Index | `0-50` to `0-80` | checkpoint-only to completed debug log | intermediate bio-soft, future geometry, overlap tolerance, and early CellUniverse2 smoke configurations | Historical only; not proof of current checkout. |

Non-promoted June 16 rebaseline note:

- Fresh `run_345_from0_to90_current_checkout_rebaseline` showed transient
  extras at frames `59` and `60` that resolved by frame `61`.
- Resume scouts `run_346_resume63_to140_from_run345` and
  `run_347_resume67_to140_threshold35` confirmed those mid-window issues were
  self-healing under the project’s accepted validation standard.
- The next persistent frontier remained frame `85`, with the same missing
  GT148 family rather than a new earlier collapse.

Additional June 16 active-tuning note:

- Fresh run
  `output_ubuntu_fluo_fresh_0-90_celluniverse2_tuning_20260616_093027`
  validated clean through frame `84` and then hit the known frame-`85` frontier
  under the accepted `matched<=35` rule.
- This is the first fresh validator-backed evidence in the current tuning
  branch that the stronger daughter-refit halfspace clamp
  (`split_daughter_refit_halfspace_min_fraction=0.18`) does not just preserve
  sparse early frames; it also survives the first split-heavy and bridge-heavy
  regions through frame `80` at the checked checkpoints.
- This is now a completed frontier check rather than an in-progress one: it
  confirms that the current halfspace-clamp tuning reaches the same old
  frame-`85` miss instead of breaking earlier clean windows.

## 2026-06-16: `output_ubuntu_fluo_fresh_0-90_celluniverse2_tuning_20260616_093027`

- Run path:
  `/run/media/blue-lobster/disk3/celluniverse_output/outputs_fluo/output_ubuntu_fluo_fresh_0-90_celluniverse2_tuning_20260616_093027`
- Continuous frames processed so far: at least `0-85` inclusive.
- Validated checked span so far: `0-84` inclusive under the accepted rule, with
  frame `85` as the first true miss.
- Run type: fresh start from frame `0`; no resume checkpoint used.
- Command shape:
  `./run_celluniverse.sh ubuntu_fluo_fresh_0-90_celluniverse2_tuning --cores 4`
- Runtime/config evidence:
  - CUDA preprocessing selected in the live log.
  - Parallel runtime confirmed `threads=4`.
  - Config echo uses the tuning YAML with
    `split_daughter_refit_halfspace_min_fraction: 0.18`.
  - Live validation confirmed no unmatched GT or predicted centers at frames
    `34`, `35`, `36`, `37`, `38`, `39`, `40`, `41`, `42`, `43`, `44`, `45`,
    `46`, `47`, `48`, `49`, `50`, `61`, `68`, `69`, and `80`.
  - Later validation found the first true miss at frame `85`:
    `pred=59 gt=60`, missing GT label `148`, nearest prediction `cell_11110`.
- Current interpretation:
  - The stronger halfspace clamp now has fresh validator-backed support well
    beyond the original early guardrail frames.
  - The branch still rejects weak-evidence bridge proposals when future
    support collapses, but it preserves valid future-supported bridge splits in
    later early/mid frames.
  - This version should be kept as an important comparison checkpoint even
    before the final frontier is known.

## Confirmed Records

### 2026-06-11: `run_003_0_to_200_dense_flat_gate`

- Run path:
  `/run/media/blue-lobster/disk3/celluniverse_output/outputs_fluo/celluniverse2_goal_0_200_20260611/run_003_0_to_200_dense_flat_gate`
- Continuous frames processed: `0-59` inclusive (`60` checkpoints retained).
- Validated continuous pass span: checkpoint-only evidence. No current
  validator-backed pass claim is attached to this run.
- Current status: stopped after saving checkpoint `frame_059.txt`; no
  `frame_060.txt` checkpoint or normal completion line was found in the
  retained output.
- Run type: fresh start from frame `0`; no resume checkpoint used.
- Command/config evidence:
  - `stdout.log` reports output folder
    `run_003_0_to_200_dense_flat_gate` and config file
    `config/config_celluniverse2.yaml`.
  - Config echo includes `celluniverse2_enabled: 1`,
    `fusionSplitPriorWindowEnabled: 1`, `fusionSplitPriorWindowSize: 3`,
    `fusionSplitPriorWindowMatchDistance: 28`,
    `fusionSplitPriorSoftGateEnabled: 1`,
    `fusionSplitPriorDynamicOverlapEnabled: 1`,
    `fusionSplitPriorPrepassFallbackEnabled: 1`, and
    `fusionReducePostSplitPerturbEnabled: 1`.
  - Runtime log shows CellUniverse2 deterministic split probes, local-bbox
    adaptive background, gamma split decay from base `1.45` toward target `1`,
    and perturb debug center export.
- Configuration identity:
  - Dense-flat gate tuning family from the June 11 branch of the CellUniverse2
    debug series.
  - This version predates the later current-midpoint rescue, no-dark
    future-axis projection gate, stale-daughter cleanup, and axis-place
    high-shape cutoff versions recorded below.
- Runtime evidence:
  - The run retained checkpoints `frame_000.txt` through `frame_059.txt`.
  - At frame `59`, the log reports `Optimize Done`, `split_attempts=12`,
    `split_accepted=8`, `final_cells=40`, and checkpoint save for
    `frame_059.txt`.
- Known caveats:
  - This is not promoted as a verified-good configuration because only the
    continuous checkpoint span was audited in this pass.
  - Later validator-backed fresh runs supersede it for current tuning decisions.

### 2026-06-13: `run_072_from0_to200_center_claim_guard`

- Run path:
  `/run/media/blue-lobster/disk3/celluniverse_output/outputs_fluo/celluniverse2_goal_0_200_20260612/run_072_from0_to200_center_claim_guard`
- Continuous frames processed: `0-85` inclusive (`86` checkpoints).
- Validated continuous pass span: `0-84` inclusive, with known one-frame GT
  timing caveats at frames `18` and `34`.
- Current status: stopped after frame `85` because validation found a true
  missing GT center.
- Run type: fresh start from frame `0`; no resume checkpoint used.
- Command shape:
  `CELLUNIVERSE_THREADS=4 CUDA_VISIBLE_DEVICES=0 ./build/celluniverse 0 200 ... config/config_celluniverse2.yaml config/embryo/initial_embryo_0_trash_labeled.csv`
- Runtime/config evidence:
  - Config echo includes `celluniverse2_enabled: 1`,
    `pca_bridge_future_window_match_distance: 28`,
    `pca_bridge_future_window_match_distance_per_frame: 4`, and
    `pca_bridge_future_window_rod_tip_axis_place_max_parent_shape: 2.2`.
  - This run was launched after building the current checkout with the
    signal-center claim guard added to local split-proposal center arbitration.
  - The guard prevents a later strong asymmetric future proposal from evicting
    a center already claimed by a clean future-supported blocker unless the
    blocker lacks clean future support or the new proposal is clearly better by
    score.
- Validation method:
  - Compared saved non-trash checkpoint centers against Fluo `01_GT/TRA`
    center labels with Hungarian assignment.
  - Treated the known one-frame early split count differences as GT timing
    caveats only when there was no unmatched GT and the following frame
    returned to exact count.
- Validated pass snapshot:
  - Frames `0-17` count-clean.
  - Frame `18`: pred `11`, GT `10`; no unmatched GT, one extra predicted
    daughter `cell_011`.
  - Frames `19-33` count-clean.
  - Frame `34`: pred `16`, GT `15`; no unmatched GT, one extra predicted
    daughter `cell_0110`.
  - Frames `35-57` count-clean.
  - Frame `58`: pred `30`, GT `30`, max assignment distance about `14.45`
    px.
  - Frame `59`: pred `37`, GT `37`, no unmatched GT or predicted centers,
    max assignment distance about `16.60` px after a seven-split burst.
  - Frames `60-64` count-clean. Frame `62` max assignment distance was about
    `28.12` px and frame `64` was about `27.87` px, both with no unmatched GT
    or predicted centers.
  - Frames `65-75` count-clean. Frames `68-74` showed higher max assignment
    distances, mostly recurring `cell_3011 -> GT672`, but had no unmatched GT
    or predicted centers. Frame `75` returned to max assignment distance about
    `19.14` px.
  - Frames `76-84` count-clean. Frame `80` reached pred `53`, GT `53`; frame
    `84` reached pred `55`, GT `55`, with max assignment distance about
    `22.82` px.
- Known caveats:
  - Frame `85` failed as pred `59`, GT `60`, unmatched GT label `[148]`.
    The worst matched assignment was no longer a hidden long-distance swap
    (`max=10.20` px), so the center-claim guard fixed the old failure shape
    but did not recover the missing daughter.

### 2026-06-13: `run_059_from0_to200_axis_place_shape_gate`

- Run path:
  `/run/media/blue-lobster/disk3/celluniverse_output/outputs_fluo/celluniverse2_goal_0_200_20260612/run_059_from0_to200_axis_place_shape_gate`
- Continuous frames processed: `0-85` inclusive (`86` checkpoints).
- Validated continuous pass span: `0-84` inclusive (`85` frames), with known
  one-frame GT timing caveats at frames `18` and `34`.
- Current status: stopped at the first true hidden assignment failure, frame
  `85`.
- Run type: fresh start from frame `0`; no resume checkpoint used.
- Command shape:
  `CELLUNIVERSE_THREADS=4 CUDA_VISIBLE_DEVICES=0 ./build/celluniverse 0 200 ... config/config_celluniverse2.yaml config/embryo/initial_embryo_0_trash_labeled.csv`
- Runtime evidence:
  - CUDA preprocessing selected in `stdout.log`.
  - Config echo includes
    `pca_bridge_future_window_rod_tip_axis_place_max_parent_shape: 2.2`.
  - Checkpoints existed through `checkpoints/frame_085.txt` when the run was
    stopped.
  - Frames `58-64` all matched GT counts; assignment-distance spot sweep over
    the same span had worst max distance about `28.12` px at frame `62`.
  - Frame `59` accepted a dense burst of seven split attempts and still matched
    GT after excluding trash: pred `37`, GT `37`.
  - Frame `60` also matched GT after the burst: pred `43`, GT `43`.
- Code/config identity:
  - Same current-checkout CellUniverse2 family as `run_058`, plus the scoped
    high-shape rod-tip axis-place cutoff:
    `pca_bridge_future_window_rod_tip_axis_place_max_parent_shape=2.20`.
  - In `Frame::trySplitCellPhased`, a future-aligned rod-tip proposal with
    `daughterSphereRadius > 0`, `gapStartBin <= -8`, aligned immediate-future
    center backing, and parent shape above the cutoff can still use the primary
    future-center seed pair, but the extra `bridge_axis_place` alternate is
    skipped.
- Validation method:
  - Compared checkpoint non-trash cell counts against GT label counts from
    `images/Fluo-N3DH-CE/01_GT/TRA/man_track%03d.tif`.
  - Monitored skip logs and the known early split families before recording the
    run as a qualified >50-frame version.
- Known caveats:
  - Frame `18`: pred `11`, GT `10`; frame `19` returned to pred `12`, GT `12`.
  - Frame `34`: pred `16`, GT `15`; frame `35` returned to pred `24`, GT `24`.
- Validated pass snapshot:
  - Frames `0-17` count-clean.
  - Frames `19-33` count-clean.
  - Frames `35-84` count-clean.
  - The frame-18 valid moderate `cell_01` `bridge_axis_place` split still
    accepted because its parent shape was below the `2.20` cutoff.
  - The frame-34 `cell_110` high-shape axis-place alternate was skipped, but
    `cell_110` still split correctly through `bridge_primary`; frame `35`
    matched GT.
  - Additional high-shape axis-place skips occurred for `cell_20` and
    `cell_201`; counts remained clean afterward.
  - A later frame-59 split burst remained count-correct, indicating the cutoff
    did not block the dense split family around frames `58-60`.
  - The historical `58-64` `cell_01100`/GT354 collapse region stayed clean by
    both count and assignment-distance spot checks.
  - The historical frame-75 through frame-77 persistent extra-cell window also
    stayed clean: frame `75` max assignment distance about `13.84` px, frame
    `76` about `24.28` px, frame `77` about `17.60` px, and frame `78` about
    `21.82` px.
  - Frames `79-84` also remained count-clean; assignment checks at `79-84` had
    max distances below `30` px except no concerning hidden swap.
- Next known failure:
  - Frame `85`: pred `60`, GT `60`, but Hungarian assignment max distance was
    about `109.07` px.
  - The bad assignment was again `cell_110101 -> GT148`, predicted at about
    `(202.5,142.9,86.1)` versus GT148 at about `(191.0,205.0,175.0)`.
  - The axis-place cutoff did fire (`Split Bridge AxisPlace Skip`), but the
    primary `rod_tip_split_fallback` still accepted by
    `futureRodTipPrimaryRescue=1` despite bad current-frame cost
    (`imageDiff=+12747.9`, `totalDiff=+12302.2`).
- Follow-up validation:
  - Added and built a cost-gate guard so high-shape rod-tip primary rescues
    using a far aligned future fallback must show current-frame image gain.
  - Focused resume run:
    `run_060_resume85_to88_rodtip_primary_gain_guard`, resumed from
    `run_059` frame `85` toward frame `88`.
  - The guard worked mechanically: `cell_11010` no longer accepted the bad
    `futureRodTipPrimaryRescue=1` current-frame placement; its cost gate
    rejected with positive image/total diff instead.
  - It did not solve the frontier: frames `85` and `86` were each short by one
    cell, leaving GT148 unmatched; frames `87` and `88` became short by seven
    and twelve cells respectively. The next fix should target the true
    `cell_11100`/GT148 split or placement, not re-enable the bad `cell_11010`
    false daughter.
  - Tried parameter-only widening in
    `run_061_resume85_to90_match30`: temporarily changed
    `pca_bridge_future_window_match_distance` from `28.0` to `30.0`.
    This did not fix frame `85` (`pred=59`, GT `60`, unmatched GT148), admitted
    broader proposals, and was reverted back to `28.0`. It is explicitly not a
    promoted configuration.
  - Added and built a delayed future-window cost rescue for PCA bridge splits
    that have clean future support but are dim or incomplete in the current
    frame. Focused resume run:
    `run_062_resume85_to90_delayed_future_cost`, resumed from `run_059`
    frame `85` toward frame `90`.
  - `run_062` fixed the persistent frame-86 GT148 miss: frame `86` matched
    GT exactly (`pred=67`, GT `67`, max assignment distance about `13.70` px).
    The accepted `cell_11100` split used `delayedFuturePcaBridgeNearMissRescue=1`
    with future support (`futureBoth=1`, `futureImmediate=0`) and current-frame
    image gain (`imageDiff=-5391.29`).
  - `run_062` still failed at the next frame: frame `87` was short by two
    cells (`pred=75`, GT `77`), with unmatched GT labels `171` and `388`.
    Because this was a resume-only frontier test and not a fresh `0-to-n`
    validation, it remains follow-up evidence under `run_059`, not a new
    qualifying long-run record.
  - Added and built a no-dark future-axis projection rescue for proposals with
    clean two-frame future support and no persistent parent. Focused resume
    run: `run_063_resume85_to90_no_dark_future_rescue`, resumed from
    `run_059` frame `85`.
  - `run_063` fixed the frame-87 misses from `run_062`: GT `171`, `388`, and
    the paired GT `164`/`395` assignments were covered by accepted splits for
    `cell_11110` and `cell_00110`. It still failed frame `87` overall because
    GT `84` was unmatched and `cell_20101` remained unsplit in the saved
    checkpoint. This remains resume-only frontier evidence, not a promoted
    long-run configuration.

### 2026-06-13: `run_058_from0_to200_stale_daughter_cleanup`

- Run path:
  `/run/media/blue-lobster/disk3/celluniverse_output/outputs_fluo/celluniverse2_goal_0_200_20260612/run_058_from0_to200_stale_daughter_cleanup`
- Continuous frames processed: `0-85` inclusive (`86` checkpoints).
- Validated continuous pass span: `0-84` inclusive (`85` frames), with
  known one-frame GT timing caveats at frames `17-18` and `34`.
- Current status: stopped at the first true assignment failure, frame `85`.
- Run type: fresh start from frame `0`; no resume checkpoint used.
- Command shape:
  `CELLUNIVERSE_THREADS=4 CUDA_VISIBLE_DEVICES=0 ./build/celluniverse 0 200 ... config/config_celluniverse2.yaml config/embryo/initial_embryo_0_trash_labeled.csv`
- Runtime evidence:
  - CUDA preprocessing selected in `stdout.log`.
  - Checkpoints existed through `checkpoints/frame_085.txt`.
  - Count validation against GT was clean for frames `70-84`, including
    frame `75`, where `run_057` had a persistent extra `cell_3011`.
- Code/config identity:
  - Same current-checkout CellUniverse2 family as `run_057`, plus the scoped
    stale split-daughter cleanup wiring:
    - `celluniverse2_stale_daughter_cleanup_enabled=true`
    - `celluniverse2_stale_daughter_cleanup_min_age=4`
    - `celluniverse2_stale_daughter_cleanup_overlap=0.50`
  - The patch records background-looking split daughters whose signal-center
    rescue cannot find a compatible center, then allows a later severe overlap
    with a live non-sibling cell to remove that stale daughter instead of
    protecting/restoring it forever.
  - Other active behavior remains the no-dark future-axis projection gate,
    dense drifting bridge protection, rod-tip fallback tuning, future-window
    support, gamma split decay, density gate, and soft overlap handling.
- Validation method:
  - Compared checkpoint non-trash cell counts against GT label counts from
    `images/Fluo-N3DH-CE/01_GT/TRA/man_track%03d.tif`.
  - Focused count checks were repeated through the old failure window and the
    current frontier. Frames `75-84` all matched GT counts.
- Known caveats:
  - Frame `17`: predicted `9`, GT `8`; frame `18`: predicted `11`, GT `10`.
    Frame `19` returned to pred `12`, GT `12`. This is the same early split
    timing family as the documented frame-18 caveat, but it starts one frame
    earlier in this run.
  - Frame `34`: predicted `16`, GT `15`; frame `35` returned to pred `24`,
    GT `24`, matching the known large split-burst timing caveat.
- Validated pass snapshot:
  - Frames `19-33` count-clean.
  - Frames `35-84` count-clean.
  - The old `run_057` frame-75 through frame-77 persistent +1 failure did not
    reproduce: frames `75-77` are all pred `51`, GT `51`.
  - Frames `80-84` are also clean:
    - frame `80`: pred `53`, GT `53`
    - frame `81`: pred `53`, GT `53`
    - frame `82`: pred `54`, GT `54`
    - frame `83`: pred `54`, GT `54`
    - frame `84`: pred `55`, GT `55`
- Causal caveat:
  - No `[CellUniverse2 Stale Daughter Cleanup]` log entry had fired by
    checkpoint `84`. The run fixes the old frame-75 count symptom in the
    observed evidence, but the cleanup path itself is not yet proven to be the
    cause. If later frames show this wiring has no positive effect or causes a
    regression, it should be reconsidered.
- Next known failure:
  - Frame `85`: pred `60`, GT `60`, but Hungarian assignment max distance was
    about `107.39`.
  - The bad assignment was `cell_110101 -> GT148`: predicted center
    approximately `(198.2,153.0,81.3)` versus GT center approximately
    `(191.0,205.0,175.0)`.
  - The local accepted split was `cell_11010`, with best label
    `bridge_axis_place`. Its PCA-bridge proposal for the same parent was
    rejected as `no_dark_future_axis_projection`, but the rod-tip fallback
    reused the same future-aligned center pair and accepted an axis-place
    alternate.
- Follow-up fix:
  - Added `pca_bridge_future_window_rod_tip_axis_place_max_parent_shape=2.20`.
  - High-shape future-aligned rod-tip proposals can still split through their
    primary future-center seed pair, but the extra `bridge_axis_place` alternate
    is skipped above that parent-shape threshold.
  - This intentionally preserves the earlier frame-18 moderate rod
    `bridge_axis_place` case while blocking the frame-85 high-shape false
    accept. Fresh validation is running in
    `run_059_from0_to200_axis_place_shape_gate`.

### 2026-06-13: `run_057_from0_to200_after_run010_followup`

- Run path:
  `/run/media/blue-lobster/disk3/celluniverse_output/outputs_fluo/celluniverse2_goal_0_200_20260612/run_057_from0_to200_after_run010_followup`
- Continuous frames processed: `0-78` inclusive (`79` checkpoints observed).
- Validated continuous pass span: `0-74` inclusive (`75` frames), with known
  one-frame GT timing caveats at frames `18` and `34`.
- Current status: stopped at the first persistent true mismatch, frame `75`.
- Run type: fresh start from frame `0`; no resume checkpoint used.
- Command shape:
  `CELLUNIVERSE_THREADS=4 CUDA_VISIBLE_DEVICES=0 ./build/celluniverse 0 200 ... config/config_celluniverse2.yaml config/embryo/initial_embryo_0_trash_labeled.csv`
- Runtime evidence:
  - CUDA preprocessing selected in `stdout.log`.
  - Checkpoints existed through `checkpoints/frame_078.txt` during the final
    inspection pass.
  - The active goal folder contains this current run and previous comparison
    `run_055_from0_to200_no_dark_future_axis_gate`.
- Code/config identity:
  - Same current-checkout CellUniverse2 family as `run_055`; no parameter or
    code changes were applied between the `run_055` validation and this
    follow-up fresh smoke.
  - Important active behavior remains the no-dark future-axis projection gate,
    dense drifting bridge protection, rod-tip fallback tuning, future-window
    support, gamma split decay, density gate, and soft overlap handling.
- Validation method:
  - Compared checkpoint non-trash cell counts against GT label counts from
    `images/Fluo-N3DH-CE/01_GT/TRA/man_track%03d.tif`.
  - Ran Hungarian assignment over predicted non-trash centers and GT
    bright-cube centers for count-matched frames, with GT z scaled into the
    runtime `0-238` coordinate range.
  - Inspected the local assignment neighborhood around `cell_3010`,
    `cell_3011`, and nearby GT labels `672`, `687`, `711`, `656`, and `643`
    across frames `72-77`.
- Known caveats:
  - Frame `18`: predicted `11`, GT `10`; frame `19` returned to pred `12`,
    GT `12`. This matches the known one-frame GT timing caveat.
  - Frame `34`: predicted `16`, GT `15`; frame `35` returned to pred `24`,
    GT `24`. This matches the known one-frame GT timing caveat around the
    large split burst.
- Validated pass snapshot:
  - Frames `0-17` count-clean.
  - Frames `19-33` count-clean.
  - Frames `35-74` count-clean.
  - Assignment-distance sweep through frame `61` found no count-matched frame
    with max assignment distance above `45`.
  - Later assignment spot checks:
    - frame `71`: pred `51`, GT `51`, max distance `44.12`; near-warning but
      still below the current inspection cutoff.
    - frame `72`: pred `51`, GT `51`, max distance `35.22`; the frame-71
      near-warning improved.
    - frame `74`: pred `51`, GT `51`, max distance `46.58`; warning focused
      on `cell_3011` vs GT `672`, but count still matched.
- Next known failure:
  - Frame `75`: pred `52`, GT `51`, and the +1 count mismatch persists through
    frames `76` and `77`.
  - The extra prediction is `cell_3011`. At frame `75`, all GT labels are
    already assigned cleanly; `cell_3011` is unassigned, nearest GT `672` at
    about `14.33`, while GT `672` is assigned to `cell_21010` at about `1.11`.
  - `cell_3011` remains unassigned at frames `76` and `77`, with nearest-GT
    distances about `17.15` and `14.78`.
  - Log evidence around frame `75` shows `cell_3011` had `background_body`
    rescue with `action=no_compatible_center`, then severe overlap with
    `cell_21010`; this looks like a stale/false daughter lifecycle problem,
    not a simple GT timing caveat.
- Debug conclusion:
  - This run independently reconfirms from a fresh frame-0 start that the
    current configuration preserves early behavior through the old June 9
    resume-run problem region and beyond.
  - The old stale-run frame-58 one-frame timing caveat is not present in this
    run; frame `58` is count-clean and assignment-clean.
  - `run_057` is now the strongest fresh evidence after the no-dark future-axis
    gate because it validates through frame `74`; the next fix should target
    the persistent frame-75 extra `cell_3011` without weakening the validated
    early splits.

### 2026-06-13: `run_055_from0_to200_no_dark_future_axis_gate`

- Run path:
  `/run/media/blue-lobster/disk3/celluniverse_output/outputs_fluo/celluniverse2_goal_0_200_20260612/run_055_from0_to200_no_dark_future_axis_gate`
- Continuous frames observed: `0-68` inclusive (`69` frames).
- Current status: interrupted/stopped after checkpoint `68`; the next fresh
  smoke in the same tuning family is
  `run_056_from0_to200_run010_followup`, which has not crossed the 50-frame
  record threshold yet.
- Run type: fresh start from frame `0`; no resume checkpoint used.
- Command shape:
  `CELLUNIVERSE_THREADS=4 CUDA_VISIBLE_DEVICES=0 ./build/celluniverse 0 200 ... config/config_celluniverse2.yaml config/embryo/initial_embryo_0_trash_labeled.csv`
- Runtime evidence:
  - CUDA preprocessing selected in `stdout.log`.
  - Checkpoints existed through `checkpoints/frame_068.txt` when this record
    was refreshed.
  - The active goal folder retained this comparison run plus the below-threshold
    follow-up run `run_056_from0_to200_run010_followup`.
- Code/config identity:
  - Same current-checkout CellUniverse2 family as `run_054`, plus the
    no-dark future-axis projection gate:
    - PCA bridge proposals now record whether the current frame had a dark
      bridge.
    - `pca_bridge_cut` proposals are rejected with
      `no_dark_future_axis_projection` when a no-dark raw PCA bridge is
      projected from a far reverted immediate-future snap.
  - The frame-18 dense drifting bridge protection and rod-tip flatness tuning
    from `run_054` remain active.
- Validation method:
  - Compared checkpoint non-trash cell counts against GT label counts from
    `images/Fluo-N3DH-CE/01_GT/TRA/man_track%03d.tif`.
  - Count scan through frame `68` matched GT for all frames except known
    one-frame timing caveats.
- Known caveats:
  - Frame `18`: predicted `11`, GT `10`; frame `19` returned to pred `12`,
    GT `12`. The frame-18 split behavior is consistent with the earlier note
    that GT can lag around this split.
  - Frame `34`: predicted `16`, GT `15`; frame `35` returned to pred `24`,
    GT `24`, matching the previously documented frame-34 timing caveat.
  - The earlier `run_053`/`run_054` frame-58 one-frame timing caveat is not
    present in this count scan: frame `58` is pred `30`, GT `30`.
  - Frame `59` is also count-clean: pred `37`, GT `37`.
  - Frames `60` and `61` are count-clean: frame `60` pred `43`, GT `43`;
    frame `61` pred `45`, GT `45`.
  - Frames `62-64` are count-clean: frame `62` pred `45`, GT `45`;
    frame `63` pred `45`, GT `45`; frame `64` pred `46`, GT `46`.
  - Frames `65-68` are count-clean: frame `65` pred `46`, GT `46`;
    frame `66` pred `47`, GT `47`; frame `67` pred `47`, GT `47`;
    frame `68` pred `48`, GT `48`.
- Debug conclusion:
  - The new gate did not break the early `0-68` behavior.
  - It correctly fired in frame `18` and frame `34` on no-dark projected PCA
    bridge proposals while allowing signal-center and rod-tip alternatives to
    preserve the expected split behavior.
  - This record is provisional until a fresh `0-200` smoke reaches the next
    true failure or completes.

### 2026-06-13: `run_054_from0_to200_current_midpoint_rescue`

- Run path:
  `/run/media/blue-lobster/disk3/celluniverse_output/outputs_fluo/celluniverse2_goal_0_200_20260612/run_054_from0_to200_current_midpoint_rescue`
- Continuous frames processed: `0-85` inclusive (`86` checkpoints observed).
- Validated continuous pass span: `0-84` inclusive (`85` frames), with the
  same GT timing caveats listed for `run_053`.
- Run type: fresh start from frame `0`; no resume checkpoint used.
- Command shape:
  `CELLUNIVERSE_THREADS=4 CUDA_VISIBLE_DEVICES=0 ./build/celluniverse 0 200 ... config/config_celluniverse2.yaml config/embryo/initial_embryo_0_trash_labeled.csv`
- Runtime evidence:
  - CUDA was requested for this run.
  - Checkpoints exist through `checkpoints/frame_085.txt`.
  - The run was stopped after the first true inspected failure, rather than
    treated as a completed `0-200` pass.
- Code/config identity:
  - Same current-checkout CellUniverse2 family as
    `run_053_from0_to65_current_midpoint_rescue`, plus the scoped frame-18
    corrections made after `run_053`:
    - rod-tip fallback flatness threshold relaxed from `1.75` to `1.70`
    - dense drifting `bridge_primary` rejection for weak no-dark future support
  - Important unchanged config values included:
    `pca_bridge_future_window_enabled=true`,
    `pca_bridge_future_window_size=3`,
    `pca_bridge_future_window_min_both_daughter_support=2`,
    `bio_min_daughter_separation_parent_fraction=1.10`,
    `bio_separation_soft_gate_enabled=true`, and
    `split_soft_geometry_gate_enabled=true`.
- Validation method:
  - Compared frame checkpoints against
    `images/Fluo-N3DH-CE/01_GT/TRA/man_track%03d.tif`.
  - Used Hungarian assignment over predicted non-trash cell centers and GT
    bright-cube centers, with z scaled to the runtime `0-238` coordinate
    range.
  - Frame `80-84` validation snapshot:
    - frame `80`: pred `53`, GT `53`, max distance `29.45`
    - frame `81`: pred `53`, GT `53`, max distance `27.02`
    - frame `82`: pred `54`, GT `54`, max distance `25.02`
    - frame `83`: pred `54`, GT `54`, max distance `23.73`
    - frame `84`: pred `55`, GT `55`, max distance `22.82`
- Known caveats:
  - Frame `34`: one-frame GT timing caveat for `cell_010`.
  - Frame `58`: one-frame GT timing caveat for `cell_0010`.
  - Frame `71`: near miss in assignment distance for `cell_00100`, but it
    improves at frame `72` and is clean by frame `73`; not treated as a true
    failure.
- Next known failure:
  - Frame `85`: pred `60`, GT `60`, but max assignment distance was about
    `107.37`.
  - Diagnosis: a false split of `cell_11010` placed both daughters near GT195,
    while the likely true split/placement around `cell_11101` / GT148 was not
    accepted. This is a placement/acceptance conflict, not a simple count
    mismatch.
- Debug conclusion:
  - This is the strongest fresh continuous evidence so far for the current
    tuning family: it extends the verified span from `0-65` to `0-84`.
  - It should be the immediate comparison run for the next frame-85 fix.

### 2026-06-13: `run_053_from0_to65_current_midpoint_rescue`

- Run path:
  `/run/media/blue-lobster/disk3/celluniverse_output/outputs_fluo/celluniverse2_goal_0_200_20260612/run_053_from0_to65_current_midpoint_rescue`
- Continuous frames: `0-65` inclusive (`66` frames).
- Run type: fresh start from frame `0`; no resume checkpoint used.
- Command shape:
  `CELLUNIVERSE_THREADS=4 CUDA_VISIBLE_DEVICES=0 ./build/celluniverse 0 65 ... config/config_celluniverse2.yaml config/embryo/initial_embryo_0_trash_labeled.csv`
- Runtime evidence:
  - CUDA preprocessing selected in `stdout.log`.
  - Completed normally with `Time elapsed: 1697.15 seconds`.
  - Checkpoints exist through `checkpoints/frame_065.txt`.
- Code/config identity:
  - Git base commit at validation time: `cd17501`.
  - Worktree had local CellUniverse2 tuning edits in
    `config/config_celluniverse2.yaml`, `includes/ConfigTypes.hpp`,
    `includes/Frame.hpp`, `src/CellUniverse.cpp`, and `src/Frame.cpp`.
  - Important active config values included:
    `pca_bridge_min_long_mid_ratio=1.02`,
    `pca_bridge_max_mid_short_ratio=2.45`,
    `pca_bridge_future_window_enabled=true`,
    `pca_bridge_future_window_size=3`,
    `pca_bridge_future_window_min_both_daughter_support=2`,
    `bio_min_daughter_separation_parent_fraction=1.10`,
    `bio_min_daughter_mean_brightness_absolute=0.05`,
    `bio_min_daughter_mean_brightness_parent_fraction=0.50`,
    `bio_separation_soft_gate_enabled=true`,
    `split_soft_geometry_gate_enabled=true`, and
    `export_perturb_cell_center_debug_images=true`.
- Validation method:
  - Compared every checkpoint frame `0-65` against
    `images/Fluo-N3DH-CE/01_GT/TRA/man_track%03d.tif`.
  - Used Hungarian assignment over predicted non-trash cell centers and GT
    bright-cube centers, with z scaled to the runtime `0-238` coordinate
    range.
  - Final validator result:
    `OK_WITH_CAVEATS through 65 frames 66`.
  - Worst non-caveat assignment summary:
    frame `65`, predicted `46`, GT `46`, max distance `31.87`, p95 `9.85`.
- Known caveats:
  - Frame `34`: predicted `16`, GT `15`, max distance `18.33`. This matches
    the known one-frame GT timing caveat for `cell_010`; at frame `35`,
    `cell_0100` and `cell_0101` resolve to separate GT centers at about
    `1.33` and `3.77` px.
  - Frame `58`: predicted `31`, GT `30`, max distance `14.81`. This matches
    the known one-frame GT timing caveat for `cell_0010`; at frame `59`,
    `cell_00100` and `cell_00101` resolve to separate GT centers at about
    `4.40` and `1.24` px.
- Debug conclusion:
  - The current checkout fixes the older June 9 resume-run frame-59 miss while
    preserving the earlier frame-18 and frame-23 behavior.
  - This record proves the current configuration can pass more than 50
    continuous frames from a fresh start, but it does not yet prove the final
    goal of `0-200`.

### 2026-06-09: `run_010_from0_0_50`

- Run path:
  `/run/media/blue-lobster/disk3/celluniverse_output/outputs_fluo/celluniverse2_smoke_0_50_goal_20260610/run_010_from0_0_50`
- Continuous frames: `0-50` inclusive (`51` frames).
- Run type: fresh start from frame `0`.
- Initial CSV: `config/embryo/initial_embryo_0_trash_labeled.csv`.
- Config family: `config/config_celluniverse2.yaml`.
- Evidence kept: checkpoints through `frame_050.txt`.
- Downstream use: source checkpoint for
  `output_ubuntu_fluo_resume51_51-200_celluniverse2_run010_20260609_214650`.
- Known caveats:
  - This run predates several later CellUniverse2 split/rescue changes.
  - Treat it as the baseline that proved the early `0-50` region, not as proof
    that the current checkout still passes `0-50`.
- Next known failure after resume:
  - Resume run from frame 51 stayed good through frame 57.
  - Frame 58 had a one-frame GT timing caveat for `cell_0010`.
  - Frame 59 exposed a real missed split around `cell_0111` / GT338 in the
    older June 9 code/config state.

## Resume Comparisons

### 2026-06-13: `run_049_resume36_to120_after_frame34_gt_timing_note`

- Run path:
  `/run/media/blue-lobster/disk3/celluniverse_output/outputs_fluo/celluniverse2_goal_0_200_20260612/run_049_resume36_to120_after_frame34_gt_timing_note`
- Continuous frames: resumed from frame `36`, not a fresh `0-to-n` proof.
- Status:
  - This run is kept as the previous comparison run for the current goal
    folder.
  - It helped show that the current checkout no longer reproduced the older
    June 9 frame-59 miss, but it is not promoted to a confirmed record because
    it depends on a prior checkpoint.
- Superseding fresh evidence:
  - `run_053_from0_to65_current_midpoint_rescue` now confirms the current
    checkout from frame `0` through frame `65`.

## Historical Fresh-Run Index

This section records older fresh `0-to-n` CellUniverse2 runs that produced more
than 50 continuous checkpoints. Most of these predate the current frame-18,
frame-34, frame-58, frame-70, and frame-85 tuning work. Use them as historical
configuration evidence only; do not treat them as proof of the current checkout.

| Date | Run | Checkpoint span | Evidence level | Notes |
| --- | --- | ---: | --- | --- |
| 2026-06-09 | `debug_bio_soft_0_60_20260609` | `0-60` | checkpoint-only | Older bio-soft debug run; consecutive checkpoints through frame `60`. |
| 2026-06-09 | `output_ubuntu_fluo_0-200_trash_labeled_20260609_114905` | `0-60` | checkpoint-only | Older trash-labeled fluo output; consecutive checkpoints through frame `60`. |
| 2026-06-10 | `celluniverse2_full_0_200_goal_20260610/run_001_from0_0_200` | `0-62` | checkpoint + debug-log tail | Last observed checkpoint frame `62`; debug tail shows frame `62` optimized and checkpoint saved. No completion line found. |
| 2026-06-10 | `celluniverse2_smoke_0_50_goal_20260610/run_010_from0_0_50` | `0-50` | checkpoint + later resume evidence | Baseline early-window proof already promoted above. |
| 2026-06-10 | `celluniverse2_smoke_0_200_goal_20260610/run_011_from0_0_50` | `0-50` | checkpoint-only | Duplicate early-window smoke family; config identity not separately validated. |
| 2026-06-10 | `celluniverse2_smoke_0_200_goal_20260610/run_012_from0_0_50` | `0-50` | checkpoint-only | Duplicate early-window smoke family; config identity not separately validated. |
| 2026-06-10 | `celluniverse2_smoke_0_80_goal_20260610/run_001_from0_0_80` | `0-64` | checkpoint + debug-log tail | Last observed checkpoint frame `64`; no completion line found. |
| 2026-06-10 | `celluniverse2_smoke_0_80_goal_20260610/run_002_from0_0_66` | `0-61` | checkpoint-only | Intermediate tuning run. |
| 2026-06-10 | `celluniverse2_smoke_0_80_goal_20260610/run_003_from0_0_66` | `0-61` | checkpoint-only | Intermediate tuning run. |
| 2026-06-10 | `celluniverse2_smoke_0_80_goal_20260610/run_004_from0_0_66` | `0-60` | checkpoint-only | Intermediate tuning run. |
| 2026-06-10 | `celluniverse2_smoke_0_80_goal_20260610/run_005_from0_0_61` | `0-61` | checkpoint-only | Intermediate tuning run. |
| 2026-06-10 | `celluniverse2_smoke_0_80_goal_20260610/run_006_from0_0_66` | `0-62` | checkpoint-only | Intermediate tuning run. |
| 2026-06-10 | `celluniverse2_smoke_0_80_goal_20260610/run_007_from0_0_66_midpoint_future` | `0-66` | completed debug log | Completed with `Time elapsed: 1897.01 seconds`; midpoint/future-support tuning family. |
| 2026-06-10 | `celluniverse2_smoke_0_80_goal_20260610/run_008_from0_0_66_pca_snap_future` | `0-66` | checkpoint-only | PCA-snap/future-support tuning family. |
| 2026-06-10 | `celluniverse2_smoke_0_80_goal_20260610/run_009_from0_0_66_future_geometry_rescue` | `0-62` | checkpoint-only | Future-geometry rescue tuning family. |
| 2026-06-10 | `celluniverse2_smoke_0_80_goal_20260610/run_010_from0_0_66_geometry_positive_cost` | `0-66` | checkpoint-only | Geometry positive-cost tuning family. |
| 2026-06-10 | `celluniverse2_smoke_0_80_goal_20260610/run_011_from0_0_80_geometry_positive_cost` | `0-80` | completed debug log | Completed with `Time elapsed: 2849.49 seconds`; final cells at frame `80`: `53`. |
| 2026-06-10 | `celluniverse2_smoke_0_80_goal_20260610/run_012_from0_0_80_signal_future_overlap` | `0-74` | checkpoint + debug-log tail | Last observed checkpoint frame `74`; no completion line found. |
| 2026-06-10 | `celluniverse2_smoke_0_80_goal_20260610/run_015_from0_0_80_overlap_tolerance_helper` | `0-80` | completed debug log | Completed with `Time elapsed: 2785.26 seconds`; overlap-tolerance helper tuning family. |
| 2026-06-11 | `celluniverse2_goal_0_200_20260611/run_003_0_to_200_dense_flat_gate` | `0-59` | detailed checkpoint-only | Promoted above as a qualifying checkpoint-span record; later superseded by current 2026-06-13 validator-backed runs. |
| 2026-06-13 | `celluniverse2_goal_0_200_20260612/run_053_from0_to65_current_midpoint_rescue` | `0-65` | completed + validator-backed | Promoted above as a confirmed record. |
| 2026-06-13 | `celluniverse2_goal_0_200_20260612/run_054_from0_to200_current_midpoint_rescue` | `0-85` processed, `0-84` validated pass | validator-backed to next true failure | Promoted above as the latest confirmed record. |
| 2026-06-13 | `celluniverse2_goal_0_200_20260612/run_055_from0_to200_no_dark_future_axis_gate` | `0-68` observed | count-validated stopped run | Promoted above as the latest stopped comparison record. |
| 2026-06-13 | `celluniverse2_goal_0_200_20260612/run_057_from0_to200_after_run010_followup` | `0-78` processed, `0-74` validated pass | validator-backed to next true failure | Promoted above; first persistent mismatch at frame `75`. |
| 2026-06-13 | `celluniverse2_goal_0_200_20260612/run_058_from0_to200_stale_daughter_cleanup` | `0-85` processed, `0-84` validated pass | validator-backed to next true failure | Promoted above; fixed the old frame-75 count symptom, then failed hidden assignment at frame `85`. |
| 2026-06-13 | `celluniverse2_goal_0_200_20260612/run_059_from0_to200_axis_place_shape_gate` | `0-85` processed, `0-84` validated pass | validator-backed to next true failure | Promoted above as current strongest fresh record; axis-place cutoff preserved prior good windows, then failed hidden assignment at frame `85`. |
| 2026-06-13 | `celluniverse2_goal_0_200_20260612/run_061_resume85_to90_match30` | resume `85-87` inspected | focused resume, not qualifying | Widening future-window match distance to `30.0` did not recover GT148 at frame `85`; setting reverted to `28.0`. |
| 2026-06-13 | `celluniverse2_goal_0_200_20260612/run_062_resume85_to90_delayed_future_cost` | resume `85-87` inspected | focused resume, not qualifying | Delayed future-window cost rescue fixed frame `86` GT148, but frame `87` remained short by GT `171` and `388`; stopped after next error. |
| 2026-06-13 | `celluniverse2_goal_0_200_20260612/run_063_resume85_to90_no_dark_future_rescue` | resume `85-87` inspected | focused resume, not qualifying | No-dark future-axis rescue fixed the `run_062` frame-87 GT `171`/`388` misses, but frame `87` still missed GT `84`; stopped at next error. |

Historical index interpretation:

- "Checkpoint-only" means the folder contains consecutive frame checkpoints,
  but the exact command line and validation result were not recovered in this
  pass.
- "Completed debug log" means the run's `debug_log.txt` includes a normal
  `Time elapsed` completion line for its requested end frame.
- "Validator-backed" means center-to-GT assignment was explicitly checked and
  caveats were inspected.
- If disk cleanup is required, preserve at least the latest validator-backed
  run and the immediately previous comparison run before deleting older
  checkpoint-only folders.

## 2026-06-16 run-window evidence

| Date | Run | Checkpoint span | Evidence level | Notes |
| --- | --- | ---: | --- | --- |
| 2026-06-16 | `celluniverse2_goal_0_200_20260612/run_from0_22_tuning_20260616_081617` | `0-22` | completed + validator-backed | Fresh CUDA four-core guardrail after separating lock-only split brightness floors from later acceptance floors. Validator checks passed for frames `4`, `6`, `7`, `8`, `9`, `11`, `16`, `17`, `18`, `19`, `20`, `21`, and `22`. |
| 2026-06-16 | `celluniverse2_goal_0_200_20260612/run_resume85_92_tuning_20260616_081619` | `85-86` observed | focused resume, not qualifying | Resume evidence only. Folder recovered checkpoints through frame `86`, not `92`. Reconfirmed the hard frontier around GT `148` plus additional crowded misses at frame `86`, so it should be treated as comparison evidence, not as a promoted run. |
| 2026-06-16 | `output_ubuntu_fluo_resume85_85-90_celluniverse2_tuning_20260616_121703` | `85-87` observed | focused resume, not qualifying | Added `split_density_cost_backed_clean_future_min_future_brightness=0.08`. This removed the frame-85 phantom daughter from `cell_11110` and restored a hard density reject there, but the window still failed by frame `86-87` because crowded signal-center pairs were mostly discarded as neighbor-owned, leaving `cell_11111` and related families mislocalized. |

2026-06-16 configuration note:

- `config/config_celluniverse2_tuning.yaml` now separates split-daughter
  position-lock brightness floors from later acceptance/rescue brightness
  floors.
- New lock-only YAML keys:
  - `split_current_locked_bridge_lock_min_brightness`
  - `split_immediate_pca_continuation_lock_min_brightness`
  - `split_one_frame_aligned_pca_continuation_lock_min_brightness`
  - `split_exact_future_center_bridge_lock_min_brightness`
- New crowded density-rescue quality key:
  - `split_density_cost_backed_clean_future_min_future_brightness`
- The fresh `0-22` guardrail run shows this split is compatible with the
  validated early window. The later `85-86` resume shows the remaining frontier
  is not solved by brightness-floor tuning alone.
