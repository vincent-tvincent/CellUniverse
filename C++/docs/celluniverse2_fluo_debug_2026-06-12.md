# CellUniverse2 Fluo Debug Log - 2026-06-12

## Goal

Make the CellUniverse2 fluo pipeline pass frames 0-200 without missing splits or
misplacing bridge-cut daughters. Smoke validation should prefer 0-to-n runs so
early logic is not accidentally broken. Short checkpoint windows may still be
used to inspect a localized mechanism, but they are not sufficient proof of the
full goal.

## Run Retention Rule

- Active debug folder should keep the current run and the immediately previous
  comparison run.
- Do not delete based only on size.
- It is acceptable to delete exported images to save volume.
- Preserve useful checkpoints that are needed for resume, partial diagnosis, or
  comparison.

## Referenced Run Inspection

Run:

`/run/media/blue-lobster/disk3/celluniverse_output/outputs_fluo/output_ubuntu_fluo_resume51_51-200_celluniverse2_run010_20260609_214650`

Observed checkpoints: frames 51-64 only.

GT count comparison:

| Frame | Pred real | GT | Delta | Note |
| --- | ---: | ---: | ---: | --- |
| 51 | 26 | 26 | 0 | OK |
| 52 | 28 | 28 | 0 | OK |
| 53 | 28 | 28 | 0 | OK |
| 54 | 28 | 28 | 0 | OK |
| 55 | 28 | 28 | 0 | OK |
| 56 | 28 | 28 | 0 | OK |
| 57 | 28 | 28 | 0 | OK |
| 58 | 31 | 30 | +1 | extra predicted `cell_00100`; likely early/false split |
| 59 | 37 | 37 | 0 | Count recovers |
| 60 | 40 | 43 | -3 | missing GT labels 114, 338, 354 |
| 61 | 43 | 45 | -2 | labels 114 and 338 recovered; 354/369 region remains wrong |
| 62 | 44 | 45 | -1 | missing GT 354 |
| 63 | 44 | 45 | -1 | missing GT 354; `cell_01100` badly misplaced |
| 64 | 45 | 46 | -1 | missing GT 354; `cell_01100`/`cell_3001` neighborhood confused |

Targeted mapping notes:

- Frame 60 misses GT labels 114, 338, 354. Labels 114 and 338 appear by frame
  61 as `cell_10111` and `cell_01101`.
- The durable problem is GT 354 and the nearby 369/656 neighborhood.
- Frame 62: `cell_01100` maps to GT 369 at distance about 30 px, while GT 354
  is unmatched.
- Frame 63: `cell_01100` maps to GT 369 at distance about 85 px, while GT 354
  is unmatched.
- Frame 64: `cell_01100` maps to GT 656, `cell_3001` maps badly to GT 369, and
  GT 354 remains unmatched.
- Logs around frame 64 show large post-PCA/overlap correction in this
  neighborhood:
  - `cell_3001` and `cell_01100` overlap heavily after PCA.
  - `cell_01100` ends as a bloated long cell near `(504,317,61)`.
  - `cell_01101` grows long near `(439,404,98)`.

Interpretation:

The referenced run's durable issue is not simply "one bridge did not cut." It
looks like a local assignment/placement collapse around `cell_01100`,
`cell_01101`, and `cell_3001`, after earlier splits partially recover counts.
Tuning should preserve the frame-51-to-59 good behavior while preventing the
frame-60-to-64 `cell_01100`/GT-354 loss.

## Current Patch Evidence From Previous Iteration

Latest focused current-state run:

`/run/media/blue-lobster/disk3/celluniverse_output/outputs_fluo/celluniverse2_goal_0_200_20260611/run_016_resume70_to71_current_bridge_overlap`

Evidence:

- Frame 70 count is fixed in this focused run: pred real 50, GT 50.
- `cell_2011` post-PCA current bridge is accepted at frame 70:
  - clean current-center snap,
  - strong bridge (`gapDensity` about 0.018, `valleyFromBright` about 0.28),
  - final accepted cost improvement about -51k.
- Frame 71 is one short in this focused run, missing GT label 602, likely
  `cell_2000`.
- Do not tune from that frame-71 miss yet because the run ends at 71, so future
  frame support is unavailable. A 0-to-n smoke with frame 72+ available is the
  correct test for that case.

## Next Debug Strategy

1. Build current code.
2. Run a CUDA 0-to-n smoke, choosing n past the known failure region. Initial
   recommended target: 0-72 or 0-80, because it covers:
   - known early frame-18/23/34/60 style split cases,
   - referenced run's 58-64 failure region,
   - current frame-70/71 post-PCA bridge behavior with future frames available.
3. Compare all checkpoint counts to GT.
4. For first mismatch, map predicted centers to GT labels and inspect the
   matching frame log before changing code.
5. Keep the latest 0-to-n run plus the immediately previous comparison run in
   the active debug folder; keep useful checkpoints even when deleting images.

## 2026-06-13 Continuation: Frame 85 No-Dark Future Projection Gate

Fresh current-checkout run:

`/run/media/blue-lobster/disk3/celluniverse_output/outputs_fluo/celluniverse2_goal_0_200_20260612/run_054_from0_to200_current_midpoint_rescue`

Validated behavior before the failure:

- Frames `0-84` were clean except the already documented GT timing caveats at
  frames `34` and `58`.
- Frame `71` had a near assignment miss for `cell_00100`, but it improved at
  frame `72` and was clean by frame `73`, so it was not treated as a true
  failure.

First true failure:

- Frame `85`: pred `60`, GT `60`, but one assignment was about `107` px wrong.
- `cell_11010` split into `cell_110100` and `cell_110101`; both daughters
  landed in the same GT195 neighborhood.
- GT148 was left uncovered; the likely competing true split/placement was
  around `cell_11101`.

False-accept evidence for `cell_11010`:

- Raw PCA bridge had no current dark bridge:
  `darkBridge=0`, `gapBins=10-10`.
- Immediate future center snap was far and was reverted:
  `maxSeedDistance=46.6697`, `trustLimit=28.7712`.
- The code then projected the far future-pair axis back onto the current frame:
  `rawSep=12.2077`, `futureSep=48.9833`,
  `projectedSep=31.8132`, `minBioSep=30.2983`.
- Final bridge evidence looked numerically clean after refit
  (`gapDensity=0.077`, `valleyFromBright=0.243`), so ordinary valley gates
  could not catch this false accept.

Conflict check:

- The only other accepted no-dark future-axis-projected PCA bridge in this
  fresh run was frame `58` / `cell_0010`, which was already a known one-frame
  GT timing caveat and passed on ordinary current-frame cost.
- Therefore the fix should not ban all no-dark bridge behavior. It should ban
  the more specific pattern: no current dark bridge plus far future snap
  reversion plus future-axis projection back into the current frame.

Fix:

- Added PCA bridge proposal metadata:
  `pcaBridgeHasDarkBridge`, `pcaBridgeFutureAxisProjected`,
  `pcaBridgeRevertedFarFutureSnap`, and the reverted snap distance/trust limit.
- `Frame::discoverPcaBridgeProposal` now records whether the current frame had
  a dark bridge.
- The CellUniverse2 proposal validator now rejects
  `pca_bridge_cut` proposals with reason
  `no_dark_future_axis_projection` when a no-dark raw PCA bridge is projected
  from a far reverted future snap.

Fresh validation run launched after this fix:

`/run/media/blue-lobster/disk3/celluniverse_output/outputs_fluo/celluniverse2_goal_0_200_20260612/run_055_from0_to200_no_dark_future_axis_gate`

Run settings:

- Fresh start: frame `0` to `200`.
- CUDA requested via `CUDA_VISIBLE_DEVICES=0`.
- Threads: `CELLUNIVERSE_THREADS=4`.
- Previous comparison run kept in the active goal folder: `run_054`.

Interim validation while `run_055` was still running:

- Checkpoints reached frame `66`, giving a fresh `0-66` continuous span.
- Count validation against GT labels was clean through frame `66` except:
  - frame `18`: pred `11`, GT `10`, resolved at frame `19`
  - frame `34`: pred `16`, GT `15`, resolved at frame `35`
- Unlike `run_053` and `run_054`, frame `58` is count-clean in this run:
  pred `30`, GT `30`.
- Frame `59` is also count-clean: pred `37`, GT `37`.
- Frames `60` and `61` are count-clean: frame `60` pred `43`, GT `43`;
  frame `61` pred `45`, GT `45`.
- Frames `62-64` are count-clean: frame `62` pred `45`, GT `45`;
  frame `63` pred `45`, GT `45`; frame `64` pred `46`, GT `46`.
- Frames `65` and `66` are count-clean: frame `65` pred `46`, GT `46`;
  frame `66` pred `47`, GT `47`.
- The new `no_dark_future_axis_projection` rejection fired in the early run,
  including frame `18` and frame `34`, but signal-center and rod-tip proposals
  still preserved the known split behavior.
- Because the run may continue toward the real `0-200` goal, do not stop it at
  frame `66`; continue to the next true error or completion.

Follow-up after the goal was reset to inspect the June 9 resume51 run first:

- The active `run_055` process was stopped manually after checkpoint `66` so the
  run would not keep consuming disk/GPU while the old run was re-inspected.
- Re-inspection of
  `output_ubuntu_fluo_resume51_51-200_celluniverse2_run010_20260609_214650`
  confirmed the old first failure family:
  - frame `59`: count matched GT, but assignment was already bad; the plausible
    `cell_0111` split had a strong bridge (`valleyFromBright=0.245953`) and was
    rejected only by `daughter_midpoint_parent_drift`, `31.191` versus the
    `29.994` limit.
  - frame `60`: GT labels `114`, `338`, and `354` were missing.
  - frames `62-64`: the durable missing cell was GT `354`, with `cell_01100`,
    `cell_01101`, and `cell_3001` drifting into a confused local neighborhood.
  - frame `63`: `cell_01100` proposed the two relevant centers, but the
    signal-center split was rejected as `d1_buried_in_cell_3001`; the PCA-bridge
    retry then had clean bridge evidence but was rejected by cost after a large
    split soft penalty.
- The current checkout already addresses this old failure family in a fresh
  `0-to-n` run: `run_055` is count-clean through frame `66`, including frames
  `59-64`. No new parameter or logic change is justified from this old evidence
  until a fresh current-checkout run reaches the next true mismatch.
- Next action: start a new fresh `0-200` CUDA smoke from frame `0`, keep
  `run_055` as the immediate previous comparison, and remove older active-goal
  runs from the debug folder according to the retention rule.

Fresh follow-up smoke:

## 2026-06-16 Continuation: Dense-Window Rod-Tip Experience

Fresh early-to-mid scout:

- `/run/media/blue-lobster/disk3/celluniverse_output/outputs_fluo/output_ubuntu_fluo_fresh_0-90_celluniverse2_tuning_20260616_093027`

Dense later-window scout:

- `/run/media/blue-lobster/disk3/celluniverse_output/outputs_fluo/output_ubuntu_fluo_resume120_120-160_celluniverse2_tuning_20260616_094539`

What is already validated:

- The tuning YAML with
  `split_daughter_refit_halfspace_min_fraction: 0.18` stayed clean in a fresh
  continuous validation span through frame `50`.
- Validated clean frames in this fresh scout currently include:
  `0-50`, with spot checks at `34`, `35`, `36`, `37`, `38`, `39`, `40`,
  `41`, `42`, `43`, `44`, `45`, `46`, `47`, `48`, `49`, and `50` all matching GT with no missing or extra
  predicted centers under the project `matched<=35` rule.
- Additional fresh validations now confirmed clean:
  - `61`
  - `68`
  - `69`
  - `80`
- This is stronger evidence than the earlier `0-22` guardrail only: the same
  clamp survived the first split-heavy region and another bridge-active region
  without reopening the old early failures.

Fresh-run mechanism notes from frames `37-45`:

- Weak-evidence PCA bridge retries are still being rejected in this branch when
  future support is absent or brightness support collapses. Example:
  frame `38`, `cell_301` was rejected by
  `pca_bridge_low_brightness_bio_near_miss_gate` with `futureBoth=0`.
- Valid bridge behavior is still preserved when the proposal has stable future
  support and only misses ordinary cost by a small amount. Example:
  frame `45`, `cell_201` was accepted by
  `stableFutureNearThresholdBridgeRescue=1` after a negative geometry-adjusted
  cost and clean two-frame future support.
- Practical interpretation: the current branch is behaving better when cost
  rescue is tied to future support and modest seed drift rather than a global
  relaxation of bridge or overlap thresholds.

Dense later-window experience already visible at frame `120`:

- Checkpoint `119` loads with `accepted_splits_total=116`, which drives gamma
  down to `1.015` under the split-decay schedule. This is direct evidence that
  gamma is already functioning as an environment-linked control rather than a
  fixed constant.
- The later crowded scene produces many future-aligned rod-tip candidates with
  strong future brightness support, but the difficult failures are not all the
  same:
  - some are still correctly rejected by
    `future_aligned_rod_tip_flatness_gate`
  - some survive the crowded flatness relaxation but are then blocked by
    `rod_tip_third_cell_future_claim_gate`
  - clean signal-center proposals with strong future support can still pass in
    the same frame
- Example crowded rejects:
  - `cell_001111`: crowded future-aligned flatness relaxation fired, but the
    proposal was still rejected by `rod_tip_third_cell_future_claim_gate`
    because another cell claimed the same future territory more plausibly
  - `cell_010000`, `cell_000100`, `cell_000000`, `cell_000001`, and
    `cell_0001100`: rejected by `future_aligned_rod_tip_flatness_gate` even
    with aligned future pairs
- Example crowded accepts:
  - `cell_110101` and `cell_000101` were accepted by signal-center splitting
    with clean future support
  - `cell_100111` was accepted by rod-tip fallback after crowded
    future-aligned relaxation plus strong future support

Current cross-window interpretation:

- `split_daughter_refit_halfspace_min_fraction` now looks globally safer at
  `0.18` than at `0.15`; it has fresh support through frame `45`, not just the
  original sparse frames.
- The later crowded-window evidence does not argue for globally loosening the
  rod-tip family. It argues for keeping the hard/global protections that stop
  third-cell theft and clearly bad future claims, while making any further
  flatness/balance relaxation explicitly conditional on crowded-scene evidence.
- The likely environment-linked inputs for future adaptive functions are:
  - local crowding / cell count
  - future daughter support count
  - future minimum daughter brightness
  - future aligned-pair snap distance
  - third-cell claim margin
- The later frame `120` logs strengthen the case that future-supported
  flatness relaxations should be scoped by crowding and claim cleanliness,
  rather than turned into a broad threshold drop.

Updated later-window distinction after the first dense rerun:

- Some frame-`120` misses are effectively one-frame late, not persistent:
  - `cell_100011`
  - `cell_111110`
- In frame `121`, both of those regain strong signal-center proposals with
  clean future support, so they should not drive a broad retune by themselves.
- The persistent crowded-window misses are different:
  - `cell_001011`
  - `cell_101000`
  - `cell_0001101`
- Those cells still fail at frame `121` through the future-aligned rod-tip
  path, typically via `future_aligned_rod_tip_flatness_gate` or crowded claim
  interaction, even though they already have immediate-future aligned pairs.
- That is the best current evidence that the next useful adaptive function is a
  dense-scene-only rod-tip flatness relaxation rather than another global split
  threshold drop.

Current follow-up applied to the tuning YAML:

- Added crowded-scene-only rod-tip relaxation keys:
  - `rod_tip_future_aligned_crowded_cell_count_min: 100`
  - `rod_tip_future_aligned_crowded_relaxed_min_shape: 1.45`
  - `rod_tip_future_aligned_crowded_relaxed_min_mid_short_ratio: 1.26`
  - `rod_tip_future_aligned_crowded_relaxed_mid_short_floor: 1.02`
  - `rod_tip_future_aligned_crowded_relaxed_mid_short_shape_span: 0.70`
  - `rod_tip_future_aligned_crowded_relaxed_max_snap_radius_scale: 2.50`
- Why this is scoped safely:
  - it requires crowded scenes (`cell_count >= 100`);
  - it only applies when the rod-tip proposal already has an immediate-future
    aligned pair;
  - the fresh `0-80` scout never enters that regime, so the early/mid clean
    frontier should remain unchanged while the dense `120+` family gets a more
    appropriate shape-dependent threshold.

Result from the immediate rerun with those dense-scene knobs:

- New rerun:
  `/run/media/blue-lobster/disk3/celluniverse_output/outputs_fluo/output_ubuntu_fluo_resume120_120-160_celluniverse2_tuning_20260616_100647`
- Frame `120` count outcome did not materially change yet; it is still
  `pred=122 gt=127`.
- But frame `121` debug evidence shows the change is real and directionally
  correct:
  - `cell_001011` now logs
    `Rod Tip Split Fallback Relaxation ... midShortRatio=1.15564 midShortThreshold=1.14054`
    instead of dying at `future_aligned_rod_tip_flatness_gate`
  - `cell_101000` now logs
    `Rod Tip Split Fallback Relaxation ... midShortRatio=1.17967 midShortThreshold=1.17369`
    instead of dying at `future_aligned_rod_tip_flatness_gate`
  - `cell_0001101` now also survives the flatness gate on frame `121`
- The remaining blocker for this trio is no longer flatness; it is now
  `rod_tip_third_cell_future_claim_gate`.
- Practical takeaway:
  - the dense-scene flatness relaxation is globally helpful because it moves
    real crowded candidates farther through the pipeline without touching the
    early clean region;
  - the next unresolved issue is center ownership / pair selection inside
    crowded windows, not the flatness threshold itself.

## 2026-06-16 Continuation: Rebaseline Through the Old Frame-85 Frontier

Fresh current-checkout scouts were used to separate transient mid-window
assignment wobble from the next real persistent failure:

- `run_345_from0_to90_current_checkout_rebaseline`
- `run_346_resume63_to140_from_run345`
- `run_347_resume67_to140_threshold35`

What changed in the validator policy:

- A hard `matched<=30` scout cutoff was too strict for this checkout. The
  existing long-run records already accept frame `65` as
  `pred=46`, `GT=46`, max assignment about `31.87`, so resume scouting was
  shifted to `matched<=35` for forward frontier checks.

Observed behavior:

- `run_345` validated clean through frame `58` with the old `<=30` watcher, but
  frame `59` and frame `60` each produced a transient `+1` extra:
  - frame `59`: extra `cell_10011`
  - frame `60`: extra `cell_2111`
- Those extras resolved by frame `61`, and frame `62` also validated clean.
- `run_346` resumed from clean frame `62` and showed another transient
  one-for-one assignment wobble at frame `65`, but frame `66` validated clean.
- `run_347`, resumed from clean frame `66` under the documented-threshold
  policy, validated clean through frame `84`:
  - frames `67`, `68`, `69`, `70`, `71`, `72`, `73`, `74`, `75`, `76`, `77`,
    `78`, `79`, `80`, `81`, `82`, `83`, and `84` all had no unmatched GT or
    predicted centers under `matched<=35`.
- The next real persistent failure remained at frame `85`:
  - `pred=59`, `GT=60`, missing GT label `148`
  - nearest predicted center stayed `cell_11110` at about `38.41` px

Current interpretation:

- The real frontier did not move earlier than the historical frame-`85`
  failure. The mid-window `59/60/65/67` issues were transient and self-healed.
- These self-healing windows should be treated as experience about validation
  policy and local ambiguity, not as immediate justification for new split
  logic.
- Gamma decay alone does not explain the transient behavior: the scout showed
  both transient wobble and later clean splits at lower effective gamma.
- The environment-dependent logic hypothesis still looks narrow:
  adaptive crowded-scene rod/flatness behavior can be useful, but third-cell
  claim and parent-balance protection still look like global guards.

Two later-window CUDA scouts were used to separate globally useful rod-tip
tuning from early-window overfitting:

- `run_331_resume121_to130_crowded_rod_tip_relax`
- `run_332_resume121_to123_crowded_rod_tip_shape_decay`

The key correction from re-reading the logs is that the old frame-85 one-frame
extra was not produced by the ordinary validated rod-tip fallback path alone.
The frame-85 proposal for `cell_11010` was marked acceptable, but the output
change that actually created `cell_110100` / `cell_110101` happened later in
the separate `post_pca_force_rod_split` stage. That means "future-supported
rod-tip accepted" and "forced post-PCA rod split accepted" must be treated as
different failure families.

Dense frame-121 observations:

- The later crowded window is dominated by three rod-tip rejection families:
  - `future_aligned_rod_tip_flatness_gate`
  - `rod_tip_third_cell_future_claim_gate`
  - `parent_distance_balance_gate`
- A fixed crowded-scene relaxation was too weak. It still left examples such as
  `cell_3110` and `cell_111100` behind the flatness gate.
- A shape-decaying crowded-scene flatness threshold worked better:
  - `cell_3110` advanced from `future_aligned_rod_tip_flatness_gate` to the
    next blocker, `rod_tip_third_cell_future_claim_gate`
  - `cell_111100` also advanced from the flatness gate to the third-cell gate
  - `cell_201010` still correctly stayed behind the flatness gate because its
    crowded future-aligned pair was both far-snapped and only weakly rod-like
    (`midShortRatio` about `1.22`)
- `cell_301000` was already a healthy dense-scene rod-tip acceptance under the
  current logic, so the crowded relaxation should not replace third-cell or
  balance gates; it should only stop the flatness gate from being the first
  failure for strong crowded rods.

Current interpretation:

- Some thresholds should stay global, especially third-cell claim and low
  parent-balance protection.
- The crowded future-aligned rod-tip flatness rule is not well modeled by a
  single constant. Later dense windows behave better when the required
  `mid/short` ratio decays as the parent becomes more strongly rod-like, while
  still requiring the scene to be crowded and the future snap distance to stay
  within a radius-scaled limit.

## 2026-06-16 Continuation: Fresh 0-22 Guardrail After Stronger Halfspace Clamp

Fresh CUDA guardrail run:

`/run/media/blue-lobster/disk3/celluniverse_output/outputs_fluo/output_ubuntu_fluo_fresh_0-22_celluniverse2_tuning_20260616_091207`

Configuration change under test:

- `split_daughter_refit_halfspace_min_fraction: 0.18`
  (up from `0.15` in `config_celluniverse2_tuning.yaml`)

Why this was tried:

- The instrumented frame-`85` frontier run showed the first true miss was a
  near-miss at `bio_separation_gate`, not an initial proposal failure.
- `cell_11100` produced a good split seed, but daughter PCA refit drift pulled
  one daughter back inward. The final rejection was only about one pixel short
  of the required separation:
  - `sep=30.3541`
  - `minBioSep=31.3178`
- That suggested the narrowest first fix was to keep post-refit daughters
  closer to the accepted seed pair before changing broader split logic.

Guardrail result:

- The full fresh `0-22` window validated clean against GT under `matched<=35`.
- Frames `0-22` all had `missing=0` and `extra=0`.
- This includes the earlier sensitive frames `18-22`, which also validated
  clean after the stronger halfspace clamp.
- A second fresh CUDA rerun,
  `output_ubuntu_fluo_fresh_0-90_celluniverse2_tuning_20260616_093027`,
  reproduced the same behavior through frame `22` while continuing forward:
  frames `12`, `16`, `18`, `19`, `20`, `21`, and `22` were rechecked live and
  all matched GT exactly. This makes the early clean span look reproducible,
  not just a one-off clean run.

Cross-window interpretation:

- The stronger halfspace clamp looks globally safe in the early sparse-to-mid
  regime. It did not reopen the early bridge or timing-caveat windows.
- The later crowded-window failures recorded elsewhere still point to
  environment-sensitive gating rather than to this clamp itself.
- Current evidence still supports separating parameters into two families:
  - likely global guards:
    - hard false-daughter density floor
    - third-cell claim protection
    - ordinary hard-overlap protection
  - likely environment-sensitive quantities:
    - future-aligned rod-tip flatness threshold
    - weak-balance acceptance thresholds
    - rescue drift tolerance when clean future support is strong but the local
      neighborhood is crowded

Working hypothesis for future adaptive tuning:

- The next adaptive function candidate is not the early halfspace clamp. It is
  the crowded-scene rod/flatness behavior, likely as a function of:
  - parent rod strength (`long/mid`, `mid/short`, or final elongation),
  - local crowding / nearest-neighbor spacing,
  - future-center snap quality relative to parent radius,
  - minimum daughter brightness support across future frames.

## 2026-06-16 Continuation: Dense `119-121` Experience From Retained Later Runs

Retained later-window evidence re-read:

- `run_107_resume119_to200_strong_cost_guard`
- `run_112_resume120_to200_clean_pca_cont_snap`

What keeps repeating in the crowded regime:

1. `future_aligned_rod_tip_flatness_gate`
   - Many parents in frames `119-121` are rejected here first.
   - Typical examples have:
     - parent shape about `1.5-1.9`
     - `long/mid` only modestly distinct
     - `mid/short` roughly `1.15-1.58`
   - This is consistent with the earlier crowded-window observation that a
     single flatness constant is too crude: later frames contain many
     plausible rods that are stronger than the sparse early false positives but
     weaker than the most obvious bridge-cut cases.

2. `parent_distance_balance_gate`
   - This remains a real later-window blocker even when future support exists.
   - The repeated failure pattern is not simply "future support missing." It is
     often:
     - `futureBoth=2`
     - `futureMissing=0`
     - but low parent-balance, sometimes together with weak future minimum
       brightness.
   - Example from the retained logs: frame `120` `cell_111100` had
     `futureBoth=2`, `activeParentBalance=0.19`, but
     `parentDistBalance=0.115` and `futureBestMinBrightness=0.048`.
   - This supports keeping balance protection global, while allowing any rescue
     logic to depend on both support count and support quality.

3. `bio_separation_gate`
   - In the later crowded runs, many PCA-bridge proposals survive far enough to
     die here rather than at the earlier proposal filters.
   - Several are not catastrophically wrong; they are near-miss crowded
     daughters whose final separation stays below the required biological
     minimum after refit.
   - This is where the early halfspace-clamp result matters: the stronger
     post-refit seed preservation may help this family without touching the
     earlier proposal scoring.

Additional useful detail:

- In retained `run_112`, ordinary non-future-backed daughters were still using
  the older halfspace clamp value `0.15`, while future-backed bridge rescues
  used the stronger `0.70` path. The fresh `0-22` guardrail now shows that
  raising the ordinary clamp to `0.18` is safe in the sparse regime.
- Clean-future claim bypasses appear frequently in the later windows. That is a
  hint that blocker ownership is real there, not a logging accident. Any later
  relaxation should therefore avoid weakening claim protection globally.

Current take-away:

- The most promising adaptive-function target is still the crowded rod-tip
  family, not the hard density floor or the ordinary claim/overlap defenses.
- If the fresh `0-90` run reaches the historical frame-`85` frontier cleanly,
  the next direct experiment should be the new `120-160` tuning preset so the
  updated halfspace clamp can be tested against this crowded `bio_separation`
  and flatness/balance mixture.

`/run/media/blue-lobster/disk3/celluniverse_output/outputs_fluo/celluniverse2_goal_0_200_20260612/run_056_from0_to200_run010_followup`

- Started from frame `0` to `200`.
- CUDA confirmed in `stdout.log`: N2V2 selected device `cuda`.
- Threads: `CELLUNIVERSE_THREADS=4`.
- Previous comparison kept: `run_055_from0_to200_no_dark_future_axis_gate`.
- Older active-goal comparison `run_054` was removed from the debug folder after
  its evidence was recorded.
- Initial checkpoints reached frame `2`; continue until the next true mismatch
  or completion.

## 2026-06-13 Continuation: Frame 75 Stale Daughter Follow-Up

Fresh current-checkout run:

`/run/media/blue-lobster/disk3/celluniverse_output/outputs_fluo/celluniverse2_goal_0_200_20260612/run_057_from0_to200_after_run010_followup`

Validated behavior before the failure:

- Fresh frame-0 start reached checkpoint `78`.
- Count validation was clean through frame `74`, except the known GT timing
  caveats at frames `18` and `34`.
- Frames `58-64` were count-clean in the current checkout, so the old June 9
  resume51 failure family stayed fixed.

First persistent count failure:

- Frame `75`: pred `52`, GT `51`; the +1 persisted through frames `76` and
  `77`.
- The extra prediction was `cell_3011`.
- Assignment inspection showed all GT labels already covered. `cell_3011` was
  unassigned, nearest GT `672` was about `14.33` px away, and GT `672` was
  already assigned to `cell_21010` at about `1.11` px.
- Local log evidence showed `cell_3011` had `background_body` signal-center
  rescue with `action=no_compatible_center`, followed by severe overlap with a
  live neighboring cell. This looked like a stale daughter lifecycle issue, not
  a delayed GT split.

Fix:

- Added scoped CellUniverse2 stale-daughter cleanup:
  - records older split daughters whose body looks like background and whose
    signal-center rescue cannot find a compatible target,
  - during hard-overlap repair, removes such a stale daughter if it severely
    overlaps a live non-sibling cell,
  - leaves ordinary hard-overlap restore/protection behavior unchanged for
    other cells.
- Added config switches in `config/config_celluniverse2.yaml`:
  - `celluniverse2_stale_daughter_cleanup_enabled: true`
  - `celluniverse2_stale_daughter_cleanup_min_age: 4`
  - `celluniverse2_stale_daughter_cleanup_overlap: 0.50`
- Build check passed with `cmake --build build -j 4`.

Fresh validation run after the fix:

`/run/media/blue-lobster/disk3/celluniverse_output/outputs_fluo/celluniverse2_goal_0_200_20260612/run_058_from0_to200_stale_daughter_cleanup`

Run settings:

- Fresh start: frame `0` to `200`.
- CUDA requested via `CUDA_VISIBLE_DEVICES=0`; `stdout.log` confirms N2V2 used
  `cuda`.
- Threads: `CELLUNIVERSE_THREADS=4`.
- Previous comparison kept in the active folder: `run_057`.

Validation result:

- Checkpoints reached frame `84`, giving a fresh `0-84` continuous span.
- Count validation against GT is clean through frame `84` except:
  - frames `17-18`: early split timing caveat; frame `19` returns to pred `12`,
    GT `12`
  - frame `34`: known large split-burst timing caveat; frame `35` returns to
    pred `24`, GT `24`
- The old frame-75 stale-daughter symptom is absent so far:
  - frame `75`: pred `51`, GT `51`
  - frame `76`: pred `51`, GT `51`
  - frame `77`: pred `51`, GT `51`
- Frames `80-84` also match GT counts:
  - frame `80`: pred `53`, GT `53`
  - frame `81`: pred `53`, GT `53`
  - frame `82`: pred `54`, GT `54`
  - frame `83`: pred `54`, GT `54`
  - frame `84`: pred `55`, GT `55`
- Important caveat: no `[CellUniverse2 Stale Daughter Cleanup]` log entry had
  fired through checkpoint `84`. The old symptom is fixed in this run's
  observed behavior, but the new cleanup path itself has not yet been proven as
  the causal mechanism.
- First true failure was frame `85`: pred `60`, GT `60`, but assignment max
  distance was about `107.39`.
- Bad assignment detail:
  - `cell_110101 -> GT148` was about `107.39` px away.
  - Predicted center was approximately `(198.2,153.0,81.3)`.
  - GT148 center was approximately `(191.0,205.0,175.0)`.
- Local log diagnosis:
  - `cell_11010` was accepted with best label `bridge_axis_place`.
  - For the same parent/frame, the PCA bridge proposal was rejected as
    `no_dark_future_axis_projection`.
  - The rod-tip fallback reused the same future-aligned center pair, then the
    generated `bridge_axis_place` alternate won by cost. This is a logic
    conflict: the no-current-bridge evidence blocked PCA bridge but not the
    rod-tip axis-place alternate.

Follow-up fix after `run_058`:

- Added config:
  `pca_bridge_future_window_rod_tip_axis_place_max_parent_shape: 2.20`.
- `Frame::trySplitCellPhased` now skips the `bridge_axis_place` alternate for
  high-shape future-aligned rod-tip proposals above that threshold.
- The primary rod-tip future-center proposal remains available.
- Rationale:
  - Frame `18` valid moderate rod case used `bridge_axis_place` at parent shape
    about `2.08`, so it stays below the cutoff.
  - Frame `85` false `cell_11010` case had parent shape about `2.34`; only the
    axis-place alternate won, while the primary rod-tip proposal did not.
- Build passed with `cmake --build build -j 4`.
- Fresh validation run started:
  `/run/media/blue-lobster/disk3/celluniverse_output/outputs_fluo/celluniverse2_goal_0_200_20260612/run_059_from0_to200_axis_place_shape_gate`
- Run settings:
  - Fresh start from frame `0` toward `200`
  - `CELLUNIVERSE_THREADS=4`
  - `CUDA_VISIBLE_DEVICES=0`
  - previous comparison kept: `run_058`
- Interim validation after crossing the record threshold:
  - Checkpoints reached frame `85`; this configuration has a fresh `0-84`
    validated continuous span and then fails hidden assignment at frame `85`.
  - Counts are clean except the known one-frame GT timing caveats:
    frame `18` resolves by frame `19`, and frame `34` resolves by frame `35`.
  - Frame `18` `cell_01` still accepted with best label `bridge_axis_place`,
    confirming that the moderate rod case stayed below the new cutoff.
  - Frame `34` `cell_110` logged `Split Bridge AxisPlace Skip` but still
    split correctly through `bridge_primary`, and frame `35` matched GT.
  - Additional high-shape axis-place skips occurred for `cell_20` and
    `cell_201`; counts remained clean afterward.
  - Frame `59` had a dense split burst with seven accepted split attempts;
    after excluding trash, frame `59` matched GT (`37` vs `37`) and frame `60`
    also matched GT (`43` vs `43`).
  - Frames `58-64` all matched GT counts. A Hungarian assignment sweep over
    count-matched frames `58-64` had worst max distance about `28.12` px, so the
    old `cell_01100`/GT354 collapse region remains fixed under this version.
  - The old frame-75 stale-daughter window also stayed fixed: frames `75-78`
    all matched GT counts, with assignment max distances about `13.84`,
    `24.28`, `17.60`, and `21.82` px respectively. Frames `79-84` also stayed
    count-clean with assignment max distances below `30` px.
  - Frame `85` still failed hidden assignment: pred `60`, GT `60`, but
    `cell_110101 -> GT148` was about `109.07` px away. The axis-place cutoff
    fired, but the primary rod-tip fallback was accepted through
    `futureRodTipPrimaryRescue=1` despite `imageDiff=+12747.9` and
    `totalDiff=+12302.2`.
  - Follow-up fix: require current-frame image gain for high-shape rod-tip
    primary rescues when the proposal uses a far aligned future fallback. Build
    passed after the guard.
  - Focused validation run
    `run_060_resume85_to88_rodtip_primary_gain_guard` resumed from `run_059`
    frame `85` to `88`. The guard blocked the false `cell_11010` current-frame
    rod-tip placement, but GT148 remained unmatched: frame `85` pred `59` vs
    GT `60`, frame `86` pred `66` vs GT `67`, frame `87` pred `70` vs GT `77`,
    and frame `88` pred `70` vs GT `82`. This means the next frontier is the
    real missing `cell_11100`/GT148 split-placement path, not the old
    `cell_11010` false daughter.

## 2026-06-13 Continuation: Frame 106/108 Split Fix

Active goal folder reused:

`/run/media/blue-lobster/disk3/celluniverse_output/outputs_fluo/celluniverse2_goal_0_200_20260612`

Current comparison runs used in this pass:

- Previous failing/current evidence: `run_038_resume103_to200_after_delayed_gt_check`
- Failed intermediate test after position lock only: `run_041_debug_cell3100_frame106_axis_seed_lock`
- Passing focused test after position lock plus scoped cost rescue: `run_043_debug_cell2100_frame108_with_future`

### Frame 102 Was Not Tuned

In `run_037_resume97_to200_birth_frame_daughter_lock`, frame 102 showed one
extra predicted object (`97` pred vs `96` GT), but by frame 103 the GT count
caught up and the daughters of `cell_20000` mapped cleanly:

- `cell_200000 -> GT606`, distance about 9.4 px
- `cell_200001 -> GT603`, distance about 6.6 px

Decision: do not tighten split gates for frame 102 because this is a one-frame
GT/timing delay, and tuning against it would risk re-breaking valid early split
support.

### Frame 106 Real Miss: `cell_3100`

`run_038_resume103_to200_after_delayed_gt_check` first real failure:

- Frame 106: `99` pred vs `100` GT.
- Missing GT label: `715` at about `(573, 291, 154)`.
- Parent: `cell_3100` followed GT712 but missed the second daughter GT715.

Focused log evidence from `run_040_debug_cell3100_frame106_future`:

- `bridge_axis_place` seed placed the missing daughter correctly:
  - seed2 about `(571.281, 293.625, 152.661)`, near GT715.
- Before the fix, split-validation PCA moved daughter 1 by about 34 px.
- The distorted final bridge then failed the bio bridge gate:
  - `valleyFromBright=0.835357`, `valleyLimit=0.8`.

Fix 1 in `src/Frame.cpp`:

- For CellUniverse2 `bridge_axis_place` split candidates, lock daughter position
  during split-validation PCA refit.
- PCA still updates shape and rotation, but the candidate centers stay on the
  validated axis-place seeds.

Result in `run_041_debug_cell3100_frame106_axis_seed_lock`:

- The PCA drift was fixed (`refineDrift1=0`, `refineDrift2=0`).
- Bridge evidence became clean:
  - `valleyFromBright=0.59479`
  - `gapDensity=0.0317367`
- Still rejected by cost gate:
  - improvement `-3499.38`
  - required adaptive threshold `-4344.36`

Fix 2 in `src/Frame.cpp`:

- Add a scoped `bridge_axis_place` clean near-threshold cost rescue.
- Conditions require:
  - `bestLabel == bridge_axis_place`
  - CellUniverse2 bridge-proposal-only path
  - max daughter seed drift <= 1 px
  - clear bridge valley (`valleyFromBright <= 0.65`)
  - low bridge density (`gapDensity <= 0.06`)
  - negligible overlap penalty
  - at least 75% of the normal adaptive cost improvement.

This avoids lowering global `split_cost_fraction` and keeps earlier false-split
density/valley protections intact.

Focused result in `run_042_debug_cell3100_frame106_axis_seed_lock_cost_rescue`:

- Frame 106: `100` pred vs `100` GT.
- `cell_31001 -> GT715`, distance about 3.4 px.
- `bridgeAxisPlaceCleanNearThresholdRescue=1` fired for `cell_3100`.

### Frame 108 Short-Window Artifact

`run_042` ended at frame 108, so frame 108 had no future-window support. It
showed `100` pred vs `101` GT and missed GT627 near `cell_2100`.

Decision: do not tune from `run_042` frame 108 because the log explicitly showed
`future_window_unavailable` for the relevant split proposal.

Future-aware rerun `run_043_debug_cell2100_frame108_with_future` covered
frames 106-110 from the same frame-105 checkpoint. Results:

| Frame | Pred real | GT | Delta | Max assign dist | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| 106 | 100 | 100 | 0 | 18.29 | OK; `cell_3100` fixed |
| 107 | 100 | 100 | 0 | 20.75 | OK |
| 108 | 101 | 101 | 0 | 24.03 | OK when future support is available |
| 109 | 102 | 102 | 0 | 23.24 | OK |
| 110 | 102 | 102 | 0 | 24.07 | OK |

Next validation should be a true 0-to-n smoke, not another middle resume. A good
next range is 0-110 or 0-120 with CUDA and `CELLUNIVERSE_THREADS=4`, because it
covers the early split cases, the original 58-64 failure family, the frame 70/71
bridge behavior, and the newly fixed 106-110 region.

## 2026-06-13 Continuation: Frame 7 Sparse Signal-Center Rescue

After the frame-106/108 focused fixes, the first continuous 0-to-120 smoke
(`run_044_0_to120_axis_seed_lock_cost_rescue`) was stopped at the first real
mismatch:

- Frame 7: `6` pred vs `7` GT.
- Frame 8 caught up to `7` pred vs `7` GT, so this was a one-frame-late split.
- Parent: `cell_2` remained unsplit at frame 7, then split into `cell_20` and
  `cell_21` at frame 8.

Frame-7 log evidence:

- The `signal_center_split` proposal for `cell_2` had strong two-frame future
  support but was rejected by `sparse_signal_center_moderate_low_separation_gate`.
- Key values were approximately:
  - `parentShape=2.145`
  - `sepRatio=1.158`
  - `parentDistanceBalance=0.751`
  - `futureBoth=2`
  - `futureMissing=0`
  - `parentPersists=0`
  - `futureBestMinBrightness=0.115`
- The fallback `pca_bridge_cut` path was not a good candidate to rescue here:
  daughter PCA drift was large and the final cost was positive.

Fix in `src/CellUniverse.cpp`:

- Add a narrow sparse-frame exception for current-frame signal-center proposals
  when a rod-like parent has two-frame future evidence.
- The exception requires:
  - sparse frame (`cells < 50`)
  - rod-like but not extreme parent shape (`2.10 <= parentShape < 2.20`)
  - near-biological daughter separation (`sep >= 1.10 * minBioSep`)
  - balanced daughters relative to the parent (`parentDistanceBalance >= 0.55`)
  - both future frames support both daughters
  - no future missing frame and no parent persistence
- This keeps the original sparse gate active for weak early false candidates.

Conflict check:

- In `run_045_0_to12_sparse_signal_twoframe_rescue`, the new rescue fired for
  frame 7 only.
- Earlier weak proposals at frame 3 (`parentShape=1.758`, `sepRatio=0.493`) and
  frame 4 (`parentShape=1.612`, `sepRatio=0.642`) were still rejected.
- Frames 0-12 all matched GT counts after the change; worst assignment distance
  stayed below 19 px.

Short validation result from `run_045_0_to12_sparse_signal_twoframe_rescue`:

| Frame | Pred real | GT | Delta | Max assign dist | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| 0 | 4 | 4 | 0 | 10.48 | OK |
| 1 | 4 | 4 | 0 | 8.64 | OK |
| 2 | 4 | 4 | 0 | 9.30 | OK |
| 3 | 4 | 4 | 0 | 9.52 | OK; weak false candidate still rejected |
| 4 | 6 | 6 | 0 | 16.76 | OK; weak false candidate still rejected |
| 5 | 6 | 6 | 0 | 8.89 | OK |
| 6 | 6 | 6 | 0 | 10.37 | OK |
| 7 | 7 | 7 | 0 | 11.16 | OK; `cell_2` split no longer late |
| 8 | 7 | 7 | 0 | 12.51 | OK |
| 9 | 7 | 7 | 0 | 18.30 | OK |
| 10 | 7 | 7 | 0 | 12.33 | OK |
| 11 | 8 | 8 | 0 | 12.59 | OK |
| 12 | 8 | 8 | 0 | 15.78 | OK |

Next validation should return to a continuous 0-to-120 run from the initial CSV,
because the short window only proves this early-frame rescue did not immediately
break neighboring cases.

## 2026-06-13 Continuation: Frame 18 Coupled False Split and Missed Split

The next continuous validation (`run_046_0_to120_sparse_signal_twoframe_rescue`)
was stopped at frame 18:

- Count matched (`10` pred vs `10` GT), but assignment showed a severe error.
- `cell_01` was still correct at frame 17, then falsely split locally at frame
  18 into `cell_010` and `cell_011`.
- The matching problem hid a simultaneous missed true split: `cell_10` should
  split to cover GT66, while the false `cell_01` daughter made the count look
  correct.

Evidence for the false `cell_01` split:

- Accepted source: `pca_bridge_cut`, `bestLabel=bridge_primary`.
- Future brightness was weak: `futureBestMinBrightness=0.0299536`.
- The bridge gap was dense: `gapDensity=0.237009`.
- The valley was only moderate: `valleyFromBright=0.516676`.
- Daughter refit drifted far from the proposal: max drift about `24.93`, with
  parent max radius about `37.52`.

Evidence for the missed true `cell_10` split:

- Rod-tip fallback was rejected only by the flatness threshold:
  - `parentShape=1.82751`
  - `longMidRatio=1.04675`
  - `midShortRatio=1.74589`
- Immediate future support was strong after allowing the proposal through:
  - `futureBoth=2`
  - `futureMissing=0`
  - `futureBestMinBrightness=0.186289`

Fixes:

- In `src/CellUniverse.cpp`, relax the rod-tip fallback flatness evidence from
  `midShortRatio >= 1.75` to `>= 1.70`. This admits the flattened-but-real
  `cell_10` split while keeping rounder cases such as frame-18 `cell_20`
  rejected (`midShortRatio=1.40366`).
- In `src/Frame.cpp`, add a scoped `dense_drifting_bridge` rejection for
  CellUniverse2 `bridge_primary` proposals when:
  - `gapDensity >= 0.20`
  - `valleyFromBright >= 0.45`
  - max daughter seed drift >= `0.45 * parentMaxRadius`
  - future support is weak, missing, or parent-persistent.

Conflict check:

- This does not affect the frame-106 `bridge_axis_place` rescue because it is
  label-scoped to `bridge_primary`, and frame 106 had a clean low-density bridge.
- It does not block the valid frame-18 `cell_00` split: that split had
  `gapDensity=0.0205`, `valleyFromBright=0.289`, and daughter drift below 1 px.
- It preserves the frame-7 sparse signal-center rescue because that path does
  not use `bridge_primary` validation.

Focused validation `run_047_resume18_18to20_dense_bridge_flat_tip_fix` resumed
from `run_046` frame 17 and covered frames 18-20:

| Frame | Pred real | GT | Delta | Max assign dist | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| 18 | 10 | 10 | 0 | 11.07 | OK; `cell_10` split accepted, bad `cell_01` split rejected |
| 19 | 12 | 12 | 0 | 11.04 | OK; `cell_01` splits correctly one frame later |
| 20 | 12 | 12 | 0 | 12.52 | OK |

Next validation should return to a continuous 0-to-120 run from the initial CSV.

## 2026-06-13 Continuation: Frame 34 GT Timing Caveat

The next continuous validation (`run_048_0_to120_dense_bridge_flat_tip_fix`)
ran cleanly through frame 33 and then flagged frame 34:

- Frame 33: `15` pred vs `15` GT, max assignment distance `12.70`.
- Frame 34: `16` pred vs `15` GT, max assignment distance `18.33`.
- Frame 35: `24` pred vs `24` GT, max assignment distance `23.16`.

The frame-34 count mismatch is likely a rare GT timing mismatch rather than a
pipeline failure:

- The extra predicted split is `cell_010 -> cell_0100/cell_0101`.
- At frame 34 both daughters are nearest to the single GT258 center:
  - `cell_0100` at distance `18.33`
  - `cell_0101` at distance `20.17`
- At frame 35 the same two daughters resolve into two distinct GT cells:
  - `cell_0100 -> GT290` at about `1.3` px
  - `cell_0101 -> GT259` at about `3.8` px
- The split had strong two-frame future support:
  - `futureBoth=2`
  - `futureMissing=0`
  - `futureBestMinBrightness=0.166236`
  - `parentPersists=0`

No pipeline tuning was applied for this case. Treat frame 34 as a GT timing
caveat unless visual inspection shows the split is biologically wrong.

## 2026-06-13 Continuation: June 9 Resume51 Run Inspection

The run
`output_ubuntu_fluo_resume51_51-200_celluniverse2_run010_20260609_214650`
was inspected as the next tuning source. It was a resume from
`celluniverse2_smoke_0_50_goal_20260610/run_010_from0_0_50` and produced
checkpoints for frames 51-64.

Checkpoint/GT comparison:

| Frame | Pred real | GT | Delta | Max assign dist | Note |
| --- | ---: | ---: | ---: | ---: | --- |
| 51 | 26 | 26 | 0 | 12.97 | OK |
| 52 | 28 | 28 | 0 | 14.50 | OK |
| 53 | 28 | 28 | 0 | 15.07 | OK |
| 54 | 28 | 28 | 0 | 22.26 | OK |
| 55 | 28 | 28 | 0 | 20.26 | OK |
| 56 | 28 | 28 | 0 | 17.26 | OK |
| 57 | 28 | 28 | 0 | 16.83 | OK |
| 58 | 31 | 30 | +1 | 15.18 | likely one-frame GT timing caveat for `cell_0010` |
| 59 | 37 | 37 | 0 | 265.53 | real mismatch in the old run |

Frame 58 caveat:

- `cell_0010` split into `cell_00100/cell_00101`.
- At frame 58 both daughters were nearest to the single GT417 label, with
  small assignment distances.
- At frame 59 the same daughters mapped cleanly to two distinct GT labels:
  `cell_00100 -> GT418` and `cell_00101 -> GT433`.
- Treat this as another likely GT timing delay, not a tuning target.

Frame 59 stale-run issue:

- GT338 was missed; its nearest prediction was `cell_0111` at about `55.2` px.
- The old run had a plausible `cell_0111` split candidate:
  - daughter near GT338: `(354.219, 406.839, 86.4651)`
  - GT338: `(355, 407, 91)`
- The split was rejected by `daughter_midpoint_parent_drift`:
  - `midpointDistance=31.191`
  - `limit=29.9939`
  - fraction `0.95`

Current-checkout cross-check:

- Current resume run
  `run_049_resume36_to120_after_frame34_gt_timing_note` reached frame 59.
- It still shows the same frame-58 GT timing caveat, but frame 59 is clean:
  `37` pred vs `37` GT, max assignment distance `16.62`.
- Therefore the old June 9 frame-59 miss appears already fixed by current
  midpoint/future-supported rescue changes.

Next validation should be a fresh `0-to-n` CUDA smoke, not another resume, so
the current checkout proves it did not break early-frame logic.

## 2026-06-13 Continuation: Fresh 0-65 Current-Checkout Proof

Fresh CUDA validation run:

`/run/media/blue-lobster/disk3/celluniverse_output/outputs_fluo/celluniverse2_goal_0_200_20260612/run_053_from0_to65_current_midpoint_rescue`

Result:

- Completed frames `0-65` from the initial CSV without resume.
- Runtime used `CELLUNIVERSE_THREADS=4` and `CUDA_VISIBLE_DEVICES=0`.
- `stdout.log` selected CUDA for N2V2 preprocessing.
- Final runtime: `1697.15` seconds.
- Final checkpoint: `checkpoints/frame_065.txt`.

Validator result:

- `OK_WITH_CAVEATS through 65 frames 66`.
- Worst assignment summary: frame `65`, predicted `46`, GT `46`, max distance
  `31.87`, p95 `9.85`.
- Frame `34` remains a GT timing caveat for `cell_010`; the two daughters
  resolve cleanly at frame `35`.
- Frame `58` remains a GT timing caveat for `cell_0010`; the two daughters
  resolve cleanly at frame `59`.
- Frame `59` no longer reproduces the old June 9 miss; predicted and GT counts
  match (`37`/`37`) with max distance `16.62`.

No tuning was applied after this run. The current configuration/code version is
now recorded as a confirmed `>50` continuous-frame version in
`docs/celluniverse2_long_run_config_records.md`.

## 2026-06-13 Continuation: `run_057` Fresh 0-200 Follow-Up

Fresh CUDA validation run:

`/run/media/blue-lobster/disk3/celluniverse_output/outputs_fluo/celluniverse2_goal_0_200_20260612/run_057_from0_to200_after_run010_followup`

Reason:

- `run_056_from0_to200_run010_followup` was a stale/interrupted below-threshold
  run and was removed after stopping its old process.
- `run_057` is the active fresh frame-0 smoke using the same current
  CellUniverse2 code/config as `run_055`, with no new tuning applied before
  launch.
- The active goal folder now keeps only previous comparison `run_055` plus
  current `run_057`.

Command shape:

`CELLUNIVERSE_THREADS=4 CUDA_VISIBLE_DEVICES=0 ./build/celluniverse 0 200 ... config/config_celluniverse2.yaml config/embryo/initial_embryo_0_trash_labeled.csv`

Validation so far:

- CUDA selected for N2V2 preprocessing.
- Checkpoints observed through frame `69` when this note was refreshed.
- Count validation against GT:
  - frames `0-17`: clean
  - frame `18`: pred `11`, GT `10`; known one-frame GT timing caveat
  - frames `19-33`: clean
  - frame `34`: pred `16`, GT `15`; known one-frame GT timing caveat
  - frames `35-69`: clean
- Assignment cross-check:
  - Hungarian assignment over predicted non-trash centers and GT bright-cube
    centers found no count-matched frame through `61` with max distance above
    `45`.
  - frame `35` max assignment distance was about `23.16`.
  - frame `36` max assignment distance was about `28.00`.
  - frame `52` max assignment distance was about `14.49`.
  - frame `58` max assignment distance was about `14.45`.
  - frame `59` max assignment distance was about `16.60`.
  - frame `60` max assignment distance was about `12.95`.
  - frame `61` max assignment distance was about `11.93`.

Cross-validation note:

- The frame `18` and frame `34` count mismatches match earlier good tracking
  records: they resolve cleanly one frame later and should not be tuned against
  unless future evidence shows spatial assignment damage.
- The stale June 9 resume-run frame-58 caveat is not present here; frame `58`
  is both count-clean and assignment-clean.
- `run_057` has now exceeded previous comparison `run_055`, which was observed
  through frame `68`.

Next action:

- Continue `run_057` until the next true count or assignment mismatch, or until
  it completes `0-200`.
- If a new mismatch appears, inspect GT assignment, compare against `run_055`
  and other known good records where available, then tune parameters first.

## 2026-06-13 Continuation: Frame-85 Resume Frontier Through `run_067`

Context:

- Strongest fresh baseline remains
  `run_059_from0_to200_axis_place_shape_gate`, validated through frame `84`
  with known GT timing caveats, then failing at frame `85`.
- Focused resume tests from `run_059` frame `85` are not promoted as
  `>50`-frame configuration records; they are frontier evidence only.

Resume evidence:

- `run_062_resume85_to90_delayed_future_cost` fixed the frame-86 GT148 miss
  but left frame `87` short by GT `171` and GT `388`.
- `run_063_resume85_to90_no_dark_future_rescue` fixed those two frame-87
  misses, but left GT `84` unmatched because `cell_20101` stayed unsplit.
- `run_064_resume85_to90_current_aligned_pair` allowed a current aligned-pair
  fallback too broadly and regressed frame `86`; this branch was stopped.
- `run_065_resume85_to90_scoped_current_aligned_pair` kept frame `86` clean
  but picked a wrong current pair for `cell_20101`, leaving frame `87` short by
  GT `84` and GT `388`.
- `run_066_resume85_to90_snapshot_bbox_fallback` restricted the fallback to
  the snapshot bbox, preserving frame `86`, but still failed `cell_20101` at
  frame `87`.
- `run_067_resume85_to90_proposal_axis_fallback` used the current PCA bridge
  proposal axis for the scoped fallback. Validation result:
  - frame `85`: pred `59`, GT `60`, unmatched GT `148`
  - frame `86`: pred `67`, GT `67`, clean count; max assignment about `12.90`
  - frame `87`: pred `76`, GT `77`, unmatched GT `388`

Conclusion:

- The proposal-axis fallback trades one frame-87 miss for another rather than
  solving the frontier, so it should not be treated as a promoted fix.
- The next fix should preserve the `run_063` recovery for GT `171`/`388` while
  adding a more targeted recovery for `cell_20101` / GT `84`, instead of
  broadening the current aligned-pair fallback across all post-PCA proposals.
- Any new candidate should be checked against frame `86` as a guardrail before
  spending time on longer runs.

## 2026-06-13 Continuation: Frame-89 Signal-Center Claim Fix

Focused resume evidence:

- `run_068_resume85_to90_near_sep_tolerance` resumed from `run_059` and kept
  frame `86` clean (`pred=67`, GT `67`), but was stopped before extending the
  frontier.
- `run_069_resume87_to90_near_sep_tolerance` resumed from `run_068` frame `87`
  and fixed the frame-87 split conflict:
  - frame `87`: `pred=77`, GT `77`, no unmatched labels, max assignment about
    `26.84`.
  - frame `88`: `pred=82`, GT `82`, no unmatched labels, but one z-heavy
    warning remained: `cell_000001 -> GT497`, about `39.59` px.
- `run_070_resume88_to100_near_sep_tolerance` continued from `run_069` frame
  `88` and exposed the next true failure:
  - frame `88`: still count-clean with the same `cell_000001 -> GT497`
    warning.
  - frame `89`: `pred=86`, GT `87`, unmatched GT `107`.

Failure mechanism:

- Frame `89` GT `107` belonged to the `cell_10111` neighborhood. The unsplit
  `cell_10111` sat between GT `100` and GT `107`, about `22-24` px from each.
- `cell_10111` had a clean signal-center split proposal:
  `futureBoth=2`, `futureMissing=0`, `parentPersists=0`,
  `futureBestMinBrightness=0.446599`, score about `0.788`.
- A neighboring `cell_10101` proposal shared one candidate center and used the
  `strong_asymmetric_future_support` local-center bypass to evict
  `cell_10111`, even though `cell_10111` had cleaner future support and a
  better score. `cell_10101` then failed in the full split gate as
  `d1_bridging_to_cell_10111`, leaving `cell_10111` unsplit.

Fix:

- Tuned the signal-center local claim bypass in `src/CellUniverse.cpp`.
- A later strong-asymmetric proposal may now evict an existing center owner
  only if the owner lacks clean future support, or the later proposal is
  clearly better by score (`proposalScore + 0.20 < blockerScore`).
- Added log fields for `score`, `blockerScore`, and `blockerCleanFuture` on
  both bypass and reject paths.

Validation:

- Build passed with `cmake --build build -j 4`.
- `run_071_resume88_to92_center_claim_guard` resumed from `run_069` frame `88`
  using CUDA and 4 threads.
- Frame `88`: `pred=82`, GT `82`, no unmatched labels; same known warning
  `cell_000001 -> GT497`, about `39.59` px.
- Frame `89`: fixed. `cell_10111` was scheduled and accepted with
  `costDiff=-149213`; validation was `pred=87`, GT `87`, no unmatched labels,
  max assignment about `20.48`.
- Frame `90`: `pred=88`, GT `88`, no unmatched labels, max assignment about
  `22.46`.
- Frame `91`: `pred=88`, GT `88`, no unmatched labels, max assignment about
  `25.24`.
- Frame `92`: `pred=88`, GT `88`, no unmatched labels, max assignment about
  `22.57`.

Conclusion:

- The local-center claim guard fixed the frame-89 GT107 miss in the focused
  frontier without regressing frames `88-92`.
- This is still resume-only evidence. It is not a promoted `>50` configuration
  record until a fresh frame-0 run crosses the threshold with this patch.

## 2026-06-14 Continuation: Frame-140 Shared-Center Rod Gate

Focused evidence:

- The user-observed frame-140 failure is a current-center ownership problem:
  multiple rod-like wrong fits can try to use the same third live cell as split
  evidence.
- `run_181_resume140_to145_owned_center_seed_gate` confirmed CUDA was active,
  but it was stopped before checkpoint. It showed the intended change:
  `cell_200110` no longer snapped to the far borrowed current center and
  instead rejected through the raw PCA/bio path.
- `run_182_resume140_gate_check` resumed from
  `run_164_resume139_to200_current_locked_rescue` at `resume_from=140` and
  completed frame `140`.

Fixes applied:

- Current-frame PCA bridge / rod-tip center snapping now skips signal centers
  that are clearly owned by another live cell unless the center is still inside
  the parent core.
- Current-frame PCA bridge aligned-pair fallback now rejects pair choices whose
  maximum seed distance exceeds `max(maxSeedSnapDistance, 1.35 * parentMaxR)`.
- The `longMovedRawPcaBridgeProposal` exception now requires at least one
  future-supported daughter pair, no parent persistence, and minimum future
  brightness, instead of waiving the future window with zero future support.

Validation:

- Build passed with `cmake --build build -j 4`.
- `run_182_resume140_gate_check` confirmed `selected_device=cuda`.
- `cell_0110000` changed from an acceptable raw PCA bridge with
  `futureBoth=0` in `run_181` to rejection in `run_182`:
  `reason=future_window_unavailable`, with `proposalsFound=0` for the main PCA
  bridge pass.
- Frame `140` completed and wrote checkpoint
  `run_182_resume140_gate_check/checkpoints/frame_140.txt`.

## 2026-06-15 Continuation: Optimized Build and Frame 87-92 Future-Window Check

Build/compiler status:

- `CELLUNIVERSE_ENABLE_LTO` is off by default to avoid the compiler/LTO
  failure.
- Release optimization remains enabled for the main target:
  `-O3 -DNDEBUG -march=native -fopenmp`.
- The GCC 15 safe optimization workaround still scopes the reduced `-O1`
  compile option to `src/N2V2Preprocessor.cpp`.
- `cmake --build build --target celluniverse -j 4` completed successfully.

Focused run:

`run_228_resume87_to90_no_false_continuation_merge`

- Resumed from `run_226` around the frame-87 frontier.
- CUDA was active.
- Frame `87` had already passed after tightening the false continuation merge
  norm limit.
- Frame `88` validated cleanly: pred `82`, GT `82`, no missing/extra, max
  assignment distance `27.18`.
- Frame `89` validated cleanly: pred `87`, GT `87`, no missing/extra, max
  assignment distance `27.95`.
- The same run ended at frame `90`, and frame `90` appeared to miss GT label
  `362` near unsplit `cell_01011`. This was not treated as a real tuning
  failure because frame `90` was the final frame of the run; the validator logs
  showed `cell_01011` rejected as `future_window_unavailable` with
  `availableFutureFrames=0`.

Boundary check:

`run_229_resume90_to92_future_window_check`

- Resumed from `run_228` so frame `90` had future frames `91` and `92`
  available.
- Frame `90` accepted the clean signal-center split for `cell_01011`.
- Validation results:
  - frame `90`: pred `88`, GT `88`, no missing/extra, max assignment distance
    `17.65`.
  - frame `91`: pred `88`, GT `88`, no missing/extra, max assignment distance
    `14.72`.
  - frame `92`: pred `88`, GT `88`, no missing/extra, max assignment distance
    `13.64`.

Conclusion:

- No code or parameter tuning is justified from the apparent frame-90 miss in
  `run_228`; it was a short-window artifact caused by ending the run at the
  frame being validated.
- Future focused validation windows must include enough lookahead frames for
  CellUniverse2 future-window gates. For a frame `n` decision with two-frame
  support, run through at least `n+2` before treating a miss as real.

Result / next problem:

- The shared-third-cell false evidence route is blocked.
- Frame `140` still has six rods with ratio `>2.2` and two with ratio `>2.5`;
  top rods remain `cell_0110000`, `cell_110100`, `cell_201001`,
  `cell_100001`, `cell_2001110`, and `cell_100100`.
- No split was accepted in frame `140`, so this is not a pass. The next fix
  should add a true future-supported rod split path for cases like
  `cell_0110000`, where signal-center future evidence exists but a current
  center is neighbor-owned. Do not undo the ownership guard just to recover
  acceptance.

## 2026-06-14 Continuation: Dim Clean Future Signal Split Cost Rescue

Focused evidence:

- The user noted that frame `140` still showed many rod-like cells, including
  two wrong rod-like cells that appeared to include the same third cell. The
  focused logs confirmed two separate cases:
  - `cell_0110000` is a shared/borrowed-center rod and should stay rejected.
  - `cell_110100` has exact two-frame signal-center support and should split,
    but was previously rejected after overlap/cost because the current frame is
    dim and crowded.

Fixes applied:

- Added a narrow `dimExactFutureSignalOverlapCostRescued` cost gate in
  `src/Frame.cpp`.
- The rescue only applies to CellUniverse2 signal-center bridge candidates
  with:
  - exact center snap (`centerSnapMaxSeedDistance <= 1e-3`),
  - two-frame daughter support, no missing daughters, no parent persistence,
  - strong parent-distance balance,
  - near-zero daughter seed drift,
  - substantial image improvement,
  - bounded overlap penalty,
  - acceptable bridge gap density and valley evidence.
- The shared-third-cell case remains blocked by the existing ownership and
  future-window gates.

Validation:

- Build passed with `cmake --build build -j 4`.
- `run_187_resume140_to142_dim_signal_cost_rescue` resumed from
  `run_164_resume139_to200_current_locked_rescue`, used CUDA
  (`selected_device=cuda`), and completed through frame `142`.
- Frame `140`:
  - `cell_0110000` signal-center path rejected with
    `reason=signal_center_neighbor_claim_gate`; PCA bridge path rejected with
    `reason=future_window_gate`.
  - `cell_110100` accepted through the new narrow gate:
    `dimExactFutureSignalOverlapRescue=1`, `futureBoth=2`,
    `futureMissing=0`, `parentPersists=0`, `parentDistBalance=0.906909`,
    `maxDaughterSeedDrift=0`, `imageDiff=-79572.9`,
    `overlapDiff=110668`, and `costDiff=-42030.8`.
  - The split continuation merge was skipped for `cell_110100` with
    `reason=preserve_clean_future_signal_daughters`.
- Frame `142`:
  - `cell_1101000` attempted to jump to a far signal center, but the
    `recent_split_daughter_move_gate` blocked that move.
  - `cell_1101001` moved to the nearby supported center before PCA and then
    shape-refit with the split-daughter position lock.
- Final checkpoint written:
  `run_187_resume140_to142_dim_signal_cost_rescue/checkpoints/frame_142.txt`.

Conclusion:

- The specific frame-140 shared-third-cell false route stayed rejected while
  the true clean future-supported `cell_110100` split was accepted.
- This is focused resume evidence only. It validates the frame-140 patch path
  through frame `142`, but it is not a promoted `>50` or `0-200` configuration
  record.

## 2026-06-15 Continuation: Frame-140 Raw Rod Primary for Shared-Center Rod

Focused evidence:

- The user observed that frame `140` still contained many rod-like wrong cells,
  and that two wrong rods could include the same third cell, making the real
  bridge/split hard to detect.
- The focused run
  `run_194_resume140_to142_high_shape_rod_cost_rescue` resumed from
  `run_164_resume139_to200_current_locked_rescue`, used CUDA
  (`selected_device=cuda`), completed frame `140`, and was stopped while
  processing frame `141` after the frame-140 checkpoint had been written.
- Previous signal-center scoring could choose a borrowed/third-cell route.
  The raw rod-body PCA pair for `cell_0110000` was centered on the rod body and
  aligned with the expected split direction:
  `rawD1=(192.238,368.2,62.7933)`,
  `rawD2=(199.569,383.646,98.8554)`.

Fixes applied:

- For high-shape signal-center proposals, the raw rod-body PCA pair can become
  the primary proposal when its separation is comparable to the signal proposal
  and its midpoint stays close to the parent body. The original signal-center
  pair is kept as an alternate seed pair for diagnostics.
- Added a narrow hard-overlap bypass for this exact pattern:
  signal-center bridge primary, `gapStartBin <= -5`, immediate future support,
  no parent persistence, high parent elongation, good parent-distance balance,
  bounded overlap, and clean gap/valley evidence.
- Added a narrow near-tie cost rescue for the same high-shape raw-rod signal
  case. It requires image improvement, bounded overlap/cost near-tie, low
  daughter seed drift, good gap/valley evidence, and future support. This
  rescue is deliberately not available to ordinary rod-tip or PCA bridge
  candidates.

Validation:

- Build passed with `cmake --build build -j 4`.
- `run_194_resume140_to142_high_shape_rod_cost_rescue` accepted:
  - `cell_0110000` through `highShapeRawRodSignalNearTieRescue=1`.
  - `cell_110100` through the earlier exact future signal overlap rescue.
  - `cell_101010` and `cell_200010` through pre-existing gates.
- Questionable neighboring proposals stayed rejected, including:
  - `cell_1001110` rod-tip path:
    `reason=rod_tip_neighbor_claim_shape_gate`.
  - `cell_2011111` rod-tip/PCA paths:
    parent-distance / bio-separation gates.
  - `cell_01101111`, `cell_100100`, `cell_2001110`,
    `cell_01111101`, and `cell_100110` through weak bridge, cost, overlap, or
    density gates.
- Frame `140` checkpoint GT-center comparison around the target cluster:
  - `cell_01100000` matched GT `262` at distance `2.2`.
  - `cell_01100001` matched GT `116` at distance `5.5`.
  - The separate third cell `cell_2011111` remained represented and matched
    GT `544` at distance `8.3`.
  - Neighboring already-good cells stayed close to their expected GT centers:
    `cell_1001110` to GT `45` at `5.5`, `cell_100110` to GT `41` at `6.0`,
    `cell_1101000` to GT `38` at `9.4`, and `cell_1101001` to GT `197` at
    `9.6`.

Conclusion:

- The frame-140 shared-third-cell rod failure is fixed in focused resume
  evidence: `cell_0110000` splits into the intended two centers while the
  nearby third cell remains independent.
- This is still not a promoted `>50` or `0-200` record. The next validation
  must be a fresh frame-0 smoke run to confirm earlier good behavior was not
  broken.

## 2026-06-15 Continuation: Reject Shared-Third-Cell Rod Bypasses

Focused evidence:

- The user re-checked frame `140` visually and observed that many rod-like
  wrong cells remained, with two wrong rods able to include the same third cell.
  This exposed a weakness in the previous focused fix: a suspicious rod-like
  neighbor was still allowed to justify claim and overlap bypasses.
- `run_196_resume140_to142_no_suspicious_rod_bypass` first removed the generic
  `clean_future_suspicious_rod_blocker` overlap bypass. This kept the true
  `cell_0110000` raw-rod split accepted but changed several shared-neighbor
  routes into hard-overlap rejects.
- `run_197_resume140_to142_strict_rod_overlap` then tightened the remaining
  rod-tip continuation overlap bypass so it only tolerates modest overlap, not
  full containment of one daughter inside a third cell.

Fixes applied:

- Proposal claim bypass for a suspicious rod blocker is no longer generic. It
  is only available to a high-shape `signal_center_split` raw-rod-primary
  proposal with strong parent-centered geometry:
  `gapStartBin <= -5`, parent elongation at least `2.55`, parent-distance
  balance at least `0.75`, future support, no parent persistence, and sufficient
  daughter separation.
- Hard-overlap bypass no longer accepts ordinary
  `clean_future_suspicious_rod_blocker` cases. The only remaining raw-rod
  overlap bypass is the narrow `high_shape_raw_rod_signal_overlap` path.
- The two-frame rod-tip continuation overlap bypass now requires
  `aInB <= 0.38` and `bInA <= 0.45`, so a proposal with one daughter fully
  contained in a third cell is rejected.

Validation:

- Build passed with `cmake --build build -j 4`.
- CUDA was used in the focused runs (`selected_device=cuda`).
- In `run_197`, the intended raw-rod split still passed:
  `cell_0110000` accepted with
  `reason=high_shape_raw_rod_signal_overlap`.
- Shared-third / borrowed-neighbor routes were rejected by hard overlap:
  - `cell_110100`: overlap with `cell_100110` / `cell_1101000`,
    `aInB=0.369781`, `bInA=0.498287`.
  - `cell_1010010`: overlap with `cell_11101001` / `cell_10100100`,
    `aInB=0.295725`, `bInA=0.658254`.
  - `cell_01101111`: overlap with `cell_2011001` / `cell_011011110`,
    `aInB=0.627367`, `bInA=0.58187`.
  - `cell_2001110`: overlap with `cell_21101` / `cell_20011101`,
    `aInB=0.458921`, `bInA=1`.
- `run_197` frame `140` summary:
  `split_attempts=19`, `split_accepted=4`, `final_cells=170`.
- GT-center validator for frame `140` stayed comparable to the previous focused
  run: `pred=170`, `gt=186`, `matched<=30=170`, `extra=0`, `mean=5.8361`,
  `max=29.8663`. This is a local center sanity check, not a full pass.

Conclusion:

- The specific shared-third-cell rod bypass is now treated as evidence against
  the split instead of evidence for it. The true high-shape raw-rod split still
  survives, while borrowed-neighbor rod proposals with large overlap or full
  containment are rejected.
- This focused frame-140 result still does not prove the whole pipeline. The
  known fresh-run regression around early frame `5`/`cell_11` remains the next
  thing to fix before another broader 0-to-200 validation.

## 2026-06-15 Continuation: Early Frame-5 Split-Daughter Move Gate

Observed regression:

- Fresh `run_195_from0_to200_raw_rod_signal_neartie` stopped at the first real
  early regression: frame `5` missed GT label `129` because `cell_11` stayed
  near `(213.156,230.181,128.739)` instead of moving to the bright daughter
  center near `(210.208,197.915,128.664)`.
- Comparing against the older good
  `run_059_from0_to200_axis_place_shape_gate` showed that the frame-4 split was
  not the root cause. Both runs initially produced the same immediate daughter
  near `(213,230,129)`. The difference was the next-frame rescue:
  `run_059` allowed the recent split daughter to move about `1.30` body units,
  while `run_195` rejected the same kind of move through
  `recent_split_daughter_move_gate` with `maxBodyUnits=0.65`.

Fix applied:

- The recent split-daughter move gate now distinguishes a well-localized bright
  daughter from a dim/background-like daughter body. If the current body signal
  is dim relative to the candidate probe, the daughter may move up to `1.35`
  body units with probe score up to `1.10`. Bright daughters keep the stricter
  `0.65` body-unit / `0.85` probe-score gate.

Validation:

- Build passed with `cmake --build build -j 4`.
- Fresh CUDA short run:
  `run_198_from0_to10_dim_daughter_move_gate`.
- Frame `5` now moves `cell_11` from `(213.156,230.181,128.739)` to
  `(210.208,197.915,128.664)` via `Signal Center Rescue`, matching the older
  good behavior.
- GT-center validator results for frames `0-10` were clean at threshold `30`:
  all frames had `missing=0` and `extra=0`.
  - Frame `5`: `pred=6`, `gt=6`, `matched<=30=6`, `max=8.8889`.
  - Frame `6`: `pred=6`, `gt=6`, `matched<=30=6`, `max=10.3224`.
  - Frame `7`: `pred=7`, `gt=7`, `matched<=30=7`, `max=11.1139`.

Conclusion:

- The early frame-5 regression introduced by the overly tight recent-daughter
  movement gate is fixed in a fresh 0-to-10 run.
- This is still a short-window result; the next step is a broader fresh run to
  confirm the frame-140 shared-third fix and the early-frame movement fix do not
  conflict with mid-run behavior.

## 2026-06-15 Continuation: Frame-140 Third-Cell Future-Claim Gate

Observation:

- Visual inspection of frame `140` showed a stronger failure pattern than a
  simple suspicious rod: two wrong rod-like proposals can borrow/include the
  same third cell, so the future-center evidence looks valid even though the
  proposed daughter is actually explained by an existing cell.
- In the previous focused run, the bad proposals were mostly caught late by
  hard overlap. That was correct as a last defense, but too late for proposal
  reasoning.

Fix applied:

- The proposal claim gate now records the blocker distances instead of keeping
  only a text label.
- A PCA bridge proposal is rejected with
  `pca_bridge_third_cell_future_claim_gate` when it uses an aligned-pair
  fallback, moves farther than the clean snap limit, and one daughter is closer
  to a third cell body than to a valid daughter region.
- A rod-tip proposal is rejected with
  `rod_tip_third_cell_future_claim_gate` when clean future support is actually
  a third-cell core/body claim, especially with poor parent-distance balance.
- The stricter gate is not applied to the intentional high-shape raw signal
  split path.

Validation:

- Build passed with `cmake --build build -j 4`.
- Focused CUDA diagnostic run:
  `run_200_resume140_to142_third_cell_claim_gate`, resumed from
  `run_164_resume139_to200_current_locked_rescue`.
- Key frame-140 results observed before stopping the diagnostic run:
  - `cell_2001110` rejected early as
    `rod_tip_third_cell_future_claim_gate`; blocker was `cell_21101`, with
    `otherDist=2.091534` and `otherRadius=26.293301`.
  - `cell_1010010` rejected early as
    `pca_bridge_third_cell_future_claim_gate`; blocker was `cell_11101000`,
    with `otherDist=38.696682` and `otherRadius=31.675600`.
  - `cell_01101111` rejected early as
    `pca_bridge_third_cell_future_claim_gate`; blocker was `cell_2011001`,
    with `otherDist=10.676574` and `otherRadius=23.178900`.
  - The intended direct raw-rod split `cell_0110000` still reached split
    construction and was accepted through
    `high_shape_raw_rod_signal_overlap`.

Note:

- `run_200` was stopped after the frame-140 gate evidence was confirmed, before
  a completed checkpoint was written. It should not be treated as a pass run.
  The next broad validation should be redirected to `stdout.log` and should run
  until the next concrete error.

## 2026-06-15 Continuation: Frame-17 Low-Brightness Bio Near-Miss Gate

Observation:

- Fresh run `run_199_from0_to200_rod_overlap_and_dim_move_gate` failed first at
  frame `17` with two false extras from `cell_10` and `cell_11`; GT had no
  missing centers.
- Both false splits were PCA bridge proposals with weak current/future
  brightness and daughter separation just under the biological separation
  threshold. They were accepted through the soft near-miss path.
- The known true frame-4 split remained different: it had strong raw bridge
  geometry and did not depend on the same low-brightness near-miss route.

Fix applied:

- Added `pca_bridge_low_brightness_bio_near_miss_gate` for PCA bridge
  proposals that are below biological separation, use a current bridge gap,
  have dim matched future centers, poor parent-distance balance, and a
  not-strongly-elongated parent.
- The gate does not apply to high-shape raw PCA bridge near-miss proposals, so
  the intentional bridge-cut path remains available.

Validation:

- Build passed with `cmake --build build -j 4`.
- CUDA fresh run:
  `run_201_from0_to24_low_brightness_bio_near_miss_gate`.
- `run_201` validated clean from frames `0` through `23`.
- At frame `16`, the old false route was blocked:
  - `cell_10` rejected as `pca_bridge_low_brightness_bio_near_miss_gate`.
  - `cell_11` rejected by the existing
    `unclean_future_aligned_pair_balance_gate`.
- Frame `17` now validates clean:
  `pred=8 gt=8 matched<=30=8 missing=0 extra=0`.
- Frame `18` and `19` also validate clean:
  `frame=18 pred=10 gt=10`, `frame=19 pred=12 gt=12`.

Boundary note:

- `run_201` showed a missing frame-24 center only because the run ended at
  frame `24`, making the future window unavailable for a real `cell_20` split.
- Boundary CUDA run `run_202_resume24_to26_boundary_check`, resumed from
  `run_201` frame `23`, gave frame `24` access to future frames and accepted
  the split into `cell_200` and `cell_201`.
- `run_202` validates clean for frames `24` through `26`:
  `frame=24 pred=13 gt=13`, `frame=25 pred=14 gt=14`,
  `frame=26 pred=14 gt=14`.

Conclusion:

- The frame-17 false extras are fixed in a fresh early-window run without
  breaking the true early splits.
- Frame-24 must not be judged from a run that ends at frame 24, because this
  pipeline intentionally requires future support for that split.
- Next validation should be a longer fresh 0-to-n run with enough future
  margin, then a resume/mid-run check around frame `140` to verify the
  third-cell future-claim gate in a completed checkpoint.

## 2026-06-15 Continuation: Signal-Center Stability Gate Cross-Check

Observation:

- Fresh CUDA run `run_203_from0_to60_low_brightness_and_third_claim_gates`
  reached frame `34` with one extra split from `cell_011`.
- Older notes had already classified this frame-34 pattern as a likely GT
  timing caveat: the two predicted daughters were nearest to one GT center at
  frame `34`, then resolved into two GT centers at frame `35`.
- A provisional stability gate was added for sparse, moderate-shape
  `signal_center_split` proposals: if the current frame is only moderately
  separated and the immediate future support is dim/far, wait for clearer
  evidence instead of accepting only because offset-2 support is strong.

Regression found:

- Fresh CUDA cross-check `run_205_from0_to60_signal_center_stability_crosscheck`
  showed that the first version of the gate was too strict.
- Frame `7` regressed to `pred=6 gt=7`, reproducing the old one-frame-late
  `cell_2` split.
- The rejection was:
  `sparse_signal_center_moderate_low_separation_gate`.
- Key evidence for the true frame-7 split:
  - `parentShape=2.14537`
  - `sepRatio=1.15759`
  - `parentDistanceBalance=0.75087`
  - `futureBoth=2`, `futureMissing=0`, `parentPersists=0`
  - `futureBestMinBrightness=0.114648`
  - `immediateFutureBrightness=0.114648`
  - `immediateFuturePairDistance=36.9157`
- This means the old good early split had bright immediate support and should
  not be blocked by the frame-34 delayed-support rule.

Fix applied:

- The immediate-future stability check now allows the immediate matched pair
  distance to be up to `1.00 * daughterSep` instead of `0.75 * daughterSep`,
  while still requiring immediate daughter support and brightness at least
  `0.06`.
- This keeps the frame-34 delayed-support behavior targeted: dim/far immediate
  support can still wait one frame, but bright immediate support in a sparse
  early split remains valid.

Validation:

- Build passed with `cmake --build build -j 4`.
- CUDA focused run:
  `run_206_from0_to12_sparse_stability_ratio_fix`.
- Frame `7` now logs:
  `CellUniverse2 Split Proposal Sparse Rescue` for `cell_2`, followed by
  `Split Accepted`.
- GT-center validator at threshold `30` is clean for frames `0-12`:
  - frame `7`: `pred=7 gt=7 matched<=30=7 missing=0 extra=0 max=11.1139`
  - frame `11`: `pred=8 gt=8 matched<=30=8 missing=0 extra=0 max=12.5928`
  - frame `12`: `pred=8 gt=8 matched<=30=8 missing=0 extra=0 max=15.7799`
- Early weak false proposals are still rejected:
  - frame `3` weak `cell_0` signal-center proposal remains rejected.
  - frame `4` weak `cell_1` signal-center proposal remains rejected.

Current rule summary:

- Sparse/moderate signal-center splits should be accepted when they have
  two-frame support plus bright immediate support near the proposed daughter
  pair.
- Sparse/moderate signal-center splits should wait when the current evidence is
  only delayed/future-offset-2 support and the immediate future is dim or far.
- This is a source-specific gate for `signal_center_split`; it should not be
  generalized to PCA bridge, rod-tip, or late crowded third-cell cases without
  separate evidence.

Cleanup note:

- Old bulky `tiff/` and `perturb_debug/` folders were removed from historical
  runs in the shared debug folder, excluding the current and previous evidence
  runs (`run_205` and `run_206`).
- Run folders, `cells.csv`, `stdout.log`, `candidate_graph`, and `checkpoints`
  were preserved so historical decisions and resume points remain available.


## 2026-06-15 Continuation: Frame98 Future-Signal Near-Threshold Rescue

Evidence:

- Previous comparison run `run_233_resume96_to102_neighbor_pca_guard` was clean
  through frames `96` and `97`, then missed GT label `550` at frame `98`.
- The missed object came from `cell_20111`. The proposal was biologically
  plausible and future-supported:
  - `source=signal_center_split`
  - `futureBoth=2`, `futureMissing=0`, `parentPersists=0`
  - daughter seeds near `(355.809,386.582,143.853)` and
    `(317.37,387.177,142.125)`.
- It was not rejected by the biological split checks. It was narrowly rejected
  by the cost gate:
  - `imageDiff=-24294.4`
  - `overlapDiff=13937.2`
  - `costDiff=-10357.2`
  - threshold `-11016.4`.
- The existing source-specific near-threshold rescue almost matched, but the
  parent balance and axis-length guards were slightly too strict for this clean
  future-supported case.

Fix applied:

- Added config copy
  `config/config_celluniverse2_run234_frame98_future_signal_near_threshold.yaml`.
- Changed only the source-specific future-signal near-threshold rescue:
  - `split_future_signal_near_threshold_min_parent_balance: 0.35`
  - `split_future_signal_near_threshold_min_axis_length_scale: 1.20`
- Generic split cost, density, and false-daughter guards were not loosened.
- Added a conservative bounding-sphere prefilter before expensive ellipsoid
  voxel-overlap checks in `Frame.cpp`. If bounding spheres cannot touch, the
  ellipsoids cannot overlap, so this is intended as a performance-only skip.

Validation:

- Build passed with `cmake --build build --target celluniverse -j 4`.
- CUDA focused run `run_237_resume98_to100_overlap_prefilter` loaded the clean
  frame-97 checkpoint from `run_233` and completed frames `98-100`.
- Frame `98` accepted `cell_20111` through the future-window cost gate with
  `signalNearThresholdRescue=1` and also accepted the paired split for
  `cell_20011`.
- GT-center validation at threshold `30`:
  - frame `98`: `pred=92 gt=92 matched<=30=92 missing=0 extra=0 max=29.0196`
  - frame `99`: `pred=93 gt=93 matched<=30=93 missing=0 extra=0 max=24.8146`
  - frame `100`: `pred=93 gt=93 matched<=30=93 missing=0 extra=0 max=20.3315`

Next:

- Resume from `run_237` frame `100` to push toward the next concrete failure.
- Keep `run_233` as previous comparison evidence and `run_237` as the current
  good checkpoint source for the frame98 fix.
- Do not claim global success until a broad 0-to-n smoke run proves the same
  config did not regress earlier frames.

## 2026-06-15 Continuation: Compiler Optimization Scope and Frame106 Post-PCA Bridge Retry

Compiler/build status:

- The GCC15 optimizer workaround is now scoped to the LibTorch-backed
  `N2V2Preprocessor.cpp` translation unit only.
- `CellUniverse.cpp` and `Frame.cpp` compile with Release `-O3 -DNDEBUG`,
  `-march=native`, and OpenMP. The N2V2 file compiles with the safer
  `-O1 -fno-tree-fre -fno-dce -fno-cprop-registers` override to avoid the
  compiler optimizer crash without slowing the hot tracking path.
- Build passed with `cmake --build build --target celluniverse -j 4`.
- Regenerated `build/compile_commands.json` confirms CUDA include paths and
  `sm_86` CUDA detection remained active during configure/build.

Focused frame-105/106 evidence:

- Current focused config copy:
  `config/config_celluniverse2_run252_frame106_post_pca_jump.yaml`.
- The latest narrow run is
  `run_252_resume105_to107_post_pca_jump`, resumed from the clean frame-104
  checkpoint in `run_243_resume103_to105_clean_bridge_no_merge`.
- CUDA was selected by N2V2 in the run log.
- Frame `105` did not split `cell_3100` early; the weak PCA bridge rejection was
  kept, which is acceptable because the daughter was not fully supported yet.
- Frame `106` exposed the real rod after final PCA. Lowering only
  `post_pca_bridge_severe_new_rod_min_elongation_jump` from `1.35` to `1.30`
  allowed post-PCA bridge retry to fire:
  - `snapshotElong=1.76715`
  - `currentElong=2.32945`
  - `elongationJump=1.3182`
  - `reason=post_pca_bridge_cut`
- The split was accepted for `cell_3100` with `costDiff=-9010.37` and daughters
  near GT labels `715` and `712`.

Validator result for the focused run:

- frame `105`: pred `98`, GT `98`, missing `0`, extra `0`.
- frame `106`: pred `100`, GT `100`, missing `0`, extra `0`.
- frame `107`: pred `100`, GT `100`, missing `0`, extra `0`.

Next validation:

- This focused success is not a global proof. The next validation must be a
  fresh `0-to-n` CUDA run using the same config copy so we can see whether the
  frame-106 post-PCA threshold conflicts with the earlier validated windows.
- If it passes more than 50 continuous frames, record it in
  `docs/celluniverse2_long_run_config_records.md` with command, config identity,
  and validator caveats.

## 2026-06-15 Continuation: Frame34 False Split from Density Waiver

Evidence from fresh run `run_253_from0_to120_post_pca_jump_crosscheck`:

- Fresh CUDA run from frame `0` using
  `config/config_celluniverse2_run252_frame106_post_pca_jump.yaml` was stopped
  at the first concrete validator error.
- Frames `0-33` validated clean against GT centers.
- Frame `34` failed as `pred=16`, `gt=15`, with missing GT label `193` and
  false extra daughters `cell_1100` / `cell_1101`.
- At frame `33`, unsplit `cell_110` matched GT `193` at distance `7.81`, so the
  split itself caused the failure.
- The bad frame-34 split was accepted through the locked future-supported PCA
  bridge density waiver even though both daughters were effectively background:
  - `d1Mean=0.00101813`
  - `d2Mean=0.00306833`
  - `hardThreshold=0.025`
  - `lockedDensityFloor=0.04`
  - `belowObviousBackgroundFloor=1`
- This matched the intended rule that future support may soften near-threshold
  bridge evidence, but must not waive an obviously empty daughter density floor.

Fix applied:

- In `Frame.cpp`, `lockedCleanFuturePcaBridgeDensityWaived` now requires
  `!belowObviousBackgroundFloor`.
- The change restores the hard floor for false daughter density while keeping
  the softer future-supported density path available for non-empty near-miss
  cases.
- Build passed with `cmake --build build --target celluniverse -j 4`.

Validation:

- Fresh CUDA run `run_254_from0_to36_density_floor_restore` completed normally
  from frame `0` to `36` with `selected_device=cuda` and `Time elapsed: 1138.27 seconds`.
- Validator result: `first_bad=none checked_through=036`, worst max assignment
  distance `28.004` at frame `36`.
- Frame `34` now validates clean: `pred=15`, `gt=15`, `missing=0`, `extra=0`,
  max assignment distance `13.746`.
- The bad `cell_110` split is now rejected as `daughter_density_brightness`,
  and unsplit `cell_110` remains near GT `193`.

Next:

- Run a longer fresh `0-to-n` validation, at least through the old frame `58/59`
  region and preferably toward `120`, because this 0-36 result fixes the
  immediate regression but is not a qualifying >50-frame or 0-200 proof.

## 2026-06-16 Continuation: Frame104 Regression from Global Force-Rod Cost Tuning

Evidence:

- `run_301_resume101_to107_force_rod_fraction_027` used
  `config/config_celluniverse2_run301_force_rod_cost_fraction_027.yaml`, which
  lowered only `post_pca_force_rod_split_min_cost_improvement_fraction` from
  `0.03` to `0.027`.
- Validator result showed the global threshold was too broad:
  - frames `101-103`: clean against GT.
  - frame `104`: `pred=98`, `GT=97`, extra unmatched prediction
    `cell_110100`.
- The failure was not a direct accepted post-PCA force-rod split. The lowered
  threshold altered the local state enough that frame `104` accepted a
  `cell_11011` split that the previous clean run did not accept.
- Current-code baseline `run_302_resume101_to104_current_code_run297` used the
  previous `0.03` config,
  `config/config_celluniverse2_run297_force_rod_cost_fraction.yaml`, from the
  same frame-100 checkpoint. It validated clean:
  - frame `101`: pred `95`, GT `95`, missing `0`, extra `0`.
  - frame `102`: pred `96`, GT `96`, missing `0`, extra `0`.
  - frame `103`: pred `97`, GT `97`, missing `0`, extra `0`.
  - frame `104`: pred `97`, GT `97`, missing `0`, extra `0`.

Fix applied:

- Reverted the experiment direction away from lowering the global
  force-rod cost fraction.

## 2026-06-16 Continuation: Rod-Tip Daughter Distance Clamp and Axis Gate Alignment

Evidence:

- `run_320_from0_to20_rodtip_distance_clamp_4core` added the rod-tip daughter
  distance clamp and used CUDA with four cores. It fixed the PCA-refit collapse
  mode at frame `18`, but still failed count validation there:
  - frame `18`: pred `9`, GT `10`, missing GT `66`.
  - `cell_10` produced a clean future-supported rod-tip proposal, then the
    clamp preserved daughter spacing:
    `seedDistance=63.088`, `refitDistance=38.9117`,
    `targetDistance=41.7847`, `driftScale=0.865617`.
  - The remaining rejection was the global cost gate, not the daughter-distance
    collapse: `totalDiff=55211.6`, `imageDiff=44923.5`,
    `overlapDiff=10288.1`.
- The cost-rescue path was still requiring
  `split_future_rod_tip_primary_min_axis_length_scale: 1.20`, while the
  configured biological daughter separation target was
  `bio_min_daughter_separation_parent_fraction: 1.10`.

Fix applied:

- In `Frame.cpp`, rod-tip future-supported daughter PCA refit now clamps the
  effective post-refit drift if the daughter separation would fall below the
  smaller of the original seed distance and the configured biological minimum.
  This keeps the wide-enough daughter placement while still allowing PCA shape
  and radius fitting.
- New config copy:
  `config/config_celluniverse2_run321_rodtip_axis_scale_110.yaml`.
- The only config change from the distance-clamp run was aligning
  `split_future_rod_tip_primary_min_axis_length_scale` to `1.10`.

Validation:

- Fresh CUDA four-core run
  `run_321_from0_to20_rodtip_axis_scale_110_4core` completed normally.
- Validator result for frames `0-20`: `STATUS OK`.
- Frame `18` now validates with pred `10`, GT `10`, max assignment distance
  `16.28`.
- Frame `18` accepted `cell_10` through
  `futureRodTipPrimaryRescue=1`; `cell_101` matched GT `66` at distance
  about `5.3`, and `cell_100` matched GT `3` at distance about `16.3`.

Next:

- Expand the same config to a fresh `0-60` CUDA smoke. Treat `run_321` as
  focused evidence for the early frame-18 regression only, not as a promoted
  >50-frame or 0-200 record.
- Added YAML-backed post-PCA force-rod near-miss controls:
  - `post_pca_force_rod_split_bright_near_miss_rescue_enabled`
  - `post_pca_force_rod_split_bright_near_miss_min_fraction`
  - `post_pca_force_rod_split_bright_near_miss_min_seed_brightness`
- Created `config/config_celluniverse2_run303_force_rod_bright_nearmiss.yaml`
  from the `run297` config:
  - kept `post_pca_force_rod_split_min_cost_improvement_fraction: 0.03`.
  - enabled the near-miss rescue with `min_fraction: 0.0275`.
  - required both daughter seeds to be at least `0.60` bright.
- Build passed with
  `cmake --build build --target celluniverse --parallel 4 --config Release`.

Validation:

- CUDA run `run_303_resume101_to107_force_rod_bright_nearmiss` resumed from
  `run_296_resume89_to110_bridge_gate_tune/checkpoints/frame_100.txt`.
- Validator result:
  - frame `101`: pred `95`, GT `95`, missing `0`, extra `0`.
  - frame `102`: pred `96`, GT `96`, missing `0`, extra `0`.
  - frame `103`: pred `97`, GT `97`, missing `0`, extra `0`.
  - frame `104`: pred `97`, GT `97`, missing `0`, extra `0`.
  - frame `105`: pred `98`, GT `98`, missing `0`, extra `0`.
  - frame `106`: pred `100`, GT `100`, missing `0`, extra `0`.
  - frame `107`: pred `100`, GT `100`, missing `0`, extra `0`.
- Important caveat: frame `106` passed via `signal_center_split` for
  `cell_2111`, not by triggering the new post-PCA force-rod near-miss rescue.
  The new rescue did not fire in this 101-107 window; it remains a scoped
  fallback for future high-brightness near-miss force-rod cases.

Next:

- Do not claim the new near-miss rescue solved the 101-107 window by itself.
  The verified claim is only that the current code plus the `run303` config is
  clean for frames `101-107` and preserves the frame-104 neighborhood.
- Before expanding toward `200`, run at least one sensitive earlier window where
  post-PCA force-rod cleanup can fire, then expand from a stable checkpoint or
  a fresh `0-to-n` run according to the current failure frontier.

## 2026-06-16: split lock-vs-acceptance brightness split

Context:

- Repeated tuning around frame `18`, frame `34`, and frame `85-86` kept showing
  the same coupling problem: current-frame daughter brightness was being used
  for two different jobs at once.
  - Job 1: decide whether a clean future-supported daughter may keep its
    snapped seed position during PCA refit.
  - Job 2: decide whether the overall split proposal is trustworthy enough to
    survive later density, overlap, and cost gates.
- The two jobs do not want the same threshold. Some real splits are dim in the
- current frame but still have strong geometry and future support, while false
  daughters in background must still fail hard.

Fix applied:

- Added separate YAML-backed lock-only brightness floors in `ConfigTypes.hpp`
  and `Frame.cpp`:
  - `split_current_locked_bridge_lock_min_brightness`
  - `split_immediate_pca_continuation_lock_min_brightness`
  - `split_one_frame_aligned_pca_continuation_lock_min_brightness`
  - `split_exact_future_center_bridge_lock_min_brightness`
- The new lock-only fields are used only by the split-daughter refit position
  lock paths.
- The existing acceptance-style fields remain responsible for later rescue,
  density, overlap, and cost decisions.
- In `config/config_celluniverse2_tuning.yaml`, the lock-only floors were set
  low (`0.02`) while the acceptance-style floors were restored to the stronger
  values that previously protected later crowded windows.

Validation:

- Build passed with `cmake --build build -j 4`.
- Fresh CUDA four-core run:
  `celluniverse2_goal_0_200_20260612/run_from0_22_tuning_20260616_081617`.
- Validator-backed early window:
  - frame `4`: pred `6`, GT `6`
  - frame `6`: pred `6`, GT `6`
  - frame `7`: pred `7`, GT `7`
  - frame `8`: pred `7`, GT `7`
  - frame `9`: pred `7`, GT `7`
  - frame `11`: pred `8`, GT `8`
  - frame `16`: pred `8`, GT `8`
  - frame `17`: pred `8`, GT `8`
  - frame `18`: pred `10`, GT `10`
  - frame `19`: pred `12`, GT `12`
  - frame `20`: pred `12`, GT `12`
  - frame `21`: pred `12`, GT `12`
  - frame `22`: pred `12`, GT `12`
- Important qualitative checks from the same fresh run:
  - Frame `16` showed several distinct rejection families in one window:
    `pca_bridge_low_brightness_bio_near_miss_gate`,
    `unclean_future_aligned_pair_balance_gate`, and
    `bio_separation_gate`.
  - Frame `18` accepted `cell_10` through `bridge_axis_place`, kept the
    axis-place seeds locked through PCA daughter refit, and validated clean.
  - Gamma split decay advanced from `1.45` down to `1.4275` by frame `18`
    after six accepted splits, without breaking the early GT-clean window.

Focused frontier note:

- Resume run
  `celluniverse2_goal_0_200_20260612/run_resume85_92_tuning_20260616_081619`
  did not reach frame `92`; the recovered folder contains frames `85-86` only.
- It is still useful evidence:
  - frame `85`: pred `59`, GT `60`, missing GT `148`
  - frame `86`: pred `65`, GT `67`, missing GT `148`, `434`, `566`, extra
    `cell_000111`
- This reinforces that the old frame-85/86 frontier is not a single
  brightness-floor bug anymore.

Working conclusions:

- The lock-vs-accept split is globally helpful. It preserved the validated
  early `0-22` window while keeping later acceptance logic stricter.
- The next adaptive-function candidates should not be "brightness only"
  thresholds. The run windows suggest three better environment-dependent
  families:
  - parent-distance-balance floors that depend on future support quality,
    current crowding, and parent rod shape;
  - low-brightness near-miss rescue/reject logic that depends on both future
    support strength and whether the proposal geometry is strong enough to
    survive later PCA/cost checks;
  - snap/drift tolerances that depend on local nearest-neighbor spacing and
    whether the daughter seeds already have stable future support.

## 2026-06-16 Cross-Window Tuning Notes

Collected experience from the early guardrail window, the old frame-58/64
bridge window, the frame-85/87 crowded frontier, and the later frame-101/106
recovery windows:

- Globally helpful patterns:
  - Splitting "lock" thresholds from "accept" thresholds is stable. The same
    low current-frame brightness can be safe for position locking while still
    being too weak for a global cost rescue.
  - Large daughter PCA drift is repeatedly a better failure predictor than raw
    brightness alone. When a proposal already has a clean supported seed, drift
    usually tells us whether refit is preserving a true split or inventing one.
  - Dense-gap bridge rejection is still important. The false cases that reopen
    after broad relaxations tend to share one of two signatures:
    1. dense bridge plus moderate/weak valley, or
    2. big daughter drift with only weak or delayed future support.

- Parameters that appear environment-dependent rather than globally fixed:
  - Parent-distance-balance thresholds:
    crowded late windows want softer balance when future support is clean,
    while false rod-tip placements still need a stronger balance floor if the
    snap is far or the neighborhood is noisy.
  - Seed snap / drift limits:
    the useful drift budget seems to shrink when nearest-neighbor spacing gets
    small, and can expand slightly when the local neighborhood is sparse and
    the future-supported centers are stable.
  - Low-brightness rescue logic:
    a dim daughter is more believable when the other daughter is clearly bright
    and the bridge geometry is clean; the same brightness should stay rejected
    when both daughters are effectively background or when the bridge is dense.

- Current frame-85/86 frontier interpretation:
  - The miss is no longer one single gate. The neighborhood mixes:
    - true late-daughter candidates that die before cost because of density or
      dense-flat checks, and
    - competing crowded placements that can still steal the region later.
  - Because of that, broad threshold lowering is more likely to trade one
    local miss for another local false split than to help globally.

- Guidance for the next change:
  - Prefer scoped logic that ties rescue/lock behavior to future support
    quality, current bridge geometry, and drift, instead of lowering global
    brightness or cost thresholds.
  - If a new adaptive function is added, base it on measured local conditions
    such as nearest-neighbor spacing, current/future support counts, and bridge
    density/valley evidence rather than frame index or one-off run identity.

## 2026-06-16 Continuation: Instrumented 85-88 Frontier Window

Instrumented run:

`/run/media/blue-lobster/disk3/celluniverse_output/outputs_fluo/output_ubuntu_fluo_resume85_85-88_celluniverse2_tuning_instrument_20260616_085451`

Resume source:

- Loaded frame `84` checkpoint from
  `run_059_from0_to200_axis_place_shape_gate`, so this window is useful for
  frontier diagnosis but is not a replacement for fresh `0-to-n` validation.

What was added for this scout:

- Future-window logging now records the actual matched pair positions:
  `bestMatchedD1`, `bestMatchedD2`, plus the best matched offset.
- This was instrumentation only. No split-accept behavior was intentionally
  changed by the patch.

GT validation for the completed window (`matched<=35`):

- frame `85`: pred `59`, GT `60`, missing GT `148`
- frame `86`: pred `65`, GT `67`, missing GT `148`, `434`, `566`, extra
  `cell_001010`
- frame `87`: pred `69`, GT `77`, missing GT `13`, `44`, `84`, `140`, `148`,
  `292`, `395`, `566`
- frame `88`: pred `69`, GT `82`, missing GT `13`, `28`, `44`, `59`, `84`,
  `122`, `140`, `148`, `234`, `292`, `331`, `395`, `566`

Important interpretation:

- Frame `85` is still the first true miss. The older frontier location is
  real.
- The later `87-88` collapse is not a separate random failure family. It is
  what happens after the crowded `148` neighborhood is missed and the next
  frame no longer contains clean immediate-future daughter support for nearby
  rod-like parents.
- Exact cell IDs drift across run lineages. In this resume window the crowded
  family shows up as `cell_11100` / `cell_11101`; in other resumes the same
  biological region appeared under nearby descendant names. The geometry and
  support pattern are more trustworthy than the literal cell name.

New evidence from the instrumented future-pair logs:

- Clean future-supported splits remain very stable when their snapped seed pair
  is already good. Examples in frame `85` include `cell_00100`, `cell_00111`,
  and `cell_01100`:
  - both daughters supported in the immediate future,
  - matched future pair recorded explicitly,
  - daughter refine drift stayed at `0`,
  - total cost still improved after PCA/refit.
- Dense fake bridges still advertise themselves clearly even when some future
  support exists. Examples include `cell_01000` and `cell_000111`, which kept
  high gap density / weak valley signatures and were correctly rejected by the
  dense-flat family.
- The main late-window cliff is the parent-distance-balance gate, not raw
  brightness by itself.
  - Frame `87` still shows future-supported PCA bridge proposals that are
    biologically plausible but die at `parent_distance_balance_gate`, such as
    `cell_10010`, `cell_101001`, and later `cell_20101`.
  - By frame `88`, once the previous misses have removed clean future support,
    many rod-like parents fall back to `parent_distance_balance_gate` with
    `futureBoth=0` and `parentDistBalance=0`, causing a cascade of missed
    splits.

What this says about globally helpful tuning:

- Keep these as hard or near-hard guards:
  - dense-flat bridge rejection,
  - hard daughter density floor below the tolerance band,
  - "do not place in the air" style current-frame image-gain requirements for
    far aligned-pair fallbacks.
- Treat these as adaptive-function candidates instead of fixed global values:
  - parent-distance-balance floor,
  - daughter seed drift limit,
  - snap-distance trust limit for future-aligned pair reuse,
  - soft brightness rescue band above the hard density floor.

Recommended adaptive variables and the environment signal they should depend on:

- Parent-distance-balance floor:
  - Depend on local evidence, not only one fixed threshold.
  - Inputs that now look meaningful:
    - `windowImmediateBothDaughtersSupported`
    - `windowBothDaughtersSupported`
    - `windowMissingDaughterCount`
    - `parentShapeElongation`
    - local crowding, preferably nearest-neighbor spacing or number of live
      centers in the parent bbox / motion-expanded bbox
    - whether the seed came from exact current centers, immediate future
      centers, or aligned-pair fallback
  - Practical rule from this window:
    - allow lower balance only when support is immediate and clean;
    - keep higher balance when support is delayed, asymmetric, or far-snapped.

- Drift / snap limits:
  - Existing code already scales some limits by `srcMaxR` and future-frame
    offset. That is useful, but this window suggests another missing signal:
    local spacing.
  - In crowded neighborhoods, the acceptable drift should shrink as nearest
    neighbor distance shrinks, because a small PCA walk can steal a sibling or
    neighbor target.
  - In sparse neighborhoods with clean future support, the drift budget can be
    slightly larger without creating false splits.

- Brightness rescue band:
  - Absolute brightness alone is not a stable global decision variable because
    gamma decay, local background, and noise all change what "dim" means.
  - A better variable is daughter brightness relative to local background and
    relative to the sibling daughter, then conditioned on bridge geometry and
    future support quality.
  - The hard daughter-density floor should remain hard below tolerance. The
    soft band should only activate when at least one daughter is clearly real
    and the split geometry is already clean.

- Crowding signal:
  - The current general clean-future rescue already has a global crowding
    relaxation using saved non-trash cell count.
  - This run suggests a more useful signal is local crowding, not just global
    embryo count. The frame-`85` family is crowded because of its immediate
    neighborhood, not because the whole embryo has crossed one universal count.

Current best summary:

- Broad threshold lowering is still the wrong move.
- The strongest globally reusable direction is:
  1. keep dense / fake / air-placement rejections hard,
  2. make the parent-balance and drift tolerances depend on local crowding plus
     immediate future support quality,
  3. keep exact or immediate future supported seed pairs locked through PCA
     whenever they already satisfy the stricter geometry and density evidence.

## 2026-06-16 Continuation: crowded-window cross-run synthesis

Fresh early/mid guardrail:

- Fresh run
  `/run/media/blue-lobster/disk3/celluniverse_output/outputs_fluo/output_ubuntu_fluo_fresh_0-90_celluniverse2_tuning_20260616_093027`
  still validates clean through frame `84` and hits the same old first true
  miss at frame `85`.
- Newly checked clean frames in this branch now include `61`, `68`, `69`, and
  `80` under the accepted `matched<=35` rule.
- Practical meaning: the stronger daughter-halfspace clamp
  (`split_daughter_refit_halfspace_min_fraction=0.18`) still looks globally
  safe for the sparse-to-moderate window. It is not the cause of the later
  crowded failures.

Later crowded reruns compared here:

- Before dense-scene rod-tip relaxation:
  `/run/media/blue-lobster/disk3/celluniverse_output/outputs_fluo/output_ubuntu_fluo_resume120_120-160_celluniverse2_tuning_20260616_095559`
- After dense-scene rod-tip relaxation:
  `/run/media/blue-lobster/disk3/celluniverse_output/outputs_fluo/output_ubuntu_fluo_resume120_120-160_celluniverse2_tuning_20260616_100647`

What the new crowded relaxation proved:

- The new `rod_tip_future_aligned_crowded_*` YAML family is not placebo.
- It moves the persistent crowded trio
  (`cell_001011`, `cell_101000`, `cell_0001101`) past the old
  `future_aligned_rod_tip_flatness_gate` failure.
- It does this without reopening the earlier fresh frontier, which is exactly
  the desired shape for a dense-scene-only function.

What the remaining failures actually are now:

- `cell_001011`
  - frame `121`: clears the old flatness gate, gets two-frame future support,
    then dies at `rod_tip_third_cell_future_claim_gate`.
  - frame `122`: the rod-tip proposal gets through claim and future support,
    but fails only at `daughter_midpoint_parent_drift` with a very narrow miss:
    midpoint distance `41.5888` vs limit `40.369`.
  - frame `123`: it no longer needs the rod-tip fallback. A
    `pca_bridge_cut` proposal becomes acceptable with one/two-frame future
    support and good balance, but the actual split still dies later at hard
    overlap against `cell_000000`.
  - Interpretation:
    - this family does self-heal past the original crowded flatness/claim
      front door,
    - but it still needs downstream geometry/overlap compatibility before it
      becomes a true accepted split.
- `cell_101000`
  - frame `121`: now reaches clean two-frame future support, but the proposed
    pair is still rejected by `rod_tip_third_cell_future_claim_gate`.
  - frame `122`: same family again, blocker distance `5.74` inside a blocker
    radius about `29.87`.
  - frame `123`: same family again, with an even farther future-aligned snap
    (`centerSnapMaxSeedDistance=103.517`) and continued third-cell claim.
  - Interpretation: this is not a harmless one-frame delay. The current
    future-supported pair is still borrowing occupied territory, so the claim
    gate remains biologically useful here.
- `cell_0001101`
  - frame `122`: signal-center split becomes acceptable with clean future
    support, but loses local center arbitration to `cell_001111`.
  - frame `123`: the same parent becomes the winning claimant instead:
    `signal_center_split` is acceptable, bypasses the continuation claim, and
    then blocks `cell_0001100` in the local claim step.
  - frame `123` downstream: once attempted, the split still dies at hard
    overlap against `cell_0011100`.
  - Interpretation: this family is a real example of the local-claim system
    self-correcting one frame later without a new global threshold change, but
    acceptance still depends on later overlap compatibility.

What this says about globally helpful tuning:

- Good dense-scene function candidate:
  - `rod_tip_future_aligned_crowded_*`
  - Inputs that now look genuinely meaningful:
    - live non-trash cell count,
    - parent shape,
    - parent mid/short ratio,
    - future-pair snap distance in parent-radius units.
- Good near-miss rescue candidate:
  - `daughter_midpoint_parent_drift` when:
    - the split already has clean future support,
    - daughter seed drift stayed locked at `0`,
    - the miss is only a small fraction over the midpoint limit.
  - `cell_001011` frame `122` is the concrete evidence point for this.
- Important non-equivalence discovered at frame `123`:
  - "proposal becomes acceptable" is not the same as "split should be globally
    promoted."
  - `cell_001011` and `cell_0001101` now show that the crowded front door can
    be fixed while hard-overlap rejections still protect the scene later in the
    accept path.
- Do not broadly weaken these yet:
  - `rod_tip_third_cell_future_claim_gate`
  - hard-overlap rejection for future-supported but geometry-conflicting
    daughters
  - generic local signal-center claim arbitration
  - Reason: `cell_101000` still shows the "borrowed third-cell core/body"
    pattern, while `cell_0001101` already demonstrates that some claim
    conflicts self-resolve on the next frame and then still need a later
    overlap check.

Current synthesis:

- The crowded-window family is no longer "flatness too strict."
- It has split into three more specific behaviors:
  1. dense-scene rod-tip flatness that really should relax with crowding,
  2. midpoint drift near-misses that can be rescued when future support is
     clean and the daughters stayed locked,
  3. true third-cell borrowing that should remain rejected.
- That is a much better foundation for the next tuning pass than another
  global threshold drop.

## 2026-06-16 Continuation: frame-85 phantom rescue removal and crowded signal-center evidence

Focused CUDA rerun:

- Run:
  `/run/media/blue-lobster/disk3/celluniverse_output/outputs_fluo/output_ubuntu_fluo_resume85_85-90_celluniverse2_tuning_20260616_121703`
- Checkpoints reached in this pass: `85`, `86`, and `87`.

What changed in code/config:

- Added a new YAML-backed gate:
  `split_density_cost_backed_clean_future_min_future_brightness: 0.08`
- Wired it into the cost-backed clean-future density waiver and the matching
  near-threshold cost rescue.
- Purpose: future-supported density rescue may only fire when the future pair
  itself is bright enough to count as real evidence, not when it is carried by
  a near-background singleton center.

What this fixed immediately:

- Frame `85`, `cell_11110` no longer takes the bad phantom rescue path.
- Before the change, the branch was accepted through
  `cost_backed_clean_future_pca_bridge_density` even though one daughter stayed
  locked at `(221.5,227.5,189)` with effectively empty current-frame support
  and future brightness only about `0.028`.
- After the change, the same proposal is rejected at the hard daughter-density
  gate:
  `reason=daughter_density_brightness`.
- Validator result for frame `85` improved from "missing `148` plus one extra"
  to "missing `148`, extra `0`". That is the desired direction: remove the
  false daughter without loosening the hard floor.

What the next true blocker now is:

- GT `148` is still missing at frame `85`, but the active miss is no longer
  `cell_11110`.
- The remaining family is `cell_11111`, which stays unsplit and becomes the
  nearest unmatched parent for GT `148` at frames `85`, `86`, and `87`.
- This is not primarily another cost-threshold problem. The dominant failure
  is localization:
  - at frame `86`, `cell_11111` attempted a `pca_bridge_cut`, but the proposed
    pair localized one daughter onto the wrong axis, then the density gate
    correctly rejected it;
  - at the same frame, the signal-center path saw many centers
    (`centersInside=15`, `candidatePairs=105`) but rejected almost all of them
    as neighbor-owned (`rejectedNeighborOwned=104`) before the weaker PCA
    bridge fallback took over.

Why this matters globally:

- The current crowded-window frontier is no longer dominated by "future support
  present or absent." It is split into two separate quality questions:
  1. whether future support is bright/clean enough to justify rescuing a dim
     current-frame daughter, and
  2. whether the current-frame signal-center ownership logic is discarding the
     biologically correct pair before the rescue logic can use it.
- The new future-brightness gate solved question (1) for the `cell_11110`
  family.
- The remaining `cell_11111`, `cell_11100`, and related frame-`86/87` misses
  point at question (2): crowded local ownership arbitration, not generic
  cost-gate tuning.

Current cross-window lesson:

- "Future support" is not a single variable. For density rescue, the
  brightness quality of the matched future pair matters more than the raw fact
  that two future centers were found.
- In the crowded `85-87` window, a new environment-linked signal is now
  visible: the fraction of signal-center candidate pairs rejected as
  neighbor-owned. When that ratio dominates the rejection pool, the pipeline
  falls back to PCA-bridge placements that are more likely to drift onto the
  wrong axis.
- This suggests the next narrow fix should not be a broad ownership relaxation.
  A better direction is a future-supported signal-center override that only
  activates when:
  - most candidate pairs are rejected as neighbor-owned,
  - separation/midpoint geometry is otherwise clean,
  - future support is bright enough, and
  - the override remains local to the crowded ownership family.

## 2026-06-16 Continuation: crowded ownership-axis rescue is environment-linked but not globally tunable

Focused CUDA rerun used for this check:

- Run:
  `/run/media/blue-lobster/disk3/celluniverse_output/outputs_fluo/output_ubuntu_fluo_resume85_85-90_celluniverse2_tuning_20260616_130128`
- Resume source:
  `/run/media/blue-lobster/disk3/celluniverse_output/outputs_fluo/output_ubuntu_fluo_fresh_0-90_celluniverse2_tuning_20260616_093027`
- Checkpoints safely written before stopping: `85` and `86`.

Tested change:

- Temporarily relaxed
  `signal_center_neighbor_owned_future_rescue_axis_fraction`
  from `0.60` to `0.58`.
- Motivation: in the previous focused rerun, frame `86` `cell_11111` missed the
  crowded ownership rescue by only about `0.011` on the derived axis floor
  (`0.75 * 0.60 = 0.45`, `bestRejectedAxis about 0.43897`).

What happened:

- The hoped-for family did move slightly, but not enough:
  - frame `86`, `cell_11111` still died at `axis_alignment_gate`
  - `candidatePairs=120`
  - `rejectedNeighborOwned=119`
  - `bestRejectedAxis=0.42762`
- So the local crowded miss remains real, but the global axis-fraction change
  did not selectively fix it.

Validator comparison against the previous focused rerun:

- Previous rerun
  (`output_ubuntu_fluo_resume85_85-90_celluniverse2_tuning_20260616_124839`):
  - frame `85`: `pred=59 gt=60`, missing `1`, extra `0`
  - frame `86`: `pred=65 gt=67`, missing `3`, extra `1`
- Temporary relaxed-axis rerun
  (`output_ubuntu_fluo_resume85_85-90_celluniverse2_tuning_20260616_130128`):
  - frame `85`: `pred=59 gt=60`, missing `1`, extra `0`
  - frame `86`: `pred=65 gt=67`, missing `4`, extra `2`

What regressed:

- The relaxed axis fraction preserved the old frame-`85` result but made frame
  `86` strictly worse:
  - missing labels became `148`, `211`, `434`, and `566`
  - extra predictions became `cell_11011` and `cell_000111`
- This means the threshold is not a clean global knob. Relaxing it helps some
  ownership-dominated pairs compete, but it also admits additional wrong
  families in the same crowded frame.

Cross-window interpretation:

- Hard guards still look globally correct:
  - hard false-daughter density floor
  - third-cell claim protection
  - hard overlap rejection
- Environment-linked quantities still look real, but they need more than a
  single fixed scalar:
  - crowded rod-tip / flatness relaxation
  - crowded ownership rescue for signal-center pairs
  - future-supported drift tolerance
  - brightness rescue quality, but only relative to local background and future
    support quality

Updated rule from this experiment:

- The crowded ownership family should not be controlled only by a global
  `axis_fraction` threshold.
- A better adaptive function should depend on multiple local signals together,
  for example:
  - neighbor-owned rejection fraction
  - best rejected axis score
  - immediate future support count and brightness
  - local crowding or nearest-neighbor spacing
- In other words, the environment signal is real, but the function should be
  conditional and local, not another broad constant relaxation.

Action taken after the test:

- Restored `signal_center_neighbor_owned_future_rescue_axis_fraction` to
  `0.60` in the tuning YAML.
- Keep the result as negative evidence: it shows why later tuning should build
  a conditional rescue function instead of pushing the same scalar lower across
  the whole dataset.

2026-06-16 crowded-window note: immediate-future dim-center guard
-----------------------------------------------------------------

Focused windows:

- baseline crowded reference:
  - `output_ubuntu_fluo_resume85_85-90_celluniverse2_tuning_20260616_124839`
- negative global-axis relaxation reference:
  - `output_ubuntu_fluo_resume85_85-90_celluniverse2_tuning_20260616_130128`
- narrow crowded-rescue experiments:
  - `output_ubuntu_fluo_resume85_85-90_celluniverse2_tuning_20260616_131553`
  - `output_ubuntu_fluo_resume85_85-90_celluniverse2_tuning_20260616_132438`
- immediate-future dim-center guard rerun:
  - `output_ubuntu_fluo_resume85_85-90_celluniverse2_tuning_20260616_133653`

What changed:

- Added YAML-backed `pca_bridge_immediate_future_aligned_min_brightness: 0.06`.
- Added `pca_bridge_immediate_future_dim_center_gate` for `pca_bridge_cut`
  proposals that:
  - use immediate-future aligned centers,
  - use aligned-pair fallback snapping,
  - and have `windowImmediateMatchedMinBrightness` below the configured floor.

Why this was added:

- The earlier crowded-rescue run showed a repeated failure family where the
  proposal used a geometrically neat frame+1 pair, but one daughter center was
  basically background:
  - frame 86 `cell_11111`: `windowImmediateMatchedMinBrightness=0.0340003`
  - the split then died later at daughter-density / bad-geometry stages
- This is exactly the class where later-frame support can look good enough to
  rescue the proposal numerically, while the actual frame+1 centers are still
  not trustworthy enough to place daughters.

What the rerun showed:

- The new guard fired on multiple bad immediate-future aligned pairs:
  - frame 85 `cell_11110`:
    - `reason=pca_bridge_immediate_future_dim_center_gate`
    - `immediateFutureBrightness=0.028268`
  - frame 85 `cell_11100`:
    - `reason=pca_bridge_immediate_future_dim_center_gate`
    - `immediateFutureBrightness=0.0274743`
  - frame 86 `cell_11011`:
    - `reason=pca_bridge_immediate_future_dim_center_gate`
    - `immediateFutureBrightness=0.0302463`
- For the main crowded target, frame 86 `cell_11111`, the failure mode improved:
  - it no longer used the obviously bad dim immediate-future pair
  - the code reverted to the raw delayed-future near-miss bridge pair:
    - `reason=raw_delayed_future_near_miss_preferred`
  - the split then failed later at:
    - `Split Reject hard overlap`
    - overlap against the neighboring split family (`cell_11110`)

Cross-window interpretation:

- This guard is directionally correct as a source-specific rule:
  it removes a genuinely untrustworthy proposal class instead of weakening
  downstream biology gates.
- But it is not a clean global win by itself:
  - frame 85 validation got worse immediately after blocking the dim frame+1
    pair for `cell_11110`
  - in other words, some dim immediate-future pairs are still the earliest
    usable evidence, even if they are too weak to place daughters safely in
    the current frame

Updated rule from this experiment:

- Immediate-future brightness is a real environment-linked signal, but it
  should not be used as a blanket accept/reject scalar.
- It behaves like a source-specific trust signal:
  - strong enough to demote or delay an exact frame+1 placement,
  - not strong enough by itself to prove the split is biologically wrong.
- The next adaptive step should combine:
  - immediate-future brightness,
  - whether support comes from frame+1 only or persists into frame+2,
  - local crowding / overlap pressure,
  - and whether the proposal is raw bridge, signal-center, or rod-tip sourced.

2026-06-16 cross-window synthesis: what actually looks global
-------------------------------------------------------------

Windows re-read together:

- fresh sparse-to-mid guardrail:
  - `output_ubuntu_fluo_fresh_0-90_celluniverse2_tuning_20260616_093027`
- crowded frontier references:
  - `output_ubuntu_fluo_resume85_85-90_celluniverse2_tuning_20260616_124839`
  - `output_ubuntu_fluo_resume85_85-90_celluniverse2_tuning_20260616_133653`
  - `output_ubuntu_fluo_resume85_85-90_celluniverse2_tuning_20260616_135225`
- checkpoint-history comparison:
  - `output_ubuntu_fluo_resume70_70-200_celluniverse2_tuning_20260616_104819`
- retained later crowded references:
  - `run_107_resume119_to200_strong_cost_guard`
  - `run_112_resume120_to200_clean_pca_cont_snap`
- focused late shared-third-cell / rod windows:
  - `run_194_resume140_to142_high_shape_rod_cost_rescue`
  - `run_187_resume140_to142_dim_signal_cost_rescue`

What this combined evidence says:

1. Broad scalar lowering is still the wrong tuning style.
   - The immediate-future dim-center guard rerun
     (`...135225`) validated exactly the same as `...133653` at frames `85`
     and `86`, and both stayed worse than `...124839`.
   - That means a frame+1 brightness floor, by itself, is not the real global
     lever. It changes proposal routing, but it does not improve the crowded
     outcome.

2. Some knobs already look globally safe.
   - The stronger daughter halfspace clamp
     (`split_daughter_refit_halfspace_min_fraction=0.18`) still holds the
     fresh `0-84` guardrail and does not appear to create the crowded failures.
   - Keeping future-supported seed pairs locked through PCA remains a good
     pattern when the daughters already have clean geometry and support.
   - Hard rejection of obviously fake / air / background daughters still looks
     necessary everywhere; nothing in the later windows argues for weakening
     that floor globally.

3. The unstable knobs are the ones that really depend on environment.
   - Absolute daughter brightness:
     gamma decay, local background, and denoising change what "dim" means.
   - Drift budget:
     in crowded neighborhoods a small PCA move can steal a sibling or neighbor,
     while the same drift is harmless in sparse neighborhoods.
   - Parent-distance balance:
     later windows show real support can exist with weak balance, but only when
     the future support is both persistent and bright enough.
   - Flatness / rod strictness:
     later windows (`119+`, `140+`) contain plausible rods that are weaker than
     the clearest early bridge-cut cases but stronger than obvious false
     positives, so one universal flatness constant is too crude.

4. Local crowding is more informative than global embryo count.
   - The frame-`85-87` family is locally crowded even before the whole embryo
     reaches the later-frame density regime.
   - The `70-200` resume crosses the same frontier but degrades harder by frame
     `87` (`pred=73 gt=77`, versus `74/77` in `...135225`), which points to
     history plus local neighborhood state, not a single global count
     threshold.

5. The post-PCA force-rod cleanup is currently blocked more by cost policy
   than by candidate generation.
   - In the `70-200` frame-`87` logs, rod-like misses such as `cell_11110`,
     `cell_11011`, `cell_10011`, and `cell_20010` do generate forced split
     candidates.
   - They then die mainly at `cost_not_improved` or
     `cost_improvement_too_small`, not because the rod logic failed to place
     seeds.
   - So if we want that stage to help globally, the next adjustment should be
     a narrow cost rule tied to rod evidence and local crowding, not another
     looser rod detector.

6. The crowded window is dynamic, not one fixed failure shape.
   - In the focused rerun `...135225`, validation shifts across frames:
     - frame `85`: missing `148`, `155`; extra `cell_11110`
     - frame `86`: missing `148`, `211`, `434`, `566`; extras `cell_11011`,
       `cell_000111`
     - frame `87`: missing `44`, `148`, `211`, `566`; extra `cell_11011`
     - frame `88`: missing `133`, `566`; no extras
     - frame `89`: missing `133`, `566`; no extras
     - frame `90`: missing `133`, `362`, `566`; no extras
   - Practical meaning:
     some early crowded misses are later recovered, while other daughters drift
     or remain merged as the neighborhood changes. That is another sign that
     the useful controls here are trajectory- and crowding-aware, not fixed
     one-frame constants.

Practical tuning rule going forward:

- Keep these as fixed global protections:
  - fake / air daughter rejection,
  - clean future-supported seed locks,
  - stronger daughter halfspace preservation,
  - hard density floor below tolerance.
- Treat these as environment-linked functions instead of fixed scalars:
  - daughter drift allowance as a function of nearest-neighbor spacing,
  - soft brightness rescue as a function of local background and sibling
    brightness ratio,
  - parent-balance relaxation as a function of future support count plus
    future support quality,
  - rod/flat rescue cost tolerance as a function of local crowding and bridge
    evidence.
- Stop spending time on broad threshold lowering that is not conditioned on
  proposal source, local crowding, or future support quality.

2026-06-16 parameter experience map: what should stay fixed vs become adaptive
----------------------------------------------------------------------------

Run windows folded into this map:

- early/sparse guardrail:
  - `output_ubuntu_fluo_fresh_0-90_celluniverse2_tuning_20260616_093027`
- crowded transition:
  - `output_ubuntu_fluo_resume85_85-90_celluniverse2_tuning_20260616_124839`
  - `output_ubuntu_fluo_resume85_85-90_celluniverse2_tuning_20260616_133653`
  - `output_ubuntu_fluo_resume85_85-90_celluniverse2_tuning_20260616_135225`
  - negative experiment:
    `output_ubuntu_fluo_resume85_85-90_celluniverse2_run343_20260616_141450`
- later dense windows:
  - `run_107_resume119_to200_strong_cost_guard`
  - `run_112_resume120_to200_clean_pca_cont_snap`
  - `run_187_resume140_to142_dim_signal_cost_rescue`
  - `run_194_resume140_to142_high_shape_rod_cost_rescue`

Observed regimes:

1. Sparse / early (`0-84`)
   - Main risk is false positives from loosening thresholds too broadly.
   - Future-supported daughter locks and the stronger halfspace clamp behave
     well here.
   - Large PCA daughter motion is usually harmless here only because neighbor
     pressure is low; this should not be interpreted as proof that wide drift
     is globally safe.

2. Crowded transition (`85-90`)
   - The difficult families are not all the same:
     - some are true bridge/split misses,
     - some are correct split candidates that die at global cost,
     - some are false daughters stealing crowded territory.
   - The failed `run343` near-miss rescue confirms that broad force-rod cost
     relaxation creates false crowded splits before it cleanly fixes the main
     misses.
   - Absolute brightness became especially unstable here because effective gamma
     had already decayed and local background varied sharply across neighbors.

3. Dense later windows (`119+`, `140+`)
   - Real daughters often have weak parent-distance balance or weaker flatness
     than the textbook early rods, but they still carry clean future support.
   - Third-cell claim conflict and occupied-territory theft remain real failure
     modes even when future support exists.
   - This is where flatness/balance rules most clearly need conditioning on
     local context instead of one global scalar.

Parameter families that now look genuinely global:

- `split_daughter_refit_halfspace_min_fraction`
  - The stronger clamp remains compatible with early good behavior and later
    future-supported locks.
  - Keep as a fixed protection unless a targeted counterexample appears.
- clean future-supported daughter position lock
  - When the seed pair is already clean and geometrically plausible, keeping
    position fixed through PCA is consistently helpful.
- hard fake / air / obvious-background rejection
  - Still necessary across all windows.
  - The floor itself can remain hard; only the rescue path above that floor
    should adapt.
- third-cell claim and hard-overlap protections
  - Repeated later-window evidence says these are real blockers for genuinely
    bad proposals, not accidental nuisances.

Parameter families that appear environment-linked:

- Daughter drift tolerance / snap trust
  - Why adaptive:
    - sparse windows tolerate larger motion,
    - crowded windows let the same motion steal a sibling or neighbor.
  - Candidate conditioning variables:
    - nearest-neighbor spacing,
    - local live-cell count inside a parent-radius neighborhood,
    - proposal source (`raw bridge`, `signal center`, `future-aligned rod tip`),
    - snapped future-pair distance measured in parent-radius units.

- Parent-distance balance rescue floors
  - Why adaptive:
    - later windows showed true daughters with weak balance but strong
      persistent future support,
    - weak-balance proposals without bright future support still produce false
      crowded placements.
  - Candidate conditioning variables:
    - `futureBoth`,
    - `futureMissing`,
    - `futureBestMinBrightness`,
    - aligned-pair snap distance,
    - local crowding.

- Brightness-based rescue / rejection thresholds
  - Why adaptive:
    - absolute brightness moved with gamma decay, denoising, and local
      background.
  - Candidate conditioning variables:
    - daughter brightness relative to parent mean signal,
    - daughter brightness relative to local background in the same bbox,
    - sibling brightness ratio,
    - future-pair minimum brightness.
  - Practical rule:
    - stop adding absolute brightness floors unless they are explicitly tied to
      local background or future-support quality.

- Rod / flat rescue cost policy
  - Why adaptive:
    - crowded windows generate plausible rod-split candidates that fail at cost,
      but loosening cost globally creates false crowded splits.
  - Candidate conditioning variables:
    - parent rod-shape strength,
    - local crowding / nearest-neighbor spacing,
    - whether the candidate stays inside a clean ownership halfspace,
    - future support quality,
    - third-cell claim cleanliness.

- Flatness / rod strictness
  - Why adaptive:
    - late windows include real splits that are weaker than the early textbook
      rods but still biologically plausible.
  - Candidate conditioning variables:
    - local crowding,
    - future-support persistence,
    - snap distance in parent-radius units,
    - whether another cell already claims the same future territory.

What to log and reuse when building adaptive functions:

- nearest-neighbor spacing around the parent,
- count of live cells within one or two parent radii,
- `futureBoth`, `futureMissing`, `parentPersists`,
- `futureBestMinBrightness`,
- aligned future-pair snap distance divided by parent max radius,
- parent mean signal, local background estimate, and daughter/parent brightness
  ratios,
- claim-conflict severity from the best competing neighbor,
- proposal source family and whether the proposal already stayed in its own
  halfspace.

Working rule for future tuning:

- Prefer:
  - fixed global protections for obviously bad geometry,
  - adaptive functions for balance, drift, rod strength, and soft brightness
    rescue.
- Avoid:
  - broad scalar threshold lowering,
  - new global rescues that do not check local crowding,
  - using absolute brightness alone as a universal decision variable.

## 2026-06-16 run357 result: lower crowded rod-tip trigger was safe early but did not move the real misses

Experiment:

- new YAML:
  `config/config_celluniverse2_run357_early_crowded_rodtip.yaml`
- only intended change:
  `rod_tip_future_aligned_crowded_cell_count_min: 100 -> 55`
- motivation:
  the crowded frame-84 to 87 frontier only had about 55 to 60 live cells, so
  the existing crowded rod-tip relaxation never fired there.

Runs:

- focused frontier:
  `output_ubuntu_fluo_resume84_84-87_celluniverse2_run357_20260616_143438`
- early guardrail:
  `output_ubuntu_fluo_0-22_celluniverse2_run357_20260616_143444`

What changed in behavior:

- the new threshold did exactly what it was supposed to do mechanically:
  crowded rod-tip relaxations started firing at frames 84 to 86 for cells such
  as `cell_3010`, `cell_01100`, `cell_11111`, and `cell_00111`.
- this means the old trigger level of 100 was too high to even expose that
  relaxation in the early crowded frontier.

What did not improve:

- focused validation stayed unchanged at the real frontier:
  - frame 85:
    `pred=59 gt=60 missing=148`
  - frame 86:
    `pred=65 gt=67 missing=148,566`
- so lowering the crowded rod-tip trigger alone is not a solution for the
  actual blocked families.

Why it failed to solve `148`:

- frame 85:
  `cell_11110` still failed at `future_window_gate`
  even after the crowded relaxation was available elsewhere.
- frame 85:
  `cell_11111` became `pca_bridge_cut` acceptable and was scheduled, but the
  frame still validated missing `148`, so exposing more crowded rod-tip rescue
  did not address the real daughter localization / downstream fit outcome.
- frame 86:
  `cell_11111` still failed later than the new trigger:
  `signal_center_split` hit `future_signal_no_current_balance_gate`,
  `rod_tip_split_fallback` hit
  `unclean_future_aligned_pair_balance_gate`,
  and `pca_bridge_cut` hit
  `future_aligned_pca_flatness_gate`.
- takeaway:
  the miss is not caused by the crowded rod-tip flatness trigger being too
  strict in global cell-count terms; it is caused by later balance / clean-snap
  / trusted-future gates on this family.

Why it failed to solve `566`:

- frame 86:
  `cell_20010` still became `pca_bridge_cut` acceptable and was scheduled.
- the blocker is downstream, not discovery:
  the family reaches split attempt territory, so simply opening rod-tip
  relaxation earlier does not help it.
- takeaway:
  `566` belongs to the final placement / cost family, not the crowded rod-tip
  trigger family.

Guardrail result:

- the early-window run remained clean on the previously good range:
  - frame 18:
    `pred=10 gt=10 missing=0 extra=0`
  - frame 19:
    `pred=12 gt=12 missing=0 extra=0`
  - frame 20:
    `pred=12 gt=12 missing=0 extra=0`
  - frame 21:
    `pred=12 gt=12 missing=0 extra=0`
  - frame 22:
    `pred=12 gt=12 missing=0 extra=0`

Decision:

- do not promote run357 as the new base just because it activates more crowded
  rod-tip rescues.
- keep the lesson:
  the crowded rod-tip trigger should probably be local- or environment-aware,
  but the current global `100 -> 55` scalar by itself is only
  safe-but-insufficient.
- next tuning should target:
  - the `148` family's later balance / clean-future trust gates,
  - the `566` family's downstream split acceptance or cost behavior,
  not the crowded rod-tip trigger again by itself.

## 2026-06-16 run358 and run359: two parameter-only probes clarified the delayed-future regime

Probe 1:

- config:
  `config/config_celluniverse2_run358_future_halfspace.yaml`
- change:
  lowered `split_clean_future_halfspace_min_parent_balance` to `0.30`
- focused run:
  `output_ubuntu_fluo_resume85_85-88_celluniverse2_tuning_instrument_20260616_151034`
  (stopped after frame `85` once the branch behavior was clear)
- frame `85` validation:
  `pred=59 gt=60 missing=148`

What it taught us:

- This was effectively a no-op for the true missing family.
- The frame-`85` missing cell was still the old `cell_11100` delayed-future
  bridge family:
  - `windowImmediateBothDaughtersSupported=0`
  - `futureBoth=1`
  - `futureMissing=2`
  - `futureBestMinBrightness=0.314322`
- Because that family does not satisfy the ordinary
  `winningCleanFutureSupportedPcaBridge` branch, lowering the
  clean-future balance floor did not activate the stronger future-backed
  daughter-halfspace clamp.
- Evidence in the log:
  `cell_11100` still refit under
  `minFraction=0.18 futureBackedBridge=0`.

Decision from run358:

- Do not keep tuning the clean-future balance floor for frame-`85` delayed
  support misses.
- This frontier is not asking for "softer clean-future balance." It is asking
  for a separate delayed-future refit-preservation rule.

Probe 2:

- config:
  `config/config_celluniverse2_run359_halfspace_030.yaml`
- change:
  raised the global `split_daughter_refit_halfspace_min_fraction` from
  `0.18` to `0.30`
- proper focused run with future context:
  `output_ubuntu_fluo_resume85_85-88_celluniverse2_tuning_instrument_20260616_151714`
- validation:
  - frame `85`: `pred=59 gt=60 missing=148`
  - frame `86`: `pred=65 gt=67 missing=148,434,566 extra=001010`
  - frame `87`: `pred=70 gt=77 missing=13,84,140,148,292,395,566`
  - frame `88`: `pred=70 gt=82 missing=13,28,59,84,122,140,148,234,292,331,395,566`

Important branch-level evidence:

- The stronger global halfspace clamp did change the bad frame-`85` split path:
  `cell_11100` moved from
  `minFraction=0.18` to `minFraction=0.30`.
- That was enough to change the final rejection mode:
  - before: hard overlap / dense-flat style failure
  - after: `daughter_density_brightness`
- But it still did not recover GT `148`.
- Worse, the broader clamp destabilized later frames in the same focused
  window, so the scalar itself is not globally helpful even though it moves the
  local geometry.

Guardrail / method note:

- A direct one-frame binary replay is not a valid way to judge these branches.
  When only frame `85` is run in isolation, the future-window logic cannot load
  later frames and many future-backed proposals collapse to `futureBoth=0`.
  Use the scripted `85-88` style window for any delayed-future diagnosis.

Decision from run359:

- Do not promote a broader global daughter-halfspace clamp above `0.18`.
- Keep `split_daughter_refit_halfspace_min_fraction=0.18` as the best general
  setting so far.
- The next useful change should be narrow and YAML-backed:
  a delayed-future-specific daughter-side preservation path that activates only
  when:
  - immediate future support is absent,
  - later future support is present,
  - parent does not persist,
  - future brightness is genuinely strong,
  - and the proposal is still geometrically plausible.

## 2026-06-16 run360: delayed-future-specific halfspace branch fired, but still did not recover the frontier

Implementation:

- code:
  added `split_delayed_future_refit_halfspace_min_fraction`
  as a new YAML-backed knob
- config:
  `config/config_celluniverse2_run360_delayed_future_halfspace.yaml`
- behavior:
  reuse the existing delayed-future PCA-bridge rescue thresholds
  (`futureBoth`, `futureMissing`, brightness, parent shape, parent balance),
  but apply them only to the daughter refit halfspace clamp rather than to a
  global cost rescue.

Focused run:

- `output_ubuntu_fluo_resume85_85-88_celluniverse2_tuning_instrument_20260616_152941`
- validation:
  - frame `85`: `pred=59 gt=60 missing=148`
  - frame `86`: `pred=64 gt=67 missing=148,203,434,566 extra=001010`
  - frame `87`: `pred=69 gt=77 missing=13,84,140,148,203,292,395,566`
  - frame `88`: `pred=69 gt=82 missing=13,28,59,84,122,140,148,203,234,292,331,395,566`

What improved mechanically:

- The new branch did activate on the intended frame-`85` target:
  `cell_11100` logged
  `minFraction=0.7 futureBackedBridge=0 delayedFutureBridge=1`.
- So the structural diagnosis from run358 was correct:
  this family really is a delayed-future branch and not an ordinary
  clean-future one.

Why it still failed biologically:

- frame `85`:
  the stronger delayed-future clamp moved the same target into a different
  rejection surface rather than recovering GT `148` cleanly:
  `cell_11100` hit `d2_buried_in_cell_10011`.
- later in the same focused window, the same family continued to mutate into
  cost / density failures instead of stabilizing.
- `cell_20010` was unaffected in the useful way:
  it still failed as a buried daughter (`d1_buried_in_cell_11011`) because its
  future brightness was too weak for the delayed-future branch and the real
  blocker remains crowded ownership / burial rather than side preservation.

Decision from run360:

- Keep the code pattern as a useful experiment, but do not promote this config.
- The delayed-future family is real, but halfspace preservation alone is not
  enough.
- The next likely helpful rule is not a bigger clamp. It is a delayed-future
  bio/ownership rule that reasons about whether the preserved daughter lands in
  another live cell's body or in obviously background-like volume before we
  accept the stronger preserve path.
