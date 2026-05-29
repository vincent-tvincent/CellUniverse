# CellLumen Fusion Center And Split Prior

**Last updated:** 2026-05-29

This document describes the active CellLumen fusion path used by the main
CellUniverse tracker. It covers center-prior placement, deterministic
CellLumen split priors, future-window evidence, fallback behavior, and how the
selected priors enter the existing CellUniverse split validator.

The short version:

```text
CellLumen proposes image-derived centers.
CellUniverse uses those centers as priors.
CellUniverse still accepts or rejects movement and split changes by its normal
image cost, overlap, geometry, bridge, and biological gates.
```

This method is active when:

```yaml
cell_lumen:
  enabled: true
  fusionEnabled: true
  fusionCenterPriorEnabled: true
  fusionSplitPriorEnabled: true
```

In the current runtime config, CellLumen fusion consumes the same prepared stack
as CellUniverse. With `simulation.preprocess_mode: n2v2`, both systems use the
N2V2-preprocessed stack rather than separate raw/preprocessed inputs.

## Main Code Locations

| Area | Location |
|---|---|
| CellLumen raw center detection | `src/CellLumen.cpp` |
| Fusion orchestration | `CellUniverse::applyCellLumenRescue` in `src/CellUniverse.cpp` |
| Future-window candidate loading | `CellUniverse::getCellLumenLookaheadCandidates` |
| Rolling prepared stack window | `CellUniverse::prepareFrameWindow` |
| Center-prior movement handoff | `CellUniverse::optimize`, perturb loop |
| Split-prior handoff to validator | `Frame::trySplitCellPhased` |
| Movement proposal scoring | `Frame::perturbCell` |
| Active config | `config/config.yaml`, `cell_lumen:` block |

## Frame-Level Flow

With CellLumen fusion enabled, the frame loop uses a rolling prepared-stack
window:

```text
for frame N:
  prepareFrameWindow(N)
    load/preprocess N, N+1, ... up to future window size
  prepareFrame(N)
    attach prepared stack for N
    build signal map
    run frame-dependent signal-center preparation
  optimize(N)
    apply CellLumen fusion on prepared stack
    run center-prior movement and split-prior validation
```

The rolling window exists so future split evidence can reuse already prepared
N2V2 stacks. With the current config:

```yaml
fusionSplitPriorWindowEnabled: true
fusionSplitPriorWindowSize: 3
```

At frame `N`, the window is sized to 3 frames total:

```text
current frame: N
future evidence offsets: N+1, N+2
```

The tracker still commits only frame `N`. Future frames are used only for
CellLumen center evidence.

## Candidate Vocabulary

The code uses "candidate" for proposed things that are not trusted yet.

| Term | Meaning |
|---|---|
| CellLumen detected cell | A raw image-derived center/component from CellLumen. |
| Center candidate | A detected center assigned to an existing CellUniverse cell as a possible continuation target. |
| Lookahead candidate | A future-frame CellLumen center used only for split evidence. |
| Split prior / split pair | Two center candidates treated as possible daughters of one parent. |
| Selected prior | A ranked split pair chosen for the current frame. |
| Accepted split | A selected prior that also passes `trySplitCellPhased` and mutates the frame. |

## CellLumen Detected Cell

CellLumen produces `CellLumen::DetectedCell` objects:

```cpp
std::string name;
cv::Point3f centerScaled;
float zForCsv;
float majorRadius;
float bRadius;
float minorRadius;
int voxelCount;
float meanIntensity;
int shellVoxelCount;
float top10Intensity;
float shellMeanIntensity;
float meanMinusShell;
float top10MinusShell;
EmbryoBrightTracker::Comp3DStat component;
```

Important fields:

| Field | Meaning |
|---|---|
| `centerScaled` | Center in CellUniverse coordinates. Z is scaled to match the main interpolated frame. |
| `zForCsv` | Z coordinate intended for CSV export. |
| `majorRadius`, `bRadius`, `minorRadius` | Estimated ellipsoid radii from the detected component. |
| `voxelCount` | Number of voxels in the detected component. Used as a size/reliability filter. |
| `meanIntensity` | Mean brightness inside the component. |
| `shellMeanIntensity` | Mean brightness in the local surrounding shell. |
| `top10Intensity` | Bright-core summary of the component. |
| `top10MinusShell` | Bright-core contrast above local shell. This is the main CellLumen signal score. |
| `component` | Connected-component stats: voxel count, weighted/unweighted center sums, intensity sum, bounding box. |

## How A Center Becomes Possible

CellLumen detects possible centers from image evidence. The detector can use
seeded watershed and/or percentile threshold sweeps, depending on config.

The high-level detection flow is:

```text
prepared stack
  -> local contrast volume
  -> bright local maxima / seeds
  -> seed spacing and valley checks
  -> component growth / seeded watershed or percentile components
  -> candidate scoring and duplicate/shape filters
  -> DetectedCell objects
```

For seeded watershed:

1. A local-contrast stack is built.
2. A seed threshold is chosen from percentile/floor config.
3. A mask threshold is chosen from percentile/floor config.
4. Local maxima above seed threshold become seed candidates.
5. Seeds too close to stronger seeds are removed unless the valley drop between
   them is large enough.
6. Regions grow from seeds into the lower mask.
7. Components are converted to `DetectedCell`.

For percentile detection, the detector tries multiple brightness percentiles,
scores each resulting component set, and keeps the best candidate set. Scoring
penalizes obvious overgrowth, tiny fragments, duplicate-like components, overly
large components, and poor shape/radius behavior.

The fusion path then applies separate filters depending on how a detected center
will be used.

### Placement Center Candidate

For center-prior placement, a detected center is stored as:

```cpp
struct CellLumenCenterCandidate {
    cv::Point3f position;
    float distance;
    int voxelCount;
    float signal;
    int candidateId;
};
```

| Field | Meaning |
|---|---|
| `position` | Candidate center in CellUniverse coordinates. |
| `distance` | Distance from the assigned existing cell center. |
| `voxelCount` | Copied from `DetectedCell::voxelCount`. |
| `signal` | Copied from `DetectedCell::top10MinusShell`. |
| `candidateId` | Index in the CellLumen candidate list; used for ownership/reservation. |

Current placement filters:

```yaml
fusionMinVoxels: 1000
fusionMinTop10MinusShell: 65.0
fusionCenterPriorMaxDistance: 24.0
```

So a detected center can guide placement only if:

```text
voxelCount >= 1000
top10MinusShell >= 65.0
distance to an existing cell <= 24 px
```

When multiple centers are near the same existing cell, the nearest center wins;
ties prefer higher signal.

### Split-Prior Center Candidate

For split-prior construction, the size/signal gate is intentionally more
permissive:

```yaml
fusionSplitPriorMinVoxels: 500
fusionSplitPriorMinTop10MinusShell: 0.0
```

So a detected center can enter split-pair ranking if:

```text
voxelCount >= 500
top10MinusShell >= 0.0
```

The split path is high recall at this stage. Weak candidates are expected to be
filtered later by parent geometry, neighbor ownership, future evidence, and the
final CellUniverse split gates.

## Center-Prior Placement

Center-prior placement is a movement proposal for existing cells. It does not
add a cell and it does not force acceptance.

For each assigned center candidate, CellUniverse tries a target position:

```text
target = old_position * (1 - fusionCenterPriorPositionBlend)
       + lumen_center * fusionCenterPriorPositionBlend
```

Current config:

```yaml
fusionCenterPriorPositionBlend: 0.35
fusionCenterPriorRadiusBlend: 0.25
```

So the first target is:

```text
target = old_position * 0.65 + lumen_center * 0.35
```

This move is passed to `Frame::perturbCell` as a forced position. Because the
target is explicit, signal-map and random position guidance are skipped for
that proposal. The proposal is accepted only if the total cost improves.

If the center-prior move is rejected, the same iteration falls back to the
normal local/random perturbation path:

```text
sample random x/y/z/theta move
apply signal-map probability bias if active
apply bright-core guidance if active
optionally run PCA shape/position refit
accept only if total cost improves
```

If the same CellLumen center is reserved as a daughter in a split prior, the
continuing cell is not allowed to consume that center as a normal placement
target. The code logs this as:

```text
cell_lumen_center_candidate_reserved_by_split
```

and locks PCA position for that continuation.

## Split-Prior Construction

A split prior is a pair of candidate centers assigned to one parent cell.

Current key switches:

```yaml
fusionSplitPriorEnabled: true
fusionSplitPriorForceSchedule: true
fusionSplitPriorSkipRandomSplits: true
fusionSplitPriorGuidedOnly: true
fusionSplitPriorFallbackToClassicOnReject: false
```

This means:

1. CellLumen-ranked split priors drive split scheduling.
2. Broad classic random split scheduling is disabled once the split-prior
   preparation has run.
3. For a parent with a prior, `trySplitCellPhased` is guided to that pair.
4. If a guided split is rejected, the current config does not fall back to
   broad classic random split search.

### Parent Assignment

Each split-prior center candidate is assigned to any parent whose catch radius
contains it.

Current config:

```yaml
fusionSplitPriorMaxParentDistance: 42.0
fusionSplitPriorParentRadiusScale: 2.8
```

The effective catch radius is:

```text
absoluteCatch = 42.0
scaledCatch = parentMaxRadius * 2.8
catchRadius = min(absoluteCatch, scaledCatch)
```

If one of those terms is disabled or zero, the code falls back to the other.

A candidate may be rejected for a parent if it is more clearly owned by another
existing continuation and the current parent is not elongated enough to carry a
soft conflict.

### Pair Enumeration

For each parent with two or more assigned candidate centers:

```text
for each pair (candidate A, candidate B):
  compute daughter separation
  compute pair midpoint
  compute distances to parent
  compute neighbor ownership penalties
  compute future-window support
  compute ranking score
```

The pair becomes a ranked split prior if it survives hard rejection gates or is
kept with soft penalties.

## Separation And Midpoint Gates

Daughter separation must be plausible.

Current config:

```yaml
fusionSplitPriorMinSeparation: 8.0
fusionSplitPriorMinSeparationRadiusScale: 0.55
fusionSplitPriorMaxSeparation: 64.0
fusionSplitPriorMaxSeparationRadiusScale: 3.4
```

Computed values:

```text
minSep = max(8.0, parentMaxRadius * 0.55)
maxSep = min(64.0, parentMaxRadius * 3.4)
```

If soft ranking gates are enabled:

```yaml
fusionSplitPriorRankingSoftGateEnabled: true
```

then near misses can survive:

```text
softMinSep = minSep * 0.70
softMaxSep = maxSep * 1.30
```

Pairs outside the soft range are rejected. Pairs outside the normal range but
inside the soft range receive a ranking penalty.

For round parents, a stricter close-daughter guard applies:

```yaml
fusionSplitPriorRoundParentMaxShape: 1.25
fusionSplitPriorRoundParentMinSeparationRadiusScale: 0.85
```

If:

```text
parentShapeElongation <= 1.25
```

then:

```text
roundParentMinSep = parentMaxRadius * 0.85
```

This prevents a nearly round parent from being split into two very close bright
peaks that are probably internal structure rather than daughters.

The pair midpoint must also stay near the parent.

Current config:

```yaml
fusionSplitPriorMaxMidpointDistance: 28.0
fusionSplitPriorMaxMidpointRadiusScale: 0.80
```

The active limit is:

```text
maxMidpointDist = max(28.0, parentMaxRadius * 0.80)
```

If midpoint distance exceeds this limit, the pair is rejected unless it is still
inside the soft midpoint range:

```text
soft midpoint limit = maxMidpointDist * 1.60
```

Near misses receive a soft midpoint penalty.

## Ranking Score

The ranking score is not the final image cost. It is a preselection cost for
choosing which CellLumen pairs should be tried as real splits.

Lower is better.

The current formula is:

```text
rawScore =
    midpointWeight * midpointDist
  + separationPenaltyWeight * sepPenalty
  - signalBonus
  + parentPersistencePenalty
  + neighborClaimPenalty
  + rankingSoftPenalty
  + continuationClaimSoftPenalty
  + windowSupportScore
  - balancedWindowBonus
  - parentAnchorBonus

finalScore =
    rawScore - elongatedParentScoreBonus, if elongated-parent rescue applies
```

### Score Terms

| Term | Current scale | Meaning |
|---|---:|---|
| `midpointDist` | pixels, weight `1.0` | Distance from old parent center to pair midpoint. |
| `sepPenalty` | pixels, weight `0.20` | Distance from target daughter separation. |
| `signalBonus` | `0.001 * (signalA + signalB)` | Small reward for stronger CellLumen signals. |
| `parentPersistencePenalty` | weighted pixels, weight `8.0` | Penalizes old-parent-center persistence and very imbalanced daughters. |
| `neighborClaimPenalty` | weighted pixels, weight `3.0` | Penalizes daughter centers better explained by neighboring cells. |
| `rankingSoftPenalty` | weighted pixels | Penalty for soft gate near misses. |
| `continuationClaimSoftPenalty` | weighted pixels, weight `6.0`, sibling block `+100` | Penalizes stealing another live continuation. |
| `windowSupportScore` | small score bonus/penalty | Future-frame evidence for or against daughters. |
| `balancedWindowBonus` | up to `25.0` | Large reward for clean balanced future support. |
| `parentAnchorBonus` | up to `8.0` | Reward for one-sided parent-anchored future-supported pairs. |
| `elongatedParentScoreBonus` | `10.0` | Reward when parent is already elongated enough to be division-like. |

Current score gates:

```yaml
fusionSplitPriorMaxScore: 22.0
fusionSplitPriorElongatedParentRescueEnabled: true
fusionSplitPriorElongatedParentMinShape: 1.55
fusionSplitPriorElongatedParentMaxScore: 80.0
fusionSplitPriorElongatedParentScoreBonus: 10.0
fusionSplitPriorGlobalSelectMaxCost: 50.0
```

Approximate interpretation:

```text
score < 0      very strong; global selector prefers it
0..10          strong
10..22         plausible under normal score gate
22..50         weak; needs future/window/rescue logic to remain selectable
>50            normally not selectable
```

This ranking score only decides what to try. A selected prior can still be
rejected by the final split validator.

## Future-Window Evidence

Future-window evidence asks: if this parent really split into these two
daughters, does CellLumen still see those two daughter-like centers in the next
images?

Current config:

```yaml
fusionSplitPriorWindowEnabled: true
fusionSplitPriorWindowSize: 3
fusionSplitPriorWindowMatchDistance: 28.0
fusionSplitPriorWindowMatchDistancePerFrame: 4.0
fusionSplitPriorWindowDaughterSupportBonus: 4.0
fusionSplitPriorWindowMissingDaughterPenalty: 5.0
fusionSplitPriorWindowParentPersistencePenalty: 6.0
fusionSplitPriorWindowBalancedDaughterBonus: 25.0
```

With window size 3, offsets 1 and 2 are checked.

For offset `k`, match radius is:

```text
matchDistance(k) = 28.0 + 4.0 * max(0, k - 1)
```

So:

```text
offset 1: 28 px
offset 2: 32 px
```

For each offset:

```text
if both daughters are matched:
  windowSupportScore -= 4.0 / offset
else:
  windowSupportScore += 5.0 * missingDaughterCount * 0.5 / offset

if old parent center persists and both daughters are not matched:
  windowSupportScore += 6.0 / offset
```

Best clean support over offsets 1 and 2:

```text
-4.0/1 - 4.0/2 = -6.0
```

If clean future support is balanced and not claimed by neighbors, the pair can
also receive:

```text
balancedWindowBonus = 25.0
```

This is intentionally much larger than the ordinary window support score. It
lets future-confirmed daughter pairs compete even when the current frame alone
is ambiguous.

Important: future-window evidence does not use future CellUniverse tracking
results. It only runs CellLumen detection on future images and checks whether
centers near the proposed daughters persist.

## Parent Anchors And Lookahead Injection

If an elongated parent has too few current-frame candidate centers, the window
logic can add nearby future CellLumen centers into the current ranking pool.
These synthetic candidate IDs are offset so they cannot be confused with real
current-frame candidate IDs.

If the current/future evidence suggests a one-sided split around a long parent,
the code may add a parent anchor candidate at the current parent center. This
lets ranking test a pair such as:

```text
old parent center + one future/current daughter-like center
```

Parent-anchored pairs are treated conservatively. They need clean future
support and strict selectable conditions before they can become selected priors.

## Global Selection

After every parent pair is ranked, CellUniverse does not simply choose the best
pair for every parent. It solves a constrained selection problem.

Constraints:

```text
at most fusionSplitPriorMaxPriorsPerFrame selected
one selected pair per parent group
the same CellLumen candidate ID cannot be consumed by two selected priors
only selectable ranked priors are eligible
```

Current config:

```yaml
fusionSplitPriorMaxPriorsPerFrame: 35
fusionSplitPriorGlobalSelectMaxCost: 50.0
```

The selector minimizes total selection cost while treating "skip this parent" as
zero cost. This is why negative or strongly future-supported priors are favored:
they make selecting a split better than skipping it.

The global selector logs:

```text
[CellLumen Fusion SplitPrior GlobalSelect]
```

with counts for ranked pairs, parent groups, selected priors, max priors,
selected score, and search truncation.

## Prepass Fallback

The prepass fallback exists for CellLumen blind spots. It does not reopen broad
classic random splitting.

Current config:

```yaml
fusionSplitPriorPrepassFallbackEnabled: true
fusionSplitPriorPrepassFallbackMaxPriors: 12
fusionSplitPriorPrepassFallbackMinKeptPixels: 50000
fusionSplitPriorPrepassFallbackMinShape: 1.20
fusionSplitPriorPrepassFallbackRejectBadLumenParent: true
```

The fallback looks at image-grounded expected daughters from the pre-pass. It
can add a prior only when:

```text
the parent has no selected CellLumen split prior
the parent was not marked as a bad/unsafe Lumen parent
there is enough kept bright-pixel support
the parent is shape-elongated enough
daughter separation ratio is plausible
both daughters are better claimed by this parent than neighbors
the fallback score is below threshold
```

If PCA-derived expected daughters drift too far from the original snapshot
seeds, a special fallback can use the snapshot seeds instead:

```yaml
fusionSplitPriorPrepassFallbackUseSnapshotSeedOnLargeDrift: true
fusionSplitPriorPrepassFallbackSeedMaxShift: 25.0
fusionSplitPriorSnapshotSeedMaxRefitDrift: 12.0
```

This catches cases where PCA centroids are contaminated by a neighboring bright
cloud, but the original snapshot split geometry remains plausible.

## Split Scheduling During Optimization

Once split priors are prepared, the optimization loop changes behavior.

Current config:

```yaml
fusionSplitPriorSkipRandomSplits: true
fusionSplitPriorGuidedOnly: true
```

When split-prior preparation ran for a frame:

```text
classic random split scheduling is disabled
selected CellLumen priors are force-scheduled
```

The log should include:

```text
[CellLumen SplitPrior Mode] random_splits_disabled=1
```

If no priors were selected:

```text
[CellLumen SplitPrior Mode] no_cell_lumen_split_priors=1 action=skip_classic_random_split
```

Normal cell movement still uses random/local perturbation. The random behavior
that is disabled here is broad random split discovery, not normal stepping.

## Final Split Validation

A selected split prior is passed into `Frame::trySplitCellPhased`.

The prior supplies:

```text
d1Pos, d2Pos
candidateIdA, candidateIdB
future-window evidence fields
parent/neighbor penalty metadata
```

When `fusionSplitPriorGuidedOnly` is true, the split attempt focuses on the
CellLumen prior pair rather than broad candidate search. The daughter pair still
runs through the existing split validation machinery:

```text
build daughters
optional burn-in/refine steps
daughter PCA refit
edge brightness / valley / bridge checks
daughter overlap checks
axis and geometry gates
image cost gate
overlap cost gate
bio volume/size checks
```

Current CellLumen-specific split-gate settings include:

```yaml
fusionSplitPriorUseDedicatedCostGate: true
fusionSplitPriorUseImageCostGate: true
fusionSplitPriorCost: 0.0
fusionSplitPriorCostFraction: 0.0
fusionSplitPriorMaxPositiveCostFraction: 0.12
fusionSplitPriorPositiveGateMinImageGain: 30.0
fusionSplitPriorMaxOverlapCostFraction: 0.15
fusionSplitPriorHighConfidenceMaxScore: 12.0
fusionSplitPriorHighConfidenceMaxOverlapCostFraction: 0.50
fusionSplitPriorHighConfidenceAxisAlignmentDegrees: 90.0
fusionSplitPriorDaughterVolumeScale: 0.68
fusionSplitPriorSoftGateEnabled: true
fusionSplitPriorHardMaxDaughterOverlapFraction: 0.82
fusionSplitPriorHardMaxValleyRatio: 2.0
fusionSplitPriorHardMaxOverlapCostFraction: 1.25
```

The key design point is that CellLumen proposes the daughter centers, but
CellUniverse still decides whether the split physically improves the frame.

## Debug Outputs And Logs

Useful log lines:

| Log prefix | Meaning |
|---|---|
| `[CellLumen Fusion Summary]` | Number of detected candidates, center-prior guided cells, additions, repairs, rejections. |
| `[CellLumen Window]` | Future-window enabled, window size, loaded offsets, match radius. |
| `[CellLumen Fusion SplitPrior Ranked]` | Top ranked split pairs and score components. |
| `[CellLumen Fusion SplitPrior]` | Priors selected for parents. |
| `[CellLumen Fusion SplitPrior GlobalSelect]` | Global constrained selection result. |
| `[CellLumen Fusion SplitPrior Summary]` | Candidate counts, ranked pairs, selected priors, rejection buckets. |
| `[CellLumen SplitPrior Ready]` | Priors handed to the optimizer loop. |
| `[CellLumen SplitPrior Mode]` | Whether random split scheduling is disabled. |
| `[CellLumen CenterCandidate Ready]` | Center-prior candidates available for movement. |
| `[Split Schedule] reason=cell_lumen_prior` | A selected prior is being attempted as an actual split. |

Candidate graph CSVs are written under:

```text
<output>/candidate_graph/frame_<N>_candidates.csv
```

Rows include:

| Column | Meaning |
|---|---|
| `kind` | `continuation`, `lumen_center`, or `split_pair`. |
| `source` | Current-cell state, CellLumen center, CellLumen split prior, or fallback source. |
| `parent` | Parent cell name for split pairs. |
| `candidateA`, `candidateB` | Candidate IDs used by a split pair. |
| `selected` | Whether the ranked pair was selected by global selection. |
| `score`, `rawScore` | Final and pre-bonus ranking scores. |
| `sep`, `minSep`, `maxSep` | Daughter separation and allowed range. |
| `midpointDist` | Pair midpoint distance from parent. |
| `parentShape` | Parent elongation ratio. |
| `parentPersistencePenalty` | Penalty for parent-like center persistence. |
| `neighborClaimPenalty` | Penalty for stealing neighbor continuations. |
| `parentDistNear`, `parentDistFar`, `parentDistBalance` | Daughter distances to parent. |
| `d1`, `d2` | Proposed daughter positions. |
| `voxA`, `voxB` | Voxel counts for daughter candidate centers. |
| `signalA`, `signalB` | `top10MinusShell` signal for daughter centers. |
| `note` | Extra fields such as window score, continuation blockers, fallback source. |

## Practical Interpretation

When a split is missed:

1. Check whether CellLumen produced centers near both daughters.
2. Check `[CellLumen Fusion SplitPrior Summary]` rejection buckets.
3. Check ranked pairs for the parent:
   - high `midpointDist`: daughter midpoint is too far from parent.
   - high `parentPersistencePenalty`: one candidate is too close to parent or pair is imbalanced.
   - high `neighborClaimPenalty`: daughters look owned by other cells.
   - positive `windowScore`: future frames did not support both daughters or old parent persisted.
   - no selected row in candidate graph: global selection skipped the pair.
4. If a prior was selected, inspect final split rejection logs from
   `trySplitCellPhased`. At that point the failure is no longer prior ranking;
   it is image/overlap/bridge/geometry validation.

When a false split appears:

1. Check whether `neighborClaimPenalty` or `continuationClaimSoftPenalty` was low
   despite the daughter being inside another live cell.
2. Check whether `balancedWindowBonus` or elongated rescue made the score too
   favorable.
3. Check whether final split gates accepted because image cost improved while
   overlap or valley evidence was too permissive.
4. Compare candidate graph selected rows against unselected rows for the same
   parent and candidate IDs.

## Current Behavior Summary

```text
Placement:
  CellLumen center-prior move is tried first.
  If rejected, normal random/local perturbation still runs.

Splitting:
  CellLumen creates deterministic daughter-center priors.
  Future CellLumen centers can support or penalize those priors.
  Global selection picks a non-conflicting set of priors.
  Broad random split scheduling is disabled in the active config.
  Selected priors still must pass CellUniverse split validation.
```

