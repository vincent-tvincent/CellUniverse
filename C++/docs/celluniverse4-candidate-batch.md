# CellUniverse 4.0 candidate-batch pipeline

CellUniverse 4.0 is an opt-in profile derived from the traditional pipeline.
It keeps the traditional preprocessing, renderer, cost function, background
model, split validator, final PCA fitting, checkpointing, and export. It
replaces the traditional sequential continuation loop with a bounded,
deterministic center-candidate batch followed by a small traditional
refinement budget.

## Selecting the mode

Use `config/config_celluniverse4.yaml` as the run configuration. That profile
enables both required switches:

```yaml
simulation:
  celluniverse4_enabled: true

candidate_batch:
  enabled: true
```

The generic `config/config.yaml` remains traditional and does not enable CU4.
CU2, CU3, CellLumen, and CU4 are mutually exclusive; invalid combinations are
rejected while loading configuration.

Example invocation:

```text
celluniverse FIRST LAST INPUT_DIR OUTPUT_DIR \
  config/config_celluniverse4.yaml INITIAL_CSV
```

## Per-frame execution contract

For every carried parent cell, CU4 performs these steps:

1. Convert the current frame's connected-component signal centers into
   immutable `ChunkEvidence`. Weighted, geometric, robust, and peak centers
   remain separate.
2. Build one `CandidateBatch` from a frozen baseline. Candidate zero is always
   the unchanged/no-op state. Eligible snapshot and chunk-center hypotheses
   are deduplicated and cheaply ranked; optional stochastic candidates use a
   keyed deterministic stream.
3. Render only the configured top-K center moves. Forced center evaluation
   copies the baseline cell and changes only its position, so radii, rotation,
   and brightness are frozen. Every evaluated candidate is rolled back.
4. Select the best candidate only if it beats no-op by the configured margin.
   Re-evaluate the winner from the baseline and atomically commit it only when
   the repeated cost agrees within the configured tolerance.
5. Run the YAML-bounded traditional perturbation refinement around the chosen
   center. Final shared PCA fitting remains a later pipeline stage.
6. Schedule at most one split opportunity per carried parent. The default
   50-iteration-equivalent keyed draw preserves the traditional opportunity
   scale. A YAML-tunable 0.05 per-opportunity urgency threshold forces a single
   attempt for strong division candidates, preventing a supported first-frame
   division from being lost to an unlucky draw. Every opportunity goes through
   `Frame::trySplitCellPhased`; CU4 does not provide an alternate
   split-acceptance path.

The default CU4 profile uses a 1,000-unit absolute split-improvement floor plus
the existing proportional floor. This prevents a deterministic schedule from
depending on random omission to suppress marginal split candidates. Dataset
profiles may override the existing `prob.split_cost` and
`prob.split_cost_fraction` fields.

## YAML controls

The `candidate_batch` block exposes every CU4 proposal and scheduling value:

- activation, shadow mode, and diagnostic export;
- total, snapshot, exact-center, top-K, refinement, and stochastic budgets;
- absolute/fractional improvement margins and winner re-evaluation tolerance;
- a separate post-center refinement improvement margin that rejects
  machine-epsilon zero-cost moves;
- candidate deduplication and parent/chunk association distances;
- chunk size, brightness, confidence, and box-count filters;
- confidence weights, size saturation, compactness exponent, and cheap-priority
  weights/biases;
- stochastic radius limits;
- split scheduling mode and reference iteration count;
- deterministic seed and namespace;
- individual weighted/geometric/robust/peak evidence-source switches.

All distances that depend on cell size have both an absolute term and a
minimum-radius-scaled term. Signed priority biases are allowed, but every
numeric value is checked for finiteness and range validity during config load.

`CELLUNIVERSE_SEED` overrides the process RNG seed when set. Otherwise CU4 uses
`candidate_batch.deterministic_seed`, making candidate generation, split
scheduling, and the traditional refinement RNG reproducible for the same
inputs and configuration.

## Diagnostics

The run log reports one `CellUniverse4 Candidate Batch Summary` per frame with
parent, chunk, proposal, expensive-evaluation, committed/no-op, and refinement
counts. Individual evaluated candidates and winners use the existing
perturb-candidate diagnostic export. `CellUniverse4 Split Schedule` records
the per-opportunity probability, probability-equivalent parent probability,
keyed draw, and decision.

Candidate diagnostic IDs are derived from the seed namespace, frame, parent,
candidate source, evidence ID, and source ordinal. They are therefore stable
for the same input and configuration, unique within a parent batch, and do not
depend on proposal insertion order.

## Validated default budget

The default remains top-K 6 with six post-center refinement attempts per
parent. On the Pos0 frames 0-12 validation window, this C6 profile preserved
the traditional 6 -> 8 -> 9 lineage and the known divisions at frames 6 and
10. It used 825 continuation renders instead of the traditional loop's 4,600
perturb calls, an 82.07% reduction, and completed in 16:25 versus 20:08 for the
traditional control.

A C3 experiment (top-K 3 and three refinements) preserved the same division
count and reduced continuation renders by 88.83%, but cell 20's center diverged
37.4 pixels from C6 by frame 12 while wall time improved by only 13 seconds.
C3 remains an experiment profile rather than the default.

## Current implementation boundary

The enabled implementation covers the one-parent/many-candidate continuation
contract and reuses the authoritative traditional split path. It does not yet
implement threshold-hierarchy graphs, chunk-support objective terms, or joint
cross-parent continuation/split ownership. Chunk evidence is therefore a soft
proposal source, not a segmentation truth label, and the normal image,
geometry, overlap, lineage, and split gates remain authoritative.
