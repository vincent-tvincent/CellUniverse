# N2V2 Raw-Before-Normalization Contract

Date: 2026-05-30

The N2V2 model is treated as a raw TIFF model. This is the only supported order for `simulation.preprocess_mode: n2v2`, for both original CellUniverse tracking and CellLumen analysis:

```text
raw TIFF stack -> N2V2 -> normalize N2V2 output -> external_preprocessed_sequence -> z interpolation/cube pooling -> downstream analysis
```

This preserves the model input contract from `yp_0520_merge_debug_05212926` and avoids feeding CellUniverse-normalized images into a model trained/designed for raw TIFF intensity scale.

Implementation notes:

- `ImageHandler::loadFrame` sends any `preprocess_mode: n2v2` run directly through the raw-before-N2V2 path.
- The previous CellLumen exclusion was removed, so CellLumen uses the same N2V2 model contract.
- The normalized-before-N2V2 path has been removed. Calling `ImageHandler::preprocessLoadedFrame` with N2V2 enabled now fails fast, because that API only has access to an already-loaded runtime stack.
