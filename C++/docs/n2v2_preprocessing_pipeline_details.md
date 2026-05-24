# N2V2 Preprocessing Pipeline Details

This document describes the active C++ N2V2 preprocessing path:

```text
raw TIFF stack -> LibTorch/TorchScript N2V2 forward pass -> background subtraction -> contrast/gamma -> TIFF/export or tracking stack
```

The executable does not train a model. Training and checkpoint creation remain
in Python/CAREamics. C++ loads a previously exported TorchScript model.

## Frame Pipeline

For each frame:

1. Read a TIFF stack with OpenCV.
2. Convert slices to `CV_32F`.
3. Compute a per-stack intensity scale from the nonzero `99.9` percentile.
4. Normalize raw pixels to `[0, 1]`.
5. Run the N2V2 TorchScript model slice by slice with tiled inference when
   `enable_network: true`.
6. Restore the original intensity scale.
7. Subtract a low-background percentile from the denoised stack.
8. Optionally quantize before contrast to match the Python compatibility path.
9. Apply contrast limits and gamma.
10. Return the processed stack to `PreprocessingHandler` or export it from the
    standalone `celluniverse_preprocess` executable.

## Configuration

Standalone runs use `config/n2v2_preprocess.yaml`. Integrated CellUniverse runs
use the nested `simulation.n2v2_preprocess` block in `config/config.yaml`.

```yaml
n2v2_preprocess:
  enable_network: true
  model_path: "models/general_n2v2_torchscript.pt"
  device: auto
  inference_batch_size: 16
  tile_size: [256, 256]
  tile_overlap: [48, 48]

  scale_percentile: 99.9
  use_nonzero_pixels: true
  fallback_scale: 65536
  careamics_mean: 0.33040523529052734
  careamics_std: 0.23699833071884863

  background_subtraction:
    enabled: true
    percentile: 20.0
    exclude_zero: true
    clip_min: 0.0

  contrast:
    enabled: true
    limit_mode: percentile
    low_limit: 10.0
    low_percentile: 75.0
    high_percentile: 99.99
    exclude_zero: true
    scope: stack
    gamma: 1.45
    preserve_zero_pixels: true

  output:
    dtype: preserve
    write_intermediate: false
    quantize_before_contrast: true
```

Important options:

- `enable_network`: when `true`, run the TorchScript N2V2 model. When `false`,
  skip the neural network and run only the classical postprocessing steps.
- `model_path`: path to the exported `.pt` model.
- `device`: `auto`, `cpu`, `cuda`, or `cuda:<index>`.
- `inference_batch_size`: number of tiles sent through the model at once.
- `tile_size`: fixed model tile size. The default exporter uses `256x256`.
- `tile_overlap`: overlap used to crop/stitch tiles without blending.
- `scale_percentile`: per-stack raw intensity scale before N2V2 normalization.
- `careamics_mean` and `careamics_std`: CAREamics normalization statistics
  learned from the prepared training patches.
- `background_subtraction.percentile`: nonzero percentile subtracted from every
  pixel after N2V2 scale restoration.
- `contrast.low_percentile` and `contrast.high_percentile`: contrast limits
  computed with NumPy-style linear percentile interpolation.
- `output.quantize_before_contrast`: keeps compatibility with the Python
  two-stage behavior by rounding/casting the N2V2/background-subtracted result
  before contrast.

## TorchScript Export

Export the model from a CAREamics checkpoint:

```bash
python3 scripts/export_n2v2_torchscript.py \
  /path/to/general_n2v2_last.ckpt \
  models/general_n2v2_torchscript.pt
```

The model receives fixed `[N, 1, 256, 256]` tiles. The C++ code handles the
normalization around the model:

```text
raw stack -> divide by percentile scale -> clip to [0, 1]
normalized tile -> (tile - careamics_mean) / careamics_std
model output -> output * careamics_std + careamics_mean
clip to [0, 1] -> multiply by percentile scale
```

## Build

Legacy-only build:

```bash
cmake -S . -B build -DCELLUNIVERSE_BUILD_N2V2_PREPROCESS=OFF
cmake --build build -j "$(nproc)"
```

LibTorch-enabled build:

```bash
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH=/path/to/libtorch
cmake --build build -j "$(nproc)"
```

If LibTorch is not found, the project still builds. Selecting N2V2 reports a
runtime error explaining that LibTorch support is required.
