# C++ N2V2 Preprocessing

The C++ N2V2 preprocessing path uses LibTorch for TorchScript inference:

```text
raw TIFF stack -> N2V2 TorchScript inference -> background subtraction -> contrast/gamma -> TIFF
```

It is available in two forms:

- a standalone executable, `celluniverse_preprocess`
- an integrated CellUniverse preprocessing engine selected with
  `simulation.preprocessing_pipeline: n2v2`

`PreprocessingHandler` calls `N2V2Preprocessor` directly for integrated runs.
The legacy `ImageHandler` iterative contrast path is skipped in that mode;
`ImageHandler` is still reused for z interpolation and cube pooling so `Frame`
receives the same post-preprocessing stack shape as before.

## Export The TorchScript Model

The C++ backend consumes a TorchScript `.pt` model. Export it once from the
CAREamics checkpoint:

```bash
python3 scripts/export_n2v2_torchscript.py \
  /path/to/general_n2v2_last.ckpt \
  models/general_n2v2_torchscript.pt
```

The exporter keeps fixed `256x256` tiles by default and writes adjacent JSON
metadata with CAREamics mean/std normalization values. The C++ code applies
those mean/std values outside the model.

## Build

The default project build does not require LibTorch and supports the legacy
preprocessing pipeline:

```bash
cmake -S . -B build -DCELLUNIVERSE_BUILD_N2V2_PREPROCESS=OFF
cmake --build build -j "$(nproc)"
```

To enable neural N2V2 inference, install standalone LibTorch and point CMake at
its root:

```bash
cmake -S . -B build \
  -DCMAKE_PREFIX_PATH=/path/to/libtorch
cmake --build build -j "$(nproc)"
```

The LibTorch directory should contain `share/cmake/Torch/TorchConfig.cmake`.
Use the CPU LibTorch package for the most portable setup; CUDA requires a
CUDA-enabled LibTorch package that matches the target computer.

If LibTorch is not found, the project still builds, but selecting
`simulation.preprocessing_pipeline: n2v2` reports a clear runtime error.

## Run

Folder input uses sorted TIFF order and the inclusive frame range:

```bash
./build/celluniverse_preprocess \
  7 7 \
  /path/to/input_tiffs \
  /tmp/n2v2_cpp_frame007 \
  config/n2v2_preprocess.yaml
```

Printf-style input patterns are also supported:

```bash
./build/celluniverse_preprocess \
  0 20 \
  /path/to/t%03d.tif \
  /tmp/n2v2_cpp_fluo_000_020 \
  config/n2v2_preprocess.yaml
```

Output layout:

```text
<output>/
  images/
  logs/
  intermediate/          # only when write_intermediate is true
  preprocess_summary.csv
```

## Parity Notes

The C++ path preserves the Python compatibility preprocessing behavior around
the model:

- per-stack `99.9` nonzero percentile scale before N2V2
- CAREamics mean/std normalization: `0.33040523529052734`,
  `0.23699833071884863`
- fixed-tile crop/stitch rules with no blending
- background subtraction before intermediate dtype casting
- optional intermediate quantization before contrast, enabled by default
- NumPy-style linear percentile interpolation for contrast limits

For integrated CellUniverse runs, use:

```yaml
simulation:
  preprocessing_pipeline: n2v2
  n2v2_preprocess:
    enable_network: true
    model_path: "models/general_n2v2_torchscript.pt"
    device: auto
```
