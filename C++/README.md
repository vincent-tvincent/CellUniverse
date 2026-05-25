# 3D CellUniverse CPP-VERSION

## Quick Start
### Package Required
- this project require package yaml-cpp
- option1: pull the package to local directory (default)
    - In the terminal type in `cd 3d/C++` to go to the root of C++ CellUniverse, then add the external YAML-CPP library
      ``` bash
        mkdir lib && cd lib
        git clone https://github.com/jbeder/yaml-cpp
      ```
- option2: install it through package manager such as homebrew(MacOS) or apt(Ubuntu), make it detect the one on this computer, no need for pull from github
  ```cmake
  # commend out the line bellow in CmakeLists.txt
   add_subdirectory(lib/yaml-cpp)
  
  # uncommend the line bellow in CmakeLists.txt
   find_package(yaml-cpp REQUIRED)
  ```

## Build Modes

The main tracker can be built in two modes:

- **Without LibTorch:** builds the legacy preprocessing pipeline and the main `celluniverse` executable. This is the default path for machines that only need the original preprocessing.
- **With LibTorch:** enables the N2V2 neural-network preprocessing path in `celluniverse` and builds the standalone `celluniverse_preprocess` and `n2v2_preprocess_unit_test` targets.

### Build Without LibTorch

Use this when running the existing legacy preprocessing pipeline:

```bash
cmake -S . -B build -DCELLUNIVERSE_BUILD_N2V2_PREPROCESS=OFF
cmake --build build -j "$(nproc)"
```

`config/config.yaml` defaults to:

```yaml
simulation:
  preprocessing_pipeline: legacy
```

This build will not run `preprocessing_pipeline: n2v2` with `enable_network: true`. If that mode is selected in the YAML, the executable reports that LibTorch support is required.

### Install LibTorch For N2V2

LibTorch is only required when using the N2V2 neural-network preprocessing
pipeline. The helper script below detects the current machine and installs the
most appropriate current LibTorch package under `external/libtorch`:

```bash
scripts/install_libtorch.sh
```

Auto-selection:

```text
Apple Silicon macOS                         -> macOS arm64 LibTorch
Linux x86_64 with NVIDIA hardware + nvcc    -> Linux CUDA LibTorch
Linux x86_64 without CUDA setup             -> Linux CPU LibTorch
```

The CUDA path is selected when NVIDIA hardware is detected and `nvcc` is
available on `PATH` or under `/usr/local/cuda*`. If the NVIDIA driver is not
currently usable, the script still installs CUDA LibTorch but prints a warning;
GPU runtime will need the driver fixed before `device: cuda` works.

Then configure and build. If `external/libtorch` is missing, CMake runs
`scripts/install_libtorch.sh` automatically. If it already exists, CMake uses
it directly:

```bash
JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)"
cmake -S . -B build
cmake --build build -j "$JOBS"
```

To skip automatic LibTorch installation, configure with:

```bash
cmake -S . -B build -DCELLUNIVERSE_AUTO_INSTALL_LIBTORCH=OFF
cmake --build build -j "$JOBS"
```

If LibTorch is installed somewhere else, pass it explicitly and disable the
local preference:

```bash
cmake -S . -B build \
  -DCELLUNIVERSE_PREFER_LOCAL_LIBTORCH=OFF \
  -DCMAKE_PREFIX_PATH="/path/to/libtorch"
cmake --build build -j "$JOBS"
```

### Manual install LibTorch (if auto script not wroking)

Manual install is also supported. Download the standalone C++/Java LibTorch
archive for your operating system from: https://pytorch.org/get-started/locally/


For the most portable setup, choose the CPU LibTorch package. CUDA LibTorch can
also be used, but it must match the CUDA/NVIDIA driver setup on that computer.
After downloading, unpack it outside git-tracked source files:

```bash
mkdir -p external
unzip libtorch-*.zip -d external
```

This creates:

```text
external/libtorch/
```

Then use the same CMake command shown above. If you unpacked to
`external/libtorch`, no extra CMake flag is needed.

Useful installer overrides:

```bash
# Force CPU even on a CUDA machine
scripts/install_libtorch.sh --target cpu

# Show what would be downloaded without installing
scripts/install_libtorch.sh --dry-run

# Force CUDA and choose a specific PyTorch CUDA package family
scripts/install_libtorch.sh --target cuda --cuda-tag cu126

# Replace an existing external/libtorch install
scripts/install_libtorch.sh --force
```

### Export The TorchScript Model

The C++ code consumes a TorchScript `.pt` file. Export it once from the CAREamics
checkpoint:

```bash
python3 scripts/export_n2v2_torchscript.py \
  /path/to/checkpoint.ckpt \
  models/general_n2v2_torchscript.pt
```

### Enable N2V2 In The Runtime Config

After building with LibTorch and exporting the model, switch the main
pipeline in `config/config.yaml`:

```yaml
simulation:
  preprocessing_pipeline: n2v2
  n2v2_preprocess:
    model_path: "models/general_n2v2_torchscript.pt"
    device: auto
```

To build only the N2V2-related targets after configuring with LibTorch:

```bash
cmake --build build -j "$(nproc)" --target celluniverse_preprocess n2v2_preprocess_unit_test
```

## Live Napari Monitor

Use `scripts/live_monitor_napari.py` to watch a running CellUniverse output
folder. The `--interval` value is in minutes, so `2.5` means refresh every two
and a half minutes:

```bash
python3 scripts/live_monitor_napari.py /path/to/output_run --interval 2.5
```

For a run launched from `scripts/run_celluniverse.sh`, the output path is printed
near startup as `Output Path`. Example:

```bash
python3 scripts/live_monitor_napari.py \
  /run/media/blue-lobster/disk3/celluniverse_output/outputs_fluo/output_ubuntu_fluo_0-150_YYYYMMDD_HHMMSS \
  --interval 2.5
```

The monitor expects TIFF export and perturb debug export folders:

```text
tiff/real
tiff/synth
perturb_debug/split_placements
perturb_debug/movement_placements
perturb_debug/cell_centers
```

Enable the corresponding config options before starting the run:

```yaml
simulation:
  export_frame_tiff: true
  export_perturb_debug_images: true
  export_perturb_cell_center_debug_images: true
```

### Using ICS openlab
- ssh into your ICS openlab. (Recommend to use vscode remote development tool: https://code.visualstudio.com/docs/remote/ssh)
    - Notice you should ssh into the circinus-28 machine to avoid dependency issue (ssh <netid>@circinus-28.ics.uci.edu)
- clone the repo and cd into the project folder
- create build directory and compile the project using cmake (assume current directory is C++/lib created in previous step)
  ```bash
    cd ..
    mkdir build && cd build
    cmake -S .. -B .
    cmake --build . -j $(nproc) // build with all available 
  ```
- Run the bash script in the examples folder
  ``` zsh
   cd ../../scripts
  # quick test run
  ./run_cell_universe.sh user_input_configurations.ini default 
  ```
- If want to select the configuration profile through the Cli UI, run it without parameters
  ``` zsh
   cd ../../scripts
  # select the configuration through cil UI
  ./run_cell_universe.sh
  ```
    - Change the mode of the bash script if you don't have the permission by running the following command
  ```zsh
  chmod 755 runcpp.sh
  ```
- Wait for celluniverse to finish executing and you can check the result in the `output/` folder inside the current
  example

### Using Mac OS:
- clone the repo and cd into the project folder
- In the terminal type in `cd C++` to go to the root of C++ CellUniverse
- Add the external YAML-CPP library
  ```zsh
  mkdir lib && cd lib
  git clone https://github.com/jbeder/yaml-cpp
  ```

- download cmake and install
- use the following command to set make path:
  ```zsh
  PATH="/Applications/CMake.app/Contents/bin":"$PATH"
  ```
- install OpenCV using commend bellow (homebrew required)
  ```zsh
   brew install opencv
  ```
- create build directory and compile the project into `build` directory using cmake (assume current directory is C++/lib created in previous steps)
  ```zsh
  cd ..
  mkdir build && cd build
  cmake -S .. -B .
  cmake --build .
  ``` 


- Run the bash script in the scripts folder
  ``` zsh
  cd ../../scripts
  # quick test run
  ./run_cell_universe.sh user_input_configurations.ini default 
  ```
- If want to select the configuration profile through the Cli UI, run it without parameters
  ``` zsh
  cd ../../scripts
  # select the configuration through cil UI
  ./run_cell_universe.sh -i
  ```
    - Change the mode of the bash script if you don't have the permission by running the following command
  ```zsh
  chmod 755 runcpp.sh
  ```

- Wait for celluniverse to finish executing and you can check the result in the `outputs/` folder
