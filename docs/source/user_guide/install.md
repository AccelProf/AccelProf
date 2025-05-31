# Installation

This section provides detailed instructions for installing **AccelProf**.

## Requirements

AccelProf has both hardware and software prerequisites:

### Hardware Requirements

- GPU (NVIDIA or AMD)
- x86 systems tested (ARM systems are likely compatible but untested)

### Software Requirements

- CUDA (Tested on version 12.1; older versions may also work)
- ROCm (Tested on version 6.3; older versions may also work)
- PyTorch (Tested on version 2.1; older versions may also work)

## Build Instructions

To compile AccelProf, follow the steps below.

### Download the Code

```shell
git clone --recursive https://github.com/AccelProf/AccelProf.git

cd AccelProf
export ACCEL_PROF_DIR=$(pwd)
```

### Compile

AccelProf can be built using an all-in-one build script, or through step-by-step compilation for easier debugging.

#### Quick Build with Integrated Script

```shell
./bin/build
```

#### Step-by-Step Compilation

**1. Build libbacktrace**

```shell
cd $ACCEL_PROF_DIR/sanalyzer/cpp_trace
if [ ! -d "$ACCEL_PROF_DIR/build/backtrace" ]; then
    source ${ACCEL_PROF_DIR}/bin/utils/build_libbacktrace
fi
```

**2. Build `cpp_trace`**

```shell
cd $ACCEL_PROF_DIR/sanalyzer/cpp_trace
make -j install DEBUG=$DEBUG BACKTRACE_DIR=$ACCEL_PROF_DIR/build/backtrace \
                             INSTALL_DIR=$ACCEL_PROF_DIR/build/sanalyzer/cpp_trace
```

**3. Build `py_frame`**

```shell
cd $ACCEL_PROF_DIR/sanalyzer/py_frame
make -j install DEBUG=$DEBUG PYBIND11_DIR=$ACCEL_PROF_DIR/third_party/pybind11 \
                             INSTALL_DIR=$ACCEL_PROF_DIR/build/sanalyzer/py_frame
```

**4. Build `sanalyzer`**

```shell
cd $ACCEL_PROF_DIR/sanalyzer
make -j install DEBUG=$DEBUG SANITIZER_TOOL_DIR=$ACCEL_PROF_DIR/nv-compute \
                NV_NVBIT_DIR=$ACCEL_PROF_DIR/nv-nvbit \
                CPP_TRACE_DIR=$ACCEL_PROF_DIR/build/sanalyzer/cpp_trace \
                PY_FRAME_DIR=$ACCEL_PROF_DIR/build/sanalyzer/py_frame \
                INSTALL_DIR=$ACCEL_PROF_DIR/build/sanalyzer
```

**5. Build `tensor_scope`**

```shell
cd $ACCEL_PROF_DIR/tensor_scope
make -j install DEBUG=$DEBUG INSTALL_DIR=$ACCEL_PROF_DIR/build/tensor_scope
```

**6. Build `nv-compute`**

```shell
cd $ACCEL_PROF_DIR/nv-compute
make -j DEBUG=$DEBUG SANALYZER_DIR=$ACCEL_PROF_DIR/build/sanalyzer \
                TORCH_SCOPE_DIR=$ACCEL_PROF_DIR/build/tensor_scope \
                PATCH_SRC_DIR=$ACCEL_PROF_DIR/nv-compute/gpu_src

cp $ACCEL_PROF_DIR/nv-compute/lib/*.so $ACCEL_PROF_DIR/lib/
```

**7. Build `nv-nvbit`**

```shell
cd $ACCEL_PROF_DIR/nv-nvbit
make -j DEBUG=$DEBUG SANALYZER_DIR=$ACCEL_PROF_DIR/build/sanalyzer \
                     TORCH_SCOPE_DIR=$ACCEL_PROF_DIR/build/tensor_scope

cp $ACCEL_PROF_DIR/nv-nvbit/lib/*.so $ACCEL_PROF_DIR/lib/
```

**8. Build `amd-rocm` (Optional, Requires AMD ROCm Development Environment)**

```shell
cd $ACCEL_PROF_DIR/amd-rocm
make -j DEBUG=$DEBUG SANALYZER_DIR=$ACCEL_PROF_DIR/build/sanalyzer

cp $ACCEL_PROF_DIR/amd-rocm/build/lib/*.so $ACCEL_PROF_DIR/lib/
```

## Configuration

To configure the environment, run the following commands, or append them to your `~/.bashrc` file:

```shell
export ACCEL_PROF_HOME=$(pwd)/AccelProf
export PATH=${ACCEL_PROF_HOME}/bin:${PATH}
```

## Quick Test

After compilation, you should find multiple `*.so` files under the `lib` directory. You can run example workloads to verify the installation. Please refer to the **Usage** section for more details.