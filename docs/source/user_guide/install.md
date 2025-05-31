# Installation Guide for AccelProf

This guide walks you through the installation of **AccelProf**, a customizable GPU profiling framework. It covers system requirements, build instructions, environment setup, and a basic test to verify successful installation.

---

## Requirements

AccelProf requires both hardware and software components to be correctly configured before use.

### Hardware Requirements

- At least one GPU (NVIDIA or AMD)
- x86-based systems are tested and supported (ARM-based systems may also work, though untested)

### Software Requirements

- **CUDA Toolkit** (Tested on version 12.1; older versions may work)
- **ROCm Toolkit** (Tested on version 6.3; older versions may work)
- **PyTorch** (Tested on version 2.1; older versions may work)

---

## Build Instructions

You can either use the integrated build script or perform a manual step-by-step build to gain finer control and debug support.

### Step 1: Download the Code

```bash
git clone --recursive https://github.com/AccelProf/AccelProf.git

cd AccelProf
export ACCEL_PROF_DIR=$(pwd)
```

### Step 2: Quick Build (Recommended for Most Users)

```bash
./bin/build
```

This script builds all major components, sets up dependencies, and copies binaries into appropriate folders.

---

### Step 3: Step-by-Step Build (Advanced Users)

Use these steps if you want to inspect or debug the compilation process.

#### 1. Build `libbacktrace`

```bash
cd $ACCEL_PROF_DIR/sanalyzer/cpp_trace
if [ ! -d "$ACCEL_PROF_DIR/build/backtrace" ]; then
    source ${ACCEL_PROF_DIR}/bin/utils/build_libbacktrace
fi
```

#### 2. Build `cpp_trace`

```bash
cd $ACCEL_PROF_DIR/sanalyzer/cpp_trace
make -j install DEBUG=$DEBUG BACKTRACE_DIR=$ACCEL_PROF_DIR/build/backtrace                              INSTALL_DIR=$ACCEL_PROF_DIR/build/sanalyzer/cpp_trace
```

#### 3. Build `py_frame`

```bash
cd $ACCEL_PROF_DIR/sanalyzer/py_frame
make -j install DEBUG=$DEBUG PYBIND11_DIR=$ACCEL_PROF_DIR/third_party/pybind11                              INSTALL_DIR=$ACCEL_PROF_DIR/build/sanalyzer/py_frame
```

#### 4. Build `sanalyzer`

```bash
cd $ACCEL_PROF_DIR/sanalyzer
make -j install DEBUG=$DEBUG SANITIZER_TOOL_DIR=$ACCEL_PROF_DIR/nv-compute                 NV_NVBIT_DIR=$ACCEL_PROF_DIR/nv-nvbit                 CPP_TRACE_DIR=$ACCEL_PROF_DIR/build/sanalyzer/cpp_trace                 PY_FRAME_DIR=$ACCEL_PROF_DIR/build/sanalyzer/py_frame                 INSTALL_DIR=$ACCEL_PROF_DIR/build/sanalyzer
```

#### 5. Build `tensor_scope`

```bash
cd $ACCEL_PROF_DIR/tensor_scope
make -j install DEBUG=$DEBUG INSTALL_DIR=$ACCEL_PROF_DIR/build/tensor_scope
```

#### 6. Build `nv-compute`

```bash
cd $ACCEL_PROF_DIR/nv-compute
make -j DEBUG=$DEBUG SANALYZER_DIR=$ACCEL_PROF_DIR/build/sanalyzer                 TORCH_SCOPE_DIR=$ACCEL_PROF_DIR/build/tensor_scope                 PATCH_SRC_DIR=$ACCEL_PROF_DIR/nv-compute/gpu_src

cp $ACCEL_PROF_DIR/nv-compute/lib/*.so $ACCEL_PROF_DIR/lib/
```

#### 7. Build `nv-nvbit`

```bash
cd $ACCEL_PROF_DIR/nv-nvbit
make -j DEBUG=$DEBUG SANALYZER_DIR=$ACCEL_PROF_DIR/build/sanalyzer                      TORCH_SCOPE_DIR=$ACCEL_PROF_DIR/build/tensor_scope

cp $ACCEL_PROF_DIR/nv-nvbit/lib/*.so $ACCEL_PROF_DIR/lib/
```

#### 8. Build `amd-rocm` (Optional – requires ROCm dev environment)

```bash
cd $ACCEL_PROF_DIR/amd-rocm
make -j DEBUG=$DEBUG SANALYZER_DIR=$ACCEL_PROF_DIR/build/sanalyzer

cp $ACCEL_PROF_DIR/amd-rocm/build/lib/*.so $ACCEL_PROF_DIR/lib/
```

---

## Configuration

To use AccelProf from the terminal or within scripts, set up the environment:

```bash
export ACCEL_PROF_HOME=$(pwd)/AccelProf
export PATH=${ACCEL_PROF_HOME}/bin:${PATH}
```

> 🔁 You can optionally add these lines to your `~/.bashrc` or `~/.zshrc` to persist them across sessions.

---

## Quick Test

Once the build completes, you should see multiple `*.so` files in the `lib/` directory. Run a basic profiling task to verify the installation works. Refer to the **Usage** section for a detailed walkthrough.

