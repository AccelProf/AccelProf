[![](./assets/banner.jpg)](https://github.com/AccelProf)
------------------------------------------------------------
[![](https://img.shields.io/badge/license-MIT-green?logo=github)](https://github.com/AccelProf/AccelProf/blob/main/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/accelprofdocs/badge/?version=latest)](https://accelprofdocs.readthedocs.io/en/latest/)
[![Static Badge](https://img.shields.io/badge/build-passing-brightgreen?logo=Linux&label=Linux)](https://github.com/AccelProf/AccelProf)
[![Static Badge](https://img.shields.io/badge/build-passing-brightgreen?logo=NVIDIA&label=NVIDIA)](https://github.com/AccelProf/AccelProf)
[![Static Badge](https://img.shields.io/badge/build-passing-brightgreen?logo=AMD&label=AMD)](https://github.com/AccelProf/AccelProf)

# AccelProf

A Modular Program Analysis Tool Framework for Emerging Accelerators.

## Overview

**AccelProf** is a modular program analysis framework for accelerator workloads spanning NVIDIA CUDA, AMD ROCm, and modern deep-learning systems. It abstracts over heterogeneous profiling APIs and deep-learning frameworks, providing a unified interface for capturing and analyzing runtime events at multiple levels. Its extensible architecture enables researchers and practitioners to rapidly prototype custom analysis tools with minimal overhead.

## Installation

```bash
# Download
git clone --recursive https://github.com/AccelProf/AccelProf.git
git submodule update --init --recursive

# Check dependences
bash ./bin/utils/check_build_env.sh

# Build and install
./bin/build

# Set env
export ACCEL_PROF_HOME=$(pwd)
export PATH=${ACCEL_PROF_HOME}/bin:${PATH}
```

## Basic Usage

Analyze an accelerator application:

```bash
accelprof -v -t app_analysis <executable> [args...]
```

## Documentation

Full user and developer documentation:
👉 **[https://accelprofdocs.readthedocs.io](https://accelprofdocs.readthedocs.io)**

## Paper

- **[CGO’26]** *PASTA: A Modular Program Analysis Tool Framework for Emerging Accelerators.*  
  Mao Lin, Hyeran Jeon, and Keren Zhou.  
  Proceedings of the 23rd ACM/IEEE International Symposium on Code Generation and Optimization (CGO 2026).

## License

Released under the **MIT License**.
See `LICENSE` for details.
