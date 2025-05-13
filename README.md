# AccelProf


## Quick Start

* Installation

```shell
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

* Basic Usage

```shell
# analyze the accelerator applications
accelprof -v -t app_analysis {executable} {excutatble args}
```