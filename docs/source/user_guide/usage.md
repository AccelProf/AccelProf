# Usage

## Basic Usage

AccelProf is a command-line tool. Users can use it to perform program analysis with pre-defined or user-defined tools.

### Command Line

```shell
accelprof -v -t app_analysis {executable} {executable args}
```

### Profile `vectoradd` Example

#### Profile the `vectoradd`

```shell
accelprof -v -t app_metric ./vectoradd
```

#### Outputs:

1. Profiling log file `vectoradd.accelprof.log`:

```shell
[ACCELPROF INFO] VERSION      : c6d15f78b30385ba30ee64ab47db1b9b4729d16c, modified 0
[ACCELPROF INFO] LD_PRELOAD   : /home/mao/AccelProf/lib/libcompute_sanitizer.so
[ACCELPROF INFO] OPTIONS      :  -v -t app_metric
[ACCELPROF INFO] COMMAND      : ./vectoradd 
[ACCELPROF INFO] START TIME   : Fri May 30 04:47:08 PM PDT 2025

[SANITIZER INFO] Enabling app_metric tool.
================================================================================
[SANITIZER INFO] Context 0x7ffd21f315a0 creation starting on device 0x7ffd21f315a8
[SANITIZER INFO] Stream 0x7ffd21f314a8 created on context 0x7ffd21f314a0
...
[SANITIZER INFO] Free memory 0x7fb4e9400000 with size 800000 (flag: 0)
Dumping traces to vectoradd_2025-05-30_16-47-09.log
[ACCELPROF INFO] END TIME     : Fri May 30 04:47:09 PM PDT 2025
[ACCELPROF INFO] ELAPSED TIME : 00:00:01
```

2. Result file `vectoradd_xxxx.log`:

```shell
Alloc(0) 0:	140414982029312 800000 (781.25 KB)
Alloc(0) 1:	140414982829568 800000 (781.25 KB)
Alloc(0) 2:	140414984126464 800000 (781.25 KB)
...
Maximum memory accesses kernel: vecAdd(double*, double*, double*, int) (Kernel ID: 0)
Maximum memory accesses per kernel: 300000 (300.00 K)
Average memory accesses per kernel: 300000 (300.00 K)
Total memory accesses: 600000 (600.00 K)
Average accesses per page: 1024
```

## Advanced Usage

This subsection shows more command-line options.

```shell
$ accelprof -h
```

### Output:

```shell
Description: A collection of CUDA application profilers.
Usage:
    -h, --help
        Print this help message.
    -t <tool_name>
        none: Do nothing.
        mem_trace: Trace memory access of CUDA kernels.
        app_metric: Collect metrics for CUDA Applications.
        code_check: Check CUDA code for potential issues.
        hot_analysis: Analyze hot memory regions.
        app_analysis: Analyze CUDA application performance.
        app_analysis_cpu: Analyze CUDA application performance on CPU.
        time_hotness_cpu: Analyze time hotness on CPU.
        uvm_advisor: Analyze UVM memory access patterns.
    -d <device_name>
        Specify the device vendor name.
        Valid values:
            - nvc (NVIDIA Compute Sanitizer, default)
            - nvbit (NVIDIA NVBit)
            - rocm (AMD)
    -v
        Verbose mode.
```

### Use Different Tools

#### Collect metrics for CUDA Applications

```shell
accelprof -v -t app_metric ./vectoradd
```

#### Analyze hot memory regions

```shell
accelprof -v -t hot_analysis ./vectoradd
```

#### Analyze CUDA application performance

```shell
accelprof -v -t app_analysis ./vectoradd
```

### Use Different Backends/Vendors

#### NVIDIA Compute Sanitizer API (default)

```shell
accelprof -v -t app_analysis ./vectoradd
```

#### NVIDIA NVBit API

```shell
accelprof -v -d nvbit -t app_analysis ./vectoradd
```

#### AMD ROCProfiler SDK

```shell
accelprof -v -d rocm -t app_analysis ./vectoradd
```

### Customized Range Inspection

AccelProf allows users to inspect only a subset of the application by using lightweight APIs provided by AccelProf.

#### Example User Code `test.py`:

```python
import accelprof

accelprof.start()
# target analysis code
accelprof.end()
```

