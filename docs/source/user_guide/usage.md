# Usage Guide for AccelProf

AccelProf is a flexible command-line tool designed for profiling GPU applications. It supports a variety of tools to collect metrics, trace memory, and analyze deep learning workloads.

---

## Basic Usage

AccelProf can be invoked directly from the terminal to analyze an application using pre-built or custom tools.

### Command Syntax

```bash
accelprof -v -t app_analysis {executable} {executable arguments}
```

### Example: Profiling `vectoradd`

Run the `vectoradd` example using the `app_metric` tool:

```bash
accelprof -v -t app_metric ./vectoradd
```

#### Output Files

1. **Profiling Log: `vectoradd.accelprof.log`**

This file logs the profiling session metadata and runtime activity:

```text
[ACCELPROF INFO] VERSION      : c6d15f78b30385ba30ee64ab47db1b9b4729d16c, modified 0
[ACCELPROF INFO] LD_PRELOAD   : /home/mao/AccelProf/lib/libcompute_sanitizer.so
[ACCELPROF INFO] OPTIONS      :  -v -t app_metric
[ACCELPROF INFO] COMMAND      : ./vectoradd 
[ACCELPROF INFO] START TIME   : Fri May 30 04:47:08 PM PDT 2025
...
[SANITIZER INFO] Free memory 0x7fb4e9400000 with size 800000 (flag: 0)
Dumping traces to vectoradd_2025-05-30_16-47-09.log
[ACCELPROF INFO] END TIME     : Fri May 30 04:47:09 PM PDT 2025
[ACCELPROF INFO] ELAPSED TIME : 00:00:01
```

2. **Analysis Results: `vectoradd_xxxx.log`**

This file contains key memory and kernel statistics:

```text
Alloc(0) 0:	140414982029312 800000 (781.25 KB)
Alloc(0) 1:	140414982829568 800000 (781.25 KB)
...
Maximum memory accesses kernel: vecAdd(double*, double*, double*, int) (Kernel ID: 0)
Maximum memory accesses per kernel: 300000 (300.00 K)
Average memory accesses per kernel: 300000 (300.00 K)
Total memory accesses: 600000 (600.00 K)
Average accesses per page: 1024
```

---

## Advanced Usage

To explore all command-line options supported by AccelProf, run:

```bash
accelprof -h
```

### Help Output

```text
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

---

## Use Different Tools

AccelProf supports multiple profiling modes:

### Metrics Collection

```bash
accelprof -v -t app_metric ./vectoradd
```

### Hot Memory Region Analysis

```bash
accelprof -v -t hot_analysis ./vectoradd
```

### Application Performance Analysis

```bash
accelprof -v -t app_analysis ./vectoradd
```

---

## Use Different Backends/Vendors

AccelProf can use various vendor APIs as profiling backends:

### NVIDIA Compute Sanitizer (Default)

```bash
accelprof -v -t app_analysis ./vectoradd
```

### NVIDIA NVBit

```bash
accelprof -v -d nvbit -t app_analysis ./vectoradd
```

### AMD ROCProfiler

```bash
accelprof -v -d rocm -t app_analysis ./vectoradd
```

---

## Customized Range Inspection

AccelProf allows fine-grained instrumentation by letting users specify code regions to analyze using lightweight Python APIs.

### Example: `test.py`

```python
import accelprof

accelprof.start()
# Insert target analysis code here
accelprof.end()
```

Use this pattern to isolate specific function calls, training loops, or inference paths for analysis.

---

This flexible usage model enables both high-level and low-level insight into application behavior across platforms.