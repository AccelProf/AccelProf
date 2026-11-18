# nv-cupti

AccelProf backend built on CUPTI. No kernel patching, no synchronization:
it reports NVTX ranges, CUDA launches and GPU timestamps to sanalyzer.
The `range_time` tool turns them into "GPU time spent in each range".

## Requirements

* CUDA >= 12.0 (developed against CUDA 13.0 headers; uses `CUpti_ActivityKernel10` on 13.x, `Kernel9` on 12.x)
* Linux

## Build

```bash
export CUDA_PATH=/path/to/cuda-13.0      # must match the CUDA of the app (e.g. torch cu130)
make ENABLE_CUPTI=1
ldd lib/libnv-cupti.so | grep cupti      # check which libcupti it uses
```

CUPTI is taken from `$CUDA_PATH/extras/CUPTI`, or from `$CUDA_PATH` if there is no `extras/CUPTI`.
Override with `make ENABLE_CUPTI=1 CUPTI_PATH=...` if needed.

Build inside the Python environment of the app (e.g. the sglang conda env):
sanalyzer links `libpython` of the `python3` found at build time, and the library is preloaded into the app.

## Run

```bash
ACCELPROF_CUPTI_CAPTURE=profiler_api accelprof -v -d cupti -t range_time python app.py
```

`accelprof -d cupti` does two things:
* `LD_PRELOAD=lib/libnv-cupti.so` (inherited by child processes, e.g. sglang workers)
* `NVTX_INJECTION64_PATH=<the libcupti.so that libnv-cupti.so links to>`.
  NVTX only sends ranges to CUPTI when this is set, and it reads it once, at the first NVTX call.

| Env var | Default | Meaning |
|---|---|---|
| `ACCELPROF_CUPTI_CAPTURE` | `all` | `all`: record the whole run, write files at exit. `profiler_api`: record only between `cudaProfilerStart` and `cudaProfilerStop`, write files at `cudaProfilerStop`. |
| `ACCELPROF_RANGE_ITER_REGEX` | `'Module': 'model\.model'` | A range whose text matches is one iteration. |
| `ACCELPROF_RANGE_GROUP_REGEX` | `'Module': '[^']*\.layers\.(\d+)\.(?:mlp\|feed_forward)\.(gate\|router\|topk)'` | Ranges to report in `groups.csv`. Group 1 = layer, group 2 = component. |
| `ACCELPROF_RANGE_TOKENS_REGEX` | `'Inputs': \[\[(\d+)` | Number of tokens, read from the iteration range text. |

The defaults match sglang's `--enable-layerwise-nvtx-marker` ranges.

With `profiler_api`, call `torch.cuda.synchronize()` before `cudaProfilerStop`:
the flush at Stop does not wait for the GPU. Use `profiler_api` for sglang,
because sglang kills its workers with SIGKILL and nothing runs at exit.

## Output

One set of files per process, in `RESULT_DIR` (`accelprof -o`, default: current directory):

| File | One row per | Main columns |
|---|---|---|
| `<app>.<pid>.range_time.ranges.csv` | range instance | `self_kern_us` (launched directly in it), `total_kern_us` (with child ranges), `cpu_us`, `gpu_span_us`, `name` |
| `<app>.<pid>.range_time.activities.csv` | kernel / memcpy / memset | `range_id` (0 = outside ranges), `has_launch`, `stream_id`, `dur_us`, `short_name` |
| `<app>.<pid>.range_time.iterations.csv` | iteration range | `num_tokens`, `kern_us` |
| `<app>.<pid>.range_time.groups.csv` | grouped range | `iter`, `num_tokens`, `layer`, `component`, `kern_us`, `kernels` |

The log also prints mean / median `kern_us` per (component, num_tokens).
These are the same quantities as `gate_kern_us` / `topk_kern_us` in `moe-router-nsys`.

## How it works

```
app thread:  nvtxRangePushA ──► CUPTI callback ──► yosemite_range_push_callback
             cudaLaunchKernel ─► CUPTI callback ──► yosemite_api_launch_callback(correlation id)
             nvtxRangePop ────► CUPTI callback ──► yosemite_range_pop_callback
CUPTI thread: kernel record (GPU start/end, correlation id) ──► yosemite_gpu_activity_callback
```

* **Callback API** (`cuptiSubscribe`, `cuptiEnableDomain`, `cuptiEnableCallback`): CUPTI calls
  our function on the app thread, during the API call. Used for NVTX push/pop, for every CUDA API
  that puts work on the GPU (picked by name with `cuptiGetCallbackName`: `*Launch*`, `*Memcpy*`,
  `*Memset*`), and for `cudaProfilerStart/Stop`.
* **Activity API** (`cuptiActivityRegisterCallbacks`, `cuptiActivityEnable`): CUPTI writes one record
  per kernel / memcpy / memset into our buffers and returns them later. We use
  `CONCURRENT_KERNEL`, not `KERNEL`: `KERNEL` serializes all kernels.
* **Correlation id**: the launch callback and the kernel record of the same launch carry the same id.
  `range_time` remembers which range was open at the launch and gives the kernel to that range.
  This is the same rule Nsight Systems uses for "kernels in an NVTX range".

## Limitations

* Eager mode only: all kernels of a CUDA graph replay share one correlation id, so they all go
  to the range around `cudaGraphLaunch`, not to the modules that were captured.
* With `profiler_api`, only processes that call `cudaProfilerStart/Stop` record anything
  (in sglang, only TP rank 0).
* NVTX messages must be ASCII (`nvtxRangePushA`, or `Ex` with an ASCII message). Ranges of all
  NVTX domains share one stack per thread.
* Only one CUPTI user per process: do not run together with nsys or torch.profiler.

## Test

See `tests/nv-cupti/`:

```bash
cd tests/nv-cupti
ACCELPROF_CUPTI_CAPTURE=profiler_api accelprof -v -d cupti -t range_time python mlp_ranges.py
python check_expected.py --app mlp_ranges --procs 1

ACCELPROF_CUPTI_CAPTURE=profiler_api accelprof -v -d cupti -t range_time python mlp_ranges_mp.py
python check_expected.py --app mlp_ranges_mp --procs 2
```
