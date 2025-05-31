# Utilities

To enable seamless integration with target applications, **AccelProf** is built as a shared library and injected at runtime using the `LD_PRELOAD` mechanism. This setup allows AccelProf to intercept both deep learning framework calls and accelerator runtime calls.

---

## Event Capture Mechanism

Once loaded, AccelProf activates the underlying event capture infrastructure through **vendor-specific APIs**, such as:

- `sanitizerEnableDomain` (Compute Sanitizer)
- `nvbit_at_cuda_event` (NVIDIA NVBit)
- `rocprofiler_configure_callback` (AMD ROCm ROCProfiler)

These APIs initialize and register the profiling hooks required to monitor accelerator behavior.

---

## Callback Integration

For every intercepted event, AccelProf uses a dedicated **callback implementation** to route data into the event handler system. This consistent routing mechanism enables modular and extensible event processing.

---

## Developer Introspection & Customization

AccelProf supports a rich set of features to enable **developer-level introspection** and **cross-language integration**:

- 🧩 `pybind11`: Enables Python-side annotations and tool development
- 🐍 `PyFrame` (CPython): Captures Python-level call stacks
- ⚙️ `libbacktrace`: Extracts symbolic stack traces for C/C++ applications

These integrations allow users to analyze runtime behavior with high fidelity, offering both framework-level and instruction-level views. The result is a powerful profiling platform for debugging, performance tuning, and workload characterization.

