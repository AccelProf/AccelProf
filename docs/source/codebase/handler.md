# Event Handler

The **Event Handler** module in AccelProf is responsible for capturing and translating runtime events from multiple sources:

- **High-level Deep Learning (DL) frameworks** (e.g., PyTorch)
  - Example callback: `c10::reportMemoryUsage`
- **Low-level instrumentation APIs**
  - Example: `SANITIZER_CBID_LAUNCH_BEGIN`

These events are then processed by a modular set of **handler functions** such as:

- `AccelProf::tensor_call_handler` for tensor allocations
- `AccelProf::kernel_call_handler` for kernel launch events

This abstraction enables AccelProf to support diverse sources of profiling data while maintaining a unified internal event format.
