# Event Processor

The **Event Processor** module serves as the intermediary stage that **preprocesses raw event data** collected by the Event Handler. It applies custom transformation and analysis routines using modular processor functions such as:

- `AccelProf::tensor_info_process`
- `AccelProf::kernel_info_process`

This design supports both CPU- and GPU-based data preprocessing, allowing:

- **Normalization** of event fields
- **Filtering and aggregation** of traces
- **Device-side analysis**, enabled through GPU offloading for high-throughput scenarios

GPU-based preprocessing is supported via patched APIs such as `sanitizerPatchModule`, allowing device-resident functions (marked with `__device__`) to process large volumes of fine-grained traces efficiently.
