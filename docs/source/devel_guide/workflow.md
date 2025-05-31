# Workflow

The core of the PASTA framework—underlying AccelProf—is a modular, stage-based workflow designed for scalable and extensible program analysis. It consists of three major components:

```
Application → [Event Handler] → [Event Processor] → [Tool Collection]
```

Each component plays a distinct role in the profiling pipeline, enabling clean separation of concerns and easy extensibility.

---

## Components

### 🔹 Application

The target workload can be:

- A deep learning model executed via a framework such as **PyTorch**
- A binary GPU-accelerated application using libraries like **cuDNN** or **MIOpen**

This application is the source of events captured during runtime.

---

### 🔹 Event Handler

Responsible for intercepting execution events. It integrates:

- **Low-level profiling APIs**  
  e.g., NVIDIA Compute Sanitizer, AMD ROCm ROCProfiler

- **High-level callbacks**  
  e.g., from DL frameworks like PyTorch

This component abstracts vendor-specific and API-level complexity through a unified internal interface, offering consistent behavior across platforms.

---

### 🔹 Event Processor

The pre-processing stage that prepares collected data for analysis:

- Can be executed on **CPU** or **GPU**
- Transforms raw events into enriched, normalized structures
- Performs filtering, bucketing, or timing-based correlation

The processor routes processed data to registered tools for final analysis.

---

### 🔹 Tool Collection

This is the analysis backend of PASTA:

- Hosts **user-defined profiling tools**
- Operates on preprocessed events
- Generates profiling results such as:
  - Kernel launch frequency
  - Memory access hotness maps
  - Tensor allocation patterns
  - Operator-level performance summaries

Each tool is modular, and developers can add new tools by inheriting from a simple interface and registering their implementation.

---

This architecture enables **flexibility**, **cross-vendor support**, and **fine-grained customization**—making PASTA a powerful backend for performance engineering on modern heterogeneous systems.