# Workflow

PASTA’s workflow is composed of three modular components: Event Handler, Event Processor, Tool Collection.

```
Application → [Event Handler] → [Event Processor] → [Tool Collection]
```

PASTA follows a modular, stage-based workflow for collecting and analyzing runtime events:

- **Application**  
  The target workload can be a deep learning model running on a framework (e.g., PyTorch) or a binary accelerator application (e.g., using CUDNN).

- **Event Handler**  
  Captures runtime events through both low-level profiling APIs and high-level DL framework callbacks. It abstracts backend-specific details through a unified interface.

- **Event Processor**  
  Performs CPU- or GPU-side preprocessing to normalize and enrich event data, then dispatches the data for further analysis.

- **Tool Collection**  
  Receives preprocessed events and runs user-defined tools to generate profiling insights such as kernel frequency, memory access patterns, or operator-level summaries.