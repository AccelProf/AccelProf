# Tool Collection

The **Tool Collection** module in AccelProf executes user-defined analysis tasks after event data has been preprocessed. It provides:

- A library of prebuilt analysis tools (e.g., kernel frequency profiling, tensor-level summaries)
- A clean interface for extending functionality via tool templates

Each tool:

- Consumes preprocessed events
- Implements customizable callbacks to extract specific performance insights
- Can be extended by overriding methods in a subclass of the analysis tool interface

This flexible system enables both standard profiling and advanced, domain-specific workload analysis.
