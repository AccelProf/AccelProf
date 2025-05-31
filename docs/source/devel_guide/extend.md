# How to Extend AccelProf

AccelProf is designed to be extensible. Developers can create custom analysis tools by building on top of the core profiling infrastructure provided by PASTA. This guide walks through how to add a new tool, reuse the tool template, and implement analysis logic.

---

## 1. Add New Tool Support

Start by registering your new tool in the tool enumeration definition. This enables the framework to recognize and activate the tool via command-line options.

### Modify the Tool Type Enum

Edit `sanalyzer/include/tools/tool_type.h` and add your new tool name to the `AnalysisTool_t` enum:

```cpp
typedef enum {
    CODE_CHECK = 0,
    APP_METRICE = 1,
    MEM_TRACE = 2,
    HOT_ANALYSIS = 3,
    UVM_ADVISOR = 4,
    TOOL_NUMS = 5  // Update TOOL_NUMS if you add more entries
} AnalysisTool_t;
```

> 🔧 You should also define a corresponding enum entry in any internal tool dispatch or configuration files used in the runtime.

---

## 2. Reuse Tool Template for Custom Analysis

AccelProf provides a base class `Tool` for implementing analysis logic. To create a new tool, inherit from this class and implement the relevant interface functions based on your analysis goals.

### Create a Custom Tool Class

Here is a minimal skeleton:

```cpp
class CustomTool final : public Tool {
public:
    CustomTool() : Tool(CUSTOM_ANALYSIS) {
        init();
    }

    ~CustomTool() {}

    void init();

    void evt_callback(EventPtr_t evt);

    void gpu_data_analysis(void* data, uint64_t size);

    void query_ranges(void* ranges, uint32_t limit, uint32_t* count);

    void flush();
};
```

This structure allows you to hook into the runtime event stream and analyze various types of memory and execution events.

---

## 3. Add Custom Analysis Code

After declaring your tool class, implement its methods to perform desired runtime analysis. You may implement any combination of the following callbacks depending on your needs.

### Example Callback Implementations

```cpp
void CustomTool::evt_callback(EventPtr_t evt) {
    // Dispatch to specific event type handlers
}

void CustomTool::kernel_start_callback(std::shared_ptr<KernelLauch_t> kernel) {
    // Perform kernel launch analysis
}

void CustomTool::kernel_end_callback(std::shared_ptr<KernelEnd_t> kernel) {
    // Perform kernel end analysis
}

void CustomTool::mem_alloc_callback(std::shared_ptr<MemAlloc_t> mem) {
    // Analyze memory allocation
}

void CustomTool::mem_free_callback(std::shared_ptr<MemFree_t> mem) {
    // Analyze memory deallocation
}

void CustomTool::mem_cpy_callback(std::shared_ptr<MemCpy_t> mem) {
    // Track memory copy operations
}

void CustomTool::mem_set_callback(std::shared_ptr<MemSet_t> mem) {
    // Analyze memory initialization
}

void CustomTool::ten_alloc_callback(std::shared_ptr<TenAlloc_t> ten) {
    // Analyze tensor allocation
}

void CustomTool::ten_free_callback(std::shared_ptr<TenFree_t> ten) {
    // Analyze tensor deallocation
}

void CustomTool::op_start_callback(std::shared_ptr<OpStart_t> op) {
    // Analyze start of a framework-level op
}

void CustomTool::op_end_callback(std::shared_ptr<OpEnd_t> op) {
    // Analyze end of a framework-level op
}

void CustomTool::gpu_data_analysis(void* data, uint64_t size) {
    // GPU-accessible memory access analysis
}

void CustomTool::query_ranges(void* ranges, uint32_t limit, uint32_t* count) {
    // Report memory range metadata
}

void CustomTool::flush() {
    // Finalize and dump results at program exit
}
```

> 🧠 You can leave unused methods empty or omit them entirely—AccelProf only calls the functions your tool overrides.

---

## Summary

Extending AccelProf with a new tool involves:

1. Declaring the tool in the appropriate header files.
2. Implementing a subclass of the `Tool` base class.
3. Registering and wiring the tool into the build and runtime system.
4. Writing custom analysis logic for the events you're interested in.

AccelProf's modular design makes it easy to experiment with new profiling strategies tailored to your research or application needs.