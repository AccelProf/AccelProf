# An Example: Adding a Custom Analysis Tool

This section demonstrates how to extend AccelProf by adding a new tool for basic GPU performance analysis. The steps include integrating the tool model, implementing analysis logic, registering the tool, adding a GPU-side patch, and enabling it in the backend runtime.

---

## Step 1: Add Tool Model

Add `app_analysis` to the tool detection logic and internal enumeration types.

### Modify the Tool Dispatcher

In `bin/accelprof`:

```diff
...
+elif [ ${TOOL} == "app_analysis" ]
+then
+    export YOSEMITE_TOOL_NAME=app_analysis
...
```

### Add to `tool_type.h`

```diff
typedef enum {
    CODE_CHECK = 0,
    APP_METRICE = 1,
    MEM_TRACE = 2,
    HOT_ANALYSIS = 3,
    UVM_ADVISOR = 4,
+    APP_ANALYSIS = 5,
    TOOL_NUMS = 6
} AnalysisTool_t;
```

### Add to `sanalyzer.h`

```diff
typedef enum {
    GPU_NO_PATCH = 0,
    GPU_PATCH_APP_METRIC = 1,
    GPU_PATCH_MEM_TRACE = 2,
    GPU_PATCH_HOT_ANALYSIS = 3,
    GPU_PATCH_UVM_ADVISOR = 4,
+    GPU_PATCH_APP_ANALYSIS = 5,
} SanitizerPatchName_t;
```

---

## Step 2: Implement Analysis Logic

Define the structure of your analysis tool by inheriting from the `Tool` base class.

### `app_analysis.h`

```cpp
// ...header file defining the AppAnalysis class
```

### `app_analysis.cpp`

```cpp
// ...implementation of event callbacks and GPU-side analysis logic
```

Each callback handles a specific type of event (e.g., memory allocations, kernel launches). You can modify or extend the callbacks for custom behavior.

---

## Step 3: Enable the Tool in Sanalyzer

In `sanalyzer.cpp`, register the new tool and its associated patch.

```diff
+ #include "tools/app_analysis.h"

+ else if (std::string(tool_name) == "app_analysis") {
+    tool = APP_ANALYSIS;
+    _tools.emplace(APP_ANALYSIS, std::make_shared<AppAnalysis>());
+ }

+ else if (tool == APP_ANALYSIS) {
+    options.patch_name = GPU_PATCH_APP_ANALYSIS;
+    options.patch_file = "gpu_patch_app_analysis.fatbin";
+ }
```

---

## Step 4: Add a GPU Patch

Write a patch file that will be injected during kernel execution.

### `gpu_patch_app_analysis.cu`

```cpp
// ...implementation of CommonCallback and memory access tracking functions
```

These patch functions log global, shared, and local memory accesses during kernel execution and update associated tracking data structures.

---

## Step 5: Enable Patch Logic in the Backend

In your backend runtime patching and tracking logic, add conditions for `GPU_PATCH_APP_ANALYSIS`.

```diff
+ else if (sanitizer_options.patch_name == GPU_PATCH_APP_ANALYSIS) {
+     // Patch instructions for memory access
+ }

+ else if (sanitizer_options.patch_name == GPU_PATCH_APP_ANALYSIS) {
+     // Memory state allocation
+ }

+ else if (sanitizer_options.patch_name == GPU_PATCH_UVM_ADVISOR) {
+     // Initialization of access state and memory copy
+ }

+ else if (sanitizer_options.patch_name == GPU_PATCH_UVM_ADVISOR) {
+     // Final analysis and data transfer from device to host
+ }
```

---

## Summary

By following these steps, you can extend AccelProf with a new tool for custom GPU analysis:

- Add enum entries and shell hooks
- Implement event-driven analysis logic
- Register and activate your tool
- Write a custom memory patch and wire it into the runtime backend

This modular design enables scalable and flexible extension of profiling capabilities.