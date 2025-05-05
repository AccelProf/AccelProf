# An Example

In this section, we show an detailed example to extend to a tool that conduct basic performance analysis on GPU applications.

## Add tool model

```diff
# In bin/accelprof
...
+elif [ ${TOOL} == "app_analysis" ]
+then
+    export YOSEMITE_TOOL_NAME=app_analysis
...


# In tool_type.h
typedef enum {
    CODE_CHECK = 0,
    APP_METRICE = 1,
    MEM_TRACE = 2,
    HOT_ANALYSIS = 3,
    UVM_ADVISOR = 4,
+    APP_ANALYSIS = 5,
    TOOL_NUMS = 6
} AnalysisTool_t;

# In sanalyzer.h
typedef enum {
    GPU_NO_PATCH = 0,
    GPU_PATCH_APP_METRIC = 1,
    GPU_PATCH_MEM_TRACE = 2,
    GPU_PATCH_HOT_ANALYSIS = 3,
    GPU_PATCH_UVM_ADVISOR = 4,
+    GPU_PATCH_APP_ANALYSIS = 5,
} SanitizerPatchName_t;
```

## Add analysis code

`app_analysis.h`:
```cpp
#ifndef YOSEMITE_TOOL_APP_ANALYSIS_H
#define YOSEMITE_TOOL_APP_ANALYSIS_H


#include "tools/tool.h"
#include "utils/event.h"

#include <map>
#include <vector>
#include <tuple>
#include <stack>
#include <memory>
#include <unordered_map>

namespace yosemite {

class AppAnalysis final : public Tool {
public:
    AppAnalysis();

    ~AppAnalysis();

    void evt_callback(EventPtr_t evt);

    void gpu_data_analysis(void* data, uint64_t size);

    void query_ranges(void* ranges, uint32_t limit, uint32_t* count);

    void query_tensors(void* ranges, uint32_t limit, uint32_t* count);

    void flush();

private :
    void init();

    void kernel_start_callback(std::shared_ptr<KernelLauch_t> kernel);

    void kernel_end_callback(std::shared_ptr<KernelEnd_t> kernel);

    void mem_alloc_callback(std::shared_ptr<MemAlloc_t> mem);

    void mem_free_callback(std::shared_ptr<MemFree_t> mem);

    void mem_cpy_callback(std::shared_ptr<MemCpy_t> mem);

    void mem_set_callback(std::shared_ptr<MemSet_t> mem);

    void ten_alloc_callback(std::shared_ptr<TenAlloc_t> ten);

    void ten_free_callback(std::shared_ptr<TenFree_t> ten);

    void op_start_callback(std::shared_ptr<OpStart_t> op);

    void op_end_callback(std::shared_ptr<OpEnd_t> op);

};  

}   // yosemite
#endif // YOSEMITE_TOOL_APP_ANALYSIS_H

```

`app_analysis.cpp`:
```cpp
#include "tools/app_analysis.h"
#include "utils/helper.h"
#include "utils/hash.h"
#include "gpu_patch.h"
#include "cpp_trace.h"
#include "py_frame.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <string>
#include <iostream>


using namespace yosemite;


AppAnalysis::AppAnalysis() : Tool(APP_ANALYSIS) {
    init();
}


AppAnalysis::~AppAnalysis() {
    fclose(out_fp);
}

void AppAnalysis::init() {
    const char* env_name = std::getenv("ACCEL_PROF_HOME");
    std::string lib_path;
    if (env_name) {
        lib_path = std::string(env_name) + "/lib/libcompute_sanitizer.so";
    }
    init_backtrace(lib_path.c_str());

}


void AppAnalysis::evt_callback(EventPtr_t evt) {
    switch (evt->evt_type) {
        case EventType_KERNEL_LAUNCH:
            kernel_start_callback(std::dynamic_pointer_cast<KernelLauch_t>(evt));
            break;
        case EventType_KERNEL_END:
            kernel_end_callback(std::dynamic_pointer_cast<KernelEnd_t>(evt));
            break;
        case EventType_MEM_ALLOC:
            mem_alloc_callback(std::dynamic_pointer_cast<MemAlloc_t>(evt));
            break;
        case EventType_MEM_FREE:
            mem_free_callback(std::dynamic_pointer_cast<MemFree_t>(evt));
            break;
        case EventType_MEM_COPY:
            mem_cpy_callback(std::dynamic_pointer_cast<MemCpy_t>(evt));
            break;
        case EventType_MEM_SET:
            mem_set_callback(std::dynamic_pointer_cast<MemSet_t>(evt));
            break;
        case EventType_TEN_ALLOC:
            ten_alloc_callback(std::dynamic_pointer_cast<TenAlloc_t>(evt));
            break;
        case EventType_TEN_FREE:
            ten_free_callback(std::dynamic_pointer_cast<TenFree_t>(evt));
            break;
        case EventType_OP_START:
            op_start_callback(std::dynamic_pointer_cast<OpStart_t>(evt));
            break;
        case EventType_OP_END:
            op_end_callback(std::dynamic_pointer_cast<OpEnd_t>(evt));
            break;
        default:
            break;
    }
}


void AppAnalysis::kernel_start_callback(std::shared_ptr<KernelLauch_t> kernel) {
    // do something

    _timer.increment(true);
}


void AppAnalysis::kernel_end_callback(std::shared_ptr<KernelEnd_t> kernel) {
    // do something

    _timer.increment(true);
}


void AppAnalysis::mem_alloc_callback(std::shared_ptr<MemAlloc_t> mem) {
    // do something

    _timer.increment(true);
}


void AppAnalysis::mem_free_callback(std::shared_ptr<MemFree_t> mem) {
    // do something

    _timer.increment(true);
}



void AppAnalysis::mem_cpy_callback(std::shared_ptr<MemCpy_t> mem) {
    // do something

    _timer.increment(true);
}


void AppAnalysis::mem_set_callback(std::shared_ptr<MemSet_t> mem) {
    // do something

    _timer.increment(true);
}


void AppAnalysis::ten_alloc_callback(std::shared_ptr<TenAlloc_t> ten) {
    // do something

    _timer.increment(true);
}


void AppAnalysis::ten_free_callback(std::shared_ptr<TenFree_t> ten) {
    // do something

    _timer.increment(true);
}


void AppAnalysis::op_start_callback(std::shared_ptr<OpStart_t> op) {
    // do something

    _timer.increment(true);
}


void AppAnalysis::op_end_callback(std::shared_ptr<OpEnd_t> op) {
    // do something

    _timer.increment(true);
}

void AppAnalysis::gpu_data_analysis(void* data, uint64_t size) {
    // do something

    _timer.increment(true);
}


void AppAnalysis::query_ranges(void* ranges, uint32_t limit, uint32_t* count) {
    // do something

    _timer.increment(true);
}

void AppAnalysis::query_tensors(void* ranges, uint32_t limit, uint32_t* count) {
    // do something

    _timer.increment(true);
}


void AppAnalysis::flush() {
    // do something

}
```

## Enable app_analysis mode

In `sanalyzer.cpp`:
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

## Add analysis patch

`gpu_patch_app_analysis.cu`:

```cpp
#include "gpu_patch.h"

#include <sanitizer_patching.h>

#include "gpu_utils.h"

struct gpu_address_comparator {
    __device__
    bool operator()(MemoryRange &l, MemoryRange &r) {
        return l.start <= r.start;
    }
};

static __device__
SanitizerPatchResult CommonCallback(
    void* userdata,
    uint64_t pc,
    void* ptr,
    uint32_t accessSize,
    uint32_t flags,
    MemoryType type)
{
    auto* pTracker = (MemoryAccessTracker*)userdata;

    uint32_t active_mask = __activemask();

    if (pTracker->access_state != nullptr) {
        MemoryAccessState* states = (MemoryAccessState*) pTracker->access_state;
        MemoryRange* start_end = states->start_end;
        MemoryRange range = {(uint64_t) ptr, 0};
        uint32_t pos = map_prev(start_end, range, states->size, gpu_address_comparator());

        if (pos != states->size) {
            if (atomic_load(states->touch + pos) == 0) {
                atomic_store(states->touch + pos, 1);
            }
        }
    }
    __syncwarp(active_mask);

    if (pTracker->tensor_access_state != nullptr) {
        TensorAccessState* tensor_states = (TensorAccessState*) pTracker->tensor_access_state;
        MemoryRange* start_end = tensor_states->start_end;
        MemoryRange range = {(uint64_t) ptr, 0};
        uint32_t tensor_pos = map_prev(start_end, range, tensor_states->size, gpu_address_comparator());

        if (tensor_pos != tensor_states->size) {
            if (atomic_load(tensor_states->touch + tensor_pos) == 0) {
                atomic_store(tensor_states->touch + tensor_pos, 1);
            }
        }
    }
    __syncwarp(active_mask);

    return SANITIZER_PATCH_SUCCESS;
}

extern "C" __device__ __noinline__
SanitizerPatchResult MemoryGlobalAccessCallback(
    void* userdata,
    uint64_t pc,
    void* ptr,
    uint32_t accessSize,
    uint32_t flags,
    const void *pData)
{
    return CommonCallback(userdata, pc, ptr, accessSize, flags, MemoryType::Global);
}

extern "C" __device__ __noinline__
SanitizerPatchResult MemorySharedAccessCallback(
    void* userdata,
    uint64_t pc,
    void* ptr,
    uint32_t accessSize,
    uint32_t flags,
    const void *pData)
{
    return CommonCallback(userdata, pc, ptr, accessSize, flags, MemoryType::Shared);
}

extern "C" __device__ __noinline__
SanitizerPatchResult MemoryLocalAccessCallback(
    void* userdata,
    uint64_t pc,
    void* ptr,
    uint32_t accessSize,
    uint32_t flags,
    const void *pData)
{
    return CommonCallback(userdata, pc, ptr, accessSize, flags, MemoryType::Local);
}

extern "C" __device__ __noinline__
SanitizerPatchResult MemcpyAsyncCallback(void* userdata, uint64_t pc, void* src, uint32_t dst, uint32_t accessSize)
{
    if (src)
    {
        CommonCallback(userdata, pc, src, accessSize, SANITIZER_MEMORY_DEVICE_FLAG_READ, MemoryType::Global);
    }

    return CommonCallback(userdata, pc, (void*)dst, accessSize, SANITIZER_MEMORY_DEVICE_FLAG_WRITE, MemoryType::Shared);
}

```

## Enable analysis patch in backend

```diff
+ else if (sanitizer_options.patch_name == GPU_PATCH_APP_ANALYSIS) {
+     SANITIZER_SAFECALL(
+         sanitizerPatchInstructions(SANITIZER_INSTRUCTION_GLOBAL_MEMORY_ACCESS, module, "MemoryGlobalAccessCallback"));
+ }

...

+ else if (sanitizer_options.patch_name == GPU_PATCH_APP_ANALYSIS) {
+     if (!device_access_state)
+         SANITIZER_SAFECALL(sanitizerAlloc(context, (void**)&device_access_state, sizeof(MemoryAccessState)));
+ 
+     if (!host_access_state) {
+         SANITIZER_SAFECALL(sanitizerAllocHost(context, (void**)&host_access_state, sizeof(MemoryAccessState)));
+     }
+     if (!device_tensor_access_state) {
+         SANITIZER_SAFECALL(sanitizerAlloc(context, (void**)&device_tensor_access_state, sizeof(TensorAccessState)));
+     }
+     if (!host_tensor_access_state) {
+         SANITIZER_SAFECALL(sanitizerAllocHost(context, (void**)&host_tensor_access_state, sizeof(TensorAccessState)));
+     }
+ }

...

+ else if (sanitizer_options.patch_name == GPU_PATCH_UVM_ADVISOR) {
+     memset(host_access_state, 0, sizeof(MemoryAccessState));
+     memset(host_tensor_access_state, 0, sizeof(TensorAccessState));
+     yosemite_query_active_ranges(host_access_state->start_end, MAX_NUM_MEMORY_RANGES, &host_access_state->size);
+     yosemite_query_active_tensors(host_tensor_access_state->start_end, MAX_NUM_TENSOR_RANGES, &host_tensor_access_state->size);
+     SANITIZER_SAFECALL(
+         sanitizerMemcpyHostToDeviceAsync(device_access_state, host_access_state, sizeof(MemoryAccessState), hstream));
+     SANITIZER_SAFECALL(
+         sanitizerMemcpyHostToDeviceAsync(device_tensor_access_state, host_tensor_access_state, sizeof(TensorAccessState), hstream));
+     host_tracker_handle->access_state = device_access_state;
+     host_tracker_handle->tensor_access_state = device_tensor_access_state;
+ }

...

+ else if (sanitizer_options.patch_name == GPU_PATCH_UVM_ADVISOR) {
+     SANITIZER_SAFECALL(sanitizerStreamSynchronize(hstream));
+     SANITIZER_SAFECALL(
+         sanitizerMemcpyDeviceToHost(host_tracker_handle, device_tracker_handle, sizeof(MemoryAccessTracker), hstream));
+     SANITIZER_SAFECALL(
+         sanitizerMemcpyDeviceToHost(host_access_state, device_access_state, sizeof(MemoryAccessState), hstream));
+     SANITIZER_SAFECALL(
+         sanitizerMemcpyDeviceToHost(host_tensor_access_state, device_tensor_access_state, sizeof(TensorAccessState), hstream));
+     host_tracker_handle->access_state = host_access_state;
+     host_tracker_handle->tensor_access_state = host_tensor_access_state;
+     yosemite_gpu_data_analysis(host_tracker_handle, host_tracker_handle->accessCount);
+ } 
```