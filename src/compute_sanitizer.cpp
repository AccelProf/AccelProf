#include "compute_sanitizer.h"

#include "sanitizer_helper.h"
#include "gpu_patch.h"
#include "sanalyzer.h"
#include "torch_scope.h"

#include <sanitizer.h>
#include <vector_types.h>

#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <vector>
#include <cstring>
#include <cassert>
#include <unordered_map>
#include <unordered_set>
#include <sstream>

#define SANITIZER_VERBOSE 1

#if SANITIZER_VERBOSE
#define PRINT(...) do { fprintf(stdout, __VA_ARGS__); fflush(stdout); } while (0)
#else
#define PRINT(...)
#endif

#define SANITIZER_SAFECALL(fn)                                      \
{                                                                   \
    SanitizerResult result = fn;                                    \
    if (result != SANITIZER_SUCCESS) {                              \
        const char *error_string;                                   \
        sanitizerGetResultString(result, &error_string);            \
        fprintf(                                                    \
            stderr,                                                 \
            "[SANITIZER ERROR] '%s' in '%s:%i' - error(%i): %s.\n", \
            #fn, __FILE__, __LINE__, result, error_string);         \
        fflush(stderr);                                             \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
}


static MemoryAccessTracker* host_tracker_handle = nullptr;
static MemoryAccessTracker* device_tracker_handle = nullptr;
static MemoryAccess* host_access_buffer = nullptr;
static MemoryAccess* device_access_buffer = nullptr;
static MemoryAccessState* host_access_state = nullptr;
static MemoryAccessState* device_access_state = nullptr;
static TensorAccessState* host_tensor_access_state = nullptr;
static TensorAccessState* device_tensor_access_state = nullptr;
static DoorBell* global_doorbell = nullptr;

static AccelProfOptions_t sanitizer_options;
// <module, is_patched>
static std::unordered_map<CUmodule, bool> sanitizer_active_modules;

// <context, device> for multi-GPU support
static std::unordered_map<CUcontext, CUdevice> sanitizer_ctx_to_device;

static std::unordered_set<std::string> sanitizer_kernel_white_list;

// 读取 whitelist 文件，按行解析 kernel 关键字
void LoadKernelWhiteList(const char* whitelist_path) {
    if (!whitelist_path) return;

    std::ifstream fin(whitelist_path);
    if (!fin.is_open()) {
        std::cerr << "[SANITIZER WARN] Failed to open whitelist file: "
                  << whitelist_path << std::endl;
        return;
    }

    auto trim = [](std::string& s) {
        const char* ws = " \t\r\n";
        auto start = s.find_first_not_of(ws);
        if (start == std::string::npos) {
            s.clear();
            return;
        }
        auto end = s.find_last_not_of(ws);
        s = s.substr(start, end - start + 1);
    };

    std::string line;
    while (std::getline(fin, line)) {
        trim(line);
        if (line.empty() || line[0] == '#') continue;  // 跳过空行与注释
        sanitizer_kernel_white_list.insert(line);
    }

    PRINT("[SANITIZER INFO] Loaded %zu kernel whitelist entries from %s\n",
          sanitizer_kernel_white_list.size(), whitelist_path);
}

bool SanitizerKernelWhiteListCheck(const std::string& functionName){
    if (sanitizer_kernel_white_list.empty()) {
        return true;
    }
    for (const auto& kernel : sanitizer_kernel_white_list) {
        if (functionName.find(kernel) != std::string::npos) {
            return true;
        }
    }
    return false;
}

void SanitizerTensorMallocCallback(uint64_t ptr, int64_t size, int64_t allocated, int64_t reserved, int device_id) {
    if (!sanitizer_options.sanitizer_callback_enabled) {
        return;
    }

    PRINT("[SANITIZER INFO] Malloc tensor %p with size %ld, allocated %ld, reserved %ld on device %d\n",
            (void*)ptr, size, allocated, reserved, device_id);
    yosemite_tensor_malloc_callback(ptr, size, allocated, reserved, device_id);
}


void SanitizerTensorFreeCallback(uint64_t ptr, int64_t size, int64_t allocated, int64_t reserved, int device_id) {
    if (!sanitizer_options.sanitizer_callback_enabled) {
        return;
    }

    PRINT("[SANITIZER INFO] Free tensor %p with size %ld, allocated %ld, reserved %ld on device %d\n",
            (void*)ptr, size, allocated, reserved, device_id);
    yosemite_tensor_free_callback(ptr, size, allocated, reserved, device_id);
}


void SanitizerOperatorStartCallback(void* ctx, std::string op_name) {
    if (!sanitizer_options.sanitizer_callback_enabled) {
        return;
    }

    PRINT("[SANITIZER INFO] Torch operator start: %s, ctx: %p\n", op_name.c_str(), ctx);
    yosemite_operator_start_callback(ctx, op_name);
}


void SanitizerOperatorEndCallback(void* ctx, std::string op_name) {
    if (!sanitizer_options.sanitizer_callback_enabled) {
        return;
    }

    PRINT("[SANITIZER INFO] Torch operator end: %s, ctx: %p\n", op_name.c_str(), ctx);
    yosemite_operator_end_callback(ctx, op_name);
}

void ModuleUnloadedCallback(CUmodule module) {
    if (sanitizer_options.patch_name == GPU_NO_PATCH) {
        return;
    }

    auto it = sanitizer_active_modules.find(module);
    assert(it != sanitizer_active_modules.end());
    if (it->second) {   // unpatch if module is patched
        SANITIZER_SAFECALL(sanitizerUnpatchModule(module));
    }
    sanitizer_active_modules.erase(it);
}


void ModuleLoadedCallback(CUmodule module)
{
    if (sanitizer_options.patch_name == GPU_NO_PATCH) {
        return;
    }

    sanitizer_active_modules.try_emplace(module, sanitizer_options.sanitizer_callback_enabled);
    if (!sanitizer_options.sanitizer_callback_enabled) {
        return;
    }

    const char* env_name = std::getenv("ACCEL_PROF_HOME");
    std::string patch_path;
    if (env_name) {
        patch_path = std::string(env_name) + "/nv-compute/lib/gpu_patch/";
    } else {
        std::cerr << "Failed to load fatbin. No patch path specified." << std::endl;
    }

    // Instrument user code
    std::string fatbin_file = patch_path + sanitizer_options.patch_file;
    SANITIZER_SAFECALL(sanitizerAddPatchesFromFile(fatbin_file.c_str(), 0));

    if (sanitizer_options.patch_name == GPU_PATCH_APP_METRIC) {
        SANITIZER_SAFECALL(
            sanitizerPatchInstructions(
                SANITIZER_INSTRUCTION_GLOBAL_MEMORY_ACCESS, module, "MemoryGlobalAccessCallback"));
        // SANITIZER_SAFECALL(
        //     sanitizerPatchInstructions(
        //         SANITIZER_INSTRUCTION_SHARED_MEMORY_ACCESS, module, "MemorySharedAccessCallback"));
        // SANITIZER_SAFECALL(
        //     sanitizerPatchInstructions(
        //         SANITIZER_INSTRUCTION_LOCAL_MEMORY_ACCESS, module, "MemoryLocalAccessCallback"));
        // SANITIZER_SAFECALL(
        //     sanitizerPatchInstructions(
        //         SANITIZER_INSTRUCTION_MEMCPY_ASYNC, module, "MemcpyAsyncCallback"));
    } else if (sanitizer_options.patch_name == GPU_PATCH_ROOFLINE_SIZE) {
        SANITIZER_SAFECALL(
            sanitizerPatchInstructions(
                SANITIZER_INSTRUCTION_GLOBAL_MEMORY_ACCESS, module, "MemoryGlobalAccessCallback"));
    } else if (sanitizer_options.patch_name == GPU_PATCH_MEM_TRACE) {
        SANITIZER_SAFECALL(
            sanitizerPatchInstructions(
                SANITIZER_INSTRUCTION_GLOBAL_MEMORY_ACCESS, module, "MemoryGlobalAccessCallback"));
        // SANITIZER_SAFECALL(
        //     sanitizerPatchInstructions(
        //         SANITIZER_INSTRUCTION_SHARED_MEMORY_ACCESS, module, "MemorySharedAccessCallback"));
        // SANITIZER_SAFECALL(
        //     sanitizerPatchInstructions(
        //         SANITIZER_INSTRUCTION_LOCAL_MEMORY_ACCESS, module, "MemoryLocalAccessCallback"));
        // SANITIZER_SAFECALL(
        //     sanitizerPatchInstructions(
        //         SANITIZER_INSTRUCTION_MEMCPY_ASYNC, module, "MemcpyAsyncCallback"));
        SANITIZER_SAFECALL(
            sanitizerPatchInstructions(SANITIZER_INSTRUCTION_BLOCK_EXIT, module, "BlockExitCallback"));
    } else if (sanitizer_options.patch_name == GPU_PATCH_HOT_ANALYSIS) {
        SANITIZER_SAFECALL(
            sanitizerPatchInstructions(
                SANITIZER_INSTRUCTION_GLOBAL_MEMORY_ACCESS, module, "MemoryGlobalAccessCallback"));
    } else if (sanitizer_options.patch_name == GPU_PATCH_UVM_ADVISOR) {
        SANITIZER_SAFECALL(
            sanitizerPatchInstructions(
                SANITIZER_INSTRUCTION_GLOBAL_MEMORY_ACCESS, module, "MemoryGlobalAccessCallback"));
    } else if (sanitizer_options.patch_name == GPU_PATCH_APP_ANALYSIS) {
        SANITIZER_SAFECALL(
            sanitizerPatchInstructions(
                SANITIZER_INSTRUCTION_GLOBAL_MEMORY_ACCESS, module, "MemoryGlobalAccessCallback"));
    } else if (sanitizer_options.patch_name == GPU_PATCH_APP_ANALYSIS_CPU) {
        SANITIZER_SAFECALL(
            sanitizerPatchInstructions(
                SANITIZER_INSTRUCTION_GLOBAL_MEMORY_ACCESS, module, "MemoryGlobalAccessCallback"));
        SANITIZER_SAFECALL(
            sanitizerPatchInstructions(
                SANITIZER_INSTRUCTION_BLOCK_EXIT, module, "BlockExitCallback"));
    } else if (sanitizer_options.patch_name == GPU_PATCH_TIME_HOTNESS_CPU) {
        SANITIZER_SAFECALL(
            sanitizerPatchInstructions(
                SANITIZER_INSTRUCTION_GLOBAL_MEMORY_ACCESS, module, "MemoryGlobalAccessCallback"));
        SANITIZER_SAFECALL(
            sanitizerPatchInstructions(SANITIZER_INSTRUCTION_BLOCK_EXIT, module, "BlockExitCallback"));
    } else if (sanitizer_options.patch_name == GPU_PATCH_HEATMAP_ANALYSIS) {
        SANITIZER_SAFECALL(
            sanitizerPatchInstructions(
                SANITIZER_INSTRUCTION_GLOBAL_MEMORY_ACCESS, module, "MemoryGlobalAccessCallback"));
        SANITIZER_SAFECALL(
            sanitizerPatchInstructions(
                SANITIZER_INSTRUCTION_SHARED_MEMORY_ACCESS, module, "MemorySharedAccessCallback"));
        SANITIZER_SAFECALL(
            sanitizerPatchInstructions(
                SANITIZER_INSTRUCTION_LOCAL_MEMORY_ACCESS, module, "MemoryLocalAccessCallback"));
        // SANITIZER_SAFECALL(
        //     sanitizerPatchInstructions(
        //         SANITIZER_INSTRUCTION_MEMCPY_ASYNC, module, "MemcpyAsyncCallback"));
        SANITIZER_SAFECALL(
            sanitizerPatchInstructions(SANITIZER_INSTRUCTION_BLOCK_EXIT, module, "BlockExitCallback"));
    } else if (sanitizer_options.patch_name == GPU_PATCH_BLOCK_DIVERGENCE_ANALYSIS) {
        SANITIZER_SAFECALL(
            sanitizerPatchInstructions(
                SANITIZER_INSTRUCTION_GLOBAL_MEMORY_ACCESS, module, "MemoryGlobalAccessCallback"));
        SANITIZER_SAFECALL(
            sanitizerPatchInstructions(
                SANITIZER_INSTRUCTION_SHARED_MEMORY_ACCESS, module, "MemorySharedAccessCallback"));
        SANITIZER_SAFECALL(
            sanitizerPatchInstructions(
                SANITIZER_INSTRUCTION_LOCAL_MEMORY_ACCESS, module, "MemoryLocalAccessCallback"));
        SANITIZER_SAFECALL(
            sanitizerPatchInstructions(SANITIZER_INSTRUCTION_BLOCK_EXIT, module, "BlockExitCallback"));
    }
    
    SANITIZER_SAFECALL(sanitizerPatchModule(module));
}


void buffer_init(CUcontext context) {
    if (!device_tracker_handle) {
        SANITIZER_SAFECALL(
            sanitizerAlloc(context, (void**)&device_tracker_handle, sizeof(MemoryAccessTracker)));
    }
    if (!host_tracker_handle) {
        SANITIZER_SAFECALL(
            sanitizerAllocHost(context, (void**)&host_tracker_handle, sizeof(MemoryAccessTracker)));
    }

    if (sanitizer_options.patch_name == GPU_PATCH_APP_METRIC) {
        if (!device_access_state)
            SANITIZER_SAFECALL(
                sanitizerAlloc(context, (void**)&device_access_state, sizeof(MemoryAccessState)));

        if (!host_access_state) {
            SANITIZER_SAFECALL(
                sanitizerAllocHost(context, (void**)&host_access_state, sizeof(MemoryAccessState)));
        }
    } else if (sanitizer_options.patch_name == GPU_PATCH_ROOFLINE_SIZE) {
        // no functions needed
    } else if (sanitizer_options.patch_name == GPU_PATCH_MEM_TRACE) {
        if (!device_access_buffer) {
            SANITIZER_SAFECALL(
                sanitizerAlloc(
                    context,
                    (void**)&device_access_buffer,
                    sizeof(MemoryAccess) * MEMORY_ACCESS_BUFFER_SIZE));
        }
        if (!host_access_buffer) {
            SANITIZER_SAFECALL(
                sanitizerAllocHost(
                    context,
                    (void**)&host_access_buffer,
                    sizeof(MemoryAccess) * MEMORY_ACCESS_BUFFER_SIZE));
        }
        if (!global_doorbell) {
            SANITIZER_SAFECALL(
                sanitizerAllocHost(context, (void**)&global_doorbell, sizeof(DoorBell)));
        }
    } else if (sanitizer_options.patch_name == GPU_PATCH_UVM_ADVISOR) {
        if (!device_access_state)
            SANITIZER_SAFECALL(
                sanitizerAlloc(context, (void**)&device_access_state, sizeof(MemoryAccessState)));

        if (!host_access_state) {
            SANITIZER_SAFECALL(
                sanitizerAllocHost(context, (void**)&host_access_state, sizeof(MemoryAccessState)));
        }
        if (!device_tensor_access_state) {
            SANITIZER_SAFECALL(
                sanitizerAlloc(context, (void**)&device_tensor_access_state, sizeof(TensorAccessState)));
        }
        if (!host_tensor_access_state) {
            SANITIZER_SAFECALL(
                sanitizerAllocHost(context, (void**)&host_tensor_access_state, sizeof(TensorAccessState)));
        }
    } else if (sanitizer_options.patch_name == GPU_PATCH_APP_ANALYSIS) {
        if (!device_access_state)
            SANITIZER_SAFECALL(
                sanitizerAlloc(context, (void**)&device_access_state, sizeof(MemoryAccessState)));

        if (!host_access_state) {
            SANITIZER_SAFECALL(
                sanitizerAllocHost(context, (void**)&host_access_state, sizeof(MemoryAccessState)));
        }
        if (!device_tensor_access_state) {
            SANITIZER_SAFECALL(
                sanitizerAlloc(context, (void**)&device_tensor_access_state, sizeof(TensorAccessState)));
        }
        if (!host_tensor_access_state) {
            SANITIZER_SAFECALL(
                sanitizerAllocHost(
                    context, (void**)&host_tensor_access_state, sizeof(TensorAccessState)));
        }
    } else if (sanitizer_options.patch_name == GPU_PATCH_APP_ANALYSIS_CPU) {
        if (!device_access_buffer) {
            SANITIZER_SAFECALL(
                sanitizerAlloc(
                    context,
                    (void**)&device_access_buffer,
                    sizeof(MemoryAccess) * MEMORY_ACCESS_BUFFER_SIZE));
        }
        if (!host_access_buffer) {
            SANITIZER_SAFECALL(
                sanitizerAllocHost(
                    context,
                    (void**)&host_access_buffer,
                    sizeof(MemoryAccess) * MEMORY_ACCESS_BUFFER_SIZE));
        }
        if (!global_doorbell) {
            SANITIZER_SAFECALL(
                sanitizerAllocHost(context, (void**)&global_doorbell, sizeof(DoorBell)));
        }
    } else if (sanitizer_options.patch_name == GPU_PATCH_TIME_HOTNESS_CPU) {
        if (!device_access_buffer) {
            SANITIZER_SAFECALL(
                sanitizerAlloc(
                    context,
                    (void**)&device_access_buffer,
                    sizeof(MemoryAccess) * MEMORY_ACCESS_BUFFER_SIZE));
        }
        if (!host_access_buffer) {
            SANITIZER_SAFECALL(
                sanitizerAllocHost(
                    context,
                    (void**)&host_access_buffer,
                    sizeof(MemoryAccess) * MEMORY_ACCESS_BUFFER_SIZE));
        }
        if (!global_doorbell) {
            SANITIZER_SAFECALL(
                sanitizerAllocHost(context, (void**)&global_doorbell, sizeof(DoorBell)));
        }
    } else if (sanitizer_options.patch_name == GPU_PATCH_HEATMAP_ANALYSIS) {
        if (!device_access_buffer) {
            SANITIZER_SAFECALL(
                sanitizerAlloc(
                    context,
                    (void**)&device_access_buffer,
                    sizeof(MemoryAccess) * MEMORY_ACCESS_BUFFER_SIZE));
        }
        if (!host_access_buffer) {
            SANITIZER_SAFECALL(
                sanitizerAllocHost(
                    context,
                    (void**)&host_access_buffer,
                    sizeof(MemoryAccess) * MEMORY_ACCESS_BUFFER_SIZE));
        }
        if (!global_doorbell) {
            SANITIZER_SAFECALL(
                sanitizerAllocHost(context, (void**)&global_doorbell, sizeof(DoorBell)));
        }
    } else if (sanitizer_options.patch_name == GPU_PATCH_BLOCK_DIVERGENCE_ANALYSIS) {
        if (!device_access_buffer) {
            SANITIZER_SAFECALL(
                sanitizerAlloc(
                    context,
                    (void**)&device_access_buffer,
                    sizeof(MemoryAccess) * MEMORY_ACCESS_BUFFER_SIZE));
        }
        if (!host_access_buffer) {
            SANITIZER_SAFECALL(
                sanitizerAllocHost(
                    context,
                    (void**)&host_access_buffer,
                    sizeof(MemoryAccess) * MEMORY_ACCESS_BUFFER_SIZE));
        }
        if (!global_doorbell) {
            SANITIZER_SAFECALL(
                sanitizerAllocHost(context, (void**)&global_doorbell, sizeof(DoorBell)));
        }
    }
}


void LaunchBeginCallback(
    CUcontext context,
    CUmodule module,
    CUfunction function,
    std::string functionName,
    Sanitizer_StreamHandle hstream,
    dim3 blockDims,
    dim3 gridDims)
{
    if (sanitizer_options.patch_name != GPU_NO_PATCH) {
        // sampling
        sanitizer_options.grid_launch_id++;
        if ((sanitizer_options.grid_launch_id % sanitizer_options.sample_rate == 0) &&
            SanitizerKernelWhiteListCheck(functionName)) {
            PRINT("[SANITIZER INFO] Monitoring kernel %s, launch id %lu\n",
                    functionName.c_str(), sanitizer_options.grid_launch_id);
            auto it = sanitizer_active_modules.find(module);
            if (!it->second) {
                ModuleLoadedCallback(module);
                it->second = true;
            }
        } else {
            PRINT("[SANITIZER INFO] Skipping kernel %s monitoring, launch id %lu\n",
                    functionName.c_str(), sanitizer_options.grid_launch_id);
            auto it = sanitizer_active_modules.find(module);
            if (it->second) {
                SANITIZER_SAFECALL(sanitizerUnpatchModule(module));
                it->second = false;
            }
            return;
        }

        buffer_init(context);

        if (sanitizer_options.patch_name == GPU_PATCH_APP_METRIC) {
            memset(host_access_state, 0, sizeof(MemoryAccessState));
            yosemite_query_active_ranges(
                host_access_state->start_end, MAX_NUM_MEMORY_RANGES, &host_access_state->size);
            SANITIZER_SAFECALL(
                sanitizerMemcpyHostToDeviceAsync(
                    device_access_state, host_access_state, sizeof(MemoryAccessState), hstream));
            host_tracker_handle->accessCount = 0;
            host_tracker_handle->access_state = device_access_state;
        } else if (sanitizer_options.patch_name == GPU_PATCH_ROOFLINE_SIZE) {
            host_tracker_handle->accessCount = 0;
            host_tracker_handle->accessSize = 0;

        } else if (sanitizer_options.patch_name == GPU_PATCH_MEM_TRACE) {
            SANITIZER_SAFECALL(
                sanitizerMemset(
                    device_access_buffer, 0, sizeof(MemoryAccess) * MEMORY_ACCESS_BUFFER_SIZE, hstream));
            host_tracker_handle->currentEntry = 0;
            host_tracker_handle->numEntries = 0;
            host_tracker_handle->access_buffer = device_access_buffer;

            uint32_t num_threads =
                        blockDims.x * blockDims.y * blockDims.z * gridDims.x * gridDims.y * gridDims.z;
            global_doorbell->num_threads = num_threads;
            global_doorbell->full = 0;
            host_tracker_handle->doorBell = global_doorbell;
        } else if (sanitizer_options.patch_name == GPU_PATCH_HOT_ANALYSIS) {
            memset(host_access_state, 0, sizeof(MemoryAccessState));
            yosemite_query_active_ranges(
                host_access_state->start_end, MAX_NUM_MEMORY_RANGES, &host_access_state->size);
            SANITIZER_SAFECALL(
                sanitizerMemcpyHostToDeviceAsync(
                    device_access_state, host_access_state, sizeof(MemoryAccessState), hstream));

            host_tracker_handle->access_state = device_access_state;
        } else if (sanitizer_options.patch_name == GPU_PATCH_UVM_ADVISOR) {
            memset(host_access_state, 0, sizeof(MemoryAccessState));
            memset(host_tensor_access_state, 0, sizeof(TensorAccessState));
            yosemite_query_active_ranges(
                host_access_state->start_end, MAX_NUM_MEMORY_RANGES, &host_access_state->size);
            yosemite_query_active_tensors(
                host_tensor_access_state->start_end, MAX_NUM_TENSOR_RANGES, &host_tensor_access_state->size);
            SANITIZER_SAFECALL(
                sanitizerMemcpyHostToDeviceAsync(
                    device_access_state, host_access_state, sizeof(MemoryAccessState), hstream));
            SANITIZER_SAFECALL(
                sanitizerMemcpyHostToDeviceAsync(
                    device_tensor_access_state, host_tensor_access_state, sizeof(TensorAccessState), hstream));

            host_tracker_handle->access_state = device_access_state;
            host_tracker_handle->tensor_access_state = device_tensor_access_state;
        } else if (sanitizer_options.patch_name == GPU_PATCH_APP_ANALYSIS) {
            memset(host_access_state, 0, sizeof(MemoryAccessState));
            memset(host_tensor_access_state, 0, sizeof(TensorAccessState));
            yosemite_query_active_ranges(
                host_access_state->start_end, MAX_NUM_MEMORY_RANGES, &host_access_state->size);
            yosemite_query_active_tensors(
                host_tensor_access_state->start_end, MAX_NUM_TENSOR_RANGES, &host_tensor_access_state->size);
            SANITIZER_SAFECALL(
                sanitizerMemcpyHostToDeviceAsync(
                    device_access_state, host_access_state, sizeof(MemoryAccessState), hstream));
            SANITIZER_SAFECALL(
                sanitizerMemcpyHostToDeviceAsync(
                    device_tensor_access_state, host_tensor_access_state, sizeof(TensorAccessState), hstream));

            host_tracker_handle->access_state = device_access_state;
            host_tracker_handle->tensor_access_state = device_tensor_access_state;
            host_tracker_handle->accessCount = 0;
        } else if (sanitizer_options.patch_name == GPU_PATCH_APP_ANALYSIS_CPU) {
            SANITIZER_SAFECALL(
                sanitizerMemset(
                    device_access_buffer, 0, sizeof(MemoryAccess) * MEMORY_ACCESS_BUFFER_SIZE, hstream));
            host_tracker_handle->currentEntry = 0;
            host_tracker_handle->numEntries = 0;
            host_tracker_handle->access_buffer = device_access_buffer;

            uint32_t num_threads =
                        blockDims.x * blockDims.y * blockDims.z * gridDims.x * gridDims.y * gridDims.z;
            global_doorbell->num_threads = num_threads;
            global_doorbell->full = 0;
            host_tracker_handle->doorBell = global_doorbell;
        } else if (sanitizer_options.patch_name == GPU_PATCH_TIME_HOTNESS_CPU) {
            SANITIZER_SAFECALL(
                sanitizerMemset(
                    device_access_buffer, 0, sizeof(MemoryAccess) * MEMORY_ACCESS_BUFFER_SIZE, hstream));
            host_tracker_handle->currentEntry = 0;
            host_tracker_handle->numEntries = 0;
            host_tracker_handle->access_buffer = device_access_buffer;

            uint32_t num_threads =
                        blockDims.x * blockDims.y * blockDims.z * gridDims.x * gridDims.y * gridDims.z;
            global_doorbell->num_threads = num_threads;
            global_doorbell->full = 0;
            host_tracker_handle->doorBell = global_doorbell;
        } else if (sanitizer_options.patch_name == GPU_PATCH_HEATMAP_ANALYSIS) {
            SANITIZER_SAFECALL(
                sanitizerMemset(
                    device_access_buffer, 0, sizeof(MemoryAccess) * MEMORY_ACCESS_BUFFER_SIZE, hstream));
            host_tracker_handle->currentEntry = 0;
            host_tracker_handle->numEntries = 0;
            host_tracker_handle->access_buffer = device_access_buffer;
            char* target_block_str = std::getenv("YOSEMITE_TARGET_BLOCK");
            if (target_block_str) {
                // target_block_str is "x,y,z"
                std::stringstream ss(target_block_str);
                std::string token;
                int i = 0;
                while (std::getline(ss, token, ',')) {
                    host_tracker_handle->target_block[i] = std::stoi(token);
                    i++;
                }
                if (i != 3) {
                    PRINT("[SANITIZER ERROR] Invalid target block format: %s\n", target_block_str);
                    exit(EXIT_FAILURE);
                }
                if (host_tracker_handle->target_block[0] < 0 || host_tracker_handle->target_block[1] < 0 || host_tracker_handle->target_block[2] < 0) {
                    PRINT("[SANITIZER ERROR] Invalid target block: %d, %d, %d\n", host_tracker_handle->target_block[0], host_tracker_handle->target_block[1], host_tracker_handle->target_block[2]);
                    exit(EXIT_FAILURE);
                }
                if (host_tracker_handle->target_block[0] >= gridDims.x || host_tracker_handle->target_block[1] >= gridDims.y || host_tracker_handle->target_block[2] >= gridDims.z) {
                    PRINT("[SANITIZER ERROR] Invalid target block: %d, %d, %d\n", host_tracker_handle->target_block[0], host_tracker_handle->target_block[1], host_tracker_handle->target_block[2]);
                    exit(EXIT_FAILURE);
                }
                PRINT("[SANITIZER INFO] Target block: %d, %d, %d\n", host_tracker_handle->target_block[0], host_tracker_handle->target_block[1], host_tracker_handle->target_block[2]);
            } else {
                host_tracker_handle->target_block[0] = 0;
                host_tracker_handle->target_block[1] = 0;
                host_tracker_handle->target_block[2] = 0;
                PRINT("[SANITIZER INFO] No target block specified, using default (0, 0, 0)\n");
            }
            // only sample one block so the doorbell need to be updated to thread amount of one block
            uint32_t num_threads = blockDims.x * blockDims.y * blockDims.z;
            global_doorbell->num_threads = num_threads;
            global_doorbell->full = 0;
            host_tracker_handle->doorBell = global_doorbell;
        } else if (sanitizer_options.patch_name == GPU_PATCH_BLOCK_DIVERGENCE_ANALYSIS) {
            SANITIZER_SAFECALL(
                sanitizerMemset(
                    device_access_buffer, 0, sizeof(MemoryAccess) * MEMORY_ACCESS_BUFFER_SIZE, hstream));
            host_tracker_handle->currentEntry = 0;
            host_tracker_handle->numEntries = 0;
            host_tracker_handle->access_buffer = device_access_buffer;

            uint32_t num_threads =
                        blockDims.x * blockDims.y * blockDims.z * gridDims.x * gridDims.y * gridDims.z;
            global_doorbell->num_threads = num_threads;
            global_doorbell->full = 0;
            host_tracker_handle->doorBell = global_doorbell;
        }

        SANITIZER_SAFECALL(
            sanitizerMemcpyHostToDeviceAsync(
                device_tracker_handle, host_tracker_handle, sizeof(MemoryAccessTracker), hstream));
        SANITIZER_SAFECALL(sanitizerSetCallbackData(function, device_tracker_handle));
    }

    int device_id = sanitizer_ctx_to_device[context];
    yosemite_kernel_start_callback(functionName, device_id);
}


void LaunchEndCallback(
    CUcontext context,
    CUstream stream,
    CUfunction function,
    std::string functionName,
    Sanitizer_StreamHandle hstream,
    Sanitizer_StreamHandle phstream)
{
    // sampling
    if (sanitizer_options.grid_launch_id % sanitizer_options.sample_rate != 0 ||
        !SanitizerKernelWhiteListCheck(functionName)) {
        return;
    }

    if (sanitizer_options.patch_name != GPU_NO_PATCH) {
        if (sanitizer_options.patch_name == GPU_PATCH_APP_METRIC) {
            SANITIZER_SAFECALL(sanitizerStreamSynchronize(hstream));
            SANITIZER_SAFECALL(
                sanitizerMemcpyDeviceToHost(
                    host_tracker_handle, device_tracker_handle, sizeof(MemoryAccessTracker), hstream));
            SANITIZER_SAFECALL(
                sanitizerMemcpyDeviceToHost(
                    host_access_state, device_access_state, sizeof(MemoryAccessState), hstream));
            host_tracker_handle->access_state = host_access_state;
            yosemite_gpu_data_analysis(host_tracker_handle, host_tracker_handle->accessCount);
        } else if (sanitizer_options.patch_name == GPU_PATCH_ROOFLINE_SIZE) {
            SANITIZER_SAFECALL(sanitizerStreamSynchronize(hstream));
            SANITIZER_SAFECALL(
                sanitizerMemcpyDeviceToHost(
                    host_tracker_handle, device_tracker_handle, sizeof(MemoryAccessTracker), hstream));
            yosemite_gpu_data_analysis(host_tracker_handle, 0);

        } else if (sanitizer_options.patch_name == GPU_PATCH_MEM_TRACE) {
            while (true)
            {
                if (global_doorbell->num_threads == 0) {
                    break;
                }

                if (global_doorbell->full) {
                    PRINT("[SANITIZER INFO] Doorbell full with size %u. Analyzing data...\n",
                                                                            MEMORY_ACCESS_BUFFER_SIZE);
                    SANITIZER_SAFECALL(
                        sanitizerMemcpyDeviceToHost(host_access_buffer, device_access_buffer,
                                            sizeof(MemoryAccess) * MEMORY_ACCESS_BUFFER_SIZE, phstream));
                    yosemite_gpu_data_analysis(host_access_buffer, MEMORY_ACCESS_BUFFER_SIZE);
                    global_doorbell->full = 0;
                }
            }
            SANITIZER_SAFECALL(sanitizerStreamSynchronize(hstream));
            SANITIZER_SAFECALL(
                sanitizerMemcpyDeviceToHost(
                    host_tracker_handle, device_tracker_handle, sizeof(MemoryAccessTracker), hstream));

            auto numEntries = host_tracker_handle->numEntries;
            SANITIZER_SAFECALL(
                sanitizerMemcpyDeviceToHost(
                    host_access_buffer, device_access_buffer, sizeof(MemoryAccess) * numEntries, hstream));

            yosemite_gpu_data_analysis(host_access_buffer, numEntries);
        } else if (sanitizer_options.patch_name == GPU_PATCH_HOT_ANALYSIS) {
            SANITIZER_SAFECALL(sanitizerStreamSynchronize(hstream));
            SANITIZER_SAFECALL(
                sanitizerMemcpyDeviceToHost(
                    host_access_state, device_access_state, sizeof(MemoryAccessState), hstream));

            host_tracker_handle->access_state = host_access_state;
            yosemite_gpu_data_analysis(host_access_state, host_access_state->size);
        } else if (sanitizer_options.patch_name == GPU_PATCH_UVM_ADVISOR) {
            SANITIZER_SAFECALL(sanitizerStreamSynchronize(hstream));
            SANITIZER_SAFECALL(
                sanitizerMemcpyDeviceToHost(
                    host_tracker_handle, device_tracker_handle, sizeof(MemoryAccessTracker), hstream));
            SANITIZER_SAFECALL(
                sanitizerMemcpyDeviceToHost(
                    host_access_state, device_access_state, sizeof(MemoryAccessState), hstream));
            SANITIZER_SAFECALL(
                sanitizerMemcpyDeviceToHost(
                    host_tensor_access_state, device_tensor_access_state, sizeof(TensorAccessState), hstream));

            host_tracker_handle->access_state = host_access_state;
            host_tracker_handle->tensor_access_state = host_tensor_access_state;
            yosemite_gpu_data_analysis(host_tracker_handle, 0);
        } else if (sanitizer_options.patch_name == GPU_PATCH_APP_ANALYSIS) {
            SANITIZER_SAFECALL(sanitizerStreamSynchronize(hstream));
            SANITIZER_SAFECALL(
                sanitizerMemcpyDeviceToHost(
                    host_tracker_handle, device_tracker_handle, sizeof(MemoryAccessTracker), hstream));
            SANITIZER_SAFECALL(
                sanitizerMemcpyDeviceToHost(
                    host_access_state, device_access_state, sizeof(MemoryAccessState), hstream));
            SANITIZER_SAFECALL(
                sanitizerMemcpyDeviceToHost(
                    host_tensor_access_state, device_tensor_access_state, sizeof(TensorAccessState), hstream));

            host_tracker_handle->access_state = host_access_state;
            host_tracker_handle->tensor_access_state = host_tensor_access_state;
            yosemite_gpu_data_analysis(host_tracker_handle, host_tracker_handle->accessCount);
        } else if (sanitizer_options.patch_name == GPU_PATCH_APP_ANALYSIS_CPU) {
            while (true)
            {
                if (global_doorbell->num_threads == 0) {
                    break;
                }

                if (global_doorbell->full) {
                    PRINT("[SANITIZER INFO] Doorbell full with size %u. Analyzing data...\n",
                                                                                    MEMORY_ACCESS_BUFFER_SIZE);
                    SANITIZER_SAFECALL(
                        sanitizerMemcpyDeviceToHost(host_access_buffer, device_access_buffer,
                                                    sizeof(MemoryAccess) * MEMORY_ACCESS_BUFFER_SIZE, phstream));
                    yosemite_gpu_data_analysis(host_access_buffer, MEMORY_ACCESS_BUFFER_SIZE);
                    SANITIZER_SAFECALL(sanitizerMemset(
                            device_access_buffer, 0, sizeof(MemoryAccess) * MEMORY_ACCESS_BUFFER_SIZE, phstream));

                    global_doorbell->full = 0;
                }
            }
            SANITIZER_SAFECALL(sanitizerStreamSynchronize(hstream));
            SANITIZER_SAFECALL(
                sanitizerMemcpyDeviceToHost(
                    host_tracker_handle, device_tracker_handle, sizeof(MemoryAccessTracker), hstream));

            auto numEntries = host_tracker_handle->numEntries;
            SANITIZER_SAFECALL(
                sanitizerMemcpyDeviceToHost(
                    host_access_buffer, device_access_buffer, sizeof(MemoryAccess) * numEntries, hstream));

            yosemite_gpu_data_analysis(host_access_buffer, numEntries);
        } else if (sanitizer_options.patch_name == GPU_PATCH_TIME_HOTNESS_CPU) {
            while (true)
            {
                if (global_doorbell->num_threads == 0) {
                    break;
                }

                if (global_doorbell->full) {
                    PRINT("[SANITIZER INFO] Doorbell full with size %u. Analyzing data...\n",
                                                                                MEMORY_ACCESS_BUFFER_SIZE);
                    SANITIZER_SAFECALL(
                        sanitizerMemcpyDeviceToHost(host_access_buffer, device_access_buffer,
                                                    sizeof(MemoryAccess) * MEMORY_ACCESS_BUFFER_SIZE, phstream));
                    yosemite_gpu_data_analysis(host_access_buffer, MEMORY_ACCESS_BUFFER_SIZE);
                    SANITIZER_SAFECALL(
                        sanitizerMemset(
                            device_access_buffer, 0, sizeof(MemoryAccess) * MEMORY_ACCESS_BUFFER_SIZE, phstream));

                    global_doorbell->full = 0;
                }
            }
            SANITIZER_SAFECALL(sanitizerStreamSynchronize(hstream));
            SANITIZER_SAFECALL(
                sanitizerMemcpyDeviceToHost(
                    host_tracker_handle, device_tracker_handle, sizeof(MemoryAccessTracker), hstream));

            auto numEntries = host_tracker_handle->numEntries;
            SANITIZER_SAFECALL(
                sanitizerMemcpyDeviceToHost(
                    host_access_buffer, device_access_buffer, sizeof(MemoryAccess) * numEntries, hstream));

            yosemite_gpu_data_analysis(host_access_buffer, numEntries);
        } else if (sanitizer_options.patch_name == GPU_PATCH_HEATMAP_ANALYSIS) {
            while (true)
            {
                if (global_doorbell->num_threads == 0) {
                    break;
                }

                if (global_doorbell->full) {
                    PRINT("[SANITIZER INFO] Doorbell full with size %u. Analyzing data...\n",
                                                                            MEMORY_ACCESS_BUFFER_SIZE);
                    SANITIZER_SAFECALL(
                        sanitizerMemcpyDeviceToHost(host_access_buffer, device_access_buffer,
                                            sizeof(MemoryAccess) * MEMORY_ACCESS_BUFFER_SIZE, phstream));
                    yosemite_gpu_data_analysis(host_access_buffer, MEMORY_ACCESS_BUFFER_SIZE);
                    global_doorbell->full = 0;
                }
            }
            SANITIZER_SAFECALL(sanitizerStreamSynchronize(hstream));
            SANITIZER_SAFECALL(
                sanitizerMemcpyDeviceToHost(
                    host_tracker_handle, device_tracker_handle, sizeof(MemoryAccessTracker), hstream));

            auto numEntries = host_tracker_handle->numEntries;
            SANITIZER_SAFECALL(
                sanitizerMemcpyDeviceToHost(
                    host_access_buffer, device_access_buffer, sizeof(MemoryAccess) * numEntries, hstream));

            yosemite_gpu_data_analysis(host_access_buffer, numEntries);
        } else if (sanitizer_options.patch_name == GPU_PATCH_BLOCK_DIVERGENCE_ANALYSIS) {
            while (true)
            {
                if (global_doorbell->num_threads == 0) {
                    break;
                }

                if (global_doorbell->full) {
                    PRINT("[SANITIZER INFO] Doorbell full with size %u. Analyzing data...\n",
                                                                            MEMORY_ACCESS_BUFFER_SIZE);
                    SANITIZER_SAFECALL(
                        sanitizerMemcpyDeviceToHost(host_access_buffer, device_access_buffer,
                                            sizeof(MemoryAccess) * MEMORY_ACCESS_BUFFER_SIZE, phstream));
                    yosemite_gpu_data_analysis(host_access_buffer, MEMORY_ACCESS_BUFFER_SIZE);
                    global_doorbell->full = 0;
                }
            }
            SANITIZER_SAFECALL(sanitizerStreamSynchronize(hstream));
            SANITIZER_SAFECALL(
                sanitizerMemcpyDeviceToHost(
                    host_tracker_handle, device_tracker_handle, sizeof(MemoryAccessTracker), hstream));

            auto numEntries = host_tracker_handle->numEntries;
            SANITIZER_SAFECALL(
                sanitizerMemcpyDeviceToHost(
                    host_access_buffer, device_access_buffer, sizeof(MemoryAccess) * numEntries, hstream));

            yosemite_gpu_data_analysis(host_access_buffer, numEntries);
        }
    } else {
        SANITIZER_SAFECALL(sanitizerStreamSynchronize(hstream));
    }

    int device_id = sanitizer_ctx_to_device[context];
    yosemite_kernel_end_callback(functionName, device_id);
}


void ComputeSanitizerCallback(
    void* userdata,
    Sanitizer_CallbackDomain domain,
    Sanitizer_CallbackId cbid,
    const void* cbdata)
{
    // Skip sanitizer kernel callback (python interface)
    if (!sanitizer_options.sanitizer_callback_enabled &&
        !((domain == SANITIZER_CB_DOMAIN_RESOURCE && cbid == SANITIZER_CBID_RESOURCE_MODULE_LOADED) ||
        (domain == SANITIZER_CB_DOMAIN_RESOURCE && cbid == SANITIZER_CBID_RESOURCE_MODULE_UNLOAD_STARTING)))
    {
        return;
    }

    if (sanitizer_cuda_api_internal()) return;

    switch (domain)
    {
        case SANITIZER_CB_DOMAIN_RESOURCE:
            switch (cbid)
            {
                case SANITIZER_CBID_RESOURCE_MODULE_LOADED:
                {
                    auto* pModuleData = (Sanitizer_ResourceModuleData*)cbdata;
                    PRINT("[SANITIZER INFO] Module %p loaded on context %p\n",
                            &pModuleData->module, &pModuleData->context);

                    ModuleLoadedCallback(pModuleData->module);
                    break;
                }
                case SANITIZER_CBID_RESOURCE_MODULE_UNLOAD_STARTING:
                {
                    auto* pModuleData = (Sanitizer_ResourceModuleData*)cbdata;
                    PRINT("[SANITIZER INFO] Module %p unload starting on context %p\n",
                            &pModuleData->module, &pModuleData->context);

                    ModuleUnloadedCallback(pModuleData->module);
                    break;
                }
                case SANITIZER_CBID_RESOURCE_CONTEXT_CREATION_STARTING:
                {
                    auto* pContextData = (Sanitizer_ResourceContextData*)cbdata;
                    sanitizer_ctx_to_device.try_emplace(pContextData->context, pContextData->device);
                    PRINT("[SANITIZER INFO] Context %p creation starting on device %p\n",
                            &pContextData->context, &pContextData->device);
                    break;
                }
                case SANITIZER_CBID_RESOURCE_CONTEXT_CREATION_FINISHED:
                {
                    auto* pContextData = (Sanitizer_ResourceContextData*)cbdata;
                    PRINT("[SANITIZER INFO] Context %p creation finished on device %p\n",
                            &pContextData->context, &pContextData->device);
                    break;
                }
                case SANITIZER_CBID_RESOURCE_CONTEXT_DESTROY_STARTING:
                {
                    auto* pContextData = (Sanitizer_ResourceContextData*)cbdata;
                    sanitizer_ctx_to_device.erase(pContextData->context);
                    PRINT("[SANITIZER INFO] Context %p destroy starting on device %p\n",
                            &pContextData->context, &pContextData->device);
                    break;
                }
                case SANITIZER_CBID_RESOURCE_CONTEXT_DESTROY_FINISHED:
                {
                    auto* pContextData = (Sanitizer_ResourceContextData*)cbdata;
                    PRINT("[SANITIZER INFO] Context %p destroy finished on device %p\n",
                            &pContextData->context, &pContextData->device);
                    break;
                }
                case SANITIZER_CBID_RESOURCE_STREAM_CREATED:
                {
                    auto* pStreamData = (Sanitizer_ResourceStreamData*)cbdata;
                    PRINT("[SANITIZER INFO] Stream %p created on context %p\n",
                            &pStreamData->stream, &pStreamData->context);
                    break;
                }
                case SANITIZER_CBID_RESOURCE_STREAM_DESTROY_STARTING:
                {
                    auto* pStreamData = (Sanitizer_ResourceStreamData*)cbdata;
                    PRINT("[SANITIZER INFO] Stream %p destroy starting on context %p\n",
                            &pStreamData->stream, &pStreamData->context);
                    break;
                }
                case SANITIZER_CBID_RESOURCE_STREAM_DESTROY_FINISHED:
                {
                    auto* pStreamData = (Sanitizer_ResourceStreamData*)cbdata;
                    PRINT("[SANITIZER INFO] Stream %p destroy finished on context %p\n",
                            &pStreamData->stream, &pStreamData->context);
                    break;
                }
                case SANITIZER_CBID_RESOURCE_DEVICE_MEMORY_ALLOC:
                {
                    auto *pModuleData = (Sanitizer_ResourceMemoryData *)cbdata;
                    if (pModuleData->flags == SANITIZER_MEMORY_FLAG_CG_RUNTIME || pModuleData->size == 0) {
                        break;
                    }

                    CUdevice device_id = sanitizer_ctx_to_device[pModuleData->context];

                    PRINT("[SANITIZER INFO] Malloc memory %p with size %lu (flag: %u) on device %d\n",
                            (void*)pModuleData->address, pModuleData->size, pModuleData->flags, device_id);
                    PRINT("[SANITIZER INFO] Sector tag: %p, end tag: %p\n", (void*)(pModuleData->address >> 5), (void*)((pModuleData->address + pModuleData->size - 1) >> 5));
                    yosemite_alloc_callback(
                            pModuleData->address, pModuleData->size, pModuleData->flags, device_id);
                    break;
                }
                case SANITIZER_CBID_RESOURCE_DEVICE_MEMORY_FREE:
                {
                    auto *pModuleData = (Sanitizer_ResourceMemoryData *)cbdata;
                    if (pModuleData->flags == SANITIZER_MEMORY_FLAG_CG_RUNTIME || pModuleData->size == 0)
                        break;

                    CUdevice device_id = sanitizer_ctx_to_device[pModuleData->context];

                    PRINT("[SANITIZER INFO] Free memory %p with size %lu (flag: %u) on device %d\n",
                            (void*)pModuleData->address, pModuleData->size, pModuleData->flags, device_id);

                    yosemite_free_callback(
                            pModuleData->address, pModuleData->size, pModuleData->flags, device_id);
                    break;
                }
                default:
                    break;
            }
            break;
        case SANITIZER_CB_DOMAIN_LAUNCH:
            switch (cbid)
            {
                case SANITIZER_CBID_LAUNCH_BEGIN:
                {
                    auto* pLaunchData = (Sanitizer_LaunchData*)cbdata;
                    dim3 blockDims, gridDims;
                    blockDims.x = pLaunchData->blockDim_x;
                    blockDims.y = pLaunchData->blockDim_y;
                    blockDims.z = pLaunchData->blockDim_z;
                    gridDims.x = pLaunchData->gridDim_x;
                    gridDims.y = pLaunchData->gridDim_y;
                    gridDims.z = pLaunchData->gridDim_z;
                    auto func_name = sanitizer_demangled_name_get(pLaunchData->functionName);

                    CUdevice device_id = sanitizer_ctx_to_device[pLaunchData->context];
                    uint64_t pc;
                    uint64_t size;
                    SANITIZER_SAFECALL(sanitizerGetFunctionPcAndSize(pLaunchData->module, pLaunchData->functionName, &pc, &size));
                    PRINT("[SANITIZER INFO] Launching kernel %s <<<(%u, %u, %u), (%u, %u, %u)>>> on device %d, pc: 0x%lx, size: %lu\n",
                            func_name,
                            pLaunchData->gridDim_x, pLaunchData->gridDim_y, pLaunchData->gridDim_z,
                            pLaunchData->blockDim_x, pLaunchData->blockDim_y, pLaunchData->blockDim_z,
                            device_id, pc, size);

                    LaunchBeginCallback(pLaunchData->context, pLaunchData->module, pLaunchData->function,
                                    func_name, pLaunchData->hStream, blockDims, gridDims);
                    break;
                }
                case SANITIZER_CBID_LAUNCH_END:
                {
                    auto* pLaunchData = (Sanitizer_LaunchData*)cbdata;
                    CUstream p_stream;
                    Sanitizer_StreamHandle p_stream_handle;
                    sanitizer_priority_stream_get(pLaunchData->context, &p_stream);
                    sanitizerGetStreamHandle(pLaunchData->context, p_stream, &p_stream_handle);
                    auto func_name = sanitizer_demangled_name_get(pLaunchData->functionName);

                    CUdevice device_id = sanitizer_ctx_to_device[pLaunchData->context];

                    LaunchEndCallback(pLaunchData->context, pLaunchData->stream, pLaunchData->function,
                                    func_name, pLaunchData->hStream, p_stream_handle);

                    PRINT("[SANITIZER INFO] Kernel %s finished on device %d\n", func_name, device_id);
                    break;
                }
                default:
                    break;
            }
            break;
        case SANITIZER_CB_DOMAIN_MEMCPY:
            switch (cbid)
            {
                case SANITIZER_CBID_MEMCPY_STARTING:
                {
                    auto* pMemcpyData = (Sanitizer_MemcpyData*)cbdata;
                    auto direction = (pMemcpyData->direction == SANITIZER_MEMCPY_DIRECTION_HOST_TO_HOST) ? "H2H" :
                                     (pMemcpyData->direction == SANITIZER_MEMCPY_DIRECTION_HOST_TO_DEVICE) ? "H2D" :
                                     (pMemcpyData->direction == SANITIZER_MEMCPY_DIRECTION_DEVICE_TO_HOST) ? "D2H" :
                                     (pMemcpyData->direction == SANITIZER_MEMCPY_DIRECTION_DEVICE_TO_DEVICE) ? "D2D" :
                                     "UNKNOWN";
                    
                    // CUdevice device_id = sanitizer_ctx_to_device[pMemcpyData->apiContext];
                    CUdevice device_id = sanitizer_ctx_to_device[pMemcpyData->srcContext];

                    PRINT("[SANITIZER INFO] Memcpy %p -> %p with size %lu, async: %d, direction: %s on device %d\n",
                            (void*)pMemcpyData->srcAddress, (void*)pMemcpyData->dstAddress, pMemcpyData->size,
                            pMemcpyData->isAsync, direction, device_id);

                    yosemite_memcpy_callback(pMemcpyData->dstAddress, pMemcpyData->srcAddress,pMemcpyData->size,
                                                pMemcpyData->isAsync, (uint32_t)pMemcpyData->direction, device_id);
                    break;
                }
                default:
                    break;
            }
            break;
        case SANITIZER_CB_DOMAIN_MEMSET:
            switch (cbid)
            {
                case SANITIZER_CBID_MEMSET_STARTING:
                {
                    auto* pMemsetData = (Sanitizer_MemsetData*)cbdata;
                    CUdevice device_id = sanitizer_ctx_to_device[pMemsetData->context];

                    PRINT("[SANITIZER INFO] Memset %p with size %lu, value %d, async: %d on device %d\n",
                            (void*)pMemsetData->address, pMemsetData->width, pMemsetData->value,
                            pMemsetData->isAsync, device_id);

                    yosemite_memset_callback(pMemsetData->address, pMemsetData->width,
                                                pMemsetData->value, pMemsetData->isAsync, device_id);
                    break;
                }
                default:
                    break;
            }
            break;
        case SANITIZER_CB_DOMAIN_SYNCHRONIZE:
            switch (cbid)
            {
                case SANITIZER_CBID_SYNCHRONIZE_STREAM_SYNCHRONIZED:
                {
                    auto* pSyncData = (Sanitizer_SynchronizeData*)cbdata;
                    CUdevice device_id = sanitizer_ctx_to_device[pSyncData->context];

                    PRINT("[SANITIZER INFO] Synchronize stream %p finished on context %p on device %d\n",
                            &pSyncData->stream, &pSyncData->context, device_id);
                    break;
                }
                case SANITIZER_CBID_SYNCHRONIZE_CONTEXT_SYNCHRONIZED:
                {
                    auto* pSyncData = (Sanitizer_SynchronizeData*)cbdata;
                    CUdevice device_id = sanitizer_ctx_to_device[pSyncData->context];

                    PRINT("[SANITIZER INFO] Synchronize context %p finished on device %d\n",
                            &pSyncData->context, device_id);
                    break;
                }
                default:
                    break;
            }
            break;
        default:
            break;
    }
}


void register_module_patches() {
    for (auto it = sanitizer_active_modules.begin(); it != sanitizer_active_modules.end(); ++it)
    {
        if (!it->second) {
            ModuleLoadedCallback(it->first);
            it->second = true;
        }
    }
}

void unregister_module_patches() {
    for (auto it = sanitizer_active_modules.begin(); it != sanitizer_active_modules.end(); ++it)
    {
        if (it->second) {
            SANITIZER_SAFECALL(sanitizerUnpatchModule(it->first));
            it->second = false;
        }
    }
}


void enable_compute_sanitizer(bool enable) {
    sanitizer_options.sanitizer_callback_enabled = enable;

    if (enable) {
        register_module_patches();
    } else {
        unregister_module_patches();
    }
}


int InitializeInjection()
{
    sanitizer_debug_wait();
    LoadKernelWhiteList(std::getenv("YOSEMITE_KERNEL_WHITELIST"));
    Sanitizer_SubscriberHandle handle;
    SANITIZER_SAFECALL(sanitizerSubscribe(&handle, ComputeSanitizerCallback, nullptr));
    SANITIZER_SAFECALL(sanitizerEnableDomain(1, handle, SANITIZER_CB_DOMAIN_RESOURCE));
    SANITIZER_SAFECALL(sanitizerEnableDomain(1, handle, SANITIZER_CB_DOMAIN_LAUNCH));
    SANITIZER_SAFECALL(sanitizerEnableDomain(1, handle, SANITIZER_CB_DOMAIN_MEMCPY));
    SANITIZER_SAFECALL(sanitizerEnableDomain(1, handle, SANITIZER_CB_DOMAIN_MEMSET));
    SANITIZER_SAFECALL(sanitizerEnableDomain(1, handle, SANITIZER_CB_DOMAIN_SYNCHRONIZE));

    yosemite_init(sanitizer_options);

    // register tensor malloc and free callback
    if (sanitizer_options.torch_prof_enabled) {
        enable_torch_scope();
        register_torch_scope_callback(
            TORCH_SCOPE_TENSOR_MALLOC, (torch_scope_callback_t)SanitizerTensorMallocCallback);
        register_torch_scope_callback(
            TORCH_SCOPE_TENSOR_FREE, (torch_scope_callback_t)SanitizerTensorFreeCallback);
        register_torch_scope_callback(
            TORCH_SCOPE_OPERATOR_START, (torch_scope_callback_t)SanitizerOperatorStartCallback);
        register_torch_scope_callback(
            TORCH_SCOPE_OPERATOR_END, (torch_scope_callback_t)SanitizerOperatorEndCallback);
    }

    return 0;
}


void cleanup(void) {
    yosemite_terminate();
}

__attribute__((constructor))
void initializer(void) {
    atexit(cleanup);
}

__attribute__((destructor))
void finalizer(void) {
}

int __global_initializer__ = InitializeInjection();
