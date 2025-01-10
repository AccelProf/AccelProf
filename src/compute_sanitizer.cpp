#include "gpu_patch.h"
#include "sanitizer_helper.h"
#include "tensor_scope.h"
#include "sanalyzer.h"

#include <sanitizer.h>
#include <vector_types.h>

#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <vector>
#include <cstring>
#include <cassert>


#define SANITIZER_VERBOSE 1

#if SANITIZER_VERBOSE
#define PRINT(...) do { fprintf(stdout, __VA_ARGS__); fflush(stdout); } while (0)
#else
#define PRINT(...)
#endif


static MemoryAccessTracker* host_tracker_handle = nullptr;
static MemoryAccessTracker* device_tracker_handle = nullptr;
static MemoryAccess* host_access_buffer = nullptr;
static MemoryAccess* device_access_buffer = nullptr;
static MemoryAccessState* host_access_state = nullptr;
static MemoryAccessState* device_access_state = nullptr;
static DoorBell* global_doorbell = nullptr;

static SanitizerOptions_t sanitizer_options;


static void tensor_malloc_callback(uint64_t ptr, int64_t size, int64_t allocated, int64_t reserved) {
    PRINT("[SANITIZER INFO] Malloc tensor %p with size %ld, allocated %ld, reserved %ld\n", ptr, size, allocated, reserved);
    yosemite_tensor_malloc_callback(ptr, size, allocated, reserved);
}


static void tensor_free_callback(uint64_t ptr, int64_t size, int64_t allocated, int64_t reserved) {
    PRINT("[SANITIZER INFO] Free tensor %p with size %ld, allocated %ld, reserved %ld\n", ptr, size, allocated, reserved);
    yosemite_tensor_free_callback(ptr, size, allocated, reserved);
}


void ModuleLoadedCallback(CUmodule module, CUcontext context)
{
    if (sanitizer_options.patch_name == GPU_NO_PATCH) {
        return;
    }
    const char* env_name = std::getenv("CU_PROF_HOME");
    std::string patch_path;
    if (env_name) {
        patch_path = std::string(env_name) + "/lib/gpu_patch/";
    } else {
        std::cerr << "Failed to load fatbin. No patch path specified." << std::endl;
    }

    // Instrument user code
    std::string fatbin_file = patch_path + sanitizer_options.patch_file;
    SanitizerResult result;
    result = sanitizerAddPatchesFromFile(fatbin_file.c_str(), 0);
    if (result != SANITIZER_SUCCESS)
        std::cerr << "Failed to load fatbin. Check its path and included SM architecture." << std::endl;

    sanitizerPatchInstructions(SANITIZER_INSTRUCTION_GLOBAL_MEMORY_ACCESS, module, "MemoryGlobalAccessCallback");
    // sanitizerPatchInstructions(SANITIZER_INSTRUCTION_SHARED_MEMORY_ACCESS, module, "MemorySharedAccessCallback");
    // sanitizerPatchInstructions(SANITIZER_INSTRUCTION_LOCAL_MEMORY_ACCESS, module, "MemoryLocalAccessCallback");
    // sanitizerPatchInstructions(SANITIZER_INSTRUCTION_MEMCPY_ASYNC, module, "MemcpyAsyncCallback");
    sanitizerPatchInstructions(SANITIZER_INSTRUCTION_BLOCK_EXIT, module, "BlockExitCallback");
    sanitizerPatchModule(module);

    if (!device_tracker_handle) {
        sanitizerAlloc(context, (void**)&device_tracker_handle, sizeof(MemoryAccessTracker));
    }
    if (!host_tracker_handle) {
        sanitizerAllocHost(context, (void**)&host_tracker_handle, sizeof(MemoryAccessTracker));
    }

    if (sanitizer_options.patch_name == GPU_PATCH_APP_METRIC) {
        if (!device_access_state)
            sanitizerAlloc(context, (void**)&device_access_state, sizeof(MemoryAccessState));

        if (!host_access_state) {
            sanitizerAllocHost(context, (void**)&host_access_state, sizeof(MemoryAccessState));
        }
        if (!global_doorbell) {
            sanitizerAllocHost(context, (void**)&global_doorbell, sizeof(DoorBell));
        }
    } else if (sanitizer_options.patch_name == GPU_PATCH_MEM_TRACE) {
        if (!device_access_buffer) {
            sanitizerAlloc(context, (void**)&device_access_buffer, sizeof(MemoryAccess) * MEMORY_ACCESS_BUFFER_SIZE);
        }
        if (!host_access_buffer) {
            sanitizerAllocHost(context, (void**)&host_access_buffer, sizeof(MemoryAccess) * MEMORY_ACCESS_BUFFER_SIZE);
        }
        if (!global_doorbell) {
            sanitizerAllocHost(context, (void**)&global_doorbell, sizeof(DoorBell));
        }
    }
}


void LaunchBeginCallback(
    CUcontext context,
    CUfunction function,
    std::string functionName,
    Sanitizer_StreamHandle hstream,
    dim3 blockDims,
    dim3 gridDims)
{
    // Skip sanitizer kernel callback (python interface), only works with app_metric for now
    if (sanitizer_options.skip_sanitizer_callback && sanitizer_options.patch_name == GPU_PATCH_APP_METRIC) {
        global_doorbell->skip_patch = 1;
        host_tracker_handle->doorBell = global_doorbell;
        sanitizerMemcpyHostToDeviceAsync(device_tracker_handle, host_tracker_handle, sizeof(MemoryAccessTracker), hstream);
        sanitizerSetCallbackData(function, device_tracker_handle);
        return;
    }
    if (sanitizer_options.patch_name != GPU_NO_PATCH) {
        if (sanitizer_options.patch_name == GPU_PATCH_APP_METRIC) {
            memset(host_access_state, 0, sizeof(MemoryAccessState));
            yosemite_query_active_ranges(host_access_state->start_end, MAX_ACTIVE_ALLOCATIONS, &host_access_state->size);
            sanitizerMemcpyHostToDeviceAsync(device_access_state, host_access_state, sizeof(MemoryAccessState), hstream);
            host_tracker_handle->accessCount = 0;
            global_doorbell->skip_patch = 0;
            host_tracker_handle->doorBell = global_doorbell;
            host_tracker_handle->access_state = device_access_state;
        } else if (sanitizer_options.patch_name == GPU_PATCH_MEM_TRACE) {
            sanitizerMemset(device_access_buffer, 0, sizeof(MemoryAccess) * MEMORY_ACCESS_BUFFER_SIZE, hstream);
            host_tracker_handle->currentEntry = 0;
            host_tracker_handle->numEntries = 0;
            host_tracker_handle->access_buffer = device_access_buffer;

            uint32_t num_threads = blockDims.x * blockDims.y * blockDims.z * gridDims.x * gridDims.y * gridDims.z;
            global_doorbell->num_threads = num_threads;
            global_doorbell->full = 0;
            host_tracker_handle->doorBell = global_doorbell;
        }        

        sanitizerMemcpyHostToDeviceAsync(device_tracker_handle, host_tracker_handle, sizeof(MemoryAccessTracker), hstream);
        sanitizerSetCallbackData(function, device_tracker_handle);
    }
    yosemite_kernel_start_callback(functionName);
}


void LaunchEndCallback(
    CUcontext context,
    CUstream stream,
    CUfunction function,
    std::string functionName,
    Sanitizer_StreamHandle hstream,
    Sanitizer_StreamHandle phstream)
{
    // Skip sanitizer kernel callback (python interface), only works with app_metric for now
    if (sanitizer_options.skip_sanitizer_callback && sanitizer_options.patch_name == GPU_PATCH_APP_METRIC) {
        sanitizerStreamSynchronize(hstream);
        return;
    }
    if (sanitizer_options.patch_name != GPU_NO_PATCH) {
        if (sanitizer_options.patch_name == GPU_PATCH_APP_METRIC) {
            sanitizerStreamSynchronize(hstream);
            sanitizerMemcpyDeviceToHost(host_tracker_handle, device_tracker_handle, sizeof(MemoryAccessTracker), hstream);
            sanitizerMemcpyDeviceToHost(host_access_state, device_access_state, sizeof(MemoryAccessState), hstream);
            host_tracker_handle->access_state = host_access_state;
            yosemite_gpu_data_analysis(host_tracker_handle, host_tracker_handle->accessCount);
        } else if (sanitizer_options.patch_name == GPU_PATCH_MEM_TRACE) {
            while (true)
            {
                if (global_doorbell->num_threads == 0) {
                    break;
                }

                if (global_doorbell->full) {
                    PRINT("[SANITIZER INFO] Doorbell full with size %u. Analyzing data...\n", MEMORY_ACCESS_BUFFER_SIZE);
                    sanitizerMemcpyDeviceToHost(host_access_buffer, device_access_buffer,
                                                    sizeof(MemoryAccess) * MEMORY_ACCESS_BUFFER_SIZE, phstream);
                    yosemite_gpu_data_analysis(host_access_buffer, MEMORY_ACCESS_BUFFER_SIZE);
                    global_doorbell->full = 0;
                }
            }
            sanitizerStreamSynchronize(hstream);
            sanitizerMemcpyDeviceToHost(host_tracker_handle, device_tracker_handle, sizeof(MemoryAccessTracker), hstream);

            auto numEntries = host_tracker_handle->numEntries;
            sanitizerMemcpyDeviceToHost(host_access_buffer, device_access_buffer, sizeof(MemoryAccess) * numEntries, hstream);
            yosemite_gpu_data_analysis(host_access_buffer, numEntries);
        }
    } else {
        sanitizerStreamSynchronize(hstream);
    }

    yosemite_kernel_end_callback(functionName);
}


void ComputeSanitizerCallback(
    void* userdata,
    Sanitizer_CallbackDomain domain,
    Sanitizer_CallbackId cbid,
    const void* cbdata)
{
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
                    ModuleLoadedCallback(pModuleData->module, pModuleData->context);
                    break;
                }
                case SANITIZER_CBID_RESOURCE_MODULE_UNLOAD_STARTING:
                {
                    auto* pModuleData = (Sanitizer_ResourceModuleData*)cbdata;
                    PRINT("[SANITIZER INFO] Module %p unload starting on context %p\n",
                            &pModuleData->module, &pModuleData->context);
                    break;
                }
                case SANITIZER_CBID_RESOURCE_CONTEXT_CREATION_STARTING:
                {
                    auto* pContextData = (Sanitizer_ResourceContextData*)cbdata;
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
                    if (pModuleData->flags == SANITIZER_MEMORY_FLAG_CG_RUNTIME || pModuleData->size == 0)
                        break;

                    PRINT("[SANITIZER INFO] Malloc memory %p with size %lu (flag: %u)\n",
                            pModuleData->address, pModuleData->size, pModuleData->flags);
                    yosemite_alloc_callback(pModuleData->address, pModuleData->size, pModuleData->flags);
                    break;
                }
                case SANITIZER_CBID_RESOURCE_DEVICE_MEMORY_FREE:
                {
                    auto *pModuleData = (Sanitizer_ResourceMemoryData *)cbdata;
                    if (pModuleData->flags == SANITIZER_MEMORY_FLAG_CG_RUNTIME || pModuleData->size == 0)
                        break;

                    PRINT("[SANITIZER INFO] Free memory %p with size %lu\n",
                            pModuleData->address, pModuleData->size);
                    yosemite_free_callback(pModuleData->address);
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

                    PRINT("[SANITIZER INFO] Launching kernel %s <<<(%u, %u, %u), (%u, %u, %u)>>>\n",
                            func_name,
                            pLaunchData->gridDim_x, pLaunchData->gridDim_y, pLaunchData->gridDim_z,
                            pLaunchData->blockDim_x, pLaunchData->blockDim_y, pLaunchData->blockDim_z);
                    LaunchBeginCallback(pLaunchData->context, pLaunchData->function, func_name,
                                    pLaunchData->hStream, blockDims, gridDims);
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

                    LaunchEndCallback(pLaunchData->context, pLaunchData->stream, pLaunchData->function,
                                    func_name, pLaunchData->hStream, p_stream_handle);
                    PRINT("[SANITIZER INFO] Kernel %s finished\n", func_name);
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
                                     (pMemcpyData->direction == SANITIZER_MEMCPY_DIRECTION_DEVICE_TO_DEVICE) ? "D2D" : "UNKNOWN";
                    PRINT("[SANITIZER INFO] Memcpy %p -> %p with size %lu, async: %d, direction: %s\n",
                            pMemcpyData->srcAddress, pMemcpyData->dstAddress, pMemcpyData->size,
                            pMemcpyData->isAsync, direction);
                    yosemite_memcpy_callback(pMemcpyData->dstAddress, pMemcpyData->srcAddress,pMemcpyData->size,
                                                pMemcpyData->isAsync, (uint32_t)pMemcpyData->direction);
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
                    PRINT("[SANITIZER INFO] Memset %p with size %u, value %d, async: %d\n",
                            pMemsetData->address, pMemsetData->elementSize, pMemsetData->value, pMemsetData->isAsync);
                    yosemite_memset_callback(pMemsetData->address, pMemsetData->elementSize,
                                                pMemsetData->value, pMemsetData->isAsync);
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
                    PRINT("[SANITIZER INFO] Synchronize stream %p finished on context %p\n",
                            &pSyncData->stream, &pSyncData->context);
                    break;
                }
                case SANITIZER_CBID_SYNCHRONIZE_CONTEXT_SYNCHRONIZED:
                {
                    auto* pSyncData = (Sanitizer_SynchronizeData*)cbdata;
                    PRINT("[SANITIZER INFO] Synchronize context %p finished\n", &pSyncData->context);
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


void enable_compute_sanitizer(bool enable) {
    // only works with app_metric for now
    if (sanitizer_options.patch_name != GPU_PATCH_APP_METRIC) {
        return ;
    }
    sanitizer_options.skip_sanitizer_callback = !enable;
}


void cleanup(void) {
    yosemite_terminate();
}


int InitializeInjection()
{
    sanitizer_debug_wait();
    Sanitizer_SubscriberHandle handle;
    sanitizerSubscribe(&handle, ComputeSanitizerCallback, nullptr);
    sanitizerEnableDomain(1, handle, SANITIZER_CB_DOMAIN_RESOURCE);
    sanitizerEnableDomain(1, handle, SANITIZER_CB_DOMAIN_LAUNCH);
    sanitizerEnableDomain(1, handle, SANITIZER_CB_DOMAIN_MEMCPY);
    sanitizerEnableDomain(1, handle, SANITIZER_CB_DOMAIN_MEMSET);
    sanitizerEnableDomain(1, handle, SANITIZER_CB_DOMAIN_SYNCHRONIZE);

    yosemite_init(sanitizer_options);

    // register tensor malloc and free callback
    if (sanitizer_options.torch_prof_enabled) {
        tensor_scope_enable();
        register_tensor_scope(tensor_malloc_callback, tensor_free_callback);
    }

    return 0;
}

__attribute__((constructor))
void initializer(void) {
    atexit(cleanup);
}

__attribute__((destructor))
void finalizer(void) {
}

int __global_initializer__ = InitializeInjection();
