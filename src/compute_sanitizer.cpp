#include "gpu_patch.h"


#include <sanitizer.h>
#include <vector_types.h>

#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <vector>
#include <cstring>
#include <cassert>

#include "sanalyzer.h"
#include "tensor_scope.h"


static MemoryAccessTracker* host_tacker_handle = nullptr;
static MemoryAccessTracker* device_tracker_handle = nullptr;
static MemoryAccess* host_access_buffer = nullptr;
static MemoryAccess* device_access_buffer = nullptr;
static MemoryAccessState* host_access_state = nullptr;
static MemoryAccessState* device_access_state = nullptr;

static SanitizerOptions_t sanitizer_options;


static std::string GetMemoryRWString(uint32_t flags)
{
    const bool isWrite = !!(flags & SANITIZER_MEMORY_DEVICE_FLAG_WRITE);
    const bool isRead = !!(flags & SANITIZER_MEMORY_DEVICE_FLAG_READ);

    if (isWrite && isRead) {return "Atomic";}
    else if (isRead) {return "Read";}
    else if (isWrite) {return "Write";}
    else {return "Unknown";}
}


static std::string GetMemoryTypeString(MemoryAccessType type)
{
    if (type == MemoryAccessType::Local) {return "local";}
    else if (type == MemoryAccessType::Shared) {return "shared";}
    else {return "global";}
}


static void tensor_malloc_callback(uint64_t ptr, int64_t size, int64_t allocated, int64_t reserved) {
    yosemite_tensor_malloc_callback(ptr, size, allocated, reserved);
}


static void tensor_free_callback(uint64_t ptr, int64_t size, int64_t allocated, int64_t reserved) {
    yosemite_tensor_free_callback(ptr, size, allocated, reserved);
}


void ModuleLoaded(CUmodule module, CUcontext context)
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
    sanitizerPatchInstructions(SANITIZER_INSTRUCTION_SHARED_MEMORY_ACCESS, module, "MemorySharedAccessCallback");
    sanitizerPatchInstructions(SANITIZER_INSTRUCTION_LOCAL_MEMORY_ACCESS, module, "MemoryLocalAccessCallback");
    sanitizerPatchInstructions(SANITIZER_INSTRUCTION_MEMCPY_ASYNC, module, "MemcpyAsyncCallback");
    sanitizerPatchInstructions(SANITIZER_INSTRUCTION_BLOCK_EXIT, module, "BlockExitCallback");
    sanitizerPatchModule(module);

    if (!device_tracker_handle) {
        sanitizerAlloc(context, (void**)&device_tracker_handle, sizeof(*device_tracker_handle));
    }
    if (!host_tacker_handle) {
        sanitizerAllocHost(context, (void**)&host_tacker_handle, sizeof(*host_tacker_handle));
    }

    if (sanitizer_options.patch_name == GPU_PATCH_APP_METRIC) {
        if (!device_access_state)
            sanitizerAlloc(context, (void**)&device_access_state, sizeof(MemoryAccessState) * MAX_ACTIVE_ALLOCATIONS);

        if (!host_access_state) {
            sanitizerAllocHost(context, (void**)&host_access_state, sizeof(MemoryAccessState) * MAX_ACTIVE_ALLOCATIONS);
        }
    } else if (sanitizer_options.patch_name == GPU_PATCH_MEM_TRACE) {
        if (!device_access_buffer) {
            sanitizerAlloc(context, (void**)&device_access_buffer, sizeof(MemoryAccess) * MEMORY_ACCESS_BUFFER_SIZE);
        }
        if (!host_access_buffer) {
            sanitizerAllocHost(context, (void**)&host_access_buffer, sizeof(MemoryAccess) * MEMORY_ACCESS_BUFFER_SIZE);
        }
    }
}


void LaunchBegin(
    CUcontext context,
    CUfunction function,
    std::string functionName,
    Sanitizer_StreamHandle hstream,
    dim3 blockDims,
    dim3 gridDims)
{
    // std::cout << std::endl << "Launch " << functionName << std::endl;
    if (sanitizer_options.patch_name != GPU_NO_PATCH) {
        if (sanitizer_options.patch_name == GPU_PATCH_APP_METRIC) {
            memset(host_access_state, 0, sizeof(MemoryAccessState) * MAX_ACTIVE_ALLOCATIONS);
            yosemite_query_active_ranges(host_access_state->start_end, MAX_ACTIVE_ALLOCATIONS, &host_access_state->size);
            sanitizerMemcpyHostToDeviceAsync(device_access_state, host_access_state, sizeof(*device_access_state), hstream);
        } else if (sanitizer_options.patch_name == GPU_PATCH_MEM_TRACE) {
            sanitizerMemset(device_access_buffer, 0, sizeof(MemoryAccess) * MEMORY_ACCESS_BUFFER_SIZE, hstream);
        }

        uint32_t num_threads = blockDims.x * blockDims.y * blockDims.z * gridDims.x * gridDims.y * gridDims.z;
        host_tacker_handle->currentEntry = 0;
        host_tacker_handle->maxEntry = MEMORY_ACCESS_BUFFER_SIZE;
        host_tacker_handle->numThreads = num_threads;
        host_tacker_handle->accessCount = 0;
        host_tacker_handle->accesses = device_access_buffer;
        host_tacker_handle->states = device_access_state;

        sanitizerMemcpyHostToDeviceAsync(device_tracker_handle, host_tacker_handle, sizeof(*device_tracker_handle), hstream);
        sanitizerSetCallbackData(function, device_tracker_handle);
    }
    yosemite_kernel_start_callback(functionName);
}


void LaunchEnd(
    CUcontext context,
    CUstream stream,
    CUfunction function,
    std::string functionName,
    Sanitizer_StreamHandle hstream)
{
    if (sanitizer_options.patch_name != GPU_NO_PATCH) {
        if (sanitizer_options.patch_name == GPU_PATCH_APP_METRIC) {
            sanitizerStreamSynchronize(hstream);
            sanitizerMemcpyDeviceToHost(host_tacker_handle, device_tracker_handle, sizeof(*device_tracker_handle), hstream);
            sanitizerMemcpyDeviceToHost(host_access_state, device_access_state, sizeof(*device_access_state), hstream);
            host_tacker_handle->states = host_access_state;

            yosemite_gpu_data_analysis(host_tacker_handle, host_tacker_handle->accessCount);
        } else if (sanitizer_options.patch_name == GPU_PATCH_MEM_TRACE) {
            while (true)
            {
                sanitizerMemcpyDeviceToHost(host_tacker_handle, device_tracker_handle, sizeof(*device_tracker_handle), hstream);
                if (host_tacker_handle->numThreads == 0) {
                    break;
                }
            }
            sanitizerStreamSynchronize(hstream);
            sanitizerMemcpyDeviceToHost(host_tacker_handle, device_tracker_handle, sizeof(*device_tracker_handle), hstream);

            auto numEntries = std::min(host_tacker_handle->currentEntry, host_tacker_handle->maxEntry);
            sanitizerMemcpyDeviceToHost(host_access_buffer, host_tacker_handle->accesses, sizeof(MemoryAccess) * numEntries, hstream);
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
    switch (domain)
    {
        case SANITIZER_CB_DOMAIN_RESOURCE:
            switch (cbid)
            {
                case SANITIZER_CBID_RESOURCE_MODULE_LOADED:
                {
                    auto* pModuleData = (Sanitizer_ResourceModuleData*)cbdata;
                    ModuleLoaded(pModuleData->module, pModuleData->context);
                    break;
                }
                case SANITIZER_CBID_RESOURCE_DEVICE_MEMORY_ALLOC:
                {
                    auto *pModuleData = (Sanitizer_ResourceMemoryData *)cbdata;
                    if (pModuleData->flags == SANITIZER_MEMORY_FLAG_CG_RUNTIME || pModuleData->size == 0) break;
                    yosemite_alloc_callback(pModuleData->address, pModuleData->size, pModuleData->flags);
                    break;
                }
                case SANITIZER_CBID_RESOURCE_DEVICE_MEMORY_FREE:
                {
                    auto *pModuleData = (Sanitizer_ResourceMemoryData *)cbdata;
                    if (pModuleData->flags == SANITIZER_MEMORY_FLAG_CG_RUNTIME || pModuleData->size == 0) break;
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
                    LaunchBegin(pLaunchData->context, pLaunchData->function, pLaunchData->functionName, pLaunchData->hStream, blockDims, gridDims);
                    break;
                }
                case SANITIZER_CBID_LAUNCH_END:
                {
                    auto* pLaunchData = (Sanitizer_LaunchData*)cbdata;
                    LaunchEnd(pLaunchData->context, pLaunchData->stream, pLaunchData->function, pLaunchData->functionName, pLaunchData->hStream);
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
                    yosemite_memcpy_callback(pMemcpyData->dstAddress, pMemcpyData->srcAddress, pMemcpyData->size, pMemcpyData->isAsync, (uint32_t)pMemcpyData->direction);
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
                    yosemite_memset_callback(pMemsetData->address, pMemsetData->elementSize, pMemsetData->value, pMemsetData->isAsync);
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


void cleanup(void) {
    yosemite_terminate();
}


int InitializeInjection()
{
    Sanitizer_SubscriberHandle handle;
    sanitizerSubscribe(&handle, ComputeSanitizerCallback, nullptr);
    sanitizerEnableDomain(1, handle, SANITIZER_CB_DOMAIN_RESOURCE);
    sanitizerEnableDomain(1, handle, SANITIZER_CB_DOMAIN_LAUNCH);
    sanitizerEnableDomain(1, handle, SANITIZER_CB_DOMAIN_MEMCPY);
    sanitizerEnableDomain(1, handle, SANITIZER_CB_DOMAIN_MEMSET);

    yosemite_init(sanitizer_options);
    // enable torch profiler?
    const char* torch_prof = std::getenv("TORCH_PROFILE_ENABLED");
    if (torch_prof && std::string(torch_prof) == "1") {
        tensor_scope_enable();
        register_tensor_scope(tensor_malloc_callback, tensor_free_callback);
        yosemite_torch_prof_enable();
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
