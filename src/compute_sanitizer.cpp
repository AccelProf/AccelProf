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


static MemoryAccessTracker* hMemAccessTracker = nullptr;
static MemoryAccessTracker* dMemAccessTracker = nullptr;
static MemoryAccess* hMemAccessBuffer = nullptr;
static MemoryAccess* dMemAccessBuffer = nullptr;
static MemoryAccessState* hMemAccessState = nullptr;
static MemoryAccessState* dMemAccessState = nullptr;

static SanitizerOptions_t sanitizer_options;

static std::map<uint64_t, MemoryRange> activeAllocations;


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


void ModuleLoaded(CUmodule module, CUcontext context)
{
    const char* env_name = std::getenv("CU_PROF_HOME");
    std::string patch_path;
    if (env_name) {
        patch_path = std::string(env_name) + "/lib/gpu_patch/";
    }

    std::string fatbin_file;
    if (sanitizer_options.enable_access_tracking) {
        fatbin_file = patch_path + "gpu_memory_access_count.fatbin";
    } else {
        fatbin_file = patch_path + "gpu_memory_access_count.fatbin";
    }

    // Instrument user code!
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

    if (!dMemAccessTracker) {
        sanitizerAlloc(context, (void**)&dMemAccessTracker, sizeof(*dMemAccessTracker));
    }
    if (!dMemAccessBuffer && sanitizer_options.enable_access_tracking) {
        sanitizerAlloc(context, (void**)&dMemAccessBuffer, sizeof(MemoryAccess) * MEMORY_ACCESS_BUFFER_SIZE);
    }
    if (!dMemAccessState && !sanitizer_options.enable_access_tracking) {
        sanitizerAlloc(context, (void**)&dMemAccessState, sizeof(MemoryAccessState) * MAX_ACTIVE_ALLOCATIONS);
    }

    if (!hMemAccessTracker) {
        sanitizerAllocHost(context, (void**)&hMemAccessTracker, sizeof(*hMemAccessTracker));
    }
    if (!hMemAccessBuffer && sanitizer_options.enable_access_tracking) {
        sanitizerAllocHost(context, (void**)&hMemAccessBuffer, sizeof(MemoryAccess) * MEMORY_ACCESS_BUFFER_SIZE);
    }
    if (!hMemAccessState && !sanitizer_options.enable_access_tracking) {
        sanitizerAllocHost(context, (void**)&hMemAccessState, sizeof(MemoryAccessState) * MAX_ACTIVE_ALLOCATIONS);
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

    uint32_t num_threads = blockDims.x * blockDims.y * blockDims.z * gridDims.x * gridDims.y * gridDims.z;

    if (sanitizer_options.enable_access_tracking) {
        sanitizerMemset(dMemAccessBuffer, 0, sizeof(MemoryAccess) * MEMORY_ACCESS_BUFFER_SIZE, hstream);
    } else {
        memset(hMemAccessState, 0, sizeof(MemoryAccessState) * MAX_ACTIVE_ALLOCATIONS);
        for (const auto& range : activeAllocations)
        {
            hMemAccessState->start_end[hMemAccessState->size].start = range.second.start;
            hMemAccessState->start_end[hMemAccessState->size].end = range.second.end;
            hMemAccessState->size++;
        }
        sanitizerMemcpyHostToDeviceAsync(dMemAccessState, hMemAccessState, sizeof(*dMemAccessState), hstream);
        hMemAccessTracker->state = dMemAccessState;
    }

    hMemAccessTracker->currentEntry = 0;
    hMemAccessTracker->maxEntry = MEMORY_ACCESS_BUFFER_SIZE;
    hMemAccessTracker->numThreads = num_threads;
    hMemAccessTracker->accesses = dMemAccessBuffer;
    hMemAccessTracker->accessCount = 0;

    sanitizerMemcpyHostToDeviceAsync(dMemAccessTracker, hMemAccessTracker, sizeof(*dMemAccessTracker), hstream);

    sanitizerSetCallbackData(function, dMemAccessTracker);

    yosemite_kernel_start_callback(functionName);
}


void LaunchEnd(
    CUcontext context,
    CUstream stream,
    CUfunction function,
    std::string functionName,
    Sanitizer_StreamHandle hstream)
{
    // while (true)
    // {
    //     sanitizerMemcpyDeviceToHost(hMemAccessTracker, dMemAccessTracker, sizeof(*dMemAccessTracker), hstream);
    //     if (hMemAccessTracker->numThreads == 0) {
    //         break;
    //     }
    // }

    sanitizerStreamSynchronize(hstream);
    sanitizerMemcpyDeviceToHost(hMemAccessTracker, dMemAccessTracker, sizeof(*dMemAccessTracker), hstream);


    std::cout << "Kernel " << functionName << " has " << hMemAccessTracker->accessCount << " memory accesses." << std::endl;

    yosemite_gpu_data_analysis(hMemAccessTracker, hMemAccessTracker->accessCount);
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

    yosemite_init(sanitizer_options);

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
