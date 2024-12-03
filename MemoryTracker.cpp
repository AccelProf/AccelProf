/* Copyright (c) 2019-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "MemoryTracker.h"

#include <sanitizer.h>

#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <vector>
#include <cstring>
#include <cassert>


static MemoryAccessTracker* hMemAccessTracker = nullptr;
static MemoryAccessTracker* dMemAccessTracker = nullptr;
static MemoryAccess* hMemAccessBuffer = nullptr;
static MemoryAccess* dMemAccessBuffer = nullptr;
static MemoryAccessState* hMemAccessState = nullptr;
static MemoryAccessState* dMemAccessState = nullptr;

static bool access_tracking_enabled = false;

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
    // Instrument user code!
    SanitizerResult result;
    if (access_tracking_enabled) {
        result = sanitizerAddPatchesFromFile("MemoryTrackerAccess.fatbin", 0);
    } else {
        result = sanitizerAddPatchesFromFile("MemoryTrackerState.fatbin", 0);
    }

    if (result != SANITIZER_SUCCESS)
        std::cerr << "Failed to load fatbin. Please check that it is in the current directory and contains the correct SM architecture" << std::endl;

    sanitizerPatchInstructions(SANITIZER_INSTRUCTION_GLOBAL_MEMORY_ACCESS, module, "MemoryGlobalAccessCallback");
    sanitizerPatchInstructions(SANITIZER_INSTRUCTION_SHARED_MEMORY_ACCESS, module, "MemorySharedAccessCallback");
    sanitizerPatchInstructions(SANITIZER_INSTRUCTION_LOCAL_MEMORY_ACCESS, module, "MemoryLocalAccessCallback");
    sanitizerPatchInstructions(SANITIZER_INSTRUCTION_MEMCPY_ASYNC, module, "MemcpyAsyncCallback");
    sanitizerPatchInstructions(SANITIZER_INSTRUCTION_BLOCK_EXIT, module, "ThreadBlockExit");
    sanitizerPatchModule(module);

    if (!dMemAccessTracker) {
        sanitizerAlloc(context, (void**)&dMemAccessTracker, sizeof(*dMemAccessTracker));
    }
    if (!dMemAccessBuffer && access_tracking_enabled) {
        sanitizerAlloc(context, (void**)&dMemAccessBuffer, sizeof(MemoryAccess) * MemoryBufferSize);
    }
    if (!dMemAccessState && !access_tracking_enabled) {
        sanitizerAlloc(context, (void**)&dMemAccessState, sizeof(MemoryAccessState) * MAX_ACTIVE_ALLOCATIONS);
    }

    if (!hMemAccessTracker) {
        sanitizerAllocHost(context, (void**)&hMemAccessTracker, sizeof(*hMemAccessTracker));
    }
    if (!hMemAccessBuffer && access_tracking_enabled) {
        sanitizerAllocHost(context, (void**)&hMemAccessBuffer, sizeof(MemoryAccess) * MemoryBufferSize);
    }
    if (!hMemAccessState && !access_tracking_enabled) {
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
    std::cout << std::endl << "Launch " << functionName << std::endl;

    uint32_t num_threads = blockDims.x * blockDims.y * blockDims.z * gridDims.x * gridDims.y * gridDims.z;

    if (access_tracking_enabled) {
        sanitizerMemset(dMemAccessBuffer, 0, sizeof(MemoryAccess) * MemoryBufferSize, hstream);
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
    hMemAccessTracker->maxEntry = MemoryBufferSize;
    hMemAccessTracker->numThreads = num_threads;
    hMemAccessTracker->accesses = dMemAccessBuffer;

    sanitizerMemcpyHostToDeviceAsync(dMemAccessTracker, hMemAccessTracker, sizeof(*dMemAccessTracker), hstream);

    sanitizerSetCallbackData(function, dMemAccessTracker);
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

    if (access_tracking_enabled) {
        uint32_t numEntries = std::min(hMemAccessTracker->currentEntry, hMemAccessTracker->maxEntry);
        sanitizerMemcpyDeviceToHost(hMemAccessBuffer, hMemAccessTracker->accesses, sizeof(MemoryAccess) * numEntries, nullptr);
        for (uint32_t i = 0; i < numEntries; ++i)
        {
            MemoryAccess& access = hMemAccessBuffer[i];

            std::cout << "  [" << i << "] " << GetMemoryRWString(access.flags)
                    << " access of " << GetMemoryTypeString(access.type)
                    << " memory by thread (" << access.threadId.x
                    << "," << access.threadId.y
                    << "," << access.threadId.z
                    << ") at address 0x" << std::hex << access.address << std::dec
                    << " (size is " << access.accessSize << " bytes)" << std::endl;
        }
    } else {
        memset(hMemAccessState, 0, sizeof(MemoryAccessState) * MAX_ACTIVE_ALLOCATIONS);
        sanitizerMemcpyDeviceToHost(hMemAccessState, hMemAccessTracker->state, sizeof(*hMemAccessState), hstream);
        for (uint32_t i = 0; i < hMemAccessState->size; ++i)
        {
            const auto& range = hMemAccessState->start_end[i];
            const auto& touch = hMemAccessState->touch[i];
            std::cout << "  Memory range 0x" << std::hex << range.start << " - 0x" << range.end << std::dec << "  " << (int)touch << std::endl;
        }
    }
    
}


void MemoryTrackerCallback(
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

                    MemoryRange range;
                    range.start = (uint64_t)pModuleData->address;
                    range.end = range.start + pModuleData->size;
                    activeAllocations.emplace(range.start, range);
                    break;
                }
                case SANITIZER_CBID_RESOURCE_DEVICE_MEMORY_FREE:
                {
                    auto *pModuleData = (Sanitizer_ResourceMemoryData *)cbdata;
                    if (pModuleData->flags == SANITIZER_MEMORY_FLAG_CG_RUNTIME || pModuleData->size == 0) break;

                    auto it = activeAllocations.find((uint64_t)pModuleData->address);
                    assert(it != activeAllocations.end());
                    if (it != activeAllocations.end())
                    {
                        activeAllocations.erase(it);
                    }
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


void cleanup(void) {}


int InitializeInjection()
{
    Sanitizer_SubscriberHandle handle;
    sanitizerSubscribe(&handle, MemoryTrackerCallback, nullptr);
    sanitizerEnableDomain(1, handle, SANITIZER_CB_DOMAIN_RESOURCE);
    sanitizerEnableDomain(1, handle, SANITIZER_CB_DOMAIN_LAUNCH);

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
