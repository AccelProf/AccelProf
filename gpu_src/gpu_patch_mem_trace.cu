#include "gpu_patch.h"

#include <sanitizer_patching.h>

#include "gpu_utils.h"
#include <cstdio>

static __device__ __inline__
uint32_t GetBufferIndex(MemoryAccessTracker* pTracker) {
    uint32_t idx = MEMORY_ACCESS_BUFFER_SIZE;

    while (idx >= MEMORY_ACCESS_BUFFER_SIZE) {
        idx = atomicAdd(&(pTracker->currentEntry), 1);

        if (idx >= MEMORY_ACCESS_BUFFER_SIZE) {
            // buffer is full, wait for last writing thread to flush
            while (*(volatile uint32_t*)&(pTracker->currentEntry) >= MEMORY_ACCESS_BUFFER_SIZE);
        }
    }

    return idx;
}

static __device__ __inline__
void IncrementNumEntries(MemoryAccessTracker* pTracker) {
    DoorBell_t* doorbell = pTracker->doorbell;
    __threadfence();
    const uint32_t numEntries = atomicAdd((int*)&(pTracker->numEntries), 1);

    if (numEntries == MEMORY_ACCESS_BUFFER_SIZE - 1) {
        // make sure everything is visible in memory
        __threadfence_system();
        doorbell->full = true;
        while (doorbell->full);

        pTracker->numEntries = 0;
        __threadfence();
        pTracker->currentEntry = 0;
    }
}

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
    uint32_t laneid = get_laneid();
    uint32_t first_laneid = __ffs(active_mask) - 1;

    MemoryAccess* accesses = nullptr;

    if (laneid == first_laneid) {
        uint32_t idx = GetBufferIndex(pTracker);
        accesses = &pTracker->accesses[idx];
        accesses->warpId = get_warpid();
        accesses->type = type;
        accesses->accessSize = accessSize;
    }

    __syncwarp(active_mask);

    accesses = (MemoryAccess*) shfl((uint64_t)accesses, first_laneid, active_mask);
    if (accesses) {
        accesses->addresses[laneid] = (uint64_t)(uintptr_t)ptr;
    }

    __syncwarp(active_mask);

    if (laneid == first_laneid) {
        IncrementNumEntries(pTracker);
    }

    return SANITIZER_PATCH_SUCCESS;
}

extern "C" __device__ __noinline__
SanitizerPatchResult MemoryGlobalAccessCallback(
    void* userdata,
    uint64_t pc,
    void* ptr,
    uint32_t accessSize,
    uint32_t flags)
{
    return CommonCallback(userdata, pc, ptr, accessSize, flags, MemoryType::Global);
}

extern "C" __device__ __noinline__
SanitizerPatchResult MemorySharedAccessCallback(
    void* userdata,
    uint64_t pc,
    void* ptr,
    uint32_t accessSize,
    uint32_t flags)
{
    return CommonCallback(userdata, pc, ptr, accessSize, flags, MemoryType::Shared);
}

extern "C" __device__ __noinline__
SanitizerPatchResult MemoryLocalAccessCallback(
    void* userdata,
    uint64_t pc,
    void* ptr,
    uint32_t accessSize,
    uint32_t flags)
{
    return CommonCallback(userdata, pc, ptr, accessSize, flags, MemoryType::Local);
}

extern "C" __device__ __noinline__
SanitizerPatchResult MemcpyAsyncCallback(void* userdata, uint64_t pc, void* src, uint32_t dst, uint32_t accessSize, uint32_t totalShmemSize)
{
    if (src)
    {
        CommonCallback(userdata, pc, src, accessSize, SANITIZER_MEMORY_DEVICE_FLAG_READ, MemoryType::Global);
    }

    return CommonCallback(userdata, pc, (void*)dst, accessSize, SANITIZER_MEMORY_DEVICE_FLAG_WRITE, MemoryType::Shared);
}

extern "C" __device__ __noinline__
SanitizerPatchResult BlockExitCallback(void* userdata, uint64_t pc)
{
    MemoryAccessTracker* tracker = (MemoryAccessTracker*)userdata;
    DoorBell_t* doorbell = tracker->doorbell;

    uint32_t active_mask = __activemask();
    uint32_t laneid = get_laneid();
    uint32_t first_laneid = __ffs(active_mask) - 1;
    int32_t pop_count = __popc(active_mask);

    if (laneid == first_laneid) {
        atomicAdd((int*)&doorbell->num_threads, -pop_count);
    }
    __syncwarp(active_mask);

    return SANITIZER_PATCH_SUCCESS;
}
