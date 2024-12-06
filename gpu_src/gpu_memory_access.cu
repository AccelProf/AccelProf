#include "gpu_patch.h"

#include <sanitizer_patching.h>

#include "gpu_utils.h"


static __device__
SanitizerPatchResult CommonCallback(
    void* userdata,
    uint64_t pc,
    void* ptr,
    uint32_t accessSize,
    uint32_t flags,
    MemoryAccessType type)
{
    auto* pTracker = (MemoryAccessTracker*)userdata;

    uint32_t active_mask = __activemask();
    uint32_t laneid = get_laneid();
    uint32_t first_laneid = __ffs(active_mask) - 1;

    MemoryAccess* accesses = nullptr;
    bool full = false;
    if (laneid == first_laneid) {
        uint32_t old = atomicAdd(&(pTracker->currentEntry), 1);

        // no more space!
        if (old >= pTracker->maxEntry)
            full = true;

        accesses = &pTracker->accesses[old];
        accesses->warpId = get_warpid();
        accesses->type = type;
        accesses->accessSize = accessSize;
        accesses->flags = flags;
    }
    
    __syncwarp(active_mask);
    
    accesses = (MemoryAccess*) shfl((uint64_t)accesses, first_laneid, active_mask);
    full = (bool) shfl(full, first_laneid, active_mask);
    if (full) {
        return SANITIZER_PATCH_SUCCESS;
    }

    if (accesses) {
        accesses->addresses[laneid] = (uint64_t)(uintptr_t)ptr;
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
    return CommonCallback(userdata, pc, ptr, accessSize, flags, MemoryAccessType::Global);
}

extern "C" __device__ __noinline__
SanitizerPatchResult MemorySharedAccessCallback(
    void* userdata,
    uint64_t pc,
    void* ptr,
    uint32_t accessSize,
    uint32_t flags)
{
    return CommonCallback(userdata, pc, ptr, accessSize, flags, MemoryAccessType::Shared);
}

extern "C" __device__ __noinline__
SanitizerPatchResult MemoryLocalAccessCallback(
    void* userdata,
    uint64_t pc,
    void* ptr,
    uint32_t accessSize,
    uint32_t flags)
{
    return CommonCallback(userdata, pc, ptr, accessSize, flags, MemoryAccessType::Local);
}

extern "C" __device__ __noinline__
SanitizerPatchResult MemcpyAsyncCallback(void* userdata, uint64_t pc, void* src, uint32_t dst, uint32_t accessSize, uint32_t totalShmemSize)
{
    if (src)
    {
        CommonCallback(userdata, pc, src, accessSize, SANITIZER_MEMORY_DEVICE_FLAG_READ, MemoryAccessType::Global);
    }

    return CommonCallback(userdata, pc, (void*)dst, accessSize, SANITIZER_MEMORY_DEVICE_FLAG_WRITE, MemoryAccessType::Shared);
}

extern "C" __device__ __noinline__
SanitizerPatchResult BlockExitCallback(void* userdata, uint64_t pc)
{
    MemoryAccessTracker* tracker = (MemoryAccessTracker*)userdata;

    uint32_t active_mask = __activemask();
    uint32_t laneid = get_laneid();
    uint32_t first_laneid = __ffs(active_mask) - 1;
    int32_t pop_count = __popc(active_mask);

    if (laneid == first_laneid) {
        atomicAdd(&tracker->numThreads, -pop_count);
    }
    __syncwarp(active_mask);

    return SANITIZER_PATCH_SUCCESS;
}
