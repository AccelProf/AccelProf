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
    MemoryAccessType type)
{
    auto* pTracker = (MemoryAccessTracker*)userdata;

    uint32_t active_mask = __activemask();
    uint32_t laneid = get_laneid();
    uint32_t first_laneid = __ffs(active_mask) - 1;

    uint32_t keep = 1;
    if (pTracker->state != nullptr) {
        MemoryAccessState* state = (MemoryAccessState*) pTracker->state;
        MemoryRange* start_end = state->start_end;
        MemoryRange range = {(uint64_t) ptr, 0};
        uint32_t pos = map_prev(start_end, range, state->size, gpu_address_comparator());

        if (pos != state->size) {
            // Find an existing range
            if (atomic_load(state->touch + pos) == 0) {
                // Update
                atomic_store(state->touch + pos, (uint8_t)1);
            } else {
                // Filter out
                keep = 0;
            }
        }
    }
    __syncwarp(active_mask);

    uint32_t all_keep = 0;
    all_keep = ballot((uint32_t)keep, active_mask);
    if (all_keep == 0) {
        // Fast path
        return SANITIZER_PATCH_SUCCESS;
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
