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
    uint32_t laneid = get_laneid();
    uint32_t first_laneid = __ffs(active_mask) - 1;

    if (pTracker->access_state != nullptr) {
        MemoryAccessState* states = (MemoryAccessState*) pTracker->access_state;
        MemoryRange* start_end = states->start_end;
        MemoryRange range = {(uint64_t) ptr, 0};
        uint32_t pos = map_prev(start_end, range, states->size, gpu_address_comparator());

        if (pos != states->size) {
            atomicAdd(&(states->touch[pos]), 1);
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

