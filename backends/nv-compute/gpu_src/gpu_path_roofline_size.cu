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
    MemoryType type)
{
    auto* pTracker = (MemoryAccessTracker*)userdata;

    uint32_t active_mask = __activemask();
    uint32_t laneid = get_laneid();
    uint32_t first_laneid = __ffs(active_mask) - 1;

    // check the number of accesses
    int active_threads = __popc(active_mask);
    if (laneid == first_laneid) {
        atomicAdd((unsigned long long int*)&pTracker->accessCount, (unsigned long long int) active_threads);
        atomicAdd((unsigned long long int*)&pTracker->accessSize, (unsigned long long int) active_threads * accessSize);
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
