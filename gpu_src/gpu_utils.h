#pragma once

/*********************************************************************
 *
 *                   Device level utility functions
 *
 **********************************************************************/

// Get the SM id
__device__ __forceinline__ unsigned int get_smid(void) {
    unsigned int ret;
    asm("mov.u32 %0, %smid;" : "=r"(ret));
    return ret;
}

// Get the warp id within the application
__device__ __forceinline__ unsigned int get_warpid(void) {
    unsigned int ret;
    asm("mov.u32 %0, %warpid;" : "=r"(ret));
    return ret;
}

// Get the line id within the warp
__device__ __forceinline__ unsigned int get_laneid(void) {
    unsigned int laneid;
    asm volatile("mov.u32 %0, %laneid;" : "=r"(laneid));
    return laneid;
}

// Get a thread's CTA ID
__device__ __forceinline__ int4 get_ctaid(void) {
    int4 ret;
    asm("mov.u32 %0, %ctaid.x;" : "=r"(ret.x));
    asm("mov.u32 %0, %ctaid.y;" : "=r"(ret.y));
    asm("mov.u32 %0, %ctaid.z;" : "=r"(ret.z));
    return ret;
}

//  Get the number of CTA ids per grid
__device__ __forceinline__ int4 get_nctaid(void) {
    int4 ret;
    asm("mov.u32 %0, %nctaid.x;" : "=r"(ret.x));
    asm("mov.u32 %0, %nctaid.y;" : "=r"(ret.y));
    asm("mov.u32 %0, %nctaid.z;" : "=r"(ret.z));
    return ret;
}

__device__ __forceinline__ uint32_t get_flat_block_id() {
    return blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
}

__device__ __forceinline__ uint32_t get_flat_thread_id() {
    return threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
}

__device__ __forceinline__ uint64_t get_unique_thread_id() {
    return get_flat_block_id() * blockDim.x * blockDim.y * blockDim.z + get_flat_thread_id();
}

__device__ __forceinline__ uint64_t get_grid_num_threads() {
    return gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
}

__device__ __forceinline__ uint64_t get_block_num_threads() {
    return blockDim.x * blockDim.y * blockDim.z;
}


template <class T>
__device__ __forceinline__ T shfl(T v, uint32_t srcline, uint32_t mask = 0xFFFFFFFF) {
    T ret;
#if (__CUDA_ARCH__ >= 300)
#if (__CUDACC_VER_MAJOR__ >= 9)
    ret = __shfl_sync(mask, v, srcline);
#else
    ret = __shfl(v, srcline);
#endif
#endif
    return ret;
}


template <class T>
__device__ __forceinline__ T shfl_xor(T v, uint32_t lane_mask, uint32_t mask = 0xFFFFFFFF) {
    T ret;
#if (__CUDA_ARCH__ >= 300)
#if (__CUDACC_VER_MAJOR__ >= 9)
    ret = __shfl_xor_sync(mask, v, lane_mask);
#else
    ret = __shfl_xor(v, lane_mask);
#endif
#endif
    return ret;
}


__device__ __forceinline__ uint32_t ballot(int32_t predicate, uint32_t mask = 0xFFFFFFFF) {
    uint32_t ret;
#if (__CUDA_ARCH__ >= 300)
#if (__CUDACC_VER_MAJOR__ >= 9)
    ret = __ballot_sync(mask, predicate);
#else
    ret = __ballot(predicate);
#endif
#endif
    return ret;
}


template <typename T>
__device__ __forceinline__ T atomic_load(const T *addr) {
    const volatile T *vaddr = addr;  // volatile to bypass cache
    __threadfence();                 // for seq_cst loads. Remove for acquire semantics.
    const T value = *vaddr;
    // fence to ensure that dependent reads are correctly ordered
    __threadfence();
    return value;
}


template <typename T>
__device__ __forceinline__ void atomic_store(T *addr, T value) {
    volatile T *vaddr = addr;  // volatile to bypass cache
    // fence to ensure that previous non-atomic stores are visible to other threads
    __threadfence();
    *vaddr = value;
}


template <typename T, typename C>
__device__ __forceinline__ uint32_t map_upper_bound(T *map, T value, uint32_t len, C cmp) {
    uint32_t low = 0;
    uint32_t high = len;
    uint32_t mid = 0;
    while (low < high) {
        mid = (high - low) / 2 + low;
        if (cmp(map[mid], value)) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low;
}


template <typename T, typename C>
__device__ __forceinline__ uint32_t map_prev(T *map, T value, uint32_t len, C cmp) {
    uint32_t pos = map_upper_bound<T, C>(map, value, len, cmp);
    if (pos != 0) {
        --pos;
    } else {
        pos = len;
    }
    return pos;
}
