#ifndef YOSEMITE_TOOL_PC_DEPENDENCY_ANALYSIS_H
#define YOSEMITE_TOOL_PC_DEPENDENCY_ANALYSIS_H


#include "tools/tool.h"
#include "utils/event.h"
#include "gpu_patch.h"
#include "parallel_hashmap/phmap.h"
#include "tools/shadow_memory.h"

#include <map>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <vector>
#include <array>
#include <cstdint>
#include <string>
#include <memory>
#include <cassert>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <cstring>
#include <sys/mman.h>


namespace yosemite {

/* we choose to use PC offset instead of PC because the PC is too long for shadow memory and it is not necessary to track the original PC.
The offset will be calculated during trace collection.

Every memory allocation will cause a shadow memory to be created.
Every memory deallocation will cause a shadow memory to be destroyed.
Shadow memory bitmask will be reset when a kernel finished. (to avoid mass shadow memory reset)

The gpu data analysis will 
1.iterate the trace buffer and query the shadow memory to get the corresponding shadow memory entry.
2. compare the last access information with the current access information with the rules below:
    0. if bitmask of this access is 0, it means the current access is a cold miss set it's acient pc to 0xFFFFFFFF.
    1. if last access and current access are from the same thread, then it is an intra thread access.
    2. if last access and current access are from the same warp, then it is an intra warp access.
    3. if last access and current access are from the same block, then it is an intra block access.
    4. if last access and current access are from the same grid, then it is an intra grid access.
3. update the pc_statistics with the current pc, ancient pc and the distance.
4. update the shadow memory entry with the current pc and the flat thread id.
*/


class PC_statisitics{
public:
    std::array<uint64_t, 5> dist = {0, 0, 0, 0, 0}; 
    // 0: intra thread
    // 1: intra instance launch
    // 2: intra warp
    // 3: intra block
    // 4: intra grid
};


class PcDependency final : public Tool {
public:
    PcDependency();

    ~PcDependency();

    void gpu_data_analysis(void* data, uint64_t size);

    void query_ranges(void* ranges, uint32_t limit, uint32_t* count) override {};

    void query_tensors(void* ranges, uint32_t limit, uint32_t* count) override {};

    void allocation_callback(uint64_t ptr, uint64_t size);

    void deallocation_callback(uint64_t ptr);

    void evt_callback(EventPtr_t evt);

    void flush();

private:
    void kernel_start_callback(std::shared_ptr<KernelLaunch_t> kernel);

    void kernel_end_callback(std::shared_ptr<KernelEnd_t> kernel);

    void mem_alloc_callback(std::shared_ptr<MemAlloc_t> mem);

    void mem_free_callback(std::shared_ptr<MemFree_t> mem);

    void ten_alloc_callback(std::shared_ptr<TenAlloc_t> ten);

    void ten_free_callback(std::shared_ptr<TenFree_t> ten);

    void kernel_trace_flush(std::shared_ptr<KernelLaunch_t> kernel);

    void unit_access(
        uint64_t ptr,
        uint32_t pc_offset,
        uint64_t current_block_id,
        uint32_t current_warp_id,
        uint32_t current_lane_id,
        memory_region& memory_region_target,
        int access_size,
        phmap::flat_hash_map<uint64_t, PC_statisitics>& local_pc_statistics
    );

    void unit_access_shared(
        uint64_t ptr,
        uint32_t pc_offset,
        uint32_t object_idx,
        uint64_t current_block_id,
        uint32_t current_warp_id,
        uint32_t current_lane_id,
        int access_size,
        phmap::flat_hash_map<uint64_t, PC_statisitics>& local_pc_statistics,
        worker_shared_shadow_state& local_shadow_memory_shared
    );

    void unit_access_local(uint64_t ptr, uint32_t pc_offset, uint64_t current_block_id, uint32_t current_warp_id, uint32_t current_lane_id, int access_size);

    // Fallback for global-memory accesses whose base allocation was not captured
    // (e.g. __device__ static globals, VMM-mapped memory).  Uses a concurrent
    // hashmap keyed by absolute device address instead of a pre-allocated shadow
    // array, so no prior registration is required.
    void unit_access_unknown(
        uint64_t abs_addr,
        uint32_t pc_offset,
        uint64_t current_block_id,
        uint32_t current_warp_id,
        uint32_t current_lane_id,
        int access_size,
        phmap::flat_hash_map<uint64_t, PC_statisitics>& local_pc_statistics
    );
    void worker_loop(uint64_t worker_idx);
    uint32_t acquire_shared_shadow_object(worker_shared_shadow_state& local_shadow_memory_shared, uint64_t cta_id);
    void release_shared_shadow_object(
        worker_shared_shadow_state& local_shadow_memory_shared,
        uint64_t cta_id,
        uint32_t exiting_threads
    );
    shared_shadow_memory_entry& get_shared_shadow_entry(
        worker_shared_shadow_state& local_shadow_memory_shared,
        uint32_t object_idx,
        uint32_t addr
    );
    uint64_t pack_pc_ancient_pairs(uint32_t current_pc_offset, uint32_t ancient_pc_offset){
        return static_cast<uint64_t>(current_pc_offset) << 32 | static_cast<uint64_t>(ancient_pc_offset);
    };
    uint32_t unpack_current_pc_offset(uint64_t pc_ancient_pairs){
        return static_cast<uint32_t>(pc_ancient_pairs >> 32);
    };
    uint32_t unpack_ancient_pc_offset(uint64_t pc_ancient_pairs){
        return static_cast<uint32_t>(pc_ancient_pairs & 0xFFFFFFFFu);
    };

/*
********************************* variables *********************************
*/
    Timer_t _timer;

    std::string output_directory;
    uint32_t kernel_id = 0;
    uint8_t _kernel_generation = 0;
    uint32_t _shared_kernel_generation = 0;
    uint64_t _current_kernel_cta_count = 0;


    std::map<uint64_t, std::shared_ptr<KernelLaunch_t>> kernel_events;
    std::map<uint64_t, std::shared_ptr<MemAlloc_t>> alloc_events;
    std::map<DevPtr, std::shared_ptr<MemAlloc_t>> active_memories;

    std::map<uint64_t, std::shared_ptr<TenAlloc>> tensor_events;
    std::map<DevPtr, std::shared_ptr<TenAlloc>> active_tensors;


    std::vector<memory_region> _memory_regions;

    std::map<memory_region, std::unique_ptr<shadow_memory>> _shadow_memories; // memory region, shadow memory

    // Per-kernel fallback shadow for addresses outside all tracked allocations.
    // Keyed by absolute device address (sampled at 4-byte stride), value is
    // the same packed shadow entry format as shadow_memory_entry::packed.
    // Use parallel_flat_hash_map_m (std::mutex per shard) so concurrent
    // worker threads can safely call try_emplace_l on the same shard.
    phmap::parallel_flat_hash_map_m<uint64_t, uint64_t> _unknown_region_shadow;

    // std::unordered_map<uint32_t, std::unordered_map<uint32_t, PC_statisitics>> _pc_statistics; // current pc offset, ancient pc offset, PC_statisitics
    // std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>> _pc_flags; // pc offset, flags, size of the access
    phmap::flat_hash_map<uint64_t, PC_statisitics> _pc_statistics; // (current pc offset<<32 || ancient pc offset), PC_statisitics
    phmap::flat_hash_map<uint32_t, std::pair<uint32_t, uint32_t>> _pc_flags; // pc offset, flags, size of the access
    // Index [0..31] stores distinct sector count 1..32.
    // Index [32..64] stores active lane count 0..32.
    // Index [0..31]:  distinct sector count 1..32.
    // Index [32..64]: active lane count 0..32.
    // Index [65..96]: distinct address count 1..32.
    std::unordered_map<uint32_t, std::array<uint64_t, 97>> _distinct_sector_count;

    // Persistent worker pool and per-worker shared-memory shadow state.
    uint64_t _worker_count = 1;
    std::vector<std::thread> _workers;
    std::vector<worker_shared_shadow_state> _worker_shadow_memory_shared;
    uint32_t _shared_shadow_object_cap_per_worker = 128;
    uint32_t _shared_shadow_bytes_per_object = 102400;
    uint32_t _current_block_thread_count = 0;

    // Per-batch job data produced by gpu_data_analysis and consumed by workers.
    const MemoryAccess* _job_accesses_buffer = nullptr;
    std::vector<std::vector<uint64_t>> _job_worker_trace_indices;
    std::vector<phmap::flat_hash_map<uint64_t, PC_statisitics>> _job_worker_pc_statistics;
    std::vector<std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>>> _job_worker_pc_flags;
    std::vector<std::unordered_map<uint32_t, std::array<uint64_t, 97>>> _job_worker_distinct_sector_count;

    std::mutex _worker_pool_mutex;
    std::condition_variable _worker_pool_cv;
    std::condition_variable _worker_pool_done_cv;
    bool _worker_pool_shutdown = false;
    uint64_t _worker_job_generation = 0;
    uint64_t _worker_pending_jobs = 0;

};

}   // yosemite
#endif // YOSEMITE_TOOL_PC_DEPENDENCY_ANALYSIS_H
