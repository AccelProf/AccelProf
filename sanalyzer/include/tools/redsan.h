#ifndef YOSEMITE_TOOL_REDSAN_H
#define YOSEMITE_TOOL_REDSAN_H

// redsan: redundant memory instruction sanitizer.
//
// Reimplements the detector from "RedSan: A Redundant Memory Instruction
// Sanitizer for GPU Programs" (SC'25) on top of AccelProf. A memory access is
// redundant when the previous access to the same address came from the same
// thread and (paper, Sec. 4.1):
//   DeadStore      : store  followed by store
//   DeadLoad       : load   followed by load
//   LoadAfterStore : store  followed by load
// Accesses from other threads are never redundant themselves, but they replace
// the last-access record and therefore break a same-thread chain.
//
// Trace collection and the last-access record reuse cuVein's design: the GPU
// patch of pc_dependency_analysis, an allocation-mirrored global shadow with a
// packed 64-bit entry per sampled byte, a per-CTA shared-memory shadow pool and
// a worker pool partitioned by CTA (see tools/shadow_memory.h).

#include "tools/tool.h"
#include "utils/event.h"
#include "gpu_patch.h"
#include "parallel_hashmap/phmap.h"
#include "tools/shadow_memory.h"

#include <array>
#include <condition_variable>
#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace yosemite {

// Access class kept in the shadow entry. The original RedSan compared the raw
// Sanitizer device flags for equality; these classes preserve that comparison
// except that atomics of different memory scopes (CTA/GPU/SYS) compare equal.
enum class RedsanAccess : uint8_t {
    None            = 0,
    Load            = 1,
    Store           = 2,
    AtomicRmw       = 3,   // Sanitizer reports atomics as READ|WRITE
    Prefetch        = 4,
    AtomicWriteOnly = 5,
    Other           = 7,
};

struct RedsanPcStats {
    uint64_t total_accesses = 0;
    uint64_t redundant_accesses = 0;   // = dead_store + dead_load + load_after_store + other_equal
    uint64_t dead_store = 0;
    uint64_t dead_load = 0;
    uint64_t load_after_store = 0;
    uint64_t other_equal = 0;          // same class repeated, neither plain load nor store
    uint32_t flags = 0;                // first-seen Sanitizer device flags
    uint32_t scope = 0;                // SANITIZER_MEMORY_GLOBAL / SHARED / LOCAL
    uint32_t access_size = 0;
};

class Redsan final : public Tool {
public:
    Redsan();
    ~Redsan();

    void gpu_data_analysis(void* data, uint64_t size) override;
    void query_ranges(void* ranges, uint32_t limit, uint32_t* count) override {};
    void query_tensors(void* ranges, uint32_t limit, uint32_t* count) override {};
    void evt_callback(EventPtr_t evt) override;
    void flush() override;

private:
    using PcStatsMap = phmap::flat_hash_map<uint32_t, RedsanPcStats>;

    void kernel_start_callback(std::shared_ptr<KernelLaunch_t> kernel);
    void kernel_end_callback(std::shared_ptr<KernelEnd_t> kernel);
    void mem_alloc_callback(std::shared_ptr<MemAlloc_t> mem);
    void mem_free_callback(std::shared_ptr<MemFree_t> mem);
    void ten_alloc_callback(std::shared_ptr<TenAlloc_t> ten);
    void ten_free_callback(std::shared_ptr<TenFree_t> ten);
    void kernel_trace_flush(std::shared_ptr<KernelLaunch_t> kernel);

    // Apply the three rules to one 4-byte segment and update `st`.
    static void classify(RedsanAccess last, RedsanAccess cur, RedsanPcStats& st);

    // Global memory inside a tracked allocation.
    void unit_access(
        uint64_t offset_in_region,
        uint32_t pc_offset,
        RedsanAccess cls,
        uint32_t current_flat_thread_id,
        const memory_region& region,
        int access_size,
        RedsanPcStats& st
    );
    // Global memory outside every tracked allocation, and local memory (whose
    // addresses the GPU patch tags with the thread id): concurrent hashmap.
    void unit_access_unknown(
        uint64_t abs_addr,
        uint32_t pc_offset,
        RedsanAccess cls,
        uint32_t current_flat_thread_id,
        int access_size,
        RedsanPcStats& st
    );
    // Shared memory: per-CTA shadow object owned by the worker.
    void unit_access_shared(
        uint64_t ptr,
        uint32_t pc_offset,
        RedsanAccess cls,
        uint32_t object_idx,
        uint64_t current_block_id,
        uint32_t current_thread_in_block,
        int access_size,
        RedsanPcStats& st,
        worker_shared_shadow_state& shared_state
    );
    void worker_loop(uint64_t worker_idx);

/*
********************************* variables *********************************
*/
    Timer_t _timer;
    std::string output_directory;
    std::string summary_path;
    uint32_t kernel_id = 0;
    uint8_t _kernel_generation = 0;          // 5 bits in the packed entry
    uint32_t _shared_kernel_generation = 0;
    uint64_t _current_kernel_cta_count = 0;
    uint32_t _current_block_thread_count = 0;

    std::map<uint64_t, std::shared_ptr<KernelLaunch_t>> kernel_events;
    std::map<uint64_t, std::shared_ptr<MemAlloc_t>> alloc_events;
    std::map<DevPtr, std::shared_ptr<MemAlloc_t>> active_memories;
    std::map<uint64_t, std::shared_ptr<TenAlloc>> tensor_events;
    std::map<DevPtr, std::shared_ptr<TenAlloc>> active_tensors;

    std::vector<memory_region> _memory_regions;
    std::map<memory_region, std::unique_ptr<shadow_memory>> _shadow_memories;
    phmap::parallel_flat_hash_map_m<uint64_t, uint64_t> _unknown_region_shadow;

    PcStatsMap _pc_stats;

    // Worker pool (same layout as pc_dependency_analysis).
    uint64_t _worker_count = 1;
    std::vector<std::thread> _workers;
    std::vector<worker_shared_shadow_state> _worker_shadow_memory_shared;
    uint32_t _shared_shadow_object_cap_per_worker = 128;
    uint32_t _shared_shadow_bytes_per_object = 102400;

    const MemoryAccess* _job_accesses_buffer = nullptr;
    std::vector<std::vector<uint64_t>> _job_worker_trace_indices;
    std::vector<PcStatsMap> _job_worker_pc_stats;

    std::mutex _worker_pool_mutex;
    std::condition_variable _worker_pool_cv;
    std::condition_variable _worker_pool_done_cv;
    bool _worker_pool_shutdown = false;
    uint64_t _worker_job_generation = 0;
    uint64_t _worker_pending_jobs = 0;
};

}   // yosemite
#endif // YOSEMITE_TOOL_REDSAN_H
