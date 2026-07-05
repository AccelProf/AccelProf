#ifndef YOSEMITE_TOOL_APP_ANALYSIS_NVBIT_H
#define YOSEMITE_TOOL_APP_ANALYSIS_NVBIT_H


#include "tools/tool.h"
#include "utils/event.h"

#include <map>
#include <vector>
#include <tuple>
#include <stack>
#include <memory>
#include <set>


namespace yosemite {

class AppAnalysisNVBIT final : public Tool {
public:
    AppAnalysisNVBIT();

    ~AppAnalysisNVBIT();

    void evt_callback(EventPtr_t evt);

    void gpu_data_analysis(void* data, uint64_t size);

    void query_ranges(void* ranges, uint32_t limit, uint32_t* count);

    void query_tensors(void* ranges, uint32_t limit, uint32_t* count);

    void flush();

private :
    void init();

    void kernel_start_callback(std::shared_ptr<KernelLaunch_t> kernel);

    void kernel_end_callback(std::shared_ptr<KernelEnd_t> kernel);

    void mem_alloc_callback(std::shared_ptr<MemAlloc_t> mem);

    void mem_free_callback(std::shared_ptr<MemFree_t> mem);

    void mem_cpy_callback(std::shared_ptr<MemCpy_t> mem);

    void mem_set_callback(std::shared_ptr<MemSet_t> mem);

    void ten_alloc_callback(std::shared_ptr<TenAlloc_t> ten);

    void ten_free_callback(std::shared_ptr<TenFree_t> ten);

    void op_start_callback(std::shared_ptr<OpStart_t> op);

    void op_end_callback(std::shared_ptr<OpEnd_t> op);

    std::shared_ptr<MemAlloc_t> query_memory_ranges_cpu(uint64_t ptr, uint64_t grid_launch_id);

    std::shared_ptr<TenAlloc_t> query_tensor_ranges_cpu(uint64_t ptr, uint64_t grid_launch_id);

    void kernel_grid_launch_id_transition();

/*
********************************* variables *********************************
*/

    Timer_t _timer;

    using active_mem_map = std::map<DevPtr, std::shared_ptr<MemAlloc_t>>;
    active_mem_map active_memories;
    std::set<std::shared_ptr<MemAlloc_t>> touched_memories;
    std::map<uint64_t, active_mem_map> active_memories_per_kernel_snapshot;

    using active_ten_map = std::map<DevPtr, std::shared_ptr<TenAlloc_t>>;
    active_ten_map active_tensors;
    std::set<std::shared_ptr<TenAlloc_t>> touched_tensors;
    std::map<uint64_t, active_ten_map> active_tensors_per_kernel_snapshot;

    struct KernelStats {
        std::shared_ptr<KernelLaunch_t> kernel_launch;
        size_t tensor_working_set_size = 0;
        size_t memory_working_set_size = 0;
        size_t tensor_footprint_size = 0;
        size_t memory_footprint_size = 0;
    };
    uint64_t kernel_id = 0;
    uint64_t current_grid_launch_id = 0;
    uint64_t previous_grid_launch_id = 0;
    uint64_t current_kernel_access_count = 0;

    std::map<uint64_t, KernelStats> kernel_stats;

    struct MemStats {
        uint64_t max_size = 0;
        uint64_t alloc_count = 0;
        uint64_t alloc_size = 0;
        uint64_t free_count = 0;
        uint64_t free_size = 0;
    };
    MemStats mem_stats;
    struct TenStats {
        uint64_t max_size = 0;
        uint64_t alloc_count = 0;
        uint64_t alloc_size = 0;
        uint64_t free_count = 0;
        uint64_t free_size = 0;
    };
    TenStats ten_stats;

};  

}   // yosemite
#endif // YOSEMITE_TOOL_APP_ANALYSIS_NVBIT_H
