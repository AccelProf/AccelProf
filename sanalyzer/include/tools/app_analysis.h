#ifndef YOSEMITE_TOOL_APP_ANALYSIS_H
#define YOSEMITE_TOOL_APP_ANALYSIS_H


#include "tools/tool.h"
#include "utils/event.h"

#include <map>
#include <vector>
#include <tuple>
#include <stack>
#include <memory>

namespace yosemite {

class AppAnalysis final : public Tool {
public:
    AppAnalysis();

    ~AppAnalysis();

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


/*
********************************* variables *********************************
*/

    Timer_t _timer;

    std::map<DevPtr, std::shared_ptr<MemAlloc_t>> active_memories;

    std::map<DevPtr, std::shared_ptr<TenAlloc_t>> active_tensors;

    struct KernelStats {
        std::shared_ptr<KernelLaunch_t> kernel_launch;
        size_t tensor_working_set_size = 0;
        size_t memory_working_set_size = 0;
        size_t tensor_footprint_size = 0;
        size_t memory_footprint_size = 0;
    };
    uint64_t kernel_id = 0;
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

    // for overhead comparison
    int max_num_kernel_monitored = -1;

};  

}   // yosemite
#endif // YOSEMITE_TOOL_APP_ANALYSIS_H
