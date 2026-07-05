#ifndef YOSEMITE_TOOL_TIME_HOTNESS_CPU_H
#define YOSEMITE_TOOL_TIME_HOTNESS_CPU_H


#include "tools/tool.h"
#include "utils/event.h"

#include <map>
#include <vector>
#include <tuple>
#include <stack>
#include <memory>
#include <set>


namespace yosemite {

class TimeHotnessCPU final : public Tool {
public:
    TimeHotnessCPU();

    ~TimeHotnessCPU();

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

    std::shared_ptr<MemAlloc_t> query_memory_ranges_cpu(uint64_t ptr);

    std::shared_ptr<TenAlloc_t> query_tensor_ranges_cpu(uint64_t ptr);

/*
********************************* variables *********************************
*/

    Timer_t _timer;

    // no free memory
    std::map<DevPtr, std::shared_ptr<MemAlloc_t>> _memories;

    struct MemStats {
        uint64_t max_size = 0;
        uint64_t alloc_count = 0;
        uint64_t alloc_size = 0;
        uint64_t free_count = 0;
        uint64_t free_size = 0;
    };
    MemStats mem_stats;

    std::map<uint64_t, uint64_t> time_series_heatmap;
    std::vector<std::map<uint64_t, uint64_t>> time_series_heatmap_list;

};  

}   // yosemite
#endif // YOSEMITE_TOOL_TIME_HOTNESS_CPU_H
