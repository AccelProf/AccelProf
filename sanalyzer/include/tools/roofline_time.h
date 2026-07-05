#ifndef YOSEMITE_TOOL_ROOFLINE_TIME_H
#define YOSEMITE_TOOL_ROOFLINE_TIME_H

#include "tools/tool.h"
#include "utils/event.h"
#include <map>

namespace yosemite {

class RooflineTime final : public Tool {
public:
    RooflineTime() : Tool(ROOFLINE_TIME) {}

    ~RooflineTime() {}

    void evt_callback(EventPtr_t evt);

    void gpu_data_analysis(void* data, uint64_t size);

    void query_ranges(void* ranges, uint32_t limit, uint32_t* count);

    void query_tensors(void* ranges, uint32_t limit, uint32_t* count) override {};

    void flush();

private:
    void kernel_start_callback(std::shared_ptr<KernelLaunch_t> kernel);

    void kernel_end_callback(std::shared_ptr<KernelEnd_t> kernel);

    void mem_alloc_callback(std::shared_ptr<MemAlloc_t> mem);

    void mem_free_callback(std::shared_ptr<MemFree_t> mem);
    
    void ten_alloc_callback(std::shared_ptr<TenAlloc_t> ten);

    void ten_free_callback(std::shared_ptr<TenFree_t> ten);

    /*
    ***************************** variables *********************************
    */
    double start_time;
    std::map<std::shared_ptr<KernelEnd_t>, double> kernel_time_map;
    uint64_t cur_mem_usage, max_mem_usage;
    int64_t cur_ten_usage, max_ten_usage;

};

}   // yosemite
#endif // YOSEMITE_TOOL_ROOFLINE_TIME_H
