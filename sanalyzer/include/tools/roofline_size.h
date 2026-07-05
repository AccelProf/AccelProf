#ifndef YOSEMITE_TOOL_ROOFLINE_SIZE_H
#define YOSEMITE_TOOL_ROOFLINE_SIZE_H
#include "tools/tool.h"
#include "utils/event.h"
#include <map>

namespace yosemite {

class RooflineSize final : public Tool {
public:
    RooflineSize() : Tool(ROOFLINE_SIZE) {}

    ~RooflineSize() {}

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

    /*
    ********************************* variables *********************************
    */
    uint64_t accessCount;
    uint64_t accessSize;
    std::map<std::shared_ptr<KernelEnd_t>, std::pair<uint64_t, uint64_t>> kernel_size_map;
    
};

}   // yosemite
#endif // YOSEMITE_TOOL_ROOFLINE_SIZE_H
