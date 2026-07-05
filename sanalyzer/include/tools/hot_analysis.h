#ifndef YOSEMITE_HOT_ANALYSIS_H
#define YOSEMITE_HOT_ANALYSIS_H


#include "tools/tool.h"
#include "utils/event.h"
#include "gpu_patch.h"
#include <map>

namespace yosemite {

class HotAnalysis final : public Tool {
public:
    HotAnalysis();

    ~HotAnalysis();

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

    void mem_cpy_callback(std::shared_ptr<MemCpy_t> mem);

    void mem_set_callback(std::shared_ptr<MemSet_t> mem);

    void ten_alloc_callback(std::shared_ptr<TenAlloc_t> ten);

    void ten_free_callback(std::shared_ptr<TenFree_t> ten);

/*
********************************* variables *********************************
*/

    std::map<DevPtr, std::shared_ptr<MemAlloc_t>> active_memories;
    std::map<DevPtr, std::shared_ptr<TenAlloc>> active_tensors;
    std::map<MemoryRange, uint32_t> range_access_counts;

    std::string output_directory;
    uint32_t global_kernel_id = 0;
};

}   // namespace yosemite
#endif // YOSEMITE_HOT_ANALYSIS_H
