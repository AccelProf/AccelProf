#ifndef YOSEMITE_TOOL_EVENT_TRACE_H
#define YOSEMITE_TOOL_EVENT_TRACE_H

#include "tools/tool.h"
#include "utils/event.h"
#include <vector>
#include <map>

namespace yosemite {

class EventTrace final : public Tool {
    public:
    EventTrace();

    ~EventTrace();

    void gpu_data_analysis(void* data, uint64_t size) override {};

    void query_ranges(void* ranges, uint32_t limit, uint32_t* count) override {};

    void query_tensors(void* ranges, uint32_t limit, uint32_t* count) override {};

    void evt_callback(EventPtr_t evt);

    void flush();

private:
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

    std::map<DevPtr, std::shared_ptr<MemAlloc_t>> _active_memories;

    int64_t _memory_size = 0;
    int64_t _tensor_size = 0;

    std::vector<int64_t> _memory_size_list;
    std::vector<int64_t> _tensor_size_list;

};

} // namespace yosemite
#endif // YOSEMITE_TOOL_EVENT_TRACE_H
