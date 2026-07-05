#ifndef YOSEMITE_TOOL_APP_METRICS_H
#define YOSEMITE_TOOL_APP_METRICS_H


#include "tools/tool.h"
#include "utils/event.h"
#include <map>

namespace yosemite {

class AppMetrics final : public Tool {
public:
    AppMetrics() : Tool(APP_METRICE) {}

    ~AppMetrics() {}

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
    typedef struct Stats{
        uint64_t num_allocs;
        uint64_t num_kernels;
        uint64_t cur_mem_usage;
        uint64_t max_mem_usage;
        uint64_t max_mem_accesses_per_kernel;
        uint64_t avg_mem_accesses;
        uint64_t tot_mem_accesses;
        std::string max_mem_accesses_kernel;
        uint64_t max_mem_access_kernel_id;
        uint64_t max_objs_per_kernel;
        uint64_t avg_objs_per_kernel;
        uint64_t tot_objs_per_kernel;
        uint64_t max_obj_size_per_kernel;
        uint64_t avg_obj_size_per_kernel;
        uint64_t tot_obj_size_per_kernel;

        Stats() = default;

        ~Stats() = default;
    } Stats_t;

    Stats_t _stats;
    uint64_t _kernel_id = 0;

    Timer_t _timer;

    std::map<uint64_t, std::shared_ptr<KernelLaunch_t>> kernel_events;
    std::map<uint64_t, std::shared_ptr<MemAlloc_t>> alloc_events;
    std::map<DevPtr, std::shared_ptr<MemAlloc_t>> active_memories;

    std::map<std::string, uint32_t> kernel_invocations;
};

}   // yosemite
#endif // YOSEMITE_TOOL_APP_METRICS_H
