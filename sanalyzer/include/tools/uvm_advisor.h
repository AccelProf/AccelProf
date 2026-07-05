#ifndef YOSEMITE_TOOL_UVM_ADVISOR_H
#define YOSEMITE_TOOL_UVM_ADVISOR_H


#include "tools/tool.h"
#include "utils/event.h"

#include <map>
#include <vector>
#include <tuple>
#include <stack>
#include <memory>
#include <unordered_map>
#include <unordered_set>

namespace yosemite {

class UVMAdvisor final : public Tool {
public:
    UVMAdvisor();

    ~UVMAdvisor();

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

    bool find_uvm_tensor(uint64_t ptr);

    void print_callstack();
/*
********************************* variables *********************************
*/

    Timer_t _timer;

    std::map<uint64_t, std::shared_ptr<MemAlloc_t>> alloc_events;
    std::map<DevPtr, std::shared_ptr<MemAlloc_t>> active_memories;

    std::map<uint64_t, std::shared_ptr<TenAlloc_t>> tenalloc_events;
    std::map<DevPtr, std::shared_ptr<TenAlloc_t>> active_tensors;

    std::vector<std::shared_ptr<KernelLaunch_t>> kernel_events;

    struct MemStats {
        uint64_t alloc_count = 0;
        uint64_t alloc_size = 0;
        uint64_t free_count = 0;
        uint64_t free_size = 0;
        uint64_t current_mem_size = 0;
        uint64_t max_mem_size = 0;
    };
    MemStats mem_stats;
    struct TenStats {
        uint64_t alloc_count = 0;
        uint64_t alloc_size = 0;
        uint64_t free_count = 0;
        uint64_t free_size = 0;
        uint64_t current_ten_size = 0;
        uint64_t max_ten_size = 0;
    };
    TenStats ten_stats;

    struct OpStats {
        uint64_t count = 0;
        uint64_t group_count = 0;
        uint64_t pending_ops = 0;
        uint64_t pending_mem_alloc = 0;
        uint64_t pending_ten_alloc = 0;
        uint64_t pending_kernels = 0;
    };
    OpStats op_stats;

    using MemAllocVec = std::vector<std::shared_ptr<MemAlloc_t>>;
    using TenAllocVec = std::vector<std::shared_ptr<TenAlloc_t>>;
    using KernelResources = std::tuple<std::shared_ptr<KernelLaunch_t>, MemAllocVec, TenAllocVec>;
    using KernelResourceVec = std::vector<KernelResources>;
    using OpResourceMap = std::map<uint64_t, std::pair<std::shared_ptr<OpStart_t>, KernelResourceVec>>;
    KernelResourceVec kernel_resources;
    OpResourceMap op_tables;
    std::stack<std::shared_ptr<OpStart_t>> op_stack;

    typedef struct {
        uint64_t op_id = 0;
        uint64_t ten_id = 0;
        uint64_t mem_id = 0;
        uint64_t kernel_id = 0;
    } opt_keys_t;
    opt_keys_t opt_keys;

    std::unordered_set<uint64_t> mem_alloc_during_this_op;
    std::unordered_set<uint64_t> ten_alloc_during_this_op;
};  

}   // yosemite
#endif // YOSEMITE_TOOL_UVM_ADVISOR_H
