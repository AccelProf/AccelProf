#ifndef YOSEMITE_TOOL_CODE_CHECK_H
#define YOSEMITE_TOOL_CODE_CHECK_H


#include "tools/tool.h"
#include "utils/event.h"
#include <map>

namespace yosemite {

class CodeCheck final : public Tool {
public:
    CodeCheck() : Tool(CODE_CHECK) {
        init();
    }

    ~CodeCheck() {}

    void evt_callback(EventPtr_t evt);

    void gpu_data_analysis(void* data, uint64_t size) override {};

    void query_ranges(void* ranges, uint32_t limit, uint32_t* count) override {};

    void query_tensors(void* ranges, uint32_t limit, uint32_t* count) override {};

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
    typedef enum {
        MEMCPY_UNKNOWN = 0,
        MEMCPY_H2H = 1,
        MEMCPY_H2D = 2,
        MEMCPY_D2H = 3,
        MEMCPY_D2D = 4,
    } MemcpyDirection_t;

    struct CpyStats {
        uint64_t count = 0;
        uint64_t size = 0;
    };

    struct SetStats {
        uint64_t count = 0;
        uint64_t size = 0;
    };

    struct MemStats {
        uint64_t alloc_count = 0;
        uint64_t alloc_size = 0;
        uint64_t free_count = 0;
        uint64_t free_size = 0;
    };

    struct TenStats {
        uint64_t alloc_count = 0;
        uint64_t alloc_size = 0;
        uint64_t free_count = 0;
        uint64_t free_size = 0;
    };


    Timer_t _timer;
    std::map<MemcpyDirection_t, CpyStats> cpy_stats;
    SetStats set_stats;
    MemStats mem_stats;
    TenStats ten_stats;
    uint64_t kernel_count = 0;
};

}   // yosemite
#endif // YOSEMITE_TOOL_CODE_CHECK_H