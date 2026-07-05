#include "tools/code_check.h"
#include "utils/helper.h"
#include "utils/hash.h"
#include "gpu_patch.h"
#include "cpp_trace.h"
#include "py_frame.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <iostream>

using namespace yosemite;


inline std::string vector2str(std::vector<std::string> &vec, int skip_first = 0, int skip_last = 0) {
    if (skip_first + skip_last > vec.size()) {
        printf("Skip first and skip last are larger than the vector size\n");
        return "";
    }
    std::string str;
    for (size_t i = skip_first; i < vec.size() - skip_last; i++) {
        str += vec[i] + "\n";
    }
    return str;
}

void CodeCheck::init() {
    const char* env_name = std::getenv("ACCEL_PROF_HOME");
    std::string lib_path;
    if (env_name) {
        lib_path = std::string(env_name) + "/lib/libcompute_sanitizer.so";
    }
    init_backtrace(lib_path.c_str());

}


void CodeCheck::evt_callback(EventPtr_t evt) {
    switch (evt->evt_type) {
        case EventType_KERNEL_LAUNCH:
            kernel_start_callback(std::dynamic_pointer_cast<KernelLaunch_t>(evt));
            break;
        case EventType_KERNEL_END:
            kernel_end_callback(std::dynamic_pointer_cast<KernelEnd_t>(evt));
            break;
        case EventType_MEM_ALLOC:
            mem_alloc_callback(std::dynamic_pointer_cast<MemAlloc_t>(evt));
            break;
        case EventType_MEM_FREE:
            mem_free_callback(std::dynamic_pointer_cast<MemFree_t>(evt));
            break;
        case EventType_MEM_COPY:
            mem_cpy_callback(std::dynamic_pointer_cast<MemCpy_t>(evt));
            break;
        case EventType_MEM_SET:
            mem_set_callback(std::dynamic_pointer_cast<MemSet_t>(evt));
            break;
        case EventType_TEN_ALLOC:
            ten_alloc_callback(std::dynamic_pointer_cast<TenAlloc_t>(evt));
            break;
        case EventType_TEN_FREE:
            ten_free_callback(std::dynamic_pointer_cast<TenFree_t>(evt));
            break;
        case EventType_OP_START:
            op_start_callback(std::dynamic_pointer_cast<OpStart_t>(evt));
            break;
        case EventType_OP_END:
            op_end_callback(std::dynamic_pointer_cast<OpEnd_t>(evt));
            break;
        default:
            break;
    }
}


void CodeCheck::kernel_start_callback(std::shared_ptr<KernelLaunch_t> kernel) {
    kernel_count++;
    _timer.increment(true);
}


void CodeCheck::kernel_end_callback(std::shared_ptr<KernelEnd_t> kernel) {
}


void CodeCheck::mem_alloc_callback(std::shared_ptr<MemAlloc_t> mem) {
    mem_stats.alloc_count++;
    mem_stats.alloc_size += mem->size;

    _timer.increment(true);
}


void CodeCheck::mem_free_callback(std::shared_ptr<MemFree_t> mem) {
    mem_stats.free_count++;
    mem_stats.free_size += mem->size;

    _timer.increment(true);
}



void CodeCheck::mem_cpy_callback(std::shared_ptr<MemCpy_t> mem) {
    // auto backtraces = get_backtrace();
    // auto py_frames = get_pyframes();
    // auto bt_str = vector2str(backtraces);
    // auto pf_str = vector2str(py_frames);

    // std::cout << "Backtrace hash: " << sha256(bt_str) << std::endl;
    // std::cout << bt_str << std::endl;
    // std::cout << "Python frame hash: " << sha256(pf_str) << std::endl;
    // std::cout << pf_str << std::endl;

    MemcpyDirection_t direction = (MemcpyDirection_t)mem->direction;
    if (cpy_stats.find(direction) == cpy_stats.end()) {
        cpy_stats[direction] = CpyStats{0, 0};
    }
    cpy_stats[direction].count++;
    cpy_stats[direction].size += mem->size;

    _timer.increment(true);
}


void CodeCheck::mem_set_callback(std::shared_ptr<MemSet_t> mem) {
    set_stats.count++;
    set_stats.size += mem->size;

    _timer.increment(true);
}


void CodeCheck::ten_alloc_callback(std::shared_ptr<TenAlloc_t> ten) {
    ten_stats.alloc_count++;
    ten_stats.alloc_size += ten->size;

    _timer.increment(true);
}


void CodeCheck::ten_free_callback(std::shared_ptr<TenFree_t> ten) {
    ten_stats.free_count++;
    ten_stats.free_size += -ten->size;

    _timer.increment(true);
}


void CodeCheck::op_start_callback(std::shared_ptr<OpStart_t> op) {
    fprintf(stdout, "Op start: %s, ctx: %p\n", op->op_name.c_str(), op->ctx);

    _timer.increment(true);
}


void CodeCheck::op_end_callback(std::shared_ptr<OpEnd_t> op) {
    fprintf(stdout, "Op end: %s, ctx: %p\n", op->op_name.c_str(), op->ctx);

    _timer.increment(true);
}


void CodeCheck::flush() {
    fprintf(stdout, "--------------------------------------------------------------------------------\n");
    fprintf(stdout, "%-12s count: %-10lu\n", "[Kernel]", kernel_count);
    fprintf(stdout, "%-12s count: %-10lu, size: %lu (%s)\n", 
            "[MemMalloc]", mem_stats.alloc_count, mem_stats.alloc_size, format_size(mem_stats.alloc_size).c_str());
    fprintf(stdout, "%-12s count: %-10lu, size: %lu (%s)\n", 
            "[MemFree]", mem_stats.free_count, mem_stats.free_size, format_size(mem_stats.free_size).c_str());
    fprintf(stdout, "%-12s count: %-10lu, size: %lu (%s)\n", 
            "[Memset]", set_stats.count, set_stats.size, format_size(set_stats.size).c_str());

    for (auto& it : cpy_stats) {
        const char* direction = it.first == MEMCPY_H2H ? "H2H" : 
                              it.first == MEMCPY_H2D ? "H2D" :
                              it.first == MEMCPY_D2H ? "D2H" : 
                              it.first == MEMCPY_D2D ? "D2D" : "N/A";
        fprintf(stdout, "[Memcpy-%s] count: %-10lu, size: %lu (%s)\n",
                direction, it.second.count, it.second.size, format_size(it.second.size).c_str());
    }

    fprintf(stdout, "%-12s count: %-10lu, size: %lu (%s)\n", 
            "[TenMalloc]", ten_stats.alloc_count, ten_stats.alloc_size, format_size(ten_stats.alloc_size).c_str());
    fprintf(stdout, "%-12s count: %-10lu, size: %lu (%s)\n", 
            "[TenFree]", ten_stats.free_count, ten_stats.free_size, format_size(ten_stats.free_size).c_str());
    fprintf(stdout, "--------------------------------------------------------------------------------\n");
}
