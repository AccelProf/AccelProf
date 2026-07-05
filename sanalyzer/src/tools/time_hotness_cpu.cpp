#include "tools/time_hotness_cpu.h"
#include "utils/helper.h"
#include "utils/hash.h"
#include "gpu_patch.h"
#include "cpp_trace.h"
#include "py_frame.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <string>
#include <iostream>
#include <unistd.h>


const uint32_t HOTNESS_GRANULARITY = 2*1024*1024;
const uint32_t SHIFT_BITS = 20;

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


TimeHotnessCPU::TimeHotnessCPU() : Tool(TIME_HOTNESS_CPU) {
    init();

}


TimeHotnessCPU::~TimeHotnessCPU() {
}

void TimeHotnessCPU::init() {
    const char* env_name = std::getenv("ACCEL_PROF_HOME");
    std::string lib_path;
    if (env_name) {
        lib_path = std::string(env_name) + "/lib/libcompute_sanitizer.so";
    }
    init_backtrace(lib_path.c_str());


    const char* env_sample_rate = std::getenv("ACCEL_PROF_ENV_SAMPLE_RATE");
    if (env_sample_rate) {
        setenv("YOSEMITE_ENV_SAMPLE_RATE", env_sample_rate, 1);
    }
}


void TimeHotnessCPU::evt_callback(EventPtr_t evt) {
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


void TimeHotnessCPU::kernel_start_callback(std::shared_ptr<KernelLaunch_t> kernel) {

}


std::shared_ptr<MemAlloc_t> TimeHotnessCPU::query_memory_ranges_cpu(uint64_t ptr) {

    return nullptr;
}


std::shared_ptr<TenAlloc_t> TimeHotnessCPU::query_tensor_ranges_cpu(uint64_t ptr) {

    return nullptr;
}


void TimeHotnessCPU::kernel_end_callback(std::shared_ptr<KernelEnd_t> kernel) {

}


void TimeHotnessCPU::mem_alloc_callback(std::shared_ptr<MemAlloc_t> mem) {
    mem_stats.alloc_count++;
    mem_stats.alloc_size += mem->size;
    mem_stats.max_size = std::max(mem_stats.max_size, mem_stats.alloc_size);

    _memories.emplace(mem->addr, mem);

    uint64_t block_ptr = mem->addr;
    int64_t size = (int64_t)mem->size;
    while (size > 0) {
        time_series_heatmap.emplace(block_ptr >> SHIFT_BITS, 0);
        size -= HOTNESS_GRANULARITY;
        block_ptr += HOTNESS_GRANULARITY;
    }

}


void TimeHotnessCPU::mem_free_callback(std::shared_ptr<MemFree_t> mem) {
    mem_stats.free_count++;
    mem_stats.free_size += mem->size;
    mem_stats.alloc_size -= mem->size;


}



void TimeHotnessCPU::mem_cpy_callback(std::shared_ptr<MemCpy_t> mem) {

}


void TimeHotnessCPU::mem_set_callback(std::shared_ptr<MemSet_t> mem) {

}


void TimeHotnessCPU::ten_alloc_callback(std::shared_ptr<TenAlloc_t> ten) {

}


void TimeHotnessCPU::ten_free_callback(std::shared_ptr<TenFree_t> ten) {

}


void TimeHotnessCPU::op_start_callback(std::shared_ptr<OpStart_t> op) {

}


void TimeHotnessCPU::op_end_callback(std::shared_ptr<OpEnd_t> op) {

}

void TimeHotnessCPU::gpu_data_analysis(void* data, uint64_t size) {
    MemoryAccess* accesses_buffer = (MemoryAccess*)data;

    for (uint32_t i = 0; i < size; i++) {
        MemoryAccess access = accesses_buffer[i];
        for (uint32_t j = 0; j < GPU_WARP_SIZE; j++) {
            if (access.addresses[j] != 0) {
                time_series_heatmap[access.addresses[j] >> SHIFT_BITS]++;
                _timer.increment(false);
                if (_timer.get() % 1000000 == 0) {
                    time_series_heatmap_list.push_back(time_series_heatmap);
                    for (auto& [key, value] : time_series_heatmap) {
                        time_series_heatmap[key] = 0;
                    }
                }
            }
        }
    }
}


void TimeHotnessCPU::query_ranges(void* ranges, uint32_t limit, uint32_t* count) {
}

void TimeHotnessCPU::query_tensors(void* ranges, uint32_t limit, uint32_t* count) {
}


void TimeHotnessCPU::flush() {
    const char* env_filename = std::getenv("YOSEMITE_APP_NAME");
    std::string filename = "output.log";
    if (env_filename) {
        filename = std::string(env_filename) + ".time_hotness_cpu.log";
    } else {
        fprintf(stdout, "No filename specified. Using default filename: %s\n", filename.c_str());
    }
    printf("Dumping traces to %s\n", filename.c_str());
    
    FILE* out_fp = fopen(filename.c_str(), "w");
    // print memory stats
    /*
    fprintf(out_fp, "Memory Stats:\n");
    fprintf(out_fp, "  Alloc Count: %lu\n", mem_stats.alloc_count);
    fprintf(out_fp, "  Alloc Size: %lu\n", mem_stats.alloc_size);
    fprintf(out_fp, "  Free Count: %lu\n", mem_stats.free_count);
    fprintf(out_fp, "  Free Size: %lu\n", mem_stats.free_size);
    fprintf(out_fp, "  Max Size: %lu\n", mem_stats.max_size);
    fprintf(out_fp, "--------------------------------\n");
    */

    
    std::map<uint64_t, uint64_t> heatmap_tmp;
    for (auto mem : _memories) {
        uint64_t block_ptr = mem.second->addr;
        int64_t size = (int64_t)mem.second->size;
        while (size > 0) {
            heatmap_tmp.emplace(block_ptr >> SHIFT_BITS, 0);
            size -= HOTNESS_GRANULARITY;
            block_ptr += HOTNESS_GRANULARITY;
        }
    }

    for (auto [key, value] : heatmap_tmp) {
        fprintf(out_fp, "%lu ", key);
    }
    fprintf(out_fp, "\n");


    for (auto& heatmap : time_series_heatmap_list) {
        for (auto [key, value] : heatmap) {
            if (heatmap_tmp.find(key) != heatmap_tmp.end()) {
                heatmap_tmp[key] = value;
            }
        }
        for (auto [key, value] : heatmap_tmp) {
            fprintf(out_fp, "%lu ", value);
            heatmap_tmp[key] = 0;
        }
        fprintf(out_fp, "\n");
    }

    fclose(out_fp);
}
