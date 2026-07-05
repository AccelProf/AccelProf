#include "tools/app_analysis.h"
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


AppAnalysis::AppAnalysis() : Tool(APP_ANALYSIS) {
    init();

}


AppAnalysis::~AppAnalysis() {
}

void AppAnalysis::init() {
    const char* env_name = std::getenv("ACCEL_PROF_HOME");
    std::string lib_path;
    if (env_name) {
        lib_path = std::string(env_name) + "/lib/libcompute_sanitizer.so";
    }
    init_backtrace(lib_path.c_str());

    const char* env_filename = std::getenv("MAX_NUM_KERNEL_MONITORED");
    if (env_filename) {
        max_num_kernel_monitored = std::stoi(env_filename);
        printf("Set max number of kernels monitored to %d\n", max_num_kernel_monitored);
    }

    const char* env_sample_rate = std::getenv("ACCEL_PROF_ENV_SAMPLE_RATE");
    if (env_sample_rate) {
        setenv("YOSEMITE_ENV_SAMPLE_RATE", env_sample_rate, 1);
    }
}


void AppAnalysis::evt_callback(EventPtr_t evt) {
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


void AppAnalysis::kernel_start_callback(std::shared_ptr<KernelLaunch_t> kernel) {
    KernelStats stats;
    stats.kernel_launch = kernel;
    stats.tensor_footprint_size = ten_stats.alloc_size;
    stats.memory_footprint_size = mem_stats.alloc_size;
    kernel_stats.emplace(kernel_id, stats);

    _timer.increment(true);
}


void AppAnalysis::kernel_end_callback(std::shared_ptr<KernelEnd_t> kernel) {
    kernel_id++;
    if (max_num_kernel_monitored != -1 && kernel_id >= max_num_kernel_monitored) {
        fprintf(stdout, "Max number of kernels monitored reached. Exiting...\n");
        fflush(stdout);
        exit(0);
    }
    _timer.increment(true);
}


void AppAnalysis::mem_alloc_callback(std::shared_ptr<MemAlloc_t> mem) {
    mem_stats.alloc_count++;
    mem_stats.alloc_size += mem->size;
    mem_stats.max_size = std::max(mem_stats.max_size, mem_stats.alloc_size);
    active_memories.emplace(mem->addr, mem);

    _timer.increment(true);
}


void AppAnalysis::mem_free_callback(std::shared_ptr<MemFree_t> mem) {
    mem_stats.free_count++;
    mem_stats.free_size += mem->size;
    mem_stats.alloc_size -= mem->size;

    auto it = active_memories.find(mem->addr);
    assert(it != active_memories.end());
    active_memories.erase(it);

    _timer.increment(true);
}



void AppAnalysis::mem_cpy_callback(std::shared_ptr<MemCpy_t> mem) {

    _timer.increment(true);
}


void AppAnalysis::mem_set_callback(std::shared_ptr<MemSet_t> mem) {

    _timer.increment(true);
}


void AppAnalysis::ten_alloc_callback(std::shared_ptr<TenAlloc_t> ten) {
    ten_stats.alloc_count++;
    ten_stats.alloc_size += ten->size;
    ten_stats.max_size = std::max(ten_stats.max_size, ten_stats.alloc_size);

    active_tensors.emplace(ten->addr, ten);

    _timer.increment(true);
}


void AppAnalysis::ten_free_callback(std::shared_ptr<TenFree_t> ten) {
    ten_stats.free_count++;
    ten_stats.free_size += -ten->size;
    ten_stats.alloc_size -= -ten->size;

    auto it = active_tensors.find(ten->addr);
    assert(it != active_tensors.end());
    active_tensors.erase(it);

    _timer.increment(true);
}


void AppAnalysis::op_start_callback(std::shared_ptr<OpStart_t> op) {

    _timer.increment(true);
}


void AppAnalysis::op_end_callback(std::shared_ptr<OpEnd_t> op) {

    _timer.increment(true);
}

void AppAnalysis::gpu_data_analysis(void* data, uint64_t size) {
    MemoryAccessTracker* tracker = (MemoryAccessTracker*)data;
    MemoryAccessState* states = tracker->access_state;
    TensorAccessState* tensor_states = tracker->tensor_access_state;

    uint64_t access_count = tracker->accessCount;

    uint64_t mem_size = 0;
    for (uint32_t i = 0; i < states->size; i++) {
        if (states->touch[i] != 0) {
            mem_size += states->start_end[i].end - states->start_end[i].start;
        }
    }

    uint64_t ten_size = 0;
    for (uint32_t i = 0; i < tensor_states->size; i++) {
        if (tensor_states->touch[i] != 0) {
            ten_size += tensor_states->start_end[i].end - tensor_states->start_end[i].start;
        }
    }

    kernel_stats[kernel_id].kernel_launch->access_count = access_count;
    kernel_stats[kernel_id].tensor_working_set_size = ten_size;
    kernel_stats[kernel_id].memory_working_set_size = mem_size;
}


void AppAnalysis::query_ranges(void* ranges, uint32_t limit, uint32_t* count) {
    MemoryRange* _ranges = (MemoryRange*)ranges;
    *count = 0;
    for (auto mem : active_memories) {
        _ranges[*count].start = mem.second->addr;
        _ranges[*count].end = mem.second->addr + mem.second->size;
        (*count)++;
        if (*count >= limit) {
            printf("Warning: query_ranges limit reached\n");
            break;
        }
    }
}

void AppAnalysis::query_tensors(void* ranges, uint32_t limit, uint32_t* count) {
    MemoryRange* _ranges = (MemoryRange*)ranges;
    *count = 0;
    for (auto ten : active_tensors) {
        _ranges[*count].start = ten.second->addr;
        _ranges[*count].end = ten.second->addr + ten.second->size;
        (*count)++;
        if (*count >= limit) {
            printf("Warning: query_tensors limit reached\n");
            break;
        }
    }
}


void AppAnalysis::flush() {
    const char* env_filename = std::getenv("YOSEMITE_APP_NAME");
    std::string filename = "output.log";
    if (env_filename) {
        filename = std::string(env_filename) + "_app_analysis.log";
    } else {
        fprintf(stdout, "No filename specified. Using default filename: %s\n", filename.c_str());
    }
    printf("Dumping traces to %s\n", filename.c_str());
    
    FILE* out_fp = fopen(filename.c_str(), "w");
    // print tensor stats
    fprintf(out_fp, "Tensor Stats:\n");
    fprintf(out_fp, "  Alloc Count: %lu\n", ten_stats.alloc_count);
    fprintf(out_fp, "  Alloc Size: %lu\n", ten_stats.alloc_size);
    fprintf(out_fp, "  Free Count: %lu\n", ten_stats.free_count);
    fprintf(out_fp, "  Free Size: %lu\n", ten_stats.free_size);
    // print memory stats
    fprintf(out_fp, "Memory Stats:\n");
    fprintf(out_fp, "  Alloc Count: %lu\n", mem_stats.alloc_count);
    fprintf(out_fp, "  Alloc Size: %lu\n", mem_stats.alloc_size);
    fprintf(out_fp, "  Free Count: %lu\n", mem_stats.free_count);
    fprintf(out_fp, "  Free Size: %lu\n", mem_stats.free_size);

    // print kernel stats
    fprintf(out_fp, "Kernel Stats:\n");
    for (auto& [kernel_id, stats] : kernel_stats) {
        fprintf(out_fp, "Kernel ID: %lu\n", kernel_id);
        fprintf(out_fp, "  Kernel Name: %s\n", stats.kernel_launch->kernel_name.c_str());
        fprintf(out_fp, "  Access Count: %lu\n", stats.kernel_launch->access_count);
        fprintf(out_fp, "  Tensor Working Set Size: %lu (%s)\n", stats.tensor_working_set_size, format_size(stats.tensor_working_set_size).c_str());
        fprintf(out_fp, "  Memory Working Set Size: %lu (%s)\n", stats.memory_working_set_size, format_size(stats.memory_working_set_size).c_str());
        fprintf(out_fp, "  Tensor Footprint Size: %lu (%s)\n", stats.tensor_footprint_size, format_size(stats.tensor_footprint_size).c_str());
        fprintf(out_fp, "  Memory Footprint Size: %lu (%s)\n", stats.memory_footprint_size, format_size(stats.memory_footprint_size).c_str());
    }

    fclose(out_fp);
}
