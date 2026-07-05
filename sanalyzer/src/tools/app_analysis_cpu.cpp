#include "tools/app_analysis_cpu.h"
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


AppAnalysisCPU::AppAnalysisCPU() : Tool(APP_ANALYSIS_CPU) {
    init();

}


AppAnalysisCPU::~AppAnalysisCPU() {
}

void AppAnalysisCPU::init() {
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


void AppAnalysisCPU::evt_callback(EventPtr_t evt) {
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


void AppAnalysisCPU::kernel_start_callback(std::shared_ptr<KernelLaunch_t> kernel) {
    KernelStats stats;
    stats.kernel_launch = kernel;
    stats.tensor_footprint_size = ten_stats.alloc_size;
    stats.memory_footprint_size = mem_stats.alloc_size;
    kernel_stats.emplace(kernel_id, stats);

    _timer.increment(true);
}


std::shared_ptr<MemAlloc_t> AppAnalysisCPU::query_memory_ranges_cpu(uint64_t ptr) {
    for (auto mem : active_memories) {
        if (mem.second->addr <= ptr && mem.second->addr + mem.second->size >= ptr) {
            return mem.second;
        }
    }
    assert(false);
    return nullptr;
}


std::shared_ptr<TenAlloc_t> AppAnalysisCPU::query_tensor_ranges_cpu(uint64_t ptr) {
    for (auto ten : active_tensors) {
        if (ten.second->addr <= ptr && ten.second->addr + ten.second->size > ptr) {
            return ten.second;
        }
    }
    assert(false);
    return nullptr;
}


void AppAnalysisCPU::kernel_end_callback(std::shared_ptr<KernelEnd_t> kernel) {
    size_t tensor_working_set_size = 0;
    for (auto ten : touched_tensors) {
        tensor_working_set_size += ten->size;
    }

    size_t memory_working_set_size = 0;
    for (auto mem : touched_memories) {
        memory_working_set_size += mem->size;
    }

    kernel_stats[kernel_id].tensor_working_set_size = tensor_working_set_size;
    kernel_stats[kernel_id].memory_working_set_size = memory_working_set_size;

    touched_tensors.clear();
    touched_memories.clear();

    kernel_id++;

    if (max_num_kernel_monitored != -1 && kernel_id >= max_num_kernel_monitored) {
        fprintf(stdout, "Max number of kernels monitored reached. Exiting...\n");
        fflush(stdout);
        _exit(0);
    }

    _timer.increment(true);
}


void AppAnalysisCPU::mem_alloc_callback(std::shared_ptr<MemAlloc_t> mem) {
    mem_stats.alloc_count++;
    mem_stats.alloc_size += mem->size;
    mem_stats.max_size = std::max(mem_stats.max_size, mem_stats.alloc_size);
    active_memories.emplace(mem->addr, mem);

    _timer.increment(true);
}


void AppAnalysisCPU::mem_free_callback(std::shared_ptr<MemFree_t> mem) {
    mem_stats.free_count++;
    mem_stats.free_size += mem->size;
    mem_stats.alloc_size -= mem->size;

    auto it = active_memories.find(mem->addr);
    assert(it != active_memories.end());
    active_memories.erase(it);

    _timer.increment(true);
}



void AppAnalysisCPU::mem_cpy_callback(std::shared_ptr<MemCpy_t> mem) {

    _timer.increment(true);
}


void AppAnalysisCPU::mem_set_callback(std::shared_ptr<MemSet_t> mem) {

    _timer.increment(true);
}


void AppAnalysisCPU::ten_alloc_callback(std::shared_ptr<TenAlloc_t> ten) {
    ten_stats.alloc_count++;
    ten_stats.alloc_size += ten->size;
    ten_stats.max_size = std::max(ten_stats.max_size, ten_stats.alloc_size);

    active_tensors.emplace(ten->addr, ten);

    _timer.increment(true);
}


void AppAnalysisCPU::ten_free_callback(std::shared_ptr<TenFree_t> ten) {
    ten_stats.free_count++;
    ten_stats.free_size += -ten->size;
    ten_stats.alloc_size -= -ten->size;

    auto it = active_tensors.find(ten->addr);
    assert(it != active_tensors.end());
    active_tensors.erase(it);

    _timer.increment(true);
}


void AppAnalysisCPU::op_start_callback(std::shared_ptr<OpStart_t> op) {

    _timer.increment(true);
}


void AppAnalysisCPU::op_end_callback(std::shared_ptr<OpEnd_t> op) {

    _timer.increment(true);
}

void AppAnalysisCPU::gpu_data_analysis(void* data, uint64_t size) {
    MemoryAccess* accesses_buffer = (MemoryAccess*)data;
    uint32_t num_accesses = 0;
    for (uint32_t i = 0; i < size; i++) {
        MemoryAccess access = accesses_buffer[i];
        for (uint32_t j = 0; j < GPU_WARP_SIZE; j++) {
            if (access.addresses[j] != 0) {
                num_accesses++;
                auto tensor = query_tensor_ranges_cpu(access.addresses[j]);
                auto memory = query_memory_ranges_cpu(access.addresses[j]);
                if (tensor != nullptr && memory != nullptr) {
                    touched_tensors.insert(tensor);
                    touched_memories.insert(memory);
                    // break;
                }
            }
        }
    }
    kernel_stats[kernel_id].kernel_launch->access_count += num_accesses;
}


void AppAnalysisCPU::query_ranges(void* ranges, uint32_t limit, uint32_t* count) {
}

void AppAnalysisCPU::query_tensors(void* ranges, uint32_t limit, uint32_t* count) {
}


void AppAnalysisCPU::flush() {
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
