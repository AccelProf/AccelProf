#include "tools/app_analysis_nvbit.h"
#include "utils/helper.h"
#include "utils/hash.h"
#include "nvbit_common.h"
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


AppAnalysisNVBIT::AppAnalysisNVBIT() : Tool(APP_ANALYSIS_NVBIT) {
    init();
}


AppAnalysisNVBIT::~AppAnalysisNVBIT() {
}

void AppAnalysisNVBIT::init() {
    const char* env_name = std::getenv("ACCEL_PROF_HOME");
    std::string lib_path;
    if (env_name) {
        lib_path = std::string(env_name) + "/lib/libcompute_sanitizer.so";
    }
    init_backtrace(lib_path.c_str());

}


void AppAnalysisNVBIT::evt_callback(EventPtr_t evt) {
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


void AppAnalysisNVBIT::kernel_start_callback(std::shared_ptr<KernelLaunch_t> kernel) {
    KernelStats stats;
    stats.kernel_launch = kernel;
    stats.tensor_footprint_size = ten_stats.alloc_size;
    stats.memory_footprint_size = mem_stats.alloc_size;
    kernel_stats.emplace(kernel_id, stats);
    active_memories_per_kernel_snapshot[kernel_id] = active_memories;
    active_tensors_per_kernel_snapshot[kernel_id] = active_tensors;

    kernel_id++;
    _timer.increment(true);
}


std::shared_ptr<MemAlloc_t> AppAnalysisNVBIT::query_memory_ranges_cpu(uint64_t ptr, uint64_t grid_launch_id) {
    auto active_memories_snapshot = active_memories_per_kernel_snapshot[grid_launch_id];
    for (auto mem : active_memories_snapshot) {
        if (mem.second->addr <= ptr && mem.second->addr + mem.second->size >= ptr) {
            return mem.second;
        }
    }
    // assert(false);
    return nullptr;
}


std::shared_ptr<TenAlloc_t> AppAnalysisNVBIT::query_tensor_ranges_cpu(uint64_t ptr, uint64_t grid_launch_id) {
    auto active_tensors_snapshot = active_tensors_per_kernel_snapshot[grid_launch_id];
    for (auto ten : active_tensors_snapshot) {
        if (ten.second->addr <= ptr && ten.second->addr + ten.second->size > ptr) {
            return ten.second;
        }
    }
    // assert(false);
    return nullptr;
}


void AppAnalysisNVBIT::kernel_grid_launch_id_transition() {
    size_t tensor_working_set_size = 0;
    for (auto ten : touched_tensors) {
        tensor_working_set_size += ten->size;
    }

    size_t memory_working_set_size = 0;
    for (auto mem : touched_memories) {
        memory_working_set_size += mem->size;
    }

    size_t memory_footprint_size = 0;
    auto active_memories_snapshot = active_memories_per_kernel_snapshot[previous_grid_launch_id];
    for (auto mem : active_memories_snapshot) {
        memory_footprint_size += mem.second->size;
    }

    size_t tensor_footprint_size = 0;
    auto active_tensors_snapshot = active_tensors_per_kernel_snapshot[previous_grid_launch_id];
    for (auto ten : active_tensors_snapshot) {
        tensor_footprint_size += ten.second->size;
    }

    kernel_stats[previous_grid_launch_id].tensor_working_set_size = tensor_working_set_size;
    kernel_stats[previous_grid_launch_id].memory_working_set_size = memory_working_set_size;
    kernel_stats[previous_grid_launch_id].memory_footprint_size = memory_footprint_size;
    kernel_stats[previous_grid_launch_id].tensor_footprint_size = tensor_footprint_size;
    kernel_stats[previous_grid_launch_id].kernel_launch->access_count = current_kernel_access_count;
    current_kernel_access_count = 0;
    touched_tensors.clear();
    touched_memories.clear();
}


void AppAnalysisNVBIT::kernel_end_callback(std::shared_ptr<KernelEnd_t> kernel) {

    _timer.increment(true);
}


void AppAnalysisNVBIT::mem_alloc_callback(std::shared_ptr<MemAlloc_t> mem) {
    mem_stats.alloc_count++;
    mem_stats.alloc_size += mem->size;
    mem_stats.max_size = std::max(mem_stats.max_size, mem_stats.alloc_size);
    active_memories.emplace(mem->addr, mem);

    _timer.increment(true);
}


void AppAnalysisNVBIT::mem_free_callback(std::shared_ptr<MemFree_t> mem) {
    mem_stats.free_count++;
    mem_stats.free_size += mem->size;
    mem_stats.alloc_size -= mem->size;

    auto it = active_memories.find(mem->addr);
    assert(it != active_memories.end());
    active_memories.erase(it);

    _timer.increment(true);
}



void AppAnalysisNVBIT::mem_cpy_callback(std::shared_ptr<MemCpy_t> mem) {

    _timer.increment(true);
}


void AppAnalysisNVBIT::mem_set_callback(std::shared_ptr<MemSet_t> mem) {

    _timer.increment(true);
}


void AppAnalysisNVBIT::ten_alloc_callback(std::shared_ptr<TenAlloc_t> ten) {
    ten_stats.alloc_count++;
    ten_stats.alloc_size += ten->size;
    ten_stats.max_size = std::max(ten_stats.max_size, ten_stats.alloc_size);

    active_tensors.emplace(ten->addr, ten);

    _timer.increment(true);
}


void AppAnalysisNVBIT::ten_free_callback(std::shared_ptr<TenFree_t> ten) {
    ten_stats.free_count++;
    ten_stats.free_size += -ten->size;
    ten_stats.alloc_size -= -ten->size;

    auto it = active_tensors.find(ten->addr);
    assert(it != active_tensors.end());
    active_tensors.erase(it);

    _timer.increment(true);
}


void AppAnalysisNVBIT::op_start_callback(std::shared_ptr<OpStart_t> op) {

    _timer.increment(true);
}


void AppAnalysisNVBIT::op_end_callback(std::shared_ptr<OpEnd_t> op) {

    _timer.increment(true);
}

void AppAnalysisNVBIT::gpu_data_analysis(void* data, uint64_t size) {
    nvbit_mem_access_t* ma = (nvbit_mem_access_t*)data;

    current_grid_launch_id = ma->grid_launch_id;
    if (current_grid_launch_id != previous_grid_launch_id) {
        kernel_grid_launch_id_transition();
        previous_grid_launch_id = current_grid_launch_id;
    }

    for (int i = 0; i < GPU_WARP_SIZE_NVBIT; i++) {
        if (ma->addrs[i] != 0) {
            current_kernel_access_count++;
            auto memory = query_memory_ranges_cpu(ma->addrs[i], current_grid_launch_id);
            if (memory != nullptr) {
                touched_memories.insert(memory);
            }
            auto tensor = query_tensor_ranges_cpu(ma->addrs[i], current_grid_launch_id);
            if (tensor != nullptr) {
                touched_tensors.insert(tensor);
            }
        }
    }
}


void AppAnalysisNVBIT::query_ranges(void* ranges, uint32_t limit, uint32_t* count) {
}

void AppAnalysisNVBIT::query_tensors(void* ranges, uint32_t limit, uint32_t* count) {
}


void AppAnalysisNVBIT::flush() {
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
