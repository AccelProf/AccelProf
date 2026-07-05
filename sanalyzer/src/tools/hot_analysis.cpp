/**
 * Hot memory analysis.
 * For PyTorch models.
 */
#include "tools/hot_analysis.h"
#include <cstring>
#include "utils/helper.h"


#include <vector>
#include <cassert>
#include <cstring>
#include <fstream>


using namespace yosemite;

constexpr uint32_t RANGE_GRANULARITY = 2 * 1024 * 1024;


HotAnalysis::HotAnalysis() : Tool(HOT_ANALYSIS) {
    const char* env_app_name = std::getenv("YOSEMITE_APP_NAME");
    if (env_app_name != nullptr) {
        output_directory = "hotness_" + std::string(env_app_name)
                            + "_" + get_current_date_n_time();
    } else {
        output_directory = "hotness_" + get_current_date_n_time();
    }
    check_folder_existance(output_directory);
}

HotAnalysis::~HotAnalysis() {
}

void HotAnalysis::kernel_start_callback(std::shared_ptr<KernelLaunch_t> kernel) {
}

void HotAnalysis::kernel_end_callback(std::shared_ptr<KernelEnd_t> kernel) {
}

void HotAnalysis::mem_alloc_callback(std::shared_ptr<MemAlloc_t> mem) {
    active_memories.emplace(mem->addr, mem);
}

void HotAnalysis::mem_free_callback(std::shared_ptr<MemFree_t> mem) {
    active_memories.erase(mem->addr);
}

void HotAnalysis::mem_cpy_callback(std::shared_ptr<MemCpy_t> mem) {
}

void HotAnalysis::mem_set_callback(std::shared_ptr<MemSet_t> mem) {
}

void HotAnalysis::ten_alloc_callback(std::shared_ptr<TenAlloc_t> ten) {
    active_tensors.emplace(ten->addr, ten);
}

void HotAnalysis::ten_free_callback(std::shared_ptr<TenFree_t> ten) {
    active_tensors.erase(ten->addr);
}

void HotAnalysis::evt_callback(EventPtr_t evt) {
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
        default:
            break;
    }
}

void HotAnalysis::gpu_data_analysis(void* data, uint64_t size) {
    MemoryAccessState* state = (MemoryAccessState*)data;

    std::string filename = output_directory + "/kernel_"
                            + std::to_string(global_kernel_id++) + ".txt";
    printf("Dumping traces to %s\n", filename.c_str());

    std::ofstream out(filename);

    auto tensor_iter = active_tensors.begin();

    for (uint32_t i = 0; i < size; ++i) {
        MemoryRange range = state->start_end[i];

        if (tensor_iter != active_tensors.end()) {
            if (tensor_iter->first == range.start) {
                out << "Tensor start ------------------------------------------" << tensor_iter->first << std::endl;
            }
        }

        out << range.start << " " << range.end << " " << state->touch[i] << std::endl;

        if (tensor_iter != active_tensors.end()) {
            if (tensor_iter->first + tensor_iter->second->size == range.end) {
                out << "Tensor end ------------------------------------------" << tensor_iter->first + tensor_iter->second->size << std::endl;
                tensor_iter++;
            }
        }

        auto it = range_access_counts.find(range);
        if (it != range_access_counts.end()) {
            it->second += state->touch[i];
        } else {
            range_access_counts.emplace(range, state->touch[i]);
        }
    }
    out << std::endl;

    for (auto active_mem : active_memories) {
        auto mem = active_mem.second;
        out << mem->addr << " " << mem->size << std::endl;
    }
    out << std::endl;

    for (auto active_ten : active_tensors) {
        auto ten = active_ten.second;
        out << ten->addr << " " << ten->size << std::endl;
    }

    out.close();
}

void HotAnalysis::query_ranges(void* ranges, uint32_t limit, uint32_t* count) {
    std::vector<MemoryRange> active_memory_ranges;
    for (auto active_mem : active_memories) {
        auto mem = active_mem.second;
        if (mem->size <= RANGE_GRANULARITY) {
            MemoryRange range = {mem->addr, mem->addr + mem->size};
            active_memory_ranges.push_back(range);
        } else {
            uint64_t start = mem->addr;
            uint64_t end = mem->addr + mem->size;
            while (start < end) {
                if (start + RANGE_GRANULARITY > end) {
                    MemoryRange range = {start, end};
                    active_memory_ranges.push_back(range);
                } else {
                    MemoryRange range = {start, start + RANGE_GRANULARITY};
                    active_memory_ranges.push_back(range);
                }
                start += RANGE_GRANULARITY;
            }
        }
    }
    fprintf(stdout, "size: %lu, limit: %u\n", active_memory_ranges.size(), MAX_NUM_MEMORY_RANGES);
    fflush(stdout);
    assert(active_memory_ranges.size() <= MAX_NUM_MEMORY_RANGES);

    MemoryRange* temp_ranges = active_memory_ranges.data();
    memcpy(ranges, temp_ranges, sizeof(MemoryRange) * active_memory_ranges.size());
    *count = active_memory_ranges.size();
}

void HotAnalysis::flush() {
    std::string filename = output_directory + "/all_kernels.txt";
    printf("Dumping traces to %s\n", filename.c_str());

    std::ofstream out(filename);

    for (auto range : range_access_counts) {
        out << range.first.start << " " << range.first.end << " " << range.second << std::endl;
    }

    out.close();
}
