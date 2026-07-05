#include "tools/block_divergence_analysis.h"
#include "utils/helper.h"

#include <cstdint>
#include <fstream>
#include <memory>
#include <cassert>
#include <algorithm>
#include <iomanip>
#include <vector>

#ifdef __has_include
#if __has_include(<sanitizer_patching.h>)
#include <sanitizer_patching.h>
#endif
#endif

#ifndef SANITIZER_MEMORY_DEVICE_FLAG_READ
#define SANITIZER_MEMORY_DEVICE_FLAG_READ 0x1
#endif

#ifndef SANITIZER_MEMORY_DEVICE_FLAG_WRITE
#define SANITIZER_MEMORY_DEVICE_FLAG_WRITE 0x2
#endif

using namespace yosemite;


BlockDivergenceAnalysis::BlockDivergenceAnalysis() : Tool(MEM_TRACE) {
    const char* torch_prof = std::getenv("TORCH_PROFILE_ENABLED");
    if (torch_prof && std::string(torch_prof) == "1") {
        fprintf(stdout, "Enabling torch profiler in BlockDivergenceAnalysis.\n");
        _torch_enabled = true;
    }

    const char* env_app_name = std::getenv("YOSEMITE_APP_NAME");
    if (env_app_name != nullptr) {
        output_directory = "block_distribution_" + std::string(env_app_name)
                            + "_" + get_current_date_n_time();
    } else {
        output_directory = "block_distribution_" + get_current_date_n_time();
    }
    check_folder_existance(output_directory);
}


BlockDivergenceAnalysis::~BlockDivergenceAnalysis() {}


void BlockDivergenceAnalysis::kernel_start_callback(std::shared_ptr<KernelLaunch_t> kernel) {

    kernel->kernel_id = kernel_id++;
    kernel_events.emplace(_timer.get(), kernel);
    _block_entries.clear();
    _unique_pcs.clear();

    _timer.increment(true);
}


void BlockDivergenceAnalysis::kernel_trace_flush(std::shared_ptr<KernelLaunch_t> kernel) {
    std::string filename = output_directory + "/kernel_"
                            + std::to_string(kernel->kernel_id) + ".csv";
    printf("Dumping traces to %s\n", filename.c_str());

    std::ofstream out(filename);
    std::vector<uint64_t> pc_list(_unique_pcs.begin(), _unique_pcs.end());
    std::sort(pc_list.begin(), pc_list.end());

    std::vector<uint64_t> block_ids;
    block_ids.reserve(_block_entries.size());
    for (const auto& entry : _block_entries) {
        block_ids.push_back(entry.first);
    }
    std::sort(block_ids.begin(), block_ids.end());

    out << "blockidx,blockidy,blockidz";
    for (const auto pc : pc_list) {
        out << ",0x" << std::hex << std::setw(16) << std::setfill('0') << pc << std::dec;
    }
    out << ",read_count,write_count" << std::endl;

    for (const auto block_id : block_ids) {
        const auto& stats = _block_entries.at(block_id);
        out << block_id << ",0,0";
        for (const auto pc : pc_list) {
            auto pc_it = stats.pc_counts.find(pc);
            uint64_t count = (pc_it != stats.pc_counts.end()) ? pc_it->second : 0;
            out << "," << count;
        }
        out << "," << stats.read_count << "," << stats.write_count << std::endl;
    }
}


void BlockDivergenceAnalysis::kernel_end_callback(std::shared_ptr<KernelEnd_t> kernel) {
    auto evt = std::prev(kernel_events.end())->second;
    evt->end_time = _timer.get();

    kernel_trace_flush(evt);

    _timer.increment(true);
}


void BlockDivergenceAnalysis::mem_alloc_callback(std::shared_ptr<MemAlloc_t> mem) {
    alloc_events.emplace(_timer.get(), mem);
    active_memories.emplace(mem->addr, mem);

    _timer.increment(true);
}


void BlockDivergenceAnalysis::mem_free_callback(std::shared_ptr<MemFree_t> mem) {
    auto it = active_memories.find(mem->addr);
    assert(it != active_memories.end());
    active_memories.erase(it);

    _timer.increment(true);
}


void BlockDivergenceAnalysis::ten_alloc_callback(std::shared_ptr<TenAlloc_t> ten) {
    tensor_events.emplace(_timer.get(), ten);
    active_tensors.emplace(ten->addr, ten);

    _timer.increment(true);
}


void BlockDivergenceAnalysis::ten_free_callback(std::shared_ptr<TenFree_t> ten) {
    auto it = active_tensors.find(ten->addr);
    assert(it != active_tensors.end());
    active_tensors.erase(it);

    _timer.increment(true);
}


void BlockDivergenceAnalysis::gpu_data_analysis(void* data, uint64_t size) {
    MemoryAccess* accesses_buffer = (MemoryAccess*)data;
    for (uint64_t i = 0; i < size; i++) {
        const MemoryAccess& trace = accesses_buffer[i];
        uint64_t executed_inst_count = static_cast<uint64_t>(__builtin_popcount(trace.active_mask));
        uint64_t pc = trace.pc;
        uint64_t cta_id = trace.ctaId;

        auto& entry = _block_entries[cta_id];
        entry.pc_counts[pc] += executed_inst_count;
        if (trace.flags & SANITIZER_MEMORY_DEVICE_FLAG_READ) {
            entry.read_count += executed_inst_count;
        }
        if (trace.flags & SANITIZER_MEMORY_DEVICE_FLAG_WRITE) {
            entry.write_count += executed_inst_count;
        }

        _unique_pcs.insert(pc);
    }

}


void BlockDivergenceAnalysis::evt_callback(EventPtr_t evt) {
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


void BlockDivergenceAnalysis::flush() {
}
