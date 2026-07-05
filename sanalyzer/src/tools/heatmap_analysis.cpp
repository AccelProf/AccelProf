#include "tools/heatmap_analysis.h"
#include "utils/helper.h"

#include <cstdint>
#include <fstream>
#include <memory>
#include <cassert>
#include <iostream>
#include <bitset>
#include <sstream>
#include <algorithm>
#include <vector>


using namespace yosemite;


HeatmapAnalysis::HeatmapAnalysis() : Tool(HEATMAP_ANALYSIS) {
    const char* torch_prof = std::getenv("TORCH_PROFILE_ENABLED");
    if (torch_prof && std::string(torch_prof) == "1") {
        fprintf(stdout, "Enabling torch profiler in HeatmapAnalysis.\n");
        _torch_enabled = true;
    }

    const char* env_app_name = std::getenv("YOSEMITE_APP_NAME");
    if (env_app_name != nullptr) {
        output_directory = "heatmap_" + std::string(env_app_name)
                            + "_" + get_current_date_n_time();
    } else {
        output_directory = "heatmap_" + get_current_date_n_time();
    }
    check_folder_existance(output_directory);
}


HeatmapAnalysis::~HeatmapAnalysis() {}


void HeatmapAnalysis::kernel_start_callback(std::shared_ptr<KernelLaunch_t> kernel) {

    kernel->kernel_id = kernel_id++;
    kernel_events.emplace(_timer.get(), kernel);
    _traces.clear();
    _heatmap_data.clear();
    _sector_pc_information.clear();

    _timer.increment(true);
}


void HeatmapAnalysis::kernel_trace_flush(std::shared_ptr<KernelLaunch_t> kernel) {
    std::string filename = output_directory + "/kernel_"
                            + std::to_string(kernel->kernel_id) + ".csv";
    printf("Dumping block 0 heatmap to %s\n", filename.c_str());

    std::ofstream out(filename);
    std::stringstream ss;

    std::vector<std::pair<uint64_t, std::array<uint32_t, 18>>> sorted_heatmap_data(_heatmap_data.begin(), _heatmap_data.end());
    std::sort(
        sorted_heatmap_data.begin(),
        sorted_heatmap_data.end(),
        [](const std::pair<uint64_t, std::array<uint32_t, 18>>& a, const std::pair<uint64_t, std::array<uint32_t, 18>>& b)
    {
        return a.first < b.first;
    });
    ss << "Sector Tag,\t\tDistinct Warp Count,\tAccess Count,\t\t\tTouched PC" << std::endl;
    for (auto& [tag, data] : sorted_heatmap_data) {
        ss << "0x"<<std::hex << tag << std::dec << ",\t\t";
        for (int i = 0; i < 9; i++) {
            ss << std::bitset<32>(data[i]).count() << ",";
        }
        ss << "\t\t";
        for (int i = 9; i < 18; i++) {
            ss << data[i] << ",";
        }
        for (auto pc : _sector_pc_information[tag]) {
            ss << "\t\t0x" << std::hex << pc << std::dec << ",";
        }
        ss << std::endl;
    }

    out << ss.str();

    out.close();
}


void HeatmapAnalysis::kernel_end_callback(std::shared_ptr<KernelEnd_t> kernel) {
    auto evt = std::prev(kernel_events.end())->second;
    evt->end_time = _timer.get();

    kernel_trace_flush(evt);

    _timer.increment(true);
}


void HeatmapAnalysis::mem_alloc_callback(std::shared_ptr<MemAlloc_t> mem) {
    alloc_events.emplace(_timer.get(), mem);
    active_memories.emplace(mem->addr, mem);

    _timer.increment(true);
}


void HeatmapAnalysis::mem_free_callback(std::shared_ptr<MemFree_t> mem) {
    auto it = active_memories.find(mem->addr);
    assert(it != active_memories.end());
    active_memories.erase(it);

    _timer.increment(true);
}


void HeatmapAnalysis::ten_alloc_callback(std::shared_ptr<TenAlloc_t> ten) {
    tensor_events.emplace(_timer.get(), ten);
    active_tensors.emplace(ten->addr, ten);

    _timer.increment(true);
}


void HeatmapAnalysis::ten_free_callback(std::shared_ptr<TenFree_t> ten) {
    auto it = active_tensors.find(ten->addr);
    assert(it != active_tensors.end());
    active_tensors.erase(it);

    _timer.increment(true);
}

// function signature:
// addr: the address of the memory access
// warp_id: the warp id of the memory access
// sector_tag: the sector tag of the memory access
// offset: the offset of the memory access
// length: the length of the memory access
// count_access_flag: whether to count the access flag
// return: void
void HeatmapAnalysis::unit_access(uint32_t warp_id, uint64_t sector_tag, uint32_t offset, uint32_t length) {
    
    // heatmap_data[tag][0-7]: distinct warp id mask for each word in this sector;
    // heatmap_data[tag][8]: distinct warp id mask for entire sector;
    // heatmap_data[tag][9-17]: access count for each word and the last is for entire sector;
    // // if count_access_flag is true, then the access count for the entire sector is incremented by 1;
    auto& sector_data = _heatmap_data[sector_tag];
    auto mask = (1u << warp_id);
    for (int i = 0; i < length; i+=4) {
        sector_data[offset+i/4] |= mask;
        sector_data[8] |= mask;
        sector_data[9+offset+i/4] += 1;
    }
    sector_data[17] += 1;
}

void HeatmapAnalysis::add_sector_pc_information(uint64_t sector_tag, uint64_t pc) {
    _sector_pc_information[sector_tag].insert(pc);
}


void HeatmapAnalysis::gpu_data_analysis(void* data, uint64_t size) {
    MemoryAccess* accesses_buffer = (MemoryAccess*)data;
    for (uint64_t i = 0; i < size; i++) {
        auto trace = accesses_buffer[i];
        for (int j = 0; j < GPU_WARP_SIZE; j++) {
            if (trace.active_mask & (1u << j)) {
                auto sector_tag = trace.addresses[j] >> SECTOR_TAG_SHIFT;
                auto offset = (trace.addresses[j] & 31) >> 2;
                unit_access(trace.warpId, sector_tag, offset, trace.accessSize);
                add_sector_pc_information(sector_tag, trace.pc);
            }
        }
    } 
}

void HeatmapAnalysis::evt_callback(EventPtr_t evt) {
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


void HeatmapAnalysis::flush() {
}
