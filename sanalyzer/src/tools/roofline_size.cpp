#include "tools/roofline_size.h"
#include "utils/helper.h"
#include "gpu_patch.h"
#include <fstream>
using namespace yosemite;

void RooflineSize::evt_callback(EventPtr_t evt) {
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
        default:
            break;
    }
}


void RooflineSize::kernel_start_callback(std::shared_ptr<KernelLaunch_t> kernel) {
    accessCount = 0;
    accessSize = 0;
    
}

void RooflineSize::kernel_end_callback(std::shared_ptr<KernelEnd_t> kernel) {
    kernel_size_map.try_emplace(kernel, std::make_pair(accessCount, accessSize));
}

void RooflineSize::mem_alloc_callback(std::shared_ptr<MemAlloc_t> mem) {

}

void RooflineSize::mem_free_callback(std::shared_ptr<MemFree_t> mem) {

}

void RooflineSize::gpu_data_analysis(void* data, uint64_t size) {
    MemoryAccessTracker* tracker = (MemoryAccessTracker*)data;
    accessCount = tracker->accessCount;
    accessSize = tracker->accessSize;
}

void RooflineSize::query_ranges(void* ranges, uint32_t limit, uint32_t* count) {

}

void RooflineSize::flush() {
    std::string file_name = "./out/roofline_size.txt";
    std::ofstream out(file_name);
    for (auto pair : kernel_size_map) {
        out << std::get<0>(pair.second) << "|" << std::get<1>(pair.second) << "|" << pair.first->kernel_name << std::endl;
    }
    out.close();
}
