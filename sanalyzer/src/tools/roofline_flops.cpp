#include "tools/roofline_flops.h"
#include "nvbit_common.h"
#include <fstream>
using namespace yosemite;


void RooflineFlops::evt_callback(EventPtr_t evt) {
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


void RooflineFlops::kernel_start_callback(std::shared_ptr<KernelLaunch_t> kernel) {
    total_flops = 0;
}

void RooflineFlops::kernel_end_callback(std::shared_ptr<KernelEnd_t> kernel) {
    kernel_flops_map.try_emplace(kernel, total_flops);
}

void RooflineFlops::mem_alloc_callback(std::shared_ptr<MemAlloc_t> mem) {
    
}

void RooflineFlops::mem_free_callback(std::shared_ptr<MemFree_t> mem) {

}

void RooflineFlops::gpu_data_analysis(void* data, uint64_t size) {
    total_flops = size;
}

void RooflineFlops::query_ranges(void* ranges, uint32_t limit, uint32_t* count) {

}

void RooflineFlops::flush() {
    std::string file_name = "./out/roofline_flops.txt";
    std::ofstream out(file_name);
    for (auto pair : kernel_flops_map) {
        out << pair.second << "|" << pair.first->kernel_name << std::endl;
    }
    out.close();

}
