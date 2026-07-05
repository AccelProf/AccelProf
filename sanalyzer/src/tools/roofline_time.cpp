#include "tools/roofline_time.h"
#include <fstream>
#include <chrono>
#include <iomanip> 
using namespace yosemite;


void RooflineTime::evt_callback(EventPtr_t evt) {
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


void RooflineTime::kernel_start_callback(std::shared_ptr<KernelLaunch_t> kernel) {
    start_time = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

}

void RooflineTime::kernel_end_callback(std::shared_ptr<KernelEnd_t> kernel) {
    double end_time = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    kernel_time_map.try_emplace(kernel, end_time - start_time);
}

void RooflineTime::mem_alloc_callback(std::shared_ptr<MemAlloc_t> mem) {
    cur_mem_usage += mem->size;
    if (cur_mem_usage > max_mem_usage) {
        max_mem_usage = cur_mem_usage;
    }
}

void RooflineTime::mem_free_callback(std::shared_ptr<MemFree_t> mem) {
    cur_mem_usage -= mem->size;
    
}

void RooflineTime::ten_alloc_callback(std::shared_ptr<TenAlloc_t> ten) {
    cur_ten_usage += ten->size;
    if (cur_ten_usage > max_ten_usage) {
        max_ten_usage = cur_ten_usage;
    }
}

void RooflineTime::ten_free_callback(std::shared_ptr<TenFree_t> ten) {
    // tensor free size is negative
    cur_ten_usage += ten->size;
}
void RooflineTime::gpu_data_analysis(void* data, uint64_t size) {

}

void RooflineTime::query_ranges(void* ranges, uint32_t limit, uint32_t* count) {

}

void RooflineTime::flush() {
    printf("Max_mem_usage: %.2f MiB\n", (double)max_mem_usage / 1024.0 / 1024.0);
    printf("Max_ten_usage: %.2f MiB\n", (double)max_ten_usage / 1024.0 / 1024.0);
    std::string file_name = "./out/roofline_time.txt";
    std::ofstream out(file_name);
    out << std::fixed << std::setprecision(6);
    for (auto pair : kernel_time_map) {
        out << pair.second << "|" << pair.first->kernel_name << std::endl;
    }
    out.close();

}
