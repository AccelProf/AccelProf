#include "tools/mem_trace.h"
#include "utils/helper.h"

#include <cstdint>
#include <fstream>
#include <memory>
#include <cassert>
#include <iostream>


using namespace yosemite;


MemTrace::MemTrace() : Tool(MEM_TRACE) {
    const char* torch_prof = std::getenv("TORCH_PROFILE_ENABLED");
    if (torch_prof && std::string(torch_prof) == "1") {
        fprintf(stdout, "Enabling torch profiler in MemTrace.\n");
        _torch_enabled = true;
    }

    const char* env_app_name = std::getenv("YOSEMITE_APP_NAME");
    if (env_app_name != nullptr) {
        output_directory = "traces_" + std::string(env_app_name)
                            + "_" + get_current_date_n_time();
    } else {
        output_directory = "traces_" + get_current_date_n_time();
    }
    check_folder_existance(output_directory);
}


MemTrace::~MemTrace() {}


void MemTrace::kernel_start_callback(std::shared_ptr<KernelLaunch_t> kernel) {

    kernel->kernel_id = kernel_id++;
    kernel_events.emplace(_timer.get(), kernel);
    _traces.clear();

    _timer.increment(true);
}


void MemTrace::kernel_trace_flush(std::shared_ptr<KernelLaunch_t> kernel) {
    std::string filename = output_directory + "/kernel_"
                            + std::to_string(kernel->kernel_id) + ".txt";
    printf("Dumping traces to %s\n", filename.c_str());

    std::ofstream out(filename);

    for (auto& trace : _traces) {
        for (int i = 0; i < GPU_WARP_SIZE; i++) {
            if (trace.addresses[i] != 0) {
                _timer.increment(false);
                out << (trace.addresses[i] >> 12) << " "
                    << trace.addresses[i] << " "
                    << trace.accessSize << " "
                    << _timer.get() << " "
                    << trace.flags << " "
                    << trace.warpId << std::endl;
            }
        }
    }

    out << std::endl;
    for (auto evt : active_memories) {
        out << "ALLOCATION: " << " " << evt.second->addr
            << " " << evt.second->size << std::endl;
    }

    out << std::endl;
    for (auto evt : active_tensors) {
        out << "TENSOR: " << " " << evt.second->addr
            << " " << evt.second->size << std::endl;
    }

    out << std::endl;
    out << "KERNEL: " << kernel->timestamp << " " << kernel->end_time << std::endl;

    out.close();
}


void MemTrace::kernel_end_callback(std::shared_ptr<KernelEnd_t> kernel) {
    auto evt = std::prev(kernel_events.end())->second;
    evt->end_time = _timer.get();

    kernel_trace_flush(evt);

    _timer.increment(true);
}


void MemTrace::mem_alloc_callback(std::shared_ptr<MemAlloc_t> mem) {
    alloc_events.emplace(_timer.get(), mem);
    active_memories.emplace(mem->addr, mem);

    _timer.increment(true);
}


void MemTrace::mem_free_callback(std::shared_ptr<MemFree_t> mem) {
    auto it = active_memories.find(mem->addr);
    assert(it != active_memories.end());
    active_memories.erase(it);

    _timer.increment(true);
}


void MemTrace::ten_alloc_callback(std::shared_ptr<TenAlloc_t> ten) {
    tensor_events.emplace(_timer.get(), ten);
    active_tensors.emplace(ten->addr, ten);

    _timer.increment(true);
}


void MemTrace::ten_free_callback(std::shared_ptr<TenFree_t> ten) {
    auto it = active_tensors.find(ten->addr);
    assert(it != active_tensors.end());
    active_tensors.erase(it);

    _timer.increment(true);
}


void MemTrace::gpu_data_analysis(void* data, uint64_t size) {
    MemoryAccess* accesses_buffer = (MemoryAccess*)data;
    for (int i = 0; i < size; i++) {
        MemoryAccess trace = accesses_buffer[i];
        _traces.push_back(trace);
    }

}


void MemTrace::evt_callback(EventPtr_t evt) {
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


void MemTrace::flush() {
}
