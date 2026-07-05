#include "tools/event_trace_mgpu.h"
#include <cassert>
#include <fstream>
#include <string>

using namespace yosemite;

#define YOSEMITE_VERBOSE 1

#if YOSEMITE_VERBOSE
#define PRINT(...) do { fprintf(stdout, __VA_ARGS__); fflush(stdout); } while (0)
#else
#define PRINT(...)
#endif

EventTraceMGPU::EventTraceMGPU() : Tool(EVENT_TRACE_MGPU) {}

EventTraceMGPU::~EventTraceMGPU() {}

void EventTraceMGPU::evt_callback(EventPtr_t evt) {
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

void EventTraceMGPU::flush() {
    // dump memory size
    std::string mem_file_name = "memory_gpu";
    for (auto &[device_id, size_list] : _memory_size_list) {
        PRINT("[YOSEMITE INFO] Memory size on device %d: ", device_id);
        std::string mem_file_name = "memory_gpu_" + std::to_string(device_id) + ".txt";
        std::ofstream mem_file(mem_file_name);
        for (auto &size : size_list) {
            mem_file << size << std::endl;
        }
        mem_file.close();
    }
    // dump tensor size list
    for (auto &[device_id, size_list] : _tensor_size_list) {
        std::string tensor_file_name = "tensor_gpu_" + std::to_string(device_id) + ".txt";
        std::ofstream tensor_file(tensor_file_name);
        for (auto &size : size_list) {
            tensor_file << size << std::endl;
        }
        tensor_file.close();
    }
}

void EventTraceMGPU::init() {}

void EventTraceMGPU::kernel_start_callback(std::shared_ptr<KernelLaunch_t> kernel) {
}

void EventTraceMGPU::kernel_end_callback(std::shared_ptr<KernelEnd_t> kernel) {
}

void EventTraceMGPU::mem_alloc_callback(std::shared_ptr<MemAlloc_t> mem) {
    auto device_id = mem->device_id;
    auto it = _memory_size.find(device_id);
    if (it == _memory_size.end()) {
        _memory_size[device_id] = 0;
    }
    _memory_size[device_id] += mem->size;

    auto it2 = _memory_size_list.find(device_id);
    if (it2 == _memory_size_list.end()) {
        _memory_size_list[device_id] = std::vector<int64_t>();
    }
    _memory_size_list[device_id].push_back(_memory_size[device_id]);
}

void EventTraceMGPU::mem_free_callback(std::shared_ptr<MemFree_t> mem) {
    // compute sanitizer pass meaningful memory size
    auto device_id = mem->device_id;
    assert(_memory_size.find(device_id) != _memory_size.end());

    _memory_size[device_id] -= mem->size;
    _memory_size_list[device_id].push_back(_memory_size[device_id]);
}

void EventTraceMGPU::ten_alloc_callback(std::shared_ptr<TenAlloc_t> ten) {
    auto device_id = ten->device_id;

    auto it = _tensor_size.find(device_id);
    if (it == _tensor_size.end()) {
        _tensor_size[device_id] = 0;
    }
    _tensor_size[device_id] += ten->size;

    auto it2 = _tensor_size_list.find(device_id);
    if (it2 == _tensor_size_list.end()) {
        _tensor_size_list[device_id] = std::vector<int64_t>();
    }
    _tensor_size_list[device_id].push_back(_tensor_size[device_id]);

    _memory_size_list[device_id].push_back(_memory_size[device_id]);
}

void EventTraceMGPU::ten_free_callback(std::shared_ptr<TenFree_t> ten) {
    auto device_id = ten->device_id;
    assert(_tensor_size.find(device_id) != _tensor_size.end());

    _tensor_size[device_id] -= (-ten->size);
    _tensor_size_list[device_id].push_back(_tensor_size[device_id]);

    _memory_size_list[device_id].push_back(_memory_size[device_id]);
}

void EventTraceMGPU::mem_cpy_callback(std::shared_ptr<MemCpy_t> mem) {
}

void EventTraceMGPU::mem_set_callback(std::shared_ptr<MemSet_t> mem) {
}

void EventTraceMGPU::op_start_callback(std::shared_ptr<OpStart_t> op) {
}

void EventTraceMGPU::op_end_callback(std::shared_ptr<OpEnd_t> op) {
}
