#include "tools/event_trace.h"
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

EventTrace::EventTrace() : Tool(EVENT_TRACE) {}

EventTrace::~EventTrace() {}

void EventTrace::evt_callback(EventPtr_t evt) {
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

void EventTrace::flush() {
    std::string mem_file_name = "memory_gpu.txt";
    std::ofstream mem_file(mem_file_name);
    for (auto &size : _memory_size_list) {
        mem_file << size << std::endl;
    }
    mem_file.close();

    std::string tensor_file_name = "tensor_gpu.txt";
    std::ofstream tensor_file(tensor_file_name);
    for (auto &size : _tensor_size_list) {
        tensor_file << size << std::endl;
    }
    tensor_file.close();
}

void EventTrace::init() {}

void EventTrace::kernel_start_callback(std::shared_ptr<KernelLaunch_t> kernel) {
}

void EventTrace::kernel_end_callback(std::shared_ptr<KernelEnd_t> kernel) {
}

void EventTrace::mem_alloc_callback(std::shared_ptr<MemAlloc_t> mem) {
    _active_memories.try_emplace(mem->addr, mem);

    _memory_size += mem->size;
    _memory_size_list.push_back(_memory_size);
}

void EventTrace::mem_free_callback(std::shared_ptr<MemFree_t> mem) {
    auto it = _active_memories.find(mem->addr);
    if (it == _active_memories.end()) {
        PRINT("[YOSEMITE INFO] Memory free callback: memory %lu not found. Active memories: %ld\n",
                mem->addr, _active_memories.size());
        // assert(false);
        return;
    }

    _memory_size -= it->second->size;
    _active_memories.erase(it);
    _memory_size_list.push_back(_memory_size);
}

void EventTrace::ten_alloc_callback(std::shared_ptr<TenAlloc_t> ten) {
    _tensor_size += ten->size;
    _tensor_size_list.push_back(_tensor_size);
}

void EventTrace::ten_free_callback(std::shared_ptr<TenFree_t> ten) {
    _tensor_size -= (-ten->size);
    _tensor_size_list.push_back(_tensor_size);
}

void EventTrace::mem_cpy_callback(std::shared_ptr<MemCpy_t> mem) {
}

void EventTrace::mem_set_callback(std::shared_ptr<MemSet_t> mem) {
}

void EventTrace::op_start_callback(std::shared_ptr<OpStart_t> op) {
}

void EventTrace::op_end_callback(std::shared_ptr<OpEnd_t> op) {
}
