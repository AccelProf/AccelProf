#include "tools/uvm_advisor.h"
#include "utils/helper.h"
#include "utils/hash.h"
#include "gpu_patch.h"
#include "cpp_trace.h"
#include "py_frame.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <string>
#include <iostream>


using namespace yosemite;

#define SANITIZER_UVM_MEMORY_FLAG 0x6
#define LARGE_TENSOR_THRESHOLD 1048576

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


UVMAdvisor::UVMAdvisor() : Tool(UVM_ADVISOR) {
    init();
}


UVMAdvisor::~UVMAdvisor() {
}

void UVMAdvisor::init() {
    const char* env_name = std::getenv("ACCEL_PROF_HOME");
    std::string lib_path;
    if (env_name) {
        lib_path = std::string(env_name) + "/lib/libcompute_sanitizer.so";
    }
    init_backtrace(lib_path.c_str());

}


void UVMAdvisor::evt_callback(EventPtr_t evt) {
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


void UVMAdvisor::kernel_start_callback(std::shared_ptr<KernelLaunch_t> kernel) {
    opt_keys.kernel_id ++;
    kernel->key = opt_keys.kernel_id;
    kernel->timestamp = _timer.get();
    kernel_events.push_back(kernel);
    op_stats.pending_kernels++;
    _timer.increment(true);
}


void UVMAdvisor::kernel_end_callback(std::shared_ptr<KernelEnd_t> kernel) {
    _timer.increment(true);
}


void UVMAdvisor::mem_alloc_callback(std::shared_ptr<MemAlloc_t> mem) {
    mem_stats.current_mem_size += mem->size;
    mem_stats.max_mem_size = std::max(mem_stats.max_mem_size, mem_stats.current_mem_size);
    if (mem->alloc_type != SANITIZER_UVM_MEMORY_FLAG) {
        return;
    }

    opt_keys.mem_id ++;
    mem->key = opt_keys.mem_id;
    mem->timestamp = _timer.get();
    op_stats.pending_mem_alloc++;
    mem_stats.alloc_count++;
    mem_stats.alloc_size += mem->size;
    alloc_events.emplace(_timer.get(), mem);
    active_memories.emplace(mem->addr, mem);

    mem_alloc_during_this_op.insert(mem->addr);

    _timer.increment(true);
}


void UVMAdvisor::mem_free_callback(std::shared_ptr<MemFree_t> mem) {
    mem_stats.current_mem_size -= mem->size;
    if (mem->alloc_type != SANITIZER_UVM_MEMORY_FLAG) {
        return;
    }

    mem_stats.free_count++;
    mem_stats.free_size += mem->size;

    auto it = active_memories.find(mem->addr);
    assert(it != active_memories.end());
    active_memories.erase(it);

    _timer.increment(true);
}



void UVMAdvisor::mem_cpy_callback(std::shared_ptr<MemCpy_t> mem) {
    _timer.increment(true);
}


void UVMAdvisor::mem_set_callback(std::shared_ptr<MemSet_t> mem) {
    _timer.increment(true);
}

bool UVMAdvisor::find_uvm_tensor(uint64_t ptr) {
    for (auto mem : active_memories) {
        if (ptr >= mem.first && ptr < mem.first + mem.second->size) {
            return true;
        }
    }
    return false;
}


void UVMAdvisor::ten_alloc_callback(std::shared_ptr<TenAlloc_t> ten) {
    ten_stats.current_ten_size += ten->size;
    ten_stats.max_ten_size = std::max(ten_stats.max_ten_size, ten_stats.current_ten_size);
    if (ten->size <= LARGE_TENSOR_THRESHOLD) {
        return;
    }
    opt_keys.ten_id ++;

    if (!find_uvm_tensor(ten->addr)) {
        return;
    }

    
    ten->key = opt_keys.ten_id;
    op_stats.pending_ten_alloc++;
    ten_stats.alloc_count++;
    ten_stats.alloc_size += ten->size;

    ten->timestamp = _timer.get();
    tenalloc_events.emplace(_timer.get(), ten);
    active_tensors.emplace(ten->addr, ten);

    ten_alloc_during_this_op.insert(ten->addr);

    _timer.increment(true);
}


void UVMAdvisor::ten_free_callback(std::shared_ptr<TenFree_t> ten) {
    ten_stats.current_ten_size -= -ten->size;
    if (-ten->size <= LARGE_TENSOR_THRESHOLD) {
        return;
    }

    if (active_tensors.find(ten->addr) == active_tensors.end()) {
        return;
    }

    ten_stats.free_count++;
    ten_stats.free_size += -ten->size;

    auto it = active_tensors.find(ten->addr);
    assert(it != active_tensors.end());
    active_tensors.erase(it);

    _timer.increment(true);
}


void UVMAdvisor::op_start_callback(std::shared_ptr<OpStart_t> op) {
    opt_keys.op_id ++;
    op->key = opt_keys.op_id;
    op->timestamp = _timer.get();
    op_stack.push(op);
    op_stats.count++;
    op_stats.pending_ops++;

    _timer.increment(true);
}


void UVMAdvisor::op_end_callback(std::shared_ptr<OpEnd_t> op) {
    auto op_start = op_stack.top();
    op_stack.pop();
    if (op_stack.empty()) {
        if (op_stats.pending_kernels > 0 && kernel_resources.size() > 0) {
            assert(op_tables.find(op_start->timestamp) == op_tables.end());
            op_start->end_time = _timer.get();
            op_start->pending_kernels = op_stats.pending_kernels;
            op_start->pending_ops = op_stats.pending_ops;
            op_start->pending_mem_alloc = op_stats.pending_mem_alloc;
            op_start->pending_ten_alloc = op_stats.pending_ten_alloc;
            op_tables[op_start->timestamp] = std::make_pair(op_start, kernel_resources);
        }
        op_stats.group_count++;
        op_stats.pending_kernels = 0;
        op_stats.pending_ops = 0;
        op_stats.pending_mem_alloc = 0;
        op_stats.pending_ten_alloc = 0;
        kernel_resources.clear();
        ten_alloc_during_this_op.clear();
        mem_alloc_during_this_op.clear();
    }

    _timer.increment(true);
}

void UVMAdvisor::gpu_data_analysis(void* data, uint64_t size) {
    MemoryAccessTracker* tracker = (MemoryAccessTracker*)data;
    MemoryAccessState* states = tracker->access_state;
    TensorAccessState* tensor_states = tracker->tensor_access_state;

    MemAllocVec mem_alloc_vec;
    TenAllocVec ten_alloc_vec;

    for (uint32_t i = 0; i < states->size; i++) {
        if (states->touch[i] == 1) {
            auto mem = active_memories.find(states->start_end[i].start);
            // not allocated during this op
            if (mem_alloc_during_this_op.find(mem->second->addr) == mem_alloc_during_this_op.end()) {
                mem_alloc_vec.push_back(mem->second);
            }
        }
    }

    for (uint32_t i = 0; i < tensor_states->size; i++) {
        if (tensor_states->touch[i] == 1) {
            auto ten = active_tensors.find(tensor_states->start_end[i].start);
            // not allocated during this op
            if (ten_alloc_during_this_op.find(ten->second->addr) == ten_alloc_during_this_op.end()) {
                ten_alloc_vec.push_back(ten->second);
            }
        }
    }

    if (mem_alloc_vec.empty() && ten_alloc_vec.empty()) {
        return;
    }

    auto kernel = kernel_events.back();
    kernel_resources.push_back(std::make_tuple(kernel, mem_alloc_vec, ten_alloc_vec));
}


void UVMAdvisor::query_ranges(void* ranges, uint32_t limit, uint32_t* count) {
    MemoryRange* _ranges = (MemoryRange*)ranges;
    *count = 0;
    for (auto mem : active_memories) {
        _ranges[*count].start = mem.second->addr;
        _ranges[*count].end = mem.second->addr + mem.second->size;
        (*count)++;
        if (*count >= limit) {
            fprintf(stdout, "Warning: query_ranges limit reached\n");
            break;
        }
    }
}

void UVMAdvisor::query_tensors(void* ranges, uint32_t limit, uint32_t* count) {
    MemoryRange* _ranges = (MemoryRange*)ranges;
    *count = 0;
    for (auto ten : active_tensors) {
        _ranges[*count].start = ten.second->addr;
        _ranges[*count].end = ten.second->addr + ten.second->size;
        (*count)++;
        if (*count >= limit) {
            fprintf(stdout, "Warning: query_tensors limit reached\n");
            break;
        }
    }
}



void UVMAdvisor::print_callstack() {
    auto backtraces = get_backtrace();
    auto py_frames = get_pyframes();
    auto bt_str = vector2str(backtraces);
    auto pf_str = vector2str(py_frames);
    std::cout << bt_str << std::endl;
    std::cout << pf_str << std::endl;
}


void UVMAdvisor::flush() {
    FILE* out;
    std::string file_name = "uvm_advisor_opt.log";
    out = fopen(file_name.c_str(), "w");
    
    fprintf(out, "--------------------------------------------------------------------------------\n");
    fprintf(out, "%-12s max_size: %lu (%s)\n", 
            "[Memory]", mem_stats.max_mem_size, format_size(mem_stats.max_mem_size).c_str());
    fprintf(out, "%-12s count: %-10lu, size: %lu (%s)\n", 
            "[MemMalloc]", mem_stats.alloc_count, mem_stats.alloc_size, format_size(mem_stats.alloc_size).c_str());
    fprintf(out, "%-12s count: %-10lu, size: %lu (%s)\n", 
            "[MemFree]", mem_stats.free_count, mem_stats.free_size, format_size(mem_stats.free_size).c_str());

    fprintf(out, "%-12s max_size: %lu (%s)\n", 
            "[Tensor]", ten_stats.max_ten_size, format_size(ten_stats.max_ten_size).c_str());
    fprintf(out, "%-12s count: %-10lu, size: %lu (%s)\n", 
            "[TenMalloc]", ten_stats.alloc_count, ten_stats.alloc_size, format_size(ten_stats.alloc_size).c_str());
    fprintf(out, "%-12s count: %-10lu, size: %lu (%s)\n", 
            "[TenFree]", ten_stats.free_count, ten_stats.free_size, format_size(ten_stats.free_size).c_str());
    fprintf(out, "%-12s count: %-10lu\n", "[Op]", op_stats.count);
    fprintf(out, "%-12s count: %-10lu\n", "[OpGroup]", op_stats.group_count);
    fprintf(out, "--------------------------------------------------------------------------------\n");

    fprintf(out, "================================================================================\n");
    for (auto& it : op_tables) {
        auto op = it.second.first;
        fprintf(out, "Op - %.30s, op_id: %lu, pending_ops: %lu, pending_kernels: %lu, pending_mem_alloc: %lu, pending_ten_alloc: %lu\n", 
                op->op_name.c_str(), op->key, op->pending_ops, op->pending_kernels, op->pending_mem_alloc, op->pending_ten_alloc);
        for (auto& kernel_tuple : it.second.second) {
            auto kernel = std::get<0>(kernel_tuple);
            auto mem_alloc_vec = std::get<1>(kernel_tuple);
            auto ten_alloc_vec = std::get<2>(kernel_tuple);
            fprintf(out, "   Kernel: %.30s, kernel_id: %lu\n", kernel->kernel_name.c_str(), kernel->key);
            fprintf(out, "       MemAlloc (%lu): ", mem_alloc_vec.size());
            for (auto& mem : mem_alloc_vec) {
                fprintf(out, "%lu:(%lu, %lu), ", mem->key, mem->addr, mem->size);
            }
            fprintf(out, "\n");
            fprintf(out, "       TenAlloc (%lu): ", ten_alloc_vec.size());
            for (auto& ten : ten_alloc_vec) {
                fprintf(out, "%lu:(%lu, %lu), ", ten->key, ten->addr, ten->size);
            }
            fprintf(out, "\n");
        }
    }
    fclose(out);
}
