
#include "tools/app_metric.h"
#include "utils/helper.h"
#include "gpu_patch.h"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <vector>
#include <string>
#include <memory>


using namespace yosemite;


void AppMetrics::evt_callback(EventPtr_t evt) {
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


void AppMetrics::kernel_start_callback(std::shared_ptr<KernelLaunch_t> kernel) {
    kernel->kernel_id = _kernel_id++;
    kernel_events.emplace(_timer.get(), kernel);
    if (kernel_invocations.find(kernel->kernel_name) == kernel_invocations.end()) {
        kernel_invocations.emplace(kernel->kernel_name, 1);
    } else {
        kernel_invocations[kernel->kernel_name]++;
    }

    _stats.num_kernels++;

    _timer.increment(true);
}


void AppMetrics::kernel_end_callback(std::shared_ptr<KernelEnd_t> kernel) {
}


void AppMetrics::mem_alloc_callback(std::shared_ptr<MemAlloc_t> mem) {
    alloc_events.emplace(_timer.get(), mem);
    active_memories.emplace(mem->addr, mem);
    
    _stats.num_allocs++;
    _stats.cur_mem_usage += mem->size;
    _stats.max_mem_usage = std::max(_stats.max_mem_usage, _stats.cur_mem_usage);

    _timer.increment(true);
}


void AppMetrics::mem_free_callback(std::shared_ptr<MemFree_t> mem) {
    auto it = active_memories.find(mem->addr);
    assert(it != active_memories.end());
    _stats.cur_mem_usage -= it->second->size;
    active_memories.erase(it);

    _timer.increment(true);
}


void AppMetrics::gpu_data_analysis(void* data, uint64_t size) {
    MemoryAccessTracker* tracker = (MemoryAccessTracker*)data;
    MemoryAccessState* states = tracker->access_state;

    uint32_t touched_objects = 0;
    uint32_t touched_objects_size = 0;
    for (uint32_t i = 0; i < states->size; i++) {
        if (states->touch[i] != 0) {
            touched_objects++;
            touched_objects_size += states->start_end[i].end - states->start_end[i].start;
        }
    }

    auto event = std::prev(kernel_events.end())->second;
    event->access_count = tracker->accessCount;
    event->touched_objects = touched_objects;
    event->touched_objects_size = touched_objects_size;
}


void AppMetrics::query_ranges(void* ranges, uint32_t limit, uint32_t* count) {
    MemoryRange* _ranges = (MemoryRange*)ranges;
    *count = 0;
    for (auto mem : active_memories) {
        _ranges[*count].start = mem.second->addr;
        _ranges[*count].end = mem.second->addr + mem.second->size;
        (*count)++;
        if (*count >= limit) {
            break;
        }
    }
}


void AppMetrics::flush() {
    const char* env_filename = std::getenv("YOSEMITE_APP_NAME");
    std::string filename;
    if (env_filename) {
        // fprintf(stdout, "YOSEMITE_APP_NAME: %s\n", env_filename);
        filename = std::string(env_filename) + "_" + get_current_date_n_time();
    } else {
        filename = "metrics_" + get_current_date_n_time();
        fprintf(stdout, "No filename specified. Using default filename: %s\n",
                filename.c_str());
    }
    filename += ".log";
    printf("Dumping traces to %s\n", filename.c_str());

    std::ofstream out(filename);

    int count = 0;
    for (auto event : alloc_events) {
        out << "Alloc(" << event.second->alloc_type << ") " << count << ":\t"
            << event.second->addr << " " << event.second->size
            << " (" << format_size(event.second->size) << ")" << std::endl;
        count++;
    }
    out << std::endl;

    for (auto event : kernel_events) {
        out << "Kernel " << event.second->kernel_id << " ("
            << "refs=" << event.second->access_count
            << ", objs=" << event.second->touched_objects
            << ", obj_size=" << event.second->touched_objects_size
            << ", " << format_size(event.second->touched_objects_size)
            << "):\t" << event.second->kernel_name << std::endl;
        _stats.tot_mem_accesses += event.second->access_count;
        if (_stats.max_mem_accesses_per_kernel < event.second->access_count) {
            _stats.max_mem_accesses_kernel = event.second->kernel_name;
            _stats.max_mem_access_kernel_id = event.second->kernel_id;
            _stats.max_mem_accesses_per_kernel = event.second->access_count;
        }

        _stats.tot_objs_per_kernel += event.second->touched_objects;
        if (_stats.max_objs_per_kernel < event.second->touched_objects) {
            _stats.max_objs_per_kernel = event.second->touched_objects;
        }

        _stats.tot_obj_size_per_kernel += event.second->touched_objects_size;
        if (_stats.max_obj_size_per_kernel < event.second->touched_objects_size) {
            _stats.max_obj_size_per_kernel = event.second->touched_objects_size;
        }

    }
    out << std::endl;

    // sort kernel_invocations by number of invocations in descending order
    std::vector<std::pair<std::string, uint32_t>> sorted_kernel_invocations(
                        kernel_invocations.begin(), kernel_invocations.end());
    std::sort(sorted_kernel_invocations.begin(), sorted_kernel_invocations.end(),
                                [](const std::pair<std::string, uint32_t>& a,
                                   const std::pair<std::string, uint32_t>& b) {
        return a.second > b.second;
    });
    for (auto kernel : sorted_kernel_invocations) {
        out << "InvCount=" << kernel.second << "\t" << kernel.first << std::endl;
    }
    out << std::endl;

    if (_stats.num_kernels > 0) {   // could be 0 when using python interface
        _stats.avg_mem_accesses = _stats.tot_mem_accesses / _stats.num_kernels;
        _stats.avg_objs_per_kernel = _stats.tot_objs_per_kernel / _stats.num_kernels;
        _stats.avg_obj_size_per_kernel = _stats.tot_obj_size_per_kernel / _stats.num_kernels;
    }
    out << "Number of allocations: " << _stats.num_allocs << std::endl;
    out << "Number of kernels: " << _stats.num_kernels << std::endl;
    out << "Maximum memory usage: " << _stats.tot_mem_accesses
        << "B (" << format_size(_stats.max_mem_usage) << ")" << std::endl;
    out << "------------------------------" << std::endl;
    out << "Maximum objects per kernel: " << _stats.max_objs_per_kernel << std::endl;
    out << "Average objects per kernel: " << _stats.avg_objs_per_kernel << std::endl;
    out << "Total objects per kernel: " << _stats.tot_objs_per_kernel << std::endl;
    out << "Maximum object size per kernel: " << _stats.max_obj_size_per_kernel
        << "B (" << format_size(_stats.max_obj_size_per_kernel) << ")" << std::endl;
    out << "Average object size per kernel: " << _stats.avg_obj_size_per_kernel
        << "B (" << format_size(_stats.avg_obj_size_per_kernel) << ")" << std::endl;
    out << "------------------------------" << std::endl;
    out << "Maximum memory accesses kernel: " << _stats.max_mem_accesses_kernel
        << " (Kernel ID: " << _stats.max_mem_access_kernel_id << ")" << std::endl;
    out << "Maximum memory accesses per kernel: " << _stats.max_mem_accesses_per_kernel
        << " (" << format_number(_stats.max_mem_accesses_per_kernel) << ")" << std::endl;
    out << "Average memory accesses per kernel: " << _stats.avg_mem_accesses
        << " (" << format_number(_stats.avg_mem_accesses) << ")"  << std::endl;
    out << "Total memory accesses: " << _stats.tot_mem_accesses
        << " (" << format_number(_stats.tot_mem_accesses) << ")"  << std::endl;

    auto avg_access_per_page = (float) _stats.tot_mem_accesses / (_stats.max_mem_usage / 4096.0f);
    out << "Average accesses per page: " << avg_access_per_page << std::endl;
    out.close();
}