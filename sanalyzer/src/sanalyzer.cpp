#include "sanalyzer.h"

#include "tools/tool.h"
#include "utils/event.h"
#include "tools/code_check.h"
#include "tools/app_metric.h"
#include "tools/roofline_flops.h"
#include "tools/roofline_size.h"
#include "tools/roofline_time.h"
#include "tools/mem_trace.h"
#include "tools/hot_analysis.h"
#include "tools/uvm_advisor.h"
#include "tools/app_analysis.h"
#include "tools/app_analysis_cpu.h"
#include "tools/app_analysis_nvbit.h"
#include "tools/time_hotness_cpu.h"
#include "tools/event_trace.h"
#include "tools/event_trace_mgpu.h"
#include "tools/heatmap_analysis.h"
#include "tools/block_divergence_analysis.h"
#include "tools/pc_dependency_analysis.h"

#include <memory>
#include <map>
#include <iostream>

using namespace yosemite;

static std::map<AnalysisTool_t, std::shared_ptr<Tool>> _tools;


YosemiteResult_t yosemite_tool_enable(AnalysisTool_t& tool) {
    const char* tool_name = std::getenv("YOSEMITE_TOOL_NAME");
    if (!tool_name) {
        fprintf(stdout, "[SANALYZER ERROR] No tool name specified.\n");
        return YOSEMITE_NOT_IMPLEMENTED;
    }

    // nvbit mode
    const char* yosemite_device_name = std::getenv("YOSEMITE_DEVICE");
    if (std::string(yosemite_device_name) == "nvbit") {
        if (std::string(tool_name) == "app_analysis") {
            tool = APP_ANALYSIS_NVBIT;
            _tools.emplace(APP_ANALYSIS_NVBIT, std::make_shared<AppAnalysisNVBIT>());
        } else  if (std::string(tool_name) == "roofline_flops") {
            tool = ROOFLINE_FLOPS;
            _tools.emplace(ROOFLINE_FLOPS, std::make_shared<RooflineFlops>());
        } else {
            fprintf(stderr, "[SANALYZER ERROR] Unsupported tool in nvbit mode, %s.\n", tool_name);
            fflush(stderr);
            return YOSEMITE_NOT_IMPLEMENTED;
        }

        fprintf(stdout, "[SANALYZER INFO] Enabling %s tool in nvbit mode.\n", tool_name);
        fflush(stdout);
        return YOSEMITE_SUCCESS;
    }

    // rocm mode
    if (std::string(yosemite_device_name) == "rocm") {
        if (std::string(tool_name) == "event_trace") {
            tool = EVENT_TRACE;
            _tools.emplace(EVENT_TRACE, std::make_shared<EventTrace>());
        } else {
            fprintf(stderr, "[SANALYZER ERROR] Unsupported tool in rocm mode, %s.\n", tool_name);
            fflush(stderr);
            return YOSEMITE_NOT_IMPLEMENTED;
        }
        fprintf(stdout, "[SANALYZER INFO] Enabling %s tool in rocm mode.\n", tool_name);
        fflush(stdout);
        return YOSEMITE_SUCCESS;
    }

    if (std::string(tool_name) == "code_check") {
        tool = CODE_CHECK;
        _tools.emplace(CODE_CHECK, std::make_shared<CodeCheck>());
    } else if (std::string(tool_name) == "app_metric") {
        tool = APP_METRICE;
        _tools.emplace(APP_METRICE, std::make_shared<AppMetrics>());
    } else if (std::string(tool_name) == "roofline_size") {
        tool = ROOFLINE_SIZE;
        _tools.emplace(ROOFLINE_SIZE, std::make_shared<RooflineSize>());
    } else if (std::string(tool_name) == "roofline_time") {
        tool = ROOFLINE_TIME;
        _tools.emplace(ROOFLINE_TIME, std::make_shared<RooflineTime>());
    } else if (std::string(tool_name) == "mem_trace") {
        tool = MEM_TRACE;
        _tools.emplace(MEM_TRACE, std::make_shared<MemTrace>());
    } else if (std::string(tool_name) == "hot_analysis") {
        tool = HOT_ANALYSIS;
        _tools.emplace(HOT_ANALYSIS, std::make_shared<HotAnalysis>());
    } else if (std::string(tool_name) == "uvm_advisor") {
        tool = UVM_ADVISOR;
        _tools.emplace(UVM_ADVISOR, std::make_shared<UVMAdvisor>());
    } else if (std::string(tool_name) == "app_analysis") {
        tool = APP_ANALYSIS;
        _tools.emplace(APP_ANALYSIS, std::make_shared<AppAnalysis>());
    } else if (std::string(tool_name) == "app_analysis_cpu") {
        tool = APP_ANALYSIS_CPU;
        _tools.emplace(APP_ANALYSIS_CPU, std::make_shared<AppAnalysisCPU>());
    } else if (std::string(tool_name) == "time_hotness_cpu") {
        tool = TIME_HOTNESS_CPU;
        _tools.emplace(TIME_HOTNESS_CPU, std::make_shared<TimeHotnessCPU>());
    } else if (std::string(tool_name) == "event_trace") {
        tool = EVENT_TRACE;
        _tools.emplace(EVENT_TRACE, std::make_shared<EventTrace>());
    } else if (std::string(tool_name) == "event_trace_mgpu") {
        tool = EVENT_TRACE_MGPU;
        _tools.emplace(EVENT_TRACE_MGPU, std::make_shared<EventTraceMGPU>());
    } else if (std::string(tool_name) == "heatmap_analysis") {
        tool = HEATMAP_ANALYSIS;
        _tools.emplace(HEATMAP_ANALYSIS, std::make_shared<HeatmapAnalysis>());
    } else if (std::string(tool_name) == "block_divergence_analysis") {
        tool = BLOCK_DIVERGENCE_ANALYSIS;
        _tools.emplace(BLOCK_DIVERGENCE_ANALYSIS, std::make_shared<BlockDivergenceAnalysis>());
    } else if (std::string(tool_name) == "pc_dependency_analysis") {
        tool = PC_DEPENDENCY_ANALYSIS;
        _tools.emplace(PC_DEPENDENCY_ANALYSIS, std::make_shared<PcDependency>());
    } else {
        fprintf(stderr, "[SANALYZER ERROR] Tool not found.\n");
        fflush(stderr);
        return YOSEMITE_NOT_IMPLEMENTED;
    }

    fprintf(stdout, "[SANALYZER INFO] Enabling %s tool.\n", tool_name);
    fflush(stdout);
    return YOSEMITE_SUCCESS;
}


YosemiteResult_t yosemite_tool_disable() {
    return YOSEMITE_SUCCESS;
}


YosemiteResult_t yosemite_flush() {
    for (auto &tool : _tools) {
        tool.second->flush();
    }
    return YOSEMITE_SUCCESS;
}


YosemiteResult_t yosemite_torch_prof_enable() {
    fprintf(stdout, "[SANALYZER INFO] Enabling torch profiler.\n");
    fflush(stdout);
    return YOSEMITE_SUCCESS;
}


/****************************************************************************************
 ********************************** Interface functions *********************************
****************************************************************************************/


YosemiteResult_t yosemite_alloc_callback(uint64_t ptr, uint64_t size, int type, int device_id) {
    for (auto &tool : _tools) {
        auto mem_alloc = std::make_shared<MemAlloc_t>(ptr, size, type, device_id);
        tool.second->evt_callback(mem_alloc);
    }
    return YOSEMITE_SUCCESS;
}


YosemiteResult_t yosemite_free_callback(uint64_t ptr, uint64_t size, int type, int device_id) {
    if (ptr == 0) {
        return YOSEMITE_CUDA_MEMFREE_ZERO;
    }
    for (auto &tool : _tools) {
        auto mem_free = std::make_shared<MemFree_t>(ptr, size, type, device_id);
        tool.second->evt_callback(mem_free);
    }
    return YOSEMITE_SUCCESS;
}


YosemiteResult_t yosemite_memcpy_callback(uint64_t dst, uint64_t src, uint64_t size, bool is_async, uint32_t direction, int device_id) {
    for (auto &tool : _tools) {
        auto mem_cpy = std::make_shared<MemCpy_t>(dst, src, size, is_async, direction, device_id);
        tool.second->evt_callback(mem_cpy);
    }
    return YOSEMITE_SUCCESS;
}


YosemiteResult_t yosemite_memset_callback(uint64_t dst, uint32_t size, int value, bool is_async, int device_id) {
    for (auto &tool : _tools) {
        auto mem_set = std::make_shared<MemSet_t>(dst, size, value, is_async, device_id);
        tool.second->evt_callback(mem_set);
    }
    return YOSEMITE_SUCCESS;
}


YosemiteResult_t yosemite_kernel_start_callback(
    std::string kernel_name,
    int device_id,
    uint32_t grid_dim_x,
    uint32_t grid_dim_y,
    uint32_t grid_dim_z,
    uint32_t block_dim_x,
    uint32_t block_dim_y,
    uint32_t block_dim_z
) {
    for (auto &tool : _tools) {
        auto grid_cta_count =
            static_cast<uint64_t>(grid_dim_x) * static_cast<uint64_t>(grid_dim_y) * static_cast<uint64_t>(grid_dim_z);
        auto block_thread_count = block_dim_x * block_dim_y * block_dim_z;

        auto kernel = std::make_shared<KernelLaunch_t>(kernel_name, device_id,
                                                        grid_dim_x, grid_dim_y, grid_dim_z,
                                                        block_dim_x, block_dim_y, block_dim_z,
                                                        grid_cta_count, block_thread_count);

        tool.second->evt_callback(kernel);
    }
    return YOSEMITE_SUCCESS;
}


YosemiteResult_t yosemite_kernel_end_callback(std::string kernel_name, int device_id) {
    for (auto &tool : _tools) {
        auto kernel = std::make_shared<KernelEnd_t>(kernel_name, device_id);
        tool.second->evt_callback(kernel);
    }
    return YOSEMITE_SUCCESS;
}


YosemiteResult_t yosemite_gpu_data_analysis(void* data, uint64_t size) {
    for (auto &tool : _tools) {
        tool.second->gpu_data_analysis(data, size);
    }
    return YOSEMITE_SUCCESS;
}


YosemiteResult_t yosemite_init(AccelProfOptions_t& options) {
    AnalysisTool_t tool;
    YosemiteResult_t res = yosemite_tool_enable(tool);
    if (res != YOSEMITE_SUCCESS) {
        return res;
    }

    if (tool == CODE_CHECK) {
        options.patch_name = GPU_NO_PATCH;
    } else if (tool == APP_METRICE) {
        options.patch_name = GPU_PATCH_APP_METRIC;
        options.patch_file = "gpu_patch_app_metric.fatbin";
    } else if (tool == ROOFLINE_FLOPS) {
        options.patch_name = GPU_PATCH_ROOFLINE_FLOPS_NVBIT;
    } else if (tool == ROOFLINE_SIZE) {
        options.patch_name = GPU_PATCH_ROOFLINE_SIZE;
        options.patch_file = "gpu_patch_roofline_size.fatbin";
    } else if (tool == ROOFLINE_TIME) {
       options.patch_name = GPU_NO_PATCH;
    } else if (tool == MEM_TRACE) {
        options.patch_name = GPU_PATCH_MEM_TRACE;
        options.patch_file = "gpu_patch_mem_trace.fatbin";
    } else if (tool == HOT_ANALYSIS) {
        options.patch_name = GPU_PATCH_HOT_ANALYSIS;
        options.patch_file = "gpu_patch_hot_analysis.fatbin";
    } else if (tool == UVM_ADVISOR) {
        options.patch_name = GPU_PATCH_UVM_ADVISOR;
        options.patch_file = "gpu_patch_uvm_advisor.fatbin";
    } else if (tool == APP_ANALYSIS) {
        options.patch_name = GPU_PATCH_APP_ANALYSIS;
        options.patch_file = "gpu_patch_app_analysis.fatbin";
    } else if (tool == APP_ANALYSIS_CPU) {
        options.patch_name = GPU_PATCH_APP_ANALYSIS_CPU;
        options.patch_file = "gpu_patch_app_analysis_cpu.fatbin";
    } else if (tool == APP_ANALYSIS_NVBIT) {
        options.patch_name = GPU_PATCH_APP_ANALYSIS_NVBIT;
    } else if (tool == TIME_HOTNESS_CPU) {
        options.patch_name = GPU_PATCH_TIME_HOTNESS_CPU;
        options.patch_file = "gpu_patch_time_hotness_cpu.fatbin";
    } else if (tool == EVENT_TRACE) {
        options.patch_name = GPU_NO_PATCH;
    } else if (tool == EVENT_TRACE_MGPU) {
        options.patch_name = GPU_NO_PATCH;
    } else if (tool == HEATMAP_ANALYSIS) {
        options.patch_name = GPU_PATCH_HEATMAP_ANALYSIS;
        options.patch_file = "gpu_patch_heatmap_analysis.fatbin";
    } else if (tool == BLOCK_DIVERGENCE_ANALYSIS) {
        options.patch_name = GPU_PATCH_BLOCK_DIVERGENCE_ANALYSIS;
        options.patch_file = "gpu_patch_block_divergence_analysis.fatbin";
    } else if (tool == PC_DEPENDENCY_ANALYSIS) {
        options.patch_name = GPU_PATCH_PC_DEPENDENCY_ANALYSIS;
        // backends/nv-compute/Makefile generates fatbins based on gpu_src/*.cu filenames.
        // The source file for this tool is backends/nv-compute/gpu_src/gpu_patch_pc_dependency.cu
        options.patch_file = "gpu_patch_pc_dependency.fatbin";
    }

    // enable torch profiler?
    const char* torch_prof = std::getenv("TORCH_PROFILE_ENABLED");
    if (torch_prof && std::string(torch_prof) == "1") {
        options.torch_prof_enabled = true;
        yosemite_torch_prof_enable();
    }

    // set sample rate
    const char* sample_rate = std::getenv("YOSEMITE_ENV_SAMPLE_RATE");
    if (sample_rate) {
        options.sample_rate = std::stoi(sample_rate);
        fprintf(stdout, "[SANALYZER INFO] Setting sample rate to %d.\n", options.sample_rate);
    }

    fprintf(stdout, "================================================================================\n");
    fflush(stdout);

    return YOSEMITE_SUCCESS;
}


YosemiteResult_t yosemite_terminate() {
    yosemite_flush();
    return YOSEMITE_SUCCESS;
}


YosemiteResult_t yosemite_tensor_malloc_callback(uint64_t ptr, int64_t alloc_size,
                                    int64_t total_allocated, int64_t total_reserved, int device_id) {
    for (auto &tool : _tools) {
        auto ten_alloc = std::make_shared<TenAlloc_t>(ptr, alloc_size, total_allocated, total_reserved, device_id);
        tool.second->evt_callback(ten_alloc);
    }
    return YOSEMITE_SUCCESS;
}


YosemiteResult_t yosemite_tensor_free_callback(uint64_t ptr, int64_t alloc_size,
                                    int64_t total_allocated, int64_t total_reserved, int device_id) {
    for (auto &tool : _tools) {
        auto ten_free = std::make_shared<TenFree_t>(ptr, alloc_size, total_allocated, total_reserved, device_id);
        tool.second->evt_callback(ten_free);
    }
    return YOSEMITE_SUCCESS;
}


YosemiteResult_t yosemite_operator_start_callback(void* ctx, std::string op_name) {
    for (auto &tool : _tools) {
        auto op_start = std::make_shared<OpStart_t>(op_name, ctx);
        tool.second->evt_callback(op_start);
    }
    return YOSEMITE_SUCCESS;
}


YosemiteResult_t yosemite_operator_end_callback(void* ctx, std::string op_name) {
    for (auto &tool : _tools) {
        auto op_end = std::make_shared<OpEnd_t>(op_name, ctx);
        tool.second->evt_callback(op_end);
    }
    return YOSEMITE_SUCCESS;
}


YosemiteResult_t yosemite_query_active_ranges(void* ranges, uint32_t limit, uint32_t* count) {
    for (auto &tool : _tools) {
        tool.second->query_ranges(ranges, limit, count);
    }
    return YOSEMITE_SUCCESS;
}


YosemiteResult_t yosemite_query_active_tensors(void* ranges, uint32_t limit, uint32_t* count) {
    for (auto &tool : _tools) {
        tool.second->query_tensors(ranges, limit, count);
    }
    return YOSEMITE_SUCCESS;
}
