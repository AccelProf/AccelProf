#include "sanalyzer.h"

#include "backend/nvbit_app_metric.h"
#include "backend/nvbit_mem_trace.h"
#include "backend/nvbit_app_analysis.h"
#include "backend/nvbit_roofline_flops.h"
#include "torch_scope.h"

/* every tool needs to include this once */
#include "nvbit_tool.h"


#define NVBIT_VERBOSE 1

#if NVBIT_VERBOSE
#define PRINT(...) do { fprintf(stdout, __VA_ARGS__); fflush(stdout); } while (0)
#else
#define PRINT(...)
#endif


static AccelProfOptions_t nvbit_options;

static volatile int nvbit_debug_wait_flag = 1;


void nvbit_tensor_malloc_callback(uint64_t ptr, int64_t size, int64_t allocated, int64_t reserved, int device_id) {
    PRINT("[NVBIT INFO] Malloc tensor %p with size %ld, allocated %ld, reserved %ld\n", (void*)ptr, size, allocated, reserved);
    yosemite_tensor_malloc_callback(ptr, size, allocated, reserved, device_id);
}

void nvbit_tensor_free_callback(uint64_t ptr, int64_t size, int64_t allocated, int64_t reserved, int device_id) {
    PRINT("[NVBIT INFO] Free tensor %p with size %ld, allocated %ld, reserved %ld\n", (void*)ptr, size, allocated, reserved);
    yosemite_tensor_free_callback(ptr, size, allocated, reserved, device_id);
}

void nvbit_operator_start_callback(void* ctx, std::string op_name) {
    PRINT("[NVBIT INFO] Torch operator start: %s, ctx: %p\n", op_name.c_str(), ctx);
    yosemite_operator_start_callback(ctx, op_name);
}

void nvbit_operator_end_callback(void* ctx, std::string op_name) {
    PRINT("[NVBIT INFO] Torch operator end: %s, ctx: %p\n", op_name.c_str(), ctx);
    yosemite_operator_end_callback(ctx, op_name);
}

/* nvbit_at_init() is executed as soon as the nvbit tool is loaded. We
 * typically do initializations in this call. In this case for instance we get
 * some environment variables values which we use as input arguments to the tool
 */
void nvbit_at_init() {

    const char* debug = std::getenv("ACCEL_PROF_DEBUG");
    if (debug) {
        while (nvbit_debug_wait_flag);
    }
    unsetenv("ACCEL_PROF_DEBUG");

    yosemite_init(nvbit_options);

    if (nvbit_options.torch_prof_enabled) {
        enable_torch_scope();
        register_torch_scope_callback(TORCH_SCOPE_TENSOR_MALLOC, (torch_scope_callback_t)nvbit_tensor_malloc_callback);
        register_torch_scope_callback(TORCH_SCOPE_TENSOR_FREE, (torch_scope_callback_t)nvbit_tensor_free_callback);
        register_torch_scope_callback(TORCH_SCOPE_OPERATOR_START, (torch_scope_callback_t)nvbit_operator_start_callback);
        register_torch_scope_callback(TORCH_SCOPE_OPERATOR_END, (torch_scope_callback_t)nvbit_operator_end_callback);
    }

    if (nvbit_options.patch_name == GPU_PATCH_APP_METRIC) {
        yosemite_app_metric::app_metric_nvbit_at_init();
    } else if (nvbit_options.patch_name == GPU_PATCH_MEM_TRACE) {
        yosemite_mem_trace::mem_trace_nvbit_at_init();
    } else if (nvbit_options.patch_name == GPU_PATCH_APP_ANALYSIS_NVBIT) {
        yosemite_app_analysis::app_analysis_nvbit_at_init();
    } else if (nvbit_options.patch_name == GPU_PATCH_ROOFLINE_FLOPS_NVBIT) {
        yosemite_roofline_flops::roofline_flops_nvbit_at_init();
    }
}

void nvbit_tool_init(CUcontext ctx) {
    if (nvbit_options.patch_name == GPU_PATCH_APP_METRIC) {
        yosemite_app_metric::app_metric_nvbit_tool_init(ctx);
    } else if (nvbit_options.patch_name == GPU_PATCH_MEM_TRACE) {
        yosemite_mem_trace::mem_trace_nvbit_tool_init(ctx);
    } else if (nvbit_options.patch_name == GPU_PATCH_APP_ANALYSIS_NVBIT) {
        yosemite_app_analysis::app_analysis_nvbit_tool_init(ctx);
    } else if (nvbit_options.patch_name == GPU_PATCH_ROOFLINE_FLOPS_NVBIT) {
        yosemite_roofline_flops::roofline_flops_nvbit_tool_init(ctx);
    }
}

void nvbit_at_ctx_init(CUcontext ctx) {
    if (nvbit_options.patch_name == GPU_PATCH_APP_METRIC) {
        yosemite_app_metric::app_metric_nvbit_at_ctx_init(ctx);
    } else if (nvbit_options.patch_name == GPU_PATCH_MEM_TRACE) {
        yosemite_mem_trace::mem_trace_nvbit_at_ctx_init(ctx);
    } else if (nvbit_options.patch_name == GPU_PATCH_APP_ANALYSIS_NVBIT) {
        yosemite_app_analysis::app_analysis_nvbit_at_ctx_init(ctx);
    } else if (nvbit_options.patch_name == GPU_PATCH_ROOFLINE_FLOPS_NVBIT) {
        yosemite_roofline_flops::roofline_flops_nvbit_at_ctx_init(ctx);
    }
}

void nvbit_at_ctx_term(CUcontext ctx) {
    if (nvbit_options.patch_name == GPU_PATCH_APP_METRIC) {
        yosemite_app_metric::app_metric_nvbit_at_ctx_term(ctx);
    } else if (nvbit_options.patch_name == GPU_PATCH_MEM_TRACE) {
        yosemite_mem_trace::mem_trace_nvbit_at_ctx_term(ctx);
    } else if (nvbit_options.patch_name == GPU_PATCH_APP_ANALYSIS_NVBIT) {
        yosemite_app_analysis::app_analysis_nvbit_at_ctx_term(ctx);
    } else if (nvbit_options.patch_name == GPU_PATCH_ROOFLINE_FLOPS_NVBIT) {
        yosemite_roofline_flops::roofline_flops_nvbit_at_ctx_term(ctx);
    }
}


/* This call-back is triggered every time a CUDA driver call is encountered.
 * Here we can look for a particular CUDA driver call by checking at the
 * call back ids  which are defined in tools_cuda_api_meta.h.
 * This call back is triggered bith at entry and at exit of each CUDA driver
 * call, is_exit=0 is entry, is_exit=1 is exit.
 * */
void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
    
    if (nvbit_options.patch_name == GPU_PATCH_APP_METRIC) {
        yosemite_app_metric::app_metric_nvbit_at_cuda_event(ctx, is_exit, cbid, name, params, pStatus);
    } else if (nvbit_options.patch_name == GPU_PATCH_MEM_TRACE) {
        yosemite_mem_trace::mem_trace_nvbit_at_cuda_event(ctx, is_exit, cbid, name, params, pStatus);
    } else if (nvbit_options.patch_name == GPU_PATCH_APP_ANALYSIS_NVBIT) {
        yosemite_app_analysis::app_analysis_nvbit_at_cuda_event(ctx, is_exit, cbid, name, params, pStatus);
    } else if (nvbit_options.patch_name == GPU_PATCH_ROOFLINE_FLOPS_NVBIT) {
        yosemite_roofline_flops::roofline_flops_nvbit_at_cuda_event(ctx, is_exit, cbid, name, params, pStatus);
    }

}

void nvbit_at_term() {
    if (nvbit_options.patch_name == GPU_PATCH_APP_METRIC) {
        yosemite_app_metric::app_metric_nvbit_at_term();
    } else if (nvbit_options.patch_name == GPU_PATCH_MEM_TRACE) {
        yosemite_mem_trace::mem_trace_nvbit_at_term();
    } else if (nvbit_options.patch_name == GPU_PATCH_APP_ANALYSIS_NVBIT) {
        yosemite_app_analysis::app_analysis_nvbit_at_term();
    } else if (nvbit_options.patch_name == GPU_PATCH_ROOFLINE_FLOPS_NVBIT) {
        yosemite_roofline_flops::roofline_flops_nvbit_at_term();
    }
}
