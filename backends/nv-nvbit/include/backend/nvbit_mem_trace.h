#ifndef _NVBIT_MEM_TRACE_H_
#define _NVBIT_MEM_TRACE_H_

#include "nvbit.h"

namespace yosemite_mem_trace {

void mem_trace_nvbit_at_init();

void mem_trace_nvbit_at_term();

void mem_trace_nvbit_at_ctx_init(CUcontext ctx);

void mem_trace_nvbit_at_ctx_term(CUcontext ctx);

void mem_trace_nvbit_tool_init(CUcontext ctx);

void mem_trace_nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus);

} // namespace yosemite_mem_trace

#endif // _NVBIT_MEM_TRACE_H_