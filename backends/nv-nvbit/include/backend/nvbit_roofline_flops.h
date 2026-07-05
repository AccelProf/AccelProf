#ifndef _NVBIT_ROOFLINE_FLOPS_H_
#define _NVBIT_ROOFLINE_FLOPS_H_

#include "nvbit.h"

namespace yosemite_roofline_flops {

void roofline_flops_nvbit_at_init();

void roofline_flops_nvbit_at_term();

void roofline_flops_nvbit_at_ctx_init(CUcontext ctx);

void roofline_flops_nvbit_at_ctx_term(CUcontext ctx);

void roofline_flops_nvbit_tool_init(CUcontext ctx);

void roofline_flops_nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus);

} // namespace yosemite_roofline_flops

#endif // _NVBIT_ROOFLINE_FLOPS_H_
