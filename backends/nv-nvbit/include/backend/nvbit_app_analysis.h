#ifndef _NVBIT_APP_ANALYSIS_H_
#define _NVBIT_APP_ANALYSIS_H_

#include "nvbit.h"

namespace yosemite_app_analysis {

void app_analysis_nvbit_at_init();

void app_analysis_nvbit_at_term();

void app_analysis_nvbit_at_ctx_init(CUcontext ctx);

void app_analysis_nvbit_at_ctx_term(CUcontext ctx);

void app_analysis_nvbit_tool_init(CUcontext ctx);

void app_analysis_nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus);

} // namespace yosemite_app_analysis

#endif // _NVBIT_APP_ANALYSIS_H_
