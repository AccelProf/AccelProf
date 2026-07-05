#ifndef _SANINITIZER_HELPER_H_
#define _SANINITIZER_HELPER_H_

#include <cuda.h>

void sanitizer_priority_stream_get(CUcontext context, CUstream* p_stream);

void sanitizer_stream_get(CUcontext context, CUstream* p_stream);

bool sanitizer_cuda_api_internal();

void sanitizer_debug_wait();

const char* sanitizer_demangled_name_get(const char* function);

#endif // _SANINITIZER_HELPER_H_
