#ifndef _SANINITIZER_HELPER_H_
#define _SANINITIZER_HELPER_H_

#include <cuda.h>

void get_priority_stream(CUcontext context, CUstream* p_stream);

void get_stream(CUcontext context, CUstream* p_stream);

bool is_cuda_api_internal();

const char* get_demangled_name(const char* function);

#endif // _SANINITIZER_HELPER_H_
