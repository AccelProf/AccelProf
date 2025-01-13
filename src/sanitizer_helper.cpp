#include "sanitizer_helper.h"
#include <cuda_runtime_api.h>

#include <map>
#include <cstdio>
#include <cxxabi.h>

static volatile int debug_wait_flag = 1;
static volatile bool cuda_api_internal = false;

static std::map<CUcontext, CUstream> context_priority_stream_map;
static std::map<CUcontext, CUstream> context_stream_map;

#define CUDA_SAFECALL(call)                                          \
    {                                                                \
        call;                                                        \
        cudaError err = cudaGetLastError();                          \
        if (cudaSuccess != err) {                                    \
            fprintf(                                                 \
                stderr,                                              \
                "[CUDA ERROR] '%s' in '%s:%i' - error: %s.\n",       \
                #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            fflush(stderr);                                          \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    }



void create_priority_stream(CUstream* p_stream) {
    cuda_api_internal = true;
    int priority_high, priority_low;
    CUDA_SAFECALL(cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high));
    CUDA_SAFECALL(cudaStreamCreateWithPriority(p_stream, cudaStreamNonBlocking, priority_high););
    cuda_api_internal = false;
}


void create_stream(CUstream* p_stream) {
    cuda_api_internal = true;
    CUDA_SAFECALL(cudaStreamCreateWithFlags(p_stream, cudaStreamNonBlocking));
    cuda_api_internal = false;
}


void sanitizer_priority_stream_get(CUcontext context, CUstream* p_stream) {
    if (context_priority_stream_map.find(context) != context_priority_stream_map.end()) {
        *p_stream = context_priority_stream_map[context];
    } else {
        create_priority_stream(p_stream);
        context_priority_stream_map[context] = *p_stream;
    }
}


void sanitizer_stream_get(CUcontext context, CUstream* p_stream) {
    if (context_stream_map.find(context) != context_stream_map.end()) {
        *p_stream = context_stream_map[context];
    } else {
        create_stream(p_stream);
        context_stream_map[context] = *p_stream;
    }
}


bool sanitizer_cuda_api_internal() {
    return cuda_api_internal;
}


void sanitizer_debug_wait() {
    const char* debug = std::getenv("COMPUTE_SANITIZER_DEBUG");
    if (debug) {
        while (debug_wait_flag);
    }
    unsetenv("COMPUTE_SANITIZER_DEBUG");
}


const char* sanitizer_demangled_name_get(const char* function) {
    /// demangle function name
    const char *func_name = function;
    int status;
    char *demangled = abi::__cxa_demangle(function, nullptr, nullptr, &status);
    if (status == 0) {
        func_name = demangled;
    }

    return func_name;
}