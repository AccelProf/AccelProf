#ifndef _TORCH_SCOPE_H_
#define _TORCH_SCOPE_H_

#include <cstdint>
#include <string>

typedef enum {
    TORCH_SCOPE_TENSOR_MALLOC = 0,
    TORCH_SCOPE_TENSOR_FREE = 1,
    TORCH_SCOPE_OPERATOR_START = 2,
    TORCH_SCOPE_OPERATOR_END = 3,
    TORCH_SCOPE_UNKNOWN = 4,
} TorchScopeType_t;


// TORCH_SCOPE_TENSOR_MALLOC, TORCH_SCOPE_TENSOR_FREE
typedef void (*tensor_callback_t)(uint64_t ptr, int64_t alloc_size,
                        int64_t total_allocated, int64_t total_reserved, int device_id);

// TORCH_SCOPE_OPERATOR_START, TORCH_SCOPE_OPERATOR_END
typedef void (*operator_callback_t)(void* ctx, std::string op_name);


typedef void* torch_scope_callback_t;

extern "C" void register_torch_scope_callback(TorchScopeType_t scope_type,
                                      torch_scope_callback_t callback_ptr);

extern "C" void enable_torch_scope();


#endif // _TORCH_SCOPE_H_
