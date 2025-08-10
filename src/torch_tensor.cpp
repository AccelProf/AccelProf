#include "torch_tensor.h"

TorchTensor& TorchTensor::getInstance() {
    static TorchTensor instance;
    return instance;
}

void TorchTensor::enable_torch_callback() {
    auto profiler = new_torch_profiler();
    c10::ThreadLocalDebugInfo::_push(c10::DebugInfoKind::PROFILER_STATE, profiler);
}

void TorchTensor::register_tensor_callback(TorchScopeType_t scope_type,
                                         tensor_callback_t callback_ptr) {
    if (scope_type == TORCH_SCOPE_TENSOR_MALLOC) {
        this->tensor_malloc_callback_ptr = callback_ptr;
    } else if (scope_type == TORCH_SCOPE_TENSOR_FREE) {
        this->tensor_free_callback_ptr = callback_ptr;
    }
}

void TorchTensor::tensor_malloc_callback(void* ptr, int64_t alloc_size, int64_t total_allocated,
                                        int64_t total_reserved, int device_id) {
    // yosemite_tensor_malloc_callback(ptr, alloc_size, total_allocated, total_reserved);
    if (this->tensor_malloc_callback_ptr == nullptr) {
        printf("tensor_malloc_callback_ptr is nullptr\n");
        return;
    }
    this->tensor_malloc_callback_ptr((uint64_t)ptr, alloc_size, total_allocated, total_reserved, device_id);
}

void TorchTensor::tensor_free_callback(void* ptr, int64_t alloc_size,
                                        int64_t total_allocated, int64_t total_reserved, int device_id) {
    // yosemite_tensor_free_callback(ptr, alloc_size, total_allocated, total_reserved);
    if (this->tensor_free_callback_ptr == nullptr) {
        printf("tensor_free_callback_ptr is nullptr\n");
        return;
    }
    this->tensor_free_callback_ptr((uint64_t)ptr, alloc_size, total_allocated, total_reserved, device_id);
}

#if TORCH_VERSION_MAJOR >= 2
void TorchTensor::TorchCallback::reportMemoryUsage(void* ptr, int64_t alloc_size, size_t total_allocated,
                                                    size_t total_reserved, c10::Device device) {
    if (device.is_cuda() || device.is_hip()) {
        if (alloc_size > 0) {
            TorchTensor::getInstance().tensor_malloc_callback(ptr, alloc_size, total_allocated, total_reserved, device.index());
        } else {
            TorchTensor::getInstance().tensor_free_callback(ptr, alloc_size, total_allocated, total_reserved, device.index());
        }
    }
}
#else
void TorchTensor::TorchCallback::reportMemoryUsage(void* ptr, int64_t alloc_size, int64_t total_allocated,
                                                    int64_t total_reserved, c10::Device device) {
    if (device.is_cuda() || device.is_hip()) {
        if (alloc_size > 0) {
            TorchTensor::getInstance().tensor_malloc_callback(ptr, alloc_size, total_allocated, total_reserved, device.index());
        } else {
            TorchTensor::getInstance().tensor_free_callback(ptr, alloc_size, total_allocated, total_reserved, device.index());
        }
    }
}
#endif
