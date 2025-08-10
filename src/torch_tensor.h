#ifndef _TORCH_TENSOR_H_
#define _TORCH_TENSOR_H_

#include <torch/extension.h>

#include "torch_scope.h"


class TorchTensor {
public:
    void enable_torch_callback();

    void register_tensor_callback(TorchScopeType_t scope_type,
                                  tensor_callback_t callback_ptr);

    void tensor_malloc_callback(void* ptr, int64_t alloc_size, int64_t total_allocated,
                                int64_t total_reserved, int device_id);

    void tensor_free_callback(void* ptr, int64_t alloc_size, int64_t total_allocated,
                                int64_t total_reserved, int device_id);

    static TorchTensor& getInstance();

private:
    TorchTensor() {}
    ~TorchTensor() {}

    class TorchCallback : public c10::MemoryReportingInfoBase {
    public:
        TorchCallback() {}

        bool memoryProfilingEnabled() const override { return true; }

#if TORCH_VERSION_MAJOR >= 2
        void reportMemoryUsage(void* ptr, int64_t alloc_size, size_t total_allocated,
                                size_t total_reserved, c10::Device device) override;
#else
        void reportMemoryUsage(void* ptr, int64_t alloc_size, int64_t total_allocated,
                                int64_t total_reserved, c10::Device device) override;
#endif
    };  // class TorchCallback

    std::shared_ptr<TorchCallback> new_torch_profiler() {
        return std::make_shared<TorchCallback>();
    }

    tensor_callback_t tensor_malloc_callback_ptr = nullptr;
    tensor_callback_t tensor_free_callback_ptr = nullptr;
};  // class TorchTensor


#endif //_TORCH_TENSOR_H_
