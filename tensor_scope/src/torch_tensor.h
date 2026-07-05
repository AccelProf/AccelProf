#ifndef _TORCH_TENSOR_H_
#define _TORCH_TENSOR_H_

#include <torch/extension.h>

#include "torch_scope.h"


class TorchTensor {
public:
    static TorchTensor* getInstance();

    void enable_torch_callback();

    void disable_torch_callback();
    
    void register_tensor_callback(TorchScopeType_t scope_type,
                                  tensor_callback_t callback_ptr);

    void tensor_malloc_callback(void* ptr, int64_t alloc_size, int64_t total_allocated,
                                int64_t total_reserved, int device_id);

    void tensor_free_callback(void* ptr, int64_t alloc_size, int64_t total_allocated,
                                int64_t total_reserved, int device_id);

    TorchTensor(const TorchTensor&) = delete;

    TorchTensor& operator=(const TorchTensor&) = delete;

private:
    TorchTensor() = default;
    ~TorchTensor() = default;

    static void ensure_profiler_installed_for_this_thread();

    class TorchCallback final : public c10::MemoryReportingInfoBase {
    public:
        TorchCallback() {}
    
        bool memoryProfilingEnabled() const override {
            return g_enabled.load(std::memory_order_acquire);
        }
    
    #if TORCH_VERSION_MAJOR >= 2
        void reportMemoryUsage(void* ptr, int64_t alloc_size, size_t total_allocated,
                                size_t total_reserved, c10::Device device) override {
            if (!device.is_cuda() && !device.is_hip()) {
                return;
            }
            auto* tt = TorchTensor::getInstance();
            if (alloc_size > 0) {
                tt->tensor_malloc_callback(ptr, alloc_size, total_allocated, total_reserved, device.index());
            } else {
                tt->tensor_free_callback(ptr, alloc_size, total_allocated, total_reserved, device.index());
            }
        }
    #else
        void reportMemoryUsage(void* ptr, int64_t alloc_size, int64_t total_allocated,
                                int64_t total_reserved, c10::Device device) override {
            if (!device.is_cuda() && !device.is_hip()) {
                return;
            }
            auto* tt = TorchTensor::getInstance();
            if (alloc_size > 0) {
                tt->tensor_malloc_callback(ptr, alloc_size, total_allocated, total_reserved, device.index());
            } else {
                tt->tensor_free_callback(ptr, alloc_size, total_allocated, total_reserved, device.index());
            }
        }
    #endif
    };  // class TorchCallback

    // -------- global state (process-wide) --------
    static std::shared_ptr<c10::DebugInfoBase> g_prof;   // immortal profiler object
    static std::atomic<bool> g_enabled;                  // runtime on/off

    // -------- per-thread state --------
    static thread_local bool t_pushed; // installed TLS on this thread?

    tensor_callback_t tensor_malloc_callback_ptr = nullptr;
    tensor_callback_t tensor_free_callback_ptr = nullptr;
};  // class TorchTensor




#endif //_TORCH_TENSOR_H_
