#ifndef _TORCH_TENSOR_H_
#define _TORCH_TENSOR_H_

#include <torch/extension.h>
#include <c10/core/CachingDeviceAllocator.h>

#include <array>
#include <atomic>
#include <cstdint>

#include "torch_scope.h"

// Observes PyTorch's CUDA caching allocator through its trace-tracker hook and
// forwards allocation / free events to the registered torch_scope callbacks.
//
// The callback convention is inherited from c10::MemoryReportingInfoBase, which
// this class used to implement: an allocation reports alloc_size > 0, a free
// reports alloc_size < 0 (the negated size), and both carry the device's running
// allocated / reserved byte totals. The consumers in sanalyzer rely on that sign.
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

public:
    // Attach the allocator trace tracker once the caching allocator exists.
    // Cheap to call repeatedly; returns true once attached. Must NOT be called
    // from inside an allocator callback (it takes the allocator lock).
    static bool ensure_tracker_attached();
private:
    static void on_allocator_trace(const c10::CachingDeviceAllocator::TraceEntry& entry);

    static constexpr int k_max_devices = 64;
    struct DeviceCounters {
        std::atomic<int64_t> allocated{0};   // bytes handed out to tensors
        std::atomic<int64_t> reserved{0};    // bytes held from the driver (segments)
    };

    static std::atomic<bool> g_enabled;                  // runtime on/off
    static std::atomic<bool> g_attached;                 // tracker installed?
    static std::array<DeviceCounters, k_max_devices> g_counters;

    tensor_callback_t tensor_malloc_callback_ptr = nullptr;
    tensor_callback_t tensor_free_callback_ptr = nullptr;
};  // class TorchTensor

#endif //_TORCH_TENSOR_H_
