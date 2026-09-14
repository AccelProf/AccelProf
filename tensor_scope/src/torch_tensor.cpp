#include "torch_tensor.h"

#ifdef TORCH_SCOPE_ROCM
#include <c10/hip/HIPCachingAllocator.h>
namespace device_allocator = c10::hip::HIPCachingAllocator;
#else
#include <c10/cuda/CUDACachingAllocator.h>
namespace device_allocator = c10::cuda::CUDACachingAllocator;
#endif

#include <mutex>

std::atomic<bool> TorchTensor::g_enabled{false};
std::atomic<bool> TorchTensor::g_attached{false};
std::array<TorchTensor::DeviceCounters, TorchTensor::k_max_devices> TorchTensor::g_counters;

TorchTensor* TorchTensor::getInstance() {
    // intentionally leaked for multi-process support
    static TorchTensor *instance = new TorchTensor();
    return instance;
}

void TorchTensor::enable_torch_callback() {
    g_enabled.store(true, std::memory_order_release);
    // Usually too early: enable_torch_scope() runs from the injection library's
    // constructor, before torch has initialized CUDA. The operator callback
    // retries until the allocator exists (see TorchOperator).
    ensure_tracker_attached();
}

void TorchTensor::disable_torch_callback() {
    g_enabled.store(false, std::memory_order_release);
}

// Register with the CUDA caching allocator exactly once per process.
//
// This replaces the previous design, which pushed a c10::MemoryReportingInfoBase
// into ThreadLocalDebugInfo under DebugInfoKind::PROFILER_STATE to receive
// reportMemoryUsage() calls. That slot is owned by torch's own profiler:
// torch::profiler::impl::getProfilerConfig() static_casts whatever sits there to
// ProfilerStateBase and reads its config, and THPFunction_apply() calls it on
// every custom torch.autograd.Function. With our object in the slot that read
// was type confusion - it surfaced as "RuntimeError: std::bad_array_new_length"
// on the first autograd.Function.apply (e.g. Triton's layer-norm tutorial), and
// was layout-dependent enough to disappear in -O0 builds. It also only saw
// allocations made on threads where the TLS had been pushed.
//
// The trace tracker is the supported hook, is process-wide, and sees every
// allocator event regardless of thread.
//
// Attaching only takes effect on per-device allocators that already exist:
// NativeCachingAllocator::attachAllocatorTraceTracker() forwards to the
// DeviceCachingAllocator objects created by init(device_count), which torch
// runs lazily on first CUDA use. Attaching before that silently registers
// nothing, so this waits for initialized() and is retried from a safe point.
bool TorchTensor::ensure_tracker_attached() {
    if (g_attached.load(std::memory_order_acquire)) {
        return true;
    }
    if (!device_allocator::get()->initialized()) {
        return false;
    }
    static std::once_flag once;
    std::call_once(once, [] {
        device_allocator::attachAllocatorTraceTracker(
            &TorchTensor::on_allocator_trace);
        g_attached.store(true, std::memory_order_release);
    });
    return true;
}

// Runs with the per-device allocator lock held, and possibly with the GIL held
// (see the comment on attachAllocatorTraceTracker in CUDACachingAllocator.h).
// Do not allocate device memory, call back into the allocator, or touch Python
// here; only bookkeeping and the forwarded callback.
void TorchTensor::on_allocator_trace(const c10::CachingDeviceAllocator::TraceEntry& entry) {
    using Action = c10::CachingDeviceAllocator::TraceEntry::Action;

    const int device = static_cast<int>(entry.device_);
    if (device < 0 || device >= k_max_devices) {
        return;
    }
    auto& counters = g_counters[static_cast<size_t>(device)];
    const int64_t size = static_cast<int64_t>(entry.size_);

    switch (entry.action_) {
        case Action::SEGMENT_ALLOC:
        case Action::SEGMENT_MAP:
            counters.reserved.fetch_add(size, std::memory_order_relaxed);
            return;
        case Action::SEGMENT_FREE:
        case Action::SEGMENT_UNMAP:
            counters.reserved.fetch_sub(size, std::memory_order_relaxed);
            return;
        case Action::ALLOC: {
            const int64_t allocated = counters.allocated.fetch_add(size, std::memory_order_relaxed) + size;
            if (!g_enabled.load(std::memory_order_acquire)) {
                return;
            }
            const int64_t reserved = counters.reserved.load(std::memory_order_relaxed);
            TorchTensor::getInstance()->tensor_malloc_callback(
                reinterpret_cast<void*>(entry.addr_), size, allocated, reserved, device);
            return;
        }
        case Action::FREE_COMPLETED: {
            // FREE_REQUESTED may precede this by a while when the block is still
            // in use on another stream; the memory is only really gone now.
            const int64_t allocated = counters.allocated.fetch_sub(size, std::memory_order_relaxed) - size;
            if (!g_enabled.load(std::memory_order_acquire)) {
                return;
            }
            const int64_t reserved = counters.reserved.load(std::memory_order_relaxed);
            TorchTensor::getInstance()->tensor_free_callback(
                reinterpret_cast<void*>(entry.addr_), -size, allocated, reserved, device);
            return;
        }
        default:
            // FREE_REQUESTED, SNAPSHOT, OOM, ANNOTATE: nothing to report.
            return;
    }
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
    if (this->tensor_malloc_callback_ptr == nullptr) {
        printf("tensor_malloc_callback_ptr is nullptr\n");
        return;
    }
    this->tensor_malloc_callback_ptr((uint64_t)ptr, alloc_size, total_allocated, total_reserved, device_id);
}

void TorchTensor::tensor_free_callback(void* ptr, int64_t alloc_size,
                                       int64_t total_allocated, int64_t total_reserved, int device_id) {
    if (this->tensor_free_callback_ptr == nullptr) {
        printf("tensor_free_callback_ptr is nullptr\n");
        return;
    }
    this->tensor_free_callback_ptr((uint64_t)ptr, alloc_size, total_allocated, total_reserved, device_id);
}
