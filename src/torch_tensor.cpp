#include "torch_tensor.h"


std::shared_ptr<c10::DebugInfoBase> TorchTensor::g_prof;
std::atomic<bool> TorchTensor::g_enabled{false};
thread_local bool TorchTensor::t_pushed = false;


TorchTensor* TorchTensor::getInstance() {
    // intentionally leaked for multi-process support
    static TorchTensor *instance = new TorchTensor();
    return instance;
}

void TorchTensor::enable_torch_callback() {
    ensure_profiler_installed_for_this_thread();
    g_enabled.store(true, std::memory_order_release);
}

void TorchTensor::disable_torch_callback() {
    g_enabled.store(false, std::memory_order_release);
}

// Install the immortal profiler into TLS of the current thread once.
void TorchTensor::ensure_profiler_installed_for_this_thread() {
    if (t_pushed) return;
    if (!g_prof) {
        // Create the immortal profiler instance exactly once per process.
        // Custom deleter is a no-op — the object lives until process exit.
        static std::once_flag once;
        std::call_once(once, [] {
            g_prof = std::shared_ptr<c10::DebugInfoBase>(
                static_cast<c10::DebugInfoBase*>(new TorchCallback()),
                +[](c10::DebugInfoBase*) { /* never delete; safe at atexit */ });
        });
    }
    c10::ThreadLocalDebugInfo::_push(c10::DebugInfoKind::PROFILER_STATE, g_prof);

    t_pushed = true;
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
