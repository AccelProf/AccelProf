#include <cstdio>
#include <cinttypes>
#include <memory>
#include <atomic>

#include <torch/extension.h>

#define VERBOSE 1

#ifdef VERBOSE
#define PRINT(fmt, ...) do {std::fprintf(stdout, fmt, ##__VA_ARGS__); std::fflush(stdout);} while (0)
#else
#define PRINT(fmt, ...) do {} while (0)
#endif

static inline bool is_cuda_or_hip(const c10::Device& d) {
  using DT = c10::DeviceType;
  return d.type() == DT::CUDA || d.type() == DT::HIP;
}

// ----------- simple logging (no globals/objects needed) -----------
static inline void tensor_malloc_callback(uint64_t ptr,
                              int64_t bytes,
                              int64_t total_alloc,
                              int64_t total_resv,
                              int device_id) {
  PRINT("Malloc tensor %lu with size %ld, allocated %ld, reserved %ld on device %d\n",
        ptr, bytes, total_alloc, total_resv, device_id);
}

static inline void tensor_free_callback(uint64_t ptr,
                            int64_t bytes,  // negative delta on frees
                            int64_t total_alloc,
                            int64_t total_resv,
                            int device_id) {
  PRINT("Free tensor %lu with size %ld, allocated %ld, reserved %ld on device %d\n",
        ptr, bytes, total_alloc, total_resv, device_id);
}

// ----------- Profiler callback -----------
class TorchCallback final : public c10::MemoryReportingInfoBase {
public:
  // Required by your PyTorch: implement the pure virtual
  bool memoryProfilingEnabled() const override { return true; }

#if defined(TORCH_VERSION_MAJOR) && (TORCH_VERSION_MAJOR >= 2)
  // PyTorch 2.x: totals are size_t
  void reportMemoryUsage(void* ptr,
                         int64_t alloc_size,
                         size_t total_allocated,
                         size_t total_reserved,
                         c10::Device device) override {
    if (!is_cuda_or_hip(device)) return;
    if (alloc_size > 0) {
      tensor_malloc_callback((uint64_t)ptr,
                 (int64_t)alloc_size,
                 (int64_t)total_allocated,
                 (int64_t)total_reserved,
                 device.index());
    } else {
      tensor_free_callback((uint64_t)ptr,
               (int64_t)alloc_size,
               (int64_t)total_allocated,
               (int64_t)total_reserved,
               device.index());
    }
  }
#else
  // PyTorch 1.x: totals are int64_t
  void reportMemoryUsage(void* ptr,
                         int64_t alloc_size,
                         int64_t total_allocated,
                         int64_t total_reserved,
                         c10::Device device) override {
    if (!is_cuda_or_hip(device)) return;
    if (alloc_size > 0) {
      tensor_malloc_callback((uint64_t)ptr,
                 (int64_t)alloc_size,
                 (int64_t)total_allocated,
                 (int64_t)total_reserved,
                 device.index());
    } else {
      tensor_free_callback((uint64_t)ptr,
               (int64_t)alloc_size,
               (int64_t)total_allocated,
               (int64_t)total_reserved,
               device.index());
    }
  }
#endif
};

// ----------- immortal profiler instance & installer -----------
// Keep-alive so the object never dies (even if TLS drops its copies).
// Custom deleter is a no-op on purpose.
static std::shared_ptr<c10::DebugInfoBase> g_prof;

static std::shared_ptr<c10::DebugInfoBase> make_profiler_never_delete() {
  static auto pinned = std::shared_ptr<c10::DebugInfoBase>(
      static_cast<c10::DebugInfoBase*>(new TorchCallback()),
      +[](c10::DebugInfoBase*) { /* never delete; safe at atexit */ });
  return pinned;
}

// Install once per process. Idempotent for safety.
__attribute__((constructor))
static void tensor_scope_on_load() {
  static std::atomic<bool> installed{false};
  if (installed.exchange(true)) return;

  PRINT("tensor_scope: constructor @%p\n", (void*)&tensor_scope_on_load);
  g_prof = make_profiler_never_delete();
  c10::ThreadLocalDebugInfo::_push(c10::DebugInfoKind::PROFILER_STATE, g_prof);

  // No destructor: do NOT pop or delete at exit.
}
