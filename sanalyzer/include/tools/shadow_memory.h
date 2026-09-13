#ifndef YOSEMITE_TOOLS_SHADOW_MEMORY_H
#define YOSEMITE_TOOLS_SHADOW_MEMORY_H

// Shadow-memory infrastructure shared by the trace-driven tools
// (pc_dependency_analysis, redsan). Originally written for cuVein's
// pc_dependency_analysis; hoisted here unchanged so other tools can reuse the
// same allocation-mirrored global shadow and per-CTA shared-memory shadow pool.

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>
#include <sys/mman.h>

constexpr uint32_t shared_memory_upper_bound = 108*1024;


#ifndef SANITIZER_MEMORY_DEVICE_FLAG_READ
#define SANITIZER_MEMORY_DEVICE_FLAG_READ 0x1
#endif

#ifndef SANITIZER_MEMORY_DEVICE_FLAG_WRITE
#define SANITIZER_MEMORY_DEVICE_FLAG_WRITE 0x2
#endif

#ifndef SANITIZER_MEMORY_DEVICE_FLAG_RED
#define SANITIZER_MEMORY_DEVICE_FLAG_RED 0x3
#endif

#ifndef SANITIZER_MEMORY_DEVICE_FLAG_ATOMIC
#define SANITIZER_MEMORY_DEVICE_FLAG_ATOMIC 0x4
#endif

#ifndef SANITIZER_MEMORY_DEVICE_FLAG_PREFETCH
#define SANITIZER_MEMORY_DEVICE_FLAG_PREFETCH 0x8
#endif

#ifndef SANITIZER_MEMORY_GLOBAL
#define SANITIZER_MEMORY_GLOBAL 0x10
#endif

#ifndef SANITIZER_MEMORY_SHARED
#define SANITIZER_MEMORY_SHARED 0x20
#endif

#ifndef SANITIZER_MEMORY_LOCAL
#define SANITIZER_MEMORY_LOCAL 0x40
#endif

namespace yosemite {

class memory_region{
public:
    memory_region() : start(0), end(0) {};
    memory_region(uint64_t start, uint64_t end) : start(start), end(end) {};
    ~memory_region() {};

    bool contains(uint64_t ptr) const {
        return ptr >= start && ptr < end;
    };

    bool operator==(const memory_region& other) const {
        return start == other.start && end == other.end;
    };

    bool operator<(const memory_region& other) const {
        // strict-weak-ordering: compare both start and end
        if (start != other.start) return start < other.start;
        return end < other.end;
    };

    uint64_t get_start() const {
        return start;
    };
    uint64_t get_end() const {
        return end;
    };

private:
    uint64_t start;
    uint64_t end;
};

class alignas(8) shadow_memory_entry{
public:
    shadow_memory_entry() {};
    ~shadow_memory_entry() {};
    // Packed representation: low 32 bits = (generation:8 | pc24:24),
    // high 32 bits = last_flat_thread_id.
    // Keeping a single 64-bit field avoids type-punning UB in atomic exchange.
    // packed == 0 means invalid/uninitialized (cold).
    uint64_t packed = 0;
};

class alignas(16) shared_shadow_memory_entry{
public:
    uint32_t pc_offset = 0;
    uint32_t flat_thread_id = 0; // [warp_id:lane_id] packed in lower 10 bits.
    uint32_t flat_block_id = 0;  // cta id
    uint32_t generation = 0;     // kernel generation
};

class shadow_memory{
public:
    shadow_memory(uint64_t size) 
    :_size(size),
    _size_celled((size + 3) / 4 * 4),
    _stride(_size_celled / 4),
    _entries_bytes(std::max<uint64_t>(1, _size_celled * sizeof(shadow_memory_entry))) {
        _shadow_memory_entries = static_cast<shadow_memory_entry*>(
            mmap(nullptr, _entries_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)
        );
        assert(_shadow_memory_entries != MAP_FAILED);

        printf("[SHADOW_MEMORY] Shadow memory entries: %lu\n", size);
        printf("[SHADOW_MEMORY] Shadow memory per entry size: %lu\n", sizeof(shadow_memory_entry));
        printf("[SHADOW_MEMORY] Shadow memory size: %lu\n", size*sizeof(shadow_memory_entry));
      };
    ~shadow_memory() {
        if (_shadow_memory_entries != nullptr && _shadow_memory_entries != MAP_FAILED) {
            munmap(_shadow_memory_entries, _entries_bytes);
            _shadow_memory_entries = nullptr;
        }
    }
    void reset_entries() {
        if (madvise(_shadow_memory_entries, _entries_bytes, MADV_DONTNEED) != 0) {
            std::memset(_shadow_memory_entries, 0, _entries_bytes);
        }
    };
    shadow_memory_entry& get_entry(uint64_t offset) {
        assert(offset < _size);
        //update layout: use offset/4 + offset%4 * _size/4 to make every 4 bytes adjacent in one cache line
        return _shadow_memory_entries[(offset/4) + (offset%4) * _stride];
        // return _shadow_memory_entries[offset];
    }
    uint64_t _size;
    uint64_t _size_celled;
    uint64_t _stride;
    uint64_t _entries_bytes;
    shadow_memory_entry* _shadow_memory_entries = nullptr;
};

struct worker_shared_shadow_state {
    static constexpr uint32_t k_invalid_object = 0xFFFFFFFFu;
    // local_cta_slot -> pooled object index, initialized on each kernel launch.
    std::vector<uint32_t> cta_slot_to_object;
    // Each object holds an array of shared_shadow_memory_entry entries indexed by 4-byte word offset.
    std::vector<shared_shadow_memory_entry*> object_entries;
    std::vector<uint64_t> object_owner_cta;
    std::vector<uint32_t> object_active_threads;
    std::vector<uint32_t> free_object_indices;

    uint64_t pool_miss_count = 0;
};

// ---- helpers shared by the tools -------------------------------------------

static inline const memory_region* find_memory_region_containing(
    const std::vector<memory_region>& regions,
    uint64_t addr
) {
    auto it = std::upper_bound(
        regions.begin(),
        regions.end(),
        addr,
        [](uint64_t value, const memory_region& region) {
            return value < region.get_start();
        }
    );
    if (it == regions.begin()) {
        return nullptr;
    }
    --it;
    return it->contains(addr) ? &(*it) : nullptr;
}

static inline uint32_t read_env_u32(const char* key, uint32_t default_value) {
    const char* raw = std::getenv(key);
    if (raw == nullptr) {
        return default_value;
    }
    char* end_ptr = nullptr;
    const unsigned long parsed = std::strtoul(raw, &end_ptr, 10);
    if (end_ptr == raw || *end_ptr != '\0') {
        return default_value;
    }
    if (parsed > std::numeric_limits<uint32_t>::max()) {
        return default_value;
    }
    return static_cast<uint32_t>(parsed);
}

// Per-CTA shared-memory shadow objects. A worker owns a fixed pool of objects;
// a CTA is mapped to one object on first touch and released once all of its
// threads have exited (BlockExit events). All three functions are pure
// bookkeeping on the worker-local state, so they are safe to call without locks
// from the worker that owns `state`.
static inline uint32_t acquire_shared_shadow_object(
    worker_shared_shadow_state& state,
    uint64_t cta_id,
    uint64_t worker_count,
    uint32_t block_thread_count
) {
    const uint64_t local_slot_u64 = cta_id / worker_count;
    if (local_slot_u64 >= state.cta_slot_to_object.size()) {
        state.cta_slot_to_object.resize(
            static_cast<size_t>(local_slot_u64 + 1u),
            worker_shared_shadow_state::k_invalid_object
        );
    }
    const uint32_t local_slot = static_cast<uint32_t>(local_slot_u64);
    const uint32_t mapped_object = state.cta_slot_to_object[local_slot];
    if (mapped_object != worker_shared_shadow_state::k_invalid_object) {
        return mapped_object;
    }
    if (state.free_object_indices.empty()) {
        state.pool_miss_count += 1;
        return std::numeric_limits<uint32_t>::max();
    }
    const uint32_t object_idx = state.free_object_indices.back();
    state.free_object_indices.pop_back();
    state.object_owner_cta[object_idx] = cta_id;
    state.object_active_threads[object_idx] = block_thread_count;
    state.cta_slot_to_object[local_slot] = object_idx;
    return object_idx;
}

static inline void release_shared_shadow_object(
    worker_shared_shadow_state& state,
    uint64_t cta_id,
    uint32_t exiting_threads,
    uint64_t worker_count
) {
    const uint64_t local_slot_u64 = cta_id / worker_count;
    if (local_slot_u64 >= state.cta_slot_to_object.size()) {
        return;
    }
    const uint32_t local_slot = static_cast<uint32_t>(local_slot_u64);
    const uint32_t object_idx = state.cta_slot_to_object[local_slot];
    if (object_idx == worker_shared_shadow_state::k_invalid_object) {
        return;
    }
    uint32_t& active_threads = state.object_active_threads[object_idx];
    if (active_threads > exiting_threads) {
        active_threads -= exiting_threads;
        return;
    }
    active_threads = 0;
    state.cta_slot_to_object[local_slot] = worker_shared_shadow_state::k_invalid_object;
    state.object_owner_cta[object_idx] = std::numeric_limits<uint64_t>::max();
    state.object_active_threads[object_idx] = 0u;
    state.free_object_indices.push_back(object_idx);
}

static inline shared_shadow_memory_entry& get_shared_shadow_entry(
    worker_shared_shadow_state& state,
    uint32_t object_idx,
    uint32_t addr,
    uint32_t bytes_per_object
) {
    assert(addr < bytes_per_object);
    (void)bytes_per_object;
    return state.object_entries[object_idx][addr];
}

}   // yosemite
#endif // YOSEMITE_TOOLS_SHADOW_MEMORY_H
