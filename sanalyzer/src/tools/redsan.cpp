#include "tools/redsan.h"
#include "utils/helper.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <thread>
#include <vector>

using namespace yosemite;

namespace {

// Sanitizer reports atomic-store instructions with this bit (sanitizer_patching.h);
// the fallback flag set in shadow_memory.h does not carry it.
constexpr uint32_t k_flag_atomic_write_only = 0x100;

// Packed global shadow entry, 64 bits:
//   [63:32] flat thread id  (block<<10 | warp<<5 | lane)
//   [31:27] kernel generation (5 bits)
//   [26:24] RedsanAccess class (3 bits)
//   [23:0]  pc offset
// packed == 0 is a cold entry. A real entry always has a non-zero class, so
// generation 0 is a valid generation and the packed value never collides
// with the cold marker.
inline uint64_t pack_entry(uint8_t generation, RedsanAccess cls, uint32_t pc24, uint32_t flat_tid) {
    const uint32_t low = (static_cast<uint32_t>(generation & 0x1Fu) << 27)
                       | (static_cast<uint32_t>(cls) << 24)
                       | (pc24 & 0x00FFFFFFu);
    return (static_cast<uint64_t>(flat_tid) << 32) | static_cast<uint64_t>(low);
}
inline uint8_t      entry_generation(uint64_t p) { return static_cast<uint8_t>((p >> 27) & 0x1Fu); }
inline RedsanAccess entry_class(uint64_t p)      { return static_cast<RedsanAccess>((p >> 24) & 0x7u); }
inline uint32_t     entry_flat_tid(uint64_t p)   { return static_cast<uint32_t>(p >> 32); }

inline RedsanAccess classify_flags(uint32_t flags) {
    if (flags & SANITIZER_MEMORY_DEVICE_FLAG_PREFETCH) return RedsanAccess::Prefetch;
    if (flags & k_flag_atomic_write_only)              return RedsanAccess::AtomicWriteOnly;
    const bool r = (flags & SANITIZER_MEMORY_DEVICE_FLAG_READ) != 0;
    const bool w = (flags & SANITIZER_MEMORY_DEVICE_FLAG_WRITE) != 0;
    if (r && w) return RedsanAccess::AtomicRmw;
    if (w)      return RedsanAccess::Store;
    if (r)      return RedsanAccess::Load;
    return RedsanAccess::Other;
}
inline bool class_reads(RedsanAccess c)  { return c == RedsanAccess::Load  || c == RedsanAccess::AtomicRmw; }
inline bool class_writes(RedsanAccess c) { return c == RedsanAccess::Store || c == RedsanAccess::AtomicRmw
                                               || c == RedsanAccess::AtomicWriteOnly; }

const char* class_name(RedsanAccess c) {
    switch (c) {
        case RedsanAccess::Load:            return "load";
        case RedsanAccess::Store:           return "store";
        case RedsanAccess::AtomicRmw:       return "atomic";
        case RedsanAccess::Prefetch:        return "prefetch";
        case RedsanAccess::AtomicWriteOnly: return "atomic_store";
        default:                            return "other";
    }
}
const char* scope_name(uint32_t scope) {
    if (scope & SANITIZER_MEMORY_SHARED) return "shared";
    if (scope & SANITIZER_MEMORY_LOCAL)  return "local";
    return "global";
}
const char* category_name(double rate) {
    if (rate >= 1.0) return "fully_redundant";
    if (rate >= 0.5) return "partially_redundant";
    if (rate > 0.0)  return "slightly_redundant";
    return "non_redundant";
}
std::string hex_u32(uint32_t v) {
    std::ostringstream oss;
    oss << "0x" << std::hex << v;
    return oss.str();
}
std::string csv_quote(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') out += "\"\"";
        else out += c;
    }
    out += "\"";
    return out;
}

} // namespace


Redsan::Redsan() : Tool(REDSAN) {
    const char* torch_prof = std::getenv("TORCH_PROFILE_ENABLED");
    if (torch_prof && std::string(torch_prof) == "1") {
        fprintf(stdout, "Enabling torch profiler in Redsan.\n");
        _torch_enabled = true;
    }

    const char* env_app_name = std::getenv("YOSEMITE_APP_NAME");
    if (env_app_name != nullptr) {
        output_directory = "redsan_" + std::string(env_app_name) + "_" + get_current_date_n_time();
    } else {
        output_directory = "redsan_" + get_current_date_n_time();
    }
    output_directory = resolve_output_path(output_directory);
    check_folder_existance(output_directory);

    summary_path = output_directory + "/summary.csv";
    {
        std::ofstream sout(summary_path);
        sout << "kernel_id,kernel_name,pc_count,fully_redundant,partially_redundant,"
                "slightly_redundant,non_redundant,total_accesses,redundant_accesses\n";
    }

    _worker_count = std::max(1u, read_env_u32("YOSEMITE_WORKER_COUNT", std::thread::hardware_concurrency()));
    const uint32_t sm_count = read_env_u32("YOSEMITE_GPU_SM_COUNT", 128);
    const uint32_t max_active_blocks_per_sm = read_env_u32("YOSEMITE_GPU_MAX_ACTIVE_BLOCKS_PER_SM", 24);
    const uint32_t pool_slack_percent = read_env_u32("YOSEMITE_SHARED_SHADOW_POOL_SLACK_PERCENT", 150);
    const uint64_t total_block_capacity =
        static_cast<uint64_t>(sm_count) * static_cast<uint64_t>(max_active_blocks_per_sm);
    const uint64_t slack_block_capacity =
        (total_block_capacity * static_cast<uint64_t>(pool_slack_percent) + 99ull) / 100ull;
    _shared_shadow_object_cap_per_worker =
        static_cast<uint32_t>(std::max<uint64_t>(32ull, (slack_block_capacity + _worker_count - 1) / _worker_count));
    _shared_shadow_bytes_per_object = read_env_u32("YOSEMITE_GPU_MAX_SHARED_MEMORY_PER_BLOCK", 102400u);
    if (_shared_shadow_bytes_per_object == 0) {
        _shared_shadow_bytes_per_object = 1;
    }

    _worker_shadow_memory_shared.resize(_worker_count);
    for (auto& worker_state : _worker_shadow_memory_shared) {
        worker_state.object_entries.resize(_shared_shadow_object_cap_per_worker, nullptr);
        worker_state.object_owner_cta.assign(_shared_shadow_object_cap_per_worker, std::numeric_limits<uint64_t>::max());
        worker_state.object_active_threads.assign(_shared_shadow_object_cap_per_worker, 0u);
        worker_state.free_object_indices.reserve(_shared_shadow_object_cap_per_worker);
        for (uint32_t idx = 0; idx < _shared_shadow_object_cap_per_worker; ++idx) {
            worker_state.free_object_indices.push_back(_shared_shadow_object_cap_per_worker - 1u - idx);
            shared_shadow_memory_entry* entries = static_cast<shared_shadow_memory_entry*>(
                mmap(nullptr,
                     static_cast<size_t>(_shared_shadow_bytes_per_object) * sizeof(shared_shadow_memory_entry),
                     PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
            assert(entries != MAP_FAILED);
            worker_state.object_entries[idx] = entries;
        }
    }
    _job_worker_trace_indices.resize(_worker_count);
    _job_worker_pc_stats.resize(_worker_count);
    _workers.reserve(_worker_count);
    for (uint64_t worker_idx = 0; worker_idx < _worker_count; ++worker_idx) {
        _workers.emplace_back(&Redsan::worker_loop, this, worker_idx);
    }
    printf("[REDSAN] Initialized with %lu worker(s)\n", _worker_count);
}


Redsan::~Redsan() {
    {
        std::lock_guard<std::mutex> guard(_worker_pool_mutex);
        _worker_pool_shutdown = true;
        ++_worker_job_generation;
    }
    _worker_pool_cv.notify_all();
    for (auto& worker : _workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    for (auto& worker_state : _worker_shadow_memory_shared) {
        for (auto* entries : worker_state.object_entries) {
            if (entries != nullptr) {
                munmap(entries,
                       static_cast<size_t>(_shared_shadow_bytes_per_object) * sizeof(shared_shadow_memory_entry));
            }
        }
    }
}


void Redsan::kernel_start_callback(std::shared_ptr<KernelLaunch_t> kernel) {
    kernel->kernel_id = kernel_id++;
    _shared_kernel_generation = kernel->kernel_id + 1u;
    _current_kernel_cta_count = kernel->grid_cta_count;
    _current_block_thread_count = kernel->block_thread_count;
    kernel_events.emplace(_timer.get(), kernel);
    _pc_stats.clear();
    _unknown_region_shadow.clear();
    for (uint64_t worker_idx = 0; worker_idx < _worker_count; ++worker_idx) {
        auto& worker_state = _worker_shadow_memory_shared[worker_idx];
        worker_state.pool_miss_count = 0;
        uint64_t worker_cta_slots = 0;
        if (_current_kernel_cta_count > worker_idx) {
            worker_cta_slots = (_current_kernel_cta_count + _worker_count - 1u - worker_idx) / _worker_count;
        }
        worker_state.cta_slot_to_object.assign(
            static_cast<size_t>(worker_cta_slots), worker_shared_shadow_state::k_invalid_object);
    }
    // 5-bit generation: wrap every 32 kernels and drop every shadow page so
    // entries from 32 launches ago cannot be mistaken for this launch.
    _kernel_generation = static_cast<uint8_t>((_kernel_generation + 1u) & 0x1Fu);
    if (_kernel_generation == 0) {
        for (auto& shadow_memory_iter : _shadow_memories) {
            shadow_memory_iter.second->reset_entries();
        }
        printf("[REDSAN] Shadow generation wrapped, resetting entries\n");
    }
    _timer.increment(true);
}


void Redsan::kernel_end_callback(std::shared_ptr<KernelEnd_t> kernel) {
    auto evt = std::prev(kernel_events.end())->second;
    evt->end_time = _timer.get();
    kernel_trace_flush(evt);
    _timer.increment(true);
}


void Redsan::mem_alloc_callback(std::shared_ptr<MemAlloc_t> mem) {
    alloc_events.emplace(_timer.get(), mem);
    active_memories.emplace(mem->addr, mem);
    memory_region region((uint64_t)mem->addr, (uint64_t)(mem->addr + mem->size));
    _memory_regions.insert(
        std::lower_bound(_memory_regions.begin(), _memory_regions.end(), region), region);
    _shadow_memories.emplace(region, std::make_unique<shadow_memory>(mem->size));
    printf("[REDSAN] Allocating shadow memory for memory region: %p - %p, size: %lu\n",
           (void*)region.get_start(), (void*)region.get_end(), mem->size);
    _timer.increment(true);
}


void Redsan::mem_free_callback(std::shared_ptr<MemFree_t> mem) {
    auto it = active_memories.find(mem->addr);
    if (it == active_memories.end()) {
        printf("[REDSAN] Memory free callback: memory %lu not found, it is not regularly allocated. Active memories: %ld\n",
               mem->addr, active_memories.size());
        return;
    }
    const uint64_t sz = it->second->size;
    active_memories.erase(it);
    memory_region r((uint64_t)mem->addr, (uint64_t)mem->addr + sz);
    auto vit = std::lower_bound(_memory_regions.begin(), _memory_regions.end(), r);
    if (vit != _memory_regions.end() && *vit == r) _memory_regions.erase(vit);
    _shadow_memories.erase(r);
    printf("[REDSAN] Freeing shadow memory for memory region: %p - %p, size: %lu\n",
           (void*)r.get_start(), (void*)r.get_end(), sz);
    _timer.increment(true);
}


void Redsan::ten_alloc_callback(std::shared_ptr<TenAlloc_t> ten) {
    tensor_events.emplace(_timer.get(), ten);
    active_tensors.emplace(ten->addr, ten);
    _timer.increment(true);
}


void Redsan::ten_free_callback(std::shared_ptr<TenFree_t> ten) {
    auto it = active_tensors.find(ten->addr);
    if (it != active_tensors.end()) {
        active_tensors.erase(it);
    }
    _timer.increment(true);
}


// The three rules from the paper, applied to one recorded (last) access and
// the current one at the same address. Callers only invoke this when the two
// accesses come from the same thread and the same kernel generation.
void Redsan::classify(RedsanAccess last, RedsanAccess cur, RedsanPcStats& st) {
    if (last == cur) {
        // Same instruction class repeated by the same thread with nobody
        // touching the address in between: the original RedSan counts any such
        // repeat as redundant (flags compared for equality).
        st.redundant_accesses += 1;
        if (cur == RedsanAccess::Store || cur == RedsanAccess::AtomicWriteOnly) {
            st.dead_store += 1;          // the *earlier* store was never read
        } else if (cur == RedsanAccess::Load) {
            st.dead_load += 1;           // reloads a value already in a register
        } else {
            st.other_equal += 1;         // atomic->atomic, prefetch->prefetch
        }
        return;
    }
    if (class_writes(last) && class_reads(cur)) {
        st.redundant_accesses += 1;
        st.load_after_store += 1;        // the value just stored is still in a register
    }
}


void Redsan::unit_access(
    uint64_t offset_in_region,
    uint32_t pc_offset,
    RedsanAccess cls,
    uint32_t current_flat_thread_id,
    const memory_region& region,
    int access_size,
    RedsanPcStats& st
) {
    auto shadow_it = _shadow_memories.find(region);
    if (shadow_it == _shadow_memories.end()) {
        printf("[REDSAN] shadow memory not found for memory region: %lu - %lu\n",
               region.get_start(), region.get_end());
        return;
    }
    auto& shadow = *(shadow_it->second);
    const uint64_t new_packed = pack_entry(_kernel_generation, cls, pc_offset, current_flat_thread_id);

    // One record per 4-byte segment, exactly like the original: an 8-byte
    // access is two accesses at addr and addr+4, a <=4-byte access is one.
    for (int i = 0; i < access_size; i += 4) {
        const uint64_t addr = offset_in_region + static_cast<uint64_t>(i);
        if (addr >= shadow._size) {
            break;
        }
        st.total_accesses += 1;
        auto& entry = shadow.get_entry(addr);
        const uint64_t old_packed = __atomic_exchange_n(&entry.packed, new_packed, __ATOMIC_ACQ_REL);
        if (old_packed == 0) continue;                                        // cold
        if (entry_generation(old_packed) != (_kernel_generation & 0x1Fu)) continue; // previous kernel
        if (entry_flat_tid(old_packed) != current_flat_thread_id) continue;   // other thread
        classify(entry_class(old_packed), cls, st);
    }
}


void Redsan::unit_access_unknown(
    uint64_t abs_addr,
    uint32_t pc_offset,
    RedsanAccess cls,
    uint32_t current_flat_thread_id,
    int access_size,
    RedsanPcStats& st
) {
    const uint64_t new_packed = pack_entry(_kernel_generation, cls, pc_offset, current_flat_thread_id);
    for (int i = 0; i < access_size; i += 4) {
        const uint64_t addr = abs_addr + static_cast<uint64_t>(i);
        st.total_accesses += 1;
        uint64_t old_packed = 0;
        const bool inserted = _unknown_region_shadow.try_emplace_l(
            addr,
            [&](auto& kv) { old_packed = kv.second; kv.second = new_packed; },
            new_packed);
        if (inserted || old_packed == 0) continue;
        if (entry_generation(old_packed) != (_kernel_generation & 0x1Fu)) continue;
        if (entry_flat_tid(old_packed) != current_flat_thread_id) continue;
        classify(entry_class(old_packed), cls, st);
    }
}


void Redsan::unit_access_shared(
    uint64_t ptr,
    uint32_t pc_offset,
    RedsanAccess cls,
    uint32_t object_idx,
    uint64_t current_block_id,
    uint32_t current_thread_in_block,
    int access_size,
    RedsanPcStats& st,
    worker_shared_shadow_state& shared_state
) {
    const uint32_t base_addr_low32 = static_cast<uint32_t>(ptr & 0xFFFFFFFFull);
    // The shared entry keeps the class in the top byte of pc_offset.
    const uint32_t encoded_pc = (static_cast<uint32_t>(cls) << 24) | (pc_offset & 0x00FFFFFFu);

    for (int i = 0; i < access_size; i += 4) {
        const uint32_t addr = base_addr_low32 + static_cast<uint32_t>(i);
        st.total_accesses += 1;
        if (addr >= _shared_shadow_bytes_per_object) {
            continue;
        }
        auto& entry = get_shared_shadow_entry(shared_state, object_idx, addr, _shared_shadow_bytes_per_object);
        const bool cold = (entry.generation != _shared_kernel_generation)
                       || (entry.flat_block_id != static_cast<uint32_t>(current_block_id));
        const bool same_thread = !cold && (entry.flat_thread_id == current_thread_in_block);
        const RedsanAccess last = static_cast<RedsanAccess>((entry.pc_offset >> 24) & 0x7u);

        entry.pc_offset = encoded_pc;
        entry.flat_thread_id = current_thread_in_block;
        entry.flat_block_id = static_cast<uint32_t>(current_block_id);
        entry.generation = _shared_kernel_generation;

        if (same_thread) {
            classify(last, cls, st);
        }
    }
}


void Redsan::worker_loop(uint64_t worker_idx) {
    uint64_t seen_generation = 0;
    while (true) {
        uint64_t current_generation = 0;
        {
            std::unique_lock<std::mutex> lock(_worker_pool_mutex);
            _worker_pool_cv.wait(lock, [&]{
                return _worker_pool_shutdown || _worker_job_generation > seen_generation;
            });
            if (_worker_pool_shutdown) {
                return;
            }
            current_generation = _worker_job_generation;
        }

        auto& local_stats = _job_worker_pc_stats[worker_idx];
        auto& shared_state = _worker_shadow_memory_shared[worker_idx];
        const auto& trace_indices = _job_worker_trace_indices[worker_idx];

        for (uint64_t i : trace_indices) {
            const MemoryAccess& trace = _job_accesses_buffer[i];
            const uint32_t pc_offset = (trace.pc & 0x00FFFFFFu);
            const uint32_t active_mask = trace.active_mask;

            if (trace.type == MemoryType::BlockExit) {
                release_shared_shadow_object(
                    shared_state, trace.ctaId, __builtin_popcount(active_mask), _worker_count);
                continue;
            }

            uint32_t scope = 0;
            switch (trace.type) {
                case MemoryType::Global: scope = SANITIZER_MEMORY_GLOBAL; break;
                case MemoryType::Shared: scope = SANITIZER_MEMORY_SHARED; break;
                case MemoryType::Local:  scope = SANITIZER_MEMORY_LOCAL;  break;
                default:
                    printf("[REDSAN] unknown memory type\n");
                    continue;
            }
            const RedsanAccess cls = classify_flags(trace.flags);
            const int access_size = static_cast<int>(trace.accessSize);

            RedsanPcStats& st = local_stats[pc_offset];
            if (st.flags == 0 && st.scope == 0) {
                st.flags = trace.flags;
                st.scope = scope;
                st.access_size = trace.accessSize;
            }

            if (active_mask == 0) {
                continue;
            }

            // Every active lane, in lane order, so the record left behind is
            // the highest lane's - the same sequential semantics as the
            // original per-thread table. Lanes are distinct threads, so two
            // lanes hitting one address is never redundant by itself.
            uint32_t object_idx = std::numeric_limits<uint32_t>::max();
            if (trace.type == MemoryType::Shared) {
                object_idx = acquire_shared_shadow_object(
                    shared_state, trace.ctaId, _worker_count, _current_block_thread_count);
                if (object_idx == std::numeric_limits<uint32_t>::max()) {
                    // Pool exhausted: count the accesses, cannot track history.
                    const uint32_t segments = (trace.accessSize + 3u) / 4u;
                    st.total_accesses += static_cast<uint64_t>(segments) * __builtin_popcount(active_mask);
                    continue;
                }
            }

            const memory_region* region_ptr = nullptr;
            if (trace.type == MemoryType::Global) {
                const uint32_t first_lane = static_cast<uint32_t>(__builtin_ctz(active_mask));
                region_ptr = find_memory_region_containing(_memory_regions, trace.addresses[first_lane]);
            }

            uint32_t remaining = active_mask;
            while (remaining != 0) {
                const uint32_t lane = static_cast<uint32_t>(__builtin_ctz(remaining));
                remaining &= (remaining - 1);
                const uint32_t thread_in_block = (trace.warpId << 5) | lane;
                const uint32_t flat_tid = static_cast<uint32_t>(
                    (trace.ctaId << 10) | static_cast<uint64_t>(thread_in_block));
                const uint64_t address = trace.addresses[lane];

                switch (trace.type) {
                    case MemoryType::Shared:
                        unit_access_shared(address, pc_offset, cls, object_idx, trace.ctaId,
                                           thread_in_block, access_size, st, shared_state);
                        break;
                    case MemoryType::Global:
                        if (region_ptr != nullptr && region_ptr->contains(address)) {
                            unit_access(address - region_ptr->get_start(), pc_offset, cls, flat_tid,
                                        *region_ptr, access_size, st);
                        } else {
                            unit_access_unknown(address, pc_offset, cls, flat_tid, access_size, st);
                        }
                        break;
                    case MemoryType::Local:
                        // The patch tags local addresses with the in-block thread id
                        // in the top bits, so per-thread windows do not alias here.
                        unit_access_unknown(address, pc_offset, cls, flat_tid, access_size, st);
                        break;
                    default:
                        break;
                }
            }
        }

        {
            std::lock_guard<std::mutex> guard(_worker_pool_mutex);
            seen_generation = current_generation;
            if (!trace_indices.empty()) {
                assert(_worker_pending_jobs > 0);
                _worker_pending_jobs -= 1;
                if (_worker_pending_jobs == 0) {
                    _worker_pool_done_cv.notify_one();
                }
            }
        }
    }
}


void Redsan::gpu_data_analysis(void* data, uint64_t size) {
    printf("[REDSAN] GPU data analysis called with size = %lu\n", size);
    MemoryAccess* accesses_buffer = (MemoryAccess*)data;
    if (size == 0) {
        return;
    }

    for (uint64_t worker_idx = 0; worker_idx < _worker_count; ++worker_idx) {
        _job_worker_trace_indices[worker_idx].clear();
        _job_worker_pc_stats[worker_idx].clear();
        _job_worker_trace_indices[worker_idx].reserve((size / _worker_count) + 1);
    }
    // Stable assignment by block id keeps intra-block (hence intra-thread)
    // trace order, which is all the same-thread rules depend on.
    for (uint64_t i = 0; i < size; ++i) {
        const uint64_t worker_idx = accesses_buffer[i].ctaId % _worker_count;
        _job_worker_trace_indices[worker_idx].push_back(i);
    }

    uint64_t pending_jobs = 0;
    for (uint64_t worker_idx = 0; worker_idx < _worker_count; ++worker_idx) {
        if (!_job_worker_trace_indices[worker_idx].empty()) {
            pending_jobs += 1;
        }
    }
    if (pending_jobs == 0) {
        return;
    }
    {
        std::lock_guard<std::mutex> guard(_worker_pool_mutex);
        _job_accesses_buffer = accesses_buffer;
        _worker_pending_jobs = pending_jobs;
        ++_worker_job_generation;
    }
    _worker_pool_cv.notify_all();
    {
        std::unique_lock<std::mutex> lock(_worker_pool_mutex);
        _worker_pool_done_cv.wait(lock, [&]{ return _worker_pending_jobs == 0; });
    }

    for (auto& local_map : _job_worker_pc_stats) {
        for (auto& [pc, local] : local_map) {
            auto& g = _pc_stats[pc];
            if (g.flags == 0 && g.scope == 0) {
                g.flags = local.flags;
                g.scope = local.scope;
                g.access_size = local.access_size;
            }
            g.total_accesses     += local.total_accesses;
            g.redundant_accesses += local.redundant_accesses;
            g.dead_store         += local.dead_store;
            g.dead_load          += local.dead_load;
            g.load_after_store   += local.load_after_store;
            g.other_equal        += local.other_equal;
        }
    }
}


void Redsan::kernel_trace_flush(std::shared_ptr<KernelLaunch_t> kernel) {
    const std::string csv_path = output_directory + "/kernel_" + std::to_string(kernel->kernel_id) + ".csv";
    std::ofstream out(csv_path);
    out << "# tool=redsan kernel_id=" << kernel->kernel_id
        << " kernel_name=" << csv_quote(kernel->kernel_name)
        << " device_id=" << kernel->device_id
        << " grid=[" << kernel->grid_dim_x << "," << kernel->grid_dim_y << "," << kernel->grid_dim_z << "]"
        << " block=[" << kernel->block_dim_x << "," << kernel->block_dim_y << "," << kernel->block_dim_z << "]"
        << " sample_stride_bytes=4\n";
    out << "pc,pc_hex,type,scope,access_size,total_accesses,redundant_accesses,"
           "dead_store,dead_load,load_after_store,other_equal,redundancy_rate,category\n";

    std::vector<uint32_t> pcs;
    pcs.reserve(_pc_stats.size());
    for (const auto& kv : _pc_stats) pcs.push_back(kv.first);
    std::sort(pcs.begin(), pcs.end());

    uint64_t fully = 0, partially = 0, slightly = 0, non = 0;
    uint64_t total = 0, redundant = 0;
    out << std::fixed << std::setprecision(6);
    for (uint32_t pc : pcs) {
        const RedsanPcStats& st = _pc_stats[pc];
        const double rate = (st.total_accesses == 0)
            ? 0.0
            : static_cast<double>(st.redundant_accesses) / static_cast<double>(st.total_accesses);
        const char* category = category_name(rate);
        if (rate >= 1.0) fully++; else if (rate >= 0.5) partially++; else if (rate > 0.0) slightly++; else non++;
        total += st.total_accesses;
        redundant += st.redundant_accesses;
        out << pc << "," << hex_u32(pc) << ","
            << class_name(classify_flags(st.flags)) << "," << scope_name(st.scope) << ","
            << st.access_size << ","
            << st.total_accesses << "," << st.redundant_accesses << ","
            << st.dead_store << "," << st.dead_load << "," << st.load_after_store << "," << st.other_equal << ","
            << rate << "," << category << "\n";
    }
    out.close();

    {
        std::ofstream sout(summary_path, std::ios::app);
        sout << kernel->kernel_id << "," << csv_quote(kernel->kernel_name) << ","
             << pcs.size() << "," << fully << "," << partially << "," << slightly << "," << non << ","
             << total << "," << redundant << "\n";
    }
    printf("[REDSAN] kernel %u: %zu pcs, fully_redundant=%lu partially_redundant=%lu slightly_redundant=%lu "
           "non_redundant=%lu (accesses=%lu redundant=%lu)\n",
           kernel->kernel_id, pcs.size(), fully, partially, slightly, non, total, redundant);
    printf("Dumping redsan report to %s\n", csv_path.c_str());
}


void Redsan::evt_callback(EventPtr_t evt) {
    switch (evt->evt_type) {
        case EventType_KERNEL_LAUNCH:
            kernel_start_callback(std::dynamic_pointer_cast<KernelLaunch_t>(evt));
            break;
        case EventType_KERNEL_END:
            kernel_end_callback(std::dynamic_pointer_cast<KernelEnd_t>(evt));
            break;
        case EventType_MEM_ALLOC:
            mem_alloc_callback(std::dynamic_pointer_cast<MemAlloc_t>(evt));
            break;
        case EventType_MEM_FREE:
            mem_free_callback(std::dynamic_pointer_cast<MemFree_t>(evt));
            break;
        case EventType_TEN_ALLOC:
            ten_alloc_callback(std::dynamic_pointer_cast<TenAlloc_t>(evt));
            break;
        case EventType_TEN_FREE:
            ten_free_callback(std::dynamic_pointer_cast<TenFree_t>(evt));
            break;
        default:
            break;
    }
}


void Redsan::flush() {
}
