#pragma once

#include <cstdint>

constexpr uint32_t GPU_WARP_SIZE = 32;

constexpr uint32_t MAX_NUM_MEMORY_RANGES = 16384;
constexpr uint32_t MAX_NUM_TENSOR_RANGES = 65536;
constexpr uint32_t MEMORY_ACCESS_BUFFER_SIZE = 1048576*2;


enum class MemoryType
{
    Global,
    Shared,
    Local,
    BlockExit,
};

// Information regarding a memory access
struct MemoryAccess
{
    uint64_t addresses[GPU_WARP_SIZE];
    uint32_t accessSize;
    uint32_t flags;
    uint64_t ctaId;
    uint64_t pc;
    uint32_t warpId;
    uint32_t distinct_sector_count;    // for pc_dependency_analysis tool
    uint32_t active_mask;
    uint32_t unique_address_mask;
    MemoryType type;

    // copy constructor
    MemoryAccess(const MemoryAccess& other)
    {
        for (int i = 0; i < GPU_WARP_SIZE; i++)
        {
            addresses[i] = other.addresses[i];
        }
        accessSize = other.accessSize;
        flags = other.flags;
        ctaId = other.ctaId;
        warpId = other.warpId;
        distinct_sector_count = other.distinct_sector_count;
        type = other.type;
        pc = other.pc;
        active_mask = other.active_mask;
        unique_address_mask = other.unique_address_mask;
    }

    MemoryAccess() = default;

    ~MemoryAccess() = default;
};


struct MemoryRange
{
    uint64_t start;
    uint64_t end;

    bool operator<(const MemoryRange& other) const {
        return start < other.start;
    }

    bool operator==(const MemoryRange& other) const {
        return start == other.start && end == other.end;
    }
};


struct MemoryAccessState
{
    uint32_t size;
    MemoryRange start_end[MAX_NUM_MEMORY_RANGES];
    uint64_t touch[MAX_NUM_MEMORY_RANGES];
};

struct TensorAccessState
{
    uint32_t size;
    MemoryRange start_end[MAX_NUM_TENSOR_RANGES];
    uint64_t touch[MAX_NUM_TENSOR_RANGES];
};

struct DoorBell
{
    volatile bool full;
    volatile uint32_t num_threads;
};

// Main tracking structure that patches get as userdata
struct MemoryAccessTracker
{
    uint32_t currentEntry;
    uint32_t numEntries;
    uint64_t accessCount;
    uint64_t accessSize;
    uint64_t kernel_pc;
    int32_t target_block[3]; // target block to sample [x, y, z]
    DoorBell* doorBell;
    MemoryAccess* access_buffer;
    MemoryAccessState* access_state;
    TensorAccessState* tensor_access_state;
};
