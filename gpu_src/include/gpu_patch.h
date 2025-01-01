#pragma once

#include <cstdint>


#define MAX_ACTIVE_ALLOCATIONS 2048
#define MEMORY_ACCESS_BUFFER_SIZE 128
#define GPU_WARP_SIZE 32

enum class MemoryAccessType
{
    Global,
    Shared,
    Local,
};

// Information regarding a memory access
struct MemoryAccess
{
    uint64_t addresses[GPU_WARP_SIZE];
    uint32_t accessSize;
    uint32_t flags;
    uint64_t warpId;
    MemoryAccessType type;
};


struct MemoryRange{
    uint64_t start;
    uint64_t end;
};


struct MemoryAccessState{
    uint32_t size;
    MemoryRange start_end[MAX_ACTIVE_ALLOCATIONS];
    uint8_t touch[MAX_ACTIVE_ALLOCATIONS];
};

// Main tracking structure that patches get as userdata
struct MemoryAccessTracker
{
    uint32_t currentEntry;
    uint32_t maxEntry;
    uint32_t numThreads;
    uint64_t accessCount;
    MemoryAccess* accesses;
    MemoryAccessState* state;
};
