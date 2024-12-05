/* Copyright (c) 2019-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <cstdint>

#include <vector_types.h>

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
