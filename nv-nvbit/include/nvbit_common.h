#pragma once
#include <stdint.h>

#define GPU_WARP_SIZE_NVBIT (32)

#define HEX(x)                                                            \
    "0x" << std::setfill('0') << std::setw(16) << std::hex << (uint64_t)x \
         << std::dec


/* information collected in the instrumentation function and passed
 * on the channel from the GPU to the CPU */
typedef struct {
    uint64_t grid_launch_id;
    int cta_id_x;
    int cta_id_y;
    int cta_id_z;
    int sm_id;
    int warp_id;
    int opcode_id;
    int mem_type;
    int size;   // bytes
    bool is_read;
    bool is_write;
    uint64_t addrs[GPU_WARP_SIZE_NVBIT];
} nvbit_mem_access_t;
