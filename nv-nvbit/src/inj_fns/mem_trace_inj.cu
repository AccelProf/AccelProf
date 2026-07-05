#include <stdint.h>
#include <stdio.h>

#include "utils/utils.h"

/* for channel */
#include "utils/channel.hpp"

/* contains definition of the nvbit_mem_access_t structure */
#include "nvbit_common.h"

extern "C" __device__ __noinline__ void instrument_mem(int pred, int opcode_id,
                                                       int mem_type,
                                                       int size,
                                                       int is_read,
                                                       int is_write,
                                                       uint64_t addr,
                                                       uint64_t grid_launch_id,
                                                       uint64_t pchannel_dev) {
    /* if thread is predicated off, return */
    if (!pred) {
        return;
    }

    int active_mask = __ballot_sync(__activemask(), 1);
    const int laneid = get_laneid();
    const int first_laneid = __ffs(active_mask) - 1;

    nvbit_mem_access_t ma;

    /* collect memory address information from other threads */
    for (int i = 0; i < 32; i++) {
        ma.addrs[i] = __shfl_sync(active_mask, addr, i);
    }

    int4 cta = get_ctaid();
    ma.grid_launch_id = grid_launch_id;
    ma.cta_id_x = cta.x;
    ma.cta_id_y = cta.y;
    ma.cta_id_z = cta.z;
    ma.sm_id = get_smid();
    ma.warp_id = get_warpid();
    ma.opcode_id = opcode_id;
    ma.mem_type = mem_type;
    ma.size = size;
    ma.is_read = (bool)is_read;
    ma.is_write = (bool)is_write;

    /* first active lane pushes information on the channel */
    if (first_laneid == laneid) {
        ChannelDev* channel_dev = (ChannelDev*)pchannel_dev;
        channel_dev->push(&ma, sizeof(nvbit_mem_access_t));
    }
}
