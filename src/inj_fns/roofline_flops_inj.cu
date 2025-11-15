#include <stdint.h>
#include <stdio.h>

#include "utils/utils.h"

extern "C" __device__ __noinline__ void count_instrs(int num_instrs,
                                                     int count_warp_level,
                                                     uint64_t pcounter) {
    /* all the active threads will compute the active mask */
    const int active_mask = __ballot_sync(__activemask(), 1);

    /* each thread will get a lane id (get_lane_id is implemented in
     * utils/utils.h) */
    const int laneid = get_laneid();

    /* get the id of the first active thread */
    const int first_laneid = __ffs(active_mask) - 1;

    /* count all the active thread */
    const int num_threads = __popc(active_mask);

    /* only the first active thread will perform the atomic */
    if (first_laneid == laneid) {
        if (count_warp_level) {
            atomicAdd((unsigned long long*)pcounter, 1 * num_instrs);
        } else {
            atomicAdd((unsigned long long*)pcounter, num_threads * num_instrs);
        }
    }
}

extern "C" __device__ __noinline__ void count_pred_off(int predicate,
                                                       int count_warp_level,
                                                       uint64_t pcounter) {
    /* all the active threads will compute the active mask */
    const int active_mask = __ballot_sync(__activemask(), 1);

    /* each thread will get a lane id (get_lane_id is implemented in
     * utils/utils.h) */
    const int laneid = get_laneid();

    /* get the id of the first active thread */
    const int first_laneid = __ffs(active_mask) - 1;

    /* get predicate mask */
    const int predicate_mask = __ballot_sync(__activemask(), predicate);

    /* get mask of threads that have their predicate off */
    const int mask_off = active_mask ^ predicate_mask;

    /* count the number of threads that have their predicate off */
    const int num_threads_off = __popc(mask_off);

    /* only the first active thread updates the counter of predicated off
     * threads */
    if (first_laneid == laneid) {
        if (count_warp_level) {
            if (predicate_mask == 0) {
                atomicAdd((unsigned long long*)pcounter, 1);
            }
        } else {
            atomicAdd((unsigned long long*)pcounter, num_threads_off);
        }
    }
}
