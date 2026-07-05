#include "backend/nvbit_app_metric.h"

#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <unordered_set>

/* nvbit interface file */
#include "nvbit.h"

/* nvbit utility functions */
#include "utils/utils.h"

#include "sanalyzer.h"


namespace yosemite_app_metric {

/* kernel id counter, maintained in system memory */
uint32_t kernel_id = 0;

/* total instruction counter, maintained in system memory, incremented by
 * "counter" every time a kernel completes  */
uint64_t tot_app_mem_refs = 0;

/* kernel instruction counter, updated by the GPU threads */
__managed__ uint64_t counter = 0;
__managed__ uint64_t counter_pred_off = 0;

/* global control variables for this tool */
uint32_t start_grid_num = 0;
uint32_t end_grid_num = UINT32_MAX;
int verbose = 0;
int count_warp_level = 0;
int exclude_pred_off = 0;
int active_from_start = 1;
bool mangled = false;

/* used to select region of insterest when active from start is off */
bool active_region = true;

/* a pthread mutex, used to prevent multiple kernels to run concurrently and
 * therefore to "corrupt" the counter variable */
pthread_mutex_t mutex;

/* nvbit_at_init() is executed as soon as the nvbit tool is loaded. We
 * typically do initializations in this call. In this case for instance we get
 * some environment variables values which we use as input arguments to the tool
 */
void app_metric_nvbit_at_init() {
    /* just make sure all managed variables are allocated on GPU */
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);

    /* we get some environment variables that are going to be use to selectively
     * instrument (within a interval of kernel indexes and instructions). By
     * default we instrument everything. */
    GET_VAR_INT(start_grid_num, "START_GRID_NUM", 0,
                "Beginning of the kernel gird launch interval where to apply "
                "instrumentation");
    GET_VAR_INT(end_grid_num, "END_GRID_NUM", UINT32_MAX,
                "End of the kernel grid launch interval where to apply "
                "instrumentation");
    GET_VAR_INT(count_warp_level, "COUNT_WARP_LEVEL", 0,
                "Count warp level or thread level instructions");
    GET_VAR_INT(exclude_pred_off, "EXCLUDE_PRED_OFF", 1,
                "Exclude predicated off instruction from count");
    GET_VAR_INT(
        active_from_start, "ACTIVE_FROM_START", 1,
        "Start instruction counting from start or wait for cuProfilerStart "
        "and cuProfilerStop");
    GET_VAR_INT(mangled, "MANGLED_NAMES", 1,
                "Print kernel names mangled or not");
    GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");
    if (active_from_start == 0) {
        active_region = false;
    }

    std::string pad(100, '-');
    printf("%s\n", pad.c_str());
}

void app_metric_nvbit_tool_init(CUcontext ctx) {
}

/* Set used to avoid re-instrumenting the same functions multiple times */
std::unordered_set<CUfunction> already_instrumented;

void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
    /* Get related functions of the kernel (device function that can be
     * called by the kernel) */
    std::vector<CUfunction> related_functions =
        nvbit_get_related_functions(ctx, func);

    /* add kernel itself to the related function vector */
    related_functions.push_back(func);

    /* iterate on function */
    for (auto f : related_functions) {
        /* "recording" function was instrumented, if set insertion failed
         * we have already encountered this function */
        if (!already_instrumented.insert(f).second) {
            continue;
        }

        /* Get the static control flow graph of instruction */
        const CFG_t &cfg = nvbit_get_CFG(ctx, f);
        if (cfg.is_degenerate) {
            printf(
                "Warning: Function %s is degenerated, we can't compute basic "
                "blocks statically",
                nvbit_get_func_name(ctx, f));
        }

        if (verbose) {
            printf("Function %s\n", nvbit_get_func_name(ctx, f));
            /* print */
            int cnt = 0;
            for (auto &bb : cfg.bbs) {
                printf("Basic block id %d - num instructions %ld\n", cnt++,
                       bb->instrs.size());
                for (auto &i : bb->instrs) {
                    i->print(" ");
                }
            }
        }

        if (verbose) {
            printf("inspecting %s - number basic blocks %ld\n",
                   nvbit_get_func_name(ctx, f), cfg.bbs.size());
        }

        /* Iterate on basic block and inject the first instruction */
        for (auto &bb : cfg.bbs) {
            uint32_t mem_refs = 0;
            for (auto i : bb->instrs) {
                if (i->getMemorySpace() == InstrType::MemorySpace::GLOBAL) {
                    mem_refs++;
                }
            }

            Instr *i = bb->instrs[0];
            /* inject device function */
            nvbit_insert_call(i, "count_mem_access", IPOINT_BEFORE);
            /* add size of basic block in number of mem refs */
            nvbit_add_call_arg_const_val32(i, mem_refs);
            /* add count warp level option */
            nvbit_add_call_arg_const_val32(i, count_warp_level);
            /* add pointer to counter location */
            nvbit_add_call_arg_const_val64(i, (uint64_t)&counter);
            if (verbose) {
                i->print("Inject count_instr before - ");
            }
        }

        if (exclude_pred_off) {
            /* iterate on instructions */
            for (auto i : nvbit_get_instrs(ctx, f)) {
                /* inject only if instruction has predicate */
                if (i->hasPred()) {
                    /* inject function */
                    nvbit_insert_call(i, "count_pred_off_mem_access", IPOINT_BEFORE);
                    /* add guard predicate as argument */
                    nvbit_add_call_arg_guard_pred_val(i);
                    /* add count warp level option */
                    nvbit_add_call_arg_const_val32(i, count_warp_level);

                    /* add pointer to counter predicate off location */
                    nvbit_add_call_arg_const_val64(i,
                                                   (uint64_t)&counter_pred_off);
                    if (verbose) {
                        i->print("Inject count func before - ");
                    }
                }
            }
        }
    }
}

/* This call-back is triggered every time a CUDA driver call is encountered.
 * Here we can look for a particular CUDA driver call by checking at the
 * call back ids  which are defined in tools_cuda_api_meta.h.
 * This call back is triggered bith at entry and at exit of each CUDA driver
 * call, is_exit=0 is entry, is_exit=1 is exit.
 * */
void app_metric_nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
    /* Identify all the possible CUDA launch events */
    if (cbid == API_CUDA_cuLaunch || cbid == API_CUDA_cuLaunchKernel_ptsz ||
        cbid == API_CUDA_cuLaunchGrid || cbid == API_CUDA_cuLaunchGridAsync ||
        cbid == API_CUDA_cuLaunchKernel ||
        cbid == API_CUDA_cuLaunchKernelEx ||
        cbid == API_CUDA_cuLaunchKernelEx_ptsz) {
        /* cast params to launch parameter based on cbid since if we are here
         * we know these are the right parameters types */
        CUfunction func;
        if (cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
            cbid == API_CUDA_cuLaunchKernelEx) {
            cuLaunchKernelEx_params* p = (cuLaunchKernelEx_params*)params;
            func = p->f;
        } else {
            cuLaunchKernel_params* p = (cuLaunchKernel_params*)params;
            func = p->f;
        }

        if (!is_exit) {
            /* if we are entering in a kernel launch:
             * 1. Lock the mutex to prevent multiple kernels to run concurrently
             * (overriding the counter) in case the user application does that
             * 2. Instrument the function if needed
             * 3. Select if we want to run the instrumented or original
             * version of the kernel
             * 4. Reset the kernel instruction counter */

            pthread_mutex_lock(&mutex);
            instrument_function_if_needed(ctx, func);

            if (active_from_start) {
                if (kernel_id >= start_grid_num && kernel_id < end_grid_num) {
                    active_region = true;
                } else {
                    active_region = false;
                }
            }

            if (active_region) {
                nvbit_enable_instrumented(ctx, func, true);
            } else {
                nvbit_enable_instrumented(ctx, func, false);
            }

            /* get function name and pc */
            const char* func_name = nvbit_get_func_name(ctx, func);
            uint64_t pc = nvbit_get_func_addr(func);

            yosemite_kernel_start_callback((std::string)func_name, 0 /* device id */);

            if (cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
                cbid == API_CUDA_cuLaunchKernelEx) {
                cuLaunchKernelEx_params* p = (cuLaunchKernelEx_params*)params;
                printf(
                    "LAUNCH KERNEL: %s\n"
                    " - grid size (%d,%d,%d) - block size (%d,%d,%d)"
                    " - CTX 0x%016lx - stream id %ld - Kernel pc 0x%016lx\n",
                    func_name,
                    p->config->gridDimX, p->config->gridDimY, p->config->gridDimZ,
                    p->config->blockDimX, p->config->blockDimY, p->config->blockDimZ,
                    (uint64_t)ctx, (uint64_t)p->config->hStream, pc);
                fflush(stdout);
            } else {
                cuLaunchKernel_params* p = (cuLaunchKernel_params*)params;
                printf(
                    "LAUNCH KERNEL: %s\n"
                    " - grid size (%d,%d,%d) - block size (%d,%d,%d)"
                    " - CTX 0x%016lx - stream id %ld - Kernel pc 0x%016lx\n",
                    func_name,
                    p->gridDimX, p->gridDimY, p->gridDimZ,
                    p->blockDimX, p->blockDimY, p->blockDimZ,
                    (uint64_t)ctx, (uint64_t)p->hStream, pc);
                fflush(stdout);
            }

            counter = 0;
            counter_pred_off = 0;
        } else {
            /* if we are exiting a kernel launch:
             * 1. Wait until the kernel is completed using
             * cudaDeviceSynchronize()
             * 2. Get number of thread blocks in the kernel
             * 3. Print the thread instruction counters
             * 4. Release the lock*/
            CUDA_SAFECALL(cudaDeviceSynchronize());
            uint64_t kernel_refs = counter - counter_pred_off;
            tot_app_mem_refs += kernel_refs;
            int num_ctas = 0;
            if (cbid == API_CUDA_cuLaunchKernel_ptsz ||
                cbid == API_CUDA_cuLaunchKernel) {
                cuLaunchKernel_params *p2 = (cuLaunchKernel_params *)params;
                num_ctas = p2->gridDimX * p2->gridDimY * p2->gridDimZ;
            } else if (cbid == API_CUDA_cuLaunchKernelEx_ptsz ||
                cbid == API_CUDA_cuLaunchKernelEx) {
                cuLaunchKernelEx_params *p2 = (cuLaunchKernelEx_params *)params;
                num_ctas = p2->config->gridDimX * p2->config->gridDimY *
                    p2->config->gridDimZ;
            }
            printf(
                "kernel %d - %s - #thread-blocks %d,  kernel "
                "memory accesses %ld, total memory accesses %ld\n",
                kernel_id++, nvbit_get_func_name(ctx, func, mangled), num_ctas,
                kernel_refs, tot_app_mem_refs);
            fflush(stdout);


            pthread_mutex_unlock(&mutex);
        }
    } else if (cbid == API_CUDA_cuProfilerStart && is_exit) {
        if (!active_from_start) {
            active_region = true;
        }
    } else if (cbid == API_CUDA_cuProfilerStop && is_exit) {
        if (!active_from_start) {
            active_region = false;
        }
    } else if (cbid == API_CUDA_cuMemAlloc ||
               cbid == API_CUDA_cuMemAlloc_v2 ||
               cbid == API_CUDA_cuMemAllocManaged ||
               cbid == API_CUDA_cuMemHostAlloc ||
               cbid == API_CUDA_cuMemHostAlloc_v2 ||
               cbid == API_CUDA_cuMemAllocHost ||
               cbid == API_CUDA_cuMemAllocHost_v2 ||
               cbid == API_CUDA_cuMemAllocAsync) {
        if (!is_exit) {
            return;
        }
        printf("ALLOC EVENT: %s - ", name);
        if (cbid == API_CUDA_cuMemAlloc) {
            cuMemAlloc_params* p = (cuMemAlloc_params*)params;
            printf("ptr %p, size %u\n", p->dptr, p->bytesize);
            yosemite_alloc_callback((uint64_t)p->dptr, (uint64_t)p->bytesize, (int)API_CUDA_cuMemAlloc, 0 /* device id */);
        } else if (cbid == API_CUDA_cuMemAlloc_v2) {    // cudaMalloc
            cuMemAlloc_v2_params* p = (cuMemAlloc_v2_params*)params;
            printf("ptr %llu, size %lu\n", *((unsigned long long*)p->dptr), p->bytesize);
            yosemite_alloc_callback((uint64_t)*(p->dptr), (uint64_t)p->bytesize, (int)API_CUDA_cuMemAlloc_v2, 0 /* device id */);
        } else if (cbid == API_CUDA_cuMemAllocManaged) {    // cudaMallocManaged
            cuMemAllocManaged_params* p = (cuMemAllocManaged_params*)params;
            printf("ptr %p, flag %d, size %ld\n", p->dptr, p->flags, p->bytesize);
            yosemite_alloc_callback((uint64_t)*(p->dptr), (uint64_t)p->bytesize, (int)API_CUDA_cuMemAllocManaged, 0 /* device id */);
        } else if (cbid == API_CUDA_cuMemHostAlloc || cbid == API_CUDA_cuMemHostAlloc_v2) {
            cuMemHostAlloc_params* p = (cuMemHostAlloc_params*)params;
            printf("ptr %p, flag %d, size %ld\n", p->pp, p->Flags, p->bytesize);
        } else if (cbid == API_CUDA_cuMemAllocAsync) {
            cuMemAllocAsync_params* p = (cuMemAllocAsync_params*)params;
            printf("ptr %p, size %ld\n", p->dptr, p->bytesize);
            yosemite_alloc_callback((uint64_t)*(p->dptr), (uint64_t)p->bytesize, (int)API_CUDA_cuMemAllocAsync, 0 /* device id */);
        } else if (cbid == API_CUDA_cuMemAllocHost) {
            cuMemAllocHost_params* p = (cuMemAllocHost_params*)params;
            printf("ptr %p, size %u\n", p->pp, p->bytesize);
            yosemite_alloc_callback((uint64_t)p->pp, (uint64_t)p->bytesize, (int)API_CUDA_cuMemAllocHost, 0 /* device id */);
        } else if (cbid == API_CUDA_cuMemAllocHost_v2) {
            cuMemAllocHost_v2_params* p = (cuMemAllocHost_v2_params*)params;
            printf("ptr %p, size %ld\n", p->pp, p->bytesize);
            yosemite_alloc_callback((uint64_t)p->pp, (uint64_t)p->bytesize, (int)API_CUDA_cuMemAllocHost_v2, 0 /* device id */);
        } else {
            printf("Not supported\n");
        }
        fflush(stdout);
    } else if (cbid == API_CUDA_cuMemFree ||
               cbid == API_CUDA_cuMemFree_v2 ||
               cbid == API_CUDA_cuMemFreeHost ||
               cbid == API_CUDA_cuMemFreeAsync) {
        if (is_exit) {
            return;
        }

        printf("FREE EVENT: %s - ", name);
        if (cbid == API_CUDA_cuMemFree) {
            cuMemFree_params* p = (cuMemFree_params*)params;
            printf("ptr %u\n", p->dptr);
            yosemite_free_callback((uint64_t)p->dptr, 0, 0, 0 /* device id */);
        } else if (cbid == API_CUDA_cuMemFree_v2) {   // cudaFree
            cuMemFree_v2_params* p = (cuMemFree_v2_params*)params;
            printf("v2 ptr %llu\n", p->dptr);
            yosemite_free_callback((uint64_t)p->dptr, 0, 0, 0 /* device id */);
        } else if (cbid == API_CUDA_cuMemFreeHost) {
            cuMemFreeHost_params* p = (cuMemFreeHost_params*)params;
            printf("ptr %p\n", p->p);
            yosemite_free_callback((uint64_t)p->p, 0, 0, 0 /* device id */);
        } else if (cbid == API_CUDA_cuMemFreeAsync) {
            cuMemFreeAsync_params* p = (cuMemFreeAsync_params*)params;
            printf("ptr %llu\n", p->dptr);
            yosemite_free_callback((uint64_t)p->dptr, 0, 0, 0 /* device id */);
        } else {
            printf("Not supported\n");
        }
        fflush(stdout);
    }
}

void app_metric_nvbit_at_ctx_init(CUcontext ctx) {}

void app_metric_nvbit_at_ctx_term(CUcontext ctx) {}

void app_metric_nvbit_at_term() {
    yosemite_terminate();
}

} // namespace yosemite_app_metric
