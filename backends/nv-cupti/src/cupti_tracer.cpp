/*
 * nv-cupti: an AccelProf backend built on CUPTI.
 *
 * This file only talks to CUPTI and forwards what it sees to sanalyzer.
 * It does no analysis; the sanalyzer tool (range_time) does that.
 *
 * CUPTI gives us data in two ways:
 *
 *   1. Callback API: CUPTI calls CuptiCallback() right away, on the
 *      application thread, whenever the app calls an API we subscribed to.
 *        - nvtxRangePush / nvtxRangePop      -> yosemite_range_push/pop_callback
 *        - CUDA APIs that put work on the GPU -> yosemite_api_launch_callback
 *          (cudaLaunchKernel, cuLaunchKernel, cudaMemcpyAsync, ...)
 *        - cudaProfilerStart / Stop           -> start / stop recording
 *
 *   2. Activity API: CUPTI writes one record per kernel / memcpy / memset,
 *      with GPU start and end timestamps, into buffers we hand to it.
 *      Full buffers come back later in BufferCompleted(), on a CUPTI thread.
 *        - kernel / memcpy / memset record    -> yosemite_gpu_activity_callback
 *
 * A launch callback and the kernel record it produces carry the same
 * "correlation id". That is how the tool knows which NVTX range a kernel
 * was launched in.
 *
 * All timestamps (CPU and GPU) are in ns on the same clock (cuptiGetTimestamp).
 */

#include "sanalyzer.h"

#include <cupti.h>
#include <generated_nvtx_meta.h>    // parameter structs of NVTX callbacks

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cxxabi.h>
#include <string>
#include <sys/syscall.h>
#include <unistd.h>

#define CUPTI_VERBOSE 1

#if CUPTI_VERBOSE
#define PRINT(...) do { fprintf(stdout, __VA_ARGS__); fflush(stdout); } while (0)
#else
#define PRINT(...)
#endif

#define CUPTI_SAFECALL(fn)                                          \
{                                                                   \
    CUptiResult result = fn;                                        \
    if (result != CUPTI_SUCCESS) {                                  \
        const char *error_string;                                   \
        cuptiGetResultString(result, &error_string);                \
        fprintf(                                                    \
            stderr,                                                 \
            "[CUPTI ERROR] '%s' in '%s:%i' - error(%i): %s.\n",     \
            #fn, __FILE__, __LINE__, result, error_string);         \
        fflush(stderr);                                             \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
}

// Each CUDA version has a newer kernel record struct (Kernel4 ... Kernel10).
// The fields used here are at the same offsets in all of them.
#if CUPTI_API_VERSION >= 130000
typedef CUpti_ActivityKernel10 CuptiKernelRecord;
#else
typedef CUpti_ActivityKernel9 CuptiKernelRecord;
#endif

static const size_t ACTIVITY_BUFFER_SIZE = 8 * 1024 * 1024;    // bytes per activity buffer

static AccelProfOptions_t cupti_options;
static CUpti_SubscriberHandle cupti_subscriber;
static bool cupti_initialized = false;

// ACCELPROF_CUPTI_CAPTURE=profiler_api : record only between cudaProfilerStart and cudaProfilerStop
// ACCELPROF_CUPTI_CAPTURE=all (default) : record the whole run
static bool cupti_capture_profiler_api = false;
static std::atomic<bool> cupti_recording(true);
static std::atomic<uint64_t> cupti_window_start(0);    // 0 = recording never started
static std::atomic<uint64_t> cupti_window_end(0);      // 0 = still recording


/****************************************************************************************
 ************************************** helpers *****************************************
****************************************************************************************/

static uint64_t get_thread_id() {
    return (uint64_t)syscall(SYS_gettid);
}


static uint64_t get_timestamp() {
    uint64_t timestamp = 0;
    cuptiGetTimestamp(&timestamp);
    return timestamp;
}


// CUPTI reports mangled kernel names, e.g. "_Z17topkGatingSoftmaxIfLi8EEvPKT_..."
static std::string demangle(const char* name) {
    if (name == nullptr) {
        return "<unknown>";
    }
    int status = 0;
    char* demangled = abi::__cxa_demangle(name, nullptr, nullptr, &status);
    if (status != 0 || demangled == nullptr) {
        return std::string(name);   // not a C++ mangled name (e.g. extern "C" kernels)
    }
    std::string result(demangled);
    free(demangled);
    return result;
}


static const char* memcpy_kind_name(uint8_t kind) {
    switch (kind) {
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD: return "HtoD";
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH: return "DtoH";
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD: return "DtoD";
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH: return "HtoH";
        case CUPTI_ACTIVITY_MEMCPY_KIND_PTOP: return "PtoP";
        default: return "Other";
    }
}


// Text of an "Ex" NVTX push. Only plain ASCII messages are supported.
static std::string nvtx_message(const nvtxEventAttributes_t* attrib) {
    if (attrib && attrib->messageType == NVTX_MESSAGE_TYPE_ASCII && attrib->message.ascii) {
        return std::string(attrib->message.ascii);
    }
    return "";
}


// Is a GPU record inside the recording window [start, end)?
static bool in_recording_window(uint64_t gpu_start) {
    if (!cupti_capture_profiler_api) {
        return true;
    }
    uint64_t start = cupti_window_start;
    uint64_t end = cupti_window_end;
    return start != 0 && gpu_start >= start && (end == 0 || gpu_start < end);
}


/****************************************************************************************
 *********************************** recording window ***********************************
****************************************************************************************/

static void start_recording() {
    if (!cupti_capture_profiler_api || cupti_recording) {
        return;
    }
    cupti_window_start = get_timestamp();
    cupti_window_end = 0;
    cupti_recording = true;
    PRINT("[CUPTI INFO] cudaProfilerStart: recording started (pid %d)\n", getpid());
}


static void stop_recording() {
    if (!cupti_capture_profiler_api || !cupti_recording) {
        return;
    }
    cupti_recording = false;
    cupti_window_end = get_timestamp();
    PRINT("[CUPTI INFO] cudaProfilerStop: recording stopped, flushing (pid %d)\n", getpid());

    // Hand all activity records to sanalyzer, then let the tools write their files.
    // This does not wait for the GPU: the app should synchronize before cudaProfilerStop.
    // Flushing here (not only at exit) matters for processes that get killed (e.g. sglang workers).
    cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
    yosemite_flush();
}


/****************************************************************************************
 ************************************ callback API **************************************
****************************************************************************************/

static void NvtxCallback(CUpti_CallbackId cbid, const CUpti_NvtxData* data) {
    if (!cupti_recording) {
        return;
    }

    switch (cbid) {
        case CUPTI_CBID_NVTX_nvtxRangePushA: {
            // torch.cuda.nvtx.range_push(msg) ends up here
            auto* params = (const nvtxRangePushA_params*)data->functionParams;
            std::string name = params->message ? params->message : "";
            yosemite_range_push_callback(name, get_thread_id(), get_timestamp());
            break;
        }
        case CUPTI_CBID_NVTX_nvtxRangePushEx: {
            auto* params = (const nvtxRangePushEx_params*)data->functionParams;
            yosemite_range_push_callback(nvtx_message(params->eventAttrib), get_thread_id(), get_timestamp());
            break;
        }
        case CUPTI_CBID_NVTX_nvtxDomainRangePushEx: {
            // ranges of all NVTX domains share one stack per thread here
            auto* params = (const nvtxDomainRangePushEx_params*)data->functionParams;
            yosemite_range_push_callback(nvtx_message(params->core.eventAttrib), get_thread_id(), get_timestamp());
            break;
        }
        case CUPTI_CBID_NVTX_nvtxRangePop:
        case CUPTI_CBID_NVTX_nvtxDomainRangePop:
            yosemite_range_pop_callback(get_thread_id(), get_timestamp());
            break;
        default:
            break;
    }
}


static void ApiCallback(CUpti_CallbackDomain domain, CUpti_CallbackId cbid, const CUpti_CallbackData* data) {
    bool is_runtime = (domain == CUPTI_CB_DOMAIN_RUNTIME_API);

    // cudaProfilerStart / cudaProfilerStop (the runtime may also call the driver version)
    if ((is_runtime && cbid == CUPTI_RUNTIME_TRACE_CBID_cudaProfilerStart_v4000) ||
        (!is_runtime && cbid == CUPTI_DRIVER_TRACE_CBID_cuProfilerStart)) {
        if (data->callbackSite == CUPTI_API_EXIT) {
            start_recording();
        }
        return;
    }
    if ((is_runtime && cbid == CUPTI_RUNTIME_TRACE_CBID_cudaProfilerStop_v4000) ||
        (!is_runtime && cbid == CUPTI_DRIVER_TRACE_CBID_cuProfilerStop)) {
        if (data->callbackSite == CUPTI_API_ENTER) {
            stop_recording();
        }
        return;
    }

    // Every other enabled API puts work on the GPU. Report it once, at API entry.
    if (data->callbackSite != CUPTI_API_ENTER || !cupti_recording) {
        return;
    }
    yosemite_api_launch_callback(data->correlationId, get_thread_id(), get_timestamp());
}


void CUPTIAPI CuptiCallback(void* userdata, CUpti_CallbackDomain domain,
                            CUpti_CallbackId cbid, const void* cbdata) {
    if (domain == CUPTI_CB_DOMAIN_NVTX) {
        NvtxCallback(cbid, (const CUpti_NvtxData*)cbdata);
    } else if (domain == CUPTI_CB_DOMAIN_RUNTIME_API || domain == CUPTI_CB_DOMAIN_DRIVER_API) {
        ApiCallback(domain, cbid, (const CUpti_CallbackData*)cbdata);
    }
}


// Enable callbacks only for APIs that put work on the GPU, plus cudaProfilerStart/Stop.
// CUPTI can tell us the name of every callback id, so we pick them by name
// instead of hard-coding a list that changes with each CUDA version.
static int EnableLaunchCallbacks(CUpti_CallbackDomain domain, uint32_t num_cbids) {
    int count = 0;
    for (uint32_t cbid = 0; cbid < num_cbids; cbid++) {
        const char* name = nullptr;
        if (cuptiGetCallbackName(domain, cbid, &name) != CUPTI_SUCCESS || name == nullptr) {
            continue;   // unused id
        }
        if (strstr(name, "Launch") || strstr(name, "Memcpy") || strstr(name, "Memset") ||
            strstr(name, "ProfilerStart") || strstr(name, "ProfilerStop")) {
            CUPTI_SAFECALL(cuptiEnableCallback(1, cupti_subscriber, domain, cbid));
            count++;
        }
    }
    return count;
}


/****************************************************************************************
 ************************************ activity API **************************************
****************************************************************************************/

// CUPTI asks for an empty buffer to write activity records into.
void CUPTIAPI BufferRequested(uint8_t** buffer, size_t* size, size_t* max_num_records) {
    *buffer = (uint8_t*)aligned_alloc(8, ACTIVITY_BUFFER_SIZE);
    *size = ACTIVITY_BUFFER_SIZE;
    *max_num_records = 0;   // 0 = as many records as fit
}


static void HandleActivityRecord(CUpti_Activity* record) {
    switch (record->kind) {
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
            auto* kernel = (CuptiKernelRecord*)record;
            if (in_recording_window(kernel->start)) {
                yosemite_gpu_activity_callback(kernel->correlationId, demangle(kernel->name),
                                               kernel->deviceId, kernel->streamId,
                                               kernel->start, kernel->end);
            }
            break;
        }
        case CUPTI_ACTIVITY_KIND_MEMCPY: {
            auto* copy = (CUpti_ActivityMemcpy6*)record;
            if (in_recording_window(copy->start)) {
                std::string name = std::string("[memcpy ") + memcpy_kind_name(copy->copyKind) + "]";
                yosemite_gpu_activity_callback(copy->correlationId, name,
                                               copy->deviceId, copy->streamId,
                                               copy->start, copy->end);
            }
            break;
        }
        case CUPTI_ACTIVITY_KIND_MEMSET: {
            auto* set = (CUpti_ActivityMemset4*)record;
            if (in_recording_window(set->start)) {
                yosemite_gpu_activity_callback(set->correlationId, "[memset]",
                                               set->deviceId, set->streamId,
                                               set->start, set->end);
            }
            break;
        }
        default:
            break;
    }
}


// CUPTI gives back a buffer filled with records (on a CUPTI thread).
void CUPTIAPI BufferCompleted(CUcontext context, uint32_t stream_id,
                              uint8_t* buffer, size_t size, size_t valid_size) {
    CUpti_Activity* record = nullptr;
    while (cuptiActivityGetNextRecord(buffer, valid_size, &record) == CUPTI_SUCCESS) {
        HandleActivityRecord(record);
    }

    size_t dropped = 0;
    cuptiActivityGetNumDroppedRecords(context, stream_id, &dropped);
    if (dropped != 0) {
        fprintf(stderr, "[CUPTI WARN] Dropped %zu activity records.\n", dropped);
        fflush(stderr);
    }
    free(buffer);
}


/****************************************************************************************
 ************************************ init / exit ***************************************
****************************************************************************************/

// Called when the library is loaded (LD_PRELOAD), and by the CUDA driver
// if the library is given through CUDA_INJECTION64_PATH.
extern "C" int InitializeInjection() {
    if (cupti_initialized) {
        return 1;
    }
    cupti_initialized = true;

    const char* capture = std::getenv("ACCELPROF_CUPTI_CAPTURE");
    cupti_capture_profiler_api = capture && std::string(capture) == "profiler_api";
    cupti_recording = !cupti_capture_profiler_api;
    PRINT("[CUPTI INFO] pid %d, capture: %s\n", getpid(),
          cupti_capture_profiler_api ? "profiler_api (cudaProfilerStart/Stop)" : "all");

    // NVTX only reports to CUPTI if NVTX_INJECTION64_PATH points to libcupti.so.
    // NVTX reads this variable once, at the first NVTX call of the process.
    const char* nvtx_injection = std::getenv("NVTX_INJECTION64_PATH");
    if (nvtx_injection) {
        PRINT("[CUPTI INFO] NVTX_INJECTION64_PATH=%s\n", nvtx_injection);
    } else {
        PRINT("[CUPTI WARN] NVTX_INJECTION64_PATH is not set, NVTX ranges will not be seen.\n");
    }

    yosemite_init(cupti_options);

    // Callback API
    CUPTI_SAFECALL(cuptiSubscribe(&cupti_subscriber, (CUpti_CallbackFunc)CuptiCallback, nullptr));
    CUPTI_SAFECALL(cuptiEnableDomain(1, cupti_subscriber, CUPTI_CB_DOMAIN_NVTX));
    int num_runtime = EnableLaunchCallbacks(CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_SIZE);
    int num_driver = EnableLaunchCallbacks(CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_SIZE);
    PRINT("[CUPTI INFO] Enabled %d runtime and %d driver API callbacks\n", num_runtime, num_driver);

    // Activity API. CONCURRENT_KERNEL (not KERNEL): KERNEL serializes all kernels.
    CUPTI_SAFECALL(cuptiActivityRegisterCallbacks(BufferRequested, BufferCompleted));
    CUPTI_SAFECALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
    CUPTI_SAFECALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
    CUPTI_SAFECALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));

    return 1;
}


void cleanup(void) {
    // Deliver the records still held by CUPTI, then let the tools write their files.
    cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
    yosemite_terminate();
}

__attribute__((constructor))
void initializer(void) {
    atexit(cleanup);
}

__attribute__((destructor))
void finalizer(void) {
}

int __global_initializer__ = InitializeInjection();
