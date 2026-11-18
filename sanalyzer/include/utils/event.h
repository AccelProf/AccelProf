#ifndef YOSEMITE_UTILS_EVENT_H
#define YOSEMITE_UTILS_EVENT_H

#include <cstdint>
#include <string>
#include <memory>

typedef uint64_t DevPtr;

typedef struct Timer{
    uint64_t access_timer;
    uint64_t event_timer;

    Timer() {
        access_timer = 0;
        event_timer = 0;
    }

    void increment(bool is_event) {
        if (is_event) {
            event_timer++;
        } else {
            access_timer++;
        }
    }

    uint64_t get() {
        return access_timer + event_timer;
    }
} Timer_t;

namespace yosemite {

typedef enum EventType {
    EventType_KERNEL_LAUNCH = 0,
    EventType_KERNEL_END = 1,
    EventType_MEM_ALLOC = 2,
    EventType_MEM_FREE = 3,
    EventType_MEM_COPY = 4,
    EventType_MEM_SET = 5,
    EventType_TEN_ALLOC = 6,
    EventType_TEN_FREE = 7,
    EventType_OP_START = 8,
    EventType_OP_END = 9,
    EventType_RANGE_PUSH = 10,
    EventType_RANGE_POP = 11,
    EventType_API_LAUNCH = 12,
    EventType_GPU_ACTIVITY = 13,
    EventTypeCount = 14,
}EventType_t;


typedef struct Event
{
    uint64_t timestamp;
    EventType_t evt_type;
    int device_id;

    Event() = default;
    virtual ~Event() = default;

    bool operator<(const Event &other) const { return timestamp < other.timestamp; }
}Event_t;

typedef std::shared_ptr<Event> EventPtr_t;


typedef struct KernelLaunch : public Event {
    uint64_t end_time;
    std::string kernel_name;
    uint32_t kernel_id;
    uint32_t grid_dim_x;
    uint32_t grid_dim_y;
    uint32_t grid_dim_z;
    uint64_t grid_cta_count;
    uint32_t block_dim_x;
    uint32_t block_dim_y;
    uint32_t block_dim_z;
    uint32_t block_thread_count;
    uint64_t access_count;
    uint32_t touched_objects;
    uint32_t touched_objects_size;
    uint64_t key;   // for UVM Advisor
    uint64_t kernel_pc;

    KernelLaunch() {
        this->evt_type = EventType_KERNEL_LAUNCH;
        grid_dim_x = 0;
        grid_dim_y = 0;
        grid_dim_z = 0;
        grid_cta_count = 0;
        block_dim_x = 0;
        block_dim_y = 0;
        block_dim_z = 0;
        block_thread_count = 0;
    }

    KernelLaunch(std::string kernel_name, int device_id)
        : kernel_name(kernel_name) {
        this->evt_type = EventType_KERNEL_LAUNCH;
        this->device_id = device_id;
        grid_dim_x = 0;
        grid_dim_y = 0;
        grid_dim_z = 0;
        grid_cta_count = 0;
        block_dim_x = 0;
        block_dim_y = 0;
        block_dim_z = 0;
        block_thread_count = 0;
    }

    KernelLaunch(std::string kernel_name, int device_id,
                 uint32_t grid_dim_x, uint32_t grid_dim_y, uint32_t grid_dim_z,
                 uint32_t block_dim_x, uint32_t block_dim_y, uint32_t block_dim_z,
                 uint64_t grid_cta_count, uint32_t block_thread_count)
        : kernel_name(kernel_name),
          grid_dim_x(grid_dim_x), grid_dim_y(grid_dim_y), grid_dim_z(grid_dim_z),
          block_dim_x(block_dim_x), block_dim_y(block_dim_y), block_dim_z(block_dim_z),
          grid_cta_count(grid_cta_count), block_thread_count(block_thread_count)
    {
        this->evt_type = EventType_KERNEL_LAUNCH;
        this->device_id = device_id;
    }

    ~KernelLaunch() = default;
}KernelLaunch_t;

typedef struct KernelEnd : public Event {
    uint64_t end_time;
    std::string kernel_name;
    uint64_t access_count;

    KernelEnd() {
        this->evt_type = EventType_KERNEL_END;
        access_count = 0;
    }

    KernelEnd(std::string kernel_name, int device_id)
        : kernel_name(kernel_name) {
            this->evt_type = EventType_KERNEL_END;
            this->device_id = device_id;
        }

    ~KernelEnd() = default;
}KernelEnd_t;

typedef struct MemAlloc : public Event {
    DevPtr addr;
    uint64_t size;
    uint64_t release_time;
    int alloc_type;
    uint64_t key;   // for UVM Advisor

    MemAlloc() {
        this->evt_type = EventType_MEM_ALLOC;
    }

    MemAlloc(DevPtr addr, uint64_t size, int alloc_type, int device_id)
        : addr(addr), size(size), alloc_type(alloc_type) {
            this->evt_type = EventType_MEM_ALLOC;
            this->device_id = device_id;
        }

    ~MemAlloc() = default;
}MemAlloc_t;

typedef struct MemFree : public Event {
    DevPtr addr;
    uint64_t size;
    int alloc_type;

    MemFree() {
        this->evt_type = EventType_MEM_FREE;
    }

    MemFree(DevPtr addr, uint64_t size, int alloc_type, int device_id)
        : addr(addr), size(size), alloc_type(alloc_type) {
            this->evt_type = EventType_MEM_FREE;
            this->device_id = device_id;
        }

    ~MemFree() = default;
}MemFree_t;

typedef struct MemCpy : public Event {
    uint64_t src_addr;
    uint64_t dst_addr;
    uint64_t size;
    bool is_async;
    uint32_t direction;

    MemCpy() {
        this->evt_type = EventType_MEM_COPY;
    }
    
    MemCpy(uint64_t src_addr, uint64_t dst_addr, uint64_t size, bool is_async, uint32_t direction, int device_id)
        : src_addr(src_addr), dst_addr(dst_addr), size(size), is_async(is_async), direction(direction) {
            this->evt_type = EventType_MEM_COPY;
            this->device_id = device_id;
        }

    ~MemCpy() = default;
}MemCpy_t;

typedef struct MemSet : public Event {

    uint64_t addr;
    uint64_t size;
    uint32_t value;
    bool is_async;

    MemSet() {
        this->evt_type = EventType_MEM_SET;
    }

    MemSet(uint64_t addr, uint64_t size, uint32_t value, bool is_async, int device_id)
        : addr(addr), size(size), value(value), is_async(is_async) {
            this->evt_type = EventType_MEM_SET;
            this->device_id = device_id;
        }

    ~MemSet() = default;
}MemSet_t;

typedef struct TenAlloc : public Event {
    DevPtr addr;
    int64_t size;
    int64_t allocated_size;
    int64_t reserved_size;
    uint64_t release_time;
    uint64_t key;   // for UVM Advisor

    TenAlloc() {
        this->evt_type = EventType_TEN_ALLOC;
    }

    TenAlloc(DevPtr addr, int64_t size, int64_t allocated_size, int64_t reserved_size, int device_id)
        : addr(addr), size(size), allocated_size(allocated_size), reserved_size(reserved_size) {
            this->evt_type = EventType_TEN_ALLOC;
            this->device_id = device_id;
        }

    ~TenAlloc() = default;
}TenAlloc_t;

typedef struct TenFree : public Event {
    DevPtr addr;
    int64_t size;
    int64_t allocated_size;
    int64_t reserved_size;

    TenFree() {
        this->evt_type = EventType_TEN_FREE;
    }

    TenFree(DevPtr addr, int64_t size, int64_t allocated_size, int64_t reserved_size, int device_id)
        : addr(addr), size(size), allocated_size(allocated_size), reserved_size(reserved_size) {
            this->evt_type = EventType_TEN_FREE;
            this->device_id = device_id;
        }

    ~TenFree() = default;
}TenFree_t;

typedef struct OpStart : public Event {
    uint64_t end_time;
    uint64_t pending_kernels;
    uint64_t pending_ops;
    uint64_t pending_mem_alloc;
    uint64_t pending_ten_alloc;
    std::string op_name;
    void* ctx;
    uint64_t key;   // for UVM Advisor
    OpStart() {
        evt_type = EventType_OP_START;
    }

    OpStart(std::string op_name, void* ctx)
        : op_name(op_name), ctx(ctx) {
            evt_type = EventType_OP_START;
        }

    ~OpStart() = default;
}OpStart_t;

typedef struct OpEnd : public Event {
    std::string op_name;
    void* ctx;

    OpEnd() {
        evt_type = EventType_OP_END;
    }

    OpEnd(std::string op_name, void* ctx)
        : op_name(op_name), ctx(ctx) {
            evt_type = EventType_OP_END;
        }

    ~OpEnd() = default;
}OpEnd_t;

// A named CPU-side range, e.g. an NVTX push. `timestamp` is the push time.
typedef struct RangePush : public Event {
    std::string name;
    uint64_t thread_id;

    RangePush() {
        evt_type = EventType_RANGE_PUSH;
    }

    RangePush(std::string name, uint64_t thread_id, uint64_t timestamp)
        : name(name), thread_id(thread_id) {
            evt_type = EventType_RANGE_PUSH;
            this->timestamp = timestamp;
        }

    ~RangePush() = default;
}RangePush_t;

// Closes the innermost open range of `thread_id`. `timestamp` is the pop time.
typedef struct RangePop : public Event {
    uint64_t thread_id;

    RangePop() {
        evt_type = EventType_RANGE_POP;
    }

    RangePop(uint64_t thread_id, uint64_t timestamp)
        : thread_id(thread_id) {
            evt_type = EventType_RANGE_POP;
            this->timestamp = timestamp;
        }

    ~RangePop() = default;
}RangePop_t;

// A CPU-side call that puts work on the GPU (kernel launch, memcpy, memset).
// It is reported on the calling thread, at call time, so it can be matched
// with the range that is open at that moment.
// `correlation_id` links this call to the GpuActivity it produces.
typedef struct ApiLaunch : public Event {
    uint64_t correlation_id;
    uint64_t thread_id;

    ApiLaunch() {
        evt_type = EventType_API_LAUNCH;
    }

    ApiLaunch(uint64_t correlation_id, uint64_t thread_id, uint64_t timestamp)
        : correlation_id(correlation_id), thread_id(thread_id) {
            evt_type = EventType_API_LAUNCH;
            this->timestamp = timestamp;
        }

    ~ApiLaunch() = default;
}ApiLaunch_t;

// Work that ran on the GPU (kernel, memcpy, memset), with GPU start/end time.
// It arrives later than the ApiLaunch that produced it, possibly from another thread.
typedef struct GpuActivity : public Event {
    uint64_t correlation_id;
    std::string name;
    uint32_t stream_id;
    uint64_t start;
    uint64_t end;

    GpuActivity() {
        evt_type = EventType_GPU_ACTIVITY;
    }

    GpuActivity(uint64_t correlation_id, std::string name, int device_id,
                uint32_t stream_id, uint64_t start, uint64_t end)
        : correlation_id(correlation_id), name(name), stream_id(stream_id),
          start(start), end(end) {
            evt_type = EventType_GPU_ACTIVITY;
            this->device_id = device_id;
            this->timestamp = start;
        }

    ~GpuActivity() = default;
}GpuActivity_t;

}   // yosemite

#endif // YOSEMITE_UTILS_EVENT_H
