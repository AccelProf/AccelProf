#ifndef YOSEMITE_TOOL_RANGE_TIME_H
#define YOSEMITE_TOOL_RANGE_TIME_H

#include "tools/tool.h"
#include "utils/event.h"

#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace yosemite {

/*
 * RangeTime: GPU time spent inside each named range (e.g. NVTX ranges).
 *
 * Inputs (from the backend):
 *   RangePush / RangePop  on the CPU thread that opens/closes the range
 *   ApiLaunch             on the CPU thread that launches GPU work
 *   GpuActivity           later, with the GPU start/end time of that work
 *
 * How a kernel is assigned to a range:
 *   1. At ApiLaunch, remember which range is open on that thread:
 *          correlation_id -> range id
 *   2. At flush, look up each GpuActivity's correlation_id in that table.
 *
 * Outputs (one set of files per process, written at flush):
 *   <app>.<pid>.range_time.ranges.csv      one row per range instance
 *   <app>.<pid>.range_time.activities.csv  one row per kernel / memcpy / memset
 *   <app>.<pid>.range_time.iterations.csv  one row per iteration range   (optional)
 *   <app>.<pid>.range_time.groups.csv      one row per grouped range     (optional)
 * The last two use the regexes described in range_time.cpp (RangeTime::init).
 */
class RangeTime final : public Tool {
public:
    RangeTime();

    ~RangeTime() {}

    void evt_callback(EventPtr_t evt);

    void gpu_data_analysis(void* data, uint64_t size) override {};

    void query_ranges(void* ranges, uint32_t limit, uint32_t* count) override {};

    void query_tensors(void* ranges, uint32_t limit, uint32_t* count) override {};

    void flush();

private:
    // One range instance. Range id = index in _ranges + 1 (0 means "no range").
    struct RangeRecord {
        std::string name;
        uint64_t parent;            // parent range id, 0 = top level
        uint64_t thread_id;
        uint32_t depth;
        uint64_t cpu_start;
        uint64_t cpu_end;           // 0 = never closed

        // filled by summarize()
        uint64_t self_kern_ns;      // GPU time launched directly in this range
        uint64_t total_kern_ns;     // self + all child ranges
        uint64_t total_activities;  // self + all child ranges
        uint64_t gpu_start;         // earliest GPU start (self + children), 0 = none
        uint64_t gpu_end;           // latest GPU end (self + children)
    };

    struct ActivityRecord {
        uint64_t correlation_id;
        std::string name;
        int device_id;
        uint32_t stream_id;
        uint64_t start;
        uint64_t end;

        // filled by summarize()
        bool has_launch;            // we saw the ApiLaunch for it
        uint64_t range;             // range open at launch time, 0 = none
    };

    void init();

    void range_push_callback(std::shared_ptr<RangePush_t> range);

    void range_pop_callback(std::shared_ptr<RangePop_t> range);

    void api_launch_callback(std::shared_ptr<ApiLaunch_t> launch);

    void gpu_activity_callback(std::shared_ptr<GpuActivity_t> activity);

    void summarize();

    void dump_ranges(const std::string& prefix);

    void dump_activities(const std::string& prefix);

    void dump_groups(const std::string& prefix);

/*
********************************* variables *********************************
*/
    std::mutex _mutex;  // events come from app threads and from the CUPTI thread

    std::vector<RangeRecord> _ranges;
    std::map<uint64_t, std::vector<uint64_t>> _open_ranges;   // thread id -> stack of range ids
    std::unordered_map<uint64_t, uint64_t> _launch_range;     // correlation id -> range id
    std::vector<ActivityRecord> _activities;
    uint64_t _unmatched_pops = 0;

    std::string _iter_regex;
    std::string _group_regex;
    std::string _tokens_regex;
};

}   // yosemite
#endif // YOSEMITE_TOOL_RANGE_TIME_H
