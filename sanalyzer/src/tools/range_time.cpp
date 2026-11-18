#include "tools/range_time.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <regex>
#include <unistd.h>

using namespace yosemite;


/*
 * Default regexes match sglang's layerwise NVTX ranges
 * (--enable-layerwise-nvtx-marker). Their text looks like:
 *   {'Module': 'model.model.layers.3.mlp.gate', 'TrainableParams': {...}, 'Inputs': [[1, 2048]], ...}
 *
 * ITER   : a range whose text matches starts one iteration (one forward pass).
 * GROUP  : ranges to report in groups.csv. Capture group 1 = layer, group 2 = component.
 * TOKENS : capture group 1 = number of tokens, read from the iteration range's text.
 */
static const char* DEFAULT_ITER_REGEX   = R"('Module': 'model\.model')";
static const char* DEFAULT_GROUP_REGEX  = R"('Module': '[^']*\.layers\.(\d+)\.(?:mlp|feed_forward)\.(gate|router|topk)')";
static const char* DEFAULT_TOKENS_REGEX = R"('Inputs': \[\[(\d+))";


static std::string get_env_or(const char* key, const char* default_value) {
    const char* value = std::getenv(key);
    return value ? std::string(value) : std::string(default_value);
}


// Wrap a field in double quotes for CSV (names contain commas).
static std::string csv_quote(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') out += '"';
        out += c;
    }
    out += "\"";
    return out;
}


// "void at::native::foo<float, 4>(float*, int)" -> "foo"
// Same idea as the "short name" column in Nsight Systems.
static std::string short_kernel_name(std::string name) {
    if (name.empty() || name[0] == '[') {
        return name;    // "[memcpy DtoH]", "[memset]"
    }
    const std::string anon = "(anonymous namespace)";
    for (size_t pos = name.find(anon); pos != std::string::npos; pos = name.find(anon)) {
        name.erase(pos, anon.size());
    }

    // drop template arguments and everything from the parameter list on
    std::string out;
    int depth = 0;
    for (char c : name) {
        if (c == '<') { depth++; continue; }
        if (c == '>') { if (depth > 0) depth--; continue; }
        if (depth > 0) continue;
        if (c == '(') break;
        out += c;
    }
    // drop return type and namespaces
    size_t pos = out.rfind(' ');
    if (pos != std::string::npos) out = out.substr(pos + 1);
    pos = out.rfind("::");
    if (pos != std::string::npos) out = out.substr(pos + 2);

    return out.empty() ? name : out;
}


static double ns_to_us(uint64_t ns) {
    return (double)ns / 1000.0;
}


RangeTime::RangeTime() : Tool(RANGE_TIME) {
    init();
}


void RangeTime::init() {
    _iter_regex   = get_env_or("ACCELPROF_RANGE_ITER_REGEX", DEFAULT_ITER_REGEX);
    _group_regex  = get_env_or("ACCELPROF_RANGE_GROUP_REGEX", DEFAULT_GROUP_REGEX);
    _tokens_regex = get_env_or("ACCELPROF_RANGE_TOKENS_REGEX", DEFAULT_TOKENS_REGEX);

    fprintf(stdout, "[RANGE_TIME INFO] iteration regex: %s\n", _iter_regex.c_str());
    fprintf(stdout, "[RANGE_TIME INFO] group regex: %s\n", _group_regex.c_str());
    fprintf(stdout, "[RANGE_TIME INFO] tokens regex: %s\n", _tokens_regex.c_str());
    fflush(stdout);
}


void RangeTime::evt_callback(EventPtr_t evt) {
    switch (evt->evt_type) {
        case EventType_RANGE_PUSH:
            range_push_callback(std::dynamic_pointer_cast<RangePush_t>(evt));
            break;
        case EventType_RANGE_POP:
            range_pop_callback(std::dynamic_pointer_cast<RangePop_t>(evt));
            break;
        case EventType_API_LAUNCH:
            api_launch_callback(std::dynamic_pointer_cast<ApiLaunch_t>(evt));
            break;
        case EventType_GPU_ACTIVITY:
            gpu_activity_callback(std::dynamic_pointer_cast<GpuActivity_t>(evt));
            break;
        default:
            break;
    }
}


void RangeTime::range_push_callback(std::shared_ptr<RangePush_t> range) {
    std::lock_guard<std::mutex> lock(_mutex);
    auto& stack = _open_ranges[range->thread_id];

    RangeRecord record = {};
    record.name = range->name;
    record.parent = stack.empty() ? 0 : stack.back();
    record.thread_id = range->thread_id;
    record.depth = stack.size();
    record.cpu_start = range->timestamp;
    record.cpu_end = 0;
    _ranges.push_back(record);

    stack.push_back(_ranges.size());    // range id = index + 1
}


void RangeTime::range_pop_callback(std::shared_ptr<RangePop_t> range) {
    std::lock_guard<std::mutex> lock(_mutex);
    auto& stack = _open_ranges[range->thread_id];
    if (stack.empty()) {
        // e.g. the range was pushed before recording started
        _unmatched_pops++;
        return;
    }
    _ranges[stack.back() - 1].cpu_end = range->timestamp;
    stack.pop_back();
}


void RangeTime::api_launch_callback(std::shared_ptr<ApiLaunch_t> launch) {
    std::lock_guard<std::mutex> lock(_mutex);
    auto& stack = _open_ranges[launch->thread_id];
    _launch_range[launch->correlation_id] = stack.empty() ? 0 : stack.back();
}


void RangeTime::gpu_activity_callback(std::shared_ptr<GpuActivity_t> activity) {
    std::lock_guard<std::mutex> lock(_mutex);
    ActivityRecord record = {};
    record.correlation_id = activity->correlation_id;
    record.name = activity->name;
    record.device_id = activity->device_id;
    record.stream_id = activity->stream_id;
    record.start = activity->start;
    record.end = activity->end;
    _activities.push_back(record);
}


// Assign each activity to a range and add up GPU time per range.
// Called at every flush, so it recomputes everything from scratch.
void RangeTime::summarize() {
    for (auto& r : _ranges) {
        r.self_kern_ns = 0;
        r.total_kern_ns = 0;
        r.total_activities = 0;
        r.gpu_start = 0;
        r.gpu_end = 0;
    }

    for (auto& a : _activities) {
        auto it = _launch_range.find(a.correlation_id);
        a.has_launch = (it != _launch_range.end());
        a.range = a.has_launch ? it->second : 0;
        if (a.range == 0) {
            continue;
        }

        auto& r = _ranges[a.range - 1];
        uint64_t duration = a.end > a.start ? a.end - a.start : 0;
        r.self_kern_ns += duration;
        r.total_kern_ns += duration;
        r.total_activities += 1;
        if (r.gpu_start == 0 || a.start < r.gpu_start) r.gpu_start = a.start;
        if (a.end > r.gpu_end) r.gpu_end = a.end;
    }

    // Add each range into its parent. A child is always created after its parent,
    // so it has a larger id: walking from the last id to the first visits every
    // child before its parent.
    for (size_t i = _ranges.size(); i-- > 0;) {
        auto& r = _ranges[i];
        if (r.parent == 0) {
            continue;
        }
        auto& p = _ranges[r.parent - 1];
        p.total_kern_ns += r.total_kern_ns;
        p.total_activities += r.total_activities;
        if (r.gpu_start != 0 && (p.gpu_start == 0 || r.gpu_start < p.gpu_start)) p.gpu_start = r.gpu_start;
        if (r.gpu_end > p.gpu_end) p.gpu_end = r.gpu_end;
    }
}


void RangeTime::dump_ranges(const std::string& prefix) {
    std::string filename = prefix + ".ranges.csv";
    std::ofstream out(filename);
    out << std::fixed << std::setprecision(3);
    out << "range_id,parent_id,thread_id,depth,cpu_start_ns,cpu_end_ns,cpu_us,"
        << "self_kern_us,total_kern_us,total_activities,gpu_span_us,name\n";

    for (size_t i = 0; i < _ranges.size(); i++) {
        const auto& r = _ranges[i];
        uint64_t cpu_ns = r.cpu_end > r.cpu_start ? r.cpu_end - r.cpu_start : 0;
        uint64_t span_ns = r.gpu_end > r.gpu_start ? r.gpu_end - r.gpu_start : 0;
        out << i + 1 << "," << r.parent << "," << r.thread_id << "," << r.depth << ","
            << r.cpu_start << "," << r.cpu_end << "," << ns_to_us(cpu_ns) << ","
            << ns_to_us(r.self_kern_ns) << "," << ns_to_us(r.total_kern_ns) << ","
            << r.total_activities << "," << ns_to_us(span_ns) << ","
            << csv_quote(r.name) << "\n";
    }
    out.close();
    printf("[RANGE_TIME INFO] Dumping %zu ranges to %s\n", _ranges.size(), filename.c_str());
}


void RangeTime::dump_activities(const std::string& prefix) {
    std::string filename = prefix + ".activities.csv";
    std::ofstream out(filename);
    out << std::fixed << std::setprecision(3);
    out << "correlation_id,range_id,has_launch,device_id,stream_id,start_ns,end_ns,dur_us,short_name,name\n";

    for (const auto& a : _activities) {
        uint64_t duration = a.end > a.start ? a.end - a.start : 0;
        out << a.correlation_id << "," << a.range << "," << (a.has_launch ? 1 : 0) << ","
            << a.device_id << "," << a.stream_id << "," << a.start << "," << a.end << ","
            << ns_to_us(duration) << "," << csv_quote(short_kernel_name(a.name)) << ","
            << csv_quote(a.name) << "\n";
    }
    out.close();
    printf("[RANGE_TIME INFO] Dumping %zu GPU activities to %s\n", _activities.size(), filename.c_str());
}


// Optional, regex based report:
//   iterations.csv  one row per iteration range (e.g. one forward pass)
//   groups.csv      one row per range that matches the group regex (e.g. mlp.gate of layer 3)
// Each grouped range is tagged with the iteration that contains it.
void RangeTime::dump_groups(const std::string& prefix) {
    std::regex iter_re(_iter_regex);
    std::regex group_re(_group_regex);
    std::regex tokens_re(_tokens_regex);

    // Iteration of each range = the closest ancestor (or itself) that matches iter_re.
    // Parents come before children in _ranges, so one forward pass is enough.
    std::vector<int64_t> iter_of(_ranges.size(), -1);
    std::vector<uint64_t> iter_range_ids;
    std::vector<int64_t> iter_tokens;
    for (size_t i = 0; i < _ranges.size(); i++) {
        const auto& r = _ranges[i];
        std::smatch m;
        if (std::regex_search(r.name, m, iter_re)) {
            iter_of[i] = iter_range_ids.size();
            iter_range_ids.push_back(i + 1);
            std::smatch t;
            iter_tokens.push_back(std::regex_search(r.name, t, tokens_re) && t.size() > 1
                                  ? std::stoll(t[1].str()) : -1);
        } else if (r.parent != 0) {
            iter_of[i] = iter_of[r.parent - 1];
        }
    }
    if (iter_range_ids.empty()) {
        printf("[RANGE_TIME INFO] No range matches the iteration regex, skip iterations/groups report.\n");
        return;
    }

    // iterations.csv
    std::string iter_filename = prefix + ".iterations.csv";
    std::ofstream iter_out(iter_filename);
    iter_out << std::fixed << std::setprecision(3);
    iter_out << "iter,range_id,num_tokens,cpu_us,kern_us,num_activities,gpu_span_us\n";
    for (size_t k = 0; k < iter_range_ids.size(); k++) {
        const auto& r = _ranges[iter_range_ids[k] - 1];
        uint64_t cpu_ns = r.cpu_end > r.cpu_start ? r.cpu_end - r.cpu_start : 0;
        uint64_t span_ns = r.gpu_end > r.gpu_start ? r.gpu_end - r.gpu_start : 0;
        iter_out << k << "," << iter_range_ids[k] << "," << iter_tokens[k] << ","
                 << ns_to_us(cpu_ns) << "," << ns_to_us(r.total_kern_ns) << ","
                 << r.total_activities << "," << ns_to_us(span_ns) << "\n";
    }
    iter_out.close();

    // Grouped ranges, and the kernel names that ran inside each of them.
    struct GroupRow {
        uint64_t range_id;
        std::string layer;
        std::string component;
        std::vector<std::string> kernels;
    };
    std::vector<GroupRow> rows;
    std::vector<int64_t> row_of(_ranges.size(), -1);
    for (size_t i = 0; i < _ranges.size(); i++) {
        std::smatch m;
        if (iter_of[i] < 0 || !std::regex_search(_ranges[i].name, m, group_re)) {
            continue;
        }
        GroupRow row;
        row.range_id = i + 1;
        row.layer = m.size() > 2 ? m[1].str() : "";
        row.component = m.size() > 2 ? m[2].str() : (m.size() > 1 ? m[1].str() : m[0].str());
        row_of[i] = rows.size();
        rows.push_back(row);
    }
    for (const auto& a : _activities) {
        // walk up from the launch range; the first grouped range found owns this activity
        for (uint64_t id = a.range; id != 0; id = _ranges[id - 1].parent) {
            if (row_of[id - 1] >= 0) {
                auto& names = rows[row_of[id - 1]].kernels;
                std::string name = short_kernel_name(a.name);
                if (std::find(names.begin(), names.end(), name) == names.end()) {
                    names.push_back(name);
                }
                break;
            }
        }
    }

    // groups.csv
    std::string group_filename = prefix + ".groups.csv";
    std::ofstream group_out(group_filename);
    group_out << std::fixed << std::setprecision(3);
    group_out << "iter,num_tokens,layer,component,range_id,cpu_us,kern_us,num_activities,gpu_span_us,kernels\n";

    // for the summary: (component, num_tokens) -> kernel times
    std::map<std::pair<std::string, int64_t>, std::vector<double>> stats;
    for (const auto& row : rows) {
        const auto& r = _ranges[row.range_id - 1];
        int64_t iter = iter_of[row.range_id - 1];
        uint64_t cpu_ns = r.cpu_end > r.cpu_start ? r.cpu_end - r.cpu_start : 0;
        uint64_t span_ns = r.gpu_end > r.gpu_start ? r.gpu_end - r.gpu_start : 0;
        std::string kernels;
        for (size_t j = 0; j < row.kernels.size(); j++) {
            kernels += (j ? "|" : "") + row.kernels[j];
        }
        group_out << iter << "," << iter_tokens[iter] << "," << row.layer << "," << row.component << ","
                  << row.range_id << "," << ns_to_us(cpu_ns) << "," << ns_to_us(r.total_kern_ns) << ","
                  << r.total_activities << "," << ns_to_us(span_ns) << "," << csv_quote(kernels) << "\n";
        stats[{row.component, iter_tokens[iter]}].push_back(ns_to_us(r.total_kern_ns));
    }
    group_out.close();

    printf("[RANGE_TIME INFO] Dumping %zu iterations to %s\n", iter_range_ids.size(), iter_filename.c_str());
    printf("[RANGE_TIME INFO] Dumping %zu grouped ranges to %s\n", rows.size(), group_filename.c_str());

    // Short summary: GPU time per grouped range, by component and number of tokens.
    printf("[RANGE_TIME INFO] %-12s %10s %8s %12s %12s\n", "component", "num_tokens", "count", "mean_us", "median_us");
    for (auto& [key, values] : stats) {
        std::sort(values.begin(), values.end());
        double sum = 0;
        for (double v : values) sum += v;
        size_t n = values.size();
        double median = n % 2 ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) / 2;
        printf("[RANGE_TIME INFO] %-12s %10ld %8zu %12.3f %12.3f\n",
               key.first.c_str(), (long)key.second, n, sum / n, median);
    }
    fflush(stdout);
}


void RangeTime::flush() {
    std::lock_guard<std::mutex> lock(_mutex);
    if (_ranges.empty() && _activities.empty()) {
        return;     // this process recorded nothing (e.g. a CPU-only helper process)
    }

    summarize();

    uint64_t in_range = 0, outside_range = 0, no_launch = 0;
    for (const auto& a : _activities) {
        if (!a.has_launch) no_launch++;
        else if (a.range == 0) outside_range++;
        else in_range++;
    }
    uint64_t still_open = 0;
    for (const auto& r : _ranges) {
        if (r.cpu_end == 0) still_open++;
    }

    printf("[RANGE_TIME INFO] pid %d: %zu ranges (%lu still open, %lu unmatched pops), "
           "%zu GPU activities (%lu in a range, %lu outside ranges, %lu without launch record)\n",
           getpid(), _ranges.size(), still_open, _unmatched_pops,
           _activities.size(), in_range, outside_range, no_launch);

    const char* result_dir = std::getenv("RESULT_DIR");
    const char* app_name = std::getenv("YOSEMITE_APP_NAME");
    std::string prefix = std::string(result_dir ? result_dir : ".") + "/"
                       + std::string(app_name ? app_name : "app") + "."
                       + std::to_string(getpid()) + ".range_time";

    dump_ranges(prefix);
    dump_activities(prefix);
    dump_groups(prefix);
    fflush(stdout);
}
