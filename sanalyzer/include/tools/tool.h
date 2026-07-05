#ifndef YOSEMITE_TOOL_H
#define YOSEMITE_TOOL_H

#include "utils/event.h"
#include "tools/tool_type.h"

namespace yosemite {

class Tool {
public:
    Tool(AnalysisTool_t tool) : _tool(tool) {}

    virtual ~Tool() = default;

    virtual void evt_callback(EventPtr_t evt) = 0;

    virtual void gpu_data_analysis(void* data, uint64_t size) = 0;

    virtual void query_ranges(void* ranges, uint32_t limit, uint32_t* count) = 0;

    virtual void query_tensors(void* ranges, uint32_t limit, uint32_t* count) = 0;

    virtual void flush() = 0;
protected:
    AnalysisTool_t _tool;

    bool _torch_enabled = false;
};

}   // yosemite
#endif // YOSEMITE_TOOL_H