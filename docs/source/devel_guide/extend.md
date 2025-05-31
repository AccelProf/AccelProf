# How to Extend?


## Add new tool support

In `sanalyzer/include/tools/tool_type.h`, add new tool name.

```C++
// Add new tool name in the following enum.
typedef enum {
    CODE_CHECK = 0,
    APP_METRICE = 1,
    MEM_TRACE = 2,
    HOT_ANALYSIS = 3,
    UVM_ADVISOR = 4,
    TOOL_NUMS = 5
} AnalysisTool_t;

```

## Reuse tool template for customized analysis

To develop a new tool to do customized analysis on PASTA. Developer can choose to inherit the base `class Tool` to do customized analysis. Within the tool template, developer can collect the needed runtime information based on their own needs. Developer collect these runtime information by implementing the (subset of) interfaces provided by PASTA.

```C++
class CustomTool final : public Tool {
public:
    CustomTool() : Tool(CUSTOM_ANALYSIS) {
        init();
    }

    void init();

    ~CustomTool() {}

    void evt_callback(EventPtr_t evt);

    void gpu_data_analysis(void* data, uint64_t size);

    void query_ranges(void* ranges, uint32_t limit, uint32_t* count);

    void flush();
};

```

## Add analysis code

After declaring the class, developer can add analysis code. User can choose all or some of the following analysis, or the combination of the following analysis, feeding their own performance analysis needs.

```C++
void CustomTool::evt_callback(EventPtr_t evt) {
    // handle different events
}


void CustomTool::kernel_start_callback(std::shared_ptr<KernelLauch_t> kernel) {
    // kernel analysis
}

void CustomTool::kernel_end_callback(std::shared_ptr<KernelLauch_t> kernel) {
    // kernel analysis
}

void CustomTool::mem_alloc_callback(std::shared_ptr<MemAlloc_t> mem) {
    // allocation analysis
}


void CustomTool::mem_free_callback(std::shared_ptr<MemFree_t> mem) {
    // reclaimation analysis
}



void CustomTool::mem_cpy_callback(std::shared_ptr<MemCpy_t> mem) {
    // memory transfer analysis
}


void CustomTool::mem_set_callback(std::shared_ptr<MemSet_t> mem) {
    // memory set analysis
}


void CustomTool::ten_alloc_callback(std::shared_ptr<TenAlloc_t> ten) {
    // tensor analysis
}


void CustomTool::ten_free_callback(std::shared_ptr<TenFree_t> ten) {
    // tensor analysis
}


void CustomTool::op_start_callback(std::shared_ptr<OpStart_t> op) {
    // op analysis
}

void CustomTool::op_end_callback(std::shared_ptr<OpStart_t> op) {
    // op analysis
}

void CustomTool::gpu_data_analysis(void* data, uint64_t size) {
    // memory access analysis
}


void CustomTool::query_ranges(void* ranges, uint32_t limit, uint32_t* count) {
    // check memory boundary
}


void CustomTool::flush() {
    // called at the end of the program
    // dump the analysis result
}
```
