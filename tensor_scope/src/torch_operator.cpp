#include "torch_operator.h"
#include "torch_tensor.h"

struct OperatorCallbackContext : at::ObserverContext {
    std::string op_name;
};

TorchOperator& TorchOperator::getInstance() {
    static TorchOperator instance;
    return instance;
}

void TorchOperator::enable_torch_callback() {
    auto callback = at::RecordFunctionCallback(
            &TorchOperator::operator_start_callback,
            &TorchOperator::operator_end_callback
        ).scopes({at::RecordScope::FUNCTION});
    operator_callback_handle_ = at::addGlobalCallback(callback);
}

void TorchOperator::register_operator_callback(TorchScopeType_t scope_type,
                                         operator_callback_t callback_ptr) {
    if (scope_type == TORCH_SCOPE_OPERATOR_START) {
        this->operator_start_callback_ptr = callback_ptr;
    } else if (scope_type == TORCH_SCOPE_OPERATOR_END) {
        this->operator_end_callback_ptr = callback_ptr;
    }
}

std::unique_ptr<at::ObserverContext> TorchOperator::operator_start_callback(const at::RecordFunction& fn) {
    // Op start is outside the caching-allocator lock, and by the first CUDA op
    // the allocator has been initialized: the right moment to hook it.
    TorchTensor::ensure_tracker_attached();
    if (getInstance().operator_start_callback_ptr == nullptr) {
        printf("operator_start_callback_ptr is nullptr\n");
    }
    auto op_ctx = std::make_unique<OperatorCallbackContext>();
    op_ctx->op_name = fn.name();
    getInstance().operator_start_callback_ptr(static_cast<void*>(op_ctx.get()), op_ctx->op_name);
    return op_ctx;
}

void TorchOperator::operator_end_callback(const at::RecordFunction& fn, at::ObserverContext* ctx) {
    if (getInstance().operator_end_callback_ptr == nullptr) {
        printf("operator_end_callback_ptr is nullptr\n");
    }
    auto op_ctx = static_cast<OperatorCallbackContext*>(ctx);
    getInstance().operator_end_callback_ptr((void*)op_ctx, op_ctx->op_name);
}
