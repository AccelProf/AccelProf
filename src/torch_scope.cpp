#include "torch_scope.h"

#include "torch_tensor.h"
#include "torch_operator.h"

void register_torch_scope_callback(TorchScopeType_t scope_type,
                           torch_scope_callback_t callback_ptr) {
    if (scope_type == TORCH_SCOPE_TENSOR_MALLOC || scope_type == TORCH_SCOPE_TENSOR_FREE) {
        TorchTensor* torch_tensor = TorchTensor::getInstance();
        torch_tensor->register_tensor_callback(scope_type, (tensor_callback_t)callback_ptr);
    } else if (scope_type == TORCH_SCOPE_OPERATOR_START || scope_type == TORCH_SCOPE_OPERATOR_END) {
        TorchOperator& torch_operator = TorchOperator::getInstance();
        torch_operator.register_operator_callback(scope_type, (operator_callback_t)callback_ptr);
    }
    fflush(stdout);
}

void enable_torch_scope() {
    TorchTensor* torch_tensor = TorchTensor::getInstance();
    torch_tensor->enable_torch_callback();
    TorchOperator& torch_operator = TorchOperator::getInstance();
    torch_operator.enable_torch_callback();
    fflush(stdout);
}
