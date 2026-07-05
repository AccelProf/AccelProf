#ifndef _TORCH_OPERATOR_H_
#define _TORCH_OPERATOR_H_

#include <ATen/record_function.h>

#include "torch_scope.h"


class TorchOperator {
public:
    static TorchOperator& getInstance();

    void enable_torch_callback();

    void register_operator_callback(TorchScopeType_t scope_type,
                                  operator_callback_t callback_ptr);

    static std::unique_ptr<at::ObserverContext> operator_start_callback(const at::RecordFunction& fn);

    static void operator_end_callback(const at::RecordFunction& fn, at::ObserverContext* ctx);

private:
    TorchOperator() {}
    ~TorchOperator() {}

    at::CallbackHandle operator_callback_handle_;

    operator_callback_t operator_start_callback_ptr = nullptr;
    operator_callback_t operator_end_callback_ptr = nullptr;
};  // class TorchOperator


#endif //_TORCH_OPERATOR_H_
