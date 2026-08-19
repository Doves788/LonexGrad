


#ifndef lonexgrad_AUTOGRAD_CONV_OPS_H_
#define lonexgrad_AUTOGRAD_CONV_OPS_H_

#include <lonexgrad/lonex_autograd.h>
#include <lonexgrad/lonex_scalar.h>
#include <lonexgrad/lonex_tensor.h>

#include <optional>
#include <string>

namespace lonexgrad::autograd {

struct TensorConv2d : public TensorFunction<TensorConv2d> {
    static constexpr std::string name = "Conv2d";
    static auto forward(
        AutogradStorage &storage,
        bool is_grad_required,
        const Tensor &tensor,
        const Tensor &weight,
        const std::optional<Tensor> &bias,
        int stride,
        int padding
    ) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorMaxPool2d : public TensorFunction<TensorMaxPool2d> {
    static constexpr std::string name = "MaxPool2d";
    static auto forward(
        AutogradStorage &storage,
        bool is_grad_required,
        const Tensor &tensor,
        int kernel_size,
        int stride,
        int padding
    ) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorMinPool2d : public TensorFunction<TensorMinPool2d> {
    static constexpr std::string name = "MinPool2d";
    static auto forward(
        AutogradStorage &storage,
        bool is_grad_required,
        const Tensor &tensor,
        int kernel_size,
        int stride,
        int padding
    ) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorAvgPool2d : public TensorFunction<TensorAvgPool2d> {
    static constexpr std::string name = "AvgPool2d";
    static auto forward(
        AutogradStorage &storage,
        bool is_grad_required,
        const Tensor &tensor,
        int kernel_size,
        int stride,
        int padding
    ) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

}    

#endif    
