


#ifndef lonexgrad_BACKEND_CPU_CONV_H_
#define lonexgrad_BACKEND_CPU_CONV_H_

#include <lonexgrad/lonex_tensor.h>

#include "tensor/backend/common/lonex_reduce.h"

#include <optional>
#include <tuple>

namespace lonexgrad::cpu {

auto batched_conv2d_forward_runner(
    const Tensor &input,
    const Tensor &weight,
    const std::optional<Tensor> &bias,
    int stride,
    int padding
) -> Tensor;

auto batched_conv2d_backward_runner(
    const Tensor &grad_output,
    const Tensor &input,
    const Tensor &weight,
    const std::optional<Tensor> &bias,
    int stride,
    int padding
) -> std::tuple<Tensor, Tensor, std::optional<Tensor>>;

template <common::reduce::ReduceOpT Op>
auto batched_pool2d_forward_runner(const Tensor &input, int kernel_size, int stride, int padding) -> Tensor;

template <common::reduce::ReduceOpT Op>
auto batched_pool2d_backward_runner(
    const Tensor &grad_output,
    const Tensor &input,
    const Tensor &result,
    int kernel_size,
    int stride,
    int padding
) -> Tensor;

}    

#endif    
