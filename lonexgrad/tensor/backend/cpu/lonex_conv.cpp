


#include "tensor/backend/cpu/lonex_conv.h"

#include <lonexgrad/lonex_concepts.h>
#include <lonexgrad/lonex_scalar.h>
#include <lonexgrad/lonex_shape.h>
#include <lonexgrad/lonex_tensor.h>

#include "tensor/backend/common/lonex_binary.h"
#include "tensor/backend/common/lonex_reduce.h"
#include "tensor/backend/common/lonex_span.h"
#include "tensor/backend/cpu/lonex_data_types.h"
#include "tensor/backend/cpu/kernel/binary.hpp"
#include "tensor/backend/cpu/kernel/conv.hpp"
#include "tensor/backend/cpu/kernel/matmul.hpp"
#include "tensor/backend/cpu/lonex_storage_cpu.h"

#include <cassert>
#include <cstddef>
#include <optional>
#include <tuple>
#include <type_traits>
#include <variant>
#include <vector>

namespace lonexgrad::cpu {

using namespace lonexgrad::common::binary;
using namespace lonexgrad::common::reduce;
using namespace kernel::conv;
using namespace kernel::matmul;
using namespace kernel::binary;

auto batched_conv2d_forward_runner(
    const Tensor &input,
    const Tensor &weight,
    const std::optional<Tensor> &bias,
    int stride,
    int padding
) -> Tensor {
    assert(input.dim() == 4 && weight.dim() == 4);
    assert(input.size(1) == weight.size(1));     
    assert(weight.size(2) == weight.size(3));    
    assert(stride > 0);
    assert(padding >= 0);
    
    const int B = input.size(0);
    const int C_in = input.size(1);
    const int H_in = input.size(2);
    const int W_in = input.size(3);
    const int kernel_size = weight.size(2);

    
    const int C_out = weight.size(0);
    const int H_out = (H_in + 2 * padding - kernel_size) / stride + 1;
    const int W_out = (W_in + 2 * padding - kernel_size) / stride + 1;

    
    const int input_unrolled_height = C_in * kernel_size * kernel_size;
    const int input_unrolled_width = H_out * W_out;
    const int input_unrolled_size = input_unrolled_height * input_unrolled_width;

    
    const auto res_shape = Shape({B, C_out, H_out, W_out});

    
    const Tensor w = weight.reshape({C_out, C_in * kernel_size * kernel_size});

    
    const auto input_cont = input.contiguous();

    
    std::optional<Tensor> expanded_bias;
    if (bias) {
        expanded_bias = bias->contiguous().reshape({C_out, 1, 1}).expand({C_out, H_out, W_out});
    }
    
    return std::visit(
        [&](auto &&tensor_storage) -> Tensor {
            using DT = std::remove_cvref_t<decltype(tensor_storage)>;    
            using T = template_parameter_t<DT>;                          
            using KernelOp = typename common::binary::OpFactory<T, BinaryOpT::add>::KernelOp;

            
            std::vector<T> result(static_cast<std::size_t>(res_shape.numel()));

            
            std::vector<T> input_unrolled(static_cast<std::size_t>(input_unrolled_size));

            
            
            const T *input_data_ptr = tensor_storage.data();
            const T *w_data_ptr = std::get<DT>(w.template get_storage<StorageCPU>().storage).data();
            input_data_ptr += input_cont.offset();    // NOLINT(*-pointer-arithmetic)
            w_data_ptr += w.offset();                 // NOLINT(*-pointer-arithmetic)
            const HostSpan<const T> input_span{input_data_ptr, static_cast<std::size_t>(input_cont.numel())};
            const HostSpan<const T> w_span{w_data_ptr, static_cast<std::size_t>(w.numel())};
            HostSpan<T> input_unrolled_span{input_unrolled};
            T *res_data_ptr = result.data();

            const int mat_N = C_out;
            const int mat_K = input_unrolled_height;
            const int mat_M = input_unrolled_width;

            
            for (int b = 0; b < B; ++b) {
                
                unroll_kernel(input_span, input_unrolled_span, b, C_in, H_in, W_in, kernel_size, stride, padding);
                
                const HostSpan<T> res_span{res_data_ptr, static_cast<std::size_t>(C_out * H_out * W_out)};
                matmul_kernel(w_span, HostSpan<const T>{input_unrolled_span}, res_span, mat_N, mat_K, mat_M);

                if (bias) {
                    DataInfo<T> res_info{res_span};
                    const DataInfo<const T> bias_info{
                        {std::get<DT>(expanded_bias->template get_storage<StorageCPU>().storage)},
                        expanded_bias->shape(),
                        expanded_bias->stride(),
                        expanded_bias->offset()
                    };
                    binary_kernel(res_info, bias_info, KernelOp{}, C_out * H_out * W_out);
                }

                
                res_data_ptr += (C_out * H_out * W_out);    // NOLINT(*-pointer-arithmetic)
            }

            return {std::make_unique<StorageCPU>(std::move(result)), to_scalar<T>::type, res_shape, input.device()};
        },
        input_cont.template get_storage<StorageCPU>().storage
    );
}

auto batched_conv2d_backward_runner(
    const Tensor &grad_output,
    const Tensor &input,
    const Tensor &weight,
    const std::optional<Tensor> &bias,
    int stride,
    int padding
) -> std::tuple<Tensor, Tensor, std::optional<Tensor>> {
    assert(input.dim() == 4 && weight.dim() == 4);
    assert(input.size(1) == weight.size(1));     
    assert(weight.size(2) == weight.size(3));    
    assert(stride > 0);
    assert(padding >= 0);
    
    const int B = input.size(0);
    const int C_in = input.size(1);
    const int H_in = input.size(2);
    const int W_in = input.size(3);
    const int kernel_size = weight.size(2);

    
    const int C_out = weight.size(0);
    const int H_out = (H_in + 2 * padding - kernel_size) / stride + 1;
    const int W_out = (W_in + 2 * padding - kernel_size) / stride + 1;

    
    const int input_unrolled_height = C_in * kernel_size * kernel_size;
    const int input_unrolled_width = H_out * W_out;
    const int input_unrolled_size = input_unrolled_height * input_unrolled_width;

    
    const auto res_shape = Shape({B, C_out, H_out, W_out});
    assert(grad_output.shape() == res_shape);

    
    const Tensor w = weight.reshape({C_out, C_in * kernel_size * kernel_size}).permute({1, 0}).contiguous();

    
    const auto input_cont = input.contiguous();

    
    return std::visit(
        [&](auto &&tensor_storage) -> std::tuple<Tensor, Tensor, std::optional<Tensor>> {
            using DT = std::remove_cvref_t<decltype(tensor_storage)>;    
            using T = template_parameter_t<DT>;                          
            using KernelOp = typename common::binary::OpFactory<T, BinaryOpT::add>::KernelOp;

            
            std::vector<T> grad_input_fold(static_cast<std::size_t>(input.numel()));
            std::vector<T> grad_w_local(static_cast<std::size_t>(weight.numel()), 0);
            std::vector<T> grad_w_global(static_cast<std::size_t>(weight.numel()), 0);

            
            std::vector<T> input_unrolled(static_cast<std::size_t>(input_unrolled_size), 0);

            
            
            const T *input_data_ptr = tensor_storage.data();
            const T *w_data_ptr = std::get<DT>(w.template get_storage<StorageCPU>().storage).data();
            input_data_ptr += input_cont.offset();    // NOLINT(*-pointer-arithmetic)
            w_data_ptr += w.offset();                 // NOLINT(*-pointer-arithmetic)
            const HostSpan<const T> input_span{input_data_ptr, static_cast<std::size_t>(input_cont.numel())};
            const HostSpan<const T> w_span{w_data_ptr, static_cast<std::size_t>(w.numel())};
            const HostSpan<const T> grad_out_span{std::get<DT>(grad_output.template get_storage<StorageCPU>().storage)};
            HostSpan<T> g_w_local_span{grad_w_local};
            HostSpan<T> g_w_global_span{grad_w_global};

            HostSpan<T> input_unrolled_span{input_unrolled};
            T *grad_input_data_ptr = grad_input_fold.data();
            const T *g_out_data_ptr = grad_out_span.get();

            
            for (int b = 0; b < B; ++b) {
                
                unroll_transpose_kernel(
                    input_span,
                    input_unrolled_span,
                    b,
                    C_in,
                    H_in,
                    W_in,
                    kernel_size,
                    stride,
                    padding
                );
                HostSpan<T> grad_input_span{grad_input_data_ptr, static_cast<std::size_t>(C_in * H_in * W_in)};
                const HostSpan<const T> g_out_slice_span{
                    g_out_data_ptr,
                    static_cast<std::size_t>(C_out * H_out * W_out)
                };

                
                
                
                
                matmul_kernel(
                    g_out_slice_span,
                    HostSpan<const T>{input_unrolled_span},
                    g_w_local_span,
                    C_out,
                    input_unrolled_width,
                    input_unrolled_height
                );
                binary_kernel(
                    DataInfo<T>{g_w_global_span},
                    DataInfo<const T>{g_w_local_span},
                    KernelOp{},
                    weight.numel()
                );

                
                
                
                matmul_kernel(
                    w_span,
                    g_out_slice_span,
                    input_unrolled_span,
                    input_unrolled_height,
                    C_out,
                    input_unrolled_width
                );
                col2im_kernel(
                    grad_input_span,
                    HostSpan<const T>{input_unrolled_span},
                    0,
                    C_in,
                    H_in,
                    W_in,
                    kernel_size,
                    stride,
                    padding
                );

                
                g_out_data_ptr += (C_out * H_out * W_out);      // NOLINT(*-pointer-arithmetic)
                grad_input_data_ptr += (C_in * H_in * W_in);    // NOLINT(*-pointer-arithmetic)
            }

            Tensor grad_input(
                std::make_unique<StorageCPU>(std::move(grad_input_fold)),
                to_scalar<T>::type,
                input.shape(),
                input.device()
            );
            Tensor grad_w(
                std::make_unique<StorageCPU>(std::move(grad_w_global)),
                to_scalar<T>::type,
                weight.shape(),
                weight.device()
            );
            return {
                grad_input,
                grad_w,
                bias.has_value() ? std::optional<Tensor>{grad_output.sum(3).sum(2).sum(0)} : std::nullopt
            };
        },
        input_cont.template get_storage<StorageCPU>().storage
    );
}

template <ReduceOpT Op>
auto batched_pool2d_forward_runner(const Tensor &input, int kernel_size, int stride, int padding) -> Tensor {
    assert(input.dim() == 4);
    assert(kernel_size > 0);
    assert(stride > 0);
    assert(padding >= 0);
    
    const int B = input.size(0);
    const int C = input.size(1);
    const int H_in = input.size(2);
    const int W_in = input.size(3);

    
    const int H_out = (H_in + 2 * padding - kernel_size) / stride + 1;
    const int W_out = (W_in + 2 * padding - kernel_size) / stride + 1;

    
    const auto res_shape = Shape({B, C, H_out, W_out});

    
    const auto input_cont = input.contiguous();

    
    return std::visit(
        [&](auto &&tensor_storage) -> Tensor {
            using DT = std::remove_cvref_t<decltype(tensor_storage)>;    
            using T = template_parameter_t<DT>;                          
            using R = common::reduce::Result<T, Op>::type;               
            using KernelOp = typename common::reduce::OpFactory<R, Op>::KernelOp;

            
            std::vector<R> result(static_cast<std::size_t>(res_shape.numel()));

            
            
            const T *input_data_ptr = tensor_storage.data();
            input_data_ptr += input_cont.offset();    // NOLINT(*-pointer-arithmetic)
            const HostSpan<const T> input_span{input_data_ptr, static_cast<std::size_t>(input_cont.numel())};

            pool_kernel(
                input_span,
                HostSpan<R>{result},
                KernelOp{static_cast<R>(kernel_size * kernel_size)},
                B,
                C,
                H_in,
                W_in,
                kernel_size,
                stride,
                padding
            );

            return {std::make_unique<StorageCPU>(std::move(result)), to_scalar<R>::type, res_shape, input.device()};
        },
        input_cont.template get_storage<StorageCPU>().storage
    );
}

template <ReduceOpT Op>
auto batched_pool2d_backward_runner(
    const Tensor &grad_output,
    const Tensor &input,
    const Tensor &result,
    int kernel_size,
    int stride,
    int padding
) -> Tensor {
    assert(input.dim() == 4);
    assert(kernel_size > 0);
    assert(stride > 0);
    assert(padding >= 0);
    
    const int B = input.size(0);
    const int C = input.size(1);
    const int H_in = input.size(2);
    const int W_in = input.size(3);

    
    const int H_out = (H_in + 2 * padding - kernel_size) / stride + 1;
    const int W_out = (W_in + 2 * padding - kernel_size) / stride + 1;

    
    const auto res_shape = Shape({B, C, H_out, W_out});

    
    const auto input_cont = input.contiguous();

    
    return std::visit(
        [&](auto &&tensor_storage) -> Tensor {
            using DT = std::remove_cvref_t<decltype(tensor_storage)>;    
            using T = template_parameter_t<DT>;                          
            using R = common::reduce::Result<T, Op>::type;               
            using DR = std::vector<R>;
            using KernelOp = typename common::reduce::OpFactory<R, Op>::KernelOp;

            
            std::vector<R> grad_input_memory(static_cast<std::size_t>(input.numel()));

            
            
            const T *input_data_ptr = tensor_storage.data();
            input_data_ptr += input_cont.offset();    // NOLINT(*-pointer-arithmetic)
            const HostSpan<const T> input_span{input_data_ptr, static_cast<std::size_t>(input_cont.numel())};
            const HostSpan<const R> res_span{std::get<DR>(result.template get_storage<StorageCPU>().storage)};
            const HostSpan<const R> grad_out_span{std::get<DR>(grad_output.template get_storage<StorageCPU>().storage)};

            pool_backward_kernel<T, R, KernelOp>(
                grad_out_span,
                HostSpan<R>{grad_input_memory},
                input_span,
                res_span,
                B,
                C,
                H_in,
                W_in,
                kernel_size,
                stride,
                padding
            );

            return {
                std::make_unique<StorageCPU>(std::move(grad_input_memory)),
                to_scalar<R>::type,
                input.shape(),
                input.device()
            };
        },
        input_cont.template get_storage<StorageCPU>().storage
    );
}

template Tensor batched_pool2d_forward_runner<ReduceOpT::min>(const Tensor &, int, int, int);
template Tensor batched_pool2d_forward_runner<ReduceOpT::max>(const Tensor &, int, int, int);
template Tensor batched_pool2d_forward_runner<ReduceOpT::mean>(const Tensor &, int, int, int);

template Tensor
    batched_pool2d_backward_runner<ReduceOpT::min>(const Tensor &, const Tensor &, const Tensor &, int, int, int);
template Tensor
    batched_pool2d_backward_runner<ReduceOpT::max>(const Tensor &, const Tensor &, const Tensor &, int, int, int);
template Tensor
    batched_pool2d_backward_runner<ReduceOpT::mean>(const Tensor &, const Tensor &, const Tensor &, int, int, int);

}    
