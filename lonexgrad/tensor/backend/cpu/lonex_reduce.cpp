


#include "tensor/backend/cpu/lonex_reduce.h"

#include <lonexgrad/lonex_concepts.h>
#include <lonexgrad/lonex_scalar.h>
#include <lonexgrad/lonex_shape.h>
#include <lonexgrad/lonex_tensor.h>

#include "tensor/backend/common/lonex_reduce.h"
#include "tensor/backend/common/lonex_span.h"
#include "tensor/backend/cpu/lonex_data_types.h"
#include "tensor/backend/cpu/kernel/reduce.hpp"
#include "tensor/backend/cpu/lonex_storage_cpu.h"

#include <cstddef>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace lonexgrad::cpu {

using namespace lonexgrad::common::reduce;
using namespace kernel::reduce;

template <ReduceOpT Op>
auto reduce_dim_runner(const Tensor &tensor, int dim) -> Tensor {
    auto res_shape = tensor.shape();
    const int RN = res_shape[dim];
    res_shape[dim] = 1;
    const int N = res_shape.numel();
    return std::visit(
        [&](auto &&tensor_storage) -> Tensor {
            using DT = std::remove_cvref_t<decltype(tensor_storage)>;    
            using T = template_parameter_t<std::remove_cvref_t<DT>>;     
            using V = Result<T, Op>::val_type;                           
            using R = Result<T, Op>::res_type;                           
            using KernelOp = typename OpFactory<V, Op>::KernelOp;

            const HostSpan<const T> a{tensor_storage};
            const auto shape = HostSpan<const int>(tensor.shape());
            const auto stride = HostSpan<const int>(tensor.stride());
            const DataInfo<const T> data_info{a, shape, stride, tensor.offset()};

            
            std::vector<R> result(static_cast<std::size_t>(N));

            reduce_dim_kernel(data_info, HostSpan<R>{result}, KernelOp{static_cast<V>(RN)}, dim, N);

            return {
                std::make_unique<StorageCPU>(std::move(result)),
                std::is_same_v<T, R> ? tensor.dtype() : to_scalar<R>::type,
                res_shape,
                tensor.device()
            };
        },
        tensor.get_storage<StorageCPU>().storage
    );
}

template <ReduceOpT Op>
auto reduce_all_runner(const Tensor &tensor) -> Tensor {
    const int N = tensor.shape().numel();
    return std::visit(
        [&](auto &&tensor_storage) -> Tensor {
            using DT = std::remove_cvref_t<decltype(tensor_storage)>;    
            using T = template_parameter_t<std::remove_cvref_t<DT>>;     
            using V = Result<T, Op>::val_type;                           
            using R = Result<T, Op>::res_type;                           
            using KernelOp = typename OpFactory<V, Op>::KernelOp;

            const HostSpan<const T> a{tensor_storage};
            const auto shape = HostSpan<const int>(tensor.shape());
            const auto stride = HostSpan<const int>(tensor.stride());
            const DataInfo<const T> data_info{a, shape, stride, tensor.offset()};

            
            std::vector<R> result(static_cast<std::size_t>(1));

            reduce_all_kernel(data_info, HostSpan<R>{result}, KernelOp{static_cast<V>(N)}, N);
            return {
                std::make_unique<StorageCPU>(std::move(result)),
                std::is_same_v<T, R> ? tensor.dtype() : to_scalar<R>::type,
                {1},
                tensor.device()
            };
        },
        tensor.get_storage<StorageCPU>().storage
    );
}

template Tensor reduce_dim_runner<ReduceOpT::min>(const Tensor &tensor, int dim);
template Tensor reduce_dim_runner<ReduceOpT::argmin>(const Tensor &tensor, int dim);
template Tensor reduce_dim_runner<ReduceOpT::max>(const Tensor &tensor, int dim);
template Tensor reduce_dim_runner<ReduceOpT::argmax>(const Tensor &tensor, int dim);
template Tensor reduce_dim_runner<ReduceOpT::sum>(const Tensor &tensor, int dim);
template Tensor reduce_dim_runner<ReduceOpT::all>(const Tensor &tensor, int dim);
template Tensor reduce_dim_runner<ReduceOpT::any>(const Tensor &tensor, int dim);

template Tensor reduce_all_runner<ReduceOpT::min>(const Tensor &tensor);
template Tensor reduce_all_runner<ReduceOpT::argmin>(const Tensor &tensor);
template Tensor reduce_all_runner<ReduceOpT::max>(const Tensor &tensor);
template Tensor reduce_all_runner<ReduceOpT::argmax>(const Tensor &tensor);
template Tensor reduce_all_runner<ReduceOpT::sum>(const Tensor &tensor);
template Tensor reduce_all_runner<ReduceOpT::all>(const Tensor &tensor);
template Tensor reduce_all_runner<ReduceOpT::any>(const Tensor &tensor);

}    
