


#include "tensor/backend/cpu/lonex_binary.h"

#include <lonexgrad/lonex_concepts.h>
#include <lonexgrad/lonex_scalar.h>
#include <lonexgrad/lonex_tensor.h>

#include "tensor/backend/common/lonex_binary.h"
#include "tensor/backend/common/lonex_span.h"
#include "tensor/backend/cpu/lonex_data_types.h"
#include "tensor/backend/cpu/kernel/binary.hpp"
#include "tensor/backend/cpu/lonex_storage_cpu.h"

#include <cstddef>
#include <type_traits>
#include <variant>
#include <vector>

namespace lonexgrad::cpu {

using namespace lonexgrad::common::binary;
using namespace kernel::binary;

template <BinaryOpT Op>
auto binary_runner(const Tensor &lhs, const Tensor &rhs) -> Tensor {
    const int N = lhs.numel();
    return std::visit(
        [&](auto &&array_storage) -> Tensor {
            using DT = std::remove_cvref_t<decltype(array_storage)>;    
            using T = template_parameter_t<DT>;                         
            using R = Result<T, Op>::type;                              
            
            constexpr bool CastBeforeOp = Result<R, Op>::CastBeforeOp;
            using KernelOp = typename OpFactory<std::conditional_t<CastBeforeOp, R, T>, Op>::KernelOp;

            
            std::vector<R> result(static_cast<std::size_t>(N));

            
            const HostSpan<const T> lhs_span{array_storage};
            const HostSpan<const T> rhs_span{std::get<DT>(rhs.template get_storage<StorageCPU>().storage)};

            
            const DataInfo<const T> l{lhs_span, lhs.shape(), lhs.stride(), lhs.offset()};
            const DataInfo<const T> r{rhs_span, rhs.shape(), rhs.stride(), rhs.offset()};

            
            binary_kernel<CastBeforeOp>(l, r, HostSpan<R>(result), KernelOp{}, N);
            return {
                std::make_unique<StorageCPU>(std::move(result)),
                Result<T, Op>::scalar(lhs.dtype()),
                lhs.shape(),
                lhs.device()
            };
        },
        lhs.template get_storage<StorageCPU>().storage
    );
}

template <BinaryOpT Op>
void binary_inplace_runner(Tensor &lhs, const Tensor &rhs) {
    const int N = lhs.numel();
    return std::visit(
        [&](auto &&array_storage) {
            using DT = std::remove_cvref_t<decltype(array_storage)>;    
            using T = template_parameter_t<DT>;                         
            using KernelOp = typename OpFactory<T, Op>::KernelOp;

            
            HostSpan<T> lhs_span{array_storage};
            const HostSpan<const T> rhs_span{std::get<DT>(rhs.template get_storage<StorageCPU>().storage)};

            
            DataInfo<T> l{lhs_span, lhs.shape(), lhs.stride(), lhs.offset()};
            const DataInfo<const T> r{rhs_span, rhs.shape(), rhs.stride(), rhs.offset()};

            
            binary_kernel(l, r, KernelOp{}, N);
        },
        lhs.template get_storage<StorageCPU>().storage
    );
}

template Tensor binary_runner<BinaryOpT::add>(const Tensor &lhs, const Tensor &rhs);
template Tensor binary_runner<BinaryOpT::subtract>(const Tensor &lhs, const Tensor &rhs);
template Tensor binary_runner<BinaryOpT::multiply>(const Tensor &lhs, const Tensor &rhs);
template Tensor binary_runner<BinaryOpT::divide>(const Tensor &lhs, const Tensor &rhs);
template Tensor binary_runner<BinaryOpT::equal>(const Tensor &lhs, const Tensor &rhs);
template Tensor binary_runner<BinaryOpT::not_equal>(const Tensor &lhs, const Tensor &rhs);
template Tensor binary_runner<BinaryOpT::less_than>(const Tensor &lhs, const Tensor &rhs);
template Tensor binary_runner<BinaryOpT::less_than_eq>(const Tensor &lhs, const Tensor &rhs);
template Tensor binary_runner<BinaryOpT::greater_than>(const Tensor &lhs, const Tensor &rhs);
template Tensor binary_runner<BinaryOpT::greater_than_eq>(const Tensor &lhs, const Tensor &rhs);
template Tensor binary_runner<BinaryOpT::logical_or>(const Tensor &lhs, const Tensor &rhs);
template Tensor binary_runner<BinaryOpT::logical_and>(const Tensor &lhs, const Tensor &rhs);
template Tensor binary_runner<BinaryOpT::bitwise_or>(const Tensor &lhs, const Tensor &rhs);
template Tensor binary_runner<BinaryOpT::bitwise_and>(const Tensor &lhs, const Tensor &rhs);
template Tensor binary_runner<BinaryOpT::bitwise_xor>(const Tensor &lhs, const Tensor &rhs);
template Tensor binary_runner<BinaryOpT::bitwise_left_shift>(const Tensor &lhs, const Tensor &rhs);
template Tensor binary_runner<BinaryOpT::bitwise_right_shift>(const Tensor &lhs, const Tensor &rhs);
template Tensor binary_runner<BinaryOpT::modulo>(const Tensor &lhs, const Tensor &rhs);
template Tensor binary_runner<BinaryOpT::minimum>(const Tensor &lhs, const Tensor &rhs);
template Tensor binary_runner<BinaryOpT::maximum>(const Tensor &lhs, const Tensor &rhs);
template Tensor binary_runner<BinaryOpT::pow>(const Tensor &lhs, const Tensor &rhs);

template void binary_inplace_runner<BinaryOpT::add>(Tensor &lhs, const Tensor &rhs);
template void binary_inplace_runner<BinaryOpT::subtract>(Tensor &lhs, const Tensor &rhs);
template void binary_inplace_runner<BinaryOpT::multiply>(Tensor &lhs, const Tensor &rhs);
template void binary_inplace_runner<BinaryOpT::divide>(Tensor &lhs, const Tensor &rhs);

}    
