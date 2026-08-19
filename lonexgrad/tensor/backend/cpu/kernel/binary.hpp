


#ifndef lonexgrad_BACKEND_CPU_KERNEL_BINARY_H_
#define lonexgrad_BACKEND_CPU_KERNEL_BINARY_H_

#include <lonexgrad/lonex_scalar.h>

#include "tensor/backend/common/lonex_span.h"
#include "tensor/backend/common/lonex_util.h"
#include "tensor/backend/cpu/lonex_data_types.h"

namespace lonexgrad::cpu::kernel::binary {

template <bool CastBeforeOp, typename T, typename R, typename OP>
void binary_kernel(const DataInfo<const T> lhs, const DataInfo<const T> rhs, HostSpan<R> res, OP op, int N) {
    for (int i = 0; i < N; ++i) {
        const auto lhs_idx = to_flat_index(i, lhs.shape, lhs.stride, lhs.offset);
        const auto rhs_idx = to_flat_index(i, rhs.shape, rhs.stride, rhs.offset);
        if constexpr (CastBeforeOp) {
            res[i] = op()(static_cast<R>(lhs.data[lhs_idx]), static_cast<R>(rhs.data[rhs_idx]));
        } else {
            res[i] = static_cast<R>(op()(lhs.data[lhs_idx], rhs.data[rhs_idx]));
        }
    }
}

template <typename T, typename OP>
void binary_kernel(DataInfo<T> lhs, const DataInfo<const T> rhs, OP op, int N) {
    for (int i = 0; i < N; ++i) {
        const auto lhs_idx = to_flat_index(i, lhs.shape, lhs.stride, lhs.offset);
        const auto rhs_idx = to_flat_index(i, rhs.shape, rhs.stride, rhs.offset);
        lhs.data[lhs_idx] = op()(lhs.data[lhs_idx], rhs.data[rhs_idx]);
    }
}

}    

#endif    
