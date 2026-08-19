// binary.cuh
// Element-wise binary kernel

#ifndef lonexgrad_BACKEND_CUDA_KERNEL_BINARY_H_
#define lonexgrad_BACKEND_CUDA_KERNEL_BINARY_H_

#include <lonexgrad/lonex_scalar.h>

#include "tensor/backend/common/lonex_util.h"
#include "tensor/backend/cuda/lonex_config.cuh"
#include "tensor/backend/cuda/lonex_data_types.cuh"

#include <nvfunctional>

namespace lonexgrad::cuda::kernel::binary {

template <bool CastBeforeOp, typename T, typename R, typename OP>
__global__ void
    binary_kernel(const DataInfo<const T> lhs, const DataInfo<const T> rhs, DeviceSpan<R> res, OP op, int N) {
    const auto i = GLOBAL_FLAT_THREAD_IDX;
    if (i < N) {
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
__global__ void binary_kernel(DataInfo<T> lhs, const DataInfo<const T> rhs, OP op, int N) {
    const auto i = GLOBAL_FLAT_THREAD_IDX;
    if (i < N) {
        const auto lhs_idx = to_flat_index(i, lhs.shape, lhs.stride, lhs.offset);
        const auto rhs_idx = to_flat_index(i, rhs.shape, rhs.stride, rhs.offset);
        lhs.data[lhs_idx] = op()(lhs.data[lhs_idx], rhs.data[rhs_idx]);
    }
}

}    // namespace lonexgrad::cuda::kernel::binary

#endif    // lonexgrad_BACKEND_CUDA_KERNEL_BINARY_H_
