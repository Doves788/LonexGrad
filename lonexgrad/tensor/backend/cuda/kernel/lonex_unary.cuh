// unary.cuh
// Element-wise unary kernel

#ifndef lonexgrad_BACKEND_CUDA_KERNEL_UNARY_H_
#define lonexgrad_BACKEND_CUDA_KERNEL_UNARY_H_

#include "tensor/backend/common/lonex_util.h"
#include "tensor/backend/cuda/lonex_config.cuh"
#include "tensor/backend/cuda/lonex_data_types.cuh"

#include <cstddef>

namespace lonexgrad::cuda::kernel::unary {

template <bool CastBeforeOp, typename T, typename R, typename OP, typename... Args>
__global__ void unary_kernel(const DataInfo<const T> di, DeviceSpan<R> res, OP op, int N, Args... args) {
    const auto i = GLOBAL_FLAT_THREAD_IDX;
    if (i < N) {
        const auto idx = static_cast<std::size_t>(to_flat_index(i, di.shape, di.stride, di.offset));
        // Some ops need to cast input before op, some after
        if constexpr (CastBeforeOp) {
            res[i] = op()(static_cast<R>(di.data[idx]), args...);
        } else {
            res[i] = static_cast<R>(op()(di.data[idx], args...));
        }
    }
}

template <typename T, typename OP, typename... Args>
__global__ void unary_kernel_inplace(DataInfo<T> di, OP op, int N, Args... args) {
    const auto i = GLOBAL_FLAT_THREAD_IDX;
    if (i < N) {
        const auto idx = to_flat_index(i, di.shape, di.stride, di.offset);
        di.data[idx] = op()(di.data[idx], args...);
    }
}

}    // namespace lonexgrad::cuda::kernel::unary

#endif    // lonexgrad_BACKEND_CUDA_KERNEL_UNARY_H_
