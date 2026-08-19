// assign.cuh
// Assignment kernels

#ifndef lonexgrad_BACKEND_CUDA_KERNEL_ASSIGN_H_
#define lonexgrad_BACKEND_CUDA_KERNEL_ASSIGN_H_

#include <lonexgrad/lonex_scalar.h>

#include "tensor/backend/common/lonex_util.h"
#include "tensor/backend/cuda/lonex_config.cuh"
#include "tensor/backend/cuda/lonex_data_types.cuh"

namespace lonexgrad::cuda::kernel::assign {

template <typename T>
__global__ void assign_kernel(DataInfo<T> lhs, const DataInfo<const T> rhs, int N) {
    const auto i = GLOBAL_FLAT_THREAD_IDX;
    if (i < N) {
        const auto lhs_idx = to_flat_index(i, lhs.shape, lhs.stride, lhs.offset);
        const auto rhs_idx = to_flat_index(i, rhs.shape, rhs.stride, rhs.offset);
        lhs.data[lhs_idx] = rhs.data[rhs_idx];
    }
}

}    // namespace lonexgrad::cuda::kernel::assign

#endif    // lonexgrad_BACKEND_CUDA_KERNEL_BINARY_H_
