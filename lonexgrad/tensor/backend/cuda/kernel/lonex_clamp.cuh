// clamp.cuh
// Element-wise clamp kernel

#ifndef lonexgrad_BACKEND_CUDA_KERNEL_CLAMP_H_
#define lonexgrad_BACKEND_CUDA_KERNEL_CLAMP_H_

#include <lonexgrad/lonex_scalar.h>

#include "tensor/backend/common/lonex_util.h"
#include "tensor/backend/cuda/lonex_config.cuh"
#include "tensor/backend/cuda/lonex_data_types.cuh"

#include <nvfunctional>

namespace lonexgrad::cuda::kernel::clamp {

template <typename T>
__global__ void clamp_kernel(DataInfo<T> array, const DataInfo<const T> min, const DataInfo<const T> max, int N) {
    const auto i = GLOBAL_FLAT_THREAD_IDX;
    if (i < N) {
        const auto idx = to_flat_index(i, array.shape, array.stride, array.offset);
        const auto min_idx = to_flat_index(i, min.shape, min.stride, min.offset);
        const auto max_idx = to_flat_index(i, max.shape, max.stride, max.offset);
        array.data[idx] = std::min(std::max(array.data[idx], min.data[min_idx]), max.data[max_idx]);
    }
}

}    // namespace lonexgrad::cuda::kernel::clamp

#endif    // lonexgrad_BACKEND_CUDA_KERNEL_CLAMP_H_
