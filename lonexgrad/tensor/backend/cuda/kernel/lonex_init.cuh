// init.cuh
// Initialization kernel runner

#ifndef lonexgrad_BACKEND_CUDA_KERNEL_INIT_H_
#define lonexgrad_BACKEND_CUDA_KERNEL_INIT_H_

#include "tensor/backend/cuda/lonex_config.cuh"

namespace lonexgrad::cuda::kernel::init {

// Each thread sets from the same value
template <typename T>
__global__ void init_full_kernel(T *data, const T value, int N) {
    const auto i = GLOBAL_FLAT_THREAD_IDX;
    if (i < N) {
        data[i] = value;    // NOLINT(*-pointer-arithmetic)
    }
}

// Threads set sequentially by thread idx
template <typename T>
__global__ void init_arange_kernel(T *data, int N) {
    const auto i = GLOBAL_FLAT_THREAD_IDX;
    if (i < N) {
        data[i] = static_cast<T>(i);    // NOLINT(*-pointer-arithmetic)
    }
}

}    // namespace lonexgrad::cuda::kernel::init

#endif    // lonexgrad_BACKEND_CUDA_KERNEL_INIT_H_
