// config.cuh
// CUDA configuration and kernel options

#ifndef lonexgrad_CUDA_CONFIG_H_
#define lonexgrad_CUDA_CONFIG_H_

#include <lonexgrad/lonex_util.h>

#include <cuda_runtime.h>

#include <cstddef>

namespace lonexgrad::cuda {

constexpr int THREADS_PER_BLOCK = 512;
constexpr unsigned int WARP_SIZE = 32;

#define GLOBAL_FLAT_THREAD_IDX static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x)

constexpr auto block_1d() -> dim3 {
    return {THREADS_PER_BLOCK, 1, 1};
}

constexpr auto grid_1d(int N, int work_per_thread_factor = 1) -> dim3 {
    return {static_cast<unsigned int>(ceil_div(N, work_per_thread_factor * THREADS_PER_BLOCK)), 1, 1};
}

constexpr auto block_2d() -> dim3 {
    return {WARP_SIZE, WARP_SIZE, 1};
}

}    // namespace lonexgrad::cuda

#endif    // lonexgrad_CUDA_KERNEL_LAUNCH_H_
