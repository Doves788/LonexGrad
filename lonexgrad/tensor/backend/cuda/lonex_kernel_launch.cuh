// kernel_launch.cuh
// Kernel launch helper

#ifndef lonexgrad_CUDA_KERNEL_LAUNCH_H_
#define lonexgrad_CUDA_KERNEL_LAUNCH_H_

#include <lonexgrad/lonex_exception.h>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#include <string>

namespace lonexgrad::cuda {

// Launch a kernel
template <typename Kernel, typename... Args>
void launch(int device_id, const Kernel &kernel, const dim3 &grid_dim, const dim3 &block_dim, Args &&...args) {
    const auto device_status = cudaSetDevice(device_id);
    if (device_status != cudaSuccess) {
        LONEXGRAD_EXCEPTION("cudaMalloc error: " + std::string(cudaGetErrorString(device_status)));
    }
    kernel<<<grid_dim, block_dim>>>(std::forward<Args>(args)...);
    const auto status = cudaGetLastError();
    if (status != cudaSuccess) {
        LONEXGRAD_ERROR("Kernel error: " + std::string(cudaGetErrorString(status)));
    }
}

}    // namespace lonexgrad::cuda

#endif    // lonexgrad_STORAGE_CUDA_H_
