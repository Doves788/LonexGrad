// distribution.cuh
// Element-wise distribution kernel

#ifndef lonexgrad_BACKEND_CUDA_KERNEL_DISTRIBUTION_H_
#define lonexgrad_BACKEND_CUDA_KERNEL_DISTRIBUTION_H_

#include <lonexgrad/lonex_concepts.h>
#include <lonexgrad/lonex_random.h>
#include <lonexgrad/lonex_scalar.h>

#include "tensor/backend/common/lonex_util.h"
#include "tensor/backend/cuda/lonex_config.cuh"
#include "tensor/backend/cuda/lonex_data_types.cuh"

#include <cstdint>
#include <nvfunctional>

namespace lonexgrad::cuda::kernel::distribution {

template <typename T, typename OP, typename... TS>
__global__ void
    variadic_param_kernel(const DeviceSpan<const uint64_t> gen_states, DataInfo<T> res, OP op, int N, TS... params) {
    const auto i = GLOBAL_FLAT_THREAD_IDX;
    if (i < N) {
        Generator gen = Generator::from_state(gen_states[i]);
        const auto idx_res = to_flat_index(i, res.shape, res.stride, res.offset);
        res.data[idx_res] = op()(params[i]..., gen);
    }
}

}    // namespace lonexgrad::cuda::kernel::distribution

#endif    // lonexgrad_BACKEND_CUDA_KERNEL_DISTRIBUTION_H_
