// misc.cuh
// Element-wise distribution kernel

#ifndef lonexgrad_BACKEND_CUDA_KERNEL_MISC_H_
#define lonexgrad_BACKEND_CUDA_KERNEL_MISC_H_

#include <lonexgrad/lonex_concepts.h>
#include <lonexgrad/lonex_scalar.h>

#include "tensor/backend/common/lonex_misc.h"
#include "tensor/backend/common/lonex_util.h"
#include "tensor/backend/cuda/lonex_config.cuh"
#include "tensor/backend/cuda/lonex_data_types.cuh"

#include <nvfunctional>

namespace lonexgrad::cuda::kernel::misc {

using namespace lonexgrad::common::misc;

template <typename T>
__global__ void where_kernel(
    DeviceSpan<T> res,
    int N,
    const DataInfo<const to_ctype_t<kBool>> cond,
    const DataInfo<const T> lhs,
    const DataInfo<const T> rhs
) {
    using KernelOp = typename OpFactory<T, MiscOpT::where>::KernelOp;
    KernelOp op{};
    const auto i = GLOBAL_FLAT_THREAD_IDX;
    if (i < N) {
        const auto lhs_idx = to_flat_index(i, lhs.shape, lhs.stride, lhs.offset);
        const auto rhs_idx = to_flat_index(i, rhs.shape, rhs.stride, rhs.offset);
        const auto cond_idx = to_flat_index(i, cond.shape, cond.stride, cond.offset);
        res[i] = op()(cond.data[cond_idx], lhs.data[lhs_idx], rhs.data[rhs_idx]);
    }
}

template <typename T>
__global__ void gather_kernel(
    DeviceSpan<T> res,
    int N,
    const DataInfo<const T> input,
    const DataInfo<const int> indices,
    const DeviceSpan<const int> res_shape,
    int dim
) {
    const auto i = GLOBAL_FLAT_THREAD_IDX;
    if (i < N) {
        const auto base_input_idx = to_flat_index(i, res_shape, input.stride, input.offset);
        const auto gather_idx = to_flat_index(i, indices.shape, indices.stride, indices.offset);
        const auto offset = input.stride[dim] * indices.data[gather_idx];
        res[i] = input.data[base_input_idx + offset];
    }
}

}    // namespace lonexgrad::cuda::kernel::misc

#endif    // lonexgrad_BACKEND_CUDA_KERNEL_MISC_H_
