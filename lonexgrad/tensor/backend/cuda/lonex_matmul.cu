// matmul.cu
// Matmul runner

#include <lonexgrad/lonex_concepts.h>
#include <lonexgrad/lonex_scalar.h>
#include <lonexgrad/lonex_shape.h>
#include <lonexgrad/lonex_tensor.h>
#include <lonexgrad/lonex_util.h>

#include "tensor/backend/cuda/lonex_data_types.cuh"
#include "tensor/backend/cuda/kernel/lonex_matmul.cuh"
#include "tensor/backend/cuda/lonex_kernel_launch.cuh"
#include "tensor/backend/cuda/lonex_matmul.h"
#include "tensor/backend/cuda/lonex_storage_cuda.h"

#include <cassert>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <variant>

namespace lonexgrad::cuda {

using namespace kernel::matmul;

auto batched_matmul_runner(const Tensor &lhs, const Tensor &rhs) -> Tensor {
    assert(lhs.dim() == 3 && rhs.dim() == 3);
    assert(lhs.size(2) == rhs.size(1));    // Inner dims match
    assert(lhs.dtype() == rhs.dtype() && lhs.device() == rhs.device());
    const int B = lhs.size(0);
    const int N = lhs.size(1);
    const int K = lhs.size(2);
    const int M = rhs.size(2);
    const int device_id = lhs.device().id;

    const auto res_shape = Shape({B, N, M});
    const auto res_device = lhs.device();

    // matmul requires contiguous data
    const auto lhs_cont = lhs.contiguous();
    const auto rhs_cont = rhs.contiguous();

    // lhs and rhs need to be same type, so visit on one to reduce codegen
    return std::visit(
        [&](auto &&tensor_dev_memory) -> Tensor {
            using DT = std::remove_cvref_t<decltype(tensor_dev_memory)>;    // Storage type
            using T = template_parameter_t<DT>;                             // Underlying type

            // Allocate for result
            auto res_dev_memory = DeviceMemory<T>::AllocateElements(device_id, static_cast<std::size_t>(B * N * M));

            // Get operand data spans
            // Data can have an offset whilst still being contiguous
            const T *lhs_data_ptr = tensor_dev_memory.data_ptr();
            const T *rhs_data_ptr = std::get<DT>(rhs_cont.template get_storage<StorageCUDA>().dev_memory).data_ptr();
            lhs_data_ptr += lhs_cont.offset();    // NOLINT(*-pointer-arithmetic)
            rhs_data_ptr += rhs_cont.offset();    // NOLINT(*-pointer-arithmetic)
            const DeviceSpan<const T> lhs_span{lhs_data_ptr, static_cast<std::size_t>(lhs_cont.numel())};
            const DeviceSpan<const T> rhs_span{rhs_data_ptr, static_cast<std::size_t>(rhs_cont.numel())};
            DeviceSpan<T> res_span{res_dev_memory};

            const dim3 block_dim{
                (properties::TILE_WIDTH * properties::TILE_HEIGHT) / (properties::TN * properties::TM)
            };
            const dim3 grid_dim{
                static_cast<unsigned int>(ceil_div(M, properties::TILE_WIDTH)),
                static_cast<unsigned int>(ceil_div(N, properties::TILE_HEIGHT)),
                static_cast<unsigned int>(B)
            };

            // Call kernel
            launch(device_id, matmul_kernel<T>, grid_dim, block_dim, lhs_span, rhs_span, res_span, N, K, M);

            return {std::make_unique<StorageCUDA>(std::move(res_dev_memory)), lhs.dtype(), res_shape, res_device};
        },
        lhs_cont.template get_storage<StorageCUDA>().dev_memory
    );
}

}    // namespace lonexgrad::cuda
