


#include "tensor/backend/cpu/lonex_index.h"

#include <lonexgrad/lonex_concepts.h>
#include <lonexgrad/lonex_scalar.h>
#include <lonexgrad/lonex_tensor.h>

#include "tensor/backend/common/lonex_span.h"
#include "tensor/backend/cpu/lonex_data_types.h"
#include "tensor/backend/cpu/kernel/index.hpp"
#include "tensor/backend/cpu/lonex_storage_cpu.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <variant>
#include <vector>

namespace lonexgrad::cpu {

using namespace kernel::index;


auto index_mask_runner(const Tensor &input, const Tensor &mask, int N) -> Tensor {
    const auto &mask_storage = std::get<std::vector<uint8_t>>(mask.get_storage<StorageCPU>().storage);

    return std::visit(
        [&](auto &&input_storage) -> Tensor {
            using DT = std::remove_cvref_t<decltype(input_storage)>;    
            using T = template_parameter_t<std::remove_cvref_t<DT>>;    

            const HostSpan<const T> input_span{input_storage};
            const DataInfo<const T> input_info{input_span, input.shape(), input.stride(), input.offset()};

            const HostSpan<const uint8_t> mask_span{mask_storage};
            const DataInfo<const uint8_t> mask_info{mask_span, mask.shape(), mask.stride(), mask.offset()};

            
            std::vector<T> result(static_cast<std::size_t>(N));
            HostSpan<T> res_span{result};

            index_mask_kernel(input_info, mask_info, DataInfo<T>{res_span}, input.numel());
            return {std::make_unique<StorageCPU>(std::move(result)), input.dtype(), {N}, input.device()};
        },
        input.get_storage<StorageCPU>().storage
    );
}


auto index_indices_runner(const Tensor &input, const Tensor &indices) -> Tensor {
    using U = to_ctype_t<kDefaultInt>;
    const auto &indices_storage = std::get<std::vector<U>>(indices.get_storage<StorageCPU>().storage);

    int N = indices.numel();
    return std::visit(
        [&](auto &&input_storage) -> Tensor {
            using DT = std::remove_cvref_t<decltype(input_storage)>;    
            using T = template_parameter_t<std::remove_cvref_t<DT>>;    

            const HostSpan<const T> input_span{input_storage};
            const DataInfo<const T> input_info{input_span, input.shape(), input.stride(), input.offset()};

            const HostSpan<const U> indices_span{indices_storage};
            const DataInfo<const U> indices_info{indices_span, indices.shape(), indices.stride(), indices.offset()};

            
            std::vector<T> result(static_cast<std::size_t>(N));
            HostSpan<T> res_span{result};

            index_indices_kernel(input_info, indices_info, DataInfo<T>{res_span}, N);
            return {std::make_unique<StorageCPU>(std::move(result)), input.dtype(), {N}, input.device()};
        },
        input.get_storage<StorageCPU>().storage
    );
}


void index_put_mask_runner(Tensor &input, const Tensor &values, const Tensor &mask) {
    const auto &mask_storage = std::get<std::vector<uint8_t>>(mask.get_storage<StorageCPU>().storage);

    std::visit(
        [&](auto &&input_storage) {
            using DT = std::remove_cvref_t<decltype(input_storage)>;    
            using T = template_parameter_t<std::remove_cvref_t<DT>>;    

            HostSpan<T> input_span{input_storage};
            DataInfo<T> input_info{input_span, input.shape(), input.stride(), input.offset()};
            const HostSpan<const T> values_span{std::get<DT>(values.get_storage<StorageCPU>().storage)};
            const DataInfo<const T> values_info{values_span, values.shape(), values.stride(), values.offset()};

            const HostSpan<const uint8_t> mask_span{mask_storage};
            const DataInfo<const uint8_t> mask_info{mask_span, mask.shape(), mask.stride(), mask.offset()};

            index_put_mask_kernel(input_info, values_info, mask_info, input.numel());
        },
        input.get_storage<StorageCPU>().storage
    );
}


void index_put_indices_runner(Tensor &input, const Tensor &values, const Tensor &indices) {
    using U = to_ctype_t<kDefaultInt>;
    const auto &indices_storage = std::get<std::vector<U>>(indices.get_storage<StorageCPU>().storage);

    std::visit(
        [&](auto &&input_storage) {
            using DT = std::remove_cvref_t<decltype(input_storage)>;    
            using T = template_parameter_t<std::remove_cvref_t<DT>>;    

            HostSpan<T> input_span{input_storage};
            DataInfo<T> input_info{input_span, input.shape(), input.stride(), input.offset()};
            const HostSpan<const T> values_span{std::get<DT>(values.get_storage<StorageCPU>().storage)};
            const DataInfo<const T> values_info{values_span, values.shape(), values.stride(), values.offset()};

            const HostSpan<const U> indices_span{indices_storage};
            const DataInfo<const U> indices_info{indices_span, indices.shape(), indices.stride(), indices.offset()};

            index_put_indices_kernel(input_info, values_info, indices_info, input.numel());
        },
        input.get_storage<StorageCPU>().storage
    );
}

}    
