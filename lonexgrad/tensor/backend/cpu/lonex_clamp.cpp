


#include "tensor/backend/cpu/lonex_clamp.h"

#include <lonexgrad/lonex_concepts.h>
#include <lonexgrad/lonex_scalar.h>
#include <lonexgrad/lonex_tensor.h>

#include "tensor/backend/common/lonex_span.h"
#include "tensor/backend/cpu/lonex_data_types.h"
#include "tensor/backend/cpu/kernel/clamp.hpp"
#include "tensor/backend/cpu/lonex_storage_cpu.h"

#include <type_traits>
#include <variant>

namespace lonexgrad::cpu {

using namespace kernel::clamp;

void clamp_inplace_runner(Tensor &tensor, const Tensor &min, const Tensor &max) {
    const int N = tensor.numel();
    return std::visit(
        [&](auto &&tensor_storage) {
            using DT = std::remove_cvref_t<decltype(tensor_storage)>;    
            using T = template_parameter_t<DT>;                          

            
            HostSpan<T> tensor_span{tensor_storage};
            const HostSpan<const T> min_span{std::get<DT>(min.template get_storage<StorageCPU>().storage)};
            const HostSpan<const T> max_span{std::get<DT>(max.template get_storage<StorageCPU>().storage)};

            
            DataInfo<T> a{tensor_span, tensor.shape(), tensor.stride(), tensor.offset()};
            const DataInfo<const T> _min{min_span, min.shape(), min.stride(), min.offset()};
            const DataInfo<const T> _max{max_span, max.shape(), max.stride(), max.offset()};

            
            clamp_kernel(a, _min, _max, N);
        },
        tensor.template get_storage<StorageCPU>().storage
    );
}

}    
