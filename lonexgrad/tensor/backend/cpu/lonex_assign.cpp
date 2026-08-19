


#include "tensor/backend/cpu/lonex_assign.h"

#include <lonexgrad/lonex_concepts.h>
#include <lonexgrad/lonex_scalar.h>
#include <lonexgrad/lonex_tensor.h>

#include "tensor/backend/common/lonex_span.h"
#include "tensor/backend/cpu/lonex_data_types.h"
#include "tensor/backend/cpu/kernel/assign.hpp"
#include "tensor/backend/cpu/lonex_storage_cpu.h"

#include <type_traits>
#include <variant>

namespace lonexgrad::cpu {

using namespace kernel::assign;

void assign_runner(Tensor &lhs, const Tensor &rhs) {
    const int N = lhs.numel();
    return std::visit(
        [&](auto &&array_storage) {
            using DT = std::remove_cvref_t<decltype(array_storage)>;    
            using T = template_parameter_t<DT>;                         

            
            HostSpan<T> lhs_span{array_storage};
            const HostSpan<const T> rhs_span{std::get<DT>(rhs.template get_storage<StorageCPU>().storage)};

            
            DataInfo<T> l{lhs_span, lhs.shape(), lhs.stride(), lhs.offset()};
            const DataInfo<const T> r{rhs_span, rhs.shape(), rhs.stride(), rhs.offset()};

            
            assign_kernel(l, r, N);
        },
        lhs.template get_storage<StorageCPU>().storage
    );
}

}    
