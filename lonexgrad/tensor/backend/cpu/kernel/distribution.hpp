


#ifndef lonexgrad_BACKEND_CPU_KERNEL_DISTRIBUTION_H_
#define lonexgrad_BACKEND_CPU_KERNEL_DISTRIBUTION_H_

#include <lonexgrad/lonex_concepts.h>
#include <lonexgrad/lonex_random.h>
#include <lonexgrad/lonex_scalar.h>

#include "tensor/backend/common/lonex_span.h"
#include "tensor/backend/common/lonex_util.h"
#include "tensor/backend/cpu/lonex_data_types.h"

#include <cstdint>

namespace lonexgrad::cpu::kernel::distribution {

template <typename T, typename OP, typename... TS>
    requires IsAllOf<T, TS...>
void variadic_param_kernel(
    const HostSpan<const uint64_t> gen_states,
    DataInfo<T> res,
    OP op,
    int N,
    const HostSpan<const TS>... params
) {
    for (int i = 0; i < N; ++i) {
        Generator gen = Generator::from_state(gen_states[i]);
        const auto idx_res = to_flat_index(i, res.shape, res.stride, res.offset);
        res.data[idx_res] = op()(params[i]..., gen);
    }
}

}    

#endif    
