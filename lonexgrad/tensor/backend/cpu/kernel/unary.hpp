


#ifndef lonexgrad_BACKEND_CPU_KERNEL_UNARY_H_
#define lonexgrad_BACKEND_CPU_KERNEL_UNARY_H_

#include "tensor/backend/common/lonex_span.h"
#include "tensor/backend/common/lonex_util.h"
#include "tensor/backend/cpu/lonex_data_types.h"

namespace lonexgrad::cpu::kernel::unary {

template <bool CastBeforeOp, typename T, typename R, typename OP, typename... Args>
void unary_kernel(const DataInfo<const T> di, HostSpan<R> res, OP op, int N, Args... args) {
    for (int i = 0; i < N; ++i) {
        const auto idx = to_flat_index(i, di.shape, di.stride, di.offset);
        
        if constexpr (CastBeforeOp) {
            res[i] = op()(static_cast<R>(di.data[idx]), args...);
        } else {
            res[i] = static_cast<R>(op()(di.data[idx], args...));
        }
    }
}

template <typename T, typename OP, typename... Args>
void unary_kernel_inplace(DataInfo<T> di, OP op, int N, Args... args) {
    for (int i = 0; i < N; ++i) {
        const auto idx = to_flat_index(i, di.shape, di.stride, di.offset);
        di.data[idx] = op()(di.data[idx], args...);
    }
}

}    

#endif    
