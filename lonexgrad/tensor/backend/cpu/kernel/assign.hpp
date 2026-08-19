


#ifndef lonexgrad_BACKEND_CPU_KERNEL_ASSIGN_H_
#define lonexgrad_BACKEND_CPU_KERNEL_ASSIGN_H_

#include <lonexgrad/lonex_scalar.h>

#include "tensor/backend/common/lonex_util.h"
#include "tensor/backend/cpu/lonex_data_types.h"

namespace lonexgrad::cpu::kernel::assign {

template <typename T>
void assign_kernel(DataInfo<T> lhs, const DataInfo<const T> rhs, int N) {
    for (int i = 0; i < N; ++i) {
        const auto lhs_idx = to_flat_index(i, lhs.shape, lhs.stride, lhs.offset);
        const auto rhs_idx = to_flat_index(i, rhs.shape, rhs.stride, rhs.offset);
        lhs.data[lhs_idx] = rhs.data[rhs_idx];
    }
}

}    

#endif    
