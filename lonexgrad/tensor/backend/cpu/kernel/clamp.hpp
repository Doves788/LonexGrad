


#ifndef lonexgrad_BACKEND_CPU_KERNEL_CLAMP_H_
#define lonexgrad_BACKEND_CPU_KERNEL_CLAMP_H_

#include <lonexgrad/lonex_scalar.h>

#include "tensor/backend/common/lonex_util.h"
#include "tensor/backend/cpu/lonex_data_types.h"

#include <cmath>

namespace lonexgrad::cpu::kernel::clamp {

template <typename T>
void clamp_kernel(DataInfo<T> array, const DataInfo<const T> min, const DataInfo<const T> max, int N) {
    for (int i = 0; i < N; ++i) {
        const auto idx = to_flat_index(i, array.shape, array.stride, array.offset);
        const auto min_idx = to_flat_index(i, min.shape, min.stride, min.offset);
        const auto max_idx = to_flat_index(i, max.shape, max.stride, max.offset);
        array.data[idx] = std::min(std::max(array.data[idx], min.data[min_idx]), max.data[max_idx]);
    }
}

}    

#endif    
