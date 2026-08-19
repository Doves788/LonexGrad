


#ifndef lonexgrad_BACKEND_COMMON_UTIL_H_
#define lonexgrad_BACKEND_COMMON_UTIL_H_

#include <lonexgrad/lonex_concepts.h>
#include <lonexgrad/lonex_macros.h>
#include <lonexgrad/lonex_shape.h>

#include <cstddef>

namespace lonexgrad {



template <typename T>
constexpr LONEXGRAD_DEVICE auto to_flat_index(int rank, const T &shape, const T &stride, int offset, int dim = -1) -> int {
    
    if (shape.size() == 0 || stride.size() == 0) {
        return rank + offset;
    }
    
    int flat_index = offset;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        const auto idx = static_cast<std::size_t>(i);
        if (dim != -1 && i == dim) {
            continue;
        }
        flat_index += (rank % shape[idx]) * stride[idx];
        rank /= shape[idx];
    }
    return flat_index;
}

}    

#endif    
