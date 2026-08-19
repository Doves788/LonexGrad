


#ifndef lonexgrad_BACKEND_CPU_DATATYPES_H_
#define lonexgrad_BACKEND_CPU_DATATYPES_H_

#include <lonexgrad/lonex_scalar.h>
#include <lonexgrad/lonex_shape.h>

#include "tensor/backend/common/lonex_span.h"

#include <type_traits>

namespace lonexgrad::cpu {


template <typename T>
    requires IsScalarType<std::remove_cvref_t<T>>
struct DataInfo {
    HostSpan<T> data;
    HostSpan<const int> shape{};
    HostSpan<const int> stride{};
    int offset = 0;
};

}    

#endif    
