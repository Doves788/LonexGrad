


#ifndef lonexgrad_BACKEND_CPU_BINARY_H_
#define lonexgrad_BACKEND_CPU_BINARY_H_

#include <lonexgrad/lonex_tensor.h>

#include "tensor/backend/common/lonex_binary.h"

namespace lonexgrad::cpu {

template <common::binary::BinaryOpT Op>
auto binary_runner(const Tensor &lhs, const Tensor &rhs) -> Tensor;

template <common::binary::BinaryOpT Op>
void binary_inplace_runner(Tensor &lhs, const Tensor &rhs);

}    

#endif    
