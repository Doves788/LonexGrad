


#ifndef lonexgrad_BACKEND_CPU_UNARY_H_
#define lonexgrad_BACKEND_CPU_UNARY_H_

#include <lonexgrad/lonex_tensor.h>

#include "tensor/backend/common/lonex_unary.h"

namespace lonexgrad::cpu {

template <common::unary::UnaryOpT Op, typename... Params>
auto unary_runner(const Tensor &tensor, Params... params) -> Tensor;

template <common::unary::UnaryOpT Op, typename... Params>
void unary_runner_inplace(Tensor &tensor, Params... params);

}    

#endif    
