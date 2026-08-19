


#ifndef lonexgrad_BACKEND_CUDA_REDUCE_H_
#define lonexgrad_BACKEND_CUDA_REDUCE_H_

#include <lonexgrad/lonex_tensor.h>

#include "tensor/backend/common/lonex_reduce.h"

namespace lonexgrad::cuda {

template <common::reduce::ReduceOpT Op>
auto reduce_all_runner(const Tensor &tensor) -> Tensor;

template <common::reduce::ReduceOpT Op>
auto reduce_dim_runner(const Tensor &tensor, int dim) -> Tensor;

}    

#endif    
