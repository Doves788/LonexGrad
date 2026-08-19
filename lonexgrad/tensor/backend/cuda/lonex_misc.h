


#ifndef lonexgrad_BACKEND_CUDA_MISC_H_
#define lonexgrad_BACKEND_CUDA_MISC_H_

#include <lonexgrad/lonex_tensor.h>

namespace lonexgrad::cuda {

auto where_runner(const Tensor &cond, const Tensor &lhs, const Tensor &rhs) -> Tensor;
auto gather_runner(const Tensor &input, const Tensor &indices, int dim) -> Tensor;

}    

#endif    
