


#ifndef lonexgrad_BACKEND_CUDA_CAST_H_
#define lonexgrad_BACKEND_CUDA_CAST_H_

#include <lonexgrad/lonex_scalar.h>
#include <lonexgrad/lonex_tensor.h>

namespace lonexgrad::cuda {

auto cast_runner(const Tensor &tensor, ScalarType dtype) -> Tensor;

}    

#endif    
