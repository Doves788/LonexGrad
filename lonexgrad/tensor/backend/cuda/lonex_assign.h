


#ifndef lonexgrad_BACKEND_CUDA_ASSIGN_H_
#define lonexgrad_BACKEND_CUDA_ASSIGN_H_

#include <lonexgrad/lonex_scalar.h>
#include <lonexgrad/lonex_tensor.h>

namespace lonexgrad::cuda {

void assign_runner(Tensor &lhs, const Tensor &rhs);

}    

#endif    
