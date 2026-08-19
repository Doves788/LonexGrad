


#ifndef lonexgrad_BACKEND_CPU_MATMUL_H_
#define lonexgrad_BACKEND_CPU_MATMUL_H_

#include <lonexgrad/lonex_tensor.h>

namespace lonexgrad::cpu {

auto batched_matmul_runner(const Tensor &lhs, const Tensor &rhs) -> Tensor;

}    

#endif    
