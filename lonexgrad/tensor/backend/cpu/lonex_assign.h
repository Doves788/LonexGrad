


#ifndef lonexgrad_BACKEND_CPU_ASSIGN_H_
#define lonexgrad_BACKEND_CPU_ASSIGN_H_

#include <lonexgrad/lonex_tensor.h>

namespace lonexgrad::cpu {

void assign_runner(Tensor &lhs, const Tensor &rhs);

}    

#endif    
