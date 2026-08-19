


#ifndef lonexgrad_BACKEND_CPU_CLAMP_H_
#define lonexgrad_BACKEND_CPU_CLAMP_H_

#include <lonexgrad/lonex_tensor.h>

namespace lonexgrad::cpu {

void clamp_inplace_runner(Tensor &tensor, const Tensor &min, const Tensor &max);

}    

#endif    
