


#ifndef lonexgrad_BACKEND_CUDA_CLAMP_H_
#define lonexgrad_BACKEND_CUDA_CLAMP_H_

#include <lonexgrad/lonex_tensor.h>

namespace lonexgrad::cuda {

void clamp_inplace_runner(Tensor &tensor, const Tensor &min, const Tensor &max);

}    

#endif    
