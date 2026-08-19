


#ifndef lonexgrad_AUTOGRAD_BACKWARD_H_
#define lonexgrad_AUTOGRAD_BACKWARD_H_

#include <lonexgrad/lonex_tensor.h>

namespace lonexgrad::autograd {

void backward(Tensor &tensor, const Tensor &grad, bool retain_graph);

}    

#endif    
