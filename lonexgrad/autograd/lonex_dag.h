


#ifndef lonexgrad_AUTOGRAD_DAG_H_
#define lonexgrad_AUTOGRAD_DAG_H_

#include <lonexgrad/lonex_tensor.h>

namespace lonexgrad::autograd {






auto build_dag(const Tensor &tensor) -> TensorList;

}    

#endif    
