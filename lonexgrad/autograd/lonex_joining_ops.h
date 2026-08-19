


#ifndef lonexgrad_AUTOGRAD_JOIN_OPS_H_
#define lonexgrad_AUTOGRAD_JOIN_OPS_H_

#include <lonexgrad/lonex_autograd.h>
#include <lonexgrad/lonex_scalar.h>
#include <lonexgrad/lonex_tensor.h>

#include <string>

namespace lonexgrad::autograd {

struct TensorCat : public TensorFunction<TensorCat> {
    static constexpr std::string name = "Cat";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const TensorList &tensors, int dim) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

}    

#endif    
