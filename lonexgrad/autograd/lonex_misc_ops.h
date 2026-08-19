


#ifndef lonexgrad_AUTOGRAD_MISC_OPS_H_
#define lonexgrad_AUTOGRAD_MISC_OPS_H_

#include <lonexgrad/lonex_autograd.h>
#include <lonexgrad/lonex_scalar.h>
#include <lonexgrad/lonex_shape.h>
#include <lonexgrad/lonex_tensor.h>

#include <string>

namespace lonexgrad::autograd {

struct TensorWhere : public TensorFunction<TensorWhere> {
    static constexpr std::string name = "Where";
    static auto forward(
        AutogradStorage &storage,
        bool is_grad_required,
        const Tensor &cond,
        const Tensor &lhs,
        const Tensor &rhs
    ) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

}    

#endif    
