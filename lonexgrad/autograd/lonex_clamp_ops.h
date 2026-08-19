


#ifndef lonexgrad_AUTOGRAD_CLAMP_OPS_H_
#define lonexgrad_AUTOGRAD_CLAMP_OPS_H_

#include <lonexgrad/lonex_autograd.h>
#include <lonexgrad/lonex_scalar.h>
#include <lonexgrad/lonex_shape.h>
#include <lonexgrad/lonex_tensor.h>

#include <string>

namespace lonexgrad::autograd {

struct TensorClamp : public TensorFunction<TensorClamp> {
    static constexpr std::string name = "Clamp";
    static auto forward(
        AutogradStorage &storage,
        bool is_grad_required,
        const Tensor &tensor,
        const Tensor &min,
        const Tensor &max
    ) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

}    

#endif    
