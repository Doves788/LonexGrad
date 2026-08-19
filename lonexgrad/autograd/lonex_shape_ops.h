


#ifndef lonexgrad_AUTOGRAD_SHAPE_OPS_H_
#define lonexgrad_AUTOGRAD_SHAPE_OPS_H_

#include <lonexgrad/lonex_autograd.h>
#include <lonexgrad/lonex_scalar.h>
#include <lonexgrad/lonex_shape.h>
#include <lonexgrad/lonex_tensor.h>

#include <string>
#include <vector>

namespace lonexgrad::autograd {

struct TensorBroadcast : public TensorFunction<TensorBroadcast> {
    static constexpr std::string name = "Reshape";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor, const Shape &shape)
        -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorReshape : public TensorFunction<TensorReshape> {
    static constexpr std::string name = "Reshape";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor, const Shape &shape)
        -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorPermute : public TensorFunction<TensorPermute> {
    static constexpr std::string name = "Permute";
    static auto
        forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor, const std::vector<int> &dims)
            -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

}    

#endif    
