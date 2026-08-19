


#ifndef lonexgrad_AUTOGRAD_REDUCE_OPS_H_
#define lonexgrad_AUTOGRAD_REDUCE_OPS_H_

#include <lonexgrad/lonex_autograd.h>
#include <lonexgrad/lonex_scalar.h>
#include <lonexgrad/lonex_shape.h>
#include <lonexgrad/lonex_tensor.h>

#include <string>

namespace lonexgrad::autograd {

struct TensorMinAll : public TensorFunction<TensorMinAll> {
    static constexpr std::string name = "MinAll";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorMinDim : public TensorFunction<TensorMinDim> {
    static constexpr std::string name = "MinDim";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor, int dim, bool keep_dim)
        -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorMaxAll : public TensorFunction<TensorMaxAll> {
    static constexpr std::string name = "MaxAll";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorMaxDim : public TensorFunction<TensorMaxDim> {
    static constexpr std::string name = "MaxDim";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor, int dim, bool keep_dim)
        -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorSumAll : public TensorFunction<TensorSumAll> {
    static constexpr std::string name = "SumAll";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorSumDim : public TensorFunction<TensorSumDim> {
    static constexpr std::string name = "SumDim";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor, int dim, bool keep_dim)
        -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorMeanAll : public TensorFunction<TensorMeanAll> {
    static constexpr std::string name = "MeanAll";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorMeanDim : public TensorFunction<TensorMeanDim> {
    static constexpr std::string name = "MeanAll";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor, int dim, bool keep_dim)
        -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

}    

#endif    
