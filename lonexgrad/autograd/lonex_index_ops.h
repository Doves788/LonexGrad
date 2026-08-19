


#ifndef lonexgrad_AUTOGRAD_INDEX_OPS_H_
#define lonexgrad_AUTOGRAD_INDEX_OPS_H_

#include <lonexgrad/lonex_autograd.h>
#include <lonexgrad/lonex_index.h>
#include <lonexgrad/lonex_scalar.h>
#include <lonexgrad/lonex_shape.h>
#include <lonexgrad/lonex_tensor.h>

#include <string>
#include <vector>

namespace lonexgrad::autograd {

struct TensorIndex : public TensorFunction<TensorIndex> {
    static constexpr std::string name = "Index";
    static auto forward(
        AutogradStorage &storage,
        bool is_grad_required,
        const Tensor &tensor,
        const indexing::IndexList &indices
    ) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorIndexMask : public TensorFunction<TensorIndexMask> {
    static constexpr std::string name = "IndexMask";
    static auto forward(
        AutogradStorage &storage,
        bool is_grad_required,
        const Tensor &tensor,
        const Tensor &mask,
        int mask_count
    ) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorIndexIndices : public TensorFunction<TensorIndexIndices> {
    static constexpr std::string name = "IndexIndices";
    static auto forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor, const Tensor &indices)
        -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorIndexPutMask : public TensorFunction<TensorIndexPutMask> {
    static constexpr std::string name = "IndexPutMask";
    static auto forward(
        AutogradStorage &storage,
        bool is_grad_required,
        const Tensor &tensor,
        const Tensor &mask,
        const Tensor &values
    ) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorIndexPutIndices : public TensorFunction<TensorIndexPutIndices> {
    static constexpr std::string name = "IndexPutIndices";
    static auto forward(
        AutogradStorage &storage,
        bool is_grad_required,
        const Tensor &tensor,
        const Tensor &indices,
        const Tensor &values
    ) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorIndexSelect : public TensorFunction<TensorIndexSelect> {
    static constexpr std::string name = "IndexSelect";
    static auto forward(
        AutogradStorage &storage,
        bool is_grad_required,
        const Tensor &tensor,
        const std::vector<int> &indices,
        int dim
    ) -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorRepeat : public TensorFunction<TensorRepeat> {
    static constexpr std::string name = "Repeat";
    static auto
        forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor, const std::vector<int> &repeats)
            -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

struct TensorGather : public TensorFunction<TensorGather> {
    static constexpr std::string name = "Gather";
    static auto
        forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor, const Tensor &indices, int dim)
            -> Tensor;
    static auto backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList;
};

}    

#endif    
