


#ifndef lonexgrad_AUTOGRAD_TYPES_H_
#define lonexgrad_AUTOGRAD_TYPES_H_

#include <lonexgrad/lonex_device.h>
#include <lonexgrad/export.h>
#include <lonexgrad/lonex_grad_mode.h>
#include <lonexgrad/lonex_index.h>
#include <lonexgrad/lonex_scalar.h>
#include <lonexgrad/lonex_shape.h>
#include <lonexgrad/lonex_tensor.h>

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace lonexgrad::autograd {

namespace detail {


template <typename T>
struct LONEXGRAD_EXPORT IterateApply {
    
    template <typename... Args>
    T &apply() {
        return self();
    }

    
    template <typename Arg, typename... Args>
    T &apply(Arg &&arg, Args &&...args) {
        self()(std::forward<Arg>(arg));
        return apply(std::forward<Args>(args)...);
    }

private:
    
    inline T &self() {
        return *static_cast<T *>(this);
    }
};


class LONEXGRAD_EXPORT ExtractTensors : public IterateApply<ExtractTensors> {
public:
    ExtractTensors(TensorList &tensors)
        : tensors_(tensors) {}
    ExtractTensors(std::vector<Tensor> &&) = delete;    

    
    void operator()(const Tensor &tensor) {
        tensors_.push_back(tensor);
    }

    
    void operator()(const TensorList &tensors) {
        for (const auto &tensor : tensors) {
            tensors_.push_back(tensor);
        }
    }

    
    void operator()(const std::optional<Tensor> &tensor) {
        if (tensor) {
            tensors_.push_back(*tensor);
        }
    }

    
    template <typename T>
    void operator()(const T &x) {
        (void)x;
    }

private:
    
    TensorList &tensors_;    // NOLINT(*avoid-const-or-ref-data-members)
};
}    


using VersionedTensor = std::tuple<Tensor, int>;
using VersionedTensorList = std::vector<std::tuple<Tensor, int>>;
using StorageItem = std::variant<
    bool,
    int,
    float,
    double,
    std::vector<int>,
    ScalarType,
    Device,
    Shape,
    Tensor,
    TensorList,
    indexing::IndexList,
    VersionedTensor,
    VersionedTensorList>;
LONEXGRAD_EXPORT inline auto make_versioned_tensor(const Tensor &tensor) -> VersionedTensor {
    return {tensor, tensor.version_count()};
}
LONEXGRAD_EXPORT inline auto make_versioned_tensor_list(const TensorList &tensors) -> VersionedTensorList {
    VersionedTensorList list;
    for (const auto &tensor : tensors) {
        list.emplace_back(tensor, tensor.version_count());
    }
    return list;
}


using AutogradStorage = std::unordered_map<std::string, StorageItem>;



using GradList = CheckedVec<std::optional<Tensor>>;
using GradFunc = std::function<GradList(AutogradStorage &storage, const Tensor &grad_output)>;


struct LONEXGRAD_EXPORT SharedGrad {
    AutogradStorage storage;
    TensorList parents;
    std::optional<Tensor> grad = std::nullopt;
    GradFunc grad_func;
    GradHook grad_hook;
    std::string grad_func_name;
    bool requires_grad = false;
};






template <typename T>
struct LONEXGRAD_EXPORT TensorFunction {
    
    static auto get_shape(Tensor &tensor) -> Shape & {
        return tensor.shape_;
    }
    static auto get_stride(Tensor &tensor) -> Shape & {
        return tensor.stride_;
    }
    static auto get_offset(Tensor &tensor) -> int & {
        return tensor.offset_;
    }
    static auto get_ctx(Tensor &tensor) -> std::shared_ptr<SharedGrad> & {
        return tensor.ctx_;
    }
    static void bump_version(const Tensor &tensor) {
        ++tensor.version_count_;
    }

    
    template <typename... Args>
    static auto apply(Args &&...args) -> Tensor {
        
        TensorList input_tensors;
        detail::ExtractTensors(input_tensors).apply(std::forward<Args>(args)...);
        bool is_grad_required =
            GradMode::is_enabled() && std::any_of(input_tensors.begin(), input_tensors.end(), [](const Tensor &t) {
                return t.requires_grad();
            });

        AutogradStorage storage;
        std::optional<Tensor> output;
        {
            
            const NoGradGuard guard;
            output = T::forward(storage, is_grad_required, std::forward<Args>(args)...);
        }

        
        
        if (is_grad_required) {
            output->ctx_->parents = std::move(input_tensors);
            output->ctx_->storage = std::move(storage);
            output->ctx_->grad_func = T::backward;
            output->ctx_->requires_grad = true;
            output->ctx_->grad_func_name = T::name;
        }
        return *output;
    }
};

}    

#endif    
