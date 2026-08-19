


#include "autograd/lonex_reduce_ops.h"

#include <lonexgrad/lonex_autograd.h>
#include <lonexgrad/lonex_exception.h>
#include <lonexgrad/lonex_grad_mode.h>
#include <lonexgrad/lonex_scalar.h>
#include <lonexgrad/lonex_shape.h>
#include <lonexgrad/lonex_tensor.h>

#include "tensor/lonex_backend_register.h"

#include <format>

namespace lonexgrad::autograd {

// NOLINTNEXTLINE(*-macro-usage)
#define CHECK_VERSION(tensor, version)                                                                               \
    if (tensor.version_count() != version) {                                                                         \
        LONEXGRAD_EXCEPTION(std::format(                                                                                    \
            "Inplace operation on tensor required for autograd detected. Tensor with version {:d} saved in forward " \
            "pass, but has version {:d} in backward pass",                                                           \
            version,                                                                                                 \
            tensor.version_count()                                                                                   \
        ));                                                                                                          \
    }



auto TensorMinAll::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input_shape"] = tensor.shape();
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->min(tensor);
}
auto TensorMinAll::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &shape_input = std::get<Shape>(storage.at("input_shape"));
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    const auto min_idx = input.argmin();
    auto mask = zeros_like(input);
    mask = index_put(mask, min_idx, 1);
    return {grad_output.expand(shape_input) * mask};
}





auto TensorMinDim::forward(
    AutogradStorage &storage,
    bool is_grad_required,
    const Tensor &tensor,
    int dim,
    bool keep_dim
) -> Tensor {
    auto result = get_backend(tensor.device())->min(tensor, dim);
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
        storage["result"] = make_versioned_tensor(result);
        storage["dim"] = dim;
        storage["keep_dim"] = keep_dim;
    }
    return keep_dim ? result : result.squeeze(dim);
}
auto TensorMinDim::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &dim = std::get<int>(storage.at("dim"));
    const auto &keep_dim = std::get<bool>(storage.at("keep_dim"));
    const auto &[input, version_input] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version_input);
    const auto &[result, version_result] = std::get<VersionedTensor>(storage.at("result"));
    CHECK_VERSION(result, version_result);

    const auto mask = (result.expand(input.shape()) == input);
    const auto _grad_output = keep_dim ? grad_output : grad_output.unsqueeze(dim);
    const auto num_min_match = mask.sum(dim, true).expand(input.shape());
    const auto masked_grad = _grad_output.expand(input.shape()) * mask;
    return {masked_grad / num_min_match};

    
    
    
    
    
    
    
    
}



auto TensorMaxAll::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input_shape"] = tensor.shape();
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->max(tensor);
}
auto TensorMaxAll::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &shape_input = std::get<Shape>(storage.at("input_shape"));
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    const auto max_idx = input.argmax();
    auto mask = zeros_like(input);
    mask = index_put(mask, max_idx, 1);
    return {grad_output.expand(shape_input) * mask};
}





auto TensorMaxDim::forward(
    AutogradStorage &storage,
    bool is_grad_required,
    const Tensor &tensor,
    int dim,
    bool keep_dim
) -> Tensor {
    auto result = get_backend(tensor.device())->max(tensor, dim);
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
        storage["result"] = make_versioned_tensor(result);
        storage["dim"] = dim;
        storage["keep_dim"] = keep_dim;
    }
    return keep_dim ? result : result.squeeze(dim);
}
auto TensorMaxDim::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &dim = std::get<int>(storage.at("dim"));
    const auto &keep_dim = std::get<bool>(storage.at("keep_dim"));
    const auto &[input, version_input] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version_input);
    const auto &[result, version_result] = std::get<VersionedTensor>(storage.at("result"));
    CHECK_VERSION(result, version_result);

    const auto mask = (result.expand(input.shape()) == input);
    const auto _grad_output = keep_dim ? grad_output : grad_output.unsqueeze(dim);
    const auto num_min_match = mask.sum(dim, true).expand(input.shape());
    const auto masked_grad = _grad_output.expand(input.shape()) * mask;
    return {masked_grad / num_min_match};

    
    
    
    
    
    
    
    
    
    
}



auto TensorSumAll::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input_shape"] = tensor.shape();
    }
    return get_backend(tensor.device())->sum(tensor);
}
auto TensorSumAll::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &shape_input = std::get<Shape>(storage.at("input_shape"));
    return {grad_output.expand(shape_input)};
}



auto TensorSumDim::forward(
    AutogradStorage &storage,
    bool is_grad_required,
    const Tensor &tensor,
    int dim,
    bool keep_dim
) -> Tensor {
    auto result = get_backend(tensor.device())->sum(tensor, dim);
    if (is_grad_required) {
        storage["input_shape"] = tensor.shape();
        storage["keep_dim_shape"] = result.shape();
    }
    return keep_dim ? result : result.squeeze(dim);
}
auto TensorSumDim::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &shape_input = std::get<Shape>(storage.at("input_shape"));
    const auto &shape_keep_dim = std::get<Shape>(storage.at("keep_dim_shape"));
    return {grad_output.reshape(shape_keep_dim).expand(shape_input)};
}



auto TensorMeanAll::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input_shape"] = tensor.shape();
    }
    return (tensor / tensor.numel()).sum();
}
auto TensorMeanAll::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &shape_input = std::get<Shape>(storage.at("input_shape"));
    int N = shape_input.numel();
    return {grad_output.expand(shape_input) / N};
}




auto TensorMeanDim::forward(
    AutogradStorage &storage,
    bool is_grad_required,
    const Tensor &tensor,
    int dim,
    bool keep_dim
) -> Tensor {
    auto result = (tensor / tensor.size(dim)).sum(dim, true);
    if (is_grad_required) {
        storage["input_shape"] = tensor.shape();
        storage["keep_dim_shape"] = result.shape();
        storage["dim"] = dim;
    }
    return keep_dim ? result : result.squeeze(dim);
}
auto TensorMeanDim::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &shape_input = std::get<Shape>(storage.at("input_shape"));
    const auto &shape_keep_dim = std::get<Shape>(storage.at("keep_dim_shape"));
    const auto &dim = std::get<int>(storage.at("dim"));
    int N = shape_input[dim];
    return {grad_output.reshape(shape_keep_dim).expand(shape_input) / N};
}

}    
