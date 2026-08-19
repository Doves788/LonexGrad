


#include "autograd/lonex_joining_ops.h"

#include <lonexgrad/lonex_autograd.h>
#include <lonexgrad/lonex_exception.h>
#include <lonexgrad/lonex_grad_mode.h>
#include <lonexgrad/lonex_index.h>
#include <lonexgrad/lonex_scalar.h>
#include <lonexgrad/lonex_shape.h>
#include <lonexgrad/lonex_tensor.h>

#include <cstddef>
#include <format>
#include <ranges>
#include <vector>

namespace lonexgrad::autograd {

// NOLINTNEXTLINE(*-macro-usage)
#define CHECK_VERSION(tensor, version)                                                                           \
    if (tensor.version_count() != version) {                                                                     \
        LONEXGRAD_EXCEPTION(                                                                                            \
            std::format(                                                                                         \
                "Inplace operation on tensor required for autograd detected. Tensor with version {:d} saved in " \
                "forward "                                                                                       \
                "pass, but has version {:d} in backward pass",                                                   \
                version,                                                                                         \
                tensor.version_count()                                                                           \
            )                                                                                                    \
        );                                                                                                       \
    }



auto TensorCat::forward(
    [[maybe_unused]] AutogradStorage &storage,
    [[maybe_unused]] bool is_grad_required,
    const TensorList &tensors,
    int dim
) -> Tensor {
    if (is_grad_required) {
        storage["tensors"] = make_versioned_tensor_list(tensors);
        storage["dim"] = dim;
    }

    
    Shape ref_shape = tensors[0].shape();
    for (const auto &tensor : std::views::drop(tensors, 1)) {
        ref_shape[dim] += tensor.shape()[dim];
    }
    ScalarType res_type = tensors[0].dtype();

    
    int start_idx = 0;
    int end_idx = 0;
    std::vector<indexing::Index> slices(static_cast<std::size_t>(ref_shape.size()), indexing::Slice());
    Tensor result = zeros(ref_shape, res_type, tensors[0].device());
    for (const auto &tensor : tensors) {
        start_idx = end_idx;
        end_idx += tensor.size(dim);
        slices[static_cast<std::size_t>(dim)] = indexing::Slice(start_idx, end_idx);
        result[slices] = tensor;
    }
    return result;
}
auto TensorCat::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &tensors = std::get<VersionedTensorList>(storage.at("tensors"));
    const auto &dim = std::get<int>(storage.at("dim"));

    
    for (const auto &[tensor, version] : tensors) {
        CHECK_VERSION(tensor, version);
    }

    
    GradList grad_inputs;
    int start_idx = 0;
    int end_idx = 0;
    std::vector<indexing::Index> slices(static_cast<std::size_t>(grad_output.dim()), indexing::Slice());
    for (auto &[tensor, _] : tensors) {
        start_idx = end_idx;
        end_idx += tensor.size(dim);
        slices[static_cast<std::size_t>(dim)] = indexing::Slice(start_idx, end_idx);
        grad_inputs.emplace_back(grad_output[slices]);
    }
    return grad_inputs;
}

}    
