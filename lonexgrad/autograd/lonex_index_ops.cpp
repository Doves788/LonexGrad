


#include "autograd/lonex_index_ops.h"

#include <lonexgrad/lonex_autograd.h>
#include <lonexgrad/lonex_concepts.h>
#include <lonexgrad/lonex_device.h>
#include <lonexgrad/lonex_exception.h>
#include <lonexgrad/lonex_grad_mode.h>
#include <lonexgrad/lonex_index.h>
#include <lonexgrad/lonex_scalar.h>
#include <lonexgrad/lonex_shape.h>
#include <lonexgrad/lonex_tensor.h>
#include <lonexgrad/lonex_util.h>

#include "tensor/lonex_backend_register.h"

#include <cassert>
#include <cstddef>
#include <format>
#include <memory>
#include <ranges>
#include <utility>
#include <variant>
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



auto TensorIndex::forward(
    AutogradStorage &storage,
    bool is_grad_required,
    const Tensor &tensor,
    const indexing::IndexList &indices
) -> Tensor {
    if (is_grad_required) {
        storage["input"] = tensor;
        storage["indices"] = indices;
    }
    Tensor result = tensor;
    get_ctx(result) = std::make_shared<SharedGrad>();
    const auto self_shape = tensor.shape();
    const auto self_stride = tensor.stride();
    Shape &result_shape = get_shape(result);
    Shape &result_stride = get_stride(result);
    int &result_offset = get_offset(result);

    for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
        
        assert(result_shape.size() == result_stride.size());
        int res_i = i - (self_shape.size() - result_shape.size());
        std::visit(
            overloaded{
                
                [&](int index) {
                    
                    if (i < self_shape.ndim() - 1 || (i == self_shape.ndim() - 1 && self_shape.ndim() > 1)) {
                        result_shape.pop(res_i);
                        const auto s = result_stride.pop(res_i);
                        result_offset += index * s;
                    } else {
                        result_shape[result_shape.size() - 1] = 1;
                        const auto s = result_stride[res_i];
                        result_offset += index * s;
                    }
                },
                [&](const indexing::Slice &slice) {
                    result_stride[res_i] *= slice.stride();
                    result_offset += slice.start() * self_stride[i];
                    result_shape[res_i] = slice.to_size(self_shape[i]);
                },
            },
            indices[static_cast<std::size_t>(i)].get_index()
        );
    }
    return result;
}
auto TensorIndex::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &input = std::get<Tensor>(storage.at("input"));
    const auto &indices = std::get<indexing::IndexList>(storage.at("indices"));
    Tensor grad_input = zeros_like(input);
    grad_input[indices] = grad_output;
    return {grad_input};
}




auto TensorIndexMask::forward(
    AutogradStorage &storage,
    bool is_grad_required,
    const Tensor &tensor,
    const Tensor &mask,
    int mask_count
) -> Tensor {
    if (is_grad_required) {
        storage["input"] = tensor;
        storage["mask"] = make_versioned_tensor(mask);
    }
    return get_backend(tensor.device())->index_mask(tensor, mask, mask_count);
}
auto TensorIndexMask::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &input = std::get<Tensor>(storage.at("input"));
    const auto &[mask, mask_version] = std::get<VersionedTensor>(storage.at("mask"));
    CHECK_VERSION(mask, mask_version);
    Tensor grad_input = zeros_like(input);
    return {index_put(grad_input, mask, grad_output)};
}



auto TensorIndexIndices::forward(
    AutogradStorage &storage,
    bool is_grad_required,
    const Tensor &tensor,
    const Tensor &indices
) -> Tensor {
    if (is_grad_required) {
        storage["input"] = tensor;
        storage["indices"] = make_versioned_tensor(indices);
    }
    return get_backend(tensor.device())->index_indices(tensor, indices);
}
auto TensorIndexIndices::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &input = std::get<Tensor>(storage.at("input"));
    const auto &[indices, indices_version] = std::get<VersionedTensor>(storage.at("indices"));
    CHECK_VERSION(indices, indices_version);
    Tensor grad_input = zeros_like(input);
    return {index_put(grad_input, indices, grad_output)};
}



auto TensorIndexPutMask::forward(
    AutogradStorage &storage,
    bool is_grad_required,
    const Tensor &tensor,
    const Tensor &mask,
    const Tensor &values
) -> Tensor {
    if (is_grad_required) {
        storage["mask"] = make_versioned_tensor(mask);
    }
    bump_version(tensor);
    auto result = tensor;
    get_ctx(result) = std::make_shared<SharedGrad>();
    get_backend(tensor.device())->index_put_mask(result, values, mask);
    return result;
}
auto TensorIndexPutMask::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[mask, mask_version] = std::get<VersionedTensor>(storage.at("mask"));
    CHECK_VERSION(mask, mask_version);
    return {where(!mask, grad_output, 0)};
}



auto TensorIndexPutIndices::forward(
    AutogradStorage &storage,
    bool is_grad_required,
    const Tensor &tensor,
    const Tensor &indices,
    const Tensor &values
) -> Tensor {
    if (is_grad_required) {
        storage["indices"] = make_versioned_tensor(indices);
    }
    bump_version(tensor);
    auto result = tensor;
    get_ctx(result) = std::make_shared<SharedGrad>();
    get_backend(tensor.device())->index_put_indices(result, values, indices);
    return result;
}
auto TensorIndexPutIndices::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[indices, indices_version] = std::get<VersionedTensor>(storage.at("indices"));
    CHECK_VERSION(indices, indices_version);
    Tensor mask = zeros_like(grad_output);
    mask = index_put(mask, indices, 1).to(kBool);
    return {where(!mask, grad_output, 0)};
}








auto TensorIndexSelect::forward(
    AutogradStorage &storage,
    bool is_grad_required,
    const Tensor &tensor,
    const std::vector<int> &indices,
    int dim
) -> Tensor {
    if (is_grad_required || true) {
        storage["input"] = tensor;
        storage["indices"] = indices;
        storage["dim"] = dim;
    }
    int N = tensor.dim();
    std::vector<to_ctype_t<kDefaultInt>> selected_indices;
    Tensor all_indices = arange(tensor.shape(), kDefaultInt, kCPU);
    std::vector<indexing::Index> slices(static_cast<std::size_t>(N), indexing::Slice());
    
    for (const auto &idx : indices) {
        slices[static_cast<std::size_t>(dim)] = idx;
        for (const auto &i : all_indices[slices].to_vec<to_ctype_t<kDefaultInt>>()) {
            selected_indices.push_back(i);
        }
        slices[static_cast<std::size_t>(dim)] = indexing::Slice();
    }

    
    Shape result_shape = tensor.shape();
    for (int i = dim; i > 0; --i) {
        result_shape[i] = result_shape[i - 1];
    }
    result_shape[0] = static_cast<int>(indices.size());

    
    Tensor result = get_backend(tensor.device())
                        ->index_indices(tensor, Tensor(selected_indices, tensor.device()))
                        .reshape(result_shape);
    
    std::vector<int> permute_dims;
    for (int i = 0; i < N; ++i) {
        permute_dims.push_back(i + 1);
    }
    permute_dims[static_cast<std::size_t>(dim)] = 0;
    for (int i = dim + 1; i < N; ++i) {
        permute_dims[static_cast<std::size_t>(i)] = i;
    }

    
    result = result.permute(permute_dims).contiguous();

    
    auto inverse_dims = permute_dims;
    for (int i = 0; i < static_cast<int>(permute_dims.size()); ++i) {
        inverse_dims[static_cast<std::size_t>(permute_dims[static_cast<std::size_t>(i)])] = i;
    }
    return result;
}
auto TensorIndexSelect::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &input = std::get<Tensor>(storage.at("input"));
    const auto &indices = std::get<std::vector<int>>(storage.at("indices"));
    const auto &dim = std::get<int>(storage.at("dim"));

    Tensor grad_input = zeros_like(input);
    std::vector<indexing::Index> slices_grad_in(static_cast<std::size_t>(input.dim()), indexing::Slice());
    std::vector<indexing::Index> slices_grad_out(static_cast<std::size_t>(input.dim()), indexing::Slice());
    int i = -1;
    for (const auto &idx : indices) {
        slices_grad_in[static_cast<std::size_t>(dim)] = idx;
        slices_grad_out[static_cast<std::size_t>(dim)] = ++i;
        grad_input[slices_grad_in] += grad_output[slices_grad_out];
        slices_grad_in[static_cast<std::size_t>(dim)] = indexing::Slice();
        slices_grad_out[static_cast<std::size_t>(dim)] = indexing::Slice();
    }
    return {grad_input};
}





auto TensorRepeat::forward(
    AutogradStorage &storage,
    bool is_grad_required,
    const Tensor &tensor,
    const std::vector<int> &repeats
) -> Tensor {
    if (is_grad_required) {
        storage["input_shape"] = tensor.shape();
        storage["repeats"] = Shape(repeats);
    }
    
    Shape result_shape = tensor.shape();
    for (int i = 0; i < tensor.dim(); ++i) {
        result_shape[i] *= repeats[static_cast<std::size_t>(i)];
    }

    
    int n = tensor.numel();
    Tensor result = tensor;
    get_ctx(result) = std::make_shared<SharedGrad>();
    for (int i = 0; i < tensor.dim(); ++i) {
        int num_repeats = repeats[static_cast<std::size_t>(i)];
        if (num_repeats < 0) {
            LONEXGRAD_EXCEPTION(
                std::format("Given a non-positive number of repeats ({:d}) at dimension ({:d})", num_repeats, i)
            );
        }
        if (num_repeats > 1) {
            result = result.reshape({result.numel() / n, n});
            result = repeat_interleave(result, num_repeats, 0);
        }
        n /= tensor.size(i);
    }
    return result.reshape(result_shape);
}
auto TensorRepeat::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &shape_input = std::get<Shape>(storage.at("input_shape"));
    const auto &repeats = std::get<Shape>(storage.at("repeats"));
    assert(shape_input.size() == repeats.size());

    std::vector<int> reshaped_dims(static_cast<std::size_t>(shape_input.size() + repeats.size()), 0);
    for (int i = 0; i < shape_input.size(); ++i) {
        reshaped_dims[static_cast<std::size_t>(2 * i)] = repeats[i];
        reshaped_dims[static_cast<std::size_t>(2 * i + 1)] = shape_input[i];
    }
    
    auto grad_input = grad_output.reshape(Shape(std::move(reshaped_dims)));
    for (int i : std::ranges::iota_view{0, shape_input.size()} | std::views::reverse) {
        grad_input = grad_input.sum(2 * i);
    }
    return {grad_input};
}




auto TensorGather::forward(
    AutogradStorage &storage,
    bool is_grad_required,
    const Tensor &tensor,
    const Tensor &indices,
    int dim
) -> Tensor {
    if (is_grad_required) {
        storage["input"] = tensor;
        storage["indices"] = indices;
        storage["dim"] = dim;
    }
    return get_backend(tensor.device())->gather(tensor, indices, dim);
}
auto TensorGather::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &input = std::get<Tensor>(storage.at("input"));
    const auto &indices = std::get<Tensor>(storage.at("indices"));
    const auto &dim = std::get<int>(storage.at("dim"));

    auto global_indices = arange(input.shape(), TensorOptions().dtype(kDefaultInt).device(input.device()));
    global_indices = gather(global_indices, indices, dim).flatten();
    auto input_grad = zeros_like(input);
    input_grad = index_put(input_grad, global_indices, grad_output.flatten());
    return {input_grad};
}

}    
