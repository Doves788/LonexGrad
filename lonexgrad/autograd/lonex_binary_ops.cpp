


#include "autograd/lonex_binary_ops.h"

#include <lonexgrad/lonex_autograd.h>
#include <lonexgrad/lonex_exception.h>
#include <lonexgrad/lonex_grad_mode.h>
#include <lonexgrad/lonex_tensor.h>

#include "tensor/lonex_backend_register.h"

#include <format>

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




auto TensorAdd::forward(
    [[maybe_unused]] AutogradStorage &storage,
    [[maybe_unused]] bool is_grad_required,
    const Tensor &lhs,
    const Tensor &rhs
) -> Tensor {
    return get_backend(lhs.device())->add(lhs, rhs);
}
auto TensorAdd::backward([[maybe_unused]] const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    return {grad_output, grad_output};
}




auto TensorSub::forward(
    [[maybe_unused]] AutogradStorage &storage,
    [[maybe_unused]] bool is_grad_required,
    const Tensor &lhs,
    const Tensor &rhs
) -> Tensor {
    return get_backend(lhs.device())->sub(lhs, rhs);
}
auto TensorSub::backward([[maybe_unused]] const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    return {grad_output, -grad_output};
}




auto TensorMul::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &lhs, const Tensor &rhs)
    -> Tensor {
    if (is_grad_required) {
        storage["lhs"] = make_versioned_tensor(lhs);
        storage["rhs"] = make_versioned_tensor(rhs);
    }
    return get_backend(lhs.device())->mul(lhs, rhs);
}
auto TensorMul::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[lhs, version_lhs] = std::get<VersionedTensor>(storage.at("lhs"));
    CHECK_VERSION(lhs, version_lhs);
    const auto &[rhs, version_rhs] = std::get<VersionedTensor>(storage.at("rhs"));
    CHECK_VERSION(rhs, version_rhs);
    return {grad_output * rhs, grad_output * lhs};
}




auto TensorDiv::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &lhs, const Tensor &rhs)
    -> Tensor {
    if (is_grad_required) {
        storage["lhs"] = make_versioned_tensor(lhs);
        storage["rhs"] = make_versioned_tensor(rhs);
    }
    return get_backend(lhs.device())->div(lhs, rhs);
}
auto TensorDiv::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    
    const auto &[u, version_lhs] = std::get<VersionedTensor>(storage.at("lhs"));
    CHECK_VERSION(u, version_lhs);
    const auto &[v, version_rhs] = std::get<VersionedTensor>(storage.at("rhs"));
    CHECK_VERSION(v, version_rhs);
    const Tensor v_recip = 1.0 / v;
    const Tensor v_recip_grad = grad_output * v_recip;
    return {v_recip_grad, -v_recip_grad * v_recip * u};
}




auto TensorMaximum::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &lhs, const Tensor &rhs)
    -> Tensor {
    if (is_grad_required) {
        storage["lhs"] = make_versioned_tensor(lhs);
        storage["rhs"] = make_versioned_tensor(rhs);
    }
    return get_backend(lhs.device())->maximum(lhs, rhs);
}
auto TensorMaximum::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[lhs, version_lhs] = std::get<VersionedTensor>(storage.at("lhs"));
    CHECK_VERSION(lhs, version_lhs);
    const auto &[rhs, version_rhs] = std::get<VersionedTensor>(storage.at("rhs"));
    CHECK_VERSION(rhs, version_rhs);
    const Tensor mask = lhs > rhs;
    return {grad_output * mask, grad_output * !mask};
}




auto TensorMinimum::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &lhs, const Tensor &rhs)
    -> Tensor {
    if (is_grad_required) {
        storage["lhs"] = make_versioned_tensor(lhs);
        storage["rhs"] = make_versioned_tensor(rhs);
    }
    return get_backend(lhs.device())->minimum(lhs, rhs);
}
auto TensorMinimum::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[lhs, version_lhs] = std::get<VersionedTensor>(storage.at("lhs"));
    CHECK_VERSION(lhs, version_lhs);
    const auto &[rhs, version_rhs] = std::get<VersionedTensor>(storage.at("rhs"));
    CHECK_VERSION(rhs, version_rhs);
    const Tensor mask = lhs < rhs;
    return {grad_output * mask, grad_output * !mask};
}




auto TensorPow::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &lhs, const Tensor &rhs)
    -> Tensor {
    if (is_grad_required) {
        storage["lhs"] = make_versioned_tensor(lhs);
        storage["rhs"] = make_versioned_tensor(rhs);
    }
    return get_backend(lhs.device())->pow(lhs, rhs);
}
auto TensorPow::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[lhs, version_lhs] = std::get<VersionedTensor>(storage.at("lhs"));
    CHECK_VERSION(lhs, version_lhs);
    const auto &[rhs, version_rhs] = std::get<VersionedTensor>(storage.at("rhs"));
    CHECK_VERSION(rhs, version_rhs);
    return {grad_output * rhs * pow(lhs, rhs - 1), grad_output * pow(lhs, rhs) * log(lhs)};
}





auto TensorBatchedMatmul::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &lhs, const Tensor &rhs)
    -> Tensor {
    if (is_grad_required) {
        storage["lhs"] = make_versioned_tensor(lhs);
        storage["rhs"] = make_versioned_tensor(rhs);
    }
    return get_backend(lhs.device())->batched_matmul(lhs, rhs);
}
auto TensorBatchedMatmul::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[lhs, version_lhs] = std::get<VersionedTensor>(storage.at("lhs"));
    CHECK_VERSION(lhs, version_lhs);
    const auto &[rhs, version_rhs] = std::get<VersionedTensor>(storage.at("rhs"));
    CHECK_VERSION(rhs, version_rhs);
    return {matmul(grad_output, rhs.permute({0, 2, 1})), matmul(lhs.permute({0, 2, 1}), grad_output)};
}

}    
