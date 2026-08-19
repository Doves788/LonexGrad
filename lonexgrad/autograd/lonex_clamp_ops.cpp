


#include "autograd/lonex_clamp_ops.h"

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



auto TensorClamp::forward(
    AutogradStorage &storage,
    bool is_grad_required,
    const Tensor &tensor,
    const Tensor &min,
    const Tensor &max
) -> Tensor {
    auto result = tensor.clone();
    get_backend(tensor.device())->clamp_(result, min, max);
    if (is_grad_required) {
        storage["result"] = make_versioned_tensor(result);
        storage["min"] = make_versioned_tensor(min);
        storage["max"] = make_versioned_tensor(max);
    }
    return result;
}
auto TensorClamp::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[result, result_version] = std::get<VersionedTensor>(storage.at("result"));
    const auto &[min, min_version] = std::get<VersionedTensor>(storage.at("min"));
    const auto &[max, max_version] = std::get<VersionedTensor>(storage.at("max"));

    CHECK_VERSION(result, result_version);
    CHECK_VERSION(min, min_version);
    CHECK_VERSION(max, max_version);
    return {where((result > min && result < max), grad_output, 0)};
}

}    
