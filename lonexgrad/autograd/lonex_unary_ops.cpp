


#include "autograd/lonex_unary_ops.h"

#include <lonexgrad/lonex_autograd.h>
#include <lonexgrad/lonex_device.h>
#include <lonexgrad/lonex_exception.h>
#include <lonexgrad/lonex_grad_mode.h>
#include <lonexgrad/lonex_scalar.h>
#include <lonexgrad/lonex_tensor.h>

#include "tensor/backend/common/lonex_dispatch.h"
#include "tensor/lonex_backend_register.h"

#include <format>
#include <numbers>

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


auto TensorClone::forward(
    [[maybe_unused]] AutogradStorage &storage,
    [[maybe_unused]] bool is_grad_required,
    const Tensor &tensor
) -> Tensor {
    return get_backend(tensor.device())->identity(tensor);
}
auto TensorClone::backward([[maybe_unused]] const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    return {grad_output};
}



auto TensorToScalarType::forward(
    [[maybe_unused]] AutogradStorage &storage,
    [[maybe_unused]] bool is_grad_required,
    const Tensor &tensor,
    ScalarType dtype
) -> Tensor {
    if (is_grad_required) {
        storage["dtype"] = tensor.dtype();
    }
    
    auto result = (dtype == kBool) ? tensor.clamp(ClampOptions().min(0).max(1)) : tensor;
    return get_backend(result.device())->to(result, dtype);
}
auto TensorToScalarType::backward([[maybe_unused]] const AutogradStorage &storage, const Tensor &grad_output)
    -> GradList {
    const auto &dtype = std::get<ScalarType>(storage.at("dtype"));
    return {grad_output.to(dtype)};
}



auto TensorToDevice::forward(
    [[maybe_unused]] AutogradStorage &storage,
    [[maybe_unused]] bool is_grad_required,
    const Tensor &tensor,
    Device device
) -> Tensor {
    if (is_grad_required) {
        storage["device"] = tensor.device();
    }
    
    if (tensor.dtype() == kBool) {
        return {tensor.to_vec<bool>(), tensor.shape(), device};
    }
    return DISPATCH_ALL_TYPES(tensor.dtype(), "Tensor::to", [&]() -> Tensor {
        return {tensor.to_vec<scalar_t>(), tensor.shape(), device};
    });
}
auto TensorToDevice::backward([[maybe_unused]] const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &device = std::get<Device>(storage.at("device"));
    return {grad_output.to(device)};
}



auto TensorIdentity::forward(
    [[maybe_unused]] AutogradStorage &storage,
    [[maybe_unused]] bool is_grad_required,
    const Tensor &tensor
) -> Tensor {
    return get_backend(tensor.device())->identity(tensor);
}
auto TensorIdentity::backward([[maybe_unused]] const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    return {grad_output};
}



auto TensorAbs::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->abs(tensor);
}
auto TensorAbs::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    return {grad_output * sign(input)};
}



auto TensorSign::forward(
    [[maybe_unused]] AutogradStorage &storage,
    [[maybe_unused]] bool is_grad_required,
    const Tensor &tensor
) -> Tensor {
    return get_backend(tensor.device())->sign(tensor);
}
auto TensorSign::backward([[maybe_unused]] const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    return {zeros_like(grad_output)};
}



auto TensorNegate::forward(
    [[maybe_unused]] AutogradStorage &storage,
    [[maybe_unused]] bool is_grad_required,
    const Tensor &tensor
) -> Tensor {
    return get_backend(tensor.device())->negate(tensor);
}
auto TensorNegate::backward([[maybe_unused]] const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    return {-grad_output};
}



auto TensorLog::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->log(tensor);
}
auto TensorLog::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    return {grad_output / input};
}



auto TensorLog2::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->log2(tensor);
}
auto TensorLog2::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    return {grad_output / (std::numbers::ln2 * input)};
}



auto TensorLog10::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->log10(tensor);
}
auto TensorLog10::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    return {grad_output / (std::numbers::ln10 * input)};
}



auto TensorLog1p::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->log1p(tensor);
}
auto TensorLog1p::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    return {grad_output / (1.0 + input)};
}



auto TensorExp::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    auto result = get_backend(tensor.device())->exp(tensor);
    if (is_grad_required) {
        storage["result"] = make_versioned_tensor(result);
    }
    return result;
}
auto TensorExp::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[result, version] = std::get<VersionedTensor>(storage.at("result"));
    CHECK_VERSION(result, version);
    return {grad_output * result};
}



auto TensorExp2::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    auto result = get_backend(tensor.device())->exp2(tensor);
    if (is_grad_required) {
        storage["result"] = make_versioned_tensor(result);
    }
    return result;
}
auto TensorExp2::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[result, version] = std::get<VersionedTensor>(storage.at("result"));
    CHECK_VERSION(result, version);
    return {grad_output * (std::numbers::ln2 * result)};
}



auto TensorExpm1::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->expm1(tensor);
}
auto TensorExpm1::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    return {grad_output * exp(input)};
}



auto TensorSqrt::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    auto result = get_backend(tensor.device())->sqrt(tensor);
    if (is_grad_required) {
        storage["result"] = make_versioned_tensor(result);
    }
    return result;
}
auto TensorSqrt::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[result, version] = std::get<VersionedTensor>(storage.at("result"));
    CHECK_VERSION(result, version);
    return {grad_output / (2 * result)};
}



auto TensorSin::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->sin(tensor);
}
auto TensorSin::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    return {grad_output * cos(input)};
}



auto TensorCos::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->cos(tensor);
}
auto TensorCos::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    return {grad_output * (-sin(input))};
}



auto TensorTan::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->tan(tensor);
}
auto TensorTan::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    return {grad_output / pow(cos(input), 2)};
}



auto TensorASin::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->asin(tensor);
}
auto TensorASin::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    return {grad_output / sqrt(1 - pow(input, 2))};
}



auto TensorACos::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->acos(tensor);
}
auto TensorACos::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    return {-grad_output / sqrt(1 - pow(input, 2))};
}



auto TensorATan::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->atan(tensor);
}
auto TensorATan::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    return {grad_output / (pow(input, 2) + 1)};
}



auto TensorSinh::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->sinh(tensor);
}
auto TensorSinh::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    return {grad_output * cosh(input)};
}



auto TensorCosh::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->cosh(tensor);
}
auto TensorCosh::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    return {grad_output * sinh(input)};
}



auto TensorTanh::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->tanh(tensor);
}
auto TensorTanh::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    return {grad_output / pow(cosh(input), 2)};
}



auto TensorASinh::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->asinh(tensor);
}
auto TensorASinh::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    return {grad_output / sqrt(pow(input, 2) + 1)};
}






auto TensorACosh::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->acosh(tensor);
}
auto TensorACosh::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    return {grad_output / sqrt(pow(input, 2) - 1)};
}



auto TensorATanh::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->atanh(tensor);
}
auto TensorATanh::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    return {grad_output / (1 - pow(input, 2))};
}




auto TensorErf::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->erf(tensor);
}
auto TensorErf::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    return {grad_output * 2 * std::numbers::inv_sqrtpi * exp(-pow(input, 2))};
}




auto TensorErfc::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->erfc(tensor);
}
auto TensorErfc::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    return {-grad_output * 2 * std::numbers::inv_sqrtpi * exp(-pow(input, 2))};
}




auto TensorTGamma::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->tgamma(tensor);
}
auto TensorTGamma::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    return {grad_output * tgamma(input) * digamma(input)};
}




auto TensorLGamma::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->lgamma(tensor);
}
auto TensorLGamma::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    return {grad_output * digamma(input)};
}



auto TensorSigmoid::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    auto result = get_backend(tensor.device())->sigmoid(tensor);
    if (is_grad_required) {
        storage["result"] = make_versioned_tensor(result);
    }
    return result;
}
auto TensorSigmoid::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[result, version] = std::get<VersionedTensor>(storage.at("result"));
    CHECK_VERSION(result, version);
    return {grad_output * result * (1 - result)};
}



auto TensorLogSigmoid::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->log_sigmoid(tensor);
}
auto TensorLogSigmoid::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    return {grad_output / (exp(input) + 1)};
}




auto TensorHardSigmoid::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->hardsigmoid(tensor);
}
auto TensorHardSigmoid::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    const auto mask = (input > -3) && (input < 3);
    return {grad_output * where(mask, Scalar(1.0 / 6, input.dtype()), Scalar(0, input.dtype()))};
}




auto TensorSoftplus::forward(
    AutogradStorage &storage,
    bool is_grad_required,
    const Tensor &tensor,
    double beta,
    double threshold
) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
        storage["beta"] = beta;
        storage["threshold"] = threshold;
    }
    return get_backend(tensor.device())->softplus(tensor, beta, threshold);
}
auto TensorSoftplus::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    const auto &beta = std::get<double>(storage.at("beta"));
    const auto &threshold = std::get<double>(storage.at("threshold"));
    const auto mask = (input < threshold);
    return {grad_output * where(mask, sigmoid(beta * input), ones_like(input))};
}




auto TensorRelu::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->relu(tensor);
}
auto TensorRelu::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    const auto mask = (input < 0);
    return {grad_output * where(mask, Scalar(0, input.dtype()), Scalar(1, input.dtype()))};
}




auto TensorRelu6::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->relu6(tensor);
}
auto TensorRelu6::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    const auto mask = (input < 0) || (input > 6);
    return {grad_output * where(mask, Scalar(0, input.dtype()), Scalar(1, input.dtype()))};
}




auto TensorLeakyRelu::forward(
    AutogradStorage &storage,
    bool is_grad_required,
    const Tensor &tensor,
    double negative_slope
) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
        storage["negative_slope"] = negative_slope;
    }
    return get_backend(tensor.device())->leaky_relu(tensor, negative_slope);
}
auto TensorLeakyRelu::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    const auto &negative_slope = std::get<double>(storage.at("negative_slope"));
    const auto mask = (input < 0);
    return {grad_output * where(mask, Scalar(negative_slope, input.dtype()), Scalar(1, input.dtype()))};
}




auto TensorElu::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor, double alpha) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
        storage["alpha"] = alpha;
    }
    return get_backend(tensor.device())->elu(tensor, alpha);
}
auto TensorElu::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    const auto &alpha = std::get<double>(storage.at("alpha"));
    const auto mask = (input < 0);
    return {grad_output * where(mask, alpha * exp(input), Scalar(1, input.dtype()))};
}




auto TensorSelu::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->selu(tensor);
}
auto TensorSelu::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    constexpr double alpha = 1.67326324235437728;
    constexpr double scale = 1.05070098735548049;
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    const auto mask = (input < 0);
    return {grad_output * where(mask, scale * alpha * exp(input), Scalar(scale, input.dtype()))};
}



auto TensorSilu::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->silu(tensor);
}
auto TensorSilu::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    const auto s = sigmoid(input);
    return {grad_output * (s + input * s * (1 - s))};
}





auto TensorHardtanh::forward(
    AutogradStorage &storage,
    bool is_grad_required,
    const Tensor &tensor,
    double min,
    double max
) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
        storage["min"] = min;
        storage["max"] = max;
    }
    return get_backend(tensor.device())->hardtanh(tensor, min, max);
}
auto TensorHardtanh::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    const auto &min = std::get<double>(storage.at("min"));
    const auto &max = std::get<double>(storage.at("max"));
    const auto mask = (input < min) || (input > max);
    return {grad_output * where(mask, Scalar(0, input.dtype()), Scalar(1, input.dtype()))};
}



auto TensorSoftsign::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor) -> Tensor {
    if (is_grad_required) {
        storage["input"] = make_versioned_tensor(tensor);
    }
    return get_backend(tensor.device())->softsign(tensor);
}
auto TensorSoftsign::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[input, version] = std::get<VersionedTensor>(storage.at("input"));
    CHECK_VERSION(input, version);
    return {grad_output / pow(1 + abs(input), 2)};
}






auto TensorSoftmax::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor, int dim) -> Tensor {
    auto result = get_backend(tensor.device())->softmax(tensor, dim);
    if (is_grad_required) {
        storage["result"] = make_versioned_tensor(result);
        storage["dim"] = dim;
    }
    return result;
}
auto TensorSoftmax::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[result, version] = std::get<VersionedTensor>(storage.at("result"));
    CHECK_VERSION(result, version);
    const auto &dim = std::get<int>(storage.at("dim"));
    Tensor _grad_output = grad_output * result;
    Tensor sum_grad = sum(_grad_output, dim, true);
    return {_grad_output - result * sum_grad.expand(result.shape())};
}



auto TensorLogSoftmax::forward(AutogradStorage &storage, bool is_grad_required, const Tensor &tensor, int dim)
    -> Tensor {
    auto result = get_backend(tensor.device())->log_softmax(tensor, dim);
    if (is_grad_required) {
        storage["result"] = make_versioned_tensor(result);
        storage["dim"] = dim;
    }
    return result;
}
auto TensorLogSoftmax::backward(const AutogradStorage &storage, const Tensor &grad_output) -> GradList {
    const auto &[result, version] = std::get<VersionedTensor>(storage.at("result"));
    CHECK_VERSION(result, version);
    const auto &dim = std::get<int>(storage.at("dim"));
    return {grad_output - exp(result) * grad_output.sum(dim, true).expand(result.shape())};
}

}    
