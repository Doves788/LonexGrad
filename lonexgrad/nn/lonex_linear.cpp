


#include <lonexgrad/lonex_device.h>
#include <lonexgrad/lonex_exception.h>
#include <lonexgrad/nn/lonex_init.h>
#include <lonexgrad/nn/lonex_linear.h>
#include <lonexgrad/lonex_scalar.h>
#include <lonexgrad/lonex_shape.h>
#include <lonexgrad/lonex_tensor.h>

#include <cmath>
#include <format>
#include <memory>
#include <optional>
#include <ostream>

namespace lonexgrad::nn {

Linear::Linear(int in_features, int out_features, bool _bias, ScalarType dtype, Device device)
    : weight(std::make_shared<Tensor>(zeros({in_features, out_features}, dtype, device, true))),
      in_features_(in_features),
      out_features_(out_features) {
    if (in_features <= 0) {
        LONEXGRAD_EXCEPTION(std::format("Expected in_features > 0, given in_features={:d}", in_features));
    }
    if (out_features <= 0) {
        LONEXGRAD_EXCEPTION(std::format("Expected out_features > 0, given out_features={:d}", out_features));
    }
    
    nn::kaiming_uniform_(*weight);
    register_param(weight);
    if (_bias) {
        bias = std::make_shared<Tensor>(zeros({out_features}, dtype, device, true));
        const auto [fan_in, _] = calc_fan_in_out(*weight);
        double bound = 1.0 / std::sqrt(fan_in);
        nn::uniform_(*bias.value(), -bound, bound);
        register_param(bias.value());
    }
}

auto Linear::forward(const Tensor &input) const -> Tensor {
    Shape result_shape = input.shape();
    result_shape[-1] = out_features_;
    Tensor result = matmul(input.flatten(0, -2), *weight).reshape(result_shape);
    if (bias) {
        result = result + bias.value()->expand(result.shape());
    }
    return result;
}

void Linear::pretty_print(std::ostream &os) const {
    os << std::format(
        "Linear(in_features={:d}, out_features={:d}, bias={:}, dtype={}, device={})",
        in_features_,
        out_features_,
        bias.has_value(),
        weight->dtype(),
        weight->device()
    );
}

}    
