


#include <lonexgrad/nn/lonex_dropout.h>
#include <lonexgrad/lonex_tensor.h>

#include <format>
#include <ostream>

namespace lonexgrad::nn {

Dropout::Dropout(double p)
    : p_(p) {}

auto Dropout::forward(const Tensor &input) const -> Tensor {
    if (is_train_ && p_ > 0) {
        Tensor mask =
            uniform_real(0, 1, input.shape(), TensorOptions().dtype(input.dtype()).device(input.device())) > p_;
        return 1.0 / (1.0 - p_) * mask * input;

    } else {
        return input;
    }
}

void Dropout::pretty_print(std::ostream &os) const {
    os << std::format("Dropout(p={:f})", p_);
}

}    
