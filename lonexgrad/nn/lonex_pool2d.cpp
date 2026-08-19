


#include <lonexgrad/lonex_device.h>
#include <lonexgrad/lonex_exception.h>
#include <lonexgrad/lonex_grad_mode.h>
#include <lonexgrad/nn/lonex_init.h>
#include <lonexgrad/nn/lonex_pool2d.h>
#include <lonexgrad/lonex_scalar.h>
#include <lonexgrad/lonex_tensor.h>

#include <cmath>
#include <format>
#include <ostream>

namespace lonexgrad::nn {



MinPool2d::MinPool2d(int kernel_size, int stride, int padding)
    : kernel_size_(kernel_size), stride_(stride), padding_(padding) {
    if (kernel_size <= 0) {
        LONEXGRAD_EXCEPTION(std::format("Expected kernel_size > 0, given kernel_size={:d}", kernel_size));
    }
    if (stride <= 0) {
        LONEXGRAD_EXCEPTION(std::format("Expected stride > 0, given stride={:d}", stride));
    }
    if (padding <= 0) {
        LONEXGRAD_EXCEPTION(std::format("Expected padding > 0, given padding={:d}", padding));
    }
}

auto MinPool2d::forward(const Tensor &input) const -> Tensor {
    return min_pool2d(input, kernel_size_, stride_, padding_);
}

void MinPool2d::pretty_print(std::ostream &os) const {
    os << std::format("MinPool2d(kernel_size={:d}, stride={:d}, padding={:d})", kernel_size_, stride_, padding_);
}



MaxPool2d::MaxPool2d(int kernel_size, int stride, int padding)
    : kernel_size_(kernel_size), stride_(stride), padding_(padding) {
    if (kernel_size <= 0) {
        LONEXGRAD_EXCEPTION(std::format("Expected kernel_size > 0, given kernel_size={:d}", kernel_size));
    }
    if (stride <= 0) {
        LONEXGRAD_EXCEPTION(std::format("Expected stride > 0, given stride={:d}", stride));
    }
    if (padding <= 0) {
        LONEXGRAD_EXCEPTION(std::format("Expected padding > 0, given padding={:d}", padding));
    }
}

auto MaxPool2d::forward(const Tensor &input) const -> Tensor {
    return max_pool2d(input, kernel_size_, stride_, padding_);
}

void MaxPool2d::pretty_print(std::ostream &os) const {
    os << std::format("MaxPool2d(kernel_size={:d}, stride={:d}, padding={:d})", kernel_size_, stride_, padding_);
}



AvgPool2d::AvgPool2d(int kernel_size, int stride, int padding)
    : kernel_size_(kernel_size), stride_(stride), padding_(padding) {
    if (kernel_size <= 0) {
        LONEXGRAD_EXCEPTION(std::format("Expected kernel_size > 0, given kernel_size={:d}", kernel_size));
    }
    if (stride <= 0) {
        LONEXGRAD_EXCEPTION(std::format("Expected stride > 0, given stride={:d}", stride));
    }
    if (padding <= 0) {
        LONEXGRAD_EXCEPTION(std::format("Expected padding > 0, given padding={:d}", padding));
    }
}

auto AvgPool2d::forward(const Tensor &input) const -> Tensor {
    return avg_pool2d(input, kernel_size_, stride_, padding_);
}

void AvgPool2d::pretty_print(std::ostream &os) const {
    os << std::format("AvgPool2d(kernel_size={:d}, stride={:d}, padding={:d})", kernel_size_, stride_, padding_);
}



}    
