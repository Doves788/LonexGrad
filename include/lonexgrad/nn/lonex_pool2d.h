


#ifndef lonexgrad_NN_POOL2D_H_
#define lonexgrad_NN_POOL2D_H_

#include <lonexgrad/export.h>
#include <lonexgrad/nn/lonex_module.h>
#include <lonexgrad/lonex_tensor.h>

#include <ostream>
#include <string>

namespace lonexgrad::nn {


class LONEXGRAD_EXPORT MinPool2d : public Module {
public:
    





    MinPool2d(int kernel_size, int stride, int padding);

    [[nodiscard]] auto forward(const Tensor &input) const -> Tensor;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "MinPool2D";
    }

private:
    int kernel_size_;
    int stride_;
    int padding_;
};


class LONEXGRAD_EXPORT MaxPool2d : public Module {
public:
    





    MaxPool2d(int kernel_size, int stride, int padding);

    [[nodiscard]] auto forward(const Tensor &input) const -> Tensor;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "MaxPool2D";
    }

private:
    int kernel_size_;
    int stride_;
    int padding_;
};


class LONEXGRAD_EXPORT AvgPool2d : public Module {
public:
    





    AvgPool2d(int kernel_size, int stride, int padding);

    [[nodiscard]] auto forward(const Tensor &input) const -> Tensor;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "AvgPool2D";
    }

private:
    int kernel_size_;
    int stride_;
    int padding_;
};

}    

#endif    
