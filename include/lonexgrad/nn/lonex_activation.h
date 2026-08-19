


#ifndef lonexgrad_NN_ACTIVATION_H_
#define lonexgrad_NN_ACTIVATION_H_

#include <lonexgrad/export.h>
#include <lonexgrad/nn/lonex_module.h>
#include <lonexgrad/lonex_tensor.h>

#include <ostream>
#include <string>

namespace lonexgrad::nn {


class LONEXGRAD_EXPORT Sigmoid : public Module {
public:
    Sigmoid() = default;

    [[nodiscard]] auto forward(const Tensor &input) const -> Tensor;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "Sigmoid";
    }
};


class LONEXGRAD_EXPORT LogSigmoid : public Module {
public:
    LogSigmoid() = default;

    [[nodiscard]] auto forward(const Tensor &input) const -> Tensor;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "LogSigmoid";
    }
};


class LONEXGRAD_EXPORT HardSigmoid : public Module {
public:
    HardSigmoid() = default;

    [[nodiscard]] auto forward(const Tensor &input) const -> Tensor;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "HardSigmoid";
    }
};


class LONEXGRAD_EXPORT Softplus : public Module {
public:
    





    Softplus(double beta = 1, double threshold = 20);

    [[nodiscard]] auto forward(const Tensor &input) const -> Tensor;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "Softplus";
    }

private:
    double beta_;
    double threshold_;
};


class LONEXGRAD_EXPORT ReLU : public Module {
public:
    ReLU() = default;

    [[nodiscard]] auto forward(const Tensor &input) const -> Tensor;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "ReLU";
    }
};


class LONEXGRAD_EXPORT ReLU6 : public Module {
public:
    ReLU6() = default;

    [[nodiscard]] auto forward(const Tensor &input) const -> Tensor;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "ReLU6";
    }
};


class LONEXGRAD_EXPORT LeakyReLU : public Module {
public:
    




    LeakyReLU(double negative_slope = 0.01);

    [[nodiscard]] auto forward(const Tensor &input) const -> Tensor;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "LeakyReLU";
    }

private:
    double negative_slope_;
};


class LONEXGRAD_EXPORT ELU : public Module {
public:
    




    ELU(double alpha = 0.01);

    [[nodiscard]] auto forward(const Tensor &input) const -> Tensor;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "ELU";
    }

private:
    double alpha_;
};


class LONEXGRAD_EXPORT SELU : public Module {
public:
    [[nodiscard]] auto forward(const Tensor &input) const -> Tensor;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "SELU";
    }
};


class LONEXGRAD_EXPORT SiLU : public Module {
public:
    [[nodiscard]] auto forward(const Tensor &input) const -> Tensor;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "SiLU";
    }
};


class LONEXGRAD_EXPORT Tanh : public Module {
public:
    [[nodiscard]] auto forward(const Tensor &input) const -> Tensor;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "Tanh";
    }
};


class LONEXGRAD_EXPORT HardTanh : public Module {
public:
    





    HardTanh(double min = -1, double max = 1);

    [[nodiscard]] auto forward(const Tensor &input) const -> Tensor;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "HardTanh";
    }

private:
    double min_;
    double max_;
};


class LONEXGRAD_EXPORT Softsign : public Module {
public:
    [[nodiscard]] auto forward(const Tensor &input) const -> Tensor;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "Softsign";
    }
};


class LONEXGRAD_EXPORT Softmax : public Module {
public:
    




    Softmax(int dim = -1);

    [[nodiscard]] auto forward(const Tensor &input) const -> Tensor;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "Softmax";
    }

private:
    int dim_;
};

}    

#endif    
