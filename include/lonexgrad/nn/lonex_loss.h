


#ifndef lonexgrad_NN_LOSS_H_
#define lonexgrad_NN_LOSS_H_

#include <lonexgrad/export.h>
#include <lonexgrad/nn/lonex_module.h>
#include <lonexgrad/lonex_tensor.h>

#include <memory>
#include <optional>
#include <ostream>
#include <string>

namespace lonexgrad::nn {


enum class LONEXGRAD_EXPORT ReductionMode {
    none,
    mean,
    sum,
    batch_mean,
};











[[nodiscard]] LONEXGRAD_EXPORT auto
    l1_loss(const Tensor &input, const Tensor &target, ReductionMode mode = ReductionMode::mean) -> Tensor;

class LONEXGRAD_EXPORT L1Loss : public Module {
public:
    



    L1Loss(ReductionMode mode = ReductionMode::mean);

    





    [[nodiscard]] auto forward(const Tensor &input, const Tensor &target) const -> Tensor {
        return l1_loss(input, target, mode_);
    }

    





    [[nodiscard]] auto operator()(const Tensor &input, const Tensor &target) const -> Tensor {
        return forward(input, target);
    }

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "L1Loss";
    }

private:
    ReductionMode mode_;
};











[[nodiscard]] LONEXGRAD_EXPORT auto
    mse_loss(const Tensor &input, const Tensor &target, ReductionMode mode = ReductionMode::mean) -> Tensor;

class LONEXGRAD_EXPORT MSELoss : public Module {
public:
    



    MSELoss(ReductionMode mode = ReductionMode::mean);

    





    [[nodiscard]] auto forward(const Tensor &input, const Tensor &target) const -> Tensor {
        return mse_loss(input, target, mode_);
    }

    





    [[nodiscard]] auto operator()(const Tensor &input, const Tensor &target) const -> Tensor {
        return forward(input, target);
    }

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "MSELoss";
    }

private:
    ReductionMode mode_;
};













[[nodiscard]] LONEXGRAD_EXPORT auto cross_entropy_loss(
    const Tensor &input,
    const Tensor &target_indices,
    const std::optional<Tensor> &weight = std::nullopt,
    ReductionMode mode = ReductionMode::mean
) -> Tensor;

class LONEXGRAD_EXPORT CrossEntropyLoss : public Module {
public:
    




    CrossEntropyLoss(const std::optional<Tensor> &weight = std::nullopt, ReductionMode mode = ReductionMode::mean);

    






    [[nodiscard]] auto forward(const Tensor &input, const Tensor &target_indices) const -> Tensor {
        return cross_entropy_loss(
            input,
            target_indices,
            weight_.has_value() ? std::make_optional<Tensor>(**weight_) : std::nullopt,
            mode_
        );
    }

    






    [[nodiscard]] auto operator()(const Tensor &input, const Tensor &target_indices) const -> Tensor {
        return forward(input, target_indices);
    }

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "CrossEntropyLoss";
    }

private:
    ReductionMode mode_;
    std::optional<std::shared_ptr<Tensor>> weight_;
};













[[nodiscard]] LONEXGRAD_EXPORT auto nll_loss(
    const Tensor &input,
    const Tensor &target_indices,
    const std::optional<Tensor> &weight = std::nullopt,
    ReductionMode mode = ReductionMode::mean
) -> Tensor;

class LONEXGRAD_EXPORT NLLLoss : public Module {
public:
    




    NLLLoss(const std::optional<Tensor> &weight = std::nullopt, ReductionMode mode = ReductionMode::mean);

    






    [[nodiscard]] auto forward(const Tensor &input, const Tensor &target_indices) const -> Tensor {
        return nll_loss(
            input,
            target_indices,
            weight_.has_value() ? std::make_optional<Tensor>(**weight_) : std::nullopt,
            mode_
        );
    }

    






    [[nodiscard]] auto operator()(const Tensor &input, const Tensor &target_indices) const -> Tensor {
        return forward(input, target_indices);
    }

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "NLLLoss";
    }

private:
    ReductionMode mode_;
    std::optional<std::shared_ptr<Tensor>> weight_;
};













[[nodiscard]] LONEXGRAD_EXPORT auto kld_loss(
    const Tensor &input,
    const Tensor &target,
    ReductionMode mode = ReductionMode::mean,
    bool log_target = false
) -> Tensor;

class LONEXGRAD_EXPORT KLDivLoss : public Module {
public:
    




    KLDivLoss(ReductionMode mode = ReductionMode::mean, bool log_target = false);

    





    [[nodiscard]] auto forward(const Tensor &input, const Tensor &target) const -> Tensor {
        return kld_loss(input, target, mode_, log_target_);
    }

    





    [[nodiscard]] auto operator()(const Tensor &input, const Tensor &target) const -> Tensor {
        return forward(input, target);
    }

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "KLDivLoss";
    }

private:
    ReductionMode mode_;
    bool log_target_;
};













[[nodiscard]] LONEXGRAD_EXPORT auto
    bce_loss(const Tensor &input, const Tensor &target, ReductionMode mode = ReductionMode::mean) -> Tensor;

class LONEXGRAD_EXPORT BCELoss : public Module {
public:
    



    BCELoss(ReductionMode mode = ReductionMode::mean);

    






    [[nodiscard]] auto forward(const Tensor &input, const Tensor &target) const -> Tensor {
        return bce_loss(input, target, mode_);
    }

    






    [[nodiscard]] auto operator()(const Tensor &input, const Tensor &target) const -> Tensor {
        return forward(input, target);
    }

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "BCELoss";
    }

private:
    ReductionMode mode_;
};













[[nodiscard]] LONEXGRAD_EXPORT auto
    bce_with_logits_loss(const Tensor &input, const Tensor &target, ReductionMode mode = ReductionMode::mean) -> Tensor;

class LONEXGRAD_EXPORT BCEWithLogitsLoss : public Module {
public:
    



    BCEWithLogitsLoss(ReductionMode mode = ReductionMode::mean);

    






    [[nodiscard]] auto forward(const Tensor &input, const Tensor &target) const -> Tensor {
        return bce_with_logits_loss(input, target, mode_);
    }

    






    [[nodiscard]] auto operator()(const Tensor &input, const Tensor &target) const -> Tensor {
        return forward(input, target);
    }

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "BCEWithLogitsLoss";
    }

private:
    ReductionMode mode_;
};












[[nodiscard]] LONEXGRAD_EXPORT auto
    huber_loss(const Tensor &input, const Tensor &target, double delta = 1.0, ReductionMode mode = ReductionMode::mean)
        -> Tensor;

class LONEXGRAD_EXPORT HuberLoss : public Module {
public:
    




    HuberLoss(double delta = 1.0, ReductionMode mode = ReductionMode::mean);

    





    [[nodiscard]] auto forward(const Tensor &input, const Tensor &target) const -> Tensor {
        return huber_loss(input, target, delta_, mode_);
    }

    





    [[nodiscard]] auto operator()(const Tensor &input, const Tensor &target) const -> Tensor {
        return forward(input, target);
    }

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "HuberLoss";
    }

private:
    double delta_;
    ReductionMode mode_;
};












[[nodiscard]] LONEXGRAD_EXPORT auto smooth_l1_loss(
    const Tensor &input,
    const Tensor &target,
    double beta = 1.0,
    ReductionMode mode = ReductionMode::mean
) -> Tensor;

class LONEXGRAD_EXPORT SmoothL1Loss : public Module {
public:
    




    SmoothL1Loss(double beta = 1.0, ReductionMode mode = ReductionMode::mean);

    





    [[nodiscard]] auto forward(const Tensor &input, const Tensor &target) const -> Tensor {
        return smooth_l1_loss(input, target, beta_, mode_);
    }

    





    [[nodiscard]] auto operator()(const Tensor &input, const Tensor &target) const -> Tensor {
        return forward(input, target);
    }

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "SmoothL1Loss";
    }

private:
    double beta_;
    ReductionMode mode_;
};











[[nodiscard]] LONEXGRAD_EXPORT auto
    soft_margin_loss(const Tensor &input, const Tensor &target, ReductionMode mode = ReductionMode::mean) -> Tensor;

class LONEXGRAD_EXPORT SoftMarginLoss : public Module {
public:
    



    SoftMarginLoss(ReductionMode mode = ReductionMode::mean);

    





    [[nodiscard]] auto forward(const Tensor &input, const Tensor &target) const -> Tensor {
        return soft_margin_loss(input, target, mode_);
    }

    





    [[nodiscard]] auto operator()(const Tensor &input, const Tensor &target) const -> Tensor {
        return forward(input, target);
    }

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "SoftMarginLoss";
    }

private:
    ReductionMode mode_;
};



}    

#endif    
