


#ifndef lonexgrad_NN_BATCHNORM_H_
#define lonexgrad_NN_BATCHNORM_H_

#include <lonexgrad/lonex_device.h>
#include <lonexgrad/export.h>
#include <lonexgrad/nn/lonex_module.h>
#include <lonexgrad/lonex_scalar.h>
#include <lonexgrad/lonex_tensor.h>

#include <memory>
#include <ostream>
#include <string>

namespace lonexgrad::nn {


struct LONEXGRAD_EXPORT BatchNormOptions {
    double eps = 1e-5;
    double momentum = 0.1;
    bool affine = true;                 
    bool track_running_stats = true;    
};

class LONEXGRAD_EXPORT BatchNorm1d : public Module {
public:
    






    BatchNorm1d(
        int num_features,
        const BatchNormOptions &options = {},
        ScalarType dtype = kDefaultFloat,
        Device device = kCPU
    );

    




    [[nodiscard]] auto forward(const Tensor &input) -> Tensor;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "BatchNorm1d";
    }

    std::shared_ptr<Tensor> gamma;
    std::shared_ptr<Tensor> beta;
    std::shared_ptr<Tensor> moving_mean;
    std::shared_ptr<Tensor> moving_var;

private:
    int num_features_;
    BatchNormOptions options_;
};
class LONEXGRAD_EXPORT BatchNorm2d : public Module {
public:
    






    BatchNorm2d(
        int num_features,
        const BatchNormOptions &options = {},
        ScalarType dtype = kDefaultFloat,
        Device device = kCPU
    );

    




    [[nodiscard]] auto forward(const Tensor &input) -> Tensor;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "BatchNorm2d";
    }

    std::shared_ptr<Tensor> gamma;
    std::shared_ptr<Tensor> beta;
    std::shared_ptr<Tensor> moving_mean;
    std::shared_ptr<Tensor> moving_var;

private:
    int num_features_;
    BatchNormOptions options_;
};

}    

#endif    
