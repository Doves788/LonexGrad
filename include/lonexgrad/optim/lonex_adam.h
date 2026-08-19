



#ifndef lonexgrad_NN_OPTIMIZER_ADAM_H_
#define lonexgrad_NN_OPTIMIZER_ADAM_H_

#include <lonexgrad/export.h>
#include <lonexgrad/optim/lonex_optimizer.h>
#include <lonexgrad/lonex_tensor.h>

#include <functional>
#include <string>
#include <vector>

namespace lonexgrad::optim {

struct LONEXGRAD_EXPORT AdamBetas {
    double beta1;
    double beta2;
};



struct LONEXGRAD_EXPORT AdamOptions {
    RegularizationMode regularization_mode = RegularizationMode::l2;
    double weight_decay = 0;
    AdamBetas betas = {.beta1 = 0.9, .beta2 = 0.999};
    double eps = 1e-8;
    bool use_amsgrad = false;
    bool maximize = false;
};

class LONEXGRAD_EXPORT Adam : public Optimizer {
    using TensorRefList = std::vector<std::reference_wrapper<Tensor>>;

public:
    






    Adam(const TensorRefList &params, double learning_rate, const AdamOptions &options = {});

    



    void save(const std::string &path) const override;

    



    void load(const std::string &path) override;

    



    void add_parameters(const std::vector<std::reference_wrapper<Tensor>> &params) override;

    



    void step() override;

protected:
    double learning_rate_;
    AdamOptions options_;
    std::vector<Tensor> first_moments_;
    std::vector<Tensor> second_moments_;
    std::vector<Tensor> second_moments_max_;
    std::vector<int> steps_;
};

}    

#endif    
