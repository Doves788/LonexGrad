



#ifndef lonexgrad_NN_OPTIMIZER_ADAMW_H_
#define lonexgrad_NN_OPTIMIZER_ADAMW_H_

#include <lonexgrad/export.h>
#include <lonexgrad/optim/lonex_optimizer.h>
#include <lonexgrad/lonex_tensor.h>

#include <functional>
#include <string>
#include <vector>

namespace lonexgrad::optim {

struct LONEXGRAD_EXPORT AdamWBetas {
    double beta1;
    double beta2;
};



struct LONEXGRAD_EXPORT AdamWOptions {
    RegularizationMode regularization_mode = RegularizationMode::l2;
    double weight_decay = 0;
    AdamWBetas betas = {.beta1 = 0.9, .beta2 = 0.999};
    double eps = 1e-8;
    bool use_amsgrad = false;
    bool maximize = false;
};

class LONEXGRAD_EXPORT AdamW : public Optimizer {
    using TensorRefList = std::vector<std::reference_wrapper<Tensor>>;

public:
    






    AdamW(const TensorRefList &params, double learning_rate, const AdamWOptions &options = {});

    



    void save(const std::string &path) const override;

    



    void load(const std::string &path) override;

    



    void add_parameters(const std::vector<std::reference_wrapper<Tensor>> &params) override;

    



    void step() override;

protected:
    double learning_rate_;
    AdamWOptions options_;
    std::vector<Tensor> first_moments_;
    std::vector<Tensor> second_moments_;
    std::vector<Tensor> second_moments_max_;
    std::vector<int> steps_;
};

}    

#endif    
