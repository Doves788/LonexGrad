


#ifndef lonexgrad_NN_OPTIMIZER_SGD_H_
#define lonexgrad_NN_OPTIMIZER_SGD_H_

#include <lonexgrad/export.h>
#include <lonexgrad/optim/lonex_optimizer.h>
#include <lonexgrad/lonex_tensor.h>

#include <functional>
#include <string>
#include <vector>

namespace lonexgrad::optim {



struct LONEXGRAD_EXPORT SGDOptions {
    RegularizationMode regularization_mode = RegularizationMode::l2;
    double weight_decay = 0;
    double momentum = 0;
    bool use_nesterov = false;
    bool maximize = false;
};

class LONEXGRAD_EXPORT SGD : public Optimizer {
    using TensorRefList = std::vector<std::reference_wrapper<Tensor>>;

public:
    






    SGD(const TensorRefList &params, double learning_rate, const SGDOptions &options = {});

    



    void save(const std::string &path) const override;

    



    void load(const std::string &path) override;

    



    void add_parameters(const std::vector<std::reference_wrapper<Tensor>> &params) override;

    



    void step() override;

public:
    double learning_rate_;
    SGDOptions options_;
    std::vector<Tensor> velocities_;
};

}    

#endif    
