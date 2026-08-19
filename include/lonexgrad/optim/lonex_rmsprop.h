



#ifndef lonexgrad_NN_OPTIMIZER_RMSPROP_H_
#define lonexgrad_NN_OPTIMIZER_RMSPROP_H_

#include <lonexgrad/export.h>
#include <lonexgrad/optim/lonex_optimizer.h>
#include <lonexgrad/lonex_tensor.h>

#include <functional>
#include <string>
#include <vector>

namespace lonexgrad::optim {



struct LONEXGRAD_EXPORT RMSpropOptions {
    RegularizationMode regularization_mode = RegularizationMode::l2;
    double weight_decay = 0;
    double momentum = 0;
    double alpha = 0.99;
    double eps = 1e-8;
    bool center = false;
    bool maximize = false;
};

class LONEXGRAD_EXPORT RMSprop : public Optimizer {
    using TensorRefList = std::vector<std::reference_wrapper<Tensor>>;

public:
    






    RMSprop(const TensorRefList &params, double learning_rate, const RMSpropOptions &options = {});

    



    void save(const std::string &path) const override;

    



    void load(const std::string &path) override;

    



    void add_parameters(const std::vector<std::reference_wrapper<Tensor>> &params) override;

    



    void step() override;

protected:
    double learning_rate_;
    RMSpropOptions options_;
    std::vector<Tensor> square_averages_;
    std::vector<Tensor> velocities_;
    std::vector<Tensor> centers_;
};

}    

#endif    
