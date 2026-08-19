



#ifndef lonexgrad_NN_OPTIMIZER_ADAGRAD_H_
#define lonexgrad_NN_OPTIMIZER_ADAGRAD_H_

#include <lonexgrad/export.h>
#include <lonexgrad/optim/lonex_optimizer.h>
#include <lonexgrad/lonex_tensor.h>

#include <functional>
#include <string>
#include <vector>

namespace lonexgrad::optim {



struct LONEXGRAD_EXPORT AdagradOptions {
    RegularizationMode regularization_mode = RegularizationMode::l2;
    double weight_decay = 0;
    double learning_rate_decay = 0;
    double eps = 1e-10;
    bool maximize = false;
};

class LONEXGRAD_EXPORT Adagrad : public Optimizer {
    using TensorRefList = std::vector<std::reference_wrapper<Tensor>>;

public:
    






    Adagrad(const TensorRefList &params, double learning_rate, const AdagradOptions &options = {});

    



    void save(const std::string &path) const override;

    



    void load(const std::string &path) override;

    



    void add_parameters(const std::vector<std::reference_wrapper<Tensor>> &params) override;

    



    void step() override;

protected:
    double learning_rate_;
    AdagradOptions options_;
    std::vector<int> steps_;
    std::vector<Tensor> state_sums_;
};

}    

#endif    
