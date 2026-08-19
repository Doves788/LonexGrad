


#ifndef lonexgrad_NN_OPTIMIZER_H_
#define lonexgrad_NN_OPTIMIZER_H_

#include <lonexgrad/export.h>
#include <lonexgrad/lonex_tensor.h>

#include <functional>
#include <string>
#include <vector>

namespace lonexgrad::optim {

enum class LONEXGRAD_EXPORT RegularizationMode {
    l1,
    l2
};

class LONEXGRAD_EXPORT Optimizer {
public:
    virtual ~Optimizer() = default;
    Optimizer() = default;

    
    Optimizer(const Optimizer &) = delete;
    Optimizer(Optimizer &&) = default;
    auto operator=(const Optimizer &) -> Optimizer & = delete;
    auto operator=(Optimizer &&) -> Optimizer & = delete;

    


    virtual void step() = 0;

    



    virtual void save(const std::string &path) const = 0;

    



    virtual void load(const std::string &path) = 0;

    


    void zero_grad();

    



    virtual void add_parameters(const std::vector<std::reference_wrapper<Tensor>> &params);

protected:
    std::vector<std::reference_wrapper<Tensor>> params_;
};

}    

#endif    
