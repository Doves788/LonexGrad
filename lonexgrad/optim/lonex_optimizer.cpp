


#include <lonexgrad/optim/lonex_optimizer.h>
#include <lonexgrad/lonex_tensor.h>

#include <functional>
#include <vector>

namespace lonexgrad::optim {

void Optimizer::zero_grad() {
    for (auto &t : params_) {
        t.get().clear_grad();
    }
}

void Optimizer::add_parameters(const std::vector<std::reference_wrapper<Tensor>> &params) {
    params_.insert(params_.end(), params.begin(), params.end());
}

}    
