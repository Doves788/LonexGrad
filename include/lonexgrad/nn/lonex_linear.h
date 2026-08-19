


#ifndef lonexgrad_NN_LINEAR_H_
#define lonexgrad_NN_LINEAR_H_

#include <lonexgrad/lonex_device.h>
#include <lonexgrad/export.h>
#include <lonexgrad/nn/lonex_module.h>
#include <lonexgrad/lonex_scalar.h>
#include <lonexgrad/lonex_tensor.h>

#include <memory>
#include <optional>
#include <ostream>
#include <string>

namespace lonexgrad::nn {


class LONEXGRAD_EXPORT Linear : public Module {
public:
    







    Linear(int in_features, int out_features, bool bias = true, ScalarType dtype = kDefaultFloat, Device device = kCPU);

    





    [[nodiscard]] auto forward(const Tensor &input) const -> Tensor;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "Linear";
    }

    std::shared_ptr<Tensor> weight;
    std::optional<std::shared_ptr<Tensor>> bias;

private:
    int in_features_;
    int out_features_;
};

}    

#endif    
