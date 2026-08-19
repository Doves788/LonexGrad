


#ifndef lonexgrad_NN_CONV2D_H_
#define lonexgrad_NN_CONV2D_H_

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


class LONEXGRAD_EXPORT Conv2d : public Module {
public:
    










    Conv2d(
        int in_channels,
        int out_channels,
        int kernel_size,
        int stride,
        int padding,
        bool bias = true,
        ScalarType dtype = kDefaultFloat,
        Device device = kCPU
    );

    





    [[nodiscard]] auto forward(const Tensor &input) const -> Tensor;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "Conv2d";
    }

    std::shared_ptr<Tensor> weight;
    std::optional<std::shared_ptr<Tensor>> bias;

private:
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int stride_;
    int padding_;
    bool has_bias_;
};

}    

#endif    
