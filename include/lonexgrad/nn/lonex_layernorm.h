


#ifndef lonexgrad_NN_LAYERNORM_H_
#define lonexgrad_NN_LAYERNORM_H_

#include <lonexgrad/lonex_device.h>
#include <lonexgrad/export.h>
#include <lonexgrad/nn/lonex_module.h>
#include <lonexgrad/lonex_scalar.h>
#include <lonexgrad/lonex_shape.h>
#include <lonexgrad/lonex_tensor.h>

#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace lonexgrad::nn {


struct LONEXGRAD_EXPORT LayerNormOptions {
    double eps = 1e-5;
    bool affine = true;    
    bool bias = true;      
};



class LONEXGRAD_EXPORT LayerNorm : public Module {
public:
    







    LayerNorm(
        const Shape &normalized_shape,
        const LayerNormOptions &options = {},
        ScalarType dtype = kDefaultFloat,
        Device device = kCPU
    );

    




    [[nodiscard]] auto forward(const Tensor &input) -> Tensor;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "LayerNorm";
    }

    std::shared_ptr<Tensor> gamma;    
    std::shared_ptr<Tensor> beta;     

private:
    Shape normalized_shape_;
    LayerNormOptions options_;
    std::vector<int> normalized_dims_;
};

}    

#endif    
