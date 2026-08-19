


#ifndef lonexgrad_NN_DROPOUT_H_
#define lonexgrad_NN_DROPOUT_H_

#include <lonexgrad/export.h>
#include <lonexgrad/nn/lonex_module.h>
#include <lonexgrad/lonex_tensor.h>

#include <ostream>
#include <string>

namespace lonexgrad::nn {


class LONEXGRAD_EXPORT Dropout : public Module {
public:
    






    Dropout(double p = 0.5);

    [[nodiscard]] auto forward(const Tensor &input) const -> Tensor;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "Dropout";
    }

private:
    double p_;
};

}    

#endif    
