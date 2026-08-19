


#ifndef lonexgrad_NN_EMBEDDING_H_
#define lonexgrad_NN_EMBEDDING_H_

#include <lonexgrad/lonex_device.h>
#include <lonexgrad/export.h>
#include <lonexgrad/nn/lonex_module.h>
#include <lonexgrad/lonex_scalar.h>
#include <lonexgrad/lonex_tensor.h>

#include <memory>
#include <ostream>
#include <string>

namespace lonexgrad::nn {


class LONEXGRAD_EXPORT Embedding : public Module {
public:
    






    Embedding(int num_embeddings, int embedding_dim, ScalarType dtype = kDefaultFloat, Device device = kCPU);

    




    [[nodiscard]] auto forward(const Tensor &input) const -> Tensor;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "Embedding";
    }

    std::shared_ptr<Tensor> weight;

private:
    int num_embeddings_;
    int embedding_dim_;
};

}    

#endif    
