


#ifndef lonexgrad_NN_LSTM_H_
#define lonexgrad_NN_LSTM_H_

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


struct LONEXGRAD_EXPORT LSTMOptions {
    int num_layers = 1;
    bool bias = true;
    bool batch_first = false;
    bool bidirectional = false;
};



class LONEXGRAD_EXPORT LSTM : public Module {
public:
    
    struct InitState {
        Tensor h;
        Tensor c;
    };

    
    struct Output {
        Tensor output;    
        Tensor h;         
        Tensor c;         
    };

    







    LSTM(
        int input_size,
        int hidden_size,
        const LSTMOptions &options = {},
        ScalarType dtype = kDefaultFloat,
        Device device = kCPU
    );

    












    [[nodiscard]] auto forward(const Tensor &input, const InitState &init_state) const -> Output;
    [[nodiscard]] auto forward(const Tensor &input) const -> Output;

    void pretty_print(std::ostream &os) const override;

    [[nodiscard]] auto name() const -> std::string override {
        return "LSTM";
    }

    CheckedVec<std::shared_ptr<Tensor>> weights_ih;
    CheckedVec<std::shared_ptr<Tensor>> weights_ih_reverse;
    CheckedVec<std::shared_ptr<Tensor>> weights_hh;
    CheckedVec<std::shared_ptr<Tensor>> weights_hh_reverse;
    CheckedVec<std::shared_ptr<Tensor>> biases_ih;
    CheckedVec<std::shared_ptr<Tensor>> biases_ih_reverse;
    CheckedVec<std::shared_ptr<Tensor>> biases_hh;
    CheckedVec<std::shared_ptr<Tensor>> biases_hh_reverse;

private:
    [[nodiscard]] auto forward(const Tensor &input, const std::optional<InitState> &init_state) const -> Output;
    int input_size_;
    int hidden_size_;
    LSTMOptions options_;
};

}    

#endif    
