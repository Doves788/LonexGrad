


#ifndef lonexgrad_NN_INIT_H_
#define lonexgrad_NN_INIT_H_

#include <lonexgrad/export.h>
#include <lonexgrad/lonex_tensor.h>

#include <string>
#include <tuple>
#include <unordered_map>

namespace lonexgrad::nn {

enum class LONEXGRAD_EXPORT GainActivation {
    linear,
    conv,
    sigmoid,
    tanh,
    relu,
    leaky_relu,
    selu
};

enum class LONEXGRAD_EXPORT FanMode {
    fan_in,
    fan_out
};

using GainActivationParams = std::unordered_map<std::string, double>;







LONEXGRAD_EXPORT auto calc_fan_in_out(const Tensor &tensor) -> std::tuple<double, double>;








LONEXGRAD_EXPORT auto calc_gain(GainActivation gain_activation, const GainActivationParams &params = {}) -> double;







LONEXGRAD_EXPORT void uniform_(Tensor tensor, double low, double high);







LONEXGRAD_EXPORT void normal_(Tensor tensor, double low, double high);






LONEXGRAD_EXPORT void constant_(Tensor tensor, double value);








LONEXGRAD_EXPORT void xavier_uniform_(Tensor tensor, double gain = 1.0);








LONEXGRAD_EXPORT void xavier_normal_(Tensor tensor, double gain = 1.0);










LONEXGRAD_EXPORT void kaiming_uniform_(Tensor tensor, double gain = 1.0, FanMode fan_mode = FanMode::fan_in);










LONEXGRAD_EXPORT void kaiming_normal_(Tensor tensor, double gain = 1.0, FanMode fan_mode = FanMode::fan_in);

}    

#endif    
