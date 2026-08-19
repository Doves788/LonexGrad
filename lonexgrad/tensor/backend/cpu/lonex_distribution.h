


#ifndef lonexgrad_BACKEND_CPU_DISTRIBUTION_H_
#define lonexgrad_BACKEND_CPU_DISTRIBUTION_H_

#include <lonexgrad/lonex_concepts.h>
#include <lonexgrad/lonex_random.h>
#include <lonexgrad/lonex_tensor.h>

#include "tensor/backend/common/lonex_distribution.h"

namespace lonexgrad::cpu {

template <common::distribution::DistributionOpT Op, typename... Params>
    requires IsAllOf<Tensor, Params...>
auto dist_runner(Generator &gen, const Params &...params) -> Tensor;

template <common::distribution::DistributionOpT Op, typename... Params>
    requires IsAllOf<Tensor, Params...>
void dist_inplace_runner(Tensor &tensor, Generator &gen, const Params &...params);

}    

#endif    
