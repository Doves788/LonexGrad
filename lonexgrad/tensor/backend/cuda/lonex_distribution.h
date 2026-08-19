


#ifndef lonexgrad_BACKEND_CUDA_DISTRIBUTION_H_
#define lonexgrad_BACKEND_CUDA_DISTRIBUTION_H_

#include <lonexgrad/lonex_concepts.h>
#include <lonexgrad/lonex_random.h>
#include <lonexgrad/lonex_tensor.h>

#include "tensor/backend/common/lonex_distribution.h"

namespace lonexgrad::cuda {
















template <common::distribution::DistributionOpT Op>
auto dist_runner(Generator &gen, const Tensor &p1) -> Tensor;

template <common::distribution::DistributionOpT Op>
auto dist_runner(Generator &gen, const Tensor &p1, const Tensor &p2) -> Tensor;

template <common::distribution::DistributionOpT Op>
void dist_inplace_runner(Tensor &tensor, Generator &gen, const Tensor &p1);

template <common::distribution::DistributionOpT Op>
void dist_inplace_runner(Tensor &tensor, Generator &gen, const Tensor &p1, const Tensor &p2);

}    

#endif    
