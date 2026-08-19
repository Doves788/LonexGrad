


#include <lonexgrad/lonex_device.h>
#include <lonexgrad/lonex_exception.h>
#include <lonexgrad/nn/lonex_embedding.h>
#include <lonexgrad/nn/lonex_init.h>
#include <lonexgrad/lonex_scalar.h>
#include <lonexgrad/lonex_tensor.h>

#include <cmath>
#include <format>
#include <memory>
#include <ostream>

namespace lonexgrad::nn {

Embedding::Embedding(int num_embeddings, int embedding_dim, ScalarType dtype, Device device)
    : weight(std::make_shared<Tensor>(zeros({num_embeddings, embedding_dim}, dtype, device, true))),
      num_embeddings_(num_embeddings),
      embedding_dim_(embedding_dim) {
    if (num_embeddings <= 0) {
        LONEXGRAD_EXCEPTION(std::format("Expected num_embeddings > 0, given num_embeddings={:d}", num_embeddings));
    }
    if (embedding_dim <= 0) {
        LONEXGRAD_EXCEPTION(std::format("Expected embedding_dim > 0, given embedding_dim={:d}", embedding_dim));
    }
    
    nn::normal_(*weight, 0, 1);
    register_param(weight);
}

auto Embedding::forward(const Tensor &input) const -> Tensor {
    return embedding(input, *weight);
}

void Embedding::pretty_print(std::ostream &os) const {
    os << std::format(
        "Embedding(num_embeddings={:d}, embedding_dim={:d}, dtype={}, device={})",
        num_embeddings_,
        embedding_dim_,
        weight->dtype(),
        weight->device()
    );
}

}    
