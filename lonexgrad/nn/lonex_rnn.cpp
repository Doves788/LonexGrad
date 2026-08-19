


#include <lonexgrad/lonex_device.h>
#include <lonexgrad/lonex_exception.h>
#include <lonexgrad/lonex_index.h>
#include <lonexgrad/nn/lonex_init.h>
#include <lonexgrad/nn/lonex_rnn.h>
#include <lonexgrad/lonex_scalar.h>
#include <lonexgrad/lonex_shape.h>
#include <lonexgrad/lonex_tensor.h>
#include <lonexgrad/lonex_util.h>

#include <algorithm>
#include <cmath>
#include <format>
#include <memory>
#include <optional>
#include <ostream>
#include <ranges>
#include <tuple>
#include <vector>

namespace lonexgrad::nn {


RNN::RNN(int input_size, int hidden_size, const RNNOptions &options, ScalarType dtype, Device device)
    : input_size_(input_size), hidden_size_(hidden_size), options_(options) {
    if (input_size <= 0) {
        LONEXGRAD_EXCEPTION(std::format("Expected input_size > 0, given input_size={:d}", input_size));
    }
    if (hidden_size <= 0) {
        LONEXGRAD_EXCEPTION(std::format("Expected hidden_size > 0, given hidden_size={:d}", hidden_size));
    }
    if (options.num_layers <= 0) {
        LONEXGRAD_EXCEPTION(std::format("Expected num_layers > 0, given num_layers={:d}", options.num_layers));
    }
    double stdev = 1.0 / std::sqrt(hidden_size);
    const auto init_w = [&](const Shape &dims, bool requires_grad) -> std::shared_ptr<Tensor> {
        auto w = std::make_shared<Tensor>(zeros(dims, dtype, device, requires_grad));
        nn::uniform_(*w, -stdev, stdev);
        return w;
    };
    for (const auto i : std::views::iota(0, options.num_layers)) {
        weights_ih.push_back(
            init_w({i == 0 ? input_size : (options_.bidirectional ? 2 : 1) * hidden_size, hidden_size}, true)
        );
        weights_hh.push_back(init_w({hidden_size, hidden_size}, true));
        biases_ih.push_back(init_w({hidden_size}, options.bias));
        biases_hh.push_back(init_w({hidden_size}, options.bias));
        if (options_.bidirectional) {
            weights_ih_reverse.push_back(init_w({i == 0 ? input_size : 2 * hidden_size, hidden_size}, true));
            weights_hh_reverse.push_back(init_w({hidden_size, hidden_size}, true));
            biases_ih_reverse.push_back(init_w({hidden_size}, options.bias));
            biases_hh_reverse.push_back(init_w({hidden_size}, options.bias));
        }
    }

    
    for (auto &w : weights_ih) {
        register_param(w);
    }
    for (auto &w : weights_hh) {
        register_param(w);
    }
    for (auto &w : biases_ih) {
        register_param(w);
    }
    for (auto &w : biases_hh) {
        register_param(w);
    }
    for (auto &w : weights_ih_reverse) {
        register_param(w);
    }
    for (auto &w : weights_hh_reverse) {
        register_param(w);
    }
    for (auto &w : biases_ih_reverse) {
        register_param(w);
    }
    for (auto &w : biases_hh_reverse) {
        register_param(w);
    }
}

namespace {
struct SortIdx {
    int i;    
    int b;    
    int t;    
    int d;    
};
}    


auto RNN::forward(const Tensor &_input, const std::optional<Tensor> &_h) const -> Output {
    if (_input.dim() != 2 && _input.dim() != 3) {
        LONEXGRAD_EXCEPTION(
            std::format("Expected input to have 2 or 3 dimensions, given input of shape {:s}", _input.shape())
        );
    }

    
    bool is_batched = _input.dim() == 3;
    Tensor input = is_batched ? _input : _input.unsqueeze(1);

    
    if (options_.batch_first) {
        input = input.permute({1, 0, 2});
    }

    if (input.size(2) != input_size_) {
        LONEXGRAD_EXCEPTION(
            std::format(
                "Expected input of shape {:s} to have input_size of {:d}, given {:d}",
                _input.shape(),
                input_size_,
                _input.size(2)
            )
        );
    }

    int seq_len = input.size(0);
    int batch_size = input.size(1);

    
    Tensor h_0 = zeros(
        {options_.bidirectional ? 2 : 1, options_.num_layers, batch_size, hidden_size_},
        TensorOptions().dtype(input.dtype()).device(input.device())

    );
    if (_h.has_value()) {
        Tensor h = _h.value();
        if (h.dim() != 2 && h.dim() != 3) {
            LONEXGRAD_EXCEPTION(std::format("Expected h to have 2 or 3 dimensions, given h of shape {:s}", h.shape()));
        }
        if (is_batched && h.dim() == 2) {
            LONEXGRAD_EXCEPTION("Given batched input but unbatched h");
        }
        h = (h.dim() == 2) ? h.unsqueeze(1) : h;
        if (h.size(2) != hidden_size_) {
            LONEXGRAD_EXCEPTION(
                std::format(
                    "Expected given h of shape {:s} to have hidden_size of {:d}, given {:d}",
                    h.shape(),
                    hidden_size_,
                    h.size(2)
                )
            );
        }
        int expected_size = options_.num_layers * (options_.bidirectional ? 2 : 1);
        if (h.size(0) != expected_size) {
            LONEXGRAD_EXCEPTION(
                std::format(
                    "Expected given h of shape {:s} to have D*num_layers of {:d}, given {:d}",
                    h.shape(),
                    expected_size,
                    h.size(0)
                )
            );
        }
        
        for (int i : std::views::iota(0, h.size(0))) {
            h_0[{options_.bidirectional ? i % 2 : 0, options_.bidirectional ? i / 2 : i}] = h[i];
        }
    }

    TensorList h_output;

    for (int layer : std::views::iota(0, options_.num_layers)) {
        std::vector<SortIdx> idx_data;
        TensorList outputs;
        
        Tensor h_t = h_0[{0, layer}];
        for (int t : std::views::iota(0, seq_len)) {
            h_t = tanh(
                matmul(input[t], *weights_ih[layer]) + biases_ih[layer]->expand({batch_size, hidden_size_})
                + matmul(h_t, *weights_hh[layer]) + biases_hh[layer]->expand({batch_size, hidden_size_})
            );
            for (int b : std::views::iota(0, batch_size)) {
                idx_data.push_back(SortIdx{.i = static_cast<int>(idx_data.size()), .b = b, .t = t, .d = 0});
            }
            outputs.push_back(h_t);
        }
        h_output.push_back(h_t);

        
        if (options_.bidirectional) {
            Tensor h_t_reverse = h_0[{1, layer}];
            for (int t : std::views::iota(0, seq_len) | std::views::reverse) {
                h_t_reverse = tanh(
                    matmul(input[t], *weights_ih_reverse[layer])
                    + biases_ih_reverse[layer]->expand({batch_size, hidden_size_})
                    + matmul(h_t_reverse, *weights_hh_reverse[layer])
                    + biases_hh_reverse[layer]->expand({batch_size, hidden_size_})
                );
                for (int b : std::views::iota(0, batch_size)) {
                    idx_data.push_back(SortIdx{.i = static_cast<int>(idx_data.size()), .b = b, .t = t, .d = 1});
                }
                outputs.push_back(h_t_reverse);
            }
            h_output.push_back(h_t_reverse);
        }
        
        
        std::ranges::sort(idx_data, [](const SortIdx &lhs, const SortIdx &rhs) -> bool {
            return std::tie(lhs.t, lhs.b, lhs.d) < std::tie(rhs.t, rhs.b, rhs.d);
        });
        std::vector<int> indices = idx_data | std::views::transform([](const SortIdx &idx) -> int { return idx.i; })
                                   | lonexgrad::to<std::vector<int>>();

        Tensor output = index_select(cat(outputs, 0), indices, 0)
                            .reshape({seq_len, batch_size, (options_.bidirectional ? 2 : 1) * hidden_size_});
        input = output;
    }
    Tensor output = input;

    
    if (options_.batch_first) {
        output = output.permute({1, 0, 2});
    }
    Tensor h_out = stack(h_output, 0);
    if (!is_batched) {
        output = output.squeeze(1);
        h_out = h_out.squeeze(1);
    }
    return {.output = output, .h = h_out};
}

void RNN::pretty_print(std::ostream &os) const {
    os << std::format(
        "RNN(in_features={:d}, hidden_size={:d}, num_layers={:d}, bias={:}, bidirectional={:}, dtype={}, device={})",
        input_size_,
        hidden_size_,
        options_.num_layers,
        options_.bias,
        options_.bidirectional,
        weights_hh[0]->dtype(),
        weights_hh[0]->device()
    );
}

}    
