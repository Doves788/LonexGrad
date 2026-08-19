


#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <lonexgrad/lonex_device.h>
#include <lonexgrad/lonex_exception.h>
#include <lonexgrad/lonex_scalar.h>
#include <lonexgrad/lonex_tensor.h>

#include <cstddef>
#include <random>
#include <vector>

#ifdef LONEXGRAD_TORCH
#include <torch/torch.h>
#undef CHECK
#endif

#include "doctest.h"
#include "test_util.h"

using namespace lonexgrad;

namespace {
template <typename T>
std::vector<T> rand_vec(int size, std::mt19937 &gen) {
    std::vector<T> res;
    res.reserve(static_cast<std::size_t>(size));
    std::uniform_real_distribution<float> dis(-5.0, 5.0);
    for (int i = 0; i < size; ++i) {
        res.push_back(static_cast<T>(dis(gen)));
    }
    return res;
}
}    

// NOLINTNEXTLINE
TEST_CASE("Unary clone autograd") {
    auto test = []<typename T>(Device device) {
        std::mt19937 gen(0);
        std::vector<T> d = rand_vec<T>(4 * 4, gen);
        std::vector<T> g = rand_vec<T>(4 * 4, gen);

        Tensor x1(d, {4, 4}, device, true);
        x1.register_hook([](Tensor &grad) { grad = grad * 2; });
        Tensor expected(g, {4, 4}, device);

        Tensor x2 = x1.clone();
        Tensor input_grad_x(g, {4, 4}, device, true);
        x2.backward(input_grad_x);

        auto close_options = CloseOptions().atol(1e-4).equal_nan();
        CHECK(allclose(x1.grad().value(), expected * 2, close_options));
    };
    runner_single_type<float>(test);
}
