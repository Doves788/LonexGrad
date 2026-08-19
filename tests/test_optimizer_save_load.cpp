


#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <lonexgrad/lonex_device.h>
#include <lonexgrad/lonex_exception.h>
#include <lonexgrad/nn/lonex_loss.h>
#include <lonexgrad/optim/lonex_adagrad.h>
#include <lonexgrad/optim/lonex_adam.h>
#include <lonexgrad/optim/lonex_adamw.h>
#include <lonexgrad/optim/lonex_rmsprop.h>
#include <lonexgrad/optim/lonex_sgd.h>
#include <lonexgrad/lonex_scalar.h>
#include <lonexgrad/lonex_tensor.h>

#include "doctest.h"
#include "test_util.h"

#include <cstddef>
#include <random>
#include <vector>

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
TEST_CASE("Optimizer Save Load SGD") {
    auto test = []<typename T>(Device device) {
        constexpr double lr = 0.1;
        constexpr double momentum = 0.2;
        std::mt19937 gen(0);
        std::vector<T> d_input = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_weight = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_target = rand_vec<T>(4 * 4 * 4, gen);

        Tensor input(d_input, {4, 4, 4}, device);
        Tensor weight(d_weight, {4, 4, 4}, device, true);
        Tensor target(d_target, {4, 4, 4}, device);
        optim::SGD sgd({weight}, lr, {.momentum = momentum, .use_nesterov = true});

        
        Tensor x = relu(input * weight);
        Tensor loss1 = nn::mse_loss(x, target, nn::ReductionMode::mean);
        loss1.backward();
        sgd.step();

        sgd.zero_grad();

        
        sgd.save("optimizer.pt");
        Tensor weight2 = weight.detach();
        weight2.set_requires_grad(true);

        
        x = relu(input * weight);
        Tensor loss2 = nn::mse_loss(x, target, nn::ReductionMode::mean);
        loss2.backward();
        sgd.step();

        
        optim::SGD sgd2({weight2}, lr, {.momentum = momentum, .use_nesterov = true});
        sgd2.load("optimizer.pt");

        
        x = relu(input * weight2);
        Tensor loss3 = nn::mse_loss(x, target, nn::ReductionMode::mean);
        loss3.backward();
        sgd2.step();

        CHECK(allclose(weight, weight2));
    };
    runner_single_type<double>(test);
}

// NOLINTNEXTLINE
TEST_CASE("Optimizer Save Load RMSprop") {
    auto test = []<typename T>(Device device) {
        constexpr double lr = 0.1;
        constexpr double momentum = 0.2;
        std::mt19937 gen(0);
        std::vector<T> d_input = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_weight = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_target = rand_vec<T>(4 * 4 * 4, gen);

        Tensor input(d_input, {4, 4, 4}, device);
        Tensor weight(d_weight, {4, 4, 4}, device, true);
        Tensor target(d_target, {4, 4, 4}, device);
        optim::RMSprop rmsprop({weight}, lr, {.momentum = momentum, .center = true});

        
        Tensor x = relu(input * weight);
        Tensor loss1 = nn::mse_loss(x, target, nn::ReductionMode::mean);
        loss1.backward();
        rmsprop.step();

        rmsprop.zero_grad();

        
        rmsprop.save("optimizer.pt");
        Tensor weight2 = weight.detach();
        weight2.set_requires_grad(true);

        
        x = relu(input * weight);
        Tensor loss2 = nn::mse_loss(x, target, nn::ReductionMode::mean);
        loss2.backward();
        rmsprop.step();

        
        optim::RMSprop rmsprop2({weight2}, lr, {.momentum = momentum, .center = true});
        rmsprop2.load("optimizer.pt");

        
        x = relu(input * weight2);
        Tensor loss3 = nn::mse_loss(x, target, nn::ReductionMode::mean);
        loss3.backward();
        rmsprop2.step();

        CHECK(allclose(weight, weight2));
    };
    runner_single_type<double>(test);
}

// NOLINTNEXTLINE
TEST_CASE("Optimizer Save Load Adam") {
    auto test = []<typename T>(Device device) {
        constexpr double lr = 0.1;
        std::mt19937 gen(0);
        std::vector<T> d_input = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_weight = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_target = rand_vec<T>(4 * 4 * 4, gen);

        Tensor input(d_input, {4, 4, 4}, device);
        Tensor weight(d_weight, {4, 4, 4}, device, true);
        Tensor target(d_target, {4, 4, 4}, device);
        optim::Adam adam({weight}, lr, {.weight_decay = 0.2, .use_amsgrad = true});

        
        Tensor x = relu(input * weight);
        Tensor loss1 = nn::mse_loss(x, target, nn::ReductionMode::mean);
        loss1.backward();
        adam.step();

        adam.zero_grad();

        
        adam.save("optimizer.pt");
        Tensor weight2 = weight.detach();
        weight2.set_requires_grad(true);

        
        x = relu(input * weight);
        Tensor loss2 = nn::mse_loss(x, target, nn::ReductionMode::mean);
        loss2.backward();
        adam.step();

        
        optim::Adam adam2({weight2}, lr, {.weight_decay = 0.2, .use_amsgrad = true});
        adam2.load("optimizer.pt");

        
        x = relu(input * weight2);
        Tensor loss3 = nn::mse_loss(x, target, nn::ReductionMode::mean);
        loss3.backward();
        adam2.step();

        CHECK(allclose(weight, weight2));
    };
    runner_single_type<double>(test);
}

// NOLINTNEXTLINE
TEST_CASE("Optimizer Save Load AdamW") {
    auto test = []<typename T>(Device device) {
        constexpr double lr = 0.1;
        std::mt19937 gen(0);
        std::vector<T> d_input = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_weight = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_target = rand_vec<T>(4 * 4 * 4, gen);

        Tensor input(d_input, {4, 4, 4}, device);
        Tensor weight(d_weight, {4, 4, 4}, device, true);
        Tensor target(d_target, {4, 4, 4}, device);
        optim::AdamW adamw({weight}, lr, {.weight_decay = 0.2, .use_amsgrad = true});

        
        Tensor x = relu(input * weight);
        Tensor loss1 = nn::mse_loss(x, target, nn::ReductionMode::mean);
        loss1.backward();
        adamw.step();

        adamw.zero_grad();

        
        adamw.save("optimizer.pt");
        Tensor weight2 = weight.detach();
        weight2.set_requires_grad(true);

        
        x = relu(input * weight);
        Tensor loss2 = nn::mse_loss(x, target, nn::ReductionMode::mean);
        loss2.backward();
        adamw.step();

        
        optim::AdamW adamw2({weight2}, lr, {.weight_decay = 0.2, .use_amsgrad = true});
        adamw2.load("optimizer.pt");

        
        x = relu(input * weight2);
        Tensor loss3 = nn::mse_loss(x, target, nn::ReductionMode::mean);
        loss3.backward();
        adamw2.step();

        CHECK(allclose(weight, weight2));
    };
    runner_single_type<double>(test);
}

// NOLINTNEXTLINE
TEST_CASE("Optimizer Save Load Adagrad") {
    auto test = []<typename T>(Device device) {
        constexpr double lr = 0.1;
        std::mt19937 gen(0);
        std::vector<T> d_input = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_weight = rand_vec<T>(4 * 4 * 4, gen);
        std::vector<T> d_target = rand_vec<T>(4 * 4 * 4, gen);

        Tensor input(d_input, {4, 4, 4}, device);
        Tensor weight(d_weight, {4, 4, 4}, device, true);
        Tensor target(d_target, {4, 4, 4}, device);
        optim::Adagrad adagrad({weight}, lr, {.weight_decay = 0.2});

        
        Tensor x = relu(input * weight);
        Tensor loss1 = nn::mse_loss(x, target, nn::ReductionMode::mean);
        loss1.backward();
        adagrad.step();

        adagrad.zero_grad();

        
        adagrad.save("optimizer.pt");
        Tensor weight2 = weight.detach();
        weight2.set_requires_grad(true);

        
        x = relu(input * weight);
        Tensor loss2 = nn::mse_loss(x, target, nn::ReductionMode::mean);
        loss2.backward();
        adagrad.step();

        
        optim::Adagrad adagrad2({weight2}, lr, {.weight_decay = 0.2});
        adagrad2.load("optimizer.pt");

        
        x = relu(input * weight2);
        Tensor loss3 = nn::mse_loss(x, target, nn::ReductionMode::mean);
        loss3.backward();
        adagrad2.step();

        CHECK(allclose(weight, weight2));
    };
    runner_single_type<double>(test);
}
