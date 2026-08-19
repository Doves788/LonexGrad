


#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <lonexgrad/lonex_grad_mode.h>
#include <lonexgrad/nn/lonex_linear.h>
#include <lonexgrad/nn/lonex_module.h>
#include <lonexgrad/lonex_tensor.h>

#include "doctest.h"
#include "test_util.h"

#include <cstddef>
#include <string>

using namespace lonexgrad;

// NOLINTNEXTLINE
TEST_CASE("save_load_tensor") {
    Tensor a = uniform_real(0, 1, {4, 3});
    save("tensor.pt", a);
    Tensor b = load("tensor.pt");
    CHECK(allclose(a, b));
}

class Head : public nn::Module {
public:
    Head()
        : layer1(3, 32), layer2(32, 16) {
        register_module(layer1);
        register_module(layer2);
    }

    auto clone() -> Head {
        Head result;

        return result;
    }

    [[nodiscard]] auto name() const -> std::string override {
        return "Head";
    }

    [[nodiscard]] auto forward(Tensor input) -> Tensor {
        Tensor result = layer1.forward(input);
        result = layer2.forward(result);
        return result;
    }

private:
    nn::Linear layer1;
    nn::Linear layer2;
};

class Tail : public nn::Module {
public:
    Tail()
        : layer1(16, 32), layer2(32, 4) {
        register_module(layer1);
        register_module(layer2);
    }

    [[nodiscard]] auto forward(Tensor input) -> Tensor {
        Tensor result = layer1.forward(input);
        result = layer2.forward(result);
        return result;
    }
    [[nodiscard]] auto name() const -> std::string override {
        return "Tail";
    }

private:
    nn::Linear layer1;
    nn::Linear layer2;
};

class Body : public nn::Module {
public:
    Body() {
        register_module(head, "Head");
        register_module(tail, "Tail");
    }

    [[nodiscard]] auto forward(Tensor input) -> Tensor {
        Tensor result = head.forward(input);
        result = tail.forward(result);
        return result;
    }

    [[nodiscard]] auto name() const -> std::string override {
        return "Body";
    }

private:
    Head head;
    Tail tail;
};

// NOLINTNEXTLINE
TEST_CASE("save_load_model") {
    Body model1;
    {
        auto params = model1.parameters();
        autograd::NoGradGuard guard;
        for (auto &p : params) {
            p.cos_();
        }
    }
    model1.save("model.pt");

    Body model2;
    
    {
        auto params1 = model1.parameters();
        auto params2 = model2.parameters();
        CHECK_EQ(params1.size(), params2.size());
        for (std::size_t i = 0; i < params1.size(); ++i) {
            CHECK_FALSE(allclose(params1[i], params2[i]));
        }
    }

    
    model2.load("model.pt");
    {
        auto params1 = model1.parameters();
        auto params2 = model2.parameters();
        CHECK_EQ(params1.size(), params2.size());
        for (std::size_t i = 0; i < params1.size(); ++i) {
            CHECK(allclose(params1[i], params2[i]));
        }
    }

    
    Body model3;
    model3.deserialize(model1.serialize());
    {
        auto params1 = model1.parameters();
        auto params3 = model3.parameters();
        CHECK_EQ(params1.size(), params3.size());
        for (std::size_t i = 0; i < params1.size(); ++i) {
            CHECK(allclose(params1[i], params3[i]));
        }
    }
}
