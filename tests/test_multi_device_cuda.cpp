


#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <lonexgrad/lonex_device.h>
#include <lonexgrad/lonex_exception.h>
#include <lonexgrad/lonex_tensor.h>

#include "doctest.h"
#include "test_util.h"

#include <tuple>

using namespace lonexgrad;

#ifdef LONEXGRAD_CUDA
// NOLINTNEXTLINE
TEST_CASE("Multi device cuda") {
    if (get_device_count(Backend::cuda) > 1) {
        
        {
            Tensor t1 = ones({4, 4}, TensorOptions().device({.backend = Backend::cuda, .id = 0}));
            Tensor t2 = ones({4, 4}, TensorOptions().device({.backend = Backend::cuda, .id = 0}));
            Tensor result = t1 + t2;
            CHECK(allclose(2 * t1, result));
        }
        
        {
            Tensor t1 = ones({4, 4}, TensorOptions().device({.backend = Backend::cuda, .id = 1}));
            Tensor t2 = ones({4, 4}, TensorOptions().device({.backend = Backend::cuda, .id = 1}));
            Tensor result = t1 + t2;
            CHECK(allclose(2 * t1, result));
        }
        
        {
            Tensor t1 = ones({4, 4}, TensorOptions().device({.backend = Backend::cuda, .id = 0}));
            Tensor t2 = ones({4, 4}, TensorOptions().device({.backend = Backend::cuda, .id = 1}));
            CHECK_THROWS_AS(std::ignore = t1 + t2, TTException);
        }
        
        {
            const Device cuda0{.backend = Backend::cuda, .id = 0};
            const Device cuda1{.backend = Backend::cuda, .id = 1};
            Tensor t1 = ones({4, 4}, TensorOptions().device(cuda0));
            Tensor t2 = ones({4, 4}, TensorOptions().device(cuda1));
            Tensor t3 = t1.to(cuda1);
            Tensor result = t3 + t2;
            CHECK(allclose(2 * t2, result));
        }
    }
}
#endif
