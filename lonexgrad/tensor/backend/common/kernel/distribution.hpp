


#ifndef lonexgrad_BACKEND_COMMON_KERNEL_DISTRIBUTION_H_
#define lonexgrad_BACKEND_COMMON_KERNEL_DISTRIBUTION_H_

#include <lonexgrad/lonex_concepts.h>
#include <lonexgrad/lonex_macros.h>
#include <lonexgrad/lonex_random.h>

#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numbers>
#include <type_traits>

#if defined(__CUDACC__)
#include <nvfunctional>
#else
#include <functional>
#endif

namespace lonexgrad::common::kernel::distribution {

namespace {

template <typename T>
struct to_wide;

template <>
struct to_wide<uint32_t> {
    using type = uint64_t;
};
template <>
struct to_wide<uint64_t> {
    using type = __uint128_t;
};

template <typename T>
using to_wide_t = to_wide<T>::type;

template <typename T>
struct float_bits;

template <>
struct float_bits<float> {
    using unsigned_t = uint32_t;
    
    constexpr static unsigned_t EXP_MASK = static_cast<unsigned_t>(127) << 23;
    constexpr static unsigned_t SHIFT = 9;
};

template <>
struct float_bits<double> {
    using unsigned_t = uint64_t;
    
    constexpr static unsigned_t EXP_MASK = static_cast<unsigned_t>(1023) << 52;
    constexpr static unsigned_t SHIFT = 12;
};
}    



LONEXGRAD_HOST_DEVICE inline uint64_t rand_in_range(uint64_t s, Generator &gen) {
    using WideT = __uint128_t;
    static_assert(sizeof(uint64_t) <= 2 * sizeof(WideT));
    WideT m = static_cast<WideT>(gen()) * static_cast<WideT>(s);
    auto l = static_cast<uint64_t>(m);
    if (l < s) {
        uint64_t t = -s % s;    
        while (l < t) {
            m = static_cast<WideT>(gen()) * static_cast<WideT>(s);
            l = static_cast<uint64_t>(m);
        }
    }
    return static_cast<uint64_t>(m >> std::numeric_limits<uint64_t>::digits);
}










template <typename T>
    requires std::is_floating_point_v<T>
LONEXGRAD_HOST_DEVICE T uniform_real_sample(T low, T high, Generator &gen) {
    assert(low < high);
    using unsigned_t = float_bits<T>::unsigned_t;
    const auto SHIFT = float_bits<T>::SHIFT;
    const auto EXP_MASK = float_bits<T>::EXP_MASK;
    const auto r = std::bit_cast<T>((static_cast<unsigned_t>(gen()) >> SHIFT) | EXP_MASK) - static_cast<T>(1.0);
    return r * (high - low) + low;
}







template <typename T>
    requires std::is_floating_point_v<T>
struct OpUniformReal {
    LONEXGRAD_STD_FUNC<T(T, T, Generator)> LONEXGRAD_HOST_DEVICE operator()() const {
        return [](T low, T high, Generator gen) { return uniform_real_sample(low, high, gen); };
    }
};






template <typename T>
    requires std::is_integral_v<T> || std::is_floating_point_v<T>
struct OpUniformInt {
    LONEXGRAD_STD_FUNC<T(T, T, Generator)> LONEXGRAD_HOST_DEVICE operator()() const {
        return [](T low, T high, Generator gen) {
            const auto range = static_cast<uint64_t>(high - low);
            const auto x = static_cast<T>(rand_in_range(range, gen));
            return static_cast<T>(x + low);
        };
    }
};





template <typename T>
    requires std::is_floating_point_v<T>
struct OpBernoulli {
    LONEXGRAD_STD_FUNC<T(T, Generator)> LONEXGRAD_HOST_DEVICE operator()() const {
        return [](T p, Generator gen) {
            const auto u = uniform_real_sample(static_cast<T>(0), static_cast<T>(1), gen);
            return static_cast<T>(u < p);
        };
    }
};






template <typename T>
    requires std::is_floating_point_v<T>
struct OpBinomial {
    LONEXGRAD_STD_FUNC<T(T, T, Generator)> LONEXGRAD_HOST_DEVICE operator()() const {
        return [](T p, T num_draws, Generator gen) {
            T sum{};
            for ([[maybe_unused]] int i = 0; i < static_cast<int>(num_draws); ++i) {
                const auto u = uniform_real_sample(static_cast<T>(0), static_cast<T>(1), gen);
                sum += static_cast<T>(u < p);
            }
            return sum;
        };
    }
};






template <typename T>
    requires std::is_floating_point_v<T>
struct OpGeometric {
    LONEXGRAD_STD_FUNC<T(T, Generator)> LONEXGRAD_HOST_DEVICE operator()() const {
        return [](T p, Generator gen) {
            const auto u = uniform_real_sample(static_cast<T>(0), static_cast<T>(1), gen);
            return p == 1 ? 1 : std::ceil(std::log1p(-u) / std::log1p(-p));
        };
    }
};








template <typename T>
    requires std::is_floating_point_v<T>
struct OpPoisson {
    LONEXGRAD_STD_FUNC<T(T, Generator)> LONEXGRAD_HOST_DEVICE operator()() const {
        return [](T lambda, Generator gen) {
            T x{};
            auto p = std::exp(-lambda);
            auto s = p;
            const auto u = uniform_real_sample(static_cast<T>(0), static_cast<T>(1), gen);
            while (u > s) {
                ++x;
                p *= lambda / x;
                s += p;
            }
            return x;
        };
    }
};







template <typename T>
    requires std::is_floating_point_v<T>
struct OpExponential {
    LONEXGRAD_STD_FUNC<T(T, Generator)> LONEXGRAD_HOST_DEVICE operator()() const {
        return [](T lambda, Generator gen) {
            const auto u = uniform_real_sample(static_cast<T>(0), static_cast<T>(1), gen);
            return -std::log1p(-u) / lambda;
        };
    }
};







template <typename T>
    requires std::is_floating_point_v<T>
struct OpNormal {
    LONEXGRAD_STD_FUNC<T(T, T, Generator)> LONEXGRAD_HOST_DEVICE operator()() const {
        return [](T mu, T std, Generator gen) {
            const auto u1 = uniform_real_sample(static_cast<T>(0), static_cast<T>(1), gen);
            const auto u2 = uniform_real_sample(static_cast<T>(0), static_cast<T>(1), gen);
            const auto r =
                std::sqrt(static_cast<T>(-2) * std::log1p(-u1));    
            const auto theta = static_cast<T>(2) * std::numbers::pi_v<T> * u2;
            const auto z1 = r * std::sin(theta);
            return std * z1 + mu;
        };
    }
};






template <typename T>
    requires std::is_floating_point_v<T>
struct OpCauchy {
    LONEXGRAD_STD_FUNC<T(T, T, Generator)> LONEXGRAD_HOST_DEVICE operator()() const {
        return [](T loc, T scale, Generator gen) {
            const auto u = uniform_real_sample(static_cast<T>(0), static_cast<T>(1), gen);
            const auto z = std::tan(std::numbers::pi_v<T> * (u - 0.5));
            return loc + scale * z;
        };
    }
};






template <typename T>
    requires std::is_floating_point_v<T>
struct OpLogNormal {
    LONEXGRAD_STD_FUNC<T(T, T, Generator)> LONEXGRAD_HOST_DEVICE operator()() const {
        return [](T mu, T std, Generator gen) {
            auto OP = OpNormal<T>{};
            return std::exp(OP()(mu, std, gen));
        };
    }
};







template <typename T>
    requires std::is_floating_point_v<T>
struct OpWeibull {
    LONEXGRAD_STD_FUNC<T(T, T, Generator)> LONEXGRAD_HOST_DEVICE operator()() const {
        return [](T scale, T shape, Generator gen) {
            const auto u = uniform_real_sample(static_cast<T>(0), static_cast<T>(1), gen);
            const auto one = static_cast<T>(1);
            return scale * std::pow(std::log(one / (one - u)), one / shape);
        };
    }
};

}    

#endif    
