


#ifndef lonexgrad_BACKEND_COMMON_KERNEL_BINARY_H_
#define lonexgrad_BACKEND_COMMON_KERNEL_BINARY_H_

#include <lonexgrad/lonex_macros.h>
#include <lonexgrad/lonex_scalar.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#if defined(__CUDACC__)
#include <nvfunctional>
#else
#include <functional>
#endif

namespace lonexgrad::common::kernel::binary {

template <typename T>
struct OpAdd {
    LONEXGRAD_STD_FUNC<T(T, T)> LONEXGRAD_HOST_DEVICE operator()() const {
        return [](auto lhs, auto rhs) { return lhs + rhs; };
    }
};

template <typename T>
struct OpSub {
    LONEXGRAD_STD_FUNC<T(T, T)> LONEXGRAD_HOST_DEVICE operator()() const {
        return [](auto lhs, auto rhs) { return lhs - rhs; };
    }
};

template <typename T>
struct OpMul {
    LONEXGRAD_STD_FUNC<T(T, T)> LONEXGRAD_HOST_DEVICE operator()() const {
        return [](auto lhs, auto rhs) { return lhs * rhs; };
    }
};

template <typename T>
struct OpDiv {
    LONEXGRAD_STD_FUNC<T(T, T)> LONEXGRAD_HOST_DEVICE operator()() const {
        return [](auto lhs, auto rhs) { return lhs / rhs; };
    }
};

template <typename T>
struct OpMinimum {
    static constexpr T padding_value = std::numeric_limits<T>::max();
    LONEXGRAD_STD_FUNC<T(T, T)> LONEXGRAD_HOST_DEVICE operator()() const {
        return [](auto lhs, auto rhs) { return (lhs < rhs) ? lhs : rhs; };
    }
};

template <typename T>
struct OpMaximum {
    static constexpr T padding_value = std::numeric_limits<T>::lowest();
    LONEXGRAD_STD_FUNC<T(T, T)> LONEXGRAD_HOST_DEVICE operator()() const {
        return [](auto lhs, auto rhs) { return (lhs > rhs) ? lhs : rhs; };
    }
};

template <typename T>
struct OpPow {
    LONEXGRAD_STD_FUNC<T(T, T)> LONEXGRAD_HOST_DEVICE operator()() const {
        return [](auto lhs, auto rhs) { return std::pow(lhs, rhs); };
    }
};

template <typename T>
struct OpEq {
    LONEXGRAD_STD_FUNC<bool(T, T)> LONEXGRAD_HOST_DEVICE operator()() const {
        return [](auto lhs, auto rhs) { return lhs == rhs; };
    }
};




template <typename T>
struct OpKahanSum {
    static constexpr T padding_value = 0;
    LONEXGRAD_STD_FUNC<T(T, T)> LONEXGRAD_HOST_DEVICE operator()() {
        return [this](auto lhs, auto rhs) {
            T rhs_with_compensation = rhs - c;
            T sum = static_cast<T>(lhs + rhs_with_compensation);    
            c = static_cast<T>((sum - lhs) - rhs_with_compensation);
            return sum;
        };
    }
    T c = padding_value;
};

template <typename T>
struct OpNe {
    LONEXGRAD_STD_FUNC<bool(T, T)> LONEXGRAD_HOST_DEVICE operator()() const {
        return [](auto lhs, auto rhs) { return lhs != rhs; };
    }
};

template <typename T>
struct OpLt {
    LONEXGRAD_STD_FUNC<bool(T, T)> LONEXGRAD_HOST_DEVICE operator()() const {
        return [](auto lhs, auto rhs) { return lhs < rhs; };
    }
};

template <typename T>
struct OpLe {
    LONEXGRAD_STD_FUNC<bool(T, T)> LONEXGRAD_HOST_DEVICE operator()() const {
        return [](auto lhs, auto rhs) { return lhs <= rhs; };
    }
};

template <typename T>
struct OpGt {
    LONEXGRAD_STD_FUNC<bool(T, T)> LONEXGRAD_HOST_DEVICE operator()() const {
        return [](auto lhs, auto rhs) { return lhs > rhs; };
    }
};

template <typename T>
struct OpGe {
    LONEXGRAD_STD_FUNC<bool(T, T)> LONEXGRAD_HOST_DEVICE operator()() const {
        return [](auto lhs, auto rhs) { return lhs >= rhs; };
    }
};

template <typename T>
struct OpLogicalOr {
    LONEXGRAD_STD_FUNC<bool(bool, bool)> LONEXGRAD_HOST_DEVICE operator()() const {
        return [](auto lhs, auto rhs) { return lhs || rhs; };
    }
};

template <typename T>
struct OpLogicalAnd {
    LONEXGRAD_STD_FUNC<bool(bool, bool)> LONEXGRAD_HOST_DEVICE operator()() const {
        return [](auto lhs, auto rhs) { return lhs && rhs; };
    }
};



template <typename T>
struct OpBitwiseOr {
    LONEXGRAD_STD_FUNC<T(T, T)> LONEXGRAD_HOST_DEVICE operator()() const {
        using U = std::conditional_t<IsScalarIntType<T>, T, uint64_t>;
        return [](auto lhs, auto rhs) { return static_cast<U>(lhs) | static_cast<U>(rhs); };
    }
};

template <typename T>
struct OpBitwiseAnd {
    LONEXGRAD_STD_FUNC<T(T, T)> LONEXGRAD_HOST_DEVICE operator()() const {
        using U = std::conditional_t<IsScalarIntType<T>, T, uint64_t>;
        return [](auto lhs, auto rhs) { return static_cast<U>(lhs) & static_cast<U>(rhs); };
    }
};

template <typename T>
struct OpBitwiseXor {
    LONEXGRAD_STD_FUNC<T(T, T)> LONEXGRAD_HOST_DEVICE operator()() const {
        using U = std::conditional_t<IsScalarIntType<T>, T, uint64_t>;
        return [](auto lhs, auto rhs) { return static_cast<U>(lhs) ^ static_cast<U>(rhs); };
    }
};

template <typename T>
struct OpBitwiseLeftShift {
    LONEXGRAD_STD_FUNC<T(T, T)> LONEXGRAD_HOST_DEVICE operator()() const {
        using U = std::conditional_t<IsScalarIntType<T>, T, uint64_t>;
        return [](auto lhs, auto rhs) { return static_cast<U>(lhs) << static_cast<U>(rhs); };
    }
};

template <typename T>
struct OpBitwiseRightShift {
    LONEXGRAD_STD_FUNC<T(T, T)> LONEXGRAD_HOST_DEVICE operator()() const {
        using U = std::conditional_t<IsScalarIntType<T>, T, uint64_t>;
        return [](auto lhs, auto rhs) { return static_cast<U>(lhs) >> static_cast<U>(rhs); };
    }
};

template <typename T>
struct OpModulo {
    LONEXGRAD_STD_FUNC<T(T, T)> LONEXGRAD_HOST_DEVICE operator()() const {
        return [](auto lhs, auto rhs) {
            if constexpr (IsScalarIntType<T>) {
                return lhs % rhs;
            } else {
                return std::fmod(lhs, rhs);
            }
        };
    }
};

}    

#endif    
