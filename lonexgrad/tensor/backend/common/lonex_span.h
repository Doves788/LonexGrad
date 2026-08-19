


#ifndef lonexgrad_BACKEND_COMMON_SPAN_H_
#define lonexgrad_BACKEND_COMMON_SPAN_H_

#include <lonexgrad/lonex_macros.h>
#include <lonexgrad/lonex_shape.h>

#include <cassert>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace lonexgrad {


template <typename T>
class NonOwningSpan {
public:
    LONEXGRAD_HOST_DEVICE NonOwningSpan() = default;
    LONEXGRAD_HOST_DEVICE explicit NonOwningSpan(T *p, std::size_t n)
        : _p(p), _n(n) {}

    [[nodiscard]] LONEXGRAD_HOST_DEVICE LONEXGRAD_INLINE auto operator[](std::size_t idx) const -> const T & {
        assert(idx < _n);
        assert(_p);
        return _p[idx];    // NOLINT(*-pointer-arithmetic)
    }
    [[nodiscard]] LONEXGRAD_HOST_DEVICE LONEXGRAD_INLINE auto operator[](std::size_t idx) -> T & {
        assert(idx < _n);
        assert(_p);
        return _p[idx];    // NOLINT(*-pointer-arithmetic)
    }
    [[nodiscard]] LONEXGRAD_HOST_DEVICE LONEXGRAD_INLINE auto operator[](int idx) const -> const T & {
        assert(idx >= 0 && static_cast<std::size_t>(idx) < _n);
        assert(_p);
        return _p[static_cast<std::size_t>(idx)];    // NOLINT(*-pointer-arithmetic)
    }
    [[nodiscard]] LONEXGRAD_HOST_DEVICE LONEXGRAD_INLINE auto operator[](int idx) -> T & {
        assert(idx >= 0 && static_cast<std::size_t>(idx) < _n);
        assert(_p);
        return _p[static_cast<std::size_t>(idx)];    // NOLINT(*-pointer-arithmetic)
    }

    [[nodiscard]] LONEXGRAD_HOST_DEVICE LONEXGRAD_INLINE auto get() const -> const T * {
        return _p;
    }

    [[nodiscard]] LONEXGRAD_HOST_DEVICE LONEXGRAD_INLINE auto get() -> T * {
        return _p;
    }

    [[nodiscard]] LONEXGRAD_HOST_DEVICE LONEXGRAD_INLINE auto size() const -> std::size_t {
        return _n;
    }

    
    template <typename U>
        requires std::is_convertible_v<U *, T *>
    LONEXGRAD_HOST_DEVICE NonOwningSpan(const NonOwningSpan<U> &dp)
        : _p(dp.get()), _n(dp.size()) {}

private:
    T *_p = nullptr;
    std::size_t _n = 0;
};


template <typename T>
struct HostSpan : public NonOwningSpan<T> {
    using NonOwningSpan<T>::NonOwningSpan;

    
    HostSpan(const std::vector<std::remove_cv_t<T>> &v)
        requires std::is_const_v<T>
        : NonOwningSpan<T>(v.data(), v.size()) {}
    HostSpan(std::vector<std::remove_cv_t<T>> &v)
        requires(!std::is_const_v<T>)
        : NonOwningSpan<T>(v.data(), v.size()) {}
    HostSpan(std::vector<std::remove_cv_t<T>> &&) = delete;

    
    HostSpan(const Shape &shape)
        : NonOwningSpan<T>(shape.to_vec().data(), static_cast<std::size_t>(shape.size())) {}
    HostSpan(Shape &&) = delete;
};

}    

#endif    
