


#ifndef lonexgrad_UTIL_H_
#define lonexgrad_UTIL_H_

#include <concepts>
#include <ranges>

namespace lonexgrad {

namespace detail {


template <typename C>
struct to_helper {};


template <typename Container, std::ranges::range R>
    requires std::convertible_to<std::ranges::range_value_t<R>, typename Container::value_type>
Container operator|(R &&r, to_helper<Container>) {
    return Container{r.begin(), r.end()};
}
}    


template <std::ranges::range Container>
    requires(!std::ranges::view<Container>)
auto to() {
    return detail::to_helper<Container>{};
}


template <class... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};

template <typename T>
constexpr auto ceil_div(T x, T y) -> T {
    return (x + y - 1) / y;
}



[[noreturn]] inline void unreachable() {
    
    
    
#if defined(_MSC_VER) && !defined(__clang__)    
    __assume(false);
#else    
    __builtin_unreachable();
#endif
}

}    

#endif    
