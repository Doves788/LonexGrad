


#ifndef lonexgrad_CONCEPTS_H_
#define lonexgrad_CONCEPTS_H_

#include <concepts>
#include <type_traits>

namespace lonexgrad {

namespace detail {


template <typename T>
struct template_parameter;

template <template <typename...> class C, typename T>
struct template_parameter<C<T>> {
    using type = T;
};


template <typename T, template <typename...> class Z>
struct is_specialization_of : std::false_type {};

template <typename... Args, template <typename...> class Z>
struct is_specialization_of<Z<Args...>, Z> : std::true_type {};

template <typename T, template <typename...> class Z>
inline constexpr bool is_specialization_of_v = is_specialization_of<T, Z>::value;

}    




template <typename T, typename... U>
concept IsAnyOf = (std::same_as<T, U> || ...);




template <typename T, typename... U>
concept IsAllOf = (std::same_as<T, U> && ...);




template <typename T>
using template_parameter_t = typename detail::template_parameter<T>::type;




template <typename T, template <typename...> class Z>
concept IsSpecialization = detail::is_specialization_of_v<T, Z>;

}    

#endif    
