


#ifndef lonexgrad_BACKEND_COMMON_DISPATCH_H_
#define lonexgrad_BACKEND_COMMON_DISPATCH_H_

#include <lonexgrad/lonex_exception.h>
#include <lonexgrad/lonex_scalar.h>

#include <sstream>

namespace lonexgrad {

// NOLINTNEXTLINE(*-macro-usage)
#define PRIVATE_DISPATCH_CASE(enum_type, ...)   \
    case enum_type: {                           \
        using scalar_t = to_ctype_t<enum_type>; \
        return __VA_ARGS__();                   \
    }

// NOLINTNEXTLINE(*-macro-usage)
#define DISPATCH_SWITCH(type, NAME, ...)                           \
    [&] {                                                          \
        switch (type) {                                            \
            __VA_ARGS__                                            \
        default:                                                   \
            std::stringstream _ss;                                 \
            _ss << "Unknown scalar type given to " << NAME << "."; \
            LONEXGRAD_ERROR(_ss.str());                                   \
        }                                                          \
    }()

// NOLINTNEXTLINE(*-macro-usage)
#define DISPATCH_CASE_FLOATING_TYPES(...)    \
    PRIVATE_DISPATCH_CASE(kF32, __VA_ARGS__) \
    PRIVATE_DISPATCH_CASE(kF64, __VA_ARGS__)

// NOLINTNEXTLINE(*-macro-usage)
#define DISPATCH_CASE_INTEGRAL_TYPES(...)    \
    PRIVATE_DISPATCH_CASE(kU8, __VA_ARGS__)  \
    PRIVATE_DISPATCH_CASE(kI16, __VA_ARGS__) \
    PRIVATE_DISPATCH_CASE(kI32, __VA_ARGS__) \
    PRIVATE_DISPATCH_CASE(kI64, __VA_ARGS__)

// NOLINTNEXTLINE(*-macro-usage)
#define DISPATCH_CASE_BOOL_TYPES(...) PRIVATE_DISPATCH_CASE(kBool, __VA_ARGS__)

// NOLINTNEXTLINE(*-macro-usage)
#define DISPATCH_CASE_ALL_TYPES(...)          \
    DISPATCH_CASE_INTEGRAL_TYPES(__VA_ARGS__) \
    DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__) \
    DISPATCH_CASE_BOOL_TYPES(__VA_ARGS__)

// NOLINTNEXTLINE(*-macro-usage)
#define DISPATCH_CASE_ALL_EXCEPT_BOOL_TYPES(...) \
    DISPATCH_CASE_INTEGRAL_TYPES(__VA_ARGS__)    \
    DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__)

// NOLINTBEGIN(*-macro-usage)
#define DISPATCH_INTEGRAL_TYPES(type, NAME, ...) DISPATCH_SWITCH(type, NAME, DISPATCH_CASE_INTEGRAL_TYPES(__VA_ARGS__))
#define DISPATCH_FLOATING_TYPES(type, NAME, ...) DISPATCH_SWITCH(type, NAME, DISPATCH_CASE_FLOATING_TYPES(__VA_ARGS__))
#define DISPATCH_ALL_TYPES(type, NAME, ...)      DISPATCH_SWITCH(type, NAME, DISPATCH_CASE_ALL_TYPES(__VA_ARGS__))
#define DISPATCH_ALL_EXCEPT_BOOL_TYPES(type, NAME, ...) \
    DISPATCH_SWITCH(type, NAME, DISPATCH_CASE_ALL_EXCEPT_BOOL_TYPES(__VA_ARGS__))
// NOLINTEND(*-macro-usage)

}    

#endif    
