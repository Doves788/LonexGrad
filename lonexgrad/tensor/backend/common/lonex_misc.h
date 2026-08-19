


#ifndef lonexgrad_BACKEND_COMMON_MISC_H_
#define lonexgrad_BACKEND_COMMON_MISC_H_

#include <lonexgrad/lonex_scalar.h>

#include "tensor/backend/common/kernel/misc.hpp"

namespace lonexgrad::common::misc {

using namespace kernel::misc;


enum class MiscOpT {
    where,
};


template <typename T, MiscOpT Op>
struct OpFactory;

// NOLINTNEXTLINE(*-macro-usage)
#define DECLARE_OP_FACTORY(MISC_OPT, KERN_OPT) \
    template <typename T>                      \
    struct OpFactory<T, MISC_OPT> {            \
        using KernelOp = KERN_OPT<T>;          \
    };

DECLARE_OP_FACTORY(MiscOpT::where, OpWhere);
#undef DECLARE_OP_FACTORY


template <typename T, MiscOpT Op>
struct Result {
    using type = T;
};


template <typename T, MiscOpT Op>
struct OpProperties;

// NOLINTNEXTLINE(*-macro-usage)
#define DECLARE_OP_PROPERTIES(MISC_OPT, SUPPORTED_CONCEPT)        \
    template <typename T>                                         \
    struct OpProperties<T, MISC_OPT> {                            \
        static constexpr bool IsSupported = SUPPORTED_CONCEPT<T>; \
    };

DECLARE_OP_PROPERTIES(MiscOpT::where, IsScalarType);
#undef DECLARE_OP_PROPERTIES

}    

#endif    
