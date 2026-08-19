


#ifndef lonexgrad_BACKEND_MISC_KERNEL_UNARY_H_
#define lonexgrad_BACKEND_MISC_KERNEL_UNARY_H_

#include <lonexgrad/lonex_macros.h>

#if defined(__CUDACC__)
#include <nvfunctional>
#else
#include <functional>
#endif

namespace lonexgrad::common::kernel::misc {


template <typename T>
struct OpWhere {
    LONEXGRAD_STD_FUNC<T(bool, T, T)> LONEXGRAD_HOST_DEVICE operator()() const {
        return [](bool cond, auto lhs, auto rhs) -> T { return cond ? lhs : rhs; };
    }
};

}    

#endif    
