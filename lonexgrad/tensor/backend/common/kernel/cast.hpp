


#ifndef lonexgrad_BACKEND_COMMON_KERNEL_CAST_H_
#define lonexgrad_BACKEND_COMMON_KERNEL_CAST_H_

#include <lonexgrad/lonex_macros.h>
#include <lonexgrad/lonex_scalar.h>

#if defined(__CUDACC__)
#include <nvfunctional>
#else
#include <functional>
#endif

namespace lonexgrad::common::kernel::cast {


template <typename T, typename R>
struct OpCast {
    LONEXGRAD_STD_FUNC<R(T)> LONEXGRAD_HOST_DEVICE operator()() const {
        return [](T val) { return static_cast<R>(val); };
    }
};

}    

#endif    
