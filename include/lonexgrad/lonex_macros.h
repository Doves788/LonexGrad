#ifndef lonexgrad_MACROS_H_



#define lonexgrad_MACROS_H_

namespace lonexgrad {

#if defined(__CUDACC__)
#define LONEXGRAD_DEVICE      __device__
#define LONEXGRAD_HOST        __host__
#define LONEXGRAD_HOST_DEVICE __host__ __device__
#define LONEXGRAD_INLINE      __forceinline__
#define LONEXGRAD_STD_FUNC    nvstd::function
#else
#define LONEXGRAD_DEVICE
#define LONEXGRAD_HOST
#define LONEXGRAD_HOST_DEVICE
#define LONEXGRAD_INLINE   inline
#define LONEXGRAD_STD_FUNC std::function
#endif

}    

#endif    
