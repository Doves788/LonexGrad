


#include "lonex_backend_register.h"

#include <lonexgrad/lonex_device.h>
#include <lonexgrad/lonex_exception.h>

#include "backend/cpu/lonex_backend_cpu.h"
#include "lonex_backend_base.h"
#ifdef LONEXGRAD_CUDA
#include "backend/cuda/lonex_backend_cuda.h"
#endif

#include <memory>

namespace lonexgrad {

BackendBase *get_backend(Backend backend) {
    switch (backend) {
    case Backend::cpu: {
        static const std::unique_ptr<cpu::BackendCPU> backend_cpu = std::make_unique<cpu::BackendCPU>();
        return backend_cpu.get();
    }
#ifdef LONEXGRAD_CUDA
    case Backend::cuda: {
        static const std::unique_ptr<cuda::BackendCUDA> backend_gpu = std::make_unique<cuda::BackendCUDA>();
        return backend_gpu.get();
    }
#endif
    default:
        LONEXGRAD_EXCEPTION("Unknown device type.");
    }
}

BackendBase *get_backend(Device device) {
    return get_backend(device.backend);
}

}    
