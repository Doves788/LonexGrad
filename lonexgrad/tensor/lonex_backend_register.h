


#ifndef lonexgrad_BACKEND_REGISTER_H_
#define lonexgrad_BACKEND_REGISTER_H_

#include <lonexgrad/lonex_device.h>

#include "lonex_backend_base.h"

namespace lonexgrad {

auto get_backend(Backend backend) -> BackendBase *;
auto get_backend(Device device) -> BackendBase *;

}    

#endif    
