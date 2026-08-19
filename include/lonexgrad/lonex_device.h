


#ifndef lonexgrad_DEVICE_H_
#define lonexgrad_DEVICE_H_

#include <lonexgrad/lonex_exception.h>
#include <lonexgrad/export.h>

#include <format>
#include <ostream>
#include <string>

namespace lonexgrad {


enum class LONEXGRAD_EXPORT Backend {
    cpu,
#ifdef LONEXGRAD_CUDA
    cuda,
#endif
};

LONEXGRAD_EXPORT constexpr auto to_string(Backend backend) -> std::string {
    switch (backend) {
    case Backend::cpu:
        return "cpu";
#ifdef LONEXGRAD_CUDA
    case Backend::cuda:
        return "cuda";
#endif
    }
    LONEXGRAD_EXCEPTION("Unknown device type.");
}


struct LONEXGRAD_EXPORT Device {
#ifdef LONEXGRAD_CUDA
    constexpr static auto CUDA(int dev_id) -> Device {
        return {.backend = Backend::cuda, .id = dev_id};
    }
#endif

    
    constexpr auto operator==(const Device &other) const -> bool {
        return backend == other.backend && id == other.id;
    }

    Backend backend;
    int id;
};


constexpr Device kCPU = Device{.backend = Backend::cpu, .id = 0};
#ifdef LONEXGRAD_CUDA
constexpr Device kCUDA = Device{.backend = Backend::cuda, .id = 0};
#endif

LONEXGRAD_EXPORT inline auto operator<<(std::ostream &os, const Device &device) -> std::ostream & {
    os << to_string(device.backend) << ":" << device.id;
    return os;
}
}    

template <>
struct LONEXGRAD_EXPORT std::formatter<lonexgrad::Device> : std::formatter<std::string> {
    auto format(const lonexgrad::Device &device, format_context &ctx) const {
        return formatter<string>::format(std::format("{}:{}", to_string(device.backend), device.id), ctx);
    }
};

#endif    
