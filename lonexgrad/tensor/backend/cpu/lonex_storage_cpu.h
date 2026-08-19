


#ifndef lonexgrad_STORAGE_CPU_H_
#define lonexgrad_STORAGE_CPU_H_

#include <lonexgrad/lonex_concepts.h>
#include <lonexgrad/lonex_scalar.h>

#include "tensor/lonex_storage_base.h"

#include <cstdint>
#include <type_traits>
#include <variant>
#include <vector>

namespace lonexgrad::cpu {


class StorageCPU : public StorageBase {
    using kBoolCType = to_ctype_t<kBool>;
    using kI16CType = to_ctype_t<kI16>;
    using kI32CType = to_ctype_t<kI32>;
    using kI64CType = to_ctype_t<kI64>;
    using kF32CType = to_ctype_t<kF32>;
    using kF64CType = to_ctype_t<kF64>;

public:
    using StorageT = std::variant<
        std::vector<kBoolCType>,
        std::vector<kI16CType>,
        std::vector<kI32CType>,
        std::vector<kI64CType>,
        std::vector<kF32CType>,
        std::vector<kF64CType>>;
    StorageCPU() = delete;

    ~StorageCPU() override {
        std::visit(
            [](auto &&tensor_storage) {
                using DT = std::remove_cvref_t<decltype(tensor_storage)>;    
                using T = template_parameter_t<DT>;                          
                current_bytes_allocated -= tensor_storage.size() * sizeof(T);
            },
            storage
        );
    }

    StorageCPU(const StorageCPU &) = delete;
    StorageCPU(StorageCPU &&) = default;
    auto operator=(const StorageCPU &) -> StorageCPU & = delete;
    auto operator=(StorageCPU &&) -> StorageCPU & = delete;

    
    template <IsScalarType T>
    StorageCPU(const std::vector<T> &data)
        : storage(data) {
        current_bytes_allocated += std::get<std::vector<T>>(storage).size() * sizeof(T);
        total_bytes_allocated += std::get<std::vector<T>>(storage).size() * sizeof(T);
    }

    
    template <IsScalarType T>
    StorageCPU(std::vector<T> &&data)
        : storage(std::move(data)) {
        current_bytes_allocated += std::get<std::vector<T>>(storage).size() * sizeof(T);
        total_bytes_allocated += std::get<std::vector<T>>(storage).size() * sizeof(T);
    }

    StorageT storage;

    inline static uint64_t current_bytes_allocated = 0;
    inline static uint64_t total_bytes_allocated = 0;
};

}    

#endif    
