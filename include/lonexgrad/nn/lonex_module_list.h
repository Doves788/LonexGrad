


#ifndef lonexgrad_NN_MODULE_LIST_H_
#define lonexgrad_NN_MODULE_LIST_H_

#include <lonexgrad/lonex_device.h>
#include <lonexgrad/export.h>
#include <lonexgrad/nn/lonex_module.h>
#include <lonexgrad/lonex_tensor.h>

#include <concepts>
#include <memory>
#include <string>
#include <type_traits>

namespace lonexgrad::nn {

class LONEXGRAD_EXPORT ModuleList : public Module {
private:
    CheckedVec<std::shared_ptr<nn::Module>> modules;
    using Iterator = decltype(modules)::Iterator;
    using ConstIterator = decltype(modules)::ConstIterator;

public:
    template <typename M>
        requires(std::derived_from<M, nn::Module> && !std::is_lvalue_reference_v<M>)
    void push_back(M &&module) {
        using T = std::remove_reference_t<M>;
        modules.push_back(std::make_shared<T>(std::forward<M>(module)));
        register_module(*modules[-1]);
    }

    auto operator[](int idx) -> nn::Module & {
        return *modules[idx];
    }

    auto operator[](int idx) const -> const nn::Module & {
        return *modules[idx];
    }

    [[nodiscard]] auto name() const -> std::string override {
        return "ModuleList";
    }

    [[nodiscard]] auto begin() -> Iterator {
        return modules.begin();
    }
    [[nodiscard]] auto begin() const -> ConstIterator {
        return modules.begin();
    }
    [[nodiscard]] auto end() -> Iterator {
        return modules.end();
    }
    [[nodiscard]] auto end() const -> ConstIterator {
        return modules.end();
    }
};

}    

#endif    
