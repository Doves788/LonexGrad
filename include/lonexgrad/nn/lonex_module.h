


#ifndef lonexgrad_NN_MODULE_H_
#define lonexgrad_NN_MODULE_H_

#include <lonexgrad/lonex_device.h>
#include <lonexgrad/lonex_exception.h>
#include <lonexgrad/export.h>
#include <lonexgrad/lonex_tensor.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace lonexgrad::nn {

class LONEXGRAD_EXPORT Module {
public:
    virtual ~Module() = default;
    Module() = default;

    
    Module(const Module &) = delete;
    Module(Module &&) = default;
    auto operator=(const Module &) -> Module & = delete;
    auto operator=(Module &&) -> Module & = delete;

    



    [[nodiscard]] virtual auto name() const -> std::string = 0;

    




    virtual void pretty_print(std::ostream &os) const;

    




    [[nodiscard]] auto parameters(bool recursive = true) const -> std::vector<Tensor>;

    







    [[nodiscard]] auto parameters_for_optimizer(bool recursive = true) const
        -> std::vector<std::reference_wrapper<Tensor>>;

    


    [[nodiscard]] auto num_params() const -> int64_t;

    




    [[nodiscard]] auto serialize() const -> std::vector<std::vector<char>>;

    



    void deserialize(const std::vector<std::vector<char>> &data);

    



    void save(const std::string &path) const;

    



    void load(const std::string &path);

    




    void to(Device device);

    


    void zero_grad();

    


    template <typename T>
    auto as() -> T * {
        return dynamic_cast<T *>(this);
    }

    



    template <typename T>
    auto as_checked() -> T & {
        T *p = dynamic_cast<T *>(this);
        if (p) {
            return *p;
        }
        LONEXGRAD_EXCEPTION("Cannot cast underlying module to given type");
    }

    




    void apply(const std::function<void(Module &)> &func, bool recursive = true);

    




    void register_param(std::shared_ptr<Tensor> param);

    




    void register_module(Module &module);

    




    void register_module(Module &module, const std::string &name);

    




    void train(bool is_train = true);

    



    void eval();

private:
    friend std::ostream &operator<<(std::ostream &os, Module &module);

    void pretty_print_recursive(std::ostream &os, const std::string &indentation) const;
    void get_params(std::vector<Tensor> &params, bool recursive) const;
    void get_params(std::vector<std::reference_wrapper<Tensor>> &params, bool recursive) const;

protected:
    std::vector<std::shared_ptr<Tensor>> params_;
    std::vector<std::pair<std::string, std::reference_wrapper<Module>>> modules_;
    bool is_train_ = true;
};

}    

#endif    
