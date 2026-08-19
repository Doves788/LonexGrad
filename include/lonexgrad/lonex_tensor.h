


#ifndef lonexgrad_TENSOR_H_
#define lonexgrad_TENSOR_H_

#include <lonexgrad/lonex_device.h>
#include <lonexgrad/lonex_exception.h>
#include <lonexgrad/export.h>
#include <lonexgrad/lonex_index.h>
#include <lonexgrad/lonex_random.h>
#include <lonexgrad/lonex_scalar.h>
#include <lonexgrad/lonex_shape.h>

#include <cstddef>
#include <cstdint>
#include <format>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace lonexgrad {


class Tensor;
class StorageBase;
template <typename T>
class CheckedVec;

using TensorList = CheckedVec<Tensor>;

namespace autograd {
struct SharedGrad;

template <typename T>
struct TensorFunction;


using GradHook = std::function<void(Tensor &grad)>;

auto build_dag(const Tensor &tensor) -> TensorList;
void backward(Tensor &tensor, const Tensor &grad, bool retain_graph);
void calc_grad_input(Tensor &tensor, bool retain_graph);
void add_grad(Tensor &tensor, const Tensor &grad);
}    



class LONEXGRAD_EXPORT TensorOptions {
public:
    TensorOptions() = default;

    
    explicit TensorOptions(Device device)
        : device_(device) {}
    explicit TensorOptions(ScalarType dtype)
        : dtype_(dtype) {}
    explicit TensorOptions(bool requires_grad)
        : requires_grad_(requires_grad) {}

    auto dtype(ScalarType dtype) -> TensorOptions & {
        dtype_ = dtype;
        return *this;
    }
    [[nodiscard]] auto dtype() const -> ScalarType {
        return dtype_;
    }
    auto device(Device device) -> TensorOptions & {
        device_ = device;
        return *this;
    }
    [[nodiscard]] auto device() const -> Device {
        return device_;
    }
    auto requires_grad(bool requires_grad) -> TensorOptions & {
        requires_grad_ = requires_grad;
        return *this;
    }
    [[nodiscard]] auto requires_grad() const -> bool {
        return requires_grad_;
    }

private:
    ScalarType dtype_ = kF32;
    Device device_ = kCPU;
    bool requires_grad_ = false;
};


class LONEXGRAD_EXPORT ClampOptions {
public:
    template <IsScalarType T>
    auto min(T min_value) -> ClampOptions & {
        _min = Scalar(min_value);
        return *this;
    }
    template <IsScalarType T>
    auto max(T max_value) -> ClampOptions & {
        _max = Scalar(max_value);
        return *this;
    }
    [[nodiscard]] inline auto min() const -> std::optional<Scalar> {
        return _min;
    }
    [[nodiscard]] inline auto max() const -> std::optional<Scalar> {
        return _max;
    }
    [[nodiscard]] auto min_to(ScalarType dtype) const -> std::optional<Scalar>;
    [[nodiscard]] auto max_to(ScalarType dtype) const -> std::optional<Scalar>;

private:
    std::optional<Scalar> _min;
    std::optional<Scalar> _max;
};


class LONEXGRAD_EXPORT Tensor {
public:
    






    template <typename T>
        requires(IsScalarType<T> || std::is_same_v<T, bool>)
    Tensor(const std::vector<T> &data, Shape shape, Device device, bool requires_grad = false);

    






    template <typename T>
        requires(IsScalarType<T> || std::is_same_v<T, bool>)
    Tensor(std::vector<T> &&data, Shape shape, Device device, bool requires_grad = false);

    






    template <typename T>
        requires(IsScalarType<T> || std::is_same_v<T, bool>)
    Tensor(const std::initializer_list<T> &data, Shape shape, Device device, bool requires_grad = false)
        : Tensor(std::vector<T>(data), shape, device, requires_grad) {}

    





    template <typename T>
        requires(IsScalarType<T> || std::is_same_v<T, bool>)
    Tensor(const std::vector<T> &data, Device device, bool requires_grad = false)
        : Tensor(data, {static_cast<int>(data.size())}, device, requires_grad) {}

    





    template <typename T>
        requires(IsScalarType<T> || std::is_same_v<T, bool>)
    Tensor(std::vector<T> &&data, Device device, bool requires_grad = false);

    





    template <typename T>
        requires(IsScalarType<T> || std::is_same_v<T, bool>)
    Tensor(const std::initializer_list<T> &data, Device device, bool requires_grad = false)
        : Tensor(std::vector<T>(data), device, requires_grad) {}

    




    Tensor(Scalar scalar, Device device, bool requires_grad = false);

    
    Tensor(std::shared_ptr<StorageBase> storage, ScalarType dtype, Shape shape, Device device);

    





    void set_from(const Tensor &other);

    ~Tensor() = default;

    


    auto serialize() const -> std::vector<char>;

    







    void deserialize(const std::vector<char> &serialized_data);

    



    [[nodiscard]] auto has_storage() const -> bool;

    


    [[nodiscard]] auto dtype() const -> ScalarType;

    


    [[nodiscard]] auto device() const -> Device;

    


    [[nodiscard]] auto offset() const -> int;

    


    [[nodiscard]] auto shape() const -> const Shape &;

    


    [[nodiscard]] auto stride() const -> const Shape &;

    


    [[nodiscard]] auto numel() const -> int;

    


    [[nodiscard]] auto dim() const -> int;

    



    [[nodiscard]] auto size(int dim) const -> int;

    


    [[nodiscard]] auto is_contiguous() const -> bool;

    




    [[nodiscard]] auto contiguous() const -> Tensor;

    



    template <typename T>
    [[nodiscard]] auto get_storage() const -> T & {
        return *static_cast<T *>(storage_.get());
    }

    


    [[nodiscard]] auto item() const -> Scalar;

    


    template <typename T>
        requires(IsScalarType<T> || std::is_same_v<T, bool>)
    [[nodiscard]] auto item() const -> T {
        return item().to<T>();
    }

    


    template <ScalarType T>
    [[nodiscard]] auto item() const -> to_ctype_t<T> {
        return item().to<std::conditional_t<T == kBool, bool, to_ctype_t<T>>>();
    }

    







    [[nodiscard]] auto to(ScalarType dtype) const -> Tensor;

    





    [[nodiscard]] auto to(Device device) const -> Tensor;

    




    [[nodiscard]] auto clone() const -> Tensor;

    




    [[nodiscard]] auto detach() const -> Tensor;

    



    [[nodiscard]] auto data_ptr() const -> uintptr_t;

    






    [[nodiscard]] auto version_count() const -> int;

    




    template <typename T>
        requires(IsScalarType<T> || std::is_same_v<T, bool>)
    [[nodiscard]] auto to_vec() const -> std::vector<T>;

    
    Tensor(const Tensor &rhs) = default;
    Tensor(Tensor &&rhs) = default;

    

    


    template <typename T>
        requires(IsScalarType<T> || std::is_same_v<T, bool>)
    auto operator=(T rhs) -> Tensor & {
        this->operator=(Scalar(rhs, scalar_type_));
        return *this;
    }

    


    template <typename T>
        requires(IsScalarType<T> || std::is_same_v<T, bool>)
    auto fill_(T rhs) -> Tensor & {
        this->operator=(Scalar(rhs, scalar_type_));
        return *this;
    }

    




    auto operator=(const Scalar &rhs) -> Tensor &;

    




    auto fill_(const Scalar &rhs) -> Tensor &;

    


    auto operator=(const Tensor &rhs) & -> Tensor &;
    auto operator=(Tensor &&rhs) & -> Tensor &;

    






    auto operator=(const Tensor &rhs) && -> Tensor &;
    auto operator=(Tensor &&rhs) && -> Tensor &;

    





    template <IsScalarType T>
    auto operator+=(T rhs) -> Tensor & {
        return this->operator+=(Scalar(rhs, scalar_type_));
    }
    auto operator+=(const Scalar &rhs) -> Tensor &;
    auto operator+=(const Tensor &rhs) -> Tensor &;

    template <IsScalarType T>
    auto operator-=(T rhs) -> Tensor & {
        return this->operator-=(Scalar(rhs, scalar_type_));
    }
    auto operator-=(const Scalar &rhs) -> Tensor &;
    auto operator-=(const Tensor &rhs) -> Tensor &;

    template <IsScalarType T>
    auto operator*=(T rhs) -> Tensor & {
        return this->operator*=(Scalar(rhs, scalar_type_));
    }
    auto operator*=(const Scalar &rhs) -> Tensor &;
    auto operator*=(const Tensor &rhs) -> Tensor &;

    template <IsScalarType T>
    auto operator/=(T rhs) -> Tensor & {
        return this->operator/=(Scalar(rhs, scalar_type_));
    }
    auto operator/=(const Scalar &rhs) -> Tensor &;
    auto operator/=(const Tensor &rhs) -> Tensor &;

    






    [[nodiscard]] auto operator[](const Tensor &mask) const -> Tensor;

    








    [[nodiscard]] auto operator[](const indexing::Index &index) const -> Tensor;

    








    [[nodiscard]] auto operator[](const std::vector<indexing::Index> &indices) const -> Tensor;

    
    
    

    class Iterator {
    public:
        Iterator(Tensor &tensor, int idx)
            : tensor_(tensor), idx_(idx) {}

        auto operator!=(const Iterator &other) const -> bool;
        auto operator++() -> Iterator;
        auto operator*() const -> Tensor;

    private:
        Tensor &tensor_;    // NOLINT(*-ref-data-members)
        int idx_;
    };

    auto begin() -> Iterator;
    auto end() -> Iterator;

    
    
    

    



    [[nodiscard]] auto requires_grad() const -> bool;

    




    void set_requires_grad(bool set_grad);

    



    [[nodiscard]] auto grad() const -> const std::optional<Tensor> &;

    


    void clear_grad();

    



    void add_grad(const Tensor &grad);

    



    void register_hook(const autograd::GradHook &hook);

    






    void backward(const std::optional<Tensor> &grad = {}, bool retain_graph = false);

    





    [[nodiscard]] auto is_leaf() const -> bool;

    
    
    

    











    [[nodiscard]] auto broadcast_to(const Shape &shape) const -> Tensor;

    










    [[nodiscard]] auto expand(const Shape &shape) const -> Tensor;

    







    [[nodiscard]] auto squeeze(int dim) const -> Tensor;

    





    [[nodiscard]] auto unsqueeze(int dim) const -> Tensor;

    







    [[nodiscard]] auto reshape(const Shape &shape) const -> Tensor;

    









    [[nodiscard]] auto flatten(int start_dim = 0, int end_dim = -1) const -> Tensor;

    







    [[nodiscard]] auto permute(const std::vector<int> &dims) const -> Tensor;

    
    
    

    



    auto abs_() -> Tensor &;

    



    auto negate_() -> Tensor &;

    





    auto logical_not_() -> Tensor &;

    




    auto sign_() -> Tensor &;

    


    auto log_() -> Tensor &;

    



    auto log10_() -> Tensor &;

    



    auto log2_() -> Tensor &;

    



    auto log1p_() -> Tensor &;

    



    auto exp_() -> Tensor &;

    



    auto exp2_() -> Tensor &;

    



    auto expm1_() -> Tensor &;

    



    auto sqrt_() -> Tensor &;

    



    auto sin_() -> Tensor &;

    



    auto cos_() -> Tensor &;

    



    auto tan_() -> Tensor &;

    



    auto asin_() -> Tensor &;

    



    auto acos_() -> Tensor &;

    



    auto atan_() -> Tensor &;

    



    auto sinh_() -> Tensor &;

    



    auto cosh_() -> Tensor &;

    



    auto tanh_() -> Tensor &;

    



    auto asinh_() -> Tensor &;

    



    auto acosh_() -> Tensor &;

    



    auto atanh_() -> Tensor &;

    




    auto erf_() -> Tensor &;

    




    auto erfc_() -> Tensor &;

    




    auto tgamma_() -> Tensor &;

    




    auto lgamma_() -> Tensor &;

    






    auto digamma_() -> Tensor &;

    




    auto ceil_() -> Tensor &;

    




    auto floor_() -> Tensor &;

    





    auto round_() -> Tensor &;

    
    
    

    




    auto sigmoid_() -> Tensor &;

    




    auto log_sigmoid_() -> Tensor &;

    




    auto hardsigmoid_() -> Tensor &;

    








    auto softplus_(double beta = 1, double threshold = 20) -> Tensor &;

    




    auto relu_() -> Tensor &;

    




    auto relu6_() -> Tensor &;

    






    auto leaky_relu_(double negative_slope = 0.01) -> Tensor &;

    





    auto elu_(double alpha = 1) -> Tensor &;

    




    auto selu_() -> Tensor &;

    




    auto silu_() -> Tensor &;

    






    auto hardtanh_(double min = -1, double max = 1) -> Tensor &;

    




    auto softsign_() -> Tensor &;

    







    auto softmax_(int dim) -> Tensor &;

    





    auto log_softmax_(int dim) -> Tensor &;

    
    
    

    




    auto add_(const Tensor &rhs) -> Tensor &;

    




    template <IsScalarType T>
    auto add_(T rhs) -> Tensor & {
        return add_(Scalar(rhs, scalar_type_));
    }

    




    auto add_(const Scalar &rhs) -> Tensor &;

    




    auto sub_(const Tensor &rhs) -> Tensor &;

    




    template <IsScalarType T>
    auto sub_(T rhs) -> Tensor & {
        return sub_(Scalar(rhs, scalar_type_));
    }

    




    auto sub_(const Scalar &rhs) -> Tensor &;

    




    auto mul_(const Tensor &rhs) -> Tensor &;

    




    template <IsScalarType T>
    auto mul_(T rhs) -> Tensor & {
        return mul_(Scalar(rhs, scalar_type_));
    }

    




    auto mul_(const Scalar &rhs) -> Tensor &;

    




    auto div_(const Tensor &rhs) -> Tensor &;

    




    template <IsScalarType T>
    auto div_(T rhs) -> Tensor & {
        return div_(Scalar(rhs, scalar_type_));
    }

    




    auto div_(const Scalar &rhs) -> Tensor &;

    
    
    

    








    auto uniform_real_(double low, double high, Generator &gen = get_default_generator()) -> Tensor &;

    









    auto uniform_real_(const Tensor &low, const Tensor &high, Generator &gen = get_default_generator()) -> Tensor &;

    








    auto uniform_int_(int64_t low, int64_t high, Generator &gen = get_default_generator()) -> Tensor &;

    








    auto uniform_int_(const Tensor &low, const Tensor &high, Generator &gen = get_default_generator()) -> Tensor &;

    






    auto bernoulli_(double p, Generator &gen = get_default_generator()) -> Tensor &;

    






    auto bernoulli_(const Tensor &p, Generator &gen = get_default_generator()) -> Tensor &;

    






    auto binomial_(double p, int num_draws, Generator &gen = get_default_generator()) -> Tensor &;

    






    auto binomial_(const Tensor &p, const Tensor &num_draws, Generator &gen = get_default_generator()) -> Tensor &;

    







    auto geometric_(double p, Generator &gen = get_default_generator()) -> Tensor &;

    







    auto geometric_(const Tensor &p, Generator &gen = get_default_generator()) -> Tensor &;

    




    auto poisson_(double lambda, Generator &gen = get_default_generator()) -> Tensor &;

    




    auto poisson_(const Tensor &lambda, Generator &gen = get_default_generator()) -> Tensor &;

    




    auto exponential_(double lambda, Generator &gen = get_default_generator()) -> Tensor &;

    




    auto exponential_(const Tensor &lambda, Generator &gen = get_default_generator()) -> Tensor &;

    





    auto normal_(double mu, double std, Generator &gen = get_default_generator()) -> Tensor &;

    





    auto normal_(const Tensor &mu, const Tensor &std, Generator &gen = get_default_generator()) -> Tensor &;

    







    auto cauchy_(double loc, double scale, Generator &gen = get_default_generator()) -> Tensor &;

    









    auto cauchy_(const Tensor &loc, const Tensor &scale, Generator &gen = get_default_generator()) -> Tensor &;

    






    auto lognormal_(double mu, double std, Generator &gen = get_default_generator()) -> Tensor &;

    






    auto lognormal_(const Tensor &mu, const Tensor &std, Generator &gen = get_default_generator()) -> Tensor &;

    






    auto weibull_(double lambda, double k, Generator &gen = get_default_generator()) -> Tensor &;

    






    auto weibull_(const Tensor &lambda, const Tensor &k, Generator &gen = get_default_generator()) -> Tensor &;

    
    
    

    




    [[nodiscard]] auto min() const -> Tensor;

    






    [[nodiscard]] auto min(int dim, bool keep_dim = false) const -> Tensor;

    






    [[nodiscard]] auto min(const std::vector<int> &dims, bool keep_dim = false) const -> Tensor;

    



    [[nodiscard]] auto argmin() const -> Tensor;

    







    [[nodiscard]] auto argmin(int dim, bool keep_dim = false) const -> Tensor;

    







    [[nodiscard]] auto argmin(const std::vector<int> &dims, bool keep_dim = false) const -> Tensor;

    



    [[nodiscard]] auto max() const -> Tensor;

    






    [[nodiscard]] auto max(int dim, bool keep_dim = false) const -> Tensor;

    






    [[nodiscard]] auto max(const std::vector<int> &dims, bool keep_dim = false) const -> Tensor;

    



    [[nodiscard]] auto argmax() const -> Tensor;

    







    [[nodiscard]] auto argmax(int dim, bool keep_dim = false) const -> Tensor;

    







    [[nodiscard]] auto argmax(const std::vector<int> &dims, bool keep_dim = false) const -> Tensor;

    



    [[nodiscard]] auto sum() const -> Tensor;

    






    [[nodiscard]] auto sum(int dim, bool keep_dim = false) const -> Tensor;

    






    [[nodiscard]] auto sum(const std::vector<int> &dims, bool keep_dim = false) const -> Tensor;

    



    [[nodiscard]] auto mean() const -> Tensor;

    






    [[nodiscard]] auto mean(int dim, bool keep_dim = false) const -> Tensor;

    






    [[nodiscard]] auto mean(const std::vector<int> &dims, bool keep_dim = false) const -> Tensor;

    







    [[nodiscard]] auto var(bool keep_dim = false, int correction = 1) const -> Tensor;

    








    [[nodiscard]] auto var(int dim, bool keep_dim = false, int correction = 1) const -> Tensor;

    








    [[nodiscard]] auto var(const std::vector<int> &dims, bool keep_dim = false, int correction = 1) const -> Tensor;

    


    [[nodiscard]] auto all() const -> bool;

    







    [[nodiscard]] auto all(int dim, bool keep_dim = false) const -> Tensor;

    







    [[nodiscard]] auto all(const std::vector<int> &dims, bool keep_dim = false) const -> Tensor;

    


    [[nodiscard]] auto any() const -> bool;

    







    [[nodiscard]] auto any(int dim, bool keep_dim = false) const -> Tensor;

    







    [[nodiscard]] auto any(const std::vector<int> &dims, bool keep_dim = false) const -> Tensor;

    
    
    

    





    auto clamp_(const ClampOptions &options = ClampOptions()) -> Tensor &;

    




    auto clamp(const ClampOptions &options = ClampOptions()) const -> Tensor;

    






    auto clamp_(const Tensor &min, const Tensor &max) -> Tensor &;

    






    auto clamp(const Tensor &min, const Tensor &max) const -> Tensor;

private:
    
    friend LONEXGRAD_EXPORT std::ostream &operator<<(std::ostream &os, const Tensor &tensor);

    template <typename T>
    friend struct autograd::TensorFunction;

    friend auto autograd::build_dag(const Tensor &tensor) -> TensorList;
    friend void autograd::backward(Tensor &tensor, const Tensor &grad, bool retain_graph);
    friend void autograd::calc_grad_input(Tensor &tensor, bool retain_graph);
    friend auto make_dot(const Tensor &tensor) -> std::string;

    
    void apply_grad_hook();

    Device device_;
    ScalarType scalar_type_;
    int offset_;
    Shape shape_;
    Shape stride_;
    std::shared_ptr<StorageBase> storage_;         
    std::shared_ptr<autograd::SharedGrad> ctx_;    
    mutable int version_count_ = 0;
};



template <typename T>
class LONEXGRAD_EXPORT CheckedVec {
    std::vector<T> data_;

public:
    using value_type = decltype(data_)::value_type;
    using Iterator = decltype(data_)::iterator;
    using ConstIterator = decltype(data_)::const_iterator;

    CheckedVec() = default;

    template <typename InputIt>
    CheckedVec(InputIt first, InputIt last)
        : data_(first, last) {}
    explicit CheckedVec(const std::vector<T> &data)
        : data_(data) {};
    explicit CheckedVec(std::vector<T> &&data)
        : data_(std::move(data)) {};
    CheckedVec(std::initializer_list<T> data)
        : data_(data) {};

    auto to_vec() const -> std::vector<T> {
        return data_;
    }

    void push_back(const T &item) {
        data_.push_back(item);
    }
    void push_back(T &&item) {
        data_.push_back(std::move(item));
    }
    template <typename... Args>
    void emplace_back(Args &&...args) {
        push_back(T(std::forward<Args>(args)...));
    }
    void clear() {
        data_.clear();
    }
    [[nodiscard]] auto empty() const -> bool {
        return data_.empty();
    }
    [[nodiscard]] auto size() const -> int {
        return static_cast<int>(data_.size());
    }
    [[nodiscard]] auto operator[](int idx) -> T & {
        int N = static_cast<int>(data_.size());
        if (idx >= N || idx < -N) {
            LONEXGRAD_EXCEPTION(std::format("Invalid idx, expected to be in range[{}, {}]", -N, N - 1));
        }
        idx = (idx < 0) ? (idx + N) % N : idx;
        return data_[static_cast<std::size_t>(idx)];
    }
    [[nodiscard]] auto operator[](int idx) const -> const T & {
        int N = static_cast<int>(data_.size());
        if (idx >= N || idx < -N) {
            LONEXGRAD_EXCEPTION(std::format("Invalid dim, expected to be in range[{}, {}]", -N, N - 1));
        }
        idx = (idx < 0) ? (idx + N) % N : idx;
        return data_[static_cast<std::size_t>(idx)];
    }
    [[nodiscard]] auto begin() -> Iterator {
        return std::begin(data_);
    }
    [[nodiscard]] auto begin() const -> ConstIterator {
        return std::begin(data_);
    }
    [[nodiscard]] auto end() -> Iterator {
        return std::end(data_);
    }
    [[nodiscard]] auto end() const -> ConstIterator {
        return std::end(data_);
    }
};














[[nodiscard]] LONEXGRAD_EXPORT auto full(Scalar value, Shape shape, Device device, bool requires_grad = false)
    -> Tensor;










template <typename T>
    requires(IsScalarType<T> || std::is_same_v<T, bool>)
[[nodiscard]] LONEXGRAD_EXPORT auto
    full(T value, Shape shape, ScalarType dtype, Device device, bool requires_grad = false) -> Tensor {
    return full(Scalar(value, dtype), shape, device, requires_grad);
}









template <typename T>
    requires(IsScalarType<T> || std::is_same_v<T, bool>)
[[nodiscard]] LONEXGRAD_EXPORT auto full(T value, Shape shape, Device device, bool requires_grad = false) -> Tensor {
    return full(Scalar(value), shape, device, requires_grad);
}









template <typename T>
    requires(IsScalarType<T> || std::is_same_v<T, bool>)
[[nodiscard]] LONEXGRAD_EXPORT auto full(T value, Shape shape, const TensorOptions &options) -> Tensor {
    return full(Scalar(value, options.dtype()), shape, options.device(), options.requires_grad());
}









[[nodiscard]] LONEXGRAD_EXPORT auto zeros(Shape shape, ScalarType dtype, Device device, bool requires_grad = false)
    -> Tensor;








[[nodiscard]] LONEXGRAD_EXPORT auto zeros(Shape shape, const TensorOptions &options = TensorOptions()) -> Tensor;








[[nodiscard]] LONEXGRAD_EXPORT auto zeros_like(const Tensor &tensor, bool requires_grad = false) -> Tensor;









[[nodiscard]] LONEXGRAD_EXPORT auto ones(Shape shape, ScalarType dtype, Device device, bool requires_grad = false)
    -> Tensor;








[[nodiscard]] LONEXGRAD_EXPORT auto ones(Shape shape, const TensorOptions &options = TensorOptions()) -> Tensor;








[[nodiscard]] LONEXGRAD_EXPORT auto ones_like(const Tensor &tensor, bool requires_grad = false) -> Tensor;










[[nodiscard]] LONEXGRAD_EXPORT auto arange(Shape shape, ScalarType dtype, Device device, bool requires_grad = false)
    -> Tensor;









[[nodiscard]] LONEXGRAD_EXPORT auto arange(Shape shape, const TensorOptions &options = TensorOptions(kDefaultInt))
    -> Tensor;














[[nodiscard]] LONEXGRAD_EXPORT auto linspace(
    double start,
    double stop,
    bool endpoint,
    Shape shape,
    ScalarType dtype,
    Device device,
    bool requires_grad = false
) -> Tensor;













[[nodiscard]] LONEXGRAD_EXPORT auto linspace(
    double start,
    double end,
    bool endpoint,
    Shape shape,
    const TensorOptions &options = TensorOptions(kDefaultFloat)
) -> Tensor;










[[nodiscard]] LONEXGRAD_EXPORT auto
    eye(int rows, int cols, ScalarType dtype, Device device, bool requires_grad = false) -> Tensor;









[[nodiscard]] LONEXGRAD_EXPORT auto eye(int rows, int cols, const TensorOptions &options = TensorOptions(kDefaultInt))
    -> Tensor;









[[nodiscard]] LONEXGRAD_EXPORT auto one_hot(Tensor indices, int num_classes) -> Tensor;














[[nodiscard]] LONEXGRAD_EXPORT auto cat(const TensorList &tensors, int dim) -> Tensor;











void LONEXGRAD_EXPORT cat(const TensorList &tensors, int dim, Tensor &out);














[[nodiscard]] LONEXGRAD_EXPORT auto stack(const TensorList &tensors, int dim) -> Tensor;














LONEXGRAD_EXPORT void stack(const TensorList &tensors, int dim, Tensor &out);
















[[nodiscard]] LONEXGRAD_EXPORT auto uniform_real(
    const Tensor &low,
    const Tensor &high,
    bool requires_grad = false,
    Generator &gen = get_default_generator()
) -> Tensor;













[[nodiscard]] LONEXGRAD_EXPORT auto uniform_real(
    double low,
    double high,
    Shape shape,
    const TensorOptions &options = TensorOptions(),
    Generator &gen = get_default_generator()
) -> Tensor;











[[nodiscard]] LONEXGRAD_EXPORT auto
    uniform_int(const Tensor &low, const Tensor &high, Generator &gen = get_default_generator()) -> Tensor;













[[nodiscard]] LONEXGRAD_EXPORT auto uniform_int(
    int64_t low,
    int64_t high,
    Shape shape,
    const TensorOptions &options = TensorOptions(),
    Generator &gen = get_default_generator()
) -> Tensor;















[[nodiscard]] LONEXGRAD_EXPORT auto
    bernoulli(const Tensor &p, bool requires_grad = false, Generator &gen = get_default_generator()) -> Tensor;












[[nodiscard]] LONEXGRAD_EXPORT auto bernoulli(
    double p,
    Shape shape,
    const TensorOptions &options = TensorOptions(),
    Generator &gen = get_default_generator()
) -> Tensor;












[[nodiscard]] LONEXGRAD_EXPORT auto binomial(
    const Tensor &p,
    const Tensor &num_draws,
    bool requires_grad = false,
    Generator &gen = get_default_generator()
) -> Tensor;













[[nodiscard]] LONEXGRAD_EXPORT auto binomial(
    double p,
    int num_draws,
    Shape shape,
    const TensorOptions &options = TensorOptions(),
    Generator &gen = get_default_generator()
) -> Tensor;













[[nodiscard]] LONEXGRAD_EXPORT auto
    geometric(const Tensor &p, bool requires_grad = false, Generator &gen = get_default_generator()) -> Tensor;














[[nodiscard]] LONEXGRAD_EXPORT auto geometric(
    double p,
    Shape shape,
    const TensorOptions &options = TensorOptions(),
    Generator &gen = get_default_generator()
) -> Tensor;














[[nodiscard]] LONEXGRAD_EXPORT auto
    poisson(const Tensor &lambda, bool requires_grad = false, Generator &gen = get_default_generator()) -> Tensor;











[[nodiscard]] LONEXGRAD_EXPORT auto poisson(
    double lambda,
    Shape shape,
    const TensorOptions &options = TensorOptions(),
    Generator &gen = get_default_generator()
) -> Tensor;










[[nodiscard]] LONEXGRAD_EXPORT auto
    exponential(const Tensor &lambda, bool requires_grad = false, Generator &gen = get_default_generator()) -> Tensor;











[[nodiscard]] LONEXGRAD_EXPORT auto exponential(
    double lambda,
    Shape shape,
    const TensorOptions &options = TensorOptions(),
    Generator &gen = get_default_generator()
) -> Tensor;















[[nodiscard]] LONEXGRAD_EXPORT auto
    normal(const Tensor &mu, const Tensor &std, bool requires_grad = false, Generator &gen = get_default_generator())
        -> Tensor;












[[nodiscard]] LONEXGRAD_EXPORT auto normal(
    double mu,
    double std,
    Shape shape,
    const TensorOptions &options = TensorOptions(),
    Generator &gen = get_default_generator()
) -> Tensor;













[[nodiscard]] LONEXGRAD_EXPORT auto
    cauchy(const Tensor &loc, const Tensor &scale, bool requires_grad = false, Generator &gen = get_default_generator())
        -> Tensor;














[[nodiscard]] LONEXGRAD_EXPORT auto cauchy(
    double loc,
    double scale,
    Shape shape,
    const TensorOptions &options = TensorOptions(),
    Generator &gen = get_default_generator()
) -> Tensor;












[[nodiscard]] LONEXGRAD_EXPORT auto
    lognormal(const Tensor &mu, const Tensor &std, bool requires_grad = false, Generator &gen = get_default_generator())
        -> Tensor;













[[nodiscard]] LONEXGRAD_EXPORT auto lognormal(
    double mu,
    double std,
    Shape shape,
    const TensorOptions &options = TensorOptions(),
    Generator &gen = get_default_generator()
) -> Tensor;












[[nodiscard]] LONEXGRAD_EXPORT auto
    weibull(const Tensor &lambda, const Tensor &k, bool requires_grad = false, Generator &gen = get_default_generator())
        -> Tensor;













[[nodiscard]] LONEXGRAD_EXPORT auto weibull(
    double lambda,
    double k,
    Shape shape,
    const TensorOptions &options = TensorOptions(),
    Generator &gen = get_default_generator()
) -> Tensor;














[[nodiscard]] LONEXGRAD_EXPORT auto broadcast_result_shape(const Shape lhs, const Shape rhs) -> Shape;












LONEXGRAD_EXPORT auto can_broadcast_to(const Shape &shape, const Shape &target_shape) -> bool;











LONEXGRAD_EXPORT auto are_broadcastable(const Shape &lhs, const Shape &rhs) -> bool;













[[nodiscard]] LONEXGRAD_EXPORT auto broadcast_to(const Tensor &tensor, const Shape &shape) -> Tensor;












[[nodiscard]] LONEXGRAD_EXPORT auto expand(const Tensor &tensor, const Shape &shape) -> Tensor;











[[nodiscard]] LONEXGRAD_EXPORT auto squeeze(const Tensor &tensor, int dim) -> Tensor;









[[nodiscard]] LONEXGRAD_EXPORT auto unsqueeze(const Tensor &tensor, int dim) -> Tensor;










[[nodiscard]] LONEXGRAD_EXPORT auto reshape(const Tensor &tensor, const Shape &shape) -> Tensor;












[[nodiscard]] LONEXGRAD_EXPORT auto flatten(const Tensor &tensor, int start_dim = 0, int end_dim = -1) -> Tensor;










[[nodiscard]] LONEXGRAD_EXPORT auto permute(const Tensor &tensor, const std::vector<int> &dims) -> Tensor;











[[nodiscard]] LONEXGRAD_EXPORT auto repeat_interleave(const Tensor &tensor, int repeats, int dim) -> Tensor;











[[nodiscard]] LONEXGRAD_EXPORT auto repeat(const Tensor &tensor, const Tensor &repeats) -> Tensor;











[[nodiscard]] LONEXGRAD_EXPORT auto repeat(const Tensor &tensor, const std::vector<int> &repeats) -> Tensor;









[[nodiscard]] LONEXGRAD_EXPORT auto gather(const Tensor &tensor, const Tensor &indices, int dim) -> Tensor;














[[nodiscard]] LONEXGRAD_EXPORT auto index(const Tensor &tensor, const Tensor &rhs) -> Tensor;









[[nodiscard]] LONEXGRAD_EXPORT auto index_select(const Tensor &tensor, const Tensor &indices, int dim) -> Tensor;









[[nodiscard]] LONEXGRAD_EXPORT auto index_select(const Tensor &tensor, const std::vector<int> &indices, int dim)
    -> Tensor;









[[nodiscard]] LONEXGRAD_EXPORT auto index_put(const Tensor &self, const Tensor &rhs, const Scalar &value) -> Tensor;











[[nodiscard]] LONEXGRAD_EXPORT auto index_put(const Tensor &self, const Tensor &rhs, const Tensor &values) -> Tensor;







template <IsScalarType T>
[[nodiscard]] LONEXGRAD_EXPORT auto index_put(const Tensor &self, const Tensor &rhs, T value) -> Tensor {
    return index_put(self, rhs, Scalar(value, self.dtype()));
}










// NOLINTNEXTLINE(*-macro-usage)
#define DECLARE_BINARY_OPS(FUNC)                                                               \
    [[nodiscard]] LONEXGRAD_EXPORT auto FUNC(const Tensor &lhs, const Tensor &rhs) -> Tensor; \
    [[nodiscard]] LONEXGRAD_EXPORT auto FUNC(const Tensor &lhs, Scalar rhs) -> Tensor;        \
    [[nodiscard]] LONEXGRAD_EXPORT auto FUNC(Scalar lhs, const Tensor &rhs) -> Tensor;        \
    template <IsScalarType T>                                                                  \
    [[nodiscard]] LONEXGRAD_EXPORT inline auto FUNC(const Tensor &lhs, T rhs) -> Tensor {     \
        Scalar s = lhs.dtype() == kBool ? cast_to_default(rhs) : Scalar(rhs, lhs.dtype());     \
        return FUNC(lhs, s);                                                                   \
    }                                                                                          \
    template <IsScalarType T>                                                                  \
    [[nodiscard]] LONEXGRAD_EXPORT auto FUNC(T lhs, const Tensor &rhs) -> Tensor {            \
        Scalar s = rhs.dtype() == kBool ? cast_to_default(lhs) : Scalar(lhs, rhs.dtype());     \
        return FUNC(s, rhs);                                                                   \
    }

// NOLINTNEXTLINE(*-macro-usage)
#define DECLARE_BINARY_GLOBAL_OPS(OP, FUNC)                                                                 \
    DECLARE_BINARY_OPS(FUNC)                                                                                \
    template <IsScalarType T>                                                                               \
    [[nodiscard]] LONEXGRAD_EXPORT auto operator OP(const Tensor &lhs, T rhs)->Tensor {                    \
        return FUNC(lhs, rhs);                                                                              \
    }                                                                                                       \
    template <IsScalarType T>                                                                               \
    [[nodiscard]] LONEXGRAD_EXPORT auto operator OP(T lhs, const Tensor &rhs)->Tensor {                    \
        return FUNC(lhs, rhs);                                                                              \
    }                                                                                                       \
    [[nodiscard]] LONEXGRAD_EXPORT inline auto operator OP(const Tensor &lhs, Scalar rhs)->Tensor {        \
        return FUNC(lhs, rhs);                                                                              \
    }                                                                                                       \
    [[nodiscard]] LONEXGRAD_EXPORT inline auto operator OP(Scalar lhs, const Tensor &rhs)->Tensor {        \
        return FUNC(lhs, rhs);                                                                              \
    }                                                                                                       \
    [[nodiscard]] LONEXGRAD_EXPORT inline auto operator OP(const Tensor &lhs, const Tensor &rhs)->Tensor { \
        return FUNC(lhs, rhs);                                                                              \
    }


DECLARE_BINARY_GLOBAL_OPS(==, eq);
DECLARE_BINARY_GLOBAL_OPS(!=, ne);
DECLARE_BINARY_GLOBAL_OPS(<, lt);
DECLARE_BINARY_GLOBAL_OPS(<=, le);
DECLARE_BINARY_GLOBAL_OPS(>, gt);
DECLARE_BINARY_GLOBAL_OPS(>=, ge);
DECLARE_BINARY_GLOBAL_OPS(||, logical_or);
DECLARE_BINARY_GLOBAL_OPS(&&, logical_and);
DECLARE_BINARY_GLOBAL_OPS(|, bitwise_or);
DECLARE_BINARY_GLOBAL_OPS(&, bitwise_and);
DECLARE_BINARY_GLOBAL_OPS(^, bitwise_xor);
DECLARE_BINARY_GLOBAL_OPS(<<, bitwise_left_shift);
DECLARE_BINARY_GLOBAL_OPS(>>, bitwise_right_shift);
DECLARE_BINARY_GLOBAL_OPS(%, modulo);
DECLARE_BINARY_GLOBAL_OPS(+, add);
DECLARE_BINARY_GLOBAL_OPS(-, sub);
DECLARE_BINARY_GLOBAL_OPS(*, mul);
DECLARE_BINARY_GLOBAL_OPS(/, div);
#undef DECLARE_BINARY_GLOBAL_OPS


DECLARE_BINARY_OPS(maximum);
DECLARE_BINARY_OPS(minimum);
DECLARE_BINARY_OPS(pow);
#undef DECLARE_BINARY_OPS












[[nodiscard]] LONEXGRAD_EXPORT auto matmul(const Tensor &lhs, const Tensor &rhs) -> Tensor;










[[nodiscard]] LONEXGRAD_EXPORT auto min(const Tensor &tensor) -> Tensor;









[[nodiscard]] LONEXGRAD_EXPORT auto min(const Tensor &tensor, int dim, bool keep_dim = false) -> Tensor;









[[nodiscard]] LONEXGRAD_EXPORT auto min(const Tensor &tensor, const std::vector<int> &dims, bool keep_dim = false)
    -> Tensor;






[[nodiscard]] LONEXGRAD_EXPORT auto argmin(const Tensor &tensor) -> Tensor;










[[nodiscard]] LONEXGRAD_EXPORT auto argmin(const Tensor &tensor, int dim, bool keep_dim = false) -> Tensor;










[[nodiscard]] LONEXGRAD_EXPORT auto argmin(const Tensor &tensor, const std::vector<int> &dims, bool keep_dim = false)
    -> Tensor;






[[nodiscard]] LONEXGRAD_EXPORT auto max(const Tensor &tensor) -> Tensor;









[[nodiscard]] LONEXGRAD_EXPORT auto max(const Tensor &tensor, int dim, bool keep_dim = false) -> Tensor;









[[nodiscard]] LONEXGRAD_EXPORT auto max(const Tensor &tensor, const std::vector<int> &dims, bool keep_dim = false)
    -> Tensor;






[[nodiscard]] LONEXGRAD_EXPORT auto argmax(const Tensor &tensor) -> Tensor;










[[nodiscard]] LONEXGRAD_EXPORT auto argmax(const Tensor &tensor, int dim, bool keep_dim = false) -> Tensor;










[[nodiscard]] LONEXGRAD_EXPORT auto argmax(const Tensor &tensor, const std::vector<int> &dims, bool keep_dim = false)
    -> Tensor;






[[nodiscard]] LONEXGRAD_EXPORT auto sum(const Tensor &tensor) -> Tensor;









[[nodiscard]] LONEXGRAD_EXPORT auto sum(const Tensor &tensor, int dim, bool keep_dim = false) -> Tensor;









[[nodiscard]] LONEXGRAD_EXPORT auto sum(const Tensor &tensor, const std::vector<int> &dims, bool keep_dim = false)
    -> Tensor;






[[nodiscard]] LONEXGRAD_EXPORT auto mean(const Tensor &tensor) -> Tensor;









[[nodiscard]] LONEXGRAD_EXPORT auto mean(const Tensor &tensor, int dim, bool keep_dim = false) -> Tensor;









[[nodiscard]] LONEXGRAD_EXPORT auto mean(const Tensor &tensor, const std::vector<int> &dims, bool keep_dim = false)
    -> Tensor;






[[nodiscard]] LONEXGRAD_EXPORT auto all(const Tensor &input) -> bool;










[[nodiscard]] LONEXGRAD_EXPORT auto all(const Tensor &tensor, int dim, bool keep_dim = false) -> Tensor;










[[nodiscard]] LONEXGRAD_EXPORT auto all(const Tensor &tensor, const std::vector<int> &dims, bool keep_dim = false)
    -> Tensor;






[[nodiscard]] LONEXGRAD_EXPORT auto any(const Tensor &input) -> bool;












[[nodiscard]] LONEXGRAD_EXPORT auto any(const Tensor &tensor, int dim, bool keep_dim = false) -> Tensor;












[[nodiscard]] LONEXGRAD_EXPORT auto any(const Tensor &tensor, const std::vector<int> &dims, bool keep_dim = false)
    -> Tensor;










[[nodiscard]] LONEXGRAD_EXPORT auto var(const Tensor &input, bool keep_dim = false, int correction = 1) -> Tensor;











[[nodiscard]] LONEXGRAD_EXPORT auto var(const Tensor &input, int dim, bool keep_dim = false, int correction = 1)
    -> Tensor;











[[nodiscard]] LONEXGRAD_EXPORT auto
    var(const Tensor &input, const std::vector<int> &dims, bool keep_dim = false, int correction = 1) -> Tensor;










[[nodiscard]] LONEXGRAD_EXPORT auto abs(const Tensor &tensor) -> Tensor;






[[nodiscard]] LONEXGRAD_EXPORT auto negate(const Tensor &tensor) -> Tensor;








[[nodiscard]] LONEXGRAD_EXPORT auto logical_not(const Tensor &tensor) -> Tensor;







[[nodiscard]] LONEXGRAD_EXPORT auto sign(const Tensor &tensor) -> Tensor;






[[nodiscard]] LONEXGRAD_EXPORT auto log(const Tensor &tensor) -> Tensor;






[[nodiscard]] LONEXGRAD_EXPORT auto log10(const Tensor &tensor) -> Tensor;






[[nodiscard]] LONEXGRAD_EXPORT auto log2(const Tensor &tensor) -> Tensor;






[[nodiscard]] LONEXGRAD_EXPORT auto log1p(const Tensor &tensor) -> Tensor;






[[nodiscard]] LONEXGRAD_EXPORT auto exp(const Tensor &tensor) -> Tensor;






[[nodiscard]] LONEXGRAD_EXPORT auto exp2(const Tensor &tensor) -> Tensor;






[[nodiscard]] LONEXGRAD_EXPORT auto expm1(const Tensor &tensor) -> Tensor;






[[nodiscard]] LONEXGRAD_EXPORT auto sqrt(const Tensor &tensor) -> Tensor;






[[nodiscard]] LONEXGRAD_EXPORT auto sin(const Tensor &tensor) -> Tensor;






[[nodiscard]] LONEXGRAD_EXPORT auto cos(const Tensor &tensor) -> Tensor;






[[nodiscard]] LONEXGRAD_EXPORT auto tan(const Tensor &tensor) -> Tensor;






[[nodiscard]] LONEXGRAD_EXPORT auto asin(const Tensor &tensor) -> Tensor;






[[nodiscard]] LONEXGRAD_EXPORT auto acos(const Tensor &tensor) -> Tensor;






[[nodiscard]] LONEXGRAD_EXPORT auto atan(const Tensor &tensor) -> Tensor;






[[nodiscard]] LONEXGRAD_EXPORT auto sinh(const Tensor &tensor) -> Tensor;






[[nodiscard]] LONEXGRAD_EXPORT auto cosh(const Tensor &tensor) -> Tensor;






[[nodiscard]] LONEXGRAD_EXPORT auto tanh(const Tensor &tensor) -> Tensor;






[[nodiscard]] LONEXGRAD_EXPORT auto asinh(const Tensor &tensor) -> Tensor;






[[nodiscard]] LONEXGRAD_EXPORT auto acosh(const Tensor &tensor) -> Tensor;






[[nodiscard]] LONEXGRAD_EXPORT auto atanh(const Tensor &tensor) -> Tensor;







[[nodiscard]] LONEXGRAD_EXPORT auto erf(const Tensor &tensor) -> Tensor;







[[nodiscard]] LONEXGRAD_EXPORT auto erfc(const Tensor &tensor) -> Tensor;







[[nodiscard]] LONEXGRAD_EXPORT auto tgamma(const Tensor &tensor) -> Tensor;







[[nodiscard]] LONEXGRAD_EXPORT auto lgamma(const Tensor &tensor) -> Tensor;









[[nodiscard]] LONEXGRAD_EXPORT auto digamma(const Tensor &tensor) -> Tensor;







[[nodiscard]] LONEXGRAD_EXPORT auto ceil(const Tensor &tensor) -> Tensor;







[[nodiscard]] LONEXGRAD_EXPORT auto floor(const Tensor &tensor) -> Tensor;








[[nodiscard]] LONEXGRAD_EXPORT auto round(const Tensor &tensor) -> Tensor;








[[nodiscard]] LONEXGRAD_EXPORT auto isinf(const Tensor &tensor) -> Tensor;









[[nodiscard]] LONEXGRAD_EXPORT auto isnan(const Tensor &tensor) -> Tensor;









[[nodiscard]] LONEXGRAD_EXPORT auto isfinite(const Tensor &tensor) -> Tensor;


LONEXGRAD_EXPORT auto operator-(const Tensor &tensor) -> Tensor;
LONEXGRAD_EXPORT auto operator!(const Tensor &tensor) -> Tensor;











[[nodiscard]] LONEXGRAD_EXPORT auto sigmoid(const Tensor &tensor) -> Tensor;







[[nodiscard]] LONEXGRAD_EXPORT auto log_sigmoid(const Tensor &tensor) -> Tensor;







[[nodiscard]] LONEXGRAD_EXPORT auto hardsigmoid(const Tensor &tensor) -> Tensor;











[[nodiscard]] LONEXGRAD_EXPORT auto softplus(const Tensor &tensor, double beta = 1, double threshold = 20) -> Tensor;







[[nodiscard]] LONEXGRAD_EXPORT auto relu(const Tensor &tensor) -> Tensor;







[[nodiscard]] LONEXGRAD_EXPORT auto relu6(const Tensor &tensor) -> Tensor;









[[nodiscard]] LONEXGRAD_EXPORT auto leaky_relu(const Tensor &tensor, double negative_slope = 0.01) -> Tensor;








[[nodiscard]] LONEXGRAD_EXPORT auto elu(const Tensor &tensor, double alpha = 1) -> Tensor;







[[nodiscard]] LONEXGRAD_EXPORT auto selu(const Tensor &tensor) -> Tensor;







[[nodiscard]] LONEXGRAD_EXPORT auto silu(const Tensor &tensor) -> Tensor;









[[nodiscard]] LONEXGRAD_EXPORT auto hardtanh(const Tensor &tensor, double min = -1, double max = 1) -> Tensor;







[[nodiscard]] LONEXGRAD_EXPORT auto softsign(const Tensor &tensor) -> Tensor;










[[nodiscard]] LONEXGRAD_EXPORT auto softmax(const Tensor &tensor, int dim) -> Tensor;








[[nodiscard]] LONEXGRAD_EXPORT auto log_softmax(const Tensor &tensor, int dim) -> Tensor;










LONEXGRAD_EXPORT void save(const std::string &path, const Tensor &tensor);








LONEXGRAD_EXPORT auto load(const std::string &path) -> Tensor;









[[nodiscard]] LONEXGRAD_EXPORT auto where(const Tensor &cond, const Tensor &lhs, const Tensor &rhs) -> Tensor;
[[nodiscard]] LONEXGRAD_EXPORT auto where(const Tensor &cond, Scalar lhs, Scalar rhs) -> Tensor;
[[nodiscard]] LONEXGRAD_EXPORT auto where(const Tensor &cond, const Tensor &lhs, Scalar rhs) -> Tensor;
[[nodiscard]] LONEXGRAD_EXPORT auto where(const Tensor &cond, Scalar lhs, const Tensor &rhs) -> Tensor;









template <IsScalarType T>
[[nodiscard]] LONEXGRAD_EXPORT auto where(const Tensor &cond, T lhs, T rhs) -> Tensor {
    return where(cond, Scalar(lhs), Scalar(rhs));
}
template <IsScalarType T>
[[nodiscard]] LONEXGRAD_EXPORT auto where(const Tensor &cond, const Tensor &lhs, T rhs) -> Tensor {
    return where(cond, lhs, Scalar(rhs));
}
template <IsScalarType T>
[[nodiscard]] LONEXGRAD_EXPORT auto where(const Tensor &cond, T lhs, const Tensor &rhs) -> Tensor {
    return where(cond, Scalar(lhs), rhs);
}


class LONEXGRAD_EXPORT CloseOptions {
public:
    auto rtol(double tol) -> CloseOptions &;
    auto atol(double tol) -> CloseOptions &;
    auto equal_nan() -> CloseOptions &;
    [[nodiscard]] inline auto rtol() const -> double {
        return _rtol;
    }
    [[nodiscard]] inline auto atol() const -> double {
        return _atol;
    }
    [[nodiscard]] inline auto equal_nan() const -> bool {
        return _equal_nan;
    }

private:
    double _rtol = 1e-5;
    double _atol = 1e-8;
    bool _equal_nan = false;
};













[[nodiscard]] LONEXGRAD_EXPORT auto
    isclose(const Tensor &lhs, const Tensor &rhs, const CloseOptions &options = CloseOptions()) -> Tensor;












template <IsScalarType T>
[[nodiscard]] LONEXGRAD_EXPORT auto isclose(const Tensor &lhs, T rhs, const CloseOptions &options = CloseOptions())
    -> Tensor {
    return isclose(lhs, Tensor(Scalar(rhs), lhs.device()).expand(lhs.shape()), options);
}












template <IsScalarType T>
[[nodiscard]] LONEXGRAD_EXPORT auto isclose(T lhs, const Tensor &rhs, const CloseOptions &options = CloseOptions())
    -> Tensor {
    return isclose(Tensor(Scalar(lhs), rhs.device()).expand(rhs.shape()), rhs, options);
}












[[nodiscard]] LONEXGRAD_EXPORT auto
    allclose(const Tensor &input, const Tensor &other, const CloseOptions &options = CloseOptions()) -> bool;







[[nodiscard]] LONEXGRAD_EXPORT auto clamp(const Tensor &input, const ClampOptions &options = ClampOptions()) -> Tensor;








[[nodiscard]] LONEXGRAD_EXPORT auto clamp(const Tensor &input, const Tensor &min, const Tensor &max) -> Tensor;













[[nodiscard]] LONEXGRAD_EXPORT auto conv2d(
    const Tensor &input,
    const Tensor &weight,
    const std::optional<Tensor> &bias = {},
    int stride = 1,
    int padding = 0
) -> Tensor;












[[nodiscard]] LONEXGRAD_EXPORT auto max_pool2d(const Tensor &input, int kernel_size, int stride, int padding = 0)
    -> Tensor;












[[nodiscard]] LONEXGRAD_EXPORT auto min_pool2d(const Tensor &input, int kernel_size, int stride, int padding = 0)
    -> Tensor;












[[nodiscard]] LONEXGRAD_EXPORT auto avg_pool2d(const Tensor &input, int kernel_size, int stride, int padding = 0)
    -> Tensor;








[[nodiscard]] LONEXGRAD_EXPORT auto embedding(const Tensor &input, const Tensor &weight) -> Tensor;






[[nodiscard]] LONEXGRAD_EXPORT auto current_memory_allocated(const Device &device) -> uint64_t;







[[nodiscard]] LONEXGRAD_EXPORT auto total_memory_allocated(const Device &device) -> uint64_t;







[[nodiscard]] LONEXGRAD_EXPORT auto get_device_count(Backend backend) -> int;






[[nodiscard]] LONEXGRAD_EXPORT auto make_dot(const Tensor &tensor) -> std::string;

}    

#endif    
