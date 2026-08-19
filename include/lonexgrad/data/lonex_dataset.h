


#ifndef lonexgrad_DATASET_H_
#define lonexgrad_DATASET_H_

#include <lonexgrad/lonex_concepts.h>
#include <lonexgrad/lonex_exception.h>
#include <lonexgrad/export.h>
#include <lonexgrad/lonex_random.h>
#include <lonexgrad/lonex_tensor.h>
#include <lonexgrad/lonex_util.h>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <format>
#include <memory>
#include <numeric>
#include <ranges>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace lonexgrad::data {


template <typename T>
concept IsDataset = requires(const T ct, int idx) {
    typename T::DataType;
    requires IsSpecialization<typename T::DataType, std::tuple>;
    { ct.size() } -> std::same_as<int>;
    { ct.get(idx) } -> std::same_as<typename T::DataType>;
};


template <typename... Ts>
    requires IsAllOf<Tensor, Ts...>
class LONEXGRAD_EXPORT TensorDataset {
public:
    static_assert(sizeof...(Ts) > 0);
    using DataType = std::tuple<Ts...>;
    TensorDataset(Ts... args)
        : tensors(std::make_tuple(args...)), N(std::get<0>(tensors).size(0)) {
        
        std::apply(
            [&](auto &&...arg) {
                auto check_dim = [&](Tensor tensor) {
                    if (tensor.size(0) != N) {
                        LONEXGRAD_EXCEPTION("Given tensors do not have the same size of the outer-most dim.");
                    }
                };
                (check_dim(arg), ...);
            },
            tensors
        );
    }

    [[nodiscard]] auto size() const -> int {
        return N;
    }

    [[nodiscard]] auto get(int idx) const -> DataType {
        return std::apply([&](auto &&...arg) { return std::make_tuple(arg[idx]...); }, tensors);
    }

private:
    std::tuple<Ts...> tensors;
    int N;
};




template <IsDataset T>
class LONEXGRAD_EXPORT DatasetView {
public:
    



    DatasetView(T &&dataset)
        : dataset_(std::make_shared<T>(std::move(dataset))) {
        indices_ = std::views::iota(0, dataset_->size()) | lonexgrad::to<decltype(indices_)>();
        check();
    }

    




    DatasetView(std::shared_ptr<T> dataset, const std::vector<int> &indices)
        : dataset_(std::move(dataset)), indices_(indices) {
        check();
    }

    



    auto shuffle(Generator &gen) {
        lonexgrad::shuffle(indices_, gen);
    }

    



    auto size() -> int {
        return static_cast<int>(indices_.size());
    }

    
    auto operator[](int idx) -> T::DataType {
        int N = static_cast<int>(indices_.size());
        if (idx >= N || idx < -N) {
            LONEXGRAD_EXCEPTION(std::format("Invalid idx, expected to be in range[{}, {}]", -N, N - 1));
        }
        
        return dataset_->get(indices_.at(static_cast<std::size_t>(idx)));
    }

private:
    void check() {
        if (static_cast<int>(indices_.size()) > dataset_->size()) {
            LONEXGRAD_EXCEPTION(
                std::format(
                    "Given indices list of size {:d}, but dataset is only of size {:d}",
                    indices_.size(),
                    dataset_->size()
                )
            );
        }
        if (indices_.empty()) {
            LONEXGRAD_EXCEPTION("Given empty indices list");
        }

        if (dataset_->size() <= 0) {
            LONEXGRAD_EXCEPTION("Dataset must contain at least one element");
        }
    }

    std::shared_ptr<T> dataset_;
    std::vector<int> indices_;
};









template <IsDataset T>
LONEXGRAD_EXPORT auto random_split(T &&dataset, Generator &gen, const std::vector<int> &splits)
    -> std::vector<DatasetView<T>> {
    auto split_sum = std::reduce(splits.begin(), splits.end());
    if (split_sum != dataset.size()) {
        LONEXGRAD_EXCEPTION(
            std::format(
                "Number of elements the split represents ({:d}) does not match the size of the dataset ({:d})",
                split_sum,
                dataset.size()
            )
        );
    }
    auto shared_dataest = std::make_shared<T>(std::forward<T>(dataset));
    std::vector<int> indices = std::views::iota(0, split_sum) | lonexgrad::to<std::vector<int>>();
    shuffle(indices, gen);

    int start_idx = 0;
    std::vector<DatasetView<T>> dataset_views;
    for (int split_size : splits) {
        if (split_size <= 0) {
            LONEXGRAD_EXCEPTION(std::format("Split sizes must be greater than 0, given split size of {:d}", split_size));
        }
        std::vector<int> view_indices =
            std::views::iota(start_idx, start_idx + split_size)
            | std::views::transform([&](int idx) { return indices.at(static_cast<std::size_t>(idx)); })
            | lonexgrad::to<std::vector<int>>();
        start_idx += split_size;
        dataset_views.emplace_back(shared_dataest, view_indices);
    }
    return dataset_views;
}









template <IsDataset T, typename... Ts>
    requires(!std::is_lvalue_reference_v<T> && IsAllOf<int, Ts...>)
LONEXGRAD_EXPORT auto random_split(T &&dataset, Generator &gen, const std::tuple<Ts...> &splits) {
    auto split_sum = std::apply([](auto... v) { return (v + ...); }, splits);
    if (split_sum != dataset.size()) {
        LONEXGRAD_EXCEPTION(
            std::format(
                "The sum of splits ({:d}) does not match the size of the dataset ({:d})",
                split_sum,
                dataset.size()
            )
        );
    }
    auto shared_dataest = std::make_shared<T>(std::forward<T>(dataset));
    std::vector<int> indices = std::views::iota(0, split_sum) | lonexgrad::to<std::vector<int>>();
    shuffle(indices, gen);

    int start_idx = 0;
    return std::apply(
        [&](auto &&...arg) {
            auto make_result = [&](int split_size) -> DatasetView<T> {
                if (split_size <= 0) {
                    LONEXGRAD_EXCEPTION(
                        std::format("Split sizes must be greater than 0, given split size of {:d}", split_size)
                    );
                }
                std::vector<int> view_indices =
                    std::views::iota(start_idx, start_idx + split_size)
                    | std::views::transform([&](int idx) { return indices.at(static_cast<std::size_t>(idx)); })
                    | lonexgrad::to<std::vector<int>>();
                start_idx += split_size;
                return {shared_dataest, view_indices};
            };
            return std::make_tuple(make_result(arg)...);
        },
        splits
    );
}









template <IsDataset T, typename... Ts>
    requires(!std::is_lvalue_reference_v<T> && IsAllOf<int, Ts...>)
LONEXGRAD_EXPORT auto random_split(T &&dataset, uint64_t seed, Ts... splits) {
    Generator gen(seed);
    return random_split(std::forward<T>(dataset), gen, std::make_tuple(splits...));
}









template <IsDataset T>
    requires(!std::is_lvalue_reference_v<T>)
LONEXGRAD_EXPORT auto random_split(T &&dataset, uint64_t seed, const std::vector<int> &splits)
    -> std::vector<DatasetView<T>> {
    Generator gen(seed);
    return random_split(std::forward<T>(dataset), gen, splits);
}




template <IsDataset T>
class LONEXGRAD_EXPORT DataLoader {
public:
    
    class Iterator {
    public:
        Iterator(std::shared_ptr<DatasetView<T>> dataview, int idx, int batch_size)
            : dataview_(std::move(dataview)), idx_(idx), batch_size_(batch_size) {}
        auto operator!=(const Iterator &other) const -> bool {
            return idx_ != other.idx_;
        }
        auto operator++() -> Iterator {
            if (idx_ * batch_size_ > dataview_->size()) {
                LONEXGRAD_ERROR(
                    "Internal iterator is already passed the end. Dataloader::Iterator should only be used implicitly "
                    "in range-based for loops"
                );
            }
            ++idx_;
            return *this;
        }
        auto operator*() const {
            if (idx_ * batch_size_ > dataview_->size()) {
                LONEXGRAD_ERROR(
                    "Internal iterator is already passed the end. Dataloader::Iterator should only be used implicitly "
                    "in range-based for loops"
                );
            }
            int idx_start = idx_ * batch_size_;
            int idx_end = std::min((idx_ + 1) * batch_size_, dataview_->size());

            
            auto batched_values = std::apply(
                [](auto &&...arg) {
                    return std::make_tuple(CheckedVec<std::remove_reference_t<decltype(arg)>>{arg}...);
                },
                (*dataview_)[idx_start]
            );

            
            for (int i : std::views::iota(idx_start + 1, idx_end)) {
                std::apply(
                    [&](auto &&...batched_value) {
                        std::apply(
                            [&](auto &&...cur_value) { (batched_value.push_back(cur_value), ...); },
                            (*dataview_)[i]
                        );
                    },
                    batched_values
                );
            }

            
            auto val = std::apply(
                [](auto &&...batched_value) {
                    auto make_result = [](auto &&arg) {
                        if constexpr (std::is_same_v<std::remove_cvref_t<decltype(arg)>, TensorList>) {
                            return stack(arg, 0);
                        } else {
                            return arg;
                        }
                    };
                    return std::make_tuple(make_result(batched_value)...);
                },
                batched_values
            );
            return val;
        }

    private:
        std::shared_ptr<DatasetView<T>> dataview_;
        int idx_;
        int batch_size_;
    };

    






    DataLoader(const DatasetView<T> &dataview, int batch_size, bool shuffle = true, uint64_t seed = 0)
        : dataview_(std::make_shared<DatasetView<T>>(dataview)),
          batch_size_(batch_size),
          shuffle_(shuffle),
          gen_(seed) {
        check();
    }

    






    DataLoader(DatasetView<T> &&dataview, int batch_size, bool shuffle = true, uint64_t seed = 0)
        : dataview_(std::make_shared<DatasetView<T>>(std::move(dataview))),
          batch_size_(batch_size),
          shuffle_(shuffle),
          gen_(seed) {
        check();
    }

    



    auto size() -> int {
        return ceil_div(dataview_->size(), batch_size_);
    }

    auto begin() -> Iterator {
        if (shuffle_) {
            dataview_->shuffle(gen_);
        }
        return {dataview_, 0, batch_size_};
    }
    auto end() -> Iterator {
        return {dataview_, ceil_div(dataview_->size(), batch_size_), batch_size_};
    }

private:
    void check() {
        if (dataview_->size() <= 0) {
            LONEXGRAD_EXCEPTION("Dataview must contain at least one element");
        }
        if (batch_size_ <= 0) {
            LONEXGRAD_EXCEPTION("batch_size must be positive");
        }
    }

    std::shared_ptr<DatasetView<T>> dataview_;
    int batch_size_;
    bool shuffle_;
    Generator gen_;
};

}    

#endif    
