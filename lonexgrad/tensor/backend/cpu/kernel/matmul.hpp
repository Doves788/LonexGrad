


#ifndef LONEXGRAD_BACKEND_CPU_KERNEL_MATMUL_H_
#define LONEXGRAD_BACKEND_CPU_KERNEL_MATMUL_H_

#include <lonexgrad/lonex_concepts.h>
#include <lonexgrad/lonex_scalar.h>

#include "tensor/backend/common/lonex_span.h"
#include "tensor/backend/cpu/kernel/lonex_simd.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <span>

#if defined(_WIN32)
#include <malloc.h>
#endif

namespace lonexgrad::cpu::kernel::matmul {

constexpr int KERNEL_HEIGHT = 8;
constexpr int KERNEL_WIDTH_VEC = 4;
constexpr int MAX_STORAGE_SIZE = 1200 * 1200;
constexpr int TILE_STRIDE = 64;


template <typename T>
struct WideScalar {
    WideScalar() = default;

    WideScalar(wide_type_t<T> value)
        : _value(value) {}

    static auto load_single(T value) -> WideScalar {
        return {wide_type<T>::load_single(value)};
    }

    static auto load_wide(T *p) -> WideScalar {
        return {wide_type<T>::load_wide(p)};
    }

    static auto load_wide(const T *p) -> WideScalar {
        return {wide_type<T>::load_wide(p)};
    }

    inline auto operator+(const WideScalar<T> &rhs) const noexcept
        -> WideScalar {
        return {wide_type<T>::add(_value, rhs._value)};
    }

    inline auto operator*(const WideScalar<T> &rhs) const noexcept
        -> WideScalar {
        return {wide_type<T>::mul(_value, rhs._value)};
    }

    inline auto operator+=(const WideScalar<T> &rhs) noexcept
        -> WideScalar & {
        _value = wide_type<T>::add(_value, rhs._value);
        return *this;
    }

    inline void store(T *p) {
        wide_type<T>::store(p, _value);
    }

    constexpr static int N = 128 / (sizeof(T) * 8);

    wide_type_t<T> _value =
        wide_type<T>::load_single(static_cast<T>(0));
};






#if defined(_WIN32)

template <typename T>
auto ALIGNED_ALLOC(std::size_t alignment, std::size_t size) -> T * {
    return static_cast<T *>(_aligned_malloc(size, alignment));
}

#define ALIGNED_FREE _aligned_free

#else

template <typename T>
auto ALIGNED_ALLOC(std::size_t alignment, std::size_t size) -> T * {
    const std::size_t padded_size =
        ((size + alignment - 1) / alignment) * alignment;

    return static_cast<T *>(std::aligned_alloc(alignment, padded_size));
}

#define ALIGNED_FREE std::free

#endif






template <typename T>
struct AlignedArray {
    explicit AlignedArray(std::size_t n)
        : data(ALIGNED_ALLOC<T>(64, sizeof(T) * n)),
          _n(n) {
        assert(data != nullptr);
        reset();
    }

    ~AlignedArray() {
        if (data != nullptr) {
            ALIGNED_FREE(data);
            data = nullptr;
        }
    }

    void reset() {
        std::memset(data, 0, sizeof(T) * _n);
    }

    AlignedArray(const AlignedArray &) = delete;

    AlignedArray &operator=(const AlignedArray &) = delete;

    AlignedArray(AlignedArray &&other) noexcept
        : data(other.data),
          _n(other._n) {
        other.data = nullptr;
        other._n = 0;
    }

    AlignedArray &operator=(AlignedArray &&other) noexcept {
        if (this != &other) {
            if (data != nullptr) {
                ALIGNED_FREE(data);
            }

            data = other.data;
            _n = other._n;

            other.data = nullptr;
            other._n = 0;
        }

        return *this;
    }

    T *data;
    std::size_t _n;
};






inline auto round_up(int n, int bucket_size) -> int {
    return ((n + bucket_size - 1) / bucket_size) * bucket_size;
}






template <typename T>
inline void tile_kernel(
    const T *__restrict A,
    const T *__restrict B,
    T *__restrict C,
    int start_row,
    int start_col,
    int l,
    int r,
    int K,
    int M
) {
    constexpr int vec_width = WideScalar<T>::N;

    std::array<
        std::array<WideScalar<T>, KERNEL_WIDTH_VEC>,
        KERNEL_HEIGHT
    > t{};

    for (int k = l; k < r; ++k) {

        for (int i = 0; i < KERNEL_HEIGHT; ++i) {

            const auto idx_A =
                (start_row + i) * K + k;

            const auto wide_A =
                WideScalar<T>::load_single(A[idx_A]);

            for (int j = 0; j < KERNEL_WIDTH_VEC; ++j) {

                const auto idx =
                    (k * M + start_col) +
                    (j * vec_width);

                const auto _i =
                    static_cast<std::size_t>(i);

                const auto _j =
                    static_cast<std::size_t>(j);

                t[_i][_j] +=
                    wide_A *
                    WideScalar<T>::load_wide(&B[idx]);
            }
        }
    }

    for (int i = 0; i < KERNEL_HEIGHT; ++i) {

        for (int j = 0; j < KERNEL_WIDTH_VEC; ++j) {

            const auto idx =
                ((start_row + i) * M + start_col) +
                (j * vec_width);

            const auto _i =
                static_cast<std::size_t>(i);

            const auto _j =
                static_cast<std::size_t>(j);

            (
                WideScalar<T>::load_wide(&C[idx]) +
                t[_i][_j]
            ).store(&C[idx]);
        }
    }
}






template <typename T>
void matmul_kernel(
    const HostSpan<const T> A,
    const HostSpan<const T> B,
    HostSpan<T> C,
    int N,
    int K,
    int M
) {
    constexpr int KERNEL_WIDTH =
        KERNEL_WIDTH_VEC * WideScalar<T>::N;

    
    const int PAD_NH =
        round_up(N, KERNEL_HEIGHT);

    const int PAD_KH =
        round_up(K, KERNEL_HEIGHT);

    const int PAD_KW =
        round_up(K, KERNEL_WIDTH);

    const int PAD_MW =
        round_up(M, KERNEL_WIDTH);


    
    static AlignedArray<T> padded_storage_A(
        MAX_STORAGE_SIZE
    );

    static AlignedArray<T> padded_storage_B(
        MAX_STORAGE_SIZE
    );

    static AlignedArray<T> padded_storage_C(
        MAX_STORAGE_SIZE
    );

    padded_storage_A.reset();
    padded_storage_B.reset();
    padded_storage_C.reset();


    auto maybe_init_storage = [](int padded_size) {

        if (padded_size > MAX_STORAGE_SIZE) {
            return std::optional<AlignedArray<T>>(
                std::in_place,
                static_cast<std::size_t>(padded_size)
            );
        }

        return std::optional<AlignedArray<T>>(
            std::nullopt
        );
    };


    auto optional_storage_A =
        maybe_init_storage(PAD_NH * PAD_KW);

    auto optional_storage_B =
        maybe_init_storage(PAD_KH * PAD_MW);

    auto optional_storage_C =
        maybe_init_storage(PAD_NH * PAD_MW);


    T *pA =
        optional_storage_A
            ? optional_storage_A->data
            : padded_storage_A.data;

    T *pB =
        optional_storage_B
            ? optional_storage_B->data
            : padded_storage_B.data;

    T *pC =
        optional_storage_C
            ? optional_storage_C->data
            : padded_storage_C.data;


    
    
    

    for (int i = 0; i < N; ++i) {

        const auto idx_pA =
            i * PAD_KW;

        const auto idx_A =
            i * K;

        std::span<const T> src{
            &A[idx_A],
            static_cast<std::size_t>(K)
        };

        std::span<T> dst{
            &pA[idx_pA],
            static_cast<std::size_t>(K)
        };

        std::copy(
            src.begin(),
            src.end(),
            dst.begin()
        );
    }


    
    
    

    for (int i = 0; i < K; ++i) {

        const auto idx_pB =
            i * PAD_MW;

        const auto idx_B =
            i * M;

        std::span<const T> src{
            &B[idx_B],
            static_cast<std::size_t>(M)
        };

        std::span<T> dst{
            &pB[idx_pB],
            static_cast<std::size_t>(M)
        };

        std::copy(
            src.begin(),
            src.end(),
            dst.begin()
        );
    }


    
    
    

    for (
        int tile_col = 0;
        tile_col < PAD_MW;
        tile_col += TILE_STRIDE
    ) {

        for (
            int tile_row = 0;
            tile_row < PAD_NH;
            tile_row += TILE_STRIDE
        ) {

            for (
                int l = 0;
                l < PAD_KH;
                l += TILE_STRIDE
            ) {

                const auto row_bound =
                    std::min(
                        tile_row + TILE_STRIDE,
                        PAD_NH
                    );

                const auto col_bound =
                    std::min(
                        tile_col + TILE_STRIDE,
                        PAD_MW
                    );

                const auto r =
                    std::min(
                        l + TILE_STRIDE,
                        K
                    );


                for (
                    int row = tile_row;
                    row < row_bound;
                    row += KERNEL_HEIGHT
                ) {

                    for (
                        int col = tile_col;
                        col < col_bound;
                        col += KERNEL_WIDTH
                    ) {

                        tile_kernel(
                            pA,
                            pB,
                            pC,
                            row,
                            col,
                            l,
                            r,
                            PAD_KW,
                            PAD_MW
                        );
                    }
                }
            }
        }
    }


    
    
    

    for (int i = 0; i < N; ++i) {

        const auto idx_pC =
            i * PAD_MW;

        const auto idx_C =
            i * M;

        std::span<T> src{
            &pC[idx_pC],
            static_cast<std::size_t>(M)
        };

        std::span<T> dst{
            &C[idx_C],
            static_cast<std::size_t>(M)
        };

        std::copy(
            src.begin(),
            src.end(),
            dst.begin()
        );
    }
}

} 

#endif 