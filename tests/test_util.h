#include <lonexgrad/lonex_device.h>
#include <lonexgrad/lonex_scalar.h>
#include <lonexgrad/lonex_tensor.h>

template <typename T, typename F, typename... Args>
void runner_single_type(const F &f, Args... args) {
    f.template operator()<T>(lonexgrad::kCPU, args...);
#ifdef LONEXGRAD_CUDA
    f.template operator()<T>(lonexgrad::kCUDA, args...);
#endif
}

template <typename F, typename... Args>
void runner_boolean(F &&f, Args... args) {
    runner_single_type<bool>(std::forward<F>(f), args...);
}
template <typename F, typename... Args>
void runner_integral(F &&f, Args... args) {
    runner_single_type<lonexgrad::to_ctype_t<lonexgrad::kU8>>(std::forward<F>(f), args...);
    runner_single_type<lonexgrad::to_ctype_t<lonexgrad::kI16>>(std::forward<F>(f), args...);
    runner_single_type<lonexgrad::to_ctype_t<lonexgrad::kI32>>(std::forward<F>(f), args...);
    runner_single_type<lonexgrad::to_ctype_t<lonexgrad::kI64>>(std::forward<F>(f), args...);
}
template <typename F, typename... Args>
void runner_signed_integral(F &&f, Args... args) {
    runner_single_type<lonexgrad::to_ctype_t<lonexgrad::kI16>>(std::forward<F>(f), args...);
    runner_single_type<lonexgrad::to_ctype_t<lonexgrad::kI32>>(std::forward<F>(f), args...);
    runner_single_type<lonexgrad::to_ctype_t<lonexgrad::kI64>>(std::forward<F>(f), args...);
}
template <typename F, typename... Args>
void runner_floating_point(F &&f, Args... args) {
    runner_single_type<lonexgrad::to_ctype_t<lonexgrad::kF32>>(std::forward<F>(f), args...);
    runner_single_type<lonexgrad::to_ctype_t<lonexgrad::kF64>>(std::forward<F>(f), args...);
}
template <typename F, typename... Args>
void runner_all(F &&f, Args... args) {
    runner_boolean(std::forward<F>(f), args...);
    runner_integral(std::forward<F>(f), args...);
    runner_floating_point(std::forward<F>(f), args...);
}
template <typename F, typename... Args>
void runner_all_except_bool(F &&f, Args... args) {
    runner_integral(std::forward<F>(f), args...);
    runner_floating_point(std::forward<F>(f), args...);
}
