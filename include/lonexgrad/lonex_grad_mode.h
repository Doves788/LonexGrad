


#ifndef lonexgrad_AUTOGRAD_GRAD_MODE_H_
#define lonexgrad_AUTOGRAD_GRAD_MODE_H_

#include <lonexgrad/export.h>

namespace lonexgrad::autograd {


class LONEXGRAD_EXPORT GradMode {
public:
    GradMode() = delete;
    static inline bool is_enabled() {
        return grad_mode;
    }
    static inline void set_enabled(bool enabled) {
        grad_mode = enabled;
    }

private:
    inline static thread_local bool grad_mode = true;    // NOLINT(*avoid-non-const-global-variables)
};


class LONEXGRAD_EXPORT NoGradGuard {
public:
    NoGradGuard()
        : prev_mode(GradMode::is_enabled()) {
        GradMode::set_enabled(false);
    }
    ~NoGradGuard() {
        GradMode::set_enabled(prev_mode);
    }

    
    NoGradGuard(const NoGradGuard &) = delete;
    NoGradGuard(NoGradGuard &&) = delete;
    NoGradGuard &operator=(const NoGradGuard &) = delete;
    NoGradGuard &operator=(NoGradGuard &&) = delete;

private:
    bool prev_mode;
};

}    

#endif    
