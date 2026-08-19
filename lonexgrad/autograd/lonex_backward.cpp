


#include "autograd/lonex_backward.h"

#include <lonexgrad/lonex_autograd.h>
#include <lonexgrad/lonex_exception.h>
#include <lonexgrad/lonex_grad_mode.h>
#include <lonexgrad/lonex_scalar.h>
#include <lonexgrad/lonex_tensor.h>

#include "autograd/lonex_dag.h"

#include <cassert>
#include <format>
#include <optional>
#include <ranges>

namespace lonexgrad::autograd {

void calc_grad_input(Tensor &tensor, bool retain_graph) {
    if (tensor.ctx_->grad_func) {
        if (!tensor.ctx_->grad) {
            LONEXGRAD_ERROR("Gradient was not propagated to this tensor");
        }
        GradList grad_inputs = tensor.ctx_->grad_func(tensor.ctx_->storage, *tensor.ctx_->grad);
        if (grad_inputs.size() > tensor.ctx_->parents.size()) {
            
            if (tensor.ctx_->parents.empty()) {
                LONEXGRAD_EXCEPTION(
                    std::format(
                        "Backward function {:s} returned {:d} gradients but has no saved parents.\n\tIf you are "
                        "wanting to "
                        "perform .backward() twice, try keeping the computation graph by using .backward(true)",
                        tensor.ctx_->grad_func_name,
                        grad_inputs.size(),
                        tensor.ctx_->parents.size()
                    )
                );
            } else {
                LONEXGRAD_ERROR(
                    std::format(
                        "Backward function {:s} returned {:d} gradients, but only only takes {:d} inputs",
                        tensor.ctx_->grad_func_name,
                        grad_inputs.size(),
                        tensor.ctx_->parents.size()
                    )
                );
            }
        }
        for (int i : std::views::iota(0, grad_inputs.size())) {
            if (grad_inputs[i]) {
                tensor.ctx_->parents[i].add_grad(grad_inputs[i].value());
            }
        }
        
        if (!retain_graph) {
            tensor.ctx_->parents.clear();
            tensor.ctx_->storage.clear();
        }
    }
}

void backward(Tensor &tensor, const Tensor &grad, bool retain_graph) {
    assert(is_float_dtype(tensor.dtype()));
    assert(tensor.device() == grad.device());
    assert(tensor.shape() == grad.shape());
    assert(tensor.dtype() == grad.dtype());

    
    tensor.ctx_->grad = grad;

    
    TensorList dag = autograd::build_dag(tensor);

    
    
    const autograd::NoGradGuard guard;
    for (auto &t : std::ranges::reverse_view(dag)) {
        calc_grad_input(t, retain_graph);
        t.apply_grad_hook();
    }
}

}    
