#include <torch/extension.h>


torch::Tensor custom_batch_norm_forward(
    torch::Tensor input,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    torch::Tensor weight,
    torch::Tensor bias,
    bool training,
    float momentum,
    float eps
) {
    return input;
}

torch::Tensor custom_batch_norm_backward(
    torch::Tensor grad
) {
    return grad;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_batch_norm_forward", &custom_batch_norm_forward, "custom batch norm forward");
    m.def("custom_batch_norm_backward", &custom_batch_norm_backward, "custom batch norm backward");
}
