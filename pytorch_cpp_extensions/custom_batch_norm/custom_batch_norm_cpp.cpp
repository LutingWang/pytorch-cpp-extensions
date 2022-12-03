#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> custom_batch_norm_cpp_forward(torch::Tensor input) {
    auto mu = input.mean(0, true);
    auto output = input - mu;
    auto sigma = output.pow(2).mean(0, true).pow(0.5);
    output = output / sigma;
    return {output, sigma};
}

torch::Tensor custom_batch_norm_cpp_backward(
    torch::Tensor grad, torch::Tensor output, torch::Tensor sigma) {
    grad = grad / sigma;
    auto mean_grad = grad.mean(0, true);
    auto mean_output_grad = (output * grad).mean(0, true);
    auto grad_input = grad - mean_grad - output * mean_output_grad;
    return grad_input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "custom_batch_norm_cpp_forward",
        &custom_batch_norm_cpp_forward,
        "custom batch norm cpp forward");
    m.def(
        "custom_batch_norm_cpp_backward",
        &custom_batch_norm_cpp_backward,
        "custom batch norm cpp backward");
}
