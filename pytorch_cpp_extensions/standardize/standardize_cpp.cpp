#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> standardize_cpp_forward(torch::Tensor input) {
    auto mu = input.mean(0, true);
    auto output = input - mu;
    auto sigma = output.pow(2).mean(0, true).pow(0.5);
    output = output / sigma;
    return {output, sigma};
}

torch::Tensor standardize_cpp_backward(
    torch::Tensor grad, torch::Tensor output, torch::Tensor sigma) {
    grad = grad / sigma;
    auto mean_grad = grad.mean(0, true);
    auto mean_output_grad = (output * grad).mean(0, true);
    auto grad_input = grad - mean_grad - output * mean_output_grad;
    return grad_input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "standardize_cpp_forward",
        &standardize_cpp_forward,
        "standardize cpp forward");
    m.def(
        "standardize_cpp_backward",
        &standardize_cpp_backward,
        "standardize cpp backward");
}
