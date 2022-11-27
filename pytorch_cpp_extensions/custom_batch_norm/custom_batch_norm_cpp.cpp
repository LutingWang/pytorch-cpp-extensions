#include <vector>
#include <torch/extension.h>


std::vector<torch::Tensor> custom_batch_norm_cpp_forward(
    torch::Tensor input
) {
    auto mu = input.mean(0, true);
    auto output = input - mu;
    auto sigma = output.pow(2).mean(0, true).pow(0.5);
    output = output / sigma;
    return {output, mu, sigma};
}


torch::Tensor custom_batch_norm_cpp_backward(
    torch::Tensor grad,
    torch::Tensor input,
    torch::Tensor mu,
    torch::Tensor sigma
) {
    auto m = grad.size(0);
    return grad / m * ((m - 1) / sigma - (input - mu).pow(2) / sigma.pow(3));
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_batch_norm_cpp_forward", &custom_batch_norm_cpp_forward, "custom batch norm cpp forward");
    m.def("custom_batch_norm_cpp_backward", &custom_batch_norm_cpp_backward, "custom batch norm cpp backward");
}