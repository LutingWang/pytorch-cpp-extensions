#include "../utils.hpp"

#include <ATen/AccumulateType.h>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>

__device__ __forceinline__ int msb(int val) { return 31 - __clz(val); }

template <typename scalar_t>
__global__ void stat(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
        input,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> mu,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
        sigma) {
    using acc_scalar_t = at::acc_type<scalar_t, true>;

    __shared__ int shared_n[WARP_SIZE];
    __shared__ acc_scalar_t shared_mean[WARP_SIZE];
    __shared__ acc_scalar_t shared_S[WARP_SIZE];

    int n = 0;
    acc_scalar_t mean = 0;
    acc_scalar_t S = 0;

    for (int i = threadIdx.x; i < input.size(0); i += blockDim.x) {
        n++;
        acc_scalar_t x = input[i][blockIdx.x];
        acc_scalar_t delta1 = x - mean;
        mean += delta1 / n;
        acc_scalar_t delta2 = x - mean;
        S += delta1 * delta2;
    }

    for (int i = 0; i < msb(warpSize); i++) {
        int other_n = __shfl_xor_sync(0xffffffff, n, 1 << i, warpSize);
        acc_scalar_t other_mean
            = __shfl_xor_sync(0xffffffff, mean, 1 << i, warpSize);
        acc_scalar_t other_S
            = __shfl_xor_sync(0xffffffff, S, 1 << i, warpSize);
        acc_scalar_t delta = other_mean - mean;
        acc_scalar_t factor = 1.0 / fmaxf(1.0, n + other_n);
        mean = (n * mean + other_n * other_mean) * factor;
        S += other_S + powf(delta, 2) * n * other_n * factor;
        n += other_n;
    }

    __syncthreads();
    if (threadIdx.x % warpSize == 0) {
        shared_n[threadIdx.x / warpSize] = n;
        shared_mean[threadIdx.x / warpSize] = mean;
        shared_S[threadIdx.x / warpSize] = S;
    }
    __syncthreads();

    if (threadIdx.x >= warpSize) {
        return;
    }

    if (threadIdx.x < (blockDim.x - 1) / warpSize + 1) {
        n = shared_n[threadIdx.x];
        mean = shared_mean[threadIdx.x];
        S = shared_S[threadIdx.x];
    } else {
        n = 0;
        mean = 0;
        S = 0;
    }

    for (int i = 0; i < msb(warpSize); i++) {
        int other_n = __shfl_xor_sync(0xffffffff, n, 1 << i, warpSize);
        acc_scalar_t other_mean
            = __shfl_xor_sync(0xffffffff, mean, 1 << i, warpSize);
        acc_scalar_t other_S
            = __shfl_xor_sync(0xffffffff, S, 1 << i, warpSize);
        acc_scalar_t delta = other_mean - mean;
        acc_scalar_t factor = 1.0 / fmaxf(1.0, n + other_n);
        mean = (n * mean + other_n * other_mean) * factor;
        S += other_S + powf(delta, 2) * n * other_n * factor;
        n += other_n;
    }

    if (threadIdx.x) {
        return;
    }

    mu[0][blockIdx.x] = mean;
    sigma[0][blockIdx.x] = sqrtf(S / n);
}

template <typename scalar_t>
__global__ void standardize_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
        input,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
        mu,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
        sigma,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
        output) {
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= input.size(0) || j >= input.size(1)) {
        return;
    }

    output[i][j] = (input[i][j] - mu[0][j]) / sigma[0][j];
}

template <typename scalar_t>
__global__ void standardize_cuda_backward_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
        grad,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
        output,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
        mean_grad,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
        mean_output_grad,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits>
        grad_input) {
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= output.size(0) || j >= output.size(1)) {
        return;
    }

    grad_input[i][j]
        = (grad[i][j] - mean_grad[0][j]
           - (output[i][j] * mean_output_grad[0][j]));
}

std::vector<torch::Tensor> standardize_cuda_forward(torch::Tensor input) {
    CHECK_INPUT(input);

    dim3 threads(
        WARP_SIZE
        * std::min(WARP_SIZE, (int) (input.size(0) - 1) / WARP_SIZE + 1));
    dim3 blocks(input.size(1));

    auto mu = torch::zeros({1, input.size(1)}, input.options());
    auto sigma = torch::zeros({1, input.size(1)}, input.options());

    AT_DISPATCH_FLOATING_TYPES(
        input.type(), "stat", ([&] {
            stat<scalar_t><<<blocks, threads>>>(
                input.packed_accessor32<
                    scalar_t,
                    2,
                    torch::RestrictPtrTraits>(),
                mu.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                sigma.packed_accessor32<
                    scalar_t,
                    2,
                    torch::RestrictPtrTraits>());
        }));

    threads = dim3(WARP_SIZE, WARP_SIZE);
    blocks = dim3(
        (input.size(1) - 1) / threads.x + 1,
        (input.size(0) - 1) / threads.y + 1);

    auto output = torch::zeros_like(input);

    AT_DISPATCH_FLOATING_TYPES(
        input.type(), "standardize_cuda_forward_kernel", ([&] {
            standardize_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
                input.packed_accessor32<
                    scalar_t,
                    2,
                    torch::RestrictPtrTraits>(),
                mu.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                sigma.packed_accessor32<
                    scalar_t,
                    2,
                    torch::RestrictPtrTraits>(),
                output.packed_accessor32<
                    scalar_t,
                    2,
                    torch::RestrictPtrTraits>());
        }));

    return {output, sigma};
}

torch::Tensor standardize_cuda_backward(
    torch::Tensor grad, torch::Tensor output, torch::Tensor sigma) {
    CHECK_INPUT(grad);
    CHECK_INPUT(output);
    CHECK_INPUT(sigma);

    grad = grad / sigma;
    auto mean_grad = grad.mean(0, true);
    auto mean_output_grad = (output * grad).mean(0, true);

    const dim3 threads(WARP_SIZE, WARP_SIZE);
    const dim3 blocks(
        (output.size(1) - 1) / threads.x + 1,
        (output.size(0) - 1) / threads.y + 1);

    auto grad_input = torch::zeros_like(output);

    AT_DISPATCH_FLOATING_TYPES(
        output.type(), "standardize_cuda_backward", ([&] {
            standardize_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
                grad.packed_accessor32<
                    scalar_t,
                    2,
                    torch::RestrictPtrTraits>(),
                output.packed_accessor32<
                    scalar_t,
                    2,
                    torch::RestrictPtrTraits>(),
                mean_grad.packed_accessor32<
                    scalar_t,
                    2,
                    torch::RestrictPtrTraits>(),
                mean_output_grad.packed_accessor32<
                    scalar_t,
                    2,
                    torch::RestrictPtrTraits>(),
                grad_input.packed_accessor32<
                    scalar_t,
                    2,
                    torch::RestrictPtrTraits>());
        }));

    return grad_input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "standardize_cuda_forward",
        &standardize_cuda_forward,
        "standardize cuda forward");
    m.def(
        "standardize_cuda_backward",
        &standardize_cuda_backward,
        "standardize cuda backward");
}
