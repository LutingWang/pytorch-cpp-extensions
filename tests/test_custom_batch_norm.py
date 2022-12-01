import pytest
import torch

from pytorch_cpp_extensions.custom_batch_norm import (
    custom_batch_norm,
    custom_batch_norm_backward,
    custom_batch_norm_cpp,
    custom_batch_norm_cpp_backward,
    custom_batch_norm_cpp_forward,
    custom_batch_norm_cuda,
    custom_batch_norm_cuda_backward,
    custom_batch_norm_cuda_forward,
    custom_batch_norm_forward,
)

M = 100
D = 256


@pytest.mark.parametrize(
    'device',
    [
        'cpu',
        pytest.param(
            'cuda',
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason='requires CUDA support',
            ),
        ),
    ],
)
class TestCustomBatchNorm:

    def test_forward(self, device: str) -> None:
        input_ = torch.tensor([[15.0], [-1.0]], device=device)
        output, mu, sigma = custom_batch_norm_forward(input_)
        assert torch.allclose(
            output,
            torch.tensor([[1.0], [-1.0]], device=device),
        )
        assert torch.allclose(mu, torch.tensor([[7.0]], device=device))
        assert torch.allclose(sigma, torch.tensor([[8.0]], device=device))

    def test_backward(self, device: str) -> None:
        grad = custom_batch_norm_backward(
            torch.ones(2, 1, device=device),
            torch.tensor([[15.0], [-1.0]], device=device),
            torch.tensor([[7.0]], device=device),
            torch.tensor([[8.0]], device=device),
        )
        assert torch.allclose(grad, torch.zeros(2, 1, device=device))

    def test_function(self, device: str) -> None:
        input_ = torch.tensor(
            [[15.0], [-1.0]],
            device=device,
            requires_grad=True,
        )
        output = custom_batch_norm(input_)
        assert torch.allclose(
            output,
            torch.tensor([[1.0], [-1.0]], device=device),
        )

        output.sum().backward()
        assert torch.allclose(input_.grad, torch.zeros(2, 1, device=device))


@pytest.mark.parametrize(
    'device',
    [
        'cpu',
        pytest.param(
            'cuda',
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason='requires CUDA support',
            ),
        ),
    ],
)
class TestCustomBatchNormCpp:

    def test_forward(self, device: str) -> None:
        input_ = torch.rand(M, D, device=device)
        output, mu, sigma = custom_batch_norm_forward(input_)
        output_cpp, mu_cpp, sigma_cpp = custom_batch_norm_cpp_forward(input_)
        assert torch.allclose(output, output_cpp)
        assert torch.allclose(mu, mu_cpp)
        assert torch.allclose(sigma, sigma_cpp)

    def test_backward(self, device: str) -> None:
        input_ = torch.rand(M, D, device=device)
        _, mu, sigma = custom_batch_norm_forward(input_)
        grad = custom_batch_norm_backward(
            torch.ones(M, D, device=device),
            input_,
            mu,
            sigma,
        )
        grad_cpp = custom_batch_norm_cpp_backward(
            torch.ones(M, D, device=device),
            input_,
            mu,
            sigma,
        )
        assert torch.allclose(grad, grad_cpp)

    def test_function(self, device: str) -> None:
        input_ = torch.rand(M, D, device=device, requires_grad=True)
        input_cpp = input_.clone().detach().requires_grad_()
        output = custom_batch_norm(input_)
        output_cpp = custom_batch_norm_cpp(input_cpp)
        assert torch.allclose(output, output_cpp)

        output.sum().backward()
        output_cpp.sum().backward()
        assert torch.allclose(input_.grad, input_cpp.grad)


class TestCustomBatchNormCuda:

    def test_forward(self) -> None:
        input_ = torch.rand(M, D, device='cuda')
        output, mu, sigma = custom_batch_norm_forward(input_)
        output_cpp, mu_cpp, sigma_cpp = custom_batch_norm_cuda_forward(input_)
        assert torch.allclose(output, output_cpp, atol=1e-5)
        assert torch.allclose(mu, mu_cpp)
        assert torch.allclose(sigma, sigma_cpp)

    def test_backward(self) -> None:
        input_ = torch.rand(M, D, device='cuda')
        _, mu, sigma = custom_batch_norm_forward(input_)
        grad = custom_batch_norm_backward(
            torch.ones(M, D, device='cuda'),
            input_,
            mu,
            sigma,
        )
        grad_cpp = custom_batch_norm_cuda_backward(
            torch.ones(M, D, device='cuda'),
            input_,
            mu,
            sigma,
        )
        assert torch.allclose(grad, grad_cpp)

    def test_function(self) -> None:
        input_ = torch.rand(M, D, device='cuda', requires_grad=True)
        input_cuda = input_.clone().detach().requires_grad_()
        output = custom_batch_norm(input_)
        output_cuda = custom_batch_norm_cuda(input_cuda)
        assert torch.allclose(output, output_cuda, atol=1e-5)

        output.sum().backward()
        output_cuda.sum().backward()
        assert torch.allclose(input_.grad, input_cuda.grad)
