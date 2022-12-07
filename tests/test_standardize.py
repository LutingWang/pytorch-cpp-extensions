from typing import Callable

import pytest
import torch
from devices import mark_parameterize_device, mark_skipif_cuda_is_unavailable

from pytorch_cpp_extensions.standardize import (
    standardize,
    standardize_backward,
    standardize_cpp,
    standardize_cpp_backward,
    standardize_cpp_forward,
    standardize_cuda,
    standardize_cuda_backward,
    standardize_cuda_forward,
    standardize_forward,
)


def mark_parameterize_size() -> pytest.mark.parametrize:
    return pytest.mark.parametrize('size', [(80, 4), (2000, 4)])


@mark_parameterize_device()
class TestCustomBatchNorm:

    def test_forward(self, device: str) -> None:
        input_ = torch.tensor([[15.0], [-1.0]], device=device)
        output, sigma = standardize_forward(input_)
        assert torch.allclose(
            output,
            torch.tensor([[1.0], [-1.0]], device=device),
        )
        assert torch.allclose(sigma, torch.tensor([[8.0]], device=device))

    def test_backward(self, device: str) -> None:
        grad = standardize_backward(
            torch.ones(2, 1, device=device),
            torch.tensor([[1.0], [-1.0]], device=device),
            torch.tensor([[8.0]], device=device),
        )
        assert torch.allclose(grad, torch.zeros(2, 1, device=device))


@mark_parameterize_device()
@mark_parameterize_size()
class TestCustomBatchNormCpp:

    def test_forward(self, device: str, size: tuple[int, int]) -> None:
        input_ = torch.rand(size, device=device)
        output, sigma = standardize_forward(input_)
        output_cpp, sigma_cpp = standardize_cpp_forward(input_)
        assert torch.allclose(output, output_cpp)
        assert torch.allclose(sigma, sigma_cpp)

    def test_backward(self, device: str, size: tuple[int, int]) -> None:
        input_ = torch.rand(size, device=device)
        output, sigma = standardize_forward(input_)
        grad = standardize_backward(
            torch.ones(size, device=device),
            output,
            sigma,
        )
        grad_cpp = standardize_cpp_backward(
            torch.ones(size, device=device),
            output,
            sigma,
        )
        assert torch.allclose(grad, grad_cpp)


@mark_skipif_cuda_is_unavailable()
@mark_parameterize_size()
class TestCustomBatchNormCuda:

    def test_forward(self, size: tuple[int, int]) -> None:
        input_ = torch.rand(size, device='cuda')
        output, sigma = standardize_forward(input_)
        output_cpp, sigma_cpp = standardize_cuda_forward(input_)
        assert torch.allclose(output, output_cpp, atol=1e-5)
        assert torch.allclose(sigma, sigma_cpp)

    def test_backward(self, size: tuple[int, int]) -> None:
        input_ = torch.rand(size, device='cuda')
        output, sigma = standardize_forward(input_)
        grad = standardize_backward(
            torch.ones(size, device='cuda'),
            output,
            sigma,
        )
        grad_cpp = standardize_cuda_backward(
            torch.ones(size, device='cuda'),
            output,
            sigma,
        )
        assert torch.allclose(grad, grad_cpp)


@pytest.mark.parametrize(
    'function_',
    [
        standardize,
        standardize_cpp,
        pytest.param(
            standardize_cuda,
            marks=mark_skipif_cuda_is_unavailable(),
        ),
    ],
)
def test_grad_check(function_: Callable[[torch.Tensor], torch.Tensor]) -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_ = torch.tensor(
        [[0.4314, 0.1143], [0.3738, 0.5579], [0.1193, 0.7181],
         [0.4514, 0.9623]],
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )
    assert torch.autograd.gradcheck(function_, input_, eps=1e-3)
