from typing import Callable

import pytest
import torch

from pytorch_cpp_extensions.custom_batch_norm import (
    CustomBatchNormTuple,
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


class TestCPU:

    @pytest.mark.parametrize(
        ['forward', 'backward', 'device'],
        [
            (custom_batch_norm_forward, custom_batch_norm_backward, 'cpu'),
            (custom_batch_norm_forward, custom_batch_norm_backward, 'cuda'),
            (
                custom_batch_norm_cpp_forward,
                custom_batch_norm_cpp_backward,
                'cpu',
            ),
            (
                custom_batch_norm_cpp_forward,
                custom_batch_norm_cpp_backward,
                'cuda',
            ),
        ],
    )
    def test_forward_backward(
        self,
        forward: Callable[[torch.Tensor], CustomBatchNormTuple],
        backward: Callable[  # yapf: disable
            [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            torch.Tensor,
        ],
        device: str,
    ) -> None:
        input_ = torch.tensor(
            [[15.0], [-1.0]],
            requires_grad=True,
            device=device,
        )
        output, mu, sigma = forward(input_)
        assert torch.allclose(
            output,
            torch.tensor([[1.0], [-1.0]], device=device),
        )
        assert torch.allclose(mu, torch.tensor([[7.0]], device=device))
        assert torch.allclose(sigma, torch.tensor([[8.0]], device=device))

        output.sum().backward()
        grad = backward(torch.ones(2, 1, device=device), input_, mu, sigma)
        assert torch.allclose(grad, input_.grad)

    @pytest.mark.parametrize(
        'function_',
        [custom_batch_norm, custom_batch_norm_cpp],
    )
    def test_function(
        self,
        function_: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        input_ = torch.tensor([[15.0], [-1.0]], requires_grad=True)
        output: torch.Tensor = function_(input_)
        assert torch.allclose(output, torch.tensor([[1.0], [-1.0]]))

        output.sum().backward()
        assert torch.allclose(input_.grad, torch.zeros(2, 1))


class TestCUDA:

    def test_forward_backward(self) -> None:
        input_ = torch.tensor(
            [[15.0], [-1.0]],
            requires_grad=True,
            device='cuda',
        )
        output, mu, sigma = custom_batch_norm_cuda_forward(input_)
        assert torch.allclose(
            output, torch.tensor([[1.0], [-1.0]], device='cuda')
        )
        assert torch.allclose(mu, torch.tensor([[7.0]], device='cuda'))
        assert torch.allclose(sigma, torch.tensor([[8.0]], device='cuda'))

        grad = custom_batch_norm_cuda_backward(
            torch.ones(2, 1, device='cuda'),
            input_,
            mu,
            sigma,
        )
        assert torch.allclose(grad, torch.zeros(2, 1, device='cuda'))

    def test_function(self) -> None:
        input_ = torch.tensor(
            [[15.0], [-1.0]],
            requires_grad=True,
            device='cuda',
        )
        output: torch.Tensor = custom_batch_norm_cuda(input_)
        assert torch.allclose(
            output,
            torch.tensor([[1.0], [-1.0]], device='cuda'),
        )

        output.sum().backward()
        assert torch.allclose(input_.grad, torch.zeros(2, 1, device='cuda'))
