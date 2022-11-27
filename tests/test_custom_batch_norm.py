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
    custom_batch_norm_forward,
)


class TestCustomBatchNorm:

    @pytest.mark.parametrize(
        ['forward', 'backward'],
        [
            (custom_batch_norm_forward, custom_batch_norm_backward),
            (custom_batch_norm_cpp_forward, custom_batch_norm_cpp_backward),
        ],
    )
    def test_forward_backward(
        self,
        forward: Callable[[torch.Tensor], CustomBatchNormTuple],
        backward: Callable[  # yapf: disable
            [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            torch.Tensor,
        ],
    ) -> None:
        input_ = torch.tensor([[15.0], [-1.0]], requires_grad=True)
        output, mu, sigma = forward(input_)
        assert torch.allclose(output, torch.tensor([[1.0], [-1.0]]))
        assert torch.allclose(mu, torch.tensor([[7.0]]))
        assert torch.allclose(sigma, torch.tensor([[8.0]]))

        output.sum().backward()
        grad = backward(torch.ones(2, 1), input_, mu, sigma)
        assert torch.allclose(grad, input_.grad)

    @pytest.mark.parametrize(
        'function',
        [custom_batch_norm, custom_batch_norm_cpp],
    )
    def test_function(
        self,
        function: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        input_ = torch.tensor([[15.0], [-1.0]], requires_grad=True)
        output: torch.Tensor = function(input_)
        assert torch.allclose(output, torch.tensor([[1.0], [-1.0]]))

        output.sum().backward()
        assert torch.allclose(input_.grad, torch.zeros(2, 1))
