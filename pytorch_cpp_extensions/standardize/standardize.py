__all__ = [
    'CustomBatchNormTuple',
    'standardize_forward',
    'standardize_backward',
    'standardize',
    'standardize_cpp_forward',
    'standardize_cpp_backward',
    'standardize_cpp',
    'standardize_cuda_forward',
    'standardize_cuda_backward',
    'standardize_cuda',
]

from typing import (
    TYPE_CHECKING,
    Callable,
    NamedTuple,
    Protocol,
    Sequence,
    cast,
)

import einops
import torch

from ..utils import Implementation


class CustomBatchNormTuple(NamedTuple):
    output: torch.Tensor  # normalized features
    sigma: torch.Tensor  # standard deviation


def standardize_forward(input_: torch.Tensor) -> CustomBatchNormTuple:
    """Standardize forward function in Python.

    Args:
        input_: :math:`\\mathbf{x} \\in \\mathcal{R}^{m \\times d}`

    Returns:
        :math:`\\mathbf{z} \\in \\mathcal{R}^{m \\times d}` and
        :math:`\\sigma \\in \\mathcal{R}^{1 \\times d}`,
    """
    mu = einops.reduce(input_, 'm d -> 1 d', 'mean')
    output = input_ - mu
    sigma = einops.reduce(output**2, 'm d -> 1 d', 'mean')**0.5
    output = output / sigma
    return CustomBatchNormTuple(output, sigma)


def standardize_backward(
    grad: torch.Tensor,
    output: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """Standardize backward function in Python.

    Args:
        grad: :math:`(m, d)`
        output: :math:`(m, d)`
        sigma: :math:`(1, d)`

    Returns:
        :math:`(m, d)` grad for input
    """
    grad = grad / sigma
    mean_grad = einops.reduce(grad, 'm d -> 1 d', 'mean')
    mean_output_grad = einops.reduce(output * grad, 'm d -> 1 d', 'mean')
    grad_input = grad - mean_grad - output * mean_output_grad
    return grad_input


class CustomBatchNormFunctionMetaProto(Protocol):
    apply: Callable[[torch.Tensor], torch.Tensor]


def standardize_function_factory(
    name: Implementation,
) -> CustomBatchNormFunctionMetaProto:
    forward: Callable[[torch.Tensor], Sequence[torch.Tensor]]
    backward: Callable[  # yapf: disable
        [torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor,
    ]
    if name is Implementation.PYTHON:
        forward = standardize_forward
        backward = standardize_backward
    elif name is Implementation.CPP:
        forward = standardize_cpp_forward
        backward = standardize_cpp_backward
    elif name is Implementation.CUDA:
        forward = standardize_cuda_forward
        backward = standardize_cuda_backward

    class CustomBatchNormFunction(torch.autograd.Function):

        @staticmethod
        def forward(
            ctx: torch.autograd.function.BackwardCFunction,
            input_: torch.Tensor,
        ) -> torch.Tensor:
            output, sigma = forward(input_)
            ctx.save_for_backward(output, sigma)
            return output

        @staticmethod
        def backward(
            ctx: torch.autograd.function.BackwardCFunction,
            grad: torch.Tensor,
        ) -> torch.Tensor:
            output, sigma = ctx.saved_tensors
            return backward(grad.contiguous(), output, sigma)

    return cast(
        CustomBatchNormFunctionMetaProto,
        CustomBatchNormFunction,
    )


standardize = standardize_function_factory(Implementation.PYTHON, ).apply

try:
    from .standardize_cpp import (
        standardize_cpp_backward,
        standardize_cpp_forward,
    )
    standardize_cpp = standardize_function_factory(Implementation.CPP, ).apply
except ImportError:
    if not TYPE_CHECKING:
        standardize_cpp_backward = None
        standardize_cpp_forward = None
        standardize_cpp = None

try:
    from .standardize_cuda import (
        standardize_cuda_backward,
        standardize_cuda_forward,
    )
    standardize_cuda = standardize_function_factory(
        Implementation.CUDA,
    ).apply
except ImportError:
    if not TYPE_CHECKING:
        standardize_cuda_backward = None
        standardize_cuda_forward = None
        standardize_cuda = None
