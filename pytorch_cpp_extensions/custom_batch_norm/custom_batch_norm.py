__all__ = [
    'CustomBatchNormTuple',
    'custom_batch_norm_forward',
    'custom_batch_norm_backward',
    'custom_batch_norm',
]

from typing import Callable, NamedTuple, Protocol, cast

import einops
import torch

from ..utils import Implementation


class CustomBatchNormTuple(NamedTuple):
    output: torch.Tensor  # normalized features
    mu: torch.Tensor  # mean
    sigma: torch.Tensor  # standard deviation


def custom_batch_norm_forward(input_: torch.Tensor) -> CustomBatchNormTuple:
    """Custom batch norm forward function in Python.

    Args:
        input_: :math:`(m, d)`

    Returns:
        :math:`(m, d)` output, and
        :math:`\\mu, \\sigma \\in \\mathcal{R}^{1 \\times d}`,
    """
    mu = einops.reduce(input_, 'm d -> 1 d', 'mean')
    output = input_ - mu
    sigma = einops.reduce(output**2, 'm d -> 1 d', 'mean')**0.5
    output = output / sigma
    return CustomBatchNormTuple(output, mu, sigma)


def custom_batch_norm_backward(
    grad: torch.Tensor,
    input_: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """Custom batch norm backward function in Python.

    Args:
        grad: :math:`(m, d)`
        input_: :math:`(m, d)`
        mu: :math:`(1, d)`
        sigma: :math:`(1, d)`

    Returns:
        :math:`(m, d)` grad for ``input_``
    """
    m = grad.shape[0]
    return grad / m * ((m - 1) / sigma - (input_ - mu)**2 / sigma**3)


class CustomBatchNormFunctionMetaProto(Protocol):
    apply: Callable[[torch.Tensor], torch.Tensor]


def custom_batch_norm_function_factory(
    name: Implementation,
) -> CustomBatchNormFunctionMetaProto:
    if name is Implementation.PYTHON:
        forward = custom_batch_norm_forward
        backward = custom_batch_norm_backward
    elif name is Implementation.CPP:
        forward = custom_batch_norm_cpp_forward
        backward = custom_batch_norm_cpp_backward
    elif name is Implementation.CUDA:
        forward = custom_batch_norm_cuda_forward
        backward = custom_batch_norm_cuda_backward

    class CustomBatchNormFunction(torch.autograd.Function):

        @staticmethod
        def forward(
            ctx: torch.autograd.function.BackwardCFunction,
            input_: torch.Tensor,
        ) -> torch.Tensor:
            output, mu, sigma = forward(input_)
            ctx.save_for_backward(input_, mu, sigma)
            return output

        @staticmethod
        def backward(
            ctx: torch.autograd.function.BackwardCFunction,
            grad: torch.Tensor,
        ) -> torch.Tensor:
            input_, mu, sigma = ctx.saved_tensors
            return backward(grad.contiguous(), input_, mu, sigma)

    return cast(
        CustomBatchNormFunctionMetaProto,
        CustomBatchNormFunction,
    )


custom_batch_norm = custom_batch_norm_function_factory(
    Implementation.PYTHON,
).apply

try:
    from .custom_batch_norm_cpp import (
        custom_batch_norm_cpp_backward,
        custom_batch_norm_cpp_forward,
    )
    __all__.extend([
        'custom_batch_norm_cpp_forward',
        'custom_batch_norm_cpp_backward',
    ])
    custom_batch_norm_cpp = custom_batch_norm_function_factory(
        Implementation.CPP,
    ).apply
    __all__.append('custom_batch_norm_cpp')
except ImportError as e:
    print(e)
    pass

try:
    from .custom_batch_norm_cuda import (
        custom_batch_norm_cuda_backward,
        custom_batch_norm_cuda_forward,
    )
    __all__.extend([
        'custom_batch_norm_cuda_forward',
        'custom_batch_norm_cuda_backward',
    ])
    custom_batch_norm_cuda = custom_batch_norm_function_factory(
        Implementation.CUDA,
    ).apply
    __all__.append('custom_batch_norm_cuda')
except ImportError:
    pass
