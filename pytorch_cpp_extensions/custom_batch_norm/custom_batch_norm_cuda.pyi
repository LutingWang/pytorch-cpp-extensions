import torch

# isort: off


def custom_batch_norm_cuda_forward(input_: torch.Tensor) -> torch.Tensor:
    pass


def custom_batch_norm_cuda_backward(
    grad: torch.Tensor,
    input_: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    pass
