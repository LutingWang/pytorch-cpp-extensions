import torch

# isort: off


def standardize_cuda_forward(input_: torch.Tensor) -> list[torch.Tensor]:
    pass


def standardize_cuda_backward(
    grad: torch.Tensor,
    output: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    pass
