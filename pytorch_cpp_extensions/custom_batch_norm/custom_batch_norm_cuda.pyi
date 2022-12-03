import torch

# isort: off


def custom_batch_norm_cuda_forward(input_: torch.Tensor) -> list[torch.Tensor]:
    pass


def custom_batch_norm_cuda_backward(
    grad: torch.Tensor,
    output: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    pass
