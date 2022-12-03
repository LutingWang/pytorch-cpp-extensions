import torch

# isort: off


def custom_batch_norm_cpp_forward(input_: torch.Tensor) -> list[torch.Tensor]:
    pass


def custom_batch_norm_cpp_backward(
    grad: torch.Tensor,
    output: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    pass
