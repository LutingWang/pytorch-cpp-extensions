import torch

# isort: off


def custom_batch_norm_cpp_forward(
    input_: torch.Tensor,
) -> torch.Tensor:
    pass


def custom_batch_norm_cpp_backward(
    grad: torch.Tensor,
    input_: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    pass
