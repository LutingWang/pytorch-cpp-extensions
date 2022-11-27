import torch


def custom_batch_norm_forward(
    input_: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    training: bool,
    momentum: float,
    eps: float,
) -> torch.Tensor:
    pass
