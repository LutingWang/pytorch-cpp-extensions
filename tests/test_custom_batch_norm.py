import torch

import custom_batch_norm


def test_custom_batch_norm() -> None:
    input_ = torch.rand(3, 4)
    running_mean = torch.rand(3, 4)
    running_var = torch.rand(3, 4)
    weight = torch.rand(3, 4)
    bias = torch.rand(3, 4)
    assert torch.allclose(
        input_,
        custom_batch_norm.custom_batch_norm_forward(
            input_,
            running_mean,
            running_var,
            weight,
            bias,
            False,
            0.5,
            1e-5,
        ),
    )
