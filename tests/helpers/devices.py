import _pytest.mark
import pytest
import torch


def mark_skipif_cuda_is_unavailable() -> pytest.mark.skipif:
    return pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason='requires CUDA support',
    )


def param_cuda() -> _pytest.mark.ParameterSet:
    return pytest.param('cuda', marks=mark_skipif_cuda_is_unavailable())


def mark_parameterize_device() -> pytest.mark.parametrize:
    return pytest.mark.parametrize('device', ['cpu', param_cuda()])
