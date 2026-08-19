import torch

from ctdg_md.temporal import ContinuousTimeEncoding, DecayedNodeMemory, PatchFourierTransform


def test_irregular_continuous_time_encoding_and_memory_gradients():
    encoding = ContinuousTimeEncoding(8)
    times = torch.tensor([0.0, 0.25, 1.75])
    features = encoding(times)
    assert features.shape == (3, 8)
    assert not torch.allclose(features[1], features[2])
    memory = DecayedNodeMemory(12)
    observation = torch.randn(5, 12, requires_grad=True)
    state = torch.randn(5, 12, requires_grad=True)
    output = memory(observation, state, torch.tensor(2.0))
    output.square().mean().backward()
    assert observation.grad is not None
    assert state.grad is not None


def test_patch_fourier_shape_nonstationarity_and_gradient():
    torch.manual_seed(2)
    layer = PatchFourierTransform(16, patch_size=4, time_dim=8, dropout=0.0)
    hidden = torch.randn(7, 5, 16, requires_grad=True)
    times = torch.tensor([0.0, 0.2, 0.7, 1.1, 3.0, 3.3, 5.0])
    output = layer(hidden, times, time_origin=torch.tensor(0.0))
    assert output.shape == hidden.shape
    assert torch.isfinite(output).all()
    output.sum().backward()
    assert hidden.grad is not None
