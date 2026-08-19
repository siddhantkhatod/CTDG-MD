from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class ContinuousTimeEncoding(nn.Module):
    """Learnable Fourier features of elapsed physical time."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        if dimension < 4 or dimension % 2:
            raise ValueError("time encoding dimension must be even and >= 4")
        half = dimension // 2
        base = torch.logspace(-4, 0, half)
        self.log_frequency = nn.Parameter(torch.log(base))
        self.phase = nn.Parameter(torch.zeros(half))

    def forward(self, delta_t: Tensor) -> Tensor:
        angle = delta_t.unsqueeze(-1) * torch.exp(self.log_frequency) + self.phase
        return torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)


class PatchFourierTransform(nn.Module):
    """Non-stationary patch Fourier filter with time-conditioned spectral gates.

    A separate gate is inferred for each temporal patch from its spectrum and center
    time. Complex-valued learned filters operate on rFFT coefficients, followed by
    irFFT and a residual normalization. This is analysis-time temporal encoding: all
    frames inside a patch are visible to that patch.
    """

    def __init__(self, hidden_dim: int, patch_size: int, time_dim: int, dropout: float) -> None:
        super().__init__()
        if patch_size < 2:
            raise ValueError("patch_size must be >= 2")
        self.patch_size = patch_size
        self.frequencies = patch_size // 2 + 1
        self.time_encoder = ContinuousTimeEncoding(time_dim)
        self.gate = nn.Sequential(
            nn.Linear(self.frequencies + time_dim, 2 * self.frequencies),
            nn.SiLU(),
            nn.Linear(2 * self.frequencies, self.frequencies),
            nn.Sigmoid(),
        )
        scale = 1.0 / math.sqrt(hidden_dim)
        self.real_weight = nn.Parameter(torch.ones(self.frequencies, hidden_dim))
        self.imag_weight = nn.Parameter(torch.randn(self.frequencies, hidden_dim) * scale)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden: Tensor, times: Tensor, time_origin: Tensor | None = None) -> Tensor:
        if hidden.ndim != 3:
            raise ValueError("hidden must be [T, N, H]")
        length, nodes, channels = hidden.shape
        if times.shape != (length,):
            raise ValueError("times must have shape [T]")
        if time_origin is None:
            time_origin = times[0]
        patches = math.ceil(length / self.patch_size)
        padded_length = patches * self.patch_size
        if padded_length != length:
            padding = hidden.new_zeros((padded_length - length, nodes, channels))
            work = torch.cat([hidden, padding], dim=0)
            if length > 1:
                dt = times[-1] - times[-2]
            else:
                dt = times.new_tensor(1.0)
            extra = times[-1] + dt * torch.arange(
                1, padded_length - length + 1, device=times.device, dtype=times.dtype
            )
            work_times = torch.cat([times, extra])
        else:
            work, work_times = hidden, times

        work = work.reshape(patches, self.patch_size, nodes, channels)
        spectra = torch.fft.rfft(work, dim=1, norm="ortho")
        power = torch.log1p(spectra.abs().mean(dim=(2, 3)))
        centers = work_times.reshape(patches, self.patch_size).mean(dim=1)
        center_delta = centers - time_origin
        time_features = self.time_encoder(center_delta)
        gates = self.gate(torch.cat([power, time_features], dim=-1))
        complex_weight = torch.complex(self.real_weight, self.imag_weight)
        filtered = spectra * complex_weight[None, :, None, :] * gates[:, :, None, None]
        reconstructed = torch.fft.irfft(
            filtered, n=self.patch_size, dim=1, norm="ortho"
        ).reshape(padded_length, nodes, channels)[:length]
        return self.norm(hidden + self.dropout(reconstructed))


class DecayedNodeMemory(nn.Module):
    """Continuous-time exponential decay followed by a GRU node update."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.log_decay = nn.Parameter(torch.full((hidden_dim,), -4.0))
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, observation: Tensor, memory: Tensor, delta_t: Tensor) -> Tensor:
        rate = torch.nn.functional.softplus(self.log_decay)
        decay = torch.exp(-delta_t.reshape(1, 1) * rate.reshape(1, -1))
        return self.gru(observation, memory * decay)
