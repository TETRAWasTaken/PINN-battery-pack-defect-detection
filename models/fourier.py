import torch
import torch.nn as nn
import numpy as np

class FourierFeatureEncoding(nn.Module):
    """
    Encodes 4D coordinates (x, y, z, t) into high-frequency Fourier features.
    Maps input coordinates using random Gaussian projection frequencies.
    """
    def __init__(self, in_features: int = 4, mapping_size: int = 32, scale: float = 5.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(in_features, mapping_size) * scale, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = 2 * np.pi * torch.matmul(x, self.B)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
