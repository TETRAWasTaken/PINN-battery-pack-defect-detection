import torch
import torch.nn as nn
import numpy as np
from typing import Optional

class FourierFeatureEncoding(nn.Module):
    """
    Encodes coordinates (e.g., 6D [x, y, z, t, I, SOH] or 4D [x, y, z, t]) into Fourier features.
    Supports:
      - Isotropic scaling (single scale parameter)
      - Anisotropic multi-scale (separate spatial, temporal, and conditioning frequency scales)
    """
    def __init__(self, in_features: int = 6, mapping_size: int = 32,
                 scale: float = 5.0,
                 scale_space: Optional[float] = None,
                 scale_time: Optional[float] = None,
                 scale_cond: Optional[float] = None):
        super().__init__()
        s_space = scale_space if scale_space is not None else scale
        s_time = scale_time if scale_time is not None else scale
        s_cond = scale_cond if scale_cond is not None else scale

        if in_features == 6:
            # 3 spatial [x, y, z], 1 temporal [t], 2 operational conditions [I, SOH]
            b_space = torch.randn(3, mapping_size) * s_space
            b_time = torch.randn(1, mapping_size) * s_time
            b_cond = torch.randn(2, mapping_size) * s_cond
            b_init = torch.cat([b_space, b_time, b_cond], dim=0)
        elif in_features == 4:
            b_space = torch.randn(3, mapping_size) * s_space
            b_time = torch.randn(1, mapping_size) * s_time
            b_init = torch.cat([b_space, b_time], dim=0)
        else:
            b_init = torch.randn(in_features, mapping_size) * scale

        self.B = nn.Parameter(b_init, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = 2 * np.pi * torch.matmul(x, self.B)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
