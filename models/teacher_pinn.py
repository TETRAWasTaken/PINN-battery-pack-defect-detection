import torch
import torch.nn as nn
from typing import Tuple, Optional

from models.fourier import FourierFeatureEncoding
from models.siren import SineActivation

class TeacherPINN(nn.Module):
    """
    Fourier-SIREN Physics-Informed Neural Network (Teacher model).
    Predicts:
      1. Temperature field T(x, y, z, t) in Kelvin with ambient baseline offset
      2. Effective thermal conductivity k(x, y, z, t) bounded in physical range [0.5, 5.0] W/(m*K)
    """
    def __init__(self, in_dim: int = 4, hidden_dim: int = 128, mapping_size: int = 32,
                 scale: float = 5.0, scale_space: Optional[float] = None, scale_time: Optional[float] = None,
                 omega_0: float = 1.0, num_layers: int = 4,
                 coordinate_scale: Tuple[float, ...] = (0.009, 0.009, 0.065, 3600.0),
                 t_ambient: float = 297.15):
        super().__init__()
        if len(coordinate_scale) != in_dim:
            raise ValueError("coordinate_scale must contain one value per input coordinate")

        self.t_ambient = t_ambient
        self.register_buffer("coordinate_scale", torch.tensor(coordinate_scale, dtype=torch.float32))

        self.fourier = FourierFeatureEncoding(
            in_features=in_dim,
            mapping_size=mapping_size,
            scale=scale,
            scale_space=scale_space,
            scale_time=scale_time
        )
        encoded_dim = mapping_size * 2

        layers = []
        layers.append(nn.Linear(encoded_dim, hidden_dim))
        layers.append(SineActivation(omega_0=omega_0))
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(SineActivation(omega_0=omega_0))
        self.backbone = nn.Sequential(*layers)

        self.temp_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            SineActivation(omega_0=omega_0),
            nn.Linear(32, 1)
        )
        self.cond_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            SineActivation(omega_0=omega_0),
            nn.Linear(32, 1)
        )

    def forward(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Differentiable coordinate normalization to O(1) domain
        normalized_coords = coords / self.coordinate_scale
        features = self.fourier(normalized_coords)
        shared = self.backbone(features)

        # Baseline ambient offset + bounded thermal rise [0, 35 K]
        temp_delta = 35.0 * torch.tanh(self.temp_head(shared))
        temperature = self.t_ambient + temp_delta

        # Physically bounded thermal conductivity [0.5, 5.0 W/(m*K)] for 18650 Li-ion cells
        conductivity = 0.5 + 4.5 * torch.sigmoid(self.cond_head(shared))
        return temperature, conductivity


# Also support standard PINN naming alias
PINN = TeacherPINN
