import torch
import torch.nn as nn
from typing import Tuple, Optional

from models.fourier import FourierFeatureEncoding
from models.siren import SineActivation

class WangGatedBackbone(nn.Module):
    """
    Wang et al. (2021) Multiplicative Gated Architecture for PINNs:
      U = phi(W_u * x + b_u)
      V = phi(W_v * x + b_v)
      For each layer l:
        Z = phi(W_l * H + b_l)
        H = (1 - Z) * U + Z * V
    Injects coordinate representations into every hidden layer to mitigate gradient pathologies.
    """
    def __init__(self, in_features: int, hidden_dim: int, num_layers: int = 4, omega_0: float = 1.0):
        super().__init__()
        self.encoder_u = nn.Linear(in_features, hidden_dim)
        self.encoder_v = nn.Linear(in_features, hidden_dim)
        self.act_u = SineActivation(omega_0=omega_0)
        self.act_v = SineActivation(omega_0=omega_0)

        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.activations = nn.ModuleList([
            SineActivation(omega_0=omega_0) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.act_u(self.encoder_u(x))
        v = self.act_v(self.encoder_v(x))
        h = u
        for layer, act in zip(self.layers, self.activations):
            z = act(layer(h))
            h = (1.0 - z) * u + z * v
        return h


class TeacherPINN(nn.Module):
    """
    Fourier-SIREN Physics-Informed Neural Network (Teacher model).
    Supports:
      1. 6D conditional inputs [x, y, z, t, I, SOH]
      2. Wang et al. Multiplicative Gated Backbone
      3. Differentiable coordinate normalization to SI units
      4. Decoupled dual-head output: T(x, y, z, t) [K] and k(x, y, z, t) [W/(m*K)]
    """
    def __init__(self, in_dim: int = 6, hidden_dim: int = 128, mapping_size: int = 32,
                 scale: float = 5.0, scale_space: Optional[float] = None,
                 scale_time: Optional[float] = None, scale_cond: Optional[float] = None,
                 omega_0: float = 1.0, num_layers: int = 4,
                 coordinate_scale: Tuple[float, ...] = (0.009, 0.009, 0.065, 3600.0, 2.0, 1.0),
                 t_ambient: float = 297.15,
                 use_gated_backbone: bool = True):
        super().__init__()
        if len(coordinate_scale) != in_dim:
            raise ValueError(f"coordinate_scale length {len(coordinate_scale)} must match in_dim {in_dim}")

        self.t_ambient = t_ambient
        self.use_gated_backbone = use_gated_backbone
        self.register_buffer("coordinate_scale", torch.tensor(coordinate_scale, dtype=torch.float32))

        self.fourier = FourierFeatureEncoding(
            in_features=in_dim,
            mapping_size=mapping_size,
            scale=scale,
            scale_space=scale_space,
            scale_time=scale_time,
            scale_cond=scale_cond
        )
        encoded_dim = mapping_size * 2

        if use_gated_backbone:
            self.backbone = WangGatedBackbone(
                in_features=encoded_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                omega_0=omega_0
            )
        else:
            layers = [nn.Linear(encoded_dim, hidden_dim), SineActivation(omega_0=omega_0)]
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
