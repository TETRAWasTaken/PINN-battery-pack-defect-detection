import torch
import torch.nn as nn

class StudentMLP(nn.Module):
    """
    Lightweight deployable Student MLP model for low-latency inference on edge BMS hardware.
    Mirrors the Teacher's physical parameterization:
      - Coordinate normalization to O(1)
      - Ambient baseline offset for temperature
      - Bounded thermal conductivity in physical range [0.5, 5.0] W/(m*K)
    """
    def __init__(self, in_dim: int = 4, hidden_dim: int = 32, out_dim: int = 2, num_layers: int = 3,
                 coordinate_scale: tuple[float, ...] = (0.009, 0.009, 0.065, 3600.0),
                 t_ambient: float = 297.15):
        super().__init__()
        if len(coordinate_scale) != in_dim:
            raise ValueError("coordinate_scale must contain one value per input coordinate")

        self.t_ambient = t_ambient
        self.register_buffer("coordinate_scale", torch.tensor(coordinate_scale, dtype=torch.float32))

        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        raw = self.net(coords / self.coordinate_scale)
        temperature = self.t_ambient + 35.0 * torch.tanh(raw[:, :1])
        conductivity = 0.5 + 4.5 * torch.sigmoid(raw[:, 1:])
        return torch.cat([temperature, conductivity], dim=-1)
