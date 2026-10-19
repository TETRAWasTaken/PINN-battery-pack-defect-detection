import torch
import torch.nn as nn

class SineActivation(nn.Module):
    """Periodic SIREN (Sinusoidal Representation Networks) activation function."""
    def __init__(self, omega_0: float = 30.0, omega_o: float = None):
        super().__init__()
        self.omega_0 = omega_o if omega_o is not None else omega_0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * x)
